from __future__ import annotations

from pathlib import Path
from typing import Any

from marginal_value.data.split_manifest import build_split_manifest, select_split
from marginal_value.logging_utils import log_event, log_progress
from marginal_value.models.ssl_encoder import build_ssl_encoder, encoder_config_from_training
from marginal_value.training.feature_scaler import FeatureScaler, fit_feature_scaler


def train_ssl_encoder(config: dict[str, Any], *, mode: str = "smoke") -> dict[str, Any]:
    """Run a small masked-reconstruction encoder training loop.

    This function intentionally imports PyTorch inside the function body so the
    local Mac can validate and package code without installing or executing the
    training stack. It is meant to run inside Modal.
    """

    import random

    import numpy as np
    import torch
    import torch.nn.functional as F

    log_event("torch_train", "start", mode=mode)
    if mode not in {"smoke", "validation", "full"}:
        raise ValueError(f"Unsupported training mode: {mode}")
    if not torch.cuda.is_available():
        raise RuntimeError("Modal training function expected a CUDA GPU, but CUDA is unavailable.")

    seed = int(config["training"].get("seed", 7))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda")
    batch_size = int(config["training"]["batch_size"])
    max_steps_by_mode = {
        "smoke": int(config["training"]["smoke_steps"]),
        "validation": int(config["training"]["validation_steps"]),
        "full": int(config["training"]["max_steps"]),
    }
    max_steps = max_steps_by_mode[mode]
    learning_rate = float(config["training"]["learning_rate"])
    d_model = int(config["training"].get("d_model", 128))
    encoder_config = encoder_config_from_training(config)
    embedding_dim = int(encoder_config["embedding_dim"])
    loss_config = encoder_config["losses"]
    augmentation_config = encoder_config["augmentation"]
    log_event(
        "torch_train",
        "configured",
        mode=mode,
        seed=seed,
        batch_size=batch_size,
        max_steps=max_steps,
        learning_rate=learning_rate,
        d_model=d_model,
        embedding_dim=embedding_dim,
        architecture=encoder_config["architecture"],
    )

    dataset = _NpzFeatureDataset(config, smoke=mode == "smoke")
    model = build_ssl_encoder(
        feature_dim=dataset.feature_dim,
        d_model=d_model,
        embedding_dim=embedding_dim,
        dropout=float(encoder_config.get("dropout", 0.1)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    log_event(
        "torch_train",
        "model_ready",
        mode=mode,
        device=torch.cuda.get_device_name(0),
        feature_dim=dataset.feature_dim,
        feature_scaler_files=dataset.scaler.n_files if dataset.scaler is not None else 0,
        feature_scaler_windows=dataset.scaler.n_windows if dataset.scaler is not None else 0,
        n_feature_files=len(dataset.paths),
    )

    losses: list[float] = []
    loss_components: list[dict[str, float]] = []
    progress_every = max(1, max_steps // 10)
    mask_probability = float(config["training"].get("mask_probability", 0.20))
    reconstruction_weight = float(loss_config.get("masked_reconstruction", 1.0))
    invariance_weight = float(loss_config.get("vicreg_invariance", 1.0))
    variance_weight = float(loss_config.get("vicreg_variance", 1.0))
    covariance_weight = float(loss_config.get("vicreg_covariance", 0.04))
    for step in range(1, max_steps + 1):
        batch = dataset.sample_batch(batch_size)
        values = torch.tensor(batch, dtype=torch.float32, device=device)
        mask = torch.rand(values.shape[:2], device=device) < mask_probability
        masked_values = values.clone()
        masked_values[mask] = 0.0

        prediction = model.reconstruct(masked_values)
        target = values
        reconstruction_loss = (
            F.smooth_l1_loss(prediction[mask], target[mask])
            if mask.any()
            else F.smooth_l1_loss(prediction, target)
        )
        view_a = _augment_values(values, augmentation_config)
        view_b = _augment_values(values, augmentation_config)
        embedding_a = model.encode(view_a)
        embedding_b = model.encode(view_b)
        invariance_loss, variance_loss, covariance_loss = _vicreg_losses(embedding_a, embedding_b)
        loss = (
            reconstruction_weight * reconstruction_loss
            + invariance_weight * invariance_loss
            + variance_weight * variance_loss
            + covariance_weight * covariance_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        component_row = {
            "loss": float(loss.detach().cpu()),
            "reconstruction_loss": float(reconstruction_loss.detach().cpu()),
            "vicreg_invariance_loss": float(invariance_loss.detach().cpu()),
            "vicreg_variance_loss": float(variance_loss.detach().cpu()),
            "vicreg_covariance_loss": float(covariance_loss.detach().cpu()),
            "embedding_std_mean": float(torch.std(embedding_a, dim=0).mean().detach().cpu()),
        }
        loss_components.append(component_row)
        log_progress(
            "torch_train",
            "step",
            index=step,
            total=max_steps,
            every=progress_every,
            mode=mode,
            loss=losses[-1],
            reconstruction_loss=component_row["reconstruction_loss"],
            vicreg_variance_loss=component_row["vicreg_variance_loss"],
            embedding_std_mean=component_row["embedding_std_mean"],
        )

    checkpoint_path: Path | None = None
    checkpoint_read_ok = False
    if mode in {"validation", "full"}:
        checkpoint_dir = Path(config["training"]["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_name = (
            config["training"]["validation_checkpoint_name"]
            if mode == "validation"
            else config["training"].get("full_checkpoint_name", "ssl_encoder_baseline.pt")
        )
        checkpoint_path = checkpoint_dir / checkpoint_name
        log_event("torch_train", "checkpoint_write_start", mode=mode, checkpoint_path=str(checkpoint_path))
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "encoder_config": encoder_config,
                "feature_scaler": dataset.scaler.to_checkpoint() if dataset.scaler is not None else None,
                "losses": losses,
                "loss_components": loss_components,
            },
            checkpoint_path,
        )
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_read_ok = (
            "model_state_dict" in loaded
            and "losses" in loaded
            and len(loaded["losses"]) == max_steps
            and "encoder_config" in loaded
            and "feature_scaler" in loaded
        )
        if not checkpoint_read_ok:
            raise RuntimeError(f"Checkpoint verification failed for {checkpoint_path}")
        log_event("torch_train", "checkpoint_verified", mode=mode, checkpoint_path=str(checkpoint_path))

    result: dict[str, Any] = {
        "mode": mode,
        "steps": max_steps,
        "last_loss": losses[-1],
        "last_reconstruction_loss": loss_components[-1]["reconstruction_loss"],
        "last_embedding_std_mean": loss_components[-1]["embedding_std_mean"],
        "device": torch.cuda.get_device_name(0),
    }
    if checkpoint_path is not None:
        result["checkpoint_path"] = str(checkpoint_path)
        result["checkpoint_read_ok"] = checkpoint_read_ok
    log_event("torch_train", "done", **result)
    return result


def _augment_values(values, config: dict[str, Any]):
    import torch

    noise_std = float(config.get("noise_std", 0.03))
    timestep_dropout = float(config.get("timestep_dropout", 0.05))
    channel_dropout = float(config.get("channel_dropout", 0.03))
    output = values
    if noise_std > 0.0:
        output = output + torch.randn_like(output) * noise_std
    if timestep_dropout > 0.0:
        mask = torch.rand(output.shape[:2], device=output.device) < timestep_dropout
        output = output.masked_fill(mask[..., None], 0.0)
    if channel_dropout > 0.0:
        mask = torch.rand(output.shape[0], 1, output.shape[2], device=output.device) < channel_dropout
        output = output.masked_fill(mask, 0.0)
    return output


def _vicreg_losses(embedding_a, embedding_b):
    import torch
    import torch.nn.functional as F

    invariance_loss = F.mse_loss(embedding_a, embedding_b)
    variance_loss = _variance_loss(embedding_a) + _variance_loss(embedding_b)
    covariance_loss = _covariance_loss(embedding_a) + _covariance_loss(embedding_b)
    return invariance_loss, variance_loss, covariance_loss


def _variance_loss(embedding):
    import torch

    std = torch.sqrt(torch.var(embedding, dim=0) + 1.0e-4)
    return torch.mean(torch.relu(1.0 - std))


def _covariance_loss(embedding):
    centered = embedding - embedding.mean(dim=0)
    n_rows = max(embedding.shape[0] - 1, 1)
    covariance = centered.T @ centered / n_rows
    return _off_diagonal(covariance).pow(2).sum() / embedding.shape[1]


def _off_diagonal(matrix):
    n_rows, n_cols = matrix.shape
    if n_rows != n_cols:
        raise ValueError("matrix must be square")
    return matrix.flatten()[:-1].view(n_rows - 1, n_rows + 1)[:, 1:].flatten()


class _NpzFeatureDataset:
    def __init__(self, config: dict[str, Any], *, smoke: bool) -> None:
        import numpy as np

        self.np = np
        encoder_config = encoder_config_from_training(config)
        normalization_config = encoder_config["normalization"]
        data_root = Path(config["data"]["root"])
        log_event("torch_train_dataset", "manifest_load_start", root=str(data_root), smoke=smoke)
        manifest = build_split_manifest(
            data_root,
            pretrain_manifest=config["data"]["pretrain_manifest"],
            val_manifest=config["data"]["val_manifest"],
            feature_glob=config["data"].get("feature_glob", "cache/features/*.npz"),
            raw_glob=config["data"].get("raw_glob", "cache/raw/*.jsonl"),
        )
        train_split = config["data"].get("train_split", "pretrain")
        self.paths = sorted(row.feature_path for row in select_split(manifest, train_split))
        if smoke:
            max_files = int(config["data"].get("max_files_for_smoke", 16))
            self.paths = self.paths[:max_files]
        if not self.paths:
            raise FileNotFoundError(
                f"No cached feature files found for split '{train_split}'. "
                "Check Modal manifests and cached features before launching training."
            )
        self.feature_dim = int(config["data"].get("feature_dim", 75))
        self.rng = np.random.default_rng(int(config["training"].get("seed", 7)))
        self.scaler: FeatureScaler | None = None
        if bool(normalization_config.get("enabled", True)):
            scaler_max_files = int(normalization_config.get("max_files_for_fit", 512))
            scaler_max_windows = int(normalization_config.get("max_windows_per_file", 256))
            log_event(
                "torch_train_dataset",
                "scaler_fit_start",
                max_files_for_fit=scaler_max_files,
                max_windows_per_file=scaler_max_windows,
                smoke=smoke,
            )
            self.scaler = fit_feature_scaler(
                self.paths,
                feature_dim=self.feature_dim,
                max_files=scaler_max_files,
                max_windows_per_file=scaler_max_windows,
            )
            log_event(
                "torch_train_dataset",
                "scaler_fit_done",
                n_files=self.scaler.n_files,
                n_windows=self.scaler.n_windows,
                mean_abs=float(np.mean(np.abs(self.scaler.mean))),
                scale_min=float(np.min(self.scaler.scale)),
                scale_max=float(np.max(self.scaler.scale)),
            )
        log_event(
            "torch_train_dataset",
            "ready",
            train_split=train_split,
            smoke=smoke,
            n_feature_files=len(self.paths),
            feature_dim=self.feature_dim,
        )

    def sample_batch(self, batch_size: int):
        windows = []
        attempts = 0
        while len(windows) < batch_size and attempts < batch_size * 20:
            attempts += 1
            path = self.paths[int(self.rng.integers(0, len(self.paths)))]
            with self.np.load(path) as data:
                values = self.np.asarray(data["window_features"], dtype="float32")
            if values.ndim != 2 or values.shape[1] != self.feature_dim:
                continue
            if self.scaler is not None:
                values = self.scaler.transform(values)
            windows.append(values)
        if len(windows) < batch_size:
            raise RuntimeError("Could not sample enough cached feature windows for a training batch.")
        return self.np.stack(windows, axis=0)
