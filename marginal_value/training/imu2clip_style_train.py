from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from marginal_value.data.split_manifest import hash_manifest_url
from marginal_value.logging_utils import log_event, log_progress
from marginal_value.models.imu2clip_encoder import IMU2CLIPStyleEncoder, option_b_checkpoint_metadata
from marginal_value.preprocessing.quality import load_modal_jsonl_imu


def train_imu2clip_style_encoder(config: Mapping[str, Any], *, mode: str = "smoke") -> dict[str, Any]:
    """Train Option B IMU2CLIP-style weights with IMU-only contrastive learning.

    This path is Modal-oriented. PyTorch is imported inside the function so local
    package checks do not require local torch installation.
    """

    import random

    import torch

    if mode not in {"smoke", "validation", "full"}:
        raise ValueError(f"Unsupported training mode: {mode}")
    if not torch.cuda.is_available():
        raise RuntimeError("IMU2CLIP-style training is expected to run on a Modal CUDA GPU.")

    training_config = dict(config.get("training", {}))
    encoder_config = dict(config.get("encoder", {}))
    execution_config = dict(config.get("execution", {}))
    run_id = str(execution_config.get("run_id", ""))
    seed = int(training_config.get("seed", 17))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    max_steps = {
        "smoke": int(training_config.get("smoke_steps", 5)),
        "validation": int(training_config.get("validation_steps", 200)),
        "full": int(training_config.get("max_steps", 5000)),
    }[mode]
    batch_size = int(training_config.get("batch_size", 32))
    learning_rate = float(training_config.get("learning_rate", 3.0e-4))
    temperature = float(training_config.get("temperature", 0.1))
    device = torch.device("cuda")

    dataset = _RawIMUWindowDataset(config, mode=mode)
    log_event(
        "imu2clip_style_train",
        "dataset_ready",
        run_id=run_id,
        mode=mode,
        n_raw_files=len(dataset.paths),
        target_hz=dataset.target_hz,
        expected_hz=dataset.expected_hz,
        raw_window_timesteps=dataset.raw_window_timesteps,
        model_window_timesteps=dataset.window_timesteps,
    )
    model = IMU2CLIPStyleEncoder(
        in_channels=6,
        hidden_dim=int(encoder_config.get("hidden_dim", 512)),
        output_dim=int(encoder_config.get("output_dim", 512)),
        gru_layers=int(encoder_config.get("gru_layers", 2)),
        cnn_layers=int(encoder_config.get("cnn_layers", 4)),
        cnn_channels=int(encoder_config.get("cnn_channels", 32)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=float(training_config.get("weight_decay", 0.01)))

    losses: list[float] = []
    diagnostics: list[dict[str, float]] = []
    configured_progress_every = int(training_config.get("progress_every_steps", 0))
    progress_every = max(1, configured_progress_every if configured_progress_every > 0 else max_steps // 10)
    log_event(
        "imu2clip_style_train",
        "start",
        run_id=run_id,
        mode=mode,
        seed=seed,
        batch_size=batch_size,
        max_steps=max_steps,
        learning_rate=learning_rate,
        temperature=temperature,
        n_raw_files=len(dataset.paths),
        device=torch.cuda.get_device_name(0),
    )

    for step in range(1, max_steps + 1):
        view_a_np, view_b_np = dataset.sample_augmented_batch(batch_size)
        view_a = torch.as_tensor(view_a_np.transpose(0, 2, 1), dtype=torch.float32, device=device)
        view_b = torch.as_tensor(view_b_np.transpose(0, 2, 1), dtype=torch.float32, device=device)
        embedding_a = model(view_a)
        embedding_b = model(view_b)
        loss = _nt_xent_loss(embedding_a, embedding_b, temperature=temperature)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            merged = torch.cat([embedding_a, embedding_b], dim=0)
            row = {
                "loss": float(loss.detach().cpu()),
                "embedding_std_mean": float(torch.std(merged, dim=0).mean().detach().cpu()),
                "effective_rank": float(_effective_rank_torch(merged).detach().cpu()),
            }
        losses.append(row["loss"])
        diagnostics.append(row)
        log_progress(
            "imu2clip_style_train",
            "step",
            index=step,
            total=max_steps,
            every=progress_every,
            mode=mode,
            run_id=run_id,
            loss=row["loss"],
            embedding_std_mean=row["embedding_std_mean"],
            effective_rank=row["effective_rank"],
        )

    result: dict[str, Any] = {
        "mode": mode,
        "run_id": run_id,
        "steps": max_steps,
        "last_loss": losses[-1],
        "last_embedding_std_mean": diagnostics[-1]["embedding_std_mean"],
        "last_effective_rank": diagnostics[-1]["effective_rank"],
        "device": torch.cuda.get_device_name(0),
    }

    if mode in {"validation", "full"}:
        checkpoint_dir = Path(str(training_config.get("checkpoint_dir", "/artifacts/checkpoints/imu2clip_style")))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_name = (
            str(training_config.get("validation_checkpoint_name", "validation_imu2clip_style_encoder.pt"))
            if mode == "validation"
            else str(training_config.get("full_checkpoint_name", "imu2clip_style_encoder.pt"))
        )
        checkpoint_path = checkpoint_dir / checkpoint_name
        log_event("imu2clip_style_train", "checkpoint_write_start", run_id=run_id, mode=mode, checkpoint_path=str(checkpoint_path))
        payload = {
            "model_state_dict": model.state_dict(),
            "config": dict(config),
            "encoder_config": {
                **option_b_checkpoint_metadata(),
                "hidden_dim": int(encoder_config.get("hidden_dim", 512)),
                "output_dim": int(encoder_config.get("output_dim", 512)),
                "gru_layers": int(encoder_config.get("gru_layers", 2)),
                "cnn_layers": int(encoder_config.get("cnn_layers", 4)),
                "cnn_channels": int(encoder_config.get("cnn_channels", 32)),
            },
            "losses": losses,
            "diagnostics": diagnostics,
        }
        torch.save(payload, checkpoint_path)
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model_state_dict" not in loaded or "encoder_config" not in loaded:
            raise RuntimeError(f"Checkpoint verification failed for {checkpoint_path}")
        result["checkpoint_path"] = str(checkpoint_path)
        result["checkpoint_read_ok"] = True
        log_event("imu2clip_style_train", "checkpoint_verified", run_id=run_id, mode=mode, checkpoint_path=str(checkpoint_path))

    log_event("imu2clip_style_train", "done", **result)
    return result


def preload_imu2clip_style_windows(config: Mapping[str, Any], *, mode: str = "validation") -> dict[str, Any]:
    """Build and persist the raw-window pool without allocating a GPU."""

    if mode not in {"smoke", "validation", "full"}:
        raise ValueError(f"Unsupported preload mode: {mode}")
    data_config = dict(config.get("data", {}))
    execution_config = dict(config.get("execution", {}))
    run_id = str(execution_config.get("run_id", ""))
    output_path = Path(str(data_config.get("preloaded_windows_path", f"/artifacts/cache/imu2clip_style/{mode}_windows.npz")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = _RawIMUWindowDataset(_config_with_data(config, {"preload_windows": True, "preloaded_windows_path": ""}), mode=mode)
    if dataset.preloaded_windows is None:
        raise RuntimeError("Preload did not create a window pool.")
    np.savez_compressed(
        output_path,
        windows=dataset.preloaded_windows.astype("float32"),
        mode=np.asarray(mode),
        run_id=np.asarray(run_id),
    )
    result = {
        "mode": mode,
        "run_id": run_id,
        "preloaded_windows_path": str(output_path),
        "windows": int(len(dataset.preloaded_windows)),
        "memory_mb": float(dataset.preloaded_windows.nbytes / 1_000_000.0),
    }
    log_event("imu2clip_style_train", "preload_artifact_written", **result)
    return result


def _nt_xent_loss(embedding_a, embedding_b, *, temperature: float):
    import torch
    import torch.nn.functional as F

    batch_size = int(embedding_a.shape[0])
    embeddings = F.normalize(torch.cat([embedding_a, embedding_b], dim=0), dim=1)
    logits = embeddings @ embeddings.T / float(temperature)
    logits.fill_diagonal_(-1.0e9)
    labels = torch.cat(
        [
            torch.arange(batch_size, 2 * batch_size, device=embedding_a.device),
            torch.arange(0, batch_size, device=embedding_a.device),
        ]
    )
    return F.cross_entropy(logits, labels)


def _effective_rank_torch(embeddings):
    import torch

    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    probabilities = singular_values / torch.clamp(singular_values.sum(), min=1.0e-12)
    entropy = -(probabilities * torch.log(torch.clamp(probabilities, min=1.0e-12))).sum()
    return torch.exp(entropy)


class _RawIMUWindowDataset:
    def __init__(self, config: Mapping[str, Any], *, mode: str) -> None:
        self.config = config
        self.mode = str(mode)
        self.data_config = dict(config.get("data", {}))
        self.training_config = dict(config.get("training", {}))
        self.execution_config = dict(config.get("execution", {}))
        self.run_id = str(self.execution_config.get("run_id", ""))
        self.root = Path(str(self.data_config.get("root", "/data")))
        self.target_hz = int(self.data_config.get("target_hz", 30))
        self.expected_hz = int(self.data_config.get("expected_hz", 200))
        self.window_timesteps = int(self.data_config.get("window_timesteps", 1000))
        self.raw_window_timesteps = max(2, int(round(float(self.data_config.get("window_len_s", 2.0)) * self.target_hz)))
        self.rng = np.random.default_rng(int(self.training_config.get("seed", 17)))
        self.paths = self._load_paths()
        if not self.paths:
            raise FileNotFoundError("No raw IMU files were found for IMU2CLIP-style training.")
        self.preloaded_windows: np.ndarray | None = None
        preloaded_path = str(self.data_config.get("preloaded_windows_path", "")).strip()
        if preloaded_path and Path(preloaded_path).exists():
            with np.load(preloaded_path) as data:
                self.preloaded_windows = np.asarray(data["windows"], dtype=np.float32)
            log_event(
                "imu2clip_style_train",
                "preloaded_from_cache",
                run_id=self.run_id,
                mode=self.mode,
                path=preloaded_path,
                windows=len(self.preloaded_windows),
                memory_mb=float(self.preloaded_windows.nbytes / 1_000_000.0),
            )
        elif bool(self.data_config.get("preload_windows", False)):
            self.preloaded_windows = self._preload_windows()

    def sample_augmented_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        from marginal_value.models.imu2clip_encoder import imu_augment

        if self.preloaded_windows is not None:
            return self._sample_preloaded_batch(batch_size, imu_augment)

        view_a = []
        view_b = []
        attempts = 0
        while len(view_a) < batch_size and attempts < batch_size * 40:
            attempts += 1
            path = self.paths[int(self.rng.integers(0, len(self.paths)))]
            try:
                values, _timestamps = load_modal_jsonl_imu(path)
            except Exception:
                continue
            if len(values) < self.raw_window_timesteps:
                continue
            start = int(self.rng.integers(0, len(values) - self.raw_window_timesteps + 1))
            window = self._normalize(values[start : start + self.raw_window_timesteps])
            view_a.append(self._resize(imu_augment(window, rng=self.rng), self.window_timesteps))
            view_b.append(self._resize(imu_augment(window, rng=self.rng), self.window_timesteps))
        if len(view_a) < batch_size:
            raise RuntimeError("Could not sample enough raw IMU windows for IMU2CLIP-style training.")
        return np.stack(view_a).astype("float32"), np.stack(view_b).astype("float32")

    def _sample_preloaded_batch(self, batch_size: int, augment_fn) -> tuple[np.ndarray, np.ndarray]:
        if self.preloaded_windows is None or len(self.preloaded_windows) == 0:
            raise RuntimeError("No preloaded IMU windows are available for sampling.")
        indices = self.rng.integers(0, len(self.preloaded_windows), size=int(batch_size))
        view_a = []
        view_b = []
        for index in indices:
            window = self.preloaded_windows[int(index)]
            view_a.append(self._resize(augment_fn(window, rng=self.rng), self.window_timesteps))
            view_b.append(self._resize(augment_fn(window, rng=self.rng), self.window_timesteps))
        return np.stack(view_a).astype("float32"), np.stack(view_b).astype("float32")

    def _preload_windows(self) -> np.ndarray:
        windows = []
        failed = 0
        progress_every = max(1, int(self.data_config.get("preload_progress_every", 512)))
        windows_per_file = max(1, int(self.data_config.get("preload_windows_per_file", 1)))
        log_event(
            "imu2clip_style_train",
            "preload_start",
            run_id=self.run_id,
            mode=self.mode,
            n_paths=len(self.paths),
            windows_per_file=windows_per_file,
        )
        for index, path in enumerate(self.paths, start=1):
            try:
                values, _timestamps = load_modal_jsonl_imu(path)
            except Exception:
                failed += 1
                continue
            if len(values) < self.raw_window_timesteps:
                failed += 1
                continue
            for _ in range(windows_per_file):
                windows.append(self._sample_window(values))
            if index % progress_every == 0 or index == len(self.paths):
                log_event(
                    "imu2clip_style_train",
                    "preload_progress",
                    run_id=self.run_id,
                    mode=self.mode,
                    index=index,
                    total=len(self.paths),
                    windows=len(windows),
                    failed=failed,
                )
        if not windows:
            raise RuntimeError("Could not preload any IMU windows for IMU2CLIP-style training.")
        output = np.stack(windows).astype("float32")
        log_event(
            "imu2clip_style_train",
            "preload_ready",
            run_id=self.run_id,
            mode=self.mode,
            windows=len(output),
            failed=failed,
            memory_mb=float(output.nbytes / 1_000_000.0),
        )
        return output

    def _sample_window(self, values: np.ndarray) -> np.ndarray:
        start = int(self.rng.integers(0, len(values) - self.raw_window_timesteps + 1))
        window = self._normalize(values[start : start + self.raw_window_timesteps])
        return self._resize(window, self.window_timesteps)

    def _load_paths(self) -> list[Path]:
        manifest_path = self.root / str(self.data_config.get("manifest", "cache/manifests/pretrain_full_cached_urls.txt"))
        max_files_by_mode = {
            "smoke": int(self.data_config.get("smoke_max_files", 32)),
            "validation": int(self.data_config.get("validation_max_files", self.data_config.get("max_files", 0))),
            "full": int(self.data_config.get("full_max_files", self.data_config.get("max_files", 0))),
        }
        max_files = max_files_by_mode.get(self.mode, 0)
        paths = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if not value:
                continue
            if value.startswith("{"):
                import json

                record = json.loads(value)
                value = str(record.get("raw_path") or record.get("path") or record.get("url") or "")
            paths.append(self._resolve_path(value))
            if max_files > 0 and len(paths) >= max_files:
                break
        return paths

    def _resolve_path(self, value: str) -> Path:
        if value.startswith("http://") or value.startswith("https://") or value.startswith("gs://"):
            return self.root / "cache" / "raw" / f"{hash_manifest_url(value)}.jsonl"
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        clip = np.asarray(values[:, :6], dtype=np.float32)
        mean = np.mean(clip, axis=0, keepdims=True)
        std = np.std(clip, axis=0, keepdims=True)
        return ((clip - mean) / np.where(std < 1.0e-6, 1.0, std)).astype("float32")

    def _resize(self, values: np.ndarray, target_len: int) -> np.ndarray:
        source = np.asarray(values[:, :6], dtype=np.float32)
        if len(source) == target_len:
            return source
        old_x = np.linspace(0.0, 1.0, num=len(source), endpoint=True)
        new_x = np.linspace(0.0, 1.0, num=target_len, endpoint=True)
        return np.stack([np.interp(new_x, old_x, source[:, channel]) for channel in range(6)], axis=1).astype("float32")


def _config_with_data(config: Mapping[str, Any], data_updates: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(config)
    data_config = dict(output.get("data", {}))
    data_config.update(data_updates)
    output["data"] = data_config
    return output
