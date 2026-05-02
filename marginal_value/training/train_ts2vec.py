from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from marginal_value.data.split_manifest import hash_manifest_url, read_manifest_urls
from marginal_value.logging_utils import log_event, log_progress
from marginal_value.preprocessing.quality import load_modal_jsonl_imu


def train_ts2vec(
    manifest_path: str,
    checkpoint_dir: str,
    output_dims: int = 320,
    hidden_dims: int = 64,
    n_epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1.0e-3,
    crop_min_len: int = 150,
    crop_max_len: int = 600,
    device: str = "cuda",
    max_steps_per_epoch: int | None = None,
    collapse_sample_size: int = 1000,
    training_log_path: str | None = None,
    alpha: float = 0.1,
    instance_warmup_epochs: int = 0,
    collapse_pooling: str = "aggregation",
    max_temporal_positions: int = 64,
    collapse_min_effective_rank: float = 10.0,
    collapse_max_mean_pairwise_cosine: float = 0.9,
) -> dict[str, Any]:
    import torch

    from marginal_value.models.ts2vec_encoder import TS2VecEncoder
    from marginal_value.models.ts2vec_loss import hierarchical_contrastive_loss_parts

    manifest = Path(manifest_path)
    urls = read_manifest_urls(manifest)
    if not urls:
        raise ValueError("TS2Vec training manifest must contain at least one URL/path.")
    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    docs_log_path = Path(training_log_path) if training_log_path is not None else Path("docs") / "ts2vec_training_log.md"
    collapse_pooling = _validate_collapse_pooling(collapse_pooling)
    config = {
        "input_dims": int(hidden_dims),
        "hidden_dims": int(hidden_dims),
        "output_dims": int(output_dims),
        "n_layers": 10,
        "manifest_path": str(manifest_path),
        "batch_size": int(batch_size),
        "crop_min_len": int(crop_min_len),
        "crop_max_len": int(crop_max_len),
        "lr": float(lr),
        "max_steps_per_epoch": None if max_steps_per_epoch is None else int(max_steps_per_epoch),
        "collapse_sample_size": int(collapse_sample_size),
        "training_log_path": str(docs_log_path),
        "alpha": float(alpha),
        "instance_warmup_epochs": int(instance_warmup_epochs),
        "collapse_pooling": collapse_pooling,
        "max_temporal_positions": int(max_temporal_positions),
        "collapse_min_effective_rank": float(collapse_min_effective_rank),
        "collapse_max_mean_pairwise_cosine": float(collapse_max_mean_pairwise_cosine),
    }
    model = TS2VecEncoder(
        input_dims=hidden_dims,
        hidden_dims=hidden_dims,
        output_dims=output_dims,
        n_layers=10,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1.0e-4)
    rng = np.random.default_rng(20260429)
    best_score = float("inf")
    best_path = checkpoint_root / "ts2vec_best.pt"
    log_event(
        "ts2vec_train",
        "start",
        n_urls=len(urls),
        n_epochs=n_epochs,
        batch_size=batch_size,
        max_steps_per_epoch=max_steps_per_epoch,
        collapse_sample_size=collapse_sample_size,
        alpha=alpha,
        instance_warmup_epochs=instance_warmup_epochs,
        collapse_pooling=collapse_pooling,
        max_temporal_positions=max_temporal_positions,
        collapse_min_effective_rank=collapse_min_effective_rank,
        collapse_max_mean_pairwise_cosine=collapse_max_mean_pairwise_cosine,
    )

    last_metrics: dict[str, float] = {}
    epoch0_embeddings = _sample_embeddings(
        model,
        manifest,
        urls,
        device=device,
        rng=rng,
        sample_size=min(int(collapse_sample_size), len(urls)),
        pooling=collapse_pooling,
    )
    epoch0_metrics = check_collapse(epoch0_embeddings)
    append_training_log(docs_log_path, epoch=0, train_loss=0.0, collapse_metrics=epoch0_metrics, instance_loss=0.0, temporal_loss=0.0)
    log_event("ts2vec_train", "epoch0_metrics", collapse_pooling=collapse_pooling, **epoch0_metrics)
    stopped_early = False
    for epoch in range(1, int(n_epochs) + 1):
        model.train()
        epoch_losses: list[float] = []
        epoch_instance_losses: list[float] = []
        epoch_temporal_losses: list[float] = []
        epoch_alpha = _alpha_for_epoch(epoch, alpha=float(alpha), instance_warmup_epochs=int(instance_warmup_epochs))
        full_steps_per_epoch = max(1, int(np.ceil(len(urls) / max(1, int(batch_size)))))
        steps_per_epoch = full_steps_per_epoch
        if max_steps_per_epoch is not None:
            steps_per_epoch = max(1, min(full_steps_per_epoch, int(max_steps_per_epoch)))
        for step in range(1, steps_per_epoch + 1):
            batch_urls = list(rng.choice(urls, size=min(int(batch_size), len(urls)), replace=len(urls) < int(batch_size)))
            crop_len = int(rng.integers(int(crop_min_len), int(crop_max_len) + 1))
            left, right, overlap_indices = _load_batch_views(manifest, batch_urls, crop_len=crop_len, rng=rng)
            left_tensor = torch.tensor(left, dtype=torch.float32, device=device)
            right_tensor = torch.tensor(right, dtype=torch.float32, device=device)
            z1 = model.encode_sequence(left_tensor, return_layers=True)
            z2 = model.encode_sequence(right_tensor, return_layers=True)
            loss_parts = hierarchical_contrastive_loss_parts(
                z1,
                z2,
                alpha=epoch_alpha,
                overlap_indices=overlap_indices,
                max_temporal_positions=int(max_temporal_positions),
            )
            loss = loss_parts["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_instance_losses.append(float(loss_parts["instance_loss"].detach().cpu()))
            epoch_temporal_losses.append(float(loss_parts["temporal_loss"].detach().cpu()))
            log_progress(
                "ts2vec_train",
                "step",
                index=step,
                total=steps_per_epoch,
                every=max(1, steps_per_epoch // 5),
                epoch=epoch,
                alpha=epoch_alpha,
                loss=epoch_losses[-1],
                instance_loss=epoch_instance_losses[-1],
                temporal_loss=epoch_temporal_losses[-1],
            )

        model.eval()
        embeddings = _sample_embeddings(
            model,
            manifest,
            urls,
            device=device,
            rng=rng,
            sample_size=min(int(collapse_sample_size), len(urls)),
            pooling=collapse_pooling,
        )
        last_metrics = check_collapse(embeddings)
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        instance_loss = float(np.mean(epoch_instance_losses)) if epoch_instance_losses else 0.0
        temporal_loss = float(np.mean(epoch_temporal_losses)) if epoch_temporal_losses else 0.0
        append_training_log(
            docs_log_path,
            epoch=epoch,
            train_loss=train_loss,
            collapse_metrics=last_metrics,
            instance_loss=instance_loss,
            temporal_loss=temporal_loss,
        )
        collapsed = _is_collapsed(
            last_metrics,
            min_effective_rank=float(collapse_min_effective_rank),
            max_mean_pairwise_cosine=float(collapse_max_mean_pairwise_cosine),
        )
        log_event(
            "ts2vec_train",
            "epoch_done",
            epoch=epoch,
            alpha=epoch_alpha,
            train_loss=train_loss,
            instance_loss=instance_loss,
            temporal_loss=temporal_loss,
            collapsed=collapsed,
            collapse_pooling=collapse_pooling,
            **last_metrics,
        )
        if collapsed:
            log_event("ts2vec_train", "collapse_detected", epoch=epoch, **last_metrics)
            stopped_early = True
            break
        checkpoint = {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "collapse_metrics": last_metrics,
        }
        if epoch % 5 == 0:
            torch.save(checkpoint, checkpoint_root / f"ts2vec_epoch_{epoch:03d}.pt")
        score = float(last_metrics["mean_pairwise_cosine"])
        if score < best_score:
            best_score = score
            torch.save(checkpoint, best_path)

    result = {
        "n_epochs": int(n_epochs),
        "checkpoint_dir": str(checkpoint_root),
        "best_checkpoint_path": str(best_path) if best_path.exists() else "",
        "collapse_metrics": last_metrics,
        "epoch0_collapse_metrics": epoch0_metrics,
        "stopped_early": stopped_early,
    }
    log_event("ts2vec_train", "done", **result)
    return result


def check_collapse(embeddings: np.ndarray) -> dict[str, float]:
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty 2D array.")
    normalized = values / (np.linalg.norm(values, axis=1, keepdims=True) + 1.0e-12)
    similarities = normalized @ normalized.T
    if len(values) > 1:
        mask = ~np.eye(len(values), dtype=bool)
        mean_pairwise = float(np.mean(similarities[mask]))
    else:
        mean_pairwise = 1.0
    centered = normalized - np.mean(normalized, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    power = singular * singular
    probabilities = power / max(float(np.sum(power)), 1.0e-12)
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1.0e-12)))
    std_per_dim = np.std(normalized, axis=0)
    return {
        "effective_rank": float(np.exp(entropy)),
        "mean_pairwise_cosine": mean_pairwise,
        "std_per_dim_mean": float(np.mean(std_per_dim)),
        "std_per_dim_std": float(np.std(std_per_dim)),
        "std_per_dim_min": float(np.min(std_per_dim)),
        "std_per_dim_max": float(np.max(std_per_dim)),
    }


def append_training_log(
    path: str | Path,
    *,
    epoch: int,
    train_loss: float,
    collapse_metrics: Mapping[str, float],
    instance_loss: float | None = None,
    temporal_loss: float | None = None,
) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = "| epoch | effective_rank | mean_pairwise_cosine | train_loss | instance_loss | temporal_loss |\n| ---: | ---: | ---: | ---: | ---: | ---: |\n"
    row = (
        f"| {int(epoch)} | {float(collapse_metrics.get('effective_rank', 0.0)):.4f} | "
        f"{float(collapse_metrics.get('mean_pairwise_cosine', 0.0)):.4f} | {float(train_loss):.6f} | "
        f"{float(instance_loss if instance_loss is not None else 0.0):.6f} | "
        f"{float(temporal_loss if temporal_loss is not None else 0.0):.6f} |\n"
    )
    if not log_path.exists() or not log_path.read_text(encoding="utf-8").strip():
        log_path.write_text("# TS2Vec Training Log\n\n" + header + row, encoding="utf-8")
    else:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(row)


def _is_collapsed(
    metrics: Mapping[str, float],
    *,
    min_effective_rank: float = 10.0,
    max_mean_pairwise_cosine: float = 0.9,
) -> bool:
    return (
        float(metrics.get("effective_rank", 0.0)) < float(min_effective_rank)
        or float(metrics.get("mean_pairwise_cosine", 1.0)) > float(max_mean_pairwise_cosine)
    )


def _alpha_for_epoch(epoch: int, *, alpha: float, instance_warmup_epochs: int) -> float:
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    if int(epoch) <= max(0, int(instance_warmup_epochs)):
        return 0.0
    return float(alpha)


def _validate_collapse_pooling(pooling: str) -> str:
    value = str(pooling)
    if value not in {"aggregation", "max", "mean", "mean_max"}:
        raise ValueError("collapse_pooling must be one of: aggregation, max, mean, mean_max.")
    return value


def _load_batch_views(
    manifest_path: Path,
    urls: Sequence[str],
    *,
    crop_len: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    left_rows = []
    right_rows = []
    overlap_indices = []
    for url in urls:
        clip = _normalize_clip(_load_manifest_clip(manifest_path, url))
        left, right, left_idx, right_idx = _two_overlapping_fixed_crops(clip, crop_len=crop_len, rng=rng)
        left_rows.append(left)
        right_rows.append(right)
        overlap_indices.append((left_idx, right_idx))
    return np.stack(left_rows).astype("float32"), np.stack(right_rows).astype("float32"), overlap_indices


def _sample_embeddings(
    model: object,
    manifest_path: Path,
    urls: Sequence[str],
    *,
    device: str,
    rng: np.random.Generator,
    sample_size: int,
    pooling: str = "max",
) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    pooling = _validate_collapse_pooling(pooling)
    selected = list(rng.choice(urls, size=int(sample_size), replace=False))
    outputs = []
    with torch.no_grad():
        for url in selected:
            clip = _normalize_clip(_load_manifest_clip(manifest_path, str(url)))
            tensor = torch.tensor(clip[None, :, :], dtype=torch.float32, device=device)
            if pooling == "aggregation":
                embedding_tensor = model(tensor)
            else:
                sequence = model.encode_sequence(tensor, return_layers=False)
                embedding_tensor = _pool_sequence_embedding(sequence, pooling)
            embedding = F.normalize(embedding_tensor, dim=-1).detach().cpu().numpy()[0]
            outputs.append(embedding)
    return np.vstack(outputs).astype("float32")


def _pool_sequence_embedding(sequence: object, pooling: str):
    import torch

    pooling = _validate_collapse_pooling(pooling)
    if pooling == "max":
        return torch.max(sequence, dim=1).values
    if pooling == "mean":
        return torch.mean(sequence, dim=1)
    return torch.cat([torch.mean(sequence, dim=1), torch.max(sequence, dim=1).values], dim=-1)


def _two_overlapping_fixed_crops(values: np.ndarray, *, crop_len: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    padded = _pad_to_length(values, int(crop_len))
    if len(padded) == crop_len:
        indices = np.arange(int(crop_len), dtype=np.int64)
        return padded.copy(), padded.copy(), indices, indices
    max_start = len(padded) - int(crop_len)
    start_a = int(rng.integers(0, max_start + 1))
    max_shift = max(1, int(crop_len // 2))
    low = max(0, start_a - max_shift)
    high = min(max_start, start_a + max_shift)
    start_b = int(rng.integers(low, high + 1))
    overlap_start = max(start_a, start_b)
    overlap_end = min(start_a + int(crop_len), start_b + int(crop_len))
    if overlap_end <= overlap_start:
        raise RuntimeError("Overlapping crop sampler produced disjoint crops.")
    left_indices = np.arange(overlap_start - start_a, overlap_end - start_a, dtype=np.int64)
    right_indices = np.arange(overlap_start - start_b, overlap_end - start_b, dtype=np.int64)
    return (
        padded[start_a : start_a + crop_len].copy(),
        padded[start_b : start_b + crop_len].copy(),
        left_indices,
        right_indices,
    )


def _pad_to_length(values: np.ndarray, length: int) -> np.ndarray:
    if len(values) >= length:
        return values
    output = np.zeros((length, values.shape[1]), dtype=np.float32)
    output[: len(values)] = values
    return output


def _normalize_clip(clip: np.ndarray) -> np.ndarray:
    values = np.asarray(clip, dtype=np.float32)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if not np.isfinite(std) or std < 1.0e-6:
        std = 1.0
    return np.nan_to_num((values - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def _load_manifest_clip(manifest_path: Path, url_or_path: str) -> np.ndarray:
    raw_path = _raw_path_for_manifest_url(manifest_path, url_or_path)
    samples, _timestamps = load_modal_jsonl_imu(raw_path)
    return samples[:, :6].astype("float32")


def _raw_path_for_manifest_url(manifest_path: Path, url_or_path: str) -> Path:
    value = str(url_or_path)
    candidate = Path(value)
    if candidate.exists():
        return candidate
    if not candidate.is_absolute() and (manifest_path.parent / candidate).exists():
        return manifest_path.parent / candidate
    data_root = manifest_path
    parts = list(manifest_path.parts)
    if "cache" in parts:
        data_root = Path(*parts[: parts.index("cache")])
    elif manifest_path.parent.name == "manifests":
        data_root = manifest_path.parent.parent.parent
    sample_id = hash_manifest_url(value)
    raw_path = data_root / "cache" / "raw" / f"{sample_id}.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing cached raw JSONL for manifest URL: {value} -> {raw_path}")
    return raw_path
