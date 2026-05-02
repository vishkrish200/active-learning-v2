from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


class TS2VecInference:
    """Load a TS2Vec checkpoint and emit L2-normalized clip embeddings."""

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        import torch

        from marginal_value.models.ts2vec_encoder import TS2VecEncoder

        self.device = torch.device(device)
        checkpoint = torch.load(Path(checkpoint_path), map_location=self.device, weights_only=False)
        config = dict(checkpoint.get("config", {}))
        self.model = TS2VecEncoder(
            input_dims=int(config.get("input_dims", 64)),
            hidden_dims=int(config.get("hidden_dims", 64)),
            output_dims=int(config.get("output_dims", 320)),
            n_layers=int(config.get("n_layers", 10)),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.output_dims = int(config.get("output_dims", 320))

    def encode_clip(self, clip: np.ndarray) -> np.ndarray:
        import torch

        values = _normalize_clip(clip)
        with torch.no_grad():
            tensor = torch.tensor(values[None, :, :], dtype=torch.float32, device=self.device)
            embedding = self.model(tensor).detach().cpu().numpy()[0]
        return _l2_normalize(embedding).astype("float32")

    def encode_batch(self, clips: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        if not clips:
            return np.zeros((0, self.output_dims), dtype="float32")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        parts = [
            self._encode_batch_chunk(clips[start : start + int(batch_size)])
            for start in range(0, len(clips), int(batch_size))
        ]
        return np.vstack(parts).astype("float32")

    def _encode_batch_chunk(self, clips: list[np.ndarray]) -> np.ndarray:
        import torch

        values = [_normalize_clip(clip) for clip in clips]
        lengths_np = np.asarray([len(clip) for clip in values], dtype=np.int64)
        max_len = int(np.max(lengths_np))
        padded = np.zeros((len(values), max_len, 6), dtype=np.float32)
        for idx, clip in enumerate(values):
            padded[idx, : len(clip)] = clip
        with torch.no_grad():
            tensor = torch.tensor(padded, dtype=torch.float32, device=self.device)
            lengths = torch.tensor(lengths_np, dtype=torch.long, device=self.device)
            if hasattr(self.model, "_encode_hidden") and hasattr(self.model, "aggregation_head"):
                hidden, _layers = self.model._encode_hidden(tensor)
                embedding = self._masked_aggregate_hidden(hidden, lengths)
            else:
                embedding = self.model(tensor)
            matrix = embedding.detach().cpu().numpy()
        return _l2_normalize_rows(matrix).astype("float32")

    def _masked_aggregate_hidden(self, hidden, lengths):
        import torch

        batch_size, max_len, _dims = hidden.shape
        mask = torch.arange(max_len, device=hidden.device)[None, :] < lengths[:, None]
        mask_f = mask.unsqueeze(-1).to(hidden.dtype)
        denom = lengths.to(hidden.dtype).clamp_min(1).unsqueeze(-1)
        mean = torch.sum(hidden * mask_f, dim=1) / denom
        masked_hidden = hidden.masked_fill(~mask.unsqueeze(-1), -torch.inf)
        maximum = torch.max(masked_hidden, dim=1).values
        centered = (hidden - mean.unsqueeze(1)) * mask_f
        std = torch.sqrt(torch.sum(centered * centered, dim=1) / denom)
        concat = torch.cat([mean, maximum, std], dim=-1)
        return self.model.aggregation_head.projection(concat)

    def encode_clip_multiscale(
        self,
        clip: np.ndarray,
        window_sizes: list[int] | None = None,
        stride_ratio: float = 0.5,
        pool: str = "max",
    ) -> np.ndarray:
        windows = window_sizes if window_sizes is not None else [900, 2700]
        if pool not in {"max", "mean"}:
            raise ValueError("pool must be 'max' or 'mean'.")
        values = np.asarray(clip, dtype=np.float32)
        parts = []
        for window_size in windows:
            embeddings = [self.encode_clip(window) for window in _sliding_windows(values, int(window_size), stride_ratio)]
            matrix = np.vstack(embeddings)
            if pool == "max":
                pooled = np.max(matrix, axis=0)
            else:
                pooled = np.mean(matrix, axis=0)
            parts.append(_l2_normalize(pooled))
        return _l2_normalize(np.concatenate(parts)).astype("float32")


def _normalize_clip(clip: np.ndarray) -> np.ndarray:
    values = np.asarray(clip, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("TS2Vec clips must be shaped (T, 6).")
    if values.shape[0] == 0:
        raise ValueError("TS2Vec clips must contain at least one timestep.")
    mean = float(np.mean(values))
    std = float(np.std(values))
    if not np.isfinite(std) or std < 1.0e-6:
        std = 1.0
    return np.nan_to_num((values - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm < 1.0e-12:
        return np.zeros_like(array, dtype="float32")
    return (array / norm).astype("float32")


def _l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(np.isfinite(norms) & (norms >= 1.0e-12), norms, 1.0)
    return (array / norms).astype("float32")


def _sliding_windows(values: np.ndarray, window_size: int, stride_ratio: float) -> list[np.ndarray]:
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if len(values) <= window_size:
        return [values]
    stride = max(1, int(round(window_size * float(stride_ratio))))
    starts = list(range(0, len(values) - window_size + 1, stride))
    if starts[-1] != len(values) - window_size:
        starts.append(len(values) - window_size)
    return [values[start : start + window_size] for start in starts]
