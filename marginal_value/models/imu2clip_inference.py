from __future__ import annotations

from typing import Sequence

import numpy as np

from marginal_value.models.imu2clip_encoder import IMU2CLIPEncoder


class IMU2CLIPInference:
    """Inference wrapper with the same shape as the TS2Vec inference interface."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        *,
        device: str = "cpu",
        target_hz: int = 30,
        window_len_s: float = 2.0,
        stride_ratio: float = 0.5,
        pool: str = "max",
        expected_hz: int = 200,
        batch_size: int = 128,
        model_window_timesteps: int = 1000,
        encoder: object | None = None,
    ):
        if encoder is not None:
            self.encoder = encoder
            return
        if checkpoint_path is None or str(checkpoint_path).strip() == "":
            raise ValueError("checkpoint_path is required for IMU2CLIP inference.")
        self.encoder = IMU2CLIPEncoder(
            checkpoint_path=str(checkpoint_path),
            device=device,
            target_hz=target_hz,
            window_len_s=window_len_s,
            stride_ratio=stride_ratio,
            pool=pool,
            expected_hz=expected_hz,
            batch_size=batch_size,
            model_window_timesteps=model_window_timesteps,
        )

    def encode_clip(self, clip: np.ndarray) -> np.ndarray:
        embedding = self.encoder.encode_clip(clip)
        return _as_float32_row(embedding, expected_dim=512)

    def encode_batch(self, clips: Sequence[np.ndarray]) -> np.ndarray:
        rows = [self.encode_clip(clip) for clip in clips]
        if not rows:
            return np.empty((0, 512), dtype=np.float32)
        return np.vstack(rows).astype("float32")

    def encode_clip_multiscale(
        self,
        clip: np.ndarray,
        window_sizes_s: list[float] | None = None,
        stride_ratio: float = 0.5,
        pool: str = "max",
    ) -> np.ndarray:
        sizes = [2.0, 5.0, 10.0] if window_sizes_s is None else window_sizes_s
        embedding = self.encoder.encode_clip_multiscale(
            clip,
            window_sizes_s=sizes,
            stride_ratio=stride_ratio,
            pool=pool,
        )
        return _as_float32_row(embedding, expected_dim=512 * len(sizes))


def _as_float32_row(values: np.ndarray, *, expected_dim: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError("IMU2CLIP embeddings must be 1D arrays.")
    if array.shape[0] != expected_dim:
        raise ValueError(f"Expected IMU2CLIP embedding dimension {expected_dim}, got {array.shape[0]}.")
    norm = float(np.linalg.norm(array))
    if norm > 1.0e-12:
        array = array / norm
    return array.astype("float32")
