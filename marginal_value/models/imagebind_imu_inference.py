from __future__ import annotations

from typing import Sequence

import numpy as np

from marginal_value.models.imagebind_imu_encoder import (
    DEFAULT_IMAGEBIND_IMU_OUTPUT_DIM,
    ImageBindIMUEncoder,
)


class ImageBindIMUInference:
    """Inference wrapper shaped like the existing encoder interfaces."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        target_hz: int = 30,
        expected_hz: int = 200,
        window_len_s: float = 10.0,
        stride_ratio: float = 0.5,
        pool: str = "max",
        batch_size: int = 32,
        pretrained: bool = True,
        checkpoint_path: str | None = None,
        checkpoint_dir: str | None = None,
        encoder: object | None = None,
    ):
        if encoder is not None:
            self.encoder = encoder
            return
        self.encoder = ImageBindIMUEncoder(
            device=device,
            target_hz=target_hz,
            expected_hz=expected_hz,
            window_len_s=window_len_s,
            stride_ratio=stride_ratio,
            pool=pool,
            batch_size=batch_size,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            checkpoint_dir=checkpoint_dir,
        )

    def encode_clip(self, clip: np.ndarray) -> np.ndarray:
        return _as_float32_row(self.encoder.encode_clip(clip))

    def encode_batch(self, clips: Sequence[np.ndarray]) -> np.ndarray:
        rows = [self.encode_clip(clip) for clip in clips]
        if not rows:
            return np.empty((0, DEFAULT_IMAGEBIND_IMU_OUTPUT_DIM), dtype=np.float32)
        return np.vstack(rows).astype("float32")


def _as_float32_row(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError("ImageBind IMU embeddings must be 1D arrays.")
    if array.shape[0] != DEFAULT_IMAGEBIND_IMU_OUTPUT_DIM:
        raise ValueError(
            f"Expected ImageBind IMU embedding dimension {DEFAULT_IMAGEBIND_IMU_OUTPUT_DIM}, got {array.shape[0]}."
        )
    norm = float(np.linalg.norm(array))
    if norm > 1.0e-12:
        array = array / norm
    return array.astype("float32")
