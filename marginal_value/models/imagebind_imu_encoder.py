from __future__ import annotations

"""
ImageBind IMU integration findings, verified against Meta's public ImageBind
repository on 2026-04-30:

- Repository: https://github.com/facebookresearch/ImageBind is public and its
  `imagebind_huge(pretrained=True)` entrypoint downloads a public checkpoint.
- The model defines `ModalityType.IMU` and includes a dedicated IMU branch.
- The IMU preprocessor expects raw tensors shaped `(B, 6, 2000)`: six IMU
  channels and 2000 timesteps. It unfolds the time axis with kernel/stride 8,
  producing 250 tokens where each token has 6 * 8 = 48 scalar inputs.
- The IMU trunk is a transformer branch with `imu_embed_dim=512`, six blocks,
  eight attention heads, and a 1024-dimensional shared-space projection head.
- The IMU postprocessor L2-normalizes and applies fixed logit scaling, so this
  wrapper L2-normalizes the pooled clip embedding again for cosine kNN.
- Unlike the unpublished IMU2CLIP weights, ImageBind weights are public. This is
  a sibling pretrained egocentric IMU lane, not a replacement for documenting
  the failed IMU2CLIP-weight lookup.
"""

from pathlib import Path
from typing import Protocol

import numpy as np


DEFAULT_IMAGEBIND_IMU_EXPECTED_HZ = 200
DEFAULT_IMAGEBIND_IMU_WINDOW_LEN_S = 10.0
DEFAULT_IMAGEBIND_IMU_OUTPUT_DIM = 1024
DEFAULT_IMAGEBIND_IMU_TIMESTEPS = 2000


class ImageBindIMUWindowBackend(Protocol):
    def encode_windows(self, windows: np.ndarray) -> np.ndarray:
        """Return one 1024-D embedding per `(6, 2000)` IMU window."""


class ImageBindIMUEncoder:
    """Window, normalize, and pool long raw IMU clips through ImageBind's IMU branch."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        target_hz: int = 30,
        expected_hz: int = DEFAULT_IMAGEBIND_IMU_EXPECTED_HZ,
        window_len_s: float = DEFAULT_IMAGEBIND_IMU_WINDOW_LEN_S,
        stride_ratio: float = 0.5,
        pool: str = "max",
        batch_size: int = 32,
        pretrained: bool = True,
        checkpoint_path: str | None = None,
        checkpoint_dir: str | None = None,
        backend: ImageBindIMUWindowBackend | None = None,
    ):
        self.device = str(device)
        self.target_hz = int(target_hz)
        self.expected_hz = int(expected_hz)
        self.window_len_s = float(window_len_s)
        self.stride_ratio = float(stride_ratio)
        self.pool = str(pool)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.model_window_timesteps = int(round(self.window_len_s * self.expected_hz))
        if self.model_window_timesteps != DEFAULT_IMAGEBIND_IMU_TIMESTEPS:
            raise ValueError(
                "ImageBind IMU expects 10 seconds at 200 Hz, i.e. 2000 timesteps. "
                "Use the defaults unless intentionally running an ablation."
            )
        self.backend = backend or TorchImageBindIMUBackend(
            device=self.device,
            pretrained=bool(pretrained),
            checkpoint_path=checkpoint_path,
            checkpoint_dir=checkpoint_dir,
        )

    def encode_clip(self, clip: np.ndarray) -> np.ndarray:
        prepared = self._prepare_clip(clip)
        windows = self._window_clip(
            prepared,
            window_timesteps=self.model_window_timesteps,
            stride_timesteps=max(1, int(round(self.model_window_timesteps * self.stride_ratio))),
        )
        if len(windows) == 0:
            return np.zeros(DEFAULT_IMAGEBIND_IMU_OUTPUT_DIM, dtype=np.float32)
        embeddings = []
        for start in range(0, len(windows), self.batch_size):
            batch = windows[start : start + self.batch_size]
            embeddings.append(_as_embedding_matrix(self.backend.encode_windows(batch)))
        return _l2_normalize(_pool_embeddings(np.vstack(embeddings), pool=self.pool))

    def _prepare_clip(self, clip: np.ndarray) -> np.ndarray:
        values = _finite_imu_clip(clip)
        resampled = self._resample(values, original_hz=self.target_hz, target_hz=self.expected_hz)
        return self._normalize_clip(resampled)

    def _normalize_clip(self, clip: np.ndarray) -> np.ndarray:
        values = _finite_imu_clip(clip)
        mean = np.mean(values, axis=0, keepdims=True)
        std = np.std(values, axis=0, keepdims=True)
        std = np.where(std < 1.0e-6, 1.0, std)
        return ((values - mean) / std).astype("float32")

    def _resample(self, clip: np.ndarray, *, original_hz: int, target_hz: int) -> np.ndarray:
        values = _finite_imu_clip(clip)
        if int(original_hz) == int(target_hz):
            return values.astype("float32")
        target_len = max(1, int(round(len(values) * float(target_hz) / float(original_hz))))
        try:
            from scipy.signal import resample

            return np.asarray(resample(values, target_len, axis=0), dtype="float32")
        except Exception:
            old_x = np.linspace(0.0, 1.0, num=len(values), endpoint=True)
            new_x = np.linspace(0.0, 1.0, num=target_len, endpoint=True)
            columns = [np.interp(new_x, old_x, values[:, channel]) for channel in range(values.shape[1])]
            return np.stack(columns, axis=1).astype("float32")

    def _window_clip(
        self,
        clip: np.ndarray,
        *,
        window_timesteps: int,
        stride_timesteps: int,
    ) -> np.ndarray:
        values = _finite_imu_clip(clip)
        window = int(window_timesteps)
        stride = int(stride_timesteps)
        if window <= 0 or stride <= 0:
            raise ValueError("window_timesteps and stride_timesteps must be positive.")
        if len(values) < window:
            return np.empty((0, values.shape[1], window), dtype=np.float32)
        starts = range(0, len(values) - window + 1, stride)
        return np.stack([values[start : start + window].T for start in starts]).astype("float32")


class TorchImageBindIMUBackend:
    """Tiny adapter around Meta ImageBind, imported only when this backend is used."""

    def __init__(
        self,
        *,
        device: str,
        pretrained: bool,
        checkpoint_path: str | None,
        checkpoint_dir: str | None,
    ):
        self.device = str(device)
        self.model, self.modality_key = self._load_model(
            pretrained=bool(pretrained),
            checkpoint_path=checkpoint_path,
            checkpoint_dir=checkpoint_dir,
        )

    def encode_windows(self, windows: np.ndarray) -> np.ndarray:
        import torch

        values = np.asarray(windows, dtype=np.float32)
        if values.ndim != 3 or values.shape[1:] != (6, DEFAULT_IMAGEBIND_IMU_TIMESTEPS):
            raise ValueError("ImageBind IMU windows must have shape (N, 6, 2000).")
        with torch.inference_mode():
            tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
            output = self.model({self.modality_key: tensor})[self.modality_key]
        return np.asarray(output.detach().cpu().numpy(), dtype=np.float32)

    def _load_model(
        self,
        *,
        pretrained: bool,
        checkpoint_path: str | None,
        checkpoint_dir: str | None,
    ):
        import torch
        from imagebind.models import imagebind_model

        model = imagebind_model.imagebind_huge(pretrained=False)
        if pretrained:
            resolved_checkpoint = _resolve_checkpoint_path(checkpoint_path, checkpoint_dir)
            if not resolved_checkpoint.exists():
                resolved_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.hub.download_url_to_file(
                    "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
                    str(resolved_checkpoint),
                    progress=True,
                )
            try:
                state = torch.load(str(resolved_checkpoint), map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(str(resolved_checkpoint), map_location="cpu")
            model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model, imagebind_model.ModalityType.IMU


def _resolve_checkpoint_path(checkpoint_path: str | None, checkpoint_dir: str | None) -> Path:
    if checkpoint_path is not None and str(checkpoint_path).strip():
        return Path(str(checkpoint_path))
    root = Path(str(checkpoint_dir)) if checkpoint_dir is not None and str(checkpoint_dir).strip() else Path(".checkpoints")
    return root / "imagebind_huge.pth"


def _finite_imu_clip(clip: np.ndarray) -> np.ndarray:
    values = np.asarray(clip, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("Expected raw IMU clip shaped (T, 6).")
    if len(values) == 0:
        raise ValueError("Expected at least one IMU sample.")
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def _as_embedding_matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != DEFAULT_IMAGEBIND_IMU_OUTPUT_DIM:
        raise ValueError(
            f"Expected ImageBind IMU embeddings shaped (N, {DEFAULT_IMAGEBIND_IMU_OUTPUT_DIM})."
        )
    return array.astype("float32")


def _pool_embeddings(embeddings: np.ndarray, *, pool: str) -> np.ndarray:
    values = _as_embedding_matrix(embeddings)
    if pool == "max":
        return np.max(values, axis=0).astype("float32")
    if pool == "mean":
        return np.mean(values, axis=0).astype("float32")
    raise ValueError("pool must be 'max' or 'mean'.")


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-12:
        return np.zeros_like(array, dtype=np.float32)
    return (array / norm).astype("float32")
