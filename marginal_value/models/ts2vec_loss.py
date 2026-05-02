from __future__ import annotations

from typing import Sequence

import numpy as np


TEMPERATURE = 0.07
MAX_TEMPORAL_POSITIONS = 512


def subsample_overlap_indices(left_indices: object, right_indices: object, *, max_positions: int = MAX_TEMPORAL_POSITIONS) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(left_indices, dtype=np.int64)
    right = np.asarray(right_indices, dtype=np.int64)
    if left.shape != right.shape:
        raise ValueError("left_indices and right_indices must have matching shapes.")
    if left.ndim != 1:
        raise ValueError("overlap indices must be 1D.")
    if len(left) <= int(max_positions):
        return left, right
    if max_positions <= 0:
        raise ValueError("max_positions must be positive.")
    positions = np.linspace(0, len(left) - 1, num=int(max_positions), dtype=np.int64)
    return left[positions], right[positions]


def create_overlapping_crops(
    x: np.ndarray,
    min_overlap: float = 0.5,
    *,
    crop_min_len: int | None = None,
    crop_max_len: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Create two random crops with a shared local overlap interval.

    The returned overlap indices are valid local indices for both crops. The
    training loop uses this as an augmentation primitive; the contrastive loss
    can then treat matching local overlap positions as positives.
    """
    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("x must be shaped (T, C).")
    if len(values) < 2:
        raise ValueError("x must contain at least two timesteps.")
    if not 0.0 < float(min_overlap) <= 1.0:
        raise ValueError("min_overlap must be in (0, 1].")
    generator = rng if rng is not None else np.random.default_rng()
    max_len = len(values) if crop_max_len is None else min(len(values), int(crop_max_len))
    min_len = max(2, int(crop_min_len) if crop_min_len is not None else max(2, int(np.ceil(max_len * min_overlap))))
    if min_len > max_len:
        min_len = max_len
    crop_len = int(generator.integers(min_len, max_len + 1))
    overlap_len = int(generator.integers(max(1, int(np.ceil(crop_len * min_overlap))), crop_len + 1))
    overlap_start = int(generator.integers(0, crop_len - overlap_len + 1))
    overlap_end = overlap_start + overlap_len
    start = int(generator.integers(0, len(values) - crop_len + 1))
    crop = values[start : start + crop_len]
    return crop.copy(), crop.copy(), overlap_start, overlap_end


def hierarchical_contrastive_loss(
    z1: Sequence[object],
    z2: Sequence[object],
    *,
    alpha: float = 0.5,
    temporal_unit: int = 0,
    temperature: float = TEMPERATURE,
    overlap_indices: Sequence[tuple[object, object]] | None = None,
    max_temporal_positions: int = MAX_TEMPORAL_POSITIONS,
):
    """Mean hierarchical instance+temporal contrastive loss over layers."""
    parts = hierarchical_contrastive_loss_parts(
        z1,
        z2,
        alpha=alpha,
        temporal_unit=temporal_unit,
        temperature=temperature,
        overlap_indices=overlap_indices,
        max_temporal_positions=max_temporal_positions,
    )
    return parts["loss"]


def hierarchical_contrastive_loss_parts(
    z1: Sequence[object],
    z2: Sequence[object],
    *,
    alpha: float = 0.5,
    temporal_unit: int = 0,
    temperature: float = TEMPERATURE,
    overlap_indices: Sequence[tuple[object, object]] | None = None,
    max_temporal_positions: int = MAX_TEMPORAL_POSITIONS,
) -> dict[str, object]:
    """Return total hierarchical contrastive loss plus component means."""
    import torch

    if len(z1) != len(z2):
        raise ValueError("z1 and z2 must contain the same number of layers.")
    if not z1:
        raise ValueError("At least one representation layer is required.")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    losses = []
    instance_losses = []
    temporal_losses = []
    for layer_index, (left, right) in enumerate(zip(z1, z2, strict=True)):
        if layer_index < int(temporal_unit):
            continue
        if left.shape != right.shape:
            raise ValueError("Per-layer TS2Vec representations must have matching shapes.")
        instance = instance_contrastive_loss(left, right, temperature=temperature)
        if float(alpha) <= 0.0:
            temporal = torch.zeros((), dtype=instance.dtype, device=instance.device)
        else:
            temporal = temporal_contrastive_loss(
                left,
                right,
                temperature=temperature,
                overlap_indices=overlap_indices,
                max_temporal_positions=max_temporal_positions,
            )
        instance_losses.append(instance)
        temporal_losses.append(temporal)
        temporal_weight = float(alpha)
        losses.append((1.0 - temporal_weight) * instance + temporal_weight * temporal)
    if not losses:
        device = z1[0].device
        zero = torch.zeros((), dtype=z1[0].dtype, device=device)
        return {"loss": zero, "instance_loss": zero, "temporal_loss": zero}
    return {
        "loss": torch.stack(losses).mean(),
        "instance_loss": torch.stack(instance_losses).mean(),
        "temporal_loss": torch.stack(temporal_losses).mean(),
    }


def instance_contrastive_loss(z1, z2, *, temperature: float = TEMPERATURE):
    import torch
    import torch.nn.functional as F

    left = F.normalize(torch.max(z1, dim=1).values, dim=-1)
    right = F.normalize(torch.max(z2, dim=1).values, dim=-1)
    logits = left @ right.T / float(temperature)
    labels = torch.arange(left.shape[0], device=left.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def temporal_contrastive_loss(
    z1,
    z2,
    *,
    temperature: float = TEMPERATURE,
    overlap_indices: Sequence[tuple[object, object]] | None = None,
    max_temporal_positions: int = MAX_TEMPORAL_POSITIONS,
):
    import torch
    import torch.nn.functional as F

    if z1.shape[1] <= 1:
        return torch.zeros((), dtype=z1.dtype, device=z1.device)
    if overlap_indices is not None and len(overlap_indices) != z1.shape[0]:
        raise ValueError("overlap_indices must contain one index pair per batch row.")
    losses = []
    for batch_index, (left_row, right_row) in enumerate(zip(z1, z2, strict=True)):
        if overlap_indices is not None:
            left_idx, right_idx = overlap_indices[batch_index]
            left_idx, right_idx = subsample_overlap_indices(left_idx, right_idx, max_positions=max_temporal_positions)
            left_idx = torch.as_tensor(left_idx, dtype=torch.long, device=z1.device)
            right_idx = torch.as_tensor(right_idx, dtype=torch.long, device=z1.device)
            if left_idx.numel() == 0 or right_idx.numel() == 0:
                continue
            if left_idx.numel() != right_idx.numel():
                raise ValueError("Each overlap index pair must have matching lengths.")
            left_row = left_row.index_select(0, left_idx)
            right_row = right_row.index_select(0, right_idx)
        left = F.normalize(left_row, dim=-1)
        right = F.normalize(right_row, dim=-1)
        logits = left @ right.T / float(temperature)
        labels = torch.arange(left.shape[0], device=left.device)
        losses.append(0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)))
    if not losses:
        return torch.zeros((), dtype=z1.dtype, device=z1.device)
    return torch.stack(losses).mean()
