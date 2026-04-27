from __future__ import annotations

import numpy as np


def robust_normalize(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples, dtype=float)
    med = np.nanmedian(array, axis=0)
    q75 = np.nanpercentile(array, 75, axis=0)
    q25 = np.nanpercentile(array, 25, axis=0)
    iqr = np.maximum(q75 - q25, 1.0e-6)
    return np.nan_to_num((array - med) / iqr)


def derive_invariant_channels(samples: np.ndarray, sample_rate: float = 30.0) -> np.ndarray:
    array = np.asarray(samples, dtype=float)
    if array.shape[1] < 6:
        raise ValueError("Need at least accel xyz and gyro xyz channels")

    acc = array[:, :3]
    gyro = array[:, 3:6]
    acc_norm = np.linalg.norm(acc, axis=1)
    gyro_norm = np.linalg.norm(gyro, axis=1)

    dt = 1.0 / sample_rate
    acc_diff = np.diff(acc, axis=0, prepend=acc[:1]) / dt
    gyro_diff = np.diff(gyro, axis=0, prepend=gyro[:1]) / dt
    jerk_norm = np.linalg.norm(acc_diff, axis=1)
    angular_jerk_norm = np.linalg.norm(gyro_diff, axis=1)

    gravity_estimate = _moving_average(acc, window=max(3, int(sample_rate)))
    linear_acc = acc - gravity_estimate
    linear_acc_norm = np.linalg.norm(linear_acc, axis=1)
    stationary_flag = ((np.abs(acc_norm - np.median(acc_norm)) < 0.05) & (gyro_norm < 0.02)).astype(
        float
    )

    return np.column_stack(
        [
            acc_norm,
            gyro_norm,
            jerk_norm,
            angular_jerk_norm,
            linear_acc_norm,
            stationary_flag,
        ]
    )


def preprocess_samples(samples: np.ndarray, sample_rate: float = 30.0) -> np.ndarray:
    array = np.asarray(samples, dtype=float)
    base = robust_normalize(array[:, :6])
    derived = robust_normalize(derive_invariant_channels(array[:, :6], sample_rate=sample_rate))
    return np.column_stack([base, derived])


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    columns = [np.convolve(values[:, idx], kernel, mode="same") for idx in range(values.shape[1])]
    return np.column_stack(columns)

