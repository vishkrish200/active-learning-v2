from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
TIMESTAMP_CANDIDATES = ("timestamp", "time", "t", "ts")


@dataclass(frozen=True)
class WorkerIMU:
    worker_id: str
    samples: np.ndarray
    timestamps: np.ndarray | None
    source_path: Path


def infer_worker_id(path: Path) -> str:
    return path.stem


def find_timestamp_column(columns: Iterable[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in TIMESTAMP_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    return None


def find_channel_columns(frame: pd.DataFrame) -> list[str]:
    lowered = {column.lower(): column for column in frame.columns}
    if all(channel in lowered for channel in DEFAULT_CHANNELS):
        return [lowered[channel] for channel in DEFAULT_CHANNELS]

    numeric_columns = [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
        and column != find_timestamp_column(frame.columns)
    ]
    if len(numeric_columns) < 6:
        raise ValueError("IMU CSV must contain at least six numeric IMU channels")
    return numeric_columns[:6]


def load_worker_csv(path: str | Path, worker_id: str | None = None) -> WorkerIMU:
    csv_path = Path(path)
    frame = pd.read_csv(csv_path)
    channel_columns = find_channel_columns(frame)
    timestamp_column = find_timestamp_column(frame.columns)

    samples = frame[channel_columns].to_numpy(dtype=float, copy=True)
    timestamps = None
    if timestamp_column is not None:
        timestamps = frame[timestamp_column].to_numpy(dtype=float, copy=True)

    return WorkerIMU(
        worker_id=worker_id or infer_worker_id(csv_path),
        samples=samples,
        timestamps=timestamps,
        source_path=csv_path,
    )


def load_imu_directory(directory: str | Path) -> list[WorkerIMU]:
    root = Path(directory)
    paths = sorted(root.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {root}")
    return [load_worker_csv(path) for path in paths]

