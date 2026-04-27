from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marginal_value.preprocessing.features import preprocess_samples


@dataclass
class HandcraftedIMUEncoder:
    """Deterministic baseline encoder used before SSL checkpoints exist."""

    sample_rate: float = 30.0
    embedding_dim: int = 512

    def encode_clip(self, samples: np.ndarray) -> np.ndarray:
        features = preprocess_samples(samples, sample_rate=self.sample_rate)
        vector = multiscale_summary(features)
        if len(vector) >= self.embedding_dim:
            return vector[: self.embedding_dim].astype(float)
        padded = np.zeros(self.embedding_dim, dtype=float)
        padded[: len(vector)] = vector
        return padded

    def encode_many(self, clips: list[np.ndarray]) -> np.ndarray:
        return np.vstack([self.encode_clip(clip) for clip in clips])


def multiscale_summary(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=float)
    if array.ndim != 2:
        raise ValueError("features must be [time, channels]")

    chunks = [array]
    if array.shape[0] >= 4:
        chunks.extend(np.array_split(array, 4))

    summaries: list[np.ndarray] = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        summaries.extend(
            [
                np.mean(chunk, axis=0),
                np.std(chunk, axis=0),
                np.percentile(chunk, 10, axis=0),
                np.percentile(chunk, 50, axis=0),
                np.percentile(chunk, 90, axis=0),
                _spectral_entropy(chunk),
            ]
        )
    return np.nan_to_num(np.concatenate(summaries))


def _spectral_entropy(chunk: np.ndarray) -> np.ndarray:
    if len(chunk) < 4:
        return np.zeros(chunk.shape[1], dtype=float)
    spectrum = np.abs(np.fft.rfft(chunk - np.mean(chunk, axis=0), axis=0)) ** 2
    power = spectrum / (np.sum(spectrum, axis=0, keepdims=True) + 1.0e-12)
    entropy = -np.sum(power * np.log(power + 1.0e-12), axis=0)
    normalizer = np.log(max(power.shape[0], 2))
    return entropy / normalizer

