from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class PrimitiveToken:
    codes: tuple[int, ...]
    duration_sec: float

    @property
    def token_id(self) -> str:
        return "_".join(str(code) for code in self.codes)


class MotionBPE:
    """Variable-duration primitive discovery with physically bounded phrases."""

    def __init__(
        self,
        *,
        num_merges: int = 8192,
        base_step_sec: float = 0.25,
        max_primitive_duration_sec: float = 12.0,
        min_count: int = 2,
    ) -> None:
        self.num_merges = num_merges
        self.base_step_sec = base_step_sec
        self.max_primitive_duration_sec = max_primitive_duration_sec
        self.min_count = min_count
        self.primitives: list[tuple[int, ...]] = []

    def fit(self, sequences: Iterable[Sequence[int]]) -> "MotionBPE":
        max_len = max(2, int(round(self.max_primitive_duration_sec / self.base_step_sec)))
        counts: Counter[tuple[int, ...]] = Counter()
        left_contexts: dict[tuple[int, ...], Counter[int | str]] = defaultdict(Counter)
        right_contexts: dict[tuple[int, ...], Counter[int | str]] = defaultdict(Counter)
        for sequence in sequences:
            seq = [int(item) for item in sequence]
            upper = min(max_len, len(seq))
            for n in range(2, upper + 1):
                for i in range(0, len(seq) - n + 1):
                    primitive = tuple(seq[i : i + n])
                    counts[primitive] += 1
                    left_contexts[primitive][seq[i - 1] if i > 0 else "<s>"] += 1
                    right_contexts[primitive][seq[i + n] if i + n < len(seq) else "</s>"] += 1

        candidates = [
            (primitive, count, left_contexts[primitive], right_contexts[primitive])
            for primitive, count in counts.items()
            if count >= self.min_count and len(primitive) * self.base_step_sec <= self.max_primitive_duration_sec
        ]
        candidates.sort(key=lambda item: self._score_candidate(item[0], item[1], item[2], item[3]), reverse=True)

        selected: list[tuple[int, ...]] = []
        for primitive, _count, _left, _right in candidates:
            if primitive in selected:
                continue
            selected.append(primitive)
            if len(selected) >= self.num_merges:
                break

        self.primitives = selected
        return self

    def encode(self, sequence: Sequence[int]) -> list[PrimitiveToken]:
        seq = [int(item) for item in sequence]
        by_length = sorted(self.primitives, key=len, reverse=True)
        encoded: list[PrimitiveToken] = []
        i = 0
        while i < len(seq):
            match = None
            for primitive in by_length:
                n = len(primitive)
                if n and tuple(seq[i : i + n]) == primitive:
                    match = primitive
                    break
            if match is None:
                match = (seq[i],)
            encoded.append(
                PrimitiveToken(
                    codes=match,
                    duration_sec=float(len(match) * self.base_step_sec),
                )
            )
            i += len(match)
        return encoded

    @staticmethod
    def _score_candidate(
        primitive: tuple[int, ...],
        count: int,
        left_context: Counter[int | str],
        right_context: Counter[int | str],
    ) -> tuple[float, int, int, tuple[int, ...]]:
        diversity_weight = 1.15 if len(set(primitive)) > 1 else 0.55
        # Reusable motion primitives should appear in more than one boundary
        # context. This avoids memorizing whole clips before learning smaller
        # reach / pause / cycle atoms.
        boundary_bonus = 1.0 + _entropy(left_context) + _entropy(right_context)
        score = float(count) * len(primitive) * diversity_weight * boundary_bonus
        return (score, len(primitive), count, tuple(-code for code in primitive))


class PatchVQTokenizer:
    """Small numpy k-means tokenizer for patch embeddings.

    This is a dependency-light stand-in for the VQ module. It is good enough for
    pipeline validation; GPU VQ training can replace it behind the same methods.
    """

    def __init__(self, codebook_size: int = 128, n_iter: int = 25, seed: int = 7) -> None:
        self.codebook_size = codebook_size
        self.n_iter = n_iter
        self.seed = seed
        self.codebook: np.ndarray | None = None

    def fit(
        self,
        patches: np.ndarray,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> "PatchVQTokenizer":
        data = np.asarray(patches, dtype=float)
        if data.ndim != 2:
            raise ValueError("patches must be [n_patches, dim]")
        if len(data) == 0:
            raise ValueError("cannot fit tokenizer on no patches")

        rng = np.random.default_rng(self.seed)
        k = min(self.codebook_size, len(data))
        centers = data[rng.choice(len(data), size=k, replace=False)].copy()
        for iteration in range(1, self.n_iter + 1):
            distances = _squared_distances(data, centers)
            labels = np.argmin(distances, axis=1)
            for idx in range(k):
                members = data[labels == idx]
                if len(members):
                    centers[idx] = np.mean(members, axis=0)
            if progress_callback is not None:
                progress_callback(iteration, self.n_iter)
        self.codebook = centers
        return self

    def encode(self, patches: np.ndarray) -> np.ndarray:
        if self.codebook is None:
            raise RuntimeError("fit must be called before encode")
        data = np.asarray(patches, dtype=float)
        return np.argmin(_squared_distances(data, self.codebook), axis=1)


def _squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.sum(left * left, axis=1, keepdims=True)
    right_norm = np.sum(right * right, axis=1, keepdims=True).T
    return np.maximum(left_norm + right_norm - 2.0 * left @ right.T, 0.0)


def _entropy(counter: Counter[int | str]) -> float:
    total = float(sum(counter.values()))
    if total <= 0:
        return 0.0
    values = np.array(list(counter.values()), dtype=float) / total
    return float(-np.sum(values * np.log(values + 1.0e-12)))
