from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Sequence

import numpy as np


class NGramMotionGrammar:
    """Smoothed n-gram language model over motion primitive token ids."""

    def __init__(self, order: int = 3, smoothing: float = 0.1, rare_threshold: int = 0) -> None:
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.smoothing = smoothing
        self.rare_threshold = rare_threshold
        self.vocabulary: set[int | str] = set()
        self.context_counts: dict[tuple[int | str, ...], Counter[int | str]] = defaultdict(Counter)
        self.ngram_counts: Counter[tuple[int | str, ...]] = Counter()

    def fit(self, sequences: Iterable[Sequence[int | str]]) -> "NGramMotionGrammar":
        for sequence in sequences:
            tokens = list(sequence)
            self.vocabulary.update(tokens)
            padded = ["<s>"] * (self.order - 1) + tokens
            for index in range(self.order - 1, len(padded)):
                context = tuple(padded[index - self.order + 1 : index]) if self.order > 1 else tuple()
                token = padded[index]
                self.context_counts[context][token] += 1
                ngram = context + (token,)
                self.ngram_counts[ngram] += 1
        return self

    def sequence_features(self, sequence: Sequence[int | str]) -> dict[str, float]:
        nll_values = self.token_nlls(sequence)
        if not nll_values:
            return {
                "token_nll_mean": 0.0,
                "token_nll_p90": 0.0,
                "token_nll_p95": 0.0,
                "transition_nll_mean": 0.0,
                "transition_nll_p95": 0.0,
                "rare_bigram_fraction": 0.0,
                "rare_trigram_fraction": 0.0,
                "rare_phrase_fraction": 0.0,
                "longest_unseen_phrase_len": 0.0,
            }

        return {
            "token_nll_mean": float(np.mean(nll_values)),
            "token_nll_p90": float(np.percentile(nll_values, 90)),
            "token_nll_p95": float(np.percentile(nll_values, 95)),
            "transition_nll_mean": float(np.mean(nll_values[1:])) if len(nll_values) > 1 else 0.0,
            "transition_nll_p95": float(np.percentile(nll_values[1:], 95)) if len(nll_values) > 1 else 0.0,
            "rare_bigram_fraction": self._rare_ngram_fraction(sequence, 2),
            "rare_trigram_fraction": self._rare_ngram_fraction(sequence, 3),
            "rare_phrase_fraction": self._rare_ngram_fraction(sequence, min(self.order, 3)),
            "longest_unseen_phrase_len": float(self._longest_unseen_phrase(sequence)),
        }

    def token_nlls(self, sequence: Sequence[int | str]) -> list[float]:
        vocab_size = max(len(self.vocabulary), 1)
        tokens = list(sequence)
        padded = ["<s>"] * (self.order - 1) + tokens
        nlls: list[float] = []
        for index in range(self.order - 1, len(padded)):
            context = tuple(padded[index - self.order + 1 : index]) if self.order > 1 else tuple()
            token = padded[index]
            counts = self.context_counts.get(context, Counter())
            numerator = counts.get(token, 0.0) + self.smoothing
            denominator = sum(counts.values()) + self.smoothing * vocab_size
            probability = numerator / max(denominator, 1.0e-12)
            nlls.append(float(-np.log(probability)))
        return nlls

    def _rare_ngram_fraction(self, sequence: Sequence[int | str], n: int) -> float:
        if n <= 0 or len(sequence) < n:
            return 0.0
        total = 0
        rare = 0
        tokens = list(sequence)
        for i in range(0, len(tokens) - n + 1):
            total += 1
            if self.ngram_counts.get(tuple(tokens[i : i + n]), 0) <= self.rare_threshold:
                rare += 1
        return float(rare / total) if total else 0.0

    def _longest_unseen_phrase(self, sequence: Sequence[int | str]) -> int:
        tokens = list(sequence)
        longest = 0
        max_n = min(len(tokens), self.order + 2)
        for n in range(1, max_n + 1):
            for i in range(0, len(tokens) - n + 1):
                if self.ngram_counts.get(tuple(tokens[i : i + n]), 0) == 0:
                    longest = max(longest, n)
        return longest

