from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from marginal_value.logging_utils import log_event, log_progress
from marginal_value.models.grammar_lm import NGramMotionGrammar
from marginal_value.tokenization.artifacts import TokenSequence, read_token_sequences_jsonl
from marginal_value.tokenization.config import validate_grammar_config


GRAMMAR_FEATURE_COLUMNS = [
    "token_nll_mean",
    "token_nll_p90",
    "token_nll_p95",
    "transition_nll_mean",
    "transition_nll_p95",
    "rare_bigram_fraction",
    "rare_trigram_fraction",
    "rare_phrase_fraction",
    "longest_unseen_phrase_len",
]


def run_grammar_pipeline(
    config: dict[str, Any],
    *,
    token_sequence_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    validate_grammar_config(config)
    mode = "smoke" if smoke else "full"
    log_event("grammar", "start", mode=mode)

    input_path = Path(token_sequence_path) if token_sequence_path is not None else _default_token_sequence_path(config, mode)
    output = Path(output_dir) if output_dir is not None else Path(config["artifacts"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    log_event("grammar", "sequence_read_start", path=str(input_path))
    sequences = read_token_sequences_jsonl(input_path)
    log_event("grammar", "sequence_read_done", path=str(input_path), n_sequences=len(sequences))
    fit_split = str(config["splits"]["fit_split"])
    score_split = str(config["splits"]["score_split"])
    fit_sequences = [sequence for sequence in sequences if sequence.split == fit_split]
    score_sequences = [sequence for sequence in sequences if sequence.split == score_split]
    if not fit_sequences:
        raise ValueError(f"No token sequences found for grammar fit_split '{fit_split}'.")
    if not score_sequences:
        raise ValueError(f"No token sequences found for grammar score_split '{score_split}'.")

    fit_counts = Counter(sequence.split for sequence in fit_sequences)
    score_counts = Counter(sequence.split for sequence in score_sequences)
    if set(fit_counts) != {fit_split}:
        raise RuntimeError(f"Grammar fit received non-fit splits: {dict(fit_counts)}")
    if set(score_counts) != {score_split}:
        raise RuntimeError(f"Grammar scoring received unexpected splits: {dict(score_counts)}")

    grammar_config = config["grammar"]
    grammar = NGramMotionGrammar(
        order=int(grammar_config["order"]),
        smoothing=float(grammar_config["smoothing"]),
        rare_threshold=int(grammar_config["rare_threshold"]),
    )
    log_event("grammar", "fit_start", n_sequences=len(fit_sequences), split=fit_split)
    grammar.fit([sequence.primitive_token_ids for sequence in fit_sequences])
    log_event("grammar", "fit_done", vocabulary_size=len(grammar.vocabulary), ngram_count=sum(grammar.ngram_counts.values()))

    rows = _score_sequences(score_sequences, grammar)
    feature_path = output / f"grammar_features_{score_split}_{mode}.csv"
    report_path = output / f"grammar_report_{mode}.json"
    _write_feature_csv(feature_path, rows)
    log_event("grammar", "feature_write_done", feature_path=str(feature_path), n_rows=len(rows))

    report = {
        "mode": mode,
        "fit_split": fit_split,
        "score_split": score_split,
        "fit_sequence_split_counts": dict(fit_counts),
        "scored_sequence_split_counts": dict(score_counts),
        "n_fit_sequences": len(fit_sequences),
        "n_scored_sequences": len(score_sequences),
        "vocabulary_size": len(grammar.vocabulary),
        "ngram_count": int(sum(grammar.ngram_counts.values())),
        "feature_summary": _feature_summary(rows),
        "leakage_audit": {
            "grammar_fit_splits": sorted(fit_counts),
            "scored_splits": sorted(score_counts),
        },
        "artifacts": {
            "features": str(feature_path),
            "report": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    result = {
        "mode": mode,
        "fit_split": fit_split,
        "score_split": score_split,
        "fit_sequence_split_counts": dict(fit_counts),
        "scored_sequence_split_counts": dict(score_counts),
        "feature_path": str(feature_path),
        "report_path": str(report_path),
    }
    log_event("grammar", "done", **result)
    return result


def _default_token_sequence_path(config: dict[str, Any], mode: str) -> Path:
    return Path(config["tokens"]["input_dir"]) / f"token_sequences_{mode}.jsonl"


def _feature_row(sequence: TokenSequence, grammar: NGramMotionGrammar) -> dict[str, object]:
    features = grammar.sequence_features(sequence.primitive_token_ids)
    row: dict[str, object] = {
        "worker_id": sequence.sample_id,
        "sample_id": sequence.sample_id,
        "split": sequence.split,
        "n_primitives": len(sequence.primitive_token_ids),
    }
    row.update(features)
    return row


def _score_sequences(sequences: Sequence[TokenSequence], grammar: NGramMotionGrammar) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    total = len(sequences)
    progress_every = max(1, total // 10) if total else 1
    log_event("grammar", "score_sequence_start", n_sequences=total)
    for index, sequence in enumerate(sequences, start=1):
        rows.append(_feature_row(sequence, grammar))
        log_progress(
            "grammar",
            "score_sequence_progress",
            index=index,
            total=total,
            every=progress_every,
            split=sequence.split,
        )
    log_event("grammar", "score_sequence_done", n_sequences=total)
    return rows


def _write_feature_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = ["worker_id", "sample_id", "split", "n_primitives"] + GRAMMAR_FEATURE_COLUMNS
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _feature_summary(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for column in GRAMMAR_FEATURE_COLUMNS:
        values = np.asarray([float(row[column]) for row in rows], dtype=float)
        summary[column] = {
            "mean": float(np.mean(values)) if values.size else 0.0,
            "p50": float(np.percentile(values, 50)) if values.size else 0.0,
            "p95": float(np.percentile(values, 95)) if values.size else 0.0,
        }
    return summary
