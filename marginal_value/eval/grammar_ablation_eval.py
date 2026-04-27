from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from marginal_value.eval.ablation_eval import summarize_ranked_scores
from marginal_value.indexing.cluster_features import kmeans
from marginal_value.logging_utils import log_event
from marginal_value.models.grammar_lm import NGramMotionGrammar
from marginal_value.tokenization.artifacts import TokenSequence, read_token_sequences_jsonl


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


def run_grammar_ablation(
    config: dict[str, Any],
    *,
    token_sequence_path: str | Path | None = None,
    candidate_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    validate_grammar_ablation_config(config)
    mode = "smoke" if smoke else "full"
    artifacts = config["artifacts"]
    token_path = Path(token_sequence_path) if token_sequence_path is not None else Path(artifacts["tokens_dir"]) / f"token_sequences_{mode}.jsonl"
    candidates_path = (
        Path(candidate_path)
        if candidate_path is not None
        else Path(artifacts["ranking_dir"]) / f"baseline_ranking_val_candidates_{mode}.csv"
    )
    output = Path(output_dir) if output_dir is not None else Path(artifacts["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    log_event(
        "grammar_ablation",
        "start",
        mode=mode,
        token_sequence_path=str(token_path),
        candidate_path=str(candidates_path),
        output_dir=str(output),
    )

    log_event("grammar_ablation", "sequence_read_start", path=str(token_path))
    sequences = read_token_sequences_jsonl(token_path)
    log_event("grammar_ablation", "sequence_read_done", n_sequences=len(sequences))
    candidates = _read_csv(candidates_path)
    log_event("grammar_ablation", "candidate_read_done", n_candidates=len(candidates))

    grammar_config = config["grammar"]
    fit_split = str(grammar_config["fit_split"])
    fit_sequences = [sequence for sequence in sequences if sequence.split == fit_split]
    fit_counts = Counter(sequence.split for sequence in fit_sequences)
    if set(fit_counts) != {fit_split}:
        raise RuntimeError(f"Grammar ablation fit received non-fit splits: {dict(fit_counts)}")
    grammar = NGramMotionGrammar(
        order=int(grammar_config["order"]),
        smoothing=float(grammar_config["smoothing"]),
        rare_threshold=int(grammar_config["rare_threshold"]),
    )
    log_event("grammar_ablation", "fit_start", split=fit_split, n_sequences=len(fit_sequences))
    grammar.fit([sequence.primitive_token_ids for sequence in fit_sequences])
    log_event("grammar_ablation", "fit_done", vocabulary_size=len(grammar.vocabulary), ngram_count=sum(grammar.ngram_counts.values()))

    sequence_by_id = {sequence.sample_id: sequence for sequence in sequences}
    scored_rows, missing_ids = _score_candidate_rows(candidates, sequence_by_id, grammar)
    log_event(
        "grammar_ablation",
        "candidate_score_done",
        n_scored=len(scored_rows),
        n_missing=len(missing_ids),
    )

    k_values = [int(k) for k in config.get("eval", {}).get("k_values", [10, 50, 100, 200])]
    low_quality_threshold = float(config.get("eval", {}).get("low_quality_threshold", 0.45))
    variants = _evaluate_variants(scored_rows, k_values=k_values, low_quality_threshold=low_quality_threshold)
    best_name, best_metrics = _best_variant(variants)

    suffix = mode
    report_path = output / f"grammar_ablation_report_{suffix}.json"
    scores_path = output / f"grammar_ablation_scores_{suffix}.csv"
    summary_path = output / f"grammar_ablation_summary_{suffix}.csv"

    report = {
        "mode": mode,
        "fit_split": fit_split,
        "candidate_path": str(candidates_path),
        "token_sequence_path": str(token_path),
        "candidate_coverage": {
            "n_candidates": len(candidates),
            "scored_candidate_count": len(scored_rows),
            "missing_token_sequence_count": len(missing_ids),
            "inherited_token_sequence_count": sum(
                1 for row in scored_rows if str(row.get("token_source_sample_id", "")) != str(row.get("sample_id", ""))
            ),
            "missing_sample_ids": missing_ids[:25],
            "candidate_split_counts": dict(Counter(str(row.get("split", "")) for row in candidates)),
            "scored_candidate_split_counts": dict(Counter(str(row.get("split", "")) for row in scored_rows)),
        },
        "leakage_audit": {
            "grammar_fit_splits": sorted(fit_counts),
            "scored_candidate_splits": sorted({str(row.get("split", "")) for row in scored_rows if str(row.get("split", ""))}),
        },
        "interpretation_warnings": _interpretation_warnings(scored_rows, fit_split=fit_split),
        "grammar": {
            "order": int(grammar_config["order"]),
            "smoothing": float(grammar_config["smoothing"]),
            "rare_threshold": int(grammar_config["rare_threshold"]),
            "vocabulary_size": len(grammar.vocabulary),
            "ngram_count": int(sum(grammar.ngram_counts.values())),
        },
        "variants": variants,
        "best_by_ndcg100": best_name,
        "best_ndcg100": _primary_ndcg(best_metrics.get("metrics", {})),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_rows(scores_path, scored_rows)
    _write_summary_csv(summary_path, variants)
    log_event(
        "grammar_ablation",
        "artifacts_written",
        report_path=str(report_path),
        scores_path=str(scores_path),
        summary_path=str(summary_path),
    )

    result = {
        "mode": mode,
        "fit_split": fit_split,
        "n_candidates": len(candidates),
        "scored_candidate_count": len(scored_rows),
        "missing_token_sequence_count": len(missing_ids),
        "best_by_ndcg100": best_name,
        "best_ndcg100": report["best_ndcg100"],
        "report_path": str(report_path),
        "scores_path": str(scores_path),
        "summary_path": str(summary_path),
    }
    log_event("grammar_ablation", "done", **result)
    return result


def run_leave_cluster_out_ablation(
    config: dict[str, Any],
    *,
    token_sequence_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    validate_grammar_ablation_config(config)
    mode = "smoke" if smoke else "full"
    artifacts = config["artifacts"]
    token_path = Path(token_sequence_path) if token_sequence_path is not None else Path(artifacts["tokens_dir"]) / f"token_sequences_{mode}.jsonl"
    output = Path(output_dir) if output_dir is not None else Path(artifacts["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    log_event(
        "grammar_cluster_ablation",
        "start",
        mode=mode,
        token_sequence_path=str(token_path),
        output_dir=str(output),
    )

    sequences = read_token_sequences_jsonl(token_path)
    grammar_config = config["grammar"]
    fit_split = str(grammar_config["fit_split"])
    source_sequences = [sequence for sequence in sequences if sequence.split == fit_split]
    if not source_sequences:
        raise ValueError(f"No token sequences found for fit_split '{fit_split}'.")

    cluster_config = config.get("cluster_ablation", {})
    n_clusters = int(cluster_config.get("n_clusters", 12 if not smoke else 4))
    n_folds = int(cluster_config.get("n_folds", min(n_clusters, 6 if not smoke else 2)))
    negative_sample_size = int(cluster_config.get("negative_sample_size", 500 if not smoke else 64))
    seed = int(cluster_config.get("seed", 23))
    k_values = [int(k) for k in config.get("eval", {}).get("k_values", [10, 50, 100, 200])]
    low_quality_threshold = float(config.get("eval", {}).get("low_quality_threshold", 0.45))

    vectors, vocabulary = _token_histogram_vectors(source_sequences)
    labels, _centers = kmeans(vectors, n_clusters=n_clusters, seed=seed)
    cluster_sizes = Counter(int(label) for label in labels.tolist())
    heldout_clusters = [
        cluster
        for cluster, _size in sorted(cluster_sizes.items(), key=lambda item: (-item[1], item[0]))[: max(1, min(n_folds, len(cluster_sizes)))]
    ]
    log_event(
        "grammar_cluster_ablation",
        "clusters_ready",
        n_sequences=len(source_sequences),
        n_clusters=len(cluster_sizes),
        heldout_clusters=heldout_clusters,
        vocabulary_size=len(vocabulary),
    )

    all_scores: list[dict[str, object]] = []
    fold_reports: list[dict[str, Any]] = []
    for fold_index, heldout_cluster in enumerate(heldout_clusters, start=1):
        fold = _leave_cluster_out_fold(
            source_sequences,
            labels=labels,
            heldout_cluster=heldout_cluster,
            fold_index=fold_index,
            grammar_config=grammar_config,
            k_values=k_values,
            low_quality_threshold=low_quality_threshold,
            negative_sample_size=negative_sample_size,
            seed=seed,
        )
        fold_reports.append(fold["report"])
        all_scores.extend(fold["rows"])
        log_event(
            "grammar_cluster_ablation",
            "fold_done",
            fold_index=fold_index,
            heldout_cluster=int(heldout_cluster),
            n_positive=fold["report"]["n_positive"],
            n_negative=fold["report"]["n_negative"],
        )

    aggregate_variants = _aggregate_fold_variants(fold_reports)
    best_name, best_metrics = _best_variant(aggregate_variants)
    suffix = mode
    report_path = output / f"grammar_leave_cluster_ablation_report_{suffix}.json"
    scores_path = output / f"grammar_leave_cluster_ablation_scores_{suffix}.csv"
    summary_path = output / f"grammar_leave_cluster_ablation_summary_{suffix}.csv"
    report = {
        "mode": mode,
        "fit_split": fit_split,
        "token_sequence_path": str(token_path),
        "source_sequence_count": len(source_sequences),
        "cluster_config": {
            "n_clusters": n_clusters,
            "n_folds": len(heldout_clusters),
            "negative_sample_size": negative_sample_size,
            "seed": seed,
        },
        "cluster_sizes": {str(cluster): int(size) for cluster, size in sorted(cluster_sizes.items())},
        "heldout_clusters": [int(cluster) for cluster in heldout_clusters],
        "leakage_audit": {
            "source_splits": sorted({sequence.split for sequence in source_sequences}),
            "heldout_fit_overlap_count": int(sum(fold["heldout_fit_overlap_count"] for fold in fold_reports)),
        },
        "folds": fold_reports,
        "aggregate_variants": aggregate_variants,
        "best_by_ndcg100": best_name,
        "best_ndcg100": _primary_ndcg(best_metrics.get("metrics", {})),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_rows(scores_path, all_scores)
    _write_summary_csv(summary_path, aggregate_variants)
    log_event(
        "grammar_cluster_ablation",
        "artifacts_written",
        report_path=str(report_path),
        scores_path=str(scores_path),
        summary_path=str(summary_path),
    )
    result = {
        "mode": mode,
        "fit_split": fit_split,
        "n_folds": len(heldout_clusters),
        "source_sequence_count": len(source_sequences),
        "best_by_ndcg100": best_name,
        "best_ndcg100": report["best_ndcg100"],
        "report_path": str(report_path),
        "scores_path": str(scores_path),
        "summary_path": str(summary_path),
    }
    log_event("grammar_cluster_ablation", "done", **result)
    return result


def load_grammar_ablation_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_grammar_ablation_config(config: dict[str, Any]) -> None:
    execution = _required_mapping(config, "execution", "Grammar ablation config")
    artifacts = _required_mapping(config, "artifacts", "Grammar ablation config")
    grammar = _required_mapping(config, "grammar", "Grammar ablation config")
    if execution.get("provider") != "modal":
        raise ValueError("Grammar ablation must run on Modal for real artifacts.")
    if not execution.get("artifacts_volume"):
        raise ValueError("Grammar ablation execution.artifacts_volume must be provided.")
    _positive_int(execution, "timeout_seconds", "Grammar ablation execution")
    for key in ("ranking_dir", "tokens_dir", "output_dir"):
        if not str(artifacts.get(key, "")).startswith("/artifacts") and not Path(str(artifacts.get(key, ""))).is_absolute():
            raise ValueError(f"Grammar ablation artifacts.{key} must be an absolute path.")
    if grammar.get("fit_split") != "pretrain":
        raise ValueError("Grammar ablation grammar.fit_split must be 'pretrain'.")
    _positive_int(grammar, "order", "Grammar ablation grammar")
    _positive_float(grammar, "smoothing", "Grammar ablation grammar")
    _non_negative_int(grammar, "rare_threshold", "Grammar ablation grammar")


def _score_candidate_rows(
    candidates: Sequence[dict[str, str]],
    sequence_by_id: dict[str, TokenSequence],
    grammar: NGramMotionGrammar,
) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    missing_ids: list[str] = []
    for candidate in candidates:
        sample_id = str(candidate.get("sample_id") or candidate.get("worker_id", ""))
        sequence, token_source_sample_id = _candidate_sequence(candidate, sequence_by_id)
        if sequence is None:
            missing_ids.append(sample_id)
            continue
        row: dict[str, object] = dict(candidate)
        row["sample_id"] = sample_id
        row.setdefault("worker_id", sample_id)
        row["token_source_sample_id"] = token_source_sample_id
        row["token_split"] = sequence.split
        row["n_primitives"] = len(sequence.primitive_token_ids)
        row.update(grammar.sequence_features(sequence.primitive_token_ids))
        rows.append(row)
    return rows, missing_ids


def _candidate_sequence(
    candidate: dict[str, str],
    sequence_by_id: dict[str, TokenSequence],
) -> tuple[TokenSequence | None, str]:
    for sample_id in _candidate_sequence_ids(candidate):
        sequence = sequence_by_id.get(sample_id)
        if sequence is not None:
            return sequence, sample_id
    return None, ""


def _candidate_sequence_ids(candidate: dict[str, str]) -> list[str]:
    ids: list[str] = []
    for key in ("sample_id", "worker_id", "source_sample_id"):
        value = str(candidate.get(key, "")).strip()
        if value and value not in ids:
            ids.append(value)
    sample_id = str(candidate.get("sample_id") or candidate.get("worker_id", "")).strip()
    if "__corrupt_" in sample_id:
        source_id = sample_id.split("__corrupt_", 1)[0]
        if source_id and source_id not in ids:
            ids.append(source_id)
    return ids


def _leave_cluster_out_fold(
    sequences: Sequence[TokenSequence],
    *,
    labels: np.ndarray,
    heldout_cluster: int,
    fold_index: int,
    grammar_config: dict[str, Any],
    k_values: Iterable[int],
    low_quality_threshold: float,
    negative_sample_size: int,
    seed: int,
) -> dict[str, Any]:
    heldout_indices = [idx for idx, label in enumerate(labels.tolist()) if int(label) == int(heldout_cluster)]
    fit_indices = [idx for idx in range(len(sequences)) if idx not in set(heldout_indices)]
    rng = np.random.default_rng(seed + fold_index)
    negative_indices = list(fit_indices)
    if len(negative_indices) > negative_sample_size:
        negative_indices = [int(idx) for idx in rng.choice(negative_indices, size=negative_sample_size, replace=False)]
    grammar = NGramMotionGrammar(
        order=int(grammar_config["order"]),
        smoothing=float(grammar_config["smoothing"]),
        rare_threshold=int(grammar_config["rare_threshold"]),
    )
    grammar.fit([sequences[idx].primitive_token_ids for idx in fit_indices])

    rows: list[dict[str, object]] = []
    for label, indices in ((1, heldout_indices), (0, negative_indices)):
        for idx in indices:
            sequence = sequences[idx]
            row: dict[str, object] = {
                "fold_index": fold_index,
                "sample_id": sequence.sample_id,
                "worker_id": sequence.sample_id,
                "split": sequence.split,
                "label": label,
                "heldout_cluster": int(heldout_cluster),
                "source_cluster": int(labels[idx]),
                "new_cluster_id": int(labels[idx]),
                "quality_score": sequence.quality_score if sequence.quality_score is not None else 1.0,
                "final_score": 0.0,
                "old_novelty_score": 1.0 if label else 0.0,
                "new_density_score": 0.0,
                "n_primitives": len(sequence.primitive_token_ids),
            }
            row.update(grammar.sequence_features(sequence.primitive_token_ids))
            rows.append(row)
    variants = _evaluate_lco_variants(rows, k_values=k_values, low_quality_threshold=low_quality_threshold)
    heldout_set = {sequences[idx].sample_id for idx in heldout_indices}
    fit_set = {sequences[idx].sample_id for idx in fit_indices}
    report = {
        "fold_index": fold_index,
        "heldout_cluster": int(heldout_cluster),
        "n_positive": len(heldout_indices),
        "n_negative": len(negative_indices),
        "n_fit_sequences": len(fit_indices),
        "heldout_fit_overlap_count": len(heldout_set & fit_set),
        "variants": variants,
    }
    return {"report": report, "rows": rows}


def _token_histogram_vectors(sequences: Sequence[TokenSequence]) -> tuple[np.ndarray, list[str]]:
    counts = Counter(token for sequence in sequences for token in sequence.primitive_token_ids)
    vocabulary = [token for token, _count in counts.most_common(512)]
    if not vocabulary:
        raise ValueError("Cannot cluster empty token sequences.")
    token_to_idx = {token: idx for idx, token in enumerate(vocabulary)}
    vectors = np.zeros((len(sequences), len(vocabulary)), dtype=float)
    for row_idx, sequence in enumerate(sequences):
        sequence_counts = Counter(sequence.primitive_token_ids)
        total = max(1, sum(sequence_counts.values()))
        for token, count in sequence_counts.items():
            col_idx = token_to_idx.get(token)
            if col_idx is not None:
                vectors[row_idx, col_idx] = count / total
    return vectors, vocabulary


def _aggregate_fold_variants(folds: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for fold in folds:
        for name, value in fold.get("variants", {}).items():
            by_variant.setdefault(name, []).append(value)
    aggregate: dict[str, dict[str, Any]] = {}
    for name, values in by_variant.items():
        metric_keys = sorted({key for value in values for key in value.get("metrics", {})})
        metrics = {
            key: float(np.mean([float(value.get("metrics", {}).get(key, 0.0)) for value in values]))
            for key in metric_keys
        }
        aggregate[name] = {
            "eligible": True,
            "n_folds": len(values),
            "metrics": metrics,
        }
    return aggregate


def _evaluate_variants(
    rows: Sequence[dict[str, object]],
    *,
    k_values: Iterable[int],
    low_quality_threshold: float,
) -> dict[str, dict[str, Any]]:
    if not rows:
        return {}
    labels = np.asarray([_to_int(row.get("label", 0)) for row in rows], dtype=int)
    clusters = np.asarray([_to_int(row.get("new_cluster_id", idx)) for idx, row in enumerate(rows)], dtype=int)
    score_inputs = _variant_scores(rows)
    variants: dict[str, dict[str, Any]] = {}
    for name, scores in score_inputs.items():
        metrics = summarize_ranked_scores(scores=scores, labels=labels, clusters=clusters, k_values=k_values)
        ranked_indices = np.argsort(-scores, kind="mergesort")
        variants[name] = {
            "eligible": True,
            "metrics": metrics,
            "quality": _top_quality_summary(rows, ranked_indices, k_values, low_quality_threshold=low_quality_threshold),
            "top_reason_code_counts": _top_reason_counts(rows, ranked_indices, k_values),
        }
    return variants


def _evaluate_lco_variants(
    rows: Sequence[dict[str, object]],
    *,
    k_values: Iterable[int],
    low_quality_threshold: float,
) -> dict[str, dict[str, Any]]:
    if not rows:
        return {}
    labels = np.asarray([_to_int(row.get("label", 0)) for row in rows], dtype=int)
    clusters = np.asarray([_to_int(row.get("new_cluster_id", idx)) for idx, row in enumerate(rows)], dtype=int)
    score_inputs = _lco_variant_scores(rows)
    variants: dict[str, dict[str, Any]] = {}
    for name, scores in score_inputs.items():
        metrics = summarize_ranked_scores(scores=scores, labels=labels, clusters=clusters, k_values=k_values)
        ranked_indices = np.argsort(-scores, kind="mergesort")
        variants[name] = {
            "eligible": True,
            "metrics": metrics,
            "quality": _top_quality_summary(rows, ranked_indices, k_values, low_quality_threshold=low_quality_threshold),
            "top_reason_code_counts": _top_reason_counts(rows, ranked_indices, k_values),
        }
    return variants


def _interpretation_warnings(rows: Sequence[dict[str, object]], *, fit_split: str) -> list[str]:
    warnings: list[str] = []
    has_fit_split_negatives = any(
        _to_int(row.get("label", 0)) == 0 and str(row.get("split", "")) == fit_split
        for row in rows
    )
    if has_fit_split_negatives:
        warnings.append("negative_candidates_include_fit_split")
    if has_fit_split_negatives and any(_to_int(row.get("label", 0)) == 1 for row in rows):
        warnings.append("grammar_metrics_may_reflect_pretrain_vs_val_split")
    return warnings


def _variant_scores(rows: Sequence[dict[str, object]]) -> dict[str, np.ndarray]:
    final_score = _column(rows, "final_score")
    old_novelty = _column(rows, "old_novelty_score")
    new_density = _column(rows, "new_density_score")
    quality = _column(rows, "quality_score", default=1.0)
    token_nll = _column(rows, "token_nll_p95")
    transition_nll = _column(rows, "transition_nll_p95")
    rare_phrase = _column(rows, "rare_phrase_fraction")
    longest_unseen = _column(rows, "longest_unseen_phrase_len")
    grammar_surprisal = _normalize(0.65 * _normalize(token_nll) + 0.25 * _normalize(transition_nll) + 0.10 * _normalize(longest_unseen))
    grammar_phrase = _normalize(0.70 * _normalize(token_nll) + 0.30 * _normalize(rare_phrase))
    phase_a = _normalize(final_score)
    return {
        "phase_a_final_score": final_score,
        "old_novelty_score": old_novelty,
        "new_density_score": new_density,
        "grammar_token_nll_p95": token_nll,
        "grammar_token_quality_gated": _normalize(token_nll) * quality,
        "grammar_surprisal_mix": grammar_surprisal,
        "grammar_phrase_mix": grammar_phrase,
        "phase_a_plus_grammar_10pct": 0.90 * phase_a + 0.10 * grammar_surprisal,
        "phase_a_plus_grammar_25pct": 0.75 * phase_a + 0.25 * grammar_surprisal,
    }


def _lco_variant_scores(rows: Sequence[dict[str, object]]) -> dict[str, np.ndarray]:
    quality = _column(rows, "quality_score", default=1.0)
    token_nll = _column(rows, "token_nll_p95")
    transition_nll = _column(rows, "transition_nll_p95")
    rare_phrase = _column(rows, "rare_phrase_fraction")
    longest_unseen = _column(rows, "longest_unseen_phrase_len")
    grammar_surprisal = _normalize(0.65 * _normalize(token_nll) + 0.25 * _normalize(transition_nll) + 0.10 * _normalize(longest_unseen))
    grammar_phrase = _normalize(0.70 * _normalize(token_nll) + 0.30 * _normalize(rare_phrase))
    return {
        "random_baseline": np.asarray([_stable_random_score(row) for row in rows], dtype=float),
        "grammar_token_nll_p95": token_nll,
        "grammar_token_quality_gated": _normalize(token_nll) * quality,
        "grammar_surprisal_mix": grammar_surprisal,
        "grammar_phrase_mix": grammar_phrase,
    }


def _stable_random_score(row: dict[str, object]) -> float:
    key = f"{row.get('fold_index', '')}:{row.get('sample_id', '')}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def _top_quality_summary(
    rows: Sequence[dict[str, object]],
    ranked_indices: np.ndarray,
    k_values: Iterable[int],
    *,
    low_quality_threshold: float,
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for k in k_values:
        indices = ranked_indices[: min(int(k), len(ranked_indices))]
        qualities = [_to_float(rows[int(idx)].get("quality_score", 1.0), default=1.0) for idx in indices]
        summary[str(k)] = {
            "mean_quality": float(np.mean(qualities)) if qualities else 0.0,
            "low_quality_count": int(sum(value < low_quality_threshold for value in qualities)),
        }
    return summary


def _top_reason_counts(
    rows: Sequence[dict[str, object]],
    ranked_indices: np.ndarray,
    k_values: Iterable[int],
) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for k in k_values:
        indices = ranked_indices[: min(int(k), len(ranked_indices))]
        output[str(k)] = dict(Counter(str(rows[int(idx)].get("reason_code", "")) for idx in indices if rows[int(idx)].get("reason_code", "")))
    return output


def _best_variant(variants: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    if not variants:
        return "", {}
    return max(
        variants.items(),
        key=lambda item: _primary_ndcg(item[1].get("metrics", {})),
    )


def _primary_ndcg(metrics: dict[str, float]) -> float:
    if "ndcg@100" in metrics:
        return float(metrics["ndcg@100"])
    ndcg_items = [
        (int(key.split("@", 1)[1]), float(value))
        for key, value in metrics.items()
        if key.startswith("ndcg@") and key.split("@", 1)[1].isdigit()
    ]
    if not ndcg_items:
        return 0.0
    return sorted(ndcg_items)[-1][1]


def _write_summary_csv(path: Path, variants: dict[str, dict[str, Any]]) -> None:
    rows: list[dict[str, object]] = []
    for name, result in sorted(variants.items()):
        row: dict[str, object] = {"variant": name, "eligible": result.get("eligible", False)}
        row.update(result.get("metrics", {}))
        for k, quality in result.get("quality", {}).items():
            row[f"mean_quality@{k}"] = quality.get("mean_quality", 0.0)
            row[f"low_quality_count@{k}"] = quality.get("low_quality_count", 0)
        rows.append(row)
    _write_rows(path, rows)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in rows]
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _column(rows: Sequence[dict[str, object]], key: str, *, default: float = 0.0) -> np.ndarray:
    return np.asarray([_to_float(row.get(key, default), default=default) for row in rows], dtype=float)


def _normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=float)
    lo = float(np.min(array[finite]))
    hi = float(np.max(array[finite]))
    if abs(hi - lo) < 1.0e-12:
        return np.zeros_like(array, dtype=float)
    normalized = (array - lo) / (hi - lo)
    return np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)


def _to_float(value: object, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object) -> int:
    return int(round(_to_float(value)))


def _required_mapping(config: dict[str, Any], key: str, label: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must include a '{key}' object.")
    return value


def _positive_int(section: dict[str, Any], key: str, label: str) -> None:
    value = section.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label}.{key} must be a positive integer.")


def _non_negative_int(section: dict[str, Any], key: str, label: str) -> None:
    value = section.get(key)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{label}.{key} must be a non-negative integer.")


def _positive_float(section: dict[str, Any], key: str, label: str) -> None:
    value = section.get(key)
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{label}.{key} must be positive.")
