from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from marginal_value.eval.ablation_eval import summarize_ranked_scores
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

SUPPORT_FEATURE_COLUMNS = [
    "token_duplicate_count",
    "token_duplicate_fraction",
    "token_max_neighbor_similarity",
    "token_mean_top5_neighbor_similarity",
    "token_neighborhood_density",
    "token_fold_centroid_similarity",
    "token_boundary_score",
]


def run_motion_phrase_holdout_eval(
    config: dict[str, Any],
    *,
    token_sequence_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    smoke: bool = False,
    allow_local_execution: bool = False,
) -> dict[str, Any]:
    validate_motion_phrase_holdout_config(config, allow_local_execution=allow_local_execution)
    mode = "smoke" if smoke else "full"
    artifacts = config["artifacts"]
    token_path = Path(token_sequence_path) if token_sequence_path is not None else Path(artifacts["tokens_dir"]) / f"token_sequences_{mode}.jsonl"
    output = Path(output_dir) if output_dir is not None else Path(artifacts["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    log_event(
        "motion_phrase_holdout",
        "start",
        mode=mode,
        token_sequence_path=str(token_path),
        output_dir=str(output),
    )
    sequences = read_token_sequences_jsonl(token_path)
    log_event("motion_phrase_holdout", "sequence_read_done", n_sequences=len(sequences))

    grammar_config = config["grammar"]
    phrase_config = config["phrase_holdout"]
    fit_split = str(grammar_config["fit_split"])
    source_sequences = [sequence for sequence in sequences if sequence.split == fit_split]
    if not source_sequences:
        raise ValueError(f"No token sequences found for fit_split '{fit_split}'.")

    max_families = int(phrase_config["max_families"])
    if smoke:
        max_families = min(max_families, 2)
    families = discover_phrase_families(
        source_sequences,
        phrase_len=int(phrase_config["phrase_len"]),
        min_support=int(phrase_config["min_support"]),
        max_families=max_families,
    )
    if not families:
        raise ValueError("No phrase families met the phrase holdout support threshold.")
    log_event("motion_phrase_holdout", "families_discovered", n_families=len(families), families=families[:5])

    all_rows: list[dict[str, object]] = []
    fold_reports: list[dict[str, Any]] = []
    for fold_index, family in enumerate(families, start=1):
        fold = build_phrase_holdout_rows(
            source_sequences,
            phrase=family["phrase"],
            fold_index=fold_index,
            grammar_order=int(grammar_config["order"]),
            smoothing=float(grammar_config["smoothing"]),
            rare_threshold=int(grammar_config["rare_threshold"]),
            negative_sample_size=int(phrase_config["negative_sample_size"]),
            artifact_negative_count=int(phrase_config["artifact_negative_count"]),
            redundancy_negative_count=int(phrase_config["redundancy_negative_count"]),
            seed=int(phrase_config["seed"]),
        )
        fold_reports.append(fold["report"])
        all_rows.extend(fold["rows"])
        log_event(
            "motion_phrase_holdout",
            "fold_done",
            fold_index=fold_index,
            phrase=fold["report"]["phrase"],
            n_positive=fold["report"]["n_positive"],
            n_negative=fold["report"]["n_negative"],
            heldout_fit_overlap_count=fold["report"]["heldout_fit_overlap_count"],
        )

    k_values = [int(k) for k in config.get("eval", {}).get("k_values", [10, 50, 100, 200])]
    low_quality_threshold = float(config.get("eval", {}).get("low_quality_threshold", 0.45))
    variants = evaluate_motion_phrase_variants(
        all_rows,
        k_values=k_values,
        low_quality_threshold=low_quality_threshold,
    )
    best_name, best_variant = _best_variant(variants)

    suffix = mode
    report_path = output / f"motion_phrase_holdout_report_{suffix}.json"
    scores_path = output / f"motion_phrase_holdout_scores_{suffix}.csv"
    summary_path = output / f"motion_phrase_holdout_summary_{suffix}.csv"
    negative_type_counts = Counter(str(row.get("negative_type", "")) for row in all_rows if str(row.get("negative_type", "")))
    report = {
        "mode": mode,
        "fit_split": fit_split,
        "token_sequence_path": str(token_path),
        "source_sequence_count": len(source_sequences),
        "phrase_holdout": {
            "phrase_len": int(phrase_config["phrase_len"]),
            "min_support": int(phrase_config["min_support"]),
            "max_families": max_families,
            "negative_sample_size": int(phrase_config["negative_sample_size"]),
            "artifact_negative_count": int(phrase_config["artifact_negative_count"]),
            "redundancy_negative_count": int(phrase_config["redundancy_negative_count"]),
            "seed": int(phrase_config["seed"]),
        },
        "grammar": {
            "order": int(grammar_config["order"]),
            "smoothing": float(grammar_config["smoothing"]),
            "rare_threshold": int(grammar_config["rare_threshold"]),
        },
        "support_feature_columns": SUPPORT_FEATURE_COLUMNS,
        "families": [_json_ready_family(family) for family in families],
        "folds": fold_reports,
        "n_rows": len(all_rows),
        "label_counts": {str(key): int(value) for key, value in Counter(int(row.get("label", 0)) for row in all_rows).items()},
        "negative_type_counts": {str(key): int(value) for key, value in sorted(negative_type_counts.items())},
        "leakage_audit": {
            "source_splits": sorted({sequence.split for sequence in source_sequences}),
            "heldout_fit_overlap_count": int(sum(int(fold["heldout_fit_overlap_count"]) for fold in fold_reports)),
            "score_rows_expose_heldout_phrase": any("contains_heldout_phrase" in row for row in all_rows),
        },
        "variants": variants,
        "best_by_ndcg100": best_name,
        "best_ndcg100": _primary_ndcg(best_variant.get("metrics", {})),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_rows(scores_path, all_rows)
    _write_summary_csv(summary_path, variants)
    log_event(
        "motion_phrase_holdout",
        "artifacts_written",
        report_path=str(report_path),
        scores_path=str(scores_path),
        summary_path=str(summary_path),
    )

    result = {
        "mode": mode,
        "fit_split": fit_split,
        "n_folds": len(fold_reports),
        "n_rows": len(all_rows),
        "best_by_ndcg100": best_name,
        "best_ndcg100": report["best_ndcg100"],
        "report_path": str(report_path),
        "scores_path": str(scores_path),
        "summary_path": str(summary_path),
    }
    log_event("motion_phrase_holdout", "done", **result)
    return result


def discover_phrase_families(
    sequences: Sequence[TokenSequence],
    *,
    phrase_len: int,
    min_support: int,
    max_families: int,
) -> list[dict[str, Any]]:
    if phrase_len <= 0:
        raise ValueError("phrase_len must be positive.")
    if min_support <= 0:
        raise ValueError("min_support must be positive.")
    if max_families <= 0:
        raise ValueError("max_families must be positive.")

    support_counts: Counter[tuple[str, ...]] = Counter()
    for sequence in sequences:
        tokens = sequence.primitive_token_ids
        unique_phrases = {
            tuple(tokens[index : index + phrase_len])
            for index in range(0, max(0, len(tokens) - phrase_len + 1))
        }
        support_counts.update(unique_phrases)

    families = [
        {"phrase": phrase, "support": int(support)}
        for phrase, support in support_counts.items()
        if int(support) >= min_support
    ]
    families.sort(key=lambda family: (-int(family["support"]), tuple(str(token) for token in family["phrase"])))
    return families[:max_families]


def build_phrase_holdout_rows(
    sequences: Sequence[TokenSequence],
    *,
    phrase: Sequence[str],
    fold_index: int,
    grammar_order: int,
    smoothing: float,
    rare_threshold: int,
    negative_sample_size: int,
    artifact_negative_count: int,
    redundancy_negative_count: int,
    seed: int,
) -> dict[str, Any]:
    phrase_tuple = tuple(str(token) for token in phrase)
    positive_sequences = [sequence for sequence in sequences if _contains_phrase(sequence.primitive_token_ids, phrase_tuple)]
    fit_sequences = [sequence for sequence in sequences if not _contains_phrase(sequence.primitive_token_ids, phrase_tuple)]
    if not positive_sequences:
        raise ValueError(f"Held-out phrase has no positive sequences: {phrase_tuple}")
    if not fit_sequences:
        raise ValueError(f"Held-out phrase leaves no fit sequences: {phrase_tuple}")

    grammar = NGramMotionGrammar(order=grammar_order, smoothing=smoothing, rare_threshold=rare_threshold)
    grammar.fit([sequence.primitive_token_ids for sequence in fit_sequences])
    rng = np.random.default_rng(seed + fold_index)
    base_negatives = _sample_sequences(fit_sequences, max_count=negative_sample_size, rng=rng)
    artifact_sources = _sample_sequences(positive_sequences, max_count=artifact_negative_count, rng=rng, replace=True)
    redundancy_sources = _sample_sequences(positive_sequences, max_count=redundancy_negative_count, rng=rng, replace=True)

    rows: list[dict[str, object]] = []
    for sequence in positive_sequences:
        rows.append(
            _score_sequence_row(
                sequence,
                grammar=grammar,
                fold_index=fold_index,
                label=1,
                phrase=phrase_tuple,
                row_kind="positive",
                quality_override=None,
                duplicate_group_id="",
            )
        )
    for sequence in base_negatives:
        rows.append(
            _score_sequence_row(
                sequence,
                grammar=grammar,
                fold_index=fold_index,
                label=0,
                phrase=phrase_tuple,
                row_kind="ordinary",
                quality_override=None,
                duplicate_group_id="",
            )
        )
    for copy_index, sequence in enumerate(artifact_sources, start=1):
        rows.append(
            _score_sequence_row(
                sequence,
                grammar=grammar,
                fold_index=fold_index,
                label=0,
                phrase=phrase_tuple,
                row_kind="artifact",
                quality_override=min(_sequence_quality(sequence), 0.20),
                duplicate_group_id=f"artifact:{fold_index}:{copy_index}:{sequence.sample_id}",
            )
        )
    for copy_index, sequence in enumerate(redundancy_sources, start=1):
        rows.append(
            _score_sequence_row(
                sequence,
                grammar=grammar,
                fold_index=fold_index,
                label=0,
                phrase=phrase_tuple,
                row_kind="redundancy",
                quality_override=None,
                duplicate_group_id=f"redundant:{fold_index}:{copy_index}:{sequence.sample_id}",
            )
        )

    add_token_support_features(rows)
    heldout_ids = {sequence.sample_id for sequence in positive_sequences}
    fit_ids = {sequence.sample_id for sequence in fit_sequences}
    report = {
        "fold_index": fold_index,
        "phrase": list(phrase_tuple),
        "n_positive": len(positive_sequences),
        "n_negative": len(rows) - len(positive_sequences),
        "n_fit_sequences": len(fit_sequences),
        "n_base_negative": len(base_negatives),
        "n_artifact_negative": len(artifact_sources),
        "n_redundancy_negative": len(redundancy_sources),
        "heldout_fit_overlap_count": len(heldout_ids & fit_ids),
    }
    return {"report": report, "rows": rows}


def add_token_support_features(rows: Sequence[dict[str, object]]) -> None:
    by_fold: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_fold.setdefault(str(row.get("fold_index", "")), []).append(row)

    for fold_rows in by_fold.values():
        token_sequences = [_row_tokens(row) for row in fold_rows]
        signatures = [tuple(tokens) for tokens in token_sequences]
        signature_counts = Counter(signatures)
        vectors = _token_histogram_vectors(token_sequences)
        n_rows = len(fold_rows)
        similarities = vectors @ vectors.T if n_rows else np.zeros((0, 0), dtype=float)
        centroid = np.mean(vectors, axis=0) if n_rows else np.zeros((0,), dtype=float)
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm > 1.0e-12:
            centroid = centroid / centroid_norm

        for row_index, row in enumerate(fold_rows):
            duplicate_count = float(signature_counts[signatures[row_index]])
            duplicate_fraction = float((duplicate_count - 1.0) / max(1, n_rows - 1))
            if n_rows > 1:
                neighbor_sims = np.delete(similarities[row_index], row_index)
                ranked = np.sort(neighbor_sims)[::-1]
                max_neighbor = float(ranked[0]) if ranked.size else 0.0
                mean_top5 = float(np.mean(ranked[: min(5, ranked.size)])) if ranked.size else 0.0
                density = float(np.mean(ranked[: min(10, ranked.size)])) if ranked.size else 0.0
            else:
                max_neighbor = 0.0
                mean_top5 = 0.0
                density = 0.0
            centroid_similarity = float(vectors[row_index] @ centroid) if centroid_norm > 1.0e-12 else 0.0
            row["token_duplicate_count"] = duplicate_count
            row["token_duplicate_fraction"] = duplicate_fraction
            row["token_max_neighbor_similarity"] = max_neighbor
            row["token_mean_top5_neighbor_similarity"] = mean_top5
            row["token_neighborhood_density"] = density
            row["token_fold_centroid_similarity"] = centroid_similarity
            row["token_boundary_score"] = float(1.0 - density)
            row.pop("_primitive_tokens", None)


def evaluate_motion_phrase_variants(
    rows: Sequence[dict[str, object]],
    *,
    k_values: Iterable[int],
    low_quality_threshold: float,
) -> dict[str, dict[str, Any]]:
    if not rows:
        return {}
    labels = np.asarray([_to_int(row.get("label", 0)) for row in rows], dtype=int)
    clusters = np.asarray([_to_int(row.get("fold_index", idx)) for idx, row in enumerate(rows)], dtype=int)
    score_inputs = _variant_scores(rows)
    variants: dict[str, dict[str, Any]] = {}
    for name, scores in score_inputs.items():
        metrics = summarize_ranked_scores(scores=scores, labels=labels, clusters=clusters, k_values=k_values)
        ranked_indices = np.argsort(-scores, kind="mergesort")
        variants[name] = {
            "eligible": True,
            "metrics": metrics,
            "quality": _top_quality_summary(rows, ranked_indices, k_values, low_quality_threshold=low_quality_threshold),
            "artifact_redundancy": _top_artifact_redundancy_summary(rows, ranked_indices, k_values),
        }
    return variants


def validate_motion_phrase_holdout_config(config: dict[str, Any], *, allow_local_execution: bool = False) -> None:
    execution = _required_mapping(config, "execution", "Motion phrase holdout config")
    artifacts = _required_mapping(config, "artifacts", "Motion phrase holdout config")
    grammar = _required_mapping(config, "grammar", "Motion phrase holdout config")
    phrase_holdout = _required_mapping(config, "phrase_holdout", "Motion phrase holdout config")
    if execution.get("provider") != "modal" and not allow_local_execution:
        raise ValueError("Motion phrase holdout must run on Modal for real artifacts.")
    if not allow_local_execution and not execution.get("artifacts_volume"):
        raise ValueError("Motion phrase holdout execution.artifacts_volume must be provided.")
    if "timeout_seconds" in execution:
        _positive_int(execution, "timeout_seconds", "Motion phrase holdout execution")
    for key in ("tokens_dir", "output_dir"):
        value = str(artifacts.get(key, ""))
        if not value:
            raise ValueError(f"Motion phrase holdout artifacts.{key} must be provided.")
        if not allow_local_execution and not value.startswith("/artifacts") and not Path(value).is_absolute():
            raise ValueError(f"Motion phrase holdout artifacts.{key} must be an absolute path.")
    if grammar.get("fit_split") != "pretrain":
        raise ValueError("Motion phrase holdout grammar.fit_split must be 'pretrain'.")
    _positive_int(grammar, "order", "Motion phrase holdout grammar")
    _positive_float(grammar, "smoothing", "Motion phrase holdout grammar")
    _non_negative_int(grammar, "rare_threshold", "Motion phrase holdout grammar")
    _positive_int(phrase_holdout, "phrase_len", "Motion phrase holdout phrase_holdout")
    _positive_int(phrase_holdout, "min_support", "Motion phrase holdout phrase_holdout")
    _positive_int(phrase_holdout, "max_families", "Motion phrase holdout phrase_holdout")
    _non_negative_int(phrase_holdout, "negative_sample_size", "Motion phrase holdout phrase_holdout")
    _non_negative_int(phrase_holdout, "artifact_negative_count", "Motion phrase holdout phrase_holdout")
    _non_negative_int(phrase_holdout, "redundancy_negative_count", "Motion phrase holdout phrase_holdout")
    if not isinstance(phrase_holdout.get("seed"), int):
        raise ValueError("Motion phrase holdout phrase_holdout.seed must be an integer.")


def load_motion_phrase_holdout_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _contains_phrase(tokens: Sequence[str], phrase: Sequence[str]) -> bool:
    if not phrase or len(tokens) < len(phrase):
        return False
    phrase_tuple = tuple(str(token) for token in phrase)
    return any(tuple(tokens[index : index + len(phrase_tuple)]) == phrase_tuple for index in range(0, len(tokens) - len(phrase_tuple) + 1))


def _sample_sequences(
    sequences: Sequence[TokenSequence],
    *,
    max_count: int,
    rng: np.random.Generator,
    replace: bool = False,
) -> list[TokenSequence]:
    if max_count <= 0 or not sequences:
        return []
    count = int(max_count) if replace else min(int(max_count), len(sequences))
    indices = rng.choice(len(sequences), size=count, replace=replace)
    return [sequences[int(index)] for index in np.atleast_1d(indices).tolist()]


def _score_sequence_row(
    sequence: TokenSequence,
    *,
    grammar: NGramMotionGrammar,
    fold_index: int,
    label: int,
    phrase: Sequence[str],
    row_kind: str,
    quality_override: float | None,
    duplicate_group_id: str,
) -> dict[str, object]:
    quality_score = _sequence_quality(sequence) if quality_override is None else float(quality_override)
    is_artifact = 1 if row_kind == "artifact" else 0
    is_redundant = 1 if row_kind == "redundancy" else 0
    row: dict[str, object] = {
        "fold_index": fold_index,
        "sample_id": sequence.sample_id,
        "worker_id": str(sequence.metadata.get("worker_id", sequence.sample_id)),
        "split": sequence.split,
        "label": int(label),
        "heldout_phrase_id": _stable_phrase_id(phrase),
        "phrase_len": len(phrase),
        "negative_type": "" if label else row_kind,
        "quality_score": quality_score,
        "is_artifact": is_artifact,
        "is_redundant": is_redundant,
        "duplicate_group_id": duplicate_group_id,
        "n_primitives": len(sequence.primitive_token_ids),
        "reason_code": _reason_code(label=label, row_kind=row_kind),
        "_primitive_tokens": tuple(sequence.primitive_token_ids),
    }
    row.update(grammar.sequence_features(sequence.primitive_token_ids))
    return row


def _row_tokens(row: dict[str, object]) -> list[str]:
    value = row.get("_primitive_tokens", ())
    if isinstance(value, str):
        return [token for token in value.split("\x1f") if token]
    try:
        return [str(token) for token in value]  # type: ignore[arg-type]
    except TypeError:
        return []


def _token_histogram_vectors(token_sequences: Sequence[Sequence[str]]) -> np.ndarray:
    vocabulary = sorted({str(token) for tokens in token_sequences for token in tokens})
    if not vocabulary:
        return np.zeros((len(token_sequences), 1), dtype=float)
    token_to_index = {token: index for index, token in enumerate(vocabulary)}
    vectors = np.zeros((len(token_sequences), len(vocabulary)), dtype=float)
    for row_index, tokens in enumerate(token_sequences):
        counts = Counter(str(token) for token in tokens)
        for token, count in counts.items():
            vectors[row_index, token_to_index[token]] = float(count)
    norms = np.linalg.norm(vectors, axis=1)
    norms = np.where(norms < 1.0e-12, 1.0, norms)
    return vectors / norms[:, None]


def _variant_scores(rows: Sequence[dict[str, object]]) -> dict[str, np.ndarray]:
    quality = _column(rows, "quality_score", default=1.0)
    token_nll = _column(rows, "token_nll_p95")
    transition_nll = _column(rows, "transition_nll_p95")
    rare_phrase = _column(rows, "rare_phrase_fraction")
    longest_unseen = _column(rows, "longest_unseen_phrase_len")
    is_artifact = _column(rows, "is_artifact")
    is_redundant = _column(rows, "is_redundant")
    duplicate_fraction = _column(rows, "token_duplicate_fraction")
    neighborhood_density = _column(rows, "token_neighborhood_density")
    grammar_surprisal = _normalize(0.65 * _normalize(token_nll) + 0.25 * _normalize(transition_nll) + 0.10 * _normalize(longest_unseen))
    phrase_mix = _normalize(0.70 * _normalize(token_nll) + 0.30 * _normalize(rare_phrase))
    artifact_redundancy_gate = np.clip(1.0 - is_artifact, 0.0, 1.0) * np.clip(1.0 - 0.75 * is_redundant, 0.0, 1.0)
    support_penalty = np.clip(1.0 - 0.50 * _normalize(duplicate_fraction) - 0.25 * _normalize(neighborhood_density), 0.0, 1.0)
    return {
        "random_baseline": np.asarray([_stable_random_score(row) for row in rows], dtype=float),
        "grammar_token_nll_p95": token_nll,
        "grammar_phrase_mix": phrase_mix,
        "grammar_surprisal_mix": grammar_surprisal,
        "grammar_surprisal_quality_gated": grammar_surprisal * np.clip(quality, 0.0, 1.0),
        "grammar_quality_support_penalized": grammar_surprisal * np.clip(quality, 0.0, 1.0) * support_penalty,
        "grammar_artifact_redundancy_gated": grammar_surprisal * np.clip(quality, 0.0, 1.0) * artifact_redundancy_gate,
    }


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


def _top_artifact_redundancy_summary(
    rows: Sequence[dict[str, object]],
    ranked_indices: np.ndarray,
    k_values: Iterable[int],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for k in k_values:
        indices = ranked_indices[: min(int(k), len(ranked_indices))]
        artifacts = [_to_int(rows[int(idx)].get("is_artifact", 0)) for idx in indices]
        redundancies = [_to_int(rows[int(idx)].get("is_redundant", 0)) for idx in indices]
        summary[str(k)] = {
            "artifact_rate": float(np.mean(artifacts)) if artifacts else 0.0,
            "redundancy_rate": float(np.mean(redundancies)) if redundancies else 0.0,
            "artifact_count": int(sum(artifacts)),
            "redundancy_count": int(sum(redundancies)),
        }
    return summary


def _write_summary_csv(path: Path, variants: dict[str, dict[str, Any]]) -> None:
    rows: list[dict[str, object]] = []
    for name, result in sorted(variants.items()):
        row: dict[str, object] = {"variant": name, "eligible": result.get("eligible", False)}
        row.update(result.get("metrics", {}))
        for k, quality in result.get("quality", {}).items():
            row[f"mean_quality@{k}"] = quality.get("mean_quality", 0.0)
            row[f"low_quality_count@{k}"] = quality.get("low_quality_count", 0)
        for k, artifact_summary in result.get("artifact_redundancy", {}).items():
            row[f"artifact_rate@{k}"] = artifact_summary.get("artifact_rate", 0.0)
            row[f"redundancy_rate@{k}"] = artifact_summary.get("redundancy_rate", 0.0)
            row[f"artifact_count@{k}"] = artifact_summary.get("artifact_count", 0)
            row[f"redundancy_count@{k}"] = artifact_summary.get("redundancy_count", 0)
        rows.append(row)
    _write_rows(path, rows)


def _write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_rows = [dict(row) for row in rows]
    fieldnames = sorted({key for row in output_rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def _json_ready_family(family: dict[str, Any]) -> dict[str, Any]:
    output = dict(family)
    output["phrase"] = list(output.get("phrase", []))
    return output


def _best_variant(variants: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    if not variants:
        return "", {}
    return max(variants.items(), key=lambda item: _primary_ndcg(item[1].get("metrics", {})))


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


def _sequence_quality(sequence: TokenSequence) -> float:
    return 1.0 if sequence.quality_score is None else float(sequence.quality_score)


def _reason_code(*, label: int, row_kind: str) -> str:
    if row_kind == "artifact":
        return "LIKELY_SENSOR_ARTIFACT"
    if row_kind == "redundancy":
        return "REDUNDANT_KNOWN_WORKFLOW"
    if label:
        return "RARE_TEMPORAL_COMPOSITION"
    return "REDUNDANT_KNOWN_WORKFLOW"


def _stable_phrase_id(phrase: Sequence[str]) -> str:
    payload = "\x1f".join(str(token) for token in phrase)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _stable_random_score(row: dict[str, object]) -> float:
    key = f"{row.get('fold_index', '')}:{row.get('sample_id', '')}:{row.get('negative_type', '')}:{row.get('duplicate_group_id', '')}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


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
