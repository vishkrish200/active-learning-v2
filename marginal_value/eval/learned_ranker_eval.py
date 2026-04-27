from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from marginal_value.eval.ablation_eval import summarize_ranked_scores
from marginal_value.logging_utils import log_event
from marginal_value.models.learned_linear_ranker import (
    LinearRankerModel,
    fit_linear_ranker,
    score_linear_ranker,
    write_linear_ranker_model,
)


GRAMMAR_FEATURE_COLUMNS = {
    "token_nll_mean",
    "token_nll_p90",
    "token_nll_p95",
    "transition_nll_mean",
    "transition_nll_p95",
    "rare_bigram_fraction",
    "rare_trigram_fraction",
    "rare_phrase_fraction",
    "longest_unseen_phrase_len",
}

FORBIDDEN_FEATURE_COLUMNS = {
    "candidate_index",
    "cluster_bonus",
    "cluster_round",
    "cluster_selection_score",
    "final_score",
    "fold_index",
    "grammar_feature_present",
    "grammar_feature_split",
    "grammar_promotion_applied",
    "grammar_promotion_delta",
    "grammar_score",
    "grammar_score_component",
    "grammar_score_variant",
    "grammar_score_weight",
    "grammar_support_gate",
    "heldout_cluster",
    "heldout_phrase_id",
    "is_artifact",
    "is_redundant",
    "label",
    "negative_type",
    "new_cluster_id",
    "new_density_score",
    "old_novelty_score",
    "phase_a_final_score",
    "phase_a_ranker_score",
    "rank",
    "ranker_score",
    "reason_code",
    "redundancy_penalty",
    "duplicate_group_id",
    "rerank_score",
    "sample_id",
    "source_cluster",
    "split",
    "worker_id",
}

DEFAULT_REQUESTED_FEATURES = [
    "quality_score",
    "old_knn_distance",
    "new_batch_density",
    "new_cluster_size",
    "is_singleton",
    "distance_to_new_cluster_medoid",
    "token_nll_mean",
    "token_nll_p95",
    "transition_nll_p95",
    "rare_phrase_fraction",
    "longest_unseen_phrase_len",
]

DEFAULT_BASELINE_SCORE_COLUMNS = [
    "phase_a_final_score",
    "final_score",
    "old_knn_distance",
    "new_batch_density",
]


class FeatureLeakageError(RuntimeError):
    """Raised when a learned-ranker feature would leak labels or held-out membership."""


@dataclass(frozen=True)
class FeatureTable:
    values: np.ndarray
    labels: np.ndarray
    sample_ids: list[str]
    feature_names: list[str]
    excluded_features: dict[str, str]
    coverage_by_label: dict[str, dict[str, float]]


@dataclass(frozen=True)
class HashSplit:
    train_indices: np.ndarray
    eval_indices: np.ndarray
    leakage_audit: dict[str, int]


def run_learned_ranker_eval(
    config: dict[str, Any],
    *,
    candidate_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    allow_local_execution: bool = False,
) -> dict[str, Any]:
    validate_learned_ranker_config(config, allow_local_execution=allow_local_execution)
    artifacts = config["artifacts"]
    candidates = Path(candidate_path) if candidate_path is not None else Path(artifacts["candidate_path"])
    output = Path(output_dir) if output_dir is not None else Path(artifacts["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    log_event("learned_ranker", "start", candidate_path=str(candidates), output_dir=str(output))
    rows = _read_csv(candidates)
    log_event("learned_ranker", "candidate_read_done", n_rows=len(rows))

    feature_config = config.get("features", {})
    table = build_feature_table(
        rows,
        requested_features=feature_config.get("requested", DEFAULT_REQUESTED_FEATURES),
        strict_feature_coverage=bool(feature_config.get("strict_feature_coverage", False)),
    )
    log_event(
        "learned_ranker",
        "feature_table_done",
        n_features=len(table.feature_names),
        excluded_features=len(table.excluded_features),
    )

    eval_config = config.get("eval", {})
    k_values = [int(value) for value in eval_config.get("k_values", [10, 50, 100, 200])]
    fold_column = str(eval_config.get("fold_column", ""))
    if fold_column:
        split_outputs = _run_fold_holdout_scoring(
            rows,
            table,
            fold_column=fold_column,
            exclude_eval_samples_from_train=bool(eval_config.get("exclude_eval_samples_from_train", True)),
        )
    else:
        split_outputs = _run_hash_split_scoring(
            table,
            eval_fraction=float(eval_config.get("eval_fraction", 0.30)),
            seed=int(eval_config.get("seed", 17)),
        )
    eval_indices = split_outputs["eval_indices"]
    eval_y = table.labels[eval_indices]
    eval_rows = [rows[int(idx)] for idx in eval_indices]
    clusters = np.asarray([_to_int(row.get("new_cluster_id", 0)) for row in eval_rows], dtype=int)
    learned_scores = split_outputs["eval_scores"]
    all_scores = split_outputs["all_scores"]

    variants = {
        "learned_linear": {
            "score_column": "learned_linear_score",
            "metrics": summarize_ranked_scores(
                scores=learned_scores,
                labels=eval_y,
                clusters=clusters,
                k_values=k_values,
            ),
        }
    }
    baseline_columns = [str(value) for value in eval_config.get("baseline_score_columns", DEFAULT_BASELINE_SCORE_COLUMNS)]
    for column in baseline_columns:
        baseline_scores = _column_scores(eval_rows, column)
        if baseline_scores is None:
            continue
        variants[column] = {
            "score_column": column,
            "metrics": summarize_ranked_scores(
                scores=baseline_scores,
                labels=eval_y,
                clusters=clusters,
                k_values=k_values,
            ),
        }

    report_path = output / "learned_ranker_eval_full.json"
    scores_path = output / "learned_ranker_scores_full.csv"
    summary_path = output / "learned_ranker_summary_full.csv"
    model_path = output / "learned_ranker_model_full.json"
    best_name, best_variant = _best_variant(variants)
    final_model = fit_linear_ranker(table.values, table.labels)
    write_linear_ranker_model(
        model_path,
        final_model,
        feature_names=table.feature_names,
        metadata={
            "candidate_path": str(candidates),
            "fit_rows": len(rows),
            "label_counts": _label_counts(table.labels),
            "final_model_fit_uses_all_rows": True,
            "eval_report_path": str(report_path),
        },
    )
    report = {
        "mode": "full",
        "candidate_path": str(candidates),
        "n_rows": len(rows),
        "feature_table": {
            "feature_names": table.feature_names,
            "excluded_features": table.excluded_features,
            "coverage_by_label": table.coverage_by_label,
        },
        "split": {
            "seed": int(eval_config.get("seed", 17)),
            "eval_fraction": float(eval_config.get("eval_fraction", 0.30)),
        }
        | split_outputs["split_report"],
        "leakage_audit": {
            **split_outputs["leakage_audit"],
            "fit_uses_only_train_indices": True,
            "forbidden_feature_request_count": int(
                sum(1 for reason in table.excluded_features.values() if reason == "forbidden_leakage_column")
            ),
            "asymmetric_feature_exclusion_count": int(
                sum(1 for reason in table.excluded_features.values() if reason == "asymmetric_grammar_feature_coverage")
            ),
        },
        "model": {
            "type": "linear_centroid",
            "n_features": len(table.feature_names),
            "model_path": str(model_path),
            "final_model_fit_uses_all_rows": True,
        },
        "variants": variants,
        "best_by_ndcg100": best_name,
        "best_ndcg100": _primary_ndcg(best_variant.get("metrics", {})),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_score_rows(scores_path, rows, table.sample_ids, table.labels, all_scores, split_outputs["score_split"])
    _write_summary_rows(summary_path, variants)

    result = {
        "n_rows": len(rows),
        "n_features": len(table.feature_names),
        "n_excluded_features": len(table.excluded_features),
        "train_count": int(split_outputs["split_report"]["train_count"]),
        "eval_count": int(split_outputs["split_report"]["eval_count"]),
        "best_by_ndcg100": best_name,
        "best_ndcg100": report["best_ndcg100"],
        "report_path": str(report_path),
        "scores_path": str(scores_path),
        "summary_path": str(summary_path),
        "model_path": str(model_path),
    }
    log_event("learned_ranker", "done", **result)
    return result


def _run_hash_split_scoring(
    table: FeatureTable,
    *,
    eval_fraction: float,
    seed: int,
) -> dict[str, Any]:
    split = make_stratified_hash_split(
        table.sample_ids,
        table.labels,
        eval_fraction=eval_fraction,
        seed=seed,
    )
    log_event(
        "learned_ranker",
        "fit_start",
        split_method="stratified_hash",
        n_train=len(split.train_indices),
        n_eval=len(split.eval_indices),
        n_features=len(table.feature_names),
    )
    model = fit_linear_ranker(table.values[split.train_indices], table.labels[split.train_indices])
    eval_scores = score_linear_ranker(model, table.values[split.eval_indices])
    all_scores = score_linear_ranker(model, table.values)
    log_event("learned_ranker", "fit_done", model_type="linear_centroid", split_method="stratified_hash")
    return {
        "eval_indices": split.eval_indices,
        "eval_scores": eval_scores,
        "all_scores": all_scores,
        "score_split": split,
        "split_report": {
            "method": "stratified_hash",
            "train_count": int(len(split.train_indices)),
            "eval_count": int(len(split.eval_indices)),
            "train_label_counts": _label_counts(table.labels[split.train_indices]),
            "eval_label_counts": _label_counts(table.labels[split.eval_indices]),
        },
        "leakage_audit": split.leakage_audit,
    }


def _run_fold_holdout_scoring(
    rows: Sequence[dict[str, object]],
    table: FeatureTable,
    *,
    fold_column: str,
    exclude_eval_samples_from_train: bool,
) -> dict[str, Any]:
    fold_splits = make_fold_holdout_splits(
        rows,
        table.sample_ids,
        table.labels,
        fold_column=fold_column,
        exclude_eval_samples_from_train=exclude_eval_samples_from_train,
    )
    all_scores = np.full(len(rows), np.nan, dtype=float)
    eval_chunks: list[np.ndarray] = []
    fold_reports: list[dict[str, object]] = []
    train_count = 0
    for fold_split in fold_splits:
        train_indices = np.asarray(fold_split["train_indices"], dtype=int)
        eval_indices = np.asarray(fold_split["eval_indices"], dtype=int)
        log_event(
            "learned_ranker",
            "fit_start",
            split_method="fold_holdout",
            fold=fold_split["fold"],
            n_train=len(train_indices),
            n_eval=len(eval_indices),
            n_features=len(table.feature_names),
        )
        model = fit_linear_ranker(table.values[train_indices], table.labels[train_indices])
        all_scores[eval_indices] = score_linear_ranker(model, table.values[eval_indices])
        eval_chunks.append(eval_indices)
        train_count += int(len(train_indices))
        fold_reports.append(
            {
                "fold": fold_split["fold"],
                "train_count": int(len(train_indices)),
                "eval_count": int(len(eval_indices)),
                "train_label_counts": _label_counts(table.labels[train_indices]),
                "eval_label_counts": _label_counts(table.labels[eval_indices]),
                "sample_overlap_count": int(fold_split["sample_overlap_count"]),
                "fold_overlap_count": int(fold_split["fold_overlap_count"]),
            }
        )
    eval_indices = np.concatenate(eval_chunks) if eval_chunks else np.asarray([], dtype=int)
    if np.any(~np.isfinite(all_scores[eval_indices])):
        raise RuntimeError("Fold holdout scoring left non-finite scores for eval rows.")
    leakage_audit = {
        "sample_overlap_count": int(sum(int(fold["sample_overlap_count"]) for fold in fold_reports)),
        "fold_overlap_count": int(sum(int(fold["fold_overlap_count"]) for fold in fold_reports)),
        "train_count": int(train_count),
        "eval_count": int(len(eval_indices)),
    }
    log_event("learned_ranker", "fit_done", model_type="linear_centroid", split_method="fold_holdout", n_folds=len(fold_reports))
    return {
        "eval_indices": eval_indices,
        "eval_scores": all_scores[eval_indices],
        "all_scores": np.nan_to_num(all_scores, nan=0.0),
        "score_split": HashSplit(train_indices=np.asarray([], dtype=int), eval_indices=eval_indices, leakage_audit=leakage_audit),
        "split_report": {
            "method": "fold_holdout",
            "fold_column": fold_column,
            "exclude_eval_samples_from_train": bool(exclude_eval_samples_from_train),
            "fold_count": len(fold_reports),
            "folds": fold_reports,
            "train_count": int(train_count),
            "eval_count": int(len(eval_indices)),
            "eval_label_counts": _label_counts(table.labels[eval_indices]),
        },
        "leakage_audit": leakage_audit,
    }


def validate_learned_ranker_config(config: dict[str, Any], *, allow_local_execution: bool = False) -> None:
    execution = _required_mapping(config, "execution")
    artifacts = _required_mapping(config, "artifacts")
    if execution.get("provider") != "modal" and not allow_local_execution:
        raise ValueError("Learned ranker fitting must be dispatched through Modal.")
    candidate_path = Path(str(artifacts.get("candidate_path", "")))
    output_dir = Path(str(artifacts.get("output_dir", "")))
    if not candidate_path.is_absolute():
        raise ValueError("artifacts.candidate_path must be absolute.")
    if not output_dir.is_absolute():
        raise ValueError("artifacts.output_dir must be absolute.")
    eval_config = config.get("eval", {})
    eval_fraction = float(eval_config.get("eval_fraction", 0.30))
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval.eval_fraction must be in (0, 1).")
    k_values = eval_config.get("k_values", [10, 50, 100, 200])
    if not k_values or any(int(value) <= 0 for value in k_values):
        raise ValueError("eval.k_values must contain positive integers.")


def build_feature_table(
    rows: Sequence[dict[str, object]],
    *,
    requested_features: Iterable[str] | None = None,
    label_column: str = "label",
    sample_id_column: str = "sample_id",
    strict_feature_coverage: bool = False,
) -> FeatureTable:
    if not rows:
        raise ValueError("rows must not be empty.")
    features = list(requested_features or DEFAULT_REQUESTED_FEATURES)
    labels = np.asarray([_to_int(row.get(label_column, 0)) for row in rows], dtype=int)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("learned ranker eval requires binary labels {0, 1}.")
    sample_ids = [str(row.get(sample_id_column) or row.get("worker_id") or idx) for idx, row in enumerate(rows)]

    selected: list[str] = []
    columns: list[np.ndarray] = []
    excluded: dict[str, str] = {}
    coverage_by_label: dict[str, dict[str, float]] = {}
    has_explicit_grammar_presence = any("grammar_feature_present" in row for row in rows)
    explicit_grammar_presence = np.asarray([_truthy(row.get("grammar_feature_present", False)) for row in rows], dtype=bool)

    for feature in features:
        if feature in FORBIDDEN_FEATURE_COLUMNS:
            excluded[feature] = "forbidden_leakage_column"
            continue

        values = np.asarray([_to_float(row.get(feature, np.nan)) for row in rows], dtype=float)
        if feature in GRAMMAR_FEATURE_COLUMNS:
            feature_presence = explicit_grammar_presence if has_explicit_grammar_presence else np.isfinite(values)
            if not _has_balanced_presence(feature_presence, labels):
                excluded[feature] = "asymmetric_grammar_feature_coverage"
                coverage_by_label[feature] = _presence_by_label(feature_presence, labels)
                if strict_feature_coverage:
                    raise FeatureLeakageError(f"Feature '{feature}' has asymmetric grammar coverage by label.")
                continue
        if not np.all(np.isfinite(values)):
            excluded[feature] = "nonfinite_values"
            if strict_feature_coverage:
                raise FeatureLeakageError(f"Feature '{feature}' has missing or non-finite values.")
            continue
        selected.append(feature)
        columns.append(values)
        coverage_by_label[feature] = {
            str(label): float(np.mean(np.isfinite(values[labels == label]))) for label in sorted(set(labels.tolist()))
        }

    if not columns:
        raise FeatureLeakageError("No usable learned-ranker features remain after leakage checks.")
    return FeatureTable(
        values=np.column_stack(columns),
        labels=labels,
        sample_ids=sample_ids,
        feature_names=selected,
        excluded_features=excluded,
        coverage_by_label=coverage_by_label,
    )


def make_stratified_hash_split(
    sample_ids: Sequence[str],
    labels: np.ndarray,
    *,
    eval_fraction: float,
    seed: int,
) -> HashSplit:
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in (0, 1).")
    label_values = np.asarray(labels, dtype=int)
    if set(label_values.tolist()) != {0, 1}:
        raise ValueError("Hash split requires binary labels {0, 1}.")
    groups: dict[str, list[int]] = {}
    for idx, sample_id in enumerate(sample_ids):
        groups.setdefault(str(sample_id), []).append(idx)
    if len(groups) < 2:
        raise ValueError("Hash split requires at least two unique sample IDs.")

    label_targets = {
        label: min(
            int(np.sum(label_values == label)) - 1,
            max(1, int(round(int(np.sum(label_values == label)) * eval_fraction))),
        )
        for label in sorted(set(label_values.tolist()))
    }
    eval_groups: set[str] = set()
    eval_label_counts = {label: 0 for label in label_targets}
    for sample_id in sorted(groups, key=lambda value: _stable_hash(f"{seed}:{value}")):
        group_labels = label_values[groups[sample_id]]
        if any(eval_label_counts[label] < target for label, target in label_targets.items() if np.any(group_labels == label)):
            eval_groups.add(sample_id)
            for label in eval_label_counts:
                eval_label_counts[label] += int(np.sum(group_labels == label))
        if all(eval_label_counts[label] >= target for label, target in label_targets.items()):
            break

    train_indices = [idx for sample_id, indices in groups.items() for idx in indices if sample_id not in eval_groups]
    eval_indices = [idx for sample_id, indices in groups.items() for idx in indices if sample_id in eval_groups]

    train = np.asarray(sorted(train_indices), dtype=int)
    eval_ = np.asarray(sorted(eval_indices), dtype=int)
    if len(train) == 0 or len(eval_) == 0:
        raise ValueError("Hash split produced an empty train or eval side.")
    if set(label_values[train].tolist()) != {0, 1} or set(label_values[eval_].tolist()) != {0, 1}:
        raise ValueError("Hash split could not preserve both labels on train and eval sides.")
    train_sample_ids = {str(sample_ids[idx]) for idx in train.tolist()}
    eval_sample_ids = {str(sample_ids[idx]) for idx in eval_.tolist()}
    overlap = train_sample_ids & eval_sample_ids
    return HashSplit(
        train_indices=train,
        eval_indices=eval_,
        leakage_audit={
            "sample_overlap_count": len(overlap),
            "train_count": int(len(train)),
            "eval_count": int(len(eval_)),
        },
    )


def make_fold_holdout_splits(
    rows: Sequence[dict[str, object]],
    sample_ids: Sequence[str],
    labels: np.ndarray,
    *,
    fold_column: str,
    exclude_eval_samples_from_train: bool = True,
) -> list[dict[str, object]]:
    label_values = np.asarray(labels, dtype=int)
    folds = sorted({str(row.get(fold_column, "")) for row in rows if str(row.get(fold_column, ""))})
    if len(folds) < 2:
        raise ValueError("Fold holdout requires at least two non-empty folds.")
    outputs: list[dict[str, object]] = []
    for fold in folds:
        eval_indices = [idx for idx, row in enumerate(rows) if str(row.get(fold_column, "")) == fold]
        eval_sample_ids = {str(sample_ids[idx]) for idx in eval_indices}
        train_indices = [
            idx
            for idx, row in enumerate(rows)
            if str(row.get(fold_column, "")) != fold
            and (not exclude_eval_samples_from_train or str(sample_ids[idx]) not in eval_sample_ids)
        ]
        if not eval_indices or not train_indices:
            continue
        if set(label_values[eval_indices].tolist()) != {0, 1}:
            continue
        if set(label_values[train_indices].tolist()) != {0, 1}:
            continue
        train_sample_ids = {str(sample_ids[idx]) for idx in train_indices}
        train_folds = {str(rows[idx].get(fold_column, "")) for idx in train_indices}
        outputs.append(
            {
                "fold": fold,
                "train_indices": np.asarray(sorted(train_indices), dtype=int),
                "eval_indices": np.asarray(sorted(eval_indices), dtype=int),
                "sample_overlap_count": len(train_sample_ids & eval_sample_ids),
                "fold_overlap_count": int(fold in train_folds),
            }
        )
    if not outputs:
        raise ValueError("Fold holdout produced no valid folds with binary train/eval labels.")
    return outputs


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_score_rows(
    path: Path,
    rows: Sequence[dict[str, object]],
    sample_ids: Sequence[str],
    labels: np.ndarray,
    learned_scores: np.ndarray,
    split: HashSplit,
) -> None:
    eval_indices = set(split.eval_indices.tolist())
    output_rows: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        output_rows.append(
            {
                "sample_id": sample_ids[idx],
                "label": int(labels[idx]),
                "fit_split": "eval" if idx in eval_indices else "train",
                "learned_linear_score": float(learned_scores[idx]),
                "phase_a_final_score": row.get("phase_a_final_score", row.get("final_score", "")),
                "old_knn_distance": row.get("old_knn_distance", ""),
                "new_cluster_id": row.get("new_cluster_id", ""),
            }
        )
    _write_rows(path, output_rows)


def _write_summary_rows(path: Path, variants: dict[str, dict[str, Any]]) -> None:
    rows = []
    for name, variant in sorted(variants.items()):
        row: dict[str, object] = {"variant": name}
        row.update(variant.get("metrics", {}))
        rows.append(row)
    _write_rows(path, rows)


def _write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _column_scores(rows: Sequence[dict[str, object]], column: str) -> np.ndarray | None:
    if any(column not in row for row in rows):
        return None
    values = np.asarray([_to_float(row.get(column, np.nan)) for row in rows], dtype=float)
    if not np.all(np.isfinite(values)):
        return None
    return values


def _best_variant(variants: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    return max(variants.items(), key=lambda item: _primary_ndcg(item[1].get("metrics", {})))


def _primary_ndcg(metrics: dict[str, float]) -> float:
    if "ndcg@100" in metrics:
        return float(metrics["ndcg@100"])
    ndcg_items = [
        (int(key.split("@", 1)[1]), float(value))
        for key, value in metrics.items()
        if key.startswith("ndcg@") and key.split("@", 1)[1].isdigit()
    ]
    return sorted(ndcg_items)[-1][1] if ndcg_items else 0.0


def _label_counts(labels: np.ndarray) -> dict[str, int]:
    counts = Counter(int(value) for value in labels.tolist())
    return {str(label): int(count) for label, count in sorted(counts.items())}


def _has_balanced_presence(presence: np.ndarray, labels: np.ndarray) -> bool:
    by_label = _presence_by_label(presence, labels)
    return all(value >= 0.99 for value in by_label.values())


def _presence_by_label(presence: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        str(label): float(np.mean(presence[labels == label])) if np.any(labels == label) else 0.0
        for label in sorted(set(labels.tolist()))
    }


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _to_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return 1.0 if value.strip().lower() == "true" else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _to_int(value: object) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Learned ranker config must include a '{key}' object.")
    return value
