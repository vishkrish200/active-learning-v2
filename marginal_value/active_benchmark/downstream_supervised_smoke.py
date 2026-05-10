from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from marginal_value.active_benchmark.representations import clip_lookup, stack_embeddings
from marginal_value.active_benchmark.schema import BenchmarkClip, BenchmarkResult
from marginal_value.active_benchmark.splits import infer_source_family_assignments


def build_downstream_supervised_rows(
    result: BenchmarkResult,
    clips: Sequence[BenchmarkClip],
    *,
    downstream_representations: Sequence[str],
    label_source: str = "source_family",
    label_representation: str = "window",
    source_family_count: int = 4,
) -> list[dict[str, object]]:
    clips_by_id = clip_lookup(clips)
    episodes = {episode.episode_id: episode for episode in result.episodes}
    labels_by_id = _labels_by_clip_id(
        clips,
        label_source=label_source,
        label_representation=label_representation,
        source_family_count=source_family_count,
    )
    rows: list[dict[str, object]] = []
    for round_result in result.rounds:
        episode = episodes[str(round_result.episode_id)]
        for representation in downstream_representations:
            baseline = nearest_centroid_classification_metrics(
                clips_by_id,
                labels_by_id=labels_by_id,
                train_ids=round_result.support_ids_before,
                target_ids=episode.target_ids,
                representation=str(representation),
            )
            after = nearest_centroid_classification_metrics(
                clips_by_id,
                labels_by_id=labels_by_id,
                train_ids=round_result.support_ids_after,
                target_ids=episode.target_ids,
                representation=str(representation),
            )
            accuracy_gain = float(after["accuracy"] - baseline["accuracy"])
            balanced_accuracy_gain = float(after["balanced_accuracy"] - baseline["balanced_accuracy"])
            nll_reduction = float(baseline["negative_log_likelihood"] - after["negative_log_likelihood"])
            baseline_nll = float(baseline["negative_log_likelihood"])
            relative_nll_reduction = float(nll_reduction / baseline_nll) if baseline_nll > 1.0e-12 else 0.0
            rows.append(
                {
                    "episode_id": str(round_result.episode_id),
                    "fold_id": int(round_result.fold_id),
                    "policy_name": str(round_result.policy_name),
                    "round_index": int(round_result.round_index),
                    "representation": str(representation),
                    "label_source": str(label_source),
                    "label_representation": str(label_representation),
                    "selected_ids": list(round_result.selected_ids),
                    "selected_source_group_ids": list(round_result.selected_source_group_ids),
                    "support_count_before": int(round_result.support_count_before),
                    "support_count_after": int(round_result.support_count_after),
                    "target_count": int(len(episode.target_ids)),
                    "baseline_accuracy": float(baseline["accuracy"]),
                    "after_accuracy": float(after["accuracy"]),
                    "accuracy_gain": float(accuracy_gain),
                    "baseline_balanced_accuracy": float(baseline["balanced_accuracy"]),
                    "after_balanced_accuracy": float(after["balanced_accuracy"]),
                    "balanced_accuracy_gain": float(balanced_accuracy_gain),
                    "baseline_negative_log_likelihood": float(baseline["negative_log_likelihood"]),
                    "after_negative_log_likelihood": float(after["negative_log_likelihood"]),
                    "nll_reduction": float(nll_reduction),
                    "relative_nll_reduction": float(relative_nll_reduction),
                    "baseline_train_class_count": int(baseline["train_class_count"]),
                    "after_train_class_count": int(after["train_class_count"]),
                    "target_class_count": int(after["target_class_count"]),
                    "baseline_known_target_fraction": float(baseline["known_target_fraction"]),
                    "after_known_target_fraction": float(after["known_target_fraction"]),
                }
            )
    return rows


def nearest_centroid_classification_metrics(
    clips_by_id: dict[str, BenchmarkClip],
    *,
    labels_by_id: dict[str, str],
    train_ids: Sequence[str],
    target_ids: Sequence[str],
    representation: str,
) -> dict[str, float | int]:
    train_ids = tuple(str(sample_id) for sample_id in train_ids if str(sample_id) in labels_by_id)
    target_ids = tuple(str(sample_id) for sample_id in target_ids if str(sample_id) in labels_by_id)
    if not train_ids or not target_ids:
        return _empty_metrics()

    train_x = stack_embeddings(clips_by_id, train_ids, representation=representation)
    target_x = stack_embeddings(clips_by_id, target_ids, representation=representation)
    train_y = np.asarray([labels_by_id[sample_id] for sample_id in train_ids], dtype=object)
    target_y = np.asarray([labels_by_id[sample_id] for sample_id in target_ids], dtype=object)
    classes = tuple(sorted({str(label) for label in train_y.tolist()}))
    if not classes:
        return _empty_metrics()

    mean = np.mean(train_x, axis=0, keepdims=True)
    scale = np.std(train_x, axis=0, keepdims=True)
    scale = np.where(scale > 1.0e-9, scale, 1.0)
    train_z = (train_x - mean) / scale
    target_z = (target_x - mean) / scale
    centroids = np.vstack([np.mean(train_z[train_y == label], axis=0) for label in classes])
    distances = np.sum((target_z[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    predictions = np.asarray([classes[int(index)] for index in np.argmin(distances, axis=1)], dtype=object)

    accuracy = float(np.mean(predictions == target_y)) if len(target_y) else 0.0
    target_classes = tuple(sorted({str(label) for label in target_y.tolist()}))
    recalls = []
    for label in target_classes:
        mask = target_y == label
        if label not in classes:
            recalls.append(0.0)
        else:
            recalls.append(float(np.mean(predictions[mask] == label)))
    balanced_accuracy = float(np.mean(recalls)) if recalls else 0.0
    probabilities = _softmax(-distances)
    class_index = {label: index for index, label in enumerate(classes)}
    eps = 1.0e-12
    true_probabilities = [
        float(probabilities[index, class_index[str(label)]]) if str(label) in class_index else eps
        for index, label in enumerate(target_y)
    ]
    negative_log_likelihood = float(np.mean([-np.log(max(probability, eps)) for probability in true_probabilities]))
    known_target_fraction = float(np.mean([str(label) in class_index for label in target_y])) if len(target_y) else 0.0
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "negative_log_likelihood": negative_log_likelihood,
        "train_class_count": int(len(classes)),
        "target_class_count": int(len(target_classes)),
        "known_target_fraction": known_target_fraction,
    }


def ridge_classification_metrics(
    clips_by_id: dict[str, BenchmarkClip],
    *,
    labels_by_id: dict[str, str],
    train_ids: Sequence[str],
    target_ids: Sequence[str],
    representation: str,
    alpha: float = 1.0,
) -> dict[str, float | int]:
    train_ids = tuple(str(sample_id) for sample_id in train_ids if str(sample_id) in labels_by_id)
    target_ids = tuple(str(sample_id) for sample_id in target_ids if str(sample_id) in labels_by_id)
    if not train_ids or not target_ids:
        return _empty_metrics()

    train_x = stack_embeddings(clips_by_id, train_ids, representation=representation)
    target_x = stack_embeddings(clips_by_id, target_ids, representation=representation)
    train_y = np.asarray([labels_by_id[sample_id] for sample_id in train_ids], dtype=object)
    target_y = np.asarray([labels_by_id[sample_id] for sample_id in target_ids], dtype=object)
    classes = tuple(sorted({str(label) for label in train_y.tolist()}))
    if not classes:
        return _empty_metrics()

    mean = np.mean(train_x, axis=0, keepdims=True)
    scale = np.std(train_x, axis=0, keepdims=True)
    scale = np.where(scale > 1.0e-9, scale, 1.0)
    train_z = (train_x - mean) / scale
    target_z = (target_x - mean) / scale
    design = np.hstack([train_z, np.ones((train_z.shape[0], 1), dtype=float)])
    target_design = np.hstack([target_z, np.ones((target_z.shape[0], 1), dtype=float)])
    class_index = {label: index for index, label in enumerate(classes)}
    y = np.zeros((design.shape[0], len(classes)), dtype=float)
    for row_index, label in enumerate(train_y):
        y[row_index, class_index[str(label)]] = 1.0
    penalty = np.eye(design.shape[1], dtype=float) * float(alpha)
    penalty[-1, -1] = 0.0
    weights = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
    scores = target_design @ weights
    predictions = np.asarray([classes[int(index)] for index in np.argmax(scores, axis=1)], dtype=object)

    accuracy = float(np.mean(predictions == target_y)) if len(target_y) else 0.0
    target_classes = tuple(sorted({str(label) for label in target_y.tolist()}))
    recalls = []
    for label in target_classes:
        mask = target_y == label
        if label not in classes:
            recalls.append(0.0)
        else:
            recalls.append(float(np.mean(predictions[mask] == label)))
    balanced_accuracy = float(np.mean(recalls)) if recalls else 0.0
    probabilities = _softmax(scores)
    eps = 1.0e-12
    true_probabilities = [
        float(probabilities[index, class_index[str(label)]]) if str(label) in class_index else eps
        for index, label in enumerate(target_y)
    ]
    negative_log_likelihood = float(np.mean([-np.log(max(probability, eps)) for probability in true_probabilities]))
    known_target_fraction = float(np.mean([str(label) in class_index for label in target_y])) if len(target_y) else 0.0
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "negative_log_likelihood": negative_log_likelihood,
        "train_class_count": int(len(classes)),
        "target_class_count": int(len(target_classes)),
        "known_target_fraction": known_target_fraction,
    }


def supervised_classification_metrics(
    clips_by_id: dict[str, BenchmarkClip],
    *,
    labels_by_id: dict[str, str],
    train_ids: Sequence[str],
    target_ids: Sequence[str],
    representation: str,
    model_name: str,
) -> dict[str, float | int]:
    if str(model_name) == "nearest_centroid":
        return nearest_centroid_classification_metrics(
            clips_by_id,
            labels_by_id=labels_by_id,
            train_ids=train_ids,
            target_ids=target_ids,
            representation=representation,
        )
    if str(model_name) == "ridge_classifier":
        return ridge_classification_metrics(
            clips_by_id,
            labels_by_id=labels_by_id,
            train_ids=train_ids,
            target_ids=target_ids,
            representation=representation,
        )
    raise ValueError(f"Unsupported downstream supervised model: {model_name}")


def summarize_downstream_supervised(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    rows_list = [dict(row) for row in rows]
    final_rows = _final_rows(rows_list)
    return {
        "row_count": int(len(rows_list)),
        "final_row_count": int(len(final_rows)),
        "policy_final_means": _policy_means(final_rows),
        "policy_round_means": _policy_round_means(rows_list),
        "final_episode_wins": _final_episode_wins(final_rows),
    }


def build_downstream_supervised_report(
    result: BenchmarkResult,
    clips: Sequence[BenchmarkClip],
    *,
    downstream_representations: Sequence[str],
    label_source: str = "source_family",
    label_representation: str = "window",
    source_family_count: int = 4,
    baseline_policy: str = "old_novelty_ts2vec",
    random_policy: str = "random_valid",
) -> dict[str, object]:
    rows = build_downstream_supervised_rows(
        result,
        clips,
        downstream_representations=downstream_representations,
        label_source=label_source,
        label_representation=label_representation,
        source_family_count=source_family_count,
    )
    summary = summarize_downstream_supervised(rows)
    return {
        "input": {
            "episode_count": int(len(result.episodes)),
            "round_count": int(len(result.rounds)),
            "policies": list(result.policies),
            "downstream_representations": [str(rep) for rep in downstream_representations],
            "label_source": str(label_source),
            "label_representation": str(label_representation),
            "source_family_count": int(source_family_count),
            "baseline_policy": str(baseline_policy),
            "random_policy": str(random_policy),
        },
        "summary": summary,
        "decision": _decision(summary, baseline_policy=str(baseline_policy), random_policy=str(random_policy)),
        "rows": rows,
    }


def write_downstream_supervised_reports(report: dict[str, object], output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "downstream_supervised_smoke_report.json"
    markdown_path = root / "downstream_supervised_smoke_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def _labels_by_clip_id(
    clips: Sequence[BenchmarkClip],
    *,
    label_source: str,
    label_representation: str,
    source_family_count: int,
) -> dict[str, str]:
    if str(label_source) != "source_family":
        raise ValueError(f"Unsupported supervised smoke label source: {label_source}")
    family_by_group = infer_source_family_assignments(
        clips,
        representation=str(label_representation),
        source_family_count=int(source_family_count),
    )
    return {clip.sample_id: str(family_by_group[str(clip.source_group_id)]) for clip in clips}


def _empty_metrics() -> dict[str, float | int]:
    return {
        "accuracy": 0.0,
        "balanced_accuracy": 0.0,
        "negative_log_likelihood": 0.0,
        "train_class_count": 0,
        "target_class_count": 0,
        "known_target_fraction": 0.0,
    }


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(values)
    denom = np.sum(exp_values, axis=1, keepdims=True)
    return exp_values / np.where(denom > 0.0, denom, 1.0)


def _final_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    max_round_by_key: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row["episode_id"]), str(row["policy_name"]))
        max_round_by_key[key] = max(max_round_by_key.get(key, -1), int(row["round_index"]))
    return [
        dict(row)
        for row in rows
        if int(row["round_index"]) == max_round_by_key[(str(row["episode_id"]), str(row["policy_name"]))]
    ]


def _policy_means(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, float | int]]:
    by_policy: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_policy[str(row["policy_name"])].append(dict(row))
    return {policy: _mean_row_values(policy_rows) for policy, policy_rows in sorted(by_policy.items())}


def _policy_round_means(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, dict[str, float | int]]]:
    by_policy_round: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_policy_round[str(row["policy_name"])][str(int(row["round_index"]))].append(dict(row))
    return {
        policy: {round_index: _mean_row_values(round_rows) for round_index, round_rows in sorted(round_map.items())}
        for policy, round_map in sorted(by_policy_round.items())
    }


def _mean_row_values(rows: Sequence[dict[str, object]]) -> dict[str, float | int]:
    return {
        "row_count": int(len(rows)),
        "mean_baseline_accuracy": _mean(rows, "baseline_accuracy"),
        "mean_after_accuracy": _mean(rows, "after_accuracy"),
        "mean_accuracy_gain": _mean(rows, "accuracy_gain"),
        "mean_baseline_balanced_accuracy": _mean(rows, "baseline_balanced_accuracy"),
        "mean_after_balanced_accuracy": _mean(rows, "after_balanced_accuracy"),
        "mean_balanced_accuracy_gain": _mean(rows, "balanced_accuracy_gain"),
        "mean_baseline_negative_log_likelihood": _mean(rows, "baseline_negative_log_likelihood"),
        "mean_after_negative_log_likelihood": _mean(rows, "after_negative_log_likelihood"),
        "mean_nll_reduction": _mean(rows, "nll_reduction"),
        "mean_relative_nll_reduction": _mean(rows, "relative_nll_reduction"),
        "mean_baseline_known_target_fraction": _mean(rows, "baseline_known_target_fraction"),
        "mean_after_known_target_fraction": _mean(rows, "after_known_target_fraction"),
    }


def _mean(rows: Sequence[dict[str, object]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in rows]
    return float(np.mean(values)) if values else 0.0


def _final_episode_wins(rows: Sequence[dict[str, object]]) -> dict[str, int]:
    by_episode_rep: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_episode_rep[(str(row["episode_id"]), str(row["representation"]))].append(dict(row))
    wins: dict[str, int] = defaultdict(int)
    for group in by_episode_rep.values():
        best = max(float(row["accuracy_gain"]) for row in group)
        for row in group:
            if abs(float(row["accuracy_gain"]) - best) <= 1.0e-12:
                wins[str(row["policy_name"])] += 1
    return dict(sorted(wins.items()))


def _decision(summary: dict[str, object], *, baseline_policy: str, random_policy: str) -> dict[str, object]:
    policy_means = summary.get("policy_final_means", {})
    if not isinstance(policy_means, dict):
        policy_means = {}
    baseline = policy_means.get(baseline_policy, {})
    random = policy_means.get(random_policy, {})
    baseline_accuracy_gain = _metric_value(baseline, "mean_accuracy_gain")
    random_accuracy_gain = _metric_value(random, "mean_accuracy_gain")
    baseline_nll_reduction = _metric_value(baseline, "mean_nll_reduction")
    random_nll_reduction = _metric_value(random, "mean_nll_reduction")
    accuracy_delta = (
        baseline_accuracy_gain - random_accuracy_gain
        if baseline_accuracy_gain is not None and random_accuracy_gain is not None
        else None
    )
    nll_delta = (
        baseline_nll_reduction - random_nll_reduction
        if baseline_nll_reduction is not None and random_nll_reduction is not None
        else None
    )
    read = "source-family pseudo-label supervised smoke ran; this is not real downstream task-label proof"
    if baseline_policy in policy_means and random_policy in policy_means:
        if (accuracy_delta is not None and accuracy_delta > 0.0) or (nll_delta is not None and nll_delta > 0.0):
            read = (
                f"{baseline_policy} beat {random_policy} on this source-family pseudo-label supervised smoke, "
                "but this is still not real downstream task-label proof"
            )
        else:
            read = (
                f"{baseline_policy} did not beat {random_policy} on this source-family pseudo-label supervised smoke, "
                "so hold downstream scaling"
            )
    return {
        "downstream_training": "tiny_smoke_only",
        "read": read,
        "baseline_policy": baseline_policy,
        "random_policy": random_policy,
        "baseline_mean_accuracy_gain": baseline_accuracy_gain,
        "random_mean_accuracy_gain": random_accuracy_gain,
        "baseline_minus_random_accuracy_gain": accuracy_delta,
        "baseline_mean_nll_reduction": baseline_nll_reduction,
        "random_mean_nll_reduction": random_nll_reduction,
        "baseline_minus_random_nll_reduction": nll_delta,
        "next_steps": [
            "Use this only as proof that acquisition choices can feed a supervised model update loop.",
            "Do not present source-family pseudo-labels as challenge labels.",
            "Only scale to real downstream training after choosing a real held-out utility target.",
        ],
    }


def _metric_value(row: object, key: str) -> float | None:
    if not isinstance(row, dict):
        return None
    if key not in row:
        return None
    return float(row[key])


def _markdown_report(report: dict[str, object]) -> str:
    summary = report.get("summary", {})
    decision = report.get("decision", {})
    policy_means = summary.get("policy_final_means", {}) if isinstance(summary, dict) else {}
    lines = [
        "# Downstream Supervised Smoke",
        "",
        "This is a tiny source-family pseudo-label supervised model-update smoke, not real downstream task-label proof.",
        "",
        "## Decision",
        "",
        f"- downstream training: `{decision.get('downstream_training')}`",
        f"- read: {decision.get('read')}",
        f"- baseline minus random accuracy gain: `{_fmt(decision.get('baseline_minus_random_accuracy_gain'))}`",
        f"- baseline minus random NLL reduction: `{_fmt(decision.get('baseline_minus_random_nll_reduction'))}`",
        "",
        "## Policy Means",
        "",
    ]
    if isinstance(policy_means, dict):
        for policy, values in sorted(policy_means.items()):
            if not isinstance(values, dict):
                continue
            lines.extend(
                [
                    f"### {policy}",
                    f"- after accuracy: `{_fmt(values.get('mean_after_accuracy'))}`",
                    f"- accuracy gain: `{_fmt(values.get('mean_accuracy_gain'))}`",
                    f"- NLL reduction: `{_fmt(values.get('mean_nll_reduction'))}`",
                    f"- after known target fraction: `{_fmt(values.get('mean_after_known_target_fraction'))}`",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return str(value)
