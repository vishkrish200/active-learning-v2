from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from marginal_value.active_benchmark.coverage_runner import CoverageRunResult
from marginal_value.active_benchmark.downstream_supervised_smoke import (
    _labels_by_clip_id,
    nearest_centroid_classification_metrics,
)
from marginal_value.active_benchmark.representations import clip_lookup
from marginal_value.active_benchmark.schema import BenchmarkClip


def build_downstream_coverage_supervised_rows(
    result: CoverageRunResult,
    clips: Sequence[BenchmarkClip],
    *,
    downstream_representations: Sequence[str],
    label_source: str = "source_family",
    label_representation: str = "window",
    source_family_count: int = 4,
) -> list[dict[str, object]]:
    clips_by_id = clip_lookup(clips)
    labels_by_id = _labels_by_clip_id(
        clips,
        label_source=label_source,
        label_representation=label_representation,
        source_family_count=source_family_count,
    )
    selected_by_key = _selected_ids_by_episode_policy_budget(result)
    budgets = (0, *tuple(int(budget) for budget in result.budgets))
    rows: list[dict[str, object]] = []
    for episode in result.episodes:
        target_labels = _labels_for_ids(labels_by_id, episode.target_ids)
        support_labels = _labels_for_ids(labels_by_id, episode.support_ids)
        candidate_bridge_ids = tuple(
            str(sample_id)
            for sample_id in episode.candidate_ids
            if labels_by_id.get(str(sample_id)) in target_labels
        )
        candidate_bridge_labels = _labels_for_ids(labels_by_id, candidate_bridge_ids)
        for policy_id in result.policies:
            for budget_k in budgets:
                selected_ids = selected_by_key.get((episode.episode_id, str(policy_id), int(budget_k)), ())
                support_ids_after = (*episode.support_ids, *selected_ids)
                selected_bridge_ids = tuple(
                    str(sample_id)
                    for sample_id in selected_ids
                    if labels_by_id.get(str(sample_id)) in target_labels
                )
                selected_bridge_labels = _labels_for_ids(labels_by_id, selected_bridge_ids)
                after_known_labels = set(support_labels)
                after_known_labels.update(_labels_for_ids(labels_by_id, selected_ids))
                discovery_rate = float(len(selected_bridge_labels) / len(target_labels)) if target_labels else 0.0
                for representation in downstream_representations:
                    baseline = nearest_centroid_classification_metrics(
                        clips_by_id,
                        labels_by_id=labels_by_id,
                        train_ids=episode.support_ids,
                        target_ids=episode.target_ids,
                        representation=str(representation),
                    )
                    after = nearest_centroid_classification_metrics(
                        clips_by_id,
                        labels_by_id=labels_by_id,
                        train_ids=support_ids_after,
                        target_ids=episode.target_ids,
                        representation=str(representation),
                    )
                    accuracy_gain = float(after["accuracy"] - baseline["accuracy"])
                    balanced_accuracy_gain = float(after["balanced_accuracy"] - baseline["balanced_accuracy"])
                    nll_reduction = float(baseline["negative_log_likelihood"] - after["negative_log_likelihood"])
                    baseline_nll = float(baseline["negative_log_likelihood"])
                    rows.append(
                        {
                            "episode_id": str(episode.episode_id),
                            "fold_id": int(episode.fold_id),
                            "policy_id": str(policy_id),
                            "budget_k": int(budget_k),
                            "representation": str(representation),
                            "label_source": str(label_source),
                            "label_representation": str(label_representation),
                            "selected_ids": list(selected_ids),
                            "support_count_before": int(len(episode.support_ids)),
                            "support_count_after": int(len(support_ids_after)),
                            "target_count": int(len(episode.target_ids)),
                            "target_family_count": int(len(target_labels)),
                            "candidate_bridge_count": int(len(candidate_bridge_ids)),
                            "candidate_bridge_family_count": int(len(candidate_bridge_labels)),
                            "selected_bridge_count": int(len(selected_bridge_ids)),
                            "selected_bridge_family_count": int(len(selected_bridge_labels)),
                            "target_family_discovery_rate": float(discovery_rate),
                            "known_target_family_count_before": int(len(target_labels & support_labels)),
                            "known_target_family_count_after": int(len(target_labels & after_known_labels)),
                            "baseline_accuracy": float(baseline["accuracy"]),
                            "after_accuracy": float(after["accuracy"]),
                            "accuracy_gain": float(accuracy_gain),
                            "baseline_balanced_accuracy": float(baseline["balanced_accuracy"]),
                            "after_balanced_accuracy": float(after["balanced_accuracy"]),
                            "balanced_accuracy_gain": float(balanced_accuracy_gain),
                            "baseline_negative_log_likelihood": float(baseline["negative_log_likelihood"]),
                            "after_negative_log_likelihood": float(after["negative_log_likelihood"]),
                            "nll_reduction": float(nll_reduction),
                            "relative_nll_reduction": float(nll_reduction / baseline_nll) if baseline_nll > 1.0e-12 else 0.0,
                            "baseline_train_class_count": int(baseline["train_class_count"]),
                            "after_train_class_count": int(after["train_class_count"]),
                            "target_class_count": int(after["target_class_count"]),
                            "baseline_known_target_fraction": float(baseline["known_target_fraction"]),
                            "after_known_target_fraction": float(after["known_target_fraction"]),
                        }
                    )
    return rows


def build_downstream_coverage_supervised_report(
    result: CoverageRunResult,
    clips: Sequence[BenchmarkClip],
    *,
    downstream_representations: Sequence[str],
    label_source: str = "source_family",
    label_representation: str = "window",
    source_family_count: int = 4,
    top_policy: str = "ts2vec_kcenter_v1",
    baseline_policy: str = "quality_stratified_random_v1",
) -> dict[str, object]:
    rows = build_downstream_coverage_supervised_rows(
        result,
        clips,
        downstream_representations=downstream_representations,
        label_source=label_source,
        label_representation=label_representation,
        source_family_count=source_family_count,
    )
    summary = summarize_downstream_coverage_supervised(rows)
    return {
        "input": {
            "episode_count": int(len(result.episodes)),
            "policy_count": int(len(result.policies)),
            "policies": list(result.policies),
            "budgets": [0, *[int(budget) for budget in result.budgets]],
            "downstream_representations": [str(rep) for rep in downstream_representations],
            "label_source": str(label_source),
            "label_representation": str(label_representation),
            "source_family_count": int(source_family_count),
            "top_policy": str(top_policy),
            "baseline_policy": str(baseline_policy),
        },
        "summary": summary,
        "decision": _decision(summary, top_policy=str(top_policy), baseline_policy=str(baseline_policy)),
        "rows": rows,
    }


def summarize_downstream_coverage_supervised(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    rows_list = [dict(row) for row in rows]
    final_budget = max((int(row["budget_k"]) for row in rows_list), default=0)
    final_rows = [row for row in rows_list if int(row["budget_k"]) == final_budget]
    return {
        "row_count": int(len(rows_list)),
        "final_budget": int(final_budget),
        "final_row_count": int(len(final_rows)),
        "policy_budget_means": _policy_budget_means(rows_list),
        "policy_final_means": _policy_means(final_rows),
    }


def write_downstream_coverage_supervised_reports(report: dict[str, object], output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "downstream_coverage_supervised_smoke_report.json"
    markdown_path = root / "downstream_coverage_supervised_smoke_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def _selected_ids_by_episode_policy_budget(result: CoverageRunResult) -> dict[tuple[str, str, int], tuple[str, ...]]:
    ranked: dict[tuple[str, str, int], list[tuple[int, str]]] = defaultdict(list)
    for row in result.selected_rows:
        ranked[(str(row.episode_id), str(row.policy_id), int(row.budget_k))].append((int(row.rank_index), str(row.sample_id)))
    return {
        key: tuple(sample_id for _rank_index, sample_id in sorted(values))
        for key, values in ranked.items()
    }


def _policy_budget_means(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, dict[str, float | int]]]:
    by_policy_budget: dict[str, dict[int, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_policy_budget[str(row["policy_id"])][int(row["budget_k"])].append(dict(row))
    return {
        policy_id: {str(budget_k): _mean_row_values(budget_rows) for budget_k, budget_rows in sorted(budget_map.items())}
        for policy_id, budget_map in sorted(by_policy_budget.items())
    }


def _policy_means(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, float | int]]:
    by_policy: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_policy[str(row["policy_id"])].append(dict(row))
    return {policy_id: _mean_row_values(policy_rows) for policy_id, policy_rows in sorted(by_policy.items())}


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
        "mean_target_family_count": _mean(rows, "target_family_count"),
        "mean_candidate_bridge_count": _mean(rows, "candidate_bridge_count"),
        "mean_selected_bridge_count": _mean(rows, "selected_bridge_count"),
        "mean_target_family_discovery_rate": _mean(rows, "target_family_discovery_rate"),
        "mean_known_target_family_count_after": _mean(rows, "known_target_family_count_after"),
    }


def _mean(rows: Sequence[dict[str, object]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in rows]
    return float(np.mean(values)) if values else 0.0


def _decision(summary: dict[str, object], *, top_policy: str, baseline_policy: str) -> dict[str, object]:
    policy_means = summary.get("policy_final_means", {})
    if not isinstance(policy_means, dict):
        policy_means = {}
    top = policy_means.get(top_policy, {})
    baseline = policy_means.get(baseline_policy, {})
    top_gain = _metric_value(top, "mean_balanced_accuracy_gain")
    baseline_gain = _metric_value(baseline, "mean_balanced_accuracy_gain")
    top_nll = _metric_value(top, "mean_nll_reduction")
    baseline_nll = _metric_value(baseline, "mean_nll_reduction")
    top_discovery = _metric_value(top, "mean_target_family_discovery_rate")
    baseline_discovery = _metric_value(baseline, "mean_target_family_discovery_rate")
    balanced_accuracy_delta = top_gain - baseline_gain if top_gain is not None and baseline_gain is not None else None
    nll_delta = top_nll - baseline_nll if top_nll is not None and baseline_nll is not None else None
    discovery_delta = top_discovery - baseline_discovery if top_discovery is not None and baseline_discovery is not None else None
    read = (
        "source-family pseudo-label bridge downstream benchmark ran; keep this as an identifiability gate and do "
        "not treat it as real challenge-label downstream proof."
    )
    return {
        "downstream_training": "hold_large_training",
        "read": read,
        "top_policy": str(top_policy),
        "baseline_policy": str(baseline_policy),
        "top_mean_balanced_accuracy_gain": top_gain,
        "baseline_mean_balanced_accuracy_gain": baseline_gain,
        "balanced_accuracy_delta_vs_baseline": balanced_accuracy_delta,
        "top_mean_target_family_discovery_rate": top_discovery,
        "baseline_mean_target_family_discovery_rate": baseline_discovery,
        "target_family_discovery_delta_vs_baseline": discovery_delta,
        "top_mean_nll_reduction": top_nll,
        "baseline_mean_nll_reduction": baseline_nll,
        "nll_delta_vs_baseline": nll_delta,
        "next_steps": [
            "Use this only to decide whether the bridge-candidate downstream probe is identifiable and non-flat.",
            "Require the TS2Vec k-center policy to beat quality-stratified random before any larger downstream run.",
            "Do not retrain TS2Vec from this smoke.",
        ],
    }


def _metric_value(row: object, key: str) -> float | None:
    if not isinstance(row, dict) or key not in row:
        return None
    return float(row[key])


def _markdown_report(report: dict[str, object]) -> str:
    input_block = report.get("input", {})
    summary = report.get("summary", {})
    decision = report.get("decision", {})
    policy_budget_means = summary.get("policy_budget_means", {}) if isinstance(summary, dict) else {}
    lines = [
        "# Downstream Coverage Supervised Smoke",
        "",
        "This is a source-family pseudo-label nearest-centroid smoke on frozen embeddings, not real challenge-label downstream proof.",
        "",
        "## Decision",
        "",
        f"- downstream training: `{decision.get('downstream_training')}`",
        f"- read: {decision.get('read')}",
        f"- top policy: `{decision.get('top_policy')}`",
        f"- baseline policy: `{decision.get('baseline_policy')}`",
        f"- target-family discovery delta vs baseline: `{_fmt(decision.get('target_family_discovery_delta_vs_baseline'))}`",
        f"- balanced accuracy delta vs baseline: `{_fmt(decision.get('balanced_accuracy_delta_vs_baseline'))}`",
        f"- NLL delta vs baseline: `{_fmt(decision.get('nll_delta_vs_baseline'))}`",
        "",
        "## Run",
        "",
        f"- episodes: `{input_block.get('episode_count')}`",
        f"- budgets: `{', '.join(str(item) for item in input_block.get('budgets', []))}`",
        f"- downstream representations: `{', '.join(str(item) for item in input_block.get('downstream_representations', []))}`",
        "",
        "## Budget Means",
        "",
        "| policy | budget | discovery rate | bridge clips | balanced accuracy gain | NLL reduction | after known target frac | rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if isinstance(policy_budget_means, dict):
        for policy_id, budget_map in sorted(policy_budget_means.items()):
            if not isinstance(budget_map, dict):
                continue
            for budget_k, values in sorted(budget_map.items(), key=lambda item: int(item[0])):
                if not isinstance(values, dict):
                    continue
                lines.append(
                    "| {policy} | {budget} | {disc} | {bridges} | {bal} | {nll} | {known} | {rows} |".format(
                        policy=policy_id,
                        budget=budget_k,
                        disc=_fmt(values.get("mean_target_family_discovery_rate")),
                        bridges=_fmt(values.get("mean_selected_bridge_count")),
                        bal=_fmt(values.get("mean_balanced_accuracy_gain")),
                        nll=_fmt(values.get("mean_nll_reduction")),
                        known=_fmt(values.get("mean_after_known_target_fraction")),
                        rows=int(values.get("row_count", 0)),
                    )
                )
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "- Budget `0` is the support-only baseline. Later budgets add selected candidate clips to support before evaluating the pseudo-label probe.",
            "- This checks whether acquisition changes a tiny downstream classifier/probe; it is still not a learned query policy or TS2Vec retraining proof.",
        ]
    )
    return "\n".join(lines) + "\n"


def _labels_for_ids(labels_by_id: dict[str, str], sample_ids: Sequence[str]) -> set[str]:
    return {str(labels_by_id[str(sample_id)]) for sample_id in sample_ids if str(sample_id) in labels_by_id}


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (float, int)):
        return f"{float(value):.6f}"
    return str(value)
