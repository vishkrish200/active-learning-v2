from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def build_downstream_supervised_autopsy(
    benchmark_report: Mapping[str, Any],
    supervised_report: Mapping[str, Any],
    *,
    baseline_policy: str | None = None,
    random_policy: str | None = None,
    accuracy_flat_tolerance: float = 1.0e-9,
) -> dict[str, object]:
    input_block = supervised_report.get("input", {})
    if not isinstance(input_block, Mapping):
        input_block = {}
    baseline = str(baseline_policy or input_block.get("baseline_policy", "old_novelty_ts2vec"))
    random = str(random_policy or input_block.get("random_policy", "random_valid"))
    policy_means = _policy_final_means(supervised_report)
    flatness = _flatness(policy_means, accuracy_flat_tolerance=float(accuracy_flat_tolerance))
    target_label_coverage = _target_label_coverage(supervised_report)
    candidate_opportunity = _candidate_opportunity(benchmark_report)
    selection_audit = _selection_audit(benchmark_report)
    comparison = _policy_comparison(policy_means, baseline_policy=baseline, random_policy=random)

    diagnosis: list[str] = []
    if flatness["accuracy_flat"]:
        diagnosis.append("accuracy_flat_across_policies")
    if flatness["accuracy_gain_flat"]:
        diagnosis.append("accuracy_gain_flat_across_policies")
    if target_label_coverage["all_baseline_known_target_fraction_one"]:
        diagnosis.append("target_labels_already_known_before_acquisition")
    if candidate_opportunity["all_candidate_farther_than_support"]:
        diagnosis.append("candidate_pool_not_closer_than_support")
    if comparison.get("baseline_minus_random_nll_reduction", 0.0) < 0.0:
        diagnosis.append("baseline_loses_random_on_nll_reduction")
    if not diagnosis:
        diagnosis.append("no_obvious_flatness_failure")

    next_gate = "build_harder_supervised_gate" if diagnosis != ["no_obvious_flatness_failure"] else "eligible_for_small_repeat"
    read = _decision_read(diagnosis, baseline=baseline, random=random)
    return {
        "input": {
            "baseline_policy": baseline,
            "random_policy": random,
            "downstream_representations": list(input_block.get("downstream_representations", [])),
        },
        "decision": {
            "downstream_training": "hold",
            "next_gate": next_gate,
            "read": read,
        },
        "diagnosis": diagnosis,
        "policy_comparison": comparison,
        "flatness": flatness,
        "target_label_coverage": target_label_coverage,
        "candidate_opportunity": candidate_opportunity,
        "selection_audit": selection_audit,
    }


def write_downstream_supervised_autopsy_reports(autopsy: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "downstream_supervised_autopsy_report.json"
    markdown_path = root / "downstream_supervised_autopsy_report.md"
    json_path.write_text(json.dumps(autopsy, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_markdown_report(autopsy), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def _policy_final_means(supervised_report: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    summary = supervised_report.get("summary", {})
    if not isinstance(summary, Mapping):
        return {}
    policy_means = summary.get("policy_final_means", {})
    if not isinstance(policy_means, Mapping):
        return {}
    out: dict[str, dict[str, float]] = {}
    for policy, values in policy_means.items():
        if not isinstance(values, Mapping):
            continue
        out[str(policy)] = {
            key: float(value)
            for key, value in values.items()
            if isinstance(value, (int, float))
        }
    return out


def _flatness(policy_means: Mapping[str, Mapping[str, float]], *, accuracy_flat_tolerance: float) -> dict[str, object]:
    after_accuracy = {
        policy: float(values.get("mean_after_accuracy", 0.0))
        for policy, values in policy_means.items()
    }
    accuracy_gain = {
        policy: float(values.get("mean_accuracy_gain", 0.0))
        for policy, values in policy_means.items()
    }
    after_values = list(after_accuracy.values())
    gain_values = list(accuracy_gain.values())
    after_range = max(after_values) - min(after_values) if after_values else 0.0
    gain_range = max(gain_values) - min(gain_values) if gain_values else 0.0
    return {
        "policy_after_accuracy": after_accuracy,
        "policy_accuracy_gain": accuracy_gain,
        "after_accuracy_range": float(after_range),
        "accuracy_gain_range": float(gain_range),
        "accuracy_flat": bool(after_values and after_range <= accuracy_flat_tolerance),
        "accuracy_gain_flat": bool(gain_values and gain_range <= accuracy_flat_tolerance),
    }


def _target_label_coverage(supervised_report: Mapping[str, Any]) -> dict[str, object]:
    rows = supervised_report.get("rows", [])
    if not isinstance(rows, Sequence):
        rows = []
    fractions = [
        float(row.get("baseline_known_target_fraction", 0.0))
        for row in rows
        if isinstance(row, Mapping) and "baseline_known_target_fraction" in row
    ]
    return {
        "row_count": int(len(fractions)),
        "mean_baseline_known_target_fraction": float(np.mean(fractions)) if fractions else 0.0,
        "min_baseline_known_target_fraction": float(min(fractions)) if fractions else 0.0,
        "all_baseline_known_target_fraction_one": bool(fractions and min(fractions) >= 1.0 - 1.0e-9),
    }


def _candidate_opportunity(benchmark_report: Mapping[str, Any]) -> dict[str, object]:
    audit_rows = benchmark_report.get("difficulty_audit", [])
    if not isinstance(audit_rows, Sequence):
        audit_rows = []
    rows: list[dict[str, object]] = []
    for audit in audit_rows:
        if not isinstance(audit, Mapping):
            continue
        baseline = audit.get("support_target_baseline_distance_by_representation", {})
        candidate = audit.get("candidate_target_nearest_distance_by_representation", {})
        if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
            continue
        for representation in sorted(set(baseline) & set(candidate)):
            support_distance = float(baseline[representation])
            candidate_distance = float(candidate[representation])
            rows.append(
                {
                    "episode_id": str(audit.get("episode_id", "")),
                    "representation": str(representation),
                    "support_target_distance": support_distance,
                    "candidate_target_nearest_distance": candidate_distance,
                    "candidate_closer_than_support": bool(candidate_distance < support_distance),
                    "candidate_minus_support": float(candidate_distance - support_distance),
                }
            )
    return {
        "rows": rows,
        "candidate_closer_fraction": float(np.mean([row["candidate_closer_than_support"] for row in rows])) if rows else 0.0,
        "all_candidate_farther_than_support": bool(rows and not any(row["candidate_closer_than_support"] for row in rows)),
    }


def _selection_audit(benchmark_report: Mapping[str, Any]) -> list[dict[str, object]]:
    rounds = benchmark_report.get("rounds", [])
    if not isinstance(rounds, Sequence):
        return []
    final_round_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rounds:
        if not isinstance(row, Mapping):
            continue
        key = (str(row.get("episode_id", "")), str(row.get("policy_name", "")))
        if not key[0] or not key[1]:
            continue
        if key not in final_round_by_key or int(row.get("round_index", -1)) > int(final_round_by_key[key].get("round_index", -1)):
            final_round_by_key[key] = row
    out = []
    for (episode_id, policy), row in sorted(final_round_by_key.items()):
        groups = [str(group) for group in row.get("selected_source_group_ids", [])]
        selected = [str(sample_id) for sample_id in row.get("selected_ids", [])]
        group_counts: dict[str, int] = defaultdict(int)
        for group in groups:
            group_counts[group] += 1
        out.append(
            {
                "episode_id": episode_id,
                "policy_name": policy,
                "round_index": int(row.get("round_index", -1)),
                "selected_count": int(len(selected)),
                "unique_selected_source_groups": int(len(group_counts)),
                "largest_selected_source_group_fraction": float(max(group_counts.values()) / len(groups)) if groups else 0.0,
                "selected_source_group_ids": groups,
                "selected_ids": selected,
            }
        )
    return out


def _policy_comparison(
    policy_means: Mapping[str, Mapping[str, float]],
    *,
    baseline_policy: str,
    random_policy: str,
) -> dict[str, object]:
    baseline = policy_means.get(baseline_policy, {})
    random = policy_means.get(random_policy, {})
    baseline_accuracy_gain = float(baseline.get("mean_accuracy_gain", 0.0))
    random_accuracy_gain = float(random.get("mean_accuracy_gain", 0.0))
    baseline_nll = float(baseline.get("mean_nll_reduction", 0.0))
    random_nll = float(random.get("mean_nll_reduction", 0.0))
    return {
        "baseline_policy": baseline_policy,
        "random_policy": random_policy,
        "baseline_accuracy_gain": baseline_accuracy_gain,
        "random_accuracy_gain": random_accuracy_gain,
        "baseline_minus_random_accuracy_gain": float(baseline_accuracy_gain - random_accuracy_gain),
        "baseline_nll_reduction": baseline_nll,
        "random_nll_reduction": random_nll,
        "baseline_minus_random_nll_reduction": float(baseline_nll - random_nll),
    }


def _decision_read(diagnosis: Sequence[str], *, baseline: str, random: str) -> str:
    if "accuracy_flat_across_policies" in diagnosis:
        return (
            "Hold downstream training: supervised accuracy is flat across policies, so the task is not "
            f"discriminating {baseline} from {random}."
        )
    if "target_labels_already_known_before_acquisition" in diagnosis:
        return "Hold downstream training: target labels are already represented before acquisition."
    if "candidate_pool_not_closer_than_support" in diagnosis:
        return "Hold downstream training: candidates are not a better target bridge than existing support."
    if "baseline_loses_random_on_nll_reduction" in diagnosis:
        return f"Hold downstream training: {baseline} loses to {random} on NLL reduction."
    return "No obvious flatness failure found; a small repeat may be eligible."


def _markdown_report(autopsy: Mapping[str, Any]) -> str:
    decision = autopsy.get("decision", {})
    diagnosis = autopsy.get("diagnosis", [])
    comparison = autopsy.get("policy_comparison", {})
    flatness = autopsy.get("flatness", {})
    coverage = autopsy.get("target_label_coverage", {})
    opportunity = autopsy.get("candidate_opportunity", {})
    lines = [
        "# Downstream Supervised Autopsy",
        "",
        "## Decision",
        "",
        f"- downstream training: `{_get(decision, 'downstream_training')}`",
        f"- next gate: `{_get(decision, 'next_gate')}`",
        f"- read: {_get(decision, 'read')}",
        "",
        "## Diagnosis",
        "",
    ]
    for item in diagnosis if isinstance(diagnosis, Sequence) and not isinstance(diagnosis, str) else []:
        lines.append(f"- `{item}`")
    lines.extend(
        [
            "",
            "## Policy Comparison",
            "",
            f"- baseline policy: `{_get(comparison, 'baseline_policy')}`",
            f"- random policy: `{_get(comparison, 'random_policy')}`",
            f"- baseline minus random accuracy gain: `{_fmt(_get(comparison, 'baseline_minus_random_accuracy_gain'))}`",
            f"- baseline minus random NLL reduction: `{_fmt(_get(comparison, 'baseline_minus_random_nll_reduction'))}`",
            "",
            "## Flatness",
            "",
            f"- accuracy flat: `{_get(flatness, 'accuracy_flat')}`",
            f"- accuracy gain flat: `{_get(flatness, 'accuracy_gain_flat')}`",
            f"- after accuracy range: `{_fmt(_get(flatness, 'after_accuracy_range'))}`",
            f"- accuracy gain range: `{_fmt(_get(flatness, 'accuracy_gain_range'))}`",
            "",
            "## Target Label Coverage",
            "",
            f"- mean baseline known target fraction: `{_fmt(_get(coverage, 'mean_baseline_known_target_fraction'))}`",
            f"- all targets known before acquisition: `{_get(coverage, 'all_baseline_known_target_fraction_one')}`",
            "",
            "## Candidate Opportunity",
            "",
            f"- candidate closer fraction: `{_fmt(_get(opportunity, 'candidate_closer_fraction'))}`",
            f"- all candidates farther than support: `{_get(opportunity, 'all_candidate_farther_than_support')}`",
            "",
            "| episode | representation | support-target distance | candidate-target distance | candidate closer |",
            "|---|---|---:|---:|---|",
        ]
    )
    rows = opportunity.get("rows", []) if isinstance(opportunity, Mapping) else []
    for row in rows if isinstance(rows, Sequence) else []:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {episode} | {rep} | {support:.6f} | {candidate:.6f} | `{closer}` |".format(
                episode=row.get("episode_id", ""),
                rep=row.get("representation", ""),
                support=float(row.get("support_target_distance", 0.0)),
                candidate=float(row.get("candidate_target_nearest_distance", 0.0)),
                closer=bool(row.get("candidate_closer_than_support", False)),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _get(mapping: object, key: str) -> object:
    if isinstance(mapping, Mapping):
        return mapping.get(key)
    return None


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return str(value)
