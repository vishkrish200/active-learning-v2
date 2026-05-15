from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

from marginal_value.active_benchmark.selection_mechanism_audit import (
    DEFAULT_COMPARISON_POLICIES,
    DEFAULT_FOCAL_POLICY,
    _load_run_source,
    _selected_by_policy_unit,
    _selected_ids,
)


MOTION_FEATURES = (
    "mean_selected_motion_energy",
    "mean_selected_gyro_norm_p95",
    "mean_selected_acc_delta_norm_p95",
    "mean_selected_gyro_delta_norm_p95",
    "mean_selected_gyro_energy_fraction",
    "mean_selected_stationary_fraction",
    "mean_selected_rotation_dominant_rate",
    "mean_selected_high_dynamic_rate",
)

OUTCOMES = ("after_mse", "relative_mse_reduction")


def build_motion_outcome_link_audit(
    run_input: str | Path,
    motion_audit_json: str | Path,
    *,
    budget_k: int | None = None,
    focal_policy: str = DEFAULT_FOCAL_POLICY,
    comparison_policies: Sequence[str] = DEFAULT_COMPARISON_POLICIES,
) -> dict[str, Any]:
    source = _load_run_source(Path(run_input))
    motion_audit = json.loads(Path(motion_audit_json).read_text(encoding="utf-8"))
    clip_features = _clip_features_by_id(motion_audit)
    selected_rows = source["selected_rows"]
    forecast_rows = source["forecast_rows"]
    if not selected_rows:
        raise ValueError(f"No selected rows found in {run_input}")
    if not forecast_rows:
        raise ValueError(f"No downstream forecast rows found in {run_input}")
    if not clip_features:
        raise ValueError(f"No clip motion features found in {motion_audit_json}")

    final_budget = int(budget_k) if budget_k is not None else max(int(row["budget_k"]) for row in selected_rows)
    selected_rows = [row for row in selected_rows if int(row.get("budget_k", -1)) == final_budget]
    forecast_rows = [row for row in forecast_rows if int(row.get("budget_k", -1)) == final_budget]
    policy_unit_rows = _policy_unit_motion_outcome_rows(selected_rows, forecast_rows, clip_features)
    if not policy_unit_rows:
        raise ValueError("No policy-unit rows could be joined between selected clips, motion features, and forecast outcomes")

    return {
        "input": {
            "run_input": str(run_input),
            "motion_audit_json": str(motion_audit_json),
            "source_kind": source["source_kind"],
            "seed_count": len(source["seed_names"]),
            "seed_names": source["seed_names"],
            "budget_k": final_budget,
            "policy_unit_row_count": len(policy_unit_rows),
            "clip_motion_feature_count": len(clip_features),
            "focal_policy": focal_policy,
            "comparison_policies": list(comparison_policies),
        },
        "policy_unit_motion_outcomes": policy_unit_rows,
        "policy_motion_outcome_profiles": _policy_motion_outcome_profiles(policy_unit_rows),
        "motion_outcome_associations": _motion_outcome_associations(policy_unit_rows),
        "focal_pairwise_motion_outcome_contrasts": _focal_pairwise_motion_outcome_contrasts(
            policy_unit_rows,
            focal_policy=focal_policy,
            comparison_policies=comparison_policies,
        ),
        "read": _interpret_link(
            policy_unit_rows,
            focal_policy=focal_policy,
            comparison_policies=comparison_policies,
        ),
    }


def write_motion_outcome_link_audit_reports(
    run_input: str | Path,
    motion_audit_json: str | Path,
    *,
    output_json: str | Path,
    output_markdown: str | Path,
    budget_k: int | None = None,
    focal_policy: str = DEFAULT_FOCAL_POLICY,
    comparison_policies: Sequence[str] = DEFAULT_COMPARISON_POLICIES,
) -> dict[str, str]:
    audit = build_motion_outcome_link_audit(
        run_input,
        motion_audit_json,
        budget_k=budget_k,
        focal_policy=focal_policy,
        comparison_policies=comparison_policies,
    )
    json_path = Path(output_json)
    markdown_path = Path(output_markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_motion_outcome_link_audit_markdown(audit), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def render_motion_outcome_link_audit_markdown(audit: Mapping[str, Any]) -> str:
    input_info = audit.get("input", {})
    lines = [
        "# Motion Outcome Link Audit",
        "",
        "## Scope",
        "",
        f"- run input: `{input_info.get('run_input')}`",
        f"- motion audit: `{input_info.get('motion_audit_json')}`",
        f"- seeds: `{input_info.get('seed_count')}`",
        f"- budget: `K={input_info.get('budget_k')}`",
        f"- policy-unit rows: `{input_info.get('policy_unit_row_count')}`",
        f"- focal policy: `{input_info.get('focal_policy')}`",
        "",
        "## Read",
        "",
    ]
    lines.extend(f"- {item}" for item in audit.get("read", []))
    lines.extend(
        [
            "",
            "## Policy Motion-Outcome Profiles",
            "",
            "| policy | rows | after MSE | rel MSE reduction | selected dynamic energy | gyro p95 | rotation rate | high-dynamic rate | missing selected |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in audit.get("policy_motion_outcome_profiles", []):
        lines.append(
            "| `{policy}` | {rows} | {mse} | {rel} | {energy} | {gyro} | {rotation} | {dynamic} | {missing} |".format(
                policy=row.get("policy_id", ""),
                rows=int(row.get("row_count", 0)),
                mse=_fmt(row.get("mean_after_mse")),
                rel=_fmt(row.get("mean_relative_mse_reduction")),
                energy=_fmt(row.get("mean_selected_motion_energy")),
                gyro=_fmt(row.get("mean_selected_gyro_norm_p95")),
                rotation=_fmt(row.get("mean_selected_rotation_dominant_rate")),
                dynamic=_fmt(row.get("mean_selected_high_dynamic_rate")),
                missing=_fmt(row.get("mean_missing_selected_count")),
            )
        )
    lines.extend(
        [
            "",
            "## Focal Pairwise Motion-Outcome Contrasts",
            "",
            "Positive MSE advantage means the focal policy has lower downstream MSE.",
            "",
            "| comparison | units | MSE advantage | median advantage | focal lower MSE | comparison lower MSE | motion delta | gyro delta | rotation-rate delta | focal more motion and lower MSE | motion-delta corr with advantage |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in audit.get("focal_pairwise_motion_outcome_contrasts", []):
        lines.append(
            "| `{comparison}` | {units} | {adv} | {median_adv} | {focal_wins} | {comparison_wins} | {motion} | {gyro} | {rotation} | {count} | {corr} |".format(
                comparison=row.get("comparison_policy", ""),
                units=int(row.get("paired_unit_count", 0)),
                adv=_fmt(row.get("mean_after_mse_advantage_focal_over_comparison")),
                median_adv=_fmt(row.get("median_after_mse_advantage_focal_over_comparison")),
                focal_wins=int(row.get("focal_lower_mse_count", 0)),
                comparison_wins=int(row.get("comparison_lower_mse_count", 0)),
                motion=_fmt(row.get("mean_motion_energy_delta_focal_minus_comparison")),
                gyro=_fmt(row.get("mean_gyro_norm_p95_delta_focal_minus_comparison")),
                rotation=_fmt(row.get("mean_rotation_dominant_rate_delta_focal_minus_comparison")),
                count=int(row.get("focal_more_motion_and_lower_mse_count", 0)),
                corr=_fmt(row.get("motion_delta_advantage_pearson_correlation")),
            )
        )
    lines.extend(
        [
            "",
            "## Descriptive Associations",
            "",
            "These correlations are descriptive diagnostics from the locked run, not prospective selector evidence.",
            "",
            "| feature | outcome | n | Pearson | Spearman |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in audit.get("motion_outcome_associations", []):
        lines.append(
            "| `{feature}` | `{outcome}` | {n} | {pearson} | {spearman} |".format(
                feature=row.get("feature", ""),
                outcome=row.get("outcome", ""),
                n=int(row.get("n", 0)),
                pearson=_fmt(row.get("pearson_correlation")),
                spearman=_fmt(row.get("spearman_correlation")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _clip_features_by_id(motion_audit: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row["sample_id"]): row
        for row in motion_audit.get("clip_motion_features", [])
        if isinstance(row, Mapping) and row.get("sample_id") is not None
    }


def _policy_unit_motion_outcome_rows(
    selected_rows: Sequence[Mapping[str, Any]],
    forecast_rows: Sequence[Mapping[str, Any]],
    clip_features: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected = _selected_by_policy_unit(selected_rows)
    rows = []
    for forecast in forecast_rows:
        policy = str(forecast["policy_id"])
        unit = str(forecast["_unit_id"])
        if unit not in selected[policy]:
            continue
        selected_ids = _selected_ids(selected[policy][unit])
        selected_features = [clip_features[sample_id] for sample_id in selected_ids if sample_id in clip_features]
        motion = _selected_motion_summary(selected_features)
        rows.append(
            {
                "unit_id": unit,
                "seed_name": str(forecast.get("_seed_name", "")),
                "episode_id": str(forecast.get("episode_id", "")),
                "fold_id": forecast.get("fold_id"),
                "policy_id": policy,
                "budget_k": int(forecast["budget_k"]),
                "baseline_mse": float(forecast["baseline_mse"]),
                "after_mse": float(forecast["after_mse"]),
                "absolute_mse_reduction": float(forecast.get("absolute_mse_reduction", 0.0)),
                "relative_mse_reduction": float(forecast.get("relative_mse_reduction", 0.0)),
                "selected_count": len(selected_ids),
                "loaded_selected_count": len(selected_features),
                "missing_selected_count": len(selected_ids) - len(selected_features),
                **motion,
            }
        )
    return sorted(rows, key=lambda row: (str(row["policy_id"]), str(row["unit_id"])))


def _selected_motion_summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, float | int | None]:
    if not samples:
        return {
            "mean_selected_motion_energy": None,
            "mean_selected_gyro_norm_p95": None,
            "mean_selected_acc_delta_norm_p95": None,
            "mean_selected_gyro_delta_norm_p95": None,
            "mean_selected_gyro_energy_fraction": None,
            "mean_selected_stationary_fraction": None,
            "mean_selected_quality_score": None,
            "selected_rotation_dominant_count": 0,
            "selected_high_dynamic_count": 0,
            "selected_low_motion_count": 0,
            "mean_selected_rotation_dominant_rate": None,
            "mean_selected_high_dynamic_rate": None,
        }
    n = len(samples)
    rotation_count = sum(1 for sample in samples if str(sample.get("regime_label")) == "rotation_dominant")
    high_dynamic_count = sum(1 for sample in samples if str(sample.get("regime_label")) == "high_dynamic")
    low_motion_count = sum(1 for sample in samples if str(sample.get("regime_label")) == "low_motion")
    return {
        "mean_selected_motion_energy": _mean_key(samples, "motion_energy"),
        "mean_selected_gyro_norm_p95": _mean_key(samples, "gyro_norm_p95"),
        "mean_selected_acc_delta_norm_p95": _mean_key(samples, "acc_delta_norm_p95"),
        "mean_selected_gyro_delta_norm_p95": _mean_key(samples, "gyro_delta_norm_p95"),
        "mean_selected_gyro_energy_fraction": _mean_key(samples, "gyro_energy_fraction"),
        "mean_selected_stationary_fraction": _mean_key(samples, "stationary_fraction"),
        "mean_selected_quality_score": _mean_key(samples, "quality_score"),
        "selected_rotation_dominant_count": rotation_count,
        "selected_high_dynamic_count": high_dynamic_count,
        "selected_low_motion_count": low_motion_count,
        "mean_selected_rotation_dominant_rate": rotation_count / n,
        "mean_selected_high_dynamic_rate": high_dynamic_count / n,
    }


def _policy_motion_outcome_profiles(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy_id"])].append(row)
    profiles = []
    for policy, policy_rows in grouped.items():
        profiles.append(
            {
                "policy_id": policy,
                "row_count": len(policy_rows),
                "mean_after_mse": _mean_key(policy_rows, "after_mse"),
                "median_after_mse": _median_key(policy_rows, "after_mse"),
                "mean_relative_mse_reduction": _mean_key(policy_rows, "relative_mse_reduction"),
                "mean_missing_selected_count": _mean_key(policy_rows, "missing_selected_count"),
                "mean_loaded_selected_count": _mean_key(policy_rows, "loaded_selected_count"),
                **{feature: _mean_key(policy_rows, feature) for feature in MOTION_FEATURES},
            }
        )
    return sorted(profiles, key=lambda row: (float(row["mean_after_mse"]), str(row["policy_id"])))


def _motion_outcome_associations(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for feature in MOTION_FEATURES:
        for outcome in OUTCOMES:
            pairs = [
                (float(row[feature]), float(row[outcome]))
                for row in rows
                if row.get(feature) is not None and row.get(outcome) is not None
            ]
            xs = [pair[0] for pair in pairs]
            ys = [pair[1] for pair in pairs]
            records.append(
                {
                    "feature": feature,
                    "outcome": outcome,
                    "n": len(pairs),
                    "pearson_correlation": _pearson(xs, ys),
                    "spearman_correlation": _spearman(xs, ys),
                }
            )
    return records


def _focal_pairwise_motion_outcome_contrasts(
    rows: Sequence[Mapping[str, Any]],
    *,
    focal_policy: str,
    comparison_policies: Sequence[str],
) -> list[dict[str, Any]]:
    by_policy_unit: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_policy_unit[str(row["policy_id"])][str(row["unit_id"])] = row

    records = []
    for comparison in comparison_policies:
        units = sorted(set(by_policy_unit[focal_policy]) & set(by_policy_unit[comparison]))
        advantages = []
        motion_deltas = []
        gyro_deltas = []
        rotation_deltas = []
        both_count = 0
        focal_more_motion_count = 0
        focal_lower_mse_count = 0
        for unit in units:
            focal = by_policy_unit[focal_policy][unit]
            other = by_policy_unit[comparison][unit]
            advantage = float(other["after_mse"]) - float(focal["after_mse"])
            motion_delta = _delta(focal, other, "mean_selected_motion_energy")
            gyro_delta = _delta(focal, other, "mean_selected_gyro_norm_p95")
            rotation_delta = _delta(focal, other, "mean_selected_rotation_dominant_rate")
            advantages.append(advantage)
            motion_deltas.append(motion_delta)
            gyro_deltas.append(gyro_delta)
            rotation_deltas.append(rotation_delta)
            focal_more_motion = motion_delta is not None and motion_delta > 1.0e-12
            focal_lower_mse = advantage > 1.0e-12
            focal_more_motion_count += int(focal_more_motion)
            focal_lower_mse_count += int(focal_lower_mse)
            both_count += int(focal_more_motion and focal_lower_mse)
        valid_motion_advantages = [
            (motion, advantage)
            for motion, advantage in zip(motion_deltas, advantages, strict=False)
            if motion is not None
        ]
        records.append(
            {
                "focal_policy": focal_policy,
                "comparison_policy": comparison,
                "paired_unit_count": len(units),
                "mean_after_mse_advantage_focal_over_comparison": _mean_values(advantages),
                "median_after_mse_advantage_focal_over_comparison": _median_values(advantages),
                "focal_lower_mse_count": focal_lower_mse_count,
                "comparison_lower_mse_count": sum(1 for advantage in advantages if advantage < -1.0e-12),
                "focal_more_motion_count": focal_more_motion_count,
                "focal_more_motion_and_lower_mse_count": both_count,
                "mean_motion_energy_delta_focal_minus_comparison": _mean_optional_values(motion_deltas),
                "mean_gyro_norm_p95_delta_focal_minus_comparison": _mean_optional_values(gyro_deltas),
                "mean_rotation_dominant_rate_delta_focal_minus_comparison": _mean_optional_values(rotation_deltas),
                "motion_delta_advantage_pearson_correlation": _pearson(
                    [pair[0] for pair in valid_motion_advantages],
                    [pair[1] for pair in valid_motion_advantages],
                ),
            }
        )
    return records


def _interpret_link(
    rows: Sequence[Mapping[str, Any]],
    *,
    focal_policy: str,
    comparison_policies: Sequence[str],
) -> list[str]:
    profiles = {str(row["policy_id"]): row for row in _policy_motion_outcome_profiles(rows)}
    contrasts = _focal_pairwise_motion_outcome_contrasts(
        rows,
        focal_policy=focal_policy,
        comparison_policies=comparison_policies,
    )
    reads = [
        "This is a descriptive mechanism-to-outcome audit on the locked downstream run, not causal proof and not a new policy-selection loop.",
        "It asks whether selected raw-motion regimes line up with lower held-out forecast MSE after acquisition.",
        "The result supports a regime-level explanation, not a scalar motion-energy rule.",
    ]
    focal = profiles.get(focal_policy)
    if focal:
        reads.append(
            f"`{focal_policy}` has mean after MSE `{float(focal['mean_after_mse']):.6f}` with selected dynamic energy `{float(focal['mean_selected_motion_energy']):.6f}` and rotation-dominant rate `{float(focal['mean_selected_rotation_dominant_rate']):.6f}`."
        )
    for contrast in contrasts:
        comparison = str(contrast["comparison_policy"])
        if comparison not in profiles:
            continue
        reads.append(
            f"`{focal_policy}` vs `{comparison}`: MSE advantage `{float(contrast['mean_after_mse_advantage_focal_over_comparison']):.6f}`, motion-energy delta `{float(contrast['mean_motion_energy_delta_focal_minus_comparison'] or 0.0):.6f}`, focal-more-motion-and-lower-MSE units `{int(contrast['focal_more_motion_and_lower_mse_count'])}/{int(contrast['paired_unit_count'])}`."
        )
    return reads


def _delta(row_a: Mapping[str, Any], row_b: Mapping[str, Any], key: str) -> float | None:
    if row_a.get(key) is None or row_b.get(key) is None:
        return None
    return float(row_a[key]) - float(row_b[key])


def _mean_key(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(mean(values)) if values else None


def _median_key(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(median(values)) if values else None


def _mean_values(values: Sequence[float]) -> float | None:
    return float(mean(values)) if values else None


def _median_values(values: Sequence[float]) -> float | None:
    return float(median(values)) if values else None


def _mean_optional_values(values: Sequence[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return float(mean(present)) if present else None


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    denom = (x_var * y_var) ** 0.5
    return float(numerator / denom) if denom > 0.0 else None


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return _pearson(_ranks(xs), _ranks(ys))


def _ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for index in range(cursor, end):
            ranks[indexed[index][0]] = rank
        cursor = end
    return ranks


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
