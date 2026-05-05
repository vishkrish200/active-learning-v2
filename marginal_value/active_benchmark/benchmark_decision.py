from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from marginal_value.active_benchmark.policy_autopsy import (
    BLEND_POLICY,
    KCENTER_WINDOW_POLICY,
    OLD_NOVELTY_POLICY,
    SOURCECAP_OLD_POLICY,
    SUBMITTED_FULL_REPLAY_POLICY,
    SUBMITTED_MINUS_TS2VEC_POLICY,
    SUBMITTED_NO_KCENTER_POLICY,
    build_policy_autopsy,
)


ORACLE_POLICY = "oracle_greedy_eval_only"
QUALITY_ONLY_POLICY = "quality_only"
RANDOM_VALID_POLICY = "random_valid"


def build_benchmark_decision_report(
    run_root: str | Path,
    *,
    aggregate_path: str | Path | None = None,
    near_zero_threshold: float = 0.10,
) -> dict[str, Any]:
    root = Path(run_root)
    autopsy = build_policy_autopsy(root, aggregate_path=aggregate_path)
    reports = _load_seed_reports(root)
    rows = [row for report in reports for row in report.get("rounds", [])]
    difficulty_by_key = _difficulty_by_episode_key(reports)
    all_episode_keys = sorted({(str(row.get("_seed_name", "")), str(row.get("episode_id", ""))) for row in rows})
    non_near_zero_keys = {
        key for key in all_episode_keys if float(difficulty_by_key.get(key, {}).get("near_zero_oracle_round_fraction", 1.0)) <= near_zero_threshold
    }

    final_means_all = _final_means(rows)
    final_means_non_near_zero = _final_means(rows, allowed_episode_keys=non_near_zero_keys)
    gates = [
        _oracle_sanity_gate(autopsy),
        _bad_control_sanity_gate(autopsy),
        _near_zero_sensitivity_gate(
            final_means_all=final_means_all,
            final_means_non_near_zero=final_means_non_near_zero,
            total_episode_count=len(all_episode_keys),
            non_near_zero_episode_count=len(non_near_zero_keys),
            near_zero_threshold=near_zero_threshold,
        ),
        _source_concentration_gate(autopsy),
        _ts2vec_incremental_value_gate(autopsy),
    ]
    decision = _decision(gates, final_means_all)
    return {
        "input": {
            "run_root": str(root),
            "aggregate_path": str(Path(aggregate_path)) if aggregate_path is not None else autopsy.get("input", {}).get("aggregate_path"),
            "report_count": len(reports),
            "near_zero_threshold": near_zero_threshold,
        },
        "decision": decision,
        "gates": gates,
        "policy_final_means": {
            "all_episodes": final_means_all,
            "non_near_zero_episodes": final_means_non_near_zero,
        },
        "policy_autopsy_refs": {
            "diagnosis": autopsy.get("diagnosis", []),
            "old_vs_blend": autopsy.get("old_vs_blend", {}),
            "source_kcenter_read": autopsy.get("source_kcenter_read", {}),
            "coverage_story": autopsy.get("coverage_story", {}),
        },
    }


def write_benchmark_decision_reports(
    run_root: str | Path,
    *,
    output_json: str | Path,
    output_markdown: str | Path,
    aggregate_path: str | Path | None = None,
    near_zero_threshold: float = 0.10,
) -> dict[str, str]:
    report = build_benchmark_decision_report(
        run_root,
        aggregate_path=aggregate_path,
        near_zero_threshold=near_zero_threshold,
    )
    json_path = Path(output_json)
    markdown_path = Path(output_markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_benchmark_decision_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def render_benchmark_decision_markdown(report: dict[str, Any]) -> str:
    decision = report.get("decision", {})
    lines = [
        "# Benchmark Decision Report",
        "",
        "## Decision",
        "",
        f"- downstream training: `{decision.get('downstream_training', '')}`",
        f"- baseline to carry: `{decision.get('baseline_to_carry', '')}`",
        f"- verdict: {decision.get('read', '')}",
        "",
        "## Next Steps",
        "",
    ]
    for step in decision.get("next_steps", []):
        lines.append(f"- {step}")

    lines.extend(
        [
            "",
            "## Gates",
            "",
            "| gate | status | read |",
            "|---|---|---|",
        ]
    )
    for gate in report.get("gates", []):
        lines.append(f"| {gate.get('name', '')} | {gate.get('status', '')} | {gate.get('read', '')} |")

    lines.extend(
        [
            "",
            "## Evidence",
            "",
            "| policy | all episodes final gain | non-near-zero final gain |",
            "|---|---:|---:|",
        ]
    )
    all_means = report.get("policy_final_means", {}).get("all_episodes", {})
    filtered_means = report.get("policy_final_means", {}).get("non_near_zero_episodes", {})
    for policy in sorted(all_means, key=lambda item: _none_low(all_means.get(item)), reverse=True):
        lines.append(f"| {policy} | {_fmt(all_means.get(policy))} | {_fmt(filtered_means.get(policy))} |")

    lines.extend(
        [
            "",
            "## Autopsy Carryover",
            "",
        ]
    )
    for item in report.get("policy_autopsy_refs", {}).get("diagnosis", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _oracle_sanity_gate(autopsy: dict[str, Any]) -> dict[str, Any]:
    outcomes = _outcomes_by_policy(autopsy)
    oracle_gain = outcomes.get(ORACLE_POLICY, {}).get("mean_final_cumulative_gain")
    non_oracle = {
        policy: row.get("mean_final_cumulative_gain")
        for policy, row in outcomes.items()
        if policy != ORACLE_POLICY and row.get("mean_final_cumulative_gain") is not None
    }
    best_non_oracle_policy, best_non_oracle_gain = _top_policy(non_oracle)
    exact = autopsy.get("coverage_story", {}).get("oracle_fraction_exact_all_rounds")
    if oracle_gain is None:
        status = "fail"
        read = "oracle_greedy_eval_only is missing, so the benchmark has no upper-bound sanity check"
    elif exact is not True:
        status = "warn"
        read = "oracle denominators were not exact for every audited round"
    elif best_non_oracle_gain is not None and float(oracle_gain) + 1.0e-12 < float(best_non_oracle_gain):
        status = "fail"
        read = f"oracle final gain {_fmt(oracle_gain)} trailed {best_non_oracle_policy} at {_fmt(best_non_oracle_gain)}"
    else:
        status = "pass"
        read = f"oracle final gain {_fmt(oracle_gain)} was at least as high as the best non-oracle policy"
    return {
        "name": "oracle_sanity",
        "status": status,
        "read": read,
        "evidence": {
            "oracle_mean_final_cumulative_gain": oracle_gain,
            "best_non_oracle_policy": best_non_oracle_policy,
            "best_non_oracle_mean_final_cumulative_gain": best_non_oracle_gain,
            "oracle_fraction_exact_all_rounds": exact,
        },
    }


def _bad_control_sanity_gate(autopsy: dict[str, Any]) -> dict[str, Any]:
    outcomes = _outcomes_by_policy(autopsy)
    kcenter_gain = outcomes.get(KCENTER_WINDOW_POLICY, {}).get("mean_final_cumulative_gain")
    quality_gain = outcomes.get(QUALITY_ONLY_POLICY, {}).get("mean_final_cumulative_gain")
    full_gain = outcomes.get(SUBMITTED_FULL_REPLAY_POLICY, {}).get("mean_final_cumulative_gain")
    no_kcenter_gain = outcomes.get(SUBMITTED_NO_KCENTER_POLICY, {}).get("mean_final_cumulative_gain")
    quality_delta = _subtract_optional(kcenter_gain, quality_gain)
    no_kcenter_delta = _subtract_optional(full_gain, no_kcenter_gain)
    missing = [name for name, value in {
        KCENTER_WINDOW_POLICY: kcenter_gain,
        QUALITY_ONLY_POLICY: quality_gain,
        SUBMITTED_FULL_REPLAY_POLICY: full_gain,
        SUBMITTED_NO_KCENTER_POLICY: no_kcenter_gain,
    }.items() if value is None]
    if missing:
        status = "warn"
        read = "missing bad-control comparisons: " + ", ".join(missing)
    elif float(quality_delta) > 0.0 and float(no_kcenter_delta) > 0.0:
        status = "pass"
        read = "stronger policies beat quality-only and no-k-center controls"
    else:
        status = "fail"
        read = "one or more bad controls matched or beat the stronger policy comparison"
    return {
        "name": "bad_control_sanity",
        "status": status,
        "read": read,
        "evidence": {
            "kcenter_minus_quality_only": quality_delta,
            "submitted_full_replay_minus_no_kcenter": no_kcenter_delta,
        },
    }


def _near_zero_sensitivity_gate(
    *,
    final_means_all: dict[str, float],
    final_means_non_near_zero: dict[str, float],
    total_episode_count: int,
    non_near_zero_episode_count: int,
    near_zero_threshold: float,
) -> dict[str, Any]:
    top_all, top_all_gain = _top_policy(_without_oracle(final_means_all))
    top_filtered, top_filtered_gain = _top_policy(_without_oracle(final_means_non_near_zero))
    if not final_means_non_near_zero:
        status = "fail"
        read = f"no episodes remained after filtering near-zero oracle episodes at threshold {near_zero_threshold:.2f}"
    elif top_all == top_filtered:
        status = "pass"
        read = "top non-oracle policy was stable after removing near-zero-oracle episodes"
    else:
        status = "warn"
        read = f"top non-oracle policy changed from {top_all} to {top_filtered} after near-zero filtering"
    return {
        "name": "near_zero_sensitivity",
        "status": status,
        "read": read,
        "evidence": {
            "total_episode_count": total_episode_count,
            "non_near_zero_episode_count": non_near_zero_episode_count,
            "near_zero_threshold": near_zero_threshold,
            "top_non_oracle_all": top_all,
            "top_non_oracle_all_gain": top_all_gain,
            "top_non_oracle_non_near_zero": top_filtered,
            "top_non_oracle_non_near_zero_gain": top_filtered_gain,
        },
    }


def _source_concentration_gate(autopsy: dict[str, Any]) -> dict[str, Any]:
    rows = {
        row.get("policy"): row
        for row in autopsy.get("source_kcenter_read", {}).get("source_concentration", [])
    }
    full_largest = rows.get(SUBMITTED_FULL_REPLAY_POLICY, {}).get("mean_largest_source_group_fraction")
    no_kcenter_largest = rows.get(SUBMITTED_NO_KCENTER_POLICY, {}).get("mean_largest_source_group_fraction")
    old_largest = rows.get(OLD_NOVELTY_POLICY, {}).get("mean_largest_source_group_fraction")
    sourcecap_largest = rows.get(SOURCECAP_OLD_POLICY, {}).get("mean_largest_source_group_fraction")
    full_improves = _less_optional(full_largest, no_kcenter_largest)
    sourcecap_improves = _less_optional(sourcecap_largest, old_largest)
    if full_improves is None and sourcecap_improves is None:
        status = "warn"
        read = "source concentration evidence is missing"
    elif full_improves is False or sourcecap_improves is False:
        status = "warn"
        read = "source concentration did not consistently improve under source cap or k-center controls"
    else:
        status = "pass"
        read = "source cap and k-center controls reduced largest-source concentration where comparable"
    return {
        "name": "source_concentration",
        "status": status,
        "read": read,
        "evidence": {
            "submitted_full_replay_largest_source_fraction": full_largest,
            "submitted_no_kcenter_largest_source_fraction": no_kcenter_largest,
            "old_novelty_largest_source_fraction": old_largest,
            "sourcecap_old_largest_source_fraction": sourcecap_largest,
        },
    }


def _ts2vec_incremental_value_gate(autopsy: dict[str, Any]) -> dict[str, Any]:
    comparisons = {
        row.get("comparison"): row
        for row in autopsy.get("source_kcenter_read", {}).get("comparisons", [])
    }
    full_minus_no_ts2vec = comparisons.get("full_replay_minus_minus_ts2vec", {})
    blend_minus_kcenter = comparisons.get("blend_minus_kcenter", {})
    deltas = [
        full_minus_no_ts2vec.get("mean_delta_final_cumulative_gain"),
        blend_minus_kcenter.get("mean_delta_final_cumulative_gain"),
    ]
    numeric_deltas = [float(value) for value in deltas if value is not None]
    if len(numeric_deltas) != 2:
        status = "warn"
        read = "TS2Vec ablation evidence is missing"
    elif all(value > 0.01 for value in numeric_deltas):
        status = "pass"
        read = "TS2Vec showed a positive incremental gain over window/k-center controls"
    else:
        status = "warn"
        read = "TS2Vec did not show clear incremental value over window/k-center controls"
    return {
        "name": "ts2vec_incremental_value",
        "status": status,
        "read": read,
        "evidence": {
            "submitted_full_replay_minus_minus_ts2vec": full_minus_no_ts2vec,
            "blend_minus_kcenter": blend_minus_kcenter,
        },
    }


def _decision(gates: list[dict[str, Any]], final_means_all: dict[str, float]) -> dict[str, Any]:
    by_name = {gate["name"]: gate for gate in gates}
    hard_failures = [gate["name"] for gate in gates if gate.get("status") == "fail"]
    ts2vec_status = by_name.get("ts2vec_incremental_value", {}).get("status")
    baseline = KCENTER_WINDOW_POLICY if KCENTER_WINDOW_POLICY in final_means_all else _top_policy(_without_oracle(final_means_all))[0]
    if hard_failures:
        downstream = "hold"
        read = "benchmark proxy has failed sanity gates; redesign or debug the proxy before model training"
    elif ts2vec_status != "pass":
        downstream = "hold"
        read = "benchmark proxy is usable for baseline selection, but TS2Vec is not earning downstream-training spend yet"
    else:
        downstream = "consider"
        read = "benchmark proxy passed gates and TS2Vec showed enough incremental value to justify a downstream-training smoke"
    return {
        "downstream_training": downstream,
        "baseline_to_carry": baseline,
        "read": read,
        "blocking_gates": hard_failures,
        "next_steps": [
            "Carry kcenter_quality_gated_window as the active-learning baseline unless a stronger proxy result displaces it.",
            "Do not launch downstream training until the benchmark either validates TS2Vec incremental value or is redesigned.",
            "If spending GCP credits next, spend them on proxy validation or a tiny downstream smoke, not a full retraining run.",
        ],
    }


def _load_seed_reports(root: Path) -> list[dict[str, Any]]:
    paths = sorted(path for path in root.glob("**/offline_active_benchmark_report.json") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No offline_active_benchmark_report.json files found under {root}")
    reports = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        seed_name = path.parent.name
        for row in report.get("rounds", []):
            row["_seed_name"] = seed_name
        for row in report.get("difficulty_audit", []):
            row["_seed_name"] = seed_name
        reports.append(report)
    return reports


def _difficulty_by_episode_key(reports: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    output = {}
    for report in reports:
        for row in report.get("difficulty_audit", []):
            output[(str(row.get("_seed_name", "")), str(row.get("episode_id", "")))] = row
    return output


def _final_means(
    rows: list[dict[str, Any]],
    *,
    allowed_episode_keys: set[tuple[str, str]] | None = None,
) -> dict[str, float]:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        episode_key = (str(row.get("_seed_name", "")), str(row.get("episode_id", "")))
        if allowed_episode_keys is not None and episode_key not in allowed_episode_keys:
            continue
        key = (*episode_key, str(row.get("policy_name", "")))
        current = latest.get(key)
        if current is None or int(row.get("round_index", 0)) > int(current.get("round_index", 0)):
            latest[key] = row
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in latest.values():
        grouped[str(row.get("policy_name", ""))].append(_float(row.get("cumulative_balanced_relative_gain")))
    return {policy: float(mean(values)) for policy, values in sorted(grouped.items()) if values}


def _outcomes_by_policy(autopsy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("policy", "")): row
        for row in autopsy.get("policy_outcomes", [])
        if row.get("policy")
    }


def _without_oracle(values: dict[str, float]) -> dict[str, float]:
    return {policy: value for policy, value in values.items() if policy != ORACLE_POLICY}


def _top_policy(values: dict[str, Any]) -> tuple[str | None, float | None]:
    numeric = {policy: _float(value) for policy, value in values.items() if value is not None}
    if not numeric:
        return None, None
    policy = max(numeric, key=lambda item: numeric[item])
    return policy, numeric[policy]


def _subtract_optional(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return _float(left) - _float(right)


def _less_optional(left: Any, right: Any) -> bool | None:
    if left is None or right is None:
        return None
    return _float(left) < _float(right)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _none_low(value: Any) -> float:
    return float("-inf") if value is None else _float(value)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)
