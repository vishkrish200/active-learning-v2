from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_BASELINE_POLICY = "quality_stratified_random_v1"
ORACLE_POLICY = "oracle_greedy_eval_view_v1"


def build_coverage_decision_report(
    reports: Sequence[Mapping[str, Any]],
    *,
    report_names: Sequence[str] | None = None,
    baseline_policy: str = DEFAULT_BASELINE_POLICY,
    oracle_policy: str = ORACLE_POLICY,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260508,
    eps: float = 1.0e-12,
) -> dict[str, Any]:
    names = _report_names(reports, report_names)
    records = _primary_view_gain_records(reports, names)
    final_budget = max((int(budget) for report in reports for budget in report.get("budgets", [])), default=0)
    final_records = [record for record in records if int(record["budget_k"]) == final_budget]
    final_by_key = _records_by_policy_unit(final_records)
    policy_final = _policy_final_summary(
        final_records,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed + 431,
    )
    budget_summary = _policy_budget_summary(records)
    pairwise = _pairwise_summaries(
        final_by_key,
        baseline_policy=baseline_policy,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    oracle_capture = _oracle_capture_summaries(
        final_by_key,
        baseline_policy=baseline_policy,
        oracle_policy=oracle_policy,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed + 9173,
        eps=eps,
    )
    gates = _decision_gates(
        final_records=final_records,
        policy_final=policy_final,
        pairwise=pairwise,
        oracle_capture=oracle_capture,
        baseline_policy=baseline_policy,
        oracle_policy=oracle_policy,
    )
    return {
        "input": {
            "report_count": len(reports),
            "report_names": list(names),
            "baseline_policy": baseline_policy,
            "oracle_policy": oracle_policy,
            "bootstrap_replicates": int(bootstrap_replicates),
            "bootstrap_seed": int(bootstrap_seed),
        },
        "coverage_units": {
            "independent_episode_count": len({str(record["unit_id"]) for record in final_records}),
            "final_budget": final_budget,
            "gain_record_count": len(records),
            "final_gain_record_count": len(final_records),
            "candidate_pool": _candidate_pool_summary(reports, names),
        },
        "policy_final_summary": policy_final,
        "policy_budget_summary": budget_summary,
        "pairwise_vs_baseline": pairwise,
        "oracle_capture_vs_baseline": oracle_capture,
        "gates": gates,
        "decision": _decision(gates),
    }


def write_coverage_decision_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    output_json: str | Path,
    output_markdown: str | Path,
    report_names: Sequence[str] | None = None,
    baseline_policy: str = DEFAULT_BASELINE_POLICY,
    oracle_policy: str = ORACLE_POLICY,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260508,
) -> dict[str, str]:
    report = build_coverage_decision_report(
        reports,
        report_names=report_names,
        baseline_policy=baseline_policy,
        oracle_policy=oracle_policy,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    json_path = Path(output_json)
    markdown_path = Path(output_markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_coverage_decision_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def render_coverage_decision_markdown(report: Mapping[str, Any]) -> str:
    decision = report.get("decision", {})
    units = report.get("coverage_units", {})
    lines = [
        "# Coverage Benchmark Decision Report",
        "",
        "## Decision",
        "",
        f"- downstream training: `{decision.get('downstream_training', '')}`",
        f"- next CPU gate: `{decision.get('next_cpu_gate', '')}`",
        f"- read: {decision.get('read', '')}",
        "",
        "## Run Shape",
        "",
        f"- reports: `{report.get('input', {}).get('report_count', 0)}`",
        f"- independent final episodes: `{units.get('independent_episode_count', 0)}`",
        f"- final budget: `{units.get('final_budget', 0)}`",
        f"- gain records: `{units.get('gain_record_count', 0)}`",
        f"- candidate pool size: `{_pool_range(units.get('candidate_pool', {}))}`",
        "",
        "## Final Leaderboard",
        "",
        "| rank | policy | mean final gain | median | CI95 low | CI95 high | episodes | target-used |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    final_summary = report.get("policy_final_summary", {})
    for rank, (policy, row) in enumerate(_ranked_items(final_summary, "mean_final_gain"), start=1):
        lines.append(
            "| {rank} | `{policy}` | {mean} | {median} | {low} | {high} | {episodes} | {target} |".format(
                rank=rank,
                policy=policy,
                mean=_fmt(row.get("mean_final_gain")),
                median=_fmt(row.get("median_final_gain")),
                low=_fmt(row.get("bootstrap_ci95_low")),
                high=_fmt(row.get("bootstrap_ci95_high")),
                episodes=int(row.get("episode_count", 0)),
                target="Y" if row.get("uses_target_for_selection") else "N",
            )
        )

    lines.extend(
        [
            "",
            "## Paired Delta Vs Baseline",
            "",
            "| policy | baseline | mean delta | median delta | CI95 low | CI95 high | win frac | paired episodes |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for policy, row in _ranked_items(report.get("pairwise_vs_baseline", {}), "mean_delta"):
        lines.append(
            "| `{policy}` | `{baseline}` | {mean} | {median} | {low} | {high} | {win} | {episodes} |".format(
                policy=policy,
                baseline=row.get("baseline_policy", ""),
                mean=_fmt(row.get("mean_delta")),
                median=_fmt(row.get("median_delta")),
                low=_fmt(row.get("bootstrap_ci95_low")),
                high=_fmt(row.get("bootstrap_ci95_high")),
                win=_fmt(row.get("win_fraction")),
                episodes=int(row.get("paired_episode_count", 0)),
            )
        )

    if report.get("oracle_capture_vs_baseline"):
        lines.extend(
            [
                "",
                "## Oracle Capture",
                "",
                "| policy | mean capture | median capture | CI95 low | CI95 high | episodes |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for policy, row in _ranked_items(report.get("oracle_capture_vs_baseline", {}), "mean_oracle_capture"):
            lines.append(
                "| `{policy}` | {mean} | {median} | {low} | {high} | {episodes} |".format(
                    policy=policy,
                    mean=_fmt(row.get("mean_oracle_capture")),
                    median=_fmt(row.get("median_oracle_capture")),
                    low=_fmt(row.get("bootstrap_ci95_low")),
                    high=_fmt(row.get("bootstrap_ci95_high")),
                    episodes=int(row.get("paired_episode_count", 0)),
                )
            )
    else:
        lines.extend(["", "## Oracle Capture", "", "- Missing: no oracle policy rows were present in these reports."])

    lines.extend(
        [
            "",
            "## Budget Curves",
            "",
            "| budget | policy | mean gain | episodes |",
            "|---:|---|---:|---:|",
        ]
    )
    for row in sorted(
        report.get("policy_budget_summary", []),
        key=lambda item: (int(item.get("budget_k", 0)), -float(item.get("mean_gain", 0.0)), str(item.get("policy_id", ""))),
    ):
        lines.append(
            "| {budget} | `{policy}` | {mean} | {episodes} |".format(
                budget=int(row.get("budget_k", 0)),
                policy=row.get("policy_id", ""),
                mean=_fmt(row.get("mean_gain")),
                episodes=int(row.get("episode_count", 0)),
            )
        )

    lines.extend(["", "## Gates", "", "| gate | status | read |", "|---|---|---|"])
    for gate in report.get("gates", []):
        lines.append(f"| {gate.get('name', '')} | {gate.get('status', '')} | {gate.get('read', '')} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Confidence intervals are episode-level paired bootstraps, not row-level bootstraps.",
            "- Oracle rows are non-deployable ceilings because they inspect the blind target set.",
            "- Gains use primary eval views but still exclude same-feature selector/eval shortcuts.",
            "",
        ]
    )
    return "\n".join(lines)


def _report_names(reports: Sequence[Mapping[str, Any]], report_names: Sequence[str] | None) -> tuple[str, ...]:
    if report_names is None:
        return tuple(f"report_{index:03d}" for index, _report in enumerate(reports))
    names = tuple(str(name) for name in report_names)
    if len(names) != len(reports):
        raise ValueError("report_names length must match reports length.")
    if len(set(names)) != len(names):
        raise ValueError("report_names must be unique.")
    return names


def _primary_view_gain_records(reports: Sequence[Mapping[str, Any]], names: Sequence[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for report, report_name in zip(reports, names):
        config = report.get("config", {})
        primary_views = {str(view) for view in config.get("primary_eval_views", [])}
        by_key: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in report.get("metric_rows", []):
            if str(row.get("metric_name", "")) != "coverage_gain_rel":
                continue
            if str(row.get("eval_view", "")) not in primary_views:
                continue
            if bool(row.get("selector_feature_overlap", False)):
                continue
            key = (
                str(report_name),
                str(row.get("episode_id", "")),
                int(row.get("budget_k", 0)),
                str(row.get("policy_id", "")),
            )
            by_key[key].append(dict(row))
        for (name, episode_id, budget_k, policy_id), rows in sorted(by_key.items()):
            records.append(
                {
                    "report_name": name,
                    "episode_id": episode_id,
                    "unit_id": f"{name}:{episode_id}",
                    "budget_k": int(budget_k),
                    "policy_id": policy_id,
                    "gain": mean(float(row.get("metric_value", 0.0)) for row in rows),
                    "view_count": len(rows),
                    "eval_views": sorted({str(row.get("eval_view", "")) for row in rows}),
                    "uses_target_for_selection": any(bool(row.get("uses_target_for_selection", False)) for row in rows),
                }
            )
    return records


def _records_by_policy_unit(records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(str(record["unit_id"]), str(record["policy_id"])): record for record in records}


def _candidate_pool_summary(reports: Sequence[Mapping[str, Any]], names: Sequence[str]) -> dict[str, float | int | None]:
    by_unit: dict[str, int] = {}
    for report, report_name in zip(reports, names):
        for row in report.get("rounds", []):
            unit_id = f"{report_name}:{row.get('episode_id', '')}"
            count = int(row.get("candidate_count", 0))
            by_unit[unit_id] = max(count, by_unit.get(unit_id, 0))
    counts = list(by_unit.values())
    if not counts:
        return {"episode_count": 0, "min": None, "median": None, "max": None}
    return {
        "episode_count": len(counts),
        "min": int(min(counts)),
        "median": float(median(counts)),
        "max": int(max(counts)),
    }


def _policy_final_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, dict[str, Any]]:
    by_policy: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_policy[str(record["policy_id"])].append(record)
    summary: dict[str, dict[str, Any]] = {}
    for index, (policy_id, policy_records) in enumerate(sorted(by_policy.items())):
        gains = [float(record["gain"]) for record in policy_records]
        ci_low, ci_high = _bootstrap_ci95(gains, seed=bootstrap_seed + index, replicates=bootstrap_replicates)
        summary[policy_id] = {
            "episode_count": len(gains),
            "mean_final_gain": mean(gains) if gains else None,
            "median_final_gain": median(gains) if gains else None,
            "bootstrap_ci95_low": ci_low,
            "bootstrap_ci95_high": ci_high,
            "uses_target_for_selection": any(bool(record.get("uses_target_for_selection", False)) for record in policy_records),
        }
    return summary


def _policy_budget_summary(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, str], list[float]] = defaultdict(list)
    target_used: dict[tuple[int, str], bool] = defaultdict(bool)
    for record in records:
        key = (int(record["budget_k"]), str(record["policy_id"]))
        by_key[key].append(float(record["gain"]))
        target_used[key] = target_used[key] or bool(record.get("uses_target_for_selection", False))
    return [
        {
            "budget_k": budget_k,
            "policy_id": policy_id,
            "episode_count": len(gains),
            "mean_gain": mean(gains) if gains else None,
            "median_gain": median(gains) if gains else None,
            "uses_target_for_selection": bool(target_used[(budget_k, policy_id)]),
        }
        for (budget_k, policy_id), gains in sorted(by_key.items())
    ]


def _pairwise_summaries(
    final_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    baseline_policy: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, dict[str, Any]]:
    policies = sorted({policy for _unit, policy in final_by_key})
    units = sorted({unit for unit, _policy in final_by_key})
    summaries: dict[str, dict[str, Any]] = {}
    for index, policy in enumerate(policies):
        if policy == baseline_policy:
            continue
        deltas = []
        for unit in units:
            policy_row = final_by_key.get((unit, policy))
            baseline_row = final_by_key.get((unit, baseline_policy))
            if policy_row is None or baseline_row is None:
                continue
            deltas.append(float(policy_row["gain"]) - float(baseline_row["gain"]))
        ci_low, ci_high = _bootstrap_ci95(deltas, seed=bootstrap_seed + index, replicates=bootstrap_replicates)
        summaries[policy] = {
            "baseline_policy": baseline_policy,
            "paired_episode_count": len(deltas),
            "mean_delta": mean(deltas) if deltas else None,
            "median_delta": median(deltas) if deltas else None,
            "bootstrap_ci95_low": ci_low,
            "bootstrap_ci95_high": ci_high,
            "win_fraction": _win_fraction(deltas),
        }
    return summaries


def _oracle_capture_summaries(
    final_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    baseline_policy: str,
    oracle_policy: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    eps: float,
) -> dict[str, dict[str, Any]]:
    units = sorted({unit for unit, _policy in final_by_key})
    policies = sorted({policy for _unit, policy in final_by_key if policy not in {baseline_policy, oracle_policy}})
    summaries: dict[str, dict[str, Any]] = {}
    for index, policy in enumerate(policies):
        captures = []
        skipped = 0
        for unit in units:
            policy_row = final_by_key.get((unit, policy))
            baseline_row = final_by_key.get((unit, baseline_policy))
            oracle_row = final_by_key.get((unit, oracle_policy))
            if policy_row is None or baseline_row is None or oracle_row is None:
                skipped += 1
                continue
            denominator = float(oracle_row["gain"]) - float(baseline_row["gain"])
            if denominator <= eps:
                skipped += 1
                continue
            captures.append((float(policy_row["gain"]) - float(baseline_row["gain"])) / denominator)
        if not captures:
            continue
        ci_low, ci_high = _bootstrap_ci95(captures, seed=bootstrap_seed + index, replicates=bootstrap_replicates)
        summaries[policy] = {
            "baseline_policy": baseline_policy,
            "oracle_policy": oracle_policy,
            "paired_episode_count": len(captures),
            "skipped_episode_count": skipped,
            "mean_oracle_capture": mean(captures),
            "median_oracle_capture": median(captures),
            "bootstrap_ci95_low": ci_low,
            "bootstrap_ci95_high": ci_high,
        }
    return summaries


def _decision_gates(
    *,
    final_records: Sequence[Mapping[str, Any]],
    policy_final: Mapping[str, Mapping[str, Any]],
    pairwise: Mapping[str, Mapping[str, Any]],
    oracle_capture: Mapping[str, Mapping[str, Any]],
    baseline_policy: str,
    oracle_policy: str,
) -> list[dict[str, Any]]:
    episode_count = len({str(record["unit_id"]) for record in final_records})
    top_policy, top_row = _top_non_target_policy(policy_final)
    top_pair = pairwise.get(top_policy, {}) if top_policy else {}
    gates = [
        {
            "name": "episode_count",
            "status": "pass" if episode_count >= 20 else "warn",
            "read": f"{episode_count} independent final episodes; use as smoke only below 20.",
            "evidence": {"independent_episode_count": episode_count},
        },
        {
            "name": "baseline_delta_ci",
            "status": "pass" if _gt(top_pair.get("bootstrap_ci95_low"), 0.04) else "warn",
            "read": (
                f"{top_policy} vs {baseline_policy} CI low is {_fmt(top_pair.get('bootstrap_ci95_low'))}; "
                "require > 0.04 before downstream spend."
            ),
            "evidence": {"top_policy": top_policy, **top_pair},
        },
        {
            "name": "oracle_present",
            "status": "pass" if oracle_policy in policy_final else "fail",
            "read": "oracle ceiling is present." if oracle_policy in policy_final else "oracle ceiling is missing.",
            "evidence": {"oracle_policy": oracle_policy},
        },
        {
            "name": "oracle_capture",
            "status": "pass" if top_policy in oracle_capture and _gt(oracle_capture[top_policy].get("mean_oracle_capture"), 0.40) else "warn",
            "read": (
                f"{top_policy} oracle capture is {_fmt(oracle_capture.get(top_policy, {}).get('mean_oracle_capture'))}; "
                "require >= 0.40 point estimate before downstream spend."
            ),
            "evidence": oracle_capture.get(top_policy, {}),
        },
    ]
    if top_row and bool(top_row.get("uses_target_for_selection", False)):
        gates.append(
            {
                "name": "deployable_top_policy",
                "status": "fail",
                "read": f"top policy {top_policy} uses target labels and is not deployable.",
                "evidence": {"top_policy": top_policy},
            }
        )
    else:
        gates.append(
            {
                "name": "deployable_top_policy",
                "status": "pass" if top_policy else "fail",
                "read": f"top non-oracle policy is {top_policy}." if top_policy else "no deployable policy found.",
                "evidence": {"top_policy": top_policy},
            }
        )
    return gates


def _decision(gates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = {str(gate.get("name")): str(gate.get("status")) for gate in gates}
    hard_fail = any(status == "fail" for name, status in statuses.items() if name in {"oracle_present", "deployable_top_policy"})
    statistical_pass = statuses.get("baseline_delta_ci") == "pass"
    oracle_pass = statuses.get("oracle_capture") == "pass"
    enough_episodes = statuses.get("episode_count") == "pass"
    if not hard_fail and statistical_pass and oracle_pass and enough_episodes:
        return {
            "downstream_training": "small-smoke-ok",
            "next_cpu_gate": "optional-robustness-scale",
            "read": "offline coverage gate is strong enough for a tiny downstream smoke, not a large training spend.",
        }
    return {
        "downstream_training": "hold",
        "next_cpu_gate": "oracle-plus-episode-bootstrap",
        "read": "keep this in CPU/offline benchmark mode until oracle, confidence, and episode-count gates pass.",
    }


def _top_non_target_policy(policy_final: Mapping[str, Mapping[str, Any]]) -> tuple[str | None, Mapping[str, Any] | None]:
    candidates = {
        policy: row
        for policy, row in policy_final.items()
        if not bool(row.get("uses_target_for_selection", False)) and row.get("mean_final_gain") is not None
    }
    if not candidates:
        return None, None
    policy = max(candidates, key=lambda item: (float(candidates[item].get("mean_final_gain", 0.0)), item))
    return policy, candidates[policy]


def _bootstrap_ci95(values: Sequence[float], *, seed: int, replicates: int) -> tuple[float | None, float | None]:
    clean = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=float)
    if clean.size == 0:
        return None, None
    if clean.size == 1 or replicates <= 0:
        value = float(np.mean(clean))
        return value, value
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, clean.size, size=(int(replicates), clean.size))
    means = np.mean(clean[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _win_fraction(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(float(value) > 0.0 for value in values) / len(values))


def _ranked_items(rows: Mapping[str, Mapping[str, Any]], metric: str) -> list[tuple[str, Mapping[str, Any]]]:
    return sorted(rows.items(), key=lambda item: _none_low(item[1].get(metric)), reverse=True)


def _none_low(value: Any) -> float:
    if value is None:
        return float("-inf")
    return float(value)


def _gt(value: Any, threshold: float) -> bool:
    return value is not None and float(value) > float(threshold)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def _pool_range(value: Any) -> str:
    if not isinstance(value, Mapping) or value.get("min") is None:
        return "unknown"
    return f"{int(value.get('min', 0))}-{int(value.get('max', 0))} clips, median {float(value.get('median', 0.0)):.1f}"
