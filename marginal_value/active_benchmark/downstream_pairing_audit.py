from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Mapping, Sequence


def build_supervised_pairing_audit(
    reports_by_seed: Mapping[int, Mapping[str, Any]],
    *,
    random_policy: str = "random_valid",
    tolerance: float = 1.0e-9,
) -> dict[str, object]:
    rows = _rows_with_seed(reports_by_seed)
    final_rows = _final_rows(rows)
    baseline_audit = _baseline_audit(rows, final_rows=final_rows, tolerance=float(tolerance))
    paired_rows = _paired_rows(rows, final_rows=final_rows, random_policy=str(random_policy))
    policy_summary = _policy_summary(paired_rows, random_policy=str(random_policy))
    return {
        "input": {
            "seed_count": int(len(reports_by_seed)),
            "seeds": sorted(int(seed) for seed in reports_by_seed),
            "random_policy": str(random_policy),
        },
        "decision": _decision(baseline_audit, policy_summary, random_policy=str(random_policy)),
        "baseline_audit": baseline_audit,
        "policy_paired_summary": policy_summary,
        "paired_rows": paired_rows,
    }


def write_supervised_pairing_audit_reports(audit: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "downstream_supervised_pairing_audit.json"
    markdown_path = root / "downstream_supervised_pairing_audit.md"
    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_markdown_report(audit), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def _rows_with_seed(reports_by_seed: Mapping[int, Mapping[str, Any]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for seed, report in sorted(reports_by_seed.items()):
        for row in report.get("rows", []):
            if not isinstance(row, Mapping):
                continue
            enriched = dict(row)
            enriched["seed"] = int(seed)
            out.append(enriched)
    return out


def _final_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    max_round_by_key: dict[tuple[int, str, str, str], int] = {}
    for row in rows:
        key = _policy_key(row)
        max_round_by_key[key] = max(max_round_by_key.get(key, -1), int(row.get("round_index", -1)))
    return [
        dict(row)
        for row in rows
        if int(row.get("round_index", -1)) == max_round_by_key[_policy_key(row)]
    ]


def _baseline_audit(
    rows: Sequence[Mapping[str, object]],
    *,
    final_rows: Sequence[Mapping[str, object]],
    tolerance: float,
) -> dict[str, object]:
    round0_rows = [row for row in rows if int(row.get("round_index", -1)) == 0]
    round0_checks = _baseline_range_rows(round0_rows, tolerance=tolerance)
    final_checks = _baseline_range_rows(final_rows, tolerance=tolerance)
    return {
        "round0_baseline_consistent": all(row["baseline_consistent"] for row in round0_checks),
        "final_round_baseline_varies": any(not row["baseline_consistent"] for row in final_checks),
        "round0_rows": round0_checks,
        "final_round_rows": final_checks,
        "read": (
            "Round-0 baselines are common across policies; final-row baseline variation means final-row "
            "accuracy_gain is a final-round incremental metric, not total acquisition gain."
        ),
    }


def _baseline_range_rows(rows: Sequence[Mapping[str, object]], *, tolerance: float) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str, str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                int(row.get("seed", 0)),
                str(row.get("episode_id", "")),
                str(row.get("representation", "")),
                int(row.get("round_index", -1)),
            )
        ].append(row)
    out = []
    for (seed, episode_id, representation, round_index), group in sorted(grouped.items()):
        acc_values = [float(row.get("baseline_accuracy", 0.0)) for row in group]
        nll_values = [float(row.get("baseline_negative_log_likelihood", 0.0)) for row in group]
        acc_range = max(acc_values) - min(acc_values) if acc_values else 0.0
        nll_range = max(nll_values) - min(nll_values) if nll_values else 0.0
        out.append(
            {
                "seed": seed,
                "episode_id": episode_id,
                "representation": representation,
                "round_index": round_index,
                "policy_count": int(len(group)),
                "baseline_accuracy_range": float(acc_range),
                "baseline_nll_range": float(nll_range),
                "baseline_consistent": bool(acc_range <= tolerance and nll_range <= tolerance),
            }
        )
    return out


def _paired_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    final_rows: Sequence[Mapping[str, object]],
    random_policy: str,
) -> list[dict[str, object]]:
    initial_by_key = _initial_baseline_by_key(rows)
    final_by_key = {
        (
            int(row.get("seed", 0)),
            str(row.get("episode_id", "")),
            str(row.get("representation", "")),
            str(row.get("policy_name", "")),
        ): row
        for row in final_rows
    }
    random_by_unit = {
        (seed, episode_id, representation): row
        for (seed, episode_id, representation, policy), row in final_by_key.items()
        if policy == random_policy
    }
    out = []
    for (seed, episode_id, representation, policy), row in sorted(final_by_key.items()):
        initial = initial_by_key.get((seed, episode_id, representation))
        random_row = random_by_unit.get((seed, episode_id, representation))
        if initial is None or random_row is None:
            continue
        policy_metrics = _total_metrics(row, initial)
        random_metrics = _total_metrics(random_row, initial)
        out.append(
            {
                "seed": seed,
                "episode_id": episode_id,
                "representation": representation,
                "policy_name": policy,
                "random_policy": random_policy,
                **policy_metrics,
                "after_accuracy_minus_random": float(policy_metrics["after_accuracy"] - random_metrics["after_accuracy"]),
                "total_accuracy_gain_minus_random": float(
                    policy_metrics["total_accuracy_gain"] - random_metrics["total_accuracy_gain"]
                ),
                "total_nll_reduction_minus_random": float(
                    policy_metrics["total_nll_reduction"] - random_metrics["total_nll_reduction"]
                ),
                "final_incremental_accuracy_gain_minus_random": float(
                    policy_metrics["final_incremental_accuracy_gain"]
                    - random_metrics["final_incremental_accuracy_gain"]
                ),
                "final_incremental_nll_reduction_minus_random": float(
                    policy_metrics["final_incremental_nll_reduction"]
                    - random_metrics["final_incremental_nll_reduction"]
                ),
            }
        )
    return out


def _initial_baseline_by_key(rows: Sequence[Mapping[str, object]]) -> dict[tuple[int, str, str], dict[str, float]]:
    grouped: dict[tuple[int, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if int(row.get("round_index", -1)) != 0:
            continue
        grouped[(int(row.get("seed", 0)), str(row.get("episode_id", "")), str(row.get("representation", "")))].append(row)
    out = {}
    for key, group in grouped.items():
        first = group[0]
        out[key] = {
            "initial_baseline_accuracy": float(first.get("baseline_accuracy", 0.0)),
            "initial_baseline_nll": float(first.get("baseline_negative_log_likelihood", 0.0)),
        }
    return out


def _total_metrics(row: Mapping[str, object], initial: Mapping[str, float]) -> dict[str, float]:
    after_accuracy = float(row.get("after_accuracy", 0.0))
    after_nll = float(row.get("after_negative_log_likelihood", 0.0))
    initial_accuracy = float(initial["initial_baseline_accuracy"])
    initial_nll = float(initial["initial_baseline_nll"])
    return {
        "after_accuracy": after_accuracy,
        "after_negative_log_likelihood": after_nll,
        "initial_baseline_accuracy": initial_accuracy,
        "initial_baseline_nll": initial_nll,
        "total_accuracy_gain": float(after_accuracy - initial_accuracy),
        "total_nll_reduction": float(initial_nll - after_nll),
        "final_incremental_accuracy_gain": float(row.get("accuracy_gain", 0.0)),
        "final_incremental_nll_reduction": float(row.get("nll_reduction", 0.0)),
    }


def _policy_summary(rows: Sequence[Mapping[str, object]], *, random_policy: str) -> dict[str, dict[str, float | int]]:
    by_policy: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_policy[str(row["policy_name"])].append(row)
    summary = {}
    for policy, policy_rows in sorted(by_policy.items()):
        summary[policy] = {
            "row_count": int(len(policy_rows)),
            "mean_after_accuracy": _mean(policy_rows, "after_accuracy"),
            "mean_total_accuracy_gain": _mean(policy_rows, "total_accuracy_gain"),
            "mean_total_nll_reduction": _mean(policy_rows, "total_nll_reduction"),
            "mean_after_accuracy_minus_random": _mean(policy_rows, "after_accuracy_minus_random"),
            "mean_total_accuracy_gain_minus_random": _mean(policy_rows, "total_accuracy_gain_minus_random"),
            "mean_total_nll_reduction_minus_random": _mean(policy_rows, "total_nll_reduction_minus_random"),
            "mean_final_incremental_accuracy_gain_minus_random": _mean(
                policy_rows, "final_incremental_accuracy_gain_minus_random"
            ),
            "mean_final_incremental_nll_reduction_minus_random": _mean(
                policy_rows, "final_incremental_nll_reduction_minus_random"
            ),
            "stdev_after_accuracy_minus_random": _stdev(policy_rows, "after_accuracy_minus_random"),
            "after_accuracy_win_count_vs_random": _win_count(policy_rows, "after_accuracy_minus_random"),
            "total_accuracy_gain_win_count_vs_random": _win_count(policy_rows, "total_accuracy_gain_minus_random"),
            "total_nll_reduction_win_count_vs_random": _win_count(policy_rows, "total_nll_reduction_minus_random"),
        }
    if random_policy in summary:
        summary[random_policy]["read"] = "reference policy"
    return summary


def _decision(
    baseline_audit: Mapping[str, object],
    policy_summary: Mapping[str, Mapping[str, float | int]],
    *,
    random_policy: str,
) -> dict[str, object]:
    best_after = _best_policy(policy_summary, "mean_after_accuracy")
    best_total_gain = _best_policy(policy_summary, "mean_total_accuracy_gain")
    quality_stratified_present = "quality_stratified_random" in policy_summary
    next_gate = "quality_stratified_random_repeat"
    read = (
        f"Do not train yet: final-row accuracy_gain is a final-round incremental metric. "
        f"Use paired total gains from the common round-0 baseline and compare against {random_policy}."
    )
    if quality_stratified_present:
        next_gate = "hold_proxy_not_promotive"
        read = (
            "Do not train: quality_stratified_random repeat is complete, and this source-family "
            "pseudo-label proxy should be treated as non-promotive unless a TS2Vec policy beats "
            f"both {random_policy} and the quality-matched control on paired total utility."
        )
    return {
        "downstream_training": "hold",
        "next_gate": next_gate,
        "best_after_accuracy_policy": best_after,
        "best_total_accuracy_gain_policy": best_total_gain,
        "quality_stratified_random_present": quality_stratified_present,
        "read": read,
        "baseline_audit_read": str(baseline_audit.get("read", "")),
    }


def _best_policy(policy_summary: Mapping[str, Mapping[str, float | int]], metric: str) -> str:
    if not policy_summary:
        return ""
    return max(policy_summary, key=lambda policy: float(policy_summary[policy].get(metric, 0.0)))


def _policy_key(row: Mapping[str, object]) -> tuple[int, str, str, str]:
    return (
        int(row.get("seed", 0)),
        str(row.get("episode_id", "")),
        str(row.get("representation", "")),
        str(row.get("policy_name", "")),
    )


def _mean(rows: Sequence[Mapping[str, object]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in rows]
    return float(fmean(values)) if values else 0.0


def _stdev(rows: Sequence[Mapping[str, object]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in rows]
    return float(stdev(values)) if len(values) > 1 else 0.0


def _win_count(rows: Sequence[Mapping[str, object]], key: str) -> int:
    return int(sum(1 for row in rows if float(row.get(key, 0.0)) > 0.0))


def _markdown_report(audit: Mapping[str, Any]) -> str:
    decision = audit.get("decision", {})
    baseline_audit = audit.get("baseline_audit", {})
    policy_summary = audit.get("policy_paired_summary", {})
    lines = [
        "# Downstream Supervised Pairing Audit",
        "",
        "## Decision",
        "",
        f"- downstream training: `{decision.get('downstream_training')}`",
        f"- next gate: `{decision.get('next_gate')}`",
        f"- read: {decision.get('read')}",
        "",
        "## Baseline Audit",
        "",
        f"- round-0 baseline consistent: `{baseline_audit.get('round0_baseline_consistent')}`",
        f"- final-round baseline varies: `{baseline_audit.get('final_round_baseline_varies')}`",
        f"- note: {baseline_audit.get('read')}",
        "",
        "## Paired Policy Summary",
        "",
        "| policy | after_acc | total_acc_gain | final_incremental_delta_vs_random | after_acc_delta_vs_random | total_gain_delta_vs_random | total_nll_delta_vs_random | wins_after |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if isinstance(policy_summary, Mapping):
        for policy, row in sorted(policy_summary.items()):
            if not isinstance(row, Mapping):
                continue
            lines.append(
                f"| `{policy}` | {_fmt(row.get('mean_after_accuracy'))} | "
                f"{_fmt(row.get('mean_total_accuracy_gain'))} | "
                f"{_fmt(row.get('mean_final_incremental_accuracy_gain_minus_random'))} | "
                f"{_fmt(row.get('mean_after_accuracy_minus_random'))} | "
                f"{_fmt(row.get('mean_total_accuracy_gain_minus_random'))} | "
                f"{_fmt(row.get('mean_total_nll_reduction_minus_random'))} | "
                f"{int(row.get('after_accuracy_win_count_vs_random', 0))} |"
            )
    return "\n".join(lines).rstrip() + "\n"


def _fmt(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return "n/a"
