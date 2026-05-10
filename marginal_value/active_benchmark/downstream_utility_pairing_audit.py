from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Mapping, Sequence


def build_utility_pairing_audit(
    reports_by_seed: Mapping[int, Mapping[str, Any]],
    *,
    random_policy: str = "random_valid",
    baseline_policy: str = "kcenter_quality_gated_ts2vec",
    quality_control_policy: str = "kcenter_quality_gated_window",
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260509,
) -> dict[str, object]:
    rows = _rows_with_seed(reports_by_seed)
    total_rows = _total_rows(rows)
    paired_rows = _paired_rows(total_rows, random_policy=str(random_policy))
    policy_summary = _policy_summary(
        paired_rows,
        random_policy=str(random_policy),
        bootstrap_replicates=int(bootstrap_replicates),
        bootstrap_seed=int(bootstrap_seed),
    )
    return {
        "input": {
            "seed_count": int(len(reports_by_seed)),
            "seeds": sorted(int(seed) for seed in reports_by_seed),
            "random_policy": str(random_policy),
            "baseline_policy": str(baseline_policy),
            "quality_control_policy": str(quality_control_policy),
            "bootstrap_replicates": int(bootstrap_replicates),
            "bootstrap_seed": int(bootstrap_seed),
        },
        "decision": _decision(
            policy_summary,
            baseline_policy=str(baseline_policy),
            random_policy=str(random_policy),
            quality_control_policy=str(quality_control_policy),
        ),
        "policy_paired_summary": policy_summary,
        "paired_rows": paired_rows,
        "total_rows": total_rows,
    }


def write_utility_pairing_audit_reports(audit: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "downstream_utility_pairing_audit.json"
    markdown_path = root / "downstream_utility_pairing_audit.md"
    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_markdown_report(audit), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def _rows_with_seed(reports_by_seed: Mapping[int, Mapping[str, Any]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed, report in sorted(reports_by_seed.items()):
        for row in report.get("rows", []):
            if not isinstance(row, Mapping):
                continue
            enriched = dict(row)
            enriched["seed"] = int(seed)
            rows.append(enriched)
    return rows


def _total_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    by_key: dict[tuple[int, str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_key[
            (
                int(row.get("seed", 0)),
                str(row.get("episode_id", "")),
                str(row.get("representation", "")),
                str(row.get("policy_name", "")),
            )
        ].append(row)

    out: list[dict[str, object]] = []
    for (seed, episode_id, representation, policy_name), group in sorted(by_key.items()):
        ordered = sorted(group, key=lambda item: int(item.get("round_index", -1)))
        if not ordered:
            continue
        first = ordered[0]
        last = ordered[-1]
        initial_error = float(first.get("baseline_reconstruction_error", 0.0))
        final_error = float(last.get("after_reconstruction_error", 0.0))
        total_absolute_gain = initial_error - final_error
        total_relative_gain = total_absolute_gain / initial_error if initial_error > 1.0e-12 else 0.0
        out.append(
            {
                "seed": seed,
                "episode_id": episode_id,
                "representation": representation,
                "policy_name": policy_name,
                "round_count": int(len(ordered)),
                "initial_reconstruction_error": initial_error,
                "final_reconstruction_error": final_error,
                "total_absolute_reconstruction_gain": float(total_absolute_gain),
                "total_relative_reconstruction_gain": float(total_relative_gain),
                "final_incremental_absolute_reconstruction_gain": float(
                    last.get("absolute_reconstruction_gain", 0.0)
                ),
                "final_incremental_relative_reconstruction_gain": float(
                    last.get("relative_reconstruction_gain", 0.0)
                ),
            }
        )
    return out


def _paired_rows(
    total_rows: Sequence[Mapping[str, object]],
    *,
    random_policy: str,
) -> list[dict[str, object]]:
    random_by_unit = {
        (int(row["seed"]), str(row["episode_id"]), str(row["representation"])): row
        for row in total_rows
        if str(row["policy_name"]) == random_policy
    }
    paired: list[dict[str, object]] = []
    for row in sorted(
        total_rows,
        key=lambda item: (
            int(item["seed"]),
            str(item["episode_id"]),
            str(item["representation"]),
            str(item["policy_name"]),
        ),
    ):
        random_row = random_by_unit.get((int(row["seed"]), str(row["episode_id"]), str(row["representation"])))
        if random_row is None:
            continue
        total_delta = float(row["total_relative_reconstruction_gain"]) - float(
            random_row["total_relative_reconstruction_gain"]
        )
        final_delta = float(row["final_incremental_relative_reconstruction_gain"]) - float(
            random_row["final_incremental_relative_reconstruction_gain"]
        )
        paired.append(
            {
                **dict(row),
                "random_policy": random_policy,
                "total_relative_reconstruction_gain_minus_random": float(total_delta),
                "final_incremental_relative_reconstruction_gain_minus_random": float(final_delta),
                "final_reconstruction_error_minus_random": float(row["final_reconstruction_error"])
                - float(random_row["final_reconstruction_error"]),
            }
        )
    return paired


def _policy_summary(
    paired_rows: Sequence[Mapping[str, object]],
    *,
    random_policy: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, dict[str, object]]:
    by_policy: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in paired_rows:
        by_policy[str(row["policy_name"])].append(row)

    summary: dict[str, dict[str, object]] = {}
    for index, (policy, rows) in enumerate(sorted(by_policy.items())):
        deltas = [float(row["total_relative_reconstruction_gain_minus_random"]) for row in rows]
        ci_low, ci_high = _bootstrap_ci95(
            deltas,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + index,
        )
        summary[policy] = {
            "row_count": int(len(rows)),
            "mean_total_relative_reconstruction_gain": _mean(rows, "total_relative_reconstruction_gain"),
            "mean_total_absolute_reconstruction_gain": _mean(rows, "total_absolute_reconstruction_gain"),
            "mean_final_incremental_relative_reconstruction_gain": _mean(
                rows, "final_incremental_relative_reconstruction_gain"
            ),
            "mean_total_relative_reconstruction_gain_minus_random": _mean(
                rows, "total_relative_reconstruction_gain_minus_random"
            ),
            "stdev_total_relative_reconstruction_gain_minus_random": _stdev(
                rows, "total_relative_reconstruction_gain_minus_random"
            ),
            "bootstrap_ci95_low": ci_low,
            "bootstrap_ci95_high": ci_high,
            "mean_final_incremental_relative_reconstruction_gain_minus_random": _mean(
                rows, "final_incremental_relative_reconstruction_gain_minus_random"
            ),
            "total_relative_win_count_vs_random": _win_count(rows, "total_relative_reconstruction_gain_minus_random"),
            "total_relative_loss_count_vs_random": _loss_count(rows, "total_relative_reconstruction_gain_minus_random"),
            "total_relative_tie_count_vs_random": _tie_count(rows, "total_relative_reconstruction_gain_minus_random"),
            "seed_mean_deltas": _seed_mean_deltas(rows),
        }
    if random_policy in summary:
        summary[random_policy]["read"] = "reference policy"
    return summary


def _decision(
    policy_summary: Mapping[str, Mapping[str, object]],
    *,
    baseline_policy: str,
    random_policy: str,
    quality_control_policy: str,
) -> dict[str, object]:
    best_total = _best_policy(policy_summary, "mean_total_relative_reconstruction_gain")
    best_delta = _best_policy(policy_summary, "mean_total_relative_reconstruction_gain_minus_random")
    baseline = policy_summary.get(baseline_policy, {})
    control = policy_summary.get(quality_control_policy, {})
    baseline_delta = _metric_value(baseline, "mean_total_relative_reconstruction_gain_minus_random")
    control_delta = _metric_value(control, "mean_total_relative_reconstruction_gain_minus_random")
    ci_low = _metric_value(baseline, "bootstrap_ci95_low")
    beats_random = baseline_delta is not None and baseline_delta > 0.0
    beats_control = (
        baseline_delta is not None
        and control_delta is not None
        and baseline_delta > control_delta
    )
    ci_clears_zero = ci_low is not None and ci_low > 0.0
    if beats_random and beats_control and ci_clears_zero:
        next_gate = "scale_active_benchmark_not_training"
        read = (
            f"{baseline_policy} clears {random_policy} and {quality_control_policy} on paired reconstruction utility "
            "with a positive bootstrap interval; this is promotive for a larger active-learning benchmark, not training proof."
        )
    elif beats_random and beats_control:
        next_gate = "repeat_or_expand_active_benchmark"
        read = (
            f"{baseline_policy} beats {random_policy} and {quality_control_policy} on mean paired reconstruction utility, "
            "but the bootstrap interval is not strong enough for downstream training."
        )
    elif beats_random:
        next_gate = "fix_quality_control_gap"
        read = (
            f"{baseline_policy} beats {random_policy} on mean paired reconstruction utility but does not clear "
            f"{quality_control_policy}; this is not training proof, so improve acquisition or benchmark before training."
        )
    else:
        next_gate = "hold_training"
        read = (
            f"{baseline_policy} does not beat {random_policy} on paired reconstruction utility; do not train."
        )
    return {
        "downstream_training": "hold",
        "next_gate": next_gate,
        "read": read,
        "best_total_relative_reconstruction_policy": best_total,
        "best_paired_delta_vs_random_policy": best_delta,
        "baseline_policy": baseline_policy,
        "random_policy": random_policy,
        "quality_control_policy": quality_control_policy,
        "baseline_mean_total_relative_reconstruction_gain": _metric_value(
            baseline, "mean_total_relative_reconstruction_gain"
        ),
        "baseline_minus_random_total_relative_reconstruction_gain": baseline_delta,
        "baseline_bootstrap_ci95_low": ci_low,
        "baseline_bootstrap_ci95_high": _metric_value(baseline, "bootstrap_ci95_high"),
        "baseline_total_win_count_vs_random": _metric_value(baseline, "total_relative_win_count_vs_random"),
        "baseline_total_loss_count_vs_random": _metric_value(baseline, "total_relative_loss_count_vs_random"),
    }


def _mean(rows: Sequence[Mapping[str, object]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in rows]
    return float(fmean(values)) if values else 0.0


def _stdev(rows: Sequence[Mapping[str, object]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in rows]
    return float(stdev(values)) if len(values) > 1 else 0.0


def _win_count(rows: Sequence[Mapping[str, object]], key: str) -> int:
    return int(sum(1 for row in rows if float(row.get(key, 0.0)) > 0.0))


def _loss_count(rows: Sequence[Mapping[str, object]], key: str) -> int:
    return int(sum(1 for row in rows if float(row.get(key, 0.0)) < 0.0))


def _tie_count(rows: Sequence[Mapping[str, object]], key: str) -> int:
    return int(sum(1 for row in rows if float(row.get(key, 0.0)) == 0.0))


def _seed_mean_deltas(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, float | int]]:
    by_seed: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(row)
    return {
        str(seed): {
            "row_count": int(len(seed_rows)),
            "mean_total_relative_reconstruction_gain_minus_random": _mean(
                seed_rows, "total_relative_reconstruction_gain_minus_random"
            ),
            "win_count_vs_random": _win_count(seed_rows, "total_relative_reconstruction_gain_minus_random"),
            "loss_count_vs_random": _loss_count(seed_rows, "total_relative_reconstruction_gain_minus_random"),
        }
        for seed, seed_rows in sorted(by_seed.items())
    }


def _bootstrap_ci95(values: Sequence[float], *, replicates: int, seed: int) -> tuple[float | None, float | None]:
    clean = [float(value) for value in values]
    if len(clean) < 2 or replicates <= 0:
        return None, None
    rng = random.Random(seed)
    means = []
    for _ in range(int(replicates)):
        sample = [clean[rng.randrange(len(clean))] for _ in clean]
        means.append(float(fmean(sample)))
    means.sort()
    low_index = int(0.025 * (len(means) - 1))
    high_index = int(0.975 * (len(means) - 1))
    return float(means[low_index]), float(means[high_index])


def _best_policy(policy_summary: Mapping[str, Mapping[str, object]], metric: str) -> str | None:
    best_policy = None
    best_value = None
    for policy, values in policy_summary.items():
        value = _metric_value(values, metric)
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_policy = policy
            best_value = value
    return best_policy


def _metric_value(row: object, key: str) -> float | int | None:
    if not isinstance(row, Mapping) or key not in row:
        return None
    value = row[key]
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    return float(value)


def _markdown_report(audit: Mapping[str, Any]) -> str:
    decision = audit.get("decision", {})
    summary = audit.get("policy_paired_summary", {})
    lines = [
        "# Downstream Utility Pairing Audit",
        "",
        "This aggregates round-loop hidden-target reconstruction utility from the common initial support baseline.",
        "",
        "## Decision",
        "",
        f"- downstream training: `{_get(decision, 'downstream_training')}`",
        f"- next gate: `{_get(decision, 'next_gate')}`",
        f"- read: {_get(decision, 'read')}",
        "",
        "## Mean Total Relative Reconstruction Gain",
        "",
        "| policy | total_gain | delta_vs_random | ci95_delta | wins/losses |",
        "|---|---:|---:|---:|---:|",
    ]
    if isinstance(summary, Mapping):
        rows = sorted(
            summary.items(),
            key=lambda item: float(item[1].get("mean_total_relative_reconstruction_gain", 0.0))
            if isinstance(item[1], Mapping)
            else 0.0,
            reverse=True,
        )
        for policy, values in rows:
            if not isinstance(values, Mapping):
                continue
            lines.append(
                f"| `{policy}` | {_fmt(values.get('mean_total_relative_reconstruction_gain'))} | "
                f"{_fmt(values.get('mean_total_relative_reconstruction_gain_minus_random'))} | "
                f"[{_fmt(values.get('bootstrap_ci95_low'))}, {_fmt(values.get('bootstrap_ci95_high'))}] | "
                f"{values.get('total_relative_win_count_vs_random')}/{values.get('total_relative_loss_count_vs_random')} |"
            )
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "- This is frozen-feature reconstruction utility. It can justify improving or scaling the active-learning benchmark; it is not challenge-label downstream training proof.",
        ]
    )
    return "\n".join(lines) + "\n"


def _get(row: object, key: str) -> object:
    return row.get(key) if isinstance(row, Mapping) else None


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (float, int)):
        return f"{float(value):.6f}"
    return str(value)
