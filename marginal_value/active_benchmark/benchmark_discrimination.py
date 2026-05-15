from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


ORACLE_POLICY = "oracle_greedy_eval_only"
RANDOM_POLICY = "random_valid"
RANDOM_REPLAY_PREFIX = "random_valid_replay_"


def build_benchmark_discrimination_report(
    run_root: str | Path,
    *,
    near_zero_threshold: float = 0.10,
    oracle_gap_fraction_floor: float = 0.01,
    minimum_oracle_gap_fraction: float = 0.20,
    minimum_random_replay_percentile: float = 0.90,
) -> dict[str, Any]:
    root = Path(run_root)
    reports = _load_seed_reports(root)
    final_rows = _final_rows(reports)
    episode_keys = sorted({key[:2] for key in final_rows})
    policies = sorted({key[2] for key in final_rows})
    random_replay_policies = sorted(policy for policy in policies if policy.startswith(RANDOM_REPLAY_PREFIX))
    comparable_policies = [
        policy
        for policy in policies
        if policy not in {ORACLE_POLICY, RANDOM_POLICY} and not policy.startswith(RANDOM_REPLAY_PREFIX)
    ]
    difficulty_by_key = _difficulty_by_episode_key(reports)
    gains_by_policy = _gains_by_policy(final_rows)
    final_means = {policy: _mean(values) for policy, values in sorted(gains_by_policy.items())}
    paired_policy_deltas = _paired_policy_deltas(final_rows, comparable_policies, oracle_gap_fraction_floor=oracle_gap_fraction_floor)
    oracle_opportunity = _oracle_opportunity(final_rows, episode_keys)
    strata = _strata(
        final_rows,
        episode_keys,
        comparable_policies,
        difficulty_by_key=difficulty_by_key,
        near_zero_threshold=near_zero_threshold,
    )
    random_replay_audit = _random_replay_audit(
        final_rows,
        comparable_policies,
        random_replay_policies=random_replay_policies,
    )
    decision = _decision(
        final_means=final_means,
        paired_policy_deltas=paired_policy_deltas,
        random_replay_audit=random_replay_audit,
        minimum_oracle_gap_fraction=minimum_oracle_gap_fraction,
        minimum_random_replay_percentile=minimum_random_replay_percentile,
    )
    return {
        "input": {
            "run_root": str(root),
            "report_count": len(reports),
            "episode_count": len(episode_keys),
            "policy_count": len(policies),
            "near_zero_threshold": float(near_zero_threshold),
            "oracle_gap_fraction_floor": float(oracle_gap_fraction_floor),
            "minimum_oracle_gap_fraction": float(minimum_oracle_gap_fraction),
            "minimum_random_replay_percentile": float(minimum_random_replay_percentile),
        },
        "decision": decision,
        "policy_final_means": final_means,
        "paired_policy_deltas": paired_policy_deltas,
        "oracle_opportunity": oracle_opportunity,
        "strata": strata,
        "random_replay_audit": random_replay_audit,
    }


def write_benchmark_discrimination_reports(
    run_root: str | Path,
    *,
    output_json: str | Path,
    output_markdown: str | Path,
    near_zero_threshold: float = 0.10,
) -> dict[str, str]:
    report = build_benchmark_discrimination_report(
        run_root,
        near_zero_threshold=near_zero_threshold,
    )
    json_path = Path(output_json)
    markdown_path = Path(output_markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_benchmark_discrimination_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def render_benchmark_discrimination_markdown(report: dict[str, Any]) -> str:
    decision = report.get("decision", {})
    random_replay = report.get("random_replay_audit", {})
    lines = [
        "# Offline Benchmark Discrimination Audit",
        "",
        "## Decision",
        "",
        f"- benchmark discrimination: `{decision.get('benchmark_discrimination', '')}`",
        f"- top non-oracle policy: `{decision.get('top_non_oracle_policy', '')}`",
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
            "## Oracle Opportunity",
            "",
            "| metric | value |",
            "|---|---:|",
        ]
    )
    for key in (
        "episode_count",
        "mean_oracle_minus_random",
        "median_oracle_minus_random",
        "min_oracle_minus_random",
        "max_oracle_minus_random",
        "near_zero_oracle_minus_random_fraction",
    ):
        lines.append(f"| {key} | {_fmt(report.get('oracle_opportunity', {}).get(key))} |")

    lines.extend(
        [
            "",
            "## Episode-Level Deltas Vs Random",
            "",
            "| comparison | episodes | mean delta | median delta | CI95 low | CI95 high | wins | losses | oracle gap fraction |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in sorted(report.get("paired_policy_deltas", {}).items()):
        lines.append(
            "| {name} | {episodes} | {mean_delta} | {median_delta} | {low} | {high} | {wins} | {losses} | {fraction} |".format(
                name=name,
                episodes=int(row.get("episode_count", 0)),
                mean_delta=_fmt(row.get("mean_delta_final_cumulative_gain")),
                median_delta=_fmt(row.get("median_delta_final_cumulative_gain")),
                low=_fmt(row.get("bootstrap_ci95_low")),
                high=_fmt(row.get("bootstrap_ci95_high")),
                wins=int(row.get("win_count", 0)),
                losses=int(row.get("loss_count", 0)),
                fraction=_fmt(row.get("mean_oracle_gap_fraction_captured")),
            )
        )

    lines.extend(
        [
            "",
            "## Random Replay Audit",
            "",
            f"- status: `{random_replay.get('status', '')}`",
            f"- replay policies: `{random_replay.get('replay_policy_count', 0)}`",
            f"- read: {random_replay.get('read', '')}",
            "",
            "| policy | episodes | mean percentile | mean minus replay mean | mean minus replay p90 | mean minus replay p95 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for policy, row in sorted(random_replay.get("policy_percentiles", {}).items()):
        lines.append(
            "| {policy} | {episodes} | {percentile} | {mean_delta} | {p90_delta} | {p95_delta} |".format(
                policy=policy,
                episodes=int(row.get("episode_count", 0)),
                percentile=_fmt(row.get("mean_random_replay_percentile")),
                mean_delta=_fmt(row.get("mean_gain_minus_random_replay_mean")),
                p90_delta=_fmt(row.get("mean_gain_minus_random_replay_p90")),
                p95_delta=_fmt(row.get("mean_gain_minus_random_replay_p95")),
            )
        )

    lines.extend(
        [
            "",
            "## Oracle-Opportunity Strata",
            "",
            "| stratum | episodes | top policy | top gain | random gain | oracle gain | top minus random | oracle minus random | fraction captured |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in report.get("strata", {}).items():
        lines.append(
            "| {name} | {episodes} | {top} | {top_gain} | {random_gain} | {oracle_gain} | {top_delta} | {oracle_delta} | {fraction} |".format(
                name=name,
                episodes=int(row.get("episode_count", 0)),
                top=row.get("top_non_oracle_policy", ""),
                top_gain=_fmt(row.get("top_non_oracle_gain")),
                random_gain=_fmt(row.get("random_valid_gain")),
                oracle_gain=_fmt(row.get("oracle_gain")),
                top_delta=_fmt(row.get("top_minus_random")),
                oracle_delta=_fmt(row.get("oracle_minus_random")),
                fraction=_fmt(row.get("top_oracle_gap_fraction")),
            )
        )

    lines.extend(
        [
            "",
            "## Policy Final Means",
            "",
            "| policy | mean final cumulative gain |",
            "|---|---:|",
        ]
    )
    final_means = report.get("policy_final_means", {})
    for policy in sorted(final_means, key=lambda item: _none_low(final_means.get(item)), reverse=True):
        lines.append(f"| {policy} | {_fmt(final_means.get(policy))} |")
    lines.append("")
    return "\n".join(lines)


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


def _final_rows(reports: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for report in reports:
        for row in report.get("rounds", []):
            key = (str(row.get("_seed_name", "")), str(row.get("episode_id", "")), str(row.get("policy_name", "")))
            current = latest.get(key)
            if current is None or int(row.get("round_index", 0)) > int(current.get("round_index", 0)):
                latest[key] = row
    return latest


def _difficulty_by_episode_key(reports: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    output = {}
    for report in reports:
        for row in report.get("difficulty_audit", []):
            output[(str(row.get("_seed_name", "")), str(row.get("episode_id", "")))] = row
    return output


def _gains_by_policy(final_rows: dict[tuple[str, str, str], dict[str, Any]]) -> dict[str, list[float]]:
    output: dict[str, list[float]] = defaultdict(list)
    for (_seed_name, _episode_id, policy), row in final_rows.items():
        output[policy].append(_float(row.get("cumulative_balanced_relative_gain")))
    return output


def _paired_policy_deltas(
    final_rows: dict[tuple[str, str, str], dict[str, Any]],
    policies: list[str],
    *,
    oracle_gap_fraction_floor: float,
) -> dict[str, dict[str, Any]]:
    episode_keys = sorted({key[:2] for key in final_rows})
    output = {}
    for policy in policies:
        deltas = []
        fractions = []
        for episode_key in episode_keys:
            random_row = final_rows.get((*episode_key, RANDOM_POLICY))
            policy_row = final_rows.get((*episode_key, policy))
            oracle_row = final_rows.get((*episode_key, ORACLE_POLICY))
            if random_row is None or policy_row is None:
                continue
            policy_gain = _float(policy_row.get("cumulative_balanced_relative_gain"))
            random_gain = _float(random_row.get("cumulative_balanced_relative_gain"))
            delta = policy_gain - random_gain
            deltas.append(delta)
            if oracle_row is not None:
                oracle_gap = _float(oracle_row.get("cumulative_balanced_relative_gain")) - random_gain
                if oracle_gap > float(oracle_gap_fraction_floor):
                    fractions.append(delta / oracle_gap)
        if not deltas:
            continue
        low, high = _bootstrap_ci95(deltas, seed=20260506 + len(output))
        output[f"{policy}_vs_{RANDOM_POLICY}"] = {
            "policy": policy,
            "baseline_policy": RANDOM_POLICY,
            "episode_count": len(deltas),
            "mean_delta_final_cumulative_gain": _mean(deltas),
            "median_delta_final_cumulative_gain": _median(deltas),
            "bootstrap_ci95_low": low,
            "bootstrap_ci95_high": high,
            "win_count": sum(1 for value in deltas if value > 0.0),
            "loss_count": sum(1 for value in deltas if value < 0.0),
            "tie_count": sum(1 for value in deltas if math.isclose(value, 0.0, rel_tol=1.0e-12, abs_tol=1.0e-12)),
            "win_fraction": sum(1 for value in deltas if value > 0.0) / len(deltas),
            "mean_oracle_gap_fraction_captured": _mean(fractions),
        }
    return output


def _oracle_opportunity(
    final_rows: dict[tuple[str, str, str], dict[str, Any]],
    episode_keys: list[tuple[str, str]],
) -> dict[str, Any]:
    gaps = []
    for episode_key in episode_keys:
        random_row = final_rows.get((*episode_key, RANDOM_POLICY))
        oracle_row = final_rows.get((*episode_key, ORACLE_POLICY))
        if random_row is None or oracle_row is None:
            continue
        gaps.append(_float(oracle_row.get("cumulative_balanced_relative_gain")) - _float(random_row.get("cumulative_balanced_relative_gain")))
    return {
        "episode_count": len(gaps),
        "mean_oracle_minus_random": _mean(gaps),
        "median_oracle_minus_random": _median(gaps),
        "min_oracle_minus_random": min(gaps) if gaps else None,
        "max_oracle_minus_random": max(gaps) if gaps else None,
        "near_zero_oracle_minus_random_fraction": sum(1 for value in gaps if value <= 0.01) / len(gaps) if gaps else None,
    }


def _strata(
    final_rows: dict[tuple[str, str, str], dict[str, Any]],
    episode_keys: list[tuple[str, str]],
    comparable_policies: list[str],
    *,
    difficulty_by_key: dict[tuple[str, str], dict[str, Any]],
    near_zero_threshold: float,
) -> dict[str, dict[str, Any]]:
    opportunity_by_key = {}
    for episode_key in episode_keys:
        random_row = final_rows.get((*episode_key, RANDOM_POLICY))
        oracle_row = final_rows.get((*episode_key, ORACLE_POLICY))
        if random_row is None or oracle_row is None:
            continue
        opportunity_by_key[episode_key] = _float(oracle_row.get("cumulative_balanced_relative_gain")) - _float(
            random_row.get("cumulative_balanced_relative_gain")
        )
    positive_opportunities = [value for value in opportunity_by_key.values() if value > 1.0e-12]
    high_threshold = _median(positive_opportunities) if positive_opportunities else None
    non_near_zero = [
        key
        for key in episode_keys
        if _float(difficulty_by_key.get(key, {}).get("near_zero_oracle_round_fraction", 1.0)) <= near_zero_threshold
    ]
    high_opportunity = [
        key
        for key, value in opportunity_by_key.items()
        if high_threshold is not None and value >= high_threshold
    ]
    low_opportunity = [
        key
        for key, value in opportunity_by_key.items()
        if high_threshold is not None and value < high_threshold
    ]
    return {
        "all": _stratum_summary(final_rows, episode_keys, comparable_policies),
        "non_near_zero": _stratum_summary(final_rows, non_near_zero, comparable_policies),
        "high_opportunity": _stratum_summary(final_rows, high_opportunity, comparable_policies),
        "low_opportunity": _stratum_summary(final_rows, low_opportunity, comparable_policies),
    }


def _stratum_summary(
    final_rows: dict[tuple[str, str, str], dict[str, Any]],
    episode_keys: list[tuple[str, str]],
    comparable_policies: list[str],
) -> dict[str, Any]:
    if not episode_keys:
        return {
            "episode_count": 0,
            "top_non_oracle_policy": None,
            "top_non_oracle_gain": None,
            "random_valid_gain": None,
            "oracle_gain": None,
            "top_minus_random": None,
            "oracle_minus_random": None,
            "top_oracle_gap_fraction": None,
        }
    policy_means = {}
    for policy in comparable_policies:
        values = [
            _float(final_rows[(*key, policy)].get("cumulative_balanced_relative_gain"))
            for key in episode_keys
            if (*key, policy) in final_rows
        ]
        if values:
            policy_means[policy] = _mean(values)
    top_policy, top_gain = _top_policy(policy_means)
    random_values = [
        _float(final_rows[(*key, RANDOM_POLICY)].get("cumulative_balanced_relative_gain"))
        for key in episode_keys
        if (*key, RANDOM_POLICY) in final_rows
    ]
    oracle_values = [
        _float(final_rows[(*key, ORACLE_POLICY)].get("cumulative_balanced_relative_gain"))
        for key in episode_keys
        if (*key, ORACLE_POLICY) in final_rows
    ]
    random_gain = _mean(random_values)
    oracle_gain = _mean(oracle_values)
    top_minus_random = _subtract_optional(top_gain, random_gain)
    oracle_minus_random = _subtract_optional(oracle_gain, random_gain)
    return {
        "episode_count": len(episode_keys),
        "top_non_oracle_policy": top_policy,
        "top_non_oracle_gain": top_gain,
        "random_valid_gain": random_gain,
        "oracle_gain": oracle_gain,
        "top_minus_random": top_minus_random,
        "oracle_minus_random": oracle_minus_random,
        "top_oracle_gap_fraction": (top_minus_random / oracle_minus_random) if oracle_minus_random and oracle_minus_random > 1.0e-12 else None,
    }


def _random_replay_audit(
    final_rows: dict[tuple[str, str, str], dict[str, Any]],
    comparable_policies: list[str],
    *,
    random_replay_policies: list[str],
) -> dict[str, Any]:
    if not random_replay_policies:
        return {
            "status": "unavailable",
            "replay_policy_count": 0,
            "read": "No random replay policies were present; this artifact can compare against the single random_valid draw but cannot estimate random replay percentiles.",
            "policy_percentiles": {},
        }
    episode_keys = sorted({key[:2] for key in final_rows})
    policy_percentiles = {}
    for policy in comparable_policies:
        percentiles = []
        minus_mean = []
        minus_p90 = []
        minus_p95 = []
        for episode_key in episode_keys:
            policy_row = final_rows.get((*episode_key, policy))
            replay_values = [
                _float(final_rows[(*episode_key, replay_policy)].get("cumulative_balanced_relative_gain"))
                for replay_policy in random_replay_policies
                if (*episode_key, replay_policy) in final_rows
            ]
            if policy_row is None or not replay_values:
                continue
            gain = _float(policy_row.get("cumulative_balanced_relative_gain"))
            percentiles.append(sum(1 for value in replay_values if value <= gain) / len(replay_values))
            minus_mean.append(gain - _mean(replay_values))
            minus_p90.append(gain - _percentile(replay_values, 0.90))
            minus_p95.append(gain - _percentile(replay_values, 0.95))
        if not percentiles:
            continue
        policy_percentiles[policy] = {
            "episode_count": len(percentiles),
            "mean_random_replay_percentile": _mean(percentiles),
            "median_random_replay_percentile": _median(percentiles),
            "mean_gain_minus_random_replay_mean": _mean(minus_mean),
            "mean_gain_minus_random_replay_p90": _mean(minus_p90),
            "mean_gain_minus_random_replay_p95": _mean(minus_p95),
        }
    return {
        "status": "available",
        "replay_policy_count": len(random_replay_policies),
        "read": "Random replay rows were available, so policy gains can be ranked against the replay distribution.",
        "policy_percentiles": policy_percentiles,
    }


def _decision(
    *,
    final_means: dict[str, float | None],
    paired_policy_deltas: dict[str, dict[str, Any]],
    random_replay_audit: dict[str, Any],
    minimum_oracle_gap_fraction: float,
    minimum_random_replay_percentile: float,
) -> dict[str, Any]:
    comparable_means = {
        policy: value
        for policy, value in final_means.items()
        if policy not in {ORACLE_POLICY, RANDOM_POLICY} and not policy.startswith(RANDOM_REPLAY_PREFIX)
    }
    top_policy, top_gain = _top_policy(comparable_means)
    random_gain = final_means.get(RANDOM_POLICY)
    oracle_gain = final_means.get(ORACLE_POLICY)
    top_minus_random = _subtract_optional(top_gain, random_gain)
    oracle_minus_random = _subtract_optional(oracle_gain, random_gain)
    gap_fraction = (top_minus_random / oracle_minus_random) if oracle_minus_random and oracle_minus_random > 1.0e-12 else None
    delta_row = paired_policy_deltas.get(f"{top_policy}_vs_{RANDOM_POLICY}", {}) if top_policy else {}
    ci_low = delta_row.get("bootstrap_ci95_low")
    replay_row = random_replay_audit.get("policy_percentiles", {}).get(str(top_policy), {})
    replay_percentile = replay_row.get("mean_random_replay_percentile")

    if random_replay_audit.get("status") != "available":
        status = "underpowered"
        read = "single random_valid draw is too thin; add random replay rows before trusting policy ranking."
        next_steps = [
            "Add random replay policies or rerun a tiny benchmark with many random_valid_replay_* rows.",
            "Use this report's episode-level deltas as a warning screen, not as a final policy ranking.",
            "Do not launch downstream training from this artifact alone.",
        ]
    elif gap_fraction is None or gap_fraction < minimum_oracle_gap_fraction:
        status = "fail"
        read = "best non-oracle policy captures too little of oracle-minus-random opportunity."
        next_steps = [
            "Redesign episodes to increase oracle opportunity before comparing TS2Vec/window policies.",
            "Increase candidate and target groups while keeping exact oracle feasible.",
            "Hold downstream training.",
        ]
    elif ci_low is None or float(ci_low) <= 0.0:
        status = "warn"
        read = "top policy is directionally above random, but paired uncertainty still crosses zero."
        next_steps = [
            "Run more source-blocked episodes or seeds before promoting a policy.",
            "Keep k-center/window as a baseline, not a proven winner.",
            "Hold downstream training until the random gap is stable.",
        ]
    elif replay_percentile is None or float(replay_percentile) < minimum_random_replay_percentile:
        status = "warn"
        read = "top policy beats the single random baseline but is not high enough in random replay percentiles."
        next_steps = [
            "Use random replay percentiles as the primary discrimination gate.",
            "Do not tune TS2Vec blends against a weak random margin.",
            "Hold downstream training.",
        ]
    else:
        status = "pass"
        read = "top policy beats random at the episode level and sits high in random replay percentiles."
        next_steps = [
            "Carry the top policy into the next benchmark decision report.",
            "Only then consider a tiny downstream utility smoke.",
            "Keep TS2Vec ablations as component evidence, not as a premise.",
        ]
    return {
        "benchmark_discrimination": status,
        "top_non_oracle_policy": top_policy,
        "top_non_oracle_mean_final_gain": top_gain,
        "random_valid_mean_final_gain": random_gain,
        "oracle_mean_final_gain": oracle_gain,
        "top_minus_random": top_minus_random,
        "oracle_minus_random": oracle_minus_random,
        "top_oracle_gap_fraction": gap_fraction,
        "top_random_replay_percentile": replay_percentile,
        "read": read,
        "next_steps": next_steps,
    }


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


def _mean(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def _bootstrap_ci95(values: list[float], *, seed: int) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = random.Random(seed)
    boot_means = []
    for _index in range(1000):
        boot_means.append(float(mean(rng.choice(values) for _item in values)))
    boot_means.sort()
    low_index = int(0.025 * (len(boot_means) - 1))
    high_index = int(0.975 * (len(boot_means) - 1))
    return boot_means[low_index], boot_means[high_index]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(math.ceil(float(percentile) * len(ordered))) - 1
    index = min(max(index, 0), len(ordered) - 1)
    return float(ordered[index])


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
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)
