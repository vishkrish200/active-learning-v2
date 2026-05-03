from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from marginal_value.logging_utils import log_event


COVERAGE_METRICS = ("relative_gain", "oracle_fraction", "absolute_gain")
HYGIENE_METRICS = (
    "artifact_rate_at_k",
    "low_quality_rate_at_k",
    "duplicate_rate_at_k",
    "spike_fail_rate_at_k",
    "trace_artifact_fail_rate_at_k",
    "trace_fail_rate_at_k",
)


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 17,
) -> dict[str, float]:
    numeric = np.asarray([float(value) for value in values], dtype=np.float64)
    if numeric.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    mean = float(np.mean(numeric))
    if numeric.size == 1 or n_bootstrap <= 0:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "n": int(numeric.size)}

    rng = np.random.default_rng(seed)
    samples = rng.choice(numeric, size=(int(n_bootstrap), int(numeric.size)), replace=True)
    sample_means = np.mean(samples, axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "mean": mean,
        "ci_low": float(np.quantile(sample_means, alpha)),
        "ci_high": float(np.quantile(sample_means, 1.0 - alpha)),
        "n": int(numeric.size),
    }


def build_active_loop_validation_report(
    *,
    coverage_by_episode_path: str | Path,
    selection_audit_path: str | Path,
    output_dir: str | Path,
    policies: Sequence[str],
    k_values: Sequence[int],
    representation: str = "balanced",
    primary_policy: str,
    compare_policies: Sequence[str],
    n_bootstrap: int = 2000,
    seed: int = 17,
    mode: str = "full",
    output_stem: str | None = None,
) -> dict[str, Any]:
    coverage_path = Path(coverage_by_episode_path)
    selection_path = Path(selection_audit_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_event(
        "active_loop_validation_report",
        "start",
        coverage_by_episode_path=str(coverage_path),
        selection_audit_path=str(selection_path),
        mode=mode,
    )

    coverage_rows = _read_csv(coverage_path)
    selection_rows = _read_csv(selection_path)
    policy_list = [str(policy) for policy in policies]
    k_list = [int(k) for k in k_values]

    policy_summaries = _coverage_summaries(
        coverage_rows,
        policies=policy_list,
        k_values=k_list,
        representation=representation,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    selection_hygiene = _selection_hygiene_summaries(
        selection_rows,
        policies=policy_list,
        k_values=k_list,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    comparisons = _policy_comparisons(
        coverage_rows,
        primary_policy=primary_policy,
        compare_policies=[str(policy) for policy in compare_policies],
        k_values=k_list,
        representation=representation,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    report = {
        "mode": mode,
        "coverage_by_episode_path": str(coverage_path),
        "selection_audit_path": str(selection_path),
        "representation": representation,
        "policies": policy_list,
        "primary_policy": primary_policy,
        "compare_policies": [str(policy) for policy in compare_policies],
        "k_values": k_list,
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "policy_summaries": policy_summaries,
        "selection_hygiene": selection_hygiene,
        "policy_comparisons": comparisons,
    }
    stem = output_stem or f"active_loop_validation_report_{mode}"
    json_path = out_dir / f"{stem}.json"
    markdown_path = out_dir / f"{stem}.md"
    report["artifacts"] = {"json": str(json_path), "markdown": str(markdown_path)}
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(format_active_loop_validation_markdown(report), encoding="utf-8")
    log_event(
        "active_loop_validation_report",
        "done",
        mode=mode,
        json_path=str(json_path),
        markdown_path=str(markdown_path),
    )
    return report


def format_active_loop_validation_markdown(report: Mapping[str, Any]) -> str:
    policies = [str(policy) for policy in report["policies"]]
    k_values = [int(k) for k in report["k_values"]]
    primary_policy = str(report["primary_policy"])
    lines = [
        "# Active-Loop Validation Report",
        "",
        f"Mode: `{report['mode']}`",
        f"Representation: `{report['representation']}`",
        f"Primary policy: `{primary_policy}`",
        "",
        "## Inputs",
        "",
        f"- Coverage rows: `{report['coverage_by_episode_path']}`",
        f"- Selection audit rows: `{report['selection_audit_path']}`",
        f"- Bootstrap samples: `{report['n_bootstrap']}`",
        "",
        "## Balanced Relative Gain",
        "",
        _coverage_table(report, policies=policies, k_values=k_values, metric="relative_gain"),
        "",
        "## Oracle Fraction",
        "",
        _coverage_table(report, policies=policies, k_values=k_values, metric="oracle_fraction"),
        "",
        "## Hygiene",
        "",
        _hygiene_table(report, policies=policies, k_values=k_values),
        "",
        "## Episode-Level Comparisons",
        "",
        _comparison_table(report, primary_policy=primary_policy, k_values=k_values),
        "",
    ]
    return "\n".join(lines)


def _coverage_summaries(
    rows: Sequence[Mapping[str, str]],
    *,
    policies: Sequence[str],
    k_values: Sequence[int],
    representation: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    policy_set = set(policies)
    k_set = set(k_values)
    for row in rows:
        policy = str(row.get("policy", ""))
        k = _int(row.get("k"))
        if policy not in policy_set or k not in k_set or row.get("representation") != representation:
            continue
        for metric in COVERAGE_METRICS:
            grouped[(policy, k)][metric].append(_float(row.get(metric)))

    summaries: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for policy in policies:
        summaries[policy] = {}
        for k in k_values:
            key = f"coverage@{int(k)}"
            summaries[policy][key] = {}
            for metric in COVERAGE_METRICS:
                summaries[policy][key][metric] = bootstrap_mean_ci(
                    grouped[(policy, int(k))][metric],
                    n_bootstrap=n_bootstrap,
                    seed=seed + int(k) + _stable_name_offset(policy, metric),
                )
    return summaries


def _selection_hygiene_summaries(
    rows: Sequence[Mapping[str, str]],
    *,
    policies: Sequence[str],
    k_values: Sequence[int],
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    policy_set = set(policies)
    k_set = set(k_values)
    for row in rows:
        policy = str(row.get("policy", ""))
        k = _int(row.get("k"))
        if policy not in policy_set or k not in k_set:
            continue
        for metric in HYGIENE_METRICS:
            if metric in row:
                grouped[(policy, k)][metric].append(_float(row.get(metric)))

    summaries: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for policy in policies:
        summaries[policy] = {}
        for k in k_values:
            key = f"coverage@{int(k)}"
            summaries[policy][key] = {}
            for metric in HYGIENE_METRICS:
                summaries[policy][key][metric] = bootstrap_mean_ci(
                    grouped[(policy, int(k))][metric],
                    n_bootstrap=n_bootstrap,
                    seed=seed + int(k) + _stable_name_offset(policy, metric),
                )
    return summaries


def _policy_comparisons(
    rows: Sequence[Mapping[str, str]],
    *,
    primary_policy: str,
    compare_policies: Sequence[str],
    k_values: Sequence[int],
    representation: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    by_key: dict[tuple[str, int, str], float] = {}
    for row in rows:
        if row.get("representation") != representation:
            continue
        by_key[(str(row.get("episode_id")), _int(row.get("k")), str(row.get("policy")))] = _float(
            row.get("relative_gain")
        )

    comparisons: dict[str, dict[str, dict[str, Any]]] = {}
    for compare_policy in compare_policies:
        comparison_key = f"{primary_policy}_vs_{compare_policy}"
        comparisons[comparison_key] = {}
        for k in k_values:
            deltas: list[float] = []
            primary_wins = 0
            compared = 0
            episodes = sorted(
                episode_id
                for episode_id, row_k, policy in by_key
                if row_k == int(k) and policy == primary_policy
            )
            for episode_id in episodes:
                primary_key = (episode_id, int(k), primary_policy)
                compare_key = (episode_id, int(k), compare_policy)
                if primary_key not in by_key or compare_key not in by_key:
                    continue
                delta = by_key[primary_key] - by_key[compare_key]
                deltas.append(delta)
                compared += 1
                if delta >= 0.0:
                    primary_wins += 1
            comparisons[comparison_key][f"coverage@{int(k)}"] = {
                "primary_win_count": int(primary_wins),
                "episode_count": int(compared),
                "primary_win_rate": float(primary_wins / compared) if compared else float("nan"),
                "relative_gain_delta": bootstrap_mean_ci(
                    deltas,
                    n_bootstrap=n_bootstrap,
                    seed=seed + int(k) + _stable_name_offset(primary_policy, compare_policy),
                ),
            }
    return comparisons


def _coverage_table(
    report: Mapping[str, Any],
    *,
    policies: Sequence[str],
    k_values: Sequence[int],
    metric: str,
) -> str:
    header = "| Policy | " + " | ".join(f"K={k}" for k in k_values) + " |"
    sep = "| --- | " + " | ".join("---:" for _ in k_values) + " |"
    rows = [header, sep]
    summaries = report["policy_summaries"]
    for policy in policies:
        cells = [policy]
        for k in k_values:
            stats = summaries[policy][f"coverage@{k}"][metric]
            cells.append(_format_mean_ci(stats))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _hygiene_table(report: Mapping[str, Any], *, policies: Sequence[str], k_values: Sequence[int]) -> str:
    rows = [
        "| Policy | K | Likely Artifact | Spike Fail | Broad Trace Fail | Duplicate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    hygiene = report["selection_hygiene"]
    for policy in policies:
        for k in k_values:
            summary = hygiene[policy][f"coverage@{k}"]
            rows.append(
                "| "
                + " | ".join(
                    [
                        policy,
                        str(k),
                        _format_mean_ci(summary["trace_artifact_fail_rate_at_k"]),
                        _format_mean_ci(summary["spike_fail_rate_at_k"]),
                        _format_mean_ci(summary["trace_fail_rate_at_k"]),
                        _format_mean_ci(summary["duplicate_rate_at_k"]),
                    ]
                )
                + " |"
            )
    return "\n".join(rows)


def _comparison_table(report: Mapping[str, Any], *, primary_policy: str, k_values: Sequence[int]) -> str:
    rows = [
        "| Comparison | K | Primary Wins | Win Rate | Relative Gain Delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for comparison_name, by_k in report["policy_comparisons"].items():
        for k in k_values:
            stats = by_k[f"coverage@{k}"]
            rows.append(
                "| "
                + " | ".join(
                    [
                        comparison_name.replace(f"{primary_policy}_vs_", f"{primary_policy} vs "),
                        str(k),
                        f"{stats['primary_win_count']}/{stats['episode_count']}",
                        _format_number(float(stats["primary_win_rate"])),
                        _format_mean_ci(stats["relative_gain_delta"]),
                    ]
                )
                + " |"
            )
    return "\n".join(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _format_mean_ci(stats: Mapping[str, float]) -> str:
    mean = float(stats["mean"])
    low = float(stats["ci_low"])
    high = float(stats["ci_high"])
    if np.isnan(mean):
        return "n/a"
    return f"{_format_number(mean)} [{_format_number(low)}, {_format_number(high)}]"


def _format_number(value: float) -> str:
    if np.isnan(value):
        return "n/a"
    return f"{value:.4f}"


def _float(value: object) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def _int(value: object) -> int:
    if value is None or value == "":
        return -1
    return int(value)


def _stable_name_offset(*parts: str) -> int:
    return sum(ord(char) for part in parts for char in part) % 100_000


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize active-loop validation outputs.")
    parser.add_argument("--coverage-by-episode", required=True)
    parser.add_argument("--selection-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policies", required=True, nargs="+")
    parser.add_argument("--primary-policy", required=True)
    parser.add_argument("--compare-policies", required=True, nargs="+")
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 25, 50, 100])
    parser.add_argument("--representation", default="balanced")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--mode", default="full")
    parser.add_argument("--output-stem")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    return build_active_loop_validation_report(
        coverage_by_episode_path=args.coverage_by_episode,
        selection_audit_path=args.selection_audit,
        output_dir=args.output_dir,
        policies=args.policies,
        k_values=args.k_values,
        representation=args.representation,
        primary_policy=args.primary_policy,
        compare_policies=args.compare_policies,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        mode=args.mode,
        output_stem=args.output_stem,
    )


if __name__ == "__main__":
    main()
