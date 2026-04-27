from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


DEFAULT_REPRESENTATIONS = ("window_mean_std_pool", "temporal_order", "raw_shape_stats")
DEFAULT_PRIMARY_REPRESENTATIONS = ("temporal_order", "raw_shape_stats")


def load_marginal_coverage_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def coverage_value(policy_report: dict[str, Any], *, k: int, representation: str) -> float:
    coverage = policy_report.get(f"coverage@{int(k)}", {})
    if not isinstance(coverage, dict):
        return 0.0
    metrics = coverage.get(representation, {})
    if not isinstance(metrics, dict):
        return 0.0
    if "relative_coverage_gain" in metrics:
        return _safe_float(metrics.get("relative_coverage_gain", 0.0))
    return _safe_float(metrics.get("relative_coverage_gain_mean", 0.0))


def fold_primary_score(
    fold: dict[str, Any],
    policy: str,
    *,
    k: int,
    representations: Sequence[str] = DEFAULT_PRIMARY_REPRESENTATIONS,
) -> float:
    policies = fold.get("policies", {})
    if not isinstance(policies, dict) or policy not in policies:
        return 0.0
    policy_report = policies[policy]
    if not isinstance(policy_report, dict):
        return 0.0
    values = [coverage_value(policy_report, k=k, representation=rep) for rep in representations]
    return _mean(values)


def mean_primary_score(
    report: dict[str, Any],
    policy: str,
    *,
    k: int,
    representations: Sequence[str] = DEFAULT_PRIMARY_REPRESENTATIONS,
) -> float:
    return _mean([fold_primary_score(fold, policy, k=k, representations=representations) for fold in report.get("folds", [])])


def policy_metric_rows(
    report: dict[str, Any],
    *,
    policies: Sequence[str],
    k_values: Sequence[int],
    representations: Sequence[str] = DEFAULT_REPRESENTATIONS,
    primary_representations: Sequence[str] = DEFAULT_PRIMARY_REPRESENTATIONS,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    folds = [fold for fold in report.get("folds", []) if isinstance(fold, dict)]
    for k in k_values:
        for policy in policies:
            if not _policy_present(report, policy):
                continue
            row: dict[str, object] = {"policy": policy, "k": int(k)}
            for representation in representations:
                row[representation] = _mean(
                    [
                        coverage_value(fold["policies"][policy], k=int(k), representation=representation)
                        for fold in folds
                        if policy in fold.get("policies", {})
                    ]
                )
            row["primary_average"] = mean_primary_score(report, policy, k=int(k), representations=primary_representations)
            row.update(_mean_selection_summary(report, policy=policy, k=int(k)))
            rows.append(row)
    return rows


def paired_delta_rows(
    report: dict[str, Any],
    *,
    challengers: Sequence[str],
    baselines: Sequence[str],
    k_values: Sequence[int],
    primary_representations: Sequence[str] = DEFAULT_PRIMARY_REPRESENTATIONS,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    folds = [fold for fold in report.get("folds", []) if isinstance(fold, dict)]
    for k in k_values:
        for challenger in challengers:
            if not _policy_present(report, challenger):
                continue
            for baseline in baselines:
                if not _policy_present(report, baseline):
                    continue
                deltas = [
                    fold_primary_score(fold, challenger, k=int(k), representations=primary_representations)
                    - fold_primary_score(fold, baseline, k=int(k), representations=primary_representations)
                    for fold in folds
                ]
                rows.append(
                    {
                        "challenger": challenger,
                        "baseline": baseline,
                        "k": int(k),
                        "mean_delta": _mean(deltas),
                        "fold_wins": int(sum(delta > 0.0 for delta in deltas)),
                        "fold_ties": int(sum(abs(delta) <= 1.0e-12 for delta in deltas)),
                        "folds": int(len(deltas)),
                    }
                )
    return rows


def render_markdown_report(
    report: dict[str, Any],
    *,
    title: str,
    policies: Sequence[str],
    challengers: Sequence[str],
    baselines: Sequence[str],
    k_values: Sequence[int],
    primary_representations: Sequence[str] = DEFAULT_PRIMARY_REPRESENTATIONS,
) -> str:
    metric_rows = policy_metric_rows(
        report,
        policies=policies,
        k_values=k_values,
        primary_representations=primary_representations,
    )
    delta_rows = paired_delta_rows(
        report,
        challengers=challengers,
        baselines=baselines,
        k_values=k_values,
        primary_representations=primary_representations,
    )
    lines = [
        f"# {title}",
        "",
        "## Run",
        "",
        f"- mode: `{report.get('mode', '')}`",
        f"- rows: `{int(report.get('n_rows', 0))}`",
        f"- source groups: `{int(report.get('n_source_groups', 0))}`",
        f"- folds: `{len(report.get('folds', []))}`",
        f"- primary representations: `{', '.join(primary_representations)}`",
        "",
        "## Mean Coverage",
        "",
        "| policy | K | window | temporal | raw | primary avg | min quality | max stationary | max abs | stationary >0.90 | max abs >60 | largest source frac |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            "| {policy} | {k} | {window:.4f} | {temporal:.4f} | {raw:.4f} | {primary:.4f} | {minq:.3f} | {maxstat:.3f} | {maxabs:.1f} | {stat90:.3f} | {abs60:.3f} | {largest:.3f} |".format(
                policy=row["policy"],
                k=row["k"],
                window=float(row.get("window_mean_std_pool", 0.0)),
                temporal=float(row.get("temporal_order", 0.0)),
                raw=float(row.get("raw_shape_stats", 0.0)),
                primary=float(row.get("primary_average", 0.0)),
                minq=float(row.get("min_quality", 0.0)),
                maxstat=float(row.get("max_stationary_fraction", 0.0)),
                maxabs=float(row.get("max_abs_value", 0.0)),
                stat90=float(row.get("stationary_fraction_over_90", 0.0)),
                abs60=float(row.get("max_abs_value_over_60", 0.0)),
                largest=float(row.get("largest_source_group_fraction", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Paired Primary Deltas",
            "",
            "| challenger | baseline | K | mean delta | fold wins | fold ties | folds |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in delta_rows:
        lines.append(
            "| {challenger} | {baseline} | {k} | {delta:.4f} | {wins} | {ties} | {folds} |".format(
                challenger=row["challenger"],
                baseline=row["baseline"],
                k=row["k"],
                delta=float(row["mean_delta"]),
                wins=row["fold_wins"],
                ties=row["fold_ties"],
                folds=row["folds"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision Notes",
            "",
            "- Treat a method as stronger only when it wins the primary aggregate and wins in most folds.",
            "- Use `window_mean_std_pool` as a sanity check, not as the sole decision metric, because it is also the current ranking space.",
            "- If a method improves minimum quality but not coverage, it is a safety control rather than a marginal-value improvement.",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown_report(
    report: dict[str, Any],
    output_path: str | Path,
    *,
    title: str,
    policies: Sequence[str],
    challengers: Sequence[str],
    baselines: Sequence[str],
    k_values: Sequence[int],
    primary_representations: Sequence[str] = DEFAULT_PRIMARY_REPRESENTATIONS,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_markdown_report(
            report,
            title=title,
            policies=policies,
            challengers=challengers,
            baselines=baselines,
            k_values=k_values,
            primary_representations=primary_representations,
        ),
        encoding="utf-8",
    )
    return output


def _mean_selection_summary(report: dict[str, Any], *, policy: str, k: int) -> dict[str, float]:
    keys = (
        "min_quality",
        "mean_quality",
        "mean_stationary_fraction",
        "max_stationary_fraction",
        "stationary_fraction_over_90",
        "mean_max_abs_value",
        "max_abs_value",
        "max_abs_value_over_60",
        "largest_source_group_fraction",
        "unique_source_groups",
    )
    values = {key: [] for key in keys}
    for fold in report.get("folds", []):
        policies = fold.get("policies", {}) if isinstance(fold, dict) else {}
        policy_report = policies.get(policy, {}) if isinstance(policies, dict) else {}
        selection = policy_report.get(f"selection@{int(k)}", {}) if isinstance(policy_report, dict) else {}
        if not isinstance(selection, dict):
            continue
        for key in keys:
            random_key = f"{key}_mean"
            values[key].append(_safe_float(selection.get(key, selection.get(random_key, 0.0))))
    return {key: _mean(items) for key, items in values.items()}


def _policy_present(report: dict[str, Any], policy: str) -> bool:
    for fold in report.get("folds", []):
        policies = fold.get("policies", {}) if isinstance(fold, dict) else {}
        if isinstance(policies, dict) and policy in policies:
            return True
    return False


def _mean(values: Sequence[float]) -> float:
    clean = [_safe_float(value) for value in values]
    return float(sum(clean) / len(clean)) if clean else 0.0


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a marginal-coverage report.")
    parser.add_argument("report_path")
    parser.add_argument("output_path")
    parser.add_argument("--title", default="Marginal Coverage Scientific Diagnostic")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=[
            "random_high_quality",
            "quality_only",
            "old_novelty_only",
            "old_novelty_raw_shape_stats",
            "quality_gated_old_novelty_raw_shape_stats_q85",
            "quality_gated_old_novelty_raw_shape_stats_q85_stat90",
            "quality_gated_old_novelty_raw_shape_stats_q85_stat90_abs60",
        ],
    )
    parser.add_argument(
        "--challengers",
        nargs="+",
        default=[
            "quality_gated_old_novelty_raw_shape_stats_q85",
            "quality_gated_old_novelty_raw_shape_stats_q85_stat90",
            "quality_gated_old_novelty_raw_shape_stats_q85_stat90_abs60",
        ],
    )
    parser.add_argument("--baselines", nargs="+", default=["random_high_quality", "quality_only", "old_novelty_only"])
    parser.add_argument("--k-values", nargs="+", type=int, default=[50, 100, 200, 400])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = load_marginal_coverage_report(args.report_path)
    write_markdown_report(
        report,
        args.output_path,
        title=args.title,
        policies=args.policies,
        challengers=args.challengers,
        baselines=args.baselines,
        k_values=args.k_values,
    )


if __name__ == "__main__":
    main()
