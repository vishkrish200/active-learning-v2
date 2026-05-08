from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from marginal_value.active_benchmark.coverage_runner import CoverageRunResult


def write_coverage_reports(result: CoverageRunResult, output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "blind_target_coverage_benchmark_report.json"
    markdown_path = output / "blind_target_coverage_benchmark_report.md"
    json_path.write_text(json.dumps(coverage_result_to_json(result), indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_coverage_markdown_report(result), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def coverage_result_to_json(result: CoverageRunResult) -> dict[str, Any]:
    return {
        "n_episodes": len(result.episodes),
        "policies": list(result.policies),
        "budgets": list(result.budgets),
        "config": asdict(result.config),
        "episodes": [asdict(episode) for episode in result.episodes],
        "rounds": [asdict(row) for row in result.rounds],
        "selected_rows": [asdict(row) for row in result.selected_rows],
        "metric_rows": [asdict(row) for row in result.metric_rows],
        "policy_summary": result.policy_summary,
    }


def render_coverage_markdown_report(result: CoverageRunResult) -> str:
    lines = [
        "# Blind Target Coverage Benchmark",
        "",
        "## Run",
        "",
        f"- episodes: `{len(result.episodes)}`",
        f"- policies: `{', '.join(result.policies)}`",
        f"- budgets: `{', '.join(str(budget) for budget in result.budgets)}`",
        f"- eval views: `{', '.join(result.config.eval_views)}`",
        f"- primary eval views: `{', '.join(result.config.primary_eval_views)}`",
        f"- distance metric: `{result.config.distance_metric}`",
        "",
        "## Policy Summary",
        "",
        "| policy | mean primary coverage_gain_rel | primary rows | diagnostic rows | underfilled rounds |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy_id, row in sorted(result.policy_summary.items()):
        lines.append(
            "| {policy} | {gain:.4f} | {primary:.0f} | {diagnostic:.0f} | {underfilled:.0f} |".format(
                policy=policy_id,
                gain=float(row.get("mean_primary_coverage_gain_rel", 0.0)),
                primary=float(row.get("primary_row_count", 0.0)),
                diagnostic=float(row.get("diagnostic_row_count", 0.0)),
                underfilled=float(row.get("underfilled_round_count", 0.0)),
            )
        )

    lines.extend(
        [
            "",
            "## Coverage Rows",
            "",
            "| episode | budget | policy | eval view | metric | value | primary | feature overlap | target-used |",
            "|---|---:|---|---|---|---:|---|---|---|",
        ]
    )
    coverage_rows = [
        row
        for row in result.metric_rows
        if row.metric_name in {"coverage_gain_rel", "coverage_gain_abs", "tau_coverage_gain"}
    ]
    for row in sorted(coverage_rows, key=lambda item: (item.episode_id, item.budget_k, item.policy_id, item.eval_view, item.metric_name)):
        lines.append(
            "| {episode} | {budget} | {policy} | {view} | {metric} | {value:.4f} | {primary} | {overlap} | {target_used} |".format(
                episode=row.episode_id,
                budget=int(row.budget_k),
                policy=row.policy_id,
                view=row.eval_view,
                metric=row.metric_name,
                value=float(row.metric_value),
                primary="Y" if row.primary_eval else "N",
                overlap="Y" if row.selector_feature_overlap else "N",
                target_used="Y" if row.uses_target_for_selection else "N",
            )
        )

    lines.extend(
        [
            "",
            "## Hygiene Rows",
            "",
            "| episode | budget | policy | metric | value |",
            "|---|---:|---|---|---:|",
        ]
    )
    hygiene_rows = [row for row in result.metric_rows if row.eval_view == "__hygiene__"]
    for row in sorted(hygiene_rows, key=lambda item: (item.episode_id, item.budget_k, item.policy_id, item.metric_name)):
        lines.append(
            "| {episode} | {budget} | {policy} | {metric} | {value:.4f} |".format(
                episode=row.episode_id,
                budget=int(row.budget_k),
                policy=row.policy_id,
                metric=row.metric_name,
                value=float(row.metric_value),
            )
        )

    lines.extend(
        [
            "",
            "## Selected Clips",
            "",
            "| episode | budget | policy | rank | sample | source | score | quality | artifact | valid |",
            "|---|---:|---|---:|---|---|---:|---:|---:|---|",
        ]
    )
    for row in sorted(result.selected_rows, key=lambda item: (item.episode_id, item.budget_k, item.policy_id, item.rank_index)):
        lines.append(
            "| {episode} | {budget} | {policy} | {rank} | {sample} | {source} | {score:.4f} | {quality:.3f} | {artifact:.3f} | {valid} |".format(
                episode=row.episode_id,
                budget=int(row.budget_k),
                policy=row.policy_id,
                rank=int(row.rank_index),
                sample=row.sample_id,
                source=row.source_group_id,
                score=float(row.score),
                quality=float(row.quality_score),
                artifact=float(row.artifact_score),
                valid="Y" if row.valid else "N",
            )
        )

    lines.extend(
        [
            "",
            "## Decision Notes",
            "",
            "- Primary rows exclude same-feature-family selector/eval comparisons.",
            "- Oracle rows, when present, are diagnostics because they inspect the target set.",
            "- Deployable v1 policies only select candidates that pass quality and artifact gates.",
            "",
        ]
    )
    return "\n".join(lines)
