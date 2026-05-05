from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from marginal_value.active_benchmark.schema import BenchmarkResult, RoundResult


def write_benchmark_reports(result: BenchmarkResult, output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "offline_active_benchmark_report.json"
    markdown_path = output / "offline_active_benchmark_report.md"
    json_path.write_text(json.dumps(result_to_json(result), indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_markdown_report(result), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def result_to_json(result: BenchmarkResult) -> dict[str, Any]:
    return {
        "n_episodes": len(result.episodes),
        "policies": list(result.policies),
        "config": asdict(result.config),
        "episodes": [asdict(episode) for episode in result.episodes],
        "rounds": [_round_to_json(round_result) for round_result in result.rounds],
        "policy_summary": result.policy_summary,
        "difficulty_audit": list(result.difficulty_audit),
    }


def render_markdown_report(result: BenchmarkResult) -> str:
    lines = [
        "# Offline Active-Learning Benchmark",
        "",
        "## Run",
        "",
        f"- episodes: `{len(result.episodes)}`",
        f"- policies: `{', '.join(result.policies)}`",
        f"- rounds: `{int(result.config.rounds)}`",
        f"- batch size: `{int(result.config.batch_size)}`",
        f"- primary representations: `{', '.join(result.config.primary_representations)}`",
        f"- oracle candidate cap: `{result.config.oracle_candidate_cap}`",
        f"- blend left/right/alpha: `{result.config.blend_left_representation}` / `{result.config.blend_right_representation}` / `{float(result.config.blend_alpha):.2f}`",
        f"- artifact demotion threshold: `{result.config.max_artifact_score}`",
        "",
        "## Policy Summary",
        "",
        "| policy | mean balanced relative gain | episode-round wins | rounds |",
        "|---|---:|---:|---:|",
    ]
    for policy, row in sorted(result.policy_summary.items()):
        lines.append(
            "| {policy} | {gain:.4f} | {wins:.0f} | {rounds:.0f} |".format(
                policy=policy,
                gain=float(row.get("mean_balanced_relative_gain", 0.0)),
                wins=float(row.get("episode_round_wins", 0.0)),
                rounds=float(row.get("round_count", 0.0)),
            )
        )
    primary_rep = result.config.primary_representations[0]
    lines.extend(
        [
            "",
            "## Difficulty Audit",
            "",
            "| episode | support | candidate | target | baseline distance | candidate-target distance | oracle max cumulative gain | near-zero oracle rounds | oracle exact rounds |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in result.difficulty_audit:
        baseline_by_rep = row.get("support_target_baseline_distance_by_representation", {})
        candidate_by_rep = row.get("candidate_target_nearest_distance_by_representation", {})
        exact_by_round = row.get("oracle_fraction_exact_by_round", [])
        exact_text = ",".join("Y" if bool(value) else "N" for value in exact_by_round) if isinstance(exact_by_round, list) else ""
        lines.append(
            "| {episode} | {support} | {candidate} | {target} | {baseline:.4f} | {candidate_target:.4f} | {oracle:.4f} | {near_zero:.3f} | {exact} |".format(
                episode=row.get("episode_id", ""),
                support=int(row.get("support_count", 0)),
                candidate=int(row.get("candidate_count", 0)),
                target=int(row.get("target_count", 0)),
                baseline=float(baseline_by_rep.get(primary_rep, 0.0)) if isinstance(baseline_by_rep, dict) else 0.0,
                candidate_target=float(candidate_by_rep.get(primary_rep, 0.0)) if isinstance(candidate_by_rep, dict) else 0.0,
                oracle=float(row.get("max_oracle_greedy_cumulative_gain", 0.0)),
                near_zero=float(row.get("near_zero_oracle_round_fraction", 1.0)),
                exact=exact_text,
            )
        )
    lines.extend(
        [
            "",
        "## Acquisition Curves",
        "",
            "| episode | round | policy | selected | round gain | cumulative gain | oracle fraction | min quality | largest source frac |",
            "|---|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for round_result in sorted(result.rounds, key=lambda row: (row.episode_id, row.round_index, row.policy_name)):
        lines.append(
            "| {episode} | {round_index} | {policy} | {selected} | {gain:.4f} | {cumulative:.4f} | {oracle:.4f} | {minq:.3f} | {largest:.3f} |".format(
                episode=round_result.episode_id,
                round_index=round_result.round_index,
                policy=round_result.policy_name,
                selected=", ".join(round_result.selected_ids),
                gain=float(round_result.balanced_relative_gain),
                cumulative=float(round_result.cumulative_balanced_relative_gain),
                oracle=float(round_result.oracle_fraction),
                minq=float(round_result.selection_summary.get("min_quality", 0.0)),
                largest=float(round_result.selection_summary.get("largest_source_group_fraction", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Selection Audit",
            "",
            "| episode | round | policy | rank | sample | source | quality | artifact | selected score | old novelty | blend score | artifact pass |",
            "|---|---:|---|---:|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for round_result in sorted(result.rounds, key=lambda row: (row.episode_id, row.round_index, row.policy_name)):
        for detail in round_result.selection_details:
            lines.append(
                "| {episode} | {round_index} | {policy} | {rank} | {sample} | {source} | {quality:.3f} | {artifact:.3f} | {selected_score} | {old_score} | {blend_score} | {artifact_pass} |".format(
                    episode=round_result.episode_id,
                    round_index=round_result.round_index,
                    policy=round_result.policy_name,
                    rank=int(detail.get("rank_index", 0)),
                    sample=detail.get("sample_id", ""),
                    source=detail.get("source_group_id", ""),
                    quality=float(detail.get("quality_score", 0.0)),
                    artifact=float(detail.get("artifact_score", 0.0)),
                    selected_score=_format_optional_float(detail.get("selected_score")),
                    old_score=_format_optional_float(detail.get("old_novelty_window_score")),
                    blend_score=_format_optional_float(detail.get("blend_score")),
                    artifact_pass="Y" if bool(detail.get("passed_artifact_gate", False)) else "N",
                )
            )
    lines.extend(
        [
            "",
            "## Decision Notes",
            "",
            "- Treat policies as useful only when they beat strong simple controls across episode rounds.",
            "- The oracle policy uses hidden targets and is a diagnostic, not a deployable acquisition policy.",
            "- When `oracle_candidate_cap` is set, the oracle is an approximate diagnostic rather than a strict upper bound.",
            "- GCP or GPU work should wait until local smoke results are mechanically clean and scientifically interpretable.",
            "",
        ]
    )
    return "\n".join(lines)


def _round_to_json(round_result: RoundResult) -> dict[str, Any]:
    return {
        "episode_id": round_result.episode_id,
        "fold_id": round_result.fold_id,
        "policy_name": round_result.policy_name,
        "round_index": round_result.round_index,
        "batch_size": round_result.batch_size,
        "support_ids_before": list(round_result.support_ids_before),
        "candidate_ids_before": list(round_result.candidate_ids_before),
        "selected_ids": list(round_result.selected_ids),
        "selected_scores": list(round_result.selected_scores),
        "selected_source_group_ids": list(round_result.selected_source_group_ids),
        "support_ids_after": list(round_result.support_ids_after),
        "candidate_ids_after": list(round_result.candidate_ids_after),
        "coverage_by_representation": round_result.coverage_by_representation,
        "balanced_relative_gain": round_result.balanced_relative_gain,
        "cumulative_balanced_relative_gain": round_result.cumulative_balanced_relative_gain,
        "oracle_fraction": round_result.oracle_fraction,
        "selection_summary": round_result.selection_summary,
        "selection_details": round_result.selection_details,
    }


def _format_optional_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""
