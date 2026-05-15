from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


BLEND_POLICY = "blend_kcenter_ts2vec_window"
ARTIFACT_GATED_BLEND_POLICY = "artifact_gate_blend_kcenter_ts2vec_window"
OLD_NOVELTY_POLICY = "old_novelty_window"
SOURCECAP_OLD_POLICY = "old_novelty_window_sourcecap2"
KCENTER_WINDOW_POLICY = "kcenter_quality_gated_window"
SUBMITTED_FULL_REPLAY_POLICY = "submitted_full_replay"
SUBMITTED_MINUS_TS2VEC_POLICY = "submitted_minus_ts2vec"
SUBMITTED_MINUS_WINDOW_POLICY = "submitted_minus_window"
SUBMITTED_NO_KCENTER_POLICY = "submitted_no_kcenter"
WINDOW_NOVELTY_SAME_GATES_POLICY = "window_novelty_same_gates_no_kcenter"
TS2VEC_NOVELTY_SAME_GATES_POLICY = "ts2vec_novelty_same_gates_no_kcenter"


def build_policy_autopsy(run_root: str | Path, *, aggregate_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(run_root)
    aggregate = _load_optional_aggregate(root, aggregate_path)
    report_paths = _find_report_paths(root)
    if not report_paths:
        raise FileNotFoundError(f"No offline_active_benchmark_report.json files found under {root}")

    reports = [_load_seed_report(path) for path in report_paths]
    rows = [row for report in reports for row in report["rounds"]]
    if not rows:
        raise ValueError(f"No round rows found under {root}")

    artifact_threshold = _artifact_threshold(reports)
    policies = sorted({str(row.get("policy_name", "")) for row in rows if row.get("policy_name")})
    ordered_policies = _ordered_policies(reports, rows)
    pairwise_overlap = _pairwise_overlap(rows, policies)
    policy_outcomes = _policy_outcomes(rows, policies)
    policy_hygiene = _policy_hygiene(rows, policies, artifact_threshold)
    paired_policy_deltas = _paired_policy_deltas(rows, ordered_policies)
    artifact_gate_effect = _artifact_gate_effect(rows, policy_hygiene)
    old_vs_blend = _old_vs_blend(policy_outcomes, pairwise_overlap)
    source_kcenter_read = _source_kcenter_read(
        policy_hygiene=policy_hygiene,
        paired_policy_deltas=paired_policy_deltas,
        artifact_gate_effect=artifact_gate_effect,
    )
    coverage_story = _coverage_story(reports, aggregate)
    diagnosis = _diagnosis(
        old_vs_blend=old_vs_blend,
        artifact_gate_effect=artifact_gate_effect,
        coverage_story=coverage_story,
    )

    return {
        "input": {
            "run_root": str(root),
            "aggregate_path": str(Path(aggregate_path)) if aggregate_path is not None else str(_default_aggregate_path(root)),
            "report_count": len(report_paths),
            "report_paths": [str(path) for path in report_paths],
            "seed_names": [str(report["seed_name"]) for report in reports],
        },
        "summary": {
            "policy_count": len(policies),
            "round_row_count": len(rows),
            "selection_detail_count": sum(len(_selection_details(row)) for row in rows),
            "artifact_threshold": artifact_threshold,
        },
        "policies": policies,
        "policy_outcomes": policy_outcomes,
        "old_vs_blend": old_vs_blend,
        "source_kcenter_read": source_kcenter_read,
        "pairwise_overlap": pairwise_overlap,
        "paired_policy_deltas": paired_policy_deltas,
        "artifact_gate_effect": artifact_gate_effect,
        "policy_hygiene": policy_hygiene,
        "coverage_story": coverage_story,
        "diagnosis": diagnosis,
    }


def write_policy_autopsy_reports(
    run_root: str | Path,
    *,
    output_json: str | Path,
    output_markdown: str | Path,
    aggregate_path: str | Path | None = None,
) -> dict[str, str]:
    autopsy = build_policy_autopsy(run_root, aggregate_path=aggregate_path)
    json_path = Path(output_json)
    markdown_path = Path(output_markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(autopsy, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_policy_autopsy_markdown(autopsy), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def render_policy_autopsy_markdown(autopsy: dict[str, Any]) -> str:
    old_vs_blend = autopsy.get("old_vs_blend", {})
    coverage = autopsy.get("coverage_story", {})
    gate = autopsy.get("artifact_gate_effect", {})
    lines = [
        "# Policy Autopsy",
        "",
        "## Run",
        "",
        f"- run root: `{autopsy.get('input', {}).get('run_root', '')}`",
        f"- seed reports: `{autopsy.get('input', {}).get('report_count', 0)}`",
        f"- round rows: `{autopsy.get('summary', {}).get('round_row_count', 0)}`",
        f"- selection details: `{autopsy.get('summary', {}).get('selection_detail_count', 0)}`",
        f"- oracle exact all rounds: `{coverage.get('oracle_fraction_exact_all_rounds')}`",
        f"- mean near-zero oracle round fraction: `{_fmt(coverage.get('mean_near_zero_oracle_round_fraction'))}`",
        "",
        "## Diagnosis",
        "",
    ]
    for item in autopsy.get("diagnosis", []):
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Why Old Novelty Vs Blend",
            "",
            "| comparison | value |",
            "|---|---:|",
            f"| old final cumulative gain | {_fmt(old_vs_blend.get('old_mean_final_cumulative_gain'))} |",
            f"| blend final cumulative gain | {_fmt(old_vs_blend.get('blend_mean_final_cumulative_gain'))} |",
            f"| old minus blend | {_fmt(old_vs_blend.get('old_minus_blend_final_gain'))} |",
            f"| mean selected-batch Jaccard | {_fmt(old_vs_blend.get('mean_jaccard'))} |",
            f"| exact batch match fraction | {_fmt(old_vs_blend.get('exact_batch_match_fraction'))} |",
            f"| read | {old_vs_blend.get('read', '')} |",
            "",
            "## Source And K-Center Read",
            "",
        ]
    )
    source_read = autopsy.get("source_kcenter_read", {})
    for item in source_read.get("summary", []):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "| comparison | policy A | policy B | episodes | mean final gain delta | CI95 low | CI95 high | A win frac | read |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in source_read.get("comparisons", []):
        lines.append(
            "| {comparison} | {a} | {b} | {episodes} | {mean_delta} | {ci_low} | {ci_high} | {win_frac} | {read} |".format(
                comparison=row.get("comparison", ""),
                a=row.get("policy_a", ""),
                b=row.get("policy_b", ""),
                episodes=int(row.get("paired_episode_count", 0)),
                mean_delta=_fmt(row.get("mean_delta_final_cumulative_gain")),
                ci_low=_fmt(row.get("bootstrap_ci95_low")),
                ci_high=_fmt(row.get("bootstrap_ci95_high")),
                win_frac=_fmt(row.get("policy_a_win_fraction")),
                read=row.get("read", ""),
            )
        )
    lines.extend(
        [
            "",
            "| policy | largest source frac | duplicate source batch rate | unique selected frac |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in source_read.get("source_concentration", []):
        lines.append(
            "| {policy} | {largest} | {duplicate} | {unique} |".format(
                policy=row.get("policy", ""),
                largest=_fmt(row.get("mean_largest_source_group_fraction")),
                duplicate=_fmt(row.get("duplicate_source_batch_rate")),
                unique=_fmt(row.get("unique_selected_fraction")),
            )
        )

    lines.extend(
        [
            "",
            "## Policy Outcomes",
            "",
            "| policy | mean final cumulative gain | final wins | mean round gain | mean oracle fraction |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in autopsy.get("policy_outcomes", []):
        lines.append(
            "| {policy} | {final_gain} | {wins} | {round_gain} | {oracle} |".format(
                policy=row.get("policy", ""),
                final_gain=_fmt(row.get("mean_final_cumulative_gain")),
                wins=int(row.get("final_episode_wins", 0)),
                round_gain=_fmt(row.get("mean_round_gain")),
                oracle=_fmt(row.get("mean_oracle_fraction")),
            )
        )

    lines.extend(
        [
            "",
            "## Paired Deltas",
            "",
            "| policy A | policy B | episodes | mean final gain delta | median final gain delta | CI95 low | CI95 high | A win frac |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in autopsy.get("paired_policy_deltas", []):
        lines.append(
            "| {a} | {b} | {episodes} | {mean_delta} | {median_delta} | {ci_low} | {ci_high} | {win_frac} |".format(
                a=row.get("policy_a", ""),
                b=row.get("policy_b", ""),
                episodes=int(row.get("paired_episode_count", 0)),
                mean_delta=_fmt(row.get("mean_delta_final_cumulative_gain")),
                median_delta=_fmt(row.get("median_delta_final_cumulative_gain")),
                ci_low=_fmt(row.get("bootstrap_ci95_low")),
                ci_high=_fmt(row.get("bootstrap_ci95_high")),
                win_frac=_fmt(row.get("policy_a_win_fraction")),
            )
        )

    lines.extend(
        [
            "",
            "## Selection Overlap",
            "",
            "| policy A | policy B | rounds | mean Jaccard | exact batch match | mean intersection |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in autopsy.get("pairwise_overlap", []):
        lines.append(
            "| {a} | {b} | {rounds} | {jaccard} | {exact} | {intersection} |".format(
                a=row.get("policy_a", ""),
                b=row.get("policy_b", ""),
                rounds=int(row.get("comparable_round_count", 0)),
                jaccard=_fmt(row.get("mean_jaccard")),
                exact=_fmt(row.get("exact_batch_match_fraction")),
                intersection=_fmt(row.get("mean_intersection_size")),
            )
        )

    lines.extend(
        [
            "",
            "## Artifact Gate",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| base policy | {gate.get('base_policy', '')} |",
            f"| gated policy | {gate.get('gated_policy', '')} |",
            f"| compared rounds | {int(gate.get('comparable_round_count', 0))} |",
            f"| changed rounds | {int(gate.get('changed_round_count', 0))} |",
            f"| exact batch match fraction | {_fmt(gate.get('exact_batch_match_fraction'))} |",
            f"| base artifact selected rate | {_fmt(gate.get('base_artifact_selected_rate'))} |",
            f"| gated artifact selected rate | {_fmt(gate.get('gated_artifact_selected_rate'))} |",
            f"| no-op | {gate.get('is_no_op')} |",
            "",
            "## Hygiene",
            "",
            "| policy | selected | mean quality | artifact rate | artifact gate fail rate | largest source frac | unique selected frac |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in autopsy.get("policy_hygiene", []):
        lines.append(
            "| {policy} | {selected} | {quality} | {artifact} | {gate_fail} | {largest} | {unique} |".format(
                policy=row.get("policy", ""),
                selected=int(row.get("selected_count", 0)),
                quality=_fmt(row.get("mean_quality")),
                artifact=_fmt(row.get("artifact_selected_rate")),
                gate_fail=_fmt(row.get("artifact_gate_fail_rate")),
                largest=_fmt(row.get("mean_largest_source_group_fraction")),
                unique=_fmt(row.get("unique_selected_fraction")),
            )
        )

    lines.extend(
        [
            "",
            "## Coverage Story",
            "",
            f"- exact oracle denominators: `{coverage.get('oracle_fraction_exact_all_rounds')}`",
            f"- exact oracle round fraction: `{_fmt(coverage.get('oracle_fraction_exact_round_fraction'))}`",
            f"- near-zero oracle round fraction: `{_fmt(coverage.get('mean_near_zero_oracle_round_fraction'))}`",
            f"- difficulty episodes: `{coverage.get('difficulty_episode_count', 0)}`",
            "",
        ]
    )
    return "\n".join(lines)


def _find_report_paths(root: Path) -> list[Path]:
    direct = sorted(root.glob("seed_*/offline_active_benchmark_report.json"))
    if direct:
        return direct
    return sorted(path for path in root.glob("**/offline_active_benchmark_report.json") if path.is_file())


def _load_seed_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    seed_name = path.parent.name
    for row in report.get("rounds", []):
        row["_seed_name"] = seed_name
    report["seed_name"] = seed_name
    return report


def _load_optional_aggregate(root: Path, aggregate_path: str | Path | None) -> dict[str, Any]:
    path = Path(aggregate_path) if aggregate_path is not None else _default_aggregate_path(root)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _default_aggregate_path(root: Path) -> Path:
    return root.parent / "aggregate_proof_summary.json"


def _artifact_threshold(reports: list[dict[str, Any]]) -> float:
    for report in reports:
        value = report.get("config", {}).get("max_artifact_score")
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return 0.05


def _policy_outcomes(rows: list[dict[str, Any]], policies: list[str]) -> list[dict[str, Any]]:
    rows_by_policy = _rows_by_policy(rows)
    round_win_counts = _win_counts(rows, value_key="cumulative_balanced_relative_gain")
    final_rows_by_policy = _final_rows_by_policy(rows)
    final_win_counts = _final_win_counts(final_rows_by_policy)
    output = []
    for policy in policies:
        policy_rows = rows_by_policy.get(policy, [])
        final_rows = final_rows_by_policy.get(policy, [])
        final_values = [_float(row.get("cumulative_balanced_relative_gain")) for row in final_rows]
        round_values = [_float(row.get("balanced_relative_gain")) for row in policy_rows]
        output.append(
            {
                "policy": policy,
                "round_count": len(policy_rows),
                "mean_round_gain": _mean(round_values),
                "mean_cumulative_gain": _mean([_float(row.get("cumulative_balanced_relative_gain")) for row in policy_rows]),
                "mean_final_cumulative_gain": _mean(final_values),
                "median_final_cumulative_gain": _median(final_values),
                "min_final_cumulative_gain": min(final_values) if final_values else None,
                "max_final_cumulative_gain": max(final_values) if final_values else None,
                "std_final_cumulative_gain": _std(final_values),
                "episode_round_wins": round_win_counts.get(policy, 0),
                "final_episode_wins": final_win_counts.get(policy, 0),
                "mean_oracle_fraction": _mean([_float(row.get("oracle_fraction")) for row in policy_rows]),
                "mean_final_oracle_fraction": _mean([_float(row.get("oracle_fraction")) for row in final_rows]),
                "mean_round_relative_coverage_by_representation": _coverage_means(policy_rows, "relative_coverage_gain"),
                "mean_round_absolute_coverage_by_representation": _coverage_means(policy_rows, "coverage_gain"),
            }
        )
    output.sort(key=lambda row: (_none_low(row.get("mean_final_cumulative_gain")), _none_low(row.get("mean_round_gain"))), reverse=True)
    return output


def _policy_hygiene(rows: list[dict[str, Any]], policies: list[str], artifact_threshold: float) -> list[dict[str, Any]]:
    output = []
    rows_by_policy = _rows_by_policy(rows)
    for policy in policies:
        policy_rows = rows_by_policy.get(policy, [])
        details = [detail for row in policy_rows for detail in _selection_details(row)]
        qualities = [_float(detail.get("quality_score")) for detail in details if _is_number(detail.get("quality_score"))]
        artifacts = [_float(detail.get("artifact_score")) for detail in details if _is_number(detail.get("artifact_score"))]
        selected_ids = [str(detail.get("sample_id")) for detail in details if detail.get("sample_id")]
        source_ids = [str(detail.get("source_group_id")) for detail in details if detail.get("source_group_id")]
        largest_source = [
            _float(row.get("selection_summary", {}).get("largest_source_group_fraction"))
            for row in policy_rows
            if _is_number(row.get("selection_summary", {}).get("largest_source_group_fraction"))
        ]
        duplicate_source_batches = 0
        comparable_batches = 0
        for row in policy_rows:
            selected_count = len(row.get("selected_ids", []))
            largest = row.get("selection_summary", {}).get("largest_source_group_fraction")
            if selected_count > 0 and _is_number(largest):
                comparable_batches += 1
                if _float(largest) > (1.0 / float(selected_count)) + 1.0e-12:
                    duplicate_source_batches += 1
        output.append(
            {
                "policy": policy,
                "selected_count": len(details),
                "unique_selected_count": len(set(selected_ids)),
                "unique_selected_fraction": (len(set(selected_ids)) / len(selected_ids)) if selected_ids else None,
                "source_group_count": len(set(source_ids)),
                "mean_quality": _mean(qualities),
                "min_quality": min(qualities) if qualities else None,
                "mean_artifact_score": _mean(artifacts),
                "max_artifact_score": max(artifacts) if artifacts else None,
                "artifact_selected_rate": _fraction(artifacts, lambda value: value > artifact_threshold),
                "artifact_gate_fail_rate": _fraction(details, lambda detail: detail.get("passed_artifact_gate") is False),
                "quality_gate_fail_rate": _fraction(details, lambda detail: detail.get("passed_quality_gate") is False),
                "mean_largest_source_group_fraction": _mean(largest_source),
                "duplicate_source_batch_rate": duplicate_source_batches / comparable_batches if comparable_batches else None,
            }
        )
    output.sort(key=lambda row: row["policy"])
    return output


def _paired_policy_deltas(rows: list[dict[str, Any]], policies: list[str]) -> list[dict[str, Any]]:
    final_by_episode_policy: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("_seed_name", "")), str(row.get("episode_id", "")), str(row.get("policy_name", "")))
        current = final_by_episode_policy.get(key)
        if current is None or int(row.get("round_index", 0)) > int(current.get("round_index", 0)):
            final_by_episode_policy[key] = row

    episode_keys = sorted({key[:2] for key in final_by_episode_policy})
    output = []
    for index, policy_a in enumerate(policies):
        for policy_b in policies[index + 1 :]:
            deltas = []
            for seed_name, episode_id in episode_keys:
                row_a = final_by_episode_policy.get((seed_name, episode_id, policy_a))
                row_b = final_by_episode_policy.get((seed_name, episode_id, policy_b))
                if row_a is None or row_b is None:
                    continue
                deltas.append(
                    _float(row_a.get("cumulative_balanced_relative_gain"))
                    - _float(row_b.get("cumulative_balanced_relative_gain"))
                )
            if not deltas:
                continue
            ci_low, ci_high = _bootstrap_ci95(deltas, seed=20260505 + index)
            output.append(
                {
                    "policy_a": policy_a,
                    "policy_b": policy_b,
                    "paired_episode_count": len(deltas),
                    "mean_delta_final_cumulative_gain": _mean(deltas),
                    "median_delta_final_cumulative_gain": _median(deltas),
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "policy_a_win_fraction": _fraction(deltas, lambda value: value > 0.0),
                    "policy_b_win_fraction": _fraction(deltas, lambda value: value < 0.0),
                    "tie_fraction": _fraction(deltas, lambda value: math.isclose(value, 0.0, rel_tol=1.0e-12, abs_tol=1.0e-12)),
                }
            )
    return output


def _pairwise_overlap(rows: list[dict[str, Any]], policies: list[str]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("_seed_name", "")),
            str(row.get("episode_id", "")),
            int(row.get("round_index", 0)),
            str(row.get("policy_name", "")),
        )
        by_key[key] = row

    episode_round_keys = sorted({key[:3] for key in by_key})
    output = []
    for index, policy_a in enumerate(policies):
        for policy_b in policies[index + 1 :]:
            jaccards = []
            exact_matches = 0
            intersections = []
            sizes_a = []
            sizes_b = []
            for base_key in episode_round_keys:
                row_a = by_key.get((*base_key, policy_a))
                row_b = by_key.get((*base_key, policy_b))
                if row_a is None or row_b is None:
                    continue
                selected_a = tuple(str(value) for value in row_a.get("selected_ids", []))
                selected_b = tuple(str(value) for value in row_b.get("selected_ids", []))
                set_a = set(selected_a)
                set_b = set(selected_b)
                union = set_a | set_b
                intersection = set_a & set_b
                jaccards.append(len(intersection) / len(union) if union else 1.0)
                intersections.append(float(len(intersection)))
                sizes_a.append(float(len(set_a)))
                sizes_b.append(float(len(set_b)))
                if selected_a == selected_b:
                    exact_matches += 1
            comparable = len(jaccards)
            if comparable:
                output.append(
                    {
                        "policy_a": policy_a,
                        "policy_b": policy_b,
                        "comparable_round_count": comparable,
                        "mean_jaccard": _mean(jaccards),
                        "exact_batch_match_fraction": exact_matches / comparable,
                        "mean_intersection_size": _mean(intersections),
                        "mean_size_a": _mean(sizes_a),
                        "mean_size_b": _mean(sizes_b),
                    }
                )
    output.sort(key=lambda row: (row["policy_a"], row["policy_b"]))
    return output


def _artifact_gate_effect(rows: list[dict[str, Any]], hygiene_rows: list[dict[str, Any]]) -> dict[str, Any]:
    base_policy = BLEND_POLICY
    gated_policy = ARTIFACT_GATED_BLEND_POLICY
    rows_by_round_policy: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for row in rows:
        rows_by_round_policy[
            (
                str(row.get("_seed_name", "")),
                str(row.get("episode_id", "")),
                int(row.get("round_index", 0)),
                str(row.get("policy_name", "")),
            )
        ] = row

    base_keys = {key[:3] for key in rows_by_round_policy if key[3] == base_policy}
    gated_keys = {key[:3] for key in rows_by_round_policy if key[3] == gated_policy}
    comparable_keys = sorted(base_keys & gated_keys)
    jaccards = []
    exact_matches = 0
    changed_rounds = 0
    for key in comparable_keys:
        base = tuple(str(value) for value in rows_by_round_policy[(*key, base_policy)].get("selected_ids", []))
        gated = tuple(str(value) for value in rows_by_round_policy[(*key, gated_policy)].get("selected_ids", []))
        base_set = set(base)
        gated_set = set(gated)
        union = base_set | gated_set
        jaccards.append(len(base_set & gated_set) / len(union) if union else 1.0)
        if base == gated:
            exact_matches += 1
        else:
            changed_rounds += 1

    hygiene_by_policy = {row["policy"]: row for row in hygiene_rows}
    comparable = len(comparable_keys)
    exact_fraction = exact_matches / comparable if comparable else None
    return {
        "base_policy": base_policy,
        "gated_policy": gated_policy,
        "comparable_round_count": comparable,
        "changed_round_count": changed_rounds,
        "mean_jaccard": _mean(jaccards),
        "exact_batch_match_fraction": exact_fraction,
        "base_artifact_selected_rate": hygiene_by_policy.get(base_policy, {}).get("artifact_selected_rate"),
        "gated_artifact_selected_rate": hygiene_by_policy.get(gated_policy, {}).get("artifact_selected_rate"),
        "base_artifact_gate_fail_rate": hygiene_by_policy.get(base_policy, {}).get("artifact_gate_fail_rate"),
        "gated_artifact_gate_fail_rate": hygiene_by_policy.get(gated_policy, {}).get("artifact_gate_fail_rate"),
        "is_no_op": bool(comparable and changed_rounds == 0 and exact_fraction == 1.0),
    }


def _old_vs_blend(policy_outcomes: list[dict[str, Any]], pairwise_overlap: list[dict[str, Any]]) -> dict[str, Any]:
    by_policy = {row["policy"]: row for row in policy_outcomes}
    old = by_policy.get(OLD_NOVELTY_POLICY, {})
    blend = by_policy.get(BLEND_POLICY, {})
    overlap = _find_pair(pairwise_overlap, BLEND_POLICY, OLD_NOVELTY_POLICY)
    old_gain = old.get("mean_final_cumulative_gain")
    blend_gain = blend.get("mean_final_cumulative_gain")
    delta = _subtract_optional(old_gain, blend_gain)
    jaccard = overlap.get("mean_jaccard")
    if delta is None:
        read = "missing old/blend outcome rows"
    elif delta > 0 and _none_low(jaccard) < 0.5:
        read = "blend selected materially different batches, but those batches had lower held-out utility"
    elif delta > 0:
        read = "blend looked mostly redundant with old novelty and still underperformed"
    elif abs(delta) <= 0.01:
        read = "blend is roughly tied with old novelty; complexity is not yet paid for"
    else:
        read = "blend beat old novelty in this run"
    return {
        "old_policy": OLD_NOVELTY_POLICY,
        "blend_policy": BLEND_POLICY,
        "old_mean_final_cumulative_gain": old_gain,
        "blend_mean_final_cumulative_gain": blend_gain,
        "old_minus_blend_final_gain": delta,
        "old_final_episode_wins": old.get("final_episode_wins"),
        "blend_final_episode_wins": blend.get("final_episode_wins"),
        "mean_jaccard": jaccard,
        "exact_batch_match_fraction": overlap.get("exact_batch_match_fraction"),
        "read": read,
    }


def _source_kcenter_read(
    *,
    policy_hygiene: list[dict[str, Any]],
    paired_policy_deltas: list[dict[str, Any]],
    artifact_gate_effect: dict[str, Any],
) -> dict[str, Any]:
    comparison_specs = [
        ("sourcecap_minus_old", SOURCECAP_OLD_POLICY, OLD_NOVELTY_POLICY),
        ("kcenter_minus_old", KCENTER_WINDOW_POLICY, OLD_NOVELTY_POLICY),
        ("blend_minus_kcenter", BLEND_POLICY, KCENTER_WINDOW_POLICY),
        ("full_replay_minus_minus_ts2vec", SUBMITTED_FULL_REPLAY_POLICY, SUBMITTED_MINUS_TS2VEC_POLICY),
        ("full_replay_minus_minus_window", SUBMITTED_FULL_REPLAY_POLICY, SUBMITTED_MINUS_WINDOW_POLICY),
        ("full_replay_minus_no_kcenter", SUBMITTED_FULL_REPLAY_POLICY, SUBMITTED_NO_KCENTER_POLICY),
        ("artifact_gate_minus_blend", ARTIFACT_GATED_BLEND_POLICY, BLEND_POLICY),
    ]
    comparisons = []
    for comparison, policy_a, policy_b in comparison_specs:
        row = _oriented_paired_delta(paired_policy_deltas, policy_a, policy_b)
        row["comparison"] = comparison
        row["read"] = _delta_read(row)
        comparisons.append(row)

    by_comparison = {row["comparison"]: row for row in comparisons}
    summary = [
        "Source cap vs old novelty: "
        + _comparison_sentence(by_comparison.get("sourcecap_minus_old"))
        + ".",
        "k-center contribution in submitted-style replay: "
        + _comparison_sentence(by_comparison.get("full_replay_minus_no_kcenter"))
        + ".",
        "TS2Vec contribution over the submitted replay ablation: "
        + _comparison_sentence(by_comparison.get("full_replay_minus_minus_ts2vec"))
        + ".",
        "Window k-center control vs TS2Vec/window blend: "
        + _comparison_sentence(by_comparison.get("blend_minus_kcenter"))
        + ".",
    ]
    if artifact_gate_effect.get("is_no_op"):
        summary.append("Artifact gate did not change any compared batches.")
    elif artifact_gate_effect.get("changed_round_count", 0):
        summary.append(
            "Artifact gate changed "
            f"{int(artifact_gate_effect.get('changed_round_count', 0))} compared batches."
        )

    hygiene_by_policy = {row["policy"]: row for row in policy_hygiene}
    concentration_policies = [
        OLD_NOVELTY_POLICY,
        SOURCECAP_OLD_POLICY,
        KCENTER_WINDOW_POLICY,
        BLEND_POLICY,
        SUBMITTED_FULL_REPLAY_POLICY,
        SUBMITTED_NO_KCENTER_POLICY,
        WINDOW_NOVELTY_SAME_GATES_POLICY,
        TS2VEC_NOVELTY_SAME_GATES_POLICY,
    ]
    source_concentration = []
    for policy in concentration_policies:
        hygiene = hygiene_by_policy.get(policy)
        if not hygiene:
            continue
        source_concentration.append(
            {
                "policy": policy,
                "mean_largest_source_group_fraction": hygiene.get("mean_largest_source_group_fraction"),
                "duplicate_source_batch_rate": hygiene.get("duplicate_source_batch_rate"),
                "unique_selected_fraction": hygiene.get("unique_selected_fraction"),
                "source_group_count": hygiene.get("source_group_count"),
            }
        )

    return {
        "summary": summary,
        "comparisons": comparisons,
        "source_concentration": source_concentration,
    }


def _oriented_paired_delta(
    paired_policy_deltas: list[dict[str, Any]],
    policy_a: str,
    policy_b: str,
) -> dict[str, Any]:
    for row in paired_policy_deltas:
        if row.get("policy_a") == policy_a and row.get("policy_b") == policy_b:
            return _copy_delta_row(row, policy_a=policy_a, policy_b=policy_b)
        if row.get("policy_a") == policy_b and row.get("policy_b") == policy_a:
            return _copy_delta_row(row, policy_a=policy_a, policy_b=policy_b, invert=True)
    return {
        "policy_a": policy_a,
        "policy_b": policy_b,
        "paired_episode_count": 0,
        "mean_delta_final_cumulative_gain": None,
        "median_delta_final_cumulative_gain": None,
        "bootstrap_ci95_low": None,
        "bootstrap_ci95_high": None,
        "policy_a_win_fraction": None,
        "policy_b_win_fraction": None,
        "tie_fraction": None,
    }


def _copy_delta_row(
    row: dict[str, Any],
    *,
    policy_a: str,
    policy_b: str,
    invert: bool = False,
) -> dict[str, Any]:
    if not invert:
        return {
            "policy_a": policy_a,
            "policy_b": policy_b,
            "paired_episode_count": row.get("paired_episode_count", 0),
            "mean_delta_final_cumulative_gain": row.get("mean_delta_final_cumulative_gain"),
            "median_delta_final_cumulative_gain": row.get("median_delta_final_cumulative_gain"),
            "bootstrap_ci95_low": row.get("bootstrap_ci95_low"),
            "bootstrap_ci95_high": row.get("bootstrap_ci95_high"),
            "policy_a_win_fraction": row.get("policy_a_win_fraction"),
            "policy_b_win_fraction": row.get("policy_b_win_fraction"),
            "tie_fraction": row.get("tie_fraction"),
        }

    ci_low = row.get("bootstrap_ci95_low")
    ci_high = row.get("bootstrap_ci95_high")
    return {
        "policy_a": policy_a,
        "policy_b": policy_b,
        "paired_episode_count": row.get("paired_episode_count", 0),
        "mean_delta_final_cumulative_gain": _negate_optional(row.get("mean_delta_final_cumulative_gain")),
        "median_delta_final_cumulative_gain": _negate_optional(row.get("median_delta_final_cumulative_gain")),
        "bootstrap_ci95_low": _negate_optional(ci_high),
        "bootstrap_ci95_high": _negate_optional(ci_low),
        "policy_a_win_fraction": row.get("policy_b_win_fraction"),
        "policy_b_win_fraction": row.get("policy_a_win_fraction"),
        "tie_fraction": row.get("tie_fraction"),
    }


def _comparison_sentence(row: dict[str, Any] | None) -> str:
    if not row:
        return "missing paired data"
    return str(row.get("read") or "missing paired data")


def _delta_read(row: dict[str, Any]) -> str:
    policy_a = str(row.get("policy_a", "policy A"))
    policy_b = str(row.get("policy_b", "policy B"))
    mean_delta = row.get("mean_delta_final_cumulative_gain")
    ci_low = row.get("bootstrap_ci95_low")
    ci_high = row.get("bootstrap_ci95_high")
    if mean_delta is None:
        return "missing paired data"
    mean_delta = float(mean_delta)
    if ci_low is not None and ci_high is not None:
        ci_low = float(ci_low)
        ci_high = float(ci_high)
        if ci_low > 0.0:
            return f"{policy_a} beat {policy_b} on paired final gain"
        if ci_high < 0.0:
            return f"{policy_a} trailed {policy_b} on paired final gain"
    if abs(mean_delta) <= 0.01:
        return f"{policy_a} and {policy_b} were effectively tied"
    if mean_delta > 0.0:
        return f"{policy_a} was directionally higher than {policy_b}, but uncertainty crosses zero"
    return f"{policy_a} was directionally lower than {policy_b}, but uncertainty crosses zero"


def _coverage_story(reports: list[dict[str, Any]], aggregate: dict[str, Any]) -> dict[str, Any]:
    difficulty_rows = [row for report in reports for row in report.get("difficulty_audit", [])]
    exact_flags = [
        bool(flag)
        for row in difficulty_rows
        for flag in row.get("oracle_fraction_exact_by_round", [])
        if isinstance(row.get("oracle_fraction_exact_by_round"), list)
    ]
    near_zero = [
        _float(row.get("near_zero_oracle_round_fraction"))
        for row in difficulty_rows
        if _is_number(row.get("near_zero_oracle_round_fraction"))
    ]
    return {
        "all_leakage_ok": aggregate.get("all_leakage_ok"),
        "oracle_fraction_exact_all_rounds": aggregate.get("oracle_fraction_exact_all_rounds", all(exact_flags) if exact_flags else None),
        "oracle_fraction_exact_round_fraction": aggregate.get(
            "oracle_fraction_exact_round_fraction",
            sum(1 for flag in exact_flags if flag) / len(exact_flags) if exact_flags else None,
        ),
        "mean_near_zero_oracle_round_fraction": aggregate.get("mean_near_zero_oracle_round_fraction", _mean(near_zero)),
        "difficulty_episode_count": len(difficulty_rows),
        "max_oracle_greedy_cumulative_gain_mean": _mean(
            [
                _float(row.get("max_oracle_greedy_cumulative_gain"))
                for row in difficulty_rows
                if _is_number(row.get("max_oracle_greedy_cumulative_gain"))
            ]
        ),
    }


def _diagnosis(
    *,
    old_vs_blend: dict[str, Any],
    artifact_gate_effect: dict[str, Any],
    coverage_story: dict[str, Any],
) -> list[str]:
    diagnosis = []
    delta = old_vs_blend.get("old_minus_blend_final_gain")
    if delta is not None:
        if float(delta) > 0.0:
            diagnosis.append(
                "old_novelty_window beat the TS2Vec/window blend by "
                f"{float(delta):.4f} mean final cumulative gain."
            )
        elif abs(float(delta)) <= 0.01:
            diagnosis.append("old_novelty_window and the TS2Vec/window blend were effectively tied.")
        else:
            diagnosis.append(
                "the TS2Vec/window blend beat old_novelty_window by "
                f"{abs(float(delta)):.4f} mean final cumulative gain."
            )
    if old_vs_blend.get("read"):
        diagnosis.append(str(old_vs_blend["read"]) + ".")

    if artifact_gate_effect.get("is_no_op"):
        diagnosis.append("artifact gate did not change any compared batches; treat it as a no-op for this run.")
    elif artifact_gate_effect.get("changed_round_count", 0):
        diagnosis.append(
            "artifact gate changed "
            f"{int(artifact_gate_effect.get('changed_round_count', 0))} compared policy-round batches."
        )

    if coverage_story.get("oracle_fraction_exact_all_rounds") is True:
        diagnosis.append("oracle-fraction denominators were exact for all audited rounds.")
    near_zero = coverage_story.get("mean_near_zero_oracle_round_fraction")
    if near_zero is not None and float(near_zero) > 0.10:
        diagnosis.append(
            "some evaluation rounds had near-zero oracle opportunity, so use win counts and curves rather than one headline mean."
        )
    return diagnosis


def _rows_by_policy(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("policy_name", ""))].append(row)
    return grouped


def _ordered_policies(reports: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for report in reports:
        for policy in report.get("policies", []):
            policy = str(policy)
            if policy and policy not in seen:
                ordered.append(policy)
                seen.add(policy)
    for row in rows:
        policy = str(row.get("policy_name", ""))
        if policy and policy not in seen:
            ordered.append(policy)
            seen.add(policy)
    return ordered


def _final_rows_by_policy(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("_seed_name", "")), str(row.get("episode_id", "")), str(row.get("policy_name", "")))
        current = latest.get(key)
        if current is None or int(row.get("round_index", 0)) > int(current.get("round_index", 0)):
            latest[key] = row
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in latest.values():
        grouped[str(row.get("policy_name", ""))].append(row)
    return grouped


def _final_win_counts(final_rows_by_policy: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    by_episode: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rows in final_rows_by_policy.values():
        for row in rows:
            by_episode[(str(row.get("_seed_name", "")), str(row.get("episode_id", "")))].append(row)
    counts: dict[str, int] = defaultdict(int)
    for rows in by_episode.values():
        if not rows:
            continue
        best = max(_float(row.get("cumulative_balanced_relative_gain")) for row in rows)
        for row in rows:
            if math.isclose(_float(row.get("cumulative_balanced_relative_gain")), best, rel_tol=1.0e-12, abs_tol=1.0e-12):
                counts[str(row.get("policy_name", ""))] += 1
    return counts


def _win_counts(rows: list[dict[str, Any]], *, value_key: str) -> dict[str, int]:
    by_round: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_round[(str(row.get("_seed_name", "")), str(row.get("episode_id", "")), int(row.get("round_index", 0)))].append(row)
    counts: dict[str, int] = defaultdict(int)
    for round_rows in by_round.values():
        best = max(_float(row.get(value_key)) for row in round_rows)
        for row in round_rows:
            if math.isclose(_float(row.get(value_key)), best, rel_tol=1.0e-12, abs_tol=1.0e-12):
                counts[str(row.get("policy_name", ""))] += 1
    return counts


def _coverage_means(rows: list[dict[str, Any]], metric: str) -> dict[str, float | None]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        coverage = row.get("coverage_by_representation", {})
        if not isinstance(coverage, dict):
            continue
        for representation, rep_row in coverage.items():
            if isinstance(rep_row, dict) and _is_number(rep_row.get(metric)):
                values[str(representation)].append(_float(rep_row.get(metric)))
    return {key: _mean(rep_values) for key, rep_values in sorted(values.items())}


def _selection_details(row: dict[str, Any]) -> list[dict[str, Any]]:
    details = row.get("selection_details", [])
    return details if isinstance(details, list) else []


def _find_pair(pairwise_overlap: list[dict[str, Any]], policy_a: str, policy_b: str) -> dict[str, Any]:
    wanted = {policy_a, policy_b}
    for row in pairwise_overlap:
        if {row.get("policy_a"), row.get("policy_b")} == wanted:
            return row
    return {}


def _subtract_optional(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return _float(left) - _float(right)


def _negate_optional(value: Any) -> float | None:
    if value is None:
        return None
    return -_float(value)


def _mean(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    value_mean = float(mean(values))
    return float(math.sqrt(sum((value - value_mean) ** 2 for value in values) / len(values)))


def _fraction(values: list[Any], predicate) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if predicate(value)) / len(values)


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


def _is_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


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
    if isinstance(value, bool):
        return str(value)
    if _is_number(value):
        return f"{float(value):.4f}"
    return str(value)
