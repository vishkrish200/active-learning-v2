from __future__ import annotations

import json
import tarfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence


DEFAULT_FOCAL_POLICY = "window_kcenter_v1"
DEFAULT_COMPARISON_POLICIES = (
    "ts2vec_kcenter_v1",
    "submitted_full_replay_v1",
    "quality_stratified_random_v1",
)


def build_selection_mechanism_audit(
    run_input: str | Path,
    *,
    budget_k: int | None = None,
    focal_policy: str = DEFAULT_FOCAL_POLICY,
    comparison_policies: Sequence[str] = DEFAULT_COMPARISON_POLICIES,
) -> dict[str, Any]:
    source = _load_run_source(Path(run_input))
    selected_rows = source["selected_rows"]
    forecast_rows = source["forecast_rows"]
    if not selected_rows:
        raise ValueError(f"No selected rows found in {run_input}")
    if not forecast_rows:
        raise ValueError(f"No downstream forecast rows found in {run_input}")

    final_budget = int(budget_k) if budget_k is not None else max(int(row["budget_k"]) for row in selected_rows)
    selected_rows = [row for row in selected_rows if int(row.get("budget_k", -1)) == final_budget]
    forecast_rows = [row for row in forecast_rows if int(row.get("budget_k", -1)) == final_budget]
    policies = sorted({str(row["policy_id"]) for row in selected_rows} | {str(row["policy_id"]) for row in forecast_rows})
    units = sorted({str(row["_unit_id"]) for row in forecast_rows})
    pairwise = _pairwise_selection_contrasts(selected_rows, forecast_rows, policies)
    diagnostics = _focal_episode_diagnostics(
        selected_rows,
        forecast_rows,
        focal_policy=focal_policy,
        comparison_policies=comparison_policies,
    )

    return {
        "input": {
            "run_input": str(run_input),
            "source_kind": source["source_kind"],
            "seed_count": len(source["seed_names"]),
            "seed_names": source["seed_names"],
            "unit_count": len(units),
            "budget_k": final_budget,
            "policies": policies,
            "focal_policy": focal_policy,
            "comparison_policies": list(comparison_policies),
        },
        "policy_selection_profiles": _policy_selection_profiles(selected_rows),
        "policy_outcome_profiles": _policy_outcome_profiles(forecast_rows),
        "pairwise_selection_contrasts": pairwise,
        "focal_pairwise_contrasts": _focal_pairwise_contrasts(
            pairwise,
            focal_policy=focal_policy,
            comparison_policies=comparison_policies,
        ),
        "focal_policy_episode_diagnostics": diagnostics,
        "read": _interpret_mechanism(pairwise, selected_rows, focal_policy=focal_policy),
    }


def write_selection_mechanism_audit_reports(
    run_input: str | Path,
    *,
    output_json: str | Path,
    output_markdown: str | Path,
    budget_k: int | None = None,
    focal_policy: str = DEFAULT_FOCAL_POLICY,
    comparison_policies: Sequence[str] = DEFAULT_COMPARISON_POLICIES,
) -> dict[str, str]:
    audit = build_selection_mechanism_audit(
        run_input,
        budget_k=budget_k,
        focal_policy=focal_policy,
        comparison_policies=comparison_policies,
    )
    json_path = Path(output_json)
    markdown_path = Path(output_markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_selection_mechanism_audit_markdown(audit), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def render_selection_mechanism_audit_markdown(audit: Mapping[str, Any]) -> str:
    input_info = audit.get("input", {})
    lines = [
        "# Selection Mechanism Audit",
        "",
        "## Scope",
        "",
        f"- run input: `{input_info.get('run_input')}`",
        f"- source kind: `{input_info.get('source_kind')}`",
        f"- seeds: `{input_info.get('seed_count')}`",
        f"- units: `{input_info.get('unit_count')}`",
        f"- budget: `K={input_info.get('budget_k')}`",
        f"- focal policy: `{input_info.get('focal_policy')}`",
        "",
        "## Read",
        "",
    ]
    lines.extend(f"- {item}" for item in audit.get("read", []))
    lines.extend(
        [
            "",
            "## Policy Selection Profiles",
            "",
            "| policy | selected rows | unique clips | unique sources | mean quality | mean artifact | duplicate-source batch rate | unique-source fraction |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in audit.get("policy_selection_profiles", []):
        lines.append(
            "| `{policy}` | {rows} | {clips} | {sources} | {quality} | {artifact} | {dupes} | {unique_frac} |".format(
                policy=row.get("policy_id", ""),
                rows=int(row.get("selected_row_count", 0)),
                clips=int(row.get("unique_selected_clip_count", 0)),
                sources=int(row.get("unique_source_group_count", 0)),
                quality=_fmt(row.get("mean_quality_score")),
                artifact=_fmt(row.get("mean_artifact_score")),
                dupes=_fmt(row.get("duplicate_source_batch_rate")),
                unique_frac=_fmt(row.get("mean_unique_source_fraction")),
            )
        )
    lines.extend(
        [
            "",
            "## Policy Outcome Profiles",
            "",
            "| policy | rows | mean after MSE | median after MSE | best/tie wins |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in audit.get("policy_outcome_profiles", []):
        lines.append(
            "| `{policy}` | {rows} | {mean_mse} | {median_mse} | {wins} |".format(
                policy=row.get("policy_id", ""),
                rows=int(row.get("row_count", 0)),
                mean_mse=_fmt(row.get("mean_after_mse")),
                median_mse=_fmt(row.get("median_after_mse")),
                wins=int(row.get("best_or_tie_count", 0)),
            )
        )
    lines.extend(
        [
            "",
            "## Focal Policy Contrasts",
            "",
            "| focal policy | comparison | units | clip Jaccard | source Jaccard | focal lower MSE | comparison lower MSE | focal advantage | Jaccard when focal wins | Jaccard when comparison wins |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in audit.get("focal_pairwise_contrasts", []):
        lines.append(
            "| `{focal}` | `{comparison}` | {units} | {clip_j} | {source_j} | {focal_wins} | {comparison_wins} | {adv} | {focal_j} | {comparison_j} |".format(
                focal=row.get("focal_policy", ""),
                comparison=row.get("comparison_policy", ""),
                units=int(row.get("paired_unit_count", 0)),
                clip_j=_fmt(row.get("mean_jaccard")),
                source_j=_fmt(row.get("mean_source_jaccard")),
                focal_wins=int(row.get("focal_lower_mse_count", 0)),
                comparison_wins=int(row.get("comparison_lower_mse_count", 0)),
                adv=_fmt(row.get("mean_after_mse_advantage_focal_over_comparison")),
                focal_j=_fmt(row.get("mean_jaccard_when_focal_wins")),
                comparison_j=_fmt(row.get("mean_jaccard_when_comparison_wins")),
            )
        )
    lines.extend(
        [
            "",
            "## Pairwise Selection Contrasts",
            "",
            "| policy A | policy B | units | clip Jaccard | source Jaccard | A lower MSE | B lower MSE | A advantage | Jaccard when A wins | Jaccard when B wins |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in audit.get("pairwise_selection_contrasts", []):
        if not row.get("highlight"):
            continue
        lines.append(
            "| `{a}` | `{b}` | {units} | {clip_j} | {source_j} | {a_wins} | {b_wins} | {adv} | {a_j} | {b_j} |".format(
                a=row.get("policy_a", ""),
                b=row.get("policy_b", ""),
                units=int(row.get("paired_unit_count", 0)),
                clip_j=_fmt(row.get("mean_jaccard")),
                source_j=_fmt(row.get("mean_source_jaccard")),
                a_wins=int(row.get("policy_a_lower_mse_count", 0)),
                b_wins=int(row.get("policy_b_lower_mse_count", 0)),
                adv=_fmt(row.get("mean_after_mse_advantage_a_over_b")),
                a_j=_fmt(row.get("mean_jaccard_when_policy_a_wins")),
                b_j=_fmt(row.get("mean_jaccard_when_policy_b_wins")),
            )
        )
    lines.extend(
        [
            "",
            "## Largest Focal Episode Contrasts",
            "",
            "| comparison | unit | direction | MSE advantage | shared clips | focal-only clips | comparison-only clips |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in audit.get("focal_policy_episode_diagnostics", [])[:20]:
        lines.append(
            "| `{comparison}` | `{unit}` | `{direction}` | {advantage} | {shared} | {focal_only} | {comparison_only} |".format(
                comparison=row.get("comparison_policy", ""),
                unit=row.get("unit_id", ""),
                direction=row.get("advantage_direction", ""),
                advantage=_fmt(row.get("after_mse_advantage")),
                shared=len(row.get("shared_selected_ids", [])),
                focal_only=len(row.get("focal_only_selected_ids", [])),
                comparison_only=len(row.get("comparison_only_selected_ids", [])),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _load_run_source(path: Path) -> dict[str, Any]:
    if path.is_file() and path.name.endswith((".tgz", ".tar.gz")):
        return _load_tar_source(path)
    return _load_directory_source(path)


def _load_directory_source(root: Path) -> dict[str, Any]:
    selected_rows: list[dict[str, Any]] = []
    forecast_rows: list[dict[str, Any]] = []
    seed_names: set[str] = set()
    for coverage_path in sorted(root.glob("**/seed_*/blind_target_coverage_benchmark_report.json")):
        seed_name = coverage_path.parent.name
        seed_names.add(seed_name)
        payload = json.loads(coverage_path.read_text(encoding="utf-8"))
        selected_rows.extend(_annotate_rows(payload.get("selected_rows", []), seed_name))
    for forecast_path in sorted(root.glob("**/seed_*/downstream_forecast_task_report.json")):
        seed_name = forecast_path.parent.name
        seed_names.add(seed_name)
        payload = json.loads(forecast_path.read_text(encoding="utf-8"))
        forecast_rows.extend(_annotate_rows(payload.get("rows", []), seed_name))
    return {
        "source_kind": "directory",
        "seed_names": sorted(seed_names),
        "selected_rows": selected_rows,
        "forecast_rows": forecast_rows,
    }


def _load_tar_source(path: Path) -> dict[str, Any]:
    selected_rows: list[dict[str, Any]] = []
    forecast_rows: list[dict[str, Any]] = []
    seed_names: set[str] = set()
    with tarfile.open(path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            member_path = Path(member.name)
            if len(member_path.parts) < 2:
                continue
            seed_name = _seed_name_from_parts(member_path.parts)
            if seed_name is None:
                continue
            if member_path.name not in {"blind_target_coverage_benchmark_report.json", "downstream_forecast_task_report.json"}:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            payload = json.loads(handle.read().decode("utf-8"))
            seed_names.add(seed_name)
            if member_path.name == "blind_target_coverage_benchmark_report.json":
                selected_rows.extend(_annotate_rows(payload.get("selected_rows", []), seed_name))
            else:
                forecast_rows.extend(_annotate_rows(payload.get("rows", []), seed_name))
    return {
        "source_kind": "tar_archive",
        "seed_names": sorted(seed_names),
        "selected_rows": selected_rows,
        "forecast_rows": forecast_rows,
    }


def _seed_name_from_parts(parts: Sequence[str]) -> str | None:
    for part in parts:
        if part.startswith("seed_"):
            return part
    return None


def _annotate_rows(rows: Sequence[Mapping[str, Any]], seed_name: str) -> list[dict[str, Any]]:
    annotated = []
    for row in rows:
        item = dict(row)
        item["_seed_name"] = seed_name
        item["_unit_id"] = _unit_id(seed_name, item)
        if "source_group_id" not in item and "sample_id" in item:
            item["source_group_id"] = _source_group(str(item["sample_id"]))
        annotated.append(item)
    return annotated


def _policy_selection_profiles(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy_id"])].append(row)
    profiles = []
    for policy, policy_rows in sorted(grouped.items()):
        batches: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in policy_rows:
            batches[str(row["_unit_id"])].append(row)
        duplicate_source_batches = 0
        largest_source_fractions = []
        unique_source_fractions = []
        for batch in batches.values():
            sources = [_row_source(row) for row in batch]
            counts = Counter(sources)
            duplicate_source_batches += int(any(count > 1 for count in counts.values()))
            largest_source_fractions.append(max(counts.values()) / len(sources) if sources else 0.0)
            unique_source_fractions.append(len(counts) / len(sources) if sources else 0.0)
        sample_ids = [str(row.get("sample_id", "")) for row in policy_rows]
        sources = [_row_source(row) for row in policy_rows]
        profiles.append(
            {
                "policy_id": policy,
                "selected_row_count": len(policy_rows),
                "unit_count": len(batches),
                "unique_selected_clip_count": len(set(sample_ids)),
                "unique_selected_clip_fraction": len(set(sample_ids)) / len(sample_ids) if sample_ids else 0.0,
                "unique_source_group_count": len(set(sources)),
                "mean_quality_score": _mean_optional(policy_rows, "quality_score"),
                "mean_artifact_score": _mean_optional(policy_rows, "artifact_score"),
                "valid_rate": _mean_bool(policy_rows, "valid"),
                "artifact_gate_pass_rate": _mean_bool(policy_rows, "passed_artifact_gate"),
                "mean_rank_index": _mean_optional(policy_rows, "rank_index"),
                "duplicate_source_batch_rate": duplicate_source_batches / len(batches) if batches else 0.0,
                "mean_largest_source_group_fraction": _mean(largest_source_fractions),
                "mean_unique_source_fraction": _mean(unique_source_fractions),
                "top_source_groups": _top_counts(sources, key_name="source_group_id"),
            }
        )
    return profiles


def _policy_outcome_profiles(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_unit: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy_id"])].append(row)
        by_unit[str(row["_unit_id"])].append(row)
    best_counts: Counter[str] = Counter()
    for unit_rows in by_unit.values():
        best = min(float(row["after_mse"]) for row in unit_rows)
        for row in unit_rows:
            if abs(float(row["after_mse"]) - best) <= 1.0e-12:
                best_counts[str(row["policy_id"])] += 1
    profiles = []
    for policy, policy_rows in grouped.items():
        profiles.append(
            {
                "policy_id": policy,
                "row_count": len(policy_rows),
                "mean_after_mse": _mean(float(row["after_mse"]) for row in policy_rows),
                "median_after_mse": _median(float(row["after_mse"]) for row in policy_rows),
                "mean_relative_mse_reduction": _mean_optional(policy_rows, "relative_mse_reduction"),
                "best_or_tie_count": best_counts.get(policy, 0),
            }
        )
    return sorted(profiles, key=lambda row: (float(row["mean_after_mse"]), str(row["policy_id"])))


def _pairwise_selection_contrasts(
    selected_rows: Sequence[Mapping[str, Any]],
    forecast_rows: Sequence[Mapping[str, Any]],
    policies: Sequence[str],
) -> list[dict[str, Any]]:
    selected = _selected_by_policy_unit(selected_rows)
    forecast = _forecast_by_policy_unit(forecast_rows)
    highlights = {
        tuple(sorted(pair))
        for pair in (
            ("window_kcenter_v1", "ts2vec_kcenter_v1"),
            ("window_kcenter_v1", "submitted_full_replay_v1"),
            ("window_kcenter_v1", "quality_stratified_random_v1"),
            ("ts2vec_kcenter_v1", "submitted_full_replay_v1"),
        )
    }
    records = []
    for policy_a, policy_b in combinations(sorted(policies), 2):
        units = sorted(set(selected[policy_a]) & set(selected[policy_b]) & set(forecast[policy_a]) & set(forecast[policy_b]))
        clip_jaccards = []
        source_jaccards = []
        shared_counts = []
        a_only_counts = []
        b_only_counts = []
        mse_deltas = []
        a_win_jaccards = []
        b_win_jaccards = []
        a_only_sources: list[str] = []
        b_only_sources: list[str] = []
        for unit in units:
            a_ids = _selected_ids(selected[policy_a][unit])
            b_ids = _selected_ids(selected[policy_b][unit])
            a_sources = {_source_group(sample_id) for sample_id in a_ids}
            b_sources = {_source_group(sample_id) for sample_id in b_ids}
            jaccard = _jaccard(set(a_ids), set(b_ids))
            clip_jaccards.append(jaccard)
            source_jaccards.append(_jaccard(a_sources, b_sources))
            shared_counts.append(len(set(a_ids) & set(b_ids)))
            a_only = set(a_ids) - set(b_ids)
            b_only = set(b_ids) - set(a_ids)
            a_only_counts.append(len(a_only))
            b_only_counts.append(len(b_only))
            a_only_sources.extend(_source_group(sample_id) for sample_id in sorted(a_only))
            b_only_sources.extend(_source_group(sample_id) for sample_id in sorted(b_only))
            delta = float(forecast[policy_a][unit]["after_mse"]) - float(forecast[policy_b][unit]["after_mse"])
            mse_deltas.append(delta)
            if delta < -1.0e-12:
                a_win_jaccards.append(jaccard)
            elif delta > 1.0e-12:
                b_win_jaccards.append(jaccard)
        records.append(
            {
                "policy_a": policy_a,
                "policy_b": policy_b,
                "paired_unit_count": len(units),
                "mean_jaccard": _mean(clip_jaccards),
                "mean_source_jaccard": _mean(source_jaccards),
                "exact_match_rate": _mean(1.0 if value == 1.0 else 0.0 for value in clip_jaccards),
                "mean_shared_clip_count": _mean(shared_counts),
                "mean_policy_a_only_clip_count": _mean(a_only_counts),
                "mean_policy_b_only_clip_count": _mean(b_only_counts),
                "mean_after_mse_delta_a_minus_b": _mean(mse_deltas),
                "mean_after_mse_advantage_a_over_b": -_mean(mse_deltas),
                "policy_a_lower_mse_count": sum(1 for delta in mse_deltas if delta < -1.0e-12),
                "policy_b_lower_mse_count": sum(1 for delta in mse_deltas if delta > 1.0e-12),
                "tie_count": sum(1 for delta in mse_deltas if abs(delta) <= 1.0e-12),
                "mean_jaccard_when_policy_a_wins": _mean(a_win_jaccards),
                "mean_jaccard_when_policy_b_wins": _mean(b_win_jaccards),
                "top_policy_a_only_source_groups": _top_counts(a_only_sources, key_name="source_group_id"),
                "top_policy_b_only_source_groups": _top_counts(b_only_sources, key_name="source_group_id"),
                "highlight": tuple(sorted((policy_a, policy_b))) in highlights,
            }
        )
    return records


def _focal_pairwise_contrasts(
    pairwise: Sequence[Mapping[str, Any]],
    *,
    focal_policy: str,
    comparison_policies: Sequence[str],
) -> list[dict[str, Any]]:
    rows = []
    for comparison in comparison_policies:
        row = _find_pairwise(pairwise, focal_policy, comparison)
        if row is None:
            continue
        focal_is_a = row["policy_a"] == focal_policy
        rows.append(
            {
                "focal_policy": focal_policy,
                "comparison_policy": comparison,
                "paired_unit_count": row["paired_unit_count"],
                "mean_jaccard": row["mean_jaccard"],
                "mean_source_jaccard": row["mean_source_jaccard"],
                "exact_match_rate": row["exact_match_rate"],
                "mean_shared_clip_count": row["mean_shared_clip_count"],
                "mean_focal_only_clip_count": row["mean_policy_a_only_clip_count"]
                if focal_is_a
                else row["mean_policy_b_only_clip_count"],
                "mean_comparison_only_clip_count": row["mean_policy_b_only_clip_count"]
                if focal_is_a
                else row["mean_policy_a_only_clip_count"],
                "mean_after_mse_advantage_focal_over_comparison": row["mean_after_mse_advantage_a_over_b"]
                if focal_is_a
                else -float(row["mean_after_mse_advantage_a_over_b"]),
                "focal_lower_mse_count": row["policy_a_lower_mse_count"]
                if focal_is_a
                else row["policy_b_lower_mse_count"],
                "comparison_lower_mse_count": row["policy_b_lower_mse_count"]
                if focal_is_a
                else row["policy_a_lower_mse_count"],
                "tie_count": row["tie_count"],
                "mean_jaccard_when_focal_wins": row["mean_jaccard_when_policy_a_wins"]
                if focal_is_a
                else row["mean_jaccard_when_policy_b_wins"],
                "mean_jaccard_when_comparison_wins": row["mean_jaccard_when_policy_b_wins"]
                if focal_is_a
                else row["mean_jaccard_when_policy_a_wins"],
                "top_focal_only_source_groups": row["top_policy_a_only_source_groups"]
                if focal_is_a
                else row["top_policy_b_only_source_groups"],
                "top_comparison_only_source_groups": row["top_policy_b_only_source_groups"]
                if focal_is_a
                else row["top_policy_a_only_source_groups"],
            }
        )
    return rows


def _find_pairwise(pairwise: Sequence[Mapping[str, Any]], policy_a: str, policy_b: str) -> Mapping[str, Any] | None:
    for row in pairwise:
        if {str(row["policy_a"]), str(row["policy_b"])} == {policy_a, policy_b}:
            return row
    return None


def _focal_episode_diagnostics(
    selected_rows: Sequence[Mapping[str, Any]],
    forecast_rows: Sequence[Mapping[str, Any]],
    *,
    focal_policy: str,
    comparison_policies: Sequence[str],
) -> list[dict[str, Any]]:
    selected = _selected_by_policy_unit(selected_rows)
    forecast = _forecast_by_policy_unit(forecast_rows)
    records = []
    comparison_order = {policy: index for index, policy in enumerate(comparison_policies)}
    for comparison in comparison_policies:
        units = sorted(set(selected[focal_policy]) & set(selected[comparison]) & set(forecast[focal_policy]) & set(forecast[comparison]))
        for unit in units:
            focal_ids = set(_selected_ids(selected[focal_policy][unit]))
            comparison_ids = set(_selected_ids(selected[comparison][unit]))
            delta = float(forecast[comparison][unit]["after_mse"]) - float(forecast[focal_policy][unit]["after_mse"])
            records.append(
                {
                    "comparison_policy": comparison,
                    "unit_id": unit,
                    "after_mse_advantage": delta,
                    "advantage_direction": _advantage_direction(delta),
                    "focal_after_mse": float(forecast[focal_policy][unit]["after_mse"]),
                    "comparison_after_mse": float(forecast[comparison][unit]["after_mse"]),
                    "jaccard": _jaccard(focal_ids, comparison_ids),
                    "shared_selected_ids": sorted(focal_ids & comparison_ids),
                    "focal_only_selected_ids": sorted(focal_ids - comparison_ids),
                    "comparison_only_selected_ids": sorted(comparison_ids - focal_ids),
                    "focal_only_source_groups": sorted({_source_group(sample_id) for sample_id in focal_ids - comparison_ids}),
                    "comparison_only_source_groups": sorted({_source_group(sample_id) for sample_id in comparison_ids - focal_ids}),
                }
            )
    return sorted(
        records,
        key=lambda row: (
            comparison_order.get(str(row["comparison_policy"]), len(comparison_order)),
            -abs(float(row["after_mse_advantage"])),
        ),
    )


def _interpret_mechanism(
    pairwise: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    *,
    focal_policy: str,
) -> list[str]:
    by_pair = {tuple(sorted((str(row["policy_a"]), str(row["policy_b"])))): row for row in pairwise}
    reads = [
        "This is a selection-mechanism audit: it explains selected-set behavior, not a new policy decision.",
        "All summaries use already-selected rows and downstream forecast rows; no target features are used for selection.",
    ]
    for comparison in DEFAULT_COMPARISON_POLICIES:
        row = by_pair.get(tuple(sorted((focal_policy, comparison))))
        if not row:
            continue
        jaccard = float(row.get("mean_jaccard", 0.0))
        if jaccard < 0.25:
            overlap_read = "largely different selected clips"
        elif jaccard < 0.60:
            overlap_read = "partially overlapping selected clips"
        else:
            overlap_read = "substantially overlapping selected clips"
        reads.append(f"`{focal_policy}` vs `{comparison}` has {overlap_read} at K with mean clip Jaccard `{jaccard:.6f}`.")
    profile = {str(row["policy_id"]): row for row in _policy_selection_profiles(selected_rows)}
    focal_profile = profile.get(focal_policy)
    if focal_profile:
        reads.append(
            f"`{focal_policy}` duplicate-source batch rate is `{float(focal_profile['duplicate_source_batch_rate']):.6f}` and mean unique-source fraction is `{float(focal_profile['mean_unique_source_fraction']):.6f}`."
        )
    return reads


def _selected_by_policy_unit(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, list[Mapping[str, Any]]]]:
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["policy_id"])][str(row["_unit_id"])].append(row)
    for by_unit in grouped.values():
        for unit_rows in by_unit.values():
            unit_rows.sort(key=lambda row: int(row.get("rank_index", 0)))
    return grouped


def _forecast_by_policy_unit(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Mapping[str, Any]]]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["policy_id"])][str(row["_unit_id"])] = row
    return grouped


def _selected_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(row.get("sample_id", "")) for row in sorted(rows, key=lambda row: int(row.get("rank_index", 0)))]


def _row_source(row: Mapping[str, Any]) -> str:
    source = row.get("source_group_id")
    if source:
        return str(source)
    return _source_group(str(row.get("sample_id", "")))


def _source_group(sample_id: str) -> str:
    return sample_id.split("_clip", 1)[0] if "_clip" in sample_id else sample_id


def _unit_id(seed_name: str, row: Mapping[str, Any]) -> str:
    return f"{seed_name}:{row.get('episode_id', '')}"


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _advantage_direction(delta: float) -> str:
    if delta > 1.0e-12:
        return "focal_lower_mse"
    if delta < -1.0e-12:
        return "comparison_lower_mse"
    return "tie"


def _top_counts(values: Sequence[str], *, key_name: str, limit: int = 10) -> list[dict[str, Any]]:
    return [{key_name: key, "count": count} for key, count in Counter(values).most_common(limit)]


def _mean(values: Sequence[float] | Any) -> float:
    materialized = list(values)
    return float(mean(materialized)) if materialized else 0.0


def _median(values: Sequence[float] | Any) -> float:
    materialized = list(values)
    return float(median(materialized)) if materialized else 0.0


def _mean_optional(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return _mean(values) if values else None


def _mean_bool(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [1.0 if bool(row[key]) else 0.0 for row in rows if row.get(key) is not None]
    return _mean(values) if values else None


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.9f}"
    return str(value)
