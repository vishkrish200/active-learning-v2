from __future__ import annotations

import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence


DEFAULT_KILLED_POLICY = "support_gap_window_probcover_v1"
DEFAULT_CHAMPION_POLICY = "window_kcenter_v1"
DEFAULT_RANDOM_POLICY = "quality_stratified_random_v1"
DEFAULT_SUBMITTED_POLICY = "submitted_full_replay_v1"
DEFAULT_TS2VEC_POLICY = "ts2vec_kcenter_v1"
DEFAULT_QUALITY_FLOOR_POLICY = "quality_only_v1"


def build_downstream_forecast_autopsy(
    run_root: str | Path,
    *,
    killed_policy: str = DEFAULT_KILLED_POLICY,
    champion_policy: str = DEFAULT_CHAMPION_POLICY,
    random_policy: str = DEFAULT_RANDOM_POLICY,
    submitted_policy: str = DEFAULT_SUBMITTED_POLICY,
    ts2vec_policy: str = DEFAULT_TS2VEC_POLICY,
    quality_floor_policy: str = DEFAULT_QUALITY_FLOOR_POLICY,
) -> dict[str, Any]:
    root = Path(run_root)
    reports = _load_seed_reports(root)
    rows = [row for report in reports for row in report["rows"]]
    if not rows:
        raise ValueError(f"No downstream forecast rows found under {root}")

    final_budget = max(int(row["budget_k"]) for row in rows)
    final_rows = [row for row in rows if int(row["budget_k"]) == final_budget]
    policies = sorted({str(row["policy_id"]) for row in rows})
    coverage_reports = _load_coverage_reports(root)
    proof_summaries = _load_proof_summaries(root)
    pairwise = _pairwise_final_deltas(final_rows, policies)
    policy_final_summary = _policy_final_summary(final_rows)

    return {
        "input": {
            "run_root": str(root),
            "seed_report_count": len(reports),
            "seed_names": [report["seed_name"] for report in reports],
            "row_count": len(rows),
            "policies": policies,
            "budgets": sorted({int(row["budget_k"]) for row in rows}),
            "final_budget": int(final_budget),
        },
        "result_card": _result_card(
            policy_final_summary=policy_final_summary,
            pairwise=pairwise,
            killed_policy=killed_policy,
            champion_policy=champion_policy,
            random_policy=random_policy,
            submitted_policy=submitted_policy,
            ts2vec_policy=ts2vec_policy,
            quality_floor_policy=quality_floor_policy,
        ),
        "policy_budget_summary": _policy_budget_summary(rows),
        "policy_final_summary": policy_final_summary,
        "pairwise_final_deltas": pairwise,
        "probcover_episode_deltas": _focused_episode_deltas(
            final_rows,
            killed_policy=killed_policy,
            comparison_policies=(champion_policy, random_policy, submitted_policy, ts2vec_policy),
        ),
        "selected_set_overlap": _selected_set_overlap(final_rows, policies),
        "selection_diagnostics": _selection_diagnostics(coverage_reports, final_budget)
        or _selection_diagnostics_from_forecast_rows(final_rows),
        "hygiene": _hygiene_summary(proof_summaries),
    }


def write_downstream_forecast_autopsy_reports(
    run_root: str | Path,
    *,
    output_json: str | Path,
    output_markdown: str | Path,
    killed_policy: str = DEFAULT_KILLED_POLICY,
    champion_policy: str = DEFAULT_CHAMPION_POLICY,
    random_policy: str = DEFAULT_RANDOM_POLICY,
    submitted_policy: str = DEFAULT_SUBMITTED_POLICY,
    ts2vec_policy: str = DEFAULT_TS2VEC_POLICY,
    quality_floor_policy: str = DEFAULT_QUALITY_FLOOR_POLICY,
) -> dict[str, str]:
    autopsy = build_downstream_forecast_autopsy(
        run_root,
        killed_policy=killed_policy,
        champion_policy=champion_policy,
        random_policy=random_policy,
        submitted_policy=submitted_policy,
        ts2vec_policy=ts2vec_policy,
        quality_floor_policy=quality_floor_policy,
    )
    json_path = Path(output_json)
    markdown_path = Path(output_markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(autopsy, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_downstream_forecast_autopsy_markdown(autopsy), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def render_downstream_forecast_autopsy_markdown(autopsy: Mapping[str, Any]) -> str:
    card = autopsy.get("result_card", {})
    lines = [
        "# Downstream Forecast Policy Decision Card",
        "",
        "## Decision",
        "",
        f"- decision: `{card.get('decision', '')}`",
        f"- killed policy: `{card.get('killed_policy', '')}`",
        f"- current champion: `{card.get('champion_policy', '')}`",
        f"- final budget: `K={autopsy.get('input', {}).get('final_budget')}`",
        f"- seed reports: `{autopsy.get('input', {}).get('seed_report_count')}`",
        f"- downstream task: `raw_imu_autoregressive_ridge_forecast`",
        "",
    ]
    lines.extend(f"- {item}" for item in card.get("reads", []))
    lines.extend(
        [
            "",
            "## Final K Policy Ranking",
            "",
            "| rank | policy | rows | mean after MSE | mean relative MSE reduction | best/tie wins | decision |",
            "|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    decisions = card.get("policy_decisions", {})
    for row in autopsy.get("policy_final_summary", []):
        lines.append(
            "| {rank} | `{policy}` | {rows} | {mse} | {rel} | {wins} | {decision} |".format(
                rank=row.get("rank", ""),
                policy=row.get("policy_id", ""),
                rows=int(row.get("row_count", 0)),
                mse=_fmt(row.get("mean_after_mse")),
                rel=_fmt(row.get("mean_relative_mse_reduction")),
                wins=int(row.get("final_episode_wins", 0)),
                decision=decisions.get(row.get("policy_id", ""), ""),
            )
        )
    lines.extend(
        [
            "",
            "## Pairwise Final K Deltas",
            "",
            "Positive advantage means policy A has lower MSE than policy B.",
            "",
            "| policy A | policy B | units | mean A-B MSE | A advantage | A wins |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in autopsy.get("pairwise_final_deltas", []):
        if row.get("highlight"):
            lines.append(
                "| `{a}` | `{b}` | {units} | {delta} | {advantage} | {wins} / {units} |".format(
                    a=row.get("policy_a", ""),
                    b=row.get("policy_b", ""),
                    units=int(row.get("paired_unit_count", 0)),
                    delta=_fmt(row.get("mean_after_mse_delta_a_minus_b")),
                    advantage=_fmt(row.get("mean_after_mse_advantage_a_over_b")),
                    wins=int(row.get("policy_a_win_count", 0)),
                )
            )
    lines.extend(
        [
            "",
            "## Acquisition Curves",
            "",
            "| policy | K | rows | mean after MSE | mean relative MSE reduction |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in autopsy.get("policy_budget_summary", []):
        lines.append(
            "| `{policy}` | {budget} | {rows} | {mse} | {rel} |".format(
                policy=row.get("policy_id", ""),
                budget=int(row.get("budget_k", 0)),
                rows=int(row.get("row_count", 0)),
                mse=_fmt(row.get("mean_after_mse")),
                rel=_fmt(row.get("mean_relative_mse_reduction")),
            )
        )
    probcover_deltas = [row for row in autopsy.get("probcover_episode_deltas", []) if int(row.get("unit_count", 0)) > 0]
    if probcover_deltas:
        lines.extend(
            [
                "",
                "## ProbCover Episode Deltas",
                "",
                "Positive advantage means ProbCover has lower MSE than the comparison.",
                "",
                "| comparison | mean advantage | median advantage | wins | worst unit | worst advantage |",
                "|---|---:|---:|---:|---|---:|",
            ]
        )
        for row in probcover_deltas:
            lines.append(
                "| vs `{comparison}` | {mean_adv} | {median_adv} | {wins} / {units} | `{worst}` | {worst_adv} |".format(
                    comparison=row.get("comparison_policy", ""),
                    mean_adv=_fmt(row.get("mean_after_mse_advantage")),
                    median_adv=_fmt(row.get("median_after_mse_advantage")),
                    wins=int(row.get("win_count", 0)),
                    units=int(row.get("unit_count", 0)),
                    worst=row.get("worst_unit_id", ""),
                    worst_adv=_fmt(row.get("worst_after_mse_advantage")),
                )
            )
    else:
        lines.extend(
            [
                "",
                "## Archived ProbCover",
                "",
                "- `support_gap_window_probcover_v1` is not present in this run. That is intentional for survivor-only confirmation.",
            ]
        )
    lines.extend(
        [
            "",
            "## Selected Set Overlap",
            "",
            "| policy A | policy B | units | mean Jaccard | exact match rate |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in autopsy.get("selected_set_overlap", []):
        if row.get("highlight"):
            lines.append(
                "| `{a}` | `{b}` | {units} | {jaccard} | {exact} |".format(
                    a=row.get("policy_a", ""),
                    b=row.get("policy_b", ""),
                    units=int(row.get("paired_unit_count", 0)),
                    jaccard=_fmt(row.get("mean_jaccard")),
                    exact=_fmt(row.get("exact_match_rate")),
                )
            )
    lines.extend(
        [
            "",
            "## Selection Diagnostics",
            "",
            "| policy | selected rows | mean quality | mean artifact | valid rate | duplicate-source batch rate | largest source frac |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in autopsy.get("selection_diagnostics", []):
        lines.append(
            "| `{policy}` | {rows} | {quality} | {artifact} | {valid} | {duplicate} | {largest} |".format(
                policy=row.get("policy_id", ""),
                rows=int(row.get("selected_row_count", 0)),
                quality=_fmt(row.get("mean_quality_score")),
                artifact=_fmt(row.get("mean_artifact_score")),
                valid=_fmt(row.get("valid_rate")),
                duplicate=_fmt(row.get("duplicate_source_batch_rate")),
                largest=_fmt(row.get("mean_largest_source_group_fraction")),
            )
        )
    hygiene = autopsy.get("hygiene", {})
    lines.extend(
        [
            "",
            "## Hygiene",
            "",
            f"- leakage all ok: `{hygiene.get('leakage_all_ok')}`",
            f"- max selected invalid rate: `{_fmt(hygiene.get('selected_invalid_rate_max'))}`",
            f"- total out-of-pool selections: `{hygiene.get('selected_out_of_pool_count_total')}`",
            f"- total target leaks: `{hygiene.get('selected_target_leak_count_total')}`",
            f"- total duplicate selected clips: `{hygiene.get('selected_duplicate_clip_count_total')}`",
            "",
            "## Frozen Next Step",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in card.get("next_steps", []))
    lines.append("")
    return "\n".join(lines)


def _load_seed_reports(root: Path) -> list[dict[str, Any]]:
    paths = sorted(root.glob("seed_*/downstream_forecast_task_report.json"))
    if not paths:
        paths = sorted(root.glob("**/seed_*/downstream_forecast_task_report.json"))
    if not paths:
        raise FileNotFoundError(f"No seed_*/downstream_forecast_task_report.json files found under {root}")
    reports = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed_name = path.parent.name
        for row in payload.get("rows", []):
            row["_seed_name"] = seed_name
            row["_unit_id"] = _unit_id(seed_name, row)
        payload["seed_name"] = seed_name
        reports.append(payload)
    return reports


def _load_coverage_reports(root: Path) -> list[dict[str, Any]]:
    reports = []
    for path in sorted(root.glob("**/seed_*/blind_target_coverage_benchmark_report.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed_name = path.parent.name
        for row in payload.get("selected_rows", []):
            row["_seed_name"] = seed_name
            row["_unit_id"] = _unit_id(seed_name, row)
        reports.append(payload)
    return reports


def _load_proof_summaries(root: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("**/seed_*/coverage_proof_summary.json"))]


def _policy_budget_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["policy_id"]), int(row["budget_k"]))].append(row)
    return [
        {"policy_id": policy, "budget_k": budget, **_forecast_means(group_rows)}
        for (policy, budget), group_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1]))
    ]


def _policy_final_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy_id"])].append(row)
    wins = _final_episode_wins(rows)
    summaries = [
        {"policy_id": policy, **_forecast_means(group_rows), "final_episode_wins": wins.get(policy, 0)}
        for policy, group_rows in grouped.items()
    ]
    summaries.sort(key=lambda row: (float(row["mean_after_mse"]), str(row["policy_id"])))
    for rank, row in enumerate(summaries, start=1):
        row["rank"] = rank
    return summaries


def _forecast_means(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": int(len(rows)),
        "mean_after_mse": _mean(rows, "after_mse"),
        "median_after_mse": _median(rows, "after_mse"),
        "mean_baseline_mse": _mean(rows, "baseline_mse"),
        "mean_absolute_mse_reduction": _mean(rows, "absolute_mse_reduction"),
        "mean_relative_mse_reduction": _mean(rows, "relative_mse_reduction"),
    }


def _final_episode_wins(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    by_unit: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_unit[str(row["_unit_id"])].append(row)
    wins: Counter[str] = Counter()
    for unit_rows in by_unit.values():
        best = min(float(row["after_mse"]) for row in unit_rows)
        for row in unit_rows:
            if abs(float(row["after_mse"]) - best) <= 1.0e-12:
                wins[str(row["policy_id"])] += 1
    return dict(wins)


def _pairwise_final_deltas(rows: Sequence[Mapping[str, Any]], policies: Sequence[str]) -> list[dict[str, Any]]:
    by_policy_unit = _rows_by_policy_unit(rows)
    highlights = _highlight_pairs()
    records = []
    for policy_a, policy_b in combinations(sorted(policies), 2):
        units = sorted(set(by_policy_unit[policy_a]) & set(by_policy_unit[policy_b]))
        deltas = [
            float(by_policy_unit[policy_a][unit]["after_mse"]) - float(by_policy_unit[policy_b][unit]["after_mse"])
            for unit in units
        ]
        records.append(
            {
                "policy_a": policy_a,
                "policy_b": policy_b,
                "paired_unit_count": len(deltas),
                "mean_after_mse_delta_a_minus_b": _mean_values(deltas),
                "median_after_mse_delta_a_minus_b": _median_values(deltas),
                "mean_after_mse_advantage_a_over_b": -_mean_values(deltas),
                "policy_a_win_count": sum(1 for delta in deltas if delta < -1.0e-12),
                "policy_b_win_count": sum(1 for delta in deltas if delta > 1.0e-12),
                "tie_count": sum(1 for delta in deltas if abs(delta) <= 1.0e-12),
                "highlight": tuple(sorted((policy_a, policy_b))) in highlights,
            }
        )
    return records


def _focused_episode_deltas(
    rows: Sequence[Mapping[str, Any]],
    *,
    killed_policy: str,
    comparison_policies: Sequence[str],
) -> list[dict[str, Any]]:
    by_policy_unit = _rows_by_policy_unit(rows)
    records = []
    for comparison in comparison_policies:
        units = sorted(set(by_policy_unit[killed_policy]) & set(by_policy_unit[comparison]))
        advantages = [
            float(by_policy_unit[comparison][unit]["after_mse"]) - float(by_policy_unit[killed_policy][unit]["after_mse"])
            for unit in units
        ]
        worst_index = min(range(len(advantages)), key=lambda index: advantages[index]) if advantages else None
        records.append(
            {
                "killed_policy": killed_policy,
                "comparison_policy": comparison,
                "unit_count": len(advantages),
                "mean_after_mse_advantage": _mean_values(advantages),
                "median_after_mse_advantage": _median_values(advantages),
                "win_count": sum(1 for advantage in advantages if advantage > 1.0e-12),
                "loss_count": sum(1 for advantage in advantages if advantage < -1.0e-12),
                "worst_unit_id": units[worst_index] if worst_index is not None else "",
                "worst_after_mse_advantage": advantages[worst_index] if worst_index is not None else None,
            }
        )
    return records


def _selected_set_overlap(rows: Sequence[Mapping[str, Any]], policies: Sequence[str]) -> list[dict[str, Any]]:
    by_policy_unit = _rows_by_policy_unit(rows)
    highlights = _highlight_pairs()
    records = []
    for policy_a, policy_b in combinations(sorted(policies), 2):
        units = sorted(set(by_policy_unit[policy_a]) & set(by_policy_unit[policy_b]))
        jaccards = []
        exact = 0
        for unit in units:
            a_ids = set(str(item) for item in by_policy_unit[policy_a][unit].get("selected_ids", []))
            b_ids = set(str(item) for item in by_policy_unit[policy_b][unit].get("selected_ids", []))
            union = a_ids | b_ids
            jaccards.append(float(len(a_ids & b_ids) / len(union)) if union else 1.0)
            exact += int(a_ids == b_ids)
        records.append(
            {
                "policy_a": policy_a,
                "policy_b": policy_b,
                "paired_unit_count": len(units),
                "mean_jaccard": _mean_values(jaccards),
                "exact_match_rate": float(exact / len(units)) if units else 0.0,
                "highlight": tuple(sorted((policy_a, policy_b))) in highlights,
            }
        )
    return records


def _selection_diagnostics(coverage_reports: Sequence[Mapping[str, Any]], final_budget: int) -> list[dict[str, Any]]:
    rows = [
        row
        for report in coverage_reports
        for row in report.get("selected_rows", [])
        if int(row.get("budget_k", -1)) == int(final_budget)
    ]
    return _selection_diagnostics_from_rows(rows) if rows else []


def _selection_diagnostics_from_forecast_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected_rows = []
    for row in rows:
        for rank_index, sample_id in enumerate(row.get("selected_ids", []), start=1):
            selected_rows.append(
                {
                    "_unit_id": row["_unit_id"],
                    "policy_id": row["policy_id"],
                    "sample_id": sample_id,
                    "source_group_id": str(sample_id).split("_clip", 1)[0],
                    "rank_index": rank_index,
                    "quality_score": None,
                    "artifact_score": None,
                    "valid": True,
                    "passed_artifact_gate": True,
                }
            )
    return _selection_diagnostics_from_rows(selected_rows)


def _selection_diagnostics_from_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_policy: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_policy[str(row["policy_id"])].append(row)
    diagnostics = []
    for policy, policy_rows in sorted(by_policy.items()):
        batch_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in policy_rows:
            batch_rows[str(row["_unit_id"])].append(row)
        duplicate_batch_count = 0
        largest_fracs = []
        unique_source_fracs = []
        for batch in batch_rows.values():
            sources = [str(row.get("source_group_id", "")) for row in batch]
            counts = Counter(sources)
            duplicate_batch_count += int(any(count > 1 for count in counts.values()))
            largest_fracs.append(max(counts.values()) / len(sources) if sources else 0.0)
            unique_source_fracs.append(len(counts) / len(sources) if sources else 0.0)
        sample_ids = [str(row.get("sample_id", "")) for row in policy_rows]
        diagnostics.append(
            {
                "policy_id": policy,
                "selected_row_count": len(policy_rows),
                "unique_selected_clip_count": len(set(sample_ids)),
                "unique_selected_clip_fraction": float(len(set(sample_ids)) / len(sample_ids)) if sample_ids else 0.0,
                "mean_quality_score": _mean_optional(policy_rows, "quality_score"),
                "mean_artifact_score": _mean_optional(policy_rows, "artifact_score"),
                "valid_rate": _mean_bool(policy_rows, "valid"),
                "artifact_gate_pass_rate": _mean_bool(policy_rows, "passed_artifact_gate"),
                "duplicate_source_batch_rate": float(duplicate_batch_count / len(batch_rows)) if batch_rows else 0.0,
                "mean_largest_source_group_fraction": _mean_values(largest_fracs),
                "mean_unique_source_fraction": _mean_values(unique_source_fracs),
            }
        )
    return diagnostics


def _hygiene_summary(proof_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not proof_summaries:
        return {}
    return {
        "proof_summary_count": len(proof_summaries),
        "leakage_all_ok": all(bool(summary.get("leakage_ok")) for summary in proof_summaries),
        "selected_invalid_rate_max": max(float(summary.get("selected_invalid_rate_max", 0.0)) for summary in proof_summaries),
        "selected_out_of_pool_count_total": sum(int(summary.get("selected_out_of_pool_count_total", 0)) for summary in proof_summaries),
        "selected_target_leak_count_total": sum(int(summary.get("selected_target_leak_count_total", 0)) for summary in proof_summaries),
        "selected_duplicate_clip_count_total": sum(
            int(summary.get("selected_duplicate_clip_count_total", 0)) for summary in proof_summaries
        ),
    }


def _result_card(
    *,
    policy_final_summary: Sequence[Mapping[str, Any]],
    pairwise: Sequence[Mapping[str, Any]],
    killed_policy: str,
    champion_policy: str,
    random_policy: str,
    submitted_policy: str,
    ts2vec_policy: str,
    quality_floor_policy: str,
) -> dict[str, Any]:
    by_policy = {str(row["policy_id"]): row for row in policy_final_summary}
    pairwise_lookup = _pairwise_lookup(pairwise)
    best_policy = str(policy_final_summary[0]["policy_id"]) if policy_final_summary else ""
    if killed_policy not in by_policy:
        decision = "survivor_confirmation_window_kcenter" if best_policy == champion_policy else "survivor_confirmation_reversal"
        reads = [
            f"{killed_policy} is intentionally excluded; this run is not a ProbCover appeal.",
            f"{champion_policy} is the locked candidate champion; compare it only to fixed survivors and baselines.",
            f"{submitted_policy} remains a submitted-system comparator.",
            f"{ts2vec_policy} remains a representation ablation / feature source.",
            "Do not add or tune policies based on this confirmation run.",
        ]
    else:
        killed_failed = (
            _advantage_for(pairwise_lookup.get((killed_policy, champion_policy), {}), killed_policy) <= 0.0
            or _advantage_for(pairwise_lookup.get((killed_policy, random_policy), {}), killed_policy) <= 0.0
        )
        decision = "kill_probcover_promote_window_kcenter" if killed_failed else "probcover_not_killed_by_rule"
        reads = [
            f"{killed_policy} failed the pre-registered K=4 downstream MSE rule; do not tune it further.",
            f"{champion_policy} is the current low-budget champion on this downstream forecast canary.",
            f"{submitted_policy} remains a defensible submitted-system comparator, not proof of downstream retraining optimality.",
            f"{ts2vec_policy} remains useful as a representation ablation / feature source, not the lead active-learning strategy.",
            "The hidden downstream target has now been used for decisions; stop designing new selectors against it.",
        ]
    policy_decisions = {
        champion_policy: "current champion" if best_policy == champion_policy else "survivor, not current best",
        submitted_policy: "defensible submitted comparator",
        ts2vec_policy: "feature source / ablation",
        random_policy: "required baseline",
        quality_floor_policy: "negative control floor",
        killed_policy: "failed; archive",
    }
    return {
        "decision": decision,
        "killed_policy": killed_policy,
        "champion_policy": champion_policy,
        "best_policy": best_policy,
        "champion_mean_after_mse": by_policy.get(champion_policy, {}).get("mean_after_mse"),
        "policy_decisions": policy_decisions,
        "reads": reads,
        "next_steps": [
            "Write/keep this result card as the frozen ProbCover decision.",
            "Run only diagnostics that explain the loss; do not alter ProbCover thresholds or graph construction.",
            "If running more compute, use a locked survivor-only confirmation with no new policies and no ProbCover rescue.",
            "Hold big neural downstream training until the survivor-only selector conclusion is stable.",
        ],
    }


def _highlight_pairs() -> set[tuple[str, str]]:
    return {
        tuple(sorted(pair))
        for pair in (
            (DEFAULT_KILLED_POLICY, DEFAULT_CHAMPION_POLICY),
            (DEFAULT_KILLED_POLICY, DEFAULT_RANDOM_POLICY),
            (DEFAULT_KILLED_POLICY, DEFAULT_SUBMITTED_POLICY),
            (DEFAULT_KILLED_POLICY, DEFAULT_TS2VEC_POLICY),
            (DEFAULT_KILLED_POLICY, DEFAULT_QUALITY_FLOOR_POLICY),
            (DEFAULT_CHAMPION_POLICY, DEFAULT_RANDOM_POLICY),
            (DEFAULT_CHAMPION_POLICY, DEFAULT_SUBMITTED_POLICY),
            (DEFAULT_CHAMPION_POLICY, DEFAULT_TS2VEC_POLICY),
        )
    }


def _pairwise_lookup(pairwise: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    lookup = {}
    for row in pairwise:
        a = str(row["policy_a"])
        b = str(row["policy_b"])
        lookup[(a, b)] = row
        lookup[(b, a)] = row
    return lookup


def _advantage_for(row: Mapping[str, Any], policy: str) -> float:
    if not row:
        return 0.0
    if str(row.get("policy_a")) == policy:
        return float(row.get("mean_after_mse_advantage_a_over_b", 0.0))
    return -float(row.get("mean_after_mse_advantage_a_over_b", 0.0))


def _rows_by_policy_unit(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Mapping[str, Any]]]:
    by_policy_unit: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_policy_unit[str(row["policy_id"])][str(row["_unit_id"])] = row
    return by_policy_unit


def _unit_id(seed_name: str, row: Mapping[str, Any]) -> str:
    return f"{seed_name}:{row.get('episode_id')}:{row.get('fold_id')}"


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return _mean_values([float(row[key]) for row in rows])


def _median(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return _median_values([float(row[key]) for row in rows])


def _mean_optional(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return _mean_values(values) if values else None


def _mean_bool(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return _mean_values([1.0 if bool(row.get(key)) else 0.0 for row in rows])


def _mean_values(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _median_values(values: Sequence[float]) -> float:
    return float(median(values)) if values else 0.0


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.9f}"
    return str(value)
