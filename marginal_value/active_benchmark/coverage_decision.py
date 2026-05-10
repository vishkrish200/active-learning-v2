from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_BASELINE_POLICY = "quality_stratified_random_v1"
ORACLE_POLICY = "oracle_greedy_eval_view_v1"
EXACT_COVERAGE_ORACLE_POLICY = "oracle_exact_coverage_v1"
TARGET_FAMILY_ORACLE_POLICY = "oracle_greedy_target_family_v1"
QUALITY_STRATIFIED_RANDOM_REPLAY_PREFIX = "quality_stratified_random_replay_"
DOWNSTREAM_BRIDGE_METRICS = (
    "target_family_discovery_rate",
    "after_known_target_fraction",
    "balanced_accuracy_gain",
    "nll_reduction",
)
EXPANDED_CPU_UNLOCK_CRITERIA = {
    "minimum_independent_episodes_per_pool": 80,
    "minimum_candidate_clips_max": 72,
    "random_delta_mean_min": 0.04,
    "random_delta_ci95_low_min": 0.025,
    "random_delta_win_fraction_min": 0.70,
    "replay_null_pvalue_max": 0.01,
    "window_kcenter_delta_mean_min": 0.015,
    "window_kcenter_delta_ci95_low_min": 0.0,
    "submitted_full_replay_delta_ci95_low_min": 0.0,
}
DOWNSTREAM_CANARY_CRITERIA = {
    "minimum_independent_episodes": 40,
    "balanced_accuracy_delta_mean_min": 0.02,
    "balanced_accuracy_delta_ci95_low_min": 0.0,
    "balanced_accuracy_win_fraction_min": 0.60,
    "target_family_discovery_delta_mean_min": 0.0,
}


def build_coverage_decision_report(
    reports: Sequence[Mapping[str, Any]],
    *,
    report_names: Sequence[str] | None = None,
    baseline_policy: str = DEFAULT_BASELINE_POLICY,
    oracle_policy: str = ORACLE_POLICY,
    downstream_reports: Sequence[Mapping[str, Any]] | None = None,
    downstream_report_names: Sequence[str] | None = None,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260508,
    eps: float = 1.0e-12,
) -> dict[str, Any]:
    names = _report_names(reports, report_names)
    downstream_reports = tuple(downstream_reports or ())
    if downstream_reports and downstream_report_names is None and len(downstream_reports) == len(names):
        downstream_names: Sequence[str] | None = names
    else:
        downstream_names = downstream_report_names
    records = _primary_view_gain_records(reports, names)
    final_budget = max((int(budget) for report in reports for budget in report.get("budgets", [])), default=0)
    final_records = [record for record in records if int(record["budget_k"]) == final_budget]
    final_by_key = _records_by_policy_unit(final_records)
    policy_final = _policy_final_summary(
        final_records,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed + 431,
    )
    budget_summary = _policy_budget_summary(records)
    pairwise = _pairwise_summaries(
        final_by_key,
        baseline_policy=baseline_policy,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    direct_pairwise = _direct_pairwise_summaries(
        final_by_key,
        policy_final,
        baseline_policy=baseline_policy,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed + 2711,
    )
    oracle_capture = _oracle_capture_summaries(
        final_by_key,
        baseline_policy=baseline_policy,
        oracle_policy=oracle_policy,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed + 9173,
        eps=eps,
    )
    oracle_stability = _oracle_stability_diagnostics(
        final_by_key,
        policy_final,
        baseline_policy=baseline_policy,
        oracle_policy=oracle_policy,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed + 11933,
        eps=eps,
    )
    oracle_diagnostics = _oracle_diagnostics(
        final_by_key,
        policy_final,
        configured_oracle_policy=oracle_policy,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed + 14149,
        eps=eps,
    )
    random_replay_diagnostics = _random_replay_diagnostics(
        final_by_key,
        policy_final,
        baseline_policy=baseline_policy,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed + 17491,
    )
    acquisition_stability = _acquisition_stability_diagnostics(
        final_by_key,
        policy_final,
        baseline_policy=baseline_policy,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed + 19301,
    )
    selected_set_audit = _selected_set_audit(reports, names, final_budget)
    downstream_bridge = _downstream_bridge_proxy_summary(
        downstream_reports,
        report_names=downstream_names,
        baseline_policy=baseline_policy,
        oracle_policy=oracle_policy,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed + 21119,
    )
    gates = _decision_gates(
        final_records=final_records,
        policy_final=policy_final,
        pairwise=pairwise,
        direct_pairwise=direct_pairwise,
        oracle_stability=oracle_stability,
        random_replay=random_replay_diagnostics,
        acquisition_stability=acquisition_stability,
        baseline_policy=baseline_policy,
        oracle_policy=oracle_policy,
    )
    decision = _decision(gates)
    return {
        "input": {
            "report_count": len(reports),
            "report_names": list(names),
            "baseline_policy": baseline_policy,
            "oracle_policy": oracle_policy,
            "bootstrap_replicates": int(bootstrap_replicates),
            "bootstrap_seed": int(bootstrap_seed),
        },
        "coverage_units": {
            "independent_episode_count": len({str(record["unit_id"]) for record in final_records}),
            "final_budget": final_budget,
            "gain_record_count": len(records),
            "final_gain_record_count": len(final_records),
            "candidate_pool": _candidate_pool_summary(reports, names),
        },
        "policy_final_summary": policy_final,
        "policy_budget_summary": budget_summary,
        "pairwise_vs_baseline": pairwise,
        "direct_pairwise_controls": direct_pairwise,
        "oracle_capture_vs_baseline": oracle_capture,
        "oracle_stability_diagnostics": oracle_stability,
        "oracle_diagnostics": oracle_diagnostics,
        "random_replay_diagnostics": random_replay_diagnostics,
        "acquisition_stability_diagnostics": acquisition_stability,
        "selected_set_audit": selected_set_audit,
        "downstream_bridge_proxy": downstream_bridge,
        "training_unlock_criteria": _training_unlock_criteria(
            baseline_policy=baseline_policy,
            focal_policy="ts2vec_kcenter_v1",
        ),
        "gates": gates,
        "decision": _decision_with_downstream_bridge(decision, downstream_bridge),
    }


def write_coverage_decision_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    output_json: str | Path,
    output_markdown: str | Path,
    report_names: Sequence[str] | None = None,
    baseline_policy: str = DEFAULT_BASELINE_POLICY,
    oracle_policy: str = ORACLE_POLICY,
    downstream_reports: Sequence[Mapping[str, Any]] | None = None,
    downstream_report_names: Sequence[str] | None = None,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260508,
) -> dict[str, str]:
    report = build_coverage_decision_report(
        reports,
        report_names=report_names,
        baseline_policy=baseline_policy,
        oracle_policy=oracle_policy,
        downstream_reports=downstream_reports,
        downstream_report_names=downstream_report_names,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    json_path = Path(output_json)
    markdown_path = Path(output_markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_coverage_decision_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def render_coverage_decision_markdown(report: Mapping[str, Any]) -> str:
    decision = report.get("decision", {})
    units = report.get("coverage_units", {})
    lines = [
        "# Coverage Benchmark Decision Report",
        "",
        "## Decision",
        "",
        f"- downstream training: `{decision.get('downstream_training', '')}`",
        f"- large training: `{decision.get('large_training', '')}`",
        f"- bounded downstream canary: `{decision.get('bounded_downstream_canary', '')}`",
        f"- tiny frozen supervised probe: `{decision.get('tiny_frozen_supervised_probe', '')}`",
        f"- TS2Vec retraining: `{decision.get('ts2vec_retraining', '')}`",
        f"- next CPU gate: `{decision.get('next_cpu_gate', '')}`",
        f"- read: {decision.get('read', '')}",
        "",
        "## Run Shape",
        "",
        f"- reports: `{report.get('input', {}).get('report_count', 0)}`",
        f"- independent final episodes: `{units.get('independent_episode_count', 0)}`",
        f"- final budget: `{units.get('final_budget', 0)}`",
        f"- gain records: `{units.get('gain_record_count', 0)}`",
        f"- candidate pool size: `{_pool_range(units.get('candidate_pool', {}))}`",
        "",
        "## Final Leaderboard",
        "",
        "| rank | policy | mean final gain | median | CI95 low | CI95 high | episodes | target-used |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    final_summary = report.get("policy_final_summary", {})
    for rank, (policy, row) in enumerate(_ranked_items(final_summary, "mean_final_gain"), start=1):
        lines.append(
            "| {rank} | `{policy}` | {mean} | {median} | {low} | {high} | {episodes} | {target} |".format(
                rank=rank,
                policy=policy,
                mean=_fmt(row.get("mean_final_gain")),
                median=_fmt(row.get("median_final_gain")),
                low=_fmt(row.get("bootstrap_ci95_low")),
                high=_fmt(row.get("bootstrap_ci95_high")),
                episodes=int(row.get("episode_count", 0)),
                target="Y" if row.get("uses_target_for_selection") else "N",
            )
        )

    unlock = report.get("training_unlock_criteria", {}).get("expanded_cpu_validation", {})
    if isinstance(unlock, Mapping) and unlock:
        lines.extend(
            [
                "",
                "## Expanded Training Unlock Criteria",
                "",
                f"- scope: {unlock.get('scope', '')}",
                f"- focal policy: `{unlock.get('focal_policy', '')}`",
                f"- baseline policy: `{unlock.get('baseline_policy', '')}`",
                f"- per-pool episodes >= `{int(unlock.get('minimum_independent_episodes_per_pool', 0))}`",
                f"- candidate clips max >= `{int(unlock.get('minimum_candidate_clips_max', 0))}`",
                f"- mean delta vs baseline >= `{_fmt(unlock.get('random_delta_mean_min'))}`",
                f"- CI95 low vs baseline >= `{_fmt(unlock.get('random_delta_ci95_low_min'))}`",
                f"- win fraction vs baseline >= `{_fmt(unlock.get('random_delta_win_fraction_min'))}`",
                f"- replay-null p-value <= `{_fmt(unlock.get('replay_null_pvalue_max'))}`",
                f"- window-kcenter mean delta >= `{_fmt(unlock.get('window_kcenter_delta_mean_min'))}` with CI95 low > `{_fmt(unlock.get('window_kcenter_delta_ci95_low_min'))}`",
                f"- submitted-full replay CI95 low > `{_fmt(unlock.get('submitted_full_replay_delta_ci95_low_min'))}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Paired Delta Vs Baseline",
            "",
            "| policy | baseline | mean delta | median delta | CI95 low | CI95 high | win frac | paired episodes |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for policy, row in _ranked_items(report.get("pairwise_vs_baseline", {}), "mean_delta"):
        lines.append(
            "| `{policy}` | `{baseline}` | {mean} | {median} | {low} | {high} | {win} | {episodes} |".format(
                policy=policy,
                baseline=row.get("baseline_policy", ""),
                mean=_fmt(row.get("mean_delta")),
                median=_fmt(row.get("median_delta")),
                low=_fmt(row.get("bootstrap_ci95_low")),
                high=_fmt(row.get("bootstrap_ci95_high")),
                win=_fmt(row.get("win_fraction")),
                episodes=int(row.get("paired_episode_count", 0)),
            )
        )

    direct_pairwise = report.get("direct_pairwise_controls", {})
    if isinstance(direct_pairwise, Mapping) and direct_pairwise:
        lines.extend(
            [
                "",
                "## Direct Paired Control Comparisons",
                "",
                "| policy | comparator | mean delta | median delta | CI95 low | CI95 high | win frac | loss frac | tie frac | paired episodes |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _key, row in _ranked_items(direct_pairwise, "mean_delta"):
            lines.append(
                "| `{policy}` | `{baseline}` | {mean} | {median} | {low} | {high} | {win} | {loss} | {tie} | {episodes} |".format(
                    policy=row.get("policy", ""),
                    baseline=row.get("comparator_policy", ""),
                    mean=_fmt(row.get("mean_delta")),
                    median=_fmt(row.get("median_delta")),
                    low=_fmt(row.get("bootstrap_ci95_low")),
                    high=_fmt(row.get("bootstrap_ci95_high")),
                    win=_fmt(row.get("win_fraction")),
                    loss=_fmt(row.get("loss_fraction")),
                    tie=_fmt(row.get("tie_fraction")),
                    episodes=int(row.get("paired_episode_count", 0)),
                )
            )

    if report.get("oracle_capture_vs_baseline"):
        lines.extend(
            [
                "",
                "## Deprecated Raw Oracle Capture",
                "",
                "Raw oracle capture is kept only as a debug metric. It is a ratio over episode-level oracle headroom and is unstable when the denominator is small or negative.",
                "",
                "| policy | mean capture | median capture | CI95 low | CI95 high | episodes |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for policy, row in _ranked_items(report.get("oracle_capture_vs_baseline", {}), "mean_oracle_capture"):
            lines.append(
                "| `{policy}` | {mean} | {median} | {low} | {high} | {episodes} |".format(
                    policy=policy,
                    mean=_fmt(row.get("mean_oracle_capture")),
                    median=_fmt(row.get("median_oracle_capture")),
                    low=_fmt(row.get("bootstrap_ci95_low")),
                    high=_fmt(row.get("bootstrap_ci95_high")),
                    episodes=int(row.get("paired_episode_count", 0)),
                )
            )
    else:
        lines.extend(["", "## Deprecated Raw Oracle Capture", "", "- Missing: no oracle policy rows were present in these reports."])

    oracle_stability = report.get("oracle_stability_diagnostics", {})
    if isinstance(oracle_stability, Mapping) and oracle_stability:
        headroom = oracle_stability.get("oracle_headroom", {})
        lines.extend(
            [
                "",
                "## Stable Oracle Diagnostics",
                "",
                f"- oracle policy: `{oracle_stability.get('oracle_policy', '')}`",
                f"- baseline source: `{oracle_stability.get('baseline_source', '')}`",
                f"- headroom epsilon: `{_fmt(oracle_stability.get('headroom_epsilon'))}`",
                f"- oracle headroom mean: `{_fmt(headroom.get('mean_delta'))}`",
                f"- oracle headroom median: `{_fmt(headroom.get('median_delta'))}`",
                f"- oracle headroom CI95: `{_fmt(headroom.get('bootstrap_ci95_low'))}` to `{_fmt(headroom.get('bootstrap_ci95_high'))}`",
                f"- oracle headroom positive fraction: `{_fmt(headroom.get('positive_fraction'))}`",
                f"- oracle headroom near-zero fraction: `{_fmt(headroom.get('near_zero_fraction'))}`",
                "",
                "| policy | oracle gap mean | oracle gap CI95 low | positive-headroom capture median | bounded capture mean | negative capture frac | overshoot frac | positive-headroom episodes |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for policy, row in _ranked_items(oracle_stability.get("policy_metrics", {}), "bounded_capture_mean"):
            gap = row.get("oracle_gap", {})
            capture = row.get("positive_headroom_capture", {})
            lines.append(
                "| `{policy}` | {gap_mean} | {gap_low} | {cap_median} | {bounded} | {neg} | {over} | {episodes} |".format(
                    policy=policy,
                    gap_mean=_fmt(gap.get("mean_delta")),
                    gap_low=_fmt(gap.get("bootstrap_ci95_low")),
                    cap_median=_fmt(capture.get("median")),
                    bounded=_fmt(row.get("bounded_capture_mean")),
                    neg=_fmt(row.get("negative_capture_fraction")),
                    over=_fmt(row.get("overshoot_fraction")),
                    episodes=int(capture.get("paired_episode_count", 0)),
                )
            )

    oracle_diagnostics = report.get("oracle_diagnostics", {})
    if isinstance(oracle_diagnostics, Mapping):
        exact_gap = oracle_diagnostics.get("exact_coverage_vs_top_deployable", {})
        lines.extend(
            [
                "",
                "## Oracle Diagnostics",
                "",
                "- The exact coverage oracle maximizes the exact final-budget coverage objective; smaller budget prefixes are deterministic ordering diagnostics.",
                "- The target-family oracle is a discovery diagnostic, not an exact ceiling for reported coverage gain.",
                f"- configured oracle policy: `{oracle_diagnostics.get('configured_oracle_policy', '')}`",
                f"- exact coverage oracle: `{oracle_diagnostics.get('exact_coverage_policy', '')}`",
                f"- target-family discovery oracle: `{oracle_diagnostics.get('target_family_oracle_policy', '')}`",
                f"- top deployable policy: `{oracle_diagnostics.get('top_deployable_policy', '')}`",
            ]
        )
        if isinstance(exact_gap, Mapping) and exact_gap:
            lines.extend(
                [
                    f"- exact-vs-top-deployable status: `{exact_gap.get('status', '')}`",
                    f"- exact-vs-top-deployable mean gap: `{_fmt(exact_gap.get('mean_gap'))}`",
                    f"- exact-vs-top-deployable CI95: `{_fmt(exact_gap.get('bootstrap_ci95_low'))}` to `{_fmt(exact_gap.get('bootstrap_ci95_high'))}`",
                ]
            )

    random_replay = report.get("random_replay_diagnostics", {})
    if isinstance(random_replay, Mapping) and int(random_replay.get("replay_count", 0) or 0) > 0:
        top_delta = random_replay.get("top_deployable_vs_replay_mean", {})
        lines.extend(
            [
                "",
                "## Random Replay Controls",
                "",
                f"- replay policy count: `{int(random_replay.get('replay_count', 0))}`",
                f"- top deployable policy: `{random_replay.get('top_deployable_policy', '')}`",
                f"- top deployable replay-percentile by mean gain: `{_fmt(random_replay.get('top_deployable_replay_percentile_by_mean_gain'))}`",
                f"- top deployable +1 replay-null p-value: `{_fmt(random_replay.get('top_deployable_replay_null_pvalue_plus_one'))}`",
            ]
        )
        if isinstance(top_delta, Mapping) and top_delta:
            lines.extend(
                [
                    f"- top-vs-replay-mean delta: `{_fmt(top_delta.get('mean_delta'))}`",
                    f"- top-vs-replay-mean CI95: `{_fmt(top_delta.get('bootstrap_ci95_low'))}` to `{_fmt(top_delta.get('bootstrap_ci95_high'))}`",
                ]
            )

    stability = report.get("acquisition_stability_diagnostics", {})
    if isinstance(stability, Mapping) and stability:
        lines.extend(
            [
                "",
                "## Acquisition Stability",
                "",
                f"- focal policy: `{stability.get('policy', '')}`",
                f"- baseline source: `{stability.get('baseline_source', '')}`",
                f"- positive seed fraction: `{_fmt(stability.get('positive_seed_fraction'))}`",
                f"- positive fold fraction: `{_fmt(stability.get('positive_fold_fraction'))}`",
                f"- min leave-one-seed-out delta: `{_fmt(stability.get('min_leave_one_seed_out_delta'))}`",
                f"- min leave-one-fold-out delta: `{_fmt(stability.get('min_leave_one_fold_out_delta'))}`",
                f"- largest single episode contribution fraction: `{_fmt(stability.get('largest_single_episode_contribution_fraction'))}`",
            ]
        )

    selected_audit = report.get("selected_set_audit", {})
    if isinstance(selected_audit, Mapping) and selected_audit:
        lines.extend(
            [
                "",
                "## Selected Set Audit",
                "",
                "| policy | selected | invalid rate | duplicate frac | mean quality | max artifact | unique source groups | top source frac |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for policy, row in _ranked_items(selected_audit, "selected_count"):
            lines.append(
                "| `{policy}` | {selected} | {invalid} | {dup} | {quality} | {artifact} | {groups} | {top_group} |".format(
                    policy=policy,
                    selected=int(row.get("selected_count", 0)),
                    invalid=_fmt(row.get("invalid_rate")),
                    dup=_fmt(row.get("duplicate_clip_fraction")),
                    quality=_fmt(row.get("mean_quality_score")),
                    artifact=_fmt(row.get("max_artifact_score")),
                    groups=int(row.get("unique_source_group_count", 0)),
                    top_group=_fmt(row.get("top_source_group_fraction")),
                )
            )

    downstream_bridge = report.get("downstream_bridge_proxy", {})
    if isinstance(downstream_bridge, Mapping) and downstream_bridge.get("units", {}).get("independent_episode_count", 0):
        decision = downstream_bridge.get("decision", {})
        units = downstream_bridge.get("units", {})
        lines.extend(
            [
                "",
                "## Downstream Bridge Proxy",
                "",
                "This is a source-family pseudo-label frozen-feature proxy on predeclared heads, not real challenge-label downstream proof.",
                "",
                f"- downstream training: `{decision.get('downstream_training', '')}`",
                f"- bounded frozen canary: `{decision.get('bounded_frozen_canary', '')}`",
                f"- large training: `{decision.get('large_training', '')}`",
                f"- TS2Vec retraining: `{decision.get('ts2vec_retraining', '')}`",
                f"- read: {decision.get('read', '')}",
                f"- focal policy: `{decision.get('focal_policy', '')}`",
                f"- top deployable policy by balanced accuracy: `{decision.get('top_deployable_policy_by_balanced_accuracy', '')}`",
                f"- independent final episodes: `{units.get('independent_episode_count', 0)}`",
                f"- final budget: `{units.get('final_budget', 0)}`",
                "",
                "### Downstream Final Means",
                "",
                "| policy | episodes | discovery | known target frac | balanced accuracy gain | NLL reduction |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for policy, row in _ranked_downstream_policy_items(downstream_bridge.get("policy_final_summary", {})):
            metrics = row.get("metrics", {})
            lines.append(
                "| `{policy}` | {episodes} | {disc} | {known} | {bal} | {nll} |".format(
                    policy=policy,
                    episodes=int(row.get("episode_count", 0)),
                    disc=_fmt_metric(metrics, "target_family_discovery_rate", "mean"),
                    known=_fmt_metric(metrics, "after_known_target_fraction", "mean"),
                    bal=_fmt_metric(metrics, "balanced_accuracy_gain", "mean"),
                    nll=_fmt_metric(metrics, "nll_reduction", "mean"),
                )
            )
        lines.extend(
            [
                "",
                "### Paired Downstream Deltas Vs Baseline",
                "",
                "| policy | metric | mean delta | median delta | CI95 low | CI95 high | win frac | paired episodes |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for policy, metrics in sorted(downstream_bridge.get("pairwise_vs_baseline", {}).items()):
            if not isinstance(metrics, Mapping):
                continue
            for metric in DOWNSTREAM_BRIDGE_METRICS:
                row = metrics.get(metric, {})
                if not isinstance(row, Mapping):
                    continue
                lines.append(
                    "| `{policy}` | `{metric}` | {mean} | {median} | {low} | {high} | {win} | {episodes} |".format(
                        policy=policy,
                        metric=metric,
                        mean=_fmt(row.get("mean_delta")),
                        median=_fmt(row.get("median_delta")),
                        low=_fmt(row.get("bootstrap_ci95_low")),
                        high=_fmt(row.get("bootstrap_ci95_high")),
                        win=_fmt(row.get("win_fraction")),
                        episodes=int(row.get("paired_episode_count", 0)),
                    )
                )
        direct_downstream = downstream_bridge.get("direct_pairwise_controls", {})
        if isinstance(direct_downstream, Mapping) and direct_downstream:
            lines.extend(
                [
                    "",
                    "### Direct Downstream Controls",
                    "",
                    "| policy | comparator | metric | mean delta | median delta | CI95 low | CI95 high | win frac | paired episodes |",
                    "|---|---|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for _key, metrics in sorted(direct_downstream.items()):
                if not isinstance(metrics, Mapping):
                    continue
                policy = metrics.get("policy", "")
                comparator = metrics.get("comparator_policy", "")
                for metric in DOWNSTREAM_BRIDGE_METRICS:
                    row = metrics.get(metric, {})
                    if not isinstance(row, Mapping):
                        continue
                    lines.append(
                        "| `{policy}` | `{comparator}` | `{metric}` | {mean} | {median} | {low} | {high} | {win} | {episodes} |".format(
                            policy=policy,
                            comparator=comparator,
                            metric=metric,
                            mean=_fmt(row.get("mean_delta")),
                            median=_fmt(row.get("median_delta")),
                            low=_fmt(row.get("bootstrap_ci95_low")),
                            high=_fmt(row.get("bootstrap_ci95_high")),
                            win=_fmt(row.get("win_fraction")),
                            episodes=int(row.get("paired_episode_count", 0)),
                        )
                    )

    lines.extend(
        [
            "",
            "## Budget Curves",
            "",
            "| budget | policy | mean gain | episodes |",
            "|---:|---|---:|---:|",
        ]
    )
    for row in sorted(
        report.get("policy_budget_summary", []),
        key=lambda item: (int(item.get("budget_k", 0)), -float(item.get("mean_gain", 0.0)), str(item.get("policy_id", ""))),
    ):
        lines.append(
            "| {budget} | `{policy}` | {mean} | {episodes} |".format(
                budget=int(row.get("budget_k", 0)),
                policy=row.get("policy_id", ""),
                mean=_fmt(row.get("mean_gain")),
                episodes=int(row.get("episode_count", 0)),
            )
        )

    lines.extend(["", "## Gates", "", "| gate | status | read |", "|---|---|---|"])
    for gate in report.get("gates", []):
        lines.append(f"| {gate.get('name', '')} | {gate.get('status', '')} | {gate.get('read', '')} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Confidence intervals are episode-level paired bootstraps, not row-level bootstraps.",
            "- Oracle rows are non-deployable ceilings because they inspect the blind target set.",
            "- Gains use primary eval views but still exclude same-feature selector/eval shortcuts.",
            "",
        ]
    )
    return "\n".join(lines)


def _report_names(reports: Sequence[Mapping[str, Any]], report_names: Sequence[str] | None) -> tuple[str, ...]:
    if report_names is None:
        return tuple(f"report_{index:03d}" for index, _report in enumerate(reports))
    names = tuple(str(name) for name in report_names)
    if len(names) != len(reports):
        raise ValueError("report_names length must match reports length.")
    if len(set(names)) != len(names):
        raise ValueError("report_names must be unique.")
    return names


def _primary_view_gain_records(reports: Sequence[Mapping[str, Any]], names: Sequence[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for report, report_name in zip(reports, names):
        config = report.get("config", {})
        primary_views = {str(view) for view in config.get("primary_eval_views", [])}
        by_key: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in report.get("metric_rows", []):
            if str(row.get("metric_name", "")) != "coverage_gain_rel":
                continue
            if str(row.get("eval_view", "")) not in primary_views:
                continue
            if bool(row.get("selector_feature_overlap", False)):
                continue
            key = (
                str(report_name),
                str(row.get("episode_id", "")),
                int(row.get("budget_k", 0)),
                str(row.get("policy_id", "")),
            )
            by_key[key].append(dict(row))
        for (name, episode_id, budget_k, policy_id), rows in sorted(by_key.items()):
            records.append(
                {
                    "report_name": name,
                    "episode_id": episode_id,
                    "fold_id": _fold_id_from_rows(rows),
                    "unit_id": f"{name}:{episode_id}",
                    "budget_k": int(budget_k),
                    "policy_id": policy_id,
                    "gain": mean(float(row.get("metric_value", 0.0)) for row in rows),
                    "view_count": len(rows),
                    "eval_views": sorted({str(row.get("eval_view", "")) for row in rows}),
                    "uses_target_for_selection": any(bool(row.get("uses_target_for_selection", False)) for row in rows),
                }
            )
    return records


def _records_by_policy_unit(records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(str(record["unit_id"]), str(record["policy_id"])): record for record in records}


def _candidate_pool_summary(reports: Sequence[Mapping[str, Any]], names: Sequence[str]) -> dict[str, float | int | None]:
    by_unit: dict[str, int] = {}
    for report, report_name in zip(reports, names):
        for row in report.get("rounds", []):
            unit_id = f"{report_name}:{row.get('episode_id', '')}"
            count = int(row.get("candidate_count", 0))
            by_unit[unit_id] = max(count, by_unit.get(unit_id, 0))
    counts = list(by_unit.values())
    if not counts:
        return {"episode_count": 0, "min": None, "median": None, "max": None}
    return {
        "episode_count": len(counts),
        "min": int(min(counts)),
        "median": float(median(counts)),
        "max": int(max(counts)),
    }


def _policy_final_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, dict[str, Any]]:
    by_policy: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_policy[str(record["policy_id"])].append(record)
    summary: dict[str, dict[str, Any]] = {}
    for index, (policy_id, policy_records) in enumerate(sorted(by_policy.items())):
        gains = [float(record["gain"]) for record in policy_records]
        ci_low, ci_high = _bootstrap_ci95(gains, seed=bootstrap_seed + index, replicates=bootstrap_replicates)
        summary[policy_id] = {
            "episode_count": len(gains),
            "mean_final_gain": mean(gains) if gains else None,
            "median_final_gain": median(gains) if gains else None,
            "bootstrap_ci95_low": ci_low,
            "bootstrap_ci95_high": ci_high,
            "uses_target_for_selection": any(bool(record.get("uses_target_for_selection", False)) for record in policy_records),
        }
    return summary


def _policy_budget_summary(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, str], list[float]] = defaultdict(list)
    target_used: dict[tuple[int, str], bool] = defaultdict(bool)
    for record in records:
        key = (int(record["budget_k"]), str(record["policy_id"]))
        by_key[key].append(float(record["gain"]))
        target_used[key] = target_used[key] or bool(record.get("uses_target_for_selection", False))
    return [
        {
            "budget_k": budget_k,
            "policy_id": policy_id,
            "episode_count": len(gains),
            "mean_gain": mean(gains) if gains else None,
            "median_gain": median(gains) if gains else None,
            "uses_target_for_selection": bool(target_used[(budget_k, policy_id)]),
        }
        for (budget_k, policy_id), gains in sorted(by_key.items())
    ]


def _pairwise_summaries(
    final_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    baseline_policy: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, dict[str, Any]]:
    policies = sorted({policy for _unit, policy in final_by_key})
    units = sorted({unit for unit, _policy in final_by_key})
    summaries: dict[str, dict[str, Any]] = {}
    for index, policy in enumerate(policies):
        if policy == baseline_policy:
            continue
        deltas = []
        for unit in units:
            policy_row = final_by_key.get((unit, policy))
            baseline_row = final_by_key.get((unit, baseline_policy))
            if policy_row is None or baseline_row is None:
                continue
            deltas.append(float(policy_row["gain"]) - float(baseline_row["gain"]))
        ci_low, ci_high = _bootstrap_ci95(deltas, seed=bootstrap_seed + index, replicates=bootstrap_replicates)
        summaries[policy] = {
            "baseline_policy": baseline_policy,
            "paired_episode_count": len(deltas),
            "mean_delta": mean(deltas) if deltas else None,
            "median_delta": median(deltas) if deltas else None,
            "bootstrap_ci95_low": ci_low,
            "bootstrap_ci95_high": ci_high,
            "win_fraction": _win_fraction(deltas),
        }
    return summaries


def _direct_pairwise_summaries(
    final_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    policy_final: Mapping[str, Mapping[str, Any]],
    *,
    baseline_policy: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, dict[str, Any]]:
    top_policy, _top_row = _top_non_target_policy(policy_final)
    focal_policy = "ts2vec_kcenter_v1" if "ts2vec_kcenter_v1" in policy_final else top_policy
    if not focal_policy:
        return {}
    desired_pairs = [
        (focal_policy, baseline_policy),
        (focal_policy, "window_kcenter_v1"),
        (focal_policy, "submitted_full_replay_v1"),
        (focal_policy, "quality_only_v1"),
        ("submitted_full_replay_v1", baseline_policy),
        ("window_kcenter_v1", baseline_policy),
    ]
    summaries: dict[str, dict[str, Any]] = {}
    seen: set[tuple[str, str]] = set()
    for index, (policy, comparator) in enumerate(desired_pairs):
        if policy == comparator or (policy, comparator) in seen:
            continue
        seen.add((policy, comparator))
        if policy not in policy_final or comparator not in policy_final:
            continue
        if bool(policy_final.get(policy, {}).get("uses_target_for_selection", False)):
            continue
        if bool(policy_final.get(comparator, {}).get("uses_target_for_selection", False)):
            continue
        deltas = _paired_deltas(final_by_key, policy, comparator)
        key = f"{policy}__vs__{comparator}"
        summaries[key] = {
            "policy": policy,
            "comparator_policy": comparator,
            **_delta_summary(
                deltas,
                seed=bootstrap_seed + index,
                replicates=bootstrap_replicates,
            ),
        }
    return summaries


def _oracle_capture_summaries(
    final_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    baseline_policy: str,
    oracle_policy: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    eps: float,
) -> dict[str, dict[str, Any]]:
    units = sorted({unit for unit, _policy in final_by_key})
    policies = sorted({policy for _unit, policy in final_by_key if policy not in {baseline_policy, oracle_policy}})
    summaries: dict[str, dict[str, Any]] = {}
    for index, policy in enumerate(policies):
        captures = []
        skipped = 0
        for unit in units:
            policy_row = final_by_key.get((unit, policy))
            baseline_row = final_by_key.get((unit, baseline_policy))
            oracle_row = final_by_key.get((unit, oracle_policy))
            if policy_row is None or baseline_row is None or oracle_row is None:
                skipped += 1
                continue
            denominator = float(oracle_row["gain"]) - float(baseline_row["gain"])
            if denominator <= eps:
                skipped += 1
                continue
            captures.append((float(policy_row["gain"]) - float(baseline_row["gain"])) / denominator)
        if not captures:
            continue
        ci_low, ci_high = _bootstrap_ci95(captures, seed=bootstrap_seed + index, replicates=bootstrap_replicates)
        summaries[policy] = {
            "baseline_policy": baseline_policy,
            "oracle_policy": oracle_policy,
            "decision_use": "deprecated_debug_only",
            "warning": "Raw oracle capture is unstable when oracle-baseline headroom is small or negative; do not gate on the mean.",
            "paired_episode_count": len(captures),
            "skipped_episode_count": skipped,
            "mean_oracle_capture": mean(captures),
            "median_oracle_capture": median(captures),
            "bootstrap_ci95_low": ci_low,
            "bootstrap_ci95_high": ci_high,
        }
    return summaries


def _oracle_stability_diagnostics(
    final_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    policy_final: Mapping[str, Mapping[str, Any]],
    *,
    baseline_policy: str,
    oracle_policy: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    eps: float,
) -> dict[str, Any]:
    selected_oracle = _select_oracle_policy(policy_final, oracle_policy)
    if not selected_oracle:
        return {}
    baseline_by_unit, baseline_source = _episode_baseline_values(final_by_key, baseline_policy)
    units = sorted({unit for unit, _policy in final_by_key})
    headroom_values = []
    skipped = 0
    for unit in units:
        oracle_row = final_by_key.get((unit, selected_oracle))
        baseline_value = baseline_by_unit.get(unit)
        if oracle_row is None or baseline_value is None:
            skipped += 1
            continue
        headroom_values.append(float(oracle_row["gain"]) - baseline_value)

    headroom_epsilon = 0.01
    headroom_summary = {
        **_delta_summary(headroom_values, seed=bootstrap_seed, replicates=bootstrap_replicates),
        "skipped_episode_count": skipped,
        "positive_fraction": _fraction(headroom_values, lambda value: value > headroom_epsilon),
        "near_zero_fraction": _fraction(headroom_values, lambda value: abs(value) < headroom_epsilon),
    }

    deployable_policies = sorted(
        policy
        for policy, row in policy_final.items()
        if not bool(row.get("uses_target_for_selection", False))
        and not _is_quality_stratified_random_replay(policy)
        and policy != baseline_policy
        and row.get("mean_final_gain") is not None
    )
    policy_metrics: dict[str, dict[str, Any]] = {}
    for index, policy in enumerate(deployable_policies):
        oracle_gaps = []
        raw_captures = []
        positive_headroom_captures = []
        bounded_captures = []
        normalized_regrets = []
        negative_count = 0
        overshoot_count = 0
        paired = 0
        skipped_policy = 0
        for unit in units:
            policy_row = final_by_key.get((unit, policy))
            oracle_row = final_by_key.get((unit, selected_oracle))
            baseline_value = baseline_by_unit.get(unit)
            if policy_row is None or oracle_row is None or baseline_value is None:
                skipped_policy += 1
                continue
            paired += 1
            policy_gain = float(policy_row["gain"])
            oracle_gain = float(oracle_row["gain"])
            oracle_gap = oracle_gain - policy_gain
            headroom = oracle_gain - baseline_value
            policy_delta = policy_gain - baseline_value
            oracle_gaps.append(oracle_gap)
            if abs(headroom) > eps:
                raw_captures.append(policy_delta / headroom)
            if headroom > headroom_epsilon:
                capture = policy_delta / headroom
                positive_headroom_captures.append(capture)
                bounded_captures.append(min(1.0, max(0.0, capture)))
                normalized_regrets.append(1.0 - capture)
                if capture < 0.0:
                    negative_count += 1
                if capture > 1.0:
                    overshoot_count += 1
        policy_metrics[policy] = {
            "baseline_source": baseline_source,
            "oracle_policy": selected_oracle,
            "paired_episode_count": paired,
            "skipped_episode_count": skipped_policy,
            "oracle_gap": _delta_summary(
                oracle_gaps,
                seed=bootstrap_seed + index * 10 + 1,
                replicates=bootstrap_replicates,
            ),
            "raw_capture_unsafe": _value_summary(
                raw_captures,
                seed=bootstrap_seed + index * 10 + 2,
                replicates=bootstrap_replicates,
            ),
            "positive_headroom_capture": _value_summary(
                positive_headroom_captures,
                seed=bootstrap_seed + index * 10 + 3,
                replicates=bootstrap_replicates,
            ),
            "bounded_capture_mean": mean(bounded_captures) if bounded_captures else None,
            "bounded_capture_median": median(bounded_captures) if bounded_captures else None,
            "negative_capture_fraction": float(negative_count / len(positive_headroom_captures)) if positive_headroom_captures else None,
            "overshoot_fraction": float(overshoot_count / len(positive_headroom_captures)) if positive_headroom_captures else None,
            "normalized_regret": _value_summary(
                normalized_regrets,
                seed=bootstrap_seed + index * 10 + 4,
                replicates=bootstrap_replicates,
            ),
        }

    return {
        "oracle_policy": selected_oracle,
        "configured_oracle_policy": oracle_policy,
        "baseline_policy": baseline_policy,
        "baseline_source": baseline_source,
        "headroom_epsilon": headroom_epsilon,
        "oracle_headroom": headroom_summary,
        "policy_metrics": policy_metrics,
    }


def _oracle_diagnostics(
    final_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    policy_final: Mapping[str, Mapping[str, Any]],
    *,
    configured_oracle_policy: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    eps: float,
) -> dict[str, Any]:
    exact_policy = EXACT_COVERAGE_ORACLE_POLICY if EXACT_COVERAGE_ORACLE_POLICY in policy_final else None
    target_family_policy = TARGET_FAMILY_ORACLE_POLICY if TARGET_FAMILY_ORACLE_POLICY in policy_final else None
    eval_view_policy = ORACLE_POLICY if ORACLE_POLICY in policy_final else None
    top_policy, _top_row = _top_non_target_policy(policy_final)
    exact_gap = {}
    if exact_policy and top_policy:
        gaps = []
        skipped = 0
        units = sorted({unit for unit, _policy in final_by_key})
        for unit in units:
            exact_row = final_by_key.get((unit, exact_policy))
            top_row = final_by_key.get((unit, top_policy))
            if exact_row is None or top_row is None:
                skipped += 1
                continue
            gaps.append(float(exact_row["gain"]) - float(top_row["gain"]))
        ci_low, ci_high = _bootstrap_ci95(gaps, seed=bootstrap_seed, replicates=bootstrap_replicates)
        mean_gap = mean(gaps) if gaps else None
        exact_gap = {
            "status": "pass" if mean_gap is not None and mean_gap >= -eps and (ci_low is None or ci_low >= -eps) else "warn",
            "paired_episode_count": len(gaps),
            "skipped_episode_count": skipped,
            "mean_gap": mean_gap,
            "median_gap": median(gaps) if gaps else None,
            "bootstrap_ci95_low": ci_low,
            "bootstrap_ci95_high": ci_high,
        }

    violating = []
    if exact_policy and policy_final.get(exact_policy, {}).get("mean_final_gain") is not None:
        exact_mean = float(policy_final[exact_policy]["mean_final_gain"])
        for policy, row in policy_final.items():
            if policy == exact_policy or bool(row.get("uses_target_for_selection", False)) or _is_quality_stratified_random_replay(policy):
                continue
            if row.get("mean_final_gain") is not None and float(row["mean_final_gain"]) > exact_mean + eps:
                violating.append(policy)

    return {
        "configured_oracle_policy": configured_oracle_policy,
        "exact_coverage_policy": exact_policy,
        "target_family_oracle_policy": target_family_policy,
        "eval_view_greedy_oracle_policy": eval_view_policy,
        "top_deployable_policy": top_policy,
        "exact_coverage_vs_top_deployable": exact_gap,
        "deployable_mean_gain_violations_vs_exact": sorted(violating),
    }


def _random_replay_diagnostics(
    final_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    policy_final: Mapping[str, Mapping[str, Any]],
    *,
    baseline_policy: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    replay_policies = sorted(policy for policy in policy_final if _is_quality_stratified_random_replay(policy))
    if not replay_policies:
        return {"replay_count": 0, "replay_policies": []}
    replay_means = [
        float(policy_final[policy]["mean_final_gain"])
        for policy in replay_policies
        if policy_final[policy].get("mean_final_gain") is not None
    ]
    top_policy, top_row = _top_non_target_policy(policy_final)
    top_mean = float(top_row["mean_final_gain"]) if top_row and top_row.get("mean_final_gain") is not None else None
    replay_percentile = None
    replay_ge_top_count = None
    replay_null_pvalue = None
    if top_mean is not None and replay_means:
        replay_percentile = float(sum(value <= top_mean for value in replay_means) / len(replay_means))
        replay_ge_top_count = int(sum(value >= top_mean for value in replay_means))
        replay_null_pvalue = float((1 + replay_ge_top_count) / (1 + len(replay_means)))

    deltas = []
    skipped = 0
    units = sorted({unit for unit, _policy in final_by_key})
    if top_policy:
        for unit in units:
            top_record = final_by_key.get((unit, top_policy))
            replay_records = [final_by_key.get((unit, policy)) for policy in replay_policies]
            replay_values = [float(record["gain"]) for record in replay_records if record is not None]
            if top_record is None or not replay_values:
                skipped += 1
                continue
            deltas.append(float(top_record["gain"]) - mean(replay_values))
    ci_low, ci_high = _bootstrap_ci95(deltas, seed=bootstrap_seed, replicates=bootstrap_replicates)

    return {
        "baseline_policy": baseline_policy,
        "replay_count": len(replay_policies),
        "replay_policies": replay_policies,
        "replay_mean_gain": {
            "min": min(replay_means) if replay_means else None,
            "median": median(replay_means) if replay_means else None,
            "mean": mean(replay_means) if replay_means else None,
            "max": max(replay_means) if replay_means else None,
        },
        "top_deployable_policy": top_policy,
        "top_deployable_replay_percentile_by_mean_gain": replay_percentile,
        "top_deployable_replay_ge_mean_count": replay_ge_top_count,
        "top_deployable_replay_null_pvalue_plus_one": replay_null_pvalue,
        "top_deployable_vs_replay_mean": {
            "paired_episode_count": len(deltas),
            "skipped_episode_count": skipped,
            "mean_delta": mean(deltas) if deltas else None,
            "median_delta": median(deltas) if deltas else None,
            "bootstrap_ci95_low": ci_low,
            "bootstrap_ci95_high": ci_high,
        },
    }


def _acquisition_stability_diagnostics(
    final_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    policy_final: Mapping[str, Mapping[str, Any]],
    *,
    baseline_policy: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    top_policy, _top_row = _top_non_target_policy(policy_final)
    focal_policy = "ts2vec_kcenter_v1" if "ts2vec_kcenter_v1" in policy_final else top_policy
    if not focal_policy:
        return {}
    baseline_by_unit, baseline_source = _episode_baseline_values(final_by_key, baseline_policy)
    rows = []
    for unit in sorted({unit for unit, _policy in final_by_key}):
        policy_row = final_by_key.get((unit, focal_policy))
        baseline_value = baseline_by_unit.get(unit)
        if policy_row is None or baseline_value is None:
            continue
        rows.append(
            {
                "unit_id": unit,
                "report_name": str(policy_row.get("report_name", "")),
                "episode_id": str(policy_row.get("episode_id", "")),
                "fold_id": str(policy_row.get("fold_id", "")),
                "policy_gain": float(policy_row["gain"]),
                "baseline_gain": float(baseline_value),
                "delta": float(policy_row["gain"]) - float(baseline_value),
            }
        )
    deltas = [float(row["delta"]) for row in rows]
    by_seed = _group_delta_summaries(rows, "report_name")
    by_fold = _group_delta_summaries(rows, "fold_id")
    leave_seed = _leave_one_out(rows, "report_name")
    leave_fold = _leave_one_out(rows, "fold_id")
    return {
        "policy": focal_policy,
        "baseline_policy": baseline_policy,
        "baseline_source": baseline_source,
        "paired_episode_count": len(rows),
        "overall": _delta_summary(deltas, seed=bootstrap_seed, replicates=bootstrap_replicates),
        "positive_seed_fraction": _positive_group_fraction(by_seed),
        "positive_fold_fraction": _positive_group_fraction(by_fold),
        "by_seed": by_seed,
        "by_fold": by_fold,
        "leave_one_seed_out": leave_seed,
        "leave_one_fold_out": leave_fold,
        "min_leave_one_seed_out_delta": _min_leave_one_delta(leave_seed),
        "min_leave_one_fold_out_delta": _min_leave_one_delta(leave_fold),
        "largest_single_episode_contribution_fraction": _largest_contribution_fraction(deltas),
        "episode_influence": sorted(rows, key=lambda row: abs(float(row["delta"])), reverse=True),
    }


def _selected_set_audit(
    reports: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    final_budget: int,
) -> dict[str, dict[str, Any]]:
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report, report_name in zip(reports, names):
        for row in report.get("selected_rows", []):
            if not isinstance(row, Mapping) or int(row.get("budget_k", 0)) != int(final_budget):
                continue
            item = dict(row)
            item["unit_id"] = f"{report_name}:{row.get('episode_id', '')}"
            by_policy[str(row.get("policy_id", ""))].append(item)
    audits: dict[str, dict[str, Any]] = {}
    for policy, rows in sorted(by_policy.items()):
        if not rows:
            continue
        source_counts: dict[str, int] = defaultdict(int)
        unit_sample_ids: dict[str, list[str]] = defaultdict(list)
        quality_values = []
        artifact_values = []
        invalid_count = 0
        for row in rows:
            source_counts[str(row.get("source_group_id", ""))] += 1
            unit_sample_ids[str(row.get("unit_id", ""))].append(str(row.get("sample_id", "")))
            quality_values.append(float(row.get("quality_score", 0.0)))
            artifact_values.append(float(row.get("artifact_score", 0.0)))
            if not bool(row.get("valid", False)):
                invalid_count += 1
        duplicate_count = sum(len(values) - len(set(values)) for values in unit_sample_ids.values())
        top_source_count = max(source_counts.values()) if source_counts else 0
        audits[policy] = {
            "selected_count": len(rows),
            "episode_count": len(unit_sample_ids),
            "invalid_rate": float(invalid_count / len(rows)),
            "duplicate_clip_fraction": float(duplicate_count / len(rows)),
            "mean_quality_score": mean(quality_values) if quality_values else None,
            "median_quality_score": median(quality_values) if quality_values else None,
            "min_quality_score": min(quality_values) if quality_values else None,
            "mean_artifact_score": mean(artifact_values) if artifact_values else None,
            "max_artifact_score": max(artifact_values) if artifact_values else None,
            "unique_source_group_count": len(source_counts),
            "top_source_group_fraction": float(top_source_count / len(rows)) if rows else None,
        }
    return audits


def _training_unlock_criteria(*, baseline_policy: str, focal_policy: str) -> dict[str, dict[str, Any]]:
    expanded = dict(EXPANDED_CPU_UNLOCK_CRITERIA)
    expanded.update(
        {
            "focal_policy": focal_policy,
            "baseline_policy": baseline_policy,
            "scope": "Unlock only a bounded frozen-feature downstream canary; keep TS2Vec retraining and large neural training held.",
            "direct_control_policies": ["window_kcenter_v1", "submitted_full_replay_v1"],
            "decision_use": "pre-registered bar for the expanded 10-seed, 8-fold, pool-stress CPU validation",
        }
    )
    return {"expanded_cpu_validation": expanded}


def _downstream_bridge_proxy_summary(
    downstream_reports: Sequence[Mapping[str, Any]],
    *,
    report_names: Sequence[str] | None,
    baseline_policy: str,
    oracle_policy: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if not downstream_reports:
        return {}
    names = _report_names(downstream_reports, report_names)
    records = _downstream_bridge_final_records(downstream_reports, names)
    if not records:
        return {}
    final_by_key = _downstream_records_by_policy_unit(records)
    policy_final = _downstream_policy_final_summary(
        records,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed + 17,
    )
    independent_episode_count = len({str(record["unit_id"]) for record in records})
    pairwise = _downstream_pairwise_summaries(
        final_by_key,
        baseline_policy=baseline_policy,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed + 101,
    )
    direct_pairwise = _downstream_direct_pairwise_summaries(
        final_by_key,
        policy_final,
        baseline_policy=baseline_policy,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed + 307,
    )
    return {
        "units": {
            "report_count": len(downstream_reports),
            "independent_episode_count": independent_episode_count,
            "final_budget": max((int(record["budget_k"]) for record in records), default=0),
            "record_count": len(records),
            "source_row_count": sum(int(record.get("source_row_count", 0)) for record in records),
        },
        "metrics": list(DOWNSTREAM_BRIDGE_METRICS),
        "policy_final_summary": policy_final,
        "pairwise_vs_baseline": pairwise,
        "direct_pairwise_controls": direct_pairwise,
        "canary_criteria": dict(DOWNSTREAM_CANARY_CRITERIA),
        "decision": _downstream_bridge_decision(
            policy_final,
            pairwise,
            independent_episode_count=independent_episode_count,
            baseline_policy=baseline_policy,
            oracle_policy=oracle_policy,
        ),
    }


def _downstream_bridge_final_records(
    downstream_reports: Sequence[Mapping[str, Any]],
    names: Sequence[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for report, report_name in zip(downstream_reports, names):
        rows = [dict(row) for row in report.get("rows", []) if isinstance(row, Mapping)]
        final_budget = max((int(row.get("budget_k", 0)) for row in rows), default=0)
        by_key: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if int(row.get("budget_k", 0)) != final_budget:
                continue
            key = (
                str(report_name),
                str(row.get("episode_id", "")),
                int(row.get("budget_k", 0)),
                str(row.get("policy_id", "")),
            )
            by_key[key].append(row)
        for (name, episode_id, budget_k, policy_id), unit_rows in sorted(by_key.items()):
            metrics = {
                metric: _mean_finite([row.get(metric) for row in unit_rows])
                for metric in DOWNSTREAM_BRIDGE_METRICS
            }
            records.append(
                {
                    "report_name": name,
                    "episode_id": episode_id,
                    "unit_id": f"{name}:{episode_id}",
                    "budget_k": int(budget_k),
                    "policy_id": policy_id,
                    "metrics": metrics,
                    "source_row_count": len(unit_rows),
                    "representations": sorted({str(row.get("representation", "")) for row in unit_rows}),
                }
            )
    return records


def _downstream_records_by_policy_unit(records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(str(record["unit_id"]), str(record["policy_id"])): record for record in records}


def _downstream_policy_final_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, dict[str, Any]]:
    by_policy: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_policy[str(record["policy_id"])].append(record)
    summary: dict[str, dict[str, Any]] = {}
    for index, (policy_id, policy_records) in enumerate(sorted(by_policy.items())):
        metric_summary = {}
        for metric in DOWNSTREAM_BRIDGE_METRICS:
            values = [
                float(record.get("metrics", {}).get(metric))
                for record in policy_records
                if record.get("metrics", {}).get(metric) is not None
            ]
            ci_low, ci_high = _bootstrap_ci95(values, seed=bootstrap_seed + index + len(metric), replicates=bootstrap_replicates)
            metric_summary[metric] = {
                "mean": mean(values) if values else None,
                "median": median(values) if values else None,
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "value_count": len(values),
            }
        summary[policy_id] = {
            "episode_count": len(policy_records),
            "metrics": metric_summary,
        }
    return summary


def _downstream_pairwise_summaries(
    final_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    baseline_policy: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, dict[str, Any]]:
    policies = sorted({policy for _unit, policy in final_by_key})
    units = sorted({unit for unit, _policy in final_by_key})
    summaries: dict[str, dict[str, Any]] = {}
    for policy_index, policy in enumerate(policies):
        if policy == baseline_policy:
            continue
        metric_summaries: dict[str, dict[str, Any]] = {}
        for metric_index, metric in enumerate(DOWNSTREAM_BRIDGE_METRICS):
            deltas = []
            for unit in units:
                policy_row = final_by_key.get((unit, policy))
                baseline_row = final_by_key.get((unit, baseline_policy))
                if policy_row is None or baseline_row is None:
                    continue
                policy_value = policy_row.get("metrics", {}).get(metric)
                baseline_value = baseline_row.get("metrics", {}).get(metric)
                if policy_value is None or baseline_value is None:
                    continue
                deltas.append(float(policy_value) - float(baseline_value))
            ci_low, ci_high = _bootstrap_ci95(
                deltas,
                seed=bootstrap_seed + policy_index * 100 + metric_index,
                replicates=bootstrap_replicates,
            )
            metric_summaries[metric] = {
                "baseline_policy": baseline_policy,
                "paired_episode_count": len(deltas),
                "mean_delta": mean(deltas) if deltas else None,
                "median_delta": median(deltas) if deltas else None,
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "win_fraction": _win_fraction(deltas),
            }
        summaries[policy] = metric_summaries
    return summaries


def _downstream_direct_pairwise_summaries(
    final_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    policy_final: Mapping[str, Mapping[str, Any]],
    *,
    baseline_policy: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, dict[str, Any]]:
    focal_policy = "ts2vec_kcenter_v1" if "ts2vec_kcenter_v1" in policy_final else _top_downstream_policy(
        policy_final,
        exclude={baseline_policy, ORACLE_POLICY, EXACT_COVERAGE_ORACLE_POLICY, TARGET_FAMILY_ORACLE_POLICY},
        metric="balanced_accuracy_gain",
    )
    if not focal_policy:
        return {}
    comparators = [
        baseline_policy,
        "window_kcenter_v1",
        "submitted_full_replay_v1",
        "quality_only_v1",
    ]
    output: dict[str, dict[str, Any]] = {}
    seen: set[tuple[str, str]] = set()
    pair_index = 0
    for comparator in comparators:
        if comparator == focal_policy or comparator not in policy_final or (focal_policy, comparator) in seen:
            continue
        seen.add((focal_policy, comparator))
        metric_summaries = {
            metric: _downstream_metric_delta_summary(
                final_by_key,
                focal_policy,
                comparator,
                metric,
                seed=bootstrap_seed + pair_index * 100 + metric_index,
                replicates=bootstrap_replicates,
            )
            for metric_index, metric in enumerate(DOWNSTREAM_BRIDGE_METRICS)
        }
        key = f"{focal_policy}__vs__{comparator}"
        output[key] = {
            "policy": focal_policy,
            "comparator_policy": comparator,
            **metric_summaries,
        }
        pair_index += 1
    return output


def _downstream_metric_delta_summary(
    final_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    policy: str,
    comparator: str,
    metric: str,
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    deltas = []
    for unit in sorted({unit for unit, _policy in final_by_key}):
        policy_row = final_by_key.get((unit, policy))
        comparator_row = final_by_key.get((unit, comparator))
        if policy_row is None or comparator_row is None:
            continue
        policy_value = policy_row.get("metrics", {}).get(metric)
        comparator_value = comparator_row.get("metrics", {}).get(metric)
        if policy_value is None or comparator_value is None:
            continue
        deltas.append(float(policy_value) - float(comparator_value))
    return _delta_summary(deltas, seed=seed, replicates=replicates)


def _downstream_bridge_decision(
    policy_final: Mapping[str, Mapping[str, Any]],
    pairwise: Mapping[str, Mapping[str, Any]],
    *,
    independent_episode_count: int,
    baseline_policy: str,
    oracle_policy: str,
) -> dict[str, Any]:
    top_policy = _top_downstream_policy(
        policy_final,
        exclude={baseline_policy, oracle_policy, ORACLE_POLICY, EXACT_COVERAGE_ORACLE_POLICY, TARGET_FAMILY_ORACLE_POLICY},
        metric="balanced_accuracy_gain",
    )
    focal_policy = "ts2vec_kcenter_v1" if "ts2vec_kcenter_v1" in policy_final else top_policy
    top_pair = pairwise.get(focal_policy or "", {}) if focal_policy else {}
    balanced_row = top_pair.get("balanced_accuracy_gain", {}) if isinstance(top_pair, Mapping) else {}
    nll_row = top_pair.get("nll_reduction", {}) if isinstance(top_pair, Mapping) else {}
    discovery_row = top_pair.get("target_family_discovery_rate", {}) if isinstance(top_pair, Mapping) else {}
    canary_pass = (
        independent_episode_count >= int(DOWNSTREAM_CANARY_CRITERIA["minimum_independent_episodes"])
        and _ge(balanced_row.get("mean_delta"), DOWNSTREAM_CANARY_CRITERIA["balanced_accuracy_delta_mean_min"])
        and _gt(balanced_row.get("bootstrap_ci95_low"), DOWNSTREAM_CANARY_CRITERIA["balanced_accuracy_delta_ci95_low_min"])
        and _ge(balanced_row.get("win_fraction"), DOWNSTREAM_CANARY_CRITERIA["balanced_accuracy_win_fraction_min"])
        and _gt(
            discovery_row.get("mean_delta"),
            DOWNSTREAM_CANARY_CRITERIA["target_family_discovery_delta_mean_min"],
        )
    )
    read = (
        "bounded frozen-feature downstream canary passed on the bridge proxy; keep this as source-family pseudo-label evidence only."
        if canary_pass
        else "bridge proxy is aggregated with episode-level paired intervals; use it as an identifiability gate, not real challenge-label downstream proof."
    )
    return {
        "downstream_training": "bounded_frozen_canary_pass" if canary_pass else "hold_large_training",
        "bounded_frozen_canary": "pass" if canary_pass else "hold",
        "large_training": "hold",
        "ts2vec_retraining": "no",
        "baseline_policy": baseline_policy,
        "oracle_policy": oracle_policy,
        "focal_policy": focal_policy,
        "top_deployable_policy_by_balanced_accuracy": top_policy,
        "independent_episode_count": int(independent_episode_count),
        "canary_criteria": dict(DOWNSTREAM_CANARY_CRITERIA),
        "balanced_accuracy_delta_vs_baseline": balanced_row.get("mean_delta") if isinstance(balanced_row, Mapping) else None,
        "balanced_accuracy_ci95_low_vs_baseline": balanced_row.get("bootstrap_ci95_low") if isinstance(balanced_row, Mapping) else None,
        "balanced_accuracy_win_fraction_vs_baseline": balanced_row.get("win_fraction") if isinstance(balanced_row, Mapping) else None,
        "nll_ci95_low_vs_baseline": nll_row.get("bootstrap_ci95_low") if isinstance(nll_row, Mapping) else None,
        "target_family_discovery_mean_delta_vs_baseline": discovery_row.get("mean_delta") if isinstance(discovery_row, Mapping) else None,
        "read": read,
        "next_steps": [
            "Use this as permission for a bounded frozen-feature downstream canary only, not larger training.",
            "Do not treat bridge pseudo-label gains as real downstream model training proof.",
        ],
    }


def _fold_id_from_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    fold_ids = sorted({str(row.get("fold_id", "")) for row in rows})
    if len(fold_ids) == 1:
        return fold_ids[0]
    return ",".join(fold_ids)


def _select_oracle_policy(policy_final: Mapping[str, Mapping[str, Any]], configured_oracle_policy: str) -> str | None:
    for policy in (configured_oracle_policy, EXACT_COVERAGE_ORACLE_POLICY, ORACLE_POLICY, TARGET_FAMILY_ORACLE_POLICY):
        if policy in policy_final:
            return policy
    return None


def _episode_baseline_values(
    final_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    baseline_policy: str,
) -> tuple[dict[str, float], str]:
    replay_policies = sorted({policy for _unit, policy in final_by_key if _is_quality_stratified_random_replay(policy)})
    units = sorted({unit for unit, _policy in final_by_key})
    if replay_policies:
        values: dict[str, float] = {}
        for unit in units:
            replay_values = [
                float(final_by_key[(unit, policy)]["gain"])
                for policy in replay_policies
                if (unit, policy) in final_by_key
            ]
            if replay_values:
                values[unit] = mean(replay_values)
        if values:
            return values, "quality_stratified_random_replay_episode_mean"

    values = {
        unit: float(final_by_key[(unit, baseline_policy)]["gain"])
        for unit in units
        if (unit, baseline_policy) in final_by_key
    }
    return values, baseline_policy


def _paired_deltas(
    final_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    policy: str,
    comparator: str,
) -> list[float]:
    deltas = []
    for unit in sorted({unit for unit, _policy in final_by_key}):
        policy_row = final_by_key.get((unit, policy))
        comparator_row = final_by_key.get((unit, comparator))
        if policy_row is None or comparator_row is None:
            continue
        deltas.append(float(policy_row["gain"]) - float(comparator_row["gain"]))
    return deltas


def _delta_summary(values: Sequence[float], *, seed: int, replicates: int) -> dict[str, Any]:
    ci_low, ci_high = _bootstrap_ci95(values, seed=seed, replicates=replicates)
    return {
        "paired_episode_count": len(values),
        "mean_delta": mean(values) if values else None,
        "median_delta": median(values) if values else None,
        "bootstrap_ci95_low": ci_low,
        "bootstrap_ci95_high": ci_high,
        "win_fraction": _win_fraction(values),
        "loss_fraction": _fraction(values, lambda value: value < 0.0),
        "tie_fraction": _fraction(values, lambda value: value == 0.0),
        "positive_episode_count": int(sum(float(value) > 0.0 for value in values)),
        "negative_episode_count": int(sum(float(value) < 0.0 for value in values)),
        "tie_episode_count": int(sum(float(value) == 0.0 for value in values)),
    }


def _value_summary(values: Sequence[float], *, seed: int, replicates: int) -> dict[str, Any]:
    ci_low, ci_high = _bootstrap_ci95(values, seed=seed, replicates=replicates)
    return {
        "paired_episode_count": len(values),
        "mean": mean(values) if values else None,
        "median": median(values) if values else None,
        "iqr_low": float(np.quantile(np.asarray(values, dtype=float), 0.25)) if values else None,
        "iqr_high": float(np.quantile(np.asarray(values, dtype=float), 0.75)) if values else None,
        "bootstrap_ci95_low": ci_low,
        "bootstrap_ci95_high": ci_high,
    }


def _group_delta_summaries(rows: Sequence[Mapping[str, Any]], group_key: str) -> list[dict[str, Any]]:
    by_group: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_group[str(row.get(group_key, ""))].append(float(row.get("delta", 0.0)))
    return [
        {
            "group": group,
            "episode_count": len(deltas),
            "mean_delta": mean(deltas) if deltas else None,
            "median_delta": median(deltas) if deltas else None,
            "win_fraction": _win_fraction(deltas),
        }
        for group, deltas in sorted(by_group.items())
    ]


def _leave_one_out(rows: Sequence[Mapping[str, Any]], group_key: str) -> list[dict[str, Any]]:
    groups = sorted({str(row.get(group_key, "")) for row in rows})
    output = []
    for group in groups:
        kept = [float(row.get("delta", 0.0)) for row in rows if str(row.get(group_key, "")) != group]
        output.append(
            {
                "dropped_group": group,
                "kept_episode_count": len(kept),
                "mean_delta_after_drop": mean(kept) if kept else None,
            }
        )
    return output


def _positive_group_fraction(rows: Sequence[Mapping[str, Any]]) -> float | None:
    if not rows:
        return None
    positive = 0
    for row in rows:
        value = row.get("mean_delta")
        if value is not None and float(value) > 0.0:
            positive += 1
    return float(positive / len(rows))


def _min_leave_one_delta(rows: Sequence[Mapping[str, Any]]) -> float | None:
    values = [float(row["mean_delta_after_drop"]) for row in rows if row.get("mean_delta_after_drop") is not None]
    return min(values) if values else None


def _largest_contribution_fraction(values: Sequence[float]) -> float | None:
    absolute = [abs(float(value)) for value in values]
    total = sum(absolute)
    if total <= 0.0:
        return None
    return float(max(absolute) / total)


def _decision_gates(
    *,
    final_records: Sequence[Mapping[str, Any]],
    policy_final: Mapping[str, Mapping[str, Any]],
    pairwise: Mapping[str, Mapping[str, Any]],
    direct_pairwise: Mapping[str, Mapping[str, Any]],
    oracle_stability: Mapping[str, Any],
    random_replay: Mapping[str, Any],
    acquisition_stability: Mapping[str, Any],
    baseline_policy: str,
    oracle_policy: str,
) -> list[dict[str, Any]]:
    episode_count = len({str(record["unit_id"]) for record in final_records})
    top_policy, top_row = _top_non_target_policy(policy_final)
    top_pair = pairwise.get(top_policy, {}) if top_policy else {}
    top_vs_replay = random_replay.get("top_deployable_vs_replay_mean", {}) if isinstance(random_replay, Mapping) else {}
    oracle_headroom = oracle_stability.get("oracle_headroom", {}) if isinstance(oracle_stability, Mapping) else {}
    focal_policy = acquisition_stability.get("policy") if isinstance(acquisition_stability, Mapping) else top_policy
    direct_control_rows = [
        row
        for row in direct_pairwise.values()
        if isinstance(row, Mapping)
        and str(row.get("policy")) == str(focal_policy)
        and str(row.get("comparator_policy")) in {"window_kcenter_v1", "submitted_full_replay_v1"}
    ]
    direct_control_pass = bool(direct_control_rows) and all(
        _gt(row.get("mean_delta"), 0.0) and _gt(row.get("median_delta"), 0.0)
        for row in direct_control_rows
    )
    stability_pass = (
        _ge(acquisition_stability.get("positive_seed_fraction") if isinstance(acquisition_stability, Mapping) else None, 1.0)
        and _ge(acquisition_stability.get("positive_fold_fraction") if isinstance(acquisition_stability, Mapping) else None, 0.75)
        and _gt(acquisition_stability.get("min_leave_one_seed_out_delta") if isinstance(acquisition_stability, Mapping) else None, 0.0)
        and _gt(acquisition_stability.get("min_leave_one_fold_out_delta") if isinstance(acquisition_stability, Mapping) else None, 0.0)
    )
    replay_pvalue = random_replay.get("top_deployable_replay_null_pvalue_plus_one") if isinstance(random_replay, Mapping) else None
    gates = [
        {
            "name": "episode_count",
            "status": "pass" if episode_count >= 20 else "warn",
            "read": f"{episode_count} independent final episodes; require at least 20 before downstream spend.",
            "evidence": {"independent_episode_count": episode_count},
        },
        {
            "name": "large_training_baseline_delta_ci",
            "status": "pass" if _gt(top_pair.get("bootstrap_ci95_low"), 0.04) else "warn",
            "read": (
                f"{top_policy} vs {baseline_policy} CI low is {_fmt(top_pair.get('bootstrap_ci95_low'))}; "
                "require > 0.04 before large downstream spend."
            ),
            "evidence": {"top_policy": top_policy, **top_pair},
        },
        {
            "name": "oracle_present",
            "status": "pass" if oracle_policy in policy_final else "fail",
            "read": "oracle ceiling is present." if oracle_policy in policy_final else "oracle ceiling is missing.",
            "evidence": {"oracle_policy": oracle_policy},
        },
        {
            "name": "oracle_headroom_sanity",
            "status": (
                "pass"
                if _gt(oracle_headroom.get("median_delta"), 0.0)
                and _ge(oracle_headroom.get("positive_fraction"), 0.70)
                else "warn"
            ),
            "read": (
                f"{oracle_policy} headroom median is {_fmt(oracle_headroom.get('median_delta'))} with positive fraction "
                f"{_fmt(oracle_headroom.get('positive_fraction'))}; require positive median and >= 0.70 positive fraction."
            ),
            "evidence": oracle_headroom,
        },
        {
            "name": "tiny_probe_random_delta",
            "status": (
                "pass"
                if _ge(top_pair.get("bootstrap_ci95_low"), 0.02)
                and _ge(top_pair.get("median_delta"), 0.025)
                and _ge(top_pair.get("win_fraction"), 0.70)
                else "warn"
            ),
            "read": (
                f"{top_policy} vs {baseline_policy}: CI low {_fmt(top_pair.get('bootstrap_ci95_low'))}, "
                f"median {_fmt(top_pair.get('median_delta'))}, win fraction {_fmt(top_pair.get('win_fraction'))}; "
                "tiny-probe threshold is CI low >= 0.02, median >= 0.025, win fraction >= 0.70."
            ),
            "evidence": {"top_policy": top_policy, **top_pair},
        },
        {
            "name": "tiny_probe_replay_null",
            "status": (
                "pass"
                if _ge(random_replay.get("top_deployable_replay_percentile_by_mean_gain"), 0.95)
                and _le(replay_pvalue, 0.05)
                and _gt(top_vs_replay.get("bootstrap_ci95_low"), 0.0)
                else "warn"
            ),
            "read": (
                f"{top_policy} replay percentile {_fmt(random_replay.get('top_deployable_replay_percentile_by_mean_gain'))}, "
                f"+1 p-value {_fmt(replay_pvalue)}, replay-mean CI low {_fmt(top_vs_replay.get('bootstrap_ci95_low'))}."
            ),
            "evidence": {
                "top_policy": top_policy,
                "top_deployable_replay_percentile_by_mean_gain": random_replay.get("top_deployable_replay_percentile_by_mean_gain")
                if isinstance(random_replay, Mapping)
                else None,
                "top_deployable_replay_null_pvalue_plus_one": replay_pvalue,
                "top_deployable_vs_replay_mean": top_vs_replay,
            },
        },
        {
            "name": "tiny_probe_direct_controls",
            "status": "pass" if direct_control_pass else "warn",
            "read": f"{focal_policy} must beat window_kcenter_v1 and submitted_full_replay_v1 by paired mean and median.",
            "evidence": {"focal_policy": focal_policy, "comparisons": direct_control_rows},
        },
        {
            "name": "tiny_probe_stability",
            "status": "pass" if stability_pass else "warn",
            "read": (
                f"{focal_policy} stability: positive seed fraction {_fmt(acquisition_stability.get('positive_seed_fraction') if isinstance(acquisition_stability, Mapping) else None)}, "
                f"positive fold fraction {_fmt(acquisition_stability.get('positive_fold_fraction') if isinstance(acquisition_stability, Mapping) else None)}, "
                f"min leave-one-seed {_fmt(acquisition_stability.get('min_leave_one_seed_out_delta') if isinstance(acquisition_stability, Mapping) else None)}, "
                f"min leave-one-fold {_fmt(acquisition_stability.get('min_leave_one_fold_out_delta') if isinstance(acquisition_stability, Mapping) else None)}."
            ),
            "evidence": {
                "policy": focal_policy,
                "positive_seed_fraction": acquisition_stability.get("positive_seed_fraction") if isinstance(acquisition_stability, Mapping) else None,
                "positive_fold_fraction": acquisition_stability.get("positive_fold_fraction") if isinstance(acquisition_stability, Mapping) else None,
                "min_leave_one_seed_out_delta": acquisition_stability.get("min_leave_one_seed_out_delta") if isinstance(acquisition_stability, Mapping) else None,
                "min_leave_one_fold_out_delta": acquisition_stability.get("min_leave_one_fold_out_delta") if isinstance(acquisition_stability, Mapping) else None,
            },
        },
    ]
    if top_row and bool(top_row.get("uses_target_for_selection", False)):
        gates.append(
            {
                "name": "deployable_top_policy",
                "status": "fail",
                "read": f"top policy {top_policy} uses target labels and is not deployable.",
                "evidence": {"top_policy": top_policy},
            }
        )
    else:
        gates.append(
            {
                "name": "deployable_top_policy",
                "status": "pass" if top_policy else "fail",
                "read": f"top non-oracle policy is {top_policy}." if top_policy else "no deployable policy found.",
                "evidence": {"top_policy": top_policy},
            }
        )
    return gates


def _decision(gates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = {str(gate.get("name")): str(gate.get("status")) for gate in gates}
    hard_fail = any(status == "fail" for name, status in statuses.items() if name in {"oracle_present", "deployable_top_policy"})
    large_statistical_pass = statuses.get("large_training_baseline_delta_ci") == "pass"
    oracle_pass = statuses.get("oracle_headroom_sanity") == "pass"
    enough_episodes = statuses.get("episode_count") == "pass"
    tiny_probe_pass = (
        not hard_fail
        and enough_episodes
        and oracle_pass
        and statuses.get("tiny_probe_random_delta") == "pass"
        and statuses.get("tiny_probe_replay_null") == "pass"
        and statuses.get("tiny_probe_direct_controls") == "pass"
        and statuses.get("tiny_probe_stability") == "pass"
    )
    bounded_canary_pass = not hard_fail and large_statistical_pass and oracle_pass and enough_episodes and tiny_probe_pass
    if tiny_probe_pass:
        return {
            "downstream_training": "bounded-frozen-canary-ok" if bounded_canary_pass else "tiny-frozen-probe-ok",
            "bounded_downstream_canary": "ok" if bounded_canary_pass else "pending-statistical-margin",
            "large_training": "hold",
            "tiny_frozen_supervised_probe": "ok",
            "ts2vec_retraining": "no",
            "next_cpu_gate": "run-predeclared-frozen-supervised-probe",
            "read": "offline coverage gate is strong enough for a bounded frozen-feature downstream canary; large neural training and TS2Vec retraining remain held.",
        }
    return {
        "downstream_training": "hold",
        "bounded_downstream_canary": "hold",
        "large_training": "hold",
        "tiny_frozen_supervised_probe": "pending-audit",
        "ts2vec_retraining": "no",
        "next_cpu_gate": "oracle-plus-episode-bootstrap",
        "read": "keep this in CPU/offline benchmark mode until oracle, confidence, direct-control, and stability gates pass.",
    }


def _decision_with_downstream_bridge(
    decision: Mapping[str, Any],
    downstream_bridge: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(decision)
    bridge_decision = downstream_bridge.get("decision", {}) if isinstance(downstream_bridge, Mapping) else {}
    if not isinstance(bridge_decision, Mapping):
        return merged
    if bridge_decision.get("bounded_frozen_canary") != "pass":
        return merged
    merged.update(
        {
            "downstream_training": "bounded-frozen-canary-passed",
            "bounded_downstream_canary": "ok",
            "large_training": "hold",
            "tiny_frozen_supervised_probe": "ok",
            "ts2vec_retraining": "no",
            "next_cpu_gate": "scale-frozen-downstream-probe",
            "read": (
                "bounded frozen-feature downstream canary passed on the downstream bridge proxy; "
                "large neural training and TS2Vec retraining remain held."
            ),
        }
    )
    return merged


def _top_non_target_policy(policy_final: Mapping[str, Mapping[str, Any]]) -> tuple[str | None, Mapping[str, Any] | None]:
    candidates = {
        policy: row
        for policy, row in policy_final.items()
        if not bool(row.get("uses_target_for_selection", False))
        and not _is_quality_stratified_random_replay(policy)
        and row.get("mean_final_gain") is not None
    }
    if not candidates:
        return None, None
    policy = max(candidates, key=lambda item: (float(candidates[item].get("mean_final_gain", 0.0)), item))
    return policy, candidates[policy]


def _top_downstream_policy(
    policy_final: Mapping[str, Mapping[str, Any]],
    *,
    exclude: set[str],
    metric: str,
) -> str | None:
    candidates: dict[str, float] = {}
    for policy, row in policy_final.items():
        if policy in exclude or _is_quality_stratified_random_replay(policy):
            continue
        metrics = row.get("metrics", {})
        if not isinstance(metrics, Mapping):
            continue
        metric_row = metrics.get(metric, {})
        if not isinstance(metric_row, Mapping) or metric_row.get("mean") is None:
            continue
        candidates[policy] = float(metric_row["mean"])
    if not candidates:
        return None
    return max(candidates, key=lambda policy: (candidates[policy], policy))


def _bootstrap_ci95(values: Sequence[float], *, seed: int, replicates: int) -> tuple[float | None, float | None]:
    clean = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=float)
    if clean.size == 0:
        return None, None
    if clean.size == 1 or replicates <= 0:
        value = float(np.mean(clean))
        return value, value
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, clean.size, size=(int(replicates), clean.size))
    means = np.mean(clean[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _win_fraction(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(float(value) > 0.0 for value in values) / len(values))


def _fraction(values: Sequence[float], predicate) -> float | None:
    if not values:
        return None
    return float(sum(1 for value in values if predicate(float(value))) / len(values))


def _ranked_items(rows: Mapping[str, Mapping[str, Any]], metric: str) -> list[tuple[str, Mapping[str, Any]]]:
    return sorted(rows.items(), key=lambda item: _none_low(item[1].get(metric)), reverse=True)


def _ranked_downstream_policy_items(rows: Any) -> list[tuple[str, Mapping[str, Any]]]:
    if not isinstance(rows, Mapping):
        return []

    def key(item: tuple[str, Mapping[str, Any]]) -> float:
        metrics = item[1].get("metrics", {})
        if not isinstance(metrics, Mapping):
            return float("-inf")
        metric_row = metrics.get("balanced_accuracy_gain", {})
        if not isinstance(metric_row, Mapping):
            return float("-inf")
        return _none_low(metric_row.get("mean"))

    return sorted(rows.items(), key=key, reverse=True)


def _is_quality_stratified_random_replay(policy_id: str) -> bool:
    if not str(policy_id).startswith(QUALITY_STRATIFIED_RANDOM_REPLAY_PREFIX):
        return False
    suffix = str(policy_id)[len(QUALITY_STRATIFIED_RANDOM_REPLAY_PREFIX) :]
    if suffix.endswith("_v1"):
        suffix = suffix[:-3]
    return suffix.isdigit()


def _none_low(value: Any) -> float:
    if value is None:
        return float("-inf")
    return float(value)


def _gt(value: Any, threshold: float) -> bool:
    return value is not None and float(value) > float(threshold)


def _ge(value: Any, threshold: float) -> bool:
    return value is not None and float(value) >= float(threshold)


def _le(value: Any, threshold: float) -> bool:
    return value is not None and float(value) <= float(threshold)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def _fmt_metric(metrics: Any, metric: str, field: str) -> str:
    if not isinstance(metrics, Mapping):
        return ""
    row = metrics.get(metric, {})
    if not isinstance(row, Mapping):
        return ""
    return _fmt(row.get(field))


def _pool_range(value: Any) -> str:
    if not isinstance(value, Mapping) or value.get("min") is None:
        return "unknown"
    return f"{int(value.get('min', 0))}-{int(value.get('max', 0))} clips, median {float(value.get('median', 0.0)):.1f}"


def _mean_finite(values: Sequence[Any]) -> float | None:
    clean = []
    for value in values:
        if value is None:
            continue
        numeric = float(value)
        if np.isfinite(numeric):
            clean.append(numeric)
    return mean(clean) if clean else None
