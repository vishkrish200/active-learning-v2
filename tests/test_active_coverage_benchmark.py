import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.active_benchmark import (
    BenchmarkClip,
    CoverageBenchmarkConfig,
    EpisodeSpec,
    build_coverage_decision_report,
    run_coverage_benchmark,
    write_coverage_reports,
)
from marginal_value.active_benchmark.coverage_decision import _decision, render_coverage_decision_markdown
from marginal_value.active_benchmark.coverage_reports import coverage_result_to_json


class ActiveCoverageBenchmarkTests(unittest.TestCase):
    def test_cpu_proxy_decision_never_unblocks_large_training(self):
        gates = [
            {"name": "episode_count", "status": "pass"},
            {"name": "large_training_baseline_delta_ci", "status": "pass"},
            {"name": "oracle_present", "status": "pass"},
            {"name": "oracle_headroom_sanity", "status": "pass"},
            {"name": "tiny_probe_random_delta", "status": "pass"},
            {"name": "tiny_probe_replay_null", "status": "pass"},
            {"name": "tiny_probe_direct_controls", "status": "pass"},
            {"name": "tiny_probe_stability", "status": "pass"},
            {"name": "deployable_top_policy", "status": "pass"},
        ]

        decision = _decision(gates)

        self.assertEqual(decision["downstream_training"], "bounded-frozen-canary-ok")
        self.assertEqual(decision["bounded_downstream_canary"], "ok")
        self.assertEqual(decision["large_training"], "hold")
        self.assertEqual(decision["ts2vec_retraining"], "no")
        self.assertIn("large neural training", decision["read"])

    def test_synthetic_cross_view_coverage_rewards_valid_target_covering_novelty(self):
        result = run_coverage_benchmark(
            _coverage_fixture_clips(),
            [_coverage_fixture_episode()],
            CoverageBenchmarkConfig(
                budgets=(1, 2),
                policies=(
                    "quality_only_v1",
                    "window_support_novelty_v1",
                    "window_kcenter_v1",
                    "ts2vec_support_novelty_v1",
                    "ts2vec_kcenter_v1",
                    "submitted_full_replay_v1",
                ),
                eval_views=("morph_stats_v1", "window", "ts2vec"),
                primary_eval_views=("morph_stats_v1",),
                ts2vec_view="ts2vec",
                window_view="window",
                eval_view_families={
                    "morph_stats_v1": "morphology",
                    "window": "window",
                    "ts2vec": "ts2vec",
                },
                quality_threshold=0.85,
                max_artifact_score=0.05,
                random_seed=11,
            ),
        )

        self.assertEqual(_selected_ids(result, "quality_only_v1", 1), ("pool_near_high_quality",))
        self.assertEqual(_selected_ids(result, "window_support_novelty_v1", 1), ("pool_target_cover",))
        self.assertEqual(_selected_ids(result, "window_kcenter_v1", 1), ("pool_target_cover",))
        self.assertEqual(_selected_ids(result, "ts2vec_support_novelty_v1", 1), ("pool_target_cover",))
        self.assertEqual(_selected_ids(result, "ts2vec_kcenter_v1", 1), ("pool_target_cover",))
        self.assertEqual(_selected_ids(result, "submitted_full_replay_v1", 1), ("pool_target_cover",))

        for policy_id in result.policies:
            self.assertNotIn("pool_far_invalid", _selected_ids(result, policy_id, 2))
            self.assertEqual(_metric(result, policy_id, 2, "__hygiene__", "selected_invalid_rate"), 0.0)

        quality_gain = _metric(result, "quality_only_v1", 1, "morph_stats_v1", "coverage_gain_rel")
        novelty_gain = _metric(result, "ts2vec_support_novelty_v1", 1, "morph_stats_v1", "coverage_gain_rel")
        kcenter_gain = _metric(result, "ts2vec_kcenter_v1", 1, "morph_stats_v1", "coverage_gain_rel")
        window_gain = _metric(result, "window_support_novelty_v1", 1, "morph_stats_v1", "coverage_gain_rel")
        submitted_gain = _metric(result, "submitted_full_replay_v1", 1, "morph_stats_v1", "coverage_gain_rel")

        self.assertLess(quality_gain, 0.05)
        self.assertGreater(novelty_gain, 0.95)
        self.assertGreater(kcenter_gain, 0.95)
        self.assertGreater(window_gain, 0.95)
        self.assertGreater(submitted_gain, 0.95)

        morph_row = _metric_row(result, "ts2vec_support_novelty_v1", 1, "morph_stats_v1", "coverage_gain_rel")
        ts2vec_row = _metric_row(result, "ts2vec_support_novelty_v1", 1, "ts2vec", "coverage_gain_rel")
        window_row = _metric_row(result, "window_support_novelty_v1", 1, "window", "coverage_gain_rel")
        self.assertTrue(morph_row.primary_eval)
        self.assertFalse(morph_row.selector_feature_overlap)
        self.assertFalse(ts2vec_row.primary_eval)
        self.assertTrue(ts2vec_row.selector_feature_overlap)
        self.assertFalse(window_row.primary_eval)
        self.assertTrue(window_row.selector_feature_overlap)

    def test_support_gap_window_probcover_selects_dense_novel_medoids_not_singleton_outliers(self):
        result = run_coverage_benchmark(
            _probcover_fixture_clips(),
            [_probcover_fixture_episode()],
            CoverageBenchmarkConfig(
                budgets=(1, 2),
                policies=("support_gap_window_probcover_v1",),
                eval_views=("window", "morph_stats_v1"),
                primary_eval_views=("morph_stats_v1",),
                eval_view_families={"window": "window", "morph_stats_v1": "morphology"},
                window_view="window",
                quality_threshold=0.85,
                max_artifact_score=0.05,
                random_seed=11,
            ),
        )

        self.assertEqual(_selected_ids(result, "support_gap_window_probcover_v1", 1), ("dense_a_center",))
        self.assertEqual(
            _selected_ids(result, "support_gap_window_probcover_v1", 2),
            ("dense_a_center", "dense_b_center"),
        )
        selected = set(_selected_ids(result, "support_gap_window_probcover_v1", 2))
        self.assertNotIn("singleton_far_outlier", selected)
        self.assertFalse(selected & {"dense_a_edge_high_quality", "near_support_high_quality"})

        selected_rows = [
            row
            for row in result.selected_rows
            if row.policy_id == "support_gap_window_probcover_v1" and row.budget_k == 2
        ]
        self.assertTrue(all(row.valid for row in selected_rows))
        self.assertTrue(all(row.passed_artifact_gate for row in selected_rows))
        self.assertEqual(_metric(result, "support_gap_window_probcover_v1", 2, "__hygiene__", "selected_invalid_rate"), 0.0)
        self.assertEqual(
            _metric(result, "support_gap_window_probcover_v1", 2, "__hygiene__", "selected_duplicate_clip_count"),
            0.0,
        )

    def test_support_gap_window_probcover_obeys_source_group_leakage_hygiene(self):
        episode = EpisodeSpec(
            episode_id="episode-smoke",
            fold_id=0,
            support_ids=("support_origin",),
            candidate_ids=("dense_a_center",),
            target_ids=("target_a",),
            support_group_ids=("support_origin_group",),
            candidate_group_ids=("shared_group",),
            target_group_ids=("shared_group",),
        )

        with self.assertRaisesRegex(ValueError, "source-group leakage"):
            run_coverage_benchmark(
                _probcover_fixture_clips(),
                [episode],
                CoverageBenchmarkConfig(
                    budgets=(1,),
                    policies=("support_gap_window_probcover_v1",),
                    eval_views=("window",),
                    primary_eval_views=("window",),
                    window_view="window",
                ),
            )

    def test_coverage_benchmark_rejects_source_group_leakage_between_roles(self):
        episode = EpisodeSpec(
            episode_id="episode-leaky",
            fold_id=0,
            support_ids=("support_left",),
            candidate_ids=("pool_target_cover",),
            target_ids=("target_a",),
            support_group_ids=("support_group",),
            candidate_group_ids=("shared_group",),
            target_group_ids=("shared_group",),
        )

        with self.assertRaisesRegex(ValueError, "source-group leakage"):
            run_coverage_benchmark(
                _coverage_fixture_clips(),
                [episode],
                CoverageBenchmarkConfig(
                    budgets=(1,),
                    policies=("ts2vec_support_novelty_v1",),
                    eval_views=("morph_stats_v1",),
                    primary_eval_views=("morph_stats_v1",),
                    ts2vec_view="ts2vec",
                    window_view="window",
                ),
            )

    def test_coverage_reports_round_trip_required_rows(self):
        result = run_coverage_benchmark(
            _coverage_fixture_clips(),
            [_coverage_fixture_episode()],
            CoverageBenchmarkConfig(
                budgets=(1,),
                policies=("quality_only_v1", "ts2vec_support_novelty_v1"),
                eval_views=("morph_stats_v1", "ts2vec"),
                primary_eval_views=("morph_stats_v1",),
                ts2vec_view="ts2vec",
                window_view="window",
                eval_view_families={"morph_stats_v1": "morphology", "ts2vec": "ts2vec"},
            ),
        )

        with TemporaryDirectory() as tmpdir:
            paths = write_coverage_reports(result, tmpdir)
            payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

        self.assertEqual(payload["n_episodes"], 1)
        self.assertIn("metric_rows", payload)
        self.assertIn("selected_rows", payload)
        self.assertIn("policy_summary", payload)
        self.assertTrue(payload["metric_rows"])
        self.assertTrue(payload["selected_rows"])
        self.assertIn("# Blind Target Coverage Benchmark", markdown)
        self.assertIn("ts2vec_support_novelty_v1", markdown)
        self.assertIn("coverage_gain_rel", markdown)

    def test_coverage_decision_report_adds_episode_bootstrap_and_oracle_capture(self):
        result = run_coverage_benchmark(
            _coverage_fixture_clips(),
            [_coverage_fixture_episode()],
            CoverageBenchmarkConfig(
                budgets=(1,),
                policies=("quality_only_v1", "ts2vec_kcenter_v1", "oracle_greedy_eval_view_v1"),
                eval_views=("morph_stats_v1", "ts2vec"),
                primary_eval_views=("morph_stats_v1",),
                ts2vec_view="ts2vec",
                window_view="window",
                eval_view_families={"morph_stats_v1": "morphology", "ts2vec": "ts2vec"},
                quality_threshold=0.85,
                max_artifact_score=0.05,
                random_seed=11,
            ),
        )

        report = build_coverage_decision_report(
            [coverage_result_to_json(result)],
            report_names=["seed_fixture"],
            baseline_policy="quality_only_v1",
            bootstrap_replicates=10,
        )

        self.assertEqual(report["coverage_units"]["independent_episode_count"], 1)
        self.assertEqual(report["coverage_units"]["final_budget"], 1)
        self.assertEqual(report["policy_final_summary"]["oracle_greedy_eval_view_v1"]["uses_target_for_selection"], True)
        self.assertGreater(report["pairwise_vs_baseline"]["ts2vec_kcenter_v1"]["mean_delta"], 0.90)
        self.assertGreater(report["oracle_capture_vs_baseline"]["ts2vec_kcenter_v1"]["mean_oracle_capture"], 0.99)
        self.assertEqual(report["decision"]["downstream_training"], "hold")

    def test_exact_coverage_oracle_selects_global_budget_set(self):
        result = run_coverage_benchmark(
            _exact_oracle_fixture_clips(),
            [_exact_oracle_fixture_episode()],
            CoverageBenchmarkConfig(
                budgets=(2,),
                policies=("quality_only_v1", "oracle_exact_coverage_v1"),
                eval_views=("morph_stats_v1",),
                primary_eval_views=("morph_stats_v1",),
                eval_view_families={"morph_stats_v1": "morphology"},
                quality_threshold=0.85,
                max_artifact_score=0.05,
                random_seed=11,
            ),
        )

        self.assertEqual(_selected_ids(result, "quality_only_v1", 2), ("candidate_mid_high_quality", "candidate_target_left"))
        self.assertEqual(
            _selected_ids(result, "oracle_exact_coverage_v1", 2),
            ("candidate_target_left", "candidate_target_right"),
        )
        quality_gain = _metric(result, "quality_only_v1", 2, "morph_stats_v1", "coverage_gain_rel")
        exact_gain = _metric(result, "oracle_exact_coverage_v1", 2, "morph_stats_v1", "coverage_gain_rel")
        self.assertLess(quality_gain, 0.70)
        self.assertGreater(exact_gain, 0.99)
        exact_row = _metric_row(result, "oracle_exact_coverage_v1", 2, "morph_stats_v1", "coverage_gain_rel")
        self.assertTrue(exact_row.uses_target_for_selection)

    def test_quality_stratified_random_replays_are_independent_named_controls(self):
        config = CoverageBenchmarkConfig(
            budgets=(2,),
            policies=(
                "quality_stratified_random_v1",
                "quality_stratified_random_replay_000_v1",
                "quality_stratified_random_replay_001_v1",
            ),
            eval_views=("morph_stats_v1",),
            primary_eval_views=("morph_stats_v1",),
            eval_view_families={"morph_stats_v1": "morphology"},
            quality_threshold=0.85,
            max_artifact_score=0.05,
            random_seed=19,
        )

        result_a = run_coverage_benchmark(_random_replay_fixture_clips(), [_random_replay_fixture_episode()], config)
        result_b = run_coverage_benchmark(_random_replay_fixture_clips(), [_random_replay_fixture_episode()], config)

        self.assertEqual(result_a.policies, config.policies)
        self.assertEqual(
            _selected_ids(result_a, "quality_stratified_random_replay_000_v1", 2),
            _selected_ids(result_b, "quality_stratified_random_replay_000_v1", 2),
        )
        self.assertNotEqual(
            _selected_ids(result_a, "quality_stratified_random_replay_000_v1", 2),
            _selected_ids(result_a, "quality_stratified_random_replay_001_v1", 2),
        )

    def test_coverage_decision_report_aggregates_downstream_bridge_proxy(self):
        result = run_coverage_benchmark(
            _coverage_fixture_clips(),
            [_coverage_fixture_episode()],
            CoverageBenchmarkConfig(
                budgets=(1,),
                policies=("quality_stratified_random_v1", "ts2vec_kcenter_v1"),
                eval_views=("morph_stats_v1",),
                primary_eval_views=("morph_stats_v1",),
                ts2vec_view="ts2vec",
                window_view="window",
                random_seed=11,
            ),
        )
        downstream_report = {
            "rows": [
                _downstream_row("quality_stratified_random_v1", "window", balanced=0.20, nll=1.0, discovery=0.0),
                _downstream_row("quality_stratified_random_v1", "raw_shape_stats", balanced=0.20, nll=1.0, discovery=0.0),
                _downstream_row("ts2vec_kcenter_v1", "window", balanced=0.70, nll=5.0, discovery=1.0),
                _downstream_row("ts2vec_kcenter_v1", "raw_shape_stats", balanced=0.50, nll=3.0, discovery=1.0),
            ]
        }

        report = build_coverage_decision_report(
            [coverage_result_to_json(result)],
            report_names=["seed_fixture"],
            baseline_policy="quality_stratified_random_v1",
            downstream_reports=[downstream_report],
            bootstrap_replicates=10,
        )
        proxy = report["downstream_bridge_proxy"]

        self.assertEqual(proxy["units"]["independent_episode_count"], 1)
        self.assertEqual(proxy["units"]["final_budget"], 1)
        self.assertEqual(proxy["decision"]["top_deployable_policy_by_balanced_accuracy"], "ts2vec_kcenter_v1")
        ts2vec_summary = proxy["policy_final_summary"]["ts2vec_kcenter_v1"]
        ts2vec_delta = proxy["pairwise_vs_baseline"]["ts2vec_kcenter_v1"]["balanced_accuracy_gain"]
        self.assertAlmostEqual(ts2vec_summary["metrics"]["balanced_accuracy_gain"]["mean"], 0.60)
        self.assertAlmostEqual(ts2vec_delta["mean_delta"], 0.40)
        self.assertAlmostEqual(ts2vec_delta["bootstrap_ci95_low"], 0.40)
        self.assertAlmostEqual(
            proxy["pairwise_vs_baseline"]["ts2vec_kcenter_v1"]["target_family_discovery_rate"]["mean_delta"],
            1.0,
        )
        self.assertIn("## Downstream Bridge Proxy", render_coverage_decision_markdown(report))

    def test_downstream_bridge_proxy_marks_bounded_frozen_canary_pass(self):
        result = run_coverage_benchmark(
            _coverage_fixture_clips(),
            [_coverage_fixture_episode()],
            CoverageBenchmarkConfig(
                budgets=(1,),
                policies=("quality_stratified_random_v1", "ts2vec_kcenter_v1"),
                eval_views=("morph_stats_v1",),
                primary_eval_views=("morph_stats_v1",),
                ts2vec_view="ts2vec",
                window_view="window",
                random_seed=11,
            ),
        )
        rows = []
        for episode_index in range(40):
            episode_id = f"episode-{episode_index:03d}"
            rows.extend(
                [
                    _downstream_row(
                        "quality_stratified_random_v1",
                        "window",
                        episode_id=episode_id,
                        balanced=0.05,
                        nll=0.01,
                        discovery=0.0,
                    ),
                    _downstream_row(
                        "quality_stratified_random_v1",
                        "raw_shape_stats",
                        episode_id=episode_id,
                        balanced=0.07,
                        nll=0.02,
                        discovery=0.0,
                    ),
                    _downstream_row(
                        "ts2vec_kcenter_v1",
                        "window",
                        episode_id=episode_id,
                        balanced=0.14,
                        nll=0.05,
                        discovery=1.0,
                    ),
                    _downstream_row(
                        "ts2vec_kcenter_v1",
                        "raw_shape_stats",
                        episode_id=episode_id,
                        balanced=0.16,
                        nll=0.06,
                        discovery=1.0,
                    ),
                ]
            )
        downstream_report = {"rows": rows}

        report = build_coverage_decision_report(
            [coverage_result_to_json(result)],
            report_names=["seed_fixture"],
            baseline_policy="quality_stratified_random_v1",
            downstream_reports=[downstream_report],
            bootstrap_replicates=10,
        )
        proxy = report["downstream_bridge_proxy"]

        self.assertEqual(proxy["units"]["independent_episode_count"], 40)
        self.assertEqual(proxy["decision"]["bounded_frozen_canary"], "pass")
        self.assertEqual(proxy["decision"]["large_training"], "hold")
        self.assertEqual(proxy["decision"]["ts2vec_retraining"], "no")
        self.assertEqual(proxy["decision"]["focal_policy"], "ts2vec_kcenter_v1")
        self.assertEqual(report["decision"]["bounded_downstream_canary"], "ok")
        self.assertEqual(report["decision"]["downstream_training"], "bounded-frozen-canary-passed")
        self.assertEqual(report["decision"]["large_training"], "hold")
        self.assertEqual(report["decision"]["ts2vec_retraining"], "no")
        self.assertGreaterEqual(proxy["decision"]["balanced_accuracy_delta_vs_baseline"], 0.08)
        self.assertGreater(proxy["decision"]["balanced_accuracy_ci95_low_vs_baseline"], 0.0)
        self.assertEqual(
            proxy["direct_pairwise_controls"]["ts2vec_kcenter_v1__vs__quality_stratified_random_v1"][
                "balanced_accuracy_gain"
            ]["paired_episode_count"],
            40,
        )
        markdown = render_coverage_decision_markdown(report)
        self.assertIn("bounded frozen canary: `pass`", markdown)
        self.assertIn("### Direct Downstream Controls", markdown)

    def test_coverage_decision_report_splits_exact_and_target_family_oracles(self):
        result = run_coverage_benchmark(
            _exact_oracle_fixture_clips(),
            [_exact_oracle_fixture_episode()],
            CoverageBenchmarkConfig(
                budgets=(2,),
                policies=("quality_only_v1", "oracle_exact_coverage_v1", "oracle_greedy_target_family_v1"),
                eval_views=("morph_stats_v1", "window"),
                primary_eval_views=("morph_stats_v1",),
                eval_view_families={"morph_stats_v1": "morphology", "window": "window"},
                source_family_count=3,
                source_family_label_view="window",
                quality_threshold=0.85,
                max_artifact_score=0.05,
                random_seed=11,
            ),
        )

        report = build_coverage_decision_report(
            [coverage_result_to_json(result)],
            report_names=["seed_fixture"],
            baseline_policy="quality_only_v1",
            oracle_policy="oracle_exact_coverage_v1",
            bootstrap_replicates=10,
        )

        diagnostics = report["oracle_diagnostics"]
        self.assertEqual(diagnostics["exact_coverage_policy"], "oracle_exact_coverage_v1")
        self.assertEqual(diagnostics["target_family_oracle_policy"], "oracle_greedy_target_family_v1")
        self.assertEqual(diagnostics["exact_coverage_vs_top_deployable"]["status"], "pass")
        self.assertGreaterEqual(diagnostics["exact_coverage_vs_top_deployable"]["mean_gap"], 0.30)
        markdown = render_coverage_decision_markdown(report)
        self.assertIn("## Oracle Diagnostics", markdown)
        self.assertIn("exact final-budget coverage objective", markdown)

    def test_coverage_decision_report_adds_stable_oracle_and_tiny_probe_diagnostics(self):
        reports = [
            _decision_diagnostic_report(
                [
                    ("episode-a", 0, {"quality_stratified_random_v1": 0.10, "quality_only_v1": 0.05, "window_kcenter_v1": 0.12, "submitted_full_replay_v1": 0.15, "ts2vec_kcenter_v1": 0.20, "oracle_exact_coverage_v1": 0.30, "quality_stratified_random_replay_000_v1": 0.08, "quality_stratified_random_replay_001_v1": 0.09}),
                    ("episode-b", 1, {"quality_stratified_random_v1": 0.09, "quality_only_v1": 0.03, "window_kcenter_v1": 0.11, "submitted_full_replay_v1": 0.13, "ts2vec_kcenter_v1": 0.18, "oracle_exact_coverage_v1": 0.25, "quality_stratified_random_replay_000_v1": 0.07, "quality_stratified_random_replay_001_v1": 0.10}),
                ]
            ),
            _decision_diagnostic_report(
                [
                    ("episode-c", 0, {"quality_stratified_random_v1": 0.10, "quality_only_v1": 0.04, "window_kcenter_v1": 0.13, "submitted_full_replay_v1": 0.16, "ts2vec_kcenter_v1": 0.19, "oracle_exact_coverage_v1": 0.28, "quality_stratified_random_replay_000_v1": 0.09, "quality_stratified_random_replay_001_v1": 0.08}),
                    ("episode-d", 1, {"quality_stratified_random_v1": 0.08, "quality_only_v1": 0.02, "window_kcenter_v1": 0.12, "submitted_full_replay_v1": 0.14, "ts2vec_kcenter_v1": 0.17, "oracle_exact_coverage_v1": 0.26, "quality_stratified_random_replay_000_v1": 0.06, "quality_stratified_random_replay_001_v1": 0.09}),
                ]
            ),
        ]

        report = build_coverage_decision_report(
            reports,
            report_names=["seed_1", "seed_2"],
            baseline_policy="quality_stratified_random_v1",
            oracle_policy="oracle_exact_coverage_v1",
            bootstrap_replicates=10,
        )

        self.assertEqual(
            report["oracle_capture_vs_baseline"]["ts2vec_kcenter_v1"]["decision_use"],
            "deprecated_debug_only",
        )
        oracle_stability = report["oracle_stability_diagnostics"]
        self.assertEqual(oracle_stability["baseline_source"], "quality_stratified_random_replay_episode_mean")
        self.assertEqual(oracle_stability["oracle_headroom"]["positive_fraction"], 1.0)
        self.assertIn("ts2vec_kcenter_v1", oracle_stability["policy_metrics"])
        self.assertGreater(oracle_stability["policy_metrics"]["ts2vec_kcenter_v1"]["bounded_capture_mean"], 0.0)

        direct = report["direct_pairwise_controls"]
        self.assertGreater(direct["ts2vec_kcenter_v1__vs__window_kcenter_v1"]["mean_delta"], 0.0)
        self.assertGreater(direct["ts2vec_kcenter_v1__vs__submitted_full_replay_v1"]["median_delta"], 0.0)
        self.assertAlmostEqual(report["random_replay_diagnostics"]["top_deployable_replay_null_pvalue_plus_one"], 1.0 / 3.0)
        self.assertEqual(report["acquisition_stability_diagnostics"]["positive_seed_fraction"], 1.0)
        self.assertEqual(report["selected_set_audit"]["ts2vec_kcenter_v1"]["invalid_rate"], 0.0)
        unlock = report["training_unlock_criteria"]["expanded_cpu_validation"]
        self.assertEqual(unlock["minimum_independent_episodes_per_pool"], 80)
        self.assertEqual(unlock["minimum_candidate_clips_max"], 72)
        self.assertEqual(unlock["random_delta_mean_min"], 0.04)
        self.assertEqual(unlock["random_delta_ci95_low_min"], 0.025)
        self.assertEqual(unlock["random_delta_win_fraction_min"], 0.70)
        self.assertEqual(unlock["replay_null_pvalue_max"], 0.01)
        self.assertEqual(unlock["window_kcenter_delta_mean_min"], 0.015)
        self.assertEqual(unlock["submitted_full_replay_delta_ci95_low_min"], 0.0)

        markdown = render_coverage_decision_markdown(report)
        self.assertIn("## Stable Oracle Diagnostics", markdown)
        self.assertIn("## Direct Paired Control Comparisons", markdown)
        self.assertIn("## Expanded Training Unlock Criteria", markdown)
        self.assertIn("candidate clips max >= `72`", markdown)
        self.assertIn("## Acquisition Stability", markdown)
        self.assertIn("## Selected Set Audit", markdown)

    def test_target_family_oracle_selects_bridge_candidates(self):
        clips = [
            _clip("support_a", "fam_a_worker_support", (0.0, 0.0)),
            _clip("candidate_bridge", "fam_c_worker_candidate", (0.0, 10.0), quality=0.91),
            _clip("candidate_distractor", "fam_b_worker_candidate", (9.0, 0.0), quality=0.99),
            _clip("target_c", "fam_c_worker_target", (0.0, 11.0)),
        ]
        episode = EpisodeSpec(
            episode_id="episode-smoke",
            fold_id=0,
            support_ids=("support_a",),
            candidate_ids=("candidate_distractor", "candidate_bridge"),
            target_ids=("target_c",),
            support_group_ids=("fam_a_worker_support",),
            candidate_group_ids=("fam_b_worker_candidate", "fam_c_worker_candidate"),
            target_group_ids=("fam_c_worker_target",),
        )

        result = run_coverage_benchmark(
            clips,
            [episode],
            CoverageBenchmarkConfig(
                budgets=(1,),
                policies=("quality_only_v1", "oracle_greedy_target_family_v1"),
                eval_views=("window",),
                primary_eval_views=("window",),
                source_family_count=3,
                source_family_label_view="window",
            ),
        )

        self.assertEqual(_selected_ids(result, "quality_only_v1", 1), ("candidate_distractor",))
        self.assertEqual(_selected_ids(result, "oracle_greedy_target_family_v1", 1), ("candidate_bridge",))
        oracle_row = _metric_row(result, "oracle_greedy_target_family_v1", 1, "window", "coverage_gain_rel")
        self.assertTrue(oracle_row.uses_target_for_selection)
        self.assertFalse(oracle_row.primary_eval)


def _coverage_fixture_episode() -> EpisodeSpec:
    return EpisodeSpec(
        episode_id="episode-smoke",
        fold_id=0,
        support_ids=("support_left", "support_right"),
        candidate_ids=("pool_near_high_quality", "pool_target_cover", "pool_far_invalid"),
        target_ids=("target_a", "target_b"),
        support_group_ids=("support_left_group", "support_right_group"),
        candidate_group_ids=("pool_near_group", "pool_target_group", "pool_invalid_group"),
        target_group_ids=("target_a_group", "target_b_group"),
    )


def _coverage_fixture_clips() -> list[BenchmarkClip]:
    return [
        _clip("support_left", "support_left_group", (-0.5, 0.0)),
        _clip("support_right", "support_right_group", (0.5, 0.0)),
        _clip("pool_near_high_quality", "pool_near_group", (0.0, 0.0), quality=0.99),
        _clip("pool_target_cover", "pool_target_group", (10.0, 0.1), quality=0.91),
        _clip("pool_far_invalid", "pool_invalid_group", (10.0, 0.0), quality=0.20, artifact=0.80),
        _clip("target_a", "target_a_group", (10.0, 0.0)),
        _clip("target_b", "target_b_group", (10.0, 0.2)),
    ]


def _exact_oracle_fixture_episode() -> EpisodeSpec:
    return EpisodeSpec(
        episode_id="episode-smoke",
        fold_id=0,
        support_ids=("support_origin",),
        candidate_ids=("candidate_mid_high_quality", "candidate_target_left", "candidate_target_right"),
        target_ids=("target_left", "target_right"),
        support_group_ids=("support_origin_group",),
        candidate_group_ids=("candidate_mid_group", "candidate_left_group", "candidate_right_group"),
        target_group_ids=("target_left_group", "target_right_group"),
    )


def _exact_oracle_fixture_clips() -> list[BenchmarkClip]:
    return [
        _clip("support_origin", "support_origin_group", (0.0, 0.0)),
        _clip("candidate_mid_high_quality", "candidate_mid_group", (5.0, 5.0), quality=0.99),
        _clip("candidate_target_left", "candidate_left_group", (10.0, 0.0), quality=0.91),
        _clip("candidate_target_right", "candidate_right_group", (0.0, 10.0), quality=0.90),
        _clip("target_left", "target_left_group", (10.0, 0.0)),
        _clip("target_right", "target_right_group", (0.0, 10.0)),
    ]


def _random_replay_fixture_episode() -> EpisodeSpec:
    return EpisodeSpec(
        episode_id="episode-smoke",
        fold_id=0,
        support_ids=("support_origin",),
        candidate_ids=("candidate_a", "candidate_b", "candidate_c", "candidate_d", "candidate_e"),
        target_ids=("target",),
        support_group_ids=("support_origin_group",),
        candidate_group_ids=("candidate_a_group", "candidate_b_group", "candidate_c_group", "candidate_d_group", "candidate_e_group"),
        target_group_ids=("target_group",),
    )


def _random_replay_fixture_clips() -> list[BenchmarkClip]:
    return [
        _clip("support_origin", "support_origin_group", (0.0, 0.0)),
        _clip("candidate_a", "candidate_a_group", (1.0, 0.0), quality=0.99),
        _clip("candidate_b", "candidate_b_group", (2.0, 0.0), quality=0.98),
        _clip("candidate_c", "candidate_c_group", (3.0, 0.0), quality=0.97),
        _clip("candidate_d", "candidate_d_group", (4.0, 0.0), quality=0.96),
        _clip("candidate_e", "candidate_e_group", (5.0, 0.0), quality=0.95),
        _clip("target", "target_group", (5.0, 0.0)),
    ]


def _probcover_fixture_episode() -> EpisodeSpec:
    return EpisodeSpec(
        episode_id="episode-smoke",
        fold_id=0,
        support_ids=("support_origin", "support_x", "support_y"),
        candidate_ids=(
            "near_support_high_quality",
            "dense_a_edge_high_quality",
            "dense_a_center",
            "dense_a_edge_two",
            "dense_b_edge_high_quality",
            "dense_b_center",
            "dense_b_edge_two",
            "singleton_far_outlier",
            "dense_invalid_artifact",
        ),
        target_ids=("target_a", "target_b"),
        support_group_ids=("support_origin_group", "support_x_group", "support_y_group"),
        candidate_group_ids=(
            "near_support_group",
            "dense_a_edge_high_quality_group",
            "dense_a_center_group",
            "dense_a_edge_two_group",
            "dense_b_edge_high_quality_group",
            "dense_b_center_group",
            "dense_b_edge_two_group",
            "singleton_far_outlier_group",
            "dense_invalid_artifact_group",
        ),
        target_group_ids=("target_a_group", "target_b_group"),
    )


def _probcover_fixture_clips() -> list[BenchmarkClip]:
    return [
        _clip("support_origin", "support_origin_group", (0.0, 0.0)),
        _clip("support_x", "support_x_group", (1.0, 0.0)),
        _clip("support_y", "support_y_group", (0.0, 1.0)),
        _clip("near_support_high_quality", "near_support_group", (0.1, 0.1), quality=0.99),
        _clip("dense_a_edge_high_quality", "dense_a_edge_high_quality_group", (5.25, 5.0), quality=0.99),
        _clip("dense_a_center", "dense_a_center_group", (5.0, 5.0), quality=0.91),
        _clip("dense_a_edge_two", "dense_a_edge_two_group", (5.0, 5.25), quality=0.92),
        _clip("dense_b_edge_high_quality", "dense_b_edge_high_quality_group", (8.25, 0.0), quality=0.98),
        _clip("dense_b_center", "dense_b_center_group", (8.0, 0.0), quality=0.91),
        _clip("dense_b_edge_two", "dense_b_edge_two_group", (8.0, 0.25), quality=0.92),
        _clip("singleton_far_outlier", "singleton_far_outlier_group", (20.0, 20.0), quality=1.0),
        _clip("dense_invalid_artifact", "dense_invalid_artifact_group", (5.05, 5.05), quality=0.99, artifact=0.50),
        _clip("target_a", "target_a_group", (5.0, 5.1)),
        _clip("target_b", "target_b_group", (8.0, 0.1)),
    ]


def _clip(
    sample_id: str,
    group_id: str,
    point: tuple[float, float],
    *,
    quality: float = 1.0,
    artifact: float = 0.0,
) -> BenchmarkClip:
    vector = np.asarray(point, dtype=float)
    return BenchmarkClip(
        sample_id=sample_id,
        source_group_id=group_id,
        embeddings={
            "ts2vec": vector,
            "window": vector,
            "morph_stats_v1": vector,
        },
        quality_score=quality,
        artifact_score=artifact,
    )


def _downstream_row(
    policy_id: str,
    representation: str,
    *,
    episode_id: str = "episode-smoke",
    balanced: float,
    nll: float,
    discovery: float,
) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "fold_id": 0,
        "policy_id": policy_id,
        "budget_k": 1,
        "representation": representation,
        "target_family_discovery_rate": discovery,
        "after_known_target_fraction": discovery,
        "balanced_accuracy_gain": balanced,
        "nll_reduction": nll,
    }


def _decision_diagnostic_report(episodes: list[tuple[str, int, dict[str, float]]]) -> dict[str, object]:
    metric_rows = []
    selected_rows = []
    rounds = []
    for episode_id, fold_id, gains in episodes:
        for policy_id, gain in gains.items():
            metric_rows.append(
                {
                    "episode_id": episode_id,
                    "fold_id": fold_id,
                    "budget_k": 4,
                    "policy_id": policy_id,
                    "eval_view": "heldout",
                    "metric_name": "coverage_gain_rel",
                    "metric_value": gain,
                    "primary_eval": True,
                    "selector_feature_overlap": False,
                    "uses_target_for_selection": policy_id == "oracle_exact_coverage_v1",
                    "higher_is_better": True,
                }
            )
            selected_rows.append(
                {
                    "episode_id": episode_id,
                    "fold_id": fold_id,
                    "budget_k": 4,
                    "policy_id": policy_id,
                    "rank_index": 1,
                    "sample_id": f"{policy_id}-{episode_id}",
                    "source_group_id": f"{policy_id}-source",
                    "quality_score": 0.95,
                    "artifact_score": 0.0,
                    "valid": True,
                    "passed_artifact_gate": True,
                    "score": gain,
                }
            )
            rounds.append(
                {
                    "episode_id": episode_id,
                    "fold_id": fold_id,
                    "budget_k": 4,
                    "policy_id": policy_id,
                    "candidate_count": 36,
                    "eligible_candidate_count": 36,
                    "selected_count": 4,
                    "primary_coverage_gain_rel": gain,
                    "status": "ok",
                }
            )
    return {
        "budgets": [4],
        "config": {
            "budgets": [4],
            "primary_eval_views": ["heldout"],
            "eval_views": ["heldout"],
        },
        "metric_rows": metric_rows,
        "selected_rows": selected_rows,
        "rounds": rounds,
    }


def _selected_ids(result, policy_id: str, budget_k: int) -> tuple[str, ...]:
    rows = [
        row
        for row in result.selected_rows
        if row.episode_id == "episode-smoke" and row.policy_id == policy_id and row.budget_k == budget_k
    ]
    return tuple(row.sample_id for row in sorted(rows, key=lambda row: row.rank_index))


def _metric(result, policy_id: str, budget_k: int, eval_view: str, metric_name: str) -> float:
    return _metric_row(result, policy_id, budget_k, eval_view, metric_name).metric_value


def _metric_row(result, policy_id: str, budget_k: int, eval_view: str, metric_name: str):
    matches = [
        row
        for row in result.metric_rows
        if row.episode_id == "episode-smoke"
        and row.policy_id == policy_id
        and row.budget_k == budget_k
        and row.eval_view == eval_view
        and row.metric_name == metric_name
    ]
    if len(matches) != 1:
        raise AssertionError(f"Expected one metric row for {(policy_id, budget_k, eval_view, metric_name)}, found {len(matches)}.")
    return matches[0]


if __name__ == "__main__":
    unittest.main()
