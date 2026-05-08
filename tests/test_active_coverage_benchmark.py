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
from marginal_value.active_benchmark.coverage_decision import render_coverage_decision_markdown
from marginal_value.active_benchmark.coverage_reports import coverage_result_to_json


class ActiveCoverageBenchmarkTests(unittest.TestCase):
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
    balanced: float,
    nll: float,
    discovery: float,
) -> dict[str, object]:
    return {
        "episode_id": "episode-smoke",
        "fold_id": 0,
        "policy_id": policy_id,
        "budget_k": 1,
        "representation": representation,
        "target_family_discovery_rate": discovery,
        "after_known_target_fraction": discovery,
        "balanced_accuracy_gain": balanced,
        "nll_reduction": nll,
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
