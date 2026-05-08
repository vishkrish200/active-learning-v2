import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.active_benchmark import (
    BenchmarkClip,
    CoverageBenchmarkConfig,
    EpisodeSpec,
    run_coverage_benchmark,
    write_coverage_reports,
)


class ActiveCoverageBenchmarkTests(unittest.TestCase):
    def test_synthetic_cross_view_coverage_rewards_valid_target_covering_novelty(self):
        result = run_coverage_benchmark(
            _coverage_fixture_clips(),
            [_coverage_fixture_episode()],
            CoverageBenchmarkConfig(
                budgets=(1, 2),
                policies=(
                    "quality_only_v1",
                    "ts2vec_support_novelty_v1",
                    "ts2vec_kcenter_v1",
                    "submitted_full_replay_v1",
                ),
                eval_views=("morph_stats_v1", "ts2vec"),
                primary_eval_views=("morph_stats_v1",),
                ts2vec_view="ts2vec",
                window_view="window",
                eval_view_families={
                    "morph_stats_v1": "morphology",
                    "ts2vec": "ts2vec",
                    "window": "window",
                },
                quality_threshold=0.85,
                max_artifact_score=0.05,
                random_seed=11,
            ),
        )

        self.assertEqual(_selected_ids(result, "quality_only_v1", 1), ("pool_near_high_quality",))
        self.assertEqual(_selected_ids(result, "ts2vec_support_novelty_v1", 1), ("pool_target_cover",))
        self.assertEqual(_selected_ids(result, "ts2vec_kcenter_v1", 1), ("pool_target_cover",))
        self.assertEqual(_selected_ids(result, "submitted_full_replay_v1", 1), ("pool_target_cover",))

        for policy_id in result.policies:
            self.assertNotIn("pool_far_invalid", _selected_ids(result, policy_id, 2))
            self.assertEqual(_metric(result, policy_id, 2, "__hygiene__", "selected_invalid_rate"), 0.0)

        quality_gain = _metric(result, "quality_only_v1", 1, "morph_stats_v1", "coverage_gain_rel")
        novelty_gain = _metric(result, "ts2vec_support_novelty_v1", 1, "morph_stats_v1", "coverage_gain_rel")
        kcenter_gain = _metric(result, "ts2vec_kcenter_v1", 1, "morph_stats_v1", "coverage_gain_rel")
        submitted_gain = _metric(result, "submitted_full_replay_v1", 1, "morph_stats_v1", "coverage_gain_rel")

        self.assertLess(quality_gain, 0.05)
        self.assertGreater(novelty_gain, 0.95)
        self.assertGreater(kcenter_gain, 0.95)
        self.assertGreater(submitted_gain, 0.95)

        morph_row = _metric_row(result, "ts2vec_support_novelty_v1", 1, "morph_stats_v1", "coverage_gain_rel")
        ts2vec_row = _metric_row(result, "ts2vec_support_novelty_v1", 1, "ts2vec", "coverage_gain_rel")
        self.assertTrue(morph_row.primary_eval)
        self.assertFalse(morph_row.selector_feature_overlap)
        self.assertFalse(ts2vec_row.primary_eval)
        self.assertTrue(ts2vec_row.selector_feature_overlap)

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
