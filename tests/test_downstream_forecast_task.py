import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.active_benchmark import BenchmarkClip, EpisodeSpec
from marginal_value.active_benchmark.coverage_runner import (
    CoverageBenchmarkConfig,
    CoverageRunResult,
    CoverageSelectedRow,
)
from marginal_value.active_benchmark.downstream_forecast_task import (
    build_downstream_forecast_report,
    build_downstream_forecast_rows,
    write_downstream_forecast_reports,
)


class DownstreamForecastTaskTests(unittest.TestCase):
    def test_forecast_rows_retrain_after_selected_clips_and_reduce_target_loss(self):
        samples_by_id = {
            "support_flat": _flat_samples(),
            "candidate_target_like": _sine_samples(phase=0.02),
            "candidate_irrelevant": _flat_samples(offset=3.0),
            "target_sine": _sine_samples(phase=0.0),
        }
        result = _coverage_result()

        rows = build_downstream_forecast_rows(
            result,
            samples_by_id,
            history_steps=4,
            horizon_steps=1,
            ridge_alpha=1.0e-4,
            max_windows_per_clip=64,
        )
        by_policy_budget = {(row["policy_id"], row["budget_k"]): row for row in rows}

        self.assertEqual(sorted({row["budget_k"] for row in rows}), [0, 1])
        self.assertEqual(by_policy_budget[("ts2vec_kcenter_v1", 0)]["selected_ids"], [])
        self.assertAlmostEqual(
            by_policy_budget[("ts2vec_kcenter_v1", 0)]["after_mse"],
            by_policy_budget[("quality_stratified_random_v1", 0)]["after_mse"],
        )
        self.assertLess(
            by_policy_budget[("ts2vec_kcenter_v1", 1)]["after_mse"],
            by_policy_budget[("quality_stratified_random_v1", 1)]["after_mse"],
        )
        self.assertGreater(
            by_policy_budget[("ts2vec_kcenter_v1", 1)]["relative_mse_reduction"],
            by_policy_budget[("quality_stratified_random_v1", 1)]["relative_mse_reduction"],
        )

    def test_forecast_report_is_marked_as_real_model_update_not_pseudo_label_proxy(self):
        samples_by_id = {
            "support_flat": _flat_samples(),
            "candidate_target_like": _sine_samples(phase=0.02),
            "candidate_irrelevant": _flat_samples(offset=3.0),
            "target_sine": _sine_samples(phase=0.0),
        }
        report = build_downstream_forecast_report(
            _coverage_result(),
            samples_by_id,
            history_steps=4,
            horizon_steps=1,
            ridge_alpha=1.0e-4,
            max_windows_per_clip=64,
            top_policy="ts2vec_kcenter_v1",
            baseline_policy="quality_stratified_random_v1",
        )

        self.assertEqual(report["decision"]["downstream_task"], "raw_imu_autoregressive_forecast")
        self.assertEqual(report["decision"]["model_update"], "actual_retrain_after_acquisition")
        self.assertIn("not a source-family pseudo-label proxy", report["decision"]["read"])
        self.assertLess(report["decision"]["top_after_mse"], report["decision"]["baseline_after_mse"])
        with TemporaryDirectory() as tmp:
            paths = write_downstream_forecast_reports(report, Path(tmp))
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["final_budget"], 1)
            self.assertIn("Actual Downstream Model Update", paths["markdown"].read_text(encoding="utf-8"))


def _coverage_result() -> CoverageRunResult:
    episode = EpisodeSpec(
        episode_id="episode_000",
        fold_id=0,
        support_ids=("support_flat",),
        candidate_ids=("candidate_target_like", "candidate_irrelevant"),
        target_ids=("target_sine",),
        support_group_ids=("support_worker",),
        candidate_group_ids=("candidate_good_worker", "candidate_bad_worker"),
        target_group_ids=("target_worker",),
    )
    return CoverageRunResult(
        episodes=(episode,),
        policies=("quality_stratified_random_v1", "ts2vec_kcenter_v1"),
        budgets=(1,),
        rounds=(),
        selected_rows=(
            _selected("quality_stratified_random_v1", "candidate_irrelevant", "candidate_bad_worker"),
            _selected("ts2vec_kcenter_v1", "candidate_target_like", "candidate_good_worker"),
        ),
        metric_rows=(),
        policy_summary={},
        config=CoverageBenchmarkConfig(
            budgets=(1,),
            policies=("quality_stratified_random_v1", "ts2vec_kcenter_v1"),
            eval_views=("window",),
            primary_eval_views=("window",),
        ),
    )


def _selected(policy_id: str, sample_id: str, source_group_id: str) -> CoverageSelectedRow:
    return CoverageSelectedRow(
        episode_id="episode_000",
        fold_id=0,
        policy_id=policy_id,
        budget_k=1,
        rank_index=1,
        sample_id=sample_id,
        source_group_id=source_group_id,
        score=1.0,
        quality_score=1.0,
        artifact_score=0.0,
        valid=True,
        passed_artifact_gate=True,
    )


def _flat_samples(n: int = 160, *, offset: float = 0.0) -> np.ndarray:
    samples = np.zeros((n, 6), dtype=float)
    samples[:, 0] = float(offset)
    return samples


def _sine_samples(n: int = 160, *, phase: float = 0.0) -> np.ndarray:
    t = np.linspace(0.0, 8.0 * np.pi, n, dtype=float) + float(phase)
    samples = np.zeros((n, 6), dtype=float)
    samples[:, 0] = np.sin(t)
    samples[:, 1] = np.cos(t)
    return samples


if __name__ == "__main__":
    unittest.main()
