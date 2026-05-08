import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.active_benchmark import (
    BenchmarkClip,
    CoverageBenchmarkConfig,
    EpisodeSpec,
    run_coverage_benchmark,
)
from marginal_value.active_benchmark.coverage_reports import coverage_result_to_json
from scripts.offline_coverage_benchmark_from_urls import _coverage_proof_summary, _parse_eval_view_families
from scripts import launch_coverage_benchmark_gcp


class CoverageBenchmarkFromUrlsTests(unittest.TestCase):
    def test_url_coverage_runner_is_directly_executable_without_pythonpath(self):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "scripts/offline_coverage_benchmark_from_urls.py", "--help"],
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--manifest", result.stdout)

    def test_coverage_decision_summarizer_is_directly_executable_without_pythonpath(self):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "scripts/summarize_coverage_benchmark_reports.py", "--help"],
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output-json", result.stdout)

    def test_gcp_three_seed_coverage_config_is_bounded_cpu_only(self):
        config_path = Path("configs/coverage_benchmark_gcp_ts2vec_3seed_cpu.json")
        self.assertTrue(config_path.exists())
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertTrue(config["execution"]["no_gpu"])
        self.assertTrue(config["execution"]["no_ts2vec_retraining"])
        self.assertTrue(config["execution"]["no_downstream_training"])
        self.assertEqual(config["data"]["selection_seeds"], [17, 23, 37])
        self.assertLessEqual(config["data"]["max_rows_per_seed"], 300)
        self.assertEqual(config["benchmark"]["budgets"], [1, 2, 4])
        self.assertEqual(config["benchmark"]["primary_eval_views"], ["window", "raw_shape_stats"])
        self.assertIn("quality_stratified_random_v1", config["benchmark"]["policies"])
        self.assertIn("window_kcenter_v1", config["benchmark"]["policies"])
        self.assertIn("ts2vec_kcenter_v1", config["benchmark"]["policies"])
        self.assertIn("submitted_full_replay_v1", config["benchmark"]["policies"])
        self.assertEqual(config["acceptance"]["minimum_completed_seeds"], 3)
        self.assertIn("same-feature selector/eval rows are diagnostic, not primary", config["acceptance"]["required_checks"])

    def test_next_coverage_gate_config_adds_oracle_ci_without_training(self):
        config_path = Path("configs/coverage_benchmark_gcp_ts2vec_oracle_ci_10seed_cpu.json")
        self.assertTrue(config_path.exists())
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertTrue(config["execution"]["no_gpu"])
        self.assertTrue(config["execution"]["no_ts2vec_retraining"])
        self.assertTrue(config["execution"]["no_downstream_training"])
        self.assertEqual(len(config["data"]["selection_seeds"]), 10)
        self.assertEqual(config["benchmark"]["folds"], 5)
        self.assertIn("oracle_greedy_eval_view_v1", config["benchmark"]["policies"])
        self.assertEqual(config["reporting"]["decision_report_script"], "scripts/summarize_coverage_benchmark_reports.py")
        self.assertEqual(config["reporting"]["bootstrap_unit"], "seed_episode")
        self.assertIn("coverage_decision_report.json", config["acceptance"]["required_artifacts"])
        self.assertIn("oracle capture is reported for top deployable policy", config["acceptance"]["required_checks"])

    def test_bridge_pilot_config_matches_advisor_gate(self):
        config_path = Path("configs/coverage_bridge_benchmark_gcp_ts2vec_kcenter_pilot.json")
        self.assertTrue(config_path.exists())
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertTrue(config["execution"]["no_gpu"])
        self.assertTrue(config["execution"]["no_ts2vec_retraining"])
        self.assertTrue(config["execution"]["no_downstream_training"])
        self.assertEqual(config["data"]["selection_seeds"], [17])
        self.assertEqual(config["data"]["clips_per_group"], 3)
        self.assertEqual(config["benchmark"]["episode_strategy"], "source_family_label_holdout")
        self.assertEqual(config["benchmark"]["candidate_groups_per_episode"], 12)
        self.assertEqual(config["benchmark"]["target_candidate_groups_per_episode"], 4)
        self.assertEqual(config["benchmark"]["target_families_per_episode"], 2)
        self.assertEqual(config["benchmark"]["budgets"], [1, 2, 4])
        self.assertIn("quality_stratified_random_v1", config["benchmark"]["policies"])
        self.assertIn("ts2vec_kcenter_v1", config["benchmark"]["policies"])
        self.assertIn("oracle_greedy_target_family_v1", config["benchmark"]["policies"])
        self.assertEqual(config["downstream"]["model"], "nearest_centroid_source_family_bridge")
        self.assertIn("oracle target-family discovery at K=4 is at least 0.50", config["acceptance"]["required_checks"])
        self.assertIn("after_known_target_fraction becomes nonzero at K=4", config["acceptance"]["required_checks"])
        self.assertIn("balanced accuracy gain moves positive for the oracle", config["acceptance"]["required_checks"])
        self.assertIn("source-group leakage is false", config["acceptance"]["required_checks"])

    def test_bridge_three_seed_config_is_bounded_cpu_only(self):
        config_path = Path("configs/coverage_bridge_benchmark_gcp_ts2vec_kcenter_3seed.json")
        self.assertTrue(config_path.exists())
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertTrue(config["execution"]["no_gpu"])
        self.assertTrue(config["execution"]["no_ts2vec_retraining"])
        self.assertTrue(config["execution"]["no_downstream_training"])
        self.assertEqual(config["data"]["selection_seeds"], [17, 23, 37])
        self.assertLessEqual(config["data"]["max_rows_per_seed"], 540)
        self.assertEqual(config["benchmark"]["episode_strategy"], "source_family_label_holdout")
        self.assertEqual(config["benchmark"]["folds"], 5)
        self.assertEqual(config["benchmark"]["candidate_groups_per_episode"], 12)
        self.assertEqual(config["benchmark"]["target_candidate_groups_per_episode"], 4)
        self.assertEqual(config["benchmark"]["target_families_per_episode"], 2)
        self.assertEqual(config["benchmark"]["budgets"], [1, 2, 4])
        self.assertIn("quality_stratified_random_v1", config["benchmark"]["policies"])
        self.assertIn("ts2vec_kcenter_v1", config["benchmark"]["policies"])
        self.assertIn("oracle_greedy_target_family_v1", config["benchmark"]["policies"])
        self.assertEqual(config["acceptance"]["minimum_completed_seeds"], 3)
        self.assertIn(
            "ts2vec_kcenter_v1 balanced-accuracy delta vs quality_stratified_random_v1 is positive",
            config["acceptance"]["required_checks"],
        )
        self.assertIn("source-group leakage is false for all seeds", config["acceptance"]["required_checks"])

    def test_bridge_three_seed_eight_fold_config_is_episode_count_gate(self):
        config_path = Path("configs/coverage_bridge_benchmark_gcp_ts2vec_kcenter_3seed_8fold.json")
        self.assertTrue(config_path.exists())
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertTrue(config["execution"]["no_gpu"])
        self.assertTrue(config["execution"]["no_ts2vec_retraining"])
        self.assertTrue(config["execution"]["no_downstream_training"])
        self.assertEqual(config["data"]["selection_seeds"], [17, 23, 37])
        self.assertLessEqual(config["data"]["max_rows_per_seed"], 540)
        self.assertEqual(config["benchmark"]["episode_strategy"], "source_family_label_holdout")
        self.assertEqual(config["benchmark"]["folds"], 8)
        self.assertEqual(config["benchmark"]["candidate_groups_per_episode"], 12)
        self.assertEqual(config["benchmark"]["target_candidate_groups_per_episode"], 4)
        self.assertEqual(config["benchmark"]["target_families_per_episode"], 2)
        self.assertEqual(config["benchmark"]["budgets"], [1, 2, 4])
        self.assertEqual(config["acceptance"]["minimum_completed_seeds"], 3)
        self.assertEqual(config["acceptance"]["minimum_episodes_per_seed"], 8)
        self.assertEqual(config["acceptance"]["minimum_independent_episodes"], 24)
        self.assertIn("downstream bridge proxy is aggregated in the coverage decision report", config["acceptance"]["required_checks"])
        self.assertIn("at least 24 independent seed-episode units are summarized", config["acceptance"]["required_checks"])

    def test_gcp_launcher_download_copy_avoids_gsutil_parallel_hang(self):
        calls = []
        original_run = launch_coverage_benchmark_gcp._run

        def fake_run(command, *, cwd, check=True):
            calls.append(command)

        try:
            launch_coverage_benchmark_gcp._run = fake_run
            launch_coverage_benchmark_gcp._copy_results(
                gcs_prefix="gs://bucket/run",
                download_dir=Path("/tmp/downstream-download"),
            )
        finally:
            launch_coverage_benchmark_gcp._run = original_run

        self.assertGreaterEqual(len(calls), 4)
        self.assertTrue(all(call[:2] == ["gcloud", "storage"] for call in calls))
        self.assertTrue(any("--recursive" in call for call in calls))

    def test_eval_view_family_parser_rejects_malformed_entries(self):
        self.assertEqual(
            _parse_eval_view_families("window:window,raw_shape_stats:raw_shape,ts2vec:ts2vec"),
            {"window": "window", "raw_shape_stats": "raw_shape", "ts2vec": "ts2vec"},
        )
        with self.assertRaisesRegex(ValueError, "view:family"):
            _parse_eval_view_families("window")

    def test_coverage_proof_summary_audits_leakage_primary_rows_and_hygiene(self):
        clips = _proof_fixture_clips()
        result = run_coverage_benchmark(
            clips,
            [_proof_fixture_episode()],
            CoverageBenchmarkConfig(
                budgets=(1,),
                policies=("quality_only_v1", "ts2vec_support_novelty_v1"),
                eval_views=("window", "ts2vec"),
                primary_eval_views=("window",),
                eval_view_families={"window": "window", "ts2vec": "ts2vec"},
                quality_threshold=0.85,
            ),
        )
        report = coverage_result_to_json(result)
        with TemporaryDirectory() as tmpdir:
            proof = _coverage_proof_summary(
                report,
                clips=clips,
                output_dir=Path(tmpdir),
                report_paths={"json": str(Path(tmpdir) / "coverage.json"), "markdown": str(Path(tmpdir) / "coverage.md")},
                elapsed_seconds=1.25,
            )

        self.assertTrue(proof["leakage_ok"])
        self.assertGreater(proof["primary_coverage_metric_row_count"], 0)
        self.assertGreater(proof["same_feature_diagnostic_metric_row_count"], 0)
        self.assertEqual(proof["selected_invalid_rate_max"], 0.0)
        self.assertEqual(proof["selected_out_of_pool_count_total"], 0.0)
        self.assertEqual(proof["selected_target_leak_count_total"], 0.0)
        self.assertIn("ts2vec_support_novelty_v1", proof["policy_summary"])


def _proof_fixture_episode() -> EpisodeSpec:
    return EpisodeSpec(
        episode_id="episode-proof",
        fold_id=0,
        support_ids=("support",),
        candidate_ids=("candidate_near", "candidate_far"),
        target_ids=("target",),
        support_group_ids=("support_group",),
        candidate_group_ids=("candidate_near_group", "candidate_far_group"),
        target_group_ids=("target_group",),
    )


def _proof_fixture_clips() -> list[BenchmarkClip]:
    return [
        _clip("support", "support_group", [0.0, 0.0]),
        _clip("candidate_near", "candidate_near_group", [0.1, 0.0], quality=0.99),
        _clip("candidate_far", "candidate_far_group", [5.0, 0.0], quality=0.90),
        _clip("target", "target_group", [5.0, 0.1]),
    ]


def _clip(sample_id: str, group_id: str, point: list[float], *, quality: float = 1.0) -> BenchmarkClip:
    vector = np.asarray(point, dtype=float)
    return BenchmarkClip(
        sample_id=sample_id,
        source_group_id=group_id,
        embeddings={"window": vector, "ts2vec": vector},
        quality_score=quality,
    )


if __name__ == "__main__":
    unittest.main()
