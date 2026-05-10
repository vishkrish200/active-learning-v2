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
from scripts.offline_coverage_benchmark_from_urls import (
    _coverage_proof_summary,
    _parse_eval_view_families,
    _proof_completion_event,
)
from scripts.summarize_coverage_benchmark_reports import _parse_names as _parse_summary_names
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

    def test_coverage_decision_summarizer_disambiguates_pool_sweep_report_names(self):
        report_paths = [
            Path("/tmp/run/pool36/seed_17/blind_target_coverage_benchmark_report.json"),
            Path("/tmp/run/pool72/seed_17/blind_target_coverage_benchmark_report.json"),
        ]

        self.assertEqual(_parse_summary_names("", report_paths), ["pool36_seed_17", "pool72_seed_17"])

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

    def test_bridge_exact_oracle_config_adds_replays_without_training(self):
        config_path = Path("configs/coverage_bridge_benchmark_gcp_ts2vec_kcenter_3seed_8fold_exact_oracle.json")
        self.assertTrue(config_path.exists())
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertTrue(config["execution"]["no_gpu"])
        self.assertTrue(config["execution"]["no_ts2vec_retraining"])
        self.assertTrue(config["execution"]["no_downstream_training"])
        self.assertEqual(config["data"]["selection_seeds"], [17, 23, 37])
        self.assertEqual(config["benchmark"]["folds"], 8)
        self.assertEqual(config["benchmark"]["budgets"], [1, 2, 4])
        self.assertGreaterEqual(config["benchmark"]["oracle_exact_combination_limit"], 58905)
        policies = config["benchmark"]["policies"]
        replays = [policy for policy in policies if policy.startswith("quality_stratified_random_replay_")]
        self.assertEqual(len(replays), 100)
        self.assertEqual(replays[0], "quality_stratified_random_replay_000_v1")
        self.assertEqual(replays[-1], "quality_stratified_random_replay_099_v1")
        self.assertIn("quality_stratified_random_v1", policies)
        self.assertIn("window_kcenter_v1", policies)
        self.assertIn("ts2vec_kcenter_v1", policies)
        self.assertIn("submitted_full_replay_v1", policies)
        self.assertIn("oracle_exact_coverage_v1", policies)
        self.assertIn("oracle_greedy_target_family_v1", policies)
        self.assertEqual(config["reporting"]["oracle_policy"], "oracle_exact_coverage_v1")
        self.assertEqual(config["downstream"]["models"], ["nearest_centroid", "ridge_classifier"])
        self.assertIn("exact coverage oracle dominates deployable policies at K=4", config["acceptance"]["required_checks"])
        self.assertIn("100 quality-stratified random replay controls are summarized", config["acceptance"]["required_checks"])

    def test_bridge_expanded_cpu_validation_config_matches_training_unlock_gate(self):
        config_path = Path("configs/coverage_bridge_benchmark_gcp_ts2vec_kcenter_10seed_8fold_pool_sweep_exact_oracle.json")
        self.assertTrue(config_path.exists())
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertTrue(config["execution"]["no_gpu"])
        self.assertTrue(config["execution"]["no_ts2vec_retraining"])
        self.assertTrue(config["execution"]["no_downstream_training"])
        self.assertEqual(config["execution"]["machine_type"], "e2-standard-8")
        self.assertGreaterEqual(config["execution"]["max_runtime_minutes"], 720)
        self.assertEqual(config["data"]["selection_seeds"], [17, 23, 37, 41, 53, 67, 79, 83, 97, 101])
        self.assertGreaterEqual(config["data"]["max_rows_per_seed"], 720)
        self.assertGreaterEqual(config["data"]["max_groups"], 240)
        self.assertEqual(config["benchmark"]["folds"], 8)
        self.assertEqual(
            config["benchmark"]["candidate_group_sweep"],
            [
                {"label": "pool36", "candidate_groups_per_episode": 12, "target_candidate_groups_per_episode": 4},
                {"label": "pool72", "candidate_groups_per_episode": 24, "target_candidate_groups_per_episode": 8},
            ],
        )
        self.assertGreaterEqual(config["benchmark"]["oracle_exact_combination_limit"], 1028790)
        policies = config["benchmark"]["policies"]
        self.assertEqual(len([policy for policy in policies if policy.startswith("quality_stratified_random_replay_")]), 100)
        self.assertEqual(config["acceptance"]["minimum_completed_seeds"], 10)
        self.assertEqual(config["acceptance"]["minimum_episodes_per_seed"], 16)
        self.assertEqual(config["acceptance"]["minimum_independent_episodes"], 160)
        self.assertIn("candidate pool stress includes both 36-clip and 72-clip episodes", config["acceptance"]["required_checks"])
        self.assertIn("mean delta vs quality_stratified_random_v1 is at least 0.04", config["acceptance"]["required_checks"])
        self.assertIn("CI95 low vs quality_stratified_random_v1 is at least 0.025", config["acceptance"]["required_checks"])
        self.assertIn("replay-null p-value is at most 0.01", config["acceptance"]["required_checks"])
        self.assertIn("window_kcenter_v1 paired mean delta is at least 0.015 with CI95 low above zero", config["acceptance"]["required_checks"])

    def test_bounded_downstream_canary_config_is_focused_cpu_only(self):
        config_path = Path("configs/downstream_canary_gcp_ts2vec_kcenter_3seed_8fold_pool_sweep.json")
        self.assertTrue(config_path.exists())
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertTrue(config["execution"]["no_gpu"])
        self.assertTrue(config["execution"]["no_ts2vec_retraining"])
        self.assertTrue(config["execution"]["no_large_downstream_training"])
        self.assertTrue(config["execution"]["frozen_feature_downstream_canary"])
        self.assertEqual(config["execution"]["machine_type"], "e2-standard-4")
        self.assertEqual(config["data"]["selection_seeds"], [17, 23, 37])
        self.assertEqual(config["benchmark"]["episode_strategy"], "source_family_label_holdout")
        self.assertEqual(config["benchmark"]["folds"], 8)
        self.assertEqual(
            config["benchmark"]["candidate_group_sweep"],
            [
                {"label": "pool36", "candidate_groups_per_episode": 12, "target_candidate_groups_per_episode": 4},
                {"label": "pool72", "candidate_groups_per_episode": 24, "target_candidate_groups_per_episode": 8},
            ],
        )
        policies = config["benchmark"]["policies"]
        self.assertEqual(
            policies,
            [
                "quality_stratified_random_v1",
                "quality_only_v1",
                "window_kcenter_v1",
                "submitted_full_replay_v1",
                "ts2vec_kcenter_v1",
            ],
        )
        self.assertEqual(config["downstream"]["models"], ["nearest_centroid", "ridge_classifier"])
        self.assertEqual(config["downstream"]["representations"], ["window", "raw_shape_stats"])
        self.assertEqual(config["downstream"]["decision"], "bounded_frozen_canary")
        self.assertEqual(config["acceptance"]["minimum_independent_episodes"], 48)
        self.assertIn("bounded_frozen_canary is pass in downstream bridge proxy decision", config["acceptance"]["required_checks"])
        self.assertIn("large_training remains hold", config["acceptance"]["required_checks"])

    def test_scaled_frozen_downstream_probe_config_is_still_bounded(self):
        config_path = Path("configs/downstream_frozen_probe_gcp_ts2vec_kcenter_3seed_12fold_pool_sweep.json")
        self.assertTrue(config_path.exists())
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertTrue(config["execution"]["no_gpu"])
        self.assertTrue(config["execution"]["no_ts2vec_retraining"])
        self.assertTrue(config["execution"]["no_large_downstream_training"])
        self.assertTrue(config["execution"]["frozen_feature_downstream_probe"])
        self.assertEqual(config["execution"]["machine_type"], "e2-standard-8")
        self.assertEqual(config["data"]["selection_seeds"], [17, 23, 37])
        self.assertGreaterEqual(config["data"]["max_rows_per_seed"], 900)
        self.assertEqual(config["benchmark"]["episode_strategy"], "source_family_label_holdout")
        self.assertEqual(config["benchmark"]["folds"], 12)
        self.assertEqual(
            config["benchmark"]["candidate_group_sweep"],
            [
                {"label": "pool36", "candidate_groups_per_episode": 12, "target_candidate_groups_per_episode": 4},
                {"label": "pool72", "candidate_groups_per_episode": 24, "target_candidate_groups_per_episode": 8},
                {"label": "pool108", "candidate_groups_per_episode": 36, "target_candidate_groups_per_episode": 12},
            ],
        )
        self.assertEqual(config["benchmark"]["budgets"], [1, 2, 4])
        self.assertEqual(config["downstream"]["models"], ["nearest_centroid", "ridge_classifier"])
        self.assertEqual(config["downstream"]["representations"], ["window", "raw_shape_stats"])
        self.assertEqual(config["downstream"]["decision"], "scale_frozen_downstream_probe")
        self.assertEqual(config["acceptance"]["minimum_independent_episodes"], 108)
        self.assertIn("three pool stresses are summarized: pool36, pool72, and pool108", config["acceptance"]["required_checks"])
        self.assertIn("large_training remains hold", config["acceptance"]["required_checks"])
        self.assertIn("ts2vec_retraining remains no", config["acceptance"]["required_checks"])

    def test_quick_heldout_utility_probe_config_is_bounded(self):
        config_path = Path("configs/downstream_utility_probe_gcp_ts2vec_kcenter_quick.json")
        self.assertTrue(config_path.exists())
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertTrue(config["execution"]["no_gpu"])
        self.assertTrue(config["execution"]["no_ts2vec_retraining"])
        self.assertTrue(config["execution"]["no_large_downstream_training"])
        self.assertTrue(config["execution"]["heldout_utility_probe"])
        self.assertEqual(config["data"]["selection_seeds"], [17])
        self.assertLessEqual(config["data"]["max_rows_per_seed"], 420)
        self.assertEqual(config["benchmark"]["episode_strategy"], "source_family_label_holdout")
        self.assertEqual(config["benchmark"]["folds"], 4)
        self.assertEqual(config["benchmark"]["budgets"], [1, 2, 4])
        self.assertEqual(config["downstream_utility"]["model"], "linear_reconstruction_pca")
        self.assertEqual(config["downstream_utility"]["top_policy"], "ts2vec_kcenter_v1")
        self.assertEqual(config["downstream_utility"]["baseline_policy"], "quality_stratified_random_v1")
        self.assertIn("downstream_coverage_utility_report.json", config["acceptance"]["required_artifacts"])
        self.assertIn("large_training remains hold", config["acceptance"]["required_checks"])

    def test_quick_tiny_training_canary_config_combines_model_heads_and_utility_probe(self):
        config_path = Path("configs/downstream_training_canary_gcp_ts2vec_kcenter_quick.json")
        self.assertTrue(config_path.exists())
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertTrue(config["execution"]["no_gpu"])
        self.assertTrue(config["execution"]["no_ts2vec_retraining"])
        self.assertTrue(config["execution"]["no_large_downstream_training"])
        self.assertTrue(config["execution"]["tiny_downstream_training_canary"])
        self.assertEqual(config["data"]["selection_seeds"], [17])
        self.assertLessEqual(config["data"]["max_rows_per_seed"], 420)
        self.assertEqual(config["benchmark"]["episode_strategy"], "source_family_label_holdout")
        self.assertEqual(config["benchmark"]["folds"], 4)
        self.assertEqual(config["benchmark"]["policies"], [
            "quality_stratified_random_v1",
            "quality_only_v1",
            "window_kcenter_v1",
            "submitted_full_replay_v1",
            "ts2vec_kcenter_v1",
        ])
        self.assertEqual(config["downstream"]["models"], ["nearest_centroid", "ridge_classifier"])
        self.assertEqual(config["downstream"]["label_source"], "source_family")
        self.assertEqual(config["downstream"]["top_policy"], "ts2vec_kcenter_v1")
        self.assertEqual(config["downstream_utility"]["model"], "linear_reconstruction_pca")
        self.assertIn("downstream_coverage_supervised_smoke_report.json", config["acceptance"]["required_artifacts"])
        self.assertIn("downstream_coverage_utility_report.json", config["acceptance"]["required_artifacts"])
        self.assertIn("not challenge-label training proof", config["acceptance"]["non_claim"])

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

    def test_gcp_launcher_runtime_overrides_support_seed_parallel_runs(self):
        config = {
            "execution": {"machine_type": "e2-standard-8"},
            "data": {"selection_seeds": [17, 23, 37]},
        }

        launch_coverage_benchmark_gcp._apply_runtime_overrides(
            config,
            machine_type="n2-standard-16",
            selection_seeds="23",
        )

        self.assertEqual(config["execution"]["machine_type"], "n2-standard-16")
        self.assertEqual(config["data"]["selection_seeds"], [23])

    def test_gcp_launcher_startup_passes_exact_oracle_limit(self):
        config = json.loads(
            Path("configs/coverage_bridge_benchmark_gcp_ts2vec_kcenter_3seed_8fold_exact_oracle.json").read_text(
                encoding="utf-8"
            )
        )

        startup = launch_coverage_benchmark_gcp._startup_script(
            config=config,
            run_id="unit-test",
            gcs_prefix="gs://bucket/unit-test",
            bundle_name="repo_bundle.tgz",
            manifest_name="manifest.txt",
            checkpoint_name="ts2vec_best.pt",
        )

        self.assertIn("--oracle-exact-combination-limit 100000", startup)
        self.assertIn("--oracle-policy oracle_exact_coverage_v1", startup)
        self.assertIn("--downstream-supervised-models nearest_centroid,ridge_classifier", startup)

    def test_gcp_launcher_startup_passes_downstream_utility_args(self):
        config = json.loads(
            Path("configs/downstream_utility_probe_gcp_ts2vec_kcenter_quick.json").read_text(encoding="utf-8")
        )

        startup = launch_coverage_benchmark_gcp._startup_script(
            config=config,
            run_id="unit-test",
            gcs_prefix="gs://bucket/unit-test",
            bundle_name="repo_bundle.tgz",
            manifest_name="manifest.txt",
            checkpoint_name="ts2vec_best.pt",
        )

        self.assertIn("--downstream-utility-enable", startup)
        self.assertIn("--downstream-utility-representations window,raw_shape_stats", startup)
        self.assertIn("--downstream-utility-top-policy ts2vec_kcenter_v1", startup)
        self.assertIn("--downstream-utility-baseline-policy quality_stratified_random_v1", startup)

    def test_downstream_forecast_task_config_wires_real_model_update_args(self):
        config = json.loads(
            Path("configs/downstream_forecast_task_gcp_ts2vec_kcenter_quick.json").read_text(encoding="utf-8")
        )

        self.assertTrue(config["execution"]["no_gpu"])
        self.assertTrue(config["execution"]["no_ts2vec_retraining"])
        self.assertEqual(config["downstream_forecast"]["model"], "raw_imu_autoregressive_ridge")
        self.assertEqual(config["downstream_forecast"]["decision"], "actual_downstream_model_update")
        self.assertIn("downstream_forecast_task_report.json", config["acceptance"]["required_artifacts"])
        self.assertIn("not a source-family pseudo-label proxy", config["acceptance"]["non_claim"])

        startup = launch_coverage_benchmark_gcp._startup_script(
            config=config,
            run_id="unit-test",
            gcs_prefix="gs://bucket/unit-test",
            bundle_name="repo_bundle.tgz",
            manifest_name="manifest.txt",
            checkpoint_name="ts2vec_best.pt",
        )

        self.assertIn("--downstream-forecast-enable", startup)
        self.assertIn("--downstream-forecast-history-steps 8", startup)
        self.assertIn("--downstream-forecast-top-policy ts2vec_kcenter_v1", startup)
        self.assertIn("--downstream-forecast-baseline-policy quality_stratified_random_v1", startup)

    def test_probcover_downstream_forecast_config_is_bounded_three_seed_replay(self):
        config_path = Path("configs/downstream_forecast_task_gcp_support_gap_probcover_3seed.json")
        self.assertTrue(config_path.exists())
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertTrue(config["execution"]["no_gpu"])
        self.assertTrue(config["execution"]["no_ts2vec_retraining"])
        self.assertTrue(config["execution"]["no_large_downstream_training"])
        self.assertEqual(config["data"]["selection_seeds"], [17, 23, 37])
        self.assertEqual(
            config["benchmark"]["policies"],
            [
                "quality_stratified_random_v1",
                "quality_only_v1",
                "window_kcenter_v1",
                "support_gap_window_probcover_v1",
                "submitted_full_replay_v1",
                "ts2vec_kcenter_v1",
            ],
        )
        self.assertEqual(config["downstream_forecast"]["model"], "raw_imu_autoregressive_ridge")
        self.assertEqual(config["downstream_forecast"]["top_policy"], "support_gap_window_probcover_v1")
        self.assertEqual(config["downstream_forecast"]["baseline_policy"], "window_kcenter_v1")
        self.assertIn("must beat window_kcenter_v1 at K=4", config["acceptance"]["promotion_rule"])
        self.assertIn("clearly beat quality_stratified_random_v1", config["acceptance"]["promotion_rule"])

        startup = launch_coverage_benchmark_gcp._startup_script(
            config=config,
            run_id="unit-test",
            gcs_prefix="gs://bucket/unit-test",
            bundle_name="repo_bundle.tgz",
            manifest_name="manifest.txt",
            checkpoint_name="ts2vec_best.pt",
        )

        self.assertIn(
            "--policies quality_stratified_random_v1,quality_only_v1,window_kcenter_v1,support_gap_window_probcover_v1,submitted_full_replay_v1,ts2vec_kcenter_v1",
            startup,
        )
        self.assertIn("--downstream-forecast-top-policy support_gap_window_probcover_v1", startup)
        self.assertIn("--downstream-forecast-baseline-policy window_kcenter_v1", startup)

    def test_survivor_downstream_forecast_config_is_locked_ten_seed_confirmation(self):
        config_path = Path("configs/downstream_forecast_task_gcp_survivors_10seed_locked.json")
        self.assertTrue(config_path.exists())
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertTrue(config["execution"]["no_gpu"])
        self.assertTrue(config["execution"]["no_ts2vec_retraining"])
        self.assertTrue(config["execution"]["no_large_downstream_training"])
        self.assertEqual(config["data"]["selection_seeds"], [17, 23, 37, 41, 53, 67, 79, 83, 97, 101])
        self.assertEqual(config["acceptance"]["minimum_independent_episodes"], 40)
        self.assertEqual(
            config["benchmark"]["policies"],
            [
                "quality_stratified_random_v1",
                "quality_only_v1",
                "window_kcenter_v1",
                "submitted_full_replay_v1",
                "ts2vec_kcenter_v1",
            ],
        )
        self.assertNotIn("support_gap_window_probcover_v1", config["benchmark"]["policies"])
        self.assertEqual(config["downstream_forecast"]["top_policy"], "window_kcenter_v1")
        self.assertEqual(config["downstream_forecast"]["baseline_policy"], "quality_stratified_random_v1")
        self.assertIn("support_gap_window_probcover_v1 is excluded rather than rescued", config["acceptance"]["required_checks"])
        self.assertIn("not policy shopping", config["acceptance"]["non_claim"])

        startup = launch_coverage_benchmark_gcp._startup_script(
            config=config,
            run_id="unit-test",
            gcs_prefix="gs://bucket/unit-test",
            bundle_name="repo_bundle.tgz",
            manifest_name="manifest.txt",
            checkpoint_name="ts2vec_best.pt",
        )

        self.assertIn(
            "--policies quality_stratified_random_v1,quality_only_v1,window_kcenter_v1,submitted_full_replay_v1,ts2vec_kcenter_v1",
            startup,
        )
        self.assertNotIn("support_gap_window_probcover_v1", startup)
        self.assertIn("--downstream-forecast-top-policy window_kcenter_v1", startup)
        self.assertIn("--downstream-forecast-baseline-policy quality_stratified_random_v1", startup)

    def test_gcp_launcher_startup_runs_round_loop_downstream_utility_runner(self):
        config = json.loads(
            Path("configs/downstream_active_loop_gcp_ts2vec_kcenter_3seed_fast.json").read_text(encoding="utf-8")
        )

        startup = launch_coverage_benchmark_gcp._startup_script(
            config=config,
            run_id="unit-test",
            gcs_prefix="gs://bucket/unit-test",
            bundle_name="repo_bundle.tgz",
            manifest_name="manifest.txt",
            checkpoint_name="ts2vec_best.pt",
        )

        self.assertIn("import scripts.downstream_utility_smoke_from_urls", startup)
        self.assertIn("python scripts/downstream_utility_smoke_from_urls.py", startup)
        self.assertIn("--rounds 2", startup)
        self.assertIn("--batch-size 2", startup)
        self.assertIn("--target-candidate-groups-per-episode 4", startup)
        self.assertIn("--target-families-per-episode 2", startup)
        self.assertIn("--policies quality_stratified_random,quality_only,kcenter_quality_gated_window,submitted_full_replay,kcenter_quality_gated_ts2vec", startup)
        self.assertIn("--downstream-baseline-policy kcenter_quality_gated_ts2vec", startup)
        self.assertIn("--downstream-random-policy quality_stratified_random", startup)
        self.assertIn("--supervised-downstream-label-source source_family", startup)
        self.assertNotIn("python scripts/offline_coverage_benchmark_from_urls.py", startup)
        self.assertNotIn("summarize_coverage_benchmark_reports.py", startup)

    def test_gcp_launcher_startup_runs_candidate_pool_sweep(self):
        config = json.loads(
            Path("configs/coverage_bridge_benchmark_gcp_ts2vec_kcenter_10seed_8fold_pool_sweep_exact_oracle.json").read_text(
                encoding="utf-8"
            )
        )

        startup = launch_coverage_benchmark_gcp._startup_script(
            config=config,
            run_id="unit-test",
            gcs_prefix="gs://bucket/unit-test",
            bundle_name="repo_bundle.tgz",
            manifest_name="manifest.txt",
            checkpoint_name="ts2vec_best.pt",
        )

        self.assertIn('POOL_SPECS=("pool36:12:4" "pool72:24:8")', startup)
        self.assertIn('for POOL_SPEC in "${POOL_SPECS[@]}"; do', startup)
        self.assertIn('IFS=: read -r POOL_LABEL CANDIDATE_GROUPS TARGET_CANDIDATE_GROUPS <<< "$POOL_SPEC"', startup)
        self.assertIn('--output-dir "$OUT/$POOL_LABEL/seed_$SEED"', startup)
        self.assertIn('--candidate-groups-per-episode "$CANDIDATE_GROUPS"', startup)
        self.assertIn('--target-candidate-groups-per-episode "$TARGET_CANDIDATE_GROUPS"', startup)
        self.assertIn("--oracle-exact-combination-limit 1100000", startup)

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

    def test_proof_completion_event_keeps_large_policy_summary_out_of_stdout(self):
        proof = {
            "output_dir": "/tmp/out",
            "json_report": "/tmp/out/report.json",
            "markdown_report": "/tmp/out/report.md",
            "n_clips": 10,
            "n_source_groups": 4,
            "n_episodes": 2,
            "n_round_rows": 8,
            "n_metric_rows": 128,
            "n_selected_rows": 16,
            "primary_coverage_metric_row_count": 64,
            "same_feature_diagnostic_metric_row_count": 16,
            "mean_primary_coverage_gain_rel": 0.25,
            "selected_invalid_rate_max": 0.0,
            "selected_out_of_pool_count_total": 0.0,
            "selected_target_leak_count_total": 0.0,
            "selected_duplicate_clip_count_total": 0.0,
            "leakage_ok": True,
            "elapsed_seconds": 12.5,
            "policies": ["quality_stratified_random_replay_000_v1"],
            "budgets": [1, 2, 4],
            "policy_summary": {
                "quality_stratified_random_replay_000_v1": {
                    str(index): {"large": "x" * 1000}
                    for index in range(100)
                }
            },
            "leakage_checks": [{"episode_id": "episode_0"}],
            "downstream_coverage_supervised_report": {
                "decision": "gate_only",
                "json": "/tmp/out/downstream.json",
                "markdown": "/tmp/out/downstream.md",
                "summary": {"large": "y" * 1000},
            },
            "downstream_coverage_utility_report": {
                "decision": "utility_gate_only",
                "json": "/tmp/out/downstream_utility.json",
                "markdown": "/tmp/out/downstream_utility.md",
                "summary": {"large": "z" * 1000},
            },
            "downstream_forecast_task_report": {
                "decision": {"model_update": "actual_retrain_after_acquisition"},
                "json": "/tmp/out/downstream_forecast.json",
                "markdown": "/tmp/out/downstream_forecast.md",
                "summary": {"large": "q" * 1000},
            },
        }

        event = _proof_completion_event(proof)

        self.assertEqual(event["event"], "done")
        self.assertEqual(event["n_metric_rows"], 128)
        self.assertNotIn("policy_summary", event)
        self.assertNotIn("leakage_checks", event)
        self.assertEqual(event["downstream_coverage_supervised_report"]["decision"], "gate_only")
        self.assertNotIn("summary", event["downstream_coverage_supervised_report"])
        self.assertEqual(event["downstream_coverage_utility_report"]["decision"], "utility_gate_only")
        self.assertNotIn("summary", event["downstream_coverage_utility_report"])
        self.assertEqual(
            event["downstream_forecast_task_report"]["decision"],
            {"model_update": "actual_retrain_after_acquisition"},
        )
        self.assertNotIn("summary", event["downstream_forecast_task_report"])


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
