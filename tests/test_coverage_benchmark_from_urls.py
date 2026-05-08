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
