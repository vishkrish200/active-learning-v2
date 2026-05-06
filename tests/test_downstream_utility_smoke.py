import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.active_benchmark import (
    BenchmarkClip,
    EpisodeSpec,
    OfflineBenchmarkConfig,
    run_offline_active_benchmark,
)
from marginal_value.active_benchmark.downstream_smoke import (
    build_downstream_utility_report,
    build_downstream_utility_rows,
    summarize_downstream_utility,
    write_downstream_utility_reports,
)


class DownstreamUtilitySmokeTests(unittest.TestCase):
    def test_gcp_tiny_ts2vec_smoke_config_is_bounded_linear_only(self):
        config_path = Path("configs/downstream_utility_smoke_gcp_ts2vec_tiny.json")
        self.assertTrue(config_path.exists())
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertTrue(config["execution"]["no_gpu"])
        self.assertTrue(config["execution"]["no_ts2vec_retraining"])
        self.assertTrue(config["execution"]["linear_model_only"])
        self.assertLessEqual(config["data"]["max_rows_per_seed"], 360)
        self.assertEqual(config["data"]["selection_seeds"], [17])
        self.assertEqual(config["benchmark"]["episode_strategy"], "source_family_shift")
        self.assertEqual(config["benchmark"]["folds"], 2)
        self.assertEqual(config["benchmark"]["candidate_groups_per_episode"], 6)
        self.assertEqual(config["benchmark"]["rounds"], 2)
        self.assertEqual(config["benchmark"]["batch_size"], 2)
        self.assertIn("old_novelty_ts2vec", config["benchmark"]["policies"])
        self.assertIn("random_valid", config["benchmark"]["policies"])
        self.assertIn("submitted_full_replay", config["benchmark"]["policies"])
        self.assertEqual(config["downstream"]["model"], "linear_reconstruction_pca")
        self.assertEqual(config["downstream"]["baseline_policy"], "old_novelty_ts2vec")
        self.assertEqual(config["downstream"]["decision"], "hold_after_smoke")

    def test_gcp_three_seed_ts2vec_smoke_config_keeps_same_bounds(self):
        config_path = Path("configs/downstream_utility_smoke_gcp_ts2vec_3seed.json")
        self.assertTrue(config_path.exists())
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertTrue(config["execution"]["no_gpu"])
        self.assertTrue(config["execution"]["no_ts2vec_retraining"])
        self.assertTrue(config["execution"]["linear_model_only"])
        self.assertEqual(config["data"]["selection_seeds"], [17, 23, 37])
        self.assertLessEqual(config["data"]["max_rows_per_seed"], 300)
        self.assertEqual(config["benchmark"]["folds"], 2)
        self.assertEqual(config["benchmark"]["rounds"], 2)
        self.assertEqual(config["benchmark"]["batch_size"], 2)
        self.assertEqual(config["downstream"]["baseline_policy"], "old_novelty_ts2vec")
        self.assertEqual(config["acceptance"]["minimum_completed_seeds"], 3)
        self.assertIn("baseline beats random on mean relative reconstruction gain", config["acceptance"]["required_checks"])

    def test_reconstruction_utility_rewards_target_like_acquisition(self):
        clips = [
            _clip("support_a", "support_a", [9.0, 0.0]),
            _clip("support_b", "support_b", [10.0, 0.0]),
            _clip("support_c", "support_c", [11.0, 0.0]),
            _clip("candidate_target_like", "candidate_target_like", [0.0, 10.0], quality=0.90),
            _clip("candidate_high_quality_irrelevant", "candidate_high_quality_irrelevant", [12.0, 0.0], quality=1.00),
            _clip("target", "target", [0.0, 11.0]),
        ]
        episode = EpisodeSpec(
            episode_id="episode_000",
            fold_id=0,
            support_ids=("support_a", "support_b", "support_c"),
            candidate_ids=("candidate_target_like", "candidate_high_quality_irrelevant"),
            target_ids=("target",),
            support_group_ids=("support_a", "support_b", "support_c"),
            candidate_group_ids=("candidate_target_like", "candidate_high_quality_irrelevant"),
            target_group_ids=("target",),
        )
        result = run_offline_active_benchmark(
            clips,
            (episode,),
            OfflineBenchmarkConfig(
                batch_size=1,
                rounds=1,
                policies=("quality_only", "old_novelty_window"),
                representations=("window",),
                primary_representations=("window",),
            ),
        )

        rows = build_downstream_utility_rows(
            result,
            clips,
            downstream_representations=("window",),
            max_components=0,
        )
        summary = summarize_downstream_utility(rows)

        self.assertEqual(summary["row_count"], 2)
        quality_gain = summary["policy_final_means"]["quality_only"]["mean_relative_reconstruction_gain"]
        novelty_gain = summary["policy_final_means"]["old_novelty_window"]["mean_relative_reconstruction_gain"]
        self.assertGreater(novelty_gain, quality_gain)
        self.assertGreater(novelty_gain, 0.0)

    def test_report_writer_marks_smoke_as_not_full_downstream_proof(self):
        clips = [
            _clip("support_a", "support_a", [1.0, 0.0]),
            _clip("candidate_a", "candidate_a", [0.0, 1.0]),
            _clip("target", "target", [0.0, 1.0]),
        ]
        episode = EpisodeSpec(
            episode_id="episode_000",
            fold_id=0,
            support_ids=("support_a",),
            candidate_ids=("candidate_a",),
            target_ids=("target",),
            support_group_ids=("support_a",),
            candidate_group_ids=("candidate_a",),
            target_group_ids=("target",),
        )
        result = run_offline_active_benchmark(
            clips,
            (episode,),
            OfflineBenchmarkConfig(
                batch_size=1,
                rounds=1,
                policies=("old_novelty_window",),
                representations=("window",),
                primary_representations=("window",),
            ),
        )
        report = build_downstream_utility_report(
            result,
            clips,
            downstream_representations=("window",),
            max_components=0,
            baseline_policy="old_novelty_window",
        )

        self.assertEqual(report["decision"]["downstream_training"], "hold")
        self.assertIn("not full downstream proof", report["decision"]["read"])
        with TemporaryDirectory() as tmp:
            paths = write_downstream_utility_reports(report, Path(tmp))
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["row_count"], 1)
            self.assertIn("linear reconstruction", paths["markdown"].read_text(encoding="utf-8"))


def _clip(sample_id: str, source_group_id: str, embedding: list[float], *, quality: float = 1.0) -> BenchmarkClip:
    return BenchmarkClip(
        sample_id=sample_id,
        source_group_id=source_group_id,
        embeddings={"window": np.asarray(embedding, dtype=float)},
        quality_score=float(quality),
    )
