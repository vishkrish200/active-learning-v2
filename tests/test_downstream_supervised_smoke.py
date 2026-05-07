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
from marginal_value.active_benchmark.downstream_supervised_smoke import (
    build_downstream_supervised_report,
    build_downstream_supervised_rows,
    summarize_downstream_supervised,
    write_downstream_supervised_reports,
)


class DownstreamSupervisedSmokeTests(unittest.TestCase):
    def test_gcp_tiny_supervised_smoke_config_is_bounded_and_frozen(self):
        config_path = Path("configs/downstream_supervised_smoke_gcp_ts2vec_tiny.json")
        self.assertTrue(config_path.exists())
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertTrue(config["execution"]["no_gpu"])
        self.assertTrue(config["execution"]["no_ts2vec_retraining"])
        self.assertEqual(config["data"]["selection_seeds"], [17])
        self.assertLessEqual(config["data"]["max_rows_per_seed"], 300)
        self.assertEqual(config["benchmark"]["episode_strategy"], "rotating")
        self.assertEqual(config["benchmark"]["folds"], 2)
        self.assertEqual(config["benchmark"]["rounds"], 2)
        self.assertEqual(config["benchmark"]["batch_size"], 2)
        self.assertEqual(config["downstream"]["model"], "nearest_centroid_source_family")
        self.assertEqual(config["downstream"]["label_source"], "source_family")
        self.assertEqual(config["downstream"]["baseline_policy"], "old_novelty_ts2vec")
        self.assertEqual(config["downstream"]["decision"], "tiny_smoke_only")

    def test_gcp_hard_supervised_smoke_config_uses_label_holdout_gate(self):
        config_path = Path("configs/downstream_supervised_smoke_gcp_ts2vec_label_holdout.json")
        self.assertTrue(config_path.exists())
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertTrue(config["execution"]["no_gpu"])
        self.assertTrue(config["execution"]["no_ts2vec_retraining"])
        self.assertEqual(config["data"]["selection_seeds"], [17])
        self.assertLessEqual(config["data"]["max_rows_per_seed"], 300)
        self.assertEqual(config["benchmark"]["episode_strategy"], "source_family_label_holdout")
        self.assertEqual(config["benchmark"]["folds"], 2)
        self.assertEqual(config["benchmark"]["candidate_groups_per_episode"], 6)
        self.assertEqual(config["benchmark"]["target_groups_per_episode"], 2)
        self.assertEqual(config["benchmark"]["rounds"], 2)
        self.assertEqual(config["benchmark"]["batch_size"], 2)
        self.assertIn("support excludes target family", config["acceptance"]["required_checks"])
        self.assertIn("candidate includes target-family bridge groups", config["acceptance"]["required_checks"])

    def test_supervised_utility_rewards_acquiring_target_family_labels(self):
        clips = [
            _clip("support_a", "fam_a_worker_support", [0.0, 0.0], quality=0.90),
            _clip("support_b", "fam_b_worker_support", [10.0, 0.0], quality=0.90),
            _clip("candidate_target_family", "fam_c_worker_candidate", [0.0, 10.0], quality=0.90),
            _clip("candidate_high_quality_known_family", "fam_b_worker_candidate", [10.0, 1.0], quality=1.00),
            _clip("target_c", "fam_c_worker_target", [0.0, 11.0], quality=0.90),
        ]
        episode = EpisodeSpec(
            episode_id="episode_000",
            fold_id=0,
            support_ids=("support_a", "support_b"),
            candidate_ids=("candidate_target_family", "candidate_high_quality_known_family"),
            target_ids=("target_c",),
            support_group_ids=("fam_a_worker_support", "fam_b_worker_support"),
            candidate_group_ids=("fam_c_worker_candidate", "fam_b_worker_candidate"),
            target_group_ids=("fam_c_worker_target",),
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

        rows = build_downstream_supervised_rows(
            result,
            clips,
            downstream_representations=("window",),
            label_source="source_family",
            label_representation="window",
            source_family_count=3,
        )
        summary = summarize_downstream_supervised(rows)

        quality_gain = summary["policy_final_means"]["quality_only"]["mean_accuracy_gain"]
        novelty_gain = summary["policy_final_means"]["old_novelty_window"]["mean_accuracy_gain"]
        self.assertGreater(novelty_gain, quality_gain)
        self.assertGreater(novelty_gain, 0.0)

    def test_report_writer_marks_supervised_smoke_as_pseudo_label_only(self):
        clips = [
            _clip("support_a", "fam_a_worker_support", [0.0, 0.0]),
            _clip("candidate_c", "fam_c_worker_candidate", [0.0, 10.0]),
            _clip("target_c", "fam_c_worker_target", [0.0, 11.0]),
        ]
        episode = EpisodeSpec(
            episode_id="episode_000",
            fold_id=0,
            support_ids=("support_a",),
            candidate_ids=("candidate_c",),
            target_ids=("target_c",),
            support_group_ids=("fam_a_worker_support",),
            candidate_group_ids=("fam_c_worker_candidate",),
            target_group_ids=("fam_c_worker_target",),
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
        report = build_downstream_supervised_report(
            result,
            clips,
            downstream_representations=("window",),
            label_source="source_family",
            label_representation="window",
            source_family_count=3,
            baseline_policy="old_novelty_window",
        )

        self.assertEqual(report["decision"]["downstream_training"], "tiny_smoke_only")
        self.assertIn("pseudo-label", report["decision"]["read"])
        with TemporaryDirectory() as tmp:
            paths = write_downstream_supervised_reports(report, Path(tmp))
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["row_count"], 1)
            self.assertIn("source-family", paths["markdown"].read_text(encoding="utf-8"))


def _clip(sample_id: str, source_group_id: str, embedding: list[float], *, quality: float = 1.0) -> BenchmarkClip:
    return BenchmarkClip(
        sample_id=sample_id,
        source_group_id=source_group_id,
        embeddings={"window": np.asarray(embedding, dtype=float)},
        quality_score=float(quality),
    )
