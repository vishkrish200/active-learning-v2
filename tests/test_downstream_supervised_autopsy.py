import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.active_benchmark.downstream_supervised_autopsy import (
    build_downstream_supervised_autopsy,
    write_downstream_supervised_autopsy_reports,
)


class DownstreamSupervisedAutopsyTests(unittest.TestCase):
    def test_autopsy_flags_flat_accuracy_known_targets_and_no_candidate_opportunity(self):
        benchmark_report = {
            "difficulty_audit": [
                {
                    "episode_id": "episode_000",
                    "support_target_baseline_distance_by_representation": {"window": 0.10, "ts2vec": 0.20},
                    "candidate_target_nearest_distance_by_representation": {"window": 0.30, "ts2vec": 0.40},
                }
            ],
            "rounds": [
                {
                    "episode_id": "episode_000",
                    "policy_name": "old_novelty_ts2vec",
                    "round_index": 1,
                    "selected_ids": ["candidate_a", "candidate_b"],
                    "selected_source_group_ids": ["worker_a", "worker_b"],
                },
                {
                    "episode_id": "episode_000",
                    "policy_name": "random_valid",
                    "round_index": 1,
                    "selected_ids": ["candidate_c", "candidate_d"],
                    "selected_source_group_ids": ["worker_c", "worker_c"],
                },
            ],
        }
        supervised_report = {
            "input": {
                "baseline_policy": "old_novelty_ts2vec",
                "random_policy": "random_valid",
                "downstream_representations": ["window"],
            },
            "summary": {
                "policy_final_means": {
                    "old_novelty_ts2vec": {
                        "mean_after_accuracy": 0.80,
                        "mean_accuracy_gain": 0.0,
                        "mean_nll_reduction": -0.02,
                        "mean_baseline_known_target_fraction": 1.0,
                    },
                    "random_valid": {
                        "mean_after_accuracy": 0.80,
                        "mean_accuracy_gain": 0.0,
                        "mean_nll_reduction": 0.01,
                        "mean_baseline_known_target_fraction": 1.0,
                    },
                }
            },
            "rows": [
                {
                    "episode_id": "episode_000",
                    "policy_name": "old_novelty_ts2vec",
                    "round_index": 1,
                    "representation": "window",
                    "baseline_known_target_fraction": 1.0,
                    "accuracy_gain": 0.0,
                    "after_accuracy": 0.80,
                },
                {
                    "episode_id": "episode_000",
                    "policy_name": "random_valid",
                    "round_index": 1,
                    "representation": "window",
                    "baseline_known_target_fraction": 1.0,
                    "accuracy_gain": 0.0,
                    "after_accuracy": 0.80,
                },
            ],
        }

        autopsy = build_downstream_supervised_autopsy(benchmark_report, supervised_report)

        self.assertEqual(autopsy["decision"]["next_gate"], "build_harder_supervised_gate")
        self.assertIn("accuracy_flat_across_policies", autopsy["diagnosis"])
        self.assertIn("target_labels_already_known_before_acquisition", autopsy["diagnosis"])
        self.assertIn("candidate_pool_not_closer_than_support", autopsy["diagnosis"])
        self.assertLess(autopsy["policy_comparison"]["baseline_minus_random_nll_reduction"], 0.0)

    def test_autopsy_writer_outputs_json_and_markdown(self):
        autopsy = {
            "decision": {"next_gate": "build_harder_supervised_gate", "read": "hold"},
            "diagnosis": ["accuracy_flat_across_policies"],
            "policy_comparison": {"baseline_policy": "old", "random_policy": "random"},
            "flatness": {"accuracy_flat": True},
            "target_label_coverage": {"all_baseline_known_target_fraction_one": True},
            "candidate_opportunity": {"all_candidate_farther_than_support": True, "rows": []},
            "selection_audit": [],
        }
        with TemporaryDirectory() as tmp:
            paths = write_downstream_supervised_autopsy_reports(autopsy, Path(tmp))
            self.assertTrue(paths["json"].exists())
            self.assertTrue(paths["markdown"].exists())
            self.assertEqual(json.loads(paths["json"].read_text(encoding="utf-8"))["decision"]["read"], "hold")
            self.assertIn("accuracy_flat_across_policies", paths["markdown"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
