import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.data.split_manifest import hash_manifest_url
from marginal_value.eval.marginal_coverage_eval import (
    coverage_gain_for_selection,
    run_marginal_coverage_eval,
)


class MarginalCoverageEvalTests(unittest.TestCase):
    def test_coverage_gain_rewards_candidates_that_cover_heldout_targets(self):
        support = np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=float)
        target = np.asarray([[10.0, 10.0], [10.0, 11.0]], dtype=float)
        candidates = np.asarray([[10.0, 10.2], [0.0, 0.2]], dtype=float)

        good = coverage_gain_for_selection(support, target, candidates, selected_indices=[0])
        bad = coverage_gain_for_selection(support, target, candidates, selected_indices=[1])

        self.assertGreater(good["coverage_gain"], bad["coverage_gain"])
        self.assertGreater(good["relative_coverage_gain"], 0.5)
        self.assertAlmostEqual(bad["coverage_gain"], 0.0, places=6)

    def test_marginal_coverage_eval_reports_ranker_and_baseline_policy_gains(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_synthetic_physical_source_dataset(root)
            config = {
                "data": {
                    "root": str(root),
                    "feature_glob": "cache/features/*.npz",
                    "raw_glob": "cache/raw/*.jsonl",
                    "pretrain_manifest": "cache/manifests/pretrain.txt",
                    "val_manifest": "cache/manifests/val.txt",
                },
                "artifacts": {"output_dir": str(root / "out")},
                "ranking": {
                    "representation": "window_mean_std_pool",
                    "k_old": 2,
                    "k_new_density": 2,
                    "novelty_weight": 0.75,
                    "reranker_method": "tiered_cluster_cap",
                    "cluster_similarity_threshold": 0.985,
                    "cluster_bonus_weight": 0.0,
                    "cluster_cap_key": "new_cluster_id",
                    "cluster_cap_min_quality": 0.45,
                    "cluster_cap_schedule": [
                        {"top_k": 4, "max_per_cluster": 2},
                        {"top_k": 8, "max_per_cluster": 4},
                    ],
                },
                "grammar_features": {"enabled": False},
                "score_guards": {"stationary_singleton": {"enabled": False}},
                "large_cluster_split": {"enabled": False},
                "quality": {"sample_rate": 30.0, "max_samples_per_clip": 90},
                "eval": {
                    "smoke_max_rows": 36,
                    "smoke_folds": 1,
                    "source_groups_per_fold": 2,
                    "candidate_fraction": 0.5,
                    "max_candidate_per_fold": 12,
                    "max_negative_per_fold": 12,
                    "max_target_per_fold": 12,
                    "embedding_load_workers": 1,
                    "k_values": [2, 4],
                    "random_seeds": [3, 7],
                    "representations": ["window_mean_std_pool", "temporal_order", "raw_shape_stats"],
                    "eval_rep_novelty_representations": ["temporal_order", "raw_shape_stats"],
                    "eval_rep_novelty_k_old": 2,
                    "quality_gate_eval_rep_novelty": True,
                    "quality_gated_old_novelty": {
                        "enabled": True,
                        "thresholds": [{"name": "q45", "mode": "fixed", "value": 0.45}],
                        "source_caps": [2],
                        "source_cap_key": "new_cluster_id",
                        "validity_gates": [
                            {"name": "stat90", "max_stationary_fraction": 0.90},
                            {"name": "stat90_abs60", "max_stationary_fraction": 0.90, "max_abs_value": 60.0},
                        ],
                    },
                    "quality_gated_random_controls": [
                        {
                            "name": "quality_gated_random_clustercap2",
                            "quality_threshold": 0.45,
                            "max_stationary_fraction": 0.90,
                            "max_abs_value": 60.0,
                            "source_cap": 2,
                            "source_cap_key": "new_cluster_id",
                        }
                    ],
                    "kcenter_greedy_controls": [
                        {
                            "name": "kcenter_greedy_quality_gated",
                            "representation": "window_mean_std_pool",
                            "quality_threshold": 0.45,
                            "max_stationary_fraction": 0.90,
                            "max_abs_value": 60.0,
                        }
                    ],
                },
                "seed": 11,
            }

            result = run_marginal_coverage_eval(config, smoke=True)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            candidates = [
                json.loads(line)
                for line in Path(result["candidates_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(result["mode"], "smoke")
        self.assertEqual(len(report["folds"]), 1)
        fold = report["folds"][0]
        self.assertGreater(fold["n_support"], 0)
        self.assertGreater(fold["n_candidate"], 0)
        self.assertGreater(fold["n_target"], 0)
        self.assertIn("ranker", fold["policies"])
        self.assertIn("quality_only", fold["policies"])
        self.assertIn("old_novelty_only", fold["policies"])
        self.assertIn("old_novelty_temporal_order", fold["policies"])
        self.assertIn("old_novelty_raw_shape_stats", fold["policies"])
        self.assertIn("quality_gated_old_novelty_temporal_order_q45", fold["policies"])
        self.assertIn("quality_gated_old_novelty_raw_shape_stats_q45", fold["policies"])
        self.assertIn("quality_gated_old_novelty_raw_shape_stats_q45_stat90", fold["policies"])
        self.assertIn("quality_gated_old_novelty_raw_shape_stats_q45_stat90_abs60", fold["policies"])
        self.assertIn("quality_gated_old_novelty_raw_shape_stats_q45_clustercap2", fold["policies"])
        self.assertIn("quality_gated_old_novelty_raw_shape_stats_q45_stat90_abs60_clustercap2", fold["policies"])
        self.assertIn("quality_gated_old_novelty_q45", fold["policies"])
        self.assertIn("quality_gated_old_novelty_q45_stat90", fold["policies"])
        self.assertIn("quality_gated_old_novelty_q45_stat90_abs60", fold["policies"])
        self.assertIn("quality_gated_old_novelty_q45_clustercap2", fold["policies"])
        self.assertIn("random_high_quality", fold["policies"])
        self.assertIn("quality_gated_random_clustercap2", fold["policies"])
        self.assertIn("kcenter_greedy_quality_gated", fold["policies"])
        self.assertIn("window_mean_std_pool", fold["policies"]["ranker"]["coverage@2"])
        self.assertIn("temporal_order", fold["policies"]["ranker"]["coverage@2"])
        self.assertIn("raw_shape_stats", fold["policies"]["ranker"]["coverage@2"])
        self.assertIn(
            "relative_coverage_gain_mean",
            fold["policies"]["quality_gated_random_clustercap2"]["coverage@2"]["raw_shape_stats"],
        )
        self.assertIn(
            "relative_coverage_gain",
            fold["policies"]["kcenter_greedy_quality_gated"]["coverage@2"]["raw_shape_stats"],
        )
        self.assertIn("max_stationary_fraction", fold["policies"]["quality_gated_old_novelty_q45_stat90"]["selection@2"])
        self.assertIn("max_abs_value", fold["policies"]["quality_gated_old_novelty_q45_stat90_abs60"]["selection@2"])
        self.assertTrue(any(int(row["is_target_source_group"]) == 1 for row in candidates))
        self.assertTrue(any(int(row["is_target_source_group"]) == 0 for row in candidates))

    def test_modal_marginal_coverage_entrypoint_dispatches_remote_job(self):
        source = Path("modal_marginal_coverage_eval.py").read_text(encoding="utf-8")
        config = json.loads(Path("configs/marginal_coverage_eval_subcluster40.json").read_text(encoding="utf-8"))

        self.assertIn("marginal-value-marginal-coverage-eval", source)
        self.assertIn("remote_marginal_coverage_eval.remote", source)
        self.assertIn("run_marginal_coverage_eval", source)
        self.assertEqual(config["data"]["pretrain_manifest"], "cache/manifests/pretrain_physical_source_urls.txt")
        self.assertEqual(config["ranking"]["reranker_method"], "tiered_cluster_cap")
        self.assertIn("raw_shape_stats", config["eval"]["representations"])

        qgate_config = json.loads(Path("configs/marginal_coverage_eval_qgate_oldnovelty.json").read_text(encoding="utf-8"))
        self.assertEqual(qgate_config["ranking"]["reranker_method"], "quality_gated_old_novelty")
        self.assertFalse(qgate_config["grammar_features"]["enabled"])
        self.assertFalse(qgate_config["large_cluster_split"]["enabled"])
        self.assertTrue(qgate_config["eval"]["quality_gated_old_novelty"]["enabled"])
        self.assertIn("raw_shape_stats", qgate_config["eval"]["eval_rep_novelty_representations"])
        self.assertTrue(qgate_config["eval"]["quality_gate_eval_rep_novelty"])
        self.assertEqual(qgate_config["eval"]["quality_gated_old_novelty"]["validity_gates"][0]["name"], "stat90")
        self.assertEqual(qgate_config["eval"]["quality_gated_old_novelty"]["validity_gates"][1]["name"], "stat90_abs60")
        self.assertEqual(qgate_config["eval"]["quality_gated_old_novelty"]["source_cap_key"], "new_cluster_id")

        controls_config = json.loads(Path("configs/marginal_coverage_eval_challenge_controls.json").read_text(encoding="utf-8"))
        self.assertEqual(controls_config["artifacts"]["output_dir"], "/artifacts/eval/marginal_coverage/challenge_controls")
        self.assertEqual(
            controls_config["eval"]["quality_gated_random_controls"][0]["name"],
            "quality_gated_random_clustercap2",
        )
        self.assertEqual(
            controls_config["eval"]["kcenter_greedy_controls"][0]["name"],
            "kcenter_greedy_quality_gated",
        )
        self.assertEqual(
            controls_config["eval"]["kcenter_greedy_controls"][0]["representation"],
            "window_shape_stats",
        )


def _write_synthetic_physical_source_dataset(root: Path) -> None:
    (root / "cache" / "features").mkdir(parents=True)
    (root / "cache" / "raw").mkdir(parents=True)
    (root / "cache" / "manifests").mkdir(parents=True)
    urls = []
    for worker_idx in range(6):
        center = 4.0 if worker_idx < 2 else -2.0 + worker_idx * 0.15
        for clip_idx in range(6):
            url = f"https://storage.googleapis.com/unit/pretrain/worker{worker_idx:05d}/clip{clip_idx:03d}.txt"
            sample_id = hash_manifest_url(url)
            urls.append(url)
            windows = np.full((6, 75), center, dtype=np.float32)
            windows[:, 0] += clip_idx * 0.01
            windows[:, 1] += np.linspace(0.0, 0.2 + worker_idx * 0.02, 6, dtype=np.float32)
            np.savez(root / "cache" / "features" / f"{sample_id}.npz", window_features=windows)
            _write_modal_jsonl(
                root / "cache" / "raw" / f"{sample_id}.jsonl",
                _make_clean_imu(90, amplitude=0.2 + worker_idx * 0.05, phase=clip_idx * 0.03),
            )
    (root / "cache" / "manifests" / "pretrain.txt").write_text("\n".join(urls) + "\n", encoding="utf-8")
    (root / "cache" / "manifests" / "val.txt").write_text("", encoding="utf-8")


def _make_clean_imu(n_samples: int, sample_rate: int = 30, amplitude: float = 0.3, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n_samples, dtype=float) / sample_rate
    return np.column_stack(
        [
            amplitude * np.sin(2 * np.pi * 0.7 * t + phase),
            0.5 * amplitude * np.cos(2 * np.pi * 0.7 * t + phase),
            9.81 + 0.1 * np.sin(2 * np.pi * 1.3 * t + phase),
            0.03 * np.cos(2 * np.pi * 0.2 * t + phase),
            0.04 * np.sin(2 * np.pi * 0.4 * t + phase),
            0.02 * np.cos(2 * np.pi * 0.3 * t + phase),
        ]
    )


def _write_modal_jsonl(path: Path, samples: np.ndarray, sample_rate: int = 30) -> None:
    lines = []
    for idx, row in enumerate(np.asarray(samples, dtype=float)):
        lines.append(
            json.dumps(
                {
                    "t_us": int(idx * 1_000_000 / sample_rate),
                    "acc": row[:3].tolist(),
                    "gyro": row[3:6].tolist(),
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
