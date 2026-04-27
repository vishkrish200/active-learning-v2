import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.data.split_manifest import hash_manifest_url
from marginal_value.eval.source_blocked_eval import run_source_blocked_eval


class SourceBlockedEvalTests(unittest.TestCase):
    def test_source_blocked_eval_uses_heldout_source_groups_and_known_negatives(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cache" / "features").mkdir(parents=True)
            (root / "cache" / "raw").mkdir(parents=True)
            (root / "cache" / "manifests").mkdir(parents=True)
            urls = []
            for worker_idx in range(6):
                center = 1.0 if worker_idx < 3 else -1.0
                for clip_idx in range(5):
                    url = f"https://storage.googleapis.com/unit/pretrain/worker{worker_idx:05d}/clip{clip_idx:03d}.txt"
                    sample_id = hash_manifest_url(url)
                    urls.append(url)
                    windows = np.full((4, 75), center, dtype=np.float32)
                    windows[:, 0] += worker_idx * 0.01 + clip_idx * 0.001
                    np.savez(root / "cache" / "features" / f"{sample_id}.npz", window_features=windows)
                    _write_modal_jsonl(
                        root / "cache" / "raw" / f"{sample_id}.jsonl",
                        _make_clean_imu(90, phase=worker_idx * 0.1 + clip_idx * 0.01),
                    )
            (root / "cache" / "manifests" / "pretrain.txt").write_text("\n".join(urls) + "\n", encoding="utf-8")
            (root / "cache" / "manifests" / "val.txt").write_text("", encoding="utf-8")
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
                    "reranker_method": "cluster_cap",
                    "cluster_cap_top_k": 8,
                    "cluster_max_per_cluster": 4,
                    "cluster_cap_key": "new_cluster_id",
                    "cluster_cap_min_quality": 0.45,
                    "cluster_bonus_weight": 0.0,
                    "mmr_lambda": 0.0,
                },
                "grammar_features": {"enabled": False},
                "score_guards": {"stationary_singleton": {"enabled": False}},
                "large_cluster_split": {"enabled": False},
                "corruption_eval": {"enabled": False},
                "quality": {"sample_rate": 30.0, "max_samples_per_clip": 90},
                "eval": {
                    "smoke_max_rows": 30,
                    "smoke_n_clusters": 2,
                    "smoke_folds": 1,
                    "clusters_per_fold": 1,
                    "source_groups_per_fold": 2,
                    "max_positive_per_fold": 6,
                    "max_negative_per_fold": 6,
                    "k_old": 2,
                    "k_new_density": 2,
                    "embedding_load_workers": 1,
                    "k_values": [4, 8],
                },
                "seed": 5,
            }

            result = run_source_blocked_eval(config, smoke=True)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            candidates = [
                json.loads(line)
                for line in Path(result["candidates_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(result["n_rows"], 30)
        self.assertEqual(result["n_source_clusters"], 2)
        self.assertGreaterEqual(result["n_source_groups"], 6)
        self.assertEqual(len(report["folds"]), 1)
        fold = report["folds"][0]
        self.assertGreater(fold["n_support"], 0)
        self.assertGreater(fold["n_positive"], 0)
        self.assertGreater(fold["n_negative"], 0)
        self.assertIn("precision@4", fold["metrics"])
        labels = {int(row["label"]) for row in candidates}
        types = {str(row["candidate_type"]) for row in candidates}
        self.assertEqual(labels, {0, 1})
        self.assertIn("heldout_source_cluster_positive", types)
        self.assertIn("source_covered_hard_negative", types)
        heldout_groups = set(str(group) for group in fold["heldout_source_groups"])
        positive_groups = {str(row["source_group_id"]) for row in candidates if int(row["label"]) == 1}
        negative_groups = {str(row["source_group_id"]) for row in candidates if int(row["label"]) == 0}
        self.assertTrue(positive_groups <= heldout_groups)
        self.assertTrue(negative_groups.isdisjoint(heldout_groups))

    def test_modal_source_blocked_entrypoint_dispatches_remote_job(self):
        source = Path("modal_source_blocked_eval.py").read_text(encoding="utf-8")
        config = json.loads(Path("configs/source_blocked_eval.json").read_text(encoding="utf-8"))
        tiered_config = json.loads(Path("configs/source_blocked_eval_tiered_childcap2_5_subcluster40.json").read_text(encoding="utf-8"))

        self.assertIn("marginal-value-source-blocked-eval", source)
        self.assertIn("remote_source_blocked_eval.remote", source)
        self.assertIn("run_source_blocked_eval", source)
        self.assertEqual(config["data"]["pretrain_manifest"], "cache/manifests/pretrain_physical_source_urls.txt")
        self.assertEqual(config["ranking"]["reranker_method"], "parent_prefix_cluster_cap")
        self.assertEqual(tiered_config["ranking"]["reranker_method"], "tiered_cluster_cap")
        self.assertEqual(tiered_config["large_cluster_split"]["target_subcluster_size"], 40)


def _make_clean_imu(n_samples: int, sample_rate: int = 30, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n_samples, dtype=float) / sample_rate
    return np.column_stack(
        [
            np.sin(2 * np.pi * 0.7 * t + phase),
            0.5 * np.cos(2 * np.pi * 0.7 * t + phase),
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

