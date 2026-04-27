import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.data.split_manifest import hash_manifest_url
from marginal_value.eval.physical_leave_cluster_eval import run_physical_leave_cluster_eval


class PhysicalLeaveClusterEvalTests(unittest.TestCase):
    def test_physical_leave_cluster_eval_holds_out_old_clusters(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cache" / "features").mkdir(parents=True)
            (root / "cache" / "raw").mkdir(parents=True)
            (root / "cache" / "manifests").mkdir(parents=True)
            urls = []
            for idx in range(24):
                url = f"s3://unit/pretrain/{idx:03d}.jsonl"
                sample_id = hash_manifest_url(url)
                urls.append(url)
                center = 1.0 if idx < 12 else -1.0
                windows = np.full((4, 75), center, dtype=np.float32)
                windows[:, 0] += idx * 0.001
                np.savez(root / "cache" / "features" / f"{sample_id}.npz", window_features=windows)
                _write_modal_jsonl(root / "cache" / "raw" / f"{sample_id}.jsonl", _make_clean_imu(90, phase=idx * 0.01))
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
                "quality": {"sample_rate": 30.0, "max_samples_per_clip": 90},
                "eval": {
                    "smoke_max_rows": 24,
                    "smoke_n_clusters": 2,
                    "smoke_folds": 1,
                    "clusters_per_fold": 1,
                    "max_positive_per_fold": 6,
                    "max_negative_per_fold": 6,
                    "k_old": 2,
                    "k_new_density": 2,
                    "embedding_load_workers": 1,
                    "k_values": [4, 8],
                },
                "seed": 3,
            }

            result = run_physical_leave_cluster_eval(config, smoke=True)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))

        self.assertEqual(result["n_rows"], 24)
        self.assertEqual(result["n_source_clusters"], 2)
        self.assertEqual(len(report["folds"]), 1)
        self.assertIn("ndcg@4", report["folds"][0]["metrics"])
        self.assertGreater(report["folds"][0]["n_support"], 0)
        self.assertGreater(report["folds"][0]["n_positive"], 0)

    def test_modal_physical_leave_cluster_entrypoint_dispatches_remote_job(self):
        source = Path("modal_physical_leave_cluster_eval.py").read_text(encoding="utf-8")
        config = json.loads(Path("configs/physical_leave_cluster_eval.json").read_text(encoding="utf-8"))

        self.assertIn("marginal-value-physical-leave-cluster-eval", source)
        self.assertIn("remote_physical_leave_cluster_eval.remote", source)
        self.assertIn("run_physical_leave_cluster_eval", source)
        self.assertEqual(config["data"]["pretrain_manifest"], "cache/manifests/pretrain_physical_source_urls.txt")
        self.assertTrue(config["artifacts"]["output_dir"].startswith("/artifacts/"))


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
