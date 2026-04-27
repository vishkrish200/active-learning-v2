import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.data.cache_new_split import build_new_split_cache
from marginal_value.data.cache_support_split import build_support_split_cache
from marginal_value.data.split_manifest import hash_manifest_url
from marginal_value.preprocessing.window_features import compute_window_feature_matrix


class NewSplitCacheTests(unittest.TestCase):
    def test_window_feature_matrix_matches_distribution_prefix_and_shape(self):
        samples = np.zeros((5400, 6), dtype=float)
        samples[:, 0] = np.linspace(0.0, 1.0, 5400)
        samples[:, 1] = 2.0
        samples[:, 2] = 9.8

        features = compute_window_feature_matrix(samples)

        self.assertEqual(features.shape, (35, 75))
        first_acc_norm = np.linalg.norm(samples[:300, :3], axis=1)
        self.assertAlmostEqual(float(features[0, 0]), float(np.mean(first_acc_norm)), places=5)
        self.assertAlmostEqual(float(features[0, 1]), float(np.std(first_acc_norm)), places=5)
        self.assertTrue(np.all(np.isfinite(features)))

    def test_build_new_split_cache_writes_manifest_raw_and_features(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "target"
            (source / "val").mkdir(parents=True)
            (target / "cache" / "manifests").mkdir(parents=True)
            url = "https://storage.googleapis.com/buildai-imu-benchmark-v1-new/val/sample000001.txt"
            (target / "cache" / "manifests" / "val_urls.txt").write_text(url + "\n", encoding="utf-8")
            _write_jsonl(source / "val" / "sample000001.txt", n=5400)

            result = build_new_split_cache(
                {
                    "execution": {"smoke_samples": 1},
                    "source": {"root": str(source), "split": "val"},
                    "target": {
                        "root": str(target),
                        "source_manifest": "cache/manifests/val_urls.txt",
                        "new_manifest": "cache/manifests/new_urls.txt",
                        "empty_val_manifest": "cache/manifests/val_unused_for_submission_urls.txt",
                    },
                },
                smoke=False,
            )

            sample_id = hash_manifest_url(url)
            self.assertEqual(result["written"], 1)
            self.assertTrue((target / "cache" / "raw" / f"{sample_id}.jsonl").exists())
            self.assertTrue((target / "cache" / "features" / f"{sample_id}.npz").exists())
            self.assertEqual((target / "cache" / "manifests" / "new_urls.txt").read_text(encoding="utf-8").strip(), url)
            self.assertEqual((target / "cache" / "manifests" / "val_unused_for_submission_urls.txt").read_text(encoding="utf-8"), "")

    def test_build_support_split_cache_selects_one_clip_per_worker(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "target"
            (target / "cache" / "manifests").mkdir(parents=True)
            urls = [
                "https://storage.googleapis.com/buildai-imu-benchmark-v1-preexisting/pretrain/worker00001/clip001.txt",
                "https://storage.googleapis.com/buildai-imu-benchmark-v1-preexisting/pretrain/worker00001/clip002.txt",
                "https://storage.googleapis.com/buildai-imu-benchmark-v1-preexisting/pretrain/worker00002/clip001.txt",
            ]
            (target / "cache" / "manifests" / "pretrain_urls.txt").write_text("\n".join(urls) + "\n", encoding="utf-8")
            for worker in ("worker00001", "worker00002"):
                worker_dir = source / "pretrain" / worker
                worker_dir.mkdir(parents=True)
                _write_jsonl(worker_dir / "clip001.txt", n=5400)
                _write_jsonl(worker_dir / "clip002.txt", n=5400)

            result = build_support_split_cache(
                {
                    "execution": {"smoke_samples": 8, "full_samples": None},
                    "source": {"root": str(source), "split": "pretrain"},
                    "target": {
                        "root": str(target),
                        "source_manifest": "cache/manifests/pretrain_urls.txt",
                    },
                    "selection": {
                        "strategy": "worker_coverage",
                        "clips_per_worker": 1,
                        "max_workers": 10000,
                    },
                },
                smoke=False,
            )

            self.assertEqual(result["n_selected"], 2)
            self.assertEqual(result["selection"]["n_workers"], 2)
            self.assertEqual(result["feature_written"], 2)
            self.assertEqual(result["raw_copied"], 2)
            for url in (urls[0], urls[2]):
                sample_id = hash_manifest_url(url)
                self.assertTrue((target / "cache" / "raw" / f"{sample_id}.jsonl").exists())
                self.assertTrue((target / "cache" / "features" / f"{sample_id}.npz").exists())

            skipped_id = hash_manifest_url(urls[1])
            self.assertFalse((target / "cache" / "features" / f"{skipped_id}.npz").exists())

    def test_build_support_split_cache_reads_flat_source_mirror(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "target"
            source.mkdir(parents=True)
            (target / "cache" / "manifests").mkdir(parents=True)
            url = "https://storage.googleapis.com/buildai-imu-benchmark-v1-preexisting/pretrain/worker09467/clip003.txt"
            flat_name = "storage.googleapis.com__buildai-imu-benchmark-v1-preexisting__pretrain__worker09467__clip003.txt"
            (target / "cache" / "manifests" / "pretrain_urls.txt").write_text(url + "\n", encoding="utf-8")
            _write_jsonl(source / flat_name, n=5400)

            result = build_support_split_cache(
                {
                    "execution": {"smoke_samples": 8, "full_samples": None},
                    "source": {"root": str(source), "split": "pretrain"},
                    "target": {
                        "root": str(target),
                        "source_manifest": "cache/manifests/pretrain_urls.txt",
                    },
                    "selection": {
                        "strategy": "worker_coverage",
                        "clips_per_worker": 1,
                        "max_workers": 10000,
                    },
                },
                smoke=False,
            )

            sample_id = hash_manifest_url(url)
            self.assertEqual(result["n_selected"], 1)
            self.assertEqual(result["missing_source"], 0)
            self.assertEqual(result["feature_written"], 1)
            self.assertTrue((target / "cache" / "raw" / f"{sample_id}.jsonl").exists())
            self.assertTrue((target / "cache" / "features" / f"{sample_id}.npz").exists())

    def test_build_support_split_cache_selects_all_source_existing_urls(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "target"
            (target / "cache" / "manifests").mkdir(parents=True)
            urls = [
                "https://storage.googleapis.com/buildai-imu-benchmark-v1-preexisting/pretrain/worker00001/clip001.txt",
                "https://storage.googleapis.com/buildai-imu-benchmark-v1-preexisting/pretrain/worker00001/clip002.txt",
                "https://storage.googleapis.com/buildai-imu-benchmark-v1-preexisting/pretrain/worker00002/clip001.txt",
            ]
            (target / "cache" / "manifests" / "pretrain_urls.txt").write_text("\n".join(urls) + "\n", encoding="utf-8")
            for relpath in ("worker00001/clip001.txt", "worker00002/clip001.txt"):
                path = source / "pretrain" / relpath
                path.parent.mkdir(parents=True, exist_ok=True)
                _write_jsonl(path, n=5400)

            result = build_support_split_cache(
                {
                    "execution": {"smoke_samples": 8, "full_samples": None},
                    "source": {"root": str(source), "split": "pretrain"},
                    "target": {
                        "root": str(target),
                        "source_manifest": "cache/manifests/pretrain_urls.txt",
                    },
                    "selection": {
                        "strategy": "source_existing_all",
                    },
                },
                smoke=False,
            )

            self.assertEqual(result["n_selected"], 2)
            self.assertEqual(result["missing_source"], 0)
            self.assertEqual(result["selection"]["n_workers"], 2)
            for url in (urls[0], urls[2]):
                sample_id = hash_manifest_url(url)
                self.assertTrue((target / "cache" / "features" / f"{sample_id}.npz").exists())
            missing_id = hash_manifest_url(urls[1])
            self.assertFalse((target / "cache" / "features" / f"{missing_id}.npz").exists())

    def test_build_support_split_cache_skips_malformed_source_clip(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "target"
            (target / "cache" / "manifests").mkdir(parents=True)
            url = "https://storage.googleapis.com/buildai-imu-benchmark-v1-preexisting/pretrain/worker00001/clip001.txt"
            source_path = source / "pretrain" / "worker00001" / "clip001.txt"
            source_path.parent.mkdir(parents=True, exist_ok=True)
            (target / "cache" / "manifests" / "pretrain_urls.txt").write_text(url + "\n", encoding="utf-8")
            source_path.write_text(json.dumps({"t_us": 0, "acc": [1.0, 2.0]}) + "\n", encoding="utf-8")

            result = build_support_split_cache(
                {
                    "execution": {"smoke_samples": 8, "full_samples": None, "progress_every": 1},
                    "source": {"root": str(source), "split": "pretrain"},
                    "target": {
                        "root": str(target),
                        "source_manifest": "cache/manifests/pretrain_urls.txt",
                    },
                    "selection": {
                        "strategy": "source_existing_all",
                    },
                },
                smoke=False,
            )

            sample_id = hash_manifest_url(url)
            self.assertEqual(result["n_selected"], 1)
            self.assertEqual(result["written"], 0)
            self.assertEqual(result["raw_copied"], 1)
            self.assertEqual(result["feature_written"], 0)
            self.assertEqual(result["malformed_source"], 1)
            self.assertEqual(result["missing_source"], 0)
            self.assertEqual(len(result["malformed_examples"]), 1)
            self.assertTrue((target / "cache" / "raw" / f"{sample_id}.jsonl").exists())
            self.assertFalse((target / "cache" / "features" / f"{sample_id}.npz").exists())

    def test_build_support_split_cache_shards_selected_urls(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "target"
            (target / "cache" / "manifests").mkdir(parents=True)
            urls = [
                f"https://storage.googleapis.com/buildai-imu-benchmark-v1-preexisting/pretrain/worker00001/clip{idx:03d}.txt"
                for idx in range(1, 5)
            ]
            (target / "cache" / "manifests" / "pretrain_urls.txt").write_text("\n".join(urls) + "\n", encoding="utf-8")
            for idx in range(1, 5):
                path = source / "pretrain" / "worker00001" / f"clip{idx:03d}.txt"
                path.parent.mkdir(parents=True, exist_ok=True)
                _write_jsonl(path, n=5400)

            result = build_support_split_cache(
                {
                    "execution": {
                        "smoke_samples": 8,
                        "full_samples": None,
                        "num_shards": 2,
                        "shard_index": 1,
                        "progress_every": 1,
                    },
                    "source": {"root": str(source), "split": "pretrain"},
                    "target": {
                        "root": str(target),
                        "source_manifest": "cache/manifests/pretrain_urls.txt",
                        "report_path": "cache/manifests/pretrain_cache_report.json",
                    },
                    "selection": {
                        "strategy": "source_existing_all",
                    },
                },
                smoke=False,
            )

            self.assertEqual(result["n_selected_before_shard"], 4)
            self.assertEqual(result["n_selected"], 2)
            self.assertEqual(result["shard"], {"num_shards": 2, "shard_index": 1})
            self.assertTrue(result["report_path"].endswith("pretrain_cache_report_shard01of02.json"))
            self.assertFalse((target / "cache" / "features" / f"{hash_manifest_url(urls[0])}.npz").exists())
            self.assertTrue((target / "cache" / "features" / f"{hash_manifest_url(urls[1])}.npz").exists())
            self.assertFalse((target / "cache" / "features" / f"{hash_manifest_url(urls[2])}.npz").exists())
            self.assertTrue((target / "cache" / "features" / f"{hash_manifest_url(urls[3])}.npz").exists())


def _write_jsonl(path: Path, *, n: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index in range(n):
            payload = {
                "t_us": int(index * 33333),
                "acc": [0.1 * index, 2.0, 9.8],
                "gyro": [0.01, 0.02, 0.03],
            }
            handle.write(json.dumps(payload) + "\n")


if __name__ == "__main__":
    unittest.main()
