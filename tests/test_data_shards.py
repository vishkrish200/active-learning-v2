import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.data.build_full_support_shards import (
    run_build_full_support_shards,
    validate_build_full_support_shards_config,
)
from marginal_value.data.shard_reader import iter_shard_arrays, load_representation_matrix, load_shard_manifest
from marginal_value.data.split_manifest import hash_manifest_url


class DataShardTests(unittest.TestCase):
    def test_build_full_support_shards_writes_representations_metadata_and_progress_log(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fixture(root, n_old=3, n_new=2)
            config = _config(root)

            result = run_build_full_support_shards(config, smoke=True)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            progress_rows = [
                json.loads(line)
                for line in Path(result["progress_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            manifest = load_shard_manifest(result["manifest_path"])
            shards = list(iter_shard_arrays(manifest))
            sample_ids, matrix = load_representation_matrix(manifest, "window_mean_std_pool")

        self.assertEqual(result["mode"], "smoke")
        self.assertEqual(result["n_clips"], 4)
        self.assertEqual(result["n_shards"], 2)
        self.assertEqual(report["status"], "done")
        self.assertEqual(report["coverage"]["pretrain"]["selected_count"], 2)
        self.assertEqual(report["coverage"]["new"]["selected_count"], 2)
        self.assertTrue(any(row["event"] == "shard_start" for row in progress_rows))
        shard_write_rows = [row for row in progress_rows if row["event"] == "shard_write"]
        self.assertTrue(shard_write_rows)
        self.assertIn("completed_fraction", shard_write_rows[-1])
        self.assertIn("eta_seconds", shard_write_rows[-1])
        self.assertTrue(any(row["event"] == "report_write" for row in progress_rows))
        self.assertEqual(report["quality_metadata"]["quality_score_present_count"], 4)
        self.assertEqual(report["quality_metadata"]["all_quality_fields_present_count"], 4)
        self.assertEqual(len(shards), 2)
        self.assertEqual(len(sample_ids), 4)
        self.assertEqual(matrix.shape[0], 4)
        self.assertIn("quality_score", shards[0])
        self.assertIn("rep__window_mean_std_pool", shards[0])
        self.assertIn("imu_samples", shards[0])
        self.assertEqual(shards[0]["imu_samples"].shape[1:], (8, 6))
        self.assertTrue(np.isfinite(shards[0]["imu_samples"]).all())

    def test_full_shard_config_can_fail_closed_on_split_caps(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fixture(root, n_old=3, n_new=1)
            config = _config(root)
            config["shards"]["max_clips_per_split"] = 2
            config["shards"]["fail_if_split_exceeds_max"] = True

            with self.assertRaisesRegex(ValueError, "exceeding shards.max_clips_per_split"):
                run_build_full_support_shards(config, smoke=False)

    def test_modal_build_full_support_shards_entrypoint_and_config_validate(self):
        source = Path("modal_build_full_support_shards.py").read_text(encoding="utf-8")
        config = json.loads(Path("configs/build_full_support_shards.json").read_text(encoding="utf-8"))

        self.assertIn("marginal-value-build-full-support-shards", source)
        self.assertIn("run_build_full_support_shards(config, smoke=smoke", source)
        self.assertIn("remote_build_full_support_shards.remote(config, smoke=True)", source)
        self.assertIn("remote_build_full_support_shards.spawn(config, smoke=False)", source)
        self.assertIn("cpu=16", source)
        self.assertNotIn("gpu=", source)
        self.assertEqual(config["shards"]["clip_splits"], ["pretrain", "new"])
        self.assertEqual(config["shards"]["progress_every_shards"], 1)
        validate_build_full_support_shards_config(config)


def _config(root: Path) -> dict[str, object]:
    return {
        "execution": {
            "provider": "modal",
            "allow_local_paths_for_tests": True,
            "smoke_max_clips_per_split": 2,
        },
        "data": {
            "root": str(root),
            "feature_glob": "cache/features/*.npz",
            "raw_glob": "cache/raw/*.jsonl",
            "quality_metadata": "quality.jsonl",
            "manifests": {
                "pretrain": "cache/manifests/pretrain_full_cached_urls.txt",
                "new": "cache/manifests/new_urls.txt",
            },
        },
        "shards": {
            "output_dir": str(root / "out" / "full_support_shards"),
            "clip_splits": ["pretrain", "new"],
            "representations": ["window_mean_std_pool", "temporal_order"],
            "sample_rate": 30.0,
            "shard_size": 2,
            "workers": 1,
            "include_imu_samples": True,
            "imu_max_samples": 8,
            "progress_every_shards": 1,
        },
    }


def _write_fixture(root: Path, *, n_old: int, n_new: int) -> None:
    old_urls = [_url("pretrain", idx) for idx in range(n_old)]
    new_urls = [_url("new", idx) for idx in range(n_new)]
    for idx, url in enumerate([*old_urls, *new_urls]):
        _write_cached_clip(root, url, center=np.asarray([float(idx + 1), 0.0, 0.0, 0.0], dtype=float))
    _write_manifest(root, "cache/manifests/pretrain_full_cached_urls.txt", old_urls)
    _write_manifest(root, "cache/manifests/new_urls.txt", new_urls)
    quality_rows = [
        {
            "sample_id": hash_manifest_url(url),
            "quality_score": 0.9 + 0.001 * idx,
            "stationary_fraction": 0.1,
            "max_abs_value": 10.0 + idx,
        }
        for idx, url in enumerate([*old_urls, *new_urls])
    ]
    (root / "quality.jsonl").write_text("\n".join(json.dumps(row) for row in quality_rows) + "\n", encoding="utf-8")


def _url(split: str, idx: int) -> str:
    return f"https://storage.googleapis.com/unit/{split}/worker{idx:05d}/clip{idx:03d}.jsonl"


def _write_manifest(root: Path, relpath: str, urls: list[str]) -> None:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(urls) + "\n", encoding="utf-8")


def _write_cached_clip(root: Path, url: str, *, center: np.ndarray) -> None:
    sample_id = hash_manifest_url(url)
    raw_dir = root / "cache" / "raw"
    feature_dir = root / "cache" / "features"
    raw_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    np.savez(feature_dir / f"{sample_id}.npz", window_features=np.vstack([center, center + 0.1]).astype("float32"))
    rows = []
    for idx in range(12):
        rows.append(
            json.dumps(
                {
                    "t_us": idx * 33333,
                    "acc": [1.0 + 0.01 * idx, 0.1, 9.81],
                    "gyro": [0.0, 0.01 * idx, 0.0],
                }
            )
        )
    (raw_dir / f"{sample_id}.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
