import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.active.support_subset_audit import run_support_subset_audit, validate_support_subset_audit_config
from marginal_value.data.split_manifest import hash_manifest_url


class ActiveSupportSubsetAuditTests(unittest.TestCase):
    def test_support_subset_audit_reports_partial_and_window_coverage(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            urls = _write_fixture(root)
            left_shard_dir = root / "left_shards"
            _write_left_shard(left_shard_dir, [hash_manifest_url(url) for url in urls[:4]])
            config = {
                "execution": {
                    "provider": "modal",
                    "allow_local_paths_for_tests": True,
                    "smoke_support_samples": 3,
                },
                "data": {
                    "root": str(root),
                    "feature_glob": "cache/features/*.npz",
                    "raw_glob": "cache/raw/*.jsonl",
                    "quality_metadata": "quality.jsonl",
                    "manifests": {"pretrain": "cache/manifests/pretrain_full_cached_urls.txt"},
                },
                "ranking": {
                    "support_split": "pretrain",
                    "left_support_shard_dir": str(left_shard_dir),
                    "right_support_max_clips": 5,
                },
                "artifacts": {"output_dir": str(root / "out")},
            }

            result = run_support_subset_audit(config, smoke=False)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            source_csv = Path(result["source_groups_path"]).read_text(encoding="utf-8")

        self.assertEqual(result["n_full_support"], 6)
        self.assertEqual(result["n_partial_ts2vec_support"], 4)
        self.assertEqual(result["n_window_support"], 5)
        self.assertEqual(report["summaries"]["partial_ts2vec_support"]["unique_workers"], 4)
        self.assertEqual(report["summaries"]["window_support_subset"]["quality_metadata_present_count"], 5)
        self.assertIn("source_group_id", source_csv)
        self.assertIn("partial_ts2vec_support_count", source_csv)

    def test_support_subset_audit_uses_seeded_window_support_subset(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            urls = _write_fixture(root)
            left_shard_dir = root / "left_shards"
            _write_left_shard(left_shard_dir, [hash_manifest_url(url) for url in urls[:4]])
            config = {
                "execution": {
                    "provider": "modal",
                    "allow_local_paths_for_tests": True,
                    "smoke_support_samples": 3,
                },
                "data": {
                    "root": str(root),
                    "feature_glob": "cache/features/*.npz",
                    "raw_glob": "cache/raw/*.jsonl",
                    "quality_metadata": "quality.jsonl",
                    "manifests": {"pretrain": "cache/manifests/pretrain_full_cached_urls.txt"},
                },
                "ranking": {
                    "support_split": "pretrain",
                    "left_support_shard_dir": str(left_shard_dir),
                    "right_support_max_clips": 3,
                    "right_support_seed": 11,
                },
                "artifacts": {"output_dir": str(root / "out")},
            }

            result = run_support_subset_audit(config, smoke=False)
            worker_rows = {row["worker_id"]: row for row in _read_csv(Path(result["workers_path"]))}
            sorted_workers = [
                f"worker{urls.index(url):05d}"
                for url in sorted(urls, key=lambda value: hash_manifest_url(value))
            ]
            expected_workers = {sorted_workers[index] for index in (0, 4, 5)}

        self.assertEqual(result["n_window_support"], 3)
        for worker, row in worker_rows.items():
            expected_count = 1 if worker in expected_workers else 0
            self.assertEqual(int(row["window_support_subset_count"]), expected_count)

    def test_modal_support_subset_audit_entrypoint_and_config_validate(self):
        source = Path("modal_active_support_subset_audit.py").read_text(encoding="utf-8")
        config = json.loads(Path("configs/active_final_blend_rank_budget_h100.json").read_text(encoding="utf-8"))

        self.assertIn("marginal-value-active-support-subset-audit", source)
        self.assertIn("run_support_subset_audit(config, smoke=smoke)", source)
        self.assertIn("timeout=1800", source)
        validate_support_subset_audit_config(config)


def _write_fixture(root: Path) -> list[str]:
    urls = [f"https://storage.googleapis.com/unit/pretrain/worker{idx:05d}/clip{idx:03d}.jsonl" for idx in range(6)]
    for url in urls:
        sample_id = hash_manifest_url(url)
        raw_dir = root / "cache" / "raw"
        feature_dir = root / "cache" / "features"
        raw_dir.mkdir(parents=True, exist_ok=True)
        feature_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / f"{sample_id}.jsonl").write_text(
            json.dumps({"t_us": 0, "acc": [0.0, 0.0, 9.8], "gyro": [0.0, 0.0, 0.0]}) + "\n",
            encoding="utf-8",
        )
        np.savez(feature_dir / f"{sample_id}.npz", window_features=np.zeros((1, 4), dtype="float32"))
    manifest = root / "cache" / "manifests" / "pretrain_full_cached_urls.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(urls) + "\n", encoding="utf-8")
    quality_rows = [
        {
            "sample_id": hash_manifest_url(url),
            "quality_score": 0.9,
            "stationary_fraction": 0.1,
            "max_abs_value": 10.0,
        }
        for url in urls
    ]
    (root / "quality.jsonl").write_text("\n".join(json.dumps(row) for row in quality_rows) + "\n", encoding="utf-8")
    return urls


def _write_left_shard(shard_dir: Path, sample_ids: list[str]) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        shard_dir / "shard_00000.npz",
        sample_ids=np.asarray(sample_ids, dtype=str),
        rep__ts2vec=np.zeros((len(sample_ids), 3), dtype="float32"),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
