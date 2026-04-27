import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.data.split_manifest import hash_manifest_url
from marginal_value.data.support_coverage_audit import (
    run_support_coverage_audit,
    validate_support_coverage_config,
)


class SupportCoverageAuditTests(unittest.TestCase):
    def test_support_coverage_audit_counts_manifest_cache_source_and_windows(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "data"
            artifacts = root / "artifacts"
            (source / "pretrain" / "worker00001").mkdir(parents=True)
            (source / "val").mkdir(parents=True)
            (target / "cache" / "manifests").mkdir(parents=True)
            (target / "cache" / "raw").mkdir(parents=True)
            (target / "cache" / "features").mkdir(parents=True)

            pretrain_urls = [
                "https://storage.googleapis.com/buildai-imu-benchmark-v1-preexisting/pretrain/worker00001/clip001.txt",
                "https://storage.googleapis.com/buildai-imu-benchmark-v1-preexisting/pretrain/worker00002/clip001.txt",
            ]
            new_urls = [
                "https://storage.googleapis.com/buildai-imu-benchmark-v1-new/val/sample000001.txt",
            ]
            (target / "cache" / "manifests" / "pretrain_urls.txt").write_text("\n".join(pretrain_urls) + "\n", encoding="utf-8")
            (target / "cache" / "manifests" / "new_urls.txt").write_text("\n".join(new_urls) + "\n", encoding="utf-8")
            (target / "cache" / "manifests" / "val_unused_for_submission_urls.txt").write_text("", encoding="utf-8")
            (source / "pretrain" / "worker00001" / "clip001.txt").write_text("source\n", encoding="utf-8")
            (source / "val" / "sample000001.txt").write_text("source\n", encoding="utf-8")

            cached_urls = [pretrain_urls[0], new_urls[0]]
            for url in cached_urls:
                sample_id = hash_manifest_url(url)
                (target / "cache" / "raw" / f"{sample_id}.jsonl").write_text("{}\n", encoding="utf-8")
                np.savez(
                    target / "cache" / "features" / f"{sample_id}.npz",
                    window_features=np.zeros((35, 75), dtype=np.float32),
                )

            result = run_support_coverage_audit(_config(source, target, artifacts), smoke=False)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))

            self.assertEqual(report["splits"]["pretrain"]["manifest_url_count"], 2)
            self.assertEqual(report["splits"]["pretrain"]["manifest_worker_count"], 2)
            self.assertEqual(report["splits"]["pretrain"]["cached_both_count"], 1)
            self.assertEqual(report["splits"]["pretrain"]["source_exists_count"], 1)
            self.assertEqual(report["splits"]["pretrain"]["feature_window_summary"]["window_count"]["mean"], 35.0)
            self.assertEqual(report["splits"]["new"]["cached_both_count"], 1)
            self.assertEqual(report["inventory"]["feature_raw_intersection_count"], 2)
            self.assertEqual(result["pretrain_cached_both_count"], 1)

    def test_support_coverage_audit_smoke_limits_expensive_checks_but_keeps_manifest_count(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "data"
            artifacts = root / "artifacts"
            (source / "pretrain" / "worker00001").mkdir(parents=True)
            (target / "cache" / "manifests").mkdir(parents=True)
            (target / "cache" / "raw").mkdir(parents=True)
            (target / "cache" / "features").mkdir(parents=True)
            urls = [
                f"https://storage.googleapis.com/buildai-imu-benchmark-v1-preexisting/pretrain/worker{idx:05d}/clip001.txt"
                for idx in range(1, 5)
            ]
            (target / "cache" / "manifests" / "pretrain_urls.txt").write_text("\n".join(urls) + "\n", encoding="utf-8")
            (target / "cache" / "manifests" / "new_urls.txt").write_text("", encoding="utf-8")
            (target / "cache" / "manifests" / "val_unused_for_submission_urls.txt").write_text("", encoding="utf-8")

            config = _config(source, target, artifacts)
            config["execution"]["smoke_manifest_samples"] = 2
            result = run_support_coverage_audit(config, smoke=True)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))

            self.assertEqual(report["splits"]["pretrain"]["manifest_url_count"], 4)
            self.assertEqual(report["splits"]["pretrain"]["inspected_url_count"], 2)

    def test_modal_support_coverage_entrypoint_uses_remote_function(self):
        source = Path("modal_support_coverage_audit.py").read_text(encoding="utf-8")

        self.assertIn("remote_support_coverage_audit.remote", source)
        self.assertIn("run_support_coverage_audit", source)
        self.assertIn("support_coverage_audit_worker_coverage.json", source)


def _config(source: Path, target: Path, artifacts: Path) -> dict:
    return {
        "execution": {
            "provider": "modal",
            "smoke_manifest_samples": 8,
            "allow_local_paths_for_tests": True,
        },
        "source": {
            "root": str(source),
            "split_by_manifest": {
                "pretrain": "pretrain",
                "new": "val",
                "val": "val",
            },
        },
        "target": {
            "root": str(target),
            "feature_glob": "cache/features/*.npz",
            "raw_glob": "cache/raw/*.jsonl",
            "max_feature_files_for_window_stats": 100,
            "manifests": {
                "pretrain": "cache/manifests/pretrain_urls.txt",
                "new": "cache/manifests/new_urls.txt",
                "val": "cache/manifests/val_unused_for_submission_urls.txt",
            },
        },
        "artifacts": {
            "output_dir": str(artifacts),
        },
        "expected": {
            "old_workers": 10000,
            "new_workers": 2000,
            "old_three_min_windows": 1150000,
        },
    }


if __name__ == "__main__":
    unittest.main()
