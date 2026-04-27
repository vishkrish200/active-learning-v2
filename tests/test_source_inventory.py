import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.data.source_inventory import run_source_inventory, validate_source_inventory_config


class SourceInventoryTests(unittest.TestCase):
    def test_source_inventory_counts_source_manifest_coverage_and_archives(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "data"
            artifacts = root / "artifacts"
            (source / "pretrain" / "worker00001").mkdir(parents=True)
            (source / "val").mkdir(parents=True)
            (target / "cache" / "manifests").mkdir(parents=True)
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
            (source / "pretrain" / "worker00001" / "clip001.txt").write_text("{}", encoding="utf-8")
            (source / "pretrain" / "worker00001" / "._clip001.txt").write_text("metadata", encoding="utf-8")
            (source / "val" / "sample000001.txt").write_text("{}", encoding="utf-8")
            (source / "pretrain_100k.tar.zst").write_bytes(b"archive")

            result = run_source_inventory(_config(source, target, artifacts), smoke=False)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))

            self.assertEqual(report["manifests"]["pretrain"]["manifest_url_count"], 2)
            self.assertEqual(report["manifests"]["pretrain"]["source_existing_count"], 1)
            self.assertEqual(report["manifests"]["new"]["source_existing_count"], 1)
            self.assertEqual(report["source"]["split_trees"]["pretrain"]["txt_file_count"], 1)
            self.assertEqual(report["source"]["split_trees"]["pretrain"]["worker_count"], 1)
            self.assertEqual(len(report["source"]["tar_files"]), 1)
            self.assertEqual(report["physical_source_manifest"]["url_count"], 1)
            self.assertTrue((target / "cache" / "manifests" / "pretrain_physical_source_urls.txt").exists())
            self.assertEqual(result["pretrain_source_existing_count"], 1)

    def test_source_inventory_smoke_limits_manifest_source_checks(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "data"
            artifacts = root / "artifacts"
            (source / "pretrain" / "worker00001").mkdir(parents=True)
            (target / "cache" / "manifests").mkdir(parents=True)
            urls = [
                f"https://storage.googleapis.com/buildai-imu-benchmark-v1-preexisting/pretrain/worker{idx:05d}/clip001.txt"
                for idx in range(1, 5)
            ]
            (target / "cache" / "manifests" / "pretrain_urls.txt").write_text("\n".join(urls) + "\n", encoding="utf-8")
            (target / "cache" / "manifests" / "new_urls.txt").write_text("", encoding="utf-8")
            (target / "cache" / "manifests" / "val_unused_for_submission_urls.txt").write_text("", encoding="utf-8")

            config = _config(source, target, artifacts)
            config["execution"]["smoke_manifest_samples"] = 2
            result = run_source_inventory(config, smoke=True)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))

            self.assertEqual(report["manifests"]["pretrain"]["manifest_url_count"], 4)
            self.assertEqual(report["manifests"]["pretrain"]["inspected_url_count"], 2)

    def test_source_inventory_rejects_non_modal_provider(self):
        config = _config(Path("/source"), Path("/data"), Path("/artifacts"))
        config["execution"]["provider"] = "local"

        with self.assertRaises(ValueError):
            validate_source_inventory_config(config)

    def test_modal_source_inventory_entrypoint_uses_remote_function(self):
        source = Path("modal_source_inventory.py").read_text(encoding="utf-8")

        self.assertIn("remote_source_inventory.remote", source)
        self.assertIn("run_source_inventory", source)
        self.assertIn("source_inventory_observe_full.json", source)


def _config(source: Path, target: Path, artifacts: Path) -> dict:
    return {
        "execution": {
            "provider": "modal",
            "allow_local_paths_for_tests": True,
            "smoke_manifest_samples": 8,
            "smoke_scan_files": 8,
        },
        "source": {
            "root": str(source),
            "scan_splits": ["pretrain", "val"],
            "split_by_manifest": {
                "pretrain": "pretrain",
                "new": "val",
                "val": "val",
            },
        },
        "target": {
            "root": str(target),
            "manifests": {
                "pretrain": "cache/manifests/pretrain_urls.txt",
                "new": "cache/manifests/new_urls.txt",
                "val": "cache/manifests/val_unused_for_submission_urls.txt",
            },
            "physical_source_manifest": "cache/manifests/pretrain_physical_source_urls.txt",
        },
        "artifacts": {
            "output_dir": str(artifacts),
        },
    }


if __name__ == "__main__":
    unittest.main()
