import json
import tarfile
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.data.source_archive_audit import run_source_archive_audit, validate_source_archive_audit_config


class SourceArchiveAuditTests(unittest.TestCase):
    def test_archive_audit_counts_manifest_overlap_and_missing_extracted_urls(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "data"
            artifacts = root / "artifacts"
            source.mkdir()
            (target / "cache" / "manifests").mkdir(parents=True)
            archive_path = source / "pretrain_fixture.tar"

            manifest_urls = [
                "https://storage.googleapis.com/buildai-imu-benchmark-v1-preexisting/pretrain/worker00001/clip001.txt",
                "https://storage.googleapis.com/buildai-imu-benchmark-v1-preexisting/pretrain/worker00002/clip001.txt",
                "https://storage.googleapis.com/buildai-imu-benchmark-v1-preexisting/pretrain/worker00004/clip001.txt",
            ]
            extracted_urls = [manifest_urls[0]]
            (target / "cache" / "manifests" / "pretrain_urls.txt").write_text(
                "\n".join(manifest_urls) + "\n",
                encoding="utf-8",
            )
            (target / "cache" / "manifests" / "pretrain_physical_source_urls.txt").write_text(
                "\n".join(extracted_urls) + "\n",
                encoding="utf-8",
            )

            with tarfile.open(archive_path, "w") as archive:
                _add_tar_text(archive, "pretrain/worker00001/clip001.txt", "{}")
                _add_tar_text(archive, "pretrain/worker00002/clip001.txt", "{}")
                _add_tar_text(archive, "pretrain/worker00003/clip001.txt", "{}")
                _add_tar_text(archive, "pretrain/worker00002/._clip002.txt", "metadata")

            result = run_source_archive_audit(_config(source, target, artifacts, archive_path.name), smoke=False)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))

            self.assertEqual(report["archive"]["data_member_count"], 3)
            self.assertEqual(report["archive"]["metadata_member_count"], 1)
            self.assertEqual(report["archive"]["worker_count"], 3)
            self.assertEqual(report["overlap"]["manifest_match_count"], 2)
            self.assertEqual(report["overlap"]["extracted_duplicate_count"], 1)
            self.assertEqual(report["overlap"]["missing_from_extracted_count"], 1)
            self.assertEqual(report["overlap"]["not_in_manifest_count"], 1)
            self.assertEqual(result["archive_missing_from_extracted_count"], 1)
            self.assertTrue((artifacts / "source_archive_members_full.txt").exists())

    def test_archive_audit_smoke_limits_data_members_but_keeps_manifest_counts(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "data"
            artifacts = root / "artifacts"
            source.mkdir()
            (target / "cache" / "manifests").mkdir(parents=True)
            archive_path = source / "pretrain_fixture.tar"
            urls = [
                f"https://storage.googleapis.com/buildai-imu-benchmark-v1-preexisting/pretrain/worker{idx:05d}/clip001.txt"
                for idx in range(1, 5)
            ]
            (target / "cache" / "manifests" / "pretrain_urls.txt").write_text("\n".join(urls) + "\n", encoding="utf-8")
            (target / "cache" / "manifests" / "pretrain_physical_source_urls.txt").write_text("", encoding="utf-8")
            with tarfile.open(archive_path, "w") as archive:
                for idx in range(1, 5):
                    _add_tar_text(archive, f"pretrain/worker{idx:05d}/clip001.txt", "{}")

            config = _config(source, target, artifacts, archive_path.name)
            config["execution"]["smoke_member_limit"] = 2
            result = run_source_archive_audit(config, smoke=True)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))

            self.assertEqual(report["manifest"]["manifest_url_count"], 4)
            self.assertEqual(report["archive"]["data_member_count"], 2)
            self.assertEqual(report["archive"]["stopped_early"], True)

    def test_archive_audit_rejects_non_modal_provider(self):
        config = _config(Path("/source"), Path("/data"), Path("/artifacts"), "pretrain.tar")
        config["execution"]["provider"] = "local"

        with self.assertRaises(ValueError):
            validate_source_archive_audit_config(config)

    def test_modal_archive_audit_entrypoint_uses_remote_function_and_zstandard(self):
        source = Path("modal_source_archive_audit.py").read_text(encoding="utf-8")

        self.assertIn("remote_source_archive_audit.remote", source)
        self.assertIn("run_source_archive_audit", source)
        self.assertIn('"zstandard==0.23.0"', source)


def _add_tar_text(archive: tarfile.TarFile, name: str, text: str) -> None:
    payload = text.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, BytesIO(payload))


def _config(source: Path, target: Path, artifacts: Path, archive_name: str) -> dict:
    return {
        "execution": {
            "provider": "modal",
            "allow_local_paths_for_tests": True,
            "smoke_member_limit": 8,
        },
        "source": {
            "root": str(source),
            "archive_path": str(source / archive_name),
            "split": "pretrain",
            "url_prefix": "https://storage.googleapis.com/buildai-imu-benchmark-v1-preexisting",
        },
        "target": {
            "root": str(target),
            "manifest": "cache/manifests/pretrain_urls.txt",
            "extracted_manifest": "cache/manifests/pretrain_physical_source_urls.txt",
        },
        "artifacts": {
            "output_dir": str(artifacts),
            "member_urls_path": "source_archive_members_full.txt",
        },
    }


if __name__ == "__main__":
    unittest.main()
