import json
import tarfile
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.data.cache_source_archive import cache_source_archive, validate_source_archive_cache_config
from marginal_value.data.split_manifest import hash_manifest_url


class CacheSourceArchiveTests(unittest.TestCase):
    def test_cache_source_archive_caches_missing_members_and_writes_union_manifest(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "data"
            artifacts = root / "artifacts"
            source.mkdir()
            (target / "cache" / "manifests").mkdir(parents=True)
            archive_path = source / "pretrain_fixture.tar"
            urls = [
                _url("worker00001", "clip001"),
                _url("worker00002", "clip001"),
                _url("worker00003", "clip001"),
            ]
            extracted_urls = [urls[0]]
            (target / "cache" / "manifests" / "pretrain_urls.txt").write_text("\n".join(urls) + "\n", encoding="utf-8")
            (target / "cache" / "manifests" / "pretrain_physical_source_urls.txt").write_text(
                "\n".join(extracted_urls) + "\n",
                encoding="utf-8",
            )

            with tarfile.open(archive_path, "w") as archive:
                _add_tar_text(archive, "pretrain/worker00001/clip001.txt", _jsonl_payload())
                _add_tar_text(archive, "pretrain/worker00002/clip001.txt", _jsonl_payload(phase=0.2))
                _add_tar_text(archive, "pretrain/worker99999/clip001.txt", _jsonl_payload(phase=0.4))

            result = cache_source_archive(_config(source, target, artifacts, archive_path.name), smoke=False)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            cached_manifest = (target / "cache" / "manifests" / "pretrain_archive_cached_urls.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            union_manifest = (target / "cache" / "manifests" / "pretrain_physical_plus_archive_urls.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            cached_id = hash_manifest_url(urls[1])

            self.assertEqual(report["selected"], 1)
            self.assertEqual(report["raw_copied"], 1)
            self.assertEqual(report["feature_written"], 1)
            self.assertEqual(report["union_url_count"], 2)
            self.assertEqual(cached_manifest, [urls[1]])
            self.assertEqual(union_manifest, [urls[0], urls[1]])
            self.assertTrue((target / "cache" / "raw" / f"{cached_id}.jsonl").exists())
            self.assertTrue((target / "cache" / "features" / f"{cached_id}.npz").exists())

    def test_cache_source_archive_smoke_limits_selected_missing_members(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "data"
            artifacts = root / "artifacts"
            source.mkdir()
            (target / "cache" / "manifests").mkdir(parents=True)
            archive_path = source / "pretrain_fixture.tar"
            urls = [_url(f"worker{idx:05d}", "clip001") for idx in range(1, 5)]
            (target / "cache" / "manifests" / "pretrain_urls.txt").write_text("\n".join(urls) + "\n", encoding="utf-8")
            (target / "cache" / "manifests" / "pretrain_physical_source_urls.txt").write_text("", encoding="utf-8")
            with tarfile.open(archive_path, "w") as archive:
                for idx, url in enumerate(urls):
                    worker = f"worker{idx + 1:05d}"
                    _add_tar_text(archive, f"pretrain/{worker}/clip001.txt", _jsonl_payload(phase=idx * 0.1))
            config = _config(source, target, artifacts, archive_path.name)
            config["execution"]["smoke_samples"] = 2

            result = cache_source_archive(config, smoke=True)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))

            self.assertEqual(report["selected"], 2)
            self.assertEqual(report["cached_archive_url_count"], 2)
            self.assertTrue((target / "cache" / "manifests" / "pretrain_physical_plus_archive_urls.smoke.txt").exists())
            self.assertFalse((target / "cache" / "manifests" / "pretrain_physical_plus_archive_urls.txt").exists())

    def test_cache_source_archive_rejects_non_modal_provider(self):
        config = _config(Path("/source"), Path("/data"), Path("/artifacts"), "pretrain.tar")
        config["execution"]["provider"] = "local"

        with self.assertRaises(ValueError):
            validate_source_archive_cache_config(config)

    def test_modal_archive_cache_entrypoint_uses_remote_function(self):
        source = Path("modal_cache_source_archive.py").read_text(encoding="utf-8")

        self.assertIn("remote_cache_source_archive.remote", source)
        self.assertIn("cache_source_archive", source)
        self.assertIn('"zstandard==0.23.0"', source)


def _url(worker: str, clip: str) -> str:
    return f"https://storage.googleapis.com/buildai-imu-benchmark-v1-preexisting/pretrain/{worker}/{clip}.txt"


def _add_tar_text(archive: tarfile.TarFile, name: str, text: str) -> None:
    payload = text.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, BytesIO(payload))


def _jsonl_payload(*, phase: float = 0.0, n_samples: int = 90, sample_rate: int = 30) -> str:
    t = np.arange(n_samples, dtype=float) / sample_rate
    samples = np.column_stack(
        [
            np.sin(2 * np.pi * 0.7 * t + phase),
            0.5 * np.cos(2 * np.pi * 0.7 * t + phase),
            9.81 + 0.1 * np.sin(2 * np.pi * 1.3 * t + phase),
            0.03 * np.cos(2 * np.pi * 0.2 * t + phase),
            0.04 * np.sin(2 * np.pi * 0.4 * t + phase),
            0.02 * np.cos(2 * np.pi * 0.3 * t + phase),
        ]
    )
    lines = []
    for idx, row in enumerate(samples):
        lines.append(
            json.dumps(
                {
                    "t_us": int(idx * 1_000_000 / sample_rate),
                    "acc": row[:3].tolist(),
                    "gyro": row[3:6].tolist(),
                }
            )
        )
    return "\n".join(lines) + "\n"


def _config(source: Path, target: Path, artifacts: Path, archive_name: str) -> dict:
    return {
        "execution": {
            "provider": "modal",
            "allow_local_paths_for_tests": True,
            "smoke_samples": 8,
            "progress_every": 2,
            "feature_workers": 2,
        },
        "source": {
            "root": str(source),
            "archive_path": str(source / archive_name),
            "split": "pretrain",
            "url_prefix": "https://storage.googleapis.com/buildai-imu-benchmark-v1-preexisting",
        },
        "target": {
            "root": str(target),
            "source_manifest": "cache/manifests/pretrain_urls.txt",
            "extracted_manifest": "cache/manifests/pretrain_physical_source_urls.txt",
            "union_manifest": "cache/manifests/pretrain_physical_plus_archive_urls.txt",
            "archive_cached_manifest": "cache/manifests/pretrain_archive_cached_urls.txt",
            "feature_dir": "cache/features",
            "raw_dir": "cache/raw",
        },
        "selection": {
            "strategy": "missing_from_extracted",
        },
        "artifacts": {
            "output_dir": str(artifacts),
        },
    }


if __name__ == "__main__":
    unittest.main()
