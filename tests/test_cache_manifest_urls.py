import json
import threading
import unittest
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.data.cache_manifest_urls import cache_manifest_urls, validate_manifest_url_cache_config
from marginal_value.data.split_manifest import hash_manifest_url


class CacheManifestUrlsTests(unittest.TestCase):
    def test_cache_manifest_urls_downloads_missing_rows_and_writes_cached_manifest(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            server_root = root / "server"
            target = root / "data"
            artifacts = root / "artifacts"
            server_root.mkdir()
            (target / "cache" / "manifests").mkdir(parents=True)
            urls, server = _start_fixture_server(server_root, ["clip001.txt", "clip002.txt", "clip003.txt"])
            self.addCleanup(server.shutdown)
            self.addCleanup(server.server_close)

            cached_id = hash_manifest_url(urls[0])
            raw_dir = target / "cache" / "raw"
            feature_dir = target / "cache" / "features"
            raw_dir.mkdir(parents=True)
            feature_dir.mkdir(parents=True)
            raw_path = raw_dir / f"{cached_id}.jsonl"
            raw_path.write_text(_jsonl_payload(phase=0.5), encoding="utf-8")
            np.savez(feature_dir / f"{cached_id}.npz", window_features=np.zeros((1, 75)), clip_features=np.zeros(75))

            (target / "cache" / "manifests" / "pretrain_urls.txt").write_text(
                "\n".join(urls) + "\n",
                encoding="utf-8",
            )

            result = cache_manifest_urls(_config(target, artifacts), smoke=False)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            cached_manifest = (target / "cache" / "manifests" / "pretrain_full_cached_urls.txt").read_text(
                encoding="utf-8"
            ).splitlines()

            self.assertEqual(report["manifest_url_count"], 3)
            self.assertEqual(report["selected"], 2)
            self.assertEqual(report["downloaded"], 2)
            self.assertEqual(report["feature_written"], 2)
            self.assertEqual(report["skipped_existing"], 1)
            self.assertEqual(report["cached_both_count"], 3)
            self.assertEqual(cached_manifest, urls)
            for url in urls:
                sample_id = hash_manifest_url(url)
                self.assertTrue((raw_dir / f"{sample_id}.jsonl").exists())
                self.assertTrue((feature_dir / f"{sample_id}.npz").exists())

    def test_cache_manifest_urls_smoke_limits_missing_selection(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            server_root = root / "server"
            target = root / "data"
            artifacts = root / "artifacts"
            server_root.mkdir()
            (target / "cache" / "manifests").mkdir(parents=True)
            urls, server = _start_fixture_server(server_root, ["clip001.txt", "clip002.txt", "clip003.txt"])
            self.addCleanup(server.shutdown)
            self.addCleanup(server.server_close)
            (target / "cache" / "manifests" / "pretrain_urls.txt").write_text(
                "\n".join(urls) + "\n",
                encoding="utf-8",
            )
            config = _config(target, artifacts)
            config["execution"]["smoke_samples"] = 2

            result = cache_manifest_urls(config, smoke=True)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))

            self.assertEqual(report["selected"], 2)
            self.assertEqual(report["cached_both_count"], 2)
            self.assertTrue((target / "cache" / "manifests" / "pretrain_full_cached_urls.smoke.txt").exists())
            self.assertFalse((target / "cache" / "manifests" / "pretrain_full_cached_urls.txt").exists())

    def test_cache_manifest_urls_limits_work_to_manifest_range(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            server_root = root / "server"
            target = root / "data"
            artifacts = root / "artifacts"
            server_root.mkdir()
            (target / "cache" / "manifests").mkdir(parents=True)
            urls, server = _start_fixture_server(
                server_root,
                ["clip001.txt", "clip002.txt", "clip003.txt", "clip004.txt", "clip005.txt"],
            )
            self.addCleanup(server.shutdown)
            self.addCleanup(server.server_close)
            (target / "cache" / "manifests" / "pretrain_urls.txt").write_text(
                "\n".join(urls) + "\n",
                encoding="utf-8",
            )
            config = _config(target, artifacts)
            config["selection"]["manifest_start_index"] = 1
            config["selection"]["manifest_end_index"] = 4
            config["execution"]["shard_id"] = "range_1_4"

            result = cache_manifest_urls(config, smoke=False)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            cached_manifest = Path(report["cached_manifest"]).read_text(encoding="utf-8").splitlines()

            self.assertEqual(report["source_manifest_url_count"], 5)
            self.assertEqual(report["manifest_start_index"], 1)
            self.assertEqual(report["manifest_end_index"], 4)
            self.assertEqual(report["manifest_url_count"], 3)
            self.assertEqual(report["selected"], 3)
            self.assertEqual(report["feature_written"], 3)
            self.assertEqual(cached_manifest, urls[1:4])
            self.assertFalse((target / "cache" / "raw" / f"{hash_manifest_url(urls[0])}.jsonl").exists())
            self.assertFalse((target / "cache" / "raw" / f"{hash_manifest_url(urls[4])}.jsonl").exists())

    def test_cache_manifest_urls_redownloads_existing_raw_when_feature_extraction_fails(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            server_root = root / "server"
            target = root / "data"
            artifacts = root / "artifacts"
            server_root.mkdir()
            (target / "cache" / "manifests").mkdir(parents=True)
            urls, server = _start_fixture_server(server_root, ["clip001.txt"])
            self.addCleanup(server.shutdown)
            self.addCleanup(server.server_close)

            sample_id = hash_manifest_url(urls[0])
            raw_dir = target / "cache" / "raw"
            raw_dir.mkdir(parents=True)
            (raw_dir / f"{sample_id}.jsonl").write_text("not-jsonl\n", encoding="utf-8")
            (target / "cache" / "manifests" / "pretrain_urls.txt").write_text(urls[0] + "\n", encoding="utf-8")

            result = cache_manifest_urls(_config(target, artifacts), smoke=False)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))

            self.assertEqual(report["selected"], 1)
            self.assertEqual(report["raw_redownloaded_after_feature_error"], 1)
            self.assertEqual(report["failed"], 0)
            self.assertEqual(report["cached_both_count"], 1)
            self.assertIn('"acc"', (raw_dir / f"{sample_id}.jsonl").read_text(encoding="utf-8"))

    def test_cache_manifest_urls_rejects_non_modal_provider(self):
        config = _config(Path("/data"), Path("/artifacts"))
        config["execution"]["provider"] = "local"

        with self.assertRaises(ValueError):
            validate_manifest_url_cache_config(config)

    def test_modal_manifest_url_cache_entrypoint_uses_remote_function(self):
        source = Path("modal_cache_manifest_urls.py").read_text(encoding="utf-8")

        self.assertIn("remote_cache_manifest_urls.remote", source)
        self.assertIn("cache_manifest_urls", source)
        self.assertIn('"numpy==2.2.6"', source)


def _start_fixture_server(server_root: Path, names: list[str]) -> tuple[list[str], ThreadingHTTPServer]:
    for idx, name in enumerate(names):
        (server_root / name).write_text(_jsonl_payload(phase=idx * 0.1), encoding="utf-8")
    handler = partial(SimpleHTTPRequestHandler, directory=str(server_root))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    urls = [f"http://{host}:{port}/{name}" for name in names]
    return urls, server


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


def _config(target: Path, artifacts: Path) -> dict:
    return {
        "execution": {
            "provider": "modal",
            "allow_local_paths_for_tests": True,
            "smoke_samples": 8,
            "progress_every": 2,
            "workers": 2,
        },
        "target": {
            "root": str(target),
            "source_manifest": "cache/manifests/pretrain_urls.txt",
            "cached_manifest": "cache/manifests/pretrain_full_cached_urls.txt",
            "feature_dir": "cache/features",
            "raw_dir": "cache/raw",
        },
        "selection": {
            "strategy": "missing_cache",
        },
        "artifacts": {
            "output_dir": str(artifacts),
        },
    }


if __name__ == "__main__":
    unittest.main()
