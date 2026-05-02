import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.active.quality_metadata import build_active_quality_metadata
from marginal_value.data.split_manifest import hash_manifest_url


class ActiveQualityMetadataTests(unittest.TestCase):
    def test_build_active_quality_metadata_writes_rows_and_registry_coverage(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            urls = [
                _url("pretrain", "worker00000", 0),
                _url("pretrain", "worker00001", 0),
                _url("pretrain", "worker00002", 0),
                _url("pretrain", "worker99999", 0),
            ]
            _write_manifest(root, "cache/manifests/pretrain_full_cached_urls.txt", urls)
            _write_cached_clip(root, urls[0], samples=_moving_samples())
            _write_cached_clip(root, urls[1], samples=np.zeros((90, 6), dtype=float))
            _write_cached_clip(root, urls[2], samples=_moving_samples(phase=0.4))
            config = {
                "execution": {
                    "provider": "modal",
                    "allow_local_paths_for_tests": True,
                    "smoke_max_clips_per_split": 3,
                },
                "data": {
                    "root": str(root),
                    "feature_glob": "cache/features/*.npz",
                    "raw_glob": "cache/raw/*.jsonl",
                    "manifests": {
                        "pretrain": "cache/manifests/pretrain_full_cached_urls.txt",
                    },
                },
                "quality": {
                    "sample_rate": 30.0,
                    "max_samples_per_clip": 90,
                    "low_quality_threshold": 0.45,
                    "selection": {
                        "splits": ["pretrain"],
                        "sort_by": "sample_id",
                    },
                },
                "artifacts": {"output_dir": str(root / "out")},
            }

            result = build_active_quality_metadata(config, smoke=True)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in Path(result["metadata_path"]).read_text(encoding="utf-8").splitlines()]
            with Path(result["metadata_csv_path"]).open(newline="") as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertEqual(result["mode"], "smoke")
        self.assertEqual(report["registry_coverage_summary"]["manifest_old_url_count"], 4)
        self.assertEqual(report["registry_coverage_summary"]["cached_old_url_count"], 3)
        self.assertEqual(report["registry_coverage_summary"]["skipped_uncached_count"], 1)
        self.assertEqual(report["selected_clip_count"], 3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(csv_rows), 3)
        self.assertGreaterEqual(report["low_quality_count"], 1)
        self.assertTrue(any(float(row["quality_score"]) <= 0.45 for row in rows))

    def test_modal_active_quality_metadata_entrypoint_uses_remote_function(self):
        source = Path("modal_active_quality_metadata.py").read_text(encoding="utf-8")
        config = json.loads(Path("configs/active_quality_metadata_smoke_full_pretrain.json").read_text(encoding="utf-8"))

        self.assertIn("marginal-value-active-quality-metadata", source)
        self.assertIn("remote_active_quality_metadata.remote(config, smoke=True)", source)
        self.assertIn("build_active_quality_metadata", source)
        self.assertIn("artifacts_volume.commit()", source)
        self.assertEqual(config["data"]["manifests"]["pretrain"], "cache/manifests/pretrain_full_cached_urls.txt")
        self.assertEqual(config["artifacts"]["output_dir"], "/artifacts/active/quality/full_pretrain_smoke")


def _url(split: str, worker: str, clip: int) -> str:
    return f"https://storage.googleapis.com/unit/{split}/{worker}/clip{clip:03d}.jsonl"


def _write_manifest(root: Path, relpath: str, urls: list[str]) -> None:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(urls) + "\n", encoding="utf-8")


def _write_cached_clip(root: Path, url: str, *, samples: np.ndarray) -> None:
    sample_id = hash_manifest_url(url)
    raw_dir = root / "cache" / "raw"
    feature_dir = root / "cache" / "features"
    raw_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    _write_raw_jsonl(raw_dir / f"{sample_id}.jsonl", samples)
    np.savez(
        feature_dir / f"{sample_id}.npz",
        window_features=np.full((4, 5), float(np.mean(samples[:, 0])), dtype=np.float32),
        clip_features=np.full(5, float(np.mean(samples[:, 0])), dtype=np.float32),
    )


def _moving_samples(n_samples: int = 90, sample_rate: int = 30, phase: float = 0.0) -> np.ndarray:
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


def _write_raw_jsonl(path: Path, samples: np.ndarray, sample_rate: int = 30) -> None:
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
