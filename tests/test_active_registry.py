import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.active.registry import (
    RegistryLeakageError,
    build_clip_registry,
    source_group_id_from_url,
)
from marginal_value.data.split_manifest import hash_manifest_url


class ActiveRegistryTests(unittest.TestCase):
    def test_build_clip_registry_uses_cached_manifest_rows_and_source_groups(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            urls = [
                _url("pretrain", "worker00001", 0),
                _url("pretrain", "worker00001", 1),
                _url("pretrain", "worker00002", 0),
            ]
            _write_manifest(root, "cache/manifests/pretrain.txt", urls)
            _write_cached_clip(root, urls[0], quality_score=0.91)
            _write_cached_clip(root, urls[2], quality_score=0.72)
            _write_quality_jsonl(root / "quality.jsonl", {hash_manifest_url(urls[0]): 0.91})

            registry = build_clip_registry(
                root,
                manifests={"pretrain": "cache/manifests/pretrain.txt"},
                quality_metadata_path="quality.jsonl",
            )

        self.assertEqual([clip.url for clip in registry.clips], [urls[0], urls[2]])
        self.assertEqual(registry.split_counts(), {"pretrain": 2})
        self.assertEqual(registry.source_group_counts("pretrain"), {"worker00001": 1, "worker00002": 1})
        self.assertEqual(registry.by_sample_id[hash_manifest_url(urls[0])].quality["quality_score"], 0.91)
        self.assertTrue(registry.by_sample_id[hash_manifest_url(urls[0])].raw_path.name.endswith(".jsonl"))
        self.assertTrue(registry.by_sample_id[hash_manifest_url(urls[0])].feature_path.name.endswith(".npz"))

    def test_build_clip_registry_rejects_manifest_overlap_between_splits(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = _url("pretrain", "worker00001", 0)
            _write_manifest(root, "cache/manifests/pretrain.txt", [shared])
            _write_manifest(root, "cache/manifests/new.txt", [shared])
            _write_cached_clip(root, shared)

            with self.assertRaises(RegistryLeakageError):
                build_clip_registry(
                    root,
                    manifests={
                        "pretrain": "cache/manifests/pretrain.txt",
                        "new": "cache/manifests/new.txt",
                    },
                )

    def test_source_group_id_prefers_worker_path_component(self):
        self.assertEqual(
            source_group_id_from_url("https://storage.googleapis.com/bucket/pretrain/worker12345/clip000.jsonl"),
            "worker12345",
        )
        self.assertEqual(
            source_group_id_from_url("https://example.test/no-worker/clip000.jsonl"),
            "example.test:clip000.jsonl",
        )


def _url(split: str, worker: str, clip: int) -> str:
    return f"https://storage.googleapis.com/unit/{split}/{worker}/clip{clip:03d}.jsonl"


def _write_manifest(root: Path, relpath: str, urls: list[str]) -> None:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(urls) + "\n", encoding="utf-8")


def _write_cached_clip(root: Path, url: str, *, quality_score: float = 1.0) -> None:
    sample_id = hash_manifest_url(url)
    raw_dir = root / "cache" / "raw"
    feature_dir = root / "cache" / "features"
    raw_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{sample_id}.jsonl").write_text(json.dumps({"acc": [0, 0, 9.8], "gyro": [0, 0, 0]}) + "\n")
    np.savez(
        feature_dir / f"{sample_id}.npz",
        window_features=np.full((3, 4), quality_score, dtype=np.float32),
        clip_features=np.full(4, quality_score, dtype=np.float32),
    )


def _write_quality_jsonl(path: Path, scores: dict[str, float]) -> None:
    rows = [
        json.dumps({"sample_id": sample_id, "quality_score": score})
        for sample_id, score in sorted(scores.items())
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
