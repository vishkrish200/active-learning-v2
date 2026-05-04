from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from marginal_value.data.split_manifest import hash_manifest_url
from marginal_value.gcp.ts2vec_embed_manifest import (
    _load_jsonl_imu_url,
    _sample_id_from_url,
    _write_manifest,
    _write_shard,
)


class GcpTS2VecEmbedManifestTests(unittest.TestCase):
    def test_sample_id_from_url(self) -> None:
        url = "https://storage.googleapis.com/bucket/path/abc123.jsonl"
        self.assertEqual(_sample_id_from_url(url), hash_manifest_url(url))

    def test_sample_id_from_url_includes_worker_path(self) -> None:
        left = "https://storage.googleapis.com/bucket/pretrain/worker00001/clip001.txt"
        right = "https://storage.googleapis.com/bucket/pretrain/worker00002/clip001.txt"
        self.assertNotEqual(_sample_id_from_url(left), _sample_id_from_url(right))

    def test_load_jsonl_imu_url_from_local_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"t_us": 1, "acc": [1, 2, 3], "gyro": [4, 5, 6]}),
                        json.dumps({"t_us": 2, "acc": [2, 3, 4], "gyro": [5, 6, 7]}),
                    ]
                ),
                encoding="utf-8",
            )
            samples = _load_jsonl_imu_url(str(path), max_samples=None)
        self.assertEqual(samples.shape, (2, 6))
        np.testing.assert_allclose(samples[0], np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float32))

    def test_write_shard_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_path = root / "shards" / "shard_00000.npz"
            matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            _write_shard(shard_path, ["a", "b"], matrix)
            with np.load(shard_path, allow_pickle=False) as data:
                self.assertEqual(data["sample_ids"].tolist(), ["a", "b"])
                np.testing.assert_allclose(data["rep__ts2vec"], matrix)

            manifest_path = root / "manifest.json"
            _write_manifest(
                manifest_path,
                sample_ids=["a", "b"],
                shards=[{"index": 0, "path": "shards/shard_00000.npz", "sample_ids": ["a", "b"], "n_clips": 2}],
                args={
                    "checkpoint": "/tmp/ts2vec_best.pt",
                    "manifest": "manifest.txt",
                    "device": "cuda",
                    "shard_size": 2,
                    "encode_batch_size": 2,
                    "load_workers": 1,
                    "max_samples": None,
                },
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["metadata"]["representations"], ["ts2vec"])
        self.assertEqual(payload["n_clips"], 2)
        self.assertEqual(payload["shards"][0]["path"], "shards/shard_00000.npz")


if __name__ == "__main__":
    unittest.main()
