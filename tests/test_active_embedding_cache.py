import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from marginal_value.active.embedding_cache import load_embedding_lookup, write_embedding_shard_cache
from marginal_value.active.registry import ClipRecord


class ActiveEmbeddingCacheTests(unittest.TestCase):
    def test_embedding_cache_reuses_pack_without_source_files(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "embedding_cache"
            clip_a = _write_clip(root, "clip-a", center=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float))
            clip_b = _write_clip(root, "clip-b", center=np.asarray([0.0, 1.0, 0.0, 0.0], dtype=float))

            first = load_embedding_lookup(
                [clip_a, clip_b],
                representations=["window_mean_std_pool", "raw_shape_stats"],
                sample_rate=30.0,
                raw_shape_max_samples=90,
                cache_dir=cache_dir,
                component="test_embedding_cache",
                mode="smoke",
            )
            self.assertEqual(first.cache_status, "miss")
            self.assertTrue(first.cache_path.exists())
            self.assertIn("clip-a", first.embeddings["window_mean_std_pool"])

            clip_a.raw_path.unlink()
            clip_a.feature_path.unlink()
            clip_b.raw_path.unlink()
            clip_b.feature_path.unlink()

            second = load_embedding_lookup(
                [clip_b, clip_a],
                representations=["window_mean_std_pool", "raw_shape_stats"],
                sample_rate=30.0,
                raw_shape_max_samples=90,
                cache_dir=cache_dir,
                component="test_embedding_cache",
                mode="smoke",
            )

        self.assertEqual(second.cache_status, "hit")
        self.assertEqual(second.cache_path, first.cache_path)
        np.testing.assert_allclose(
            second.embeddings["raw_shape_stats"]["clip-a"],
            first.embeddings["raw_shape_stats"]["clip-a"],
        )
        np.testing.assert_allclose(
            second.embeddings["window_mean_std_pool"]["clip-b"],
            first.embeddings["window_mean_std_pool"]["clip-b"],
        )

    def test_embedding_cache_loads_precomputed_shards_without_source_files(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "embedding_cache"
            clips = [
                _write_clip(root, "clip-a", center=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)),
                _write_clip(root, "clip-b", center=np.asarray([0.0, 1.0, 0.0, 0.0], dtype=float)),
                _write_clip(root, "clip-c", center=np.asarray([0.0, 0.0, 1.0, 0.0], dtype=float)),
            ]

            result = write_embedding_shard_cache(
                clips,
                representations=["window_mean_std_pool", "raw_shape_stats"],
                sample_rate=30.0,
                raw_shape_max_samples=90,
                cache_dir=cache_dir,
                shard_size=1,
                component="test_embedding_cache",
                mode="smoke",
            )
            self.assertEqual(result["n_shards"], 3)
            self.assertTrue(Path(result["manifest_path"]).exists())
            for clip in clips:
                clip.raw_path.unlink()
                clip.feature_path.unlink()

            loaded = load_embedding_lookup(
                list(reversed(clips)),
                representations=["window_mean_std_pool", "raw_shape_stats"],
                sample_rate=30.0,
                raw_shape_max_samples=90,
                cache_dir=cache_dir,
                component="test_embedding_cache",
                mode="smoke",
            )

        self.assertEqual(loaded.cache_status, "shard_hit")
        self.assertTrue(str(loaded.cache_path).endswith(".shards.json"))
        self.assertEqual(set(loaded.embeddings["raw_shape_stats"]), {"clip-a", "clip-b", "clip-c"})

    def test_shard_cache_invokes_callback_for_new_shards_only(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "embedding_cache"
            clips = [
                _write_clip(root, "clip-a", center=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)),
                _write_clip(root, "clip-b", center=np.asarray([0.0, 1.0, 0.0, 0.0], dtype=float)),
                _write_clip(root, "clip-c", center=np.asarray([0.0, 0.0, 1.0, 0.0], dtype=float)),
            ]
            callback_calls: list[str] = []

            write_embedding_shard_cache(
                clips,
                representations=["window_mean_std_pool"],
                sample_rate=30.0,
                raw_shape_max_samples=90,
                cache_dir=cache_dir,
                shard_size=2,
                component="test_embedding_cache",
                mode="smoke",
                on_shard_written=lambda: callback_calls.append("called"),
            )
            self.assertEqual(callback_calls, ["called", "called"])

            write_embedding_shard_cache(
                clips,
                representations=["window_mean_std_pool"],
                sample_rate=30.0,
                raw_shape_max_samples=90,
                cache_dir=cache_dir,
                shard_size=2,
                component="test_embedding_cache",
                mode="smoke",
                on_shard_written=lambda: callback_calls.append("called_again"),
            )

        self.assertEqual(callback_calls, ["called", "called"])

    def test_shard_cache_parallel_workers_preserve_embedding_rows(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            clips = [
                _write_clip(root, "clip-a", center=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)),
                _write_clip(root, "clip-b", center=np.asarray([0.0, 1.0, 0.0, 0.0], dtype=float)),
                _write_clip(root, "clip-c", center=np.asarray([0.0, 0.0, 1.0, 0.0], dtype=float)),
            ]
            serial_cache_dir = root / "serial_embedding_cache"
            parallel_cache_dir = root / "parallel_embedding_cache"

            write_embedding_shard_cache(
                clips,
                representations=["window_mean_std_pool", "raw_shape_stats"],
                sample_rate=30.0,
                raw_shape_max_samples=90,
                cache_dir=serial_cache_dir,
                shard_size=2,
                component="test_embedding_cache",
                mode="smoke",
                workers=1,
            )
            write_embedding_shard_cache(
                list(reversed(clips)),
                representations=["window_mean_std_pool", "raw_shape_stats"],
                sample_rate=30.0,
                raw_shape_max_samples=90,
                cache_dir=parallel_cache_dir,
                shard_size=2,
                component="test_embedding_cache",
                mode="smoke",
                workers=2,
            )
            serial = load_embedding_lookup(
                clips,
                representations=["window_mean_std_pool", "raw_shape_stats"],
                sample_rate=30.0,
                raw_shape_max_samples=90,
                cache_dir=serial_cache_dir,
                component="test_embedding_cache",
                mode="smoke",
            )
            parallel = load_embedding_lookup(
                clips,
                representations=["window_mean_std_pool", "raw_shape_stats"],
                sample_rate=30.0,
                raw_shape_max_samples=90,
                cache_dir=parallel_cache_dir,
                component="test_embedding_cache",
                mode="smoke",
            )

        self.assertEqual(serial.cache_status, "shard_hit")
        self.assertEqual(parallel.cache_status, "shard_hit")
        for representation in ("window_mean_std_pool", "raw_shape_stats"):
            for clip in clips:
                np.testing.assert_allclose(
                    parallel.embeddings[representation][clip.sample_id],
                    serial.embeddings[representation][clip.sample_id],
                )

    def test_ts2vec_uses_full_clip_when_raw_shape_is_sample_limited(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = _write_clip(root, "clip-a", center=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float))
            seen_lengths: list[int] = []

            class FakeInference:
                def __init__(self, checkpoint_path, device="cpu"):
                    self.checkpoint_path = checkpoint_path
                    self.device = device

                def encode_clip(self, values):
                    seen_lengths.append(len(values))
                    return np.asarray([float(len(values))], dtype="float32")

            with patch("marginal_value.active.embedding_cache.TS2VecInference", FakeInference):
                result = load_embedding_lookup(
                    [clip],
                    representations=["raw_shape_stats", "ts2vec"],
                    sample_rate=30.0,
                    raw_shape_max_samples=10,
                    cache_dir=None,
                    component="test_embedding_cache",
                    mode="smoke",
                    representation_options={"ts2vec_checkpoint_path": "/tmp/ts2vec.pt"},
                )

        self.assertEqual(seen_lengths, [90])
        np.testing.assert_allclose(result.embeddings["ts2vec"]["clip-a"], np.asarray([90.0], dtype="float32"))


def _write_clip(root: Path, sample_id: str, *, center: np.ndarray) -> ClipRecord:
    raw_dir = root / "cache" / "raw"
    feature_dir = root / "cache" / "features"
    raw_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{sample_id}.jsonl"
    feature_path = feature_dir / f"{sample_id}.npz"
    trend = np.linspace(0.0, 0.02, 5, dtype=np.float32)[:, None]
    windows = np.asarray(center, dtype=np.float32)[None, :] + trend
    np.savez(feature_path, window_features=windows)
    _write_raw_jsonl(raw_path, center=center)
    return ClipRecord(
        sample_id=sample_id,
        split="pretrain",
        url=f"https://storage.googleapis.com/unit/pretrain/worker00000/{sample_id}.jsonl",
        source_group_id="worker00000",
        worker_id="worker00000",
        raw_path=raw_path,
        feature_path=feature_path,
    )


def _write_raw_jsonl(path: Path, *, center: np.ndarray, n_samples: int = 90, sample_rate: int = 30) -> None:
    axis = int(np.argmax(np.abs(center[:3]))) if np.any(center[:3]) else 0
    t = np.arange(n_samples, dtype=float) / sample_rate
    samples = np.zeros((n_samples, 6), dtype=float)
    samples[:, 2] = 9.81
    samples[:, axis] = np.sin(2.0 * np.pi * 0.7 * t)
    samples[:, 3 + (axis % 3)] = 0.05 * np.cos(2.0 * np.pi * 0.4 * t)
    lines = []
    for idx, row in enumerate(samples):
        lines.append(
            (
                '{"t_us": %d, "acc": [%f, %f, %f], "gyro": [%f, %f, %f]}'
                % (
                    int(idx * 1_000_000 / sample_rate),
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                )
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
