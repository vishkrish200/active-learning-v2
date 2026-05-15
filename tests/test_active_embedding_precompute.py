import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.active.embedding_cache import load_embedding_lookup
from marginal_value.active.embedding_precompute import run_active_embedding_precompute
from marginal_value.active.registry import build_clip_registry
from marginal_value.data.split_manifest import hash_manifest_url


class ActiveEmbeddingPrecomputeTests(unittest.TestCase):
    def test_precompute_writes_shards_for_episode_clip_set(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_ids = _write_fixture(root)
            config = {
                "execution": {
                    "provider": "modal",
                    "allow_local_paths_for_tests": True,
                    "smoke_max_episodes": 1,
                },
                "data": {
                    "root": str(root),
                    "feature_glob": "cache/features/*.npz",
                    "raw_glob": "cache/raw/*.jsonl",
                    "manifests": {"pretrain": "cache/manifests/pretrain_full_cached_urls.txt"},
                },
                "episodes": {"path": "episodes.jsonl"},
                "embeddings": {"cache_dir": str(root / "embedding_cache")},
                "precompute": {
                    "representations": ["window_mean_std_pool", "raw_shape_stats"],
                    "sample_rate": 30.0,
                    "raw_shape_max_samples": 90,
                    "shard_size": 2,
                },
            }

            result = run_active_embedding_precompute(config, smoke=True)
            self.assertTrue(Path(result["manifest_path"]).exists())
            registry = build_clip_registry(root, manifests=config["data"]["manifests"])
            clips = [registry.by_sample_id[sample_id] for sample_id in sample_ids]
            for clip in clips:
                clip.raw_path.unlink()
                clip.feature_path.unlink()
            loaded = load_embedding_lookup(
                clips,
                representations=["window_mean_std_pool", "raw_shape_stats"],
                sample_rate=30.0,
                raw_shape_max_samples=90,
                cache_dir=root / "embedding_cache",
                component="test_embedding_precompute",
                mode="smoke",
            )

        self.assertEqual(result["mode"], "smoke")
        self.assertEqual(result["n_episodes"], 1)
        self.assertEqual(result["n_clips"], len(sample_ids))
        self.assertEqual(result["n_shards"], 2)
        self.assertEqual(loaded.cache_status, "shard_hit")

    def test_modal_active_embedding_precompute_entrypoint(self):
        source = Path("modal_active_embedding_precompute.py").read_text(encoding="utf-8")
        config = json.loads(Path("configs/active_embedding_precompute_scale_pretrain.json").read_text(encoding="utf-8"))

        self.assertIn("marginal-value-active-embedding-precompute", source)
        self.assertIn("remote_active_embedding_precompute.remote(config, smoke=True)", source)
        self.assertIn("run_active_embedding_precompute(config, smoke=smoke, on_shard_written=artifacts_volume.commit)", source)
        self.assertEqual(config["embeddings"]["cache_dir"], "/artifacts/active/embedding_cache/scale_pretrain")
        self.assertEqual(config["precompute"]["shard_size"], 512)
        self.assertEqual(config["precompute"]["workers"], 16)
        self.assertIn("raw_shape_stats", config["precompute"]["representations"])


def _write_fixture(root: Path) -> list[str]:
    urls = [_url(idx) for idx in range(4)]
    for idx, url in enumerate(urls):
        _write_cached_clip(root, url, center=np.asarray([float(idx + 1), 0.0, 0.0, 0.0], dtype=float))
    _write_manifest(root, "cache/manifests/pretrain_full_cached_urls.txt", urls)
    episode = {
        "episode_id": "episode_00000",
        "support_clip_ids": [hash_manifest_url(urls[0])],
        "candidate_clip_ids": [hash_manifest_url(urls[1]), hash_manifest_url(urls[2])],
        "hidden_target_clip_ids": [hash_manifest_url(urls[3])],
        "candidate_roles": {
            hash_manifest_url(urls[1]): "known_like",
            hash_manifest_url(urls[2]): "heldout_novel",
        },
    }
    (root / "episodes.jsonl").write_text(json.dumps(episode) + "\n", encoding="utf-8")
    return sorted(hash_manifest_url(url) for url in urls)


def _url(clip: int) -> str:
    return f"https://storage.googleapis.com/unit/pretrain/worker00000/clip{clip:03d}.jsonl"


def _write_manifest(root: Path, relpath: str, urls: list[str]) -> None:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(urls) + "\n", encoding="utf-8")


def _write_cached_clip(root: Path, url: str, *, center: np.ndarray) -> None:
    sample_id = hash_manifest_url(url)
    raw_dir = root / "cache" / "raw"
    feature_dir = root / "cache" / "features"
    raw_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    np.savez(feature_dir / f"{sample_id}.npz", window_features=np.asarray(center, dtype=np.float32)[None, :])
    rows = []
    for idx in range(90):
        rows.append(json.dumps({"t_us": idx * 33333, "acc": [1.0, 0.0, 9.81], "gyro": [0.0, 0.0, 0.0]}))
    (raw_dir / f"{sample_id}.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
