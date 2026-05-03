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
        self.assertIn("wait_full: bool = False", source)
        self.assertIn("remote_active_embedding_precompute.remote(config, smoke=True)", source)
        self.assertIn("remote_active_embedding_precompute.remote(config, smoke=False)", source)
        self.assertIn("remote_active_embedding_precompute.spawn(config, smoke=False)", source)
        self.assertIn("run_active_embedding_precompute(config, smoke=smoke, on_shard_written=artifacts_volume.commit)", source)
        self.assertIn('gpu="H100"', source)
        self.assertIn("timeout=3600", source)
        self.assertEqual(config["embeddings"]["cache_dir"], "/artifacts/active/embedding_cache/scale_pretrain")
        self.assertEqual(config["precompute"]["shard_size"], 512)
        self.assertEqual(config["precompute"]["workers"], 16)
        self.assertIn("raw_shape_stats", config["precompute"]["representations"])
        full_new_config = json.loads(Path("configs/active_embedding_precompute_ts2vec_full_new.json").read_text(encoding="utf-8"))
        self.assertEqual(full_new_config["precompute"]["clip_splits"], ["pretrain", "new"])
        self.assertIn("ts2vec", full_new_config["precompute"]["representations"])
        self.assertEqual(full_new_config["precompute"]["representation_options"]["ts2vec_device"], "cuda")
        new_only_config = json.loads(Path("configs/active_embedding_precompute_ts2vec_new_only_h100.json").read_text(encoding="utf-8"))
        self.assertEqual(new_only_config["precompute"]["clip_splits"], ["new"])
        self.assertEqual(new_only_config["precompute"]["max_clips_per_split"], 2500)
        self.assertTrue(new_only_config["precompute"]["fail_if_split_exceeds_max"])
        self.assertIn("ts2vec", new_only_config["precompute"]["representations"])

    def test_precompute_can_write_shards_for_configured_registry_splits_without_episodes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_split_fixture(root)
            config = {
                "execution": {
                    "provider": "modal",
                    "allow_local_paths_for_tests": True,
                    "smoke_max_clips_per_split": 2,
                },
                "data": {
                    "root": str(root),
                    "feature_glob": "cache/features/*.npz",
                    "raw_glob": "cache/raw/*.jsonl",
                    "manifests": {
                        "pretrain": "cache/manifests/pretrain_full_cached_urls.txt",
                        "new": "cache/manifests/new_urls.txt",
                    },
                },
                "embeddings": {"cache_dir": str(root / "embedding_cache")},
                "precompute": {
                    "clip_splits": ["pretrain", "new"],
                    "representations": ["window_mean_std_pool"],
                    "sample_rate": 30.0,
                    "shard_size": 3,
                },
            }

            result = run_active_embedding_precompute(config, smoke=True)
            manifest_exists = Path(result["manifest_path"]).exists()

        self.assertEqual(result["mode"], "smoke")
        self.assertEqual(result["selection_mode"], "splits")
        self.assertEqual(result["n_episodes"], 0)
        self.assertEqual(result["n_clips"], 4)
        self.assertTrue(manifest_exists)

    def test_precompute_split_cap_can_fail_closed_for_budget_guardrail(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_split_fixture(root)
            config = {
                "execution": {
                    "provider": "modal",
                    "allow_local_paths_for_tests": True,
                    "smoke_max_clips_per_split": 2,
                },
                "data": {
                    "root": str(root),
                    "feature_glob": "cache/features/*.npz",
                    "raw_glob": "cache/raw/*.jsonl",
                    "manifests": {
                        "new": "cache/manifests/new_urls.txt",
                    },
                },
                "embeddings": {"cache_dir": str(root / "embedding_cache")},
                "precompute": {
                    "clip_splits": ["new"],
                    "max_clips_per_split": 2,
                    "fail_if_split_exceeds_max": True,
                    "representations": ["window_mean_std_pool"],
                    "sample_rate": 30.0,
                    "shard_size": 3,
                },
            }

            with self.assertRaisesRegex(ValueError, "exceeding precompute.max_clips_per_split"):
                run_active_embedding_precompute(config, smoke=False)


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


def _write_split_fixture(root: Path) -> None:
    pretrain_urls = [_url(idx) for idx in range(3)]
    new_urls = [f"https://storage.googleapis.com/unit/new/worker00001/clip{idx:03d}.jsonl" for idx in range(3)]
    for url in [*pretrain_urls, *new_urls]:
        _write_cached_clip(root, url, center=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float))
    _write_manifest(root, "cache/manifests/pretrain_full_cached_urls.txt", pretrain_urls)
    _write_manifest(root, "cache/manifests/new_urls.txt", new_urls)


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
