import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.active.final_blend_rank import (
    _right_support_clips_for_budgeted_rank,
    run_active_final_blend_rank,
    validate_active_final_blend_rank_config,
)
from marginal_value.active.exact_window_blend_rank import (
    run_active_exact_window_blend_rank,
    validate_active_exact_window_blend_rank_config,
)
from marginal_value.active.registry import ClipRecord
from marginal_value.data.build_full_support_shards import run_build_full_support_shards
from marginal_value.data.split_manifest import hash_manifest_url


class ActiveFinalBlendRankTests(unittest.TestCase):
    def test_final_blend_rank_writes_submission_diagnostics_and_finalized_ids(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            urls = _write_fixture(root)
            config = {
                "execution": {
                    "provider": "modal",
                    "allow_local_paths_for_tests": True,
                    "smoke_support_samples": 3,
                    "smoke_query_samples": 3,
                },
                "data": {
                    "root": str(root),
                    "feature_glob": "cache/features/*.npz",
                    "raw_glob": "cache/raw/*.jsonl",
                    "quality_metadata": "quality.jsonl",
                    "manifests": {
                        "pretrain": "cache/manifests/pretrain_full_cached_urls.txt",
                        "new": "cache/manifests/new_urls.txt",
                    },
                },
                "embeddings": {"cache_dir": str(root / "embedding_cache")},
                "ranking": {
                    "support_split": "pretrain",
                    "query_split": "new",
                    "left_representation": "window_mean_std_pool",
                    "right_representation": "temporal_order",
                    "alpha": 0.5,
                    "old_novelty_k": 2,
                    "quality_threshold": 0.85,
                    "max_stationary_fraction": 0.90,
                    "max_abs_value": 60.0,
                    "cluster_similarity_threshold": 0.995,
                    "top_k_values": [2, 3],
                },
                "artifacts": {"output_dir": str(root / "out")},
            }

            result = run_active_final_blend_rank(config, smoke=True)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            submission = _read_csv(Path(result["submission_path"]))
            diagnostics = _read_csv(Path(result["diagnostics_path"]))
            finalized = _read_csv(Path(result["finalized_worker_id_path"]))

        self.assertEqual(result["mode"], "smoke")
        self.assertEqual(report["selector"], "blend_kcenter_window_mean_std_pool_temporal_order_a05")
        self.assertEqual(report["n_support"], 3)
        self.assertEqual(report["n_query"], 3)
        self.assertEqual(report["topk_quality"]["k2"]["quality_failure_rate"], 0.0)
        self.assertAlmostEqual(report["topk_quality"]["k3"]["quality_failure_rate"], 1.0 / 3.0)
        self.assertEqual(len(submission), 3)
        self.assertEqual([int(row["rank"]) for row in submission], [1, 2, 3])
        self.assertEqual(submission[-1]["worker_id"], hash_manifest_url(urls["low_quality_new"]))
        self.assertEqual(diagnostics[-1]["quality_gate_pass"], "False")
        self.assertIn(finalized[0]["worker_id"], {"clip-new-good-a", "clip-new-good-b"})
        self.assertEqual(finalized[-1]["worker_id"], "clip-new-bad")

    def test_budgeted_candidate_only_rank_uses_partial_left_support_cache(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            urls = _write_fixture(root)
            left_shard_dir = root / "partial_left_shards"
            _write_partial_embedding_shard(
                left_shard_dir,
                sample_ids=[hash_manifest_url(url) for url in urls["pretrain_urls"][:2]],
                representation="window_mean_std_pool",
                values=np.asarray(
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1],
                        [0.0, 1.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1],
                    ],
                    dtype="float32",
                ),
            )
            config = {
                "execution": {
                    "provider": "modal",
                    "allow_local_paths_for_tests": True,
                    "smoke_support_samples": 3,
                    "smoke_query_samples": 3,
                },
                "data": {
                    "root": str(root),
                    "feature_glob": "cache/features/*.npz",
                    "raw_glob": "cache/raw/*.jsonl",
                    "quality_metadata": "quality.jsonl",
                    "manifests": {
                        "pretrain": "cache/manifests/pretrain_full_cached_urls.txt",
                        "new": "cache/manifests/new_urls.txt",
                    },
                },
                "embeddings": {"cache_dir": str(root / "query_cache")},
                "ranking": {
                    "budgeted_candidate_only": True,
                    "support_split": "pretrain",
                    "query_split": "new",
                    "left_representation": "window_mean_std_pool",
                    "right_representation": "temporal_order",
                    "left_support_shard_dir": str(left_shard_dir),
                    "min_left_support_clips": 2,
                    "candidate_cache_dir": str(root / "query_cache"),
                    "right_support_cache_dir": str(root / "right_support_cache"),
                    "right_embedding_workers": 2,
                    "max_query_clips": 5,
                    "alpha": 0.5,
                    "old_novelty_k": 1,
                    "quality_threshold": 0.85,
                    "max_stationary_fraction": 0.90,
                    "max_abs_value": 60.0,
                    "cluster_similarity_threshold": 0.995,
                    "top_k_values": [2, 3],
                },
                "artifacts": {"output_dir": str(root / "out")},
            }

            result = run_active_final_blend_rank(config, smoke=True)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            submission = _read_csv(Path(result["submission_path"]))

        self.assertEqual(result["ranking_mode"], "budgeted_candidate_only")
        self.assertEqual(report["ranking_mode"], "budgeted_candidate_only")
        self.assertEqual(report["left_support_cache"]["status"], "partial_shard_hit")
        self.assertEqual(report["n_left_support"], 2)
        self.assertEqual(report["n_query"], 3)
        self.assertEqual(len(submission), 3)

    def test_budgeted_candidate_only_rank_can_seed_right_support_subset(self):
        clips = [
            ClipRecord(
                sample_id=f"clip-{idx}",
                split="pretrain",
                url=f"https://storage.googleapis.com/unit/pretrain/worker{idx:05d}/clip.jsonl",
                source_group_id=f"worker{idx:05d}",
                worker_id=f"worker{idx:05d}",
                raw_path=Path(f"/tmp/clip-{idx}.jsonl"),
                feature_path=Path(f"/tmp/clip-{idx}.npz"),
                quality={},
            )
            for idx in range(6)
        ]
        config = {
            "execution": {"smoke_support_samples": 2},
            "ranking": {
                "right_support_max_clips": 3,
                "right_support_seed": 11,
            },
        }

        seeded = _right_support_clips_for_budgeted_rank(clips, config=config, smoke=False)
        unseeded = _right_support_clips_for_budgeted_rank(
            clips,
            config={"execution": {"smoke_support_samples": 2}, "ranking": {"right_support_max_clips": 3}},
            smoke=False,
        )
        smoke = _right_support_clips_for_budgeted_rank(clips, config=config, smoke=True)

        self.assertEqual([clip.sample_id for clip in seeded], ["clip-0", "clip-4", "clip-5"])
        self.assertEqual([clip.sample_id for clip in unseeded], ["clip-0", "clip-1", "clip-2"])
        self.assertEqual([clip.sample_id for clip in smoke], ["clip-0", "clip-1"])

    def test_final_blend_rank_config_validates_required_representations(self):
        config = {
            "execution": {"provider": "modal", "allow_local_paths_for_tests": True},
            "data": {
                "root": "/tmp/unit",
                "manifests": {
                    "pretrain": "cache/manifests/pretrain_full_cached_urls.txt",
                    "new": "cache/manifests/new_urls.txt",
                },
            },
            "ranking": {
                "left_representation": "ts2vec",
                "right_representation": "window_mean_std_pool",
            },
            "artifacts": {"output_dir": "/tmp/unit/out"},
        }

        validate_active_final_blend_rank_config(config)

    def test_modal_final_blend_entrypoint_and_production_config(self):
        source = Path("modal_active_final_blend_rank.py").read_text(encoding="utf-8")
        config = json.loads(Path("configs/active_final_blend_rank_new.json").read_text(encoding="utf-8"))
        budget_config = json.loads(Path("configs/active_final_blend_rank_budget_h100.json").read_text(encoding="utf-8"))

        self.assertIn("marginal-value-active-final-blend-rank", source)
        self.assertIn('gpu="H100"', source)
        self.assertIn("timeout=3600", source)
        self.assertIn("run_active_final_blend_rank(config, smoke=smoke)", source)
        self.assertEqual(config["embeddings"]["cache_dir"], "/artifacts/active/embedding_cache/ts2vec_window_full_new")
        self.assertEqual(config["ranking"]["left_representation"], "ts2vec")
        self.assertEqual(config["ranking"]["right_representation"], "window_mean_std_pool")
        self.assertEqual(config["ranking"]["alpha"], 0.5)
        self.assertEqual(config["ranking"]["quality_threshold"], 0.85)
        self.assertEqual(config["ranking"]["max_stationary_fraction"], 0.90)
        self.assertEqual(config["ranking"]["max_abs_value"], 60.0)
        validate_active_final_blend_rank_config(config)
        self.assertTrue(budget_config["ranking"]["budgeted_candidate_only"])
        self.assertEqual(budget_config["ranking"]["max_query_clips"], 2500)
        self.assertEqual(budget_config["ranking"]["min_left_support_clips"], 20000)
        self.assertEqual(budget_config["ranking"]["right_support_max_clips"], 25000)
        self.assertEqual(budget_config["ranking"]["right_embedding_workers"], 16)
        validate_active_final_blend_rank_config(budget_config)

    def test_exact_window_blend_rank_uses_window_shards_for_full_right_support(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            urls = _write_fixture(root)
            shard_result = run_build_full_support_shards(_window_shard_config(root), smoke=True)
            left_shard_dir = root / "partial_left_shards"
            _write_partial_embedding_shard(
                left_shard_dir,
                sample_ids=[hash_manifest_url(url) for url in urls["pretrain_urls"][:2]],
                representation="window_mean_std_pool",
                values=np.asarray(
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1],
                        [0.0, 1.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1],
                    ],
                    dtype="float32",
                ),
            )
            left_query_shard_dir = root / "partial_left_query_shards"
            _write_partial_embedding_shard(
                left_query_shard_dir,
                sample_ids=[hash_manifest_url(url) for url in urls["new_urls"]],
                representation="window_mean_std_pool",
                values=np.asarray(
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2],
                        [0.0, 1.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2],
                        [0.0, 0.0, 1.0, 0.0, 0.2, 0.2, 0.2, 0.2],
                    ],
                    dtype="float32",
                ),
            )
            config = {
                "execution": {
                    "provider": "modal",
                    "allow_local_paths_for_tests": True,
                    "smoke_query_samples": 3,
                    "smoke_window_support_samples": 3,
                },
                "data": {
                    "root": str(root),
                    "feature_glob": "cache/features/*.npz",
                    "raw_glob": "cache/raw/*.jsonl",
                    "quality_metadata": "quality.jsonl",
                    "manifests": {
                        "pretrain": "cache/manifests/pretrain_full_cached_urls.txt",
                        "new": "cache/manifests/new_urls.txt",
                    },
                },
                "embeddings": {"cache_dir": str(root / "query_cache")},
                "ranking": {
                    "support_split": "pretrain",
                    "query_split": "new",
                    "left_representation": "window_mean_std_pool",
                    "right_representation": "window_mean_std_pool",
                    "left_support_shard_dir": str(left_shard_dir),
                    "left_query_shard_dir": str(left_query_shard_dir),
                    "right_support_shard_manifest": shard_result["manifest_path"],
                    "candidate_cache_dir": str(root / "query_cache"),
                    "min_left_support_clips": 2,
                    "alpha": 0.5,
                    "old_novelty_k": 1,
                    "quality_threshold": 0.85,
                    "max_stationary_fraction": 0.90,
                    "max_abs_value": 60.0,
                    "cluster_similarity_threshold": 0.995,
                    "top_k_values": [2, 3],
                },
                "artifacts": {"output_dir": str(root / "exact_out")},
            }

            result = run_active_exact_window_blend_rank(config, smoke=True)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            diagnostics = _read_csv(Path(result["diagnostics_path"]))

        self.assertEqual(result["mode"], "smoke")
        self.assertEqual(result["ranking_mode"], "partial_left_exact_window_right")
        self.assertEqual(report["right_support_cache"]["status"], "full_support_shard_hit")
        self.assertEqual(report["n_right_support"], 3)
        self.assertEqual(report["n_query"], 3)
        self.assertEqual(report["topk_quality"]["k2"]["quality_failure_rate"], 0.0)
        self.assertEqual(len(diagnostics), 3)

    def test_modal_exact_window_blend_rank_entrypoint_and_config_validate(self):
        source = Path("modal_active_exact_window_blend_rank.py").read_text(encoding="utf-8")
        config = json.loads(Path("configs/active_exact_window_blend_rank.json").read_text(encoding="utf-8"))

        self.assertIn("marginal-value-active-exact-window-blend-rank", source)
        self.assertIn("run_active_exact_window_blend_rank(config, smoke=smoke)", source)
        self.assertIn("remote_active_exact_window_blend_rank.remote(config, smoke=True)", source)
        self.assertIn("remote_active_exact_window_blend_rank.spawn(config, smoke=False)", source)
        self.assertNotIn("gpu=", source)
        self.assertEqual(config["ranking"]["right_support_shard_dir"], "/artifacts/active/full_support_shards/window_mean_std_v1")
        validate_active_exact_window_blend_rank_config(config)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_fixture(root: Path) -> dict[str, str]:
    pretrain_urls = [
        "https://storage.googleapis.com/unit/pretrain/worker00000/clip-old-a.jsonl",
        "https://storage.googleapis.com/unit/pretrain/worker00001/clip-old-b.jsonl",
        "https://storage.googleapis.com/unit/pretrain/worker00002/clip-old-c.jsonl",
    ]
    new_urls = [
        "https://storage.googleapis.com/unit/new/worker10000/clip-new-good-a.jsonl",
        "https://storage.googleapis.com/unit/new/worker10001/clip-new-good-b.jsonl",
        "https://storage.googleapis.com/unit/new/worker10002/clip-new-bad.jsonl",
    ]
    for idx, url in enumerate(pretrain_urls):
        _write_cached_clip(root, url, center=np.asarray([float(idx), 0.0, 0.0, 0.0], dtype=float))
    _write_cached_clip(root, new_urls[0], center=np.asarray([3.0, 0.0, 0.0, 0.0], dtype=float))
    _write_cached_clip(root, new_urls[1], center=np.asarray([0.0, 3.0, 0.0, 0.0], dtype=float))
    _write_cached_clip(root, new_urls[2], center=np.asarray([9.0, 9.0, 0.0, 0.0], dtype=float))
    _write_manifest(root, "cache/manifests/pretrain_full_cached_urls.txt", pretrain_urls)
    _write_manifest(root, "cache/manifests/new_urls.txt", new_urls)
    quality_rows = [
        {"sample_id": hash_manifest_url(new_urls[0]), "quality_score": 0.95, "stationary_fraction": 0.10, "max_abs_value": 12.0},
        {"sample_id": hash_manifest_url(new_urls[1]), "quality_score": 0.93, "stationary_fraction": 0.20, "max_abs_value": 13.0},
        {"sample_id": hash_manifest_url(new_urls[2]), "quality_score": 0.20, "stationary_fraction": 0.99, "max_abs_value": 12.0},
    ]
    (root / "quality.jsonl").write_text("\n".join(json.dumps(row) for row in quality_rows) + "\n", encoding="utf-8")
    return {"low_quality_new": new_urls[2], "pretrain_urls": pretrain_urls, "new_urls": new_urls}


def _write_manifest(root: Path, relpath: str, urls: list[str]) -> None:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(urls) + "\n", encoding="utf-8")


def _window_shard_config(root: Path) -> dict[str, object]:
    return {
        "execution": {
            "provider": "modal",
            "allow_local_paths_for_tests": True,
            "smoke_max_clips_per_split": 3,
        },
        "data": {
            "root": str(root),
            "feature_glob": "cache/features/*.npz",
            "raw_glob": "cache/raw/*.jsonl",
            "quality_metadata": "quality.jsonl",
            "manifests": {
                "pretrain": "cache/manifests/pretrain_full_cached_urls.txt",
                "new": "cache/manifests/new_urls.txt",
            },
        },
        "shards": {
            "output_dir": str(root / "window_shards"),
            "clip_splits": ["pretrain", "new"],
            "representations": ["window_mean_std_pool"],
            "shard_size": 2,
            "workers": 1,
            "include_imu_samples": False,
            "progress_every_shards": 1,
        },
    }


def _write_cached_clip(root: Path, url: str, *, center: np.ndarray) -> None:
    sample_id = hash_manifest_url(url)
    raw_dir = root / "cache" / "raw"
    feature_dir = root / "cache" / "features"
    raw_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    window_features = np.vstack([center, center + 0.1, center + 0.2]).astype("float32")
    np.savez(feature_dir / f"{sample_id}.npz", window_features=window_features)
    rows = []
    for idx in range(90):
        rows.append(json.dumps({"t_us": idx * 33333, "acc": [1.0, 0.0, 9.81], "gyro": [0.0, 0.0, 0.0]}))
    (raw_dir / f"{sample_id}.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_partial_embedding_shard(
    shard_dir: Path,
    *,
    sample_ids: list[str],
    representation: str,
    values: np.ndarray,
) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        shard_dir / "shard_00000.npz",
        sample_ids=np.asarray(sample_ids, dtype=str),
        **{f"rep__{representation}": np.asarray(values, dtype="float32")},
    )


if __name__ == "__main__":
    unittest.main()
