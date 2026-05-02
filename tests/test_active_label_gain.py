import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.active.label_gain import compute_gated_labels, run_active_label_gain
from marginal_value.data.split_manifest import hash_manifest_url


class ActiveLabelGainTests(unittest.TestCase):
    def test_label_gain_rewards_candidate_that_covers_hidden_target(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _write_label_gain_fixture(root)
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
                    "quality_metadata": "quality.jsonl",
                    "manifests": {
                        "pretrain": "cache/manifests/pretrain_full_cached_urls.txt",
                    },
                },
                "episodes": {"path": "episodes.jsonl"},
                "embeddings": {"cache_dir": str(root / "embedding_cache")},
                "labels": {
                    "representations": ["window_mean_std_pool", "window_shape_stats"],
                    "raw_shape_max_samples": 90,
                },
                "artifacts": {"output_dir": str(root / "out")},
            }

            result = run_active_label_gain(config, smoke=True)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            label_rows = [json.loads(line) for line in Path(result["labels_path"]).read_text(encoding="utf-8").splitlines()]
            with Path(result["labels_csv_path"]).open(newline="") as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertEqual(result["mode"], "smoke")
        self.assertEqual(result["n_episodes"], 1)
        self.assertEqual(result["n_labels"], 3)
        self.assertEqual(len(csv_rows), 3)
        by_id = {row["sample_id"]: row for row in label_rows}
        good = by_id[fixture["good_candidate"]]
        redundant = by_id[fixture["redundant_candidate"]]
        distractor = by_id[fixture["distractor_candidate"]]

        self.assertGreater(good["balanced_gain"], redundant["balanced_gain"])
        self.assertGreater(good["balanced_relative_gain"], 0.5)
        self.assertAlmostEqual(redundant["balanced_gain"], 0.0, places=6)
        self.assertIn("gain_window_mean_std_pool", good)
        self.assertIn("relative_gain_window_shape_stats", good)
        self.assertEqual(good["greedy_prefix_rank"], 1)
        self.assertGreater(good["balanced_gain_after_greedy_prefix"], 0.0)
        self.assertEqual(distractor["candidate_role"], "low_quality")
        self.assertIn("new_cluster_similarity", good)
        self.assertTrue(good["hygiene_gate"])
        self.assertEqual(distractor["hygiene_gate"], False)
        self.assertAlmostEqual(distractor["gated_balanced_gain"], 0.0, places=6)
        self.assertEqual(report["label_summary"]["candidate_count"], 3)
        self.assertGreater(report["label_summary"]["balanced_gain"]["max"], 0.0)
        self.assertIn("gated_balanced_gain", report["label_summary"])
        self.assertEqual(report["registry_coverage_summary"]["registry_old_clip_count"], 5)
        self.assertEqual(report["embedding_cache"]["status"], "miss")
        self.assertTrue(report["embedding_cache"]["path"].endswith(".npz"))

    def test_compute_gated_labels_hard_zeros_dirty_candidates(self):
        rows = [
            {
                "sample_id": "clean",
                "balanced_gain": 1.0,
                "balanced_relative_gain": 0.5,
                "quality_score": 0.9,
                "stationary_fraction": 0.1,
                "max_abs_value": 12.0,
                "new_cluster_similarity": 0.2,
            },
            {
                "sample_id": "low_quality",
                "balanced_gain": 2.0,
                "balanced_relative_gain": 0.6,
                "quality_score": 0.2,
                "stationary_fraction": 0.1,
                "max_abs_value": 12.0,
                "new_cluster_similarity": 0.2,
            },
            {
                "sample_id": "stationary",
                "balanced_gain": 3.0,
                "balanced_relative_gain": 0.7,
                "quality_score": 0.9,
                "stationary_fraction": 0.95,
                "max_abs_value": 12.0,
                "new_cluster_similarity": 0.2,
            },
            {
                "sample_id": "abs_value",
                "balanced_gain": 4.0,
                "balanced_relative_gain": 0.8,
                "quality_score": 0.9,
                "stationary_fraction": 0.1,
                "max_abs_value": 120.0,
                "new_cluster_similarity": 0.2,
            },
            {
                "sample_id": "duplicate",
                "balanced_gain": 5.0,
                "balanced_relative_gain": 0.9,
                "quality_score": 0.9,
                "stationary_fraction": 0.1,
                "max_abs_value": 12.0,
                "new_cluster_similarity": 0.99,
            },
        ]

        gated = compute_gated_labels(rows)

        by_id = {row["sample_id"]: row for row in gated}
        self.assertTrue(by_id["clean"]["hygiene_gate"])
        self.assertEqual(by_id["clean"]["gated_balanced_gain"], 1.0)
        for sample_id in ("low_quality", "stationary", "abs_value", "duplicate"):
            self.assertFalse(by_id[sample_id]["hygiene_gate"])
            self.assertEqual(by_id[sample_id]["gated_balanced_gain"], 0.0)
            self.assertEqual(by_id[sample_id]["gated_balanced_relative_gain"], 0.0)

    def test_compute_gated_labels_requires_quality_columns(self):
        with self.assertRaisesRegex(ValueError, "stationary_fraction"):
            compute_gated_labels(
                [
                    {
                        "balanced_gain": 1.0,
                        "balanced_relative_gain": 0.5,
                        "quality_score": 0.9,
                        "max_abs_value": 12.0,
                    }
                ]
            )

    def test_compute_gated_labels_can_leave_duplicate_control_to_selector(self):
        rows = [
            {
                "sample_id": "duplicate_but_clean",
                "balanced_gain": 1.0,
                "balanced_relative_gain": 0.5,
                "quality_score": 0.9,
                "stationary_fraction": 0.1,
                "max_abs_value": 12.0,
                "new_cluster_similarity": 0.999,
            }
        ]

        gated = compute_gated_labels(rows, duplicate_cosine_threshold=None)

        self.assertTrue(gated[0]["hygiene_gate"])
        self.assertEqual(gated[0]["gated_balanced_gain"], 1.0)

    def test_modal_active_label_gain_entrypoint_uses_remote_gpu_job(self):
        source = Path("modal_active_label_gain.py").read_text(encoding="utf-8")
        config = json.loads(Path("configs/active_label_gain_smoke_full_pretrain.json").read_text(encoding="utf-8"))

        self.assertIn("marginal-value-active-label-gain", source)
        self.assertIn("remote_active_label_gain.remote(config, smoke=True)", source)
        self.assertIn("run_active_label_gain", source)
        self.assertIn('"torch==2.8.0"', source)
        self.assertIn('gpu="H100"', source)
        self.assertEqual(config["data"]["manifests"]["pretrain"], "cache/manifests/pretrain_full_cached_urls.txt")
        self.assertEqual(config["data"]["manifests"]["new"], "cache/manifests/new_urls.txt")
        self.assertEqual(
            config["data"]["quality_metadata"],
            "/artifacts/active/quality/full_pretrain_smoke/active_quality_smoke.jsonl",
        )
        self.assertEqual(config["embeddings"]["cache_dir"], "/artifacts/active/embedding_cache/full_pretrain_smoke")
        self.assertIn("raw_shape_stats", config["labels"]["representations"])
        scale_config = json.loads(Path("configs/active_label_gain_scale_pretrain.json").read_text(encoding="utf-8"))
        self.assertEqual(scale_config["embeddings"]["cache_dir"], "/artifacts/active/embedding_cache/scale_pretrain")
        self.assertEqual(scale_config["execution"]["smoke_max_episodes"], 2)


def _write_label_gain_fixture(root: Path) -> dict[str, str]:
    urls = [
        _url("worker00000", 0),
        _url("worker00000", 1),
        _url("worker00001", 0),
        _url("worker00001", 1),
        _url("worker00002", 0),
    ]
    centers = {
        urls[0]: np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
        urls[1]: np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
        urls[2]: np.asarray([0.0, 1.0, 0.0, 0.0], dtype=float),
        urls[3]: np.asarray([0.0, 1.0, 0.0, 0.0], dtype=float),
        urls[4]: np.asarray([-1.0, 0.0, 0.0, 0.0], dtype=float),
    }
    for url, center in centers.items():
        _write_cached_clip(root, url, center=center)
    _write_manifest(root, "cache/manifests/pretrain_full_cached_urls.txt", urls)
    quality_rows = []
    for url in urls:
        quality_score = 0.2 if url == urls[4] else 0.9
        quality_rows.append(
            json.dumps(
                {
                    "sample_id": hash_manifest_url(url),
                    "quality_score": quality_score,
                    "stationary_fraction": 0.1,
                    "max_abs_value": 12.0,
                }
            )
        )
    (root / "quality.jsonl").write_text("\n".join(quality_rows) + "\n", encoding="utf-8")
    episode = {
        "episode_id": "episode_00000",
        "seed": 17,
        "support_clip_ids": [hash_manifest_url(urls[0])],
        "candidate_clip_ids": [
            hash_manifest_url(urls[1]),
            hash_manifest_url(urls[2]),
            hash_manifest_url(urls[4]),
        ],
        "hidden_target_clip_ids": [hash_manifest_url(urls[3])],
        "distractor_clip_ids": [hash_manifest_url(urls[1]), hash_manifest_url(urls[4])],
        "heldout_source_groups": ["worker00001"],
        "known_source_groups": ["worker00000"],
        "candidate_roles": {
            hash_manifest_url(urls[1]): "known_like",
            hash_manifest_url(urls[2]): "heldout_novel",
            hash_manifest_url(urls[4]): "low_quality",
        },
        "low_quality_clip_ids": [hash_manifest_url(urls[4])],
    }
    (root / "episodes.jsonl").write_text(json.dumps(episode) + "\n", encoding="utf-8")
    return {
        "redundant_candidate": hash_manifest_url(urls[1]),
        "good_candidate": hash_manifest_url(urls[2]),
        "distractor_candidate": hash_manifest_url(urls[4]),
    }


def _url(worker: str, clip: int) -> str:
    return f"https://storage.googleapis.com/unit/pretrain/{worker}/clip{clip:03d}.jsonl"


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
    trend = np.linspace(0.0, 0.02, 5, dtype=np.float32)[:, None]
    windows = np.asarray(center, dtype=np.float32)[None, :] + trend
    np.savez(
        feature_dir / f"{sample_id}.npz",
        window_features=windows,
        clip_features=np.asarray(center, dtype=np.float32),
    )
    _write_raw_jsonl(raw_dir / f"{sample_id}.jsonl", center=center)


def _write_raw_jsonl(path: Path, *, center: np.ndarray, n_samples: int = 90, sample_rate: int = 30) -> None:
    axis = int(np.argmax(np.abs(center[:3]))) if np.any(center[:3]) else 0
    t = np.arange(n_samples, dtype=float) / sample_rate
    samples = np.zeros((n_samples, 6), dtype=float)
    samples[:, 2] = 9.81
    samples[:, axis] = np.sin(2.0 * np.pi * 0.7 * t) * np.sign(center[axis] if center[axis] else 1.0)
    samples[:, 3 + (axis % 3)] = 0.05 * np.cos(2.0 * np.pi * 0.4 * t)
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
