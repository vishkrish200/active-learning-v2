import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.active.support_sampling_stability import (
    run_support_sampling_stability,
    summarize_rank_stability,
    validate_support_sampling_stability_config,
)
from marginal_value.data.split_manifest import hash_manifest_url


class ActiveSupportSamplingStabilityTests(unittest.TestCase):
    def test_summarize_rank_stability_reports_overlap_rank_agreement_and_hygiene(self):
        run_a = {
            "label": "seed_1",
            "seed": 1,
            "diagnostics_rows": [
                _rank_row("a", 1, 1.0, cluster=0),
                _rank_row("b", 2, 0.9, cluster=1),
                _rank_row("c", 3, 0.8, cluster=2, quality_gate_pass=False),
                _rank_row("d", 4, 0.7, cluster=3),
            ],
        }
        run_b = {
            "label": "seed_2",
            "seed": 2,
            "diagnostics_rows": [
                _rank_row("a", 1, 0.95, cluster=0),
                _rank_row("c", 2, 0.85, cluster=2),
                _rank_row("b", 3, 0.75, cluster=1),
                _rank_row("d", 4, 0.65, cluster=1, physical_validity_pass=False),
            ],
        }

        summary = summarize_rank_stability([run_a, run_b], k_values=[2, 3, 4])

        self.assertEqual(summary["n_runs"], 2)
        self.assertAlmostEqual(summary["pairwise"]["top2_overlap_mean"], 0.5)
        self.assertAlmostEqual(summary["pairwise"]["top3_overlap_mean"], 1.0)
        self.assertLess(summary["pairwise"]["rank_spearman_mean"], 1.0)
        self.assertGreater(summary["pairwise"]["rank_spearman_mean"], 0.0)
        self.assertGreater(summary["pairwise"]["score_mean_abs_delta_mean"], 0.0)
        self.assertAlmostEqual(summary["runs"]["seed_1"]["topk_quality"]["k3"]["quality_failure_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["runs"]["seed_2"]["topk_quality"]["k4"]["duplicate_rate"], 0.25)

    def test_support_sampling_stability_runner_writes_reports(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            urls = _write_fixture(root)
            left_shard_dir = root / "partial_left_shards"
            _write_partial_embedding_shard(
                left_shard_dir,
                sample_ids=[hash_manifest_url(url) for url in urls["pretrain_urls"][:3]],
                values=np.asarray(
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1],
                        [0.0, 1.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1],
                        [0.0, 0.0, 1.0, 0.0, 0.1, 0.1, 0.1, 0.1],
                    ],
                    dtype="float32",
                ),
            )
            config = {
                "execution": {
                    "provider": "modal",
                    "allow_local_paths_for_tests": True,
                    "smoke_support_samples": 2,
                    "smoke_query_samples": 2,
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
                    "budgeted_candidate_only": True,
                    "support_split": "pretrain",
                    "query_split": "new",
                    "left_representation": "window_mean_std_pool",
                    "right_representation": "temporal_order",
                    "left_support_shard_dir": str(left_shard_dir),
                    "min_left_support_clips": 2,
                    "candidate_cache_dir": str(root / "query_cache"),
                    "right_support_cache_dir": str(root / "right_support_cache"),
                    "right_support_max_clips": 2,
                    "max_query_clips": 3,
                    "alpha": 0.5,
                    "old_novelty_k": 1,
                    "quality_threshold": 0.85,
                    "max_stationary_fraction": 0.90,
                    "max_abs_value": 60.0,
                    "cluster_similarity_threshold": 0.995,
                    "top_k_values": [2, 3],
                },
                "stability": {
                    "right_support_seeds": [1, 2],
                    "k_values": [2, 3],
                },
                "artifacts": {"output_dir": str(root / "out")},
            }

            result = run_support_sampling_stability(config, smoke=False)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))

        self.assertEqual(result["mode"], "full")
        self.assertEqual(report["summary"]["n_runs"], 2)
        self.assertIn("top2_overlap_mean", report["summary"]["pairwise"])
        self.assertTrue(Path(result["markdown_path"]).name.endswith(".md"))

    def test_modal_support_sampling_stability_entrypoint_and_config_validate(self):
        source = Path("modal_active_support_sampling_stability.py").read_text(encoding="utf-8")
        config = json.loads(Path("configs/active_support_sampling_stability_budget_cpu.json").read_text(encoding="utf-8"))

        self.assertIn("marginal-value-active-support-sampling-stability", source)
        self.assertNotIn('gpu="H100"', source)
        self.assertIn("cpu=16", source)
        self.assertEqual(config["stability"]["right_support_seeds"], [1, 2, 3])
        self.assertEqual(config["execution"]["smoke_max_seeds"], 2)
        validate_support_sampling_stability_config(config)


def _rank_row(
    worker_id: str,
    rank: int,
    score: float,
    *,
    cluster: int,
    quality_gate_pass: bool = True,
    physical_validity_pass: bool = True,
) -> dict[str, object]:
    return {
        "worker_id": worker_id,
        "rank": rank,
        "score": score,
        "quality_gate_pass": quality_gate_pass,
        "physical_validity_pass": physical_validity_pass,
        "new_cluster_id": cluster,
    }


def _write_fixture(root: Path) -> dict[str, list[str]]:
    pretrain_urls = [
        f"https://storage.googleapis.com/unit/pretrain/worker{idx:05d}/clip-old-{idx}.jsonl"
        for idx in range(4)
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
    return {"pretrain_urls": pretrain_urls, "new_urls": new_urls}


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
    values: np.ndarray,
) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        shard_dir / "shard_00000.npz",
        sample_ids=np.asarray(sample_ids, dtype=str),
        rep__window_mean_std_pool=np.asarray(values, dtype="float32"),
    )


if __name__ == "__main__":
    unittest.main()
