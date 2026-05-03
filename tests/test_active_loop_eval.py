import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.active.baselines import trace_artifact_rerank_order, trace_hygiene_rerank_order
from marginal_value.active.evaluate_active_loop import (
    ORACLE_POLICY_NAME,
    run_active_loop_eval,
    validate_active_loop_eval_config,
)
from marginal_value.data.split_manifest import hash_manifest_url


class ActiveLoopEvalTests(unittest.TestCase):
    def test_active_loop_eval_measures_hidden_target_gain_and_oracle_bound(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _write_active_loop_fixture(root)
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
                        "new": "cache/manifests/new_urls.txt",
                    },
                },
                "embeddings": {"cache_dir": str(root / "embedding_cache")},
                "episodes": {
                    "path": "episodes.jsonl",
                },
                "evaluation": {
                    "representations": [
                        "window_mean_std_pool",
                        "temporal_order",
                        "raw_shape_stats",
                        "window_shape_stats",
                    ],
                    "primary_representation": "window_mean_std_pool",
                    "policies": [
                        "random_valid",
                        "quality_only",
                        "old_novelty_only",
                        "kcenter_greedy_quality_gated",
                        "window_shape_stats_q85_stat90_abs60_clustercap2",
                        "oracle_greedy_eval_only",
                    ],
                    "k_values": [1, 2, 4],
                    "quality_threshold": 0.45,
                    "random_seed": 13,
                    "raw_shape_max_samples": 90,
                    "cluster_similarity_threshold": 0.995,
                },
                "artifacts": {"output_dir": str(root / "out")},
            }

            result = run_active_loop_eval(config, smoke=True)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            with Path(result["coverage_by_episode_path"]).open(newline="") as handle:
                coverage_rows = list(csv.DictReader(handle))
            with Path(result["selection_audit_path"]).open(newline="") as handle:
                audit_rows = list(csv.DictReader(handle))

        self.assertEqual(result["mode"], "smoke")
        self.assertEqual(result["n_episodes"], 1)
        self.assertIn(ORACLE_POLICY_NAME, report["policies"])
        self.assertEqual(report["registry_coverage"]["old"]["manifest_url_count"], 21)
        self.assertEqual(report["registry_coverage"]["old"]["registry_clip_count"], 20)
        self.assertEqual(report["registry_coverage"]["old"]["skipped_uncached_count"], 1)
        self.assertEqual(report["registry_coverage"]["new"]["manifest_url_count"], 2)
        self.assertEqual(report["registry_coverage_summary"]["manifest_old_url_count"], 21)
        self.assertEqual(report["registry_coverage_summary"]["cached_old_url_count"], 20)
        self.assertEqual(report["registry_coverage_summary"]["registry_old_clip_count"], 20)
        self.assertEqual(report["registry_coverage_summary"]["skipped_uncached_count"], 1)
        self.assertEqual(report["embedding_cache"]["status"], "miss")
        self.assertTrue(report["embedding_cache"]["path"].endswith(".npz"))
        self.assertEqual(report["episode_diagnostics"]["support_target_same_group_violations"], 0)
        self.assertGreater(
            report["policies"][ORACLE_POLICY_NAME]["coverage@1"]["balanced"]["relative_gain_mean"],
            report["policies"]["quality_only"]["coverage@1"]["balanced"]["relative_gain_mean"],
        )
        self.assertGreater(
            report["policies"][ORACLE_POLICY_NAME]["coverage@1"]["window_mean_std_pool"]["absolute_gain_mean"],
            0.0,
        )
        self.assertAlmostEqual(
            report["policies"][ORACLE_POLICY_NAME]["coverage@1"]["balanced"]["oracle_fraction_mean"],
            1.0,
            places=6,
        )
        self.assertLess(
            report["policies"]["quality_only"]["coverage@1"]["balanced"]["oracle_fraction_mean"],
            1.0,
        )
        self.assertTrue(
            any(
                row["policy"] == ORACLE_POLICY_NAME
                and row["k"] == "1"
                and row["representation"] == "balanced"
                and abs(float(row["oracle_fraction"]) - 1.0) < 1.0e-6
                for row in coverage_rows
            )
        )
        audit_for_k4 = [
            row
            for row in audit_rows
            if row["policy"] == ORACLE_POLICY_NAME and row["k"] == "4"
        ][0]
        role_mix = json.loads(audit_for_k4["candidate_role_mix_json"])
        self.assertIn("heldout_novel", role_mix)
        self.assertIn("low_quality", role_mix)
        self.assertGreaterEqual(float(audit_for_k4["low_quality_rate_at_k"]), 0.0)
        self.assertGreaterEqual(float(audit_for_k4["trace_fail_rate_at_k"]), 0.0)
        self.assertGreaterEqual(float(audit_for_k4["trace_artifact_fail_rate_at_k"]), 0.0)
        self.assertGreaterEqual(int(audit_for_k4["unique_source_groups_at_k"]), 1)
        self.assertEqual(fixture["heldout_worker"], "worker00002")

    def test_active_loop_eval_can_cap_full_episode_count(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_active_loop_fixture(root, episode_count=3)
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
                        "new": "cache/manifests/new_urls.txt",
                    },
                },
                "embeddings": {"cache_dir": str(root / "embedding_cache")},
                "episodes": {
                    "path": "episodes.jsonl",
                },
                "evaluation": {
                    "representations": [
                        "window_mean_std_pool",
                        "raw_shape_stats",
                    ],
                    "primary_representation": "window_mean_std_pool",
                    "policies": [
                        "quality_only",
                        "oracle_greedy_eval_only",
                    ],
                    "k_values": [1],
                    "quality_threshold": 0.45,
                    "random_seed": 13,
                    "raw_shape_max_samples": 90,
                    "max_episodes": 2,
                },
                "artifacts": {"output_dir": str(root / "out")},
            }

            result = run_active_loop_eval(config, smoke=False)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))

        self.assertEqual(result["mode"], "full")
        self.assertEqual(result["n_episodes"], 2)
        self.assertEqual(report["n_episodes"], 2)

    def test_modal_active_loop_eval_entrypoint_dispatches_remote_gpu_job(self):
        source = Path("modal_active_loop_eval.py").read_text(encoding="utf-8")
        config = json.loads(Path("configs/active_loop_eval_smoke_full_pretrain.json").read_text(encoding="utf-8"))

        self.assertIn("marginal-value-active-loop-eval", source)
        self.assertIn("remote_active_loop_eval.remote(config, smoke=True)", source)
        self.assertIn("run_active_loop_eval", source)
        self.assertIn('"torch==2.8.0"', source)
        self.assertIn('gpu="H100"', source)
        self.assertEqual(config["data"]["manifests"]["pretrain"], "cache/manifests/pretrain_full_cached_urls.txt")
        self.assertEqual(
            config["data"]["quality_metadata"],
            "/artifacts/active/quality/full_pretrain_smoke/active_quality_smoke.jsonl",
        )
        self.assertEqual(config["embeddings"]["cache_dir"], "/artifacts/active/embedding_cache/full_pretrain_smoke")
        self.assertIn("oracle_greedy_eval_only", config["evaluation"]["policies"])
        self.assertIn("raw_shape_stats", config["evaluation"]["representations"])
        scale_config = json.loads(Path("configs/active_loop_eval_scale_pretrain.json").read_text(encoding="utf-8"))
        self.assertEqual(scale_config["evaluation"]["k_values"], [5, 10, 25, 50, 100])
        self.assertEqual(scale_config["embeddings"]["cache_dir"], "/artifacts/active/embedding_cache/scale_pretrain")
        fixed_config = json.loads(Path("configs/active_loop_eval_ts2vec_fixed_window_blend_scale.json").read_text(encoding="utf-8"))
        self.assertEqual(
            fixed_config["evaluation"]["representation_options"]["ts2vec_checkpoint_path"],
            "/artifacts/checkpoints/ts2vec_fixed_crops_candidate_eval/ts2vec_best.pt",
        )
        self.assertEqual(
            fixed_config["embeddings"]["cache_dir"],
            "/artifacts/active/embedding_cache/ts2vec_fixed_crop_candidate_scale",
        )
        self.assertEqual(
            fixed_config["artifacts"]["output_dir"],
            "/artifacts/active/eval/ts2vec_fixed_crop_window_blend_scale",
        )
        validate_active_loop_eval_config(fixed_config)
        fixed_medium_config = json.loads(
            Path("configs/active_loop_eval_ts2vec_fixed_window_blend_medium.json").read_text(encoding="utf-8")
        )
        self.assertEqual(fixed_medium_config["evaluation"]["max_episodes"], 8)
        self.assertEqual(
            fixed_medium_config["embeddings"]["cache_dir"],
            "/artifacts/active/embedding_cache/ts2vec_fixed_crop_candidate_medium",
        )
        self.assertEqual(
            fixed_medium_config["artifacts"]["output_dir"],
            "/artifacts/active/eval/ts2vec_fixed_crop_window_blend_medium",
        )
        validate_active_loop_eval_config(fixed_medium_config)
        current_medium_config = json.loads(
            Path("configs/active_loop_eval_ts2vec_window_blend_medium.json").read_text(encoding="utf-8")
        )
        self.assertEqual(current_medium_config["evaluation"]["max_episodes"], 8)
        self.assertEqual(
            current_medium_config["evaluation"]["representation_options"]["ts2vec_checkpoint_path"],
            "/artifacts/checkpoints/ts2vec_candidate_eval/ts2vec_best.pt",
        )
        self.assertEqual(
            current_medium_config["embeddings"]["cache_dir"],
            "/artifacts/active/embedding_cache/ts2vec_candidate_medium",
        )
        self.assertEqual(
            current_medium_config["artifacts"]["output_dir"],
            "/artifacts/active/eval/ts2vec_window_blend_medium",
        )
        validate_active_loop_eval_config(current_medium_config)
        trace_gate_config = json.loads(
            Path("configs/active_loop_eval_trace_gate_window_blend_scale.json").read_text(encoding="utf-8")
        )
        self.assertIn(
            "trace_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05",
            trace_gate_config["evaluation"]["policies"],
        )
        self.assertEqual(
            trace_gate_config["artifacts"]["output_dir"],
            "/artifacts/active/eval/trace_gate_ts2vec_window_blend_scale",
        )
        validate_active_loop_eval_config(trace_gate_config)
        artifact_gate_config = json.loads(
            Path("configs/active_loop_eval_artifact_gate_window_blend_scale.json").read_text(encoding="utf-8")
        )
        self.assertIn(
            "artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05",
            artifact_gate_config["evaluation"]["policies"],
        )
        self.assertEqual(
            artifact_gate_config["artifacts"]["output_dir"],
            "/artifacts/active/eval/artifact_gate_ts2vec_window_blend_scale",
        )
        validate_active_loop_eval_config(artifact_gate_config)

    def test_blended_ts2vec_window_policy_names_validate(self):
        config = {
            "execution": {"provider": "modal", "allow_local_paths_for_tests": True},
            "data": {
                "root": "/tmp/unit",
                "manifests": {"pretrain": "cache/manifests/pretrain_full_cached_urls.txt"},
            },
            "episodes": {"path": "episodes.jsonl"},
            "evaluation": {
                "representations": ["ts2vec", "window_mean_std_pool"],
                "primary_representation": "ts2vec",
                "policies": [
                    "blend_old_novelty_ts2vec_window_mean_std_pool_a03",
                    "blend_kcenter_ts2vec_window_mean_std_pool_a07",
                    "trace_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05",
                    "artifact_gate_blend_kcenter_ts2vec_window_mean_std_pool_a05",
                ],
                "k_values": [5],
            },
            "artifacts": {"output_dir": "/tmp/unit/out"},
        }

        validate_active_loop_eval_config(config)

    def test_trace_hygiene_rerank_demotes_trace_failed_rows_after_base_order(self):
        rows = [
            {
                "sample_id": "spike",
                "quality_score": 0.99,
                "quality__spike_rate": 0.030,
                "trace__verdict": "likely_artifact",
            },
            {
                "sample_id": "clean",
                "quality_score": 0.91,
                "quality__spike_rate": 0.000,
                "trace__verdict": "plausible_motion",
            },
            {
                "sample_id": "stationary",
                "quality_score": 0.95,
                "quality__spike_rate": 0.000,
                "trace__verdict": "mostly_stationary",
            },
        ]

        order = trace_hygiene_rerank_order([0, 1, 2], rows, spike_rate_threshold=0.025)

        self.assertEqual(order, [1, 0, 2])

    def test_trace_artifact_rerank_only_demotes_likely_artifacts(self):
        rows = [
            {
                "sample_id": "stationary",
                "quality_score": 0.95,
                "quality__spike_rate": 0.000,
                "trace__verdict": "mostly_stationary",
            },
            {
                "sample_id": "clean",
                "quality_score": 0.91,
                "quality__spike_rate": 0.000,
                "trace__verdict": "plausible_motion",
            },
            {
                "sample_id": "artifact",
                "quality_score": 0.99,
                "quality__spike_rate": 0.030,
                "trace__verdict": "likely_artifact",
            },
        ]

        order = trace_artifact_rerank_order([2, 0, 1], rows)

        self.assertEqual(order, [0, 1, 2])

    def test_active_loop_eval_rejects_nonpositive_max_episodes(self):
        config = {
            "execution": {"provider": "modal", "allow_local_paths_for_tests": True},
            "data": {
                "root": "/tmp/unit",
                "manifests": {"pretrain": "cache/manifests/pretrain_full_cached_urls.txt"},
            },
            "episodes": {"path": "episodes.jsonl"},
            "evaluation": {
                "representations": ["window_mean_std_pool"],
                "primary_representation": "window_mean_std_pool",
                "k_values": [5],
                "max_episodes": 0,
            },
            "artifacts": {"output_dir": "/tmp/unit/out"},
        }

        with self.assertRaisesRegex(ValueError, "evaluation.max_episodes must be positive"):
            validate_active_loop_eval_config(config)


def _write_active_loop_fixture(root: Path, episode_count: int = 1) -> dict[str, str]:
    pretrain_urls: list[str] = []
    quality_rows: list[dict[str, float | str]] = []
    centers = {
        "worker00000": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
        "worker00001": np.asarray([1.0, 0.1, 0.0, 0.0], dtype=float),
        "worker00002": np.asarray([0.0, 1.0, 0.0, 0.0], dtype=float),
        "worker00003": np.asarray([-1.0, 0.0, 0.0, 0.0], dtype=float),
        "worker00004": np.asarray([0.0, 0.0, 1.0, 0.0], dtype=float),
    }
    for worker, center in centers.items():
        for clip_idx in range(4):
            url = _url("pretrain", worker, clip_idx)
            pretrain_urls.append(url)
            sample_id = hash_manifest_url(url)
            score = 0.20 if worker == "worker00003" and clip_idx == 0 else 0.90
            quality_rows.append(
                {
                    "sample_id": sample_id,
                    "quality_score": score,
                    "stationary_fraction": 0.99 if score < 0.45 else 0.10,
                    "max_abs_value": 12.0,
                }
            )
            _write_cached_clip(root, url, center=center + clip_idx * 0.01, raw_axis=_raw_axis_for_worker(worker))
    missing_url = _url("pretrain", "worker99999", 0)
    pretrain_urls.append(missing_url)
    _write_manifest(root, "cache/manifests/pretrain_full_cached_urls.txt", pretrain_urls)

    new_urls = [_url("new", "worker10000", 0), _url("new", "worker10001", 0)]
    _write_manifest(root, "cache/manifests/new_urls.txt", new_urls)
    for url in new_urls:
        _write_cached_clip(root, url, center=np.asarray([0.0, 0.0, 1.0, 0.0], dtype=float), raw_axis=2)
        quality_rows.append({"sample_id": hash_manifest_url(url), "quality_score": 0.90})

    quality_path = root / "quality.jsonl"
    quality_path.write_text("\n".join(json.dumps(row) for row in quality_rows) + "\n", encoding="utf-8")
    episode = {
        "episode_id": "episode_00000",
        "seed": 123,
        "support_clip_ids": [
            hash_manifest_url(_url("pretrain", "worker00000", 0)),
            hash_manifest_url(_url("pretrain", "worker00000", 1)),
            hash_manifest_url(_url("pretrain", "worker00001", 0)),
            hash_manifest_url(_url("pretrain", "worker00001", 1)),
        ],
        "candidate_clip_ids": [
            hash_manifest_url(_url("pretrain", "worker00000", 2)),
            hash_manifest_url(_url("pretrain", "worker00002", 0)),
            hash_manifest_url(_url("pretrain", "worker00002", 1)),
            hash_manifest_url(_url("pretrain", "worker00003", 0)),
        ],
        "hidden_target_clip_ids": [
            hash_manifest_url(_url("pretrain", "worker00002", 2)),
            hash_manifest_url(_url("pretrain", "worker00002", 3)),
        ],
        "distractor_clip_ids": [
            hash_manifest_url(_url("pretrain", "worker00000", 2)),
            hash_manifest_url(_url("pretrain", "worker00002", 1)),
            hash_manifest_url(_url("pretrain", "worker00003", 0)),
        ],
        "heldout_source_groups": ["worker00002"],
        "known_source_groups": ["worker00000", "worker00001"],
        "candidate_roles": {
            hash_manifest_url(_url("pretrain", "worker00000", 2)): "known_like",
            hash_manifest_url(_url("pretrain", "worker00002", 0)): "heldout_novel",
            hash_manifest_url(_url("pretrain", "worker00002", 1)): "near_duplicate",
            hash_manifest_url(_url("pretrain", "worker00003", 0)): "low_quality",
        },
        "low_quality_clip_ids": [hash_manifest_url(_url("pretrain", "worker00003", 0))],
    }
    episode_rows = []
    for episode_idx in range(episode_count):
        row = dict(episode)
        row["episode_id"] = f"episode_{episode_idx:05d}"
        episode_rows.append(json.dumps(row))
    (root / "episodes.jsonl").write_text("\n".join(episode_rows) + "\n", encoding="utf-8")
    return {"heldout_worker": "worker00002"}


def _url(split: str, worker: str, clip: int) -> str:
    return f"https://storage.googleapis.com/unit/{split}/{worker}/clip{clip:03d}.jsonl"


def _write_manifest(root: Path, relpath: str, urls: list[str]) -> None:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(urls) + "\n", encoding="utf-8")


def _write_cached_clip(root: Path, url: str, *, center: np.ndarray, raw_axis: int) -> None:
    sample_id = hash_manifest_url(url)
    raw_dir = root / "cache" / "raw"
    feature_dir = root / "cache" / "features"
    raw_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    trend = np.linspace(0.0, 0.03, 5, dtype=np.float32)[:, None]
    windows = np.asarray(center, dtype=np.float32)[None, :] + trend
    np.savez(
        feature_dir / f"{sample_id}.npz",
        window_features=windows,
        clip_features=np.asarray(center, dtype=np.float32),
    )
    _write_raw_jsonl(raw_dir / f"{sample_id}.jsonl", axis=raw_axis)


def _write_raw_jsonl(path: Path, *, axis: int, n_samples: int = 90, sample_rate: int = 30) -> None:
    t = np.arange(n_samples, dtype=float) / sample_rate
    samples = np.zeros((n_samples, 6), dtype=float)
    samples[:, 2] = 9.81
    samples[:, axis] = np.sin(2.0 * np.pi * 0.7 * t)
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


def _raw_axis_for_worker(worker: str) -> int:
    if worker == "worker00002":
        return 1
    if worker == "worker00004":
        return 2
    return 0


if __name__ == "__main__":
    unittest.main()
