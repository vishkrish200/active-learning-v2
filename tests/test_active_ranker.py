import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.active.evaluate_active_loop import load_active_episodes
from marginal_value.active.baselines import learned_score_quality_gated_kcenter_order
from marginal_value.active.ranker import (
    FORBIDDEN_FEATURE_NAMES,
    HYBRID_POLICY_NAME,
    build_active_ranker_feature_table,
    episode_level_split,
    run_active_ranker_train_eval,
)
from marginal_value.active.registry import ClipRecord, ClipRegistry
from marginal_value.data.split_manifest import hash_manifest_url


class ActiveRankerTests(unittest.TestCase):
    def test_hybrid_policy_quality_gates_then_diversifies_learned_scores(self):
        support = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
        candidates = np.asarray(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.999, 0.001],
                [0.0, 0.0, 1.0],
                [0.0, 0.8, 0.2],
            ],
            dtype=np.float32,
        )
        rows = [
            {"sample_id": "good_duplicate_a", "quality_score": 0.95, "stationary_fraction": 0.1, "max_abs_value": 12.0, "learned_score": 0.95},
            {"sample_id": "good_duplicate_b", "quality_score": 0.95, "stationary_fraction": 0.1, "max_abs_value": 12.0, "learned_score": 0.94},
            {"sample_id": "good_diverse", "quality_score": 0.95, "stationary_fraction": 0.1, "max_abs_value": 12.0, "learned_score": 0.80},
            {"sample_id": "bad_high_score", "quality_score": 0.20, "stationary_fraction": 0.1, "max_abs_value": 12.0, "learned_score": 0.99},
        ]

        order = learned_score_quality_gated_kcenter_order(
            support,
            candidates,
            rows,
            quality_threshold=0.85,
            max_stationary_fraction=0.90,
            max_abs_value=60.0,
            max_selected=2,
            pool_multiplier=2.0,
        )

        self.assertEqual(order[:3], [0, 2, 1])
        self.assertEqual(order[-1], 3)

    def test_feature_table_uses_deployable_features_and_episode_split(self):
        registry, episodes, labels, embeddings = _memory_fixture()

        table = build_active_ranker_feature_table(
            episodes=episodes,
            registry=registry,
            embeddings=embeddings,
            label_rows=labels,
            representations=["window_mean_std_pool", "window_shape_stats", "raw_shape_stats", "ts2vec"],
            old_novelty_k=1,
            cluster_similarity_threshold=0.99,
        )
        split = episode_level_split([episode.episode_id for episode in episodes], train_count=2, validation_count=0, test_count=1)

        self.assertEqual(table.values.shape[0], 6)
        self.assertEqual(len(table.feature_names), table.values.shape[1])
        self.assertTrue(set(split["train"]).isdisjoint(split["test"]))
        self.assertEqual(split["train"], ["episode_00000", "episode_00001"])
        self.assertEqual(split["test"], ["episode_00002"])
        self.assertIn("quality_score", table.feature_names)
        self.assertIn("old_novelty_window_mean_std_pool", table.feature_names)
        self.assertIn("old_novelty_ts2vec_k5", table.feature_names)
        self.assertIn("old_novelty_ts2vec_k10", table.feature_names)
        self.assertIn("ts2vec_x_raw_shape_old_novelty_score", table.feature_names)
        self.assertIn("candidate_nn_distance_window_mean_std_pool", table.feature_names)
        for forbidden in FORBIDDEN_FEATURE_NAMES:
            self.assertNotIn(forbidden, table.feature_names)
        self.assertNotIn("candidate_role", table.feature_names)
        good_rows = [idx for idx, row in enumerate(table.rows) if row["sample_id"].endswith("_good")]
        bad_rows = [idx for idx, row in enumerate(table.rows) if row["sample_id"].endswith("_known")]
        self.assertGreater(float(np.mean(table.labels[good_rows])), float(np.mean(table.labels[bad_rows])))

    def test_feature_table_requires_configured_label_column(self):
        registry, episodes, labels, embeddings = _memory_fixture(include_gated_labels=False)

        with self.assertRaisesRegex(ValueError, "compute_gated_labels"):
            build_active_ranker_feature_table(
                episodes=episodes,
                registry=registry,
                embeddings=embeddings,
                label_rows=labels,
                representations=["window_mean_std_pool", "window_shape_stats"],
                label_column="gated_balanced_gain",
                old_novelty_k=1,
                cluster_similarity_threshold=0.99,
            )

    def test_run_active_ranker_train_eval_writes_holdout_active_loop_report(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            good_by_episode = _write_ranker_fixture(root)
            config = {
                "execution": {
                    "provider": "modal",
                    "allow_local_paths_for_tests": True,
                    "smoke_max_episodes": 3,
                },
                "data": {
                    "root": str(root),
                    "feature_glob": "cache/features/*.npz",
                    "raw_glob": "cache/raw/*.jsonl",
                    "quality_metadata": "quality.jsonl",
                    "manifests": {"pretrain": "cache/manifests/pretrain_full_cached_urls.txt"},
                },
                "episodes": {"path": "episodes.jsonl"},
                "labels": {"path": "labels.csv"},
                "embeddings": {"cache_dir": str(root / "embedding_cache")},
                "ranker": {
                    "representations": ["window_mean_std_pool", "window_shape_stats"],
                    "primary_representation": "window_mean_std_pool",
                    "label_column": "gated_balanced_gain",
                    "ridge_l2": 0.1,
                    "old_novelty_k": 1,
                    "cluster_similarity_threshold": 0.99,
                    "quality_threshold": 0.45,
                    "k_values": [1, 2],
                    "train_episode_count": 2,
                    "validation_episode_count": 0,
                    "test_episode_count": 1,
                    "baseline_policies": ["random_valid", "old_novelty_only", "oracle_greedy_eval_only"],
                    "hybrid_policy": {
                        "enabled": True,
                        "quality_threshold": 0.85,
                        "max_stationary_fraction": 0.90,
                        "max_abs_value": 60.0,
                        "pool_multiplier": 2.0,
                    },
                },
                "artifacts": {"output_dir": str(root / "out")},
            }

            result = run_active_ranker_train_eval(config, smoke=True)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            with Path(result["scores_csv_path"]).open(newline="") as handle:
                score_rows = list(csv.DictReader(handle))

        self.assertEqual(result["mode"], "smoke")
        self.assertEqual(report["split"]["train"], ["episode_00000", "episode_00001"])
        self.assertEqual(report["split"]["test"], ["episode_00002"])
        self.assertEqual(report["feature_leakage_check"]["forbidden_feature_count"], 0)
        self.assertIn("learned_ridge_balanced_gain", report["splits"]["test"]["policies"])
        self.assertIn(HYBRID_POLICY_NAME, report["splits"]["test"]["policies"])
        test_scores = [row for row in score_rows if row["split"] == "test"]
        by_sample = {row["sample_id"]: float(row["learned_score"]) for row in test_scores}
        self.assertGreater(by_sample[good_by_episode["episode_00002"]], max(score for sample_id, score in by_sample.items() if sample_id != good_by_episode["episode_00002"]))

    def test_modal_active_ranker_train_entrypoint_uses_remote_gpu_job(self):
        source = Path("modal_active_ranker_train.py").read_text(encoding="utf-8")
        config = json.loads(Path("configs/active_ranker_train_scale_pretrain.json").read_text(encoding="utf-8"))

        self.assertIn("marginal-value-active-ranker-train", source)
        self.assertIn("remote_active_ranker_train.remote(config, smoke=True)", source)
        self.assertIn("run_active_ranker_train_eval", source)
        self.assertIn('"torch==2.8.0"', source)
        self.assertIn('gpu="H100"', source)
        self.assertEqual(config["labels"]["path"], "/artifacts/active/labels/ranker_hygiene_fix/active_label_gain_full.csv")
        self.assertIn("ts2vec", config["ranker"]["representations"])
        self.assertEqual(config["ranker"]["primary_representation"], "ts2vec")
        self.assertEqual(config["ranker"]["label_column"], "gated_balanced_gain")
        self.assertEqual(config["ranker"]["hygiene_gate"]["quality_threshold"], 0.85)
        self.assertEqual(config["ranker"]["train_episode_count"], 48)
        self.assertEqual(config["ranker"]["validation_episode_count"], 8)
        self.assertEqual(config["ranker"]["test_episode_count"], 8)
        self.assertTrue(config["ranker"]["hybrid_policy"]["enabled"])


def _memory_fixture(*, include_gated_labels: bool = True):
    clips = []
    embeddings = {"window_mean_std_pool": {}, "window_shape_stats": {}, "ts2vec": {}, "raw_shape_stats": {}}
    labels = []
    episode_rows = []
    for episode_idx in range(3):
        episode_id = f"episode_{episode_idx:05d}"
        support = f"{episode_id}_support"
        known = f"{episode_id}_known"
        good = f"{episode_id}_good"
        target = f"{episode_id}_target"
        for sample_id, vector, group in (
            (support, [1.0, 0.0, 0.0, 0.0], f"known_{episode_idx}"),
            (known, [0.98, 0.02, 0.0, 0.0], f"known_{episode_idx}"),
            (good, [0.0, 1.0, 0.0, 0.0], f"novel_{episode_idx}"),
            (target, [0.0, 1.0, 0.0, 0.0], f"novel_{episode_idx}"),
        ):
            clips.append(_clip(sample_id, group=group, quality=0.9))
            for representation in embeddings:
                embeddings[representation][sample_id] = np.asarray(vector, dtype=np.float32)
        episode_rows.append(
            {
                "episode_id": episode_id,
                "support_clip_ids": [support],
                "candidate_clip_ids": [known, good],
                "hidden_target_clip_ids": [target],
                "candidate_roles": {known: "known_like", good: "heldout_novel"},
                "heldout_source_groups": [f"novel_{episode_idx}"],
                "known_source_groups": [f"known_{episode_idx}"],
            }
        )
        known_label = {"episode_id": episode_id, "sample_id": known, "candidate_role": "known_like", "balanced_gain": 0.0}
        good_label = {"episode_id": episode_id, "sample_id": good, "candidate_role": "heldout_novel", "balanced_gain": 0.25}
        if include_gated_labels:
            known_label["gated_balanced_gain"] = 0.0
            good_label["gated_balanced_gain"] = 0.25
        labels.append(known_label)
        labels.append(good_label)
    registry = ClipRegistry(root=Path("."), clips=tuple(clips))
    episodes_path = Path("/tmp/active_ranker_memory_episodes.jsonl")
    episodes_path.write_text("\n".join(json.dumps(row) for row in episode_rows) + "\n", encoding="utf-8")
    episodes = load_active_episodes(episodes_path)
    episodes_path.unlink()
    return registry, episodes, labels, embeddings


def _write_ranker_fixture(root: Path) -> dict[str, str]:
    rows = []
    quality_rows = []
    label_rows = []
    good_by_episode = {}
    urls = []
    for episode_idx in range(3):
        episode_id = f"episode_{episode_idx:05d}"
        urls_by_role = {
            "support": f"https://storage.googleapis.com/unit/pretrain/worker{episode_idx:05d}a/clip000.jsonl",
            "known": f"https://storage.googleapis.com/unit/pretrain/worker{episode_idx:05d}a/clip001.jsonl",
            "good": f"https://storage.googleapis.com/unit/pretrain/worker{episode_idx:05d}b/clip000.jsonl",
            "target": f"https://storage.googleapis.com/unit/pretrain/worker{episode_idx:05d}b/clip001.jsonl",
        }
        ids = {role: hash_manifest_url(url) for role, url in urls_by_role.items()}
        vectors = {
            ids["support"]: [1.0, 0.0, 0.0, 0.0],
            ids["known"]: [0.98, 0.02, 0.0, 0.0],
            ids["good"]: [0.0, 1.0, 0.0, 0.0],
            ids["target"]: [0.0, 1.0, 0.0, 0.0],
        }
        url_by_id = {ids[role]: url for role, url in urls_by_role.items()}
        for sample_id, vector in vectors.items():
            url = url_by_id[sample_id]
            urls.append(url)
            _write_cached_clip(root, sample_id, url=url, center=np.asarray(vector, dtype=float))
            quality_rows.append(json.dumps({"sample_id": sample_id, "quality_score": 0.9, "stationary_fraction": 0.1, "max_abs_value": 12.0}))
        rows.append(
            {
                "episode_id": episode_id,
                "support_clip_ids": [ids["support"]],
                "candidate_clip_ids": [ids["known"], ids["good"]],
                "hidden_target_clip_ids": [ids["target"]],
                "candidate_roles": {ids["known"]: "known_like", ids["good"]: "heldout_novel"},
                "heldout_source_groups": [f"worker{episode_idx:05d}b"],
                "known_source_groups": [f"worker{episode_idx:05d}a"],
            }
        )
        label_rows.append({"episode_id": episode_id, "sample_id": ids["known"], "candidate_role": "known_like", "balanced_gain": 0.0, "gated_balanced_gain": 0.0})
        label_rows.append({"episode_id": episode_id, "sample_id": ids["good"], "candidate_role": "heldout_novel", "balanced_gain": 0.25, "gated_balanced_gain": 0.25})
        good_by_episode[episode_id] = ids["good"]
    _write_manifest(root, "cache/manifests/pretrain_full_cached_urls.txt", urls)
    (root / "episodes.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    (root / "quality.jsonl").write_text("\n".join(quality_rows) + "\n", encoding="utf-8")
    _write_csv(root / "labels.csv", label_rows)
    return good_by_episode


def _clip(sample_id: str, *, group: str, quality: float) -> ClipRecord:
    return ClipRecord(
        sample_id=sample_id,
        split="pretrain",
        url=f"https://storage.googleapis.com/unit/pretrain/{group}/{sample_id}.jsonl",
        source_group_id=group,
        worker_id=group,
        raw_path=Path(f"{sample_id}.jsonl"),
        feature_path=Path(f"{sample_id}.npz"),
        quality={"quality_score": quality, "stationary_fraction": 0.1, "max_abs_value": 12.0},
    )


def _write_manifest(root: Path, relpath: str, urls: list[str]) -> None:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(urls) + "\n", encoding="utf-8")


def _write_cached_clip(root: Path, sample_id: str, *, url: str, center: np.ndarray) -> None:
    raw_dir = root / "cache" / "raw"
    feature_dir = root / "cache" / "features"
    raw_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    np.savez(feature_dir / f"{sample_id}.npz", window_features=np.asarray(center, dtype=np.float32)[None, :])
    rows = []
    for idx in range(90):
        rows.append(json.dumps({"t_us": idx * 33333, "acc": [1.0, 0.0, 9.81], "gyro": [0.0, 0.0, 0.0], "url": url}))
    (raw_dir / f"{sample_id}.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
