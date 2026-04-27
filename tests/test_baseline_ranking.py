import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json

import numpy as np

from marginal_value.data.split_manifest import SplitSample
from marginal_value.models.learned_linear_ranker import fit_linear_ranker, write_linear_ranker_model
from marginal_value.preprocessing.quality import compute_quality_from_jsonl, quality_scores_for_rows
from marginal_value.ranking.config import (
    RankingLeakageError,
    load_ranking_config,
    validate_ranking_config,
)
from marginal_value.ranking.baseline_ranker import (
    apply_grammar_score_promotion,
    apply_stationary_singleton_guard,
    build_reason_codes,
    build_scored_rows,
    cluster_cap_rank_rows,
    cluster_aware_rank_rows,
    compute_batch_clusters,
    combine_novelty_density,
    annotate_cluster_features,
    mmr_rank_rows,
    parent_prefix_cluster_cap_rank_rows,
    quality_only_rank_rows,
    quality_gated_old_novelty_rank_rows,
    raw_shape_stats_embedding,
    split_large_clusters,
    temporal_order_embedding,
    tiered_cluster_cap_rank_rows,
    window_mean_std_embedding,
)
from marginal_value.ranking.modal_baseline_rank import (
    GRAMMAR_FEATURE_COLUMNS,
    _build_candidate_eval,
    _build_corruption_eval,
    _build_embedding_lookup,
    _candidate_grammar_promotion_allowed,
    _join_grammar_features,
    _join_quality_metadata,
    _load_grammar_features,
    _load_embeddings,
    _maybe_apply_learned_ranker_scores,
    _maybe_apply_score_guards,
    _maybe_split_large_clusters,
    _selection_trace_rows,
    _submission_rows,
)


class BaselineRankingTests(unittest.TestCase):
    def test_window_mean_std_embedding_concatenates_statistics(self):
        windows = np.array([[1.0, 3.0], [3.0, 7.0], [5.0, 11.0]])

        embedding = window_mean_std_embedding(windows)

        self.assertEqual(embedding.shape, (4,))
        np.testing.assert_allclose(embedding[:2], [3.0, 7.0])
        np.testing.assert_allclose(embedding[2:], np.std(windows, axis=0))

    def test_alternate_deterministic_embeddings_have_stable_shapes(self):
        windows = np.arange(4 * 3, dtype=float).reshape(4, 3)
        samples = np.arange(8 * 6, dtype=float).reshape(8, 6)

        temporal = temporal_order_embedding(windows, n_segments=2)
        raw_shape = raw_shape_stats_embedding(samples, sample_rate=30.0)

        self.assertEqual(temporal.shape, (21,))
        self.assertEqual(raw_shape.shape, (120,))

    def test_combine_novelty_density_rewards_supported_novelty(self):
        novelty = np.array([0.9, 0.9, 0.2])
        density = np.array([0.8, 0.1, 0.9])

        scores = combine_novelty_density(novelty, density, novelty_weight=0.75)

        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[0], scores[2])

    def test_reason_codes_distinguish_supported_singleton_and_redundant(self):
        rows = [
            {"quality_score": 0.9, "old_novelty_score": 0.9, "new_density_score": 0.8},
            {"quality_score": 0.9, "old_novelty_score": 0.9, "new_density_score": 0.1},
            {"quality_score": 0.9, "old_novelty_score": 0.1, "new_density_score": 0.9},
            {"quality_score": 0.2, "old_novelty_score": 0.9, "new_density_score": 0.8},
        ]

        self.assertEqual(
            build_reason_codes(rows),
            [
                "COHESIVE_NEW_WORKFLOW",
                "HIGH_NOVELTY_SINGLETON",
                "REDUNDANT_KNOWN_WORKFLOW",
                "LOW_QUALITY",
            ],
        )

    def test_rare_temporal_reason_code_uses_calibrated_grammar_thresholds(self):
        rows = [
            {
                "quality_score": 0.9,
                "old_novelty_score": 0.25,
                "new_density_score": 0.35,
                "grammar_score_component": 0.75,
                "grammar_promotion_delta": 0.08,
            },
            {
                "quality_score": 0.9,
                "old_novelty_score": 0.25,
                "new_density_score": 0.35,
                "grammar_score_component": 0.74,
                "grammar_promotion_delta": 0.20,
            },
            {
                "quality_score": 0.9,
                "old_novelty_score": 0.25,
                "new_density_score": 0.35,
                "grammar_score_component": 0.90,
                "grammar_promotion_delta": 0.07,
            },
        ]

        self.assertEqual(
            build_reason_codes(rows),
            [
                "RARE_TEMPORAL_COMPOSITION",
                "RARE_MOTION_PRIMITIVES",
                "RARE_MOTION_PRIMITIVES",
            ],
        )

    def test_stationary_singleton_guard_penalizes_unsupported_stationary_singleton(self):
        rows = [
            {
                "sample_id": "stationary-singleton",
                "quality_score": 0.98,
                "old_novelty_score": 1.0,
                "new_density_score": 0.05,
                "grammar_score_component": 0.70,
                "ranker_score": 0.80,
                "final_score": 0.784,
                "stationary_fraction": 0.95,
                "reason_code": "HIGH_NOVELTY_SINGLETON",
            },
            {
                "sample_id": "grammar-supported",
                "quality_score": 0.98,
                "old_novelty_score": 1.0,
                "new_density_score": 0.05,
                "grammar_score_component": 0.92,
                "ranker_score": 0.80,
                "final_score": 0.784,
                "stationary_fraction": 0.95,
                "reason_code": "HIGH_NOVELTY_SINGLETON",
            },
            {
                "sample_id": "moving-singleton",
                "quality_score": 0.98,
                "old_novelty_score": 1.0,
                "new_density_score": 0.05,
                "grammar_score_component": 0.70,
                "ranker_score": 0.80,
                "final_score": 0.784,
                "stationary_fraction": 0.20,
                "reason_code": "HIGH_NOVELTY_SINGLETON",
            },
        ]

        guarded, summary = apply_stationary_singleton_guard(
            rows,
            stationary_threshold=0.9,
            max_new_density_score=0.35,
            min_grammar_score=0.85,
            penalty_multiplier=0.35,
        )
        by_id = {row["sample_id"]: row for row in guarded}

        self.assertEqual(summary["applied_count"], 1)
        self.assertTrue(by_id["stationary-singleton"]["stationary_singleton_guard_applied"])
        self.assertLess(by_id["stationary-singleton"]["final_score"], rows[0]["final_score"])
        self.assertFalse(by_id["grammar-supported"]["stationary_singleton_guard_applied"])
        self.assertFalse(by_id["moving-singleton"]["stationary_singleton_guard_applied"])

    def test_quality_metadata_join_exposes_stationary_features_to_ranking_rows(self):
        rows = [{"sample_id": "a", "quality_score": 0.9, "final_score": 0.5}]
        metadata = [
            {
                "sample_id": "a",
                "split": "new",
                "stationary_fraction": 0.91,
                "spike_rate": 0.01,
                "max_abs_value": 12.0,
            }
        ]

        joined = _join_quality_metadata(rows, metadata)

        self.assertTrue(joined[0]["quality_metadata_present"])
        self.assertEqual(joined[0]["quality_metadata_split"], "new")
        self.assertAlmostEqual(joined[0]["stationary_fraction"], 0.91)

    def test_score_guard_config_helper_applies_stationary_guard(self):
        rows = [
            {
                "sample_id": "a",
                "quality_score": 1.0,
                "old_novelty_score": 0.9,
                "new_density_score": 0.0,
                "grammar_score_component": 0.5,
                "ranker_score": 0.8,
                "final_score": 0.8,
                "stationary_fraction": 0.95,
                "reason_code": "HIGH_NOVELTY_SINGLETON",
            }
        ]
        config = {
            "score_guards": {
                "stationary_singleton": {
                    "enabled": True,
                    "stationary_threshold": 0.9,
                    "max_new_density_score": 0.35,
                    "min_grammar_score": 0.85,
                    "penalty_multiplier": 0.5,
                }
            }
        }

        guarded, summary = _maybe_apply_score_guards(config, rows, label="query")

        self.assertTrue(summary["applied"])
        self.assertEqual(summary["label"], "query")
        self.assertAlmostEqual(guarded[0]["final_score"], 0.4)

    def test_quality_from_modal_jsonl_scores_corrupt_trace_lower(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            clean = _make_clean_imu(120)
            corrupt = clean.copy()
            corrupt[::8, 3] = 500.0
            clean_path = tmp_path / "clean.jsonl"
            corrupt_path = tmp_path / "corrupt.jsonl"
            _write_modal_jsonl(clean_path, clean)
            _write_modal_jsonl(corrupt_path, corrupt)

            clean_features = compute_quality_from_jsonl(clean_path, sample_rate=30)
            corrupt_features = compute_quality_from_jsonl(corrupt_path, sample_rate=30)

            self.assertEqual(clean_features["n_samples"], 120.0)
            self.assertGreater(clean_features["quality_score"], corrupt_features["quality_score"] + 0.15)

    def test_quality_penalizes_sparse_impossible_spikes(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            clean = _make_clean_imu(540)
            corrupt = clean.copy()
            corrupt[::45, 0] += 500.0
            clean_path = tmp_path / "clean.jsonl"
            corrupt_path = tmp_path / "sparse-spikes.jsonl"
            _write_modal_jsonl(clean_path, clean)
            _write_modal_jsonl(corrupt_path, corrupt)

            clean_features = compute_quality_from_jsonl(clean_path, sample_rate=30)
            corrupt_features = compute_quality_from_jsonl(corrupt_path, sample_rate=30)

            self.assertLess(corrupt_features["quality_score"], 0.45)
            self.assertGreater(clean_features["quality_score"], 0.9)
            self.assertGreater(corrupt_features["max_abs_value"], 250.0)

    def test_quality_from_jsonl_can_be_bounded_for_remote_ranking(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "long.jsonl"
            _write_modal_jsonl(path, _make_clean_imu(120))

            features = compute_quality_from_jsonl(path, sample_rate=30, max_samples=20)

            self.assertEqual(features["n_samples"], 20.0)

    def test_quality_scores_for_rows_follow_raw_paths(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            clean_path = tmp_path / "clean.jsonl"
            flat_path = tmp_path / "flat.jsonl"
            feature_path = tmp_path / "dummy.npz"
            _write_modal_jsonl(clean_path, _make_clean_imu(90))
            _write_modal_jsonl(flat_path, np.zeros((90, 6)))
            feature_path.write_bytes(b"")
            rows = [
                SplitSample("clean", "val", "clean-url", clean_path, feature_path),
                SplitSample("flat", "val", "flat-url", flat_path, feature_path),
            ]

            scores, metadata = quality_scores_for_rows(rows, sample_rate=30)

            self.assertEqual(scores.shape, (2,))
            self.assertEqual(metadata[0]["sample_id"], "clean")
            self.assertGreater(scores[0], scores[1])

    def test_corruption_eval_builds_low_quality_negative_candidates(self):
        rows = [
            SplitSample(f"pos-{idx}", "val", f"url-{idx}", Path(f"/raw/{idx}.jsonl"), Path(f"/features/{idx}.npz"))
            for idx in range(5)
        ]
        embeddings = np.arange(20, dtype=float).reshape(5, 4) + 1.0
        config = {
            "corruption_eval": {
                "enabled": True,
                "sample_size": 3,
                "quality_score": 0.05,
                "modes": ["flatline", "spike"],
            }
        }

        result = _build_corruption_eval(
            positive_rows=rows,
            positive_embeddings=embeddings,
            config=config,
            rng=np.random.default_rng(7),
        )

        self.assertEqual(result["embeddings"].shape, (3, 4))
        self.assertTrue(np.all(result["labels"] == 0))
        self.assertTrue(np.all(result["is_corruption"]))
        self.assertTrue(np.allclose(result["quality_scores"], 0.05))
        self.assertEqual(result["summary"]["n_corruptions"], 3)
        self.assertEqual(set(result["modes"]), {"flatline", "spike"})
        self.assertTrue(all("__corrupt_" in sample_id for sample_id in result["sample_ids"]))

    def test_raw_signal_corruption_eval_recomputes_embeddings_and_quality(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = []
            for idx in range(4):
                raw_path = tmp_path / f"pos-{idx}.jsonl"
                _write_modal_jsonl(raw_path, _make_clean_imu(120))
                rows.append(SplitSample(f"pos-{idx}", "val", f"url-{idx}", raw_path, tmp_path / f"pos-{idx}.npz"))
            embeddings = np.ones((4, 150), dtype=float)
            config = {
                "ranking": {"representation": "window_mean_std_pool"},
                "corruption_eval": {
                    "enabled": True,
                    "sample_size": 4,
                    "raw_signal": True,
                    "modes": ["flatline", "spike", "saturation", "jitter"],
                },
            }

            result = _build_corruption_eval(
                positive_rows=rows,
                positive_embeddings=embeddings,
                config=config,
                rng=np.random.default_rng(7),
                sample_rate=30.0,
            )

        self.assertEqual(result["embeddings"].shape[0], 4)
        self.assertEqual(result["embeddings"].shape[1], 150)
        self.assertEqual(result["summary"]["quality_score_source"], "computed_from_corrupted_raw")
        self.assertTrue(result["summary"]["raw_signal"])
        self.assertFalse(np.allclose(result["quality_scores"], 0.05))
        self.assertLess(float(np.min(result["quality_scores"])), 0.6)
        self.assertTrue(all(row["quality_score_source"] == "computed_from_corrupted_raw" for row in result["quality_metadata"]))

    def test_candidate_eval_includes_corruption_rate_metrics(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            feature_path = tmp_path / "dummy.npz"
            feature_path.write_bytes(b"")
            raw_paths = []
            for idx in range(6):
                raw_path = tmp_path / f"sample-{idx}.jsonl"
                _write_modal_jsonl(raw_path, _make_clean_imu(90))
                raw_paths.append(raw_path)
            positives = [
                SplitSample(f"val-{idx}", "val", f"val-url-{idx}", raw_paths[idx], feature_path)
                for idx in range(3)
            ]
            negatives = [
                SplitSample(f"pretrain-{idx}", "pretrain", f"pretrain-url-{idx}", raw_paths[idx + 3], feature_path)
                for idx in range(3)
            ]
            support_embeddings = np.asarray(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, 0.0],
                    [0.0, -1.0],
                ],
                dtype=float,
            )
            positive_embeddings = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]], dtype=float)
            negative_embeddings = np.asarray([[1.0, 0.0], [0.98, 0.02], [-1.0, 0.0]], dtype=float)

            result = _build_candidate_eval(
                support_embeddings=support_embeddings,
                positive_rows=positives,
                positive_embeddings=positive_embeddings,
                negative_rows=negatives,
                negative_embeddings=negative_embeddings,
                k_old=2,
                k_new_density=1,
                novelty_weight=0.75,
                mmr_lambda=0.25,
                reranker_method="cluster_aware",
                cluster_similarity_threshold=0.985,
                cluster_bonus_weight=0.25,
                cluster_cap_top_k=10,
                cluster_max_per_cluster=2,
                cluster_cap_key="new_cluster_id",
                cluster_cap_min_quality=0.0,
                prefix_cluster_cap_top_k=3,
                prefix_cluster_cap_key="new_cluster_parent_id",
                prefix_cluster_max_per_cluster=2,
                k_values=[1, 8],
                sample_rate=30.0,
                max_quality_samples=None,
                grammar_features={},
                config={
                    "ranking": {"seed": 11},
                    "corruption_eval": {
                        "enabled": True,
                        "sample_size": 2,
                        "quality_score": 0.05,
                        "modes": ["flatline", "jitter"],
                    },
                },
            )

        corruption_rows = [row for row in result["ranked_rows"] if row.get("is_corruption")]
        self.assertEqual(len(corruption_rows), 2)
        self.assertTrue(all(row["label"] == 0 for row in corruption_rows))
        self.assertTrue(all(row["quality_score"] == 0.05 for row in corruption_rows))
        self.assertTrue(all(row["reason_code"] == "LOW_QUALITY" for row in corruption_rows))
        self.assertEqual(result["corruption_eval"]["n_corruptions"], 2)
        self.assertIn("corruption_rate@1", result["metrics"])
        self.assertIn("corruption_rate@8", result["metrics"])
        self.assertGreater(result["metrics"]["corruption_rate@8"], 0.0)

    def test_encoder_artifact_embedding_loader_preserves_requested_row_order(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            support_embeddings = root / "support.npy"
            query_embeddings = root / "query.npy"
            support_manifest = root / "support.csv"
            query_manifest = root / "query.csv"
            np.save(support_embeddings, np.asarray([[1.0, 0.0], [0.5, 0.5]], dtype="float32"))
            np.save(query_embeddings, np.asarray([[0.0, 1.0]], dtype="float32"))
            support_manifest.write_text(
                "sample_id,split,url,raw_path,feature_path\n"
                "pretrain-a,pretrain,u,raw,feat\n"
                "pretrain-b,pretrain,u,raw,feat\n",
                encoding="utf-8",
            )
            query_manifest.write_text(
                "sample_id,split,url,raw_path,feature_path\n"
                "val-a,val,u,raw,feat\n",
                encoding="utf-8",
            )
            config = {
                "ranking": {"representation": "encoder_artifact"},
                "encoder_embeddings": {
                    "support_embeddings": str(support_embeddings),
                    "support_manifest": str(support_manifest),
                    "query_embeddings": str(query_embeddings),
                    "query_manifest": str(query_manifest),
                },
            }
            rows = [
                SplitSample("pretrain-b", "pretrain", "u", Path("raw"), Path("feat")),
                SplitSample("val-a", "val", "u", Path("raw"), Path("feat")),
                SplitSample("pretrain-a", "pretrain", "u", Path("raw"), Path("feat")),
            ]

            lookup = _build_embedding_lookup(config)
            embeddings = _load_embeddings(config, rows, label="unit", embedding_lookup=lookup)

            np.testing.assert_allclose(embeddings, [[0.5, 0.5], [0.0, 1.0], [1.0, 0.0]])

    def test_learned_ranker_scores_replace_phase_a_score_when_enabled(self):
        with TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "learned_ranker_model_full.json"
            train_x = np.asarray(
                [
                    [0.9, 0.8],
                    [0.8, 0.7],
                    [0.1, 0.2],
                    [0.2, 0.1],
                ],
                dtype=float,
            )
            train_y = np.asarray([1, 1, 0, 0], dtype=int)
            model = fit_linear_ranker(train_x, train_y)
            write_linear_ranker_model(
                model_path,
                model,
                feature_names=["old_knn_distance", "new_batch_density"],
                metadata={"source": "unit-test"},
            )
            rows = [
                {
                    "sample_id": "learned-high",
                    "worker_id": "learned-high",
                    "quality_score": 1.0,
                    "ranker_score": 0.1,
                    "final_score": 0.1,
                    "old_knn_distance": 0.85,
                    "new_batch_density": 0.75,
                    "old_novelty_score": 0.1,
                    "new_density_score": 0.1,
                },
                {
                    "sample_id": "learned-low",
                    "worker_id": "learned-low",
                    "quality_score": 1.0,
                    "ranker_score": 0.9,
                    "final_score": 0.9,
                    "old_knn_distance": 0.15,
                    "new_batch_density": 0.15,
                    "old_novelty_score": 0.1,
                    "new_density_score": 0.1,
                },
            ]

            scored, summary = _maybe_apply_learned_ranker_scores(
                {
                    "learned_ranker": {
                        "enabled": True,
                        "model_path": str(model_path),
                        "score_transform": "sigmoid",
                        "score_weight": 1.0,
                    }
                },
                rows,
                label="query",
            )

        by_id = {row["sample_id"]: row for row in scored}
        self.assertTrue(summary["applied"])
        self.assertEqual(summary["model_feature_count"], 2)
        self.assertGreater(by_id["learned-high"]["final_score"], by_id["learned-low"]["final_score"])
        self.assertEqual(by_id["learned-high"]["pre_learned_final_score"], 0.1)
        self.assertTrue(by_id["learned-high"]["learned_ranker_applied"])

    def test_build_scored_rows_multiplies_quality_gate(self):
        rows = build_scored_rows(
            sample_ids=["good", "bad"],
            embeddings=np.eye(2),
            old_knn_distance=np.array([0.9, 0.5]),
            new_density=np.array([0.8, 0.4]),
            quality_scores=np.array([1.0, 0.25]),
        )

        self.assertGreater(rows[0]["final_score"], rows[1]["final_score"])
        self.assertEqual(rows[1]["reason_code"], "LOW_QUALITY")

    def test_grammar_score_promotion_is_quality_and_support_gated(self):
        rows = [
            {
                "sample_id": "supported-grammar",
                "quality_score": 1.0,
                "ranker_score": 0.20,
                "final_score": 0.20,
                "old_novelty_score": 0.25,
                "new_density_score": 1.0,
                "grammar_feature_present": True,
                "token_nll_p95": 9.0,
                "transition_nll_p95": 7.0,
                "longest_unseen_phrase_len": 6.0,
                "rare_phrase_fraction": 0.9,
            },
            {
                "sample_id": "unsupported-grammar",
                "quality_score": 1.0,
                "ranker_score": 0.20,
                "final_score": 0.20,
                "old_novelty_score": 0.25,
                "new_density_score": 0.1,
                "grammar_feature_present": True,
                "token_nll_p95": 9.0,
                "transition_nll_p95": 7.0,
                "longest_unseen_phrase_len": 6.0,
                "rare_phrase_fraction": 0.9,
            },
            {
                "sample_id": "low-quality-grammar",
                "quality_score": 0.2,
                "ranker_score": 0.20,
                "final_score": 0.04,
                "old_novelty_score": 0.25,
                "new_density_score": 1.0,
                "grammar_feature_present": True,
                "token_nll_p95": 9.0,
                "transition_nll_p95": 7.0,
                "longest_unseen_phrase_len": 6.0,
                "rare_phrase_fraction": 0.9,
            },
            {
                "sample_id": "phase-a",
                "quality_score": 1.0,
                "ranker_score": 0.40,
                "final_score": 0.40,
                "old_novelty_score": 0.25,
                "new_density_score": 0.6,
                "grammar_feature_present": False,
                "token_nll_p95": 0.0,
                "transition_nll_p95": 0.0,
                "longest_unseen_phrase_len": 0.0,
                "rare_phrase_fraction": 0.0,
            },
        ]

        promoted = apply_grammar_score_promotion(
            rows,
            score_variant="grammar_surprisal_mix",
            score_weight=0.35,
            min_quality=0.45,
            min_new_density_score=0.35,
        )
        by_id = {row["sample_id"]: row for row in promoted}

        self.assertEqual(by_id["supported-grammar"]["phase_a_final_score"], 0.20)
        self.assertGreaterEqual(by_id["supported-grammar"]["grammar_promotion_delta"], 0.08)
        self.assertGreater(by_id["supported-grammar"]["final_score"], by_id["phase-a"]["final_score"])
        self.assertLess(by_id["unsupported-grammar"]["final_score"], by_id["phase-a"]["final_score"])
        self.assertLess(by_id["low-quality-grammar"]["final_score"], by_id["phase-a"]["final_score"])
        self.assertEqual(by_id["supported-grammar"]["reason_code"], "RARE_TEMPORAL_COMPOSITION")
        self.assertTrue(all("grammar_score_component" in row for row in promoted))

    def test_grammar_reason_code_requires_meaningful_promotion_delta(self):
        rows = [
            {
                "sample_id": "phase-a-already-high",
                "quality_score": 1.0,
                "ranker_score": 0.90,
                "final_score": 0.90,
                "old_novelty_score": 0.25,
                "new_density_score": 1.0,
                "grammar_feature_present": True,
                "token_nll_p95": 9.0,
                "transition_nll_p95": 7.0,
                "longest_unseen_phrase_len": 6.0,
                "rare_phrase_fraction": 0.9,
            },
            {
                "sample_id": "low-phase-a-grammar",
                "quality_score": 1.0,
                "ranker_score": 0.20,
                "final_score": 0.20,
                "old_novelty_score": 0.25,
                "new_density_score": 1.0,
                "grammar_feature_present": True,
                "token_nll_p95": 9.0,
                "transition_nll_p95": 7.0,
                "longest_unseen_phrase_len": 6.0,
                "rare_phrase_fraction": 0.9,
            },
            {
                "sample_id": "low-grammar-reference",
                "quality_score": 1.0,
                "ranker_score": 0.20,
                "final_score": 0.20,
                "old_novelty_score": 0.25,
                "new_density_score": 1.0,
                "grammar_feature_present": True,
                "token_nll_p95": 0.0,
                "transition_nll_p95": 0.0,
                "longest_unseen_phrase_len": 0.0,
                "rare_phrase_fraction": 0.0,
            },
        ]

        promoted = apply_grammar_score_promotion(
            rows,
            score_variant="grammar_surprisal_mix",
            score_weight=0.15,
            min_quality=0.45,
            min_new_density_score=0.35,
        )
        by_id = {row["sample_id"]: row for row in promoted}

        self.assertLess(by_id["phase-a-already-high"]["grammar_promotion_delta"], 0.08)
        self.assertNotEqual(by_id["phase-a-already-high"]["reason_code"], "RARE_TEMPORAL_COMPOSITION")
        self.assertEqual(by_id["phase-a-already-high"]["reason_code"], "REDUNDANT_KNOWN_WORKFLOW")
        self.assertGreaterEqual(by_id["low-phase-a-grammar"]["grammar_promotion_delta"], 0.08)
        self.assertEqual(by_id["low-phase-a-grammar"]["reason_code"], "RARE_TEMPORAL_COMPOSITION")

    def test_old_novelty_reason_codes_take_priority_over_grammar_reason_code(self):
        rows = [
            {
                "sample_id": "novel-singleton",
                "quality_score": 1.0,
                "ranker_score": 0.20,
                "final_score": 0.20,
                "old_novelty_score": 0.80,
                "new_density_score": 0.44,
                "grammar_feature_present": True,
                "token_nll_p95": 9.0,
                "transition_nll_p95": 7.0,
                "longest_unseen_phrase_len": 6.0,
                "rare_phrase_fraction": 0.9,
            },
            {
                "sample_id": "cohesive-novel",
                "quality_score": 1.0,
                "ranker_score": 0.20,
                "final_score": 0.20,
                "old_novelty_score": 0.80,
                "new_density_score": 1.0,
                "grammar_feature_present": True,
                "token_nll_p95": 9.0,
                "transition_nll_p95": 7.0,
                "longest_unseen_phrase_len": 6.0,
                "rare_phrase_fraction": 0.9,
            },
            {
                "sample_id": "low-grammar-reference",
                "quality_score": 1.0,
                "ranker_score": 0.20,
                "final_score": 0.20,
                "old_novelty_score": 0.25,
                "new_density_score": 1.0,
                "grammar_feature_present": True,
                "token_nll_p95": 0.0,
                "transition_nll_p95": 0.0,
                "longest_unseen_phrase_len": 0.0,
                "rare_phrase_fraction": 0.0,
            },
        ]

        promoted = apply_grammar_score_promotion(
            rows,
            score_variant="grammar_surprisal_mix",
            score_weight=0.35,
            min_quality=0.45,
            min_new_density_score=0.0,
        )
        by_id = {row["sample_id"]: row for row in promoted}

        self.assertGreaterEqual(by_id["novel-singleton"]["grammar_promotion_delta"], 0.08)
        self.assertEqual(by_id["novel-singleton"]["reason_code"], "HIGH_NOVELTY_SINGLETON")
        self.assertGreaterEqual(by_id["cohesive-novel"]["grammar_promotion_delta"], 0.08)
        self.assertEqual(by_id["cohesive-novel"]["reason_code"], "COHESIVE_NEW_WORKFLOW")

    def test_mmr_rank_rows_adds_monotone_rerank_scores(self):
        rows = [
            {"worker_id": "a1", "final_score": 0.95},
            {"worker_id": "a2", "final_score": 0.94},
            {"worker_id": "b1", "final_score": 0.90},
        ]
        embeddings = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])

        ranked = mmr_rank_rows(rows, embeddings, lambda_redundancy=0.20)

        rerank_scores = [row["rerank_score"] for row in ranked]
        self.assertEqual([row["rank"] for row in ranked], [1, 2, 3])
        self.assertTrue(all(left >= right for left, right in zip(rerank_scores, rerank_scores[1:])))
        self.assertEqual(ranked[1]["worker_id"], "b1")

    def test_compute_batch_clusters_groups_near_duplicates(self):
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
                [0.01, 0.99],
                [-1.0, 0.0],
            ]
        )

        cluster_ids = compute_batch_clusters(embeddings, similarity_threshold=0.98)

        self.assertEqual(len(set(cluster_ids.tolist())), 3)
        self.assertEqual(cluster_ids[0], cluster_ids[1])
        self.assertEqual(cluster_ids[2], cluster_ids[3])
        self.assertNotEqual(cluster_ids[0], cluster_ids[4])

    def test_annotate_cluster_features_adds_sizes_and_medoid_distance(self):
        rows = [{"worker_id": "a"}, {"worker_id": "b"}, {"worker_id": "c"}]
        embeddings = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
        cluster_ids = np.array([0, 0, 1])

        annotated = annotate_cluster_features(rows, embeddings, cluster_ids)

        self.assertEqual(annotated[0]["new_cluster_size"], 2)
        self.assertEqual(annotated[2]["new_cluster_size"], 1)
        self.assertTrue(annotated[2]["is_singleton"])
        self.assertIn("distance_to_new_cluster_medoid", annotated[0])

    def test_cluster_aware_rank_rows_takes_cluster_representatives_before_duplicates(self):
        rows = [
            {"worker_id": "a1", "final_score": 0.95, "new_cluster_id": 0},
            {"worker_id": "a2", "final_score": 0.94, "new_cluster_id": 0},
            {"worker_id": "b1", "final_score": 0.80, "new_cluster_id": 1},
            {"worker_id": "c1", "final_score": 0.70, "new_cluster_id": 2},
        ]
        embeddings = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [-1.0, 0.0]])

        ranked = cluster_aware_rank_rows(rows, embeddings, lambda_redundancy=0.20)

        self.assertEqual([row["worker_id"] for row in ranked[:3]], ["a1", "b1", "c1"])
        self.assertEqual(ranked[3]["worker_id"], "a2")
        self.assertTrue(
            all(left >= right for left, right in zip([row["rerank_score"] for row in ranked], [row["rerank_score"] for row in ranked][1:]))
        )
        self.assertTrue(all(row["reranker"] == "cluster_aware" for row in ranked))

    def test_cluster_cap_rank_rows_limits_top_k_cluster_repeats_before_relaxing(self):
        rows = [
            {"worker_id": "a1", "final_score": 0.99, "new_cluster_id": 0},
            {"worker_id": "a2", "final_score": 0.98, "new_cluster_id": 0},
            {"worker_id": "a3", "final_score": 0.97, "new_cluster_id": 0},
            {"worker_id": "b1", "final_score": 0.80, "new_cluster_id": 1},
            {"worker_id": "b2", "final_score": 0.79, "new_cluster_id": 1},
            {"worker_id": "c1", "final_score": 0.70, "new_cluster_id": 2},
        ]
        embeddings = np.eye(6)

        ranked = cluster_cap_rank_rows(
            rows,
            embeddings,
            lambda_redundancy=0.0,
            cluster_bonus_weight=0.0,
            cluster_cap_top_k=5,
            cluster_max_per_cluster=2,
        )

        top5_clusters = [row["new_cluster_id"] for row in ranked[:5]]
        self.assertLessEqual(top5_clusters.count(0), 2)
        self.assertIn(2, top5_clusters)
        self.assertEqual(ranked[5]["worker_id"], "a3")
        self.assertTrue(all(row["reranker"] == "cluster_cap" for row in ranked))
        self.assertTrue(all("cluster_count_before_selection" in row for row in ranked))
        self.assertTrue(all("was_cluster_cap_active" in row for row in ranked[:5]))
        self.assertFalse(ranked[5]["was_cluster_cap_active"])

    def test_cluster_cap_rank_rows_can_cap_by_parent_cluster(self):
        rows = [
            {"worker_id": "a1", "final_score": 0.99, "new_cluster_id": 10, "new_cluster_parent_id": 0},
            {"worker_id": "a2", "final_score": 0.98, "new_cluster_id": 11, "new_cluster_parent_id": 0},
            {"worker_id": "a3", "final_score": 0.97, "new_cluster_id": 12, "new_cluster_parent_id": 0},
            {"worker_id": "b1", "final_score": 0.80, "new_cluster_id": 20, "new_cluster_parent_id": 1},
            {"worker_id": "c1", "final_score": 0.70, "new_cluster_id": 30, "new_cluster_parent_id": 2},
        ]
        embeddings = np.eye(5)

        ranked = cluster_cap_rank_rows(
            rows,
            embeddings,
            lambda_redundancy=0.0,
            cluster_bonus_weight=0.0,
            cluster_cap_top_k=3,
            cluster_max_per_cluster=1,
            cluster_key="new_cluster_parent_id",
        )

        self.assertEqual([row["worker_id"] for row in ranked[:3]], ["a1", "b1", "c1"])
        self.assertEqual(ranked[0]["cluster_cap_key"], "new_cluster_parent_id")
        self.assertEqual(ranked[0]["cluster_cap_cluster_id"], 0)
        self.assertEqual(ranked[1]["cluster_count_before_selection"], 0)
        self.assertTrue(all(row["was_cluster_cap_active"] for row in ranked[:3]))
        self.assertFalse(ranked[3]["was_cluster_cap_active"])

    def test_cluster_cap_quality_floor_does_not_promote_low_quality_singletons(self):
        rows = [
            {"worker_id": "a1", "final_score": 0.99, "quality_score": 0.99, "new_cluster_id": 0},
            {"worker_id": "a2", "final_score": 0.98, "quality_score": 0.99, "new_cluster_id": 0},
            {"worker_id": "bad", "final_score": 0.97, "quality_score": 0.05, "new_cluster_id": 1},
            {"worker_id": "b1", "final_score": 0.80, "quality_score": 0.99, "new_cluster_id": 2},
        ]
        embeddings = np.eye(4)

        ranked = cluster_cap_rank_rows(
            rows,
            embeddings,
            lambda_redundancy=0.0,
            cluster_bonus_weight=0.0,
            cluster_cap_top_k=3,
            cluster_max_per_cluster=1,
            cluster_cap_min_quality=0.45,
        )

        self.assertEqual([row["worker_id"] for row in ranked[:3]], ["a1", "b1", "a2"])
        self.assertEqual(ranked[2]["was_selected_by_fallback"], True)
        self.assertEqual(ranked[2]["cluster_cap_min_quality"], 0.45)
        self.assertEqual(ranked[3]["worker_id"], "bad")

    def test_quality_gated_old_novelty_ranks_passed_rows_by_novelty(self):
        rows = [
            {"sample_id": "lowq-most-novel", "quality_score": 0.20, "old_novelty_score": 0.99, "final_score": 0.99},
            {"sample_id": "passed-most-novel", "quality_score": 0.50, "old_novelty_score": 0.80, "final_score": 0.40},
            {"sample_id": "passed-cleaner", "quality_score": 0.95, "old_novelty_score": 0.70, "final_score": 0.67},
        ]

        ranked = quality_gated_old_novelty_rank_rows(rows, quality_threshold=0.45)

        self.assertEqual([row["sample_id"] for row in ranked], ["passed-most-novel", "passed-cleaner", "lowq-most-novel"])
        self.assertTrue(ranked[0]["quality_gate_pass"])
        self.assertFalse(ranked[-1]["quality_gate_pass"])
        self.assertEqual(ranked[0]["reranker"], "quality_gated_old_novelty")
        self.assertAlmostEqual(ranked[0]["rerank_score"], 0.80)

    def test_quality_gated_old_novelty_can_require_physical_validity(self):
        rows = [
            {
                "sample_id": "stationary-most-novel",
                "quality_score": 0.99,
                "old_novelty_score": 0.99,
                "stationary_fraction": 0.98,
            },
            {
                "sample_id": "moving-novel",
                "quality_score": 0.92,
                "old_novelty_score": 0.80,
                "stationary_fraction": 0.25,
            },
            {
                "sample_id": "low-quality-moving",
                "quality_score": 0.70,
                "old_novelty_score": 1.00,
                "stationary_fraction": 0.20,
            },
        ]

        ranked = quality_gated_old_novelty_rank_rows(
            rows,
            quality_threshold=0.85,
            max_stationary_fraction=0.90,
        )

        self.assertEqual([row["sample_id"] for row in ranked], ["moving-novel", "stationary-most-novel", "low-quality-moving"])
        self.assertTrue(ranked[0]["physical_validity_pass"])
        self.assertFalse(ranked[1]["physical_validity_pass"])
        self.assertIn("stationary_fraction", ranked[1]["physical_validity_failure_reasons"])

    def test_quality_gated_old_novelty_can_reject_large_spikes(self):
        rows = [
            {
                "sample_id": "spiky-most-novel",
                "quality_score": 0.99,
                "old_novelty_score": 0.99,
                "stationary_fraction": 0.05,
                "max_abs_value": 142.0,
            },
            {
                "sample_id": "clean-novel",
                "quality_score": 0.92,
                "old_novelty_score": 0.80,
                "stationary_fraction": 0.25,
                "max_abs_value": 24.0,
            },
        ]

        ranked = quality_gated_old_novelty_rank_rows(
            rows,
            quality_threshold=0.85,
            max_stationary_fraction=0.90,
            max_abs_value=60.0,
        )

        self.assertEqual([row["sample_id"] for row in ranked], ["clean-novel", "spiky-most-novel"])
        self.assertTrue(ranked[0]["physical_validity_pass"])
        self.assertFalse(ranked[1]["physical_validity_pass"])
        self.assertIn("max_abs_value", ranked[1]["physical_validity_failure_reasons"])
        self.assertEqual(ranked[1]["physical_validity_max_abs_value"], 60.0)

    def test_quality_only_rank_rows_uses_novelty_only_as_tie_break(self):
        rows = [
            {"sample_id": "clean-less-novel", "quality_score": 0.90, "old_novelty_score": 0.10},
            {"sample_id": "clean-more-novel", "quality_score": 0.90, "old_novelty_score": 0.80},
            {"sample_id": "lower-quality", "quality_score": 0.80, "old_novelty_score": 1.00},
        ]

        ranked = quality_only_rank_rows(rows)

        self.assertEqual([row["sample_id"] for row in ranked], ["clean-more-novel", "clean-less-novel", "lower-quality"])
        self.assertEqual(ranked[0]["reranker"], "quality_only")
        self.assertAlmostEqual(ranked[0]["rerank_score"], 0.90)

    def test_quality_gated_old_novelty_source_cap_limits_early_repeats(self):
        rows = [
            {"sample_id": "a1", "source_group_id": "worker-a", "quality_score": 0.9, "old_novelty_score": 0.99},
            {"sample_id": "a2", "source_group_id": "worker-a", "quality_score": 0.9, "old_novelty_score": 0.98},
            {"sample_id": "b1", "source_group_id": "worker-b", "quality_score": 0.8, "old_novelty_score": 0.70},
            {"sample_id": "bad", "source_group_id": "worker-c", "quality_score": 0.1, "old_novelty_score": 1.00},
        ]

        ranked = quality_gated_old_novelty_rank_rows(
            rows,
            quality_threshold=0.45,
            source_cap=1,
            source_key="source_group_id",
        )

        self.assertEqual([row["sample_id"] for row in ranked], ["a1", "b1", "a2", "bad"])
        self.assertEqual(ranked[0]["quality_gate_source_cap"], 1)
        self.assertTrue(ranked[0]["was_source_cap_active"])
        self.assertTrue(ranked[2]["was_selected_by_source_cap_fallback"])
        self.assertEqual(ranked[2]["reranker"], "quality_gated_old_novelty_sourcecap")

    def test_quality_gated_old_novelty_source_cap_can_use_cluster_key(self):
        rows = [
            {"sample_id": "a1", "new_cluster_id": 0, "quality_score": 0.9, "old_novelty_score": 0.99},
            {"sample_id": "a2", "new_cluster_id": 0, "quality_score": 0.9, "old_novelty_score": 0.98},
            {"sample_id": "a3", "new_cluster_id": 0, "quality_score": 0.9, "old_novelty_score": 0.97},
            {"sample_id": "b1", "new_cluster_id": 1, "quality_score": 0.9, "old_novelty_score": 0.70},
        ]

        ranked = quality_gated_old_novelty_rank_rows(
            rows,
            quality_threshold=0.85,
            source_cap=2,
            source_key="new_cluster_id",
        )

        self.assertEqual([row["sample_id"] for row in ranked], ["a1", "a2", "b1", "a3"])
        self.assertEqual(ranked[0]["quality_gate_source_key"], "new_cluster_id")
        self.assertEqual(ranked[3]["quality_gate_source_id"], "0")
        self.assertTrue(ranked[3]["was_selected_by_source_cap_fallback"])

    def test_quality_gated_old_novelty_source_cap_can_use_parent_cluster_key(self):
        rows = [
            {"sample_id": "a1", "new_cluster_parent_id": 0, "quality_score": 0.9, "old_novelty_score": 0.99},
            {"sample_id": "a2", "new_cluster_parent_id": 0, "quality_score": 0.9, "old_novelty_score": 0.98},
            {"sample_id": "a3", "new_cluster_parent_id": 0, "quality_score": 0.9, "old_novelty_score": 0.97},
            {"sample_id": "b1", "new_cluster_parent_id": 1, "quality_score": 0.9, "old_novelty_score": 0.70},
        ]

        ranked = quality_gated_old_novelty_rank_rows(
            rows,
            quality_threshold=0.85,
            source_cap=2,
            source_key="new_cluster_parent_id",
        )

        self.assertEqual([row["sample_id"] for row in ranked], ["a1", "a2", "b1", "a3"])
        self.assertEqual(ranked[0]["quality_gate_source_key"], "new_cluster_parent_id")
        self.assertEqual(ranked[3]["quality_gate_source_id"], "0")
        self.assertTrue(ranked[3]["was_selected_by_source_cap_fallback"])

    def test_quality_gated_old_novelty_source_cap_parent_key_falls_back_to_cluster_key(self):
        rows = [
            {"sample_id": "a1", "new_cluster_id": 0, "quality_score": 0.9, "old_novelty_score": 0.99},
            {"sample_id": "a2", "new_cluster_id": 0, "quality_score": 0.9, "old_novelty_score": 0.98},
            {"sample_id": "a3", "new_cluster_id": 0, "quality_score": 0.9, "old_novelty_score": 0.97},
            {"sample_id": "b1", "new_cluster_id": 1, "quality_score": 0.9, "old_novelty_score": 0.70},
        ]

        ranked = quality_gated_old_novelty_rank_rows(
            rows,
            quality_threshold=0.85,
            source_cap=2,
            source_key="new_cluster_parent_id",
        )

        self.assertEqual([row["sample_id"] for row in ranked], ["a1", "a2", "b1", "a3"])
        self.assertEqual(ranked[0]["quality_gate_source_key"], "new_cluster_parent_id")
        self.assertEqual(ranked[3]["quality_gate_source_id"], "0")
        self.assertEqual(ranked[3]["new_cluster_parent_id"], 0)
        self.assertTrue(ranked[3]["was_selected_by_source_cap_fallback"])

    def test_quality_gated_old_novelty_source_cap_rejects_missing_explicit_key(self):
        rows = [
            {"sample_id": "a1", "quality_score": 0.9, "old_novelty_score": 0.99},
        ]

        with self.assertRaisesRegex(ValueError, "source_key missing_key is missing"):
            quality_gated_old_novelty_rank_rows(
                rows,
                quality_threshold=0.85,
                source_cap=2,
                source_key="missing_key",
            )

    def test_tiered_cluster_cap_relaxes_duplicate_limit_after_first_cutoff(self):
        rows = [
            {"worker_id": "a1", "final_score": 0.99, "quality_score": 0.99, "new_cluster_id": 0},
            {"worker_id": "a2", "final_score": 0.98, "quality_score": 0.99, "new_cluster_id": 0},
            {"worker_id": "a3", "final_score": 0.97, "quality_score": 0.99, "new_cluster_id": 0},
            {"worker_id": "b1", "final_score": 0.80, "quality_score": 0.99, "new_cluster_id": 1},
            {"worker_id": "c1", "final_score": 0.70, "quality_score": 0.99, "new_cluster_id": 2},
            {"worker_id": "d1", "final_score": 0.60, "quality_score": 0.99, "new_cluster_id": 3},
        ]
        embeddings = np.eye(6)

        ranked = tiered_cluster_cap_rank_rows(
            rows,
            embeddings,
            cap_schedule=[{"top_k": 3, "max_per_cluster": 1}, {"top_k": 5, "max_per_cluster": 2}],
            lambda_redundancy=0.0,
            cluster_bonus_weight=0.0,
            cluster_cap_min_quality=0.45,
        )

        self.assertEqual([row["worker_id"] for row in ranked[:3]], ["a1", "b1", "c1"])
        self.assertEqual(ranked[3]["worker_id"], "a2")
        self.assertEqual(ranked[3]["cluster_count_before_selection"], 1)
        self.assertEqual(ranked[3]["cluster_cap_top_k"], 5)
        self.assertEqual(ranked[3]["cluster_max_per_cluster"], 2)
        self.assertEqual(ranked[4]["worker_id"], "d1")
        self.assertFalse(any(row["was_selected_by_fallback"] for row in ranked[:5]))
        self.assertTrue(all(row["reranker"] == "tiered_cluster_cap" for row in ranked))

    def test_parent_prefix_cluster_cap_uses_parent_prefix_then_child_fill(self):
        rows = [
            {"worker_id": "a1", "final_score": 0.99, "quality_score": 0.99, "new_cluster_id": 10, "new_cluster_parent_id": 0},
            {"worker_id": "a2", "final_score": 0.98, "quality_score": 0.99, "new_cluster_id": 11, "new_cluster_parent_id": 0},
            {"worker_id": "b1", "final_score": 0.80, "quality_score": 0.99, "new_cluster_id": 20, "new_cluster_parent_id": 1},
            {"worker_id": "c1", "final_score": 0.70, "quality_score": 0.99, "new_cluster_id": 30, "new_cluster_parent_id": 2},
        ]
        embeddings = np.eye(4)

        ranked = parent_prefix_cluster_cap_rank_rows(
            rows,
            embeddings,
            prefix_top_k=3,
            prefix_cluster_key="new_cluster_parent_id",
            prefix_max_per_cluster=1,
            fill_cluster_key="new_cluster_id",
            fill_max_per_cluster=2,
            cluster_cap_top_k=4,
            lambda_redundancy=0.0,
            cluster_bonus_weight=0.0,
            cluster_cap_min_quality=0.45,
        )

        self.assertEqual([row["worker_id"] for row in ranked], ["a1", "b1", "c1", "a2"])
        self.assertTrue(all(row["reranker"] == "parent_prefix_cluster_cap" for row in ranked))
        self.assertTrue(all(row["hybrid_prefix_selected"] for row in ranked[:3]))
        self.assertFalse(ranked[3]["hybrid_prefix_selected"])
        self.assertEqual(ranked[0]["hybrid_prefix_cluster_key"], "new_cluster_parent_id")
        self.assertEqual(ranked[3]["hybrid_fill_cluster_key"], "new_cluster_id")

    def test_selection_trace_rows_exposes_cluster_cap_fallback_state(self):
        ranked = [
            {
                "worker_id": "a",
                "sample_id": "a",
                "rank": 129,
                "phase_a_final_score": 0.8,
                "final_score": 0.9,
                "rerank_score": 0.7,
                "new_cluster_id": 3,
                "new_cluster_parent_id": 1,
                "new_cluster_size": 1500,
                "cluster_cap_key": "new_cluster_parent_id",
                "cluster_cap_cluster_id": 1,
                "cluster_cap_min_quality": 0.45,
                "cluster_count_before_selection": 4,
                "was_cluster_cap_active": False,
                "was_selected_by_fallback": True,
                "eligible_under_cluster_cap_count": 0,
                "cluster_cap_top_k": 200,
                "cluster_max_per_cluster": 4,
                "quality_score": 0.99,
                "reason_code": "RARE_TEMPORAL_COMPOSITION",
            }
        ]

        trace = _selection_trace_rows(ranked)

        self.assertEqual(trace[0]["rank"], 129)
        self.assertEqual(trace[0]["cluster_count_before_selection"], 4)
        self.assertTrue(trace[0]["was_selected_by_fallback"])
        self.assertEqual(trace[0]["cluster_cap_top_k"], 200)
        self.assertEqual(trace[0]["cluster_cap_key"], "new_cluster_parent_id")
        self.assertEqual(trace[0]["cluster_cap_cluster_id"], 1)
        self.assertEqual(trace[0]["cluster_cap_min_quality"], 0.45)

    def test_baseline_ranking_config_uses_pretrain_support_and_val_query(self):
        config = load_ranking_config("configs/baseline_ranking.json")

        validate_ranking_config(config)

        self.assertEqual(config["execution"]["provider"], "modal")
        self.assertEqual(config["splits"]["support_split"], "pretrain")
        self.assertEqual(config["splits"]["query_split"], "val")
        self.assertEqual(config["ranking"]["representation"], "window_mean_std_pool")
        self.assertEqual(config["ranking"]["reranker_method"], "cluster_aware")
        self.assertNotIn("checkpoint", config)

    def test_baseline_ranking_with_grammar_config_is_diagnostics_only(self):
        config = load_ranking_config("configs/baseline_ranking_with_grammar.json")

        validate_ranking_config(config)

        self.assertTrue(config["grammar_features"]["enabled"])
        self.assertFalse(config["grammar_features"]["use_in_score"])
        self.assertIn("{mode}", config["grammar_features"]["path_template"])

    def test_baseline_ranking_promoted_grammar_config_is_guarded(self):
        config = load_ranking_config("configs/baseline_ranking_grammar_promoted.json")

        validate_ranking_config(config)

        grammar = config["grammar_features"]
        self.assertTrue(grammar["enabled"])
        self.assertTrue(grammar["use_in_score"])
        self.assertEqual(grammar["score_variant"], "grammar_surprisal_mix")
        self.assertLessEqual(grammar["score_weight"], 0.35)
        self.assertIn("/artifacts/eval/grammar_ablation/", grammar["ablation_report_path"])

    def test_submission_ranking_config_allows_pretrain_support_and_new_query_with_expected_count(self):
        config = load_ranking_config("configs/baseline_ranking_new_quality_gated_grammar.json")

        validate_ranking_config(config)

        self.assertEqual(config["splits"]["support_split"], "pretrain")
        self.assertEqual(config["splits"]["query_split"], "new")
        self.assertEqual(config["splits"]["negative_split"], "pretrain")
        self.assertEqual(config["ranking"]["run_mode"], "submission")
        self.assertEqual(config["ranking"]["reranker_method"], "cluster_cap")
        self.assertEqual(config["ranking"]["cluster_cap_top_k"], 200)
        self.assertEqual(config["ranking"]["cluster_max_per_cluster"], 8)
        self.assertEqual(config["ranking"]["cluster_bonus_weight"], 0.0)
        self.assertEqual(config["ranking"]["mmr_lambda"], 0.0)
        self.assertEqual(config["ranking"]["min_query_samples"], 2000)
        self.assertEqual(config["grammar_features"]["score_variant"], "quality_gated_grammar")
        self.assertIn("extra_path_templates", config["grammar_features"])
        self.assertTrue(any(template.endswith("grammar_features_pretrain_full.csv") for template in config["grammar_features"]["extra_path_templates"]))
        self.assertTrue(config["large_cluster_split"]["enabled"])
        self.assertTrue(config["corruption_eval"]["enabled"])

    def test_submission_ranking_config_supports_quality_gated_old_novelty_reset(self):
        config = load_ranking_config("configs/baseline_ranking_new_qgate_oldnovelty_knn5.json")

        validate_ranking_config(config)

        self.assertEqual(config["ranking"]["reranker_method"], "quality_gated_old_novelty")
        self.assertEqual(config["ranking"]["k_old"], 5)
        self.assertEqual(config["ranking"]["novelty_weight"], 1.0)
        self.assertFalse(config["grammar_features"]["enabled"])
        self.assertFalse(config["large_cluster_split"]["enabled"])
        self.assertFalse(config["score_guards"]["stationary_singleton"]["enabled"])

    def test_submission_ranking_config_supports_quality_only_control(self):
        config = load_ranking_config("configs/baseline_ranking_new_quality_only.json")

        validate_ranking_config(config)

        self.assertEqual(config["ranking"]["reranker_method"], "quality_only")
        self.assertEqual(config["ranking"]["run_mode"], "submission")
        self.assertFalse(config["grammar_features"]["enabled"])
        self.assertFalse(config["large_cluster_split"]["enabled"])
        self.assertFalse(config["score_guards"]["stationary_singleton"]["enabled"])

    def test_parent_cap_submission_config_caps_by_real_parent_cluster(self):
        config = load_ranking_config("configs/baseline_ranking_new_quality_gated_grammar_worker_coverage_w03_parentcap.json")

        validate_ranking_config(config)

        self.assertEqual(config["ranking"]["cluster_cap_key"], "new_cluster_parent_id")
        self.assertEqual(config["ranking"]["cluster_cap_min_quality"], 0.45)
        self.assertEqual(config["ranking"]["cluster_max_per_cluster"], 8)
        self.assertTrue(config["large_cluster_split"]["enabled"])
        self.assertIn("parentcap", config["artifacts"]["output_dir"])

    def test_hybrid_submission_config_uses_parent_prefix_then_child_cap(self):
        config = load_ranking_config("configs/baseline_ranking_new_quality_gated_grammar_worker_coverage_w03_hybrid75.json")

        validate_ranking_config(config)

        self.assertEqual(config["ranking"]["reranker_method"], "parent_prefix_cluster_cap")
        self.assertEqual(config["ranking"]["prefix_cluster_cap_top_k"], 75)
        self.assertEqual(config["ranking"]["prefix_cluster_cap_key"], "new_cluster_parent_id")
        self.assertEqual(config["ranking"]["cluster_cap_key"], "new_cluster_id")
        self.assertEqual(config["ranking"]["cluster_cap_min_quality"], 0.45)
        self.assertIn("hybrid75", config["artifacts"]["output_dir"])

    def test_physical_source_hardened_config_uses_feature_subclusters_and_raw_corruption(self):
        config = load_ranking_config(
            "configs/baseline_ranking_new_quality_gated_grammar_physical_source_hybrid75_feature_subcluster_rawcorr.json"
        )

        validate_ranking_config(config)

        self.assertEqual(config["data"]["pretrain_manifest"], "cache/manifests/pretrain_physical_source_urls.txt")
        self.assertEqual(config["large_cluster_split"]["method"], "feature_kmeans")
        self.assertEqual(config["large_cluster_split"]["score_feature_weight"], 0.0)
        self.assertTrue(config["corruption_eval"]["raw_signal"])
        self.assertIn("feature_subcluster_rawcorr", config["artifacts"]["output_dir"])

    def test_stationary_guard_config_uses_narrow_singleton_guard(self):
        config = load_ranking_config(
            "configs/baseline_ranking_new_quality_gated_grammar_physical_source_hybrid75_feature_subcluster_rawcorr_stationary_guard.json"
        )

        validate_ranking_config(config)

        guard = config["score_guards"]["stationary_singleton"]
        self.assertTrue(guard["enabled"])
        self.assertEqual(guard["stationary_threshold"], 0.90)
        self.assertEqual(guard["max_new_density_score"], 0.35)
        self.assertEqual(guard["min_grammar_score"], 0.85)
        self.assertEqual(guard["penalty_multiplier"], 0.35)
        self.assertIn("stationary_guard", config["artifacts"]["output_dir"])

    def test_parentcap200_config_extends_parent_cap_through_top_200(self):
        config = load_ranking_config(
            "configs/baseline_ranking_new_quality_gated_grammar_physical_source_hybrid75_feature_subcluster_rawcorr_stationary_guard_parentcap200_p8.json"
        )

        validate_ranking_config(config)

        ranking = config["ranking"]
        self.assertEqual(ranking["reranker_method"], "parent_prefix_cluster_cap")
        self.assertEqual(ranking["prefix_cluster_cap_top_k"], 200)
        self.assertEqual(ranking["prefix_cluster_cap_key"], "new_cluster_parent_id")
        self.assertEqual(ranking["prefix_cluster_max_per_cluster"], 8)
        self.assertEqual(ranking["cluster_cap_top_k"], 200)
        self.assertEqual(ranking["cluster_cap_key"], "new_cluster_id")
        self.assertTrue(config["score_guards"]["stationary_singleton"]["enabled"])
        self.assertTrue(config["large_cluster_split"]["enabled"])
        self.assertTrue(config["corruption_eval"]["raw_signal"])
        self.assertIn("parentcap200_p8", config["artifacts"]["output_dir"])

    def test_tiered_childcap_config_caps_subclusters_at_top50_and_top200(self):
        config = load_ranking_config(
            "configs/baseline_ranking_new_quality_gated_grammar_physical_source_hybrid75_feature_subcluster_rawcorr_stationary_guard_tiered_childcap2_5.json"
        )

        validate_ranking_config(config)

        ranking = config["ranking"]
        self.assertEqual(ranking["reranker_method"], "tiered_cluster_cap")
        self.assertEqual(ranking["cluster_cap_key"], "new_cluster_id")
        self.assertEqual(
            ranking["cluster_cap_schedule"],
            [{"top_k": 50, "max_per_cluster": 2}, {"top_k": 200, "max_per_cluster": 5}],
        )
        self.assertTrue(config["score_guards"]["stationary_singleton"]["enabled"])
        self.assertTrue(config["large_cluster_split"]["enabled"])
        self.assertTrue(config["corruption_eval"]["raw_signal"])
        self.assertIn("tiered_childcap2_5", config["artifacts"]["output_dir"])

    def test_tiered_childcap_subcluster40_config_increases_subcluster_capacity(self):
        config = load_ranking_config(
            "configs/baseline_ranking_new_quality_gated_grammar_physical_source_hybrid75_feature_subcluster_rawcorr_stationary_guard_tiered_childcap2_5_subcluster40.json"
        )

        validate_ranking_config(config)

        self.assertEqual(config["ranking"]["reranker_method"], "tiered_cluster_cap")
        self.assertEqual(config["ranking"]["cluster_cap_schedule"][1]["max_per_cluster"], 5)
        self.assertEqual(config["large_cluster_split"]["target_subcluster_size"], 40)
        self.assertIn("subcluster40", config["artifacts"]["output_dir"])

    def test_raw_shape_and_temporal_qgate_configs_are_submission_capable(self):
        raw_config = load_ranking_config("configs/baseline_ranking_new_qgate_oldnovelty_raw_shape_knn5.json")
        raw_q85_config = load_ranking_config("configs/baseline_ranking_new_qgate_oldnovelty_raw_shape_q85_knn5.json")
        temporal_config = load_ranking_config("configs/baseline_ranking_new_qgate_oldnovelty_temporal_knn5.json")

        validate_ranking_config(raw_config)
        validate_ranking_config(raw_q85_config)
        validate_ranking_config(temporal_config)

        self.assertEqual(raw_config["ranking"]["representation"], "raw_shape_stats")
        self.assertEqual(raw_q85_config["ranking"]["quality_gate_threshold"], 0.85)
        self.assertEqual(temporal_config["ranking"]["representation"], "temporal_order")
        self.assertEqual(raw_config["ranking"]["reranker_method"], "quality_gated_old_novelty")
        self.assertEqual(temporal_config["ranking"]["reranker_method"], "quality_gated_old_novelty")

    def test_raw_shape_q85_physical_validity_config_is_submission_capable(self):
        config = load_ranking_config("configs/baseline_ranking_new_qgate_oldnovelty_raw_shape_q85_stat90_knn5.json")
        spike_config = load_ranking_config(
            "configs/baseline_ranking_new_qgate_oldnovelty_raw_shape_q85_stat90_abs60_knn5.json"
        )
        cap_config = load_ranking_config(
            "configs/baseline_ranking_new_qgate_oldnovelty_raw_shape_q85_stat90_abs60_clustercap2_knn5.json"
        )
        parent_cap_config = load_ranking_config(
            "configs/baseline_ranking_new_qgate_oldnovelty_raw_shape_q85_stat90_abs60_parentcap2_knn5.json"
        )

        validate_ranking_config(config)
        validate_ranking_config(spike_config)
        validate_ranking_config(cap_config)
        validate_ranking_config(parent_cap_config)

        self.assertEqual(config["ranking"]["representation"], "raw_shape_stats")
        self.assertEqual(config["ranking"]["quality_gate_threshold"], 0.85)
        self.assertEqual(config["ranking"]["max_stationary_fraction"], 0.90)
        self.assertIn("stat90", config["artifacts"]["output_dir"])
        self.assertEqual(spike_config["ranking"]["max_abs_value"], 60.0)
        self.assertIn("abs60", spike_config["artifacts"]["output_dir"])
        self.assertEqual(cap_config["ranking"]["reranker_method"], "quality_gated_old_novelty_sourcecap")
        self.assertEqual(cap_config["ranking"]["source_cap"], 2)
        self.assertEqual(cap_config["ranking"]["source_cap_key"], "new_cluster_id")
        self.assertIn("clustercap2", cap_config["artifacts"]["output_dir"])
        self.assertEqual(parent_cap_config["ranking"]["reranker_method"], "quality_gated_old_novelty_sourcecap")
        self.assertEqual(parent_cap_config["ranking"]["source_cap"], 2)
        self.assertEqual(parent_cap_config["ranking"]["source_cap_key"], "new_cluster_parent_id")
        self.assertIn("parentcap2", parent_cap_config["artifacts"]["output_dir"])

    def test_baseline_ranking_config_rejects_bad_stationary_guard_threshold(self):
        config = load_ranking_config(
            "configs/baseline_ranking_new_quality_gated_grammar_physical_source_hybrid75_feature_subcluster_rawcorr_stationary_guard.json"
        )
        config["score_guards"]["stationary_singleton"]["stationary_threshold"] = 1.5

        with self.assertRaises(ValueError):
            validate_ranking_config(config)

    def test_learned_ranking_config_points_at_prefit_model_artifact(self):
        config = load_ranking_config("configs/baseline_ranking_learned.json")

        validate_ranking_config(config)

        self.assertTrue(config["learned_ranker"]["enabled"])
        self.assertTrue(config["learned_ranker"]["model_path"].startswith("/artifacts/"))
        self.assertEqual(config["learned_ranker"]["score_transform"], "sigmoid")
        self.assertEqual(config["splits"]["query_split"], "val")

    def test_encoder_artifact_ranking_config_uses_precomputed_pretrain_val_embeddings(self):
        config = load_ranking_config("configs/baseline_ranking_encoder_anticollapse.json")

        validate_ranking_config(config)

        self.assertEqual(config["ranking"]["representation"], "encoder_artifact")
        self.assertEqual(config["splits"]["support_split"], "pretrain")
        self.assertEqual(config["splits"]["query_split"], "val")
        self.assertTrue(config["encoder_embeddings"]["support_embeddings"].endswith(".npy"))

    def test_submission_rows_preserve_rank_order_when_cap_rerank_breaks_score_sorting(self):
        rows = [
            {"worker_id": "a", "rank": 1, "rerank_score": 0.5, "final_score": 0.5, "quality_score": 1.0},
            {"worker_id": "b", "rank": 2, "rerank_score": 0.9, "final_score": 0.9, "quality_score": 1.0},
            {"worker_id": "c", "rank": 3, "rerank_score": 0.4, "final_score": 0.4, "quality_score": 1.0},
        ]

        submission = _submission_rows(rows)
        scores = [float(row["score"]) for row in submission]

        self.assertTrue(all(left > right for left, right in zip(scores, scores[1:])))
        self.assertEqual([row["rank"] for row in submission], [1, 2, 3])

    def test_baseline_ranking_config_rejects_support_query_leakage(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_ranking.json"
            path.write_text(
                json.dumps(
                    {
                        "execution": {"provider": "modal"},
                        "data": {"root": "/data", "feature_dim": 75},
                        "artifacts": {"root": "/artifacts"},
                        "splits": {"support_split": "val", "query_split": "val"},
                        "ranking": {"representation": "window_mean_std_pool", "k_old": 10},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(RankingLeakageError):
                validate_ranking_config(load_ranking_config(path))

    def test_baseline_ranking_config_rejects_unbounded_grammar_score_promotion(self):
        config = load_ranking_config("configs/baseline_ranking_with_grammar.json")
        config["grammar_features"]["use_in_score"] = True
        config["grammar_features"]["score_variant"] = "grammar_surprisal_mix"
        config["grammar_features"]["score_weight"] = 0.75
        config["grammar_features"]["min_quality"] = 0.45
        config["grammar_features"]["min_new_density_score"] = 0.35
        config["grammar_features"]["ablation_report_path"] = "/artifacts/eval/grammar_ablation/grammar_leave_cluster_ablation_report_full.json"

        with self.assertRaises(ValueError):
            validate_ranking_config(config)

    def test_baseline_ranking_config_rejects_unknown_corruption_mode(self):
        config = load_ranking_config("configs/baseline_ranking.json")
        config["corruption_eval"]["modes"] = ["flatline", "unknown"]

        with self.assertRaises(ValueError):
            validate_ranking_config(config)

    def test_baseline_ranking_config_rejects_bad_learned_ranker_path(self):
        config = load_ranking_config("configs/baseline_ranking_learned.json")
        config["learned_ranker"]["model_path"] = "relative-model.json"

        with self.assertRaises(ValueError):
            validate_ranking_config(config)

    def test_quality_gated_grammar_score_variant_replaces_phase_a_score(self):
        rows = [
            {
                "sample_id": "a",
                "quality_score": 1.0,
                "ranker_score": 0.1,
                "final_score": 0.1,
                "new_density_score": 0.0,
                "token_nll_p95": 10.0,
                "transition_nll_p95": 9.0,
                "longest_unseen_phrase_len": 4.0,
                "grammar_feature_present": True,
            },
            {
                "sample_id": "b",
                "quality_score": 1.0,
                "ranker_score": 0.9,
                "final_score": 0.9,
                "new_density_score": 1.0,
                "token_nll_p95": 1.0,
                "transition_nll_p95": 1.0,
                "longest_unseen_phrase_len": 1.0,
                "grammar_feature_present": True,
            },
        ]

        promoted = apply_grammar_score_promotion(
            rows,
            score_variant="quality_gated_grammar",
            score_weight=1.0,
            min_quality=0.45,
            min_new_density_score=0.0,
        )

        self.assertGreater(promoted[0]["final_score"], promoted[1]["final_score"])
        self.assertEqual(promoted[0]["grammar_score_variant"], "quality_gated_grammar")
        self.assertTrue(promoted[0]["grammar_promotion_applied"])

    def test_load_grammar_features_reads_mode_template_csv(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "grammar_features_val_smoke.csv"
            path.write_text(
                "sample_id,split,token_nll_mean,token_nll_p95,rare_phrase_fraction\n"
                "worker-a,val,1.5,2.5,0.25\n",
                encoding="utf-8",
            )
            config = {
                "grammar_features": {
                    "enabled": True,
                    "path_template": str(Path(tmp) / "grammar_features_val_{mode}.csv"),
                }
            }

            features, source_path = _load_grammar_features(config, mode="smoke")

            self.assertEqual(source_path, path)
            self.assertEqual(set(features), {"worker-a"})
            self.assertEqual(features["worker-a"]["token_nll_mean"], 1.5)
            self.assertEqual(features["worker-a"]["grammar_feature_split"], "val")
            self.assertEqual(features["worker-a"]["grammar_feature_source_path"], str(path))

    def test_load_grammar_features_reads_extra_templates_for_candidate_negatives(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary = root / "grammar_features_new_smoke.csv"
            extra = root / "grammar_features_pretrain_smoke.csv"
            static_extra = root / "grammar_features_pretrain_full.csv"
            primary.write_text(
                "sample_id,split,token_nll_mean,token_nll_p95,transition_nll_p95,rare_phrase_fraction\n"
                "new-a,new,3.0,4.0,5.0,0.8\n",
                encoding="utf-8",
            )
            extra.write_text(
                "sample_id,split,token_nll_mean,token_nll_p95,transition_nll_p95,rare_phrase_fraction\n"
                "pretrain-a,pretrain,1.0,2.0,3.0,0.1\n",
                encoding="utf-8",
            )
            static_extra.write_text(
                "sample_id,split,token_nll_mean,token_nll_p95,transition_nll_p95,rare_phrase_fraction\n"
                "pretrain-a,pretrain,1.0,2.0,3.0,0.1\n"
                "pretrain-b,pretrain,1.2,2.2,3.2,0.2\n",
                encoding="utf-8",
            )
            config = {
                "grammar_features": {
                    "enabled": True,
                    "path_template": str(root / "grammar_features_new_{mode}.csv"),
                    "extra_path_templates": [
                        str(root / "grammar_features_pretrain_{mode}.csv"),
                        str(root / "grammar_features_pretrain_full.csv"),
                    ],
                }
            }

            features, source_path = _load_grammar_features(config, mode="smoke")

            self.assertEqual(source_path, primary)
            self.assertEqual(set(features), {"new-a", "pretrain-a", "pretrain-b"})
            self.assertEqual(features["new-a"]["grammar_feature_split"], "new")
            self.assertEqual(features["pretrain-a"]["grammar_feature_split"], "pretrain")
            self.assertEqual(features["pretrain-b"]["grammar_feature_split"], "pretrain")

    def test_join_grammar_features_adds_diagnostics_without_changing_scores(self):
        rows = [
            {"sample_id": "worker-a", "final_score": 0.2, "ranker_score": 0.3},
            {"sample_id": "worker-b", "final_score": 0.8, "ranker_score": 0.9},
        ]
        features = {
            "worker-a": {
                "token_nll_mean": 1.5,
                "token_nll_p95": 2.5,
                "rare_phrase_fraction": 0.25,
                "grammar_feature_split": "val",
            }
        }

        joined = _join_grammar_features(rows, features)

        self.assertEqual([row["final_score"] for row in joined], [0.2, 0.8])
        self.assertEqual([row["ranker_score"] for row in joined], [0.3, 0.9])
        self.assertTrue(joined[0]["grammar_feature_present"])
        self.assertFalse(joined[1]["grammar_feature_present"])
        self.assertEqual(joined[0]["token_nll_mean"], 1.5)
        self.assertEqual(joined[0]["grammar_feature_split"], "val")
        for column in GRAMMAR_FEATURE_COLUMNS:
            self.assertIn(column, joined[1])

    def test_join_grammar_features_inherits_source_features_for_corruptions(self):
        rows = [
            {"sample_id": "new-a__corrupt_flatline_0000", "source_sample_id": "new-a", "final_score": 0.05},
            {"sample_id": "pretrain-a", "final_score": 0.8},
        ]
        features = {
            "new-a": {
                "token_nll_p95": 9.0,
                "transition_nll_p95": 7.0,
                "grammar_feature_split": "new",
            },
            "pretrain-a": {
                "token_nll_p95": 2.0,
                "transition_nll_p95": 1.0,
                "grammar_feature_split": "pretrain",
            },
        }

        joined = _join_grammar_features(rows, features)

        self.assertTrue(joined[0]["grammar_feature_present"])
        self.assertEqual(joined[0]["grammar_feature_lookup_key"], "new-a")
        self.assertEqual(joined[0]["token_nll_p95"], 9.0)
        self.assertEqual(joined[1]["grammar_feature_lookup_key"], "pretrain-a")

    def test_candidate_grammar_promotion_requires_feature_coverage_for_both_labels(self):
        rows = [
            {"sample_id": "val-a", "label": 1, "grammar_feature_present": True},
            {"sample_id": "val-b", "label": 1, "grammar_feature_present": True},
            {"sample_id": "pretrain-a", "label": 0, "grammar_feature_present": False},
        ]

        allowed, reason = _candidate_grammar_promotion_allowed(rows)

        self.assertFalse(allowed)
        self.assertEqual(reason, "missing_grammar_features_for_label_0")

    def test_candidate_grammar_promotion_allows_symmetric_new_pretrain_and_corruption_features(self):
        rows = [
            {"sample_id": "new-a", "label": 1, "grammar_feature_present": True},
            {"sample_id": "pretrain-a", "label": 0, "grammar_feature_present": True},
            {"sample_id": "new-a__corrupt_flatline_0000", "label": 0, "grammar_feature_present": True},
        ]

        allowed, reason = _candidate_grammar_promotion_allowed(rows)

        self.assertTrue(allowed)
        self.assertEqual(reason, "all_labels_have_grammar_features")

    def test_split_large_clusters_deterministically_splits_mega_cluster_and_recomputes_sizes(self):
        rows = [
            {
                "sample_id": f"mega-{idx:03d}",
                "worker_id": f"mega-{idx:03d}",
                "new_cluster_id": 7,
                "new_cluster_size": 10,
                "grammar_score": 1.0 - idx * 0.01,
                "old_novelty_score": 0.8,
                "new_density_score": 0.9,
                "final_score": 0.9 - idx * 0.01,
            }
            for idx in range(10)
        ]
        rows.extend(
            [
                {
                    "sample_id": "small-a",
                    "worker_id": "small-a",
                    "new_cluster_id": 8,
                    "new_cluster_size": 1,
                    "final_score": 0.1,
                }
            ]
        )

        split_rows, summary = split_large_clusters(rows, max_cluster_size=4, target_subcluster_size=3)

        mega_rows = [row for row in split_rows if row["sample_id"].startswith("mega-")]
        mega_clusters = {row["new_cluster_id"] for row in mega_rows}
        self.assertEqual(summary["n_split_parent_clusters"], 1)
        self.assertGreater(len(mega_clusters), 1)
        self.assertLessEqual(max(row["new_cluster_size"] for row in mega_rows), 3)
        self.assertTrue(all(row["new_cluster_parent_id"] == 7 for row in mega_rows))
        self.assertTrue(all(row["large_cluster_split_applied"] for row in mega_rows))
        self.assertFalse([row for row in split_rows if row["sample_id"] == "small-a"][0]["large_cluster_split_applied"])

    def test_split_large_clusters_uses_feature_space_not_score_round_robin(self):
        rows = []
        embeddings = []
        for idx in range(8):
            rows.append(
                {
                    "sample_id": f"left-{idx}",
                    "worker_id": f"left-{idx}",
                    "new_cluster_id": 3,
                    "new_cluster_size": 16,
                    "final_score": 1.0 if idx % 2 == 0 else 0.1,
                }
            )
            embeddings.append([1.0, 0.02 * idx])
        for idx in range(8):
            rows.append(
                {
                    "sample_id": f"right-{idx}",
                    "worker_id": f"right-{idx}",
                    "new_cluster_id": 3,
                    "new_cluster_size": 16,
                    "final_score": 1.0 if idx % 2 else 0.1,
                }
            )
            embeddings.append([-1.0, 0.02 * idx])

        split_rows, summary = split_large_clusters(
            rows,
            np.asarray(embeddings, dtype=float),
            max_cluster_size=4,
            target_subcluster_size=8,
            kmeans_iterations=12,
        )

        by_id = {row["sample_id"]: row for row in split_rows}
        left_clusters = {by_id[f"left-{idx}"]["new_cluster_id"] for idx in range(8)}
        right_clusters = {by_id[f"right-{idx}"]["new_cluster_id"] for idx in range(8)}
        self.assertEqual(len(left_clusters), 1)
        self.assertEqual(len(right_clusters), 1)
        self.assertTrue(left_clusters.isdisjoint(right_clusters))
        self.assertEqual(summary["feature_split_parent_clusters"], 1)
        self.assertEqual(summary["fallback_split_parent_clusters"], 0)
        self.assertTrue(summary["embedding_features_used"])

    def test_maybe_split_large_clusters_reports_disabled_and_applied_states(self):
        rows = [
            {"sample_id": f"a{idx}", "new_cluster_id": 0, "new_cluster_size": 5, "final_score": 1.0 - idx * 0.1}
            for idx in range(5)
        ]

        unchanged, disabled = _maybe_split_large_clusters({"large_cluster_split": {"enabled": False}}, rows, label="query")
        split_rows, applied = _maybe_split_large_clusters(
            {"large_cluster_split": {"enabled": True, "max_cluster_size": 2, "target_subcluster_size": 2}},
            rows,
            label="query",
        )

        self.assertEqual(unchanged, rows)
        self.assertFalse(disabled["applied"])
        self.assertTrue(applied["applied"])
        self.assertGreater(len({row["new_cluster_id"] for row in split_rows}), 1)

    def test_modal_rank_uses_remote_baseline_job(self):
        source = Path("modal_rank.py").read_text(encoding="utf-8")

        self.assertIn("remote_baseline_rank.remote", source)
        self.assertIn("run_baseline_ranking", source)
        self.assertIn("baseline_ranking.json", source)

    def test_ranker_core_does_not_import_submission_pandas_path(self):
        source = Path("marginal_value/ranking/baseline_ranker.py").read_text(encoding="utf-8")

        self.assertNotIn("marginal_value.submit", source)


def _make_clean_imu(n_samples: int, sample_rate: int = 30) -> np.ndarray:
    t = np.arange(n_samples, dtype=float) / sample_rate
    return np.column_stack(
        [
            np.sin(2 * np.pi * 0.7 * t),
            0.5 * np.cos(2 * np.pi * 0.7 * t),
            9.81 + 0.1 * np.sin(2 * np.pi * 1.3 * t),
            0.03 * np.cos(2 * np.pi * 0.2 * t),
            0.04 * np.sin(2 * np.pi * 0.4 * t),
            0.02 * np.cos(2 * np.pi * 0.3 * t),
        ]
    )


def _write_modal_jsonl(path: Path, samples: np.ndarray, sample_rate: int = 30) -> None:
    lines = []
    for idx, row in enumerate(np.asarray(samples, dtype=float)):
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
