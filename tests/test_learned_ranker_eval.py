import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.models.learned_linear_ranker import load_linear_ranker_model
from marginal_value.eval.learned_ranker_eval import (
    FeatureLeakageError,
    build_feature_table,
    fit_linear_ranker,
    make_fold_holdout_splits,
    make_stratified_hash_split,
    run_learned_ranker_eval,
    score_linear_ranker,
)


class LearnedRankerEvalTests(unittest.TestCase):
    def test_feature_table_excludes_forbidden_and_asymmetric_grammar_features(self):
        rows = [
            _candidate_row("pos-a", 1, 0.9, grammar_present=True, token_nll_p95=8.0),
            _candidate_row("pos-b", 1, 0.8, grammar_present=True, token_nll_p95=7.0),
            _candidate_row("neg-a", 0, 0.2, grammar_present=False, token_nll_p95=0.0),
            _candidate_row("neg-b", 0, 0.1, grammar_present=False, token_nll_p95=0.0),
        ]

        table = build_feature_table(
            rows,
            requested_features=["quality_score", "old_knn_distance", "token_nll_p95", "rank"],
            strict_feature_coverage=False,
        )

        self.assertEqual(table.feature_names, ["quality_score", "old_knn_distance"])
        self.assertIn("token_nll_p95", table.excluded_features)
        self.assertEqual(table.excluded_features["token_nll_p95"], "asymmetric_grammar_feature_coverage")
        self.assertIn("rank", table.excluded_features)
        self.assertEqual(table.excluded_features["rank"], "forbidden_leakage_column")
        self.assertEqual(table.values.shape, (4, 2))

    def test_feature_table_can_reject_asymmetric_features_in_strict_mode(self):
        rows = [
            _candidate_row("pos-a", 1, 0.9, grammar_present=True, token_nll_p95=8.0),
            _candidate_row("neg-a", 0, 0.1, grammar_present=False, token_nll_p95=0.0),
        ]

        with self.assertRaises(FeatureLeakageError):
            build_feature_table(
                rows,
                requested_features=["old_knn_distance", "token_nll_p95"],
                strict_feature_coverage=True,
            )

    def test_feature_table_accepts_symmetric_grammar_features_without_presence_flag(self):
        rows = [
            _leave_cluster_row("sample-a", 1, token_nll_p95=9.0),
            _leave_cluster_row("sample-b", 1, token_nll_p95=8.5),
            _leave_cluster_row("sample-c", 0, token_nll_p95=7.0),
            _leave_cluster_row("sample-d", 0, token_nll_p95=7.5),
        ]

        table = build_feature_table(
            rows,
            requested_features=["quality_score", "token_nll_p95", "rare_phrase_fraction", "n_primitives"],
            strict_feature_coverage=False,
        )

        self.assertEqual(table.feature_names, ["quality_score", "token_nll_p95", "rare_phrase_fraction", "n_primitives"])
        self.assertNotIn("token_nll_p95", table.excluded_features)

    def test_feature_table_excludes_leave_cluster_label_helper_columns(self):
        rows = [
            _leave_cluster_row("sample-a", 1, token_nll_p95=9.0),
            _leave_cluster_row("sample-b", 0, token_nll_p95=7.0),
        ]

        table = build_feature_table(
            rows,
            requested_features=["token_nll_p95", "old_novelty_score", "source_cluster", "heldout_cluster", "fold_index"],
            strict_feature_coverage=False,
        )

        self.assertEqual(table.feature_names, ["token_nll_p95"])
        self.assertEqual(table.excluded_features["old_novelty_score"], "forbidden_leakage_column")
        self.assertEqual(table.excluded_features["source_cluster"], "forbidden_leakage_column")
        self.assertEqual(table.excluded_features["heldout_cluster"], "forbidden_leakage_column")
        self.assertEqual(table.excluded_features["fold_index"], "forbidden_leakage_column")

    def test_feature_table_excludes_motion_phrase_oracle_helper_columns(self):
        rows = [
            _motion_phrase_row("fold1-pos-a", 1, token_nll_p95=9.0, is_artifact=0, is_redundant=0),
            _motion_phrase_row("fold1-neg-a", 0, token_nll_p95=6.0, is_artifact=1, is_redundant=0),
            _motion_phrase_row("fold2-pos-a", 1, token_nll_p95=8.8, is_artifact=0, is_redundant=0),
            _motion_phrase_row("fold2-neg-a", 0, token_nll_p95=6.2, is_artifact=0, is_redundant=1),
        ]

        table = build_feature_table(
            rows,
            requested_features=[
                "quality_score",
                "token_nll_p95",
                "is_artifact",
                "is_redundant",
                "negative_type",
                "duplicate_group_id",
                "heldout_phrase_id",
                "reason_code",
                "fold_index",
            ],
            strict_feature_coverage=False,
        )

        self.assertEqual(table.feature_names, ["quality_score", "token_nll_p95"])
        for feature in (
            "is_artifact",
            "is_redundant",
            "negative_type",
            "duplicate_group_id",
            "heldout_phrase_id",
            "reason_code",
            "fold_index",
        ):
            self.assertEqual(table.excluded_features[feature], "forbidden_leakage_column")

    def test_stratified_hash_split_has_no_sample_overlap_and_preserves_labels(self):
        sample_ids = [f"pos-{idx}" for idx in range(6)] + [f"neg-{idx}" for idx in range(6)]
        labels = np.array([1] * 6 + [0] * 6)

        split = make_stratified_hash_split(sample_ids, labels, eval_fraction=0.34, seed=7)

        train_ids = {sample_ids[idx] for idx in split.train_indices}
        eval_ids = {sample_ids[idx] for idx in split.eval_indices}
        self.assertFalse(train_ids & eval_ids)
        self.assertEqual(set(labels[split.train_indices].tolist()), {0, 1})
        self.assertEqual(set(labels[split.eval_indices].tolist()), {0, 1})
        self.assertEqual(split.leakage_audit["sample_overlap_count"], 0)

    def test_hash_split_keeps_repeated_sample_ids_in_one_side(self):
        sample_ids = ["shared-a", "shared-a", "shared-b", "shared-b", "pos-c", "neg-d"]
        labels = np.array([1, 0, 1, 0, 1, 0])

        split = make_stratified_hash_split(sample_ids, labels, eval_fraction=0.5, seed=3)

        train_ids = {sample_ids[idx] for idx in split.train_indices}
        eval_ids = {sample_ids[idx] for idx in split.eval_indices}
        self.assertFalse(train_ids & eval_ids)
        self.assertEqual(split.leakage_audit["sample_overlap_count"], 0)

    def test_fold_holdout_splits_exclude_eval_fold_and_eval_samples_from_train(self):
        rows = [
            {"sample_id": "a", "fold_index": 1, "label": 1},
            {"sample_id": "b", "fold_index": 1, "label": 0},
            {"sample_id": "a", "fold_index": 2, "label": 0},
            {"sample_id": "c", "fold_index": 2, "label": 1},
            {"sample_id": "d", "fold_index": 2, "label": 0},
            {"sample_id": "e", "fold_index": 3, "label": 1},
            {"sample_id": "f", "fold_index": 3, "label": 0},
        ]
        sample_ids = [row["sample_id"] for row in rows]
        labels = np.array([row["label"] for row in rows])

        splits = make_fold_holdout_splits(
            rows,
            sample_ids,
            labels,
            fold_column="fold_index",
            exclude_eval_samples_from_train=True,
        )

        self.assertEqual({split["fold"] for split in splits}, {"1", "2", "3"})
        for split in splits:
            train_ids = {sample_ids[idx] for idx in split["train_indices"]}
            eval_ids = {sample_ids[idx] for idx in split["eval_indices"]}
            train_folds = {str(rows[idx]["fold_index"]) for idx in split["train_indices"]}
            self.assertFalse(train_ids & eval_ids)
            self.assertNotIn(split["fold"], train_folds)

    def test_linear_ranker_learns_from_train_split_and_scores_eval_split(self):
        train_x = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.8], [0.2, 0.7]])
        train_y = np.array([1, 1, 0, 0])
        eval_x = np.array([[0.85, 0.2], [0.15, 0.75]])

        model = fit_linear_ranker(train_x, train_y)
        scores = score_linear_ranker(model, eval_x)

        self.assertGreater(scores[0], scores[1])
        self.assertEqual(model.feature_mean.shape, (2,))
        self.assertEqual(model.weights.shape, (2,))

    def test_run_learned_ranker_eval_writes_safe_feature_report(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_path = root / "candidates.csv"
            output_dir = root / "out"
            rows = [
                _candidate_row(f"pos-{idx}", 1, 0.80 + idx * 0.01, grammar_present=True, token_nll_p95=8.0)
                for idx in range(8)
            ] + [
                _candidate_row(f"neg-{idx}", 0, 0.10 + idx * 0.01, grammar_present=False, token_nll_p95=0.0)
                for idx in range(8)
            ]
            _write_rows(candidate_path, rows)
            config = {
                "execution": {"provider": "modal"},
                "artifacts": {
                    "candidate_path": str(candidate_path),
                    "output_dir": str(output_dir),
                },
                "features": {
                    "requested": ["quality_score", "old_knn_distance", "token_nll_p95"],
                    "strict_feature_coverage": False,
                },
                "eval": {
                    "eval_fraction": 0.25,
                    "seed": 11,
                    "k_values": [2, 4],
                },
            }

            result = run_learned_ranker_eval(config, allow_local_execution=True)

            self.assertEqual(result["n_rows"], 16)
            self.assertTrue(Path(result["report_path"]).exists())
            self.assertTrue(Path(result["scores_path"]).exists())
            self.assertTrue(Path(result["model_path"]).exists())
            _model, feature_names, metadata = load_linear_ranker_model(result["model_path"])
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(report["feature_table"]["feature_names"], ["quality_score", "old_knn_distance"])
            self.assertEqual(report["feature_table"]["excluded_features"]["token_nll_p95"], "asymmetric_grammar_feature_coverage")
            self.assertEqual(report["leakage_audit"]["sample_overlap_count"], 0)
            self.assertEqual(feature_names, ["quality_score", "old_knn_distance"])
            self.assertTrue(metadata["final_model_fit_uses_all_rows"])
            self.assertEqual(report["model"]["model_path"], result["model_path"])
            self.assertGreaterEqual(
                report["variants"]["learned_linear"]["metrics"]["ndcg@4"],
                report["variants"]["phase_a_final_score"]["metrics"]["ndcg@4"],
            )

    def test_run_learned_ranker_eval_can_use_fold_holdout_mode(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_path = root / "lco.csv"
            output_dir = root / "out"
            rows = [
                _leave_cluster_row(f"fold1-pos-{idx}", 1, token_nll_p95=9.0 + idx * 0.1) | {"fold_index": 1}
                for idx in range(4)
            ] + [
                _leave_cluster_row(f"fold1-neg-{idx}", 0, token_nll_p95=6.0 + idx * 0.1) | {"fold_index": 1}
                for idx in range(4)
            ] + [
                _leave_cluster_row(f"fold2-pos-{idx}", 1, token_nll_p95=9.0 + idx * 0.1) | {"fold_index": 2}
                for idx in range(4)
            ] + [
                _leave_cluster_row(f"fold2-neg-{idx}", 0, token_nll_p95=6.0 + idx * 0.1) | {"fold_index": 2}
                for idx in range(4)
            ]
            _write_rows(candidate_path, rows)
            config = {
                "execution": {"provider": "modal"},
                "artifacts": {
                    "candidate_path": str(candidate_path),
                    "output_dir": str(output_dir),
                },
                "features": {
                    "requested": ["token_nll_p95", "rare_phrase_fraction", "fold_index"],
                    "strict_feature_coverage": False,
                },
                "eval": {
                    "fold_column": "fold_index",
                    "exclude_eval_samples_from_train": True,
                    "k_values": [2, 4],
                    "baseline_score_columns": ["token_nll_p95"],
                },
            }

            result = run_learned_ranker_eval(config, allow_local_execution=True)

            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(report["split"]["method"], "fold_holdout")
            self.assertEqual(report["leakage_audit"]["fold_overlap_count"], 0)
            self.assertEqual(report["leakage_audit"]["sample_overlap_count"], 0)
            self.assertEqual(report["feature_table"]["excluded_features"]["fold_index"], "forbidden_leakage_column")
            self.assertEqual(report["variants"]["learned_linear"]["metrics"]["precision@4"], 1.0)


def _candidate_row(
    sample_id: str,
    label: int,
    old_knn_distance: float,
    *,
    grammar_present: bool,
    token_nll_p95: float,
) -> dict[str, object]:
    quality = 0.95 if label else 0.90
    return {
        "sample_id": sample_id,
        "worker_id": sample_id,
        "label": label,
        "split": "val" if label else "pretrain",
        "rank": 1,
        "quality_score": quality,
        "old_knn_distance": old_knn_distance,
        "old_novelty_score": old_knn_distance,
        "new_batch_density": 0.7 if label else 0.4,
        "new_density_score": 0.7 if label else 0.4,
        "new_cluster_size": 2 if label else 4,
        "is_singleton": False,
        "distance_to_new_cluster_medoid": 0.05 if label else 0.10,
        "phase_a_final_score": 1.0 - old_knn_distance,
        "final_score": 1.0 - old_knn_distance,
        "grammar_feature_present": grammar_present,
        "token_nll_p95": token_nll_p95,
        "transition_nll_p95": token_nll_p95 / 2.0,
        "rare_phrase_fraction": 0.5 if grammar_present else 0.0,
        "new_cluster_id": label,
    }


def _leave_cluster_row(sample_id: str, label: int, *, token_nll_p95: float) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "worker_id": sample_id,
        "label": label,
        "split": "pretrain",
        "fold_index": 1,
        "heldout_cluster": 3,
        "source_cluster": 3 if label else 1,
        "new_cluster_id": 3 if label else 1,
        "quality_score": 1.0,
        "old_novelty_score": 1.0 if label else 0.0,
        "new_density_score": 0.0,
        "final_score": 0.0,
        "n_primitives": 20 if label else 12,
        "token_nll_mean": token_nll_p95 - 0.5,
        "token_nll_p95": token_nll_p95,
        "transition_nll_p95": token_nll_p95 / 2.0,
        "rare_phrase_fraction": 1.0 if label else 0.5,
        "longest_unseen_phrase_len": 5.0 if label else 3.0,
    }


def _motion_phrase_row(
    sample_id: str,
    label: int,
    *,
    token_nll_p95: float,
    is_artifact: int,
    is_redundant: int,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "worker_id": sample_id,
        "label": label,
        "split": "pretrain",
        "fold_index": 1 if sample_id.startswith("fold1") else 2,
        "heldout_phrase_id": "abc123",
        "negative_type": "artifact" if is_artifact else "redundancy" if is_redundant else "",
        "duplicate_group_id": sample_id if is_redundant else "",
        "reason_code": "LIKELY_SENSOR_ARTIFACT" if is_artifact else "RARE_TEMPORAL_COMPOSITION",
        "quality_score": 0.2 if is_artifact else 1.0,
        "is_artifact": is_artifact,
        "is_redundant": is_redundant,
        "n_primitives": 20,
        "token_nll_mean": token_nll_p95 - 0.5,
        "token_nll_p95": token_nll_p95,
        "transition_nll_p95": token_nll_p95 / 2.0,
        "rare_phrase_fraction": 1.0 if label else 0.5,
        "longest_unseen_phrase_len": 5.0 if label else 3.0,
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
