import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.eval.score_calibration_eval import (
    evaluate_score_calibration_dataset,
    run_score_calibration_eval,
    validate_score_calibration_config,
)


class ScoreCalibrationEvalTests(unittest.TestCase):
    def test_quality_gated_grammar_downranks_artifact_without_using_oracle_flags(self):
        rows = [
            _row("artifact", label=0, quality=0.20, token_nll=10.0, transition_nll=9.5),
            _row("positive", label=1, quality=1.00, token_nll=8.0, transition_nll=7.5),
            _row("ordinary", label=0, quality=1.00, token_nll=3.0, transition_nll=2.5),
        ]
        dataset = {
            "name": "phrase_holdout",
            "cluster_column": "new_cluster_id",
            "forbidden_score_features": ["is_artifact", "is_redundant", "negative_type", "label"],
        }

        report = evaluate_score_calibration_dataset(
            rows,
            dataset_config=dataset,
            k_values=[1, 2],
            low_quality_threshold=0.45,
        )

        raw = report["variants"]["grammar_surprisal_mix"]["metrics"]
        gated = report["variants"]["quality_gated_grammar"]["metrics"]
        self.assertEqual(raw["precision@1"], 0.0)
        self.assertEqual(gated["precision@1"], 1.0)
        self.assertEqual(gated["artifact_rate@1"], 0.0)
        self.assertNotIn("is_artifact", report["variants"]["quality_gated_grammar"]["used_features"])
        self.assertFalse(report["leakage_audit"]["leaking_variant_features"])

    def test_cluster_aware_variant_spreads_top_k_without_reusing_label_columns(self):
        rows = [
            _row("a1", label=1, final_score=0.95, cluster_id=0, old_novelty=0.95, support=0.90),
            _row("a2", label=1, final_score=0.94, cluster_id=0, old_novelty=0.94, support=0.88),
            _row("b1", label=1, final_score=0.80, cluster_id=1, old_novelty=0.80, support=0.80),
            _row("c1", label=0, final_score=0.20, cluster_id=2, old_novelty=0.20, support=0.20),
        ]
        dataset = {
            "name": "baseline",
            "cluster_column": "new_cluster_id",
            "allow_cluster_aware_variants": True,
        }

        report = evaluate_score_calibration_dataset(
            rows,
            dataset_config=dataset,
            k_values=[2],
            low_quality_threshold=0.45,
        )

        raw = report["variants"]["current_final_score"]["metrics"]
        cluster_aware = report["variants"]["current_final_score_cluster_aware"]["metrics"]
        self.assertEqual(raw["unique_cluster_count@2"], 1)
        self.assertEqual(raw["cluster_repeat_count@2"], 1)
        self.assertEqual(cluster_aware["unique_cluster_count@2"], 2)
        self.assertEqual(cluster_aware["cluster_repeat_count@2"], 0)
        self.assertNotIn("label", report["variants"]["current_final_score_cluster_aware"]["used_features"])

    def test_run_score_calibration_eval_writes_report_and_summary(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_path = root / "baseline.csv"
            output_dir = root / "out"
            _write_rows(
                candidate_path,
                [
                    _row("artifact", label=0, quality=0.20, token_nll=10.0, transition_nll=9.5),
                    _row("positive", label=1, quality=1.00, token_nll=8.0, transition_nll=7.5),
                    _row("ordinary", label=0, quality=1.00, token_nll=3.0, transition_nll=2.5),
                ],
            )
            config = {
                "execution": {"provider": "modal"},
                "artifacts": {
                    "output_dir": str(output_dir),
                    "datasets": [
                        {
                            "name": "unit",
                            "path": str(candidate_path),
                            "cluster_column": "new_cluster_id",
                            "forbidden_score_features": ["is_artifact", "label"],
                        }
                    ],
                },
                "eval": {
                    "k_values": [1, 2],
                    "low_quality_threshold": 0.45,
                },
            }

            result = run_score_calibration_eval(config, allow_local_execution=True)

            self.assertEqual(result["dataset_count"], 1)
            self.assertTrue(Path(result["report_path"]).exists())
            self.assertTrue(Path(result["summary_path"]).exists())
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            self.assertIn("unit", report["datasets"])
            self.assertIn("quality_gated_grammar", report["datasets"]["unit"]["variants"])
            self.assertIn("quality_gated_grammar", Path(result["summary_path"]).read_text(encoding="utf-8"))

    def test_aggregate_reports_best_common_variant_not_single_dataset_outlier(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_path = root / "baseline.csv"
            phrase_path = root / "phrase.csv"
            output_dir = root / "out"
            _write_rows(
                baseline_path,
                [
                    _row("b-pos", label=1, final_score=1.0, token_nll=8.0, transition_nll=7.0),
                    _row("b-neg", label=0, final_score=0.0, token_nll=3.0, transition_nll=2.0),
                ],
            )
            _write_rows(
                phrase_path,
                [
                    _row("p-artifact", label=0, quality=0.2, token_nll=10.0, transition_nll=9.0),
                    _row("p-pos", label=1, quality=1.0, token_nll=8.0, transition_nll=7.0),
                    _row("p-neg", label=0, quality=1.0, token_nll=2.0, transition_nll=1.0),
                ],
            )
            config = {
                "execution": {"provider": "modal"},
                "artifacts": {
                    "output_dir": str(output_dir),
                    "datasets": [
                        {"name": "baseline", "path": str(baseline_path), "cluster_column": "new_cluster_id"},
                        {"name": "phrase", "path": str(phrase_path), "cluster_column": "new_cluster_id"},
                    ],
                },
                "eval": {"k_values": [1, 2], "primary_k": 1},
            }

            result = run_score_calibration_eval(config, allow_local_execution=True)

            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(report["aggregate"]["best_common_variant"], "quality_gated_grammar")
            self.assertEqual(report["aggregate"]["best_common_dataset_count"], 2)
            self.assertEqual(result["best_common_variant"], "quality_gated_grammar")

    def test_validate_score_calibration_config_requires_modal_and_absolute_artifacts(self):
        valid_config = {
            "execution": {"provider": "modal"},
            "artifacts": {
                "output_dir": "/artifacts/eval/score_calibration",
                "datasets": [{"name": "baseline", "path": "/artifacts/ranking/candidates.csv"}],
            },
        }
        validate_score_calibration_config(valid_config)

        with self.assertRaises(ValueError):
            validate_score_calibration_config(valid_config | {"execution": {"provider": "local"}})

        with self.assertRaises(ValueError):
            validate_score_calibration_config(
                {
                    "execution": {"provider": "modal"},
                    "artifacts": {
                        "output_dir": "/artifacts/eval/score_calibration",
                        "datasets": [{"name": "bad", "path": "relative.csv"}],
                    },
                }
            )


def _row(
    sample_id: str,
    *,
    label: int,
    quality: float = 1.0,
    token_nll: float = 5.0,
    transition_nll: float = 5.0,
    final_score: float = 0.0,
    cluster_id: int = 0,
    old_novelty: float = 0.5,
    support: float = 0.5,
    redundancy: float = 0.0,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "worker_id": sample_id,
        "label": label,
        "quality_score": quality,
        "final_score": final_score,
        "old_novelty_score": old_novelty,
        "new_density_score": support,
        "new_batch_density": support,
        "new_cluster_id": cluster_id,
        "token_nll_p95": token_nll,
        "transition_nll_p95": transition_nll,
        "longest_unseen_phrase_len": token_nll / 2.0,
        "rare_phrase_fraction": 1.0 if label else 0.0,
        "redundancy_penalty": redundancy,
        "is_artifact": 1 if quality < 0.45 else 0,
        "is_redundant": 0,
        "negative_type": "artifact" if quality < 0.45 else "ordinary" if not label else "",
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
