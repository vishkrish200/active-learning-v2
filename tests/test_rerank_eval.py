import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.eval.rerank_eval import (
    evaluate_rerank_variants,
    run_rerank_eval,
    validate_rerank_eval_config,
)


class RerankEvalTests(unittest.TestCase):
    def test_rerank_eval_compares_raw_sort_to_existing_rerank_and_tracks_cluster_repeats(self):
        rows = [
            _candidate_row("a1", label=1, final_score=0.95, rank=1, cluster_id=0),
            _candidate_row("a2", label=0, final_score=0.94, rank=3, cluster_id=0),
            _candidate_row("b1", label=1, final_score=0.80, rank=2, cluster_id=1),
            _candidate_row("c1", label=0, final_score=0.70, rank=4, cluster_id=2, quality_score=0.30),
        ]

        report = evaluate_rerank_variants(rows, k_values=[2, 3], low_quality_threshold=0.45)

        raw = report["variants"]["raw_final_score"]["metrics"]
        reranked = report["variants"]["existing_rerank"]["metrics"]
        self.assertEqual(raw["unique_cluster_count@2"], 1)
        self.assertEqual(raw["cluster_repeat_count@2"], 1)
        self.assertEqual(reranked["unique_cluster_count@2"], 2)
        self.assertEqual(reranked["cluster_repeat_count@2"], 0)
        self.assertEqual(raw["precision@2"], 0.5)
        self.assertEqual(reranked["precision@2"], 1.0)
        self.assertLess(reranked["score_retention_vs_raw@2"], 1.0)
        self.assertEqual(report["best_by_cluster_repeat_at_2"], "existing_rerank")

    def test_run_rerank_eval_writes_report_and_summary(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_path = root / "candidates.csv"
            output_dir = root / "out"
            rows = [
                _candidate_row("a1", label=1, final_score=0.95, rank=1, cluster_id=0),
                _candidate_row("a2", label=0, final_score=0.94, rank=3, cluster_id=0),
                _candidate_row("b1", label=1, final_score=0.80, rank=2, cluster_id=1),
                _candidate_row("c1", label=0, final_score=0.70, rank=4, cluster_id=2),
            ]
            _write_rows(candidate_path, rows)
            config = {
                "execution": {"provider": "modal"},
                "artifacts": {
                    "candidate_path": str(candidate_path),
                    "output_dir": str(output_dir),
                },
                "eval": {
                    "k_values": [2, 3],
                    "low_quality_threshold": 0.45,
                },
            }

            result = run_rerank_eval(config, allow_local_execution=True)

            self.assertEqual(result["n_rows"], 4)
            self.assertTrue(Path(result["report_path"]).exists())
            self.assertTrue(Path(result["summary_path"]).exists())
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            self.assertIn("raw_final_score", report["variants"])
            self.assertIn("existing_rerank", report["variants"])
            summary = Path(result["summary_path"]).read_text(encoding="utf-8")
            self.assertIn("raw_final_score", summary)
            self.assertIn("existing_rerank", summary)

    def test_validate_rerank_eval_config_requires_modal_and_absolute_artifacts(self):
        valid_config = {
            "execution": {"provider": "modal"},
            "artifacts": {
                "candidate_path": "/artifacts/ranking/candidates.csv",
                "output_dir": "/artifacts/eval/rerank",
            },
        }
        validate_rerank_eval_config(valid_config)

        with self.assertRaises(ValueError):
            validate_rerank_eval_config(
                valid_config | {"execution": {"provider": "local"}},
                allow_local_execution=False,
            )

        with self.assertRaises(ValueError):
            validate_rerank_eval_config(
                {
                    "execution": {"provider": "modal"},
                    "artifacts": {
                        "candidate_path": "relative.csv",
                        "output_dir": "/artifacts/eval/rerank",
                    },
                }
            )


def _candidate_row(
    worker_id: str,
    *,
    label: int,
    final_score: float,
    rank: int,
    cluster_id: int,
    quality_score: float = 0.95,
) -> dict[str, object]:
    return {
        "worker_id": worker_id,
        "sample_id": worker_id,
        "label": label,
        "final_score": final_score,
        "rank": rank,
        "rerank_score": 1.0 / rank,
        "new_cluster_id": cluster_id,
        "new_cluster_size": 2 if cluster_id == 0 else 1,
        "quality_score": quality_score,
        "reason_code": "COHESIVE_NEW_WORKFLOW" if label else "REDUNDANT_KNOWN_WORKFLOW",
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
