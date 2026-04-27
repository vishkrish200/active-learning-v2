import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.ranking.audit_submission import build_ranking_model_card, write_ranking_model_card


class AuditSubmissionTests(unittest.TestCase):
    def test_model_card_reports_topk_diversity_quality_and_leakage_checks(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            submission_path = root / "submission.csv"
            diagnostics_path = root / "diagnostics.csv"
            candidate_path = root / "candidates.csv"
            report_path = root / "report.json"
            config_path = root / "config.json"
            _write_rows(
                submission_path,
                [
                    {"worker_id": "a", "rank": 1, "score": 0.9, "quality_score": 0.99, "reason_code": "RARE_TEMPORAL_COMPOSITION"},
                    {"worker_id": "b", "rank": 2, "score": 0.8, "quality_score": 0.97, "reason_code": "COHESIVE_NEW_WORKFLOW"},
                    {"worker_id": "c", "rank": 3, "score": 0.7, "quality_score": 0.20, "reason_code": "LOW_QUALITY"},
                ],
            )
            _write_rows(
                diagnostics_path,
                [
                    _diagnostic("a", 1, 0, 0.99, 0.9, 0.7, 0.8, False, "RARE_TEMPORAL_COMPOSITION"),
                    _diagnostic("b", 2, 1, 0.97, 0.8, 0.6, 0.9, False, "COHESIVE_NEW_WORKFLOW", parent_cluster=10),
                    _diagnostic("c", 3, 1, 0.20, 0.1, 0.2, 0.3, True, "LOW_QUALITY", parent_cluster=10),
                ],
            )
            _write_rows(
                candidate_path,
                [
                    {
                        "sample_id": "a",
                        "worker_id": "a",
                        "rank": 1,
                        "split": "new",
                        "label": 1,
                        "new_cluster_id": 0,
                        "quality_score": 0.99,
                        "grammar_feature_present": True,
                        "token_nll_p95": 5.0,
                    },
                    {
                        "sample_id": "n",
                        "worker_id": "n",
                        "rank": 2,
                        "split": "pretrain",
                        "label": 0,
                        "new_cluster_id": 1,
                        "quality_score": 0.97,
                        "grammar_feature_present": False,
                        "token_nll_p95": 0.0,
                    },
                ],
            )
            report_path.write_text(json.dumps({"support_split": "pretrain", "query_split": "new", "metrics": {"ndcg@100": 0.7}}), encoding="utf-8")
            config_path.write_text(
                json.dumps(
                    {
                        "splits": {"support_split": "pretrain", "query_split": "new"},
                        "ranking": {"representation": "window_mean_std_pool", "reranker_method": "cluster_cap"},
                        "grammar_features": {"enabled": True, "score_variant": "quality_gated_grammar", "use_in_score": True},
                    }
                ),
                encoding="utf-8",
            )

            card = build_ranking_model_card(
                submission_path=submission_path,
                diagnostics_path=diagnostics_path,
                candidate_path=candidate_path,
                run_report_path=report_path,
                config_path=config_path,
                top_ks=(2, 3),
                run_name="unit",
            )

            self.assertEqual(card["run_name"], "unit")
            self.assertEqual(card["top_k"]["2"]["unique_clusters"], 2)
            self.assertEqual(card["top_k"]["3"]["largest_cluster_count"], 2)
            self.assertEqual(card["top_k"]["3"]["unique_parent_clusters"], 2)
            self.assertEqual(card["top_k"]["3"]["parent_largest_cluster_count"], 2)
            self.assertGreater(card["top_k"]["3"]["cluster_gini"], 0.0)
            self.assertEqual(card["top_k"]["3"]["corruption_negative_count"], 1)
            self.assertAlmostEqual(card["top_k"]["2"]["mean_grammar_score"], 0.85)
            self.assertFalse(card["leakage_checks"]["worker_overlap"])
            self.assertTrue(card["leakage_checks"]["candidate_eval_score_leakage_risk"])
            self.assertEqual(card["run_config"]["grammar_variant"], "quality_gated_grammar")
            self.assertEqual(card["source_run_metrics"]["ndcg@100"], 0.7)

            output_path = write_ranking_model_card(card, root / "model_card.json")
            self.assertTrue(output_path.exists())


def _diagnostic(
    worker_id: str,
    rank: int,
    cluster: int,
    quality: float,
    grammar: float,
    novelty: float,
    support: float,
    is_corruption: bool,
    reason: str,
    parent_cluster: int | None = None,
) -> dict[str, object]:
    row = {
        "worker_id": worker_id,
        "sample_id": worker_id,
        "rank": rank,
        "final_score": grammar * quality,
        "rerank_score": grammar,
        "quality_score": quality,
        "new_cluster_id": cluster,
        "grammar_score": grammar,
        "old_novelty_score": novelty,
        "new_density_score": support,
        "is_corruption": str(is_corruption),
        "reason_code": reason,
    }
    if parent_cluster is not None:
        row["new_cluster_parent_id"] = parent_cluster
        row["large_cluster_split_applied"] = "True"
    return row


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
