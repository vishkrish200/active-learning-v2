import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.eval.ranking_audit import audit_ranking_artifacts, write_audit_artifacts


class RankingAuditTests(unittest.TestCase):
    def test_audit_summarizes_top_quality_reason_counts_and_low_quality_examples(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            submission_path = root / "submission.csv"
            diagnostics_path = root / "diagnostics.csv"
            candidate_path = root / "candidates.csv"
            quality_path = root / "quality.csv"
            _write_csv(
                submission_path,
                ["worker_id", "rank", "score", "quality_score", "reason_code"],
                [
                    ["a", 1, 0.90, 0.98, "COHESIVE_NEW_WORKFLOW"],
                    ["b", 2, 0.80, 0.20, "LOW_QUALITY"],
                    ["c", 3, 0.70, 0.92, "RARE_MOTION_PRIMITIVES"],
                ],
            )
            _write_csv(
                diagnostics_path,
                ["worker_id", "rank", "final_score", "quality_score", "reason_code", "new_cluster_id"],
                [
                    ["a", 1, 0.90, 0.98, "COHESIVE_NEW_WORKFLOW", 0],
                    ["b", 2, 0.80, 0.20, "LOW_QUALITY", 1],
                    ["c", 3, 0.70, 0.92, "RARE_MOTION_PRIMITIVES", 1],
                ],
            )
            _write_csv(
                candidate_path,
                ["worker_id", "rank", "label", "quality_score", "reason_code", "new_cluster_id"],
                [
                    ["a", 1, 1, 0.98, "COHESIVE_NEW_WORKFLOW", 0],
                    ["b", 2, 0, 0.20, "LOW_QUALITY", 1],
                    ["c", 3, 1, 0.92, "RARE_MOTION_PRIMITIVES", 1],
                ],
            )
            _write_csv(
                quality_path,
                ["sample_id", "quality_score", "spike_rate"],
                [["a", 0.98, 0.0], ["b", 0.20, 0.1], ["c", 0.92, 0.0]],
            )

            report = audit_ranking_artifacts(
                submission_path=submission_path,
                diagnostics_path=diagnostics_path,
                candidate_path=candidate_path,
                quality_metadata_path=quality_path,
                top_ks=(2,),
            )

            self.assertTrue(report["submission"]["rank_contiguous"])
            self.assertTrue(report["submission"]["score_nonincreasing"])
            self.assertEqual(report["top_k"]["2"]["low_quality_count"], 1)
            self.assertEqual(report["top_k"]["2"]["unique_cluster_count"], 2)
            self.assertEqual(report["top_k"]["2"]["largest_cluster_count"], 1)
            self.assertEqual(report["top_k"]["2"]["unique_parent_cluster_count"], 2)
            self.assertEqual(report["top_k"]["2"]["parent_largest_cluster_count"], 1)
            self.assertIn("cluster_gini", report["top_k"]["2"])
            self.assertEqual(report["top_k"]["2"]["reason_code_counts"]["LOW_QUALITY"], 1)
            self.assertEqual(report["candidate_eval"]["top_k"]["2"]["positive_count"], 1)
            self.assertEqual(report["candidate_eval"]["top_k"]["2"]["unique_cluster_count"], 2)
            self.assertEqual(report["candidate_eval"]["top_k"]["2"]["unique_parent_cluster_count"], 2)
            self.assertIn("largest_cluster_fraction", report["candidate_eval"]["top_k"]["2"])
            self.assertEqual(report["low_quality_examples"][0]["worker_id"], "b")

    def test_audit_reports_parent_cluster_diversity_after_large_cluster_split(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            submission_path = root / "submission.csv"
            diagnostics_path = root / "diagnostics.csv"
            quality_path = root / "quality.csv"
            _write_csv(
                submission_path,
                ["worker_id", "rank", "score", "quality_score", "reason_code"],
                [
                    ["a", 1, 0.90, 0.98, "RARE_TEMPORAL_COMPOSITION"],
                    ["b", 2, 0.80, 0.97, "RARE_TEMPORAL_COMPOSITION"],
                    ["c", 3, 0.70, 0.96, "RARE_TEMPORAL_COMPOSITION"],
                ],
            )
            _write_csv(
                diagnostics_path,
                [
                    "worker_id",
                    "rank",
                    "final_score",
                    "quality_score",
                    "reason_code",
                    "new_cluster_id",
                    "new_cluster_parent_id",
                    "large_cluster_split_applied",
                ],
                [
                    ["a", 1, 0.90, 0.98, "RARE_TEMPORAL_COMPOSITION", 10, 0, "True"],
                    ["b", 2, 0.80, 0.97, "RARE_TEMPORAL_COMPOSITION", 11, 0, "True"],
                    ["c", 3, 0.70, 0.96, "RARE_TEMPORAL_COMPOSITION", 12, 0, "True"],
                ],
            )
            _write_csv(quality_path, ["sample_id", "quality_score", "spike_rate"], [["a", 0.98, 0.0]])

            report = audit_ranking_artifacts(
                submission_path=submission_path,
                diagnostics_path=diagnostics_path,
                quality_metadata_path=quality_path,
                top_ks=(3,),
            )

            top3 = report["top_k"]["3"]
            self.assertEqual(top3["unique_cluster_count"], 3)
            self.assertEqual(top3["unique_parent_cluster_count"], 1)
            self.assertEqual(top3["large_cluster_split_count"], 3)
            self.assertEqual(top3["parent_largest_cluster_count"], 3)

    def test_write_audit_artifacts_creates_json_and_review_csvs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = {
                "submission": {"n_rows": 1},
                "top_examples": [{"worker_id": "a", "rank": 1}],
                "low_quality_examples": [{"worker_id": "b", "rank": 5}],
                "reason_code_counts": {"LOW_QUALITY": 1},
            }

            paths = write_audit_artifacts(report, root, suffix="unit")

            self.assertTrue(paths["report"].exists())
            self.assertTrue(paths["top_examples"].exists())
            self.assertTrue(paths["low_quality_examples"].exists())
            self.assertEqual(json.loads(paths["report"].read_text())["submission"]["n_rows"], 1)

    def test_audit_summarizes_grammar_diagnostics_when_present(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            submission_path = root / "submission.csv"
            diagnostics_path = root / "diagnostics.csv"
            quality_path = root / "quality.csv"
            _write_csv(
                submission_path,
                ["worker_id", "rank", "score", "quality_score", "reason_code"],
                [
                    ["a", 1, 0.90, 0.98, "COHESIVE_NEW_WORKFLOW"],
                    ["b", 2, 0.80, 0.30, "LOW_QUALITY"],
                    ["c", 3, 0.70, 0.95, "RARE_MOTION_PRIMITIVES"],
                    ["d", 4, 0.60, 0.99, "RARE_TEMPORAL_COMPOSITION"],
                ],
            )
            _write_csv(
                diagnostics_path,
                [
                    "worker_id",
                    "rank",
                    "final_score",
                    "quality_score",
                    "reason_code",
                    "new_cluster_id",
                    "grammar_feature_present",
                    "token_nll_p95",
                    "rare_phrase_fraction",
                    "transition_nll_p95",
                    "longest_unseen_phrase_len",
                ],
                [
                    ["a", 1, 0.90, 0.98, "COHESIVE_NEW_WORKFLOW", 0, "True", 1.0, 0.0, 1.1, 0],
                    ["b", 2, 0.80, 0.30, "LOW_QUALITY", 1, "True", 9.0, 0.9, 8.5, 5],
                    ["c", 3, 0.70, 0.95, "RARE_MOTION_PRIMITIVES", 2, "False", 0.0, 0.0, 0.0, 0],
                    ["d", 4, 0.60, 0.99, "RARE_TEMPORAL_COMPOSITION", 3, "True", 7.0, 0.7, 6.5, 4],
                ],
            )
            _write_csv(
                quality_path,
                ["sample_id", "quality_score", "spike_rate"],
                [["a", 0.98, 0.0], ["b", 0.30, 0.2], ["c", 0.95, 0.0], ["d", 0.99, 0.0]],
            )

            report = audit_ranking_artifacts(
                submission_path=submission_path,
                diagnostics_path=diagnostics_path,
                quality_metadata_path=quality_path,
                top_ks=(2,),
                n_examples=2,
            )

            grammar = report["grammar_diagnostics"]
            self.assertEqual(grammar["present_count"], 3)
            self.assertEqual(grammar["top_surprisal_low_quality_count"], 1)
            self.assertEqual(grammar["top_surprisal_examples"][0]["worker_id"], "b")
            self.assertLess(grammar["quality_correlation_token_nll_p95"], 0.0)
            self.assertEqual(grammar["top_surprisal_reason_code_counts"]["LOW_QUALITY"], 1)

            paths = write_audit_artifacts(report, root / "out", suffix="unit")
            self.assertTrue(paths["top_grammar_examples"].exists())

    def test_modal_audit_uses_remote_function(self):
        source = Path("modal_audit.py").read_text(encoding="utf-8")

        self.assertIn("remote_ranking_audit.remote", source)
        self.assertIn("run_ranking_audit", source)
        self.assertIn("ranking_audit.json", source)


def _write_csv(path: Path, fieldnames: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
