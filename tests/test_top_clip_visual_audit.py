import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from marginal_value.eval import top_clip_visual_audit
from marginal_value.eval.top_clip_visual_audit import validate_visual_audit_config


class TopClipVisualAuditTests(unittest.TestCase):
    def test_visual_audit_writes_report_without_plotting(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ranking_dir = root / "artifacts" / "ranking"
            output_dir = ranking_dir / "visual_audit"
            ranking_dir.mkdir(parents=True)
            _write_csv(
                ranking_dir / "baseline_diagnostics_val_full.csv",
                [
                    _diagnostic_row("sample-a", 1, parent="p-big", cluster="p-big/0", score=0.9),
                    _diagnostic_row("sample-b", 2, parent="p-small", cluster="p-small/0", score=0.8),
                    _diagnostic_row("sample-c", 3, parent="p-big", cluster="p-big/1", score=0.7),
                    _diagnostic_row("sample-d", 4, parent="p-big", cluster="p-big/2", score=0.6),
                ],
            )
            _write_csv(
                ranking_dir / "baseline_quality_metadata_full.csv",
                [
                    {"sample_id": "sample-a", "raw_path": str(root / "a.jsonl"), "quality_score": "0.99"},
                    {"sample_id": "sample-b", "raw_path": str(root / "b.jsonl"), "quality_score": "0.98"},
                    {"sample_id": "sample-c", "raw_path": str(root / "c.jsonl"), "quality_score": "0.97"},
                    {"sample_id": "sample-d", "raw_path": str(root / "d.jsonl"), "quality_score": "0.96"},
                ],
            )
            config = {
                "execution": {"provider": "modal"},
                "artifacts": {
                    "ranking_dir": str(ranking_dir).replace(str(root / "artifacts"), "/artifacts"),
                    "output_dir": str(output_dir).replace(str(root / "artifacts"), "/artifacts"),
                },
                "visual_audit": {
                    "top_n": 2,
                    "dominant_parent_top_k": 4,
                    "dominant_parent_examples": 2,
                    "top_ks": [2, 4],
                    "generate_plots": False,
                },
            }
            # The production validator requires mounted Modal paths; the runner can use local
            # temp paths in unit tests once validation has been exercised.
            validate_visual_audit_config(config)
            config["artifacts"]["ranking_dir"] = str(ranking_dir)
            config["artifacts"]["output_dir"] = str(output_dir)

            with patch.object(top_clip_visual_audit, "validate_visual_audit_config", lambda _config: None):
                result = top_clip_visual_audit.run_top_clip_visual_audit(config, smoke=False)

            self.assertEqual(result["dominant_parent_cluster_id"], "p-big")
            self.assertEqual(result["n_selected_for_plot"], 3)
            self.assertEqual(result["n_plots_written"], 0)
            report = json.loads((output_dir / "top_clip_visual_audit_report_full.json").read_text())
            self.assertEqual(report["top_summary"]["2"]["unique_parent_clusters"], 2)
            self.assertAlmostEqual(report["top_summary"]["2"]["stationary_fraction_over_90"], 0.5)
            self.assertEqual(report["dominant_parent_summary"]["dominant_parent_count"], 3)
            with (output_dir / "top_clip_visual_audit_top_rows_full.csv").open(newline="", encoding="utf-8") as handle:
                top_rows = list(csv.DictReader(handle))
            self.assertIn("physical_validity_pass", top_rows[0])
            self.assertIn("physical_validity_max_abs_value", top_rows[0])
            self.assertTrue((output_dir / "top_clip_visual_audit_index_full.html").exists())

    def test_visual_audit_rejects_non_modal_config(self):
        with self.assertRaises(ValueError):
            validate_visual_audit_config(
                {
                    "execution": {"provider": "local"},
                    "artifacts": {"ranking_dir": "/artifacts/ranking"},
                }
            )


def _diagnostic_row(sample_id: str, rank: int, *, parent: str, cluster: str, score: float) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "worker_id": sample_id,
        "rank": str(rank),
        "final_score": str(score),
        "quality_score": "0.99",
        "stationary_fraction": "0.95" if rank == 1 else "0.10",
        "max_abs_value": "12.0",
        "quality_gate_pass": "True",
        "quality_threshold_pass": "True",
        "physical_validity_pass": "False" if rank == 1 else "True",
        "physical_validity_failure_reasons": "stationary_fraction" if rank == 1 else "",
        "physical_validity_max_stationary_fraction": "0.9",
        "physical_validity_max_abs_value": "60.0",
        "reason_code": "RARE_TEMPORAL_COMPOSITION",
        "new_cluster_parent_id": parent,
        "new_cluster_id": cluster,
        "old_novelty_score": "0.7",
        "grammar_score_component": "0.8",
        "grammar_score": "0.8",
        "new_density_score": "0.6",
        "old_knn_distance": "0.2",
        "new_batch_density": "0.4",
        "large_cluster_split_applied": "False",
        "large_cluster_split_strategy": "not_split",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
