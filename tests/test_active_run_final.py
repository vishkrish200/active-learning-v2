import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.active.run_final import package_final_submission


class ActiveRunFinalTests(unittest.TestCase):
    def test_packages_promoted_artifacts_into_evaluator_ready_directory(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "final"
            docs = root / "docs"
            source.mkdir()
            docs.mkdir()
            _write_csv(
                source / "spike_hygiene_ablation_artifact_gate_submission_full_new_worker_id.csv",
                [
                    {"rank": 1, "score": 1.0, "new_worker_id": "sample000003"},
                    {"rank": 2, "score": 0.5, "new_worker_id": "sample000001"},
                    {"rank": 3, "score": 0.1, "new_worker_id": "sample000002"},
                ],
            )
            _write_csv(
                source / "spike_hygiene_ablation_artifact_gate_submission_full_worker_id.csv",
                [
                    {"rank": 1, "score": 1.0, "worker_id": "worker003"},
                    {"rank": 2, "score": 0.5, "worker_id": "worker001"},
                    {"rank": 3, "score": 0.1, "worker_id": "worker002"},
                ],
            )
            _write_csv(
                source / "spike_hygiene_ablation_artifact_gate_diagnostics_full.csv",
                [
                    {
                        "rank": 1,
                        "sample_id": "sample000003",
                        "score": 1.0,
                        "quality_score": 0.99,
                        "trace_artifact_pass": True,
                    },
                    {
                        "rank": 2,
                        "sample_id": "sample000001",
                        "score": 0.5,
                        "quality_score": 0.98,
                        "trace_artifact_pass": True,
                    },
                    {
                        "rank": 3,
                        "sample_id": "sample000002",
                        "score": 0.1,
                        "quality_score": 0.97,
                        "trace_artifact_pass": True,
                    },
                ],
            )
            (source / "spike_hygiene_ablation_report_full.json").write_text(
                json.dumps({"summary": {"artifact_gate": {"topk": {"10": {"trace_artifact_fail_rate": 0.0}}}}}),
                encoding="utf-8",
            )
            validation_report = docs / "active_loop_validation_report.json"
            validation_report.write_text(json.dumps({"mode": "full", "n_episodes": 64}), encoding="utf-8")
            support_audit = docs / "support_audit.json"
            support_audit.write_text(json.dumps({"worker_coverage": 0.93}), encoding="utf-8")
            stability_report = docs / "stability.json"
            stability_report.write_text(json.dumps({"rank_spearman_mean": 0.95}), encoding="utf-8")

            result = package_final_submission(
                {
                    "method": {
                        "name": "artifact-gate exact-window blend",
                        "primary_submission_id_column": "new_worker_id",
                        "claim": "Budgeted TS2Vec/window blended k-center selector.",
                    },
                    "inputs": {
                        "old_manifest": "cache/manifests/pretrain_full_cached_urls.txt",
                        "new_manifest": "cache/manifests/new_urls.txt",
                    },
                    "source_artifacts": {
                        "source_dir": str(source),
                        "primary_submission": "spike_hygiene_ablation_artifact_gate_submission_full_new_worker_id.csv",
                        "backup_worker_submission": "spike_hygiene_ablation_artifact_gate_submission_full_worker_id.csv",
                        "diagnostics": "spike_hygiene_ablation_artifact_gate_diagnostics_full.csv",
                        "selector_report": "spike_hygiene_ablation_report_full.json",
                        "validation_report": str(validation_report),
                        "support_audit": str(support_audit),
                        "stability_report": str(stability_report),
                    },
                    "artifacts": {"output_dir": str(output)},
                    "validation": {"expected_count": 3},
                }
            )

            self.assertEqual(result["n_rows"], 3)
            self.assertTrue(Path(result["artifacts"]["primary_submission"]).exists())
            self.assertTrue(Path(result["artifacts"]["backup_worker_submission"]).exists())
            self.assertTrue(Path(result["artifacts"]["diagnostics"]).exists())
            self.assertTrue(Path(result["artifacts"]["selector_config"]).exists())
            self.assertTrue(Path(result["artifacts"]["feature_schema"]).exists())
            self.assertTrue(Path(result["artifacts"]["validation_report"]).exists())
            self.assertIn("README_final.md", result["artifacts"]["readme"])

            packaged_rows = _read_csv(Path(result["artifacts"]["primary_submission"]))
            self.assertEqual([row["new_worker_id"] for row in packaged_rows], ["sample000003", "sample000001", "sample000002"])
            readme = Path(result["artifacts"]["readme"]).read_text(encoding="utf-8")
            self.assertIn("artifact-gate exact-window blend", readme)
            self.assertIn("not an exact full-200k TS2Vec search", readme)

    def test_rejects_submission_with_missing_rank(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            _write_csv(
                source / "primary.csv",
                [
                    {"rank": 1, "score": 1.0, "new_worker_id": "sample000001"},
                    {"rank": 3, "score": 0.5, "new_worker_id": "sample000002"},
                ],
            )
            _write_csv(source / "backup.csv", [{"rank": 1, "score": 1.0, "worker_id": "worker001"}])
            _write_csv(source / "diagnostics.csv", [{"rank": 1, "sample_id": "sample000001"}])
            (source / "report.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "contiguous ranks"):
                package_final_submission(
                    {
                        "source_artifacts": {
                            "source_dir": str(source),
                            "primary_submission": "primary.csv",
                            "backup_worker_submission": "backup.csv",
                            "diagnostics": "diagnostics.csv",
                            "selector_report": "report.json",
                        },
                        "artifacts": {"output_dir": str(root / "out")},
                        "validation": {"expected_count": 2},
                    }
                )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
