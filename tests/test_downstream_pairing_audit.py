import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.active_benchmark.downstream_pairing_audit import (
    build_supervised_pairing_audit,
    write_supervised_pairing_audit_reports,
)
from scripts.downstream_supervised_pairing_audit_report import load_seed_reports


class DownstreamPairingAuditTests(unittest.TestCase):
    def test_audit_recomputes_total_paired_deltas_from_common_initial_baseline(self):
        report = {
            "rows": [
                _row("random_valid", round_index=0, baseline_accuracy=0.0, after_accuracy=0.8, baseline_nll=10.0, after_nll=3.0),
                _row("quality_only", round_index=0, baseline_accuracy=0.0, after_accuracy=0.4, baseline_nll=10.0, after_nll=6.0),
                _row("random_valid", round_index=1, baseline_accuracy=0.8, after_accuracy=0.8, baseline_nll=3.0, after_nll=3.0),
                _row("quality_only", round_index=1, baseline_accuracy=0.4, after_accuracy=0.7, baseline_nll=6.0, after_nll=2.0),
            ]
        }

        audit = build_supervised_pairing_audit({17: report}, random_policy="random_valid")

        self.assertTrue(audit["baseline_audit"]["round0_baseline_consistent"])
        self.assertTrue(audit["baseline_audit"]["final_round_baseline_varies"])
        quality = audit["policy_paired_summary"]["quality_only"]
        self.assertAlmostEqual(quality["mean_final_incremental_accuracy_gain_minus_random"], 0.3)
        self.assertAlmostEqual(quality["mean_total_accuracy_gain_minus_random"], -0.1)
        self.assertAlmostEqual(quality["mean_after_accuracy_minus_random"], -0.1)
        self.assertEqual(quality["after_accuracy_win_count_vs_random"], 0)
        self.assertIn("final-round incremental", audit["decision"]["read"])

    def test_quality_stratified_repeat_result_stops_repeat_recommendation(self):
        report = {
            "rows": [
                _row("random_valid", round_index=0, baseline_accuracy=0.0, after_accuracy=0.7, baseline_nll=10.0, after_nll=4.0),
                _row("quality_stratified_random", round_index=0, baseline_accuracy=0.0, after_accuracy=0.8, baseline_nll=10.0, after_nll=3.0),
                _row("old_novelty_ts2vec", round_index=0, baseline_accuracy=0.0, after_accuracy=0.6, baseline_nll=10.0, after_nll=5.0),
            ]
        }

        audit = build_supervised_pairing_audit({17: report}, random_policy="random_valid")

        self.assertEqual(audit["decision"]["next_gate"], "hold_proxy_not_promotive")
        self.assertIn("quality_stratified_random repeat is complete", audit["decision"]["read"])
        self.assertEqual(audit["decision"]["best_total_accuracy_gain_policy"], "quality_stratified_random")

    def test_writer_outputs_json_and_markdown(self):
        report = {
            "rows": [
                _row("random_valid", round_index=0, baseline_accuracy=0.0, after_accuracy=1.0, baseline_nll=10.0, after_nll=1.0),
                _row("quality_only", round_index=0, baseline_accuracy=0.0, after_accuracy=1.0, baseline_nll=10.0, after_nll=1.0),
            ]
        }
        audit = build_supervised_pairing_audit({17: report}, random_policy="random_valid")
        with TemporaryDirectory() as tmp:
            paths = write_supervised_pairing_audit_reports(audit, Path(tmp))
            self.assertTrue(paths["json"].exists())
            self.assertTrue(paths["markdown"].exists())
            self.assertEqual(json.loads(paths["json"].read_text(encoding="utf-8"))["input"]["seed_count"], 1)
            markdown = paths["markdown"].read_text(encoding="utf-8")
            self.assertIn("quality_only", markdown)
            self.assertIn("final_incremental_delta_vs_random", markdown)

    def test_loads_seed_reports_from_gcp_run_root(self):
        report = {
            "rows": [
                _row("random_valid", round_index=0, baseline_accuracy=0.0, after_accuracy=1.0, baseline_nll=10.0, after_nll=1.0),
            ]
        }
        with TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            seed_dir = run_root / "seed_17"
            seed_dir.mkdir()
            (seed_dir / "downstream_supervised_smoke_report.json").write_text(json.dumps(report), encoding="utf-8")

            reports = load_seed_reports(run_root)

        self.assertEqual(sorted(reports), [17])
        self.assertEqual(reports[17]["rows"], report["rows"])

    def test_cli_runs_without_manual_pythonpath(self):
        report = {
            "rows": [
                _row("random_valid", round_index=0, baseline_accuracy=0.0, after_accuracy=1.0, baseline_nll=10.0, after_nll=1.0),
            ]
        }
        with TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            seed_dir = run_root / "seed_17"
            seed_dir.mkdir()
            (seed_dir / "downstream_supervised_smoke_report.json").write_text(json.dumps(report), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/downstream_supervised_pairing_audit_report.py",
                    "--run-root",
                    str(run_root),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((run_root / "downstream_supervised_pairing_audit.json").exists())


def _row(
    policy_name: str,
    *,
    round_index: int,
    baseline_accuracy: float,
    after_accuracy: float,
    baseline_nll: float,
    after_nll: float,
) -> dict[str, object]:
    return {
        "episode_id": "episode_000",
        "fold_id": 0,
        "policy_name": policy_name,
        "round_index": round_index,
        "representation": "window",
        "baseline_accuracy": baseline_accuracy,
        "after_accuracy": after_accuracy,
        "accuracy_gain": after_accuracy - baseline_accuracy,
        "baseline_balanced_accuracy": baseline_accuracy,
        "after_balanced_accuracy": after_accuracy,
        "balanced_accuracy_gain": after_accuracy - baseline_accuracy,
        "baseline_negative_log_likelihood": baseline_nll,
        "after_negative_log_likelihood": after_nll,
        "nll_reduction": baseline_nll - after_nll,
        "selected_ids": [f"{policy_name}_selected_{round_index}"],
        "selected_source_group_ids": [f"{policy_name}_worker_{round_index}"],
    }


if __name__ == "__main__":
    unittest.main()
