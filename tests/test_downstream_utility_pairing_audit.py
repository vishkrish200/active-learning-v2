import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.active_benchmark.downstream_utility_pairing_audit import (
    build_utility_pairing_audit,
    write_utility_pairing_audit_reports,
)
from scripts.downstream_utility_pairing_audit_report import load_seed_reports


class DownstreamUtilityPairingAuditTests(unittest.TestCase):
    def test_audit_uses_common_initial_baseline_for_total_reconstruction_gain(self):
        report = {
            "rows": [
                _row("quality_stratified_random", round_index=0, baseline_error=10.0, after_error=8.0),
                _row("kcenter_quality_gated_ts2vec", round_index=0, baseline_error=10.0, after_error=7.0),
                _row("quality_stratified_random", round_index=1, baseline_error=8.0, after_error=6.0),
                _row("kcenter_quality_gated_ts2vec", round_index=1, baseline_error=7.0, after_error=4.0),
            ]
        }

        audit = build_utility_pairing_audit(
            {17: report},
            random_policy="quality_stratified_random",
            baseline_policy="kcenter_quality_gated_ts2vec",
            bootstrap_replicates=25,
        )

        ts2vec = audit["policy_paired_summary"]["kcenter_quality_gated_ts2vec"]
        self.assertAlmostEqual(ts2vec["mean_total_relative_reconstruction_gain"], 0.6)
        self.assertAlmostEqual(ts2vec["mean_total_relative_reconstruction_gain_minus_random"], 0.2)
        self.assertEqual(ts2vec["total_relative_win_count_vs_random"], 1)
        self.assertEqual(audit["decision"]["best_paired_delta_vs_random_policy"], "kcenter_quality_gated_ts2vec")
        self.assertIn("not training proof", audit["decision"]["read"])
        self.assertIn("bootstrap_ci95_low", ts2vec)

    def test_writer_outputs_markdown_with_total_utility_table(self):
        report = {
            "rows": [
                _row("quality_stratified_random", round_index=0, baseline_error=10.0, after_error=8.0),
                _row("kcenter_quality_gated_window", round_index=0, baseline_error=10.0, after_error=5.0),
            ]
        }
        audit = build_utility_pairing_audit({17: report}, random_policy="quality_stratified_random")
        with TemporaryDirectory() as tmp:
            paths = write_utility_pairing_audit_reports(audit, Path(tmp))

            self.assertTrue(paths["json"].exists())
            self.assertTrue(paths["markdown"].exists())
            markdown = paths["markdown"].read_text(encoding="utf-8")
            self.assertIn("Mean Total Relative Reconstruction Gain", markdown)
            self.assertIn("kcenter_quality_gated_window", markdown)

    def test_loads_seed_reports_from_gcp_run_root(self):
        report = {
            "rows": [
                _row("quality_stratified_random", round_index=0, baseline_error=10.0, after_error=8.0),
            ]
        }
        with TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            seed_dir = run_root / "seed_17"
            seed_dir.mkdir()
            (seed_dir / "downstream_utility_smoke_report.json").write_text(json.dumps(report), encoding="utf-8")

            reports = load_seed_reports(run_root)

        self.assertEqual(sorted(reports), [17])
        self.assertEqual(reports[17]["rows"], report["rows"])

    def test_cli_runs_without_manual_pythonpath(self):
        report = {
            "rows": [
                _row("quality_stratified_random", round_index=0, baseline_error=10.0, after_error=8.0),
            ]
        }
        with TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            seed_dir = run_root / "seed_17"
            seed_dir.mkdir()
            (seed_dir / "downstream_utility_smoke_report.json").write_text(json.dumps(report), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/downstream_utility_pairing_audit_report.py",
                    "--run-root",
                    str(run_root),
                    "--random-policy",
                    "quality_stratified_random",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((run_root / "downstream_utility_pairing_audit.json").exists())


def _row(
    policy_name: str,
    *,
    round_index: int,
    baseline_error: float,
    after_error: float,
) -> dict[str, object]:
    return {
        "episode_id": "episode_000",
        "fold_id": 0,
        "policy_name": policy_name,
        "round_index": round_index,
        "representation": "window",
        "baseline_reconstruction_error": baseline_error,
        "after_reconstruction_error": after_error,
        "absolute_reconstruction_gain": baseline_error - after_error,
        "relative_reconstruction_gain": (baseline_error - after_error) / baseline_error,
        "selected_ids": [f"{policy_name}_selected_{round_index}"],
        "selected_source_group_ids": [f"{policy_name}_worker_{round_index}"],
    }


if __name__ == "__main__":
    unittest.main()
