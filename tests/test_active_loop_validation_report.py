import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.active.active_loop_validation_report import (
    bootstrap_mean_ci,
    build_active_loop_validation_report,
)


class ActiveLoopValidationReportTests(unittest.TestCase):
    def test_bootstrap_mean_ci_is_deterministic_and_contains_mean(self):
        first = bootstrap_mean_ci([0.1, 0.2, 0.3, 0.4], n_bootstrap=200, seed=7)
        second = bootstrap_mean_ci([0.1, 0.2, 0.3, 0.4], n_bootstrap=200, seed=7)

        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean"], 0.25, places=8)
        self.assertLessEqual(first["ci_low"], first["mean"])
        self.assertGreaterEqual(first["ci_high"], first["mean"])

    def test_report_summarizes_coverage_wins_and_hygiene(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            coverage_path = root / "coverage.csv"
            selection_path = root / "selection.csv"
            output_dir = root / "out"
            _write_csv(
                coverage_path,
                [
                    _coverage_row("episode_1", "artifact", 10, 0.20, 0.50),
                    _coverage_row("episode_2", "artifact", 10, 0.10, 0.25),
                    _coverage_row("episode_1", "plain", 10, 0.10, 0.25),
                    _coverage_row("episode_2", "plain", 10, 0.15, 0.40),
                    _coverage_row("episode_1", "trace", 10, 0.05, 0.15),
                    _coverage_row("episode_2", "trace", 10, 0.08, 0.20),
                ],
            )
            _write_csv(
                selection_path,
                [
                    _selection_row("episode_1", "artifact", 10, 0.0, 0.1, 0.0, 0.2),
                    _selection_row("episode_2", "artifact", 10, 0.0, 0.3, 0.0, 0.0),
                    _selection_row("episode_1", "plain", 10, 0.2, 0.4, 0.1, 0.1),
                    _selection_row("episode_2", "plain", 10, 0.1, 0.2, 0.2, 0.1),
                    _selection_row("episode_1", "trace", 10, 0.0, 0.0, 0.0, 0.0),
                    _selection_row("episode_2", "trace", 10, 0.0, 0.0, 0.0, 0.0),
                ],
            )

            report = build_active_loop_validation_report(
                coverage_by_episode_path=coverage_path,
                selection_audit_path=selection_path,
                output_dir=output_dir,
                policies=["artifact", "plain", "trace"],
                k_values=[10],
                representation="balanced",
                primary_policy="artifact",
                compare_policies=["plain", "trace"],
                n_bootstrap=100,
                seed=11,
                mode="unit",
            )
            self.assertTrue(Path(report["artifacts"]["json"]).exists())
            markdown = Path(report["artifacts"]["markdown"]).read_text(encoding="utf-8")
            saved = json.loads(Path(report["artifacts"]["json"]).read_text(encoding="utf-8"))

        artifact_summary = report["policy_summaries"]["artifact"]["coverage@10"]
        self.assertAlmostEqual(artifact_summary["relative_gain"]["mean"], 0.15, places=8)
        self.assertAlmostEqual(artifact_summary["oracle_fraction"]["mean"], 0.375, places=8)
        self.assertAlmostEqual(
            report["selection_hygiene"]["artifact"]["coverage@10"]["trace_fail_rate_at_k"]["mean"],
            0.2,
            places=8,
        )
        self.assertEqual(
            report["policy_comparisons"]["artifact_vs_plain"]["coverage@10"]["primary_win_count"],
            1,
        )
        self.assertEqual(
            report["policy_comparisons"]["artifact_vs_trace"]["coverage@10"]["primary_win_count"],
            2,
        )
        self.assertIn("Active-Loop Validation Report", markdown)
        self.assertEqual(saved["mode"], "unit")


def _coverage_row(episode_id: str, policy: str, k: int, relative_gain: float, oracle_fraction: float) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "policy": policy,
        "k": k,
        "representation": "balanced",
        "relative_gain": relative_gain,
        "oracle_fraction": oracle_fraction,
        "absolute_gain": relative_gain / 10.0,
        "coverage_before": 1.0,
        "coverage_after": 1.0 - relative_gain / 10.0,
        "oracle_absolute_gain": 0.1,
        "selected_count": k,
    }


def _selection_row(
    episode_id: str,
    policy: str,
    k: int,
    artifact_rate: float,
    trace_fail_rate: float,
    spike_fail_rate: float,
    duplicate_rate: float,
) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "policy": policy,
        "k": k,
        "selected_count": k,
        "artifact_rate_at_k": artifact_rate,
        "low_quality_rate_at_k": artifact_rate,
        "duplicate_rate_at_k": duplicate_rate,
        "spike_fail_rate_at_k": spike_fail_rate,
        "trace_artifact_fail_rate_at_k": artifact_rate,
        "trace_fail_rate_at_k": trace_fail_rate,
        "unique_new_clusters_at_k": k,
        "unique_source_groups_at_k": k,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
