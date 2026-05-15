import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.active_benchmark.benchmark_discrimination import (
    build_benchmark_discrimination_report,
    write_benchmark_discrimination_reports,
)


class BenchmarkDiscriminationReportTests(unittest.TestCase):
    def test_discrimination_report_uses_episode_level_random_replay_percentiles(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "run"
            _write_seed_report(run_root / "seed_17" / "offline_active_benchmark_report.json", include_replays=True)

            report = build_benchmark_discrimination_report(run_root)

        self.assertEqual(report["decision"]["benchmark_discrimination"], "pass")
        self.assertEqual(report["random_replay_audit"]["status"], "available")
        self.assertEqual(report["random_replay_audit"]["replay_policy_count"], 2)
        kcenter_random = report["paired_policy_deltas"]["kcenter_quality_gated_window_vs_random_valid"]
        self.assertAlmostEqual(kcenter_random["mean_delta_final_cumulative_gain"], 0.275)
        self.assertEqual(kcenter_random["episode_count"], 2)
        kcenter_replay = report["random_replay_audit"]["policy_percentiles"]["kcenter_quality_gated_window"]
        self.assertAlmostEqual(kcenter_replay["mean_random_replay_percentile"], 1.0)
        self.assertGreater(report["oracle_opportunity"]["mean_oracle_minus_random"], 0.0)
        self.assertIn("high_opportunity", report["strata"])

    def test_discrimination_report_marks_random_replays_unavailable_without_replay_rows(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "run"
            _write_seed_report(run_root / "seed_17" / "offline_active_benchmark_report.json", include_replays=False)

            report = build_benchmark_discrimination_report(run_root)

        self.assertEqual(report["random_replay_audit"]["status"], "unavailable")
        self.assertEqual(report["random_replay_audit"]["replay_policy_count"], 0)
        self.assertIn("random replay", report["decision"]["next_steps"][0])

    def test_discrimination_report_writes_json_markdown_and_cli_runs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "run"
            _write_seed_report(run_root / "seed_17" / "offline_active_benchmark_report.json", include_replays=True)
            output_json = root / "discrimination.json"
            output_md = root / "discrimination.md"

            paths = write_benchmark_discrimination_reports(
                run_root,
                output_json=output_json,
                output_markdown=output_md,
            )
            saved = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/benchmark_discrimination_report.py",
                    "--run-root",
                    str(run_root),
                    "--output-json",
                    str(root / "cli.json"),
                    "--output-markdown",
                    str(root / "cli.md"),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(saved["decision"]["benchmark_discrimination"], "pass")
        self.assertIn("## Random Replay Audit", markdown)
        self.assertIn("kcenter_quality_gated_window", markdown)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("benchmark_discrimination_written", completed.stdout)


def _write_seed_report(path: Path, *, include_replays: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_seed_report(include_replays=include_replays), indent=2), encoding="utf-8")


def _seed_report(*, include_replays: bool) -> dict[str, object]:
    policies = [
        "random_valid",
        "kcenter_quality_gated_window",
        "submitted_full_replay",
        "oracle_greedy_eval_only",
    ]
    if include_replays:
        policies.extend(["random_valid_replay_000", "random_valid_replay_001"])
    gains_by_episode = {
        "episode_000": {
            "random_valid": 0.20,
            "random_valid_replay_000": 0.18,
            "random_valid_replay_001": 0.22,
            "kcenter_quality_gated_window": 0.50,
            "submitted_full_replay": 0.45,
            "oracle_greedy_eval_only": 0.70,
        },
        "episode_001": {
            "random_valid": 0.30,
            "random_valid_replay_000": 0.28,
            "random_valid_replay_001": 0.32,
            "kcenter_quality_gated_window": 0.55,
            "submitted_full_replay": 0.35,
            "oracle_greedy_eval_only": 0.65,
        },
    }
    rounds = []
    for episode_id, gains in gains_by_episode.items():
        for policy in policies:
            rounds.append(
                {
                    "episode_id": episode_id,
                    "fold_id": int(episode_id[-1]),
                    "policy_name": policy,
                    "round_index": 0,
                    "batch_size": 2,
                    "candidate_ids_before": ["candidate_a", "candidate_b", "candidate_c", "candidate_d"],
                    "candidate_ids_after": ["candidate_c", "candidate_d"],
                    "selected_ids": ["candidate_a", "candidate_b"],
                    "selected_scores": [1.0, 0.5],
                    "selected_source_group_ids": ["worker_a", "worker_b"],
                    "support_ids_before": ["support_a"],
                    "support_ids_after": ["support_a", "candidate_a", "candidate_b"],
                    "coverage_by_representation": {
                        "window": {
                            "relative_coverage_gain": gains[policy],
                            "coverage_gain": gains[policy] / 10.0,
                        }
                    },
                    "balanced_relative_gain": gains[policy],
                    "cumulative_balanced_relative_gain": gains[policy],
                    "oracle_fraction": gains[policy] / gains["oracle_greedy_eval_only"],
                    "selection_summary": {"min_quality": 0.9, "largest_source_group_fraction": 0.5},
                    "selection_details": [],
                }
            )
    return {
        "config": {"primary_representations": ["window"]},
        "difficulty_audit": [
            {
                "episode_id": "episode_000",
                "near_zero_oracle_round_fraction": 0.0,
                "oracle_fraction_exact_all_rounds": True,
                "oracle_fraction_exact_by_round": [True],
                "max_oracle_greedy_cumulative_gain": 0.70,
            },
            {
                "episode_id": "episode_001",
                "near_zero_oracle_round_fraction": 0.0,
                "oracle_fraction_exact_all_rounds": True,
                "oracle_fraction_exact_by_round": [True],
                "max_oracle_greedy_cumulative_gain": 0.65,
            },
        ],
        "episodes": [],
        "n_episodes": 2,
        "policies": policies,
        "policy_summary": {},
        "rounds": rounds,
    }


if __name__ == "__main__":
    unittest.main()
