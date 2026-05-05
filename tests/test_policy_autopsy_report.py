import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.active_benchmark.policy_autopsy import (
    build_policy_autopsy,
    render_policy_autopsy_markdown,
    write_policy_autopsy_reports,
)


class PolicyAutopsyReportTests(unittest.TestCase):
    def test_policy_autopsy_summarizes_overlap_gate_effect_and_hygiene(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "gcp_run"
            _write_seed_report(run_root / "seed_17" / "offline_active_benchmark_report.json")
            aggregate_path = root / "aggregate_proof_summary.json"
            aggregate_path.write_text(
                json.dumps(
                    {
                        "all_leakage_ok": True,
                        "oracle_fraction_exact_all_rounds": True,
                        "mean_near_zero_oracle_round_fraction": 0.0,
                    }
                ),
                encoding="utf-8",
            )

            autopsy = build_policy_autopsy(run_root, aggregate_path=aggregate_path)

        outcomes = {row["policy"]: row for row in autopsy["policy_outcomes"]}
        self.assertGreater(
            outcomes["old_novelty_window"]["mean_final_cumulative_gain"],
            outcomes["blend_kcenter_ts2vec_window"]["mean_final_cumulative_gain"],
        )
        self.assertEqual(outcomes["old_novelty_window"]["final_episode_wins"], 0)
        self.assertEqual(outcomes["old_novelty_window_sourcecap2"]["final_episode_wins"], 1)

        overlaps = {
            (row["policy_a"], row["policy_b"]): row
            for row in autopsy["pairwise_overlap"]
        }
        blend_old = overlaps[("blend_kcenter_ts2vec_window", "old_novelty_window")]
        self.assertAlmostEqual(blend_old["mean_jaccard"], 1.0 / 3.0)
        self.assertAlmostEqual(blend_old["exact_batch_match_fraction"], 0.0)

        gate = autopsy["artifact_gate_effect"]
        self.assertTrue(gate["is_no_op"])
        self.assertEqual(gate["changed_round_count"], 0)
        self.assertAlmostEqual(gate["exact_batch_match_fraction"], 1.0)

        hygiene = {row["policy"]: row for row in autopsy["policy_hygiene"]}
        self.assertAlmostEqual(hygiene["blend_kcenter_ts2vec_window"]["artifact_selected_rate"], 0.5)
        self.assertEqual(hygiene["artifact_gate_blend_kcenter_ts2vec_window"]["selected_count"], 2)
        self.assertEqual(autopsy["coverage_story"]["oracle_fraction_exact_all_rounds"], True)
        paired = {
            (row["policy_a"], row["policy_b"]): row
            for row in autopsy["paired_policy_deltas"]
        }
        old_vs_blend_delta = paired[("old_novelty_window", "blend_kcenter_ts2vec_window")]
        self.assertAlmostEqual(old_vs_blend_delta["mean_delta_final_cumulative_gain"], 0.18)
        self.assertIn("bootstrap_ci95_low", old_vs_blend_delta)
        self.assertEqual(old_vs_blend_delta["paired_episode_count"], 1)
        source_read = autopsy["source_kcenter_read"]
        comparisons = {row["comparison"]: row for row in source_read["comparisons"]}
        self.assertAlmostEqual(comparisons["sourcecap_minus_old"]["mean_delta_final_cumulative_gain"], 0.04)
        self.assertAlmostEqual(comparisons["full_replay_minus_no_kcenter"]["mean_delta_final_cumulative_gain"], 0.12)
        self.assertAlmostEqual(comparisons["full_replay_minus_minus_ts2vec"]["mean_delta_final_cumulative_gain"], -0.08)
        self.assertIn("k-center", " ".join(source_read["summary"]))
        self.assertIn("artifact gate did not change any compared batches", "\n".join(autopsy["diagnosis"]))

    def test_policy_autopsy_writes_json_and_markdown_reports(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "gcp_run"
            _write_seed_report(run_root / "seed_17" / "offline_active_benchmark_report.json")
            output_json = root / "policy_autopsy.json"
            output_md = root / "policy_autopsy.md"

            paths = write_policy_autopsy_reports(
                run_root,
                output_json=output_json,
                output_markdown=output_md,
            )

            report = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

        self.assertEqual(report["input"]["report_count"], 1)
        self.assertIn("## Why Old Novelty Vs Blend", markdown)
        self.assertIn("## Source And K-Center Read", markdown)
        self.assertIn("## Selection Overlap", markdown)
        self.assertIn("## Paired Deltas", markdown)
        self.assertIn("## Artifact Gate", markdown)
        self.assertIn("old_novelty_window", markdown)

    def test_policy_autopsy_script_runs_directly_from_repo_root(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "gcp_run"
            _write_seed_report(run_root / "seed_17" / "offline_active_benchmark_report.json")
            output_json = root / "policy_autopsy.json"
            output_md = root / "policy_autopsy.md"

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/policy_autopsy_report.py",
                    "--run-root",
                    str(run_root),
                    "--output-json",
                    str(output_json),
                    "--output-markdown",
                    str(output_md),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("policy_autopsy_written", completed.stdout)


def _write_seed_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_seed_report(), indent=2), encoding="utf-8")


def _seed_report() -> dict[str, object]:
    policies = (
        "old_novelty_window",
        "old_novelty_window_sourcecap2",
        "blend_kcenter_ts2vec_window",
        "artifact_gate_blend_kcenter_ts2vec_window",
        "kcenter_quality_gated_window",
        "submitted_full_replay",
        "submitted_minus_ts2vec",
        "submitted_minus_window",
        "submitted_no_kcenter",
        "window_novelty_same_gates_no_kcenter",
        "ts2vec_novelty_same_gates_no_kcenter",
    )
    rounds = []
    selected_by_policy = {
        "old_novelty_window": ("clip_a", "clip_b"),
        "old_novelty_window_sourcecap2": ("clip_a", "clip_d"),
        "blend_kcenter_ts2vec_window": ("clip_a", "clip_c"),
        "artifact_gate_blend_kcenter_ts2vec_window": ("clip_a", "clip_c"),
        "kcenter_quality_gated_window": ("clip_d", "clip_e"),
        "submitted_full_replay": ("clip_a", "clip_c"),
        "submitted_minus_ts2vec": ("clip_d", "clip_e"),
        "submitted_minus_window": ("clip_a", "clip_f"),
        "submitted_no_kcenter": ("clip_a", "clip_g"),
        "window_novelty_same_gates_no_kcenter": ("clip_a", "clip_b"),
        "ts2vec_novelty_same_gates_no_kcenter": ("clip_h", "clip_i"),
    }
    final_gain_by_policy = {
        "old_novelty_window": 0.60,
        "old_novelty_window_sourcecap2": 0.64,
        "blend_kcenter_ts2vec_window": 0.42,
        "artifact_gate_blend_kcenter_ts2vec_window": 0.42,
        "kcenter_quality_gated_window": 0.50,
        "submitted_full_replay": 0.42,
        "submitted_minus_ts2vec": 0.50,
        "submitted_minus_window": 0.41,
        "submitted_no_kcenter": 0.30,
        "window_novelty_same_gates_no_kcenter": 0.60,
        "ts2vec_novelty_same_gates_no_kcenter": 0.41,
    }
    for policy in policies:
        selected = selected_by_policy[policy]
        rounds.append(
            {
                "episode_id": "episode_000",
                "fold_id": 0,
                "policy_name": policy,
                "round_index": 0,
                "batch_size": 2,
                "candidate_ids_before": ["clip_a", "clip_b", "clip_c", "clip_d", "clip_e"],
                "candidate_ids_after": [clip for clip in ["clip_a", "clip_b", "clip_c", "clip_d", "clip_e"] if clip not in selected],
                "selected_ids": list(selected),
                "selected_scores": [1.0, 0.5],
                "selected_source_group_ids": [f"worker_{clip[-1]}" for clip in selected],
                "support_ids_before": ["support_a"],
                "support_ids_after": ["support_a", *selected],
                "coverage_by_representation": {
                    "window": {
                        "relative_coverage_gain": final_gain_by_policy[policy],
                        "coverage_gain": final_gain_by_policy[policy] / 10.0,
                    },
                    "ts2vec": {
                        "relative_coverage_gain": final_gain_by_policy[policy] / 2.0,
                        "coverage_gain": final_gain_by_policy[policy] / 20.0,
                    },
                },
                "balanced_relative_gain": final_gain_by_policy[policy],
                "cumulative_balanced_relative_gain": final_gain_by_policy[policy],
                "oracle_fraction": final_gain_by_policy[policy] / 0.75,
                "selection_summary": {
                    "min_quality": 0.91,
                    "largest_source_group_fraction": 0.5,
                },
                "selection_details": [
                    _detail(selected[0], rank=0, artifact_score=0.0),
                    _detail(selected[1], rank=1, artifact_score=0.40 if selected[1] == "clip_c" else 0.0),
                ],
            }
        )
    return {
        "config": {
            "max_artifact_score": 0.05,
            "primary_representations": ["window", "ts2vec"],
        },
        "difficulty_audit": [
            {
                "episode_id": "episode_000",
                "near_zero_oracle_round_fraction": 0.0,
                "oracle_fraction_exact_all_rounds": True,
                "oracle_fraction_exact_by_round": [True],
                "max_oracle_greedy_cumulative_gain": 0.75,
            }
        ],
        "episodes": [],
        "n_episodes": 1,
        "policies": list(policies),
        "policy_summary": {},
        "rounds": rounds,
    }


def _detail(sample_id: str, *, rank: int, artifact_score: float) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "source_group_id": f"worker_{sample_id[-1]}",
        "rank_index": rank,
        "quality_score": 0.95 if artifact_score == 0.0 else 0.91,
        "artifact_score": artifact_score,
        "stationary_fraction": 0.0,
        "max_abs_value": 10.0,
        "old_novelty_window_score": 0.7 - rank * 0.1,
        "blend_left_novelty_score": 0.6 - rank * 0.1,
        "blend_right_novelty_score": 0.7 - rank * 0.1,
        "blend_score": 0.65 - rank * 0.1,
        "selected_score": 1.0 - rank * 0.5,
        "passed_quality_gate": True,
        "passed_artifact_gate": artifact_score <= 0.05,
    }


if __name__ == "__main__":
    unittest.main()
