import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.active_benchmark.benchmark_decision import (
    build_benchmark_decision_report,
    write_benchmark_decision_reports,
)


class BenchmarkDecisionReportTests(unittest.TestCase):
    def test_decision_report_gates_benchmark_before_downstream_training(self):
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

            report = build_benchmark_decision_report(run_root, aggregate_path=aggregate_path)

        gates = {gate["name"]: gate for gate in report["gates"]}
        self.assertEqual(report["decision"]["downstream_training"], "hold")
        self.assertEqual(report["decision"]["baseline_to_carry"], "old_novelty_window_sourcecap2")
        self.assertEqual(gates["oracle_sanity"]["status"], "pass")
        self.assertEqual(gates["bad_control_sanity"]["status"], "pass")
        self.assertEqual(gates["near_zero_sensitivity"]["status"], "pass")
        self.assertEqual(gates["source_concentration"]["status"], "pass")
        self.assertEqual(gates["ts2vec_incremental_value"]["status"], "warn")
        self.assertIn("TS2Vec", gates["ts2vec_incremental_value"]["read"])
        self.assertIn("downstream", " ".join(report["decision"]["next_steps"]))

    def test_decision_report_writes_json_and_markdown(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "gcp_run"
            _write_seed_report(run_root / "seed_17" / "offline_active_benchmark_report.json")
            output_json = root / "benchmark_decision.json"
            output_md = root / "benchmark_decision.md"

            paths = write_benchmark_decision_reports(
                run_root,
                output_json=output_json,
                output_markdown=output_md,
            )

            report = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

        self.assertEqual(report["decision"]["downstream_training"], "hold")
        self.assertIn("## Decision", markdown)
        self.assertIn("## Gates", markdown)
        self.assertIn("TS2Vec", markdown)
        self.assertIn("old_novelty_window_sourcecap2", markdown)

    def test_decision_report_script_runs_directly_from_repo_root(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "gcp_run"
            _write_seed_report(run_root / "seed_17" / "offline_active_benchmark_report.json")
            output_json = root / "benchmark_decision.json"
            output_md = root / "benchmark_decision.md"

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/benchmark_decision_report.py",
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
        self.assertIn("benchmark_decision_written", completed.stdout)


def _write_seed_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_seed_report(), indent=2), encoding="utf-8")


def _seed_report() -> dict[str, object]:
    policies = (
        "random_valid",
        "quality_only",
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
        "oracle_greedy_eval_only",
    )
    selected_by_policy = {
        "random_valid": ("clip_j", "clip_k"),
        "quality_only": ("clip_l", "clip_m"),
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
        "oracle_greedy_eval_only": ("clip_o", "clip_p"),
    }
    final_gain_by_policy = {
        "random_valid": 0.20,
        "quality_only": 0.25,
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
        "oracle_greedy_eval_only": 0.75,
    }
    rounds = []
    for policy in policies:
        selected = selected_by_policy[policy]
        largest_source_fraction = 1.0 if policy in {"submitted_no_kcenter", "old_novelty_window"} else 0.5
        rounds.append(
            {
                "episode_id": "episode_000",
                "fold_id": 0,
                "policy_name": policy,
                "round_index": 0,
                "batch_size": 2,
                "candidate_ids_before": [f"clip_{letter}" for letter in "abcdefghijklmnop"],
                "candidate_ids_after": [clip for clip in [f"clip_{letter}" for letter in "abcdefghijklmnop"] if clip not in selected],
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
                    "largest_source_group_fraction": largest_source_fraction,
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
        "selected_score": 1.0 - rank * 0.5,
        "passed_quality_gate": True,
        "passed_artifact_gate": artifact_score <= 0.05,
    }


if __name__ == "__main__":
    unittest.main()
