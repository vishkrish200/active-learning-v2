import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.active_benchmark.scientific_claim_memo import build_scientific_claim_memo


class ScientificClaimMemoTests(unittest.TestCase):
    def test_memo_freezes_claims_non_claims_and_advisor_read(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            decision = root / "decision.json"
            selection = root / "selection.json"
            motion = root / "motion.json"
            outcome = root / "outcome.json"
            advisor = root / "advisor.md"
            _write_inputs(decision, selection, motion, outcome)
            advisor.write_text(
                "Advisor says: claim bounded downstream-canary evidence; do not claim causal proof or tune more policies.",
                encoding="utf-8",
            )

            memo = build_scientific_claim_memo(
                decision,
                selection,
                motion,
                outcome,
                advisor_markdown=advisor,
            )

        self.assertEqual(memo["decision"], "freeze_window_kcenter_as_downstream_canary_mean_risk_incumbent")
        self.assertIn("window_kcenter_v1", memo["claims"][0])
        self.assertTrue(any("not a learned query policy" in item for item in memo["non_claims"]))
        self.assertTrue(any("not a scalar motion-energy ranker" in item for item in memo["non_claims"]))
        self.assertTrue(any("policy shopping" in item for item in memo["self_deception_risks"]))
        self.assertIn("Advisor says", memo["advisor"]["summary"])

    def test_memo_script_writes_markdown_and_json(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            decision = root / "decision.json"
            selection = root / "selection.json"
            motion = root / "motion.json"
            outcome = root / "outcome.json"
            advisor = root / "advisor.md"
            output_json = root / "claim_memo.json"
            output_md = root / "claim_memo.md"
            _write_inputs(decision, selection, motion, outcome)
            advisor.write_text("External advisor: freeze the claim; no more selector tuning.", encoding="utf-8")
            repo_root = Path(__file__).resolve().parents[1]

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/scientific_claim_memo_report.py",
                    "--decision-card-json",
                    str(decision),
                    "--selection-mechanism-json",
                    str(selection),
                    "--selected-motion-json",
                    str(motion),
                    "--motion-outcome-json",
                    str(outcome),
                    "--advisor-markdown",
                    str(advisor),
                    "--output-json",
                    str(output_json),
                    "--output-markdown",
                    str(output_md),
                ],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output_json.exists())
            rendered = output_md.read_text(encoding="utf-8")
            self.assertIn("Frozen Scientific Claim Memo", rendered)
            self.assertIn("What This Does Not Claim", rendered)
            self.assertIn("scientific_claim_memo_written", result.stdout)


def _write_inputs(decision: Path, selection: Path, motion: Path, outcome: Path) -> None:
    decision.write_text(
        json.dumps(
            {
                "result_card": {
                    "decision": "survivor_confirmation_window_kcenter",
                    "champion_policy": "window_kcenter_v1",
                    "policy_decisions": {
                        "window_kcenter_v1": "current champion",
                        "ts2vec_kcenter_v1": "feature source / ablation",
                        "submitted_full_replay_v1": "defensible submitted comparator",
                        "quality_stratified_random_v1": "required baseline",
                    },
                },
                "policy_final_summary": [
                    {
                        "rank": 1,
                        "policy_id": "window_kcenter_v1",
                        "row_count": 40,
                        "mean_after_mse": 0.118,
                        "mean_relative_mse_reduction": 0.086,
                        "final_episode_wins": 11,
                    },
                    {
                        "rank": 2,
                        "policy_id": "ts2vec_kcenter_v1",
                        "row_count": 40,
                        "mean_after_mse": 0.121,
                        "mean_relative_mse_reduction": 0.079,
                        "final_episode_wins": 12,
                    },
                ],
                "pairwise_final_deltas": [
                    {
                        "policy_a": "ts2vec_kcenter_v1",
                        "policy_b": "window_kcenter_v1",
                        "paired_unit_count": 40,
                        "mean_after_mse_advantage_a_over_b": -0.0025,
                        "policy_a_win_count": 24,
                        "policy_b_win_count": 16,
                        "highlight": True,
                    }
                ],
                "hygiene": {
                    "leakage_all_ok": True,
                    "selected_target_leak_count_total": 0,
                    "selected_out_of_pool_count_total": 0,
                    "selected_duplicate_clip_count_total": 0,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    selection.write_text(
        json.dumps(
            {
                "focal_pairwise_contrasts": [
                    {
                        "comparison_policy": "ts2vec_kcenter_v1",
                        "mean_jaccard": 0.225,
                        "mean_source_jaccard": 0.388,
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    motion.write_text(
        json.dumps(
            {
                "policy_motion_profiles": [
                    {
                        "policy_id": "window_kcenter_v1",
                        "mean_motion_energy": 8.65,
                        "mean_gyro_norm_p95": 0.868,
                        "regime_counts": {"rotation_dominant": 68},
                    },
                    {
                        "policy_id": "ts2vec_kcenter_v1",
                        "mean_motion_energy": 3.10,
                        "mean_gyro_norm_p95": 0.508,
                        "regime_counts": {"rotation_dominant": 49},
                    },
                ],
                "focal_motion_contrasts": [
                    {
                        "comparison_policy": "ts2vec_kcenter_v1",
                        "mean_focal_only_motion_energy": 10.90,
                        "mean_comparison_only_motion_energy": 2.52,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    outcome.write_text(
        json.dumps(
            {
                "policy_motion_outcome_profiles": [
                    {
                        "policy_id": "window_kcenter_v1",
                        "mean_after_mse": 0.118,
                        "mean_selected_motion_energy": 8.65,
                        "mean_selected_rotation_dominant_rate": 0.425,
                    },
                    {
                        "policy_id": "ts2vec_kcenter_v1",
                        "mean_after_mse": 0.121,
                        "mean_selected_motion_energy": 3.10,
                        "mean_selected_rotation_dominant_rate": 0.306,
                    },
                ],
                "focal_pairwise_motion_outcome_contrasts": [
                    {
                        "comparison_policy": "ts2vec_kcenter_v1",
                        "mean_after_mse_advantage_focal_over_comparison": 0.0025,
                        "median_after_mse_advantage_focal_over_comparison": -0.000083,
                        "focal_lower_mse_count": 16,
                        "comparison_lower_mse_count": 24,
                        "mean_motion_energy_delta_focal_minus_comparison": 5.55,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
