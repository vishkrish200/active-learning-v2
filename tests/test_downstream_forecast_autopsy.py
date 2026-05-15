import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.active_benchmark.downstream_forecast_autopsy import build_downstream_forecast_autopsy


class DownstreamForecastAutopsyTests(unittest.TestCase):
    def test_autopsy_kills_probcover_and_keeps_window_champion(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_seed(root, "seed_17", baseline=1.0, probcover=0.92, window=0.80, random=0.90)
            _write_seed(root, "seed_23", baseline=1.0, probcover=0.91, window=0.82, random=0.89)

            autopsy = build_downstream_forecast_autopsy(root)

        self.assertEqual(autopsy["result_card"]["decision"], "kill_probcover_promote_window_kcenter")
        self.assertEqual(autopsy["policy_final_summary"][0]["policy_id"], "window_kcenter_v1")
        self.assertEqual(autopsy["result_card"]["policy_decisions"]["support_gap_window_probcover_v1"], "failed; archive")
        self.assertTrue(autopsy["hygiene"]["leakage_all_ok"])
        self.assertEqual(autopsy["hygiene"]["selected_target_leak_count_total"], 0)

        probcover_vs_window = _pairwise(autopsy, "support_gap_window_probcover_v1", "window_kcenter_v1")
        self.assertLess(probcover_vs_window["mean_after_mse_advantage_a_over_b"], 0.0)
        self.assertEqual(probcover_vs_window["policy_a_win_count"], 0)

        probcover_deltas = {row["comparison_policy"]: row for row in autopsy["probcover_episode_deltas"]}
        self.assertLess(probcover_deltas["window_kcenter_v1"]["mean_after_mse_advantage"], 0.0)
        self.assertLess(probcover_deltas["quality_stratified_random_v1"]["mean_after_mse_advantage"], 0.0)

        probcover_selection = {
            row["policy_id"]: row
            for row in autopsy["selection_diagnostics"]
        }["support_gap_window_probcover_v1"]
        self.assertEqual(probcover_selection["valid_rate"], 1.0)
        self.assertEqual(probcover_selection["mean_artifact_score"], 0.0)

    def test_autopsy_script_is_directly_executable_without_pythonpath(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            _write_seed(root, "seed_17", baseline=1.0, probcover=0.92, window=0.80, random=0.90)
            output_json = Path(tmp) / "autopsy.json"
            output_md = Path(tmp) / "autopsy.md"
            repo_root = Path(__file__).resolve().parents[1]

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/downstream_forecast_autopsy_report.py",
                    "--run-root",
                    str(root),
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
            self.assertIn("Downstream Forecast Policy Decision Card", output_md.read_text(encoding="utf-8"))
            self.assertIn("downstream_forecast_autopsy_written", result.stdout)

    def test_survivor_only_autopsy_has_validation_next_steps_not_probcover_tuning(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_seed(root, "seed_109", baseline=1.0, probcover=None, window=0.80, random=0.90)
            _write_seed(root, "seed_127", baseline=1.0, probcover=None, window=0.82, random=0.89)

            autopsy = build_downstream_forecast_autopsy(root)

        self.assertEqual(autopsy["result_card"]["decision"], "survivor_confirmation_window_kcenter")
        self.assertNotIn("support_gap_window_probcover_v1", autopsy["input"]["policies"])
        self.assertTrue(any("not a ProbCover appeal" in item for item in autopsy["result_card"]["reads"]))
        next_steps = "\n".join(autopsy["result_card"]["next_steps"])
        self.assertIn("validation-only", next_steps)
        self.assertIn("do not add or tune policies", next_steps)
        self.assertNotIn("ProbCover thresholds", next_steps)


def _write_seed(root: Path, seed_name: str, *, baseline: float, probcover: float | None, window: float, random: float) -> None:
    seed_dir = root / seed_name
    seed_dir.mkdir(parents=True, exist_ok=True)
    policies = {
        "quality_stratified_random_v1": random,
        "quality_only_v1": baseline,
        "window_kcenter_v1": window,
        "submitted_full_replay_v1": window + 0.01,
        "ts2vec_kcenter_v1": random - 0.02,
    }
    if probcover is not None:
        policies["support_gap_window_probcover_v1"] = probcover
    rows = []
    for episode_index in range(2):
        episode_id = f"episode_{episode_index:03d}"
        for policy, final_mse in policies.items():
            rows.append(_forecast_row(seed_name, episode_id, policy, budget=0, baseline=baseline, after=baseline, selected=[]))
            rows.append(
                _forecast_row(
                    seed_name,
                    episode_id,
                    policy,
                    budget=4,
                    baseline=baseline,
                    after=final_mse + episode_index * 0.01,
                    selected=[f"{seed_name}_{policy}_clip{rank}" for rank in range(4)],
                )
            )
    (seed_dir / "downstream_forecast_task_report.json").write_text(
        json.dumps({"rows": rows, "summary": {}, "decision": {}, "input": {}}, indent=2),
        encoding="utf-8",
    )
    selected_rows = []
    for row in rows:
        if row["budget_k"] != 4:
            continue
        for rank, sample_id in enumerate(row["selected_ids"], start=1):
            selected_rows.append(
                {
                    "episode_id": row["episode_id"],
                    "fold_id": row["fold_id"],
                    "policy_id": row["policy_id"],
                    "budget_k": 4,
                    "rank_index": rank,
                    "sample_id": sample_id,
                    "source_group_id": sample_id.rsplit("_clip", 1)[0],
                    "score": 1.0,
                    "quality_score": 1.0,
                    "artifact_score": 0.0,
                    "valid": True,
                    "passed_artifact_gate": True,
                }
            )
    (seed_dir / "blind_target_coverage_benchmark_report.json").write_text(
        json.dumps({"selected_rows": selected_rows}, indent=2),
        encoding="utf-8",
    )
    (seed_dir / "coverage_proof_summary.json").write_text(
        json.dumps(
            {
                "leakage_ok": True,
                "selected_invalid_rate_max": 0.0,
                "selected_out_of_pool_count_total": 0,
                "selected_target_leak_count_total": 0,
                "selected_duplicate_clip_count_total": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _forecast_row(
    seed_name: str,
    episode_id: str,
    policy: str,
    *,
    budget: int,
    baseline: float,
    after: float,
    selected: list[str],
) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "fold_id": int(episode_id.rsplit("_", 1)[1]),
        "policy_id": policy,
        "budget_k": budget,
        "selected_ids": selected,
        "support_count_before": 10,
        "support_count_after": 10 + len(selected),
        "target_count": 2,
        "history_steps": 4,
        "horizon_steps": 1,
        "ridge_alpha": 0.01,
        "max_windows_per_clip": 8,
        "baseline_mse": baseline,
        "after_mse": after,
        "absolute_mse_reduction": baseline - after,
        "relative_mse_reduction": (baseline - after) / baseline,
        "_seed_name": seed_name,
    }


def _pairwise(autopsy: dict[str, object], policy_a: str, policy_b: str) -> dict[str, object]:
    for row in autopsy["pairwise_final_deltas"]:
        if row["policy_a"] == policy_a and row["policy_b"] == policy_b:
            return row
        if row["policy_a"] == policy_b and row["policy_b"] == policy_a:
            flipped = dict(row)
            flipped["policy_a"] = policy_a
            flipped["policy_b"] = policy_b
            flipped["mean_after_mse_advantage_a_over_b"] = -float(row["mean_after_mse_advantage_a_over_b"])
            flipped["policy_a_win_count"] = int(row["policy_b_win_count"])
            return flipped
    raise AssertionError(f"Missing pairwise row for {policy_a} vs {policy_b}")


if __name__ == "__main__":
    unittest.main()
