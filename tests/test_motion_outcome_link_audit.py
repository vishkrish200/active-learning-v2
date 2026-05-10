import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.active_benchmark.motion_outcome_link_audit import build_motion_outcome_link_audit


class MotionOutcomeLinkAuditTests(unittest.TestCase):
    def test_links_selected_motion_profiles_to_downstream_outcomes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            _write_seed(root, "seed_17", motion_suffix="a")
            _write_seed(root, "seed_23", motion_suffix="b")
            motion_path = Path(tmp) / "motion.json"
            _write_motion_audit(motion_path)

            audit = build_motion_outcome_link_audit(root, motion_path)

        window = _profile(audit, "window_kcenter_v1")
        ts2vec = _profile(audit, "ts2vec_kcenter_v1")
        random = _profile(audit, "quality_stratified_random_v1")

        self.assertLess(window["mean_after_mse"], ts2vec["mean_after_mse"])
        self.assertLess(window["mean_after_mse"], random["mean_after_mse"])
        self.assertGreater(window["mean_selected_motion_energy"], ts2vec["mean_selected_motion_energy"])
        self.assertGreater(window["mean_selected_rotation_dominant_rate"], ts2vec["mean_selected_rotation_dominant_rate"])

        window_vs_ts2vec = _contrast(audit, "ts2vec_kcenter_v1")
        self.assertGreater(window_vs_ts2vec["mean_after_mse_advantage_focal_over_comparison"], 0.0)
        self.assertGreater(window_vs_ts2vec["mean_motion_energy_delta_focal_minus_comparison"], 0.0)
        self.assertEqual(window_vs_ts2vec["focal_more_motion_and_lower_mse_count"], 4)

        assoc = _association(audit, "mean_selected_motion_energy", "after_mse")
        self.assertLess(assoc["pearson_correlation"], 0.0)
        reduction_assoc = _association(audit, "mean_selected_motion_energy", "relative_mse_reduction")
        self.assertGreater(reduction_assoc["pearson_correlation"], 0.0)
        self.assertIn("descriptive", " ".join(audit["read"]).lower())

    def test_outcome_link_script_is_directly_executable_without_pythonpath(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            _write_seed(root, "seed_17", motion_suffix="a")
            motion_path = Path(tmp) / "motion.json"
            _write_motion_audit(motion_path)
            output_json = Path(tmp) / "outcome_link.json"
            output_md = Path(tmp) / "outcome_link.md"
            repo_root = Path(__file__).resolve().parents[1]

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/motion_outcome_link_audit_report.py",
                    "--run-root",
                    str(root),
                    "--motion-audit-json",
                    str(motion_path),
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
            self.assertIn("Motion Outcome Link Audit", rendered)
            self.assertIn("focal lower MSE", rendered)
            self.assertIn("not a scalar motion-energy rule", rendered)
            self.assertIn("motion_outcome_link_audit_written", result.stdout)


def _write_seed(root: Path, seed_name: str, *, motion_suffix: str) -> None:
    seed_dir = root / seed_name
    seed_dir.mkdir(parents=True, exist_ok=True)
    policies = {
        "window_kcenter_v1": (0.70, [f"window_hi_{motion_suffix}_1", f"window_hi_{motion_suffix}_2"]),
        "ts2vec_kcenter_v1": (0.92, [f"ts2vec_lo_{motion_suffix}_1", f"ts2vec_lo_{motion_suffix}_2"]),
        "quality_stratified_random_v1": (0.95, [f"random_lo_{motion_suffix}_1", f"random_mid_{motion_suffix}_2"]),
        "submitted_full_replay_v1": (0.78, [f"submitted_mid_{motion_suffix}_1", f"window_hi_{motion_suffix}_1"]),
    }
    forecast_rows = []
    selected_rows = []
    for episode_index in range(2):
        episode_id = f"episode_{episode_index:03d}"
        for policy, (after_mse, selected_ids) in policies.items():
            adjusted_after = after_mse + episode_index * 0.01
            forecast_rows.append(
                {
                    "episode_id": episode_id,
                    "fold_id": episode_index,
                    "policy_id": policy,
                    "budget_k": 4,
                    "selected_ids": selected_ids,
                    "support_count_before": 10,
                    "support_count_after": 12,
                    "target_count": 2,
                    "history_steps": 4,
                    "horizon_steps": 1,
                    "ridge_alpha": 0.01,
                    "max_windows_per_clip": 8,
                    "baseline_mse": 1.0,
                    "after_mse": adjusted_after,
                    "absolute_mse_reduction": 1.0 - adjusted_after,
                    "relative_mse_reduction": 1.0 - adjusted_after,
                }
            )
            for rank, sample_id in enumerate(selected_ids, start=1):
                selected_rows.append(
                    {
                        "episode_id": episode_id,
                        "fold_id": episode_index,
                        "policy_id": policy,
                        "budget_k": 4,
                        "rank_index": rank,
                        "sample_id": sample_id,
                        "source_group_id": sample_id.rsplit("_", 2)[0],
                        "score": 1.0,
                        "quality_score": 1.0,
                        "artifact_score": 0.0,
                        "valid": True,
                        "passed_artifact_gate": True,
                    }
                )
    (seed_dir / "downstream_forecast_task_report.json").write_text(
        json.dumps({"rows": forecast_rows, "summary": {}, "decision": {}, "input": {}}, indent=2),
        encoding="utf-8",
    )
    (seed_dir / "blind_target_coverage_benchmark_report.json").write_text(
        json.dumps({"selected_rows": selected_rows}, indent=2),
        encoding="utf-8",
    )


def _write_motion_audit(path: Path) -> None:
    rows = []
    for suffix in ("a", "b"):
        rows.extend(
            [
                _motion(f"window_hi_{suffix}_1", 12.0, 1.20, "rotation_dominant"),
                _motion(f"window_hi_{suffix}_2", 10.0, 1.10, "rotation_dominant"),
                _motion(f"ts2vec_lo_{suffix}_1", 2.0, 0.20, "low_motion"),
                _motion(f"ts2vec_lo_{suffix}_2", 2.5, 0.25, "mixed_motion"),
                _motion(f"random_lo_{suffix}_1", 3.0, 0.30, "low_motion"),
                _motion(f"random_mid_{suffix}_2", 5.0, 0.55, "mixed_motion"),
                _motion(f"submitted_mid_{suffix}_1", 6.0, 0.65, "mixed_motion"),
            ]
        )
    path.write_text(
        json.dumps(
            {
                "input": {"budget_k": 4},
                "clip_motion_features": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _motion(sample_id: str, motion_energy: float, gyro_p95: float, regime: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "motion_energy": motion_energy,
        "gyro_norm_p95": gyro_p95,
        "acc_delta_norm_p95": motion_energy / 5.0,
        "gyro_delta_norm_p95": gyro_p95 / 5.0,
        "gyro_energy_fraction": 0.6 if regime == "rotation_dominant" else 0.2,
        "stationary_fraction": 0.0 if regime != "low_motion" else 0.8,
        "quality_score": 1.0,
        "regime_label": regime,
    }


def _profile(audit: dict[str, object], policy_id: str) -> dict[str, object]:
    for row in audit["policy_motion_outcome_profiles"]:
        if row["policy_id"] == policy_id:
            return row
    raise AssertionError(f"Missing profile for {policy_id}")


def _contrast(audit: dict[str, object], comparison_policy: str) -> dict[str, object]:
    for row in audit["focal_pairwise_motion_outcome_contrasts"]:
        if row["comparison_policy"] == comparison_policy:
            return row
    raise AssertionError(f"Missing contrast for {comparison_policy}")


def _association(audit: dict[str, object], feature: str, outcome: str) -> dict[str, object]:
    for row in audit["motion_outcome_associations"]:
        if row["feature"] == feature and row["outcome"] == outcome:
            return row
    raise AssertionError(f"Missing association for {feature} vs {outcome}")


if __name__ == "__main__":
    unittest.main()
