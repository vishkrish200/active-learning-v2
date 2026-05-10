import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.active_benchmark.selected_motion_regime_audit import build_selected_motion_regime_audit


class SelectedMotionRegimeAuditTests(unittest.TestCase):
    def test_motion_regime_audit_explains_window_only_vs_ts2vec_only_clips(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            _write_motion_clip(raw_dir / "win_high_rot.jsonl", gyro_scale=5.0, acc_scale=1.2)
            _write_motion_clip(raw_dir / "win_high_acc.jsonl", gyro_scale=2.0, acc_scale=3.0)
            _write_motion_clip(raw_dir / "shared_smooth.jsonl", gyro_scale=0.6, acc_scale=0.8)
            _write_motion_clip(raw_dir / "ts_low_1.jsonl", gyro_scale=0.05, acc_scale=0.1)
            _write_motion_clip(raw_dir / "ts_low_2.jsonl", gyro_scale=0.05, acc_scale=0.1)
            _write_motion_clip(raw_dir / "random_low.jsonl", gyro_scale=0.02, acc_scale=0.05)
            _write_seed(root)

            audit = build_selected_motion_regime_audit(
                root,
                raw_dirs=[raw_dir],
                focal_policy="window_kcenter_v1",
                comparison_policies=["ts2vec_kcenter_v1"],
            )

        self.assertEqual(audit["input"]["loaded_clip_count"], 6)
        window = _policy(audit, "window_kcenter_v1")
        ts2vec = _policy(audit, "ts2vec_kcenter_v1")
        self.assertGreater(window["mean_gyro_norm_p95"], ts2vec["mean_gyro_norm_p95"])
        self.assertGreater(window["mean_motion_energy"], ts2vec["mean_motion_energy"])

        contrast = audit["focal_motion_contrasts"][0]
        self.assertEqual(contrast["comparison_policy"], "ts2vec_kcenter_v1")
        self.assertGreater(contrast["mean_focal_only_gyro_norm_p95"], contrast["mean_comparison_only_gyro_norm_p95"])
        self.assertGreater(contrast["mean_focal_only_motion_energy"], contrast["mean_comparison_only_motion_energy"])
        self.assertIn("rotation_dominant", contrast["focal_only_regime_counts"])

    def test_motion_regime_cli_writes_markdown(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            for sample_id in ("win_high_rot", "shared_smooth", "ts_low_1", "random_low"):
                _write_motion_clip(
                    raw_dir / f"{sample_id}.jsonl",
                    gyro_scale=5.0 if sample_id == "win_high_rot" else 0.1,
                    acc_scale=2.0 if sample_id == "win_high_rot" else 0.2,
                )
            _write_seed(root, compact=True)
            output_json = root / "motion.json"
            output_md = root / "motion.md"
            repo_root = Path(__file__).resolve().parents[1]

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/selected_motion_regime_audit_report.py",
                    "--run-root",
                    str(root),
                    "--raw-dir",
                    str(raw_dir),
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
            self.assertIn("Selected Motion Regime Audit", output_md.read_text(encoding="utf-8"))
            self.assertIn("selected_motion_regime_audit_written", result.stdout)


def _write_seed(root: Path, *, compact: bool = False) -> None:
    seed_dir = root / "seed_17"
    seed_dir.mkdir()
    window_ids = ["win_high_rot", "win_high_acc", "shared_smooth"] if not compact else ["win_high_rot", "shared_smooth"]
    ts2vec_ids = ["shared_smooth", "ts_low_1", "ts_low_2"] if not compact else ["shared_smooth", "ts_low_1"]
    random_ids = ["random_low", "ts_low_1", "ts_low_2"] if not compact else ["random_low", "ts_low_1"]
    policy_ids = {
        "window_kcenter_v1": window_ids,
        "ts2vec_kcenter_v1": ts2vec_ids,
        "quality_stratified_random_v1": random_ids,
    }
    selected_rows = []
    forecast_rows = []
    for policy, ids in policy_ids.items():
        forecast_rows.append(
            {
                "episode_id": "episode_000",
                "fold_id": 0,
                "policy_id": policy,
                "budget_k": len(ids),
                "selected_ids": ids,
                "baseline_mse": 1.0,
                "after_mse": 0.75 if policy == "window_kcenter_v1" else 0.90,
                "absolute_mse_reduction": 0.25 if policy == "window_kcenter_v1" else 0.10,
                "relative_mse_reduction": 0.25 if policy == "window_kcenter_v1" else 0.10,
            }
        )
        for rank, sample_id in enumerate(ids, start=1):
            selected_rows.append(
                {
                    "episode_id": "episode_000",
                    "fold_id": 0,
                    "policy_id": policy,
                    "budget_k": len(ids),
                    "rank_index": rank,
                    "sample_id": sample_id,
                    "source_group_id": sample_id.split("_", 1)[0],
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
    (seed_dir / "downstream_forecast_task_report.json").write_text(
        json.dumps({"rows": forecast_rows}, indent=2),
        encoding="utf-8",
    )


def _write_motion_clip(path: Path, *, gyro_scale: float, acc_scale: float) -> None:
    rows = []
    for idx in range(64):
        phase = idx / 8.0
        rows.append(
            {
                "t_us": idx * 33333,
                "acc": [
                    9.81 + acc_scale * ((idx % 5) - 2),
                    acc_scale * phase,
                    acc_scale * ((idx % 3) - 1),
                ],
                "gyro": [
                    gyro_scale * ((idx % 7) - 3),
                    gyro_scale * 0.5,
                    gyro_scale * ((idx % 4) - 1.5),
                ],
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _policy(audit: dict[str, object], policy_id: str) -> dict[str, object]:
    for row in audit["policy_motion_profiles"]:
        if row["policy_id"] == policy_id:
            return row
    raise AssertionError(f"Missing policy profile for {policy_id}")


if __name__ == "__main__":
    unittest.main()
