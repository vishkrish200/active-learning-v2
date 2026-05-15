import json
import subprocess
import sys
import tarfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.active_benchmark.selection_mechanism_audit import build_selection_mechanism_audit


class SelectionMechanismAuditTests(unittest.TestCase):
    def test_mechanism_audit_links_selection_overlap_to_downstream_outcomes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_seed(
                root,
                "seed_17",
                [
                    _unit(
                        "episode_000",
                        window_ids=["worker001_clip001", "worker002_clip001", "worker003_clip001", "worker004_clip001"],
                        ts2vec_ids=["worker001_clip001", "worker005_clip001", "worker006_clip001", "worker007_clip001"],
                        submitted_ids=[
                            "worker001_clip001",
                            "worker002_clip001",
                            "worker003_clip001",
                            "worker004_clip001",
                        ],
                        random_ids=["worker008_clip001", "worker008_clip002", "worker009_clip001", "worker010_clip001"],
                        window_mse=0.70,
                        ts2vec_mse=0.90,
                        submitted_mse=0.72,
                        random_mse=0.95,
                    ),
                    _unit(
                        "episode_001",
                        window_ids=["worker011_clip001", "worker012_clip001", "worker013_clip001", "worker014_clip001"],
                        ts2vec_ids=["worker011_clip001", "worker012_clip001", "worker013_clip001", "worker015_clip001"],
                        submitted_ids=[
                            "worker011_clip001",
                            "worker012_clip001",
                            "worker013_clip001",
                            "worker014_clip001",
                        ],
                        random_ids=["worker016_clip001", "worker016_clip002", "worker017_clip001", "worker018_clip001"],
                        window_mse=0.84,
                        ts2vec_mse=0.78,
                        submitted_mse=0.84,
                        random_mse=0.88,
                    ),
                ],
            )

            audit = build_selection_mechanism_audit(root)

        self.assertEqual(audit["input"]["unit_count"], 2)
        window_profile = _profile(audit, "window_kcenter_v1")
        random_profile = _profile(audit, "quality_stratified_random_v1")
        self.assertEqual(window_profile["selected_row_count"], 8)
        self.assertEqual(window_profile["duplicate_source_batch_rate"], 0.0)
        self.assertEqual(random_profile["duplicate_source_batch_rate"], 1.0)

        contrast = _contrast(audit, "window_kcenter_v1", "ts2vec_kcenter_v1")
        self.assertEqual(contrast["paired_unit_count"], 2)
        self.assertAlmostEqual(contrast["mean_jaccard"], ((1 / 7) + (3 / 5)) / 2)
        self.assertEqual(contrast["policy_a_lower_mse_count"], 1)
        self.assertEqual(contrast["policy_b_lower_mse_count"], 1)
        self.assertLess(contrast["mean_jaccard_when_policy_a_wins"], contrast["mean_jaccard_when_policy_b_wins"])
        self.assertEqual(contrast["top_policy_a_only_source_groups"][0]["source_group_id"], "worker002")

        diagnostic = audit["focal_policy_episode_diagnostics"][0]
        self.assertEqual(diagnostic["comparison_policy"], "ts2vec_kcenter_v1")
        self.assertIn(diagnostic["advantage_direction"], {"focal_lower_mse", "comparison_lower_mse"})
        self.assertEqual(len(diagnostic["focal_only_selected_ids"]), 3)

    def test_mechanism_audit_cli_reads_gcp_style_archive(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "payload" / "survivors-run"
            _write_seed(
                root,
                "seed_17",
                [
                    _unit(
                        "episode_000",
                        window_ids=["worker001_clip001", "worker002_clip001", "worker003_clip001", "worker004_clip001"],
                        ts2vec_ids=["worker001_clip001", "worker005_clip001", "worker006_clip001", "worker007_clip001"],
                        submitted_ids=[
                            "worker001_clip001",
                            "worker002_clip001",
                            "worker003_clip001",
                            "worker004_clip001",
                        ],
                        random_ids=["worker008_clip001", "worker008_clip002", "worker009_clip001", "worker010_clip001"],
                        window_mse=0.70,
                        ts2vec_mse=0.90,
                        submitted_mse=0.72,
                        random_mse=0.95,
                    )
                ],
            )
            archive = Path(tmp) / "results.tgz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(root, arcname="survivors-run")
            output_json = Path(tmp) / "mechanism.json"
            output_md = Path(tmp) / "mechanism.md"
            repo_root = Path(__file__).resolve().parents[1]

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/selection_mechanism_audit_report.py",
                    "--run-archive",
                    str(archive),
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
            self.assertIn("Selection Mechanism Audit", output_md.read_text(encoding="utf-8"))
            self.assertIn("selection_mechanism_audit_written", result.stdout)


def _write_seed(root: Path, seed_name: str, units: list[dict[str, object]]) -> None:
    seed_dir = root / seed_name
    seed_dir.mkdir(parents=True, exist_ok=True)
    coverage_rows = []
    forecast_rows = []
    for fold_id, unit in enumerate(units):
        episode_id = str(unit["episode_id"])
        policy_ids = {
            "window_kcenter_v1": unit["window_ids"],
            "ts2vec_kcenter_v1": unit["ts2vec_ids"],
            "submitted_full_replay_v1": unit["submitted_ids"],
            "quality_stratified_random_v1": unit["random_ids"],
            "quality_only_v1": unit["random_ids"][:4],
        }
        mses = {
            "window_kcenter_v1": unit["window_mse"],
            "ts2vec_kcenter_v1": unit["ts2vec_mse"],
            "submitted_full_replay_v1": unit["submitted_mse"],
            "quality_stratified_random_v1": unit["random_mse"],
            "quality_only_v1": 1.0,
        }
        for policy_id, selected_ids in policy_ids.items():
            forecast_rows.append(
                {
                    "episode_id": episode_id,
                    "fold_id": fold_id,
                    "policy_id": policy_id,
                    "budget_k": 4,
                    "selected_ids": selected_ids,
                    "baseline_mse": 1.0,
                    "after_mse": mses[policy_id],
                    "absolute_mse_reduction": 1.0 - float(mses[policy_id]),
                    "relative_mse_reduction": 1.0 - float(mses[policy_id]),
                }
            )
            for rank, sample_id in enumerate(selected_ids, start=1):
                coverage_rows.append(
                    {
                        "episode_id": episode_id,
                        "fold_id": fold_id,
                        "policy_id": policy_id,
                        "budget_k": 4,
                        "rank_index": rank,
                        "sample_id": sample_id,
                        "source_group_id": sample_id.split("_clip", 1)[0],
                        "score": 1.0,
                        "quality_score": 1.0,
                        "artifact_score": 0.0,
                        "valid": True,
                        "passed_artifact_gate": True,
                    }
                )
    (seed_dir / "blind_target_coverage_benchmark_report.json").write_text(
        json.dumps({"selected_rows": coverage_rows}, indent=2),
        encoding="utf-8",
    )
    (seed_dir / "downstream_forecast_task_report.json").write_text(
        json.dumps({"rows": forecast_rows}, indent=2),
        encoding="utf-8",
    )


def _unit(
    episode_id: str,
    *,
    window_ids: list[str],
    ts2vec_ids: list[str],
    submitted_ids: list[str],
    random_ids: list[str],
    window_mse: float,
    ts2vec_mse: float,
    submitted_mse: float,
    random_mse: float,
) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "window_ids": window_ids,
        "ts2vec_ids": ts2vec_ids,
        "submitted_ids": submitted_ids,
        "random_ids": random_ids,
        "window_mse": window_mse,
        "ts2vec_mse": ts2vec_mse,
        "submitted_mse": submitted_mse,
        "random_mse": random_mse,
    }


def _profile(audit: dict[str, object], policy_id: str) -> dict[str, object]:
    for row in audit["policy_selection_profiles"]:
        if row["policy_id"] == policy_id:
            return row
    raise AssertionError(f"Missing profile for {policy_id}")


def _contrast(audit: dict[str, object], policy_a: str, policy_b: str) -> dict[str, object]:
    for row in audit["pairwise_selection_contrasts"]:
        if row["policy_a"] == policy_a and row["policy_b"] == policy_b:
            return row
        if row["policy_a"] == policy_b and row["policy_b"] == policy_a:
            flipped = dict(row)
            flipped["policy_a"] = policy_a
            flipped["policy_b"] = policy_b
            flipped["policy_a_lower_mse_count"] = row["policy_b_lower_mse_count"]
            flipped["policy_b_lower_mse_count"] = row["policy_a_lower_mse_count"]
            flipped["mean_jaccard_when_policy_a_wins"] = row["mean_jaccard_when_policy_b_wins"]
            flipped["mean_jaccard_when_policy_b_wins"] = row["mean_jaccard_when_policy_a_wins"]
            flipped["top_policy_a_only_source_groups"] = row["top_policy_b_only_source_groups"]
            flipped["top_policy_b_only_source_groups"] = row["top_policy_a_only_source_groups"]
            return flipped
    raise AssertionError(f"Missing contrast for {policy_a} vs {policy_b}")


if __name__ == "__main__":
    unittest.main()
