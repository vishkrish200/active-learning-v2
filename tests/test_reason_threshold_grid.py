import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.eval.reason_threshold_grid import (
    evaluate_reason_threshold_grid,
    run_reason_threshold_grid,
    validate_reason_threshold_grid_config,
)


class ReasonThresholdGridTests(unittest.TestCase):
    def test_reason_threshold_grid_sweeps_component_and_delta_thresholds(self):
        rows = _diagnostic_rows()

        report = evaluate_reason_threshold_grid(
            rows,
            component_thresholds=[0.5, 0.8],
            delta_thresholds=[0.02, 0.05],
            top_ks=[3],
            low_quality_threshold=0.45,
            min_new_density_score=0.35,
            target_top_k=3,
            target_rare_temporal_count=1,
        )

        loose = report["variants"]["component_0.500_delta_0.020"]
        strict = report["variants"]["component_0.800_delta_0.050"]
        self.assertEqual(loose["reason_code_counts"]["RARE_TEMPORAL_COMPOSITION"], 2)
        self.assertEqual(strict["reason_code_counts"]["RARE_TEMPORAL_COMPOSITION"], 1)
        self.assertEqual(loose["top_k"]["3"]["low_quality_count"], 0)
        self.assertEqual(strict["top_k"]["3"]["reason_code_counts"]["COHESIVE_NEW_WORKFLOW"], 1)
        self.assertEqual(report["recommended"]["variant"], "component_0.800_delta_0.050")

    def test_reason_threshold_grid_accepts_generators(self):
        report = evaluate_reason_threshold_grid(
            _diagnostic_rows(),
            component_thresholds=(value for value in [0.5, 0.8]),
            delta_thresholds=(value for value in [0.02, 0.05]),
            top_ks=(value for value in [3]),
            low_quality_threshold=0.45,
            min_new_density_score=0.35,
            target_top_k=3,
            target_rare_temporal_count=1,
        )

        self.assertEqual(report["grid"]["grammar_component_thresholds"], [0.5, 0.8])
        self.assertEqual(report["grid"]["grammar_delta_thresholds"], [0.02, 0.05])
        self.assertEqual(report["grid"]["top_ks"], [3])
        self.assertEqual(len(report["variants"]), 4)

    def test_run_reason_threshold_grid_writes_report_and_summary(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            diagnostics_path = root / "diagnostics.csv"
            output_dir = root / "out"
            _write_diagnostics(diagnostics_path, _diagnostic_rows())
            config = {
                "execution": {"provider": "local"},
                "artifacts": {
                    "diagnostics_path": str(diagnostics_path),
                    "output_dir": str(output_dir),
                },
                "grid": {
                    "grammar_component_thresholds": [0.5, 0.8],
                    "grammar_delta_thresholds": [0.02, 0.05],
                    "min_new_density_score": 0.35,
                    "target_top_k": 3,
                    "target_rare_temporal_count": 1,
                },
                "audit": {
                    "top_ks": [3],
                    "low_quality_threshold": 0.45,
                },
            }

            result = run_reason_threshold_grid(config)

            self.assertEqual(result["n_rows"], 5)
            self.assertTrue(Path(result["report_path"]).exists())
            self.assertTrue(Path(result["summary_path"]).exists())
            written = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(written["recommended"]["variant"], "component_0.800_delta_0.050")
            with Path(result["summary_path"]).open("r", encoding="utf-8") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(len(summary_rows), 4)

    def test_reason_threshold_grid_config_requires_absolute_artifact_paths(self):
        with self.assertRaises(ValueError):
            validate_reason_threshold_grid_config(
                {
                    "execution": {"provider": "local"},
                    "artifacts": {"diagnostics_path": "diagnostics.csv", "output_dir": "/tmp/out"},
                }
            )


def _diagnostic_rows() -> list[dict[str, object]]:
    return [
        {
            "worker_id": "cohesive",
            "rank": 1,
            "quality_score": 0.99,
            "old_novelty_score": 0.80,
            "new_density_score": 0.80,
            "grammar_score_component": 0.95,
            "grammar_promotion_delta": 0.10,
            "new_cluster_id": 0,
        },
        {
            "worker_id": "rare-strong",
            "rank": 2,
            "quality_score": 0.98,
            "old_novelty_score": 0.20,
            "new_density_score": 0.70,
            "grammar_score_component": 0.90,
            "grammar_promotion_delta": 0.08,
            "new_cluster_id": 1,
        },
        {
            "worker_id": "rare-borderline",
            "rank": 3,
            "quality_score": 0.97,
            "old_novelty_score": 0.20,
            "new_density_score": 0.70,
            "grammar_score_component": 0.60,
            "grammar_promotion_delta": 0.03,
            "new_cluster_id": 2,
        },
        {
            "worker_id": "redundant",
            "rank": 4,
            "quality_score": 0.96,
            "old_novelty_score": 0.10,
            "new_density_score": 0.90,
            "grammar_score_component": 0.90,
            "grammar_promotion_delta": 0.0,
            "new_cluster_id": 2,
        },
        {
            "worker_id": "low-quality",
            "rank": 5,
            "quality_score": 0.20,
            "old_novelty_score": 0.80,
            "new_density_score": 0.80,
            "grammar_score_component": 0.95,
            "grammar_promotion_delta": 0.10,
            "new_cluster_id": 3,
        },
    ]


def _write_diagnostics(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
