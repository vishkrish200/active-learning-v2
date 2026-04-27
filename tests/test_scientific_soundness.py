import tempfile
import unittest
from pathlib import Path

from marginal_value.eval.scientific_soundness import (
    DEFAULT_CANDIDATE_POLICY,
    evaluate_scientific_soundness,
    render_scientific_soundness_markdown,
    write_scientific_soundness_report,
)


class ScientificSoundnessTests(unittest.TestCase):
    def test_verdict_separates_raw_shape_support_from_broad_behavior_failure(self):
        verdict = evaluate_scientific_soundness(_report())

        self.assertEqual(verdict["claim_status"]["artifact_safe_raw_shape_supported"], "pass")
        self.assertEqual(verdict["claim_status"]["broad_behavior_discovery_supported"], "fail")
        self.assertEqual(verdict["overall_status"], "conditional")

        gates = {gate["name"]: gate for gate in verdict["gates"]}
        self.assertEqual(gates["primary_vs_simple_controls"]["status"], "pass")
        self.assertEqual(gates["physical_validity"]["status"], "pass")
        self.assertEqual(gates["uncapped_regression"]["status"], "pass")
        self.assertEqual(gates["independent_temporal_vs_controls"]["status"], "fail")

    def test_markdown_renders_claims_and_failed_independent_gate(self):
        markdown = render_scientific_soundness_markdown(evaluate_scientific_soundness(_report()))

        self.assertIn("# Scientific Soundness Verdict", markdown)
        self.assertIn("artifact_safe_raw_shape_supported", markdown)
        self.assertIn("broad_behavior_discovery_supported", markdown)
        self.assertIn("| independent_temporal_vs_controls | fail |", markdown)

    def test_write_report_persists_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "verdict.json"
            md_path = Path(tmp) / "verdict.md"

            result = write_scientific_soundness_report(_report(), json_path=json_path, markdown_path=md_path)

            self.assertEqual(result, {"json_path": str(json_path), "markdown_path": str(md_path)})
            self.assertIn('"overall_status": "conditional"', json_path.read_text(encoding="utf-8"))
            self.assertIn("Scientific Soundness Verdict", md_path.read_text(encoding="utf-8"))


def _policy(
    temporal: float,
    raw: float,
    window: float = 0.0,
    min_quality: float = 0.9,
    stationary_over_90: float = 0.0,
    max_abs_over_60: float = 0.0,
):
    return {
        "coverage@100": {
            "window_mean_std_pool": {"relative_coverage_gain": window},
            "temporal_order": {"relative_coverage_gain": temporal},
            "raw_shape_stats": {"relative_coverage_gain": raw},
        },
        "coverage@200": {
            "window_mean_std_pool": {"relative_coverage_gain": window + 0.01},
            "temporal_order": {"relative_coverage_gain": temporal + 0.01},
            "raw_shape_stats": {"relative_coverage_gain": raw + 0.01},
        },
        "selection@100": {
            "min_quality": min_quality,
            "mean_quality": 0.95,
            "stationary_fraction_over_90": stationary_over_90,
            "max_abs_value_over_60": max_abs_over_60,
            "largest_source_group_fraction": 0.1,
        },
        "selection@200": {
            "min_quality": min_quality,
            "mean_quality": 0.95,
            "stationary_fraction_over_90": stationary_over_90,
            "max_abs_value_over_60": max_abs_over_60,
            "largest_source_group_fraction": 0.1,
        },
    }


def _report():
    candidate = DEFAULT_CANDIDATE_POLICY
    uncapped = "quality_gated_old_novelty_raw_shape_stats_q85_stat90_abs60"
    return {
        "mode": "unit",
        "n_rows": 40,
        "n_source_groups": 8,
        "folds": [
            {
                "policies": {
                    candidate: _policy(temporal=0.02, raw=0.30, window=0.02),
                    uncapped: _policy(temporal=0.019, raw=0.298, window=0.02),
                    "quality_only": _policy(temporal=0.03, raw=0.10, window=0.01),
                    "old_novelty_only": _policy(temporal=0.05, raw=0.10, window=0.01),
                    "random_high_quality": _policy(temporal=0.01, raw=0.09, window=0.01),
                }
            },
            {
                "policies": {
                    candidate: _policy(temporal=0.021, raw=0.31, window=0.021),
                    uncapped: _policy(temporal=0.020, raw=0.308, window=0.021),
                    "quality_only": _policy(temporal=0.031, raw=0.11, window=0.011),
                    "old_novelty_only": _policy(temporal=0.051, raw=0.11, window=0.011),
                    "random_high_quality": _policy(temporal=0.011, raw=0.10, window=0.011),
                }
            },
            {
                "policies": {
                    candidate: _policy(temporal=0.019, raw=0.29, window=0.019),
                    uncapped: _policy(temporal=0.018, raw=0.288, window=0.019),
                    "quality_only": _policy(temporal=0.029, raw=0.09, window=0.009),
                    "old_novelty_only": _policy(temporal=0.049, raw=0.09, window=0.009),
                    "random_high_quality": _policy(temporal=0.009, raw=0.08, window=0.009),
                }
            },
            {
                "policies": {
                    candidate: _policy(temporal=0.022, raw=0.32, window=0.022),
                    uncapped: _policy(temporal=0.021, raw=0.318, window=0.022),
                    "quality_only": _policy(temporal=0.032, raw=0.12, window=0.012),
                    "old_novelty_only": _policy(temporal=0.052, raw=0.12, window=0.012),
                    "random_high_quality": _policy(temporal=0.012, raw=0.11, window=0.012),
                }
            },
        ],
    }


if __name__ == "__main__":
    unittest.main()
