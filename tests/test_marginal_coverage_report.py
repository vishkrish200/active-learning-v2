import unittest

from marginal_value.eval.marginal_coverage_report import (
    coverage_value,
    paired_delta_rows,
    policy_metric_rows,
    render_markdown_report,
)


class MarginalCoverageReportTests(unittest.TestCase):
    def test_coverage_value_handles_deterministic_and_random_metrics(self):
        deterministic = {"coverage@100": {"raw_shape_stats": {"relative_coverage_gain": 0.12}}}
        random = {"coverage@100": {"raw_shape_stats": {"relative_coverage_gain_mean": 0.08}}}

        self.assertEqual(coverage_value(deterministic, k=100, representation="raw_shape_stats"), 0.12)
        self.assertEqual(coverage_value(random, k=100, representation="raw_shape_stats"), 0.08)

    def test_policy_rows_and_paired_deltas_use_primary_representations(self):
        report = _report()

        rows = policy_metric_rows(report, policies=["a", "b"], k_values=[100])
        deltas = paired_delta_rows(report, challengers=["a"], baselines=["b"], k_values=[100])

        by_policy = {row["policy"]: row for row in rows}
        self.assertAlmostEqual(by_policy["a"]["primary_average"], 0.3)
        self.assertAlmostEqual(by_policy["b"]["primary_average"], 0.2)
        self.assertAlmostEqual(deltas[0]["mean_delta"], 0.1)
        self.assertEqual(deltas[0]["fold_wins"], 2)
        self.assertEqual(deltas[0]["folds"], 2)

    def test_render_markdown_report_includes_mean_and_delta_tables(self):
        markdown = render_markdown_report(
            _report(),
            title="Unit Diagnostic",
            policies=["a", "b"],
            challengers=["a"],
            baselines=["b"],
            k_values=[100],
        )

        self.assertIn("# Unit Diagnostic", markdown)
        self.assertIn("## Mean Coverage", markdown)
        self.assertIn("## Paired Primary Deltas", markdown)
        self.assertIn("| a | 100 |", markdown)


def _policy(temporal: float, raw: float, window: float = 0.0, min_quality: float = 0.5):
    return {
        "coverage@100": {
            "window_mean_std_pool": {"relative_coverage_gain": window},
            "temporal_order": {"relative_coverage_gain": temporal},
            "raw_shape_stats": {"relative_coverage_gain": raw},
        },
        "selection@100": {
            "min_quality": min_quality,
            "mean_quality": 0.9,
            "largest_source_group_fraction": 0.1,
            "unique_source_groups": 10,
        },
    }


def _report():
    return {
        "mode": "unit",
        "n_rows": 10,
        "n_source_groups": 4,
        "folds": [
            {"policies": {"a": _policy(0.2, 0.4), "b": _policy(0.1, 0.3)}},
            {"policies": {"a": _policy(0.3, 0.3), "b": _policy(0.2, 0.2)}},
        ],
    }


if __name__ == "__main__":
    unittest.main()
