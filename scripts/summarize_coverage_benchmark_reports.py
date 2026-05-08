from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from marginal_value.active_benchmark.coverage_decision import write_coverage_decision_reports


def main() -> None:
    args = _parse_args()
    report_paths = [Path(path) for path in args.reports]
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in report_paths]
    report_names = _parse_names(args.report_names, report_paths)
    paths = write_coverage_decision_reports(
        reports,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        report_names=report_names,
        baseline_policy=args.baseline_policy,
        oracle_policy=args.oracle_policy,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps({"event": "coverage_decision_report_done", **paths}, sort_keys=True), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize one or more blind-target coverage benchmark JSON reports.")
    parser.add_argument("reports", nargs="+", help="blind_target_coverage_benchmark_report.json paths.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--report-names", default="", help="Optional comma-separated names matching the report paths.")
    parser.add_argument("--baseline-policy", default="quality_stratified_random_v1")
    parser.add_argument("--oracle-policy", default="oracle_greedy_eval_view_v1")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260508)
    return parser.parse_args()


def _parse_names(value: str, report_paths: list[Path]) -> list[str]:
    if str(value).strip():
        names = [item.strip() for item in str(value).split(",") if item.strip()]
        if len(names) != len(report_paths):
            raise ValueError("--report-names must have the same number of entries as reports.")
        return names
    return [path.parent.name or f"report_{index:03d}" for index, path in enumerate(report_paths)]


if __name__ == "__main__":
    main()
