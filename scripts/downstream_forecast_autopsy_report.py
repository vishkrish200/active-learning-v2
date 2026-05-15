from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from marginal_value.active_benchmark.downstream_forecast_autopsy import write_downstream_forecast_autopsy_reports


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    paths = write_downstream_forecast_autopsy_reports(
        args.run_root,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        killed_policy=args.killed_policy,
        champion_policy=args.champion_policy,
        random_policy=args.random_policy,
        submitted_policy=args.submitted_policy,
        ts2vec_policy=args.ts2vec_policy,
        quality_floor_policy=args.quality_floor_policy,
    )
    print(json.dumps({"event": "downstream_forecast_autopsy_written", **paths}, sort_keys=True), flush=True)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a downstream forecast policy decision card and diagnostics.")
    parser.add_argument("--run-root", required=True, help="Directory containing seed_*/downstream_forecast_task_report.json files.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--killed-policy", default="support_gap_window_probcover_v1")
    parser.add_argument("--champion-policy", default="window_kcenter_v1")
    parser.add_argument("--random-policy", default="quality_stratified_random_v1")
    parser.add_argument("--submitted-policy", default="submitted_full_replay_v1")
    parser.add_argument("--ts2vec-policy", default="ts2vec_kcenter_v1")
    parser.add_argument("--quality-floor-policy", default="quality_only_v1")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
