from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from marginal_value.active_benchmark.policy_autopsy import write_policy_autopsy_reports


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    paths = write_policy_autopsy_reports(
        args.run_root,
        aggregate_path=args.aggregate_path,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )
    print(json.dumps({"event": "policy_autopsy_written", **paths}, sort_keys=True))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a policy autopsy report from offline active-learning benchmark artifacts.")
    parser.add_argument("--run-root", required=True, help="Directory containing seed_*/offline_active_benchmark_report.json files.")
    parser.add_argument("--aggregate-path", default=None, help="Optional aggregate_proof_summary.json path.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
