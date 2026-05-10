#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from marginal_value.active_benchmark.scientific_claim_memo import write_scientific_claim_memo_reports  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a frozen scientific claim memo from locked IMU benchmark artifacts.")
    parser.add_argument("--decision-card-json", type=Path, required=True)
    parser.add_argument("--selection-mechanism-json", type=Path, required=True)
    parser.add_argument("--selected-motion-json", type=Path, required=True)
    parser.add_argument("--motion-outcome-json", type=Path, required=True)
    parser.add_argument("--advisor-markdown", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args(argv)

    outputs = write_scientific_claim_memo_reports(
        args.decision_card_json,
        args.selection_mechanism_json,
        args.selected_motion_json,
        args.motion_outcome_json,
        advisor_markdown=args.advisor_markdown,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )
    print(json.dumps({"event": "scientific_claim_memo_written", **outputs}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
