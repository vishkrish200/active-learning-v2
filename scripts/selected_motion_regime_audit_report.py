#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from marginal_value.active_benchmark.selected_motion_regime_audit import (  # noqa: E402
    DEFAULT_COMPARISON_POLICIES,
    DEFAULT_FOCAL_POLICY,
    write_selected_motion_regime_audit_reports,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Join selected clips back to raw IMU motion-regime summaries.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-root", type=Path, help="Run directory containing seed_*/ reports.")
    source.add_argument("--run-archive", type=Path, help="GCP results .tgz archive containing seed_*/ reports.")
    parser.add_argument("--raw-dir", action="append", default=[], help="Directory with sample_id.jsonl raw clips.")
    parser.add_argument("--manifest", action="append", default=[], help="Manifest URL file used to resolve selected sample ids.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--budget-k", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=900)
    parser.add_argument("--sample-rate", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--focal-policy", default=DEFAULT_FOCAL_POLICY)
    parser.add_argument("--comparison-policy", action="append", dest="comparison_policies")
    args = parser.parse_args(argv)

    comparison_policies = args.comparison_policies or list(DEFAULT_COMPARISON_POLICIES)
    outputs = write_selected_motion_regime_audit_reports(
        args.run_root or args.run_archive,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        raw_dirs=args.raw_dir,
        manifest_paths=args.manifest,
        budget_k=args.budget_k,
        max_samples=args.max_samples,
        sample_rate=args.sample_rate,
        workers=args.workers,
        timeout_seconds=args.timeout_seconds,
        focal_policy=args.focal_policy,
        comparison_policies=comparison_policies,
    )
    print(json.dumps({"event": "selected_motion_regime_audit_written", **outputs}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
