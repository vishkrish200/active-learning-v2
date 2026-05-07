from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from marginal_value.active_benchmark.downstream_pairing_audit import (
    build_supervised_pairing_audit,
    write_supervised_pairing_audit_reports,
)


def main() -> None:
    args = _parse_args()
    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir) if args.output_dir else run_root
    reports_by_seed = load_seed_reports(run_root)
    audit = build_supervised_pairing_audit(reports_by_seed, random_policy=args.random_policy)
    paths = write_supervised_pairing_audit_reports(audit, output_dir)
    print(
        json.dumps(
            {
                "event": "downstream_supervised_pairing_audit_written",
                "run_root": str(run_root.resolve()),
                "seed_count": len(reports_by_seed),
                "json": str(paths["json"].resolve()),
                "markdown": str(paths["markdown"].resolve()),
                "decision": audit["decision"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def load_seed_reports(run_root: str | Path) -> dict[int, dict[str, Any]]:
    root = Path(run_root)
    reports: dict[int, dict[str, Any]] = {}
    for seed_dir in sorted(root.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        seed_text = seed_dir.name.removeprefix("seed_")
        if not seed_text.isdigit():
            continue
        report_path = seed_dir / "downstream_supervised_smoke_report.json"
        if not report_path.exists():
            continue
        reports[int(seed_text)] = json.loads(report_path.read_text(encoding="utf-8"))
    if not reports:
        raise FileNotFoundError(f"No seed_*/downstream_supervised_smoke_report.json files found under {root}.")
    return reports


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit supervised downstream rows with paired common-baseline deltas.")
    parser.add_argument("--run-root", required=True, help="Run root containing seed_*/downstream_supervised_smoke_report.json files.")
    parser.add_argument("--output-dir", default="", help="Output directory for audit JSON/Markdown. Defaults to --run-root.")
    parser.add_argument("--random-policy", default="random_valid")
    return parser.parse_args()


if __name__ == "__main__":
    main()
