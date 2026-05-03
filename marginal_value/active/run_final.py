from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from marginal_value.logging_utils import log_event


DEFAULT_PRIMARY_ID_COLUMN = "new_worker_id"
DEFAULT_PRIMARY_NAME = "ranked_new_clips.csv"
DEFAULT_PRIMARY_NEW_WORKER_NAME = "ranked_new_clips_new_worker_id.csv"
DEFAULT_BACKUP_WORKER_NAME = "ranked_new_clips_worker_id.csv"


def package_final_submission(config: Mapping[str, Any]) -> dict[str, Any]:
    source_config = _required_mapping(config, "source_artifacts")
    artifacts_config = _required_mapping(config, "artifacts")
    method_config = dict(config.get("method", {}))
    validation_config = dict(config.get("validation", {}))

    source_dir = Path(str(source_config.get("source_dir", "")))
    output_dir = Path(str(artifacts_config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_path = _resolve_artifact(source_dir, source_config, "primary_submission")
    backup_path = _resolve_artifact(source_dir, source_config, "backup_worker_submission")
    diagnostics_path = _resolve_artifact(source_dir, source_config, "diagnostics")
    selector_report_path = _resolve_artifact(source_dir, source_config, "selector_report")
    primary_id_column = str(method_config.get("primary_submission_id_column", DEFAULT_PRIMARY_ID_COLUMN))
    expected_count = _optional_int(validation_config.get("expected_count", 2000))

    log_event(
        "active_run_final",
        "start",
        source_dir=str(source_dir),
        output_dir=str(output_dir),
        primary_submission=str(primary_path),
        expected_count=expected_count,
    )

    primary_rows = _read_csv(primary_path)
    backup_rows = _read_csv(backup_path)
    diagnostic_rows = _read_csv(diagnostics_path)
    _validate_submission_rows(primary_rows, id_column=primary_id_column, expected_count=expected_count)
    _validate_submission_rows(backup_rows, id_column="worker_id", expected_count=expected_count)

    primary_out = output_dir / DEFAULT_PRIMARY_NAME
    primary_new_worker_out = output_dir / DEFAULT_PRIMARY_NEW_WORKER_NAME
    backup_worker_out = output_dir / DEFAULT_BACKUP_WORKER_NAME
    diagnostics_out = output_dir / "diagnostics.csv"
    selector_report_out = output_dir / "selector_report.json"
    selector_config_out = output_dir / "selector_config.json"
    feature_schema_out = output_dir / "feature_schema.json"
    package_report_out = output_dir / "final_package_report.json"
    readme_out = output_dir / "README_final.md"

    _copy(primary_path, primary_out)
    _copy(primary_path, primary_new_worker_out)
    _copy(backup_path, backup_worker_out)
    _copy(diagnostics_path, diagnostics_out)
    _copy(selector_report_path, selector_report_out)

    optional_outputs = _copy_optional_reports(
        source_config,
        output_dir=output_dir,
        keys={
            "validation_report": "validation_report.json",
            "support_audit": "support_audit.json",
            "stability_report": "stability_report.json",
        },
    )
    selector_config = _selector_config(
        config,
        n_rows=len(primary_rows),
        primary_id_column=primary_id_column,
        output_files={
            "primary_submission": primary_out,
            "primary_new_worker_submission": primary_new_worker_out,
            "backup_worker_submission": backup_worker_out,
            "diagnostics": diagnostics_out,
            "selector_report": selector_report_out,
            **optional_outputs,
        },
    )
    selector_config_out.write_text(json.dumps(selector_config, indent=2, sort_keys=True), encoding="utf-8")
    feature_schema = _feature_schema(primary_rows, backup_rows, diagnostic_rows)
    feature_schema_out.write_text(json.dumps(feature_schema, indent=2, sort_keys=True), encoding="utf-8")

    package_report = {
        "status": "ready",
        "n_rows": len(primary_rows),
        "primary_id_column": primary_id_column,
        "method_name": method_config.get("name", "artifact-gate exact-window blend"),
        "artifacts": {
            "primary_submission": str(primary_out),
            "primary_new_worker_submission": str(primary_new_worker_out),
            "backup_worker_submission": str(backup_worker_out),
            "diagnostics": str(diagnostics_out),
            "selector_report": str(selector_report_out),
            "selector_config": str(selector_config_out),
            "feature_schema": str(feature_schema_out),
            "readme": str(readme_out),
            "package_report": str(package_report_out),
            **{key: str(path) for key, path in optional_outputs.items()},
        },
    }
    package_report_out.write_text(json.dumps(package_report, indent=2, sort_keys=True), encoding="utf-8")
    readme_out.write_text(_readme_text(config, package_report, feature_schema), encoding="utf-8")
    log_event("active_run_final", "done", output_dir=str(output_dir), n_rows=len(primary_rows))
    return package_report


def validate_final_package(package_dir: str | Path, *, expected_count: int = 2000) -> dict[str, Any]:
    root = Path(package_dir)
    primary_path = root / DEFAULT_PRIMARY_NAME
    backup_path = root / DEFAULT_BACKUP_WORKER_NAME
    diagnostics_path = root / "diagnostics.csv"
    report_path = root / "final_package_report.json"
    primary_rows = _read_csv(primary_path)
    backup_rows = _read_csv(backup_path)
    diagnostic_rows = _read_csv(diagnostics_path)
    _validate_submission_rows(primary_rows, id_column=DEFAULT_PRIMARY_ID_COLUMN, expected_count=expected_count)
    _validate_submission_rows(backup_rows, id_column="worker_id", expected_count=expected_count)
    if len(diagnostic_rows) < len(primary_rows):
        raise ValueError("diagnostics.csv has fewer rows than the primary submission.")
    if not report_path.exists():
        raise FileNotFoundError(f"Missing final package report: {report_path}")
    return {
        "status": "ready",
        "n_rows": len(primary_rows),
        "package_dir": str(root),
        "primary_submission": str(primary_path),
        "backup_worker_submission": str(backup_path),
    }


def load_final_package_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_submission_rows(rows: Sequence[Mapping[str, str]], *, id_column: str, expected_count: int | None) -> None:
    if not rows:
        raise ValueError("Submission has no rows.")
    required = {"rank", "score", id_column}
    missing = sorted(required - set(rows[0].keys()))
    if missing:
        raise ValueError(f"Submission is missing required columns: {missing}")
    ranks = [_int(row.get("rank")) for row in rows]
    expected_ranks = list(range(1, len(rows) + 1))
    if ranks != expected_ranks:
        raise ValueError("Submission ranks must be contiguous ranks from 1..N in file order.")
    ids = [str(row.get(id_column, "")).strip() for row in rows]
    if any(not value for value in ids):
        raise ValueError(f"Submission has blank IDs in column '{id_column}'.")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Submission has duplicate IDs in column '{id_column}'.")
    _ = [_float(row.get("score")) for row in rows]
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"Submission row count is {len(rows)}, expected {expected_count}.")


def _selector_config(
    config: Mapping[str, Any],
    *,
    n_rows: int,
    primary_id_column: str,
    output_files: Mapping[str, Path],
) -> dict[str, Any]:
    method = dict(config.get("method", {}))
    inputs = dict(config.get("inputs", {}))
    return {
        "method": {
            "name": method.get("name", "artifact-gate exact-window blend"),
            "primary_submission_id_column": primary_id_column,
            "claim": method.get(
                "claim",
                "Partial-TS2Vec / exact-window blended k-center selector with artifact-aware trace rerank.",
            ),
            "known_limitations": method.get(
                "known_limitations",
                [
                    "The TS2Vec old-support view is partial, not exact full-support.",
                    "The window-stat old-support view is exact full-support.",
                    "The final artifact is not a validated clean TS2Vec active-learning paper result.",
                ],
            ),
        },
        "inputs": {
            "old_manifest": inputs.get("old_manifest"),
            "new_manifest": inputs.get("new_manifest"),
        },
        "n_ranked_clips": int(n_rows),
        "outputs": {key: str(value) for key, value in output_files.items()},
    }


def _feature_schema(
    primary_rows: Sequence[Mapping[str, str]],
    backup_rows: Sequence[Mapping[str, str]],
    diagnostic_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    return {
        "primary_submission_columns": _column_schema(primary_rows),
        "backup_worker_submission_columns": _column_schema(backup_rows),
        "diagnostic_columns": _column_schema(diagnostic_rows),
        "forbidden_training_features_used": False,
        "notes": [
            "Submission files contain only rank, score, and an ID column.",
            "Diagnostics are for audit and are not required by the external evaluator.",
        ],
    }


def _column_schema(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    if not rows:
        return []
    first = rows[0]
    return [{"name": str(key), "inferred_type": _infer_type(first.get(key, ""))} for key in first.keys()]


def _readme_text(config: Mapping[str, Any], package_report: Mapping[str, Any], feature_schema: Mapping[str, Any]) -> str:
    method = dict(config.get("method", {}))
    inputs = dict(config.get("inputs", {}))
    method_name = method.get("name", "artifact-gate exact-window blend")
    claim = method.get(
        "claim",
        "Partial-TS2Vec / exact-window blended k-center selector with artifact-aware trace rerank.",
    )
    artifacts = package_report["artifacts"]
    limitations = method.get(
        "known_limitations",
        [
            "This is not an exact full-200k TS2Vec search.",
            "The TS2Vec old-support cache is partial.",
            "The external held-out evaluator remains the real test.",
        ],
    )
    limitation_lines = "\n".join(f"- {item}" for item in limitations)
    diagnostic_columns = ", ".join(column["name"] for column in feature_schema["diagnostic_columns"][:12])
    return f"""# Final IMU Clip Ranking Package

## Primary Submission

Use `{Path(str(artifacts["primary_submission"])).name}` as the primary submission file.
It contains `{package_report["n_rows"]}` ranked new clips with ID column
`{package_report["primary_id_column"]}`.

Backup ID-format file: `{Path(str(artifacts["backup_worker_submission"])).name}`.

## Method

{method_name}

{claim}

This package ranks all new clips using the currently promoted conservative
selector. It combines partial TS2Vec old-support novelty, exact full-support
window-stat novelty, quality/physical-validity gates, k-center-style redundancy
control, and an artifact-aware trace rerank.

## Inputs

- Old manifest: `{inputs.get("old_manifest", "not recorded")}`
- New manifest: `{inputs.get("new_manifest", "not recorded")}`

## Files

- `{Path(str(artifacts["primary_submission"])).name}`: primary CSV to submit.
- `{Path(str(artifacts["backup_worker_submission"])).name}`: backup worker-ID variant.
- `{Path(str(artifacts["diagnostics"])).name}`: full ranking diagnostics.
- `{Path(str(artifacts["selector_config"])).name}`: machine-readable method and output config.
- `{Path(str(artifacts["feature_schema"])).name}`: output and diagnostic column schema.
- `{Path(str(artifacts["selector_report"])).name}`: final selector report.
- `{Path(str(artifacts["package_report"])).name}`: package readiness report.

## Diagnostics

The diagnostic table begins with these columns:

`{diagnostic_columns}`

Diagnostics are for review only. The external submission file should remain the
ranked CSV.

## Known Limitations

{limitation_lines}

## Reproduce The Package

```bash
python -m marginal_value.active.run_final \\
  --config-path configs/final_package_artifact_gate.json
```

This command packages existing promoted artifacts. It does not launch Modal,
train a model, or recompute embeddings.
"""


def _copy_optional_reports(
    source_config: Mapping[str, Any],
    *,
    output_dir: Path,
    keys: Mapping[str, str],
) -> dict[str, Path]:
    copied: dict[str, Path] = {}
    for key, filename in keys.items():
        value = str(source_config.get(key, "")).strip()
        if not value:
            continue
        src = Path(value)
        if not src.exists():
            continue
        dst = output_dir / filename
        _copy(src, dst)
        copied[key] = dst
    return copied


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _resolve_artifact(source_dir: Path, source_config: Mapping[str, Any], key: str) -> Path:
    raw_value = str(source_config.get(key, "")).strip()
    if not raw_value:
        raise ValueError(f"source_artifacts.{key} is required.")
    path = Path(raw_value)
    if not path.is_absolute():
        path = source_dir / path
    if not path.exists():
        raise FileNotFoundError(f"Missing source artifact for {key}: {path}")
    return path


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Final package config requires mapping '{key}'.")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _infer_type(value: object) -> str:
    text = str(value)
    if text.lower() in {"true", "false"}:
        return "bool"
    try:
        int(text)
        return "int"
    except ValueError:
        pass
    try:
        float(text)
        return "float"
    except ValueError:
        return "string"


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _int(value: object) -> int:
    if value is None or value == "":
        raise ValueError("Expected integer value, got blank.")
    return int(value)


def _float(value: object) -> float:
    if value is None or value == "":
        raise ValueError("Expected numeric value, got blank.")
    return float(value)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package the promoted final IMU ranking artifacts.")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--source-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = load_final_package_config(args.config_path)
    if args.source_dir:
        config.setdefault("source_artifacts", {})["source_dir"] = args.source_dir
    if args.output_dir:
        config.setdefault("artifacts", {})["output_dir"] = args.output_dir
    if args.expected_count is not None:
        config.setdefault("validation", {})["expected_count"] = args.expected_count
    if args.validate_only:
        output_dir = config.get("artifacts", {}).get("output_dir")
        expected = int(config.get("validation", {}).get("expected_count", 2000))
        return validate_final_package(output_dir, expected_count=expected)
    return package_final_submission(config)


if __name__ == "__main__":
    main()
