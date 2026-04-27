from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from marginal_value.logging_utils import log_event


def run_reason_threshold_grid(
    config: dict[str, Any],
    *,
    diagnostics_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    validate_reason_threshold_grid_config(config)
    artifacts = config["artifacts"]
    diagnostics = Path(diagnostics_path) if diagnostics_path is not None else Path(artifacts["diagnostics_path"])
    output = Path(output_dir) if output_dir is not None else Path(artifacts["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    log_event(
        "reason_threshold_grid",
        "start",
        diagnostics_path=str(diagnostics),
        output_dir=str(output),
    )
    rows = _read_csv(diagnostics)
    grid = config.get("grid", {})
    audit = config.get("audit", {})
    report = evaluate_reason_threshold_grid(
        rows,
        component_thresholds=[float(value) for value in grid.get("grammar_component_thresholds", [0.55, 0.65, 0.75, 0.85])],
        delta_thresholds=[float(value) for value in grid.get("grammar_delta_thresholds", [0.02, 0.03, 0.05, 0.08])],
        top_ks=[int(value) for value in audit.get("top_ks", [10, 50, 100])],
        low_quality_threshold=float(audit.get("low_quality_threshold", 0.45)),
        min_new_density_score=float(grid.get("min_new_density_score", 0.35)),
        target_top_k=int(grid.get("target_top_k", 100)),
        target_rare_temporal_count=int(grid.get("target_rare_temporal_count", 20)),
    )
    report["diagnostics_path"] = str(diagnostics)

    report_path = output / "grammar_reason_threshold_grid_report.json"
    summary_path = output / "grammar_reason_threshold_grid_summary.csv"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_rows(summary_path, _summary_rows(report))
    result = {
        "n_rows": len(rows),
        "n_variants": len(report["variants"]),
        "recommended_variant": report["recommended"]["variant"],
        "report_path": str(report_path),
        "summary_path": str(summary_path),
    }
    log_event("reason_threshold_grid", "done", **result)
    return result


def evaluate_reason_threshold_grid(
    rows: Sequence[dict[str, object]],
    *,
    component_thresholds: Iterable[float],
    delta_thresholds: Iterable[float],
    top_ks: Iterable[int],
    low_quality_threshold: float,
    min_new_density_score: float,
    target_top_k: int,
    target_rare_temporal_count: int,
) -> dict[str, Any]:
    component_values = [float(value) for value in component_thresholds]
    delta_values = [float(value) for value in delta_thresholds]
    top_k_values = [int(value) for value in top_ks]
    ranked = sorted((dict(row) for row in rows), key=lambda row: _to_float(row.get("rank", 0.0)))
    variants: dict[str, dict[str, Any]] = {}
    for component_threshold in component_values:
        for delta_threshold in delta_values:
            name = _variant_name(component_threshold, delta_threshold)
            relabeled = [
                row
                | {
                    "threshold_reason_code": _assign_threshold_reason_code(
                        row,
                        grammar_component_threshold=component_threshold,
                        grammar_delta_threshold=delta_threshold,
                        low_quality_threshold=low_quality_threshold,
                        min_new_density_score=min_new_density_score,
                    )
                }
                for row in ranked
            ]
            variants[name] = {
                "grammar_component_threshold": float(component_threshold),
                "grammar_delta_threshold": float(delta_threshold),
                "reason_code_counts": dict(Counter(str(row["threshold_reason_code"]) for row in relabeled)),
                "top_k": {
                    str(k): _top_k_summary(
                        relabeled[: int(k)],
                        low_quality_threshold=low_quality_threshold,
                    )
                    for k in top_k_values
                },
            }

    recommended = _recommend_variant(
        variants,
        target_top_k=target_top_k,
        target_rare_temporal_count=target_rare_temporal_count,
    )
    return {
        "n_rows": len(ranked),
        "grid": {
            "grammar_component_thresholds": component_values,
            "grammar_delta_thresholds": delta_values,
            "min_new_density_score": float(min_new_density_score),
            "low_quality_threshold": float(low_quality_threshold),
            "top_ks": top_k_values,
            "target_top_k": int(target_top_k),
            "target_rare_temporal_count": int(target_rare_temporal_count),
        },
        "variants": variants,
        "recommended": recommended,
    }


def validate_reason_threshold_grid_config(config: dict[str, Any]) -> None:
    execution = config.get("execution", {})
    artifacts = config.get("artifacts", {})
    if execution.get("provider") != "local":
        raise ValueError("Reason threshold grid runs locally against an existing diagnostics CSV.")
    diagnostics_path = Path(str(artifacts.get("diagnostics_path", "")))
    output_dir = Path(str(artifacts.get("output_dir", "")))
    if not diagnostics_path.is_absolute():
        raise ValueError("artifacts.diagnostics_path must be absolute.")
    if not output_dir.is_absolute():
        raise ValueError("artifacts.output_dir must be absolute.")
    grid = config.get("grid", {})
    for key in ("grammar_component_thresholds", "grammar_delta_thresholds"):
        values = grid.get(key, [])
        if not isinstance(values, list) or not values:
            raise ValueError(f"grid.{key} must be a non-empty list.")
        if any(float(value) < 0.0 for value in values):
            raise ValueError(f"grid.{key} must contain non-negative values.")


def _assign_threshold_reason_code(
    row: dict[str, object],
    *,
    grammar_component_threshold: float,
    grammar_delta_threshold: float,
    low_quality_threshold: float,
    min_new_density_score: float,
) -> str:
    quality = _to_float(row.get("quality_score", 1.0))
    novelty = _to_float(row.get("old_novelty_score", 0.0))
    density = _to_float(row.get("new_density_score", 0.0))
    grammar_component = _to_float(row.get("grammar_score_component", 0.0))
    grammar_delta = _to_float(row.get("grammar_promotion_delta", 0.0))

    if quality < low_quality_threshold:
        return "LOW_QUALITY"
    if novelty >= 0.65 and density >= 0.45:
        return "COHESIVE_NEW_WORKFLOW"
    if novelty >= 0.65:
        return "HIGH_NOVELTY_SINGLETON"
    if (
        grammar_component >= grammar_component_threshold
        and grammar_delta >= grammar_delta_threshold
        and density >= min_new_density_score
    ):
        return "RARE_TEMPORAL_COMPOSITION"
    if novelty < 0.35 and density >= 0.65:
        return "REDUNDANT_KNOWN_WORKFLOW"
    return "RARE_MOTION_PRIMITIVES"


def _top_k_summary(rows: Sequence[dict[str, object]], *, low_quality_threshold: float) -> dict[str, Any]:
    qualities = [_to_float(row.get("quality_score", 1.0)) for row in rows]
    clusters = [str(row.get("new_cluster_id", "")) for row in rows if str(row.get("new_cluster_id", ""))]
    reasons = Counter(str(row.get("threshold_reason_code", "")) for row in rows)
    return {
        "n_rows": len(rows),
        "rare_temporal_count": int(reasons.get("RARE_TEMPORAL_COMPOSITION", 0)),
        "low_quality_count": int(sum(value < low_quality_threshold for value in qualities)),
        "mean_quality": _mean(qualities),
        "unique_cluster_count": len(set(clusters)),
        "reason_code_counts": dict(reasons),
    }


def _recommend_variant(
    variants: dict[str, dict[str, Any]],
    *,
    target_top_k: int,
    target_rare_temporal_count: int,
) -> dict[str, Any]:
    target_key = str(target_top_k)

    def objective(item: tuple[str, dict[str, Any]]) -> tuple[float, float, float, float]:
        _name, variant = item
        top_summary = variant.get("top_k", {}).get(target_key)
        if top_summary is None:
            available = sorted(variant.get("top_k", {}).items(), key=lambda value: int(value[0]))
            top_summary = available[-1][1] if available else {}
        rare_count = float(top_summary.get("rare_temporal_count", 0.0))
        low_quality_count = float(top_summary.get("low_quality_count", 0.0))
        component_threshold = float(variant.get("grammar_component_threshold", 0.0))
        delta_threshold = float(variant.get("grammar_delta_threshold", 0.0))
        return (
            abs(rare_count - float(target_rare_temporal_count)) + 10.0 * low_quality_count,
            -component_threshold,
            -delta_threshold,
            rare_count,
        )

    name, variant = min(variants.items(), key=objective)
    return {
        "variant": name,
        "grammar_component_threshold": variant["grammar_component_threshold"],
        "grammar_delta_threshold": variant["grammar_delta_threshold"],
        "top_k": variant["top_k"],
        "reason_code_counts": variant["reason_code_counts"],
    }


def _summary_rows(report: dict[str, Any]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, variant in sorted(report.get("variants", {}).items()):
        row: dict[str, object] = {
            "variant": name,
            "grammar_component_threshold": variant.get("grammar_component_threshold", 0.0),
            "grammar_delta_threshold": variant.get("grammar_delta_threshold", 0.0),
        }
        for reason, count in variant.get("reason_code_counts", {}).items():
            row[f"reason_{reason}"] = count
        for k, summary in variant.get("top_k", {}).items():
            row[f"top{k}_rare_temporal_count"] = summary.get("rare_temporal_count", 0)
            row[f"top{k}_low_quality_count"] = summary.get("low_quality_count", 0)
            row[f"top{k}_mean_quality"] = summary.get("mean_quality", 0.0)
            row[f"top{k}_unique_cluster_count"] = summary.get("unique_cluster_count", 0)
        rows.append(row)
    return rows


def _variant_name(component_threshold: float, delta_threshold: float) -> str:
    return f"component_{component_threshold:.3f}_delta_{delta_threshold:.3f}"


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: Iterable[float]) -> float:
    numbers = [float(value) for value in values]
    return float(sum(numbers) / len(numbers)) if numbers else 0.0
