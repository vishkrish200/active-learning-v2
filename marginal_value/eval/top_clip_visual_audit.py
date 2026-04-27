from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from marginal_value.logging_utils import log_event, log_progress
from marginal_value.preprocessing.quality import compute_quality_features, load_modal_jsonl_imu


DEFAULT_SCORE_COLUMNS = (
    "final_score",
    "quality_score",
    "stationary_fraction",
    "max_abs_value",
    "physical_validity_pass",
    "physical_validity_max_abs_value",
    "old_novelty_score",
    "grammar_score_component",
    "new_density_score",
    "old_knn_distance",
    "new_batch_density",
)


def run_top_clip_visual_audit(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    validate_visual_audit_config(config)
    suffix = "smoke" if smoke else "full"
    artifacts = config["artifacts"]
    audit_config = config.get("visual_audit", {})
    ranking_dir = Path(artifacts["ranking_dir"])
    output_dir = Path(artifacts.get("output_dir", ranking_dir / "visual_audit"))
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_path = ranking_dir / f"baseline_diagnostics_val_{suffix}.csv"
    quality_metadata_path = ranking_dir / f"baseline_quality_metadata_{suffix}.csv"
    if smoke and not diagnostics_path.exists():
        diagnostics_path = ranking_dir / "baseline_diagnostics_val_full.csv"
    if smoke and not quality_metadata_path.exists():
        quality_metadata_path = ranking_dir / "baseline_quality_metadata_full.csv"

    top_n = int(audit_config.get("top_n_smoke" if smoke else "top_n", 50))
    parent_top_k = int(audit_config.get("dominant_parent_top_k_smoke" if smoke else "dominant_parent_top_k", 200))
    parent_examples = int(audit_config.get("dominant_parent_examples_smoke" if smoke else "dominant_parent_examples", 25))
    max_samples = audit_config.get("max_samples_per_clip", config.get("quality", {}).get("max_samples_per_clip"))
    max_samples = int(max_samples) if max_samples is not None else None
    sample_rate = float(config.get("quality", {}).get("sample_rate", 30.0))
    max_plot_points = int(audit_config.get("max_plot_points", 1800))
    generate_plots = bool(audit_config.get("generate_plots", True))

    log_event(
        "top_clip_visual_audit",
        "start",
        smoke=smoke,
        diagnostics_path=str(diagnostics_path),
        quality_metadata_path=str(quality_metadata_path),
        output_dir=str(output_dir),
        top_n=top_n,
        dominant_parent_top_k=parent_top_k,
        dominant_parent_examples=parent_examples,
        generate_plots=generate_plots,
    )

    diagnostics = _ranked_rows(_read_csv(diagnostics_path))
    quality_by_id = _quality_metadata_by_id(_read_csv(quality_metadata_path))
    if not diagnostics:
        raise ValueError(f"No diagnostics rows found in {diagnostics_path}")

    top_rows = diagnostics[: min(top_n, len(diagnostics))]
    dominant_parent_id = _dominant_parent_cluster(diagnostics[: min(parent_top_k, len(diagnostics))])
    dominant_parent_rows = [row for row in diagnostics if _parent_cluster_id(row) == dominant_parent_id]
    parent_rows = _diverse_parent_examples(dominant_parent_rows, max_examples=parent_examples)

    selected: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for group, rows in (("top", top_rows), ("dominant_parent", parent_rows)):
        for row in rows:
            sample_id = _sample_id(row)
            if sample_id in seen_ids:
                continue
            output = dict(row)
            output["audit_group"] = group
            selected.append(output)
            seen_ids.add(sample_id)

    plot_dir = output_dir / f"top_clip_trace_plots_{suffix}"
    plot_rows: list[dict[str, Any]] = []
    if generate_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)
        progress_every = max(1, len(selected) // 10) if selected else 1
        for index, row in enumerate(selected, start=1):
            sample_id = _sample_id(row)
            metadata = quality_by_id.get(sample_id)
            if metadata is None:
                plot_rows.append(_plot_error_row(row, "missing_quality_metadata"))
                continue
            raw_path = Path(metadata.get("raw_path", ""))
            if not raw_path.exists():
                plot_rows.append(_plot_error_row(row, f"missing_raw_path:{raw_path}"))
                continue
            try:
                samples, timestamps = load_modal_jsonl_imu(raw_path, max_samples=max_samples)
                quality_features = compute_quality_features(samples, timestamps=timestamps, sample_rate=sample_rate)
                plot_path = plot_dir / _plot_filename(row)
                _write_trace_plot(
                    plot_path,
                    row=row,
                    metadata=metadata,
                    samples=samples,
                    timestamps=timestamps,
                    sample_rate=sample_rate,
                    max_plot_points=max_plot_points,
                    quality_features=quality_features,
                )
                plot_rows.append(_plot_success_row(row, plot_path, raw_path, quality_features))
            except Exception as exc:  # pragma: no cover - retained for remote audit robustness
                plot_rows.append(_plot_error_row(row, f"{type(exc).__name__}:{exc}"))
            log_progress(
                "top_clip_visual_audit",
                "plot_progress",
                index=index,
                total=len(selected),
                every=progress_every,
                rank=int(float(row.get("rank", 0) or 0)),
            )

    top_summary = _top_summary(diagnostics, top_ks=tuple(int(k) for k in audit_config.get("top_ks", [10, 25, 50, 100, 200])))
    parent_summary = _parent_cluster_summary(diagnostics, parent_top_k=parent_top_k, dominant_parent_id=dominant_parent_id)
    report = {
        "mode": suffix,
        "diagnostics_path": str(diagnostics_path),
        "quality_metadata_path": str(quality_metadata_path),
        "n_diagnostics": len(diagnostics),
        "n_quality_metadata": len(quality_by_id),
        "n_selected_for_plot": len(selected),
        "n_plots_written": int(sum(1 for row in plot_rows if row.get("plot_status") == "ok")),
        "dominant_parent_cluster_id": dominant_parent_id,
        "top_summary": top_summary,
        "dominant_parent_summary": parent_summary,
        "score_columns": _score_column_summary(diagnostics, DEFAULT_SCORE_COLUMNS),
        "artifacts": {},
    }

    report_path = output_dir / f"top_clip_visual_audit_report_{suffix}.json"
    top_csv_path = output_dir / f"top_clip_visual_audit_top_rows_{suffix}.csv"
    parent_csv_path = output_dir / f"top_clip_visual_audit_dominant_parent_rows_{suffix}.csv"
    plot_csv_path = output_dir / f"top_clip_visual_audit_plot_index_{suffix}.csv"
    html_path = output_dir / f"top_clip_visual_audit_index_{suffix}.html"

    _write_rows(top_csv_path, _summary_rows(top_rows, "top"))
    _write_rows(parent_csv_path, _summary_rows(parent_rows, "dominant_parent"))
    _write_rows(plot_csv_path, plot_rows)
    _write_html_index(html_path, report=report, plot_rows=plot_rows, output_dir=output_dir)
    report["artifacts"] = {
        "report": str(report_path),
        "top_rows": str(top_csv_path),
        "dominant_parent_rows": str(parent_csv_path),
        "plot_index": str(plot_csv_path),
        "html_index": str(html_path),
        "plot_dir": str(plot_dir),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    result = {
        "mode": suffix,
        "report_path": str(report_path),
        "html_index": str(html_path),
        "plot_dir": str(plot_dir),
        "n_selected_for_plot": len(selected),
        "n_plots_written": report["n_plots_written"],
        "dominant_parent_cluster_id": dominant_parent_id,
        "top10_mean_quality": top_summary.get("10", {}).get("mean_quality", 0.0),
        "top100_parent_largest_fraction": top_summary.get("100", {}).get("parent_largest_cluster_fraction", 0.0),
    }
    log_event("top_clip_visual_audit", "done", **result)
    return result


def load_visual_audit_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_visual_audit_config(config: dict[str, Any]) -> None:
    execution = config.get("execution", {})
    artifacts = config.get("artifacts", {})
    if execution.get("provider") != "modal":
        raise ValueError("Top-clip visual audit must run on Modal.")
    if not str(artifacts.get("ranking_dir", "")).startswith("/artifacts"):
        raise ValueError("visual audit ranking_dir must be mounted under /artifacts.")
    if not str(artifacts.get("output_dir", artifacts.get("ranking_dir", ""))).startswith("/artifacts"):
        raise ValueError("visual audit output_dir must be mounted under /artifacts.")


def _write_trace_plot(
    path: Path,
    *,
    row: dict[str, str],
    metadata: dict[str, str],
    samples: np.ndarray,
    timestamps: np.ndarray | None,
    sample_rate: float,
    max_plot_points: int,
    quality_features: dict[str, float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(samples, dtype=float)
    time_axis = _time_axis(values, timestamps, sample_rate=sample_rate)
    stride = max(1, int(np.ceil(len(time_axis) / max(1, max_plot_points))))
    idx = np.arange(0, len(time_axis), stride)
    t = time_axis[idx]
    plotted = values[idx, :6]
    acc = values[:, :3]
    gyro = values[:, 3:6]
    acc_norm = np.linalg.norm(acc, axis=1)[idx]
    gyro_norm = np.linalg.norm(gyro, axis=1)[idx]
    acc_jerk = np.linalg.norm(np.diff(acc, axis=0, prepend=acc[:1]), axis=1)[idx] * sample_rate
    gyro_jerk = np.linalg.norm(np.diff(gyro, axis=0, prepend=gyro[:1]), axis=1)[idx] * sample_rate

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(_plot_title(row), fontsize=11)
    for channel, label in enumerate(("acc_x", "acc_y", "acc_z")):
        axes[0].plot(t, plotted[:, channel], linewidth=0.8, label=label)
    axes[0].set_ylabel("accel")
    axes[0].legend(loc="upper right", ncols=3, fontsize=8)

    for channel, label in enumerate(("gyro_x", "gyro_y", "gyro_z"), start=3):
        axes[1].plot(t, plotted[:, channel], linewidth=0.8, label=label)
    axes[1].set_ylabel("gyro")
    axes[1].legend(loc="upper right", ncols=3, fontsize=8)

    axes[2].plot(t, acc_norm, linewidth=0.9, label="acc_norm")
    axes[2].plot(t, gyro_norm, linewidth=0.9, label="gyro_norm")
    axes[2].set_ylabel("norm")
    axes[2].legend(loc="upper right", ncols=2, fontsize=8)

    axes[3].plot(t, acc_jerk, linewidth=0.8, label="acc_jerk")
    axes[3].plot(t, gyro_jerk, linewidth=0.8, label="gyro_jerk")
    axes[3].set_ylabel("jerk")
    axes[3].set_xlabel("seconds")
    axes[3].legend(loc="upper right", ncols=2, fontsize=8)

    for axis in axes:
        axis.grid(True, color="#dddddd", linewidth=0.5)
        axis.margins(x=0)

    text = _metrics_text(row, metadata, quality_features)
    fig.text(0.01, 0.005, text, ha="left", va="bottom", family="monospace", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _time_axis(samples: np.ndarray, timestamps: np.ndarray | None, *, sample_rate: float) -> np.ndarray:
    if timestamps is not None and len(timestamps) == len(samples):
        ts = np.asarray(timestamps, dtype=float)
        finite = np.isfinite(ts)
        if np.any(finite):
            first = float(ts[finite][0])
            out = ts - first
            if np.all(np.isfinite(out[finite])):
                return out
    return np.arange(len(samples), dtype=float) / sample_rate


def _plot_title(row: dict[str, str]) -> str:
    rank = int(float(row.get("rank", 0) or 0))
    sample = _sample_id(row)[:12]
    return (
        f"rank {rank:03d} | {sample} | score={_float(row.get('final_score')):.4f} | "
        f"q={_float(row.get('quality_score')):.4f} | reason={row.get('reason_code', '')} | "
        f"parent={_parent_cluster_id(row)} | cluster={row.get('new_cluster_id', '')}"
    )


def _metrics_text(row: dict[str, str], metadata: dict[str, str], quality_features: dict[str, float]) -> str:
    pieces = [
        f"old_novelty={_float(row.get('old_novelty_score')):.4f}",
        f"grammar={_float(row.get('grammar_score_component', row.get('grammar_score'))):.4f}",
        f"new_density={_float(row.get('new_density_score')):.4f}",
        f"old_knn={_float(row.get('old_knn_distance')):.4f}",
        f"new_batch_density={_float(row.get('new_batch_density')):.4f}",
        f"split={row.get('large_cluster_split_applied', '')}/{row.get('large_cluster_split_strategy', '')}",
        f"raw_q={float(quality_features.get('quality_score', 0.0)):.4f}",
        f"spike={float(quality_features.get('spike_rate', 0.0)):.5f}",
        f"stationary={float(quality_features.get('stationary_fraction', 0.0)):.4f}",
        f"max_abs={float(quality_features.get('max_abs_value', 0.0)):.2f}",
        f"raw={Path(metadata.get('raw_path', '')).name}",
    ]
    return " | ".join(pieces)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _ranked_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return sorted((dict(row) for row in rows), key=lambda row: int(float(row.get("rank", 0) or 0)))


def _quality_metadata_by_id(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = row.get("sample_id", "")
        if sample_id:
            output[sample_id] = dict(row)
    return output


def _sample_id(row: dict[str, str]) -> str:
    return str(row.get("sample_id") or row.get("worker_id") or "")


def _parent_cluster_id(row: dict[str, str]) -> str:
    return str(row.get("new_cluster_parent_id") or row.get("new_cluster_id") or "")


def _dominant_parent_cluster(rows: Iterable[dict[str, str]]) -> str:
    counts = Counter(_parent_cluster_id(row) for row in rows)
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def _diverse_parent_examples(rows: list[dict[str, str]], *, max_examples: int) -> list[dict[str, str]]:
    if max_examples <= 0:
        return []
    selected: list[dict[str, str]] = []
    seen_clusters: set[str] = set()
    ranked = _ranked_rows(rows)
    for row in ranked:
        cluster_id = str(row.get("new_cluster_id", ""))
        if cluster_id in seen_clusters:
            continue
        selected.append(row)
        seen_clusters.add(cluster_id)
        if len(selected) >= max_examples:
            return selected
    for row in ranked:
        if _sample_id(row) in {_sample_id(item) for item in selected}:
            continue
        selected.append(row)
        if len(selected) >= max_examples:
            break
    return selected


def _top_summary(rows: list[dict[str, str]], *, top_ks: tuple[int, ...]) -> dict[str, dict[str, Any]]:
    return {str(k): _slice_summary(rows[: min(k, len(rows))]) for k in top_ks}


def _slice_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    parent_ids = [_parent_cluster_id(row) for row in rows]
    split_ids = [str(row.get("new_cluster_id", "")) for row in rows]
    parent_counts = Counter(parent_ids)
    split_counts = Counter(split_ids)
    parent_largest = parent_counts.most_common(1)[0] if parent_counts else ("", 0)
    split_largest = split_counts.most_common(1)[0] if split_counts else ("", 0)
    quality = [_float(row.get("quality_score")) for row in rows]
    stationary = [_float(row.get("stationary_fraction")) for row in rows]
    max_abs = [_float(row.get("max_abs_value")) for row in rows]
    return {
        "n_rows": len(rows),
        "mean_quality": _mean(quality),
        "min_quality": min(quality) if quality else 0.0,
        "mean_stationary_fraction": _mean(stationary),
        "max_stationary_fraction": max(stationary) if stationary else 0.0,
        "stationary_fraction_over_90": _mean([1.0 if value > 0.90 else 0.0 for value in stationary]),
        "max_abs_value": max(max_abs) if max_abs else 0.0,
        "unique_parent_clusters": len(parent_counts),
        "parent_largest_cluster_id": parent_largest[0],
        "parent_largest_cluster_count": parent_largest[1],
        "parent_largest_cluster_fraction": parent_largest[1] / len(rows) if rows else 0.0,
        "unique_split_clusters": len(split_counts),
        "split_largest_cluster_id": split_largest[0],
        "split_largest_cluster_count": split_largest[1],
        "split_largest_cluster_fraction": split_largest[1] / len(rows) if rows else 0.0,
        "reason_code_counts": dict(Counter(str(row.get("reason_code", "")) for row in rows)),
        "mean_final_score": _mean([_float(row.get("final_score")) for row in rows]),
        "mean_old_novelty_score": _mean([_float(row.get("old_novelty_score")) for row in rows]),
        "mean_grammar_score": _mean([_float(row.get("grammar_score_component", row.get("grammar_score"))) for row in rows]),
        "mean_new_density_score": _mean([_float(row.get("new_density_score")) for row in rows]),
    }


def _parent_cluster_summary(rows: list[dict[str, str]], *, parent_top_k: int, dominant_parent_id: str) -> dict[str, Any]:
    top_rows = rows[: min(parent_top_k, len(rows))]
    parent_counts = Counter(_parent_cluster_id(row) for row in top_rows)
    dominant_rows = [row for row in top_rows if _parent_cluster_id(row) == dominant_parent_id]
    return {
        "top_k": len(top_rows),
        "dominant_parent_cluster_id": dominant_parent_id,
        "dominant_parent_count": parent_counts.get(dominant_parent_id, 0),
        "dominant_parent_fraction": parent_counts.get(dominant_parent_id, 0) / len(top_rows) if top_rows else 0.0,
        "parent_cluster_counts": dict(parent_counts.most_common(10)),
        "dominant_parent_reason_code_counts": dict(Counter(str(row.get("reason_code", "")) for row in dominant_rows)),
        "dominant_parent_split_cluster_count": len({str(row.get("new_cluster_id", "")) for row in dominant_rows}),
        "dominant_parent_mean_quality": _mean([_float(row.get("quality_score")) for row in dominant_rows]),
        "dominant_parent_mean_score": _mean([_float(row.get("final_score")) for row in dominant_rows]),
    }


def _score_column_summary(rows: list[dict[str, str]], columns: Iterable[str]) -> dict[str, dict[str, float]]:
    return {column: _numeric_summary([_float(row.get(column)) for row in rows]) for column in columns}


def _numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def _summary_rows(rows: Iterable[dict[str, str]], group: str) -> list[dict[str, object]]:
    output = []
    for row in rows:
        output.append(
            {
                "audit_group": group,
                "rank": int(float(row.get("rank", 0) or 0)),
                "sample_id": _sample_id(row),
                "final_score": _float(row.get("final_score")),
                "quality_score": _float(row.get("quality_score")),
                "stationary_fraction": _float(row.get("stationary_fraction")),
                "max_abs_value": _float(row.get("max_abs_value")),
                "quality_threshold_pass": row.get("quality_threshold_pass", ""),
                "quality_gate_pass": row.get("quality_gate_pass", ""),
                "physical_validity_pass": row.get("physical_validity_pass", ""),
                "physical_validity_failure_reasons": row.get("physical_validity_failure_reasons", ""),
                "physical_validity_max_stationary_fraction": row.get("physical_validity_max_stationary_fraction", ""),
                "physical_validity_max_abs_value": row.get("physical_validity_max_abs_value", ""),
                "reason_code": row.get("reason_code", ""),
                "new_cluster_parent_id": _parent_cluster_id(row),
                "new_cluster_id": row.get("new_cluster_id", ""),
                "old_novelty_score": _float(row.get("old_novelty_score")),
                "grammar_score_component": _float(row.get("grammar_score_component", row.get("grammar_score"))),
                "new_density_score": _float(row.get("new_density_score")),
                "old_knn_distance": _float(row.get("old_knn_distance")),
                "new_batch_density": _float(row.get("new_batch_density")),
                "large_cluster_split_applied": row.get("large_cluster_split_applied", ""),
                "large_cluster_split_strategy": row.get("large_cluster_split_strategy", ""),
            }
        )
    return output


def _plot_success_row(
    row: dict[str, str],
    plot_path: Path,
    raw_path: Path,
    quality_features: dict[str, float],
) -> dict[str, Any]:
    output = _summary_rows([row], str(row.get("audit_group", "")))[0]
    output.update(
        {
            "plot_status": "ok",
            "plot_path": str(plot_path),
            "plot_file": plot_path.name,
            "raw_path": str(raw_path),
            "computed_quality_score": float(quality_features.get("quality_score", 0.0)),
            "computed_spike_rate": float(quality_features.get("spike_rate", 0.0)),
            "computed_stationary_fraction": float(quality_features.get("stationary_fraction", 0.0)),
            "computed_max_abs_value": float(quality_features.get("max_abs_value", 0.0)),
        }
    )
    return output


def _plot_error_row(row: dict[str, str], error: str) -> dict[str, Any]:
    output = _summary_rows([row], str(row.get("audit_group", "")))[0]
    output.update({"plot_status": "error", "plot_error": error, "plot_path": "", "plot_file": ""})
    return output


def _plot_filename(row: dict[str, str]) -> str:
    rank = int(float(row.get("rank", 0) or 0))
    sample = _sample_id(row)[:16] or "unknown"
    group = str(row.get("audit_group", "clip")).replace("/", "_")
    return f"{group}_rank_{rank:03d}_{sample}.png"


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_html_index(path: Path, *, report: dict[str, Any], plot_rows: list[dict[str, Any]], output_dir: Path) -> None:
    cards = []
    for row in plot_rows:
        plot_path = str(row.get("plot_path", ""))
        rel_plot = Path(plot_path).relative_to(output_dir) if plot_path and Path(plot_path).is_absolute() else Path(plot_path)
        status = row.get("plot_status", "")
        image_html = f'<img src="{html.escape(str(rel_plot))}" alt="rank {row.get("rank", "")}">' if status == "ok" else ""
        cards.append(
            "<section>"
            f"<h2>Rank {html.escape(str(row.get('rank', '')))} | {html.escape(str(row.get('reason_code', '')))}</h2>"
            f"<p>score={html.escape(str(row.get('final_score', '')))} "
            f"quality={html.escape(str(row.get('quality_score', '')))} "
            f"parent={html.escape(str(row.get('new_cluster_parent_id', '')))} "
            f"cluster={html.escape(str(row.get('new_cluster_id', '')))} "
            f"group={html.escape(str(row.get('audit_group', '')))}</p>"
            f"{image_html}"
            f"<p>{html.escape(str(row.get('plot_error', '')))}</p>"
            "</section>"
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Top Clip Visual Audit</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #111; }}
    section {{ margin: 0 0 32px; padding-bottom: 24px; border-bottom: 1px solid #ddd; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
    pre {{ white-space: pre-wrap; background: #f6f6f6; padding: 12px; }}
  </style>
</head>
<body>
  <h1>Top Clip Visual Audit</h1>
  <pre>{html.escape(json.dumps({key: value for key, value in report.items() if key != "score_columns"}, indent=2, sort_keys=True))}</pre>
  {''.join(cards)}
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
