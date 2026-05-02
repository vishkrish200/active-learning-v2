from __future__ import annotations

import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from marginal_value.active.registry import load_clip_registry_from_config
from marginal_value.logging_utils import log_event, log_progress
from marginal_value.preprocessing.quality import compute_quality_features, load_modal_jsonl_imu


DEFAULT_TOP_KS = (10, 50, 100, 200)


def run_topk_audit_pack(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    validate_topk_audit_pack_config(config)
    mode = "smoke" if smoke else "full"
    audit_config = config.get("audit", {})
    artifacts = config["artifacts"]
    output_dir = Path(str(artifacts["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    exact_path = Path(str(artifacts["exact_diagnostics_path"]))
    consensus_path = Path(str(artifacts["consensus_diagnostics_path"]))
    exact_rows = _ranked_rows(_read_csv(exact_path))
    consensus_rows = _ranked_rows(_read_csv(consensus_path))
    if not exact_rows:
        raise ValueError(f"No exact-window diagnostics rows found: {exact_path}")
    if not consensus_rows:
        raise ValueError(f"No consensus diagnostics rows found: {consensus_path}")

    top_n = int(audit_config.get("top_n_smoke" if smoke else "top_n", 50))
    disagreement_n = int(audit_config.get("disagreement_n_smoke" if smoke else "disagreement_n", 30))
    rejected_n = int(audit_config.get("rejected_n_smoke" if smoke else "rejected_n", 30))
    max_plot_points = int(audit_config.get("max_plot_points", 1800))
    max_samples = audit_config.get("max_samples_per_clip", config.get("quality", {}).get("max_samples_per_clip", 5400))
    max_samples = int(max_samples) if max_samples is not None else None
    sample_rate = float(config.get("quality", {}).get("sample_rate", 30.0))
    generate_plots = bool(audit_config.get("generate_plots", True))
    top_ks = tuple(int(k) for k in audit_config.get("top_ks", DEFAULT_TOP_KS))

    log_event(
        "active_topk_audit_pack",
        "start",
        mode=mode,
        exact_diagnostics_path=str(exact_path),
        consensus_diagnostics_path=str(consensus_path),
        output_dir=str(output_dir),
        top_n=top_n,
        disagreement_n=disagreement_n,
        rejected_n=rejected_n,
        generate_plots=generate_plots,
    )

    registry_lookup: dict[str, Any] = {}
    if config.get("data"):
        registry = load_clip_registry_from_config(config)
        registry_lookup = registry.by_sample_id

    selected = select_audit_candidates(
        exact_rows,
        consensus_rows,
        top_n=top_n,
        disagreement_n=disagreement_n,
        rejected_n=rejected_n,
        quality_threshold=float(audit_config.get("quality_threshold", config.get("ranking", {}).get("quality_threshold", 0.85))),
        max_stationary_fraction=float(
            audit_config.get("max_stationary_fraction", config.get("ranking", {}).get("max_stationary_fraction", 0.90))
        ),
        max_abs_value=float(audit_config.get("max_abs_value", config.get("ranking", {}).get("max_abs_value", 60.0))),
    )
    plot_dir = output_dir / f"trace_plots_{mode}"
    plot_rows: list[dict[str, Any]] = []
    if generate_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)
    progress_every = max(1, len(selected) // 10) if selected else 1
    for index, row in enumerate(selected, start=1):
        sample_id = str(row["sample_id"])
        raw_path = _raw_path_for_row(row, registry_lookup)
        if raw_path is None or not raw_path.exists():
            plot_rows.append(_error_row(row, "missing_raw_path", raw_path))
            continue
        try:
            samples, timestamps = load_modal_jsonl_imu(raw_path, max_samples=max_samples)
            quality = compute_quality_features(samples, timestamps=timestamps, sample_rate=sample_rate)
            trace_summary = summarize_trace(samples, timestamps=timestamps, quality=quality, sample_rate=sample_rate)
            plot_path = plot_dir / f"{int(row['audit_index']):03d}_{_safe_filename(sample_id)}.png"
            if generate_plots:
                write_trace_plot(
                    plot_path,
                    row=row,
                    samples=samples,
                    timestamps=timestamps,
                    quality=quality,
                    trace_summary=trace_summary,
                    sample_rate=sample_rate,
                    max_plot_points=max_plot_points,
                )
            plot_rows.append(_success_row(row, raw_path, plot_path if generate_plots else None, quality, trace_summary))
        except Exception as exc:  # pragma: no cover - retained for robust remote audit jobs
            plot_rows.append(_error_row(row, f"{type(exc).__name__}:{exc}", raw_path))
        log_progress(
            "active_topk_audit_pack",
            "plot_progress",
            index=index,
            total=len(selected),
            every=progress_every,
            sample_id=sample_id,
        )

    summary = summarize_audit(
        exact_rows,
        consensus_rows,
        plot_rows,
        top_ks=top_ks,
        quality_threshold=float(audit_config.get("quality_threshold", config.get("ranking", {}).get("quality_threshold", 0.85))),
        max_stationary_fraction=float(
            audit_config.get("max_stationary_fraction", config.get("ranking", {}).get("max_stationary_fraction", 0.90))
        ),
        max_abs_value=float(audit_config.get("max_abs_value", config.get("ranking", {}).get("max_abs_value", 60.0))),
    )

    selected_path = output_dir / f"topk_audit_selected_{mode}.csv"
    plot_index_path = output_dir / f"topk_audit_plot_index_{mode}.csv"
    report_path = output_dir / f"topk_audit_report_{mode}.json"
    markdown_path = output_dir / f"topk_audit_report_{mode}.md"
    html_path = output_dir / f"topk_audit_index_{mode}.html"

    _write_rows(selected_path, selected)
    _write_rows(plot_index_path, plot_rows)
    report = {
        "mode": mode,
        "exact_diagnostics_path": str(exact_path),
        "consensus_diagnostics_path": str(consensus_path),
        "n_exact_rows": len(exact_rows),
        "n_consensus_rows": len(consensus_rows),
        "n_selected": len(selected),
        "n_plots_written": int(sum(1 for row in plot_rows if row.get("plot_status") == "ok" and row.get("plot_path"))),
        "summary": summary,
        "artifacts": {
            "selected": str(selected_path),
            "plot_index": str(plot_index_path),
            "report": str(report_path),
            "markdown": str(markdown_path),
            "html_index": str(html_path),
            "plot_dir": str(plot_dir),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_markdown_report(report, plot_rows), encoding="utf-8")
    _write_html_index(html_path, report, plot_rows, output_dir=output_dir)

    result = {
        "mode": mode,
        "report_path": str(report_path),
        "markdown_path": str(markdown_path),
        "html_index_path": str(html_path),
        "plot_index_path": str(plot_index_path),
        "n_selected": len(selected),
        "n_plots_written": report["n_plots_written"],
        "likely_artifact_count": int(summary["verdict_counts"].get("likely_artifact", 0)),
        "mostly_stationary_count": int(summary["verdict_counts"].get("mostly_stationary", 0)),
    }
    log_event("active_topk_audit_pack", "done", **result)
    return result


def select_audit_candidates(
    exact_rows: Sequence[Mapping[str, Any]],
    consensus_rows: Sequence[Mapping[str, Any]],
    *,
    top_n: int,
    disagreement_n: int,
    rejected_n: int,
    quality_threshold: float,
    max_stationary_fraction: float,
    max_abs_value: float,
) -> list[dict[str, Any]]:
    exact_by_id = _by_sample_id(exact_rows)
    consensus_by_id = _by_sample_id(consensus_rows)
    selected: dict[str, dict[str, Any]] = {}

    def add(row: Mapping[str, Any], group: str, source: str) -> None:
        sample_id = _sample_id(row)
        if not sample_id:
            return
        out = selected.get(sample_id)
        if out is None:
            exact = exact_by_id.get(sample_id, row)
            consensus = consensus_by_id.get(sample_id)
            out = dict(exact)
            out["sample_id"] = sample_id
            out["audit_groups"] = []
            out["exact_rank"] = _optional_int(exact.get("rank"))
            out["consensus_rank"] = _optional_int(consensus.get("rank")) if consensus is not None else None
            out["rank_delta_exact_minus_consensus"] = _rank_delta(out.get("exact_rank"), out.get("consensus_rank"))
            selected[sample_id] = out
        groups = list(out.get("audit_groups", []))
        label = f"{source}:{group}"
        if label not in groups:
            groups.append(label)
        out["audit_groups"] = groups

    for row in exact_rows[:top_n]:
        add(row, "top", "exact")
    for row in consensus_rows[:top_n]:
        add(row, "top", "consensus")

    exact_rank = {sample_id: idx for idx, sample_id in enumerate((_sample_id(row) for row in exact_rows), start=1)}
    consensus_rank = {sample_id: idx for idx, sample_id in enumerate((_sample_id(row) for row in consensus_rows), start=1)}
    common_ids = sorted(set(exact_rank) & set(consensus_rank))
    disagreement_ids = sorted(
        common_ids,
        key=lambda sample_id: abs(exact_rank[sample_id] - consensus_rank[sample_id]),
        reverse=True,
    )[: max(0, disagreement_n)]
    for sample_id in disagreement_ids:
        add(exact_by_id[sample_id], "rank_disagreement", "exact_vs_consensus")

    rejected = [
        row
        for row in exact_rows
        if _fails_hygiene(
            row,
            quality_threshold=quality_threshold,
            max_stationary_fraction=max_stationary_fraction,
            max_abs_value=max_abs_value,
        )
    ]
    rejected.sort(key=lambda row: (_score_for_rejected(row), -float(_optional_int(row.get("rank")) or 0)), reverse=True)
    for row in rejected[: max(0, rejected_n)]:
        add(row, "high_novelty_rejected", "exact")

    rows = list(selected.values())
    rows.sort(
        key=lambda row: (
            _group_priority(row.get("audit_groups", [])),
            _optional_int(row.get("exact_rank")) or 1_000_000,
            _optional_int(row.get("consensus_rank")) or 1_000_000,
            str(row.get("sample_id", "")),
        )
    )
    for index, row in enumerate(rows, start=1):
        row["audit_index"] = index
        row["audit_groups"] = "|".join(str(group) for group in row.get("audit_groups", []))
    return rows


def summarize_trace(
    samples: np.ndarray,
    *,
    timestamps: np.ndarray | None,
    quality: Mapping[str, float],
    sample_rate: float,
) -> dict[str, Any]:
    values = np.asarray(samples, dtype=float)
    acc = values[:, :3]
    gyro = values[:, 3:6]
    acc_norm = np.linalg.norm(acc, axis=1)
    gyro_norm = np.linalg.norm(gyro, axis=1)
    acc_diff = np.diff(acc, axis=0, prepend=acc[:1])
    gyro_diff = np.diff(gyro, axis=0, prepend=gyro[:1])
    acc_jerk = np.linalg.norm(acc_diff, axis=1) * sample_rate
    gyro_jerk = np.linalg.norm(gyro_diff, axis=1) * sample_rate
    flags = trace_flags(quality, acc_norm=acc_norm, gyro_norm=gyro_norm, acc_jerk=acc_jerk, gyro_jerk=gyro_jerk)
    verdict = trace_verdict(flags)
    return {
        "n_samples": int(values.shape[0]),
        "duration_sec": float(values.shape[0] / sample_rate),
        "acc_norm_mean": _nan_stat(acc_norm, "mean"),
        "acc_norm_p95": _nan_stat(acc_norm, "p95"),
        "gyro_norm_mean": _nan_stat(gyro_norm, "mean"),
        "gyro_norm_p95": _nan_stat(gyro_norm, "p95"),
        "acc_jerk_p95": _nan_stat(acc_jerk, "p95"),
        "gyro_jerk_p95": _nan_stat(gyro_jerk, "p95"),
        "axis_std_min": float(np.nanmin(np.nanstd(values[:, :6], axis=0))),
        "axis_std_max": float(np.nanmax(np.nanstd(values[:, :6], axis=0))),
        "periodicity_acc_norm": _dominant_periodicity(acc_norm, sample_rate=sample_rate),
        "periodicity_gyro_norm": _dominant_periodicity(gyro_norm, sample_rate=sample_rate),
        "flags": flags,
        "verdict": verdict,
    }


def trace_flags(
    quality: Mapping[str, float],
    *,
    acc_norm: np.ndarray,
    gyro_norm: np.ndarray,
    acc_jerk: np.ndarray,
    gyro_jerk: np.ndarray,
) -> list[str]:
    flags: list[str] = []
    if float(quality.get("quality_score", 1.0)) < 0.85:
        flags.append("low_quality_score")
    if float(quality.get("stationary_fraction", 0.0)) > 0.90:
        flags.append("mostly_stationary")
    if float(quality.get("max_abs_value", 0.0)) > 60.0:
        flags.append("physical_abs_value_outlier")
    if float(quality.get("spike_rate", 0.0)) > 0.02 or float(quality.get("extreme_value_fraction", 0.0)) > 0.0:
        flags.append("spiky_or_extreme")
    if float(quality.get("flatline_fraction", 0.0)) > 0.20 or float(quality.get("missing_rate", 0.0)) > 0.0:
        flags.append("flatline_or_missing")
    if float(quality.get("axis_imbalance", 1.0)) > 25.0:
        flags.append("axis_imbalance")
    if float(quality.get("timestamp_jitter_fraction", 0.0)) > 0.10 or float(quality.get("repeated_timestamp_fraction", 0.0)) > 0.01:
        flags.append("timestamp_irregularity")
    if _nan_stat(gyro_norm, "p95") < 0.02 and _nan_stat(acc_jerk, "p95") < 1.0:
        flags.append("low_motion_energy")
    if _nan_stat(gyro_jerk, "p95") > 25.0 or _nan_stat(acc_jerk, "p95") > 250.0:
        flags.append("high_jerk")
    return flags or ["clean_motion"]


def trace_verdict(flags: Sequence[str]) -> str:
    flag_set = set(flags)
    artifact_flags = {
        "low_quality_score",
        "physical_abs_value_outlier",
        "spiky_or_extreme",
        "flatline_or_missing",
        "axis_imbalance",
        "timestamp_irregularity",
    }
    if flag_set & artifact_flags:
        return "likely_artifact"
    if "mostly_stationary" in flag_set or "low_motion_energy" in flag_set:
        return "mostly_stationary"
    return "plausible_motion"


def summarize_audit(
    exact_rows: Sequence[Mapping[str, Any]],
    consensus_rows: Sequence[Mapping[str, Any]],
    plot_rows: Sequence[Mapping[str, Any]],
    *,
    top_ks: Sequence[int],
    quality_threshold: float,
    max_stationary_fraction: float,
    max_abs_value: float,
) -> dict[str, Any]:
    exact_ids = [_sample_id(row) for row in exact_rows]
    consensus_ids = [_sample_id(row) for row in consensus_rows]
    topk = {}
    for k in top_ks:
        exact_top = set(exact_ids[: min(k, len(exact_ids))])
        consensus_top = set(consensus_ids[: min(k, len(consensus_ids))])
        exact_slice = exact_rows[: min(k, len(exact_rows))]
        topk[str(k)] = {
            "exact_consensus_overlap": _fraction(len(exact_top & consensus_top), min(k, len(exact_top), len(consensus_top))),
            "exact_quality_fail_rate": _fraction(
                sum(1 for row in exact_slice if _float(row.get("quality_score"), 1.0) < quality_threshold),
                len(exact_slice),
            ),
            "exact_physical_fail_rate": _fraction(
                sum(
                    1
                    for row in exact_slice
                    if _float(row.get("stationary_fraction"), 0.0) > max_stationary_fraction
                    or _float(row.get("max_abs_value"), 0.0) > max_abs_value
                ),
                len(exact_slice),
            ),
            "exact_unique_clusters": len({str(row.get("new_cluster_id", "")) for row in exact_slice}),
        }
    verdict_counts = Counter(str(row.get("verdict", "unknown")) for row in plot_rows if row.get("plot_status") == "ok")
    group_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    for row in plot_rows:
        for group in str(row.get("audit_groups", "")).split("|"):
            if group:
                group_counts[group] += 1
        for flag in str(row.get("flags", "")).split("|"):
            if flag:
                flag_counts[flag] += 1
    return {
        "topk": topk,
        "group_counts": dict(group_counts),
        "verdict_counts": dict(verdict_counts),
        "flag_counts": dict(flag_counts),
        "plot_error_count": int(sum(1 for row in plot_rows if row.get("plot_status") != "ok")),
        "review_recommendation": _review_recommendation(verdict_counts),
    }


def write_trace_plot(
    path: Path,
    *,
    row: Mapping[str, Any],
    samples: np.ndarray,
    timestamps: np.ndarray | None,
    quality: Mapping[str, float],
    trace_summary: Mapping[str, Any],
    sample_rate: float,
    max_plot_points: int,
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
    acc_norm = np.linalg.norm(acc, axis=1)
    gyro_norm = np.linalg.norm(gyro, axis=1)
    acc_jerk = np.linalg.norm(np.diff(acc, axis=0, prepend=acc[:1]), axis=1) * sample_rate
    gyro_jerk = np.linalg.norm(np.diff(gyro, axis=0, prepend=gyro[:1]), axis=1) * sample_rate

    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
    fig.suptitle(_plot_title(row, trace_summary), fontsize=11)
    for channel, label in enumerate(("acc_x", "acc_y", "acc_z")):
        axes[0].plot(t, plotted[:, channel], linewidth=0.8, label=label)
    axes[0].set_ylabel("accel")
    axes[0].legend(loc="upper right", ncols=3, fontsize=8)

    for channel, label in enumerate(("gyro_x", "gyro_y", "gyro_z"), start=3):
        axes[1].plot(t, plotted[:, channel], linewidth=0.8, label=label)
    axes[1].set_ylabel("gyro")
    axes[1].legend(loc="upper right", ncols=3, fontsize=8)

    axes[2].plot(t, acc_norm[idx], linewidth=0.9, label="acc_norm")
    axes[2].plot(t, gyro_norm[idx], linewidth=0.9, label="gyro_norm")
    axes[2].set_ylabel("norm")
    axes[2].legend(loc="upper right", ncols=2, fontsize=8)

    axes[3].plot(t, acc_jerk[idx], linewidth=0.8, label="acc_jerk")
    axes[3].plot(t, gyro_jerk[idx], linewidth=0.8, label="gyro_jerk")
    axes[3].set_ylabel("jerk")
    axes[3].set_xlabel("seconds")
    axes[3].legend(loc="upper right", ncols=2, fontsize=8)

    for axis in axes:
        axis.grid(True, color="#dddddd", linewidth=0.5)
        axis.margins(x=0)

    fig.text(0.01, 0.006, _metrics_text(row, quality, trace_summary), ha="left", va="bottom", family="monospace", fontsize=8)
    fig.tight_layout(rect=(0, 0.075, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=135)
    plt.close(fig)


def load_topk_audit_pack_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_topk_audit_pack_config(config: Mapping[str, Any]) -> None:
    execution = _required_mapping(config, "execution")
    artifacts = _required_mapping(config, "artifacts")
    if execution.get("provider") != "modal":
        raise ValueError("Top-K audit pack must run on Modal.")
    allow_local_paths = bool(execution.get("allow_local_paths_for_tests", False))
    required = ("exact_diagnostics_path", "consensus_diagnostics_path", "output_dir")
    missing = [key for key in required if not str(artifacts.get(key, "")).strip()]
    if missing:
        raise ValueError(f"Top-K audit artifacts missing required paths: {missing}")
    if not allow_local_paths:
        for key in required:
            if not str(artifacts.get(key, "")).startswith("/artifacts"):
                raise ValueError(f"artifacts.{key} must be mounted under /artifacts.")
        data = _required_mapping(config, "data")
        if not str(data.get("root", "")).startswith("/data"):
            raise ValueError("data.root must be mounted under /data.")


def _raw_path_for_row(row: Mapping[str, Any], registry_lookup: Mapping[str, Any]) -> Path | None:
    raw_path = str(row.get("raw_path", "")).strip()
    if raw_path:
        return Path(raw_path)
    clip = registry_lookup.get(str(row.get("sample_id", "")))
    if clip is None:
        return None
    return Path(clip.raw_path)


def _success_row(
    row: Mapping[str, Any],
    raw_path: Path,
    plot_path: Path | None,
    quality: Mapping[str, float],
    trace_summary: Mapping[str, Any],
) -> dict[str, Any]:
    out = _base_plot_row(row)
    out.update(
        {
            "plot_status": "ok",
            "plot_path": "" if plot_path is None else str(plot_path),
            "raw_path": str(raw_path),
            "verdict": str(trace_summary.get("verdict", "unknown")),
            "flags": "|".join(str(flag) for flag in trace_summary.get("flags", [])),
        }
    )
    for key, value in quality.items():
        out[f"quality__{key}"] = float(value)
    for key, value in trace_summary.items():
        if key in {"flags", "verdict"}:
            continue
        out[f"trace__{key}"] = json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
    return out


def _error_row(row: Mapping[str, Any], error: str, raw_path: Path | None) -> dict[str, Any]:
    out = _base_plot_row(row)
    out.update(
        {
            "plot_status": "error",
            "plot_error": error,
            "plot_path": "",
            "raw_path": "" if raw_path is None else str(raw_path),
            "verdict": "unknown",
            "flags": "",
        }
    )
    return out


def _base_plot_row(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "audit_index",
        "audit_groups",
        "sample_id",
        "worker_id",
        "exact_rank",
        "consensus_rank",
        "rank_delta_exact_minus_consensus",
        "final_score",
        "blend_old_novelty_score",
        "quality_score",
        "stationary_fraction",
        "max_abs_value",
        "new_cluster_id",
        "new_cluster_size",
        "physical_validity_pass",
        "physical_validity_failure_reasons",
    ]
    return {key: row.get(key, "") for key in keys}


def _markdown_report(report: Mapping[str, Any], plot_rows: Sequence[Mapping[str, Any]]) -> str:
    summary = report["summary"]
    lines = [
        "# Exact-Window Top-K Audit Pack",
        "",
        f"Mode: `{report['mode']}`",
        "",
        "## Verdict",
        "",
        str(summary.get("review_recommendation", "")),
        "",
        "## Top-K Agreement And Hygiene",
        "",
        "| K | Exact/Consensus Overlap | Exact Quality Fail | Exact Physical Fail | Exact Unique Clusters |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for k, row in summary["topk"].items():
        lines.append(
            f"| {k} | {float(row['exact_consensus_overlap']):.4f} | "
            f"{float(row['exact_quality_fail_rate']):.4f} | {float(row['exact_physical_fail_rate']):.4f} | "
            f"{int(row['exact_unique_clusters'])} |"
        )
    lines.extend(
        [
            "",
            "## Automated Trace Review",
            "",
            f"Selected clips: {report['n_selected']}",
            f"Plots written: {report['n_plots_written']}",
            f"Verdict counts: `{dict(summary['verdict_counts'])}`",
            f"Flag counts: `{dict(summary['flag_counts'])}`",
            "",
            "## Highest-Priority Review Rows",
            "",
            "| Audit Index | Groups | Exact Rank | Consensus Rank | Verdict | Flags | Plot |",
            "| ---: | --- | ---: | ---: | --- | --- | --- |",
        ]
    )
    for row in plot_rows[:30]:
        plot = Path(str(row.get("plot_path", ""))).name if row.get("plot_path") else ""
        plot_link = f"[{plot}](trace_plots_{report['mode']}/{plot})" if plot else ""
        lines.append(
            f"| {row.get('audit_index', '')} | {row.get('audit_groups', '')} | {row.get('exact_rank', '')} | "
            f"{row.get('consensus_rank', '')} | {row.get('verdict', '')} | {row.get('flags', '')} | {plot_link} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "```text",
            json.dumps(report["artifacts"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _write_html_index(path: Path, report: Mapping[str, Any], plot_rows: Sequence[Mapping[str, Any]], *, output_dir: Path) -> None:
    cards = []
    for row in plot_rows:
        plot_path = str(row.get("plot_path", ""))
        if not plot_path:
            continue
        relative = Path(plot_path).relative_to(output_dir)
        title = html.escape(
            f"#{row.get('audit_index')} rank {row.get('exact_rank')} / consensus {row.get('consensus_rank')} "
            f"{row.get('verdict')}"
        )
        meta = html.escape(f"{row.get('audit_groups')} | {row.get('flags')}")
        cards.append(
            f"<section><h2>{title}</h2><p>{meta}</p><img src='{html.escape(str(relative))}' loading='lazy'></section>"
        )
    body = "\n".join(cards)
    path.write_text(
        f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Exact-Window Top-K Audit</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; background: #f7f7f5; color: #1f1f1d; }}
section {{ margin: 0 0 28px; padding: 16px; background: white; border: 1px solid #ddd; }}
h1, h2 {{ margin: 0 0 8px; }}
p {{ margin: 0 0 12px; color: #555; }}
img {{ width: 100%; max-width: 1500px; border: 1px solid #ccc; background: white; }}
pre {{ white-space: pre-wrap; }}
</style>
</head>
<body>
<h1>Exact-Window Top-K Audit</h1>
<pre>{html.escape(json.dumps(report.get("summary", {}), indent=2, sort_keys=True))}</pre>
{body}
</body>
</html>
""",
        encoding="utf-8",
    )


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _ranked_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    output.sort(key=lambda row: _optional_int(row.get("rank")) or 1_000_000)
    return output


def _by_sample_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {_sample_id(row): row for row in rows if _sample_id(row)}


def _sample_id(row: Mapping[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("worker_id") or "")


def _fails_hygiene(
    row: Mapping[str, Any],
    *,
    quality_threshold: float,
    max_stationary_fraction: float,
    max_abs_value: float,
) -> bool:
    if str(row.get("quality_gate_pass", "")).lower() == "false":
        return True
    if str(row.get("physical_validity_pass", "")).lower() == "false":
        return True
    return (
        _float(row.get("quality_score"), 1.0) < quality_threshold
        or _float(row.get("stationary_fraction"), 0.0) > max_stationary_fraction
        or _float(row.get("max_abs_value"), 0.0) > max_abs_value
    )


def _score_for_rejected(row: Mapping[str, Any]) -> float:
    for key in ("blend_old_novelty_score", "ranker_score", "final_score", "score"):
        if key in row:
            return _float(row.get(key), 0.0)
    return 0.0


def _group_priority(groups: object) -> int:
    text = "|".join(str(group) for group in groups) if isinstance(groups, list) else str(groups)
    if "exact:top" in text:
        return 0
    if "consensus:top" in text:
        return 1
    if "rank_disagreement" in text:
        return 2
    if "high_novelty_rejected" in text:
        return 3
    return 4


def _rank_delta(left: object, right: object) -> int | None:
    left_int = _optional_int(left)
    right_int = _optional_int(right)
    if left_int is None or right_int is None:
        return None
    return int(left_int - right_int)


def _plot_title(row: Mapping[str, Any], trace_summary: Mapping[str, Any]) -> str:
    return (
        f"audit {int(row.get('audit_index', 0)):03d} | exact rank={row.get('exact_rank')} | "
        f"consensus rank={row.get('consensus_rank')} | verdict={trace_summary.get('verdict')} | "
        f"sample={str(row.get('sample_id', ''))[:12]}"
    )


def _metrics_text(row: Mapping[str, Any], quality: Mapping[str, float], trace_summary: Mapping[str, Any]) -> str:
    pieces = [
        f"groups={row.get('audit_groups', '')}",
        f"score={_float(row.get('final_score'), 0.0):.4f}",
        f"blend={_float(row.get('blend_old_novelty_score'), 0.0):.4f}",
        f"q={float(quality.get('quality_score', 0.0)):.4f}",
        f"stationary={float(quality.get('stationary_fraction', 0.0)):.4f}",
        f"max_abs={float(quality.get('max_abs_value', 0.0)):.2f}",
        f"spike={float(quality.get('spike_rate', 0.0)):.5f}",
        f"acc_p95={float(trace_summary.get('acc_norm_p95', 0.0)):.3f}",
        f"gyro_p95={float(trace_summary.get('gyro_norm_p95', 0.0)):.3f}",
        f"flags={','.join(str(flag) for flag in trace_summary.get('flags', []))}",
    ]
    return " | ".join(pieces)


def _time_axis(samples: np.ndarray, timestamps: np.ndarray | None, *, sample_rate: float) -> np.ndarray:
    if timestamps is not None and len(timestamps) == len(samples):
        ts = np.asarray(timestamps, dtype=float)
        finite = np.isfinite(ts)
        if np.any(finite):
            first = float(ts[finite][0])
            return ts - first
    return np.arange(len(samples), dtype=float) / sample_rate


def _dominant_periodicity(values: np.ndarray, *, sample_rate: float) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    if len(x) < int(sample_rate * 4):
        return {"period_sec": 0.0, "strength": 0.0}
    x = x - float(np.nanmean(x))
    norm = float(np.linalg.norm(x))
    if not np.isfinite(norm) or norm <= 1.0e-8:
        return {"period_sec": 0.0, "strength": 0.0}
    x = x / norm
    min_lag = max(1, int(0.35 * sample_rate))
    max_lag = min(len(x) - 1, int(3.0 * sample_rate))
    if max_lag <= min_lag:
        return {"period_sec": 0.0, "strength": 0.0}
    corrs = [float(np.dot(x[:-lag], x[lag:])) for lag in range(min_lag, max_lag + 1)]
    best_offset = int(np.argmax(corrs))
    best_lag = min_lag + best_offset
    return {"period_sec": float(best_lag / sample_rate), "strength": float(corrs[best_offset])}


def _nan_stat(values: np.ndarray, stat: str) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return 0.0
    if stat == "mean":
        return float(np.nanmean(arr))
    if stat == "p95":
        return float(np.nanpercentile(arr, 95))
    raise ValueError(f"Unsupported stat: {stat}")


def _review_recommendation(verdict_counts: Mapping[str, int]) -> str:
    artifacts = int(verdict_counts.get("likely_artifact", 0))
    stationary = int(verdict_counts.get("mostly_stationary", 0))
    plausible = int(verdict_counts.get("plausible_motion", 0))
    total = artifacts + stationary + plausible
    if total == 0:
        return "No plots were successfully reviewed; rerun the audit before drawing conclusions."
    if artifacts > max(2, total * 0.10):
        return "Automated review found a non-trivial artifact count. Inspect flagged plots before promoting the ranking."
    if stationary > max(3, total * 0.15):
        return "Automated review found many mostly-stationary clips. Check whether the selector is rewarding low-motion novelty."
    return "Automated trace review is broadly clean: most audited clips look like plausible motion rather than obvious artifacts."


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:80]


def _optional_int(value: object) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Top-K audit config requires object field '{key}'.")
    return value
