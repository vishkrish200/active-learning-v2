from __future__ import annotations

import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from marginal_value.active.registry import load_clip_registry_from_config
from marginal_value.active.topk_audit_pack import summarize_trace, write_trace_plot
from marginal_value.logging_utils import log_event, log_progress
from marginal_value.preprocessing.quality import compute_quality_features, load_modal_jsonl_imu


def run_trace_gate_audit_pack(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    validate_trace_gate_audit_pack_config(config)
    mode = "smoke" if smoke else "full"
    artifacts = config["artifacts"]
    audit = config.get("audit", {})
    output_dir = Path(str(artifacts["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    current_rows = _ranked_rows(_read_csv(artifacts["current_diagnostics_path"]))
    trace_rows = _ranked_rows(_read_csv(artifacts["trace_gate_diagnostics_path"]))
    if not current_rows:
        raise ValueError(f"No current diagnostics rows found: {artifacts['current_diagnostics_path']}")
    if not trace_rows:
        raise ValueError(f"No trace-gate diagnostics rows found: {artifacts['trace_gate_diagnostics_path']}")

    top_n = int(audit.get("top_n_smoke" if smoke else "top_n", 10))
    compare_k = int(audit.get("compare_k", 50))
    max_plot_points = int(audit.get("max_plot_points", 1800))
    sample_rate = float(config.get("quality", {}).get("sample_rate", 30.0))
    max_samples = config.get("quality", {}).get("max_samples_per_clip", 5400)
    max_samples = int(max_samples) if max_samples is not None else None
    generate_plots = bool(audit.get("generate_plots", True))

    log_event(
        "trace_gate_audit_pack",
        "start",
        mode=mode,
        current_diagnostics_path=str(artifacts["current_diagnostics_path"]),
        trace_gate_diagnostics_path=str(artifacts["trace_gate_diagnostics_path"]),
        output_dir=str(output_dir),
        top_n=top_n,
        compare_k=compare_k,
        generate_plots=generate_plots,
    )

    selected = select_trace_gate_audit_candidates(
        current_rows,
        trace_rows,
        top_n=top_n,
        compare_k=compare_k,
    )
    if smoke:
        selected = selected[: int(audit.get("smoke_max_selected", 8))]

    registry_lookup: dict[str, Any] = {}
    if config.get("data") and config.get("data", {}).get("manifests"):
        registry = load_clip_registry_from_config(config)
        registry_lookup = registry.by_sample_id

    plot_dir = output_dir / f"trace_gate_plots_{mode}"
    if generate_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)
    plot_rows: list[dict[str, Any]] = []
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
        except Exception as exc:  # pragma: no cover - remote audit should record all row-level failures
            plot_rows.append(_error_row(row, f"{type(exc).__name__}:{exc}", raw_path))
        log_progress(
            "trace_gate_audit_pack",
            "plot_progress",
            index=index,
            total=len(selected),
            every=progress_every,
            sample_id=sample_id,
        )

    selected_path = output_dir / f"trace_gate_audit_selected_{mode}.csv"
    plot_index_path = output_dir / f"trace_gate_audit_plot_index_{mode}.csv"
    report_path = output_dir / f"trace_gate_audit_report_{mode}.json"
    markdown_path = output_dir / f"trace_gate_audit_report_{mode}.md"
    html_path = output_dir / f"trace_gate_audit_index_{mode}.html"

    summary = summarize_trace_gate_audit(plot_rows)
    _write_rows(selected_path, selected)
    _write_rows(plot_index_path, plot_rows)
    report = {
        "mode": mode,
        "current_diagnostics_path": str(artifacts["current_diagnostics_path"]),
        "trace_gate_diagnostics_path": str(artifacts["trace_gate_diagnostics_path"]),
        "n_selected": len(selected),
        "n_plots_written": int(sum(1 for row in plot_rows if row.get("plot_status") == "ok" and row.get("plot_path"))),
        "top_n": top_n,
        "compare_k": compare_k,
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
        "promoted_added_artifact_count": int(summary["group_verdict_counts"].get("trace_gate:added_top50", {}).get("likely_artifact", 0)),
        "trace_top_artifact_count": int(summary["group_verdict_counts"].get("trace_gate:top", {}).get("likely_artifact", 0)),
    }
    log_event("trace_gate_audit_pack", "done", **result)
    return result


def select_trace_gate_audit_candidates(
    current_rows: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
    *,
    top_n: int,
    compare_k: int,
) -> list[dict[str, Any]]:
    current = _ranked_rows(current_rows)
    trace = _ranked_rows(trace_rows)
    current_by_id = {_sample_id(row): row for row in current if _sample_id(row)}
    trace_by_id = {_sample_id(row): row for row in trace if _sample_id(row)}
    current_top = [_sample_id(row) for row in current[:compare_k]]
    trace_top = [_sample_id(row) for row in trace[:compare_k]]
    current_top_set = set(current_top)
    trace_top_set = set(trace_top)
    selected: dict[str, dict[str, Any]] = {}

    def add(sample_id: str, group: str) -> None:
        if not sample_id:
            return
        out = selected.get(sample_id)
        current_row = current_by_id.get(sample_id)
        trace_row = trace_by_id.get(sample_id)
        source = trace_row or current_row
        if source is None:
            return
        if out is None:
            out = dict(source)
            out["sample_id"] = sample_id
            out["current_rank"] = _optional_int(current_row.get("rank")) if current_row is not None else None
            out["trace_gate_rank"] = _optional_int(trace_row.get("rank")) if trace_row is not None else None
            out["exact_rank"] = out["current_rank"]
            out["consensus_rank"] = out["trace_gate_rank"]
            out["rank_delta_exact_minus_consensus"] = _rank_delta(out.get("current_rank"), out.get("trace_gate_rank"))
            out["audit_groups"] = []
            selected[sample_id] = out
        groups = list(out.get("audit_groups", []))
        if group not in groups:
            groups.append(group)
        out["audit_groups"] = groups

    for row in trace[:top_n]:
        add(_sample_id(row), "trace_gate:top")
    for sample_id in current_top:
        if sample_id not in trace_top_set:
            add(sample_id, "current:removed_top50")
    for sample_id in trace_top:
        if sample_id not in current_top_set:
            add(sample_id, "trace_gate:added_top50")

    rows = list(selected.values())
    rows.sort(
        key=lambda row: (
            _group_priority(row.get("audit_groups", [])),
            _optional_int(row.get("trace_gate_rank")) or 1_000_000,
            _optional_int(row.get("current_rank")) or 1_000_000,
            str(row.get("sample_id", "")),
        )
    )
    for index, row in enumerate(rows, start=1):
        row["audit_index"] = index
        row["audit_groups"] = "|".join(str(group) for group in row.get("audit_groups", []))
    return rows


def summarize_trace_gate_audit(plot_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    verdict_counts = Counter(str(row.get("verdict", "unknown")) for row in plot_rows if row.get("plot_status") == "ok")
    group_counts: Counter[str] = Counter()
    group_verdict_counts: dict[str, Counter[str]] = defaultdict(Counter)
    flag_counts: Counter[str] = Counter()
    for row in plot_rows:
        verdict = str(row.get("verdict", "unknown"))
        for group in str(row.get("audit_groups", "")).split("|"):
            if not group:
                continue
            group_counts[group] += 1
            group_verdict_counts[group][verdict] += 1
        for flag in str(row.get("flags", "")).split("|"):
            if flag:
                flag_counts[flag] += 1
    return {
        "verdict_counts": dict(verdict_counts),
        "group_counts": dict(group_counts),
        "group_verdict_counts": {group: dict(counts) for group, counts in group_verdict_counts.items()},
        "flag_counts": dict(flag_counts),
        "plot_error_count": int(sum(1 for row in plot_rows if row.get("plot_status") != "ok")),
    }


def validate_trace_gate_audit_pack_config(config: Mapping[str, Any]) -> None:
    execution = _required_mapping(config, "execution")
    artifacts = _required_mapping(config, "artifacts")
    if execution.get("provider") != "modal":
        raise ValueError("Trace-gate audit pack must run on Modal.")
    allow_local_paths = bool(execution.get("allow_local_paths_for_tests", False))
    required = ("current_diagnostics_path", "trace_gate_diagnostics_path", "output_dir")
    missing = [key for key in required if not str(artifacts.get(key, "")).strip()]
    if missing:
        raise ValueError(f"Trace-gate audit artifacts missing required paths: {missing}")
    if not allow_local_paths:
        for key in required:
            if not str(artifacts.get(key, "")).startswith("/artifacts"):
                raise ValueError(f"artifacts.{key} must be mounted under /artifacts.")
        data = _required_mapping(config, "data")
        if not str(data.get("root", "")).startswith("/data"):
            raise ValueError("data.root must be mounted under /data.")


def load_trace_gate_audit_pack_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
        "current_rank",
        "trace_gate_rank",
        "exact_rank",
        "consensus_rank",
        "rank_delta_exact_minus_consensus",
        "original_rank",
        "original_score",
        "final_score",
        "blend_old_novelty_score",
        "quality_score",
        "stationary_fraction",
        "max_abs_value",
        "quality__spike_rate",
        "new_cluster_id",
        "new_cluster_size",
        "physical_validity_pass",
        "physical_validity_failure_reasons",
    ]
    return {key: row.get(key, "") for key in keys}


def _raw_path_for_row(row: Mapping[str, Any], registry_lookup: Mapping[str, Any]) -> Path | None:
    raw_path = str(row.get("raw_path", "")).strip()
    if raw_path:
        return Path(raw_path)
    clip = registry_lookup.get(str(row.get("sample_id", "")))
    if clip is None:
        return None
    return Path(clip.raw_path)


def _markdown_report(report: Mapping[str, Any], plot_rows: Sequence[Mapping[str, Any]]) -> str:
    summary = report["summary"]
    lines = [
        "# Trace-Gate Targeted Audit Pack",
        "",
        f"Mode: `{report['mode']}`",
        f"Selected clips: `{report['n_selected']}`",
        f"Plots written: `{report['n_plots_written']}`",
        "",
        "## Summary",
        "",
        f"Verdict counts: `{dict(summary['verdict_counts'])}`",
        f"Group counts: `{dict(summary['group_counts'])}`",
        f"Group verdict counts: `{dict(summary['group_verdict_counts'])}`",
        f"Flag counts: `{dict(summary['flag_counts'])}`",
        "",
        "## Review Rows",
        "",
        "| Audit Index | Groups | Current Rank | Trace-Gate Rank | Verdict | Flags | Plot |",
        "| ---: | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in plot_rows:
        plot = Path(str(row.get("plot_path", ""))).name if row.get("plot_path") else ""
        plot_link = f"[{plot}](trace_gate_plots_{report['mode']}/{plot})" if plot else ""
        lines.append(
            f"| {row.get('audit_index', '')} | {row.get('audit_groups', '')} | {row.get('current_rank', '')} | "
            f"{row.get('trace_gate_rank', '')} | {row.get('verdict', '')} | {row.get('flags', '')} | {plot_link} |"
        )
    lines.extend(["", "## Artifacts", "", "```text", json.dumps(report["artifacts"], indent=2, sort_keys=True), "```", ""])
    return "\n".join(lines)


def _write_html_index(path: Path, report: Mapping[str, Any], plot_rows: Sequence[Mapping[str, Any]], *, output_dir: Path) -> None:
    cards = []
    for row in plot_rows:
        plot_path = str(row.get("plot_path", ""))
        if not plot_path:
            continue
        relative = Path(plot_path).relative_to(output_dir)
        title = html.escape(
            f"#{row.get('audit_index')} current {row.get('current_rank')} / trace {row.get('trace_gate_rank')} "
            f"{row.get('verdict')}"
        )
        meta = html.escape(f"{row.get('audit_groups')} | {row.get('flags')}")
        cards.append(
            f"<section><h2>{title}</h2><p>{meta}</p><img src='{html.escape(str(relative))}' loading='lazy'></section>"
        )
    path.write_text(
        f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Trace-Gate Targeted Audit</title>
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
<h1>Trace-Gate Targeted Audit</h1>
<pre>{html.escape(json.dumps(report.get("summary", {}), indent=2, sort_keys=True))}</pre>
{chr(10).join(cards)}
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


def _sample_id(row: Mapping[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("worker_id") or "")


def _group_priority(groups: object) -> int:
    text = "|".join(str(group) for group in groups) if isinstance(groups, list) else str(groups)
    if "trace_gate:top" in text:
        return 0
    if "trace_gate:added_top50" in text:
        return 1
    if "current:removed_top50" in text:
        return 2
    return 3


def _rank_delta(left: object, right: object) -> int | None:
    left_int = _optional_int(left)
    right_int = _optional_int(right)
    if left_int is None or right_int is None:
        return None
    return int(left_int - right_int)


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:80]


def _optional_int(value: object) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Trace-gate audit config requires object field '{key}'.")
    return value
