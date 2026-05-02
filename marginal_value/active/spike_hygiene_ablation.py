from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from marginal_value.active.registry import load_clip_registry_from_config
from marginal_value.active.topk_audit_pack import summarize_trace
from marginal_value.logging_utils import log_event, log_progress
from marginal_value.preprocessing.quality import compute_quality_features, load_modal_jsonl_imu
from marginal_value.submit.finalize_submission import finalize_submission_ids


DEFAULT_TOP_KS = (10, 50, 100, 200)
SPIKE_FEATURE_COLUMNS = (
    "quality__spike_rate",
    "quality__extreme_value_fraction",
    "quality__high_frequency_energy",
    "trace__acc_norm_p95",
    "trace__gyro_norm_p95",
    "trace__acc_jerk_p95",
    "trace__gyro_jerk_p95",
)


def run_spike_hygiene_ablation(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    validate_spike_hygiene_ablation_config(config)
    mode = "smoke" if smoke else "full"
    artifacts = config["artifacts"]
    ablation = config.get("ablation", {})
    output_dir = Path(str(artifacts["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    exact_path = Path(str(artifacts["exact_diagnostics_path"]))
    rows = _ranked_rows(_read_csv(exact_path))
    if smoke:
        rows = rows[: int(config.get("execution", {}).get("smoke_max_rows", ablation.get("smoke_max_rows", 128)))]
    if not rows:
        raise ValueError(f"No exact-window diagnostics rows found: {exact_path}")

    spike_rate_threshold = float(ablation.get("spike_rate_threshold", 0.025))
    top_k_values = [int(k) for k in ablation.get("top_k_values", DEFAULT_TOP_KS)]
    sample_rate = float(config.get("quality", {}).get("sample_rate", 30.0))
    max_samples = config.get("quality", {}).get("max_samples_per_clip", 5400)
    max_samples = int(max_samples) if max_samples is not None else None
    generate_submissions = bool(ablation.get("generate_submissions", True)) and not smoke

    log_event(
        "spike_hygiene_ablation",
        "start",
        mode=mode,
        exact_diagnostics_path=str(exact_path),
        output_dir=str(output_dir),
        n_rows=len(rows),
        spike_rate_threshold=spike_rate_threshold,
        generate_submissions=generate_submissions,
    )

    registry_lookup: dict[str, Any] = {}
    if config.get("data"):
        registry = load_clip_registry_from_config(config)
        registry_lookup = registry.by_sample_id

    enriched_rows = enrich_spike_features(
        rows,
        registry_lookup=registry_lookup,
        sample_rate=sample_rate,
        max_samples=max_samples,
    )
    hard_rows = build_spike_ablation_rows(enriched_rows, mode="hard_gate", spike_rate_threshold=spike_rate_threshold)
    soft_rows = build_spike_ablation_rows(enriched_rows, mode="soft_penalty", spike_rate_threshold=spike_rate_threshold)
    trace_rows = build_spike_ablation_rows(enriched_rows, mode="trace_gate", spike_rate_threshold=spike_rate_threshold)

    enriched_path = output_dir / f"spike_hygiene_ablation_enriched_current_{mode}.csv"
    hard_path = output_dir / f"spike_hygiene_ablation_hard_gate_diagnostics_{mode}.csv"
    soft_path = output_dir / f"spike_hygiene_ablation_soft_penalty_diagnostics_{mode}.csv"
    trace_path = output_dir / f"spike_hygiene_ablation_trace_gate_diagnostics_{mode}.csv"
    report_path = output_dir / f"spike_hygiene_ablation_report_{mode}.json"
    markdown_path = output_dir / f"spike_hygiene_ablation_report_{mode}.md"

    _write_rows(enriched_path, enriched_rows)
    _write_rows(hard_path, hard_rows)
    _write_rows(soft_path, soft_rows)
    _write_rows(trace_path, trace_rows)

    artifacts_out: dict[str, str] = {
        "enriched_current_diagnostics": str(enriched_path),
        "hard_gate_diagnostics": str(hard_path),
        "soft_penalty_diagnostics": str(soft_path),
        "trace_gate_diagnostics": str(trace_path),
        "report": str(report_path),
        "markdown": str(markdown_path),
    }
    if generate_submissions:
        artifacts_out.update(_write_submission_artifacts(config, output_dir, hard_rows, mode=mode, label="hard_gate"))
        artifacts_out.update(_write_submission_artifacts(config, output_dir, soft_rows, mode=mode, label="soft_penalty"))
        artifacts_out.update(_write_submission_artifacts(config, output_dir, trace_rows, mode=mode, label="trace_gate"))

    summary = {
        "current": summarize_spike_ranking(
            enriched_rows,
            baseline_rows=enriched_rows,
            top_k_values=top_k_values,
            spike_rate_threshold=spike_rate_threshold,
        ),
        "hard_gate": summarize_spike_ranking(
            hard_rows,
            baseline_rows=enriched_rows,
            top_k_values=top_k_values,
            spike_rate_threshold=spike_rate_threshold,
        ),
        "soft_penalty": summarize_spike_ranking(
            soft_rows,
            baseline_rows=enriched_rows,
            top_k_values=top_k_values,
            spike_rate_threshold=spike_rate_threshold,
        ),
        "trace_gate": summarize_spike_ranking(
            trace_rows,
            baseline_rows=enriched_rows,
            top_k_values=top_k_values,
            spike_rate_threshold=spike_rate_threshold,
        ),
    }
    report = {
        "mode": mode,
        "exact_diagnostics_path": str(exact_path),
        "n_rows": len(enriched_rows),
        "spike_rate_threshold": spike_rate_threshold,
        "top_k_values": top_k_values,
        "summary": summary,
        "artifacts": artifacts_out,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")

    result = {
        "mode": mode,
        "report_path": str(report_path),
        "markdown_path": str(markdown_path),
        "hard_gate_diagnostics_path": str(hard_path),
        "soft_penalty_diagnostics_path": str(soft_path),
        "trace_gate_diagnostics_path": str(trace_path),
        "hard_gate_top50_spike_fail_rate": summary["hard_gate"]["topk"].get("50", {}).get("spike_fail_rate", 0.0),
        "soft_penalty_top50_spike_fail_rate": summary["soft_penalty"]["topk"].get("50", {}).get("spike_fail_rate", 0.0),
        "trace_gate_top50_trace_fail_rate": summary["trace_gate"]["topk"].get("50", {}).get("trace_fail_rate", 0.0),
    }
    log_event("spike_hygiene_ablation", "done", **result)
    return result


def enrich_spike_features(
    rows: Sequence[Mapping[str, Any]],
    *,
    registry_lookup: Mapping[str, Any],
    sample_rate: float,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    total = len(rows)
    progress_every = max(1, total // 10) if total else 1
    for index, row in enumerate(rows, start=1):
        out = dict(row)
        out["original_rank"] = _optional_int(row.get("rank")) or index
        if _has_spike_features(out):
            enriched.append(_normalize_spike_columns(out))
        else:
            raw_path = _raw_path_for_row(out, registry_lookup)
            if raw_path is None or not raw_path.exists():
                raise FileNotFoundError(f"Missing raw path for sample_id={_sample_id(out)}: {raw_path}")
            samples, timestamps = load_modal_jsonl_imu(raw_path, max_samples=max_samples)
            quality = compute_quality_features(samples, timestamps=timestamps, sample_rate=sample_rate)
            trace = summarize_trace(samples, timestamps=timestamps, quality=quality, sample_rate=sample_rate)
            for key, value in quality.items():
                out[f"quality__{key}"] = float(value)
            for key, value in trace.items():
                if key in {"flags", "verdict"}:
                    out[f"trace__{key}"] = "|".join(str(item) for item in value) if isinstance(value, list) else str(value)
                elif isinstance(value, dict):
                    out[f"trace__{key}"] = json.dumps(value, sort_keys=True)
                else:
                    out[f"trace__{key}"] = value
            enriched.append(_normalize_spike_columns(out))
        log_progress(
            "spike_hygiene_ablation",
            "feature_progress",
            index=index,
            total=total,
            every=progress_every,
            sample_id=_sample_id(out),
        )
    return enriched


def build_spike_ablation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    spike_rate_threshold: float,
    soft_penalty_strength: float = 5.0,
    soft_penalty_min_multiplier: float = 0.25,
) -> list[dict[str, Any]]:
    ranked = []
    for row in _ranked_rows(rows):
        out = dict(row)
        spike_rate = _float(out.get("quality__spike_rate"), 0.0)
        spike_pass = spike_rate <= spike_rate_threshold
        trace_pass = spike_pass and not _trace_artifact_fail(out)
        multiplier = spike_penalty_multiplier(
            spike_rate,
            threshold=spike_rate_threshold,
            strength=soft_penalty_strength,
            minimum=soft_penalty_min_multiplier,
        )
        out["spike_hygiene_mode"] = mode
        out["spike_rate_threshold"] = float(spike_rate_threshold)
        out["spike_hygiene_pass"] = bool(spike_pass)
        out["trace_hygiene_pass"] = bool(trace_pass)
        out["spike_penalty_multiplier"] = float(multiplier)
        out["original_rank"] = _optional_int(out.get("original_rank")) or _optional_int(out.get("rank")) or 1_000_000
        out["original_score"] = _float(out.get("score", out.get("rerank_score", out.get("final_score"))), 0.0)
        out["spike_adjusted_score"] = float(out["original_score"]) * multiplier
        ranked.append(out)

    if mode == "hard_gate":
        ranked.sort(
            key=lambda row: (
                not bool(row["spike_hygiene_pass"]),
                _optional_int(row.get("original_rank")) or 1_000_000,
                _sample_id(row),
            )
        )
    elif mode == "soft_penalty":
        ranked.sort(
            key=lambda row: (
                -_float(row.get("spike_adjusted_score"), 0.0),
                _optional_int(row.get("original_rank")) or 1_000_000,
                _sample_id(row),
            )
        )
    elif mode == "trace_gate":
        ranked.sort(
            key=lambda row: (
                not bool(row["trace_hygiene_pass"]),
                _optional_int(row.get("original_rank")) or 1_000_000,
                _sample_id(row),
            )
        )
    else:
        raise ValueError(f"Unsupported spike hygiene ablation mode: {mode}")

    total = max(1, len(ranked))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = int(rank)
        row["selector"] = f"spike_hygiene_{mode}"
        row["reranker"] = f"spike_hygiene_{mode}"
        row["score"] = float((total - rank + 1) / total)
        row["rerank_score"] = row["score"]
    return ranked


def spike_penalty_multiplier(spike_rate: float, *, threshold: float, strength: float, minimum: float) -> float:
    if threshold <= 0:
        return 1.0
    if spike_rate <= threshold:
        return 1.0
    excess_ratio = (spike_rate - threshold) / threshold
    return float(max(minimum, 1.0 - strength * excess_ratio))


def summarize_spike_ranking(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_rows: Sequence[Mapping[str, Any]],
    top_k_values: Sequence[int],
    spike_rate_threshold: float,
) -> dict[str, Any]:
    ranked = _ranked_rows(rows)
    baseline = _ranked_rows(baseline_rows)
    baseline_ids = [_sample_id(row) for row in baseline]
    baseline_top50 = set(baseline_ids[: min(50, len(baseline_ids))])
    baseline_spike_top50 = {
        _sample_id(row)
        for row in baseline[: min(50, len(baseline))]
        if _float(row.get("quality__spike_rate"), 0.0) > spike_rate_threshold
    }
    baseline_trace_top50 = {
        _sample_id(row)
        for row in baseline[: min(50, len(baseline))]
        if _trace_artifact_fail(row)
    }
    topk = {}
    for k in top_k_values:
        subset = ranked[: min(int(k), len(ranked))]
        ids = {_sample_id(row) for row in subset}
        topk[str(int(k))] = {
            "n_rows": len(subset),
            "spike_fail_rate": _fraction(
                sum(1 for row in subset if _float(row.get("quality__spike_rate"), 0.0) > spike_rate_threshold),
                len(subset),
            ),
            "trace_fail_rate": _fraction(sum(1 for row in subset if _trace_artifact_fail(row)), len(subset)),
            "quality_fail_rate": _fraction(sum(1 for row in subset if _float(row.get("quality_score"), 1.0) < 0.85), len(subset)),
            "physical_fail_rate": _fraction(
                sum(
                    1
                    for row in subset
                    if str(row.get("physical_validity_pass", "")).lower() == "false"
                    or _float(row.get("stationary_fraction"), 0.0) > 0.90
                    or _float(row.get("max_abs_value"), 0.0) > 60.0
                ),
                len(subset),
            ),
            "unique_clusters": len({str(row.get("new_cluster_id", "")) for row in subset}),
            "mean_original_rank": _mean([float(_optional_int(row.get("original_rank")) or 0) for row in subset]),
            "mean_spike_rate": _mean([_float(row.get("quality__spike_rate"), 0.0) for row in subset]),
            "mean_stationary_fraction": _mean([_float(row.get("stationary_fraction"), 0.0) for row in subset]),
            "overlap_with_current": _fraction(len(ids & set(baseline_ids[: min(int(k), len(baseline_ids))])), min(int(k), len(ids))),
        }
    return {
        "topk": topk,
        "baseline_top50_spike_fail_count": len(baseline_spike_top50),
        "baseline_top50_spike_fail_removed_count": len(baseline_spike_top50 - {_sample_id(row) for row in ranked[: min(50, len(ranked))]}),
        "baseline_top50_trace_fail_count": len(baseline_trace_top50),
        "baseline_top50_trace_fail_removed_count": len(baseline_trace_top50 - {_sample_id(row) for row in ranked[: min(50, len(ranked))]}),
        "verdict_counts_top50": dict(Counter(_spike_verdict(row, spike_rate_threshold=spike_rate_threshold) for row in ranked[: min(50, len(ranked))])),
        "trace_verdict_counts_top50": dict(Counter(str(row.get("trace__verdict", "")) for row in ranked[: min(50, len(ranked))])),
    }


def validate_spike_hygiene_ablation_config(config: Mapping[str, Any]) -> None:
    execution = _required_mapping(config, "execution")
    artifacts = _required_mapping(config, "artifacts")
    if execution.get("provider") != "modal":
        raise ValueError("Spike hygiene ablation must run on Modal.")
    allow_local_paths = bool(execution.get("allow_local_paths_for_tests", False))
    for key in ("exact_diagnostics_path", "output_dir"):
        if not str(artifacts.get(key, "")).strip():
            raise ValueError(f"artifacts.{key} is required.")
        if not allow_local_paths and not str(artifacts[key]).startswith("/artifacts"):
            raise ValueError(f"artifacts.{key} must be mounted under /artifacts.")
    if config.get("data") and not allow_local_paths and not str(config["data"].get("root", "")).startswith("/data"):
        raise ValueError("data.root must be mounted under /data.")


def load_spike_hygiene_ablation_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_submission_artifacts(
    config: Mapping[str, Any],
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    label: str,
) -> dict[str, str]:
    submission_path = output_dir / f"spike_hygiene_ablation_{label}_submission_{mode}.csv"
    worker_id_path = output_dir / f"spike_hygiene_ablation_{label}_submission_{mode}_worker_id.csv"
    new_worker_id_path = output_dir / f"spike_hygiene_ablation_{label}_submission_{mode}_new_worker_id.csv"
    _write_rows(submission_path, _submission_rows(rows))
    manifest_path = _manifest_path(config)
    finalize_submission_ids(
        submission_path=submission_path,
        manifest_path=manifest_path,
        output_path=worker_id_path,
        input_id_column="worker_id",
        output_id_column="worker_id",
    )
    finalize_submission_ids(
        submission_path=submission_path,
        manifest_path=manifest_path,
        output_path=new_worker_id_path,
        input_id_column="worker_id",
        output_id_column="new_worker_id",
    )
    return {
        f"{label}_submission": str(submission_path),
        f"{label}_submission_worker_id": str(worker_id_path),
        f"{label}_submission_new_worker_id": str(new_worker_id_path),
    }


def _submission_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
    return [
        {
            "rank": int(_optional_int(row.get("rank")) or idx),
            "worker_id": _sample_id(row),
            "score": float(_float(row.get("score"), 0.0)),
        }
        for idx, row in enumerate(rows, start=1)
    ]


def _manifest_path(config: Mapping[str, Any]) -> Path:
    data = _required_mapping(config, "data")
    manifests = _required_mapping(data, "manifests")
    split = str(config.get("ranking", {}).get("query_split", "new"))
    if split not in manifests:
        raise ValueError(f"data.manifests must include query split '{split}' to finalize submissions.")
    root = Path(str(data["root"]))
    return root / str(manifests[split])


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Spike Hygiene Ablation",
        "",
        f"Mode: `{report['mode']}`",
        f"Spike-rate threshold: `{report['spike_rate_threshold']}`",
        "",
        "## Top-K Summary",
        "",
        "| Variant | K | Spike Fail | Trace Fail | Current Overlap | Unique Clusters | Mean Original Rank | Removed Top-50 Spike | Removed Top-50 Trace |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, summary in report["summary"].items():
        for k, row in summary["topk"].items():
            lines.append(
                f"| {variant} | {k} | {float(row['spike_fail_rate']):.4f} | "
                f"{float(row['trace_fail_rate']):.4f} | "
                f"{float(row['overlap_with_current']):.4f} | {int(row['unique_clusters'])} | "
                f"{float(row['mean_original_rank']):.2f} | {int(summary['baseline_top50_spike_fail_removed_count'])} | "
                f"{int(summary['baseline_top50_trace_fail_removed_count'])} |"
            )
    lines.extend(["", "## Artifacts", "", "```text", json.dumps(report["artifacts"], indent=2, sort_keys=True), "```", ""])
    return "\n".join(lines)


def _normalize_spike_columns(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in SPIKE_FEATURE_COLUMNS:
        out[key] = _float(out.get(key), 0.0)
    if "quality__quality_score" in out:
        out["quality_score"] = _float(out.get("quality_score", out["quality__quality_score"]), _float(out["quality__quality_score"]))
    if "quality__stationary_fraction" in out:
        out["stationary_fraction"] = _float(
            out.get("stationary_fraction", out["quality__stationary_fraction"]),
            _float(out["quality__stationary_fraction"]),
        )
    if "quality__max_abs_value" in out:
        out["max_abs_value"] = _float(out.get("max_abs_value", out["quality__max_abs_value"]), _float(out["quality__max_abs_value"]))
    return out


def _has_spike_features(row: Mapping[str, Any]) -> bool:
    return "quality__spike_rate" in row and str(row.get("quality__spike_rate", "")).strip() != ""


def _raw_path_for_row(row: Mapping[str, Any], registry_lookup: Mapping[str, Any]) -> Path | None:
    raw_path = str(row.get("raw_path", "")).strip()
    if raw_path:
        return Path(raw_path)
    clip = registry_lookup.get(_sample_id(row))
    if clip is None:
        return None
    return Path(clip.raw_path)


def _spike_verdict(row: Mapping[str, Any], *, spike_rate_threshold: float) -> str:
    return "spike_fail" if _float(row.get("quality__spike_rate"), 0.0) > spike_rate_threshold else "spike_pass"


def _trace_artifact_fail(row: Mapping[str, Any]) -> bool:
    verdict = str(row.get("trace__verdict", "")).strip().lower()
    return verdict in {"likely_artifact", "mostly_stationary"}


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


def _mean(values: Sequence[float]) -> float:
    clean = [float(value) for value in values if np.isfinite(float(value))]
    return float(sum(clean) / len(clean)) if clean else 0.0


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Spike hygiene ablation config requires object field '{key}'.")
    return value
