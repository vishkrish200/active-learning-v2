from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from marginal_value.active.final_blend_rank import run_active_final_blend_rank, validate_active_final_blend_rank_config
from marginal_value.logging_utils import log_event


DEFAULT_STABILITY_K_VALUES = (10, 50, 100)


def run_support_sampling_stability(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    validate_support_sampling_stability_config(config)
    mode = "smoke" if smoke else "full"
    output_dir = Path(str(config["artifacts"]["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    stability = config["stability"]
    seeds = [int(seed) for seed in stability.get("right_support_seeds", [1, 2, 3])]
    if smoke:
        seeds = seeds[: int(config["execution"].get("smoke_max_seeds", 1))]
    k_values = [int(k) for k in stability.get("k_values", DEFAULT_STABILITY_K_VALUES)]
    log_event("support_sampling_stability", "start", mode=mode, seeds=seeds)

    runs: list[dict[str, object]] = []
    for seed in seeds:
        seed_config = copy.deepcopy(config)
        seed_config["ranking"]["right_support_seed"] = int(seed)
        seed_config["artifacts"]["output_dir"] = str(output_dir / f"seed_{seed:06d}")
        result = run_active_final_blend_rank(seed_config, smoke=smoke)
        diagnostics_path = Path(str(result["diagnostics_path"]))
        runs.append(
            {
                "label": f"seed_{seed}",
                "seed": int(seed),
                "result": result,
                "diagnostics_path": str(diagnostics_path),
                "diagnostics_rows": _read_rows(diagnostics_path),
            }
        )

    summary = summarize_rank_stability(runs, k_values=k_values)
    report_path = output_dir / f"support_sampling_stability_report_{mode}.json"
    markdown_path = output_dir / f"support_sampling_stability_report_{mode}.md"
    report = {
        "mode": mode,
        "seeds": seeds,
        "k_values": k_values,
        "summary": summary,
        "runs": [
            {
                "label": str(run["label"]),
                "seed": int(run["seed"]),
                "diagnostics_path": str(run["diagnostics_path"]),
                "result": run["result"],
            }
            for run in runs
        ],
        "artifacts": {
            "report": str(report_path),
            "markdown": str(markdown_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_stability_markdown(report), encoding="utf-8")
    result = {
        "mode": mode,
        "n_runs": len(runs),
        "seeds": seeds,
        "report_path": str(report_path),
        "markdown_path": str(markdown_path),
    }
    log_event("support_sampling_stability", "done", **result)
    return result


def summarize_rank_stability(runs: Sequence[Mapping[str, object]], *, k_values: Sequence[int]) -> dict[str, object]:
    if len(runs) < 2:
        raise ValueError("Rank stability requires at least two runs.")
    ordered_runs = [_run_view(run, k_values=k_values) for run in runs]
    pair_rows = []
    for left_idx in range(len(ordered_runs)):
        for right_idx in range(left_idx + 1, len(ordered_runs)):
            pair_rows.append(_pair_summary(ordered_runs[left_idx], ordered_runs[right_idx], k_values=k_values))
    pairwise: dict[str, object] = {
        "n_pairs": len(pair_rows),
        "rank_spearman_mean": _mean([float(row["rank_spearman"]) for row in pair_rows]),
        "rank_spearman_min": _min([float(row["rank_spearman"]) for row in pair_rows]),
        "score_mean_abs_delta_mean": _mean([float(row["score_mean_abs_delta"]) for row in pair_rows]),
        "score_mean_abs_delta_max": _max([float(row["score_mean_abs_delta"]) for row in pair_rows]),
    }
    for k in k_values:
        values = [float(row[f"top{k}_overlap"]) for row in pair_rows]
        pairwise[f"top{k}_overlap_mean"] = _mean(values)
        pairwise[f"top{k}_overlap_min"] = _min(values)
    return {
        "n_runs": len(ordered_runs),
        "pairwise": pairwise,
        "pairs": pair_rows,
        "runs": {
            str(run["label"]): {
                "seed": int(run["seed"]) if run.get("seed") is not None else None,
                "n_rows": len(run["rows"]),
                "topk_quality": run["topk_quality"],
            }
            for run in ordered_runs
        },
    }


def validate_support_sampling_stability_config(config: Mapping[str, Any]) -> None:
    validate_active_final_blend_rank_config(config)
    stability = config.get("stability")
    if not isinstance(stability, Mapping):
        raise ValueError("support sampling stability config requires object field 'stability'.")
    seeds = stability.get("right_support_seeds", [])
    if not isinstance(seeds, list | tuple) or len(seeds) < 2:
        raise ValueError("stability.right_support_seeds must contain at least two seeds.")
    for seed in seeds:
        int(seed)
    for k in stability.get("k_values", DEFAULT_STABILITY_K_VALUES):
        if int(k) <= 0:
            raise ValueError("stability.k_values must contain positive integers.")


def _run_view(run: Mapping[str, object], *, k_values: Sequence[int]) -> dict[str, object]:
    rows = [dict(row) for row in run["diagnostics_rows"]]  # type: ignore[index]
    rows.sort(key=lambda row: int(row.get("rank", 0)))
    return {
        "label": str(run.get("label", run.get("seed", "run"))),
        "seed": run.get("seed"),
        "rows": rows,
        "rank_by_id": {str(row["worker_id"]): int(row["rank"]) for row in rows},
        "score_by_id": {str(row["worker_id"]): _safe_float(row.get("score", row.get("final_score", 0.0))) for row in rows},
        "topk": {int(k): {str(row["worker_id"]) for row in rows[: min(int(k), len(rows))]} for k in k_values},
        "topk_quality": _topk_quality(rows, k_values=k_values),
    }


def _pair_summary(left: Mapping[str, object], right: Mapping[str, object], *, k_values: Sequence[int]) -> dict[str, object]:
    left_ranks: Mapping[str, int] = left["rank_by_id"]  # type: ignore[assignment]
    right_ranks: Mapping[str, int] = right["rank_by_id"]  # type: ignore[assignment]
    common_ids = sorted(set(left_ranks) & set(right_ranks))
    left_scores: Mapping[str, float] = left["score_by_id"]  # type: ignore[assignment]
    right_scores: Mapping[str, float] = right["score_by_id"]  # type: ignore[assignment]
    row: dict[str, object] = {
        "left": str(left["label"]),
        "right": str(right["label"]),
        "common_count": len(common_ids),
        "rank_spearman": _rank_correlation([left_ranks[sample_id] for sample_id in common_ids], [right_ranks[sample_id] for sample_id in common_ids]),
        "score_mean_abs_delta": _mean([abs(left_scores[sample_id] - right_scores[sample_id]) for sample_id in common_ids]),
    }
    for k in k_values:
        left_top = left["topk"][int(k)]  # type: ignore[index]
        right_top = right["topk"][int(k)]  # type: ignore[index]
        denominator = min(int(k), len(left_top), len(right_top))
        row[f"top{int(k)}_overlap"] = float(len(left_top & right_top) / denominator) if denominator else 0.0
    return row


def _topk_quality(rows: Sequence[Mapping[str, object]], *, k_values: Sequence[int]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for k in k_values:
        selected = list(rows[: min(int(k), len(rows))])
        n = len(selected)
        clusters = {str(row.get("new_cluster_id", "")) for row in selected}
        quality_fail = sum(not _as_bool(row.get("quality_gate_pass", False)) for row in selected)
        physical_fail = sum(not _as_bool(row.get("physical_validity_pass", False)) for row in selected)
        output[f"k{int(k)}"] = {
            "selected_count": float(n),
            "quality_failure_rate": _fraction(quality_fail, n),
            "physical_failure_rate": _fraction(physical_fail, n),
            "duplicate_rate": 1.0 - _fraction(len(clusters), n),
            "unique_new_clusters": float(len(clusters)),
        }
    return output


def _rank_correlation(left: Sequence[int], right: Sequence[int]) -> float:
    if len(left) < 2 or len(right) < 2:
        return 1.0
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    left_array = left_array - float(left_array.mean())
    right_array = right_array - float(right_array.mean())
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    if denominator == 0.0:
        return 1.0
    return float(np.dot(left_array, right_array) / denominator)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _stability_markdown(report: Mapping[str, object]) -> str:
    summary: Mapping[str, object] = report["summary"]  # type: ignore[assignment]
    pairwise: Mapping[str, object] = summary["pairwise"]  # type: ignore[index,assignment]
    k_values = [int(k) for k in report["k_values"]]  # type: ignore[index]
    lines = [
        "# Support Sampling Stability",
        "",
        f"Mode: `{report['mode']}`",
        f"Seeds: `{', '.join(str(seed) for seed in report['seeds'])}`",  # type: ignore[index]
        "",
        "## Pairwise Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| rank_spearman_mean | {float(pairwise['rank_spearman_mean']):.4f} |",
        f"| rank_spearman_min | {float(pairwise['rank_spearman_min']):.4f} |",
        f"| score_mean_abs_delta_mean | {float(pairwise['score_mean_abs_delta_mean']):.6f} |",
    ]
    for k in k_values:
        lines.append(f"| top{k}_overlap_mean | {float(pairwise[f'top{k}_overlap_mean']):.4f} |")
        lines.append(f"| top{k}_overlap_min | {float(pairwise[f'top{k}_overlap_min']):.4f} |")
    lines.extend(["", "## Run Hygiene", "", "| Run | K | Quality Fail | Physical Fail | Duplicate |", "|---|---:|---:|---:|---:|"])
    runs: Mapping[str, object] = summary["runs"]  # type: ignore[index,assignment]
    for label, run_summary in runs.items():
        topk_quality: Mapping[str, Mapping[str, float]] = run_summary["topk_quality"]  # type: ignore[index,assignment]
        for k in k_values:
            row = topk_quality[f"k{k}"]
            lines.append(
                f"| {label} | {k} | {row['quality_failure_rate']:.4f} | "
                f"{row['physical_failure_rate']:.4f} | {row['duplicate_rate']:.4f} |"
            )
    lines.append("")
    return "\n".join(lines)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(result):
        return default
    return result


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float))) if values else 0.0


def _min(values: Sequence[float]) -> float:
    return float(np.min(np.asarray(values, dtype=float))) if values else 0.0


def _max(values: Sequence[float]) -> float:
    return float(np.max(np.asarray(values, dtype=float))) if values else 0.0


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0
