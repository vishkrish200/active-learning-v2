from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from marginal_value.active.final_blend_rank import run_active_final_blend_rank, validate_active_final_blend_rank_config
from marginal_value.logging_utils import log_event
from marginal_value.submit.finalize_submission import finalize_submission_ids


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
    consensus = write_consensus_artifacts(config, runs, output_dir=output_dir, mode=mode, k_values=k_values)
    summary["consensus"] = consensus["summary"]
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
            "consensus_submission": consensus["submission_path"],
            "consensus_diagnostics": consensus["diagnostics_path"],
            "consensus_finalized_worker_id": consensus["finalized_worker_id_path"],
            "consensus_finalized_new_worker_id": consensus["finalized_new_worker_id_path"],
            "consensus_report": consensus["report_path"],
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
        "consensus_submission_path": consensus["submission_path"],
        "consensus_diagnostics_path": consensus["diagnostics_path"],
        "consensus_finalized_worker_id_path": consensus["finalized_worker_id_path"],
        "consensus_finalized_new_worker_id_path": consensus["finalized_new_worker_id_path"],
        "consensus_report_path": consensus["report_path"],
    }
    log_event("support_sampling_stability", "done", **result)
    return result


def build_consensus_ranking(
    runs: Sequence[Mapping[str, object]],
    *,
    selector_name: str,
) -> list[dict[str, object]]:
    if not runs:
        raise ValueError("Consensus ranking requires at least one run.")
    run_rows = []
    candidate_sets = []
    for run in runs:
        rows = [dict(row) for row in run["diagnostics_rows"]]  # type: ignore[index]
        rows.sort(key=lambda row: int(row.get("rank", 0)))
        by_id = {str(row["worker_id"]): row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError(f"Run '{run.get('label', run.get('seed', 'run'))}' contains duplicate worker IDs.")
        run_rows.append(by_id)
        candidate_sets.append(set(by_id))
    expected = candidate_sets[0]
    for idx, candidate_set in enumerate(candidate_sets[1:], start=2):
        if candidate_set != expected:
            missing = sorted(expected - candidate_set)[:5]
            extra = sorted(candidate_set - expected)[:5]
            raise ValueError(
                "Consensus ranking requires every support-sample run to rank the same candidates; "
                f"run {idx} differs. missing={missing}, extra={extra}"
            )

    n_candidates = len(expected)
    n_runs = len(run_rows)
    consensus_rows = []
    for worker_id in sorted(expected):
        member_rows = [by_id[worker_id] for by_id in run_rows]
        ranks = [int(row["rank"]) for row in member_rows]
        scores = [_safe_float(row.get("score", row.get("final_score", 0.0))) for row in member_rows]
        borda_score = float(sum(n_candidates - rank + 1 for rank in ranks))
        best_row = dict(min(member_rows, key=lambda row: int(row["rank"])))
        mean_rank = _mean([float(rank) for rank in ranks])
        mean_score = _mean(scores)
        best_row["selector"] = selector_name
        best_row["reranker"] = selector_name
        best_row["support_seed_count"] = int(n_runs)
        best_row["consensus_mean_rank"] = mean_rank
        best_row["consensus_rank_std"] = _std([float(rank) for rank in ranks])
        best_row["consensus_best_rank"] = int(min(ranks))
        best_row["consensus_worst_rank"] = int(max(ranks))
        best_row["consensus_mean_score"] = mean_score
        best_row["consensus_borda_score"] = borda_score
        best_row["score"] = float(borda_score / max(1.0, float(n_runs * n_candidates)))
        best_row["rerank_score"] = best_row["score"]
        best_row["reason_code"] = _consensus_reason_code(best_row)
        consensus_rows.append(best_row)

    consensus_rows.sort(
        key=lambda row: (
            _safe_float(row["consensus_mean_rank"]),
            -_safe_float(row["consensus_borda_score"]),
            -_safe_float(row["consensus_mean_score"]),
            str(row["worker_id"]),
        )
    )
    for rank, row in enumerate(consensus_rows, start=1):
        row["rank"] = int(rank)
    return consensus_rows


def write_consensus_artifacts(
    config: Mapping[str, Any],
    runs: Sequence[Mapping[str, object]],
    *,
    output_dir: Path,
    mode: str,
    k_values: Sequence[int],
) -> dict[str, object]:
    ranking_config = config["ranking"]
    selector_name = _consensus_selector_name(
        str(ranking_config["left_representation"]),
        str(ranking_config["right_representation"]),
        float(ranking_config.get("alpha", 0.5)),
    )
    rows = build_consensus_ranking(runs, selector_name=selector_name)
    submission_path = output_dir / f"active_final_blend_consensus_submission_{mode}.csv"
    diagnostics_path = output_dir / f"active_final_blend_consensus_diagnostics_{mode}.csv"
    report_path = output_dir / f"active_final_blend_consensus_report_{mode}.json"
    finalized_worker_id_path = output_dir / f"active_final_blend_consensus_submission_{mode}_worker_id.csv"
    finalized_new_worker_id_path = output_dir / f"active_final_blend_consensus_submission_{mode}_new_worker_id.csv"

    _write_rows(submission_path, _submission_rows(rows))
    _write_rows(diagnostics_path, rows)
    manifest_path = _manifest_path(config, str(ranking_config.get("query_split", "new")))
    finalize_submission_ids(
        submission_path=submission_path,
        manifest_path=manifest_path,
        output_path=finalized_worker_id_path,
        input_id_column="worker_id",
        output_id_column="worker_id",
    )
    finalize_submission_ids(
        submission_path=submission_path,
        manifest_path=manifest_path,
        output_path=finalized_new_worker_id_path,
        input_id_column="worker_id",
        output_id_column="new_worker_id",
    )

    summary = {
        "selector": selector_name,
        "n_runs": len(runs),
        "n_rows": len(rows),
        "topk_quality": _topk_quality(rows, k_values=k_values),
        "rank_std_mean": _mean([_safe_float(row.get("consensus_rank_std")) for row in rows]),
        "rank_std_top10_mean": _mean([_safe_float(row.get("consensus_rank_std")) for row in rows[: min(10, len(rows))]]),
        "mean_rank_top10_mean": _mean([_safe_float(row.get("consensus_mean_rank")) for row in rows[: min(10, len(rows))]]),
    }
    report = {
        "mode": mode,
        "selector": selector_name,
        "aggregation": "mean_rank_then_borda",
        "seeds": [int(run["seed"]) for run in runs if run.get("seed") is not None],
        "n_runs": len(runs),
        "n_rows": len(rows),
        "summary": summary,
        "artifacts": {
            "submission": str(submission_path),
            "diagnostics": str(diagnostics_path),
            "finalized_worker_id": str(finalized_worker_id_path),
            "finalized_new_worker_id": str(finalized_new_worker_id_path),
            "report": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "summary": summary,
        "submission_path": str(submission_path),
        "diagnostics_path": str(diagnostics_path),
        "finalized_worker_id_path": str(finalized_worker_id_path),
        "finalized_new_worker_id_path": str(finalized_new_worker_id_path),
        "report_path": str(report_path),
    }


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


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    row_list = [dict(row) for row in rows]
    fieldnames = sorted({key for row in row_list for key in row})
    if "worker_id" in fieldnames:
        fieldnames = ["worker_id", *[field for field in fieldnames if field != "worker_id"]]
    if "rank" in fieldnames:
        fieldnames = [field for field in fieldnames if field != "rank"]
        insert_at = 1 if fieldnames and fieldnames[0] == "worker_id" else 0
        fieldnames.insert(insert_at, "rank")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_list)


def _submission_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    previous_score: float | None = None
    for row in rows:
        score = _safe_float(row.get("rerank_score", row.get("score", 0.0)))
        if previous_score is not None and score >= previous_score:
            score = previous_score - 1.0e-12
        previous_score = score
        output.append(
            {
                "worker_id": row["worker_id"],
                "rank": int(row["rank"]),
                "score": float(score),
                "quality_score": _safe_float(row.get("quality_score", 1.0), default=1.0),
                "reason_code": row.get("reason_code", ""),
            }
        )
    return output


def _manifest_path(config: Mapping[str, Any], split: str) -> Path:
    data = config["data"]
    path = Path(str(data["manifests"][split]))
    if path.is_absolute():
        return path
    return Path(str(data["root"])) / path


def _consensus_selector_name(left_representation: str, right_representation: str, alpha: float) -> str:
    alpha_int = int(round(float(alpha) * 10))
    return f"consensus_blend_kcenter_{left_representation}_{right_representation}_a{alpha_int:02d}"


def _consensus_reason_code(row: Mapping[str, object]) -> str:
    base = str(row.get("reason_code", "")).strip()
    if not base:
        return "CONSENSUS"
    parts = [part for part in base.split("|") if part]
    if "CONSENSUS" not in parts:
        parts.append("CONSENSUS")
    return "|".join(parts)


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
    consensus = summary.get("consensus")
    if isinstance(consensus, Mapping):
        lines.extend(
            [
                "",
                "## Consensus",
                "",
                f"Selector: `{consensus['selector']}`",
                "",
                "| K | Quality Fail | Physical Fail | Duplicate |",
                "|---:|---:|---:|---:|",
            ]
        )
        topk_quality = consensus["topk_quality"]  # type: ignore[index]
        for k in k_values:
            row = topk_quality[f"k{k}"]
            lines.append(
                f"| {k} | {row['quality_failure_rate']:.4f} | "
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


def _std(values: Sequence[float]) -> float:
    return float(np.std(np.asarray(values, dtype=float))) if values else 0.0


def _min(values: Sequence[float]) -> float:
    return float(np.min(np.asarray(values, dtype=float))) if values else 0.0


def _max(values: Sequence[float]) -> float:
    return float(np.max(np.asarray(values, dtype=float))) if values else 0.0


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0
