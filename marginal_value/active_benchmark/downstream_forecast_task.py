from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Mapping, Sequence

import numpy as np

from marginal_value.active_benchmark.coverage_runner import CoverageRunResult


def build_downstream_forecast_rows(
    result: CoverageRunResult,
    samples_by_id: Mapping[str, np.ndarray],
    *,
    history_steps: int = 8,
    horizon_steps: int = 1,
    ridge_alpha: float = 1.0e-2,
    max_windows_per_clip: int = 128,
) -> list[dict[str, object]]:
    _validate_forecast_config(
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        ridge_alpha=ridge_alpha,
        max_windows_per_clip=max_windows_per_clip,
    )
    selected_by_key = _selected_ids_by_episode_policy_budget(result)
    budgets = (0, *tuple(int(budget) for budget in result.budgets))
    rows: list[dict[str, object]] = []
    for episode in result.episodes:
        baseline_mse = autoregressive_forecast_mse(
            samples_by_id,
            train_ids=episode.support_ids,
            target_ids=episode.target_ids,
            history_steps=history_steps,
            horizon_steps=horizon_steps,
            ridge_alpha=ridge_alpha,
            max_windows_per_clip=max_windows_per_clip,
        )
        for policy_id in result.policies:
            for budget_k in budgets:
                selected_ids = selected_by_key.get((episode.episode_id, str(policy_id), int(budget_k)), ())
                train_ids_after = (*episode.support_ids, *selected_ids)
                after_mse = autoregressive_forecast_mse(
                    samples_by_id,
                    train_ids=train_ids_after,
                    target_ids=episode.target_ids,
                    history_steps=history_steps,
                    horizon_steps=horizon_steps,
                    ridge_alpha=ridge_alpha,
                    max_windows_per_clip=max_windows_per_clip,
                )
                absolute_reduction = float(baseline_mse - after_mse)
                relative_reduction = float(absolute_reduction / baseline_mse) if baseline_mse > 1.0e-12 else 0.0
                rows.append(
                    {
                        "episode_id": str(episode.episode_id),
                        "fold_id": int(episode.fold_id),
                        "policy_id": str(policy_id),
                        "budget_k": int(budget_k),
                        "selected_ids": list(selected_ids),
                        "support_count_before": int(len(episode.support_ids)),
                        "support_count_after": int(len(train_ids_after)),
                        "target_count": int(len(episode.target_ids)),
                        "history_steps": int(history_steps),
                        "horizon_steps": int(horizon_steps),
                        "ridge_alpha": float(ridge_alpha),
                        "max_windows_per_clip": int(max_windows_per_clip),
                        "baseline_mse": float(baseline_mse),
                        "after_mse": float(after_mse),
                        "absolute_mse_reduction": float(absolute_reduction),
                        "relative_mse_reduction": float(relative_reduction),
                    }
                )
    return rows


def build_downstream_forecast_report(
    result: CoverageRunResult,
    samples_by_id: Mapping[str, np.ndarray],
    *,
    history_steps: int = 8,
    horizon_steps: int = 1,
    ridge_alpha: float = 1.0e-2,
    max_windows_per_clip: int = 128,
    top_policy: str = "ts2vec_kcenter_v1",
    baseline_policy: str = "quality_stratified_random_v1",
) -> dict[str, object]:
    rows = build_downstream_forecast_rows(
        result,
        samples_by_id,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        ridge_alpha=ridge_alpha,
        max_windows_per_clip=max_windows_per_clip,
    )
    summary = summarize_downstream_forecast(rows)
    return {
        "input": {
            "episode_count": int(len(result.episodes)),
            "policy_count": int(len(result.policies)),
            "policies": list(result.policies),
            "budgets": [0, *[int(budget) for budget in result.budgets]],
            "history_steps": int(history_steps),
            "horizon_steps": int(horizon_steps),
            "ridge_alpha": float(ridge_alpha),
            "max_windows_per_clip": int(max_windows_per_clip),
            "top_policy": str(top_policy),
            "baseline_policy": str(baseline_policy),
        },
        "summary": summary,
        "decision": _decision(summary, top_policy=str(top_policy), baseline_policy=str(baseline_policy)),
        "rows": rows,
    }


def write_downstream_forecast_reports(report: Mapping[str, object], output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "downstream_forecast_task_report.json"
    markdown_path = root / "downstream_forecast_task_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def summarize_downstream_forecast(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    rows_list = [dict(row) for row in rows]
    final_budget = max((int(row["budget_k"]) for row in rows_list), default=0)
    final_rows = [row for row in rows_list if int(row["budget_k"]) == final_budget]
    return {
        "row_count": int(len(rows_list)),
        "final_budget": int(final_budget),
        "final_row_count": int(len(final_rows)),
        "policy_budget_means": _policy_budget_means(rows_list),
        "policy_final_means": _policy_means(final_rows),
        "final_episode_wins": _final_episode_wins(final_rows),
    }


def autoregressive_forecast_mse(
    samples_by_id: Mapping[str, np.ndarray],
    *,
    train_ids: Sequence[str],
    target_ids: Sequence[str],
    history_steps: int = 8,
    horizon_steps: int = 1,
    ridge_alpha: float = 1.0e-2,
    max_windows_per_clip: int = 128,
) -> float:
    train_x, train_y = _stack_autoregressive_examples(
        samples_by_id,
        train_ids,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        max_windows_per_clip=max_windows_per_clip,
    )
    target_x, target_y = _stack_autoregressive_examples(
        samples_by_id,
        target_ids,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        max_windows_per_clip=max_windows_per_clip,
    )
    if target_x.size == 0:
        return 0.0
    if train_x.size == 0:
        return float("inf")
    model = _fit_ridge_forecaster(train_x, train_y, ridge_alpha=float(ridge_alpha))
    prediction = _predict_ridge_forecaster(model, target_x)
    return float(np.mean((target_y - prediction) ** 2))


def _stack_autoregressive_examples(
    samples_by_id: Mapping[str, np.ndarray],
    sample_ids: Sequence[str],
    *,
    history_steps: int,
    horizon_steps: int,
    max_windows_per_clip: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for sample_id in sample_ids:
        if str(sample_id) not in samples_by_id:
            raise ValueError(f"Missing raw samples for downstream forecast clip: {sample_id}")
        x_clip, y_clip = _autoregressive_examples_for_clip(
            np.asarray(samples_by_id[str(sample_id)], dtype=float),
            history_steps=history_steps,
            horizon_steps=horizon_steps,
            max_windows_per_clip=max_windows_per_clip,
        )
        if x_clip.size:
            x_parts.append(x_clip)
            y_parts.append(y_clip)
    if not x_parts:
        return np.empty((0, 0), dtype=float), np.empty((0, 0), dtype=float)
    return np.vstack(x_parts), np.vstack(y_parts)


def _autoregressive_examples_for_clip(
    samples: np.ndarray,
    *,
    history_steps: int,
    horizon_steps: int,
    max_windows_per_clip: int,
) -> tuple[np.ndarray, np.ndarray]:
    if samples.ndim != 2:
        raise ValueError("Raw IMU samples must be a 2D array of shape [time, channels].")
    if samples.shape[1] <= 0:
        raise ValueError("Raw IMU samples must contain at least one channel.")
    clean = np.asarray(samples, dtype=float)
    if not np.all(np.isfinite(clean)):
        clean = np.nan_to_num(clean, nan=0.0, posinf=0.0, neginf=0.0)
    n_windows = clean.shape[0] - int(history_steps) - int(horizon_steps) + 1
    if n_windows <= 0:
        return np.empty((0, clean.shape[1] * int(history_steps)), dtype=float), np.empty((0, clean.shape[1]), dtype=float)
    starts = np.arange(n_windows, dtype=int)
    if int(max_windows_per_clip) > 0 and starts.size > int(max_windows_per_clip):
        starts = np.unique(np.linspace(0, starts.size - 1, num=int(max_windows_per_clip), dtype=int))
    x = np.vstack([clean[start : start + int(history_steps)].reshape(1, -1) for start in starts])
    y_indices = starts + int(history_steps) + int(horizon_steps) - 1
    y = clean[y_indices]
    return x, y


def _fit_ridge_forecaster(train_x: np.ndarray, train_y: np.ndarray, *, ridge_alpha: float) -> dict[str, np.ndarray]:
    x_mean = np.mean(train_x, axis=0, keepdims=True)
    x_std = np.std(train_x, axis=0, keepdims=True)
    x_std = np.where(x_std > 1.0e-8, x_std, 1.0)
    y_mean = np.mean(train_y, axis=0, keepdims=True)
    x_norm = (train_x - x_mean) / x_std
    y_centered = train_y - y_mean
    eye = np.eye(x_norm.shape[1], dtype=float)
    weights = np.linalg.solve(x_norm.T @ x_norm + float(ridge_alpha) * eye, x_norm.T @ y_centered)
    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "weights": weights}


def _predict_ridge_forecaster(model: Mapping[str, np.ndarray], target_x: np.ndarray) -> np.ndarray:
    x_norm = (target_x - model["x_mean"]) / model["x_std"]
    return x_norm @ model["weights"] + model["y_mean"]


def _selected_ids_by_episode_policy_budget(result: CoverageRunResult) -> dict[tuple[str, str, int], tuple[str, ...]]:
    ranked: dict[tuple[str, str, int], list[tuple[int, str]]] = defaultdict(list)
    for row in result.selected_rows:
        ranked[(str(row.episode_id), str(row.policy_id), int(row.budget_k))].append((int(row.rank_index), str(row.sample_id)))
    return {
        key: tuple(sample_id for _rank_index, sample_id in sorted(values))
        for key, values in ranked.items()
    }


def _policy_budget_means(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, dict[str, float | int]]]:
    by_policy_budget: dict[str, dict[int, list[Mapping[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_policy_budget[str(row["policy_id"])][int(row["budget_k"])].append(row)
    return {
        policy: {str(budget): _mean_row_values(budget_rows) for budget, budget_rows in sorted(budget_map.items())}
        for policy, budget_map in sorted(by_policy_budget.items())
    }


def _policy_means(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, float | int]]:
    by_policy: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_policy[str(row["policy_id"])].append(row)
    return {policy: _mean_row_values(policy_rows) for policy, policy_rows in sorted(by_policy.items())}


def _mean_row_values(rows: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    return {
        "row_count": int(len(rows)),
        "mean_baseline_mse": _mean(rows, "baseline_mse"),
        "mean_after_mse": _mean(rows, "after_mse"),
        "median_after_mse": _median(rows, "after_mse"),
        "mean_absolute_mse_reduction": _mean(rows, "absolute_mse_reduction"),
        "mean_relative_mse_reduction": _mean(rows, "relative_mse_reduction"),
    }


def _final_episode_wins(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    by_episode: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_episode[str(row["episode_id"])].append(row)
    wins: dict[str, int] = defaultdict(int)
    for episode_rows in by_episode.values():
        best = min(float(row["after_mse"]) for row in episode_rows)
        for row in episode_rows:
            if abs(float(row["after_mse"]) - best) <= 1.0e-12:
                wins[str(row["policy_id"])] += 1
    return dict(sorted(wins.items()))


def _decision(summary: Mapping[str, object], *, top_policy: str, baseline_policy: str) -> dict[str, object]:
    policy_means = summary.get("policy_final_means", {})
    if not isinstance(policy_means, Mapping):
        policy_means = {}
    top = policy_means.get(top_policy, {})
    baseline = policy_means.get(baseline_policy, {})
    top_after = _metric(top, "mean_after_mse")
    baseline_after = _metric(baseline, "mean_after_mse")
    top_gain = _metric(top, "mean_relative_mse_reduction")
    baseline_gain = _metric(baseline, "mean_relative_mse_reduction")
    mse_delta = baseline_after - top_after if top_after is not None and baseline_after is not None else None
    gain_delta = top_gain - baseline_gain if top_gain is not None and baseline_gain is not None else None
    return {
        "downstream_task": "raw_imu_autoregressive_forecast",
        "model_update": "actual_retrain_after_acquisition",
        "large_training": "hold_until_repeated_real_task_runs",
        "top_policy": str(top_policy),
        "baseline_policy": str(baseline_policy),
        "top_after_mse": top_after,
        "baseline_after_mse": baseline_after,
        "after_mse_delta_vs_baseline": mse_delta,
        "top_mean_relative_mse_reduction": top_gain,
        "baseline_mean_relative_mse_reduction": baseline_gain,
        "relative_mse_reduction_delta_vs_baseline": gain_delta,
        "read": (
            "Actual downstream model-update task: raw IMU autoregressive forecaster is retrained after acquisition; "
            "this is not a source-family pseudo-label proxy."
        ),
        "next_steps": [
            "Run this on the same blocked seed/pool episodes before any neural downstream model.",
            "Use paired target MSE deltas as the primary downstream metric.",
            "Keep TS2Vec retraining separate from this downstream task.",
        ],
    }


def _markdown_report(report: Mapping[str, object]) -> str:
    decision = report.get("decision", {})
    summary = report.get("summary", {})
    policy_means = summary.get("policy_final_means", {}) if isinstance(summary, Mapping) else {}
    budget_means = summary.get("policy_budget_means", {}) if isinstance(summary, Mapping) else {}
    lines = [
        "# Downstream Forecast Task",
        "",
        "Actual Downstream Model Update: selected raw IMU clips are added to support, a ridge autoregressive forecaster is retrained, and held-out target forecast MSE is measured.",
        "",
        "## Decision",
        "",
        f"- downstream task: `{decision.get('downstream_task')}`",
        f"- model update: `{decision.get('model_update')}`",
        f"- large training: `{decision.get('large_training')}`",
        f"- top policy: `{decision.get('top_policy')}`",
        f"- baseline policy: `{decision.get('baseline_policy')}`",
        f"- after-MSE delta vs baseline: `{_fmt(decision.get('after_mse_delta_vs_baseline'))}`",
        f"- relative reduction delta vs baseline: `{_fmt(decision.get('relative_mse_reduction_delta_vs_baseline'))}`",
        f"- read: {decision.get('read')}",
        "",
        "## Final Means",
        "",
        "| policy | after MSE | relative reduction | rows |",
        "|---|---:|---:|---:|",
    ]
    if isinstance(policy_means, Mapping):
        for policy, row in sorted(policy_means.items(), key=lambda item: _metric(item[1], "mean_after_mse") or float("inf")):
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "| `{policy}` | {after} | {rel} | {rows} |".format(
                    policy=policy,
                    after=_fmt(row.get("mean_after_mse")),
                    rel=_fmt(row.get("mean_relative_mse_reduction")),
                    rows=int(row.get("row_count", 0)),
                )
            )
    lines.extend(["", "## Budget Means", "", "| policy | budget | after MSE | relative reduction | rows |", "|---|---:|---:|---:|---:|"])
    if isinstance(budget_means, Mapping):
        for policy, budget_map in sorted(budget_means.items()):
            if not isinstance(budget_map, Mapping):
                continue
            for budget, row in sorted(budget_map.items(), key=lambda item: int(item[0])):
                if not isinstance(row, Mapping):
                    continue
                lines.append(
                    "| `{policy}` | {budget} | {after} | {rel} | {rows} |".format(
                        policy=policy,
                        budget=budget,
                        after=_fmt(row.get("mean_after_mse")),
                        rel=_fmt(row.get("mean_relative_mse_reduction")),
                        rows=int(row.get("row_count", 0)),
                    )
                )
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "- This is a real model update task, but it is still a small linear forecaster, not final challenge-label training.",
            "- Budget `0` is the support-only model. Later budgets retrain after adding selected candidate clips.",
        ]
    )
    return "\n".join(lines) + "\n"


def _validate_forecast_config(
    *,
    history_steps: int,
    horizon_steps: int,
    ridge_alpha: float,
    max_windows_per_clip: int,
) -> None:
    if int(history_steps) <= 0:
        raise ValueError("history_steps must be positive.")
    if int(horizon_steps) <= 0:
        raise ValueError("horizon_steps must be positive.")
    if float(ridge_alpha) < 0.0:
        raise ValueError("ridge_alpha must be non-negative.")
    if int(max_windows_per_clip) < 0:
        raise ValueError("max_windows_per_clip must be non-negative.")


def _mean(rows: Sequence[Mapping[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
    return float(mean(values)) if values else 0.0


def _median(rows: Sequence[Mapping[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
    return float(median(values)) if values else 0.0


def _metric(row: object, key: str) -> float | None:
    if not isinstance(row, Mapping) or key not in row:
        return None
    value = float(row[key])
    return value if np.isfinite(value) else None


def _fmt(value: object) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"
