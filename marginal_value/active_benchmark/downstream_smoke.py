from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from marginal_value.active_benchmark.representations import clip_lookup, stack_embeddings
from marginal_value.active_benchmark.schema import BenchmarkClip, BenchmarkResult


def build_downstream_utility_rows(
    result: BenchmarkResult,
    clips: Sequence[BenchmarkClip],
    *,
    downstream_representations: Sequence[str],
    max_components: int = 8,
) -> list[dict[str, object]]:
    clips_by_id = clip_lookup(clips)
    episodes = {episode.episode_id: episode for episode in result.episodes}
    rows: list[dict[str, object]] = []
    for round_result in result.rounds:
        episode = episodes[str(round_result.episode_id)]
        for representation in downstream_representations:
            baseline_error = linear_reconstruction_error(
                clips_by_id,
                train_ids=round_result.support_ids_before,
                target_ids=episode.target_ids,
                representation=str(representation),
                max_components=max_components,
            )
            after_error = linear_reconstruction_error(
                clips_by_id,
                train_ids=round_result.support_ids_after,
                target_ids=episode.target_ids,
                representation=str(representation),
                max_components=max_components,
            )
            absolute_gain = float(baseline_error - after_error)
            relative_gain = float(absolute_gain / baseline_error) if baseline_error > 1.0e-12 else 0.0
            rows.append(
                {
                    "episode_id": str(round_result.episode_id),
                    "fold_id": int(round_result.fold_id),
                    "policy_name": str(round_result.policy_name),
                    "round_index": int(round_result.round_index),
                    "representation": str(representation),
                    "selected_ids": list(round_result.selected_ids),
                    "selected_source_group_ids": list(round_result.selected_source_group_ids),
                    "support_count_before": int(round_result.support_count_before),
                    "support_count_after": int(round_result.support_count_after),
                    "target_count": int(len(episode.target_ids)),
                    "max_components": int(max_components),
                    "baseline_reconstruction_error": float(baseline_error),
                    "after_reconstruction_error": float(after_error),
                    "absolute_reconstruction_gain": float(absolute_gain),
                    "relative_reconstruction_gain": float(relative_gain),
                }
            )
    return rows


def linear_reconstruction_error(
    clips_by_id: dict[str, BenchmarkClip],
    *,
    train_ids: Sequence[str],
    target_ids: Sequence[str],
    representation: str,
    max_components: int = 8,
) -> float:
    train = stack_embeddings(clips_by_id, train_ids, representation=representation)
    target = stack_embeddings(clips_by_id, target_ids, representation=representation)
    if train.size == 0 or target.size == 0:
        return 0.0
    mean = np.mean(train, axis=0, keepdims=True)
    centered = train - mean
    n_components = min(max(0, int(max_components)), max(0, train.shape[0] - 1), train.shape[1])
    if n_components > 0 and np.any(np.abs(centered) > 1.0e-12):
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:n_components]
        target_centered = target - mean
        reconstructed = mean + target_centered @ components.T @ components
    else:
        reconstructed = np.repeat(mean, repeats=target.shape[0], axis=0)
    return float(np.mean((target - reconstructed) ** 2))


def summarize_downstream_utility(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    rows_list = [dict(row) for row in rows]
    final_rows = _final_rows(rows_list)
    return {
        "row_count": int(len(rows_list)),
        "final_row_count": int(len(final_rows)),
        "policy_final_means": _policy_means(final_rows),
        "policy_round_means": _policy_round_means(rows_list),
        "final_episode_wins": _final_episode_wins(final_rows),
    }


def build_downstream_utility_report(
    result: BenchmarkResult,
    clips: Sequence[BenchmarkClip],
    *,
    downstream_representations: Sequence[str],
    max_components: int = 8,
    baseline_policy: str = "old_novelty_ts2vec",
    random_policy: str = "random_valid",
) -> dict[str, object]:
    rows = build_downstream_utility_rows(
        result,
        clips,
        downstream_representations=downstream_representations,
        max_components=max_components,
    )
    summary = summarize_downstream_utility(rows)
    return {
        "input": {
            "episode_count": int(len(result.episodes)),
            "round_count": int(len(result.rounds)),
            "policies": list(result.policies),
            "downstream_representations": [str(rep) for rep in downstream_representations],
            "max_components": int(max_components),
            "baseline_policy": str(baseline_policy),
            "random_policy": str(random_policy),
        },
        "summary": summary,
        "decision": _decision(summary, baseline_policy=str(baseline_policy), random_policy=str(random_policy)),
        "rows": rows,
    }


def write_downstream_utility_reports(report: dict[str, object], output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "downstream_utility_smoke_report.json"
    markdown_path = root / "downstream_utility_smoke_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def _final_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    max_round_by_key: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row["episode_id"]), str(row["policy_name"]))
        max_round_by_key[key] = max(max_round_by_key.get(key, -1), int(row["round_index"]))
    return [
        dict(row)
        for row in rows
        if int(row["round_index"]) == max_round_by_key[(str(row["episode_id"]), str(row["policy_name"]))]
    ]


def _policy_means(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, float | int]]:
    by_policy: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_policy[str(row["policy_name"])].append(dict(row))
    return {policy: _mean_row_values(policy_rows) for policy, policy_rows in sorted(by_policy.items())}


def _policy_round_means(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, dict[str, float | int]]]:
    by_policy_round: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_policy_round[str(row["policy_name"])][str(int(row["round_index"]))].append(dict(row))
    return {
        policy: {round_index: _mean_row_values(round_rows) for round_index, round_rows in sorted(round_map.items())}
        for policy, round_map in sorted(by_policy_round.items())
    }


def _mean_row_values(rows: Sequence[dict[str, object]]) -> dict[str, float | int]:
    return {
        "row_count": int(len(rows)),
        "mean_baseline_reconstruction_error": _mean(rows, "baseline_reconstruction_error"),
        "mean_after_reconstruction_error": _mean(rows, "after_reconstruction_error"),
        "mean_absolute_reconstruction_gain": _mean(rows, "absolute_reconstruction_gain"),
        "mean_relative_reconstruction_gain": _mean(rows, "relative_reconstruction_gain"),
    }


def _mean(rows: Sequence[dict[str, object]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in rows]
    return float(np.mean(values)) if values else 0.0


def _final_episode_wins(rows: Sequence[dict[str, object]]) -> dict[str, int]:
    by_episode_rep: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_episode_rep[(str(row["episode_id"]), str(row["representation"]))].append(dict(row))
    wins: dict[str, int] = defaultdict(int)
    for group in by_episode_rep.values():
        best = max(float(row["relative_reconstruction_gain"]) for row in group)
        for row in group:
            if abs(float(row["relative_reconstruction_gain"]) - best) <= 1.0e-12:
                wins[str(row["policy_name"])] += 1
    return dict(sorted(wins.items()))


def _decision(summary: dict[str, object], *, baseline_policy: str, random_policy: str) -> dict[str, object]:
    policy_means = summary.get("policy_final_means", {})
    if not isinstance(policy_means, dict):
        policy_means = {}
    baseline = policy_means.get(baseline_policy, {})
    random = policy_means.get(random_policy, {})
    baseline_gain = _metric_value(baseline, "mean_relative_reconstruction_gain")
    random_gain = _metric_value(random, "mean_relative_reconstruction_gain")
    baseline_minus_random = baseline_gain - random_gain if baseline_gain is not None and random_gain is not None else None
    read = "linear reconstruction smoke ran, but it is not full downstream proof"
    if baseline_policy in policy_means and random_policy in policy_means:
        if baseline_minus_random is not None and baseline_minus_random > 0.0:
            read = f"{baseline_policy} beat {random_policy} on this tiny linear reconstruction smoke, but this is not full downstream proof"
        else:
            read = f"{baseline_policy} did not beat {random_policy} on this tiny linear reconstruction smoke, so do not scale downstream training"
    return {
        "downstream_training": "hold",
        "read": read,
        "baseline_policy": baseline_policy,
        "random_policy": random_policy,
        "baseline_mean_relative_reconstruction_gain": baseline_gain,
        "random_mean_relative_reconstruction_gain": random_gain,
        "baseline_minus_random": baseline_minus_random,
        "next_steps": [
            "Treat this as a cheap model-update smoke, not a semantic downstream benchmark.",
            "Only scale if the carried acquisition baseline beats random on this smoke and the offline acquisition report remains healthy.",
            "Do not launch full TS2Vec retraining from this artifact alone.",
        ],
    }


def _metric_value(row: object, key: str) -> float | None:
    if not isinstance(row, dict):
        return None
    if key not in row:
        return None
    return float(row[key])


def _markdown_report(report: dict[str, object]) -> str:
    summary = report.get("summary", {})
    decision = report.get("decision", {})
    policy_means = summary.get("policy_final_means", {}) if isinstance(summary, dict) else {}
    lines = [
        "# Downstream Utility Smoke",
        "",
        "This is a tiny linear reconstruction model-update smoke, not full downstream proof.",
        "",
        "## Decision",
        "",
        f"- downstream training: `{decision.get('downstream_training')}`",
        f"- read: {decision.get('read')}",
        f"- baseline minus random: `{_fmt(decision.get('baseline_minus_random'))}`",
        "",
        "## Policy Means",
        "",
    ]
    if isinstance(policy_means, dict):
        for policy, values in sorted(policy_means.items()):
            if not isinstance(values, dict):
                continue
            lines.append(
                f"- `{policy}`: relative reconstruction gain `{_fmt(values.get('mean_relative_reconstruction_gain'))}`, "
                f"after error `{_fmt(values.get('mean_after_reconstruction_error'))}`"
            )
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "- This trains a small linear reconstruction surrogate on frozen embeddings. It checks whether acquisition changes a model's held-out target utility; it does not prove a task-label downstream model will improve.",
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (float, int)):
        return f"{float(value):.6f}"
    return str(value)
