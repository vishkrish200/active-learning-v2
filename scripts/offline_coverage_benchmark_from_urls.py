from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from marginal_value.active_benchmark import CoverageBenchmarkConfig, run_coverage_benchmark, write_coverage_reports
from marginal_value.active_benchmark.coverage_reports import coverage_result_to_json
from marginal_value.active_benchmark.downstream_coverage_smoke import (
    build_downstream_coverage_supervised_report,
    write_downstream_coverage_supervised_reports,
)
from scripts.offline_active_benchmark_from_urls import (
    _build_episodes_from_clips,
    _load_clips_from_urls,
    _parse_csv_list,
    _parse_optional_float,
    _print_progress_event,
    _select_group_balanced_urls,
    _source_group_id,
)


def main() -> None:
    args = _parse_args()
    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    urls = _select_group_balanced_urls(
        Path(args.manifest),
        max_rows=args.max_rows,
        max_groups=args.max_groups,
        clips_per_group=args.clips_per_group,
        sampling_mode=args.sampling_mode,
        selection_seed=args.selection_seed,
    )
    selected_manifest_path = output_dir / "selected_urls.txt"
    selected_manifest_path.write_text("\n".join(urls) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "url_selection_done",
                "n_urls": len(urls),
                "n_source_groups": len({_source_group_id(url) for url in urls}),
                "manifest": str(Path(args.manifest).resolve()),
                "sampling_mode": args.sampling_mode,
                "selection_seed": args.selection_seed,
                "selected_manifest": str(selected_manifest_path.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    representations = _parse_csv_list(args.representations)
    eval_views = _parse_csv_list(args.eval_views)
    primary_eval_views = _parse_csv_list(args.primary_eval_views)
    _validate_loaded_views(representations=representations, eval_views=eval_views, primary_eval_views=primary_eval_views)

    clips = _load_clips_from_urls(
        urls,
        max_samples=args.max_samples,
        sample_rate=args.sample_rate,
        workers=args.download_workers,
        timeout_seconds=args.download_timeout_seconds,
        representations=representations,
        ts2vec_checkpoint=args.ts2vec_checkpoint,
        ts2vec_device=args.ts2vec_device,
        ts2vec_batch_size=args.ts2vec_batch_size,
    )
    episodes = _build_episodes_from_clips(
        clips,
        episode_strategy=args.episode_strategy,
        folds=args.folds,
        candidate_groups_per_episode=args.candidate_groups_per_episode,
        target_groups_per_episode=args.target_groups_per_episode,
        max_support_groups=args.max_support_groups,
        episode_representation=args.episode_representation,
        source_family_count=args.source_family_count,
    )
    config = CoverageBenchmarkConfig(
        budgets=_parse_int_csv(args.budgets),
        policies=_parse_csv_list(args.policies),
        eval_views=eval_views,
        primary_eval_views=primary_eval_views,
        random_seed=args.seed,
        quality_threshold=args.quality_threshold,
        max_stationary_fraction=args.max_stationary_fraction,
        max_abs_value=args.max_abs_value,
        max_artifact_score=args.max_artifact_score,
        ts2vec_view=args.ts2vec_view,
        window_view=args.window_view,
        blend_alpha=args.blend_alpha,
        eval_view_families=_parse_eval_view_families(args.eval_view_families),
        distance_metric=args.distance_metric,
    )
    result = run_coverage_benchmark(clips, episodes, config, progress_callback=_print_progress_event)
    paths = write_coverage_reports(result, output_dir)
    report = coverage_result_to_json(result)
    proof = _coverage_proof_summary(
        report,
        clips=clips,
        output_dir=output_dir,
        report_paths=paths,
        elapsed_seconds=time.time() - started,
    )
    if args.downstream_supervised_label_source != "none":
        downstream_report = build_downstream_coverage_supervised_report(
            result,
            clips,
            downstream_representations=_parse_csv_list(args.downstream_supervised_representations),
            label_source=args.downstream_supervised_label_source,
            label_representation=args.downstream_supervised_label_representation,
            source_family_count=args.downstream_supervised_source_family_count,
            top_policy=args.downstream_supervised_top_policy,
            baseline_policy=args.downstream_supervised_baseline_policy,
        )
        downstream_paths = write_downstream_coverage_supervised_reports(downstream_report, output_dir)
        proof["downstream_coverage_supervised_report"] = {
            "json": str(downstream_paths["json"]),
            "markdown": str(downstream_paths["markdown"]),
            "decision": downstream_report["decision"],
            "summary": downstream_report["summary"],
        }
    proof_path = output_dir / "coverage_proof_summary.json"
    proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "done", **proof}, sort_keys=True), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run blind-target coverage benchmark from public IMU URL manifests.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=300)
    parser.add_argument("--max-groups", type=int, default=100)
    parser.add_argument("--clips-per-group", type=int, default=3)
    parser.add_argument("--sampling-mode", choices=["first", "random"], default="random")
    parser.add_argument("--selection-seed", type=int, default=17)
    parser.add_argument("--max-samples", type=int, default=900)
    parser.add_argument("--sample-rate", type=float, default=30.0)
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--download-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument(
        "--episode-strategy",
        choices=["rotating", "hard", "opportunity", "source_family_shift", "source_family_label_holdout"],
        default="source_family_shift",
    )
    parser.add_argument("--episode-representation", default="window")
    parser.add_argument("--source-family-count", type=int, default=4)
    parser.add_argument("--candidate-groups-per-episode", type=int, default=6)
    parser.add_argument("--target-groups-per-episode", type=int, default=2)
    parser.add_argument("--max-support-groups", type=int, default=64)
    parser.add_argument("--budgets", default="1,2,4")
    parser.add_argument("--quality-threshold", type=float, default=0.85)
    parser.add_argument("--max-stationary-fraction", type=float, default=0.90)
    parser.add_argument("--max-abs-value", type=float, default=60.0)
    parser.add_argument("--max-artifact-score", type=_parse_optional_float, default=0.05)
    parser.add_argument(
        "--policies",
        default=(
            "random_valid_v1,quality_stratified_random_v1,quality_only_v1,"
            "window_support_novelty_v1,window_kcenter_v1"
        ),
    )
    parser.add_argument("--representations", default="window,raw_shape_stats")
    parser.add_argument("--eval-views", default="window,raw_shape_stats")
    parser.add_argument("--primary-eval-views", default="window,raw_shape_stats")
    parser.add_argument("--eval-view-families", default="window:window,raw_shape_stats:raw_shape,ts2vec:ts2vec")
    parser.add_argument("--window-view", default="window")
    parser.add_argument("--ts2vec-view", default="ts2vec")
    parser.add_argument("--ts2vec-checkpoint", default="")
    parser.add_argument("--ts2vec-device", default="cpu")
    parser.add_argument("--ts2vec-batch-size", type=int, default=32)
    parser.add_argument("--blend-alpha", type=float, default=0.5)
    parser.add_argument("--distance-metric", choices=["euclidean", "cosine"], default="euclidean")
    parser.add_argument("--downstream-supervised-label-source", choices=["none", "source_family"], default="none")
    parser.add_argument("--downstream-supervised-label-representation", default="window")
    parser.add_argument("--downstream-supervised-source-family-count", type=int, default=4)
    parser.add_argument("--downstream-supervised-representations", default="window,raw_shape_stats")
    parser.add_argument("--downstream-supervised-top-policy", default="ts2vec_kcenter_v1")
    parser.add_argument("--downstream-supervised-baseline-policy", default="quality_stratified_random_v1")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def _coverage_proof_summary(
    report: dict,
    *,
    clips,
    output_dir: Path,
    report_paths: dict[str, str],
    elapsed_seconds: float,
) -> dict:
    leakage_ok = True
    leakage_checks = []
    for episode in report["episodes"]:
        support = set(episode["support_group_ids"])
        candidate = set(episode["candidate_group_ids"])
        target = set(episode["target_group_ids"])
        row = {
            "episode_id": episode["episode_id"],
            "support_candidate_overlap": sorted(support & candidate),
            "support_target_overlap": sorted(support & target),
            "candidate_target_overlap": sorted(candidate & target),
        }
        leakage_ok = leakage_ok and not row["support_candidate_overlap"] and not row["support_target_overlap"] and not row["candidate_target_overlap"]
        leakage_checks.append(row)

    metric_rows = list(report["metric_rows"])
    hygiene_rows = [row for row in metric_rows if row["eval_view"] == "__hygiene__"]
    primary_coverage_rows = [
        row
        for row in metric_rows
        if row["metric_name"] == "coverage_gain_rel" and bool(row["primary_eval"])
    ]
    same_feature_diagnostic_rows = [
        row
        for row in metric_rows
        if row["metric_name"] == "coverage_gain_rel"
        and bool(row["selector_feature_overlap"])
        and not bool(row["primary_eval"])
    ]
    return {
        "output_dir": str(output_dir.resolve()),
        "json_report": str(Path(report_paths["json"]).resolve()),
        "markdown_report": str(Path(report_paths["markdown"]).resolve()),
        "n_clips": len(clips),
        "n_source_groups": len({clip.source_group_id for clip in clips}),
        "n_episodes": report["n_episodes"],
        "n_round_rows": len(report["rounds"]),
        "n_metric_rows": len(metric_rows),
        "n_selected_rows": len(report["selected_rows"]),
        "policies": report["policies"],
        "budgets": report["budgets"],
        "policy_summary": report["policy_summary"],
        "primary_coverage_metric_row_count": len(primary_coverage_rows),
        "same_feature_diagnostic_metric_row_count": len(same_feature_diagnostic_rows),
        "mean_primary_coverage_gain_rel": _mean_metric(primary_coverage_rows),
        "selected_invalid_rate_max": _max_hygiene_metric(hygiene_rows, "selected_invalid_rate"),
        "selected_out_of_pool_count_total": _sum_hygiene_metric(hygiene_rows, "selected_out_of_pool_count"),
        "selected_target_leak_count_total": _sum_hygiene_metric(hygiene_rows, "selected_target_leak_count"),
        "selected_duplicate_clip_count_total": _sum_hygiene_metric(hygiene_rows, "selected_duplicate_clip_count"),
        "leakage_ok": bool(leakage_ok),
        "leakage_checks": leakage_checks,
        "elapsed_seconds": float(elapsed_seconds),
    }


def _parse_eval_view_families(value: str) -> dict[str, str]:
    if not str(value).strip():
        return {}
    mapping: dict[str, str] = {}
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("Eval view families must be comma-separated view:family entries.")
        view, family = [part.strip() for part in item.split(":", 1)]
        if not view or not family:
            raise ValueError("Eval view families must be comma-separated view:family entries.")
        mapping[view] = family
    return mapping


def _parse_int_csv(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not values:
        raise ValueError("Expected at least one comma-separated integer.")
    if any(item <= 0 for item in values):
        raise ValueError("Integer CSV values must be positive.")
    return values


def _validate_loaded_views(
    *,
    representations: tuple[str, ...],
    eval_views: tuple[str, ...],
    primary_eval_views: tuple[str, ...],
) -> None:
    missing_eval = sorted(set(eval_views) - set(representations))
    if missing_eval:
        raise ValueError(f"Coverage eval views must be loaded as representations: {missing_eval}")
    missing_primary = sorted(set(primary_eval_views) - set(eval_views))
    if missing_primary:
        raise ValueError(f"Coverage primary eval views must be included in eval views: {missing_primary}")


def _mean_metric(rows: list[dict]) -> float:
    values = [float(row["metric_value"]) for row in rows]
    return float(np.mean(values)) if values else 0.0


def _max_hygiene_metric(rows: list[dict], metric_name: str) -> float:
    values = [float(row["metric_value"]) for row in rows if row["metric_name"] == metric_name]
    return max(values) if values else 0.0


def _sum_hygiene_metric(rows: list[dict], metric_name: str) -> float:
    return float(sum(float(row["metric_value"]) for row in rows if row["metric_name"] == metric_name))


if __name__ == "__main__":
    main()
