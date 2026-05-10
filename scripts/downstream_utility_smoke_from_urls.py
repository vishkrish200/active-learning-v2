from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from marginal_value.active_benchmark import (
    OfflineBenchmarkConfig,
    run_offline_active_benchmark,
    write_benchmark_reports,
)
from marginal_value.active_benchmark.downstream_supervised_smoke import (
    build_downstream_supervised_report,
    write_downstream_supervised_reports,
)
from marginal_value.active_benchmark.downstream_smoke import build_downstream_utility_report, write_downstream_utility_reports
from marginal_value.active_benchmark.reports import result_to_json
from scripts.offline_active_benchmark_from_urls import (
    _build_episodes_from_clips,
    _load_clips_from_urls,
    _parse_csv_list,
    _parse_optional_float,
    _print_progress_event,
    _proof_summary,
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
    primary_representations = _parse_csv_list(args.primary_representations)
    downstream_representations = _parse_csv_list(args.downstream_representations)
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
        target_candidate_groups_per_episode=args.target_candidate_groups_per_episode,
        target_families_per_episode=args.target_families_per_episode,
        max_support_groups=args.max_support_groups,
        episode_representation=args.episode_representation,
        source_family_count=args.source_family_count,
    )
    config = OfflineBenchmarkConfig(
        batch_size=args.batch_size,
        rounds=args.rounds,
        policies=_parse_csv_list(args.policies),
        representations=representations,
        primary_representations=primary_representations,
        random_seed=args.seed,
        quality_threshold=args.quality_threshold,
        max_stationary_fraction=args.max_stationary_fraction,
        max_abs_value=args.max_abs_value,
        oracle_candidate_cap=args.oracle_candidate_cap,
        oracle_exact_combination_limit=args.oracle_exact_combination_limit,
        blend_left_representation=args.blend_left_representation,
        blend_right_representation=args.blend_right_representation,
        blend_alpha=args.blend_alpha,
        max_artifact_score=args.max_artifact_score,
    )
    result = run_offline_active_benchmark(clips, episodes, config, progress_callback=_print_progress_event)
    benchmark_paths = write_benchmark_reports(result, output_dir)
    downstream_report = build_downstream_utility_report(
        result,
        clips,
        downstream_representations=downstream_representations,
        max_components=args.downstream_max_components,
        baseline_policy=args.downstream_baseline_policy,
        random_policy=args.downstream_random_policy,
    )
    downstream_paths = write_downstream_utility_reports(downstream_report, output_dir)
    proof = _proof_summary(
        result_to_json(result),
        clips=clips,
        output_dir=output_dir,
        report_paths=benchmark_paths,
        elapsed_seconds=time.time() - started,
    )
    proof["downstream_report"] = {
        "json": str(downstream_paths["json"]),
        "markdown": str(downstream_paths["markdown"]),
        "decision": downstream_report["decision"],
        "summary": downstream_report["summary"],
    }
    if args.supervised_downstream_label_source != "none":
        supervised_representations = _parse_csv_list(args.supervised_downstream_representations)
        supervised_report = build_downstream_supervised_report(
            result,
            clips,
            downstream_representations=supervised_representations,
            label_source=args.supervised_downstream_label_source,
            label_representation=args.supervised_downstream_label_representation,
            source_family_count=args.supervised_downstream_source_family_count,
            baseline_policy=args.supervised_downstream_baseline_policy,
            random_policy=args.supervised_downstream_random_policy,
        )
        supervised_paths = write_downstream_supervised_reports(supervised_report, output_dir)
        proof["downstream_supervised_report"] = {
            "json": str(supervised_paths["json"]),
            "markdown": str(supervised_paths["markdown"]),
            "decision": supervised_report["decision"],
            "summary": supervised_report["summary"],
        }
    proof_path = output_dir / "proof_summary.json"
    proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "done", **proof}, sort_keys=True), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tiny downstream utility smoke from public IMU URL manifests.")
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
    parser.add_argument("--episode-strategy", choices=["rotating", "hard", "opportunity", "source_family_shift", "source_family_label_holdout"], default="source_family_shift")
    parser.add_argument("--episode-representation", default="window")
    parser.add_argument("--source-family-count", type=int, default=4)
    parser.add_argument("--candidate-groups-per-episode", type=int, default=6)
    parser.add_argument("--target-groups-per-episode", type=int, default=2)
    parser.add_argument("--target-candidate-groups-per-episode", type=int, default=None)
    parser.add_argument("--target-families-per-episode", type=int, default=1)
    parser.add_argument("--max-support-groups", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--quality-threshold", type=float, default=0.85)
    parser.add_argument("--max-stationary-fraction", type=float, default=0.90)
    parser.add_argument("--max-abs-value", type=float, default=60.0)
    parser.add_argument(
        "--policies",
        default=(
            "random_valid,quality_only,old_novelty_window,old_novelty_ts2vec,"
            "kcenter_quality_gated_ts2vec,submitted_full_replay,submitted_minus_ts2vec,"
            "submitted_minus_window,ts2vec_novelty_same_gates_no_kcenter"
        ),
    )
    parser.add_argument("--representations", default="ts2vec,window,raw_shape_stats")
    parser.add_argument("--primary-representations", default="window,raw_shape_stats")
    parser.add_argument("--ts2vec-checkpoint", required=True)
    parser.add_argument("--ts2vec-device", default="cpu")
    parser.add_argument("--ts2vec-batch-size", type=int, default=32)
    parser.add_argument("--blend-left-representation", default="ts2vec")
    parser.add_argument("--blend-right-representation", default="window")
    parser.add_argument("--blend-alpha", type=float, default=0.5)
    parser.add_argument("--max-artifact-score", type=_parse_optional_float, default=0.05)
    parser.add_argument("--oracle-candidate-cap", type=int, default=None)
    parser.add_argument("--oracle-exact-combination-limit", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--downstream-representations", default="window,raw_shape_stats")
    parser.add_argument("--downstream-max-components", type=int, default=4)
    parser.add_argument("--downstream-baseline-policy", default="old_novelty_ts2vec")
    parser.add_argument("--downstream-random-policy", default="random_valid")
    parser.add_argument("--supervised-downstream-label-source", choices=["none", "source_family"], default="none")
    parser.add_argument("--supervised-downstream-label-representation", default="window")
    parser.add_argument("--supervised-downstream-source-family-count", type=int, default=4)
    parser.add_argument("--supervised-downstream-representations", default="window,raw_shape_stats")
    parser.add_argument("--supervised-downstream-baseline-policy", default="old_novelty_ts2vec")
    parser.add_argument("--supervised-downstream-random-policy", default="random_valid")
    return parser.parse_args()


if __name__ == "__main__":
    main()
