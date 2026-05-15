from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np

from marginal_value.active_benchmark import (
    BenchmarkClip,
    OfflineBenchmarkConfig,
    build_difficulty_targeted_episodes,
    build_opportunity_targeted_episodes,
    build_source_family_label_holdout_episodes,
    build_source_family_shift_episodes,
    build_source_blocked_episodes,
    run_offline_active_benchmark,
    write_benchmark_reports,
)
from marginal_value.active_benchmark.reports import result_to_json
from marginal_value.preprocessing.quality import compute_quality_features
from marginal_value.ranking.baseline_ranker import raw_shape_stats_embedding, temporal_order_embedding, window_mean_std_embedding


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
    if "window" not in representations:
        raise ValueError("The URL benchmark runner requires the 'window' representation for V1 acquisition policies.")
    missing_primary = set(primary_representations) - set(representations)
    if missing_primary:
        raise ValueError(f"Primary representations must be loaded: {sorted(missing_primary)}")
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
    paths = write_benchmark_reports(result, output_dir)
    proof = _proof_summary(
        result_to_json(result),
        clips=clips,
        output_dir=output_dir,
        report_paths=paths,
        elapsed_seconds=time.time() - started,
    )
    proof_path = output_dir / "proof_summary.json"
    proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "done", **proof}, sort_keys=True), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline active-learning benchmark from public IMU URL manifests.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=10_000)
    parser.add_argument("--max-groups", type=int, default=400)
    parser.add_argument("--clips-per-group", type=int, default=25)
    parser.add_argument("--sampling-mode", choices=["first", "random"], default="random")
    parser.add_argument("--selection-seed", type=int, default=17)
    parser.add_argument("--max-samples", type=int, default=900)
    parser.add_argument("--sample-rate", type=float, default=30.0)
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--download-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--episode-strategy", choices=["rotating", "hard", "opportunity", "source_family_shift", "source_family_label_holdout"], default="rotating")
    parser.add_argument("--episode-representation", default="window")
    parser.add_argument("--source-family-count", type=int, default=4)
    parser.add_argument("--candidate-groups-per-episode", type=int, default=3)
    parser.add_argument("--target-groups-per-episode", type=int, default=3)
    parser.add_argument("--target-candidate-groups-per-episode", type=int, default=None)
    parser.add_argument("--target-families-per-episode", type=int, default=1)
    parser.add_argument("--max-support-groups", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--quality-threshold", type=float, default=0.85)
    parser.add_argument("--max-stationary-fraction", type=float, default=0.90)
    parser.add_argument("--max-abs-value", type=float, default=60.0)
    parser.add_argument(
        "--policies",
        default="random_valid,quality_only,old_novelty_window,kcenter_quality_gated_window,oracle_greedy_eval_only",
    )
    parser.add_argument("--representations", default="window,raw_shape_stats")
    parser.add_argument("--primary-representations", default="window,raw_shape_stats")
    parser.add_argument("--ts2vec-checkpoint", default="")
    parser.add_argument("--ts2vec-device", default="cpu")
    parser.add_argument("--ts2vec-batch-size", type=int, default=32)
    parser.add_argument("--blend-left-representation", default="ts2vec")
    parser.add_argument("--blend-right-representation", default="window")
    parser.add_argument("--blend-alpha", type=float, default=0.5)
    parser.add_argument("--max-artifact-score", type=_parse_optional_float, default=0.05)
    parser.add_argument("--oracle-candidate-cap", type=int, default=None)
    parser.add_argument("--oracle-exact-combination-limit", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def _build_episodes_from_clips(
    clips: list[BenchmarkClip],
    *,
    episode_strategy: str,
    folds: int,
    candidate_groups_per_episode: int,
    target_groups_per_episode: int,
    target_candidate_groups_per_episode: int | None = None,
    target_families_per_episode: int = 1,
    max_support_groups: int | None = None,
    episode_representation: str = "window",
    source_family_count: int = 4,
):
    if episode_strategy == "rotating":
        return build_source_blocked_episodes(
            clips,
            n_folds=folds,
            candidate_groups_per_episode=candidate_groups_per_episode,
            target_groups_per_episode=target_groups_per_episode,
            max_support_groups=max_support_groups,
        )
    if episode_strategy == "hard":
        return build_difficulty_targeted_episodes(
            clips,
            n_folds=folds,
            candidate_groups_per_episode=candidate_groups_per_episode,
            target_groups_per_episode=target_groups_per_episode,
            max_support_groups=max_support_groups,
            representation=episode_representation,
        )
    if episode_strategy == "opportunity":
        return build_opportunity_targeted_episodes(
            clips,
            n_folds=folds,
            candidate_groups_per_episode=candidate_groups_per_episode,
            target_groups_per_episode=target_groups_per_episode,
            max_support_groups=max_support_groups,
            representation=episode_representation,
        )
    if episode_strategy == "source_family_shift":
        return build_source_family_shift_episodes(
            clips,
            n_folds=folds,
            candidate_groups_per_episode=candidate_groups_per_episode,
            target_groups_per_episode=target_groups_per_episode,
            max_support_groups=max_support_groups,
            representation=episode_representation,
            source_family_count=source_family_count,
        )
    if episode_strategy == "source_family_label_holdout":
        return build_source_family_label_holdout_episodes(
            clips,
            n_folds=folds,
            candidate_groups_per_episode=candidate_groups_per_episode,
            target_groups_per_episode=target_groups_per_episode,
            target_candidate_groups_per_episode=target_candidate_groups_per_episode,
            target_families_per_episode=target_families_per_episode,
            max_support_groups=max_support_groups,
            representation=episode_representation,
            source_family_count=source_family_count,
        )
    raise ValueError(f"Unsupported episode strategy: {episode_strategy}")


def _select_group_balanced_urls(
    manifest_path: Path,
    *,
    max_rows: int,
    max_groups: int,
    clips_per_group: int,
    sampling_mode: str = "random",
    selection_seed: int = 17,
) -> list[str]:
    if max_rows <= 0:
        raise ValueError("max_rows must be positive.")
    if max_groups <= 0:
        raise ValueError("max_groups must be positive.")
    if clips_per_group <= 0:
        raise ValueError("clips_per_group must be positive.")

    grouped: dict[str, list[str]] = {}
    for url in _iter_manifest_urls(manifest_path):
        group = _source_group_id(url)
        grouped.setdefault(group, []).append(url)

    group_ids = list(grouped)
    if sampling_mode == "first":
        selected_groups = group_ids[:max_groups]
        urls = [url for group in selected_groups for url in grouped[group][:clips_per_group]]
        return urls[:max_rows]
    if sampling_mode != "random":
        raise ValueError(f"Unsupported sampling mode: {sampling_mode}")

    rng = np.random.default_rng(int(selection_seed))
    selected_groups = list(rng.permutation(group_ids))[:max_groups]
    urls: list[str] = []
    for group in selected_groups:
        group_urls = grouped[str(group)]
        if len(group_urls) <= clips_per_group:
            selected_urls = list(group_urls)
        else:
            selected_urls = [group_urls[int(index)] for index in rng.choice(len(group_urls), size=clips_per_group, replace=False)]
        urls.extend(selected_urls)
    return urls[:max_rows]


def _iter_manifest_urls(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            url = line.strip()
            if url:
                yield url


def _print_progress_event(event: dict[str, object]) -> None:
    print(json.dumps(event, sort_keys=True), flush=True)


def _load_clips_from_urls(
    urls: list[str],
    *,
    max_samples: int,
    sample_rate: float,
    workers: int,
    timeout_seconds: float,
    representations: tuple[str, ...],
    ts2vec_checkpoint: str = "",
    ts2vec_device: str = "cpu",
    ts2vec_batch_size: int = 32,
) -> list[BenchmarkClip]:
    clips, _samples_by_id = _load_clips_and_samples_from_urls(
        urls,
        max_samples=max_samples,
        sample_rate=sample_rate,
        workers=workers,
        timeout_seconds=timeout_seconds,
        representations=representations,
        ts2vec_checkpoint=ts2vec_checkpoint,
        ts2vec_device=ts2vec_device,
        ts2vec_batch_size=ts2vec_batch_size,
    )
    return clips


def _load_clips_and_samples_from_urls(
    urls: list[str],
    *,
    max_samples: int,
    sample_rate: float,
    workers: int,
    timeout_seconds: float,
    representations: tuple[str, ...],
    ts2vec_checkpoint: str = "",
    ts2vec_device: str = "cpu",
    ts2vec_batch_size: int = 32,
) -> tuple[list[BenchmarkClip], dict[str, np.ndarray]]:
    if "ts2vec" in representations and not str(ts2vec_checkpoint).strip():
        raise ValueError("--ts2vec-checkpoint is required when representations include ts2vec.")
    base_representations = tuple(representation for representation in representations if representation != "ts2vec")
    clips: list[BenchmarkClip] = []
    samples_by_id: dict[str, np.ndarray] = {}
    failed: list[dict[str, str]] = []
    progress_every = max(1, len(urls) // 20)
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {
            pool.submit(
                _clip_from_url,
                url,
                max_samples=max_samples,
                sample_rate=sample_rate,
                timeout_seconds=timeout_seconds,
                representations=base_representations,
            ): url
            for url in urls
        }
        for index, future in enumerate(as_completed(futures), start=1):
            url = futures[future]
            try:
                clip, samples = future.result()
                clips.append(clip)
                samples_by_id[clip.sample_id] = samples
            except Exception as exc:  # noqa: BLE001 - keep GCP batch robust and report failures.
                failed.append({"url": url, "error": repr(exc)})
            if index == len(urls) or index % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "event": "clip_load_progress",
                            "loaded": len(clips),
                            "failed": len(failed),
                            "seen": index,
                            "total": len(urls),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    if failed:
        print(json.dumps({"event": "clip_load_failures", "failures": failed[:20], "n_failed": len(failed)}, sort_keys=True), flush=True)
    if len(clips) < 10:
        raise RuntimeError(f"Only loaded {len(clips)} clips from {len(urls)} URLs.")
    clips = sorted(clips, key=lambda clip: clip.sample_id)
    if "ts2vec" in representations:
        clips = _attach_ts2vec_embeddings(
            clips,
            samples_by_id=samples_by_id,
            checkpoint_path=str(ts2vec_checkpoint),
            device=str(ts2vec_device),
            batch_size=int(ts2vec_batch_size),
        )
        print(
            json.dumps(
                {
                    "event": "ts2vec_embedding_done",
                    "n_clips": len(clips),
                    "checkpoint": str(ts2vec_checkpoint),
                    "device": str(ts2vec_device),
                    "batch_size": int(ts2vec_batch_size),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return clips, samples_by_id


def _attach_ts2vec_embeddings(
    clips: list[BenchmarkClip],
    *,
    samples_by_id: dict[str, np.ndarray],
    checkpoint_path: str,
    device: str,
    batch_size: int,
    encoder_factory=None,
) -> list[BenchmarkClip]:
    if encoder_factory is None:
        from marginal_value.models.ts2vec_inference import TS2VecInference

        encoder_factory = TS2VecInference
    missing = [clip.sample_id for clip in clips if clip.sample_id not in samples_by_id]
    if missing:
        raise ValueError(f"Cannot attach TS2Vec embeddings; missing raw samples for {len(missing)} clips: {missing[:5]}")
    if int(batch_size) <= 0:
        raise ValueError("ts2vec batch size must be positive.")
    encoder = encoder_factory(str(checkpoint_path), device=str(device))
    sample_batches = [np.asarray(samples_by_id[clip.sample_id], dtype=np.float32) for clip in clips]
    matrix = np.asarray(encoder.encode_batch(sample_batches, batch_size=int(batch_size)), dtype=np.float32)
    if matrix.shape[0] != len(clips):
        raise ValueError(f"TS2Vec encoder returned {matrix.shape[0]} embeddings for {len(clips)} clips.")
    updated: list[BenchmarkClip] = []
    for clip, embedding in zip(clips, matrix, strict=True):
        embeddings = dict(clip.embeddings)
        embeddings["ts2vec"] = np.asarray(embedding, dtype=np.float32)
        updated.append(
            BenchmarkClip(
                sample_id=clip.sample_id,
                source_group_id=clip.source_group_id,
                embeddings=embeddings,
                quality_score=clip.quality_score,
                stationary_fraction=clip.stationary_fraction,
                max_abs_value=clip.max_abs_value,
                artifact_score=clip.artifact_score,
            )
        )
    return updated


def _clip_from_url(
    url: str,
    *,
    max_samples: int,
    sample_rate: float,
    timeout_seconds: float,
    representations: tuple[str, ...],
) -> tuple[BenchmarkClip, np.ndarray]:
    samples, timestamps = _load_url_jsonl_imu(url, max_samples=max_samples, timeout_seconds=timeout_seconds)
    quality = compute_quality_features(samples, timestamps=timestamps, sample_rate=sample_rate)
    embeddings = _embeddings_for_representations(samples, sample_rate=sample_rate, representations=representations)
    return (
        BenchmarkClip(
            sample_id=_sample_id(url),
            source_group_id=_source_group_id(url),
            embeddings=embeddings,
            quality_score=float(quality["quality_score"]),
            stationary_fraction=float(quality["stationary_fraction"]),
            max_abs_value=float(quality["max_abs_value"]),
            artifact_score=float(
                max(
                    quality["missing_rate"],
                    quality["flatline_fraction"],
                    quality["saturation_fraction"],
                    quality["extreme_value_fraction"],
                    quality["spike_rate"],
                )
            ),
        ),
        samples,
    )


def _embeddings_for_representations(
    samples: np.ndarray,
    *,
    sample_rate: float,
    representations: tuple[str, ...],
) -> dict[str, np.ndarray]:
    embeddings: dict[str, np.ndarray] = {}
    for representation in representations:
        if representation == "window":
            embeddings[representation] = window_mean_std_embedding(samples)
        elif representation == "temporal_order":
            embeddings[representation] = temporal_order_embedding(samples)
        elif representation == "raw_shape_stats":
            embeddings[representation] = raw_shape_stats_embedding(samples, sample_rate=sample_rate)
        else:
            raise ValueError(f"Unsupported URL benchmark representation: {representation}")
    return embeddings


def _load_url_jsonl_imu(
    url: str,
    *,
    max_samples: int,
    timeout_seconds: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    samples: list[list[float]] = []
    timestamps: list[float] = []
    saw_timestamp = False
    with urlopen(url, timeout=timeout_seconds) as response:
        for raw_line in response:
            if len(samples) >= max_samples:
                break
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            record = json.loads(line)
            acc = record.get("acc")
            gyro = record.get("gyro")
            if not isinstance(acc, list) or not isinstance(gyro, list) or len(acc) < 3 or len(gyro) < 3:
                raise ValueError(f"Unexpected IMU JSONL record shape for {url}")
            samples.append([float(value) for value in [*acc[:3], *gyro[:3]]])
            if "t_us" in record:
                timestamps.append(float(record["t_us"]) / 1_000_000.0)
                saw_timestamp = True
            else:
                timestamps.append(np.nan)
    if not samples:
        raise ValueError(f"No IMU samples read from {url}")
    return np.asarray(samples, dtype=float), np.asarray(timestamps, dtype=float) if saw_timestamp else None


def _source_group_id(url: str) -> str:
    parts = [part for part in urlparse(url).path.split("/") if part]
    for part in parts:
        if part.startswith("worker"):
            return part
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"


def _sample_id(url: str) -> str:
    parts = [part for part in urlparse(url).path.split("/") if part]
    clip = Path(parts[-1]).stem if parts else "clip"
    return f"{_source_group_id(url)}_{clip}"


def _proof_summary(
    report: dict,
    *,
    clips: list[BenchmarkClip],
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

    support_deltas = [
        len(row["support_ids_after"]) - len(row["support_ids_before"])
        for row in report["rounds"]
    ]
    candidate_deltas = [
        len(row["candidate_ids_after"]) - len(row["candidate_ids_before"])
        for row in report["rounds"]
    ]
    oracle_fractions = [float(row["oracle_fraction"]) for row in report["rounds"]]
    difficulty_audit = report.get("difficulty_audit", [])
    near_zero_fractions = [
        float(row.get("near_zero_oracle_round_fraction", 1.0))
        for row in difficulty_audit
        if isinstance(row, dict)
    ]
    oracle_exact_flags = [
        bool(flag)
        for row in difficulty_audit
        if isinstance(row, dict)
        for flag in row.get("oracle_fraction_exact_by_round", [])
    ]
    return {
        "output_dir": str(output_dir.resolve()),
        "json_report": str(Path(report_paths["json"]).resolve()),
        "markdown_report": str(Path(report_paths["markdown"]).resolve()),
        "n_clips": len(clips),
        "n_source_groups": len({clip.source_group_id for clip in clips}),
        "n_episodes": report["n_episodes"],
        "n_round_rows": len(report["rounds"]),
        "policies": report["policies"],
        "policy_summary": report["policy_summary"],
        "difficulty_audit": difficulty_audit,
        "mean_near_zero_oracle_round_fraction": float(np.mean(near_zero_fractions)) if near_zero_fractions else 1.0,
        "oracle_fraction_exact_all_rounds": bool(all(oracle_exact_flags)) if oracle_exact_flags else False,
        "oracle_fraction_exact_round_fraction": float(np.mean(oracle_exact_flags)) if oracle_exact_flags else 0.0,
        "leakage_ok": bool(leakage_ok),
        "leakage_checks": leakage_checks,
        "support_delta_values": sorted(set(support_deltas)),
        "candidate_delta_values": sorted(set(candidate_deltas)),
        "oracle_fraction_min": min(oracle_fractions) if oracle_fractions else 0.0,
        "oracle_fraction_max": max(oracle_fractions) if oracle_fractions else 0.0,
        "elapsed_seconds": float(elapsed_seconds),
    }


def _parse_csv_list(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not items:
        raise ValueError("Expected at least one comma-separated value.")
    return items


def _parse_optional_float(value: str | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip().lower()
    if text in {"", "none", "null", "disabled", "off"}:
        return None
    return float(text)


if __name__ == "__main__":
    main()
