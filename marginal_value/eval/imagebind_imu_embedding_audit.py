from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from marginal_value.eval.imu2clip_embedding_audit import (
    _baseline_embedding,
    _collapse_metrics,
    _knn_novelty,
    _load_preloaded_audit_clips,
    _nearest_neighbor_rows,
    _rank_correlation,
    _read_manifest,
    _resolve_raw_path,
)
from marginal_value.logging_utils import log_event, log_progress
from marginal_value.models.imagebind_imu_inference import ImageBindIMUInference
from marginal_value.preprocessing.quality import load_modal_jsonl_imu


def run_imagebind_imu_embedding_audit(
    manifest_path: str,
    n_clips: int = 500,
    output_path: str = "docs/imagebind_imu_embedding_audit.md",
    preloaded_clips_path: str | None = None,
    checkpoint_path: str | None = None,
    checkpoint_dir: str | None = None,
    batch_size: int = 32,
    run_id: str = "",
) -> dict[str, object]:
    if preloaded_clips_path:
        records, clips, clip_ids = _load_preloaded_audit_clips(preloaded_clips_path)
        records = records[: int(n_clips)]
        clips = clips[: int(n_clips)]
        clip_ids = clip_ids[: int(n_clips)]
        log_event(
            "imagebind_imu_embedding_audit",
            "preloaded_from_cache",
            run_id=run_id,
            path=preloaded_clips_path,
            n_clips=len(clips),
        )
    else:
        records = _read_manifest(manifest_path)[: int(n_clips)]
        clips = []
        clip_ids = []
    if not records:
        raise ValueError("ImageBind IMU embedding audit requires at least one manifest row.")
    log_event(
        "imagebind_imu_embedding_audit",
        "start",
        run_id=run_id,
        manifest_path=manifest_path,
        n_clips=len(records),
        output_path=output_path,
        preloaded_clips_path=preloaded_clips_path or "",
        checkpoint_path=checkpoint_path or "",
        checkpoint_dir=checkpoint_dir or "",
        batch_size=int(batch_size),
    )
    progress_every = max(1, min(50, len(records) // 10 or 1))
    if not clips:
        for index, row in enumerate(records, start=1):
            samples, _timestamps = load_modal_jsonl_imu(_resolve_raw_path(row["path"]))
            clips.append(samples)
            clip_ids.append(row["clip_id"])
            log_progress(
                "imagebind_imu_embedding_audit",
                "clip_load_progress",
                index=index,
                total=len(records),
                every=progress_every,
            )

    device = "cuda" if _cuda_available() else "cpu"
    inference = ImageBindIMUInference(
        device=device,
        batch_size=int(batch_size),
        checkpoint_path=checkpoint_path,
        checkpoint_dir=checkpoint_dir,
    )
    log_event("imagebind_imu_embedding_audit", "encoder_ready", run_id=run_id, device=device)

    imagebind_rows: list[np.ndarray] = []
    for index, clip in enumerate(clips, start=1):
        imagebind_rows.append(inference.encode_clip(clip))
        log_progress(
            "imagebind_imu_embedding_audit",
            "imagebind_encode_progress",
            index=index,
            total=len(clips),
            every=progress_every,
        )
    imagebind_embeddings = np.vstack(imagebind_rows).astype("float32")

    baseline_rows: list[np.ndarray] = []
    for index, (row, clip) in enumerate(zip(records, clips), start=1):
        baseline_rows.append(_baseline_embedding(row, clip))
        log_progress(
            "imagebind_imu_embedding_audit",
            "baseline_encode_progress",
            index=index,
            total=len(clips),
            every=progress_every,
        )
    baseline_embeddings = np.vstack(baseline_rows).astype("float32")

    collapse = _collapse_metrics(imagebind_embeddings)
    nn_rows = _nearest_neighbor_rows(clip_ids, imagebind_embeddings)
    imagebind_novelty = _knn_novelty(imagebind_embeddings)
    baseline_novelty = _knn_novelty(baseline_embeddings)
    rank_correlation = _rank_correlation(imagebind_novelty, baseline_novelty)
    passed_effective_rank_gate = bool(float(collapse["effective_rank"]) > 20.0)
    passed_cosine_gate = bool(float(collapse["mean_pairwise_cosine"]) < 0.5)

    report = {
        "collapse": collapse,
        **collapse,
        "rank_correlation_vs_window_mean_std_pool": float(rank_correlation),
        "nearest_neighbors": nn_rows,
        "passed_effective_rank_gate": passed_effective_rank_gate,
        "passed_cosine_gate": passed_cosine_gate,
        "passed_embedding_audit": bool(passed_effective_rank_gate and passed_cosine_gate),
        "window_contract": {
            "input_shape": "(B, 6, 2000)",
            "window_len_s": 10.0,
            "expected_hz": 200,
            "pool": "max",
        },
    }
    _write_report(Path(output_path), report, manifest_path=manifest_path, n_clips=len(clips), run_id=run_id)
    log_event(
        "imagebind_imu_embedding_audit",
        "done",
        run_id=run_id,
        output_path=output_path,
        effective_rank=collapse["effective_rank"],
        mean_pairwise_cosine=collapse["mean_pairwise_cosine"],
        rank_correlation_vs_window_mean_std_pool=rank_correlation,
        passed_embedding_audit=report["passed_embedding_audit"],
    )
    return report


def preload_imagebind_imu_audit_clips(
    manifest_path: str,
    n_clips: int = 500,
    output_path: str = "/artifacts/cache/imagebind_imu/audit_clips_500.npz",
    run_id: str = "",
) -> dict[str, object]:
    records = _read_manifest(manifest_path)[: int(n_clips)]
    if not records:
        raise ValueError("ImageBind IMU audit preload requires at least one manifest row.")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    clip_ids: list[str] = []
    raw_paths: list[str] = []
    feature_paths: list[str] = []
    progress_every = max(1, min(50, len(records) // 10 or 1))
    log_event(
        "imagebind_imu_embedding_audit",
        "preload_start",
        run_id=run_id,
        manifest_path=manifest_path,
        n_clips=len(records),
        output_path=str(path),
    )
    for index, row in enumerate(records, start=1):
        samples, _timestamps = load_modal_jsonl_imu(_resolve_raw_path(row["path"]))
        arrays[f"clip_{index - 1:06d}"] = np.asarray(samples, dtype=np.float32)
        clip_ids.append(row["clip_id"])
        raw_paths.append(row["path"])
        feature_paths.append(str(row.get("feature_path", "")))
        log_progress(
            "imagebind_imu_embedding_audit",
            "preload_progress",
            index=index,
            total=len(records),
            every=progress_every,
        )
    np.savez_compressed(
        path,
        clip_ids=np.asarray(clip_ids),
        raw_paths=np.asarray(raw_paths),
        feature_paths=np.asarray(feature_paths),
        **arrays,
    )
    result = {
        "preloaded_clips_path": str(path),
        "n_clips": len(records),
        "memory_mb": sum(float(array.nbytes) for array in arrays.values()) / 1.0e6,
        "run_id": run_id,
    }
    log_event("imagebind_imu_embedding_audit", "preload_done", **result)
    return result


def _write_report(path: Path, report: Mapping[str, object], *, manifest_path: str, n_clips: int, run_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ImageBind IMU Embedding Audit",
        "",
        f"- run_id: `{run_id}`",
        f"- manifest: `{manifest_path}`",
        f"- clips embedded: `{n_clips}`",
        f"- effective_rank: `{float(report['effective_rank']):.3f}`",
        f"- mean_pairwise_cosine: `{float(report['mean_pairwise_cosine']):.3f}`",
        f"- rank_correlation_vs_window_mean_std_pool: `{float(report['rank_correlation_vs_window_mean_std_pool']):.3f}`",
        f"- passed_effective_rank_gate: `{bool(report['passed_effective_rank_gate'])}`",
        f"- passed_cosine_gate: `{bool(report['passed_cosine_gate'])}`",
        f"- passed_embedding_audit: `{bool(report['passed_embedding_audit'])}`",
        "",
        "## Window Contract",
        "",
        "ImageBind IMU is fixed at `(B, 6, 2000)`, i.e. 10 seconds at 200 Hz. Long clips are resampled, "
        "normalized per clip/per channel, windowed with 50% overlap, max pooled, and L2 normalized.",
        "",
        "```json",
        json.dumps(report["window_contract"], indent=2, sort_keys=True),
        "```",
        "",
        "## Nearest Neighbors",
        "",
        "```json",
        json.dumps(report["nearest_neighbors"], indent=2, sort_keys=True),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False
