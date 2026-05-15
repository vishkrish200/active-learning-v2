from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from marginal_value.indexing.cosine_search import cosine_knn
from marginal_value.logging_utils import log_event, log_progress
from marginal_value.models.imu2clip_inference import IMU2CLIPInference
from marginal_value.preprocessing.quality import load_modal_jsonl_imu
from marginal_value.ranking.baseline_ranker import raw_shape_stats_embedding, window_mean_std_embedding


def run_embedding_audit(
    manifest_path: str,
    checkpoint_path: str,
    n_clips: int = 500,
    output_path: str = "docs/imu2clip_embedding_audit.md",
    preloaded_clips_path: str | None = None,
) -> dict[str, object]:
    if preloaded_clips_path:
        records, clips, clip_ids = _load_preloaded_audit_clips(preloaded_clips_path)
        records = records[: int(n_clips)]
        clips = clips[: int(n_clips)]
        clip_ids = clip_ids[: int(n_clips)]
        log_event(
            "imu2clip_embedding_audit",
            "preloaded_from_cache",
            path=preloaded_clips_path,
            n_clips=len(clips),
        )
    else:
        records = _read_manifest(manifest_path)[: int(n_clips)]
        clips = []
        clip_ids = []
    if not records:
        raise ValueError("Embedding audit requires at least one manifest row.")
    log_event(
        "imu2clip_embedding_audit",
        "start",
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        n_clips=len(records),
        output_path=output_path,
        preloaded_clips_path=preloaded_clips_path or "",
    )
    progress_every = max(1, min(50, len(records) // 10 or 1))
    if not clips:
        for index, row in enumerate(records, start=1):
            samples, _timestamps = load_modal_jsonl_imu(_resolve_raw_path(row["path"]))
            clips.append(samples)
            clip_ids.append(row["clip_id"])
            log_progress(
                "imu2clip_embedding_audit",
                "clip_load_progress",
                index=index,
                total=len(records),
                every=progress_every,
            )

    device = "cuda" if _cuda_available() else "cpu"
    inference = IMU2CLIPInference(checkpoint_path=checkpoint_path, device=device)
    log_event("imu2clip_embedding_audit", "encoder_ready", device=device)

    imu2clip_rows: list[np.ndarray] = []
    for index, clip in enumerate(clips, start=1):
        imu2clip_rows.append(inference.encode_clip(clip))
        log_progress(
            "imu2clip_embedding_audit",
            "imu2clip_encode_progress",
            index=index,
            total=len(clips),
            every=progress_every,
        )
    imu2clip_embeddings = np.vstack(imu2clip_rows).astype("float32")

    baseline_rows: list[np.ndarray] = []
    for index, (row, clip) in enumerate(zip(records, clips), start=1):
        baseline_rows.append(_baseline_embedding(row, clip))
        log_progress(
            "imu2clip_embedding_audit",
            "baseline_encode_progress",
            index=index,
            total=len(clips),
            every=progress_every,
        )
    baseline_embeddings = np.vstack(baseline_rows).astype("float32")

    collapse = _collapse_metrics(imu2clip_embeddings)
    log_event(
        "imu2clip_embedding_audit",
        "collapse_done",
        effective_rank=collapse["effective_rank"],
        mean_pairwise_cosine=collapse["mean_pairwise_cosine"],
    )
    nn_rows = _nearest_neighbor_rows(clip_ids, imu2clip_embeddings)
    imu_novelty = _knn_novelty(imu2clip_embeddings)
    baseline_novelty = _knn_novelty(baseline_embeddings)
    rank_correlation = _rank_correlation(imu_novelty, baseline_novelty)
    log_event("imu2clip_embedding_audit", "novelty_comparison_done", rank_correlation=rank_correlation)
    sensitivity = _window_sensitivity(inference, clips[: min(50, len(clips))])

    report = {
        "collapse": collapse,
        **collapse,
        "rank_correlation_vs_window_mean_std_pool": float(rank_correlation),
        "window_sensitivity": sensitivity,
        "nearest_neighbors": nn_rows,
        "passed_effective_rank_gate": bool(float(collapse["effective_rank"]) > 20.0),
    }
    _write_report(Path(output_path), report, checkpoint_path=checkpoint_path, manifest_path=manifest_path, n_clips=len(clips))
    log_event(
        "imu2clip_embedding_audit",
        "done",
        output_path=output_path,
        effective_rank=collapse["effective_rank"],
        mean_pairwise_cosine=collapse["mean_pairwise_cosine"],
        passed_effective_rank_gate=report["passed_effective_rank_gate"],
    )
    return report


def preload_embedding_audit_clips(
    manifest_path: str,
    n_clips: int = 500,
    output_path: str = "/artifacts/cache/imu2clip_style/audit_clips.npz",
) -> dict[str, object]:
    records = _read_manifest(manifest_path)[: int(n_clips)]
    if not records:
        raise ValueError("Audit preload requires at least one manifest row.")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    clip_ids: list[str] = []
    raw_paths: list[str] = []
    feature_paths: list[str] = []
    progress_every = max(1, min(50, len(records) // 10 or 1))
    log_event(
        "imu2clip_embedding_audit",
        "preload_start",
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
            "imu2clip_embedding_audit",
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
    }
    log_event("imu2clip_embedding_audit", "preload_done", **result)
    return result


def _read_manifest(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                record = json.loads(line)
                raw_path = str(record.get("raw_path") or record.get("path") or record.get("url") or "")
                clip_id = str(record.get("sample_id") or record.get("clip_id") or Path(raw_path).stem or index)
                feature_path = str(record.get("feature_path") or "")
            else:
                raw_path = line
                clip_id = Path(line).stem or str(index)
                feature_path = ""
            rows.append({"clip_id": clip_id, "path": raw_path, "feature_path": feature_path})
    return rows


def _load_preloaded_audit_clips(path: str | Path) -> tuple[list[dict[str, str]], list[np.ndarray], list[str]]:
    records: list[dict[str, str]] = []
    clips: list[np.ndarray] = []
    with np.load(path, allow_pickle=False) as data:
        clip_ids = [str(value) for value in data["clip_ids"]]
        raw_paths = [str(value) for value in data["raw_paths"]]
        feature_paths = [str(value) for value in data["feature_paths"]]
        for index, clip_id in enumerate(clip_ids):
            records.append({"clip_id": clip_id, "path": raw_paths[index], "feature_path": feature_paths[index]})
            clips.append(np.asarray(data[f"clip_{index:06d}"], dtype=np.float32))
    return records, clips, clip_ids


def _baseline_embedding(row: Mapping[str, str], clip: np.ndarray) -> np.ndarray:
    feature_path = str(row.get("feature_path", ""))
    if feature_path and Path(feature_path).exists():
        with np.load(feature_path) as data:
            return window_mean_std_embedding(np.asarray(data["window_features"], dtype="float32"))
    return raw_shape_stats_embedding(clip, sample_rate=30.0)


def _resolve_raw_path(path_or_url: str) -> Path:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://") or path_or_url.startswith("gs://"):
        from marginal_value.data.split_manifest import hash_manifest_url

        return Path("/data/cache/raw") / f"{hash_manifest_url(path_or_url)}.jsonl"
    return Path(path_or_url)


def _collapse_metrics(embeddings: np.ndarray) -> dict[str, object]:
    vectors = _normalize_rows(embeddings)
    centered = vectors - np.mean(vectors, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    power = singular * singular
    probs = power / max(float(np.sum(power)), 1.0e-12)
    entropy = -float(np.sum(probs * np.log(probs + 1.0e-12)))
    similarities = vectors @ vectors.T
    mask = ~np.eye(len(vectors), dtype=bool)
    pairwise = similarities[mask] if len(vectors) > 1 else np.asarray([0.0])
    return {
        "effective_rank": float(np.exp(entropy)),
        "mean_pairwise_cosine": float(np.mean(pairwise)),
        "embedding_dim_std_mean": float(np.mean(np.std(vectors, axis=0))),
        "embedding_dim_std_min": float(np.min(np.std(vectors, axis=0))),
        "embedding_dim_std_max": float(np.max(np.std(vectors, axis=0))),
    }


def _nearest_neighbor_rows(clip_ids: Sequence[str], embeddings: np.ndarray) -> list[dict[str, object]]:
    if len(embeddings) < 2:
        return []
    vectors = _normalize_rows(embeddings)
    count = min(10, len(vectors))
    rows: list[dict[str, object]] = []
    distances, indices = cosine_knn(vectors, vectors[:count], k=min(6, len(vectors)), backend="numpy")
    for row_index in range(count):
        neighbor_pairs = [
            {"clip_id": clip_ids[int(idx)], "distance": float(dist)}
            for dist, idx in zip(distances[row_index], indices[row_index])
            if int(idx) != row_index
        ][:5]
        rows.append({"query_clip_id": clip_ids[row_index], "neighbors": neighbor_pairs})
    return rows


def _knn_novelty(embeddings: np.ndarray, *, k: int = 10) -> np.ndarray:
    vectors = _normalize_rows(embeddings)
    if len(vectors) <= 1:
        return np.zeros(len(vectors), dtype=float)
    distances, _indices = cosine_knn(vectors, vectors, k=min(k + 1, len(vectors)), backend="numpy")
    return np.mean(distances[:, 1:], axis=1)


def _window_sensitivity(inference: IMU2CLIPInference, clips: Sequence[np.ndarray]) -> dict[str, float]:
    if not clips:
        return {"cosine_2s_vs_5s_mean": 0.0, "cosine_2s_vs_10s_mean": 0.0}
    log_event("imu2clip_embedding_audit", "window_sensitivity_start", n_clips=len(clips))
    cos_5: list[float] = []
    cos_10: list[float] = []
    progress_every = max(1, min(10, len(clips) // 5 or 1))
    for index, clip in enumerate(clips, start=1):
        emb_2 = inference.encode_clip_multiscale(clip, window_sizes_s=[2.0])
        emb_5 = inference.encode_clip_multiscale(clip, window_sizes_s=[5.0])
        emb_10 = inference.encode_clip_multiscale(clip, window_sizes_s=[10.0])
        cos_5.append(float(emb_2 @ emb_5))
        cos_10.append(float(emb_2 @ emb_10))
        log_progress(
            "imu2clip_embedding_audit",
            "window_sensitivity_progress",
            index=index,
            total=len(clips),
            every=progress_every,
        )
    result = {
        "cosine_2s_vs_5s_mean": float(np.mean(cos_5)),
        "cosine_2s_vs_10s_mean": float(np.mean(cos_10)),
    }
    log_event("imu2clip_embedding_audit", "window_sensitivity_done", **result)
    return result


def _write_report(path: Path, report: Mapping[str, object], *, checkpoint_path: str, manifest_path: str, n_clips: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# IMU2CLIP Embedding Audit",
        "",
        f"- manifest: `{manifest_path}`",
        f"- checkpoint: `{checkpoint_path}`",
        f"- clips embedded: `{n_clips}`",
        f"- effective_rank: `{float(report['effective_rank']):.3f}`",
        f"- mean_pairwise_cosine: `{float(report['mean_pairwise_cosine']):.3f}`",
        f"- rank_correlation_vs_window_mean_std_pool: `{float(report['rank_correlation_vs_window_mean_std_pool']):.3f}`",
        f"- passed_effective_rank_gate: `{bool(report['passed_effective_rank_gate'])}`",
        "",
        "## Window Sensitivity",
        "",
        "```json",
        json.dumps(report["window_sensitivity"], indent=2, sort_keys=True),
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


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.size < 2:
        return 0.0
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    left_centered = left_rank - np.mean(left_rank)
    right_centered = right_rank - np.mean(right_rank)
    denom = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    return 0.0 if denom < 1.0e-12 else float(left_centered @ right_centered / denom)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1.0e-12)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False
