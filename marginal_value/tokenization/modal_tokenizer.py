from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from marginal_value.data.split_manifest import SplitSample, select_split
from marginal_value.logging_utils import log_event, log_progress
from marginal_value.models.tokenizer import MotionBPE, PatchVQTokenizer
from marginal_value.tokenization.artifacts import (
    TokenSequence,
    write_bpe_merges_json,
    write_token_sequences_jsonl,
)
from marginal_value.tokenization.config import validate_tokenizer_config
from marginal_value.tokenization.patches import extract_patch_vectors, load_window_features


def run_tokenizer_pipeline(
    config: dict[str, Any],
    manifest: Sequence[SplitSample],
    *,
    output_dir: str | Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    validate_tokenizer_config(config)
    mode = "smoke" if smoke else "full"
    output = Path(output_dir) if output_dir is not None else Path(config["artifacts"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    log_event("tokenizer", "start", mode=mode)

    fit_split = str(config["splits"]["fit_split"])
    transform_splits = [str(split) for split in config["splits"]["transform_splits"]]
    patch_size_windows, stride_windows = _patch_window_shape(config)

    fit_rows = _bounded_rows(
        select_split(manifest, fit_split),
        limit=int(config["patches"]["smoke_fit_samples"] if smoke else config["patches"]["full_fit_samples"]),
    )
    log_event(
        "tokenizer",
        "fit_rows_selected",
        split=fit_split,
        n_rows=len(fit_rows),
        patch_size_windows=patch_size_windows,
        stride_windows=stride_windows,
    )
    fit_patches, fit_patch_metadata = _fit_patch_vectors_for_rows(
        fit_rows,
        patch_size_windows=patch_size_windows,
        stride_windows=stride_windows,
    )
    fit_patch_counts = _count_metadata_splits(fit_patch_metadata)
    if set(fit_patch_counts) != {fit_split}:
        raise RuntimeError(f"Tokenizer VQ fit received non-fit splits: {fit_patch_counts}")
    log_event("tokenizer", "vq_fit_start", n_patches=len(fit_patches), split=fit_split)
    vq_progress_every = max(1, int(config["vq"]["n_iter"]) // 5)
    vq = PatchVQTokenizer(
        codebook_size=int(config["vq"]["codebook_size"]),
        n_iter=int(config["vq"]["n_iter"]),
        seed=int(config["vq"]["seed"]),
    ).fit(
        fit_patches,
        progress_callback=lambda index, total: log_progress(
            "tokenizer",
            "vq_fit_iteration_progress",
            index=index,
            total=total,
            every=vq_progress_every,
            n_patches=len(fit_patches),
        ),
    )
    log_event("tokenizer", "vq_fit_done", codebook_size=0 if vq.codebook is None else len(vq.codebook))

    transform_rows_by_split = {
        split: _bounded_rows(
            select_split(manifest, split),
            limit=int(config["patches"]["smoke_transform_samples_per_split"]) if smoke else None,
        )
        for split in transform_splits
    }
    base_sequences = _encode_base_sequences(
        transform_rows_by_split,
        vq=vq,
        patch_size_windows=patch_size_windows,
        stride_windows=stride_windows,
    )
    pretrain_sequences = [sequence.base_token_ids for sequence in base_sequences if sequence.split == fit_split]
    bpe_fit_counts = Counter(sequence.split for sequence in base_sequences if sequence.split == fit_split)
    if set(bpe_fit_counts) != {fit_split}:
        raise RuntimeError(f"Tokenizer BPE fit received non-fit splits: {dict(bpe_fit_counts)}")

    log_event("tokenizer", "bpe_fit_start", n_sequences=len(pretrain_sequences), split=fit_split)
    bpe = MotionBPE(
        num_merges=int(config["bpe"]["num_merges"]),
        base_step_sec=float(config["bpe"]["base_step_sec"]),
        max_primitive_duration_sec=float(config["bpe"]["max_primitive_duration_sec"]),
        min_count=int(config["bpe"]["min_count"]),
    ).fit(pretrain_sequences)
    log_event("tokenizer", "bpe_fit_done", n_primitives=len(bpe.primitives))

    token_sequences = [_with_primitives(sequence, bpe) for sequence in base_sequences]
    sequence_counts = Counter(sequence.split for sequence in token_sequences)

    suffix = mode
    codebook_path = output / f"vq_codebook_{suffix}.npz"
    bpe_merges_path = output / f"bpe_merges_{suffix}.json"
    sequence_path = output / f"token_sequences_{suffix}.jsonl"
    report_path = output / f"tokenizer_report_{suffix}.json"

    if vq.codebook is None:
        raise RuntimeError("VQ codebook was not fit.")
    np.savez(codebook_path, codebook=vq.codebook.astype(np.float32))
    write_bpe_merges_json(bpe_merges_path, bpe.primitives)
    write_token_sequences_jsonl(sequence_path, token_sequences)

    report = {
        "mode": mode,
        "fit_split": fit_split,
        "transform_splits": transform_splits,
        "fit_patch_split_counts": dict(fit_patch_counts),
        "bpe_fit_sequence_split_counts": dict(bpe_fit_counts),
        "sequence_split_counts": dict(sequence_counts),
        "n_fit_rows": len(fit_rows),
        "n_sequences": len(token_sequences),
        "n_primitives": len(bpe.primitives),
        "diagnostics": _diagnostics(vq_codebook=vq.codebook, sequences=token_sequences),
        "leakage_audit": {
            "vq_fit_splits": sorted(fit_patch_counts),
            "bpe_fit_splits": sorted(bpe_fit_counts),
            "transformed_splits": sorted(sequence_counts),
        },
        "artifacts": {
            "codebook": str(codebook_path),
            "bpe_merges": str(bpe_merges_path),
            "token_sequences": str(sequence_path),
            "report": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    result = {
        "mode": mode,
        "fit_split": fit_split,
        "transform_splits": transform_splits,
        "fit_patch_split_counts": dict(fit_patch_counts),
        "bpe_fit_sequence_split_counts": dict(bpe_fit_counts),
        "sequence_split_counts": dict(sequence_counts),
        "codebook_path": str(codebook_path),
        "bpe_merges_path": str(bpe_merges_path),
        "sequence_path": str(sequence_path),
        "report_path": str(report_path),
    }
    log_event("tokenizer", "done", **result)
    return result


def _patch_window_shape(config: dict[str, Any]) -> tuple[int, int]:
    patch_len = float(config["patches"]["patch_len_sec"])
    patch_stride = float(config["patches"]["patch_stride_sec"])
    if patch_stride <= 0.0:
        raise ValueError("patch_stride_sec must be positive.")
    patch_size_windows = max(1, int(round(patch_len / patch_stride)))
    stride_windows = 1
    return patch_size_windows, stride_windows


def _bounded_rows(rows: Sequence[SplitSample], *, limit: int | None) -> list[SplitSample]:
    row_list = list(rows)
    if limit is None:
        return row_list
    if limit <= 0:
        raise ValueError("limit must be positive when provided.")
    return row_list[:limit]


def _fit_patch_vectors_for_rows(
    rows: Sequence[SplitSample],
    *,
    patch_size_windows: int,
    stride_windows: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    row_list = list(rows)
    if not row_list:
        raise ValueError("No rows were provided for patch extraction.")

    patch_batches: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    total = len(row_list)
    progress_every = max(1, total // 10)
    n_patches_total = 0
    log_event("tokenizer", "fit_patch_extract_start", n_rows=total)
    for index, row in enumerate(row_list, start=1):
        patches = extract_patch_vectors(
            load_window_features(row.feature_path),
            patch_size_windows=patch_size_windows,
            stride_windows=stride_windows,
        )
        patch_batches.append(patches)
        n_patches_total += len(patches)
        for patch_index in range(len(patches)):
            metadata.append(
                {
                    "sample_id": row.sample_id,
                    "split": row.split,
                    "patch_index": patch_index,
                }
            )
        log_progress(
            "tokenizer",
            "fit_patch_extract_progress",
            index=index,
            total=total,
            every=progress_every,
            split=row.split,
            n_patches_total=n_patches_total,
        )

    log_event("tokenizer", "fit_patch_extract_done", n_rows=total, n_patches=n_patches_total)
    return np.vstack(patch_batches), metadata


def _encode_base_sequences(
    rows_by_split: dict[str, Sequence[SplitSample]],
    *,
    vq: PatchVQTokenizer,
    patch_size_windows: int,
    stride_windows: int,
) -> list[TokenSequence]:
    sequences: list[TokenSequence] = []
    for split, rows in rows_by_split.items():
        row_list = list(rows)
        total = len(row_list)
        progress_every = max(1, total // 10) if total else 1
        log_event("tokenizer", "transform_split_start", split=split, n_rows=total)
        for index, row in enumerate(row_list, start=1):
            window_features = load_window_features(row.feature_path)
            patches = extract_patch_vectors(
                window_features,
                patch_size_windows=patch_size_windows,
                stride_windows=stride_windows,
            )
            base_ids = vq.encode(patches).astype(int).tolist()
            sequences.append(
                TokenSequence(
                    sample_id=row.sample_id,
                    split=split,
                    base_token_ids=base_ids,
                    primitive_token_ids=[str(token_id) for token_id in base_ids],
                    primitive_durations_sec=[float(stride_windows) for _ in base_ids],
                    metadata={
                        "feature_path": str(row.feature_path),
                        "n_patches": len(base_ids),
                    },
                )
            )
            log_progress(
                "tokenizer",
                "transform_split_progress",
                index=index,
                total=total,
                every=progress_every,
                split=split,
                n_patches=len(base_ids),
            )
        log_event("tokenizer", "transform_split_done", split=split, n_rows=total)
    return sequences


def _with_primitives(sequence: TokenSequence, bpe: MotionBPE) -> TokenSequence:
    primitives = bpe.encode(sequence.base_token_ids)
    return TokenSequence(
        sample_id=sequence.sample_id,
        split=sequence.split,
        base_token_ids=sequence.base_token_ids,
        primitive_token_ids=[primitive.token_id for primitive in primitives],
        primitive_durations_sec=[primitive.duration_sec for primitive in primitives],
        quality_score=sequence.quality_score,
        metadata=sequence.metadata,
    )


def _count_metadata_splits(metadata: Iterable[dict[str, object]]) -> Counter[str]:
    return Counter(str(item["split"]) for item in metadata)


def _diagnostics(*, vq_codebook: np.ndarray, sequences: Sequence[TokenSequence]) -> dict[str, float]:
    token_counts = Counter(token_id for sequence in sequences for token_id in sequence.base_token_ids)
    used_codes = len(token_counts)
    codebook_size = len(vq_codebook)
    total_tokens = sum(token_counts.values())
    if total_tokens:
        probs = np.asarray(list(token_counts.values()), dtype=float) / float(total_tokens)
        entropy = float(-np.sum(probs * np.log(probs + 1.0e-12)))
        perplexity = float(np.exp(entropy))
    else:
        entropy = 0.0
        perplexity = 0.0
    primitive_durations = [
        duration
        for sequence in sequences
        for duration in sequence.primitive_durations_sec
    ]
    return {
        "code_usage_entropy": entropy,
        "codebook_perplexity": perplexity,
        "dead_code_fraction": float((codebook_size - used_codes) / max(codebook_size, 1)),
        "mean_primitive_duration_sec": float(np.mean(primitive_durations)) if primitive_durations else 0.0,
        "max_primitive_duration_sec": float(np.max(primitive_durations)) if primitive_durations else 0.0,
    }
