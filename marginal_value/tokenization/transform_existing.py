from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from marginal_value.data.split_manifest import SplitSample, build_split_manifest, select_split, split_counts
from marginal_value.logging_utils import log_event, log_progress
from marginal_value.models.tokenizer import MotionBPE, PatchVQTokenizer
from marginal_value.tokenization.artifacts import (
    TokenSequence,
    read_token_sequences_jsonl,
    write_bpe_merges_json,
    write_token_sequences_jsonl,
)
from marginal_value.tokenization.patches import extract_patch_vectors, load_window_features


def run_existing_tokenizer_transform(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    mode = "smoke" if smoke else "full"
    data_root = Path(config["data"]["root"])
    tokens = config["tokens"]
    input_dir = Path(tokens["input_dir"])
    output_dir = Path(tokens["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    source_mode = str(tokens.get("source_mode", "full"))

    log_event("tokenizer_transform", "start", mode=mode, source_mode=source_mode)
    manifest = build_split_manifest(
        data_root,
        pretrain_manifest=config["data"]["pretrain_manifest"],
        val_manifest=config["data"]["val_manifest"],
        extra_manifests={"new": config["data"]["new_manifest"]},
        feature_glob=config["data"].get("feature_glob", "cache/features/*.npz"),
        raw_glob=config["data"].get("raw_glob", "cache/raw/*.jsonl"),
    )
    log_event("tokenizer_transform", "manifest_ready", split_counts=split_counts(manifest))
    transform_split = str(config["splits"].get("transform_split", "new"))
    transform_rows = select_split(manifest, transform_split)
    if smoke:
        transform_rows = transform_rows[: int(config["patches"].get("smoke_transform_samples", 64))]
    if not transform_rows:
        raise ValueError(f"No cached rows found for transform_split '{transform_split}'.")

    source_sequences = read_token_sequences_jsonl(input_dir / f"token_sequences_{source_mode}.jsonl")
    fit_split = str(config["splits"].get("fit_split", "pretrain"))
    fit_sequences = [sequence for sequence in source_sequences if sequence.split == fit_split]
    if smoke:
        fit_sequences = fit_sequences[: int(config["patches"].get("smoke_fit_sequences", 64))]
    if not fit_sequences:
        raise ValueError(f"No source token sequences found for fit_split '{fit_split}'.")

    vq = _load_vq(input_dir / f"vq_codebook_{source_mode}.npz")
    bpe = _load_bpe(
        input_dir / f"bpe_merges_{source_mode}.json",
        base_step_sec=float(config["bpe"].get("base_step_sec", 0.25)),
        max_primitive_duration_sec=float(config["bpe"].get("max_primitive_duration_sec", 12.0)),
    )
    patch_size_windows, stride_windows = _patch_window_shape(config)
    new_sequences = _encode_rows(
        transform_rows,
        vq=vq,
        bpe=bpe,
        split=transform_split,
        patch_size_windows=patch_size_windows,
        stride_windows=stride_windows,
    )

    output_sequences = [*fit_sequences, *new_sequences]
    sequence_path = output_dir / f"token_sequences_{mode}.jsonl"
    report_path = output_dir / f"tokenizer_transform_report_{mode}.json"
    write_token_sequences_jsonl(sequence_path, output_sequences)
    write_bpe_merges_json(output_dir / f"bpe_merges_{mode}.json", bpe.primitives)
    np.savez(output_dir / f"vq_codebook_{mode}.npz", codebook=vq.codebook.astype(np.float32))

    report = {
        "mode": mode,
        "source_mode": source_mode,
        "fit_split": fit_split,
        "transform_split": transform_split,
        "n_fit_sequences": len(fit_sequences),
        "n_transformed_sequences": len(new_sequences),
        "sequence_path": str(sequence_path),
        "leakage_audit": {
            "fit_sequences": fit_split,
            "transformed_split": transform_split,
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    log_event("tokenizer_transform", "done", **report)
    return report


def _patch_window_shape(config: dict[str, Any]) -> tuple[int, int]:
    patch_len = float(config["patches"]["patch_len_sec"])
    patch_stride = float(config["patches"]["patch_stride_sec"])
    if patch_stride <= 0.0:
        raise ValueError("patch_stride_sec must be positive.")
    return max(1, int(round(patch_len / patch_stride))), 1


def _load_vq(path: Path) -> PatchVQTokenizer:
    with np.load(path) as data:
        codebook = np.asarray(data["codebook"], dtype=np.float32)
    vq = PatchVQTokenizer(codebook_size=len(codebook))
    vq.codebook = codebook
    return vq


def _load_bpe(path: Path, *, base_step_sec: float, max_primitive_duration_sec: float) -> MotionBPE:
    rows = json.loads(path.read_text(encoding="utf-8"))
    bpe = MotionBPE(base_step_sec=base_step_sec, max_primitive_duration_sec=max_primitive_duration_sec)
    bpe.primitives = [tuple(int(code) for code in row["codes"]) for row in rows]
    return bpe


def _encode_rows(
    rows: Sequence[SplitSample],
    *,
    vq: PatchVQTokenizer,
    bpe: MotionBPE,
    split: str,
    patch_size_windows: int,
    stride_windows: int,
) -> list[TokenSequence]:
    sequences: list[TokenSequence] = []
    total = len(rows)
    progress_every = max(1, total // 10) if total else 1
    for index, row in enumerate(rows, start=1):
        window_features = load_window_features(row.feature_path)
        patches = extract_patch_vectors(
            window_features,
            patch_size_windows=patch_size_windows,
            stride_windows=stride_windows,
        )
        base_ids = vq.encode(patches).astype(int).tolist()
        primitives = bpe.encode(base_ids)
        sequences.append(
            TokenSequence(
                sample_id=row.sample_id,
                split=split,
                base_token_ids=base_ids,
                primitive_token_ids=[primitive.token_id for primitive in primitives],
                primitive_durations_sec=[primitive.duration_sec for primitive in primitives],
                metadata={"feature_path": str(row.feature_path), "n_patches": len(base_ids), "source": "existing_vq_bpe"},
            )
        )
        log_progress(
            "tokenizer_transform",
            "transform_progress",
            index=index,
            total=total,
            every=progress_every,
            split=split,
        )
    return sequences
