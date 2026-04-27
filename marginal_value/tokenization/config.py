from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TokenizationLeakageError(RuntimeError):
    """Raised when tokenizer or grammar config would leak holdout data into fitting."""


def load_tokenizer_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_grammar_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_tokenizer_config(config: dict[str, Any]) -> None:
    execution = _required_mapping(config, "execution", "Tokenizer config")
    data = _required_mapping(config, "data", "Tokenizer config")
    artifacts = _required_mapping(config, "artifacts", "Tokenizer config")
    splits = _required_mapping(config, "splits", "Tokenizer config")
    patches = _required_mapping(config, "patches", "Tokenizer config")
    vq = _required_mapping(config, "vq", "Tokenizer config")
    bpe = _required_mapping(config, "bpe", "Tokenizer config")
    diagnostics = config.get("diagnostics", {})
    if diagnostics is not None and not isinstance(diagnostics, dict):
        raise ValueError("Tokenizer config diagnostics section must be an object when provided.")

    _validate_modal_execution(execution, require_gpu=True, label="Tokenizer")
    _validate_remote_data(data, label="Tokenizer")
    _validate_artifacts(artifacts, label="Tokenizer")

    fit_split = splits.get("fit_split")
    transform_splits = splits.get("transform_splits")
    if fit_split != "pretrain":
        raise TokenizationLeakageError("Tokenizer fit_split must be 'pretrain'.")
    if not isinstance(transform_splits, list) or not transform_splits:
        raise ValueError("Tokenizer splits.transform_splits must be a non-empty list.")
    if "pretrain" not in transform_splits:
        raise TokenizationLeakageError("Tokenizer transform_splits must include 'pretrain' for fit artifact auditability.")
    if "val" not in transform_splits:
        raise ValueError("Tokenizer transform_splits must include 'val' so held-out clips can be scored.")
    if len(set(transform_splits)) != len(transform_splits):
        raise ValueError("Tokenizer transform_splits must not contain duplicates.")

    if patches.get("feature_source") != "npz_features":
        raise ValueError("Tokenizer patches.feature_source must be 'npz_features' for the current cache.")
    _positive_float(patches, "patch_len_sec", "Tokenizer patches")
    _positive_float(patches, "patch_stride_sec", "Tokenizer patches")
    _positive_float(patches, "sample_rate", "Tokenizer patches")
    _positive_int(patches, "smoke_fit_samples", "Tokenizer patches")
    _positive_int(patches, "smoke_transform_samples_per_split", "Tokenizer patches")
    _positive_int(patches, "full_fit_samples", "Tokenizer patches")

    _positive_int(vq, "codebook_size", "Tokenizer vq")
    _positive_int(vq, "n_iter", "Tokenizer vq")
    _non_negative_int(vq, "seed", "Tokenizer vq")

    _positive_int(bpe, "num_merges", "Tokenizer bpe")
    _positive_int(bpe, "min_count", "Tokenizer bpe")
    _positive_float(bpe, "base_step_sec", "Tokenizer bpe")
    _positive_float(bpe, "max_primitive_duration_sec", "Tokenizer bpe")
    if float(bpe["max_primitive_duration_sec"]) < float(bpe["base_step_sec"]):
        raise ValueError("Tokenizer bpe.max_primitive_duration_sec must be at least bpe.base_step_sec.")
    if diagnostics:
        _positive_int(diagnostics, "top_k_primitives", "Tokenizer diagnostics")


def validate_grammar_config(config: dict[str, Any]) -> None:
    execution = _required_mapping(config, "execution", "Grammar config")
    tokens = _required_mapping(config, "tokens", "Grammar config")
    artifacts = _required_mapping(config, "artifacts", "Grammar config")
    splits = _required_mapping(config, "splits", "Grammar config")
    grammar = _required_mapping(config, "grammar", "Grammar config")

    _validate_modal_execution(execution, require_gpu=False, label="Grammar")
    _validate_artifacts(artifacts, label="Grammar")
    if not str(tokens.get("input_dir", "")).startswith("/artifacts/tokens/"):
        raise ValueError("Grammar tokens.input_dir must be under /artifacts/tokens/.")

    fit_split = splits.get("fit_split")
    score_split = splits.get("score_split")
    allow_fit_split_scoring = bool(splits.get("allow_fit_split_scoring", False))
    if fit_split != "pretrain":
        raise TokenizationLeakageError("Grammar fit_split must be 'pretrain'.")
    if score_split == fit_split and not allow_fit_split_scoring:
        raise TokenizationLeakageError("Grammar score_split must be different from fit_split.")
    if score_split not in {"pretrain", "val", "new"}:
        raise ValueError("Grammar score_split must be 'pretrain' for support diagnostics, 'val' for pseudo-holdout evaluation, or 'new' for submission scoring.")
    if score_split != fit_split and allow_fit_split_scoring:
        raise ValueError("Grammar splits.allow_fit_split_scoring is only valid when score_split equals fit_split.")

    if grammar.get("model") != "ngram":
        raise ValueError("Grammar config currently supports grammar.model='ngram'.")
    _positive_int(grammar, "order", "Grammar")
    _positive_float(grammar, "smoothing", "Grammar")
    _non_negative_int(grammar, "rare_threshold", "Grammar")


def build_modal_tokenizer_command(config_path: str | Path, *, run_full: bool) -> list[str]:
    command = [
        "modal",
        "run",
        "modal_tokenizer.py",
        "--config-path",
        str(config_path),
    ]
    if run_full:
        command.append("--run-full")
    return command


def build_modal_grammar_command(config_path: str | Path, *, run_full: bool) -> list[str]:
    command = [
        "modal",
        "run",
        "modal_grammar.py",
        "--config-path",
        str(config_path),
    ]
    if run_full:
        command.append("--run-full")
    return command


def _validate_modal_execution(execution: dict[str, Any], *, require_gpu: bool, label: str) -> None:
    if execution.get("provider") != "modal":
        raise ValueError(f"{label} provider must be 'modal'; local fitting is disabled.")
    if require_gpu and execution.get("gpu") != "H100":
        raise ValueError(f"{label} fitting must request Modal gpu='H100'.")
    if not execution.get("artifacts_volume"):
        raise ValueError(f"{label} execution.artifacts_volume must be provided.")
    if require_gpu and not execution.get("data_volume"):
        raise ValueError(f"{label} execution.data_volume must be provided.")
    _positive_int(execution, "timeout_seconds", f"{label} execution")


def _validate_remote_data(data: dict[str, Any], *, label: str) -> None:
    if not str(data.get("root", "")).startswith("/data"):
        raise ValueError(f"{label} data.root must be mounted under /data.")
    if data.get("format") != "npz_features":
        raise ValueError(f"{label} data.format must be 'npz_features'.")
    if not str(data.get("feature_glob", "")).endswith("*.npz"):
        raise ValueError(f"{label} data.feature_glob must point at cached .npz feature files.")
    if not str(data.get("raw_glob", "")).endswith("*.jsonl"):
        raise ValueError(f"{label} data.raw_glob must point at cached .jsonl raw files.")
    for key in ("pretrain_manifest", "val_manifest"):
        if not str(data.get(key, "")).startswith("cache/manifests/"):
            raise ValueError(f"{label} data.{key} must point at a cache manifest.")
    _positive_int(data, "feature_dim", f"{label} data")


def _validate_artifacts(artifacts: dict[str, Any], *, label: str) -> None:
    if not str(artifacts.get("root", "")).startswith("/artifacts"):
        raise ValueError(f"{label} artifacts.root must be mounted under /artifacts.")
    if not str(artifacts.get("output_dir", "")).startswith("/artifacts/tokens/"):
        raise ValueError(f"{label} artifacts.output_dir must be under /artifacts/tokens/.")


def _required_mapping(config: dict[str, Any], key: str, label: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must include a '{key}' object.")
    return value


def _positive_int(section: dict[str, Any], key: str, label: str) -> None:
    value = section.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label}.{key} must be a positive integer.")


def _non_negative_int(section: dict[str, Any], key: str, label: str) -> None:
    value = section.get(key)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{label}.{key} must be a non-negative integer.")


def _positive_float(section: dict[str, Any], key: str, label: str) -> None:
    value = section.get(key)
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{label}.{key} must be positive.")
