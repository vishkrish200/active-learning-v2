from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LocalTrainingDisabledError(RuntimeError):
    """Raised when a caller tries to train outside Modal."""


def load_training_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return config


def validate_training_dispatch(config: dict[str, Any]) -> None:
    execution = _required_mapping(config, "execution")
    data = _required_mapping(config, "data")
    training = _required_mapping(config, "training")
    encoder = config.get("encoder", {})
    if encoder is not None and not isinstance(encoder, dict):
        raise ValueError("Training config encoder section must be an object when provided.")

    provider = execution.get("provider")
    if provider != "modal":
        raise ValueError("Training provider must be 'modal'; local training is disabled.")

    gpu = execution.get("gpu")
    if gpu != "H100":
        raise ValueError("Training must request Modal gpu='H100' for the fast GPU path.")

    data_root = str(data.get("root", ""))
    checkpoint_dir = str(training.get("checkpoint_dir", ""))
    if not data_root.startswith("/data"):
        raise ValueError("Remote training data root must live under the Modal /data volume mount.")
    if not checkpoint_dir.startswith("/artifacts"):
        raise ValueError("Remote checkpoints must be written under the Modal /artifacts volume mount.")
    if data.get("format") != "npz_features":
        raise ValueError("Training config must use data.format='npz_features' for the current Modal cache.")
    if not str(data.get("feature_glob", "")).endswith("*.npz"):
        raise ValueError("Training config must point data.feature_glob at cached .npz feature files.")
    if data.get("train_split") == data.get("holdout_split"):
        raise ValueError("train_split and holdout_split must be different to prevent leakage.")
    if data.get("train_split") != "pretrain":
        raise ValueError("Training is currently restricted to train_split='pretrain'.")
    if data.get("holdout_split") != "val":
        raise ValueError("Evaluation holdout must be holdout_split='val'.")

    _positive_int(training, "batch_size")
    _positive_int(training, "max_steps")
    _positive_int(training, "smoke_steps")
    _positive_int(training, "validation_steps")
    _positive_int(data, "feature_dim")
    _positive_float(training, "learning_rate")
    if int(training["validation_steps"]) <= int(training["smoke_steps"]):
        raise ValueError("validation_steps must be greater than smoke_steps.")
    if encoder:
        architecture = str(encoder.get("architecture", "normalized_vicreg_mlp"))
        if architecture != "normalized_vicreg_mlp":
            raise ValueError("Training encoder.architecture must be 'normalized_vicreg_mlp'.")
        embedding_dim = int(encoder.get("embedding_dim", 0))
        dropout = float(encoder.get("dropout", -1.0))
        if embedding_dim <= 0:
            raise ValueError("Training encoder.embedding_dim must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Training encoder.dropout must be in [0, 1).")
        normalization = encoder.get("normalization", {})
        if normalization and not isinstance(normalization, dict):
            raise ValueError("Training encoder.normalization must be an object.")
        if isinstance(normalization, dict) and bool(normalization.get("enabled", True)):
            if int(normalization.get("max_files_for_fit", 1)) <= 0:
                raise ValueError("Training encoder.normalization.max_files_for_fit must be positive.")
            if int(normalization.get("max_windows_per_file", 1)) <= 0:
                raise ValueError("Training encoder.normalization.max_windows_per_file must be positive.")


def refuse_local_training() -> None:
    raise LocalTrainingDisabledError(
        "Local Mac training is disabled for this project. Run validation locally, then launch "
        "training with `modal run modal_train.py --config-path configs/modal_training.json --run-full`."
    )


def build_modal_run_command(
    config_path: str | Path,
    *,
    run_validation: bool = False,
    run_full: bool,
) -> list[str]:
    command = [
        "modal",
        "run",
        "modal_train.py",
        "--config-path",
        str(config_path),
    ]
    if run_validation:
        command.append("--run-validation")
    if run_full:
        command.append("--run-full")
    return command


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Training config must include a '{key}' object.")
    return value


def _positive_int(section: dict[str, Any], key: str) -> None:
    value = section.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Training config field '{key}' must be a positive integer.")


def _positive_float(section: dict[str, Any], key: str) -> None:
    value = section.get(key)
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"Training config field '{key}' must be positive.")
