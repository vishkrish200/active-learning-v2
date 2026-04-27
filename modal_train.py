from __future__ import annotations

from pathlib import Path

import modal

from marginal_value.logging_utils import log_event
from marginal_value.training.config import load_training_config, validate_training_dispatch


APP_NAME = "marginal-value-imu-training"
DATA_VOLUME_NAME = "imu-novelty-subset-data"
ARTIFACTS_VOLUME_NAME = "activelearning-imu-rebuild-cache"
FULL_TRAIN_TIMEOUT_SECONDS = 21600

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "numpy==2.2.6",
        "pandas==2.2.3",
        "torch==2.8.0",
    )
    .add_local_python_source("marginal_value")
)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=False)


@app.function(
    image=image,
    gpu="H100",
    timeout=1800,
    volumes={"/data": data_volume, "/artifacts": artifacts_volume},
)
def remote_split_audit(config: dict) -> dict:
    from pathlib import Path

    from marginal_value.data.split_manifest import (
        build_split_manifest,
        split_counts,
        write_manifest_csv,
        write_manifest_report,
    )

    log_event("modal_train", "split_audit_start")
    validate_training_dispatch(config)
    data_root = Path(config["data"]["root"])
    manifest = build_split_manifest(
        data_root,
        pretrain_manifest=config["data"]["pretrain_manifest"],
        val_manifest=config["data"]["val_manifest"],
        feature_glob=config["data"].get("feature_glob", "cache/features/*.npz"),
        raw_glob=config["data"].get("raw_glob", "cache/raw/*.jsonl"),
    )
    output_dir = Path("/artifacts/eval")
    write_manifest_csv(manifest, data_root, output_dir / "split_manifest.csv")
    write_manifest_report(manifest, output_dir / "split_manifest_report.json")
    artifacts_volume.commit()
    log_event("modal_train", "split_audit_done", n_samples=len(manifest), split_counts=split_counts(manifest))
    return {
        "n_samples": len(manifest),
        "split_counts": split_counts(manifest),
        "manifest_path": "/artifacts/eval/split_manifest.csv",
        "report_path": "/artifacts/eval/split_manifest_report.json",
    }


@app.function(
    image=image,
    gpu="H100",
    timeout=1800,
    volumes={"/data": data_volume, "/artifacts": artifacts_volume},
)
def remote_smoke_train(config: dict) -> dict:
    from marginal_value.training.torch_train import train_ssl_encoder

    log_event("modal_train", "smoke_train_start")
    validate_training_dispatch(config)
    result = train_ssl_encoder(config, mode="smoke")
    log_event("modal_train", "smoke_train_done", **result)
    return result


@app.function(
    image=image,
    gpu="H100",
    timeout=1800,
    volumes={"/data": data_volume, "/artifacts": artifacts_volume},
)
def remote_validation_train(config: dict) -> dict:
    from marginal_value.training.torch_train import train_ssl_encoder

    log_event("modal_train", "validation_train_start")
    validate_training_dispatch(config)
    result = train_ssl_encoder(config, mode="validation")
    artifacts_volume.commit()
    log_event("modal_train", "validation_train_done", **result)
    return result


@app.function(
    image=image,
    gpu="H100",
    timeout=FULL_TRAIN_TIMEOUT_SECONDS,
    volumes={"/data": data_volume, "/artifacts": artifacts_volume},
)
def remote_full_train(config: dict) -> dict:
    from marginal_value.training.torch_train import train_ssl_encoder

    log_event("modal_train", "full_train_start")
    validate_training_dispatch(config)
    result = train_ssl_encoder(config, mode="full")
    artifacts_volume.commit()
    log_event("modal_train", "full_train_done", **result)
    return result


@app.local_entrypoint()
def train(
    config_path: str = "configs/modal_training.json",
    run_validation: bool = False,
    run_full: bool = False,
) -> None:
    config = load_training_config(Path(config_path))
    validate_training_dispatch(config)
    log_event("modal_train", "local_dispatch_start", run_validation=run_validation, run_full=run_full)

    audit_result = remote_split_audit.remote(config)
    print(f"Remote split audit passed: {audit_result}")

    smoke_result = remote_smoke_train.remote(config)
    print(f"Remote smoke training passed on {smoke_result['device']}: {smoke_result}")

    if run_validation or run_full:
        validation_result = remote_validation_train.remote(config)
        print(f"Remote validation training passed on {validation_result['device']}: {validation_result}")

    if not run_full:
        print("Full training was not launched. Re-run with --run-full after reviewing smoke output.")
        log_event("modal_train", "local_dispatch_done", run_validation=run_validation, run_full=run_full)
        return

    full_result = remote_full_train.remote(config)
    print(f"Remote full training completed: {full_result}")
    log_event("modal_train", "local_dispatch_done", run_validation=run_validation, run_full=run_full)
