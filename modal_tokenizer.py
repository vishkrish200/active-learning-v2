from __future__ import annotations

from pathlib import Path

import modal

from marginal_value.logging_utils import log_event
from marginal_value.tokenization.config import load_tokenizer_config, validate_tokenizer_config


APP_NAME = "marginal-value-motion-tokenizer"
DATA_VOLUME_NAME = "imu-novelty-subset-data"
ARTIFACTS_VOLUME_NAME = "activelearning-imu-rebuild-cache"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy==2.2.6")
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
def remote_tokenizer_smoke(config: dict) -> dict:
    from marginal_value.data.split_manifest import build_split_manifest, split_counts
    from marginal_value.tokenization.modal_tokenizer import run_tokenizer_pipeline

    log_event("modal_tokenizer", "remote_tokenizer_smoke_start")
    validate_tokenizer_config(config)
    manifest = _build_manifest(config)
    log_event("modal_tokenizer", "manifest_ready", split_counts=split_counts(manifest), mode="smoke")
    result = run_tokenizer_pipeline(config, manifest, smoke=True)
    artifacts_volume.commit()
    log_event("modal_tokenizer", "remote_tokenizer_smoke_done", **result)
    return result


@app.function(
    image=image,
    gpu="H100",
    timeout=7200,
    volumes={"/data": data_volume, "/artifacts": artifacts_volume},
)
def remote_tokenizer_full(config: dict) -> dict:
    from marginal_value.data.split_manifest import build_split_manifest, split_counts
    from marginal_value.tokenization.modal_tokenizer import run_tokenizer_pipeline

    log_event("modal_tokenizer", "remote_tokenizer_full_start")
    validate_tokenizer_config(config)
    manifest = _build_manifest(config)
    log_event("modal_tokenizer", "manifest_ready", split_counts=split_counts(manifest), mode="full")
    result = run_tokenizer_pipeline(config, manifest, smoke=False)
    artifacts_volume.commit()
    log_event("modal_tokenizer", "remote_tokenizer_full_done", **result)
    return result


@app.local_entrypoint()
def tokenize(
    config_path: str = "configs/modal_tokenizer.json",
    run_full: bool = False,
    skip_smoke: bool = False,
) -> None:
    config = load_tokenizer_config(Path(config_path))
    validate_tokenizer_config(config)
    if skip_smoke and not run_full:
        raise ValueError("--skip-smoke is only valid when --run-full is also set.")
    log_event("modal_tokenizer", "local_dispatch_start", run_full=run_full, skip_smoke=skip_smoke)

    if skip_smoke:
        log_event("modal_tokenizer", "local_smoke_skipped", reason="previous_smoke_passed")
    else:
        smoke_result = remote_tokenizer_smoke.remote(config)
        print(f"Remote tokenizer smoke completed: {smoke_result}")

    if not run_full:
        print("Full tokenization was not launched. Re-run with --run-full after reviewing smoke output.")
        log_event("modal_tokenizer", "local_dispatch_done", run_full=run_full, skip_smoke=skip_smoke)
        return

    full_result = remote_tokenizer_full.remote(config)
    print(f"Remote tokenizer full completed: {full_result}")
    log_event("modal_tokenizer", "local_dispatch_done", run_full=run_full, skip_smoke=skip_smoke)


def _build_manifest(config: dict):
    from marginal_value.data.split_manifest import build_split_manifest

    data_root = Path(config["data"]["root"])
    return build_split_manifest(
        data_root,
        pretrain_manifest=config["data"]["pretrain_manifest"],
        val_manifest=config["data"]["val_manifest"],
        feature_glob=config["data"].get("feature_glob", "cache/features/*.npz"),
        raw_glob=config["data"].get("raw_glob", "cache/raw/*.jsonl"),
    )
