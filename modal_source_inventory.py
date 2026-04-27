from __future__ import annotations

from pathlib import Path

import modal

from marginal_value.logging_utils import log_event


APP_NAME = "marginal-value-source-inventory"
SOURCE_VOLUME_NAME = "activelearning-observe-full-cache"
TARGET_VOLUME_NAME = "imu-novelty-subset-data"
ARTIFACT_VOLUME_NAME = "imu-novelty-artifacts"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy==2.2.6")
    .add_local_python_source("marginal_value")
)

source_volume = modal.Volume.from_name(SOURCE_VOLUME_NAME, create_if_missing=False)
target_volume = modal.Volume.from_name(TARGET_VOLUME_NAME, create_if_missing=False)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    timeout=3600,
    volumes={"/source": source_volume, "/data": target_volume, "/artifacts": artifact_volume},
)
def remote_source_inventory(config: dict, smoke: bool = False) -> dict:
    from marginal_value.data.source_inventory import run_source_inventory

    log_event("modal_source_inventory", "remote_start", smoke=smoke)
    result = run_source_inventory(config, smoke=smoke)
    target_volume.commit()
    artifact_volume.commit()
    log_event("modal_source_inventory", "remote_done", **result)
    return result


@app.local_entrypoint()
def source_inventory(
    config_path: str = "configs/source_inventory_observe_full.json",
    run_full: bool = False,
    skip_smoke: bool = False,
) -> None:
    import json

    if skip_smoke and not run_full:
        raise ValueError("--skip-smoke is only valid when --run-full is also set.")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    log_event("modal_source_inventory", "local_dispatch_start", run_full=run_full, skip_smoke=skip_smoke)
    if skip_smoke:
        log_event("modal_source_inventory", "local_smoke_skipped", reason="previous_smoke_passed")
    else:
        smoke_result = remote_source_inventory.remote(config, smoke=True)
        print(f"Remote source inventory smoke completed: {smoke_result}")
    if not run_full:
        print("Full source inventory was not launched. Re-run with --run-full after reviewing smoke output.")
        log_event("modal_source_inventory", "local_dispatch_done", run_full=run_full, skip_smoke=skip_smoke)
        return
    full_result = remote_source_inventory.remote(config, smoke=False)
    print(f"Remote source inventory full completed: {full_result}")
    log_event("modal_source_inventory", "local_dispatch_done", run_full=run_full, skip_smoke=skip_smoke)
