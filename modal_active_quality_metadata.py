from __future__ import annotations

from pathlib import Path

import modal

from marginal_value.logging_utils import log_event


APP_NAME = "marginal-value-active-quality-metadata"
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
    timeout=7200,
    cpu=16.0,
    memory=32768,
    volumes={"/data": data_volume, "/artifacts": artifacts_volume},
)
def remote_active_quality_metadata(config: dict, smoke: bool = False) -> dict:
    from marginal_value.active.quality_metadata import build_active_quality_metadata

    log_event("modal_active_quality_metadata", "remote_start", smoke=smoke)
    result = build_active_quality_metadata(config, smoke=smoke)
    artifacts_volume.commit()
    log_event("modal_active_quality_metadata", "remote_done", **result)
    return result


@app.local_entrypoint()
def active_quality_metadata(
    config_path: str = "configs/active_quality_metadata_smoke_full_pretrain.json",
    run_full: bool = False,
    skip_smoke: bool = False,
) -> None:
    import json

    if skip_smoke and not run_full:
        raise ValueError("--skip-smoke is only valid when --run-full is also set.")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    log_event("modal_active_quality_metadata", "local_dispatch_start", run_full=run_full, skip_smoke=skip_smoke)
    if skip_smoke:
        log_event("modal_active_quality_metadata", "local_smoke_skipped", reason="previous_smoke_passed")
    else:
        smoke_result = remote_active_quality_metadata.remote(config, smoke=True)
        print(f"Remote active quality metadata smoke completed: {smoke_result}")
    if not run_full:
        print("Full active quality metadata was not launched. Re-run with --run-full after reviewing smoke output.")
        log_event("modal_active_quality_metadata", "local_dispatch_done", run_full=run_full, skip_smoke=skip_smoke)
        return
    full_result = remote_active_quality_metadata.remote(config, smoke=False)
    print(f"Remote active quality metadata full completed: {full_result}")
    log_event("modal_active_quality_metadata", "local_dispatch_done", run_full=run_full, skip_smoke=skip_smoke)
