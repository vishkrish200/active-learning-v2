from __future__ import annotations

from pathlib import Path

import modal

from marginal_value.data.source_archive_audit import (
    load_source_archive_audit_config,
    validate_source_archive_audit_config,
)
from marginal_value.logging_utils import log_event


APP_NAME = "marginal-value-source-archive-audit"
SOURCE_VOLUME_NAME = "activelearning-observe-full-cache"
DATA_VOLUME_NAME = "imu-novelty-subset-data"
ARTIFACTS_VOLUME_NAME = "activelearning-imu-rebuild-cache"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("zstandard==0.23.0")
    .add_local_python_source("marginal_value")
)

source_volume = modal.Volume.from_name(SOURCE_VOLUME_NAME, create_if_missing=False)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=False)


@app.function(
    image=image,
    timeout=7200,
    cpu=8,
    memory=32768,
    volumes={"/source": source_volume, "/data": data_volume, "/artifacts": artifacts_volume},
)
def remote_source_archive_audit(config: dict, smoke: bool = False) -> dict:
    from marginal_value.data.source_archive_audit import run_source_archive_audit

    log_event("modal_source_archive_audit", "remote_start", smoke=smoke)
    validate_source_archive_audit_config(config)
    result = run_source_archive_audit(config, smoke=smoke)
    artifacts_volume.commit()
    log_event("modal_source_archive_audit", "remote_done", **result)
    return result


@app.local_entrypoint()
def source_archive_audit(
    config_path: str = "configs/source_archive_audit_pretrain_100k.json",
    run_full: bool = False,
    skip_smoke: bool = False,
) -> None:
    if skip_smoke and not run_full:
        raise ValueError("--skip-smoke is only valid when --run-full is also set.")
    config = load_source_archive_audit_config(Path(config_path))
    validate_source_archive_audit_config(config)
    log_event("modal_source_archive_audit", "local_dispatch_start", run_full=run_full, skip_smoke=skip_smoke)
    if skip_smoke:
        log_event("modal_source_archive_audit", "local_smoke_skipped", reason="previous_smoke_passed")
    else:
        smoke_result = remote_source_archive_audit.remote(config, smoke=True)
        print(f"Remote source archive smoke completed: {smoke_result}")
    if not run_full:
        print("Full source archive audit was not launched. Re-run with --run-full after reviewing smoke output.")
        log_event("modal_source_archive_audit", "local_dispatch_done", run_full=run_full, skip_smoke=skip_smoke)
        return
    full_result = remote_source_archive_audit.remote(config, smoke=False)
    print(f"Remote source archive full completed: {full_result}")
    log_event("modal_source_archive_audit", "local_dispatch_done", run_full=run_full, skip_smoke=skip_smoke)
