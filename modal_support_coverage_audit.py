from __future__ import annotations

from pathlib import Path

import modal

from marginal_value.data.support_coverage_audit import (
    load_support_coverage_config,
    validate_support_coverage_config,
)
from marginal_value.logging_utils import log_event


APP_NAME = "marginal-value-support-coverage-audit"
SOURCE_VOLUME_NAME = "activelearning-observe-full-cache"
DATA_VOLUME_NAME = "imu-novelty-subset-data"
ARTIFACTS_VOLUME_NAME = "activelearning-imu-rebuild-cache"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy==2.2.6")
    .add_local_python_source("marginal_value")
)

source_volume = modal.Volume.from_name(SOURCE_VOLUME_NAME, create_if_missing=False)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=False)


@app.function(
    image=image,
    timeout=3600,
    volumes={"/source": source_volume, "/data": data_volume, "/artifacts": artifacts_volume},
)
def remote_support_coverage_audit(config: dict, smoke: bool = False) -> dict:
    from marginal_value.data.support_coverage_audit import run_support_coverage_audit

    log_event("modal_support_coverage_audit", "remote_start", smoke=smoke)
    validate_support_coverage_config(config)
    result = run_support_coverage_audit(config, smoke=smoke)
    artifacts_volume.commit()
    log_event("modal_support_coverage_audit", "remote_done", **result)
    return result


@app.local_entrypoint()
def support_coverage_audit(
    config_path: str = "configs/support_coverage_audit_worker_coverage.json",
    run_full: bool = False,
    skip_smoke: bool = False,
) -> None:
    if skip_smoke and not run_full:
        raise ValueError("--skip-smoke is only valid when --run-full is also set.")
    config = load_support_coverage_config(Path(config_path))
    validate_support_coverage_config(config)
    log_event("modal_support_coverage_audit", "local_dispatch_start", run_full=run_full, skip_smoke=skip_smoke)
    if skip_smoke:
        log_event("modal_support_coverage_audit", "local_smoke_skipped", reason="previous_smoke_passed")
    else:
        smoke_result = remote_support_coverage_audit.remote(config, smoke=True)
        print(f"Remote support coverage smoke completed: {smoke_result}")
    if not run_full:
        print("Full support coverage audit was not launched. Re-run with --run-full after reviewing smoke output.")
        log_event("modal_support_coverage_audit", "local_dispatch_done", run_full=run_full, skip_smoke=skip_smoke)
        return
    full_result = remote_support_coverage_audit.remote(config, smoke=False)
    print(f"Remote support coverage full completed: {full_result}")
    log_event("modal_support_coverage_audit", "local_dispatch_done", run_full=run_full, skip_smoke=skip_smoke)
