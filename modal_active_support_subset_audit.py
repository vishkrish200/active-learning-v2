from __future__ import annotations

import json
from pathlib import Path

import modal

from marginal_value.logging_utils import log_event


APP_NAME = "marginal-value-active-support-subset-audit"
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
    timeout=1800,
    cpu=8,
    memory=32768,
    volumes={"/data": data_volume, "/artifacts": artifacts_volume},
)
def remote_active_support_subset_audit(config: dict, smoke: bool = False) -> dict:
    from marginal_value.active.support_subset_audit import run_support_subset_audit

    log_event("modal_active_support_subset_audit", "remote_start", smoke=smoke)
    result = run_support_subset_audit(config, smoke=smoke)
    artifacts_volume.commit()
    log_event("modal_active_support_subset_audit", "remote_done", **result)
    return result


@app.local_entrypoint()
def active_support_subset_audit(
    config_path: str = "configs/active_final_blend_rank_budget_h100.json",
    run_full: bool = False,
    skip_smoke: bool = False,
) -> None:
    if skip_smoke and not run_full:
        raise ValueError("--skip-smoke is only valid when --run-full is also set.")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    log_event("modal_active_support_subset_audit", "local_dispatch_start", run_full=run_full, skip_smoke=skip_smoke)
    if skip_smoke:
        log_event("modal_active_support_subset_audit", "local_smoke_skipped", reason="previous_smoke_passed")
    else:
        smoke_result = remote_active_support_subset_audit.remote(config, smoke=True)
        print(f"Remote active support subset audit smoke completed: {smoke_result}")
    if not run_full:
        print("Full active support subset audit was not launched. Re-run with --run-full after reviewing smoke output.")
        log_event("modal_active_support_subset_audit", "local_dispatch_done", run_full=run_full, skip_smoke=skip_smoke)
        return
    full_result = remote_active_support_subset_audit.remote(config, smoke=False)
    print(f"Remote active support subset audit full completed: {full_result}")
    log_event("modal_active_support_subset_audit", "local_dispatch_done", run_full=run_full, skip_smoke=skip_smoke)
