from __future__ import annotations

from pathlib import Path

import modal

from marginal_value.active.topk_audit_pack import (
    load_topk_audit_pack_config,
    validate_topk_audit_pack_config,
)
from marginal_value.logging_utils import log_event


APP_NAME = "marginal-value-active-topk-audit-pack"
DATA_VOLUME_NAME = "imu-novelty-subset-data"
ARTIFACTS_VOLUME_NAME = "activelearning-imu-rebuild-cache"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy==2.2.6", "matplotlib==3.10.3")
    .add_local_python_source("marginal_value")
)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=False)


@app.function(
    image=image,
    timeout=1800,
    cpu=8,
    memory=16384,
    volumes={"/data": data_volume, "/artifacts": artifacts_volume},
)
def remote_active_topk_audit_pack(config: dict, smoke: bool = False) -> dict:
    from marginal_value.active.topk_audit_pack import run_topk_audit_pack

    log_event("modal_active_topk_audit_pack", "remote_start", smoke=smoke)
    validate_topk_audit_pack_config(config)
    result = run_topk_audit_pack(config, smoke=smoke)
    artifacts_volume.commit()
    log_event("modal_active_topk_audit_pack", "remote_done", **result)
    return result


@app.local_entrypoint()
def active_topk_audit_pack(
    config_path: str = "configs/active_exact_window_topk_audit.json",
    smoke: bool = False,
) -> None:
    config = load_topk_audit_pack_config(Path(config_path))
    validate_topk_audit_pack_config(config)
    log_event("modal_active_topk_audit_pack", "local_dispatch_start", smoke=smoke, config_path=config_path)
    result = remote_active_topk_audit_pack.remote(config, smoke=smoke)
    print(f"Remote active top-k audit pack completed: {result}")
    log_event("modal_active_topk_audit_pack", "local_dispatch_done", smoke=smoke)
