from __future__ import annotations

from pathlib import Path

import modal

from marginal_value.eval.top_clip_visual_audit import (
    load_visual_audit_config,
    validate_visual_audit_config,
)
from marginal_value.logging_utils import log_event


APP_NAME = "marginal-value-top-clip-visual-audit"
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
    timeout=900,
    cpu=8,
    memory=16384,
    volumes={"/data": data_volume, "/artifacts": artifacts_volume},
)
def remote_top_clip_visual_audit(config: dict, smoke: bool = False) -> dict:
    from marginal_value.eval.top_clip_visual_audit import run_top_clip_visual_audit

    log_event("modal_top_clip_visual_audit", "remote_start", smoke=smoke)
    validate_visual_audit_config(config)
    result = run_top_clip_visual_audit(config, smoke=smoke)
    artifacts_volume.commit()
    log_event("modal_top_clip_visual_audit", "remote_done", **result)
    return result


@app.local_entrypoint()
def top_clip_visual_audit(
    config_path: str = "configs/top_clip_visual_audit_quality_gated_grammar_new_physical_source_hybrid75_feature_subcluster_rawcorr.json",
    smoke: bool = False,
) -> None:
    config = load_visual_audit_config(Path(config_path))
    validate_visual_audit_config(config)
    log_event("modal_top_clip_visual_audit", "local_dispatch_start", smoke=smoke, config_path=config_path)
    result = remote_top_clip_visual_audit.remote(config, smoke=smoke)
    print(f"Remote top-clip visual audit completed: {result}")
    log_event("modal_top_clip_visual_audit", "local_dispatch_done", smoke=smoke)
