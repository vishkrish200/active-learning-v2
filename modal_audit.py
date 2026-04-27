from __future__ import annotations

from pathlib import Path

import modal

from marginal_value.logging_utils import log_event
from marginal_value.eval.ranking_audit import load_audit_config, validate_audit_config


APP_NAME = "marginal-value-ranking-audit"
ARTIFACTS_VOLUME_NAME = "activelearning-imu-rebuild-cache"

app = modal.App(APP_NAME)

image = modal.Image.debian_slim(python_version="3.12").add_local_python_source("marginal_value")

artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=False)


@app.function(
    image=image,
    timeout=600,
    volumes={"/artifacts": artifacts_volume},
)
def remote_ranking_audit(config: dict, smoke: bool = False) -> dict:
    from marginal_value.eval.ranking_audit import run_ranking_audit

    log_event("modal_audit", "remote_ranking_audit_start", smoke=smoke)
    validate_audit_config(config)
    result = run_ranking_audit(config, smoke=smoke)
    artifacts_volume.commit()
    log_event("modal_audit", "remote_ranking_audit_done", **result)
    return result


@app.local_entrypoint()
def audit_ranking(config_path: str = "configs/ranking_audit.json", smoke: bool = False) -> None:
    config = load_audit_config(Path(config_path))
    validate_audit_config(config)
    log_event("modal_audit", "local_dispatch_start", smoke=smoke)
    result = remote_ranking_audit.remote(config, smoke=smoke)
    print(f"Remote ranking audit completed: {result}")
    log_event("modal_audit", "local_dispatch_done", smoke=smoke)
