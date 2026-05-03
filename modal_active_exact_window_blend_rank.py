from __future__ import annotations

import json
import os
from pathlib import Path

import modal

from marginal_value.logging_utils import log_event


APP_NAME = "marginal-value-active-exact-window-blend-rank"
DATA_VOLUME_NAME = "imu-novelty-subset-data"
ARTIFACTS_VOLUME_NAME = "activelearning-imu-rebuild-cache"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy==2.2.6")
    .add_local_python_source("marginal_value")
)

data_volume = modal.Volume.from_name(os.environ.get("MV_DATA_VOLUME", DATA_VOLUME_NAME), create_if_missing=True)
artifacts_volume = modal.Volume.from_name(
    os.environ.get("MV_ARTIFACTS_VOLUME", ARTIFACTS_VOLUME_NAME),
    create_if_missing=True,
)


@app.function(
    image=image,
    timeout=3600 * 2,
    cpu=16,
    memory=65536,
    volumes={"/data": data_volume, "/artifacts": artifacts_volume},
)
def remote_active_exact_window_blend_rank(config: dict, smoke: bool = False) -> dict:
    from marginal_value.active.exact_window_blend_rank import run_active_exact_window_blend_rank

    log_event("modal_active_exact_window_blend_rank", "remote_start", smoke=smoke)
    result = run_active_exact_window_blend_rank(config, smoke=smoke)
    artifacts_volume.commit()
    log_event("modal_active_exact_window_blend_rank", "remote_done", **result)
    return result


@app.local_entrypoint()
def active_exact_window_blend_rank(
    config_path: str = "configs/active_exact_window_blend_rank.json",
    run_full: bool = False,
    skip_smoke: bool = False,
    wait_full: bool = False,
) -> None:
    if skip_smoke and not run_full:
        raise ValueError("--skip-smoke is only valid when --run-full is also set.")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    log_event(
        "modal_active_exact_window_blend_rank",
        "local_dispatch_start",
        run_full=run_full,
        skip_smoke=skip_smoke,
        wait_full=wait_full,
    )
    if skip_smoke:
        log_event("modal_active_exact_window_blend_rank", "local_smoke_skipped", reason="previous_smoke_passed")
    else:
        smoke_result = remote_active_exact_window_blend_rank.remote(config, smoke=True)
        print(f"Remote active exact-window blend rank smoke completed: {smoke_result}")
    if not run_full:
        print("Full active exact-window blend rank was not launched. Re-run with --run-full after reviewing smoke output.")
        log_event(
            "modal_active_exact_window_blend_rank",
            "local_dispatch_done",
            run_full=run_full,
            skip_smoke=skip_smoke,
            wait_full=wait_full,
        )
        return
    if wait_full:
        full_result = remote_active_exact_window_blend_rank.remote(config, smoke=False)
        print(f"Remote active exact-window blend rank completed: {full_result}")
        log_event(
            "modal_active_exact_window_blend_rank",
            "local_dispatch_done",
            run_full=run_full,
            skip_smoke=skip_smoke,
            wait_full=wait_full,
        )
        return
    full_call = remote_active_exact_window_blend_rank.spawn(config, smoke=False)
    call_id = getattr(full_call, "object_id", str(full_call))
    print(f"Remote active exact-window blend rank full spawned: {call_id}")
    log_event(
        "modal_active_exact_window_blend_rank",
        "local_dispatch_done",
        run_full=run_full,
        skip_smoke=skip_smoke,
        wait_full=wait_full,
    )
