from __future__ import annotations

import json
from pathlib import Path

import modal

from marginal_value.logging_utils import log_event


APP_NAME = "marginal-value-build-full-support-shards"
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
    timeout=3600 * 12,
    cpu=16,
    memory=65536,
    volumes={"/data": data_volume, "/artifacts": artifacts_volume},
)
def remote_build_full_support_shards(config: dict, smoke: bool = False) -> dict:
    from marginal_value.data.build_full_support_shards import run_build_full_support_shards

    log_event("modal_build_full_support_shards", "remote_start", smoke=smoke)
    result = run_build_full_support_shards(config, smoke=smoke, on_shard_written=artifacts_volume.commit)
    artifacts_volume.commit()
    log_event("modal_build_full_support_shards", "remote_done", **result)
    return result


@app.local_entrypoint()
def build_full_support_shards(
    config_path: str = "configs/build_full_support_shards.json",
    run_full: bool = False,
    skip_smoke: bool = False,
) -> None:
    if skip_smoke and not run_full:
        raise ValueError("--skip-smoke is only valid when --run-full is also set.")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    log_event("modal_build_full_support_shards", "local_dispatch_start", run_full=run_full, skip_smoke=skip_smoke)
    if skip_smoke:
        log_event("modal_build_full_support_shards", "local_smoke_skipped", reason="previous_smoke_passed")
    else:
        smoke_result = remote_build_full_support_shards.remote(config, smoke=True)
        print(f"Remote full-support shard smoke completed: {smoke_result}")
    if not run_full:
        print("Full full-support shard build was not launched. Re-run with --run-full after reviewing smoke output.")
        log_event("modal_build_full_support_shards", "local_dispatch_done", run_full=run_full, skip_smoke=skip_smoke)
        return
    full_call = remote_build_full_support_shards.spawn(config, smoke=False)
    call_id = getattr(full_call, "object_id", str(full_call))
    print(f"Remote full-support shard build spawned: {call_id}")
    log_event("modal_build_full_support_shards", "local_dispatch_done", run_full=run_full, skip_smoke=skip_smoke)
