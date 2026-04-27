from __future__ import annotations

from pathlib import Path

import modal

from marginal_value.logging_utils import log_event


APP_NAME = "marginal-value-cache-new-split"
SOURCE_VOLUME_NAME = "activelearning-observe-full-cache"
TARGET_VOLUME_NAME = "imu-novelty-subset-data"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy==2.2.6")
    .add_local_python_source("marginal_value")
)

source_volume = modal.Volume.from_name(SOURCE_VOLUME_NAME, create_if_missing=False)
target_volume = modal.Volume.from_name(TARGET_VOLUME_NAME, create_if_missing=False)


@app.function(
    image=image,
    timeout=1200,
    volumes={"/source": source_volume, "/data": target_volume},
)
def remote_cache_new_split(config: dict, smoke: bool = False) -> dict:
    from marginal_value.data.cache_new_split import build_new_split_cache

    log_event("modal_cache_new_split", "remote_start", smoke=smoke)
    result = build_new_split_cache(config, smoke=smoke)
    target_volume.commit()
    log_event("modal_cache_new_split", "remote_done", **result)
    return result


@app.local_entrypoint()
def cache_new_split(
    config_path: str = "configs/cache_new_split.json",
    run_full: bool = False,
    skip_smoke: bool = False,
) -> None:
    import json

    if skip_smoke and not run_full:
        raise ValueError("--skip-smoke is only valid when --run-full is also set.")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    log_event("modal_cache_new_split", "local_dispatch_start", run_full=run_full, skip_smoke=skip_smoke)
    if skip_smoke:
        log_event("modal_cache_new_split", "local_smoke_skipped", reason="previous_smoke_passed")
    else:
        smoke_result = remote_cache_new_split.remote(config, smoke=True)
        print(f"Remote new-split cache smoke completed: {smoke_result}")
    if not run_full:
        print("Full new-split cache build was not launched. Re-run with --run-full after reviewing smoke output.")
        log_event("modal_cache_new_split", "local_dispatch_done", run_full=run_full, skip_smoke=skip_smoke)
        return
    full_result = remote_cache_new_split.remote(config, smoke=False)
    print(f"Remote new-split cache full completed: {full_result}")
    log_event("modal_cache_new_split", "local_dispatch_done", run_full=run_full, skip_smoke=skip_smoke)
