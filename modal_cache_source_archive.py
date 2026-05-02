from __future__ import annotations

from pathlib import Path

import modal

from marginal_value.logging_utils import log_event


APP_NAME = "marginal-value-cache-source-archive"
SOURCE_VOLUME_NAME = "activelearning-observe-full-cache"
TARGET_VOLUME_NAME = "imu-novelty-subset-data"
ARTIFACTS_VOLUME_NAME = "activelearning-imu-rebuild-cache"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy==2.2.6", "zstandard==0.23.0")
    .add_local_python_source("marginal_value")
)

source_volume = modal.Volume.from_name(SOURCE_VOLUME_NAME, create_if_missing=False)
target_volume = modal.Volume.from_name(TARGET_VOLUME_NAME, create_if_missing=False)
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=False)


@app.function(
    image=image,
    timeout=21600,
    cpu=8.0,
    volumes={"/source": source_volume, "/data": target_volume, "/artifacts": artifacts_volume},
)
def remote_cache_source_archive(config: dict, smoke: bool = False) -> dict:
    from marginal_value.data.cache_source_archive import cache_source_archive

    log_event("modal_cache_source_archive", "remote_start", smoke=smoke)
    result = cache_source_archive(config, smoke=smoke)
    target_volume.commit()
    artifacts_volume.commit()
    log_event("modal_cache_source_archive", "remote_done", **result)
    return result


@app.local_entrypoint()
def cache_source_archive(
    config_path: str = "configs/cache_pretrain_archive_missing.json",
    run_full: bool = False,
    skip_smoke: bool = False,
) -> None:
    import json

    if skip_smoke and not run_full:
        raise ValueError("--skip-smoke is only valid when --run-full is also set.")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    log_event("modal_cache_source_archive", "local_dispatch_start", run_full=run_full, skip_smoke=skip_smoke)
    if skip_smoke:
        log_event("modal_cache_source_archive", "local_smoke_skipped", reason="previous_smoke_passed")
    else:
        smoke_result = remote_cache_source_archive.remote(config, smoke=True)
        print(f"Remote source archive cache smoke completed: {smoke_result}")
    if not run_full:
        print("Full source archive cache was not launched. Re-run with --run-full after reviewing smoke output.")
        log_event("modal_cache_source_archive", "local_dispatch_done", run_full=run_full, skip_smoke=skip_smoke)
        return
    full_result = remote_cache_source_archive.remote(config, smoke=False)
    print(f"Remote source archive cache full completed: {full_result}")
    log_event("modal_cache_source_archive", "local_dispatch_done", run_full=run_full, skip_smoke=skip_smoke)
