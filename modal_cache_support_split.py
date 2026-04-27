from __future__ import annotations

import copy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import modal

from marginal_value.logging_utils import log_event


APP_NAME = "marginal-value-cache-support-split"
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
    timeout=21600,
    volumes={"/source": source_volume, "/data": target_volume},
)
def remote_cache_support_split(config: dict, smoke: bool = False) -> dict:
    from marginal_value.data.cache_support_split import build_support_split_cache

    log_event("modal_cache_support_split", "remote_start", smoke=smoke)
    result = build_support_split_cache(config, smoke=smoke)
    target_volume.commit()
    log_event("modal_cache_support_split", "remote_done", **result)
    return result


@app.local_entrypoint()
def cache_support_split(
    config_path: str = "configs/cache_pretrain_worker_coverage.json",
    run_full: bool = False,
    skip_smoke: bool = False,
    num_shards: int = 1,
) -> None:
    import json

    if skip_smoke and not run_full:
        raise ValueError("--skip-smoke is only valid when --run-full is also set.")
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    log_event("modal_cache_support_split", "local_dispatch_start", run_full=run_full, skip_smoke=skip_smoke, num_shards=num_shards)
    if skip_smoke:
        log_event("modal_cache_support_split", "local_smoke_skipped", reason="previous_smoke_passed")
    else:
        smoke_result = remote_cache_support_split.remote(config, smoke=True)
        print(f"Remote support cache smoke completed: {smoke_result}")
    if not run_full:
        print("Full support cache build was not launched. Re-run with --run-full after reviewing smoke output.")
        log_event("modal_cache_support_split", "local_dispatch_done", run_full=run_full, skip_smoke=skip_smoke)
        return
    if num_shards == 1:
        full_result = remote_cache_support_split.remote(config, smoke=False)
    else:
        full_result = _run_sharded_full_cache(config, num_shards=num_shards)
    print(f"Remote support cache full completed: {full_result}")
    log_event("modal_cache_support_split", "local_dispatch_done", run_full=run_full, skip_smoke=skip_smoke, num_shards=num_shards)


def _run_sharded_full_cache(config: dict, *, num_shards: int) -> dict:
    log_event("modal_cache_support_split", "sharded_dispatch_start", num_shards=num_shards)
    shard_results: list[dict] = []
    with ThreadPoolExecutor(max_workers=num_shards) as executor:
        futures = {}
        for shard_index in range(num_shards):
            shard_config = copy.deepcopy(config)
            shard_config.setdefault("execution", {})
            shard_config["execution"]["num_shards"] = num_shards
            shard_config["execution"]["shard_index"] = shard_index
            futures[executor.submit(remote_cache_support_split.remote, shard_config, False)] = shard_index
        for future in as_completed(futures):
            shard_index = futures[future]
            result = future.result()
            result["shard_index"] = shard_index
            shard_results.append(result)
            log_event("modal_cache_support_split", "shard_done", **result)
    shard_results.sort(key=lambda row: int(row.get("shard_index", 0)))
    summary = {
        "mode": "full",
        "num_shards": num_shards,
        "n_manifest_urls": max((int(row.get("n_manifest_urls", 0)) for row in shard_results), default=0),
        "n_selected": sum(int(row.get("n_selected", 0)) for row in shard_results),
        "written": sum(int(row.get("written", 0)) for row in shard_results),
        "skipped": sum(int(row.get("skipped", 0)) for row in shard_results),
        "missing_source": sum(int(row.get("missing_source", 0)) for row in shard_results),
        "malformed_source": sum(int(row.get("malformed_source", 0)) for row in shard_results),
        "raw_copied": sum(int(row.get("raw_copied", 0)) for row in shard_results),
        "feature_written": sum(int(row.get("feature_written", 0)) for row in shard_results),
        "shards": shard_results,
    }
    log_event("modal_cache_support_split", "sharded_dispatch_done", **summary)
    return summary
