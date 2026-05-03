from __future__ import annotations

import argparse
from collections import Counter
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from marginal_value.active.embedding_cache import _cache_metadata, _shard_dir
from marginal_value.active.embedding_precompute import validate_active_embedding_precompute_config
from marginal_value.active.exact_window_blend_rank import validate_active_exact_window_blend_rank_config
from marginal_value.active.registry import ClipRecord
from marginal_value.active.spike_hygiene_ablation import validate_spike_hygiene_ablation_config
from marginal_value.data.build_full_support_shards import validate_build_full_support_shards_config
from marginal_value.data.cache_manifest_urls import validate_manifest_url_cache_config
from marginal_value.data.split_manifest import hash_manifest_url, read_manifest_urls
from marginal_value.logging_utils import log_event


DEFAULT_RUN_ID = "hidden_test"
DEFAULT_DATA_VOLUME = "imu-novelty-subset-data"
DEFAULT_ARTIFACTS_VOLUME = "activelearning-imu-rebuild-cache"
DEFAULT_LEFT_SUPPORT_SHARD_DIR = "/artifacts/active/embedding_cache/ts2vec_window_full_new/embeddings_7481b57ede264d17002b4316_shards"
DEFAULT_TS2VEC_CHECKPOINT_PATH = "/artifacts/checkpoints/ts2vec_candidate_eval/ts2vec_best.pt"
DEFAULT_MIN_LEFT_SUPPORT_CLIPS = 20000
DEFAULT_MAX_QUERY_CLIPS = 2500
DEFAULT_ALPHA = 0.5
DEFAULT_TOP_K_VALUES = [10, 50, 100, 200]


def prepare_hidden_test_run(config: Mapping[str, Any]) -> dict[str, Any]:
    """Prepare a reproducible hidden-test run package without launching compute.

    The prepared package binds arbitrary old/new manifests to the currently
    promoted exact-window artifact-gated selector. It writes Modal stage configs
    and a commands file, but it does not upload manifests, run Modal, train, or
    compute embeddings by itself.
    """

    inputs = _required_mapping(config, "inputs")
    artifacts = _required_mapping(config, "artifacts")
    modal = dict(config.get("modal", {}))
    method = dict(config.get("method", {}))

    old_manifest = Path(str(inputs["old_manifest"]))
    new_manifest = Path(str(inputs["new_manifest"]))
    old_urls = _read_valid_manifest(old_manifest, label="old_manifest")
    new_urls = _read_valid_manifest(new_manifest, label="new_manifest")
    _validate_manifest_disjoint(old_urls, new_urls)

    run_id = str(config.get("run_id") or artifacts.get("run_id") or DEFAULT_RUN_ID)
    run_dir = Path(str(artifacts["run_dir"]))
    manifest_dir = run_dir / "manifests"
    config_dir = run_dir / "configs"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    local_old_manifest = manifest_dir / "pretrain_urls.txt"
    local_new_manifest = manifest_dir / "new_urls.txt"
    _copy_manifest(old_manifest, local_old_manifest)
    _copy_manifest(new_manifest, local_new_manifest)

    data_volume = str(modal.get("data_volume", DEFAULT_DATA_VOLUME))
    artifacts_volume = str(modal.get("artifacts_volume", DEFAULT_ARTIFACTS_VOLUME))
    remote_manifest_dir = str(modal.get("remote_manifest_dir", f"cache/manifests/hidden_test/{run_id}"))
    remote_artifact_dir = str(modal.get("remote_artifact_dir", f"/artifacts/active/hidden_test/{run_id}"))
    remote_old_manifest = f"{remote_manifest_dir.rstrip('/')}/pretrain_urls.txt"
    remote_new_manifest = f"{remote_manifest_dir.rstrip('/')}/new_urls.txt"
    remote_cached_old_manifest = f"{remote_manifest_dir.rstrip('/')}/pretrain_cached_urls.txt"
    remote_cached_new_manifest = f"{remote_manifest_dir.rstrip('/')}/new_cached_urls.txt"
    query_cache_dir = f"{remote_artifact_dir.rstrip('/')}/query_ts2vec"

    representation_options = _ts2vec_representation_options(method)
    query_shard_dir = _expected_query_shard_dir(
        new_urls,
        cache_dir=query_cache_dir,
        representation_options=representation_options,
    )
    output_dirs = {
        "manifest_url_cache": f"{remote_artifact_dir.rstrip('/')}/manifest_url_cache",
        "window_shards": f"{remote_artifact_dir.rstrip('/')}/window_shards",
        "exact_window_old_novelty_rank": f"{remote_artifact_dir.rstrip('/')}/exact_window_old_novelty_rank",
        "exact_window_rank": f"{remote_artifact_dir.rstrip('/')}/exact_window_rank",
        "artifact_hygiene_ablation": f"{remote_artifact_dir.rstrip('/')}/artifact_hygiene_ablation",
    }

    configs = _stage_configs(
        data_volume=data_volume,
        artifacts_volume=artifacts_volume,
        remote_old_manifest=remote_old_manifest,
        remote_new_manifest=remote_new_manifest,
        remote_cached_old_manifest=remote_cached_old_manifest,
        remote_cached_new_manifest=remote_cached_new_manifest,
        query_cache_dir=query_cache_dir,
        query_shard_dir=str(query_shard_dir),
        output_dirs=output_dirs,
        method=method,
        representation_options=representation_options,
        new_count=len(new_urls),
    )
    config_paths = _write_configs(config_dir, configs)
    plan = _run_plan(
        run_id=run_id,
        old_manifest=old_manifest,
        new_manifest=new_manifest,
        old_count=len(old_urls),
        new_count=len(new_urls),
        run_dir=run_dir,
        data_volume=data_volume,
        artifacts_volume=artifacts_volume,
        remote_manifest_dir=remote_manifest_dir,
        remote_artifact_dir=remote_artifact_dir,
        query_shard_dir=str(query_shard_dir),
        output_dirs=output_dirs,
        config_paths=config_paths,
    )
    plan_path = run_dir / "hidden_test_run_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    commands_path = run_dir / "commands.sh"
    commands_path.write_text(_commands_text(plan), encoding="utf-8")
    commands_path.chmod(commands_path.stat().st_mode | 0o111)
    (run_dir / "README_hidden_test.md").write_text(_readme_text(plan), encoding="utf-8")

    result = {
        "status": "prepared",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "old_manifest_count": len(old_urls),
        "new_manifest_count": len(new_urls),
        "plan_path": str(plan_path),
        "commands_path": str(commands_path),
        "query_shard_dir": str(query_shard_dir),
    }
    log_event("active_run_hidden_test", "prepared", **result)
    return result


def validate_hidden_test_run_package(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    plan_path = root / "hidden_test_run_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing hidden-test run plan: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    required_files = [
        root / "manifests" / "pretrain_urls.txt",
        root / "manifests" / "new_urls.txt",
        root / "commands.sh",
        root / "README_hidden_test.md",
        root / "configs" / "cache_old_manifest_urls.json",
        root / "configs" / "cache_new_manifest_urls.json",
        root / "configs" / "build_full_support_window_shards.json",
        root / "configs" / "active_exact_window_old_novelty_rank.json",
        root / "configs" / "active_embedding_precompute_ts2vec_new.json",
        root / "configs" / "active_exact_window_blend_rank.json",
        root / "configs" / "active_spike_hygiene_ablation_artifact_gate.json",
        root / "configs" / "final_package_artifact_gate.json",
        root / "configs" / "final_package_exact_window_old_novelty.json",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Hidden-test run package is missing files: {missing}")
    old_urls = _read_valid_manifest(root / "manifests" / "pretrain_urls.txt", label="pretrain_urls")
    new_urls = _read_valid_manifest(root / "manifests" / "new_urls.txt", label="new_urls")
    if int(plan["old_manifest_count"]) != len(old_urls):
        raise ValueError("Hidden-test run plan old_manifest_count does not match copied manifest.")
    if int(plan["new_manifest_count"]) != len(new_urls):
        raise ValueError("Hidden-test run plan new_manifest_count does not match copied manifest.")
    _validate_manifest_disjoint(old_urls, new_urls)
    stage_config_validation = _validate_stage_configs(root, plan)
    return {
        "status": "prepared",
        "run_dir": str(root),
        "old_manifest_count": len(old_urls),
        "new_manifest_count": len(new_urls),
        "query_shard_dir": str(plan["query_shard_dir"]),
        "stage_config_validation": stage_config_validation,
    }


def load_hidden_test_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_stage_configs(root: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    configs = _read_stage_configs(root / "configs")
    cache_old = configs["cache_old_manifest_urls"]
    cache_new = configs["cache_new_manifest_urls"]
    build = configs["build_full_support_window_shards"]
    old_novelty = configs["active_exact_window_old_novelty_rank"]
    precompute = configs["active_embedding_precompute_ts2vec_new"]
    exact = configs["active_exact_window_blend_rank"]
    ablation = configs["active_spike_hygiene_ablation_artifact_gate"]
    package = configs["final_package_artifact_gate"]
    old_novelty_package = configs["final_package_exact_window_old_novelty"]

    validate_manifest_url_cache_config(cache_old)
    validate_manifest_url_cache_config(cache_new)
    validate_build_full_support_shards_config(build)
    validate_active_exact_window_blend_rank_config(old_novelty)
    validate_active_embedding_precompute_config(precompute)
    validate_active_exact_window_blend_rank_config(exact)
    validate_spike_hygiene_ablation_config(ablation)
    _validate_final_package_config_shape(package)
    _validate_final_package_config_shape(old_novelty_package)

    expected_source_pretrain = f"{str(plan['remote_manifest_dir']).rstrip('/')}/pretrain_urls.txt"
    expected_source_new = f"{str(plan['remote_manifest_dir']).rstrip('/')}/new_urls.txt"
    expected_pretrain = f"{str(plan['remote_manifest_dir']).rstrip('/')}/pretrain_cached_urls.txt"
    expected_new = f"{str(plan['remote_manifest_dir']).rstrip('/')}/new_cached_urls.txt"
    if cache_old["target"].get("source_manifest") != expected_source_pretrain:
        raise ValueError("cache_old config source_manifest does not match run plan.")
    if cache_old["target"].get("cached_manifest") != expected_pretrain:
        raise ValueError("cache_old config cached_manifest does not match run plan.")
    if cache_new["target"].get("source_manifest") != expected_source_new:
        raise ValueError("cache_new config source_manifest does not match run plan.")
    if cache_new["target"].get("cached_manifest") != expected_new:
        raise ValueError("cache_new config cached_manifest does not match run plan.")
    for label, config in (("cache_old", cache_old), ("cache_new", cache_new)):
        if not bool(config["execution"].get("fail_if_incomplete", False)):
            raise ValueError(f"{label} config must fail if the fresh manifest cache is incomplete.")
    for label, config in (("build", build), ("old_novelty", old_novelty), ("exact", exact), ("ablation", ablation)):
        manifests = config["data"]["manifests"]
        if manifests.get("pretrain") != expected_pretrain:
            raise ValueError(f"{label} config pretrain manifest does not match run plan.")
        if manifests.get("new") != expected_new:
            raise ValueError(f"{label} config new manifest does not match run plan.")
    if precompute["data"]["manifests"].get("new") != expected_new:
        raise ValueError("precompute config new manifest does not match run plan.")

    query_cache_dir = str(precompute["embeddings"]["cache_dir"]).rstrip("/")
    query_shard_dir = str(plan["query_shard_dir"])
    if not query_shard_dir.startswith(f"{query_cache_dir}/embeddings_") or not query_shard_dir.endswith("_shards"):
        raise ValueError("query_shard_dir is not compatible with precompute embeddings.cache_dir.")
    if str(exact["ranking"]["left_query_shard_dir"]) != query_shard_dir:
        raise ValueError("exact ranking.left_query_shard_dir does not match the run plan query_shard_dir.")
    if str(exact["ranking"]["right_support_shard_dir"]) != str(build["shards"]["output_dir"]):
        raise ValueError("exact ranking.right_support_shard_dir must match build shards.output_dir.")
    if str(old_novelty["ranking"].get("selector_mode")) != "old_novelty_only":
        raise ValueError("old_novelty ranking.selector_mode must be old_novelty_only.")
    if str(old_novelty["ranking"]["right_support_shard_dir"]) != str(build["shards"]["output_dir"]):
        raise ValueError("old_novelty ranking.right_support_shard_dir must match build shards.output_dir.")
    if "left_support_shard_dir" in old_novelty["ranking"]:
        raise ValueError("old_novelty config must not depend on a partial left support shard.")

    exact_output_dir = str(exact["artifacts"]["output_dir"]).rstrip("/")
    expected_diagnostics = f"{exact_output_dir}/active_exact_window_blend_diagnostics_full.csv"
    if str(ablation["artifacts"]["exact_diagnostics_path"]) != expected_diagnostics:
        raise ValueError("ablation artifacts.exact_diagnostics_path does not match exact ranking output.")
    if str(package["source_artifacts"]["source_dir"]) != str(ablation["artifacts"]["output_dir"]):
        raise ValueError("final package source_artifacts.source_dir must match ablation artifacts.output_dir.")
    if str(old_novelty_package["source_artifacts"]["source_dir"]) != str(old_novelty["artifacts"]["output_dir"]):
        raise ValueError("old-novelty final package source_artifacts.source_dir must match old-novelty output.")
    if int(package["validation"]["expected_count"]) != int(plan["new_manifest_count"]):
        raise ValueError("final package validation.expected_count does not match new manifest count.")
    if int(old_novelty_package["validation"]["expected_count"]) != int(plan["new_manifest_count"]):
        raise ValueError("old-novelty final package validation.expected_count does not match new manifest count.")

    return {
        "status": "valid",
        "validated_configs": sorted(configs),
    }


def _read_stage_configs(config_dir: Path) -> dict[str, dict[str, Any]]:
    names = {
        "cache_old_manifest_urls": "cache_old_manifest_urls.json",
        "cache_new_manifest_urls": "cache_new_manifest_urls.json",
        "build_full_support_window_shards": "build_full_support_window_shards.json",
        "active_exact_window_old_novelty_rank": "active_exact_window_old_novelty_rank.json",
        "active_embedding_precompute_ts2vec_new": "active_embedding_precompute_ts2vec_new.json",
        "active_exact_window_blend_rank": "active_exact_window_blend_rank.json",
        "active_spike_hygiene_ablation_artifact_gate": "active_spike_hygiene_ablation_artifact_gate.json",
        "final_package_artifact_gate": "final_package_artifact_gate.json",
        "final_package_exact_window_old_novelty": "final_package_exact_window_old_novelty.json",
    }
    configs: dict[str, dict[str, Any]] = {}
    for key, filename in names.items():
        path = config_dir / filename
        configs[key] = json.loads(path.read_text(encoding="utf-8"))
    return configs


def _validate_final_package_config_shape(config: Mapping[str, Any]) -> None:
    source = _required_mapping(config, "source_artifacts")
    artifacts = _required_mapping(config, "artifacts")
    validation = _required_mapping(config, "validation")
    for key in ("source_dir", "primary_submission", "backup_worker_submission", "diagnostics", "selector_report"):
        if not str(source.get(key, "")).strip():
            raise ValueError(f"final package source_artifacts.{key} is required.")
    if not str(artifacts.get("output_dir", "")).strip():
        raise ValueError("final package artifacts.output_dir is required.")
    if int(validation.get("expected_count", 0)) <= 0:
        raise ValueError("final package validation.expected_count must be positive.")


def _stage_configs(
    *,
    data_volume: str,
    artifacts_volume: str,
    remote_old_manifest: str,
    remote_new_manifest: str,
    remote_cached_old_manifest: str,
    remote_cached_new_manifest: str,
    query_cache_dir: str,
    query_shard_dir: str,
    output_dirs: Mapping[str, str],
    method: Mapping[str, Any],
    representation_options: Mapping[str, Any],
    new_count: int,
) -> dict[str, dict[str, Any]]:
    common_execution = {
        "provider": "modal",
        "target_volume": data_volume,
        "artifacts_volume": artifacts_volume,
    }
    common_data = {
        "root": "/data",
        "feature_glob": "cache/features/*.npz",
        "raw_glob": "cache/raw/*.jsonl",
        "manifests": {
            "pretrain": remote_cached_old_manifest,
            "new": remote_cached_new_manifest,
        },
    }
    exact_output_dir = output_dirs["exact_window_rank"]
    artifact_gate_dir = output_dirs["artifact_hygiene_ablation"]
    return {
        "cache_old_manifest_urls": {
            "execution": {
                "provider": "modal",
                "target_volume": data_volume,
                "artifacts_volume": artifacts_volume,
                "smoke_samples": int(method.get("cache_smoke_samples", 8)),
                "progress_every": int(method.get("cache_progress_every", 1000)),
                "workers": int(method.get("cache_workers", 16)),
                "max_pending": int(method.get("cache_max_pending", 128)),
                "download_timeout_seconds": float(method.get("cache_download_timeout_seconds", 60.0)),
                "fail_if_incomplete": True,
            },
            "target": {
                "root": "/data",
                "source_manifest": remote_old_manifest,
                "cached_manifest": remote_cached_old_manifest,
                "feature_dir": "cache/features",
                "raw_dir": "cache/raw",
            },
            "selection": {"strategy": "missing_cache"},
            "artifacts": {"output_dir": f"{output_dirs['manifest_url_cache'].rstrip('/')}/pretrain"},
        },
        "cache_new_manifest_urls": {
            "execution": {
                "provider": "modal",
                "target_volume": data_volume,
                "artifacts_volume": artifacts_volume,
                "smoke_samples": int(method.get("cache_smoke_samples", 8)),
                "progress_every": int(method.get("cache_progress_every", 1000)),
                "workers": int(method.get("cache_workers", 16)),
                "max_pending": int(method.get("cache_max_pending", 128)),
                "download_timeout_seconds": float(method.get("cache_download_timeout_seconds", 60.0)),
                "fail_if_incomplete": True,
            },
            "target": {
                "root": "/data",
                "source_manifest": remote_new_manifest,
                "cached_manifest": remote_cached_new_manifest,
                "feature_dir": "cache/features",
                "raw_dir": "cache/raw",
            },
            "selection": {"strategy": "missing_cache"},
            "artifacts": {"output_dir": f"{output_dirs['manifest_url_cache'].rstrip('/')}/new"},
        },
        "build_full_support_window_shards": {
            "execution": {
                **common_execution,
                "timeout_seconds": int(method.get("window_shard_timeout_seconds", 43200)),
                "smoke_max_clips_per_split": int(method.get("smoke_max_clips_per_split", 128)),
            },
            "data": common_data,
            "shards": {
                "output_dir": output_dirs["window_shards"],
                "clip_splits": ["pretrain", "new"],
                "representations": ["window_mean_std_pool"],
                "sample_rate": float(method.get("sample_rate", 30.0)),
                "raw_shape_max_samples": int(method.get("raw_shape_max_samples", 5400)),
                "shard_size": int(method.get("window_shard_size", 4096)),
                "workers": int(method.get("window_shard_workers", 16)),
                "include_imu_samples": False,
                "imu_max_samples": int(method.get("raw_shape_max_samples", 5400)),
                "compress": False,
                "progress_every_shards": 1,
            },
        },
        "active_exact_window_old_novelty_rank": {
            "execution": {
                **common_execution,
                "timeout_seconds": int(method.get("old_novelty_ranking_timeout_seconds", 7200)),
                "smoke_query_samples": int(method.get("smoke_query_samples", 64)),
                "smoke_window_support_samples": int(method.get("smoke_window_support_samples", 512)),
            },
            "data": common_data,
            "embeddings": {"cache_dir": query_cache_dir},
            "ranking": {
                "selector_mode": "old_novelty_only",
                "support_split": "pretrain",
                "query_split": "new",
                "left_representation": "window_mean_std_pool",
                "right_representation": "window_mean_std_pool",
                "old_novelty_k": int(method.get("old_novelty_k", 10)),
                "quality_threshold": float(method.get("quality_threshold", 0.85)),
                "max_stationary_fraction": float(method.get("max_stationary_fraction", 0.90)),
                "max_abs_value": float(method.get("max_abs_value", 60.0)),
                "cluster_similarity_threshold": float(method.get("cluster_similarity_threshold", 0.995)),
                "right_support_shard_dir": output_dirs["window_shards"],
                "max_query_clips": int(method.get("max_query_clips", DEFAULT_MAX_QUERY_CLIPS)),
                "fail_if_query_exceeds_max": True,
                "top_k_values": [int(value) for value in method.get("top_k_values", DEFAULT_TOP_K_VALUES)],
            },
            "quality": {
                "sample_rate": float(method.get("sample_rate", 30.0)),
                "max_samples_per_clip": int(method.get("raw_shape_max_samples", 5400)),
            },
            "artifacts": {"output_dir": output_dirs["exact_window_old_novelty_rank"]},
        },
        "active_embedding_precompute_ts2vec_new": {
            "execution": {
                **common_execution,
                "timeout_seconds": int(method.get("query_embedding_timeout_seconds", 3600)),
                "smoke_max_clips_per_split": int(method.get("smoke_max_clips_per_split", 8)),
            },
            "data": {
                **common_data,
                "manifests": {"new": remote_cached_new_manifest},
            },
            "embeddings": {"cache_dir": query_cache_dir},
            "precompute": {
                "clip_splits": ["new"],
                "max_clips_per_split": int(method.get("max_query_clips", DEFAULT_MAX_QUERY_CLIPS)),
                "fail_if_split_exceeds_max": True,
                "representations": ["ts2vec", "window_mean_std_pool"],
                "sample_rate": float(method.get("sample_rate", 30.0)),
                "raw_shape_max_samples": int(method.get("raw_shape_max_samples", 5400)),
                "shard_size": int(method.get("query_embedding_shard_size", 1024)),
                "workers": int(method.get("query_embedding_workers", 1)),
                "representation_options": representation_options,
            },
        },
        "active_exact_window_blend_rank": {
            "execution": {
                **common_execution,
                "timeout_seconds": int(method.get("ranking_timeout_seconds", 7200)),
                "smoke_query_samples": int(method.get("smoke_query_samples", 64)),
                "smoke_window_support_samples": int(method.get("smoke_window_support_samples", 512)),
            },
            "data": common_data,
            "embeddings": {"cache_dir": query_cache_dir},
            "ranking": {
                "support_split": "pretrain",
                "query_split": "new",
                "left_representation": "ts2vec",
                "right_representation": "window_mean_std_pool",
                "alpha": float(method.get("alpha", DEFAULT_ALPHA)),
                "old_novelty_k": int(method.get("old_novelty_k", 10)),
                "quality_threshold": float(method.get("quality_threshold", 0.85)),
                "max_stationary_fraction": float(method.get("max_stationary_fraction", 0.90)),
                "max_abs_value": float(method.get("max_abs_value", 60.0)),
                "cluster_similarity_threshold": float(method.get("cluster_similarity_threshold", 0.995)),
                "left_support_shard_dir": str(method.get("left_support_shard_dir", DEFAULT_LEFT_SUPPORT_SHARD_DIR)),
                "left_query_shard_dir": query_shard_dir,
                "min_left_support_clips": int(method.get("min_left_support_clips", DEFAULT_MIN_LEFT_SUPPORT_CLIPS)),
                "candidate_cache_dir": query_cache_dir,
                "right_support_shard_dir": output_dirs["window_shards"],
                "max_query_clips": int(method.get("max_query_clips", DEFAULT_MAX_QUERY_CLIPS)),
                "fail_if_query_exceeds_max": True,
                "top_k_values": [int(value) for value in method.get("top_k_values", DEFAULT_TOP_K_VALUES)],
                "representation_options": representation_options,
            },
            "quality": {
                "sample_rate": float(method.get("sample_rate", 30.0)),
                "max_samples_per_clip": int(method.get("raw_shape_max_samples", 5400)),
            },
            "artifacts": {"output_dir": exact_output_dir},
        },
        "active_spike_hygiene_ablation_artifact_gate": {
            "execution": {
                "provider": "modal",
                "data_volume": data_volume,
                "artifacts_volume": artifacts_volume,
                "timeout_seconds": int(method.get("artifact_gate_timeout_seconds", 900)),
                "smoke_max_rows": int(method.get("artifact_gate_smoke_max_rows", 128)),
            },
            "data": common_data,
            "ranking": {"query_split": "new"},
            "quality": {
                "sample_rate": float(method.get("sample_rate", 30.0)),
                "max_samples_per_clip": int(method.get("raw_shape_max_samples", 5400)),
            },
            "ablation": {
                "spike_rate_threshold": float(method.get("spike_rate_threshold", 0.025)),
                "top_k_values": [int(value) for value in method.get("top_k_values", DEFAULT_TOP_K_VALUES)],
                "generate_submissions": True,
            },
            "artifacts": {
                "exact_diagnostics_path": f"{exact_output_dir.rstrip('/')}/active_exact_window_blend_diagnostics_full.csv",
                "output_dir": artifact_gate_dir,
            },
        },
        "final_package_artifact_gate": {
            "source_artifacts": {
                "source_dir": artifact_gate_dir,
                "primary_submission": "spike_hygiene_ablation_artifact_gate_submission_full_new_worker_id.csv",
                "backup_worker_submission": "spike_hygiene_ablation_artifact_gate_submission_full_worker_id.csv",
                "diagnostics": "spike_hygiene_ablation_artifact_gate_diagnostics_full.csv",
                "selector_report": "spike_hygiene_ablation_report_full.json",
            },
            "artifacts": {"output_dir": "final_package"},
            "inputs": {
                "old_manifest": remote_cached_old_manifest,
                "new_manifest": remote_cached_new_manifest,
            },
            "validation": {"expected_count": int(new_count)},
            "method": {
                "name": "artifact-gate exact-window blend hidden-test runner",
                "primary_submission_id_column": "new_worker_id",
                "claim": (
                    "Partial-TS2Vec / exact full-window blended k-center selector with "
                    "artifact-aware trace rerank."
                ),
                "known_limitations": [
                    "The TS2Vec old-support view uses the frozen partial cache, not exact full-support TS2Vec.",
                    "The window-stat old-support view is rebuilt exactly for the provided old manifest.",
                    "This runner does not use hidden targets or labels.",
                ],
            },
        },
        "final_package_exact_window_old_novelty": {
            "source_artifacts": {
                "source_dir": output_dirs["exact_window_old_novelty_rank"],
                "primary_submission": "active_exact_window_blend_submission_full_new_worker_id.csv",
                "backup_worker_submission": "active_exact_window_blend_submission_full_worker_id.csv",
                "diagnostics": "active_exact_window_blend_diagnostics_full.csv",
                "selector_report": "active_exact_window_blend_report_full.json",
            },
            "artifacts": {"output_dir": "final_package_exact_window_old_novelty"},
            "inputs": {
                "old_manifest": remote_cached_old_manifest,
                "new_manifest": remote_cached_new_manifest,
            },
            "validation": {"expected_count": int(new_count)},
            "method": {
                "name": "exact-window old-novelty baseline",
                "primary_submission_id_column": "new_worker_id",
                "claim": (
                    "Quality-gated old-corpus novelty ranking using exact full-window-stat "
                    "support rebuilt from the supplied old manifest."
                ),
                "known_limitations": [
                    "This baseline does not use TS2Vec.",
                    "It is included as a cold-runnable provenance baseline for held-out evaluation.",
                    "It ranks by exact full-window old-corpus novelty rather than learned temporal embeddings.",
                ],
            },
        },
    }


def _write_configs(config_dir: Path, configs: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    names = {
        "cache_old_manifest_urls": "cache_old_manifest_urls.json",
        "cache_new_manifest_urls": "cache_new_manifest_urls.json",
        "build_full_support_window_shards": "build_full_support_window_shards.json",
        "active_exact_window_old_novelty_rank": "active_exact_window_old_novelty_rank.json",
        "active_embedding_precompute_ts2vec_new": "active_embedding_precompute_ts2vec_new.json",
        "active_exact_window_blend_rank": "active_exact_window_blend_rank.json",
        "active_spike_hygiene_ablation_artifact_gate": "active_spike_hygiene_ablation_artifact_gate.json",
        "final_package_artifact_gate": "final_package_artifact_gate.json",
        "final_package_exact_window_old_novelty": "final_package_exact_window_old_novelty.json",
    }
    written: dict[str, str] = {}
    for key, filename in names.items():
        path = config_dir / filename
        path.write_text(json.dumps(configs[key], indent=2, sort_keys=True), encoding="utf-8")
        written[key] = str(path)
    return written


def _run_plan(
    *,
    run_id: str,
    old_manifest: Path,
    new_manifest: Path,
    old_count: int,
    new_count: int,
    run_dir: Path,
    data_volume: str,
    artifacts_volume: str,
    remote_manifest_dir: str,
    remote_artifact_dir: str,
    query_shard_dir: str,
    output_dirs: Mapping[str, str],
    config_paths: Mapping[str, str],
) -> dict[str, Any]:
    artifact_gate_dir = output_dirs["artifact_hygiene_ablation"]
    old_novelty_dir = output_dirs["exact_window_old_novelty_rank"]
    return {
        "status": "prepared",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "old_manifest": str(old_manifest),
        "new_manifest": str(new_manifest),
        "old_manifest_count": int(old_count),
        "new_manifest_count": int(new_count),
        "data_volume": data_volume,
        "artifacts_volume": artifacts_volume,
        "remote_manifest_dir": remote_manifest_dir,
        "remote_artifact_dir": remote_artifact_dir,
        "query_shard_dir": query_shard_dir,
        "source_artifact_dir": artifact_gate_dir,
        "source_artifact_volume_path": _artifacts_volume_path(artifact_gate_dir),
        "old_novelty_source_artifact_dir": old_novelty_dir,
        "old_novelty_source_artifact_volume_path": _artifacts_volume_path(old_novelty_dir),
        "config_paths": dict(config_paths),
        "final_outputs": {
            "primary_submission": "final_package/ranked_new_clips.csv",
            "primary_new_worker_submission": "final_package/ranked_new_clips_new_worker_id.csv",
            "backup_worker_submission": "final_package/ranked_new_clips_worker_id.csv",
            "diagnostics": "final_package/diagnostics.csv",
            "selector_report": "final_package/selector_report.json",
            "exact_window_old_novelty_primary": "final_package_exact_window_old_novelty/ranked_new_clips.csv",
            "exact_window_old_novelty_new_worker": "final_package_exact_window_old_novelty/ranked_new_clips_new_worker_id.csv",
            "exact_window_old_novelty_worker": "final_package_exact_window_old_novelty/ranked_new_clips_worker_id.csv",
        },
    }


def _commands_text(plan: Mapping[str, Any]) -> str:
    data_volume = str(plan["data_volume"])
    artifacts_volume = str(plan["artifacts_volume"])
    remote_manifest_dir = str(plan["remote_manifest_dir"]).rstrip("/")
    source_artifact_volume_path = str(plan["source_artifact_volume_path"])
    old_novelty_volume_path = str(plan["old_novelty_source_artifact_volume_path"])
    return f"""#!/usr/bin/env bash
set -euo pipefail

# Run this file from the prepared hidden-test run directory.
# It does not train a model; it caches raw/features from the provided
# manifests, rebuilds exact window-stat support, embeds the new clips with the
# frozen TS2Vec checkpoint, ranks, applies the artifact gate, and packages CSVs.

RUN_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
REPO_ROOT="${{REPO_ROOT:-$(git -C "$RUN_DIR" rev-parse --show-toplevel 2>/dev/null || pwd)}}"
cd "$REPO_ROOT"

export MV_DATA_VOLUME="{data_volume}"
export MV_ARTIFACTS_VOLUME="{artifacts_volume}"

modal volume put "$MV_DATA_VOLUME" "$RUN_DIR/manifests/pretrain_urls.txt" {remote_manifest_dir}/pretrain_urls.txt --force
modal volume put "$MV_DATA_VOLUME" "$RUN_DIR/manifests/new_urls.txt" {remote_manifest_dir}/new_urls.txt --force

modal run modal_cache_manifest_urls.py --config-path "$RUN_DIR/configs/cache_old_manifest_urls.json" --run-full
modal run modal_cache_manifest_urls.py --config-path "$RUN_DIR/configs/cache_new_manifest_urls.json" --run-full

modal run modal_build_full_support_shards.py --config-path "$RUN_DIR/configs/build_full_support_window_shards.json" --run-full --wait-full
modal run modal_active_exact_window_blend_rank.py --config-path "$RUN_DIR/configs/active_exact_window_old_novelty_rank.json" --run-full --skip-smoke --wait-full
modal run modal_active_embedding_precompute.py --config-path "$RUN_DIR/configs/active_embedding_precompute_ts2vec_new.json" --run-full --skip-smoke --wait-full

modal run modal_active_exact_window_blend_rank.py --config-path "$RUN_DIR/configs/active_exact_window_blend_rank.json" --run-full --skip-smoke --wait-full
modal run modal_active_spike_hygiene_ablation.py --config-path "$RUN_DIR/configs/active_spike_hygiene_ablation_artifact_gate.json"

rm -rf "$RUN_DIR/source_artifacts/exact_window_old_novelty"
mkdir -p "$RUN_DIR/source_artifacts/exact_window_old_novelty"
modal volume get {artifacts_volume} {old_novelty_volume_path}/active_exact_window_blend_submission_full_new_worker_id.csv "$RUN_DIR/source_artifacts/exact_window_old_novelty/active_exact_window_blend_submission_full_new_worker_id.csv" --force
modal volume get {artifacts_volume} {old_novelty_volume_path}/active_exact_window_blend_submission_full_worker_id.csv "$RUN_DIR/source_artifacts/exact_window_old_novelty/active_exact_window_blend_submission_full_worker_id.csv" --force
modal volume get {artifacts_volume} {old_novelty_volume_path}/active_exact_window_blend_diagnostics_full.csv "$RUN_DIR/source_artifacts/exact_window_old_novelty/active_exact_window_blend_diagnostics_full.csv" --force
modal volume get {artifacts_volume} {old_novelty_volume_path}/active_exact_window_blend_report_full.json "$RUN_DIR/source_artifacts/exact_window_old_novelty/active_exact_window_blend_report_full.json" --force

rm -rf "$RUN_DIR/source_artifacts/artifact_hygiene_ablation"
mkdir -p "$RUN_DIR/source_artifacts/artifact_hygiene_ablation"
modal volume get {artifacts_volume} {source_artifact_volume_path}/spike_hygiene_ablation_artifact_gate_submission_full_new_worker_id.csv "$RUN_DIR/source_artifacts/artifact_hygiene_ablation/spike_hygiene_ablation_artifact_gate_submission_full_new_worker_id.csv" --force
modal volume get {artifacts_volume} {source_artifact_volume_path}/spike_hygiene_ablation_artifact_gate_submission_full_worker_id.csv "$RUN_DIR/source_artifacts/artifact_hygiene_ablation/spike_hygiene_ablation_artifact_gate_submission_full_worker_id.csv" --force
modal volume get {artifacts_volume} {source_artifact_volume_path}/spike_hygiene_ablation_artifact_gate_diagnostics_full.csv "$RUN_DIR/source_artifacts/artifact_hygiene_ablation/spike_hygiene_ablation_artifact_gate_diagnostics_full.csv" --force
modal volume get {artifacts_volume} {source_artifact_volume_path}/spike_hygiene_ablation_report_full.json "$RUN_DIR/source_artifacts/artifact_hygiene_ablation/spike_hygiene_ablation_report_full.json" --force
modal volume get {artifacts_volume} {source_artifact_volume_path}/spike_hygiene_ablation_report_full.md "$RUN_DIR/source_artifacts/artifact_hygiene_ablation/spike_hygiene_ablation_report_full.md" --force

python3 -m marginal_value.active.run_final \\
  --config-path "$RUN_DIR/configs/final_package_artifact_gate.json" \\
  --source-dir "$RUN_DIR/source_artifacts/artifact_hygiene_ablation" \\
  --output-dir "$RUN_DIR/final_package" \\
  --expected-count {int(plan["new_manifest_count"])}

python3 -m marginal_value.active.run_final \\
  --config-path "$RUN_DIR/configs/final_package_exact_window_old_novelty.json" \\
  --source-dir "$RUN_DIR/source_artifacts/exact_window_old_novelty" \\
  --output-dir "$RUN_DIR/final_package_exact_window_old_novelty" \\
  --expected-count {int(plan["new_manifest_count"])}
"""


def _readme_text(plan: Mapping[str, Any]) -> str:
    return f"""# Hidden-Test IMU Ranking Run Package

This package prepares the current scientifically strongest non-training path:
artifact-gated exact-window blend ranking.

It binds the provided old/new manifests into Modal stage configs and records the
expected artifacts. Running `commands.sh` ranks the new manifest relative to the
old manifest. The method does not use hidden targets, labels, candidate roles,
or evaluator feedback.

## What It Does

1. Uploads the copied manifests to `{plan["data_volume"]}`.
2. Caches raw JSONL and feature NPZ files for the supplied old/new manifest URLs.
3. Rebuilds exact `window_mean_std_pool` support shards for the cached old and
   new manifests.
4. Emits an exact-window old-novelty baseline that does not depend on TS2Vec.
5. Computes frozen-checkpoint TS2Vec embeddings for the new clips only.
6. Runs the partial-TS2Vec / exact-window blended k-center selector.
7. Applies the artifact-aware hygiene rerank.
8. Packages submission CSVs and diagnostics under `final_package/` and
   `final_package_exact_window_old_novelty/`.

## Inputs

- Old manifest rows: `{plan["old_manifest_count"]}`
- New manifest rows: `{plan["new_manifest_count"]}`
- Remote manifest directory: `{plan["remote_manifest_dir"]}`
- Remote artifact directory: `{plan["remote_artifact_dir"]}`

## Important Limitations

- The TS2Vec old-support view is still the frozen partial old-support cache.
- The window-stat old-support view is rebuilt exactly for the supplied old
  manifest.
- This is a reproducible hidden-test runner, not TS2Vec retraining and not a
  claim of exact full-support TS2Vec.

## Run

```bash
./commands.sh
```

Primary output after the command finishes:

`{plan["final_outputs"]["primary_submission"]}`

Cold-runnable exact-window baseline:

`{plan["final_outputs"]["exact_window_old_novelty_primary"]}`
"""


def _expected_query_shard_dir(
    urls: Sequence[str],
    *,
    cache_dir: str,
    representation_options: Mapping[str, Any],
) -> Path:
    clips = [
        ClipRecord(
            sample_id=hash_manifest_url(url),
            split="new",
            url=url,
            source_group_id="",
            worker_id="",
            raw_path=Path(),
            feature_path=Path(),
        )
        for url in urls
    ]
    ordered_clips = sorted(clips, key=lambda clip: clip.sample_id)
    metadata = _cache_metadata(
        ordered_clips,
        representations=["ts2vec", "window_mean_std_pool"],
        sample_rate=30.0,
        raw_shape_max_samples=5400,
        representation_options=representation_options,
    )
    return _shard_dir(cache_dir, metadata)


def _ts2vec_representation_options(method: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ts2vec_checkpoint_path": str(method.get("ts2vec_checkpoint_path", DEFAULT_TS2VEC_CHECKPOINT_PATH)),
        "ts2vec_device": str(method.get("ts2vec_device", "cuda")),
        "ts2vec_batch_size": int(method.get("ts2vec_batch_size", 128)),
    }


def _read_valid_manifest(path: Path, *, label: str) -> list[str]:
    urls = read_manifest_urls(path)
    if not urls:
        raise ValueError(f"{label} has no URLs: {path}")
    counts = Counter(urls)
    duplicates = sorted(url for url, count in counts.items() if count > 1)
    if duplicates:
        preview = ", ".join(duplicates[:3])
        raise ValueError(f"{label} has duplicate URLs: {preview}")
    blank_count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if not line.strip())
    if blank_count:
        raise ValueError(f"{label} contains blank lines; remove them before preparing a hidden-test run.")
    return urls


def _validate_manifest_disjoint(old_urls: Sequence[str], new_urls: Sequence[str]) -> None:
    old_ids = {hash_manifest_url(url) for url in old_urls}
    new_ids = {hash_manifest_url(url) for url in new_urls}
    overlap = old_ids & new_ids
    if overlap:
        preview = ", ".join(sorted(overlap)[:5])
        raise ValueError(f"old/new manifests overlap on {len(overlap)} sample IDs: {preview}")


def _copy_manifest(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Hidden-test config requires mapping '{key}'.")
    return value


def _artifacts_volume_path(path: str) -> str:
    normalized = path.strip()
    if normalized.startswith("/artifacts/"):
        return normalized[len("/artifacts/") :]
    if normalized == "/artifacts":
        return "."
    return normalized.lstrip("/")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a hidden-test run package for the final IMU selector.")
    parser.add_argument("--config-path")
    parser.add_argument("--old-manifest")
    parser.add_argument("--new-manifest")
    parser.add_argument("--run-dir")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--remote-manifest-dir")
    parser.add_argument("--remote-artifact-dir")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    if args.validate_only:
        if not args.run_dir:
            raise ValueError("--run-dir is required with --validate-only.")
        return validate_hidden_test_run_package(args.run_dir)
    if args.config_path:
        config = load_hidden_test_config(args.config_path)
    else:
        if not (args.old_manifest and args.new_manifest and args.run_dir):
            raise ValueError("--old-manifest, --new-manifest, and --run-dir are required without --config-path.")
        config = {
            "inputs": {
                "old_manifest": args.old_manifest,
                "new_manifest": args.new_manifest,
            },
            "artifacts": {
                "run_dir": args.run_dir,
                "run_id": args.run_id,
            },
        }
    if args.remote_manifest_dir:
        config.setdefault("modal", {})["remote_manifest_dir"] = args.remote_manifest_dir
    if args.remote_artifact_dir:
        config.setdefault("modal", {})["remote_artifact_dir"] = args.remote_artifact_dir
    return prepare_hidden_test_run(config)


if __name__ == "__main__":
    main()
