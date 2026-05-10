from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_BUCKET_PREFIX = "gs://active-learning-v2-802636843791/offline-active-benchmark"


def main() -> None:
    args = _parse_args()
    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _apply_runtime_overrides(
        config,
        machine_type=args.machine_type_override,
        selection_seeds=args.selection_seeds,
    )
    run_id = args.run_id or _default_run_id(config_path)
    zone = str(config["execution"]["zone"])
    vm_name = args.vm_name or _default_vm_name(run_id)
    local_output = repo_root / str(config["artifacts"]["local_output_dir"])
    launch_dir = local_output / "launch" / run_id
    launch_dir.mkdir(parents=True, exist_ok=True)
    gcs_prefix = f"{args.bucket_prefix.rstrip('/')}/{run_id}"

    bundle_path = launch_dir / "repo_bundle.tgz"
    startup_path = launch_dir / "startup.sh"
    manifest_path = repo_root / str(config["data"]["manifest"])
    checkpoint_path = Path(str(config["ts2vec"]["checkpoint_path"]))

    _assert_inputs(repo_root=repo_root, manifest_path=manifest_path, checkpoint_path=checkpoint_path)
    _write_repo_bundle(repo_root, bundle_path)
    startup_path.write_text(
        _startup_script(
            config=config,
            run_id=run_id,
            gcs_prefix=gcs_prefix,
            bundle_name=bundle_path.name,
            manifest_name=manifest_path.name,
            checkpoint_name=checkpoint_path.name,
        ),
        encoding="utf-8",
    )

    _run(["gsutil", "cp", str(bundle_path), f"{gcs_prefix}/inputs/{bundle_path.name}"], cwd=repo_root)
    _run(["gsutil", "cp", str(manifest_path), f"{gcs_prefix}/inputs/{manifest_path.name}"], cwd=repo_root)
    _run(["gsutil", "cp", str(checkpoint_path), f"{gcs_prefix}/inputs/{checkpoint_path.name}"], cwd=repo_root)

    create_cmd = [
        "gcloud",
        "compute",
        "instances",
        "create",
        vm_name,
        "--zone",
        zone,
        "--machine-type",
        str(config["execution"]["machine_type"]),
        "--image-family",
        args.image_family,
        "--image-project",
        args.image_project,
        "--boot-disk-size",
        args.boot_disk_size,
        "--scopes",
        "https://www.googleapis.com/auth/cloud-platform",
        "--metadata",
        f"run-id={run_id},gcs-prefix={gcs_prefix}",
        f"--metadata-from-file=startup-script={startup_path}",
    ]
    if args.preemptible:
        create_cmd.append("--preemptible")
    if args.no_launch:
        print(json.dumps({"event": "dry_run", "run_id": run_id, "vm_name": vm_name, "gcs_prefix": gcs_prefix}, sort_keys=True))
        print(" ".join(shlex.quote(part) for part in create_cmd))
        return

    _run(create_cmd, cwd=repo_root)
    print(json.dumps({"event": "vm_created", "run_id": run_id, "vm_name": vm_name, "zone": zone, "gcs_prefix": gcs_prefix}), flush=True)

    if args.detach:
        return

    final_state = _poll_status(
        gcs_prefix=gcs_prefix,
        run_id=run_id,
        timeout_seconds=int(config["execution"]["max_runtime_minutes"]) * 60,
        poll_seconds=args.poll_seconds,
    )
    download_dir = local_output / "gcp_downloads" / run_id
    download_dir.mkdir(parents=True, exist_ok=True)
    _copy_results(gcs_prefix=gcs_prefix, download_dir=download_dir)

    if bool(config["execution"].get("delete_vm_after_copy", True)):
        _run(["gcloud", "compute", "instances", "delete", vm_name, "--zone", zone, "--quiet"], cwd=repo_root)
        print(json.dumps({"event": "vm_deleted", "vm_name": vm_name, "zone": zone}), flush=True)

    if final_state != "success":
        raise SystemExit(f"GCP coverage benchmark ended with state={final_state}. Results copied to {download_dir}")
    print(json.dumps({"event": "done", "state": final_state, "download_dir": str(download_dir)}, sort_keys=True), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a CPU-only blind-target coverage benchmark on one GCP VM.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--bucket-prefix", default=DEFAULT_BUCKET_PREFIX)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--vm-name", default="")
    parser.add_argument("--image-family", default="debian-12")
    parser.add_argument("--image-project", default="debian-cloud")
    parser.add_argument("--boot-disk-size", default="50GB")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--machine-type-override", default="", help="Override config execution.machine_type.")
    parser.add_argument(
        "--selection-seeds",
        default="",
        help="Comma-separated override for config data.selection_seeds, useful for seed-parallel GCP runs.",
    )
    parser.add_argument("--preemptible", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--no-launch", action="store_true", help="Stage inputs and print the VM create command without launching.")
    return parser.parse_args()


def _apply_runtime_overrides(config: dict[str, Any], *, machine_type: str, selection_seeds: str) -> None:
    if machine_type:
        config["execution"]["machine_type"] = machine_type
    if selection_seeds:
        seeds = [int(seed.strip()) for seed in selection_seeds.split(",") if seed.strip()]
        if not seeds:
            raise ValueError("--selection-seeds must contain at least one integer seed.")
        config["data"]["selection_seeds"] = seeds


def _assert_inputs(*, repo_root: Path, manifest_path: Path, checkpoint_path: Path) -> None:
    for path in (manifest_path, checkpoint_path):
        if not path.exists():
            raise FileNotFoundError(path)
    for relative in (
        "pyproject.toml",
        "marginal_value",
        "scripts/offline_coverage_benchmark_from_urls.py",
        "scripts/downstream_utility_smoke_from_urls.py",
    ):
        if not (repo_root / relative).exists():
            raise FileNotFoundError(repo_root / relative)


def _write_repo_bundle(repo_root: Path, bundle_path: Path) -> None:
    include_paths = [
        "pyproject.toml",
        "marginal_value",
        "scripts/offline_active_benchmark_from_urls.py",
        "scripts/offline_coverage_benchmark_from_urls.py",
        "scripts/downstream_utility_smoke_from_urls.py",
        "scripts/summarize_coverage_benchmark_reports.py",
    ]
    with tarfile.open(bundle_path, "w:gz") as archive:
        for relative in include_paths:
            archive.add(repo_root / relative, arcname=relative)


def _startup_script(
    *,
    config: dict[str, Any],
    run_id: str,
    gcs_prefix: str,
    bundle_name: str,
    manifest_name: str,
    checkpoint_name: str,
) -> str:
    runner = str(config.get("execution", {}).get("runner", "coverage"))
    if runner == "downstream_utility_smoke":
        return _downstream_utility_smoke_startup_script(
            config=config,
            run_id=run_id,
            gcs_prefix=gcs_prefix,
            bundle_name=bundle_name,
            manifest_name=manifest_name,
            checkpoint_name=checkpoint_name,
        )
    if runner != "coverage":
        raise ValueError(f"Unsupported GCP benchmark runner: {runner}")

    data = config["data"]
    benchmark = config["benchmark"]
    reporting = config["reporting"]
    seeds = " ".join(str(seed) for seed in data["selection_seeds"])
    policies = ",".join(str(policy) for policy in benchmark["policies"])
    representations = ",".join(str(item) for item in benchmark["representations"])
    eval_views = ",".join(str(item) for item in benchmark["eval_views"])
    primary_eval_views = ",".join(str(item) for item in benchmark["primary_eval_views"])
    budgets = ",".join(str(item) for item in benchmark["budgets"])
    eval_view_families = ",".join(f"{key}:{value}" for key, value in benchmark["eval_view_families"].items())
    max_artifact_score = benchmark.get("max_artifact_score")
    artifact_arg = "none" if max_artifact_score is None else str(max_artifact_score)
    target_families = int(benchmark.get("target_families_per_episode", 1))
    oracle_exact_combination_limit = int(benchmark.get("oracle_exact_combination_limit", 100_000))
    pool_specs = _candidate_group_sweep_specs(benchmark)
    if len(pool_specs) == 1 and not pool_specs[0][0]:
        _label, candidate_groups, target_candidate_groups = pool_specs[0]
        pool_loop_header = (
            f"CANDIDATE_GROUPS={candidate_groups}\n"
            f"TARGET_CANDIDATE_GROUPS={target_candidate_groups}\n"
            f"for SEED in {seeds}; do"
        )
        pool_loop_footer = "done"
        output_dir = '"$OUT/seed_$SEED"'
    else:
        pool_spec_words = " ".join(f'"{label}:{candidate}:{target}"' for label, candidate, target in pool_specs)
        pool_loop_header = (
            f"POOL_SPECS=({pool_spec_words})\n"
            'for POOL_SPEC in "${POOL_SPECS[@]}"; do\n'
            '  IFS=: read -r POOL_LABEL CANDIDATE_GROUPS TARGET_CANDIDATE_GROUPS <<< "$POOL_SPEC"\n'
            f"  for SEED in {seeds}; do"
        )
        pool_loop_footer = "  done\ndone"
        output_dir = '"$OUT/$POOL_LABEL/seed_$SEED"'
    downstream_args = _downstream_args(config)

    return f"""#!/usr/bin/env bash
set -euo pipefail

RUN_ID={shlex.quote(run_id)}
GCS_PREFIX={shlex.quote(gcs_prefix)}
WORK=/opt/blind-target-coverage
OUT="$WORK/out/$RUN_ID"
STATUS="$WORK/status"
LOG="$OUT/run_all.log"
mkdir -p "$WORK" "$OUT" "$STATUS"

finish_failure() {{
  code=$?
  set +e
  echo '{{"run_id":"'$RUN_ID'","state":"failure","exit_code":'$code',"ts":"'"$(date -Iseconds)"'"}}' > "$STATUS/failure.json"
  if [ -f "$LOG" ]; then cp "$LOG" "$STATUS/run_all.log"; fi
  tar -czf "$WORK/partial_${{RUN_ID}}.tgz" -C "$WORK" out status
  gsutil cp "$STATUS/failure.json" "$GCS_PREFIX/status/failure.json" >/dev/null 2>&1 || true
  if [ -f "$STATUS/run_all.log" ]; then gsutil cp "$STATUS/run_all.log" "$GCS_PREFIX/status/run_all.log" >/dev/null 2>&1 || true; fi
  gsutil cp "$WORK/partial_${{RUN_ID}}.tgz" "$GCS_PREFIX/partial_${{RUN_ID}}.tgz" >/dev/null 2>&1 || true
  exit "$code"
}}
trap finish_failure ERR

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-pip curl ca-certificates gnupg tar build-essential libglib2.0-0

gsutil cp "$GCS_PREFIX/inputs/{bundle_name}" "$WORK/repo_bundle.tgz"
gsutil cp "$GCS_PREFIX/inputs/{manifest_name}" "$WORK/manifest.txt"
gsutil cp "$GCS_PREFIX/inputs/{checkpoint_name}" "$WORK/ts2vec_best.pt"

cd "$WORK"
tar -xzf repo_bundle.tgz
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch

python - <<'PY'
import importlib
import torch
import marginal_value
import scripts.offline_coverage_benchmark_from_urls
import scripts.summarize_coverage_benchmark_reports
print({{"event":"import_smoke_ok","torch": torch.__version__}})
PY

echo '{{"event":"policy_list","policy_count":{len(benchmark["policies"])},"budgets":{json.dumps(benchmark["budgets"])},"representations":{json.dumps(benchmark["representations"])},"no_gpu":true,"no_ts2vec_retraining":true,"pool_spec_count":{len(pool_specs)},"ts":"'"$(date -u +%Y-%m-%dT%H:%M:%SZ)"'"}}' | tee -a "$LOG"
{pool_loop_header}
  echo '{{"event":"seed_start","seed":'$SEED',"ts":"'"$(date -u +%Y-%m-%dT%H:%M:%SZ)"'"}}' | tee -a "$LOG"
  python scripts/offline_coverage_benchmark_from_urls.py \\
    --manifest "$WORK/manifest.txt" \\
    --output-dir {output_dir} \\
    --max-rows {int(data["max_rows_per_seed"])} \\
    --max-groups {int(data["max_groups"])} \\
    --clips-per-group {int(data["clips_per_group"])} \\
    --sampling-mode {shlex.quote(str(data["sampling_mode"]))} \\
    --selection-seed "$SEED" \\
    --max-samples {int(data["max_samples_per_clip"])} \\
    --download-workers 16 \\
    --folds {int(benchmark["folds"])} \\
    --episode-strategy {shlex.quote(str(benchmark["episode_strategy"]))} \\
    --episode-representation {shlex.quote(str(benchmark["episode_representation"]))} \\
    --source-family-count {int(benchmark["source_family_count"])} \\
    --candidate-groups-per-episode "$CANDIDATE_GROUPS" \\
    --target-groups-per-episode {int(benchmark["target_groups_per_episode"])} \\
    --target-candidate-groups-per-episode "$TARGET_CANDIDATE_GROUPS" \\
    --target-families-per-episode {target_families} \\
    --max-support-groups {int(benchmark["max_support_groups"])} \\
    --budgets {shlex.quote(budgets)} \\
    --quality-threshold {float(benchmark["quality_threshold"])} \\
    --max-stationary-fraction {float(benchmark["max_stationary_fraction"])} \\
    --max-abs-value {float(benchmark["max_abs_value"])} \\
    --max-artifact-score {shlex.quote(artifact_arg)} \\
    --policies {shlex.quote(policies)} \\
    --representations {shlex.quote(representations)} \\
    --eval-views {shlex.quote(eval_views)} \\
    --primary-eval-views {shlex.quote(primary_eval_views)} \\
    --eval-view-families {shlex.quote(eval_view_families)} \\
    --window-view {shlex.quote(str(benchmark["window_view"]))} \\
    --ts2vec-view {shlex.quote(str(benchmark["ts2vec_view"]))} \\
    --ts2vec-checkpoint "$WORK/ts2vec_best.pt" \\
    --ts2vec-device cpu \\
    --ts2vec-batch-size {int(config["ts2vec"]["batch_size"])} \\
    --blend-alpha {float(benchmark["blend_alpha"])} \\
    --distance-metric {shlex.quote(str(benchmark["distance_metric"]))} \\
    --oracle-exact-combination-limit {oracle_exact_combination_limit} \\
{downstream_args}
    --seed "$SEED" 2>&1 | tee -a "$LOG"
  echo '{{"event":"seed_done","seed":'$SEED',"ts":"'"$(date -u +%Y-%m-%dT%H:%M:%SZ)"'"}}' | tee -a "$LOG"
{pool_loop_footer}

mapfile -t REPORTS < <(find "$OUT" -path '*/blind_target_coverage_benchmark_report.json' | sort)
python scripts/summarize_coverage_benchmark_reports.py "${{REPORTS[@]}}" \\
  --output-json "$OUT/coverage_decision_report.json" \\
  --output-markdown "$OUT/coverage_decision_report.md" \\
  --baseline-policy {shlex.quote(str(reporting["baseline_policy"]))} \\
  --oracle-policy {shlex.quote(str(reporting["oracle_policy"]))} \\
  --bootstrap-replicates {int(reporting["bootstrap_replicates"])}

tar -czf "$WORK/results_${{RUN_ID}}.tgz" -C "$WORK/out" "$RUN_ID"
gsutil -m cp -r "$OUT" "$GCS_PREFIX/results/"
gsutil cp "$WORK/results_${{RUN_ID}}.tgz" "$GCS_PREFIX/results_${{RUN_ID}}.tgz"
echo '{{"run_id":"'$RUN_ID'","state":"success","exit_code":0,"ts":"'"$(date -Iseconds)"'"}}' > "$STATUS/success.json"
gsutil cp "$STATUS/success.json" "$GCS_PREFIX/status/success.json"
"""


def _downstream_utility_smoke_startup_script(
    *,
    config: dict[str, Any],
    run_id: str,
    gcs_prefix: str,
    bundle_name: str,
    manifest_name: str,
    checkpoint_name: str,
) -> str:
    data = config["data"]
    benchmark = config["benchmark"]
    downstream = config.get("downstream", {})
    downstream_utility = config.get("downstream_utility", downstream)
    seeds = " ".join(str(seed) for seed in data["selection_seeds"])
    policies = ",".join(str(policy) for policy in benchmark["policies"])
    representations = ",".join(str(item) for item in benchmark["representations"])
    primary_representations = ",".join(str(item) for item in benchmark["primary_representations"])
    downstream_representations = ",".join(
        str(item) for item in downstream_utility.get("representations", ["window", "raw_shape_stats"])
    )
    supervised_representations = ",".join(
        str(item) for item in downstream.get("representations", ["window", "raw_shape_stats"])
    )
    max_artifact_score = benchmark.get("max_artifact_score")
    artifact_arg = "none" if max_artifact_score is None else str(max_artifact_score)
    target_candidate_groups = int(
        benchmark.get("target_candidate_groups_per_episode", max(1, int(benchmark["candidate_groups_per_episode"]) // 2))
    )
    target_families = int(benchmark.get("target_families_per_episode", 1))
    oracle_candidate_arg = ""
    if benchmark.get("oracle_candidate_cap") is not None:
        oracle_candidate_arg = f"    --oracle-candidate-cap {int(benchmark['oracle_candidate_cap'])} \\\n"
    supervised_args = ""
    if isinstance(downstream, dict) and str(downstream.get("label_source", "none")) != "none":
        supervised_args = f"""    --supervised-downstream-label-source {shlex.quote(str(downstream["label_source"]))} \\
    --supervised-downstream-label-representation {shlex.quote(str(downstream.get("label_representation", "window")))} \\
    --supervised-downstream-source-family-count {int(downstream.get("source_family_count", 4))} \\
    --supervised-downstream-representations {shlex.quote(supervised_representations)} \\
    --supervised-downstream-baseline-policy {shlex.quote(str(downstream.get("baseline_policy", "old_novelty_ts2vec")))} \\
    --supervised-downstream-random-policy {shlex.quote(str(downstream.get("random_policy", "random_valid")))} \\
"""

    return f"""#!/usr/bin/env bash
set -euo pipefail

RUN_ID={shlex.quote(run_id)}
GCS_PREFIX={shlex.quote(gcs_prefix)}
WORK=/opt/offline-active-loop
OUT="$WORK/out/$RUN_ID"
STATUS="$WORK/status"
LOG="$OUT/run_all.log"
mkdir -p "$WORK" "$OUT" "$STATUS"

finish_failure() {{
  code=$?
  set +e
  echo '{{"run_id":"'$RUN_ID'","state":"failure","exit_code":'$code',"ts":"'"$(date -Iseconds)"'"}}' > "$STATUS/failure.json"
  if [ -f "$LOG" ]; then cp "$LOG" "$STATUS/run_all.log"; fi
  tar -czf "$WORK/partial_${{RUN_ID}}.tgz" -C "$WORK" out status
  gsutil cp "$STATUS/failure.json" "$GCS_PREFIX/status/failure.json" >/dev/null 2>&1 || true
  if [ -f "$STATUS/run_all.log" ]; then gsutil cp "$STATUS/run_all.log" "$GCS_PREFIX/status/run_all.log" >/dev/null 2>&1 || true; fi
  gsutil cp "$WORK/partial_${{RUN_ID}}.tgz" "$GCS_PREFIX/partial_${{RUN_ID}}.tgz" >/dev/null 2>&1 || true
  exit "$code"
}}
trap finish_failure ERR

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-pip curl ca-certificates gnupg tar build-essential libglib2.0-0

gsutil cp "$GCS_PREFIX/inputs/{bundle_name}" "$WORK/repo_bundle.tgz"
gsutil cp "$GCS_PREFIX/inputs/{manifest_name}" "$WORK/manifest.txt"
gsutil cp "$GCS_PREFIX/inputs/{checkpoint_name}" "$WORK/ts2vec_best.pt"

cd "$WORK"
tar -xzf repo_bundle.tgz
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch

python - <<'PY'
import torch
import marginal_value
import scripts.downstream_utility_smoke_from_urls
print({{"event":"import_smoke_ok","torch": torch.__version__}})
PY

echo '{{"event":"round_loop_policy_list","policy_count":{len(benchmark["policies"])},"representations":{json.dumps(benchmark["representations"])},"no_gpu":true,"no_ts2vec_retraining":true,"ts":"'"$(date -u +%Y-%m-%dT%H:%M:%SZ)"'"}}' | tee -a "$LOG"
for SEED in {seeds}; do
  echo '{{"event":"seed_start","seed":'$SEED',"ts":"'"$(date -u +%Y-%m-%dT%H:%M:%SZ)"'"}}' | tee -a "$LOG"
  python scripts/downstream_utility_smoke_from_urls.py \\
    --manifest "$WORK/manifest.txt" \\
    --output-dir "$OUT/seed_$SEED" \\
    --max-rows {int(data["max_rows_per_seed"])} \\
    --max-groups {int(data["max_groups"])} \\
    --clips-per-group {int(data["clips_per_group"])} \\
    --sampling-mode {shlex.quote(str(data["sampling_mode"]))} \\
    --selection-seed "$SEED" \\
    --max-samples {int(data["max_samples_per_clip"])} \\
    --download-workers 16 \\
    --folds {int(benchmark["folds"])} \\
    --episode-strategy {shlex.quote(str(benchmark["episode_strategy"]))} \\
    --episode-representation {shlex.quote(str(benchmark["episode_representation"]))} \\
    --source-family-count {int(benchmark["source_family_count"])} \\
    --candidate-groups-per-episode {int(benchmark["candidate_groups_per_episode"])} \\
    --target-groups-per-episode {int(benchmark["target_groups_per_episode"])} \\
    --target-candidate-groups-per-episode {target_candidate_groups} \\
    --target-families-per-episode {target_families} \\
    --max-support-groups {int(benchmark["max_support_groups"])} \\
    --rounds {int(benchmark["rounds"])} \\
    --batch-size {int(benchmark["batch_size"])} \\
    --quality-threshold {float(benchmark["quality_threshold"])} \\
    --max-stationary-fraction {float(benchmark["max_stationary_fraction"])} \\
    --max-abs-value {float(benchmark["max_abs_value"])} \\
    --max-artifact-score {shlex.quote(artifact_arg)} \\
    --policies {shlex.quote(policies)} \\
    --representations {shlex.quote(representations)} \\
    --primary-representations {shlex.quote(primary_representations)} \\
    --ts2vec-checkpoint "$WORK/ts2vec_best.pt" \\
    --ts2vec-device cpu \\
    --ts2vec-batch-size {int(config["ts2vec"]["batch_size"])} \\
    --blend-left-representation {shlex.quote(str(benchmark["blend_left_representation"]))} \\
    --blend-right-representation {shlex.quote(str(benchmark["blend_right_representation"]))} \\
    --blend-alpha {float(benchmark["blend_alpha"])} \\
{oracle_candidate_arg}    --oracle-exact-combination-limit {int(benchmark.get("oracle_exact_combination_limit", 100_000))} \\
    --downstream-representations {shlex.quote(downstream_representations)} \\
    --downstream-max-components {int(downstream_utility.get("max_components", 4))} \\
    --downstream-baseline-policy {shlex.quote(str(downstream_utility.get("baseline_policy", downstream.get("baseline_policy", "old_novelty_ts2vec"))))} \\
    --downstream-random-policy {shlex.quote(str(downstream_utility.get("random_policy", downstream.get("random_policy", "random_valid"))))} \\
{supervised_args}    --seed "$SEED" 2>&1 | tee -a "$LOG"
  echo '{{"event":"seed_done","seed":'$SEED',"ts":"'"$(date -u +%Y-%m-%dT%H:%M:%SZ)"'"}}' | tee -a "$LOG"
done

tar -czf "$WORK/results_${{RUN_ID}}.tgz" -C "$WORK/out" "$RUN_ID"
gsutil -m cp -r "$OUT" "$GCS_PREFIX/results/"
gsutil cp "$WORK/results_${{RUN_ID}}.tgz" "$GCS_PREFIX/results_${{RUN_ID}}.tgz"
echo '{{"run_id":"'$RUN_ID'","state":"success","exit_code":0,"ts":"'"$(date -Iseconds)"'"}}' > "$STATUS/success.json"
gsutil cp "$STATUS/success.json" "$GCS_PREFIX/status/success.json"
"""


def _candidate_group_sweep_specs(benchmark: dict[str, Any]) -> list[tuple[str, int, int]]:
    sweep = benchmark.get("candidate_group_sweep")
    if isinstance(sweep, list) and sweep:
        specs = []
        for item in sweep:
            if not isinstance(item, dict):
                raise ValueError("candidate_group_sweep entries must be objects.")
            label = str(item.get("label", "")).strip()
            if not label or any(char.isspace() or char == ":" for char in label):
                raise ValueError("candidate_group_sweep labels must be non-empty and contain no spaces or colons.")
            candidate_groups = int(item["candidate_groups_per_episode"])
            target_candidate_groups = int(
                item.get("target_candidate_groups_per_episode", max(1, candidate_groups // 2))
            )
            specs.append((label, candidate_groups, target_candidate_groups))
        return specs

    candidate_groups = int(benchmark["candidate_groups_per_episode"])
    target_candidate_groups = int(
        benchmark.get("target_candidate_groups_per_episode", max(1, candidate_groups // 2))
    )
    return [("", candidate_groups, target_candidate_groups)]


def _poll_status(*, gcs_prefix: str, run_id: str, timeout_seconds: int, poll_seconds: int) -> str:
    deadline = time.time() + timeout_seconds
    last_report = 0.0
    while time.time() < deadline:
        if _remote_status_exists(gcs_prefix, "success.json"):
            print(json.dumps({"event": "remote_success", "run_id": run_id}), flush=True)
            return "success"
        if _remote_status_exists(gcs_prefix, "failure.json"):
            print(json.dumps({"event": "remote_failure", "run_id": run_id}), flush=True)
            return "failure"
        now = time.time()
        if now - last_report >= max(1, poll_seconds):
            print(json.dumps({"event": "poll", "run_id": run_id, "elapsed_seconds": round(now - (deadline - timeout_seconds), 1)}), flush=True)
            last_report = now
        time.sleep(max(1, int(poll_seconds)))
    return "timeout"


def _copy_results(*, gcs_prefix: str, download_dir: Path) -> None:
    _run(["gcloud", "storage", "cp", "--recursive", f"{gcs_prefix}/status", str(download_dir)], cwd=Path.cwd(), check=False)
    _run(["gcloud", "storage", "cp", "--recursive", f"{gcs_prefix}/results", str(download_dir)], cwd=Path.cwd(), check=False)
    _run(["gcloud", "storage", "cp", f"{gcs_prefix}/results_*.tgz", str(download_dir)], cwd=Path.cwd(), check=False)
    _run(["gcloud", "storage", "cp", f"{gcs_prefix}/partial_*.tgz", str(download_dir)], cwd=Path.cwd(), check=False)


def _gsutil_stat(path: str) -> bool:
    return subprocess.run(["gsutil", "-q", "stat", path], check=False).returncode == 0


def _remote_status_exists(gcs_prefix: str, filename: str) -> bool:
    return _gsutil_stat(f"{gcs_prefix}/status/{filename}") or _gsutil_stat(f"{gcs_prefix}/status/status/{filename}")


def _downstream_args(config: dict[str, Any]) -> str:
    args: list[str] = []
    downstream = config.get("downstream", {})
    continuation = "\\"
    if isinstance(downstream, dict) and str(downstream.get("label_source", "none")) != "none":
        args.extend(
            [
                f"--downstream-supervised-label-source {shlex.quote(str(downstream['label_source']))}",
                f"--downstream-supervised-label-representation {shlex.quote(str(downstream.get('label_representation', 'window')))}",
                f"--downstream-supervised-source-family-count {int(downstream.get('source_family_count', 4))}",
                f"--downstream-supervised-representations {shlex.quote(','.join(str(item) for item in downstream.get('representations', ['window', 'raw_shape_stats'])))}",
                f"--downstream-supervised-models {shlex.quote(','.join(str(item) for item in downstream.get('models', ['nearest_centroid'])))}",
                f"--downstream-supervised-top-policy {shlex.quote(str(downstream.get('top_policy', 'ts2vec_kcenter_v1')))}",
                f"--downstream-supervised-baseline-policy {shlex.quote(str(downstream.get('baseline_policy', 'quality_stratified_random_v1')))}",
            ]
        )
    utility = config.get("downstream_utility", {})
    if isinstance(utility, dict) and str(utility.get("model", "none")) != "none":
        args.extend(
            [
                "--downstream-utility-enable",
                f"--downstream-utility-representations {shlex.quote(','.join(str(item) for item in utility.get('representations', ['window', 'raw_shape_stats'])))}",
                f"--downstream-utility-max-components {int(utility.get('max_components', 4))}",
                f"--downstream-utility-top-policy {shlex.quote(str(utility.get('top_policy', 'ts2vec_kcenter_v1')))}",
                f"--downstream-utility-baseline-policy {shlex.quote(str(utility.get('baseline_policy', 'quality_stratified_random_v1')))}",
            ]
        )
    forecast = config.get("downstream_forecast", {})
    if isinstance(forecast, dict) and str(forecast.get("model", "none")) != "none":
        args.extend(
            [
                "--downstream-forecast-enable",
                f"--downstream-forecast-history-steps {int(forecast.get('history_steps', 8))}",
                f"--downstream-forecast-horizon-steps {int(forecast.get('horizon_steps', 1))}",
                f"--downstream-forecast-ridge-alpha {float(forecast.get('ridge_alpha', 1.0e-2))}",
                f"--downstream-forecast-max-windows-per-clip {int(forecast.get('max_windows_per_clip', 128))}",
                f"--downstream-forecast-top-policy {shlex.quote(str(forecast.get('top_policy', 'ts2vec_kcenter_v1')))}",
                f"--downstream-forecast-baseline-policy {shlex.quote(str(forecast.get('baseline_policy', 'quality_stratified_random_v1')))}",
            ]
        )
    return "\n".join(f"    {arg} {continuation}" for arg in args)


def _run(command: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(json.dumps({"event": "run", "cmd": command}), flush=True)
    result = subprocess.run(command, cwd=cwd, text=True, check=False)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    return result


def _default_run_id(config_path: Path) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%MZ")
    return f"{config_path.stem}-{stamp}"


def _default_vm_name(run_id: str) -> str:
    cleaned = "".join(char if char.isalnum() or char == "-" else "-" for char in run_id.lower())
    return f"cov-bench-{cleaned[-38:]}"[:63].rstrip("-")


if __name__ == "__main__":
    main()
