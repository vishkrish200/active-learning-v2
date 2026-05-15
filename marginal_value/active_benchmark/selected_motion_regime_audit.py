from __future__ import annotations

import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np

from marginal_value.active_benchmark.selection_mechanism_audit import (
    DEFAULT_COMPARISON_POLICIES,
    DEFAULT_FOCAL_POLICY,
    _load_run_source,
    _selected_by_policy_unit,
    _selected_ids,
)
from marginal_value.preprocessing.quality import compute_quality_features, load_modal_jsonl_imu


def build_selected_motion_regime_audit(
    run_input: str | Path,
    *,
    raw_dirs: Sequence[str | Path] = (),
    manifest_paths: Sequence[str | Path] = (),
    budget_k: int | None = None,
    max_samples: int = 900,
    sample_rate: float = 30.0,
    workers: int = 8,
    timeout_seconds: float = 20.0,
    focal_policy: str = DEFAULT_FOCAL_POLICY,
    comparison_policies: Sequence[str] = DEFAULT_COMPARISON_POLICIES,
) -> dict[str, Any]:
    source = _load_run_source(Path(run_input))
    selected_rows = source["selected_rows"]
    if not selected_rows:
        raise ValueError(f"No selected rows found in {run_input}")
    final_budget = int(budget_k) if budget_k is not None else max(int(row["budget_k"]) for row in selected_rows)
    selected_rows = [row for row in selected_rows if int(row.get("budget_k", -1)) == final_budget]
    selected_ids = sorted({str(row["sample_id"]) for row in selected_rows})
    raw_sources = _resolve_raw_sources(
        selected_ids,
        raw_dirs=[Path(path) for path in raw_dirs],
        manifest_paths=[Path(path) for path in manifest_paths],
    )
    missing = sorted(set(selected_ids) - set(raw_sources))
    if missing:
        raise FileNotFoundError(f"Missing raw source for {len(missing)} selected clips: {missing[:10]}")

    clip_features = _load_clip_features(
        raw_sources,
        max_samples=max_samples,
        sample_rate=sample_rate,
        workers=workers,
        timeout_seconds=timeout_seconds,
    )
    _assign_regime_labels(clip_features)
    return {
        "input": {
            "run_input": str(run_input),
            "source_kind": source["source_kind"],
            "seed_count": len(source["seed_names"]),
            "seed_names": source["seed_names"],
            "budget_k": final_budget,
            "selected_unique_clip_count": len(selected_ids),
            "loaded_clip_count": len(clip_features),
            "raw_dir_count": len(raw_dirs),
            "manifest_count": len(manifest_paths),
            "max_samples": int(max_samples),
            "sample_rate": float(sample_rate),
            "focal_policy": focal_policy,
            "comparison_policies": list(comparison_policies),
        },
        "policy_motion_profiles": _policy_motion_profiles(selected_rows, clip_features),
        "focal_motion_contrasts": _focal_motion_contrasts(
            selected_rows,
            clip_features,
            focal_policy=focal_policy,
            comparison_policies=comparison_policies,
        ),
        "clip_motion_features": [clip_features[sample_id] for sample_id in sorted(clip_features)],
        "read": _interpret_motion(
            selected_rows,
            clip_features,
            focal_policy=focal_policy,
            comparison_policies=comparison_policies,
        ),
    }


def write_selected_motion_regime_audit_reports(
    run_input: str | Path,
    *,
    output_json: str | Path,
    output_markdown: str | Path,
    raw_dirs: Sequence[str | Path] = (),
    manifest_paths: Sequence[str | Path] = (),
    budget_k: int | None = None,
    max_samples: int = 900,
    sample_rate: float = 30.0,
    workers: int = 8,
    timeout_seconds: float = 20.0,
    focal_policy: str = DEFAULT_FOCAL_POLICY,
    comparison_policies: Sequence[str] = DEFAULT_COMPARISON_POLICIES,
) -> dict[str, str]:
    audit = build_selected_motion_regime_audit(
        run_input,
        raw_dirs=raw_dirs,
        manifest_paths=manifest_paths,
        budget_k=budget_k,
        max_samples=max_samples,
        sample_rate=sample_rate,
        workers=workers,
        timeout_seconds=timeout_seconds,
        focal_policy=focal_policy,
        comparison_policies=comparison_policies,
    )
    json_path = Path(output_json)
    markdown_path = Path(output_markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_selected_motion_regime_audit_markdown(audit), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def render_selected_motion_regime_audit_markdown(audit: Mapping[str, Any]) -> str:
    input_info = audit.get("input", {})
    lines = [
        "# Selected Motion Regime Audit",
        "",
        "## Scope",
        "",
        f"- run input: `{input_info.get('run_input')}`",
        f"- seeds: `{input_info.get('seed_count')}`",
        f"- budget: `K={input_info.get('budget_k')}`",
        f"- selected unique clips: `{input_info.get('selected_unique_clip_count')}`",
        f"- loaded raw clips: `{input_info.get('loaded_clip_count')}`",
        f"- max samples per clip: `{input_info.get('max_samples')}`",
        f"- focal policy: `{input_info.get('focal_policy')}`",
        "",
        "## Read",
        "",
    ]
    lines.extend(f"- {item}" for item in audit.get("read", []))
    lines.extend(
        [
            "",
            "## Policy Motion Profiles",
            "",
            "Dynamic energy is centered acceleration energy plus gyro energy, so gravity magnitude does not dominate this column.",
            "",
            "| policy | selected rows | unique clips | dynamic energy | acc p95 | gyro p95 | acc delta p95 | gyro delta p95 | stationary | top regimes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in audit.get("policy_motion_profiles", []):
        lines.append(
            "| `{policy}` | {rows} | {clips} | {energy} | {acc} | {gyro} | {acc_delta} | {gyro_delta} | {stationary} | {regimes} |".format(
                policy=row.get("policy_id", ""),
                rows=int(row.get("selected_row_count", 0)),
                clips=int(row.get("unique_clip_count", 0)),
                energy=_fmt(row.get("mean_motion_energy")),
                acc=_fmt(row.get("mean_acc_norm_p95")),
                gyro=_fmt(row.get("mean_gyro_norm_p95")),
                acc_delta=_fmt(row.get("mean_acc_delta_norm_p95")),
                gyro_delta=_fmt(row.get("mean_gyro_delta_norm_p95")),
                stationary=_fmt(row.get("mean_stationary_fraction")),
                regimes=_format_regimes(row.get("regime_counts", {})),
            )
        )
    lines.extend(
        [
            "",
            "## Focal-Only Motion Contrasts",
            "",
            "| comparison | units | focal-only clips | comparison-only clips | focal dynamic energy | comparison dynamic energy | focal gyro p95 | comparison gyro p95 | focal regimes | comparison regimes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in audit.get("focal_motion_contrasts", []):
        lines.append(
            "| `{comparison}` | {units} | {focal_n} | {comp_n} | {focal_energy} | {comp_energy} | {focal_gyro} | {comp_gyro} | {focal_regimes} | {comp_regimes} |".format(
                comparison=row.get("comparison_policy", ""),
                units=int(row.get("paired_unit_count", 0)),
                focal_n=int(row.get("focal_only_clip_observation_count", 0)),
                comp_n=int(row.get("comparison_only_clip_observation_count", 0)),
                focal_energy=_fmt(row.get("mean_focal_only_motion_energy")),
                comp_energy=_fmt(row.get("mean_comparison_only_motion_energy")),
                focal_gyro=_fmt(row.get("mean_focal_only_gyro_norm_p95")),
                comp_gyro=_fmt(row.get("mean_comparison_only_gyro_norm_p95")),
                focal_regimes=_format_regimes(row.get("focal_only_regime_counts", {})),
                comp_regimes=_format_regimes(row.get("comparison_only_regime_counts", {})),
            )
        )
    lines.extend(
        [
            "",
            "## Top Focal-Only Clip Examples",
            "",
            "| comparison | sample | regime | motion energy | gyro p95 | acc delta p95 |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in audit.get("focal_motion_contrasts", []):
        for example in row.get("top_focal_only_motion_examples", [])[:8]:
            lines.append(
                "| `{comparison}` | `{sample}` | `{regime}` | {energy} | {gyro} | {acc_delta} |".format(
                    comparison=row.get("comparison_policy", ""),
                    sample=example.get("sample_id", ""),
                    regime=example.get("regime_label", ""),
                    energy=_fmt(example.get("motion_energy")),
                    gyro=_fmt(example.get("gyro_norm_p95")),
                    acc_delta=_fmt(example.get("acc_delta_norm_p95")),
                )
            )
    lines.append("")
    return "\n".join(lines)


def _resolve_raw_sources(
    sample_ids: Sequence[str],
    *,
    raw_dirs: Sequence[Path],
    manifest_paths: Sequence[Path],
) -> dict[str, str]:
    sample_id_set = set(sample_ids)
    sources: dict[str, str] = {}
    for raw_dir in raw_dirs:
        for sample_id in sample_ids:
            path = raw_dir / f"{sample_id}.jsonl"
            if path.exists():
                sources[sample_id] = str(path)
    if len(sources) == len(sample_id_set):
        return sources
    for manifest_path in manifest_paths:
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                url = line.strip()
                if not url:
                    continue
                sample_id = _sample_id_from_url(url)
                if sample_id in sample_id_set and sample_id not in sources:
                    sources[sample_id] = url
    return sources


def _load_clip_features(
    raw_sources: Mapping[str, str],
    *,
    max_samples: int,
    sample_rate: float,
    workers: int,
    timeout_seconds: float,
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {
            pool.submit(
                _clip_motion_features,
                sample_id,
                source,
                max_samples=max_samples,
                sample_rate=sample_rate,
                timeout_seconds=timeout_seconds,
            ): sample_id
            for sample_id, source in raw_sources.items()
        }
        for future in as_completed(futures):
            sample_id = futures[future]
            features[sample_id] = future.result()
    return features


def _clip_motion_features(
    sample_id: str,
    source: str,
    *,
    max_samples: int,
    sample_rate: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    samples, timestamps = _load_samples(source, max_samples=max_samples, timeout_seconds=timeout_seconds)
    quality = compute_quality_features(samples, timestamps=timestamps, sample_rate=sample_rate)
    acc = samples[:, :3]
    gyro = samples[:, 3:6]
    centered_acc = acc - np.mean(acc, axis=0, keepdims=True)
    acc_norm = np.linalg.norm(acc, axis=1)
    gyro_norm = np.linalg.norm(gyro, axis=1)
    acc_deltas = np.linalg.norm(np.diff(acc, axis=0), axis=1) if len(acc) > 1 else np.asarray([0.0])
    gyro_deltas = np.linalg.norm(np.diff(gyro, axis=0), axis=1) if len(gyro) > 1 else np.asarray([0.0])
    acc_dynamic_energy = float(np.mean(np.sum(centered_acc * centered_acc, axis=1)))
    gyro_energy = float(np.mean(gyro_norm * gyro_norm))
    return {
        "sample_id": sample_id,
        "raw_source": source,
        "n_samples": int(samples.shape[0]),
        "duration_sec": float(samples.shape[0] / sample_rate),
        "acc_norm_mean": float(np.mean(acc_norm)),
        "acc_norm_std": float(np.std(acc_norm)),
        "acc_norm_p95": float(np.percentile(acc_norm, 95)),
        "gyro_norm_mean": float(np.mean(gyro_norm)),
        "gyro_norm_std": float(np.std(gyro_norm)),
        "gyro_norm_p95": float(np.percentile(gyro_norm, 95)),
        "acc_delta_norm_mean": float(np.mean(acc_deltas)),
        "acc_delta_norm_p95": float(np.percentile(acc_deltas, 95)),
        "gyro_delta_norm_mean": float(np.mean(gyro_deltas)),
        "gyro_delta_norm_p95": float(np.percentile(gyro_deltas, 95)),
        "motion_energy": float(acc_dynamic_energy + gyro_energy),
        "acc_dynamic_energy": float(acc_dynamic_energy),
        "gyro_energy": float(gyro_energy),
        "gyro_energy_fraction": float(gyro_energy / max(acc_dynamic_energy + gyro_energy, 1.0e-12)),
        "quality_score": float(quality["quality_score"]),
        "stationary_fraction": float(quality["stationary_fraction"]),
        "high_frequency_energy": float(quality["high_frequency_energy"]),
    }


def _load_samples(source: str, *, max_samples: int, timeout_seconds: float) -> tuple[np.ndarray, np.ndarray | None]:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https", "file"}:
        return _load_url_jsonl_imu(source, max_samples=max_samples, timeout_seconds=timeout_seconds)
    return load_modal_jsonl_imu(source, max_samples=max_samples)


def _load_url_jsonl_imu(source: str, *, max_samples: int, timeout_seconds: float) -> tuple[np.ndarray, np.ndarray | None]:
    samples: list[list[float]] = []
    timestamps: list[float] = []
    saw_timestamp = False
    with urlopen(source, timeout=timeout_seconds) as response:
        for raw_line in response:
            if len(samples) >= max_samples:
                break
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            record = json.loads(line)
            acc = record.get("acc")
            gyro = record.get("gyro")
            if not isinstance(acc, list) or not isinstance(gyro, list) or len(acc) < 3 or len(gyro) < 3:
                raise ValueError(f"Unexpected IMU JSONL record shape for {source}")
            samples.append([float(value) for value in [*acc[:3], *gyro[:3]]])
            if "t_us" in record:
                timestamps.append(float(record["t_us"]) / 1_000_000.0)
                saw_timestamp = True
            else:
                timestamps.append(np.nan)
    if not samples:
        raise ValueError(f"No IMU samples read from {source}")
    return np.asarray(samples, dtype=float), np.asarray(timestamps, dtype=float) if saw_timestamp else None


def _assign_regime_labels(clip_features: Mapping[str, dict[str, Any]]) -> None:
    values = list(clip_features.values())
    if not values:
        return
    q = {
        "motion_lo": _percentile(values, "motion_energy", 25),
        "motion_hi": _percentile(values, "motion_energy", 75),
        "gyro_frac_hi": _percentile(values, "gyro_energy_fraction", 75),
        "gyro_hi": _percentile(values, "gyro_norm_p95", 75),
        "acc_delta_hi": _percentile(values, "acc_delta_norm_p95", 75),
    }
    for row in values:
        if float(row["stationary_fraction"]) >= 0.80 or float(row["motion_energy"]) <= q["motion_lo"]:
            label = "low_motion"
        elif float(row["gyro_energy_fraction"]) >= q["gyro_frac_hi"] or float(row["gyro_norm_p95"]) >= q["gyro_hi"]:
            label = "rotation_dominant"
        elif float(row["motion_energy"]) >= q["motion_hi"] or float(row["acc_delta_norm_p95"]) >= q["acc_delta_hi"]:
            label = "high_dynamic"
        else:
            label = "mixed_motion"
        row["regime_label"] = label


def _policy_motion_profiles(
    selected_rows: Sequence[Mapping[str, Any]],
    clip_features: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        if str(row["sample_id"]) in clip_features:
            grouped[str(row["policy_id"])].append(row)
    profiles = []
    for policy, rows in grouped.items():
        samples = [clip_features[str(row["sample_id"])] for row in rows]
        profiles.append(
            {
                "policy_id": policy,
                "selected_row_count": len(rows),
                "unique_clip_count": len({str(row["sample_id"]) for row in rows}),
                "regime_counts": dict(Counter(str(sample["regime_label"]) for sample in samples)),
                **_motion_means(samples),
            }
        )
    return sorted(profiles, key=lambda row: str(row["policy_id"]))


def _focal_motion_contrasts(
    selected_rows: Sequence[Mapping[str, Any]],
    clip_features: Mapping[str, Mapping[str, Any]],
    *,
    focal_policy: str,
    comparison_policies: Sequence[str],
) -> list[dict[str, Any]]:
    selected = _selected_by_policy_unit(selected_rows)
    rows = []
    for comparison in comparison_policies:
        units = sorted(set(selected[focal_policy]) & set(selected[comparison]))
        focal_only_ids: list[str] = []
        comparison_only_ids: list[str] = []
        for unit in units:
            focal_ids = set(_selected_ids(selected[focal_policy][unit]))
            comparison_ids = set(_selected_ids(selected[comparison][unit]))
            focal_only_ids.extend(sorted(focal_ids - comparison_ids))
            comparison_only_ids.extend(sorted(comparison_ids - focal_ids))
        focal_samples = [clip_features[sample_id] for sample_id in focal_only_ids if sample_id in clip_features]
        comparison_samples = [clip_features[sample_id] for sample_id in comparison_only_ids if sample_id in clip_features]
        rows.append(
            {
                "focal_policy": focal_policy,
                "comparison_policy": comparison,
                "paired_unit_count": len(units),
                "focal_only_clip_observation_count": len(focal_samples),
                "comparison_only_clip_observation_count": len(comparison_samples),
                "focal_only_unique_clip_count": len(set(focal_only_ids)),
                "comparison_only_unique_clip_count": len(set(comparison_only_ids)),
                "focal_only_regime_counts": dict(Counter(str(sample["regime_label"]) for sample in focal_samples)),
                "comparison_only_regime_counts": dict(Counter(str(sample["regime_label"]) for sample in comparison_samples)),
                "top_focal_only_motion_examples": _top_motion_examples(focal_samples),
                "top_comparison_only_motion_examples": _top_motion_examples(comparison_samples),
                **_prefixed_motion_means("focal_only", focal_samples),
                **_prefixed_motion_means("comparison_only", comparison_samples),
            }
        )
    return rows


def _interpret_motion(
    selected_rows: Sequence[Mapping[str, Any]],
    clip_features: Mapping[str, Mapping[str, Any]],
    *,
    focal_policy: str,
    comparison_policies: Sequence[str],
) -> list[str]:
    profiles = {row["policy_id"]: row for row in _policy_motion_profiles(selected_rows, clip_features)}
    reads = [
        "This audit joins selected clip IDs back to raw IMU motion summaries; it does not change policy rankings.",
        "Regime labels are relative to the selected-clip pool, so they explain this run rather than defining universal activity classes.",
        "Motion energy is centered acceleration energy plus gyro energy; raw gravity magnitude is reported separately through acceleration percentiles.",
    ]
    focal = profiles.get(focal_policy)
    if focal:
        reads.append(
            f"`{focal_policy}` selected clips have mean gyro p95 `{float(focal['mean_gyro_norm_p95']):.6f}` and mean motion energy `{float(focal['mean_motion_energy']):.6f}`."
        )
    contrasts = _focal_motion_contrasts(
        selected_rows,
        clip_features,
        focal_policy=focal_policy,
        comparison_policies=comparison_policies,
    )
    for contrast in contrasts:
        comparison = str(contrast["comparison_policy"])
        focal_energy = float(contrast.get("mean_focal_only_motion_energy") or 0.0)
        comparison_energy = float(contrast.get("mean_comparison_only_motion_energy") or 0.0)
        focal_gyro = float(contrast.get("mean_focal_only_gyro_norm_p95") or 0.0)
        comparison_gyro = float(contrast.get("mean_comparison_only_gyro_norm_p95") or 0.0)
        reads.append(
            f"`{focal_policy}`-only clips vs `{comparison}`-only clips: motion energy `{focal_energy:.6f}` vs `{comparison_energy:.6f}`, gyro p95 `{focal_gyro:.6f}` vs `{comparison_gyro:.6f}`."
        )
    return reads


def _motion_means(samples: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    return {
        "mean_motion_energy": _mean_key(samples, "motion_energy"),
        "median_motion_energy": _median_key(samples, "motion_energy"),
        "mean_acc_norm_p95": _mean_key(samples, "acc_norm_p95"),
        "mean_gyro_norm_p95": _mean_key(samples, "gyro_norm_p95"),
        "mean_acc_delta_norm_p95": _mean_key(samples, "acc_delta_norm_p95"),
        "mean_gyro_delta_norm_p95": _mean_key(samples, "gyro_delta_norm_p95"),
        "mean_gyro_energy_fraction": _mean_key(samples, "gyro_energy_fraction"),
        "mean_stationary_fraction": _mean_key(samples, "stationary_fraction"),
        "mean_quality_score": _mean_key(samples, "quality_score"),
    }


def _prefixed_motion_means(prefix: str, samples: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    return {f"mean_{prefix}_{key.removeprefix('mean_')}": value for key, value in _motion_means(samples).items() if key.startswith("mean_")}


def _top_motion_examples(samples: Sequence[Mapping[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    by_sample: dict[str, Mapping[str, Any]] = {}
    for row in samples:
        by_sample[str(row["sample_id"])] = row
    rows = sorted(by_sample.values(), key=lambda row: float(row["motion_energy"]), reverse=True)
    return [
        {
            "sample_id": row["sample_id"],
            "regime_label": row["regime_label"],
            "motion_energy": row["motion_energy"],
            "gyro_norm_p95": row["gyro_norm_p95"],
            "acc_delta_norm_p95": row["acc_delta_norm_p95"],
        }
        for row in rows[:limit]
    ]


def _mean_key(samples: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(sample[key]) for sample in samples if sample.get(key) is not None]
    return float(mean(values)) if values else None


def _median_key(samples: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(sample[key]) for sample in samples if sample.get(key) is not None]
    return float(median(values)) if values else None


def _percentile(samples: Sequence[Mapping[str, Any]], key: str, q: float) -> float:
    values = [float(sample[key]) for sample in samples]
    return float(np.percentile(values, q)) if values else 0.0


def _sample_id_from_url(url: str) -> str:
    parts = [part for part in urlparse(url).path.split("/") if part]
    clip = Path(parts[-1]).stem if parts else "clip"
    source = "unknown"
    for part in parts:
        if part.startswith("worker"):
            source = part
            break
    return f"{source}_{clip}"


def _format_regimes(regimes: object) -> str:
    if not isinstance(regimes, Mapping):
        return ""
    return ", ".join(f"{key}:{value}" for key, value in sorted(regimes.items(), key=lambda item: (-int(item[1]), str(item[0]))))


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
