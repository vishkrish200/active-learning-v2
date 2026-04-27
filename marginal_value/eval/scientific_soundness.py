from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from marginal_value.eval.marginal_coverage_report import coverage_value, fold_primary_score


DEFAULT_CANDIDATE_POLICY = "quality_gated_old_novelty_raw_shape_stats_q85_stat90_abs60_clustercap2"
DEFAULT_UNCAPPED_POLICY = "quality_gated_old_novelty_raw_shape_stats_q85_stat90_abs60"
DEFAULT_SIMPLE_CONTROLS = ("random_high_quality", "quality_only", "old_novelty_only")
DEFAULT_PRIMARY_REPRESENTATIONS = ("temporal_order", "raw_shape_stats")
DEFAULT_INDEPENDENT_REPRESENTATIONS = ("temporal_order",)
DEFAULT_K_VALUES = (100, 200)


def evaluate_scientific_soundness(
    report: dict[str, Any],
    *,
    candidate_policy: str = DEFAULT_CANDIDATE_POLICY,
    uncapped_policy: str | None = DEFAULT_UNCAPPED_POLICY,
    simple_controls: Sequence[str] = DEFAULT_SIMPLE_CONTROLS,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    primary_representations: Sequence[str] = DEFAULT_PRIMARY_REPRESENTATIONS,
    independent_representations: Sequence[str] = DEFAULT_INDEPENDENT_REPRESENTATIONS,
    min_fold_wins: int = 3,
    material_loss_tolerance: float = 0.002,
) -> dict[str, Any]:
    """Turn marginal-coverage metrics into explicit scientific claim gates.

    The main purpose is to avoid overclaiming. A raw-shape ranking policy can
    pass the narrow artifact-safe raw-shape claim while still failing the
    broader behavior-discovery claim if independent temporal coverage loses to
    simple controls.
    """

    folds = _folds(report)
    gates = [
        _primary_vs_controls_gate(
            folds=folds,
            candidate_policy=candidate_policy,
            controls=simple_controls,
            k_values=k_values,
            primary_representations=primary_representations,
            min_fold_wins=min_fold_wins,
        ),
        _physical_validity_gate(folds=folds, candidate_policy=candidate_policy, k_values=k_values),
        _uncapped_regression_gate(
            folds=folds,
            candidate_policy=candidate_policy,
            uncapped_policy=uncapped_policy,
            k_values=k_values,
            primary_representations=primary_representations,
            material_loss_tolerance=material_loss_tolerance,
        ),
        _independent_representation_gate(
            folds=folds,
            candidate_policy=candidate_policy,
            controls=simple_controls,
            k_values=k_values,
            independent_representations=independent_representations,
            min_fold_wins=min_fold_wins,
        ),
    ]
    by_name = {str(gate["name"]): gate for gate in gates}
    artifact_safe = _status_if_all_pass(
        by_name,
        ["primary_vs_simple_controls", "physical_validity", "uncapped_regression"],
    )
    broad_behavior = "pass" if by_name["independent_temporal_vs_controls"]["status"] == "pass" else "fail"
    if artifact_safe == "pass" and broad_behavior == "pass":
        overall = "pass"
    elif artifact_safe == "pass":
        overall = "conditional"
    else:
        overall = "fail"
    return {
        "overall_status": overall,
        "candidate_policy": candidate_policy,
        "uncapped_policy": uncapped_policy,
        "k_values": [int(k) for k in k_values],
        "primary_representations": list(primary_representations),
        "independent_representations": list(independent_representations),
        "claim_status": {
            "artifact_safe_raw_shape_supported": artifact_safe,
            "broad_behavior_discovery_supported": broad_behavior,
            "hidden_test_ready": "needs_blind_external_test" if overall != "fail" else "no",
        },
        "gates": gates,
        "interpretation": _interpretation(artifact_safe=artifact_safe, broad_behavior=broad_behavior),
    }


def render_scientific_soundness_markdown(verdict: dict[str, Any]) -> str:
    lines = [
        "# Scientific Soundness Verdict",
        "",
        "## Summary",
        "",
        f"- candidate: `{verdict.get('candidate_policy', '')}`",
        f"- overall status: `{verdict.get('overall_status', '')}`",
        f"- primary representations: `{', '.join(verdict.get('primary_representations', []))}`",
        f"- independent representations: `{', '.join(verdict.get('independent_representations', []))}`",
        "",
        "## Claim Status",
        "",
        "| claim | status |",
        "|---|---|",
    ]
    for claim, status in verdict.get("claim_status", {}).items():
        lines.append(f"| {claim} | {status} |")
    lines.extend(
        [
            "",
            "## Gates",
            "",
            "| gate | status | key result |",
            "|---|---|---|",
        ]
    )
    for gate in verdict.get("gates", []):
        lines.append(f"| {gate.get('name', '')} | {gate.get('status', '')} | {gate.get('summary', '')} |")
    lines.extend(["", "## Interpretation", "", str(verdict.get("interpretation", "")), ""])
    return "\n".join(lines)


def write_scientific_soundness_report(
    report: dict[str, Any] | str | Path,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
    **kwargs: Any,
) -> dict[str, str]:
    loaded = _load_report(report)
    verdict = evaluate_scientific_soundness(loaded, **kwargs)
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")
    markdown_output.write_text(render_scientific_soundness_markdown(verdict), encoding="utf-8")
    return {"json_path": str(json_output), "markdown_path": str(markdown_output)}


def _primary_vs_controls_gate(
    *,
    folds: list[dict[str, Any]],
    candidate_policy: str,
    controls: Sequence[str],
    k_values: Sequence[int],
    primary_representations: Sequence[str],
    min_fold_wins: int,
) -> dict[str, Any]:
    checks = []
    for k in k_values:
        for control in controls:
            deltas = _primary_deltas(
                folds,
                candidate_policy=candidate_policy,
                baseline_policy=control,
                k=int(k),
                representations=primary_representations,
            )
            checks.append(_comparison_check(k=int(k), baseline=control, deltas=deltas, min_fold_wins=min_fold_wins))
    passed = all(check["mean_delta"] > 0.0 and check["fold_wins"] >= min_fold_wins for check in checks)
    return {
        "name": "primary_vs_simple_controls",
        "status": "pass" if passed else "fail",
        "summary": _summarize_checks(checks),
        "checks": checks,
    }


def _physical_validity_gate(
    *,
    folds: list[dict[str, Any]],
    candidate_policy: str,
    k_values: Sequence[int],
) -> dict[str, Any]:
    checks = []
    for k in k_values:
        selections = [
            fold.get("policies", {}).get(candidate_policy, {}).get(f"selection@{int(k)}", {})
            for fold in folds
        ]
        stationary_over_90 = max(_safe_float(selection.get("stationary_fraction_over_90", 0.0)) for selection in selections)
        max_abs_over_60 = max(_safe_float(selection.get("max_abs_value_over_60", 0.0)) for selection in selections)
        min_quality = min(_safe_float(selection.get("min_quality", 0.0)) for selection in selections)
        checks.append(
            {
                "k": int(k),
                "max_stationary_fraction_over_90": stationary_over_90,
                "max_abs_value_over_60": max_abs_over_60,
                "min_quality": min_quality,
            }
        )
    passed = all(
        check["max_stationary_fraction_over_90"] <= 0.0 and check["max_abs_value_over_60"] <= 0.0
        for check in checks
    )
    return {
        "name": "physical_validity",
        "status": "pass" if passed else "fail",
        "summary": _summarize_physical_checks(checks),
        "checks": checks,
    }


def _uncapped_regression_gate(
    *,
    folds: list[dict[str, Any]],
    candidate_policy: str,
    uncapped_policy: str | None,
    k_values: Sequence[int],
    primary_representations: Sequence[str],
    material_loss_tolerance: float,
) -> dict[str, Any]:
    if not uncapped_policy:
        return {
            "name": "uncapped_regression",
            "status": "warn",
            "summary": "No uncapped control was configured.",
            "checks": [],
        }
    checks = []
    for k in k_values:
        deltas = _primary_deltas(
            folds,
            candidate_policy=candidate_policy,
            baseline_policy=uncapped_policy,
            k=int(k),
            representations=primary_representations,
        )
        check = _comparison_check(k=int(k), baseline=uncapped_policy, deltas=deltas, min_fold_wins=0)
        check["material_loss_tolerance"] = float(material_loss_tolerance)
        checks.append(check)
    passed = all(check["mean_delta"] >= -float(material_loss_tolerance) for check in checks)
    return {
        "name": "uncapped_regression",
        "status": "pass" if passed else "fail",
        "summary": _summarize_checks(checks),
        "checks": checks,
    }


def _independent_representation_gate(
    *,
    folds: list[dict[str, Any]],
    candidate_policy: str,
    controls: Sequence[str],
    k_values: Sequence[int],
    independent_representations: Sequence[str],
    min_fold_wins: int,
) -> dict[str, Any]:
    checks = []
    for k in k_values:
        for representation in independent_representations:
            for control in controls:
                deltas = _coverage_deltas(
                    folds,
                    candidate_policy=candidate_policy,
                    baseline_policy=control,
                    k=int(k),
                    representation=representation,
                )
                check = _comparison_check(k=int(k), baseline=control, deltas=deltas, min_fold_wins=min_fold_wins)
                check["representation"] = representation
                checks.append(check)
    passed = all(check["mean_delta"] > 0.0 and check["fold_wins"] >= min_fold_wins for check in checks)
    return {
        "name": "independent_temporal_vs_controls",
        "status": "pass" if passed else "fail",
        "summary": _summarize_checks(checks),
        "checks": checks,
    }


def _primary_deltas(
    folds: Sequence[dict[str, Any]],
    *,
    candidate_policy: str,
    baseline_policy: str,
    k: int,
    representations: Sequence[str],
) -> list[float]:
    return [
        fold_primary_score(fold, candidate_policy, k=k, representations=representations)
        - fold_primary_score(fold, baseline_policy, k=k, representations=representations)
        for fold in folds
        if _has_policy(fold, candidate_policy) and _has_policy(fold, baseline_policy)
    ]


def _coverage_deltas(
    folds: Sequence[dict[str, Any]],
    *,
    candidate_policy: str,
    baseline_policy: str,
    k: int,
    representation: str,
) -> list[float]:
    deltas = []
    for fold in folds:
        policies = fold.get("policies", {})
        if not isinstance(policies, dict) or candidate_policy not in policies or baseline_policy not in policies:
            continue
        candidate = coverage_value(policies[candidate_policy], k=k, representation=representation)
        baseline = coverage_value(policies[baseline_policy], k=k, representation=representation)
        deltas.append(float(candidate - baseline))
    return deltas


def _comparison_check(*, k: int, baseline: str, deltas: Sequence[float], min_fold_wins: int) -> dict[str, Any]:
    return {
        "k": int(k),
        "baseline": baseline,
        "mean_delta": _mean(deltas),
        "fold_wins": int(sum(delta > 0.0 for delta in deltas)),
        "fold_ties": int(sum(abs(delta) <= 1.0e-12 for delta in deltas)),
        "folds": int(len(deltas)),
        "min_fold_wins": int(min_fold_wins),
    }


def _summarize_checks(checks: Sequence[dict[str, Any]]) -> str:
    if not checks:
        return "No checks were computed."
    weakest = min(checks, key=lambda check: (float(check.get("mean_delta", 0.0)), int(check.get("fold_wins", 0))))
    return "weakest mean delta {delta:+.4f} vs {baseline} at K={k} ({wins}/{folds} fold wins)".format(
        delta=float(weakest.get("mean_delta", 0.0)),
        baseline=weakest.get("baseline", ""),
        k=weakest.get("k", ""),
        wins=weakest.get("fold_wins", 0),
        folds=weakest.get("folds", 0),
    )


def _summarize_physical_checks(checks: Sequence[dict[str, Any]]) -> str:
    if not checks:
        return "No physical checks were computed."
    max_stationary = max(float(check["max_stationary_fraction_over_90"]) for check in checks)
    max_abs = max(float(check["max_abs_value_over_60"]) for check in checks)
    min_quality = min(float(check["min_quality"]) for check in checks)
    return f"max stationary>0.90={max_stationary:.3f}, max abs>60={max_abs:.3f}, min quality={min_quality:.3f}"


def _interpretation(*, artifact_safe: str, broad_behavior: str) -> str:
    if artifact_safe == "pass" and broad_behavior == "pass":
        return (
            "The candidate passes the narrow safety/coverage gates and the independent-representation gate. "
            "It is still not a substitute for a blind external or downstream training test."
        )
    if artifact_safe == "pass":
        return (
            "The candidate is defensible as an artifact-safe raw-shape/media selector, but it is not yet "
            "scientifically supported as broad behavior discovery because independent temporal coverage does not pass."
        )
    return (
        "The candidate does not pass the basic artifact-safe marginal-coverage gates. Do not promote it without "
        "changing the method or the claim."
    )


def _status_if_all_pass(gates: dict[str, dict[str, Any]], names: Sequence[str]) -> str:
    return "pass" if all(gates[name]["status"] == "pass" for name in names) else "fail"


def _folds(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [fold for fold in report.get("folds", []) if isinstance(fold, dict)]


def _has_policy(fold: dict[str, Any], policy: str) -> bool:
    policies = fold.get("policies", {})
    return isinstance(policies, dict) and policy in policies


def _mean(values: Sequence[float]) -> float:
    clean = [float(value) for value in values]
    return float(sum(clean) / len(clean)) if clean else 0.0


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _load_report(report: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(report, dict):
        return report
    return json.loads(Path(report).read_text(encoding="utf-8"))
