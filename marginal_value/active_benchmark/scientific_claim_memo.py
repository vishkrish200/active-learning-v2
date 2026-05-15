from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def build_scientific_claim_memo(
    decision_card_json: str | Path,
    selection_mechanism_json: str | Path,
    selected_motion_json: str | Path,
    motion_outcome_json: str | Path,
    *,
    advisor_markdown: str | Path | None = None,
) -> dict[str, Any]:
    decision = _read_json(decision_card_json)
    selection = _read_json(selection_mechanism_json)
    motion = _read_json(selected_motion_json)
    outcome = _read_json(motion_outcome_json)
    advisor_text = _read_optional_text(advisor_markdown)

    champion = str(decision.get("result_card", {}).get("champion_policy") or "window_kcenter_v1")
    policy_profiles = {str(row.get("policy_id")): row for row in decision.get("policy_final_summary", [])}
    motion_profiles = {str(row.get("policy_id")): row for row in motion.get("policy_motion_profiles", [])}
    outcome_profiles = {str(row.get("policy_id")): row for row in outcome.get("policy_motion_outcome_profiles", [])}

    return {
        "decision": "freeze_window_kcenter_as_downstream_canary_mean_risk_incumbent",
        "inputs": {
            "decision_card_json": str(decision_card_json),
            "selection_mechanism_json": str(selection_mechanism_json),
            "selected_motion_json": str(selected_motion_json),
            "motion_outcome_json": str(motion_outcome_json),
            "advisor_markdown": str(advisor_markdown) if advisor_markdown else None,
        },
        "claims": _claims(champion, policy_profiles, motion_profiles, outcome_profiles),
        "non_claims": _non_claims(),
        "evidence": _evidence(champion, decision, selection, motion, outcome),
        "self_deception_risks": _self_deception_risks(),
        "advisor": _advisor(advisor_text),
        "next_steps": _next_steps(),
    }


def write_scientific_claim_memo_reports(
    decision_card_json: str | Path,
    selection_mechanism_json: str | Path,
    selected_motion_json: str | Path,
    motion_outcome_json: str | Path,
    *,
    output_json: str | Path,
    output_markdown: str | Path,
    advisor_markdown: str | Path | None = None,
) -> dict[str, str]:
    memo = build_scientific_claim_memo(
        decision_card_json,
        selection_mechanism_json,
        selected_motion_json,
        motion_outcome_json,
        advisor_markdown=advisor_markdown,
    )
    json_path = Path(output_json)
    markdown_path = Path(output_markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(memo, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_scientific_claim_memo_markdown(memo), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def render_scientific_claim_memo_markdown(memo: Mapping[str, Any]) -> str:
    lines = [
        "# Frozen Scientific Claim Memo",
        "",
        "## Decision",
        "",
        f"- decision: `{memo.get('decision')}`",
        "- status: frozen synthesis; no new selector decision is made here",
        "",
        "## What This Claims",
        "",
    ]
    lines.extend(f"- {item}" for item in memo.get("claims", []))
    lines.extend(["", "## What This Does Not Claim", ""])
    lines.extend(f"- {item}" for item in memo.get("non_claims", []))
    lines.extend(["", "## Evidence Anchors", ""])
    for item in memo.get("evidence", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Self-Deception Risks", ""])
    lines.extend(f"- {item}" for item in memo.get("self_deception_risks", []))
    advisor = memo.get("advisor", {})
    lines.extend(["", "## GPT-5.5 Pro Advisor Read", ""])
    lines.append(f"- source: `{advisor.get('source', '')}`")
    lines.append(f"- summary: {advisor.get('summary', '')}")
    if advisor.get("verdict_bullets"):
        lines.extend(f"- {item}" for item in advisor.get("verdict_bullets", []))
    lines.extend(["", "## Frozen Next Step", ""])
    lines.extend(f"- {item}" for item in memo.get("next_steps", []))
    lines.append("")
    return "\n".join(lines)


def _claims(
    champion: str,
    policy_profiles: Mapping[str, Mapping[str, Any]],
    motion_profiles: Mapping[str, Mapping[str, Any]],
    outcome_profiles: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    champion_profile = policy_profiles.get(champion, {})
    ts2vec = policy_profiles.get("ts2vec_kcenter_v1", {})
    submitted = policy_profiles.get("submitted_full_replay_v1", {})
    random = policy_profiles.get("quality_stratified_random_v1", {})
    champion_motion = motion_profiles.get(champion, {})
    champion_outcome = outcome_profiles.get(champion, {})
    return [
        f"`{champion}` is the current downstream forecast canary mean-risk incumbent by mean K=4 after-MSE (`{_fmt(champion_profile.get('mean_after_mse'))}` over `{champion_profile.get('row_count', '')}` policy-unit rows).",
        f"`ts2vec_kcenter_v1` remains scientifically useful as a representation ablation or feature source, but it is not the lead acquisition story in this benchmark (`mean after-MSE {_fmt(ts2vec.get('mean_after_mse'))}`).",
        f"`submitted_full_replay_v1` remains a defensible submitted-system comparator (`mean after-MSE {_fmt(submitted.get('mean_after_mse'))}`), not the proof target.",
        f"`quality_stratified_random_v1` is still the required baseline (`mean after-MSE {_fmt(random.get('mean_after_mse'))}`); the incumbent must be described relative to it.",
        f"The best mechanism-level read is that `{champion}` covers different raw-motion regimes: selected dynamic energy `{_fmt(champion_motion.get('mean_motion_energy'))}` and selected rotation-dominant rate `{_fmt(champion_outcome.get('mean_selected_rotation_dominant_rate'))}`.",
    ]


def _non_claims() -> list[str]:
    return [
        "This is not a learned query policy.",
        "This is not proof that neural downstream training would improve with the same selected clips.",
        "This is not evidence that TS2Vec is useless; TS2Vec remains a useful feature source / ablation.",
        "This is not a scalar motion-energy ranker: the motion audit is explanatory, and per-unit scalar correlations are weak or mixed.",
        "This is not a license to tune another selector on the same hidden downstream target.",
        "This is not a claim that `window_kcenter_v1` wins every paired unit; the TS2Vec comparison is heterogeneous.",
    ]


def _evidence(
    champion: str,
    decision: Mapping[str, Any],
    selection: Mapping[str, Any],
    motion: Mapping[str, Any],
    outcome: Mapping[str, Any],
) -> list[str]:
    evidence = []
    for row in decision.get("pairwise_final_deltas", []):
        if {str(row.get("policy_a")), str(row.get("policy_b"))} == {champion, "ts2vec_kcenter_v1"}:
            advantage = _oriented_advantage(row, champion)
            champion_wins = _oriented_win_count(row, champion)
            other_wins = _other_win_count(row, champion)
            evidence.append(
                f"Against TS2Vec, `{champion}` has positive mean MSE advantage `{_fmt(advantage)}`, but paired wins are heterogeneous: `{champion_wins}` vs `{other_wins}`."
            )
        if {str(row.get("policy_a")), str(row.get("policy_b"))} == {champion, "quality_stratified_random_v1"}:
            evidence.append(
                f"Against quality-stratified random, `{champion}` is cleaner: mean MSE advantage `{_fmt(_oriented_advantage(row, champion))}`."
            )
    for row in selection.get("focal_pairwise_contrasts", []):
        if row.get("comparison_policy") == "ts2vec_kcenter_v1":
            evidence.append(
                f"Selected-set mechanism differs from TS2Vec: clip Jaccard `{_fmt(row.get('mean_jaccard'))}`, source Jaccard `{_fmt(row.get('mean_source_jaccard'))}`."
            )
    for row in motion.get("focal_motion_contrasts", []):
        if row.get("comparison_policy") == "ts2vec_kcenter_v1":
            evidence.append(
                f"Window-only vs TS2Vec-only clips are much more dynamic: energy `{_fmt(row.get('mean_focal_only_motion_energy'))}` vs `{_fmt(row.get('mean_comparison_only_motion_energy'))}`."
            )
    for row in outcome.get("focal_pairwise_motion_outcome_contrasts", []):
        if row.get("comparison_policy") == "ts2vec_kcenter_v1":
            evidence.append(
                f"Motion-outcome link is not monotone: median advantage `{_fmt(row.get('median_after_mse_advantage_focal_over_comparison'))}`, focal lower-MSE count `{row.get('focal_lower_mse_count')}` vs comparison `{row.get('comparison_lower_mse_count')}`."
            )
    hygiene = decision.get("hygiene", {})
    evidence.append(
        f"Hygiene remains clean: leakage_all_ok `{hygiene.get('leakage_all_ok')}`, target leaks `{hygiene.get('selected_target_leak_count_total')}`, out-of-pool selections `{hygiene.get('selected_out_of_pool_count_total')}`."
    )
    return evidence


def _self_deception_risks() -> list[str]:
    return [
        "policy shopping: the hidden downstream target has already influenced decisions, so do not introduce or tune new selectors against this same canary.",
        "Mean-vs-win-count overread: mean MSE favors window, but the TS2Vec pairwise result is heterogeneous and must not be phrased as a universal win.",
        "Mechanism overread: higher dynamic/rotation-heavy support is a plausible regime-level explanation, not a causal scalar feature rule.",
        "Submission overclaim: the submitted system can be described as defensible, but not retroactively proven as a full active-learning loop.",
        "Complexity creep: no TS2Vec retraining, nonlinear ranker, or neural downstream training should be started from this memo alone.",
    ]


def _advisor(advisor_text: str | None) -> dict[str, Any]:
    if not advisor_text:
        return {
            "source": "not provided",
            "summary": "No external advisor text was provided for this memo.",
            "verdict_bullets": [],
        }
    lines = _markdown_content_lines(advisor_text)
    summary = _first_sentence_or_prefix(advisor_text)
    verdict_bullets = lines[1:] if lines and summary == lines[0] else lines
    return {
        "source": "GPT-5.5 Pro / Extended Pro web consultation",
        "summary": summary,
        "verdict_bullets": verdict_bullets[:8],
    }


def _next_steps() -> list[str]:
    return [
        "Freeze this memo as the current scientific interpretation of the downstream canary.",
        "Do not tune or add selectors on this same downstream target.",
        "Use the memo to decide whether a genuinely different downstream objective/model family is worth a separate pre-registered experiment.",
        "Keep TS2Vec framed as a feature source / ablation and window geometry as the current low-budget canary mean-risk incumbent.",
    ]


def _oriented_advantage(row: Mapping[str, Any], focal_policy: str) -> float | None:
    value = row.get("mean_after_mse_advantage_a_over_b")
    if value is None:
        return None
    if row.get("policy_a") == focal_policy:
        return float(value)
    return -float(value)


def _oriented_win_count(row: Mapping[str, Any], focal_policy: str) -> int:
    return int(row.get("policy_a_win_count" if row.get("policy_a") == focal_policy else "policy_b_win_count", 0))


def _other_win_count(row: Mapping[str, Any], focal_policy: str) -> int:
    return int(row.get("policy_b_win_count" if row.get("policy_a") == focal_policy else "policy_a_win_count", 0))


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_optional_text(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return Path(path).read_text(encoding="utf-8").strip()


def _first_sentence_or_prefix(text: str, limit: int = 360) -> str:
    content_lines = _markdown_content_lines(text)
    compact = " ".join(content_lines)
    for sep in (". ", "\n"):
        if sep in compact:
            return compact.split(sep, 1)[0].strip() + "."
    return compact[:limit].strip()


def _markdown_content_lines(text: str) -> list[str]:
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if (
            not line
            or line.lstrip().startswith("#")
            or line.startswith("Source:")
            or line.startswith("Consult date:")
        ):
            continue
        lines.append(line.removeprefix("- ").strip())
    return lines


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.9f}"
    return str(value)
