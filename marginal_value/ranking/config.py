from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RankingLeakageError(RuntimeError):
    """Raised when a ranking config would leak held-out data into old support."""


def load_ranking_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_ranking_config(config: dict[str, Any]) -> None:
    execution = _required_mapping(config, "execution")
    data = _required_mapping(config, "data")
    artifacts = _required_mapping(config, "artifacts")
    splits = _required_mapping(config, "splits")
    ranking = _required_mapping(config, "ranking")
    quality = config.get("quality", {})
    grammar_features = config.get("grammar_features", {"enabled": False})
    large_cluster_split = config.get("large_cluster_split", {"enabled": False})
    corruption_eval = config.get("corruption_eval", {"enabled": False})
    learned_ranker = config.get("learned_ranker", {"enabled": False})
    score_guards = config.get("score_guards", {"enabled": False})
    if quality is not None and not isinstance(quality, dict):
        raise ValueError("Ranking config quality section must be an object when provided.")
    if grammar_features is not None and not isinstance(grammar_features, dict):
        raise ValueError("Ranking config grammar_features section must be an object when provided.")
    if large_cluster_split is not None and not isinstance(large_cluster_split, dict):
        raise ValueError("Ranking config large_cluster_split section must be an object when provided.")
    if corruption_eval is not None and not isinstance(corruption_eval, dict):
        raise ValueError("Ranking config corruption_eval section must be an object when provided.")
    if learned_ranker is not None and not isinstance(learned_ranker, dict):
        raise ValueError("Ranking config learned_ranker section must be an object when provided.")
    if score_guards is not None and not isinstance(score_guards, dict):
        raise ValueError("Ranking config score_guards section must be an object when provided.")

    if "checkpoint" in config:
        raise RankingLeakageError("Phase A baseline ranking must not depend on a trained checkpoint.")
    if execution.get("provider") != "modal":
        raise ValueError("Baseline ranking must run on Modal.")
    if not str(data.get("root", "")).startswith("/data"):
        raise ValueError("Ranking data root must be mounted under /data.")
    if not str(artifacts.get("root", "")).startswith("/artifacts"):
        raise ValueError("Ranking artifacts root must be mounted under /artifacts.")

    support_split = splits.get("support_split")
    query_split = splits.get("query_split")
    if support_split == query_split:
        raise RankingLeakageError("support_split and query_split must be different.")
    run_mode = str(ranking.get("run_mode", "eval"))
    if support_split != "pretrain":
        raise RankingLeakageError("Current ranking must use pretrain support.")
    if query_split == "val":
        pass
    elif query_split == "new" and run_mode == "submission":
        if not data.get("new_manifest"):
            raise ValueError("Submission ranking with query_split='new' requires data.new_manifest.")
    else:
        raise RankingLeakageError("Ranking query_split must be val for eval or new for submission.")

    negative_split = splits.get("negative_split", "pretrain")
    if negative_split != "pretrain":
        raise RankingLeakageError("Pseudo-negative examples must come from pretrain only.")

    representation = ranking.get("representation")
    if representation not in {"window_mean_std_pool", "temporal_order", "raw_shape_stats", "encoder_artifact"}:
        raise ValueError(
            "Baseline ranking currently supports window_mean_std_pool, temporal_order, "
            "raw_shape_stats, or encoder_artifact representations."
        )
    if representation == "encoder_artifact":
        encoder_embeddings = _required_mapping(config, "encoder_embeddings")
        for key in ("support_embeddings", "support_manifest", "query_embeddings", "query_manifest"):
            value = str(encoder_embeddings.get(key, ""))
            if not value.startswith("/artifacts/"):
                raise ValueError(f"encoder_embeddings.{key} must point under /artifacts/.")
        for key in ("support_embeddings", "query_embeddings"):
            if not str(encoder_embeddings.get(key, "")).endswith(".npy"):
                raise ValueError(f"encoder_embeddings.{key} must be a .npy file.")
        for key in ("support_manifest", "query_manifest"):
            if not str(encoder_embeddings.get(key, "")).endswith(".csv"):
                raise ValueError(f"encoder_embeddings.{key} must be a .csv file.")
    if ranking.get("reranker_method", "cluster_aware") not in {
        "cluster_aware",
        "cluster_cap",
        "mmr",
        "parent_prefix_cluster_cap",
        "quality_only",
        "quality_gated_old_novelty",
        "quality_gated_old_novelty_sourcecap",
        "tiered_cluster_cap",
    }:
        raise ValueError(
            "ranking.reranker_method must be 'cluster_aware', 'cluster_cap', "
            "'parent_prefix_cluster_cap', 'quality_only', 'quality_gated_old_novelty', "
            "'quality_gated_old_novelty_sourcecap', 'tiered_cluster_cap', or 'mmr'."
        )
    for key in ("k_old", "k_new_density"):
        if int(ranking.get(key, 0)) <= 0:
            raise ValueError(f"ranking.{key} must be a positive integer.")
    novelty_weight = float(ranking.get("novelty_weight", 0.75))
    mmr_lambda = float(ranking.get("mmr_lambda", 0.25))
    cluster_bonus_weight = float(ranking.get("cluster_bonus_weight", 0.25))
    cluster_cap_top_k = int(ranking.get("cluster_cap_top_k", 200))
    cluster_max_per_cluster = int(ranking.get("cluster_max_per_cluster", 3))
    cluster_cap_key = str(ranking.get("cluster_cap_key", "new_cluster_id"))
    cluster_cap_min_quality = float(ranking.get("cluster_cap_min_quality", 0.0))
    prefix_cluster_cap_top_k = int(ranking.get("prefix_cluster_cap_top_k", 75))
    prefix_cluster_cap_key = str(ranking.get("prefix_cluster_cap_key", "new_cluster_parent_id"))
    prefix_cluster_max_per_cluster = int(ranking.get("prefix_cluster_max_per_cluster", cluster_max_per_cluster))
    cluster_cap_schedule = ranking.get("cluster_cap_schedule")
    cluster_similarity_threshold = float(ranking.get("cluster_similarity_threshold", 0.985))
    quality_gate_threshold = float(ranking.get("quality_gate_threshold", 0.45))
    max_stationary_fraction = ranking.get("max_stationary_fraction")
    max_abs_value = ranking.get("max_abs_value")
    source_cap = ranking.get("source_cap")
    source_cap_key = str(ranking.get("source_cap_key", "source_group_id"))
    if not 0.0 <= novelty_weight <= 1.0:
        raise ValueError("ranking.novelty_weight must be in [0, 1].")
    if not 0.0 <= mmr_lambda <= 1.0:
        raise ValueError("ranking.mmr_lambda must be in [0, 1].")
    if cluster_bonus_weight < 0.0:
        raise ValueError("ranking.cluster_bonus_weight must be non-negative.")
    if cluster_cap_top_k <= 0:
        raise ValueError("ranking.cluster_cap_top_k must be positive.")
    if cluster_max_per_cluster <= 0:
        raise ValueError("ranking.cluster_max_per_cluster must be positive.")
    if cluster_cap_key not in {"new_cluster_id", "new_cluster_parent_id"}:
        raise ValueError("ranking.cluster_cap_key must be 'new_cluster_id' or 'new_cluster_parent_id'.")
    if not 0.0 <= cluster_cap_min_quality <= 1.0:
        raise ValueError("ranking.cluster_cap_min_quality must be in [0, 1].")
    if prefix_cluster_cap_top_k <= 0:
        raise ValueError("ranking.prefix_cluster_cap_top_k must be positive.")
    if prefix_cluster_cap_key not in {"new_cluster_id", "new_cluster_parent_id"}:
        raise ValueError("ranking.prefix_cluster_cap_key must be 'new_cluster_id' or 'new_cluster_parent_id'.")
    if prefix_cluster_max_per_cluster <= 0:
        raise ValueError("ranking.prefix_cluster_max_per_cluster must be positive.")
    if cluster_cap_schedule is not None:
        if not isinstance(cluster_cap_schedule, list) or not cluster_cap_schedule:
            raise ValueError("ranking.cluster_cap_schedule must be a non-empty list when provided.")
        previous_top_k = 0
        for tier in cluster_cap_schedule:
            if not isinstance(tier, dict):
                raise ValueError("ranking.cluster_cap_schedule tiers must be objects.")
            top_k = int(tier.get("top_k", 0))
            max_per_cluster = int(tier.get("max_per_cluster", 0))
            if top_k <= 0:
                raise ValueError("ranking.cluster_cap_schedule tiers must have positive top_k.")
            if max_per_cluster <= 0:
                raise ValueError("ranking.cluster_cap_schedule tiers must have positive max_per_cluster.")
            if top_k <= previous_top_k:
                raise ValueError("ranking.cluster_cap_schedule top_k values must be strictly increasing.")
            previous_top_k = top_k
    if "embedding_load_workers" in ranking and int(ranking.get("embedding_load_workers", 0)) <= 0:
        raise ValueError("ranking.embedding_load_workers must be positive when provided.")
    if "min_query_samples" in ranking and int(ranking.get("min_query_samples", 0)) <= 0:
        raise ValueError("ranking.min_query_samples must be positive when provided.")
    if not -1.0 <= cluster_similarity_threshold <= 1.0:
        raise ValueError("ranking.cluster_similarity_threshold must be in [-1, 1].")
    if not 0.0 <= quality_gate_threshold <= 1.0:
        raise ValueError("ranking.quality_gate_threshold must be in [0, 1].")
    if max_stationary_fraction is not None and not 0.0 <= float(max_stationary_fraction) <= 1.0:
        raise ValueError("ranking.max_stationary_fraction must be in [0, 1] when provided.")
    if max_abs_value is not None and float(max_abs_value) < 0.0:
        raise ValueError("ranking.max_abs_value must be non-negative when provided.")
    if ranking.get("reranker_method") == "quality_gated_old_novelty_sourcecap":
        if source_cap is None or int(source_cap) <= 0:
            raise ValueError("ranking.source_cap must be positive for quality_gated_old_novelty_sourcecap.")
        if not source_cap_key:
            raise ValueError("ranking.source_cap_key must be non-empty for source-cap ranking.")
    if int(data.get("feature_dim", 0)) <= 0:
        raise ValueError("data.feature_dim must be positive.")
    if quality and float(quality.get("sample_rate", 30.0)) <= 0:
        raise ValueError("quality.sample_rate must be positive.")
    if quality and int(quality.get("max_samples_per_clip", 1)) <= 0:
        raise ValueError("quality.max_samples_per_clip must be positive when provided.")
    if grammar_features and bool(grammar_features.get("enabled", False)):
        path_template = str(grammar_features.get("path_template", ""))
        if not path_template.startswith("/artifacts/tokens/"):
            raise ValueError("grammar_features.path_template must be under /artifacts/tokens/.")
        if "{mode}" not in path_template:
            raise ValueError("grammar_features.path_template must include '{mode}'.")
        for path_template in _grammar_feature_path_templates(grammar_features):
            if not path_template.startswith("/artifacts/tokens/"):
                raise ValueError("grammar_features path templates must be under /artifacts/tokens/.")
        if bool(grammar_features.get("use_in_score", False)):
            if grammar_features.get("score_variant") not in {"grammar_surprisal_mix", "grammar_phrase_mix", "token_nll_p95", "quality_gated_grammar"}:
                raise ValueError("grammar_features.score_variant must be a supported grammar score.")
            score_weight = float(grammar_features.get("score_weight", -1.0))
            min_quality = float(grammar_features.get("min_quality", -1.0))
            min_new_density_score = float(grammar_features.get("min_new_density_score", -1.0))
            max_score_weight = 1.0 if grammar_features.get("score_variant") == "quality_gated_grammar" else 0.35
            if not 0.0 <= score_weight <= max_score_weight:
                raise ValueError(f"grammar_features.score_weight must be in [0, {max_score_weight}].")
            if not 0.0 <= min_quality <= 1.0:
                raise ValueError("grammar_features.min_quality must be in [0, 1].")
            if not 0.0 <= min_new_density_score <= 1.0:
                raise ValueError("grammar_features.min_new_density_score must be in [0, 1].")
            ablation_report_path = str(grammar_features.get("ablation_report_path", ""))
            if not ablation_report_path.startswith("/artifacts/eval/grammar_ablation/") or not ablation_report_path.endswith(".json"):
                raise ValueError("grammar_features.ablation_report_path must point at the Modal grammar ablation report.")
    if large_cluster_split and bool(large_cluster_split.get("enabled", False)):
        max_cluster_size = int(large_cluster_split.get("max_cluster_size", 0))
        target_subcluster_size = int(large_cluster_split.get("target_subcluster_size", 0))
        score_columns = large_cluster_split.get("score_columns", [])
        split_method = str(large_cluster_split.get("method", large_cluster_split.get("split_method", "feature_kmeans")))
        score_feature_weight = float(large_cluster_split.get("score_feature_weight", 0.0))
        kmeans_iterations = int(large_cluster_split.get("kmeans_iterations", 16))
        if max_cluster_size <= 0:
            raise ValueError("large_cluster_split.max_cluster_size must be positive when enabled.")
        if target_subcluster_size <= 0:
            raise ValueError("large_cluster_split.target_subcluster_size must be positive when enabled.")
        if score_columns is not None and (not isinstance(score_columns, list) or not all(isinstance(column, str) for column in score_columns)):
            raise ValueError("large_cluster_split.score_columns must be a list of strings when provided.")
        if split_method not in {"feature_kmeans", "score_round_robin"}:
            raise ValueError("large_cluster_split.method must be 'feature_kmeans' or 'score_round_robin'.")
        if score_feature_weight < 0.0:
            raise ValueError("large_cluster_split.score_feature_weight must be non-negative.")
        if kmeans_iterations <= 0:
            raise ValueError("large_cluster_split.kmeans_iterations must be positive.")
    if corruption_eval and bool(corruption_eval.get("enabled", False)):
        sample_size = int(corruption_eval.get("sample_size", 0))
        raw_signal = bool(corruption_eval.get("raw_signal", False))
        quality_score = float(corruption_eval.get("quality_score", 0.05 if raw_signal else -1.0))
        modes = corruption_eval.get("modes", [])
        if sample_size <= 0:
            raise ValueError("corruption_eval.sample_size must be positive when enabled.")
        if not 0.0 <= quality_score <= 1.0:
            raise ValueError("corruption_eval.quality_score must be in [0, 1].")
        if not isinstance(modes, list) or not modes:
            raise ValueError("corruption_eval.modes must be a non-empty list when enabled.")
        supported_modes = {"flatline", "spike", "saturation", "jitter"}
        unsupported = sorted({str(mode) for mode in modes} - supported_modes)
        if unsupported:
            raise ValueError(f"corruption_eval.modes contains unsupported modes: {unsupported}.")
    if learned_ranker and bool(learned_ranker.get("enabled", False)):
        model_path = str(learned_ranker.get("model_path", ""))
        score_weight = float(learned_ranker.get("score_weight", -1.0))
        score_transform = str(learned_ranker.get("score_transform", "sigmoid"))
        if not model_path.startswith("/artifacts/") or not model_path.endswith(".json"):
            raise ValueError("learned_ranker.model_path must point at a JSON artifact under /artifacts/.")
        if not 0.0 <= score_weight <= 1.0:
            raise ValueError("learned_ranker.score_weight must be in [0, 1].")
        if score_transform not in {"sigmoid", "minmax"}:
            raise ValueError("learned_ranker.score_transform must be 'sigmoid' or 'minmax'.")
    if score_guards:
        stationary_singleton = score_guards.get("stationary_singleton", {})
        if stationary_singleton and not isinstance(stationary_singleton, dict):
            raise ValueError("score_guards.stationary_singleton must be an object when provided.")
        if isinstance(stationary_singleton, dict) and bool(stationary_singleton.get("enabled", False)):
            for key in ("stationary_threshold", "max_new_density_score", "min_grammar_score", "penalty_multiplier"):
                value = float(stationary_singleton.get(key, -1.0))
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"score_guards.stationary_singleton.{key} must be in [0, 1].")


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Ranking config must include a '{key}' object.")
    return value


def _grammar_feature_path_templates(grammar_features: dict[str, Any]) -> list[str]:
    path_template = str(grammar_features.get("path_template", ""))
    templates = [path_template] if path_template else []
    extra_templates = grammar_features.get("extra_path_templates", [])
    if extra_templates:
        if not isinstance(extra_templates, list):
            raise ValueError("grammar_features.extra_path_templates must be a list when provided.")
        templates.extend(str(template) for template in extra_templates)
    if not templates:
        raise ValueError("grammar_features.path_template is required when grammar features are enabled.")
    return templates
