import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.active_benchmark import (
    BenchmarkClip,
    EpisodeSpec,
    OfflineBenchmarkConfig,
    build_difficulty_targeted_episodes,
    build_opportunity_targeted_episodes,
    build_source_family_label_holdout_episodes,
    build_source_family_shift_episodes,
    build_source_blocked_episodes,
    run_offline_active_benchmark,
    write_benchmark_reports,
)
from scripts.offline_active_benchmark_from_urls import _attach_ts2vec_embeddings, _build_episodes_from_clips, _select_group_balanced_urls


class OfflineActiveBenchmarkTests(unittest.TestCase):
    def test_old_window_focus_gcp_config_is_window_only_and_bounded(self):
        config_path = Path("configs/offline_active_benchmark_gcp_old_window_focus_exact.json")
        self.assertTrue(config_path.exists())
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertTrue(config["execution"]["no_gpu"])
        self.assertTrue(config["execution"]["no_training"])
        self.assertNotIn("ts2vec", config)

        data = config["data"]
        benchmark = config["benchmark"]
        policies = benchmark["policies"]
        random_replays = [policy for policy in policies if policy.startswith("random_valid_replay_")]

        self.assertEqual(config["acceptance"]["minimum_completed_seeds"], 5)
        self.assertEqual(len(data["selection_seeds"]), 5)
        self.assertEqual(len(random_replays), 50)
        self.assertEqual(random_replays[0], "random_valid_replay_000")
        self.assertEqual(random_replays[-1], "random_valid_replay_049")
        self.assertNotIn("ts2vec", benchmark["representations"])
        self.assertNotIn("ts2vec", benchmark["primary_representations"])
        self.assertEqual(benchmark["representations"], ["window", "raw_shape_stats"])
        self.assertEqual(benchmark["primary_representations"], ["window", "raw_shape_stats"])
        self.assertEqual(benchmark["blend_left_representation"], "window")
        self.assertEqual(benchmark["blend_right_representation"], "window")
        self.assertEqual(benchmark["episode_strategy"], "opportunity")
        self.assertEqual(benchmark["episode_representation"], "window")

        required_policies = {
            "random_valid",
            "quality_only",
            "old_novelty_window",
            "old_novelty_window_sourcecap2",
            "kcenter_quality_gated_window",
            "submitted_minus_ts2vec",
            "submitted_no_kcenter",
            "window_novelty_same_gates_no_kcenter",
            "oracle_greedy_eval_only",
        }
        ts2vec_premise_policies = {
            "blend_kcenter_ts2vec_window",
            "artifact_gate_blend_kcenter_ts2vec_window",
            "submitted_full_replay",
            "submitted_minus_window",
            "ts2vec_novelty_same_gates_no_kcenter",
        }
        self.assertTrue(required_policies <= set(policies))
        self.assertFalse(ts2vec_premise_policies & set(policies))

        candidate_clip_count = int(benchmark["candidate_groups_per_episode"]) * int(data["clips_per_group"])
        final_budget = int(benchmark["rounds"]) * int(benchmark["batch_size"])
        self.assertLessEqual(
            math.comb(candidate_clip_count, final_budget),
            int(benchmark["oracle_exact_combination_limit"]),
        )

    def test_source_shift_ts2vec_gcp_config_is_frozen_and_bounded(self):
        config_path = Path("configs/offline_active_benchmark_gcp_source_shift_ts2vec_exact.json")
        self.assertTrue(config_path.exists())
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertTrue(config["execution"]["no_gpu"])
        self.assertTrue(config["execution"]["no_training"])
        self.assertIn("ts2vec", config)

        data = config["data"]
        benchmark = config["benchmark"]
        policies = benchmark["policies"]
        random_replays = [policy for policy in policies if policy.startswith("random_valid_replay_")]

        self.assertEqual(config["acceptance"]["minimum_completed_seeds"], 5)
        self.assertEqual(len(data["selection_seeds"]), 5)
        self.assertEqual(len(random_replays), 50)
        self.assertEqual(benchmark["episode_strategy"], "source_family_shift")
        self.assertEqual(benchmark["episode_representation"], "window")
        self.assertEqual(benchmark["source_family_count"], 4)
        self.assertIn("ts2vec", benchmark["representations"])
        self.assertEqual(benchmark["primary_representations"], ["window", "raw_shape_stats"])
        self.assertEqual(benchmark["blend_left_representation"], "ts2vec")
        self.assertEqual(benchmark["blend_right_representation"], "window")

        required_policies = {
            "random_valid",
            "quality_only",
            "old_novelty_window",
            "old_novelty_window_sourcecap2",
            "old_novelty_ts2vec",
            "kcenter_quality_gated_window",
            "kcenter_quality_gated_ts2vec",
            "blend_kcenter_ts2vec_window",
            "artifact_gate_blend_kcenter_ts2vec_window",
            "submitted_full_replay",
            "submitted_minus_ts2vec",
            "submitted_minus_window",
            "submitted_no_kcenter",
            "window_novelty_same_gates_no_kcenter",
            "ts2vec_novelty_same_gates_no_kcenter",
            "oracle_greedy_eval_only",
        }
        self.assertTrue(required_policies <= set(policies))

        candidate_clip_count = int(benchmark["candidate_groups_per_episode"]) * int(data["clips_per_group"])
        final_budget = int(benchmark["rounds"]) * int(benchmark["batch_size"])
        self.assertLessEqual(
            math.comb(candidate_clip_count, final_budget),
            int(benchmark["oracle_exact_combination_limit"]),
        )

    def test_source_shift_ts2vec_v2_gcp_config_expands_candidates_but_stays_exact(self):
        config_path = Path("configs/offline_active_benchmark_gcp_source_shift_ts2vec_v2_exact.json")
        self.assertTrue(config_path.exists())
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertTrue(config["execution"]["no_gpu"])
        self.assertTrue(config["execution"]["no_training"])
        self.assertIn("ts2vec", config)

        data = config["data"]
        benchmark = config["benchmark"]
        policies = benchmark["policies"]
        random_replays = [policy for policy in policies if policy.startswith("random_valid_replay_")]

        self.assertEqual(config["acceptance"]["minimum_completed_seeds"], 5)
        self.assertEqual(len(data["selection_seeds"]), 5)
        self.assertEqual(len(random_replays), 50)
        self.assertEqual(random_replays[0], "random_valid_replay_000")
        self.assertEqual(random_replays[-1], "random_valid_replay_049")
        self.assertEqual(benchmark["episode_strategy"], "source_family_shift")
        self.assertEqual(benchmark["candidate_groups_per_episode"], 6)
        self.assertEqual(data["clips_per_group"], 3)
        self.assertEqual(benchmark["rounds"], 2)
        self.assertEqual(benchmark["batch_size"], 2)
        self.assertIn("ts2vec", benchmark["representations"])
        self.assertEqual(benchmark["primary_representations"], ["window", "raw_shape_stats"])
        self.assertEqual(benchmark["blend_left_representation"], "ts2vec")
        self.assertEqual(benchmark["blend_right_representation"], "window")

        candidate_clip_count = int(benchmark["candidate_groups_per_episode"]) * int(data["clips_per_group"])
        final_budget = int(benchmark["rounds"]) * int(benchmark["batch_size"])
        self.assertEqual(candidate_clip_count, 18)
        self.assertEqual(final_budget, 4)
        self.assertLessEqual(
            math.comb(candidate_clip_count, final_budget),
            int(benchmark["oracle_exact_combination_limit"]),
        )

        required_checks = set(config["acceptance"]["required_checks"])
        self.assertIn("support deltas are exactly +2", required_checks)
        self.assertIn("candidate deltas are exactly -2", required_checks)
        self.assertIn("near-zero oracle sensitivity improves over source_shift_ts2vec_exact_20260506T174755Z", required_checks)

    def test_synthetic_two_round_benchmark_tracks_policy_state_and_reports(self):
        clips = _synthetic_clips()
        episodes = build_source_blocked_episodes(
            clips,
            n_folds=2,
            candidate_groups_per_episode=2,
            target_groups_per_episode=2,
            max_support_groups=4,
        )
        config = OfflineBenchmarkConfig(
            batch_size=2,
            rounds=2,
            policies=[
                "random_valid",
                "quality_only",
                "old_novelty_window",
                "kcenter_quality_gated_window",
                "oracle_greedy_eval_only",
            ],
            representations=["window"],
            primary_representations=["window"],
            random_seed=13,
        )

        result = run_offline_active_benchmark(clips, episodes, config)

        self.assertEqual(len(result.episodes), 2)
        self.assertEqual(
            set(result.policies),
            {
                "random_valid",
                "quality_only",
                "old_novelty_window",
                "kcenter_quality_gated_window",
                "oracle_greedy_eval_only",
            },
        )
        for episode in result.episodes:
            self.assertFalse(set(episode.support_group_ids) & set(episode.candidate_group_ids))
            self.assertFalse(set(episode.support_group_ids) & set(episode.target_group_ids))
            self.assertFalse(set(episode.candidate_group_ids) & set(episode.target_group_ids))

        by_key = {
            (round_result.episode_id, round_result.policy_name, round_result.round_index): round_result
            for round_result in result.rounds
        }
        for episode in result.episodes:
            for policy_name in result.policies:
                for round_index in range(2):
                    round_result = by_key[(episode.episode_id, policy_name, round_index)]
                    self.assertEqual(len(round_result.selected_ids), 2)
                    self.assertEqual(
                        round_result.support_count_after,
                        round_result.support_count_before + 2,
                    )
                    self.assertEqual(
                        round_result.candidate_count_after,
                        round_result.candidate_count_before - 2,
                    )
                    self.assertIn("window", round_result.coverage_by_representation)
                    self.assertGreaterEqual(round_result.oracle_fraction, 0.0)
                    self.assertLessEqual(round_result.oracle_fraction, 1.0)

                self.assertNotEqual(
                    by_key[(episode.episode_id, policy_name, 0)].support_ids_before,
                    by_key[(episode.episode_id, policy_name, 1)].support_ids_before,
                )

            for round_index in range(2):
                oracle = by_key[(episode.episode_id, "oracle_greedy_eval_only", round_index)]
                for policy_name in result.policies:
                    challenger = by_key[(episode.episode_id, policy_name, round_index)]
                    self.assertGreaterEqual(
                        oracle.cumulative_balanced_relative_gain + 1.0e-12,
                        challenger.cumulative_balanced_relative_gain,
                    )

        self.assertTrue(result.policy_summary["oracle_greedy_eval_only"]["mean_balanced_relative_gain"] >= 0.0)
        self.assertEqual(len(result.difficulty_audit), 2)
        for episode_difficulty in result.difficulty_audit:
            self.assertIn("support_target_baseline_distance_by_representation", episode_difficulty)
            self.assertIn("candidate_target_nearest_distance_by_representation", episode_difficulty)
            self.assertIn("oracle_greedy_cumulative_gain_by_round", episode_difficulty)
            self.assertIn("near_zero_oracle_round_fraction", episode_difficulty)
            self.assertIn("window", episode_difficulty["support_target_baseline_distance_by_representation"])

        with TemporaryDirectory() as tmp:
            paths = write_benchmark_reports(result, tmp)
            report = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

        self.assertEqual(report["n_episodes"], 2)
        self.assertEqual(len(report["difficulty_audit"]), 2)
        self.assertIn("kcenter_quality_gated_window", report["policy_summary"])
        self.assertIn("## Difficulty Audit", markdown)
        self.assertIn("## Acquisition Curves", markdown)
        self.assertIn("oracle_greedy_eval_only", markdown)
        self.assertTrue(Path("configs/offline_active_benchmark_smoke.json").exists())

    def test_quality_stratified_random_samples_from_quality_matched_stratum(self):
        clips = [
            _quality_clip("support_a", "support_a", [0.0, 0.0], quality=0.90),
            _quality_clip("target_a", "target_a", [1.0, 1.0], quality=0.90),
            _quality_clip("top_a", "candidate_a", [2.0, 0.0], quality=0.99),
            _quality_clip("top_b", "candidate_b", [2.1, 0.0], quality=0.99),
            _quality_clip("top_c", "candidate_c", [2.2, 0.0], quality=0.99),
            _quality_clip("top_d", "candidate_d", [2.3, 0.0], quality=0.99),
            _quality_clip("low_a", "candidate_e", [9.0, 0.0], quality=0.20),
            _quality_clip("low_b", "candidate_f", [9.1, 0.0], quality=0.10),
        ]
        episode = EpisodeSpec(
            episode_id="episode_000",
            fold_id=0,
            support_ids=("support_a",),
            candidate_ids=("top_a", "top_b", "top_c", "top_d", "low_a", "low_b"),
            target_ids=("target_a",),
            support_group_ids=("support_a",),
            candidate_group_ids=("candidate_a", "candidate_b", "candidate_c", "candidate_d", "candidate_e", "candidate_f"),
            target_group_ids=("target_a",),
        )

        result = run_offline_active_benchmark(
            clips,
            (episode,),
            OfflineBenchmarkConfig(
                batch_size=2,
                rounds=1,
                policies=("quality_only", "quality_stratified_random"),
                representations=("window",),
                primary_representations=("window",),
                random_seed=3,
                quality_threshold=0.0,
            ),
        )

        by_policy = {row.policy_name: row.selected_ids for row in result.rounds}
        self.assertEqual(by_policy["quality_only"], ("top_a", "top_b"))
        self.assertTrue(set(by_policy["quality_stratified_random"]) <= {"top_a", "top_b", "top_c", "top_d"})
        self.assertNotEqual(by_policy["quality_stratified_random"], by_policy["quality_only"])

    def test_url_sampling_randomizes_workers_and_clips_reproducibly(self):
        with TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "urls.txt"
            urls = [
                f"https://storage.googleapis.com/buildai-imu-benchmark-v1-preexisting/pretrain/worker{worker:05d}/clip{clip:03d}.txt"
                for worker in range(12)
                for clip in range(8)
            ]
            manifest.write_text("\n".join(urls) + "\n", encoding="utf-8")

            selected_a = _select_group_balanced_urls(
                manifest,
                max_rows=20,
                max_groups=5,
                clips_per_group=4,
                sampling_mode="random",
                selection_seed=7,
            )
            selected_b = _select_group_balanced_urls(
                manifest,
                max_rows=20,
                max_groups=5,
                clips_per_group=4,
                sampling_mode="random",
                selection_seed=7,
            )
            selected_c = _select_group_balanced_urls(
                manifest,
                max_rows=20,
                max_groups=5,
                clips_per_group=4,
                sampling_mode="random",
                selection_seed=11,
            )
            prefix_selected = _select_group_balanced_urls(
                manifest,
                max_rows=20,
                max_groups=5,
                clips_per_group=4,
                sampling_mode="first",
                selection_seed=7,
            )

        self.assertEqual(selected_a, selected_b)
        self.assertNotEqual(selected_a, selected_c)
        self.assertNotEqual(selected_a, prefix_selected)
        self.assertEqual(len(selected_a), 20)
        self.assertEqual(len({_worker_id(url) for url in selected_a}), 5)
        self.assertEqual(
            sorted({sum(1 for url in selected_a if _worker_id(url) == worker) for worker in {_worker_id(url) for url in selected_a}}),
            [4],
        )

    def test_difficulty_targeted_episodes_put_candidates_closer_to_targets_than_support(self):
        clips = _separated_source_group_clips()

        episodes = build_difficulty_targeted_episodes(
            clips,
            n_folds=3,
            candidate_groups_per_episode=2,
            target_groups_per_episode=1,
            max_support_groups=2,
            representation="window",
        )

        self.assertEqual(len(episodes), 3)
        for episode in episodes:
            self.assertEqual(len(episode.support_group_ids), 2)
            self.assertEqual(len(episode.candidate_group_ids), 2)
            self.assertEqual(len(episode.target_group_ids), 1)
            self.assertFalse(set(episode.support_group_ids) & set(episode.candidate_group_ids))
            self.assertFalse(set(episode.support_group_ids) & set(episode.target_group_ids))
            self.assertFalse(set(episode.candidate_group_ids) & set(episode.target_group_ids))
            support_distance = _mean_group_distance(clips, episode.target_group_ids, episode.support_group_ids)
            candidate_distance = _mean_group_distance(clips, episode.target_group_ids, episode.candidate_group_ids)
            self.assertLess(candidate_distance, support_distance)

    def test_opportunity_targeted_episodes_raise_oracle_over_random_replay_gap(self):
        clips = _opportunity_source_group_clips()
        hard = build_difficulty_targeted_episodes(
            clips,
            n_folds=1,
            candidate_groups_per_episode=4,
            target_groups_per_episode=1,
            max_support_groups=4,
            representation="window",
        )
        opportunity = build_opportunity_targeted_episodes(
            clips,
            n_folds=1,
            candidate_groups_per_episode=4,
            target_groups_per_episode=1,
            max_support_groups=4,
            representation="window",
        )

        hard_gap = _oracle_minus_random_replay_mean(clips, hard)
        opportunity_gap = _oracle_minus_random_replay_mean(clips, opportunity)

        self.assertGreater(opportunity_gap, hard_gap + 0.15)
        candidate_distances = _group_distances_to_targets(clips, opportunity[0].target_group_ids, opportunity[0].candidate_group_ids)
        hard_candidate_distances = _group_distances_to_targets(clips, hard[0].target_group_ids, hard[0].candidate_group_ids)
        self.assertLess(min(candidate_distances), 0.05)
        self.assertGreater(max(candidate_distances), max(hard_candidate_distances) + 0.50)

    def test_source_family_shift_episodes_hold_out_target_family(self):
        clips = _source_family_shift_clips()

        episodes = build_source_family_shift_episodes(
            clips,
            n_folds=4,
            candidate_groups_per_episode=4,
            target_groups_per_episode=2,
            max_support_groups=4,
            representation="window",
            source_family_count=4,
        )

        self.assertEqual(len(episodes), 4)
        for episode in episodes:
            target_families = {_literal_family(group) for group in episode.target_group_ids}
            candidate_families = {_literal_family(group) for group in episode.candidate_group_ids}
            support_families = {_literal_family(group) for group in episode.support_group_ids}
            self.assertEqual(len(target_families), 1)
            self.assertFalse(target_families & candidate_families)
            self.assertFalse(target_families & support_families)
            self.assertFalse(set(episode.support_group_ids) & set(episode.candidate_group_ids))
            self.assertFalse(set(episode.support_group_ids) & set(episode.target_group_ids))
            self.assertFalse(set(episode.candidate_group_ids) & set(episode.target_group_ids))
            self.assertEqual(len(episode.target_group_ids), 2)
            self.assertEqual(len(episode.candidate_group_ids), 4)
            self.assertEqual(len(episode.support_group_ids), 4)

    def test_source_family_shift_skips_target_families_that_would_exhaust_support(self):
        clips = _imbalanced_source_family_shift_clips()

        episodes = build_source_family_shift_episodes(
            clips,
            n_folds=2,
            candidate_groups_per_episode=4,
            target_groups_per_episode=2,
            max_support_groups=4,
            representation="window",
            source_family_count=3,
        )

        self.assertEqual(len(episodes), 2)
        for episode in episodes:
            self.assertNotIn("familyA", {_literal_family(group) for group in episode.target_group_ids})
            self.assertGreaterEqual(len(episode.support_group_ids), 1)

    def test_source_family_shift_adapts_cluster_count_for_unlabeled_workers(self):
        clips = _clustered_worker_shift_clips()

        episodes = build_source_family_shift_episodes(
            clips,
            n_folds=2,
            candidate_groups_per_episode=4,
            target_groups_per_episode=2,
            max_support_groups=4,
            representation="window",
            source_family_count=4,
        )

        self.assertEqual(len(episodes), 2)
        for episode in episodes:
            self.assertEqual(len(episode.target_group_ids), 2)
            self.assertEqual(len(episode.candidate_group_ids), 4)
            self.assertGreaterEqual(len(episode.support_group_ids), 1)

    def test_source_family_label_holdout_candidates_include_target_family_not_support(self):
        clips = _source_family_label_holdout_clips()

        episodes = build_source_family_label_holdout_episodes(
            clips,
            n_folds=2,
            candidate_groups_per_episode=4,
            target_groups_per_episode=2,
            target_candidate_groups_per_episode=2,
            max_support_groups=4,
            representation="window",
            source_family_count=4,
        )

        self.assertEqual(len(episodes), 2)
        for episode in episodes:
            target_families = {_literal_family(group) for group in episode.target_group_ids}
            candidate_families = {_literal_family(group) for group in episode.candidate_group_ids}
            support_families = {_literal_family(group) for group in episode.support_group_ids}

            self.assertEqual(len(target_families), 1)
            self.assertTrue(target_families & candidate_families)
            self.assertFalse(target_families & support_families)
            self.assertEqual(len([group for group in episode.candidate_group_ids if _literal_family(group) in target_families]), 2)
            self.assertFalse(set(episode.support_group_ids) & set(episode.candidate_group_ids))
            self.assertFalse(set(episode.support_group_ids) & set(episode.target_group_ids))
            self.assertFalse(set(episode.candidate_group_ids) & set(episode.target_group_ids))

    def test_url_runner_can_select_difficulty_targeted_episode_strategy(self):
        clips = _separated_source_group_clips()

        hard = _build_episodes_from_clips(
            clips,
            episode_strategy="hard",
            folds=2,
            candidate_groups_per_episode=2,
            target_groups_per_episode=1,
            max_support_groups=2,
            episode_representation="window",
        )
        rotating = _build_episodes_from_clips(
            clips,
            episode_strategy="rotating",
            folds=2,
            candidate_groups_per_episode=2,
            target_groups_per_episode=1,
            max_support_groups=2,
            episode_representation="window",
        )

        self.assertNotEqual(hard[0].target_group_ids, rotating[0].target_group_ids)
        self.assertLess(
            _mean_group_distance(clips, hard[0].target_group_ids, hard[0].candidate_group_ids),
            _mean_group_distance(clips, hard[0].target_group_ids, hard[0].support_group_ids),
        )

        opportunity = _build_episodes_from_clips(
            clips,
            episode_strategy="opportunity",
            folds=2,
            candidate_groups_per_episode=2,
            target_groups_per_episode=1,
            max_support_groups=2,
            episode_representation="window",
        )
        self.assertEqual(len(opportunity), 2)
        self.assertFalse(set(opportunity[0].candidate_group_ids) & set(opportunity[0].target_group_ids))

        source_shift = _build_episodes_from_clips(
            _source_family_shift_clips(),
            episode_strategy="source_family_shift",
            folds=2,
            candidate_groups_per_episode=4,
            target_groups_per_episode=2,
            max_support_groups=4,
            episode_representation="window",
            source_family_count=4,
        )
        self.assertEqual(len(source_shift), 2)
        self.assertFalse({_literal_family(group) for group in source_shift[0].target_group_ids} & {_literal_family(group) for group in source_shift[0].candidate_group_ids})

        label_holdout = _build_episodes_from_clips(
            _source_family_label_holdout_clips(),
            episode_strategy="source_family_label_holdout",
            folds=2,
            candidate_groups_per_episode=4,
            target_groups_per_episode=2,
            max_support_groups=4,
            episode_representation="window",
            source_family_count=4,
        )
        self.assertEqual(len(label_holdout), 2)
        label_holdout_target_families = {_literal_family(group) for group in label_holdout[0].target_group_ids}
        self.assertTrue(label_holdout_target_families & {_literal_family(group) for group in label_holdout[0].candidate_group_ids})
        self.assertFalse(label_holdout_target_families & {_literal_family(group) for group in label_holdout[0].support_group_ids})

    def test_progress_events_and_oracle_candidate_cap_are_exposed(self):
        clips = _candidate_cap_clips()
        episodes = (_candidate_cap_episode(clips),)
        config = OfflineBenchmarkConfig(
            batch_size=2,
            rounds=2,
            policies=["oracle_greedy_eval_only"],
            representations=["window"],
            primary_representations=["window"],
            oracle_candidate_cap=3,
            oracle_exact_combination_limit=10,
        )
        events: list[dict[str, object]] = []

        result = run_offline_active_benchmark(clips, episodes, config, progress_callback=events.append)

        self.assertEqual(result.policy_summary["oracle_greedy_eval_only"]["round_count"], 2.0)
        self.assertEqual(
            [event["event"] for event in events if event["event"] in {"episode_start", "policy_start", "round_done", "benchmark_done"}],
            ["episode_start", "policy_start", "round_done", "round_done", "benchmark_done"],
        )
        round_events = [event for event in events if event["event"] == "round_done"]
        self.assertEqual(round_events[0]["oracle_candidate_cap"], 3)
        self.assertEqual(round_events[0]["candidate_count_before"], 18)
        self.assertEqual(round_events[0]["selected_count"], 2)
        selected = [round_result.selected_ids for round_result in result.rounds if round_result.policy_name == "oracle_greedy_eval_only"]
        self.assertTrue(any("candidate_good" in sample_id for batch in selected for sample_id in batch))

    def test_exact_oracle_calibration_marks_round_denominators_exact(self):
        clips = _candidate_cap_clips()
        episodes = (_candidate_cap_episode(clips),)
        config = OfflineBenchmarkConfig(
            batch_size=2,
            rounds=2,
            policies=["random_valid", "oracle_greedy_eval_only"],
            representations=["window"],
            primary_representations=["window"],
            oracle_candidate_cap=None,
            oracle_exact_combination_limit=10_000,
        )

        result = run_offline_active_benchmark(clips, episodes, config)

        self.assertEqual(len(result.difficulty_audit), 1)
        audit = result.difficulty_audit[0]
        self.assertTrue(audit["oracle_fraction_exact_all_rounds"])
        self.assertEqual(audit["oracle_fraction_exact_by_round"], [True, True])
        self.assertEqual(
            [row["combination_count"] for row in audit["oracle_fraction_budget_audit"]],
            [153, 3060],
        )
        self.assertEqual(
            [row["budget"] for row in audit["oracle_fraction_budget_audit"]],
            [2, 4],
        )

    def test_random_replay_policies_are_supported_and_seeded_independently(self):
        clips = _candidate_cap_clips()
        episodes = (_candidate_cap_episode(clips),)
        config = OfflineBenchmarkConfig(
            batch_size=2,
            rounds=1,
            policies=["random_valid", "random_valid_replay_000", "random_valid_replay_001"],
            representations=["window"],
            primary_representations=["window"],
            random_seed=101,
        )

        result_a = run_offline_active_benchmark(clips, episodes, config)
        result_b = run_offline_active_benchmark(clips, episodes, config)

        selected_a = {row.policy_name: row.selected_ids for row in result_a.rounds}
        selected_b = {row.policy_name: row.selected_ids for row in result_b.rounds}
        self.assertEqual(selected_a, selected_b)
        self.assertIn("random_valid_replay_000", result_a.policy_summary)
        self.assertIn("random_valid_replay_001", result_a.policy_summary)
        self.assertNotEqual(selected_a["random_valid_replay_000"], selected_a["random_valid_replay_001"])

    def test_artifact_gate_blend_policy_demotes_artifacts_after_ts2vec_window_blend(self):
        clips = _blend_policy_clips()
        episode = _blend_policy_episode(clips)
        config = OfflineBenchmarkConfig(
            batch_size=1,
            rounds=1,
            policies=["artifact_gate_blend_kcenter_ts2vec_window"],
            representations=["ts2vec", "window"],
            primary_representations=["window"],
            blend_left_representation="ts2vec",
            blend_right_representation="window",
            blend_alpha=0.5,
            max_artifact_score=0.05,
        )

        result = run_offline_active_benchmark(clips, (episode,), config)

        row = result.rounds[0]
        self.assertEqual(row.policy_name, "artifact_gate_blend_kcenter_ts2vec_window")
        self.assertEqual(row.selected_ids, ("candidate_clean_clip000",))
        self.assertIn("artifact_gate_blend_kcenter_ts2vec_window", result.policy_summary)

    def test_source_capped_old_novelty_forces_batch_source_diversity(self):
        clips = _source_cap_clips()
        episode = _source_cap_episode(clips)
        config = OfflineBenchmarkConfig(
            batch_size=3,
            rounds=1,
            policies=["old_novelty_window", "old_novelty_window_sourcecap2"],
            representations=["window"],
            primary_representations=["window"],
        )

        result = run_offline_active_benchmark(clips, (episode,), config)

        by_policy = {row.policy_name: row for row in result.rounds}
        self.assertEqual(
            by_policy["old_novelty_window"].selected_source_group_ids,
            ("candidate_a", "candidate_a", "candidate_a"),
        )
        self.assertEqual(
            by_policy["old_novelty_window_sourcecap2"].selected_source_group_ids,
            ("candidate_a", "candidate_a", "candidate_b"),
        )
        self.assertLessEqual(
            max(by_policy["old_novelty_window_sourcecap2"].selected_source_group_ids.count(group) for group in {"candidate_a", "candidate_b"}),
            2,
        )

    def test_submitted_style_ablation_policies_separate_representation_and_kcenter(self):
        clips = _submitted_ablation_clips()
        episode = _submitted_ablation_episode(clips)
        config = OfflineBenchmarkConfig(
            batch_size=2,
            rounds=1,
            policies=[
                "submitted_full_replay",
                "submitted_no_kcenter",
                "submitted_minus_ts2vec",
                "submitted_minus_window",
                "window_novelty_same_gates_no_kcenter",
                "ts2vec_novelty_same_gates_no_kcenter",
            ],
            representations=["ts2vec", "window"],
            primary_representations=["window", "ts2vec"],
            blend_left_representation="ts2vec",
            blend_right_representation="window",
            blend_alpha=0.5,
        )

        result = run_offline_active_benchmark(clips, (episode,), config)

        by_policy = {row.policy_name: row for row in result.rounds}
        self.assertEqual(by_policy["submitted_no_kcenter"].selected_ids, ("candidate_a_clip000", "candidate_a_clip001"))
        self.assertEqual(by_policy["submitted_full_replay"].selected_ids, ("candidate_a_clip000", "candidate_b_clip000"))
        self.assertEqual(by_policy["submitted_minus_ts2vec"].selected_ids, ("candidate_a_clip000", "candidate_b_clip000"))
        self.assertEqual(by_policy["submitted_minus_window"].selected_ids, ("candidate_b_clip000", "candidate_a_clip000"))
        self.assertEqual(by_policy["window_novelty_same_gates_no_kcenter"].selected_ids, ("candidate_a_clip000", "candidate_a_clip001"))
        self.assertEqual(by_policy["ts2vec_novelty_same_gates_no_kcenter"].selected_ids, ("candidate_b_clip000", "candidate_a_clip000"))

    def test_explicit_ts2vec_policy_aliases_match_ts2vec_ablation_behavior(self):
        clips = _submitted_ablation_clips()
        episode = _submitted_ablation_episode(clips)
        config = OfflineBenchmarkConfig(
            batch_size=2,
            rounds=1,
            policies=[
                "old_novelty_ts2vec",
                "kcenter_quality_gated_ts2vec",
                "ts2vec_novelty_same_gates_no_kcenter",
                "submitted_minus_window",
            ],
            representations=["ts2vec", "window"],
            primary_representations=["window", "ts2vec"],
            blend_left_representation="ts2vec",
            blend_right_representation="window",
        )

        result = run_offline_active_benchmark(clips, (episode,), config)

        by_policy = {row.policy_name: row for row in result.rounds}
        self.assertEqual(by_policy["old_novelty_ts2vec"].selected_ids, ("candidate_b_clip000", "candidate_a_clip000"))
        self.assertEqual(by_policy["ts2vec_novelty_same_gates_no_kcenter"].selected_ids, by_policy["old_novelty_ts2vec"].selected_ids)
        self.assertEqual(by_policy["kcenter_quality_gated_ts2vec"].selected_ids, by_policy["submitted_minus_window"].selected_ids)

    def test_selected_clip_audit_records_scores_and_gate_status(self):
        clips = _blend_policy_clips()
        episode = _blend_policy_episode(clips)
        config = OfflineBenchmarkConfig(
            batch_size=1,
            rounds=1,
            policies=["artifact_gate_blend_kcenter_ts2vec_window"],
            representations=["ts2vec", "window"],
            primary_representations=["window"],
            blend_left_representation="ts2vec",
            blend_right_representation="window",
            blend_alpha=0.5,
            max_artifact_score=0.05,
        )

        result = run_offline_active_benchmark(clips, (episode,), config)
        row = result.rounds[0]

        self.assertEqual(len(row.selection_details), 1)
        detail = row.selection_details[0]
        self.assertEqual(detail["sample_id"], "candidate_clean_clip000")
        self.assertEqual(detail["rank_index"], 0)
        self.assertTrue(detail["passed_quality_gate"])
        self.assertTrue(detail["passed_artifact_gate"])
        self.assertIn("old_novelty_window_score", detail)
        self.assertIn("blend_left_novelty_score", detail)
        self.assertIn("blend_right_novelty_score", detail)
        self.assertIn("blend_score", detail)
        self.assertEqual(detail["selected_score"], row.selected_scores[0])

        with TemporaryDirectory() as tmp:
            paths = write_benchmark_reports(result, tmp)
            report = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

        self.assertEqual(report["rounds"][0]["selection_details"][0]["sample_id"], "candidate_clean_clip000")
        self.assertIn("## Selection Audit", markdown)
        self.assertIn("candidate_clean_clip000", markdown)

    def test_ts2vec_embedding_attachment_uses_checkpoint_encoder_batch(self):
        FakeTS2VecEncoder.calls = []
        clips = [
            BenchmarkClip("b", "worker00002", {"window": np.asarray([0.0, 1.0])}),
            BenchmarkClip("a", "worker00001", {"window": np.asarray([1.0, 0.0])}),
        ]
        samples = {
            "a": np.ones((4, 6), dtype=np.float32),
            "b": np.full((4, 6), 2.0, dtype=np.float32),
        }

        updated = _attach_ts2vec_embeddings(
            clips,
            samples_by_id=samples,
            checkpoint_path="/tmp/fake-ts2vec.pt",
            device="cpu",
            batch_size=2,
            encoder_factory=FakeTS2VecEncoder,
        )

        self.assertEqual(FakeTS2VecEncoder.calls, [("/tmp/fake-ts2vec.pt", "cpu")])
        self.assertEqual([clip.sample_id for clip in updated], ["b", "a"])
        np.testing.assert_allclose(updated[0].embeddings["ts2vec"], np.asarray([48.0, 2.0], dtype=np.float32))
        np.testing.assert_allclose(updated[1].embeddings["ts2vec"], np.asarray([24.0, 2.0], dtype=np.float32))


def _synthetic_clips() -> list[BenchmarkClip]:
    centers = {
        "worker00000": 10.0,
        "worker00001": 10.2,
        "worker00002": 10.1,
        "worker00003": -4.0,
        "worker00004": -3.9,
        "worker00005": 20.0,
        "worker00006": 0.0,
        "worker00007": 1.0,
    }
    clips: list[BenchmarkClip] = []
    for group_index, (group_id, center) in enumerate(centers.items()):
        for clip_index in range(5):
            sample_id = f"{group_id}_clip{clip_index:03d}"
            value = center + clip_index * 0.01
            embedding = np.asarray([value, 1.0, value * 0.05], dtype=float)
            clips.append(
                BenchmarkClip(
                    sample_id=sample_id,
                    source_group_id=group_id,
                    embeddings={"window": embedding},
                    quality_score=0.95 - 0.01 * (clip_index % 2),
                    stationary_fraction=0.10 + 0.01 * (group_index % 3),
                    max_abs_value=20.0 + group_index,
                )
            )
    return clips


def _quality_clip(sample_id: str, source_group_id: str, embedding: list[float], *, quality: float) -> BenchmarkClip:
    return BenchmarkClip(
        sample_id=sample_id,
        source_group_id=source_group_id,
        embeddings={"window": np.asarray(embedding, dtype=float)},
        quality_score=float(quality),
    )


def _separated_source_group_clips() -> list[BenchmarkClip]:
    centers = {
        "target_like_a": 10.0,
        "target_like_b": 10.2,
        "target_like_c": 9.8,
        "near_a": 9.9,
        "near_b": 10.1,
        "near_c": 10.3,
        "far_a": -10.0,
        "far_b": -9.5,
        "far_c": -11.0,
    }
    clips: list[BenchmarkClip] = []
    for group_id, center in centers.items():
        for clip_index in range(3):
            sample_id = f"{group_id}_clip{clip_index:03d}"
            embedding = np.asarray([center + clip_index * 0.01, 1.0], dtype=float)
            clips.append(BenchmarkClip(sample_id=sample_id, source_group_id=group_id, embeddings={"window": embedding}))
    return clips


def _opportunity_source_group_clips() -> list[BenchmarkClip]:
    angles_by_group = {
        "target_like_00": 0.0,
        "target_like_01": 2.0,
        "target_like_02": 4.0,
        "target_like_03": 6.0,
        "target_like_04": 8.0,
        "distractor_00": 178.0,
        "distractor_01": 182.0,
        "support_like_00": 170.0,
        "support_like_01": 174.0,
        "support_like_02": 186.0,
        "support_like_03": 190.0,
        "neutral_00": 90.0,
        "neutral_01": 270.0,
    }
    clips: list[BenchmarkClip] = []
    for group_id, angle in angles_by_group.items():
        for clip_index in range(2):
            radians = np.deg2rad(angle + clip_index * 0.25)
            embedding = np.asarray([np.cos(radians), np.sin(radians)], dtype=float)
            clips.append(
                BenchmarkClip(
                    sample_id=f"{group_id}_clip{clip_index:03d}",
                    source_group_id=group_id,
                    embeddings={"window": embedding},
                    quality_score=1.0,
                    stationary_fraction=0.0,
                    max_abs_value=1.0,
                )
            )
    return clips


def _source_family_shift_clips() -> list[BenchmarkClip]:
    centers = {
        "familyA_worker00": np.asarray([1.0, 0.0], dtype=float),
        "familyA_worker01": np.asarray([0.98, 0.08], dtype=float),
        "familyA_worker02": np.asarray([0.96, -0.10], dtype=float),
        "familyA_worker03": np.asarray([0.94, 0.18], dtype=float),
        "familyB_worker00": np.asarray([0.0, 1.0], dtype=float),
        "familyB_worker01": np.asarray([0.08, 0.98], dtype=float),
        "familyB_worker02": np.asarray([-0.10, 0.96], dtype=float),
        "familyB_worker03": np.asarray([0.18, 0.94], dtype=float),
        "familyC_worker00": np.asarray([-1.0, 0.0], dtype=float),
        "familyC_worker01": np.asarray([-0.98, 0.08], dtype=float),
        "familyC_worker02": np.asarray([-0.96, -0.10], dtype=float),
        "familyC_worker03": np.asarray([-0.94, 0.18], dtype=float),
        "familyD_worker00": np.asarray([0.0, -1.0], dtype=float),
        "familyD_worker01": np.asarray([0.08, -0.98], dtype=float),
        "familyD_worker02": np.asarray([-0.10, -0.96], dtype=float),
        "familyD_worker03": np.asarray([0.18, -0.94], dtype=float),
    }
    clips: list[BenchmarkClip] = []
    for group_id, center in centers.items():
        for clip_index in range(2):
            offset = 0.01 * clip_index
            window = np.asarray([center[0] + offset, center[1] - offset], dtype=float)
            ts2vec = np.asarray([center[1] - offset, center[0] + offset], dtype=float)
            clips.append(
                BenchmarkClip(
                    sample_id=f"{group_id}_clip{clip_index:03d}",
                    source_group_id=group_id,
                    embeddings={"window": window, "ts2vec": ts2vec},
                    quality_score=1.0,
                    stationary_fraction=0.0,
                    max_abs_value=1.0,
                )
            )
    return clips


def _source_family_label_holdout_clips() -> list[BenchmarkClip]:
    return _source_family_shift_clips()


def _imbalanced_source_family_shift_clips() -> list[BenchmarkClip]:
    centers: dict[str, np.ndarray] = {}
    for index in range(8):
        centers[f"familyA_worker{index:02d}"] = np.asarray([1.0, 0.02 * index], dtype=float)
    for index in range(2):
        centers[f"familyB_worker{index:02d}"] = np.asarray([0.0, 1.0 - 0.02 * index], dtype=float)
        centers[f"familyC_worker{index:02d}"] = np.asarray([-1.0, 0.02 * index], dtype=float)
    clips: list[BenchmarkClip] = []
    for group_id, center in centers.items():
        clips.append(
            BenchmarkClip(
                sample_id=f"{group_id}_clip000",
                source_group_id=group_id,
                embeddings={"window": center, "ts2vec": center[::-1].copy()},
                quality_score=1.0,
            )
        )
    return clips


def _clustered_worker_shift_clips() -> list[BenchmarkClip]:
    clips: list[BenchmarkClip] = []
    for group_index in range(20):
        angle = np.deg2rad((group_index % 5) * 10.0 + (group_index // 5) * 90.0)
        center = np.asarray([np.cos(angle), np.sin(angle)], dtype=float)
        clips.append(
            BenchmarkClip(
                sample_id=f"worker{group_index:05d}_clip000",
                source_group_id=f"worker{group_index:05d}",
                embeddings={"window": center, "ts2vec": center[::-1].copy()},
                quality_score=1.0,
            )
        )
    return clips


def _candidate_cap_clips() -> list[BenchmarkClip]:
    centers = {
        "support_far_a": -5.0,
        "support_far_b": -4.5,
        "target": 5.0,
        "candidate_bad_a": -4.0,
        "candidate_bad_b": -3.5,
        "candidate_bad_c": -3.0,
        "candidate_bad_d": -2.5,
        "candidate_bad_e": -2.0,
        "candidate_good": 5.05,
    }
    clips: list[BenchmarkClip] = []
    for group_id, center in centers.items():
        for clip_index in range(3):
            sample_id = f"{group_id}_clip{clip_index:03d}"
            embedding = np.asarray([center + clip_index * 0.01, 1.0], dtype=float)
            clips.append(BenchmarkClip(sample_id=sample_id, source_group_id=group_id, embeddings={"window": embedding}))
    return clips


def _candidate_cap_episode(clips: list[BenchmarkClip]) -> EpisodeSpec:
    by_group: dict[str, list[str]] = {}
    for clip in clips:
        by_group.setdefault(clip.source_group_id, []).append(clip.sample_id)
    support_groups = ("support_far_a", "support_far_b")
    candidate_groups = (
        "candidate_bad_a",
        "candidate_bad_b",
        "candidate_bad_c",
        "candidate_bad_d",
        "candidate_bad_e",
        "candidate_good",
    )
    target_groups = ("target",)
    return EpisodeSpec(
        episode_id="candidate_cap_episode",
        fold_id=0,
        support_ids=tuple(sample_id for group in support_groups for sample_id in sorted(by_group[group])),
        candidate_ids=tuple(sample_id for group in candidate_groups for sample_id in sorted(by_group[group])),
        target_ids=tuple(sample_id for group in target_groups for sample_id in sorted(by_group[group])),
        support_group_ids=support_groups,
        candidate_group_ids=candidate_groups,
        target_group_ids=target_groups,
    )


def _blend_policy_clips() -> list[BenchmarkClip]:
    centers = {
        "support": np.asarray([1.0, 0.0], dtype=float),
        "target": np.asarray([0.0, 1.0], dtype=float),
        "candidate_artifact": np.asarray([-1.0, 0.0], dtype=float),
        "candidate_clean": np.asarray([0.0, 1.0], dtype=float),
    }
    clips: list[BenchmarkClip] = []
    for group_id, center in centers.items():
        sample_id = f"{group_id}_clip000"
        clips.append(
            BenchmarkClip(
                sample_id=sample_id,
                source_group_id=group_id,
                embeddings={"ts2vec": center.copy(), "window": center.copy()},
                quality_score=0.99 if group_id.startswith("candidate") else 1.0,
                stationary_fraction=0.0,
                max_abs_value=1.0,
                artifact_score=0.90 if group_id == "candidate_artifact" else 0.0,
            )
        )
    return clips


def _blend_policy_episode(clips: list[BenchmarkClip]) -> EpisodeSpec:
    by_group = {clip.source_group_id: clip.sample_id for clip in clips}
    return EpisodeSpec(
        episode_id="blend_policy",
        fold_id=0,
        support_ids=(by_group["support"],),
        candidate_ids=(by_group["candidate_artifact"], by_group["candidate_clean"]),
        target_ids=(by_group["target"],),
        support_group_ids=("support",),
        candidate_group_ids=("candidate_artifact", "candidate_clean"),
        target_group_ids=("target",),
    )


def _source_cap_clips() -> list[BenchmarkClip]:
    centers = {
        "support": [0.0],
        "target": [1.0],
        "candidate_a": [10.0, 9.0, 8.0],
        "candidate_b": [3.0, 2.0, 1.5],
    }
    clips: list[BenchmarkClip] = []
    for group_id, values in centers.items():
        for clip_index, value in enumerate(values):
            clips.append(
                BenchmarkClip(
                    sample_id=f"{group_id}_clip{clip_index:03d}",
                    source_group_id=group_id,
                    embeddings={"window": np.asarray([float(value), 1.0], dtype=float)},
                    quality_score=1.0,
                )
            )
    return clips


def _source_cap_episode(clips: list[BenchmarkClip]) -> EpisodeSpec:
    by_group: dict[str, list[str]] = {}
    for clip in clips:
        by_group.setdefault(clip.source_group_id, []).append(clip.sample_id)
    return EpisodeSpec(
        episode_id="source_cap",
        fold_id=0,
        support_ids=tuple(sorted(by_group["support"])),
        candidate_ids=tuple(sorted([*by_group["candidate_a"], *by_group["candidate_b"]])),
        target_ids=tuple(sorted(by_group["target"])),
        support_group_ids=("support",),
        candidate_group_ids=("candidate_a", "candidate_b"),
        target_group_ids=("target",),
    )


def _submitted_ablation_clips() -> list[BenchmarkClip]:
    specs = {
        "support": [([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])],
        "target": [([0.0, 1.0, 0.0], [0.0, 1.0, 0.0])],
        "candidate_a": [
            ([0.0, 1.0, 0.0], [0.10, 0.0, 0.995]),
            ([0.02, 0.9998, 0.0], [0.12, 0.0, 0.9928]),
        ],
        "candidate_b": [
            ([0.42, 0.0, 0.9075], [0.0, 1.0, 0.0]),
            ([0.45, 0.0, 0.8930], [0.15, 0.9887, 0.0]),
        ],
    }
    clips: list[BenchmarkClip] = []
    for group_id, rows in specs.items():
        for clip_index, (window, ts2vec) in enumerate(rows):
            clips.append(
                BenchmarkClip(
                    sample_id=f"{group_id}_clip{clip_index:03d}",
                    source_group_id=group_id,
                    embeddings={
                        "window": np.asarray(window, dtype=float),
                        "ts2vec": np.asarray(ts2vec, dtype=float),
                    },
                    quality_score=1.0,
                )
            )
    return clips


def _submitted_ablation_episode(clips: list[BenchmarkClip]) -> EpisodeSpec:
    by_group: dict[str, list[str]] = {}
    for clip in clips:
        by_group.setdefault(clip.source_group_id, []).append(clip.sample_id)
    return EpisodeSpec(
        episode_id="submitted_ablation",
        fold_id=0,
        support_ids=tuple(sorted(by_group["support"])),
        candidate_ids=tuple(sorted([*by_group["candidate_a"], *by_group["candidate_b"]])),
        target_ids=tuple(sorted(by_group["target"])),
        support_group_ids=("support",),
        candidate_group_ids=("candidate_a", "candidate_b"),
        target_group_ids=("target",),
    )


class FakeTS2VecEncoder:
    calls: list[tuple[str, str]] = []

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        self.calls.append((checkpoint_path, device))

    def encode_batch(self, clips: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        return np.asarray([[float(clip.sum()), float(batch_size)] for clip in clips], dtype=np.float32)


def _mean_group_distance(clips: list[BenchmarkClip], target_groups: tuple[str, ...], comparison_groups: tuple[str, ...]) -> float:
    distances = _group_distances_to_targets(clips, target_groups, comparison_groups)
    return float(np.mean(distances))


def _group_distances_to_targets(
    clips: list[BenchmarkClip],
    target_groups: tuple[str, ...],
    comparison_groups: tuple[str, ...],
) -> list[float]:
    by_group: dict[str, list[np.ndarray]] = {}
    for clip in clips:
        by_group.setdefault(clip.source_group_id, []).append(np.asarray(clip.embeddings["window"], dtype=float))
    target = np.vstack([np.mean(by_group[group], axis=0) for group in target_groups])
    comparison = np.vstack([np.mean(by_group[group], axis=0) for group in comparison_groups])
    target = target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1.0e-12)
    comparison = comparison / np.maximum(np.linalg.norm(comparison, axis=1, keepdims=True), 1.0e-12)
    distances = []
    for row in comparison:
        distances.append(float(1.0 - np.max(target @ row.reshape(-1, 1))))
    return distances


def _oracle_minus_random_replay_mean(clips: list[BenchmarkClip], episodes: tuple[EpisodeSpec, ...]) -> float:
    replay_policies = [f"random_valid_replay_{index:03d}" for index in range(12)]
    config = OfflineBenchmarkConfig(
        batch_size=2,
        rounds=1,
        policies=[*replay_policies, "oracle_greedy_eval_only"],
        representations=["window"],
        primary_representations=["window"],
        random_seed=29,
        oracle_exact_combination_limit=10_000,
    )
    result = run_offline_active_benchmark(clips, episodes, config)
    final_by_policy = {row.policy_name: float(row.cumulative_balanced_relative_gain) for row in result.rounds}
    random_mean = float(np.mean([final_by_policy[policy] for policy in replay_policies]))
    return final_by_policy["oracle_greedy_eval_only"] - random_mean


def _worker_id(url: str) -> str:
    return next(part for part in url.split("/") if part.startswith("worker"))


def _literal_family(group_id: str) -> str:
    return group_id.split("_worker", 1)[0]


if __name__ == "__main__":
    unittest.main()
