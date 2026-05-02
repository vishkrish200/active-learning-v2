import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.active.baselines import (
    BASELINE_POLICY_NAMES,
    blended_kcenter_greedy_quality_gated_order,
    blended_old_novelty_order,
    kcenter_greedy_quality_gated_order,
    old_novelty_only_order,
    quality_only_order,
    random_valid_order,
)
from marginal_value.active.episodes import (
    EpisodeConfig,
    generate_active_episodes,
    run_active_episode_smoke,
)
from marginal_value.active.registry import build_clip_registry
from marginal_value.data.split_manifest import hash_manifest_url


class ActiveEpisodeTests(unittest.TestCase):
    def test_generate_active_episodes_is_deterministic_source_blocked_and_mixed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            urls = _write_episode_fixture(root)
            registry = build_clip_registry(
                root,
                manifests={"pretrain": "cache/manifests/pretrain_full_cached_urls.txt"},
                quality_metadata_path="quality.jsonl",
            )
            config = EpisodeConfig(
                n_episodes=2,
                seed=41,
                split="pretrain",
                support_source_groups_per_episode=3,
                heldout_source_groups_per_episode=2,
                support_clips_per_group=3,
                known_like_candidates_per_group=1,
                heldout_candidates_per_group=1,
                hidden_targets_per_group=2,
                near_duplicate_candidates=2,
                low_quality_candidates=1,
                low_quality_threshold=0.45,
            )

            first = generate_active_episodes(registry, config)
            second = generate_active_episodes(registry, config)

        self.assertEqual([episode.to_dict() for episode in first], [episode.to_dict() for episode in second])
        self.assertEqual(len(first), 2)
        episode = first[0]
        self.assertTrue(set(episode.support_clip_ids).isdisjoint(episode.candidate_clip_ids))
        self.assertTrue(set(episode.support_clip_ids).isdisjoint(episode.hidden_target_clip_ids))
        self.assertTrue(set(episode.candidate_clip_ids).isdisjoint(episode.hidden_target_clip_ids))
        self.assertTrue(set(episode.known_source_groups).isdisjoint(episode.heldout_source_groups))
        hidden_groups = {registry.by_sample_id[sample_id].source_group_id for sample_id in episode.hidden_target_clip_ids}
        support_groups = {registry.by_sample_id[sample_id].source_group_id for sample_id in episode.support_clip_ids}
        self.assertTrue(support_groups.isdisjoint(hidden_groups))
        self.assertIn("known_like", set(episode.candidate_roles.values()))
        self.assertIn("heldout_novel", set(episode.candidate_roles.values()))
        self.assertIn("near_duplicate", set(episode.candidate_roles.values()))
        self.assertIn("low_quality", set(episode.candidate_roles.values()))
        self.assertEqual(len(episode.low_quality_clip_ids), 1)
        self.assertTrue(set(episode.distractor_clip_ids).issubset(set(episode.candidate_clip_ids)))
        self.assertEqual(len(set(urls)), 36)

    def test_run_active_episode_smoke_writes_report_and_jsonl(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_episode_fixture(root)
            config = {
                "execution": {"provider": "modal", "allow_local_paths_for_tests": True, "smoke_episodes": 1},
                "data": {
                    "root": str(root),
                    "manifests": {"pretrain": "cache/manifests/pretrain_full_cached_urls.txt"},
                    "quality_metadata": "quality.jsonl",
                },
                "episodes": {
                    "n_episodes": 3,
                    "seed": 7,
                    "split": "pretrain",
                    "support_source_groups_per_episode": 3,
                    "heldout_source_groups_per_episode": 2,
                    "support_clips_per_group": 3,
                    "known_like_candidates_per_group": 1,
                    "heldout_candidates_per_group": 1,
                    "hidden_targets_per_group": 2,
                    "near_duplicate_candidates": 1,
                    "low_quality_candidates": 1,
                },
                "diagnostics": {
                    "required_candidate_roles": ["known_like", "heldout_novel", "near_duplicate", "low_quality"],
                    "fail_on_violation": True,
                },
                "artifacts": {"output_dir": str(root / "artifacts")},
            }

            result = run_active_episode_smoke(config, smoke=True)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            episode_rows = [
                json.loads(line)
                for line in Path(result["episodes_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(result["mode"], "smoke")
        self.assertEqual(result["n_episodes"], 1)
        self.assertEqual(report["registry"]["split_counts"]["pretrain"], 36)
        self.assertTrue(report["diagnostics_gate"]["passed"])
        self.assertEqual(report["diagnostics_gate"]["violations"], [])
        self.assertEqual(len(episode_rows), 1)
        self.assertIn("heldout_novel", episode_rows[0]["candidate_role_counts"])

    def test_baseline_controls_expose_expected_orders(self):
        self.assertEqual(
            BASELINE_POLICY_NAMES,
            (
                "random_valid",
                "quality_only",
                "old_novelty_only",
                "kcenter_greedy_quality_gated",
                "window_shape_stats_q85_stat90_abs60_clustercap2",
            ),
        )
        rows = [
            {"sample_id": "a", "quality_score": 0.2, "old_novelty_score": 0.8},
            {"sample_id": "b", "quality_score": 0.9, "old_novelty_score": 0.1},
            {"sample_id": "c", "quality_score": 0.7, "old_novelty_score": 0.6},
        ]
        support = np.asarray([[1.0, 0.0]], dtype=float)
        candidates = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)

        self.assertEqual(quality_only_order(rows), [1, 2, 0])
        self.assertEqual(random_valid_order(rows, seed=5, quality_threshold=0.45)[:2], [2, 1])
        old_order, old_distance = old_novelty_only_order(support, candidates, k=1)
        self.assertEqual(old_order, [2, 1, 0])
        self.assertGreater(old_distance[2], old_distance[0])
        self.assertEqual(
            kcenter_greedy_quality_gated_order(
                support,
                candidates,
                rows,
                quality_threshold=0.45,
            )[:2],
            [2, 1],
        )

    def test_blended_baselines_respect_alpha_between_representations(self):
        rows = [
            {"sample_id": "a", "quality_score": 0.9},
            {"sample_id": "b", "quality_score": 0.9},
            {"sample_id": "c", "quality_score": 0.9},
        ]
        novelty_left = np.asarray([1.0, 0.0, 0.5], dtype=float)
        novelty_right = np.asarray([0.0, 1.0, 0.5], dtype=float)

        self.assertEqual(blended_old_novelty_order(novelty_left, novelty_right, alpha=1.0), [0, 2, 1])
        self.assertEqual(blended_old_novelty_order(novelty_left, novelty_right, alpha=0.0), [1, 2, 0])

        support_left = np.asarray([[1.0, 0.0]], dtype=float)
        candidates_left = np.asarray([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
        support_right = np.asarray([[1.0, 0.0]], dtype=float)
        candidates_right = np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]], dtype=float)

        self.assertEqual(
            blended_kcenter_greedy_quality_gated_order(
                support_left,
                candidates_left,
                support_right,
                candidates_right,
                rows,
                alpha=1.0,
                quality_threshold=0.45,
            )[:1],
            [0],
        )
        self.assertEqual(
            blended_kcenter_greedy_quality_gated_order(
                support_left,
                candidates_left,
                support_right,
                candidates_right,
                rows,
                alpha=0.0,
                quality_threshold=0.45,
            )[:1],
            [1],
        )

    def test_modal_active_episode_smoke_entrypoint_uses_full_pretrain_manifest(self):
        source = Path("modal_active_episode_smoke.py").read_text(encoding="utf-8")
        config = json.loads(Path("configs/active_episode_smoke_full_pretrain.json").read_text(encoding="utf-8"))

        self.assertIn("remote_active_episode_smoke.remote(config, smoke=True)", source)
        self.assertIn("run_active_episode_smoke", source)
        self.assertIn("artifacts_volume.commit()", source)
        self.assertEqual(config["data"]["manifests"]["pretrain"], "cache/manifests/pretrain_full_cached_urls.txt")
        self.assertEqual(config["data"]["manifests"]["new"], "cache/manifests/new_urls.txt")
        self.assertEqual(
            config["data"]["quality_metadata"],
            "/artifacts/active/quality/full_pretrain_smoke/active_quality_smoke.jsonl",
        )
        self.assertEqual(config["episodes"]["split"], "pretrain")
        self.assertIn("low_quality", config["diagnostics"]["required_candidate_roles"])
        scale_config = json.loads(Path("configs/active_episode_scale_pretrain.json").read_text(encoding="utf-8"))
        self.assertEqual(scale_config["execution"]["smoke_episodes"], 2)
        self.assertEqual(scale_config["episodes"]["n_episodes"], 64)
        self.assertGreaterEqual(
            scale_config["episodes"]["known_like_candidates_per_group"]
            * scale_config["episodes"]["support_source_groups_per_episode"]
            + scale_config["episodes"]["heldout_candidates_per_group"]
            * scale_config["episodes"]["heldout_source_groups_per_episode"]
            + scale_config["episodes"]["near_duplicate_candidates"]
            + scale_config["episodes"]["low_quality_candidates"],
            150,
        )


def _write_episode_fixture(root: Path) -> list[str]:
    urls: list[str] = []
    scores: dict[str, float] = {}
    for worker_idx in range(6):
        worker = f"worker{worker_idx:05d}"
        for clip_idx in range(6):
            url = _url("pretrain", worker, clip_idx)
            urls.append(url)
            score = 0.2 if worker_idx == 5 and clip_idx < 3 else 0.82 + worker_idx * 0.01
            scores[hash_manifest_url(url)] = score
            _write_cached_clip(root, url, value=worker_idx + clip_idx * 0.01)
    _write_manifest(root, "cache/manifests/pretrain_full_cached_urls.txt", urls)
    _write_quality_jsonl(root / "quality.jsonl", scores)
    return urls


def _url(split: str, worker: str, clip: int) -> str:
    return f"https://storage.googleapis.com/unit/{split}/{worker}/clip{clip:03d}.jsonl"


def _write_manifest(root: Path, relpath: str, urls: list[str]) -> None:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(urls) + "\n", encoding="utf-8")


def _write_cached_clip(root: Path, url: str, *, value: float) -> None:
    sample_id = hash_manifest_url(url)
    raw_dir = root / "cache" / "raw"
    feature_dir = root / "cache" / "features"
    raw_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{sample_id}.jsonl").write_text(json.dumps({"acc": [0, 0, 9.8], "gyro": [0, 0, 0]}) + "\n")
    np.savez(
        feature_dir / f"{sample_id}.npz",
        window_features=np.full((4, 5), value, dtype=np.float32),
        clip_features=np.full(5, value, dtype=np.float32),
    )


def _write_quality_jsonl(path: Path, scores: dict[str, float]) -> None:
    rows = [
        json.dumps({"sample_id": sample_id, "quality_score": score})
        for sample_id, score in sorted(scores.items())
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
