import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.eval.shadow_ranking_eval import (
    _submission_rows,
    build_shadow_ranked_rows,
    run_shadow_ranking_eval,
    validate_shadow_ranking_config,
)


class ShadowRankingEvalTests(unittest.TestCase):
    def test_quality_gated_shadow_score_downranks_low_quality_surprisal_without_oracle_features(self):
        rows = [
            _row("artifact", label=0, quality=0.20, token_nll=10.0, transition_nll=9.5, cluster_id=0),
            _row("positive", label=1, quality=1.00, token_nll=8.0, transition_nll=7.5, cluster_id=1),
            _row("ordinary", label=0, quality=1.00, token_nll=3.0, transition_nll=2.5, cluster_id=2),
        ]

        ranked, metadata = build_shadow_ranked_rows(rows, cluster_bonus_weight=0.0)

        self.assertEqual(ranked[0]["worker_id"], "positive")
        self.assertLess(float(ranked[0]["shadow_score"]), float(rows[0]["token_nll_p95"]))
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[0]["reason_code"], "RARE_TEMPORAL_COMPOSITION")
        self.assertIn("quality_score", metadata["used_features"])
        self.assertNotIn("label", metadata["used_features"])
        self.assertNotIn("is_artifact", metadata["used_features"])
        self.assertFalse(metadata["forbidden_used"])

    def test_shadow_cluster_aware_order_spreads_top_k(self):
        rows = [
            _row("a1", label=1, quality=1.0, token_nll=10.0, transition_nll=9.0, cluster_id=0),
            _row("a2", label=1, quality=1.0, token_nll=9.9, transition_nll=8.9, cluster_id=0),
            _row("b1", label=1, quality=1.0, token_nll=8.5, transition_nll=8.0, cluster_id=1),
            _row("c1", label=0, quality=1.0, token_nll=1.0, transition_nll=1.0, cluster_id=2),
        ]

        ranked, _metadata = build_shadow_ranked_rows(rows, cluster_bonus_weight=0.25)

        self.assertEqual([row["worker_id"] for row in ranked[:3]], ["a1", "b1", "a2"])
        self.assertEqual(ranked[0]["shadow_cluster_round"], 0)
        self.assertEqual(ranked[2]["shadow_cluster_round"], 1)

    def test_shadow_cluster_cap_limits_top_k_repeats_before_relaxing(self):
        rows = [
            _row("a1", label=1, quality=1.0, token_nll=10.0, transition_nll=9.0, cluster_id=0),
            _row("a2", label=1, quality=1.0, token_nll=9.9, transition_nll=8.9, cluster_id=0),
            _row("a3", label=1, quality=1.0, token_nll=9.8, transition_nll=8.8, cluster_id=0),
            _row("b1", label=1, quality=1.0, token_nll=8.5, transition_nll=8.0, cluster_id=1),
            _row("b2", label=1, quality=1.0, token_nll=8.4, transition_nll=7.9, cluster_id=1),
            _row("c1", label=0, quality=1.0, token_nll=7.5, transition_nll=7.0, cluster_id=2),
        ]

        ranked, metadata = build_shadow_ranked_rows(
            rows,
            diversity_method="cluster_cap",
            cluster_bonus_weight=0.0,
            cluster_cap_top_k=5,
            cluster_max_per_cluster=2,
        )

        top5_clusters = [row["new_cluster_id"] for row in ranked[:5]]
        self.assertEqual(metadata["diversity_method"], "cluster_cap")
        self.assertLessEqual(top5_clusters.count(0), 2)
        self.assertIn(2, top5_clusters)
        self.assertEqual(ranked[5]["worker_id"], "a3")

    def test_shadow_round_robin_takes_one_per_cluster_before_repeating(self):
        rows = [
            _row("a1", label=1, quality=1.0, token_nll=10.0, transition_nll=9.0, cluster_id=0),
            _row("a2", label=1, quality=1.0, token_nll=9.9, transition_nll=8.9, cluster_id=0),
            _row("b1", label=1, quality=1.0, token_nll=8.5, transition_nll=8.0, cluster_id=1),
            _row("b2", label=1, quality=1.0, token_nll=8.4, transition_nll=7.9, cluster_id=1),
            _row("c1", label=0, quality=1.0, token_nll=7.5, transition_nll=7.0, cluster_id=2),
        ]

        ranked, metadata = build_shadow_ranked_rows(
            rows,
            diversity_method="cluster_round_robin",
            cluster_bonus_weight=0.0,
        )

        self.assertEqual(metadata["diversity_method"], "cluster_round_robin")
        self.assertEqual([row["worker_id"] for row in ranked[:3]], ["a1", "b1", "c1"])
        self.assertEqual([row["shadow_cluster_round"] for row in ranked[:5]], [0, 0, 0, 1, 1])

    def test_shadow_cluster_mmr_and_cap_then_mmr_penalize_repeated_clusters(self):
        rows = [
            _row("a1", label=1, quality=1.0, token_nll=10.0, transition_nll=9.0, cluster_id=0),
            _row("a2", label=1, quality=1.0, token_nll=9.9, transition_nll=8.9, cluster_id=0),
            _row("b1", label=1, quality=1.0, token_nll=8.5, transition_nll=8.0, cluster_id=1),
        ]

        mmr_ranked, mmr_metadata = build_shadow_ranked_rows(
            rows,
            diversity_method="cluster_mmr",
            lambda_redundancy=1.0,
        )
        cap_mmr_ranked, cap_mmr_metadata = build_shadow_ranked_rows(
            rows,
            diversity_method="cluster_cap_then_cluster_mmr",
            cluster_cap_top_k=3,
            cluster_max_per_cluster=2,
            lambda_redundancy=1.0,
        )

        self.assertEqual(mmr_metadata["diversity_method"], "cluster_mmr")
        self.assertEqual(mmr_ranked[1]["worker_id"], "b1")
        self.assertEqual(cap_mmr_metadata["diversity_method"], "cluster_cap_then_cluster_mmr")
        self.assertEqual(cap_mmr_ranked[1]["worker_id"], "b1")

    def test_run_shadow_ranking_eval_writes_shadow_artifacts_and_audit(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            diagnostics_path = root / "diagnostics.csv"
            candidate_path = root / "candidates.csv"
            quality_path = root / "quality.csv"
            output_dir = root / "shadow"
            rows = [
                _row("artifact", label=0, quality=0.20, token_nll=10.0, transition_nll=9.5, cluster_id=0),
                _row("positive", label=1, quality=1.00, token_nll=8.0, transition_nll=7.5, cluster_id=1),
                _row("ordinary", label=0, quality=1.00, token_nll=3.0, transition_nll=2.5, cluster_id=2),
            ]
            _write_rows(diagnostics_path, [row | {"rank": idx + 1} for idx, row in enumerate(rows)])
            _write_rows(candidate_path, [row | {"rank": idx + 1} for idx, row in enumerate(rows)])
            _write_rows(
                quality_path,
                [{"sample_id": row["sample_id"], "quality_score": row["quality_score"], "spike_rate": 0.0} for row in rows],
            )
            config = {
                "execution": {"provider": "modal"},
                "artifacts": {
                    "diagnostics_path": str(diagnostics_path),
                    "candidate_path": str(candidate_path),
                    "quality_metadata_path": str(quality_path),
                    "output_dir": str(output_dir),
                },
                "shadow": {
                    "score_variant": "quality_gated_grammar",
                    "cluster_bonus_weight": 0.0,
                },
                "audit": {
                    "top_ks": [1, 2],
                    "low_quality_threshold": 0.45,
                    "n_examples": 2,
                },
            }

            result = run_shadow_ranking_eval(config, allow_local_execution=True)

            self.assertTrue(Path(result["submission_path"]).exists())
            self.assertTrue(Path(result["diagnostics_path"]).exists())
            self.assertTrue(Path(result["candidate_path"]).exists())
            self.assertTrue(Path(result["report_path"]).exists())
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(report["shadow"]["score_variant"], "quality_gated_grammar")
            self.assertEqual(report["candidate_comparison"]["shadow"]["top_k"]["1"]["positive_fraction"], 1.0)
            self.assertEqual(report["audit"]["top_k"]["1"]["low_quality_count"], 0)
            self.assertIn("dominant_cluster_fraction", report["audit"]["top_k"]["2"])

    def test_shadow_submission_score_is_monotonic_for_cluster_aware_order(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            diagnostics_path = root / "diagnostics.csv"
            candidate_path = root / "candidates.csv"
            quality_path = root / "quality.csv"
            output_dir = root / "shadow"
            rows = [
                _row("a1", label=1, quality=1.0, token_nll=10.0, transition_nll=9.0, cluster_id=0),
                _row("a2", label=1, quality=1.0, token_nll=9.9, transition_nll=8.9, cluster_id=0),
                _row("b1", label=1, quality=1.0, token_nll=8.5, transition_nll=8.0, cluster_id=1),
            ]
            _write_rows(diagnostics_path, [row | {"rank": idx + 1} for idx, row in enumerate(rows)])
            _write_rows(candidate_path, [row | {"rank": idx + 1} for idx, row in enumerate(rows)])
            _write_rows(
                quality_path,
                [{"sample_id": row["sample_id"], "quality_score": row["quality_score"], "spike_rate": 0.0} for row in rows],
            )
            config = {
                "execution": {"provider": "modal"},
                "artifacts": {
                    "diagnostics_path": str(diagnostics_path),
                    "candidate_path": str(candidate_path),
                    "quality_metadata_path": str(quality_path),
                    "output_dir": str(output_dir),
                },
                "shadow": {"score_variant": "quality_gated_grammar", "cluster_bonus_weight": 0.25},
                "audit": {"top_ks": [2], "low_quality_threshold": 0.45, "n_examples": 2},
            }

            result = run_shadow_ranking_eval(config, allow_local_execution=True)

            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            self.assertTrue(report["audit"]["submission"]["score_nonincreasing"])
            with Path(result["diagnostics_path"]).open(newline="", encoding="utf-8") as handle:
                diagnostics = list(csv.DictReader(handle))
            self.assertIn("shadow_score", diagnostics[0])
            self.assertIn("shadow_rerank_score", diagnostics[0])

    def test_run_shadow_ranking_eval_writes_multi_variant_comparison(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            diagnostics_path = root / "diagnostics.csv"
            candidate_path = root / "candidates.csv"
            quality_path = root / "quality.csv"
            output_dir = root / "shadow"
            rows = [
                _row("a1", label=1, quality=1.0, token_nll=10.0, transition_nll=9.0, cluster_id=0),
                _row("a2", label=1, quality=1.0, token_nll=9.9, transition_nll=8.9, cluster_id=0),
                _row("a3", label=1, quality=1.0, token_nll=9.8, transition_nll=8.8, cluster_id=0),
                _row("b1", label=1, quality=1.0, token_nll=8.5, transition_nll=8.0, cluster_id=1),
                _row("c1", label=0, quality=1.0, token_nll=7.5, transition_nll=7.0, cluster_id=2),
            ]
            _write_rows(diagnostics_path, [row | {"rank": idx + 1} for idx, row in enumerate(rows)])
            _write_rows(candidate_path, [row | {"rank": idx + 1} for idx, row in enumerate(rows)])
            _write_rows(
                quality_path,
                [{"sample_id": row["sample_id"], "quality_score": row["quality_score"], "spike_rate": 0.0} for row in rows],
            )
            config = {
                "execution": {"provider": "modal"},
                "artifacts": {
                    "diagnostics_path": str(diagnostics_path),
                    "candidate_path": str(candidate_path),
                    "quality_metadata_path": str(quality_path),
                    "output_dir": str(output_dir),
                },
                "shadow": {
                    "score_variant": "quality_gated_grammar",
                    "diversity_variants": [
                        {"name": "bonus", "diversity_method": "cluster_bonus", "cluster_bonus_weight": 0.0},
                        {
                            "name": "cap",
                            "diversity_method": "cluster_cap",
                            "cluster_bonus_weight": 0.0,
                            "cluster_cap_top_k": 4,
                            "cluster_max_per_cluster": 1,
                        },
                    ],
                },
                "audit": {"top_ks": [4], "low_quality_threshold": 0.45, "n_examples": 2},
            }

            result = run_shadow_ranking_eval(config, allow_local_execution=True)

            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(set(report["variants"]), {"bonus", "cap"})
            self.assertLess(
                report["variants"]["cap"]["candidate_comparison"]["shadow"]["top_k"]["4"]["cluster_repeat_count"],
                report["variants"]["bonus"]["candidate_comparison"]["shadow"]["top_k"]["4"]["cluster_repeat_count"],
            )
            self.assertIn(
                "dominant_cluster_fraction",
                report["variants"]["bonus"]["candidate_comparison"]["shadow"]["top_k"]["4"],
            )
            self.assertTrue((output_dir / "bonus" / "shadow_submission_val_full.csv").exists())
            self.assertTrue((output_dir / "cap" / "shadow_submission_val_full.csv").exists())

    def test_multi_variant_report_selects_candidate_that_passes_top_k_thresholds(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            diagnostics_path = root / "diagnostics.csv"
            candidate_path = root / "candidates.csv"
            quality_path = root / "quality.csv"
            output_dir = root / "shadow"
            rows = [
                _row(f"a{idx}", label=1, quality=1.0, token_nll=10.0 - idx * 0.01, transition_nll=9.0, cluster_id=0)
                for idx in range(8)
            ]
            rows.extend(
                [
                    _row("b1", label=1, quality=1.0, token_nll=8.5, transition_nll=8.0, cluster_id=1),
                    _row("c1", label=1, quality=1.0, token_nll=8.0, transition_nll=7.5, cluster_id=2),
                    _row("d1", label=1, quality=1.0, token_nll=7.5, transition_nll=7.0, cluster_id=3),
                ]
            )
            _write_rows(diagnostics_path, [row | {"rank": idx + 1} for idx, row in enumerate(rows)])
            _write_rows(candidate_path, [row | {"rank": idx + 1} for idx, row in enumerate(rows)])
            _write_rows(
                quality_path,
                [{"sample_id": row["sample_id"], "quality_score": row["quality_score"], "spike_rate": 0.0} for row in rows],
            )
            config = {
                "execution": {"provider": "modal"},
                "artifacts": {
                    "diagnostics_path": str(diagnostics_path),
                    "candidate_path": str(candidate_path),
                    "quality_metadata_path": str(quality_path),
                    "output_dir": str(output_dir),
                },
                "shadow": {
                    "score_variant": "quality_gated_grammar",
                    "diversity_variants": [
                        {"name": "bonus", "diversity_method": "cluster_bonus", "cluster_bonus_weight": 0.0},
                        {
                            "name": "cap",
                            "diversity_method": "cluster_cap",
                            "cluster_bonus_weight": 0.0,
                            "cluster_cap_top_k": 10,
                            "cluster_max_per_cluster": 2,
                        },
                    ],
                },
                "selection_criteria": {
                    "candidate_top_k": 10,
                    "min_positive_fraction": 0.90,
                    "min_unique_clusters": 4,
                    "max_low_quality_count": 0,
                },
                "audit": {"top_ks": [10], "low_quality_threshold": 0.45, "n_examples": 2},
            }

            result = run_shadow_ranking_eval(config, allow_local_execution=True)

            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(report["selected_variant"], "cap")
            self.assertTrue(report["selection_criteria"]["passed"])
            self.assertEqual(result["best_variant"], "cap")

    def test_shadow_submission_uses_rerank_score_for_monotonic_scores(self):
        rows = [
            {"worker_id": "a", "rank": 1, "shadow_score": 1.0, "shadow_rerank_score": 1.25, "quality_score": 1.0},
            {"worker_id": "b", "rank": 2, "shadow_score": 0.8, "shadow_rerank_score": 1.05, "quality_score": 1.0},
            {"worker_id": "c", "rank": 3, "shadow_score": 0.9, "shadow_rerank_score": 0.90, "quality_score": 1.0},
        ]

        submission = _submission_rows(rows)

        scores = [float(row["score"]) for row in submission]
        self.assertTrue(all(left >= right for left, right in zip(scores, scores[1:])))

    def test_validate_shadow_ranking_config_requires_modal_and_absolute_artifacts(self):
        valid_config = {
            "execution": {"provider": "modal"},
            "artifacts": {
                "diagnostics_path": "/artifacts/ranking/diagnostics.csv",
                "candidate_path": "/artifacts/ranking/candidates.csv",
                "quality_metadata_path": "/artifacts/ranking/quality.csv",
                "output_dir": "/artifacts/eval/shadow",
            },
        }
        validate_shadow_ranking_config(valid_config)

        with self.assertRaises(ValueError):
            validate_shadow_ranking_config(valid_config | {"execution": {"provider": "local"}})

        with self.assertRaises(ValueError):
            validate_shadow_ranking_config(
                {
                    "execution": {"provider": "modal"},
                    "artifacts": {
                        "diagnostics_path": "relative.csv",
                        "candidate_path": "/artifacts/ranking/candidates.csv",
                        "quality_metadata_path": "/artifacts/ranking/quality.csv",
                        "output_dir": "/artifacts/eval/shadow",
                    },
                }
            )

    def test_new_split_shadow_diversity_config_is_valid(self):
        config = json.loads(Path("configs/shadow_quality_gated_grammar_new_diversity.json").read_text(encoding="utf-8"))

        validate_shadow_ranking_config(config)

        methods = {variant["diversity_method"] for variant in config["shadow"]["diversity_variants"]}
        self.assertIn("score_sort", methods)
        self.assertIn("cluster_cap_then_cluster_mmr", methods)


def _row(
    worker_id: str,
    *,
    label: int,
    quality: float,
    token_nll: float,
    transition_nll: float,
    cluster_id: int,
) -> dict[str, object]:
    return {
        "worker_id": worker_id,
        "sample_id": worker_id,
        "label": label,
        "quality_score": quality,
        "final_score": token_nll / 10.0,
        "old_novelty_score": token_nll / 10.0,
        "new_density_score": 0.8,
        "new_cluster_id": cluster_id,
        "token_nll_p95": token_nll,
        "transition_nll_p95": transition_nll,
        "longest_unseen_phrase_len": token_nll / 2.0,
        "rare_phrase_fraction": 1.0 if label else 0.0,
        "grammar_feature_present": True,
        "reason_code": "LOW_QUALITY" if quality < 0.45 else "RARE_MOTION_PRIMITIVES",
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
