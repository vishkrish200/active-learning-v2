import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.eval.grammar_ablation_eval import run_grammar_ablation, run_leave_cluster_out_ablation
from marginal_value.tokenization.artifacts import TokenSequence, write_token_sequences_jsonl


class GrammarAblationEvalTests(unittest.TestCase):
    def test_grammar_ablation_scores_candidates_without_val_fit_leakage(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            token_path = root / "tokens" / "token_sequences_full.jsonl"
            candidate_path = root / "ranking" / "baseline_ranking_val_candidates_full.csv"
            output_dir = root / "ablation"
            write_token_sequences_jsonl(token_path, _token_sequences())
            _write_candidates(candidate_path)

            result = run_grammar_ablation(
                _config(root),
                token_sequence_path=token_path,
                candidate_path=candidate_path,
                output_dir=output_dir,
                smoke=False,
            )

            self.assertEqual(result["mode"], "full")
            self.assertEqual(result["fit_split"], "pretrain")
            self.assertEqual(result["n_candidates"], 4)
            self.assertEqual(result["missing_token_sequence_count"], 0)
            self.assertTrue(Path(result["report_path"]).exists())
            self.assertTrue(Path(result["scores_path"]).exists())
            self.assertTrue(Path(result["summary_path"]).exists())

            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(report["leakage_audit"]["grammar_fit_splits"], ["pretrain"])
            self.assertEqual(report["leakage_audit"]["scored_candidate_splits"], ["pretrain", "val"])
            self.assertIn("negative_candidates_include_fit_split", report["interpretation_warnings"])
            self.assertIn("phase_a_final_score", report["variants"])
            self.assertIn("grammar_token_nll_p95", report["variants"])
            self.assertIn("phase_a_plus_grammar_10pct", report["variants"])
            self.assertGreaterEqual(
                report["variants"]["grammar_token_nll_p95"]["metrics"]["ndcg@2"],
                report["variants"]["phase_a_final_score"]["metrics"]["ndcg@2"],
            )

            scores = _read_csv(Path(result["scores_path"]))
            rare = next(row for row in scores if row["sample_id"] == "val-rare")
            familiar = next(row for row in scores if row["sample_id"] == "pre-familiar")
            self.assertGreater(float(rare["token_nll_p95"]), float(familiar["token_nll_p95"]))

    def test_grammar_ablation_reports_missing_candidate_tokens(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            token_path = root / "tokens" / "token_sequences_full.jsonl"
            candidate_path = root / "ranking" / "baseline_ranking_val_candidates_full.csv"
            output_dir = root / "ablation"
            write_token_sequences_jsonl(token_path, _token_sequences()[:-1])
            _write_candidates(candidate_path)

            result = run_grammar_ablation(
                _config(root),
                token_sequence_path=token_path,
                candidate_path=candidate_path,
                output_dir=output_dir,
                smoke=False,
            )

            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(result["missing_token_sequence_count"], 1)
            self.assertEqual(report["candidate_coverage"]["missing_token_sequence_count"], 1)
            self.assertEqual(report["candidate_coverage"]["scored_candidate_count"], 3)

    def test_grammar_ablation_inherits_source_tokens_for_corruptions(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            token_path = root / "tokens" / "token_sequences_full.jsonl"
            candidate_path = root / "ranking" / "baseline_ranking_val_candidates_full.csv"
            output_dir = root / "ablation"
            write_token_sequences_jsonl(token_path, _token_sequences())
            _write_corruption_candidate(candidate_path)

            result = run_grammar_ablation(
                _config(root),
                token_sequence_path=token_path,
                candidate_path=candidate_path,
                output_dir=output_dir,
                smoke=False,
            )

            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(result["missing_token_sequence_count"], 0)
            self.assertEqual(report["candidate_coverage"]["inherited_token_sequence_count"], 1)
            scores = _read_csv(Path(result["scores_path"]))
            self.assertEqual(scores[0]["sample_id"], "val-rare__corrupt_spike_0001")
            self.assertEqual(scores[0]["token_source_sample_id"], "val-rare")
            self.assertEqual(scores[0]["token_split"], "val")

    def test_leave_cluster_out_ablation_fits_without_heldout_clusters(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            token_path = root / "tokens" / "token_sequences_full.jsonl"
            output_dir = root / "cluster_ablation"
            write_token_sequences_jsonl(token_path, _cluster_token_sequences())
            config = _config(root)
            config["cluster_ablation"] = {
                "n_clusters": 2,
                "n_folds": 2,
                "negative_sample_size": 2,
                "seed": 3,
            }

            result = run_leave_cluster_out_ablation(
                config,
                token_sequence_path=token_path,
                output_dir=output_dir,
                smoke=False,
            )

            self.assertEqual(result["mode"], "full")
            self.assertEqual(result["fit_split"], "pretrain")
            self.assertEqual(result["n_folds"], 2)
            self.assertTrue(Path(result["report_path"]).exists())
            self.assertTrue(Path(result["summary_path"]).exists())
            self.assertTrue(Path(result["scores_path"]).exists())

            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(report["leakage_audit"]["source_splits"], ["pretrain"])
            self.assertEqual(report["leakage_audit"]["heldout_fit_overlap_count"], 0)
            self.assertIn("grammar_token_nll_p95", report["aggregate_variants"])
            self.assertIn("random_baseline", report["aggregate_variants"])
            self.assertNotIn("phase_a_final_score", report["aggregate_variants"])
            self.assertNotIn("old_novelty_score", report["aggregate_variants"])
            self.assertEqual(len(report["folds"]), 2)
            for fold in report["folds"]:
                self.assertEqual(fold["heldout_fit_overlap_count"], 0)
                self.assertGreater(fold["n_positive"], 0)
                self.assertGreater(fold["n_negative"], 0)


def _token_sequences() -> list[TokenSequence]:
    return [
        TokenSequence(
            sample_id="pre-familiar",
            split="pretrain",
            base_token_ids=[1, 2, 3, 1],
            primitive_token_ids=["reach", "inspect", "return", "reach"],
            primitive_durations_sec=[0.5, 0.5, 0.5, 0.5],
        ),
        TokenSequence(
            sample_id="pre-familiar-2",
            split="pretrain",
            base_token_ids=[1, 2, 3, 1],
            primitive_token_ids=["reach", "inspect", "return", "reach"],
            primitive_durations_sec=[0.5, 0.5, 0.5, 0.5],
        ),
        TokenSequence(
            sample_id="val-familiar",
            split="val",
            base_token_ids=[1, 2, 3, 1],
            primitive_token_ids=["reach", "inspect", "return", "reach"],
            primitive_durations_sec=[0.5, 0.5, 0.5, 0.5],
        ),
        TokenSequence(
            sample_id="val-rare",
            split="val",
            base_token_ids=[9, 8, 7, 9],
            primitive_token_ids=["twist", "pause", "scan", "twist"],
            primitive_durations_sec=[0.5, 0.5, 0.5, 0.5],
        ),
    ]


def _cluster_token_sequences() -> list[TokenSequence]:
    sequences: list[TokenSequence] = []
    specs = [
        ("reach-a", ["reach", "inspect", "return", "reach"]),
        ("reach-b", ["reach", "inspect", "return", "reach"]),
        ("reach-c", ["reach", "reach", "inspect", "return"]),
        ("twist-a", ["twist", "pause", "scan", "twist"]),
        ("twist-b", ["twist", "pause", "scan", "twist"]),
        ("twist-c", ["twist", "twist", "pause", "scan"]),
    ]
    for sample_id, tokens in specs:
        sequences.append(
            TokenSequence(
                sample_id=sample_id,
                split="pretrain",
                base_token_ids=list(range(len(tokens))),
                primitive_token_ids=tokens,
                primitive_durations_sec=[0.5 for _ in tokens],
            )
        )
    return sequences


def _write_candidates(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "worker_id",
        "split",
        "label",
        "final_score",
        "old_novelty_score",
        "new_density_score",
        "quality_score",
        "new_cluster_id",
        "new_cluster_size",
    ]
    rows = [
        {
            "sample_id": "pre-familiar",
            "worker_id": "pre-familiar",
            "split": "pretrain",
            "label": 0,
            "final_score": 0.95,
            "old_novelty_score": 0.05,
            "new_density_score": 0.8,
            "quality_score": 1.0,
            "new_cluster_id": 1,
            "new_cluster_size": 2,
        },
        {
            "sample_id": "pre-familiar-2",
            "worker_id": "pre-familiar-2",
            "split": "pretrain",
            "label": 0,
            "final_score": 0.90,
            "old_novelty_score": 0.08,
            "new_density_score": 0.7,
            "quality_score": 0.9,
            "new_cluster_id": 1,
            "new_cluster_size": 2,
        },
        {
            "sample_id": "val-familiar",
            "worker_id": "val-familiar",
            "split": "val",
            "label": 1,
            "final_score": 0.10,
            "old_novelty_score": 0.2,
            "new_density_score": 0.5,
            "quality_score": 1.0,
            "new_cluster_id": 2,
            "new_cluster_size": 1,
        },
        {
            "sample_id": "val-rare",
            "worker_id": "val-rare",
            "split": "val",
            "label": 1,
            "final_score": 0.05,
            "old_novelty_score": 0.4,
            "new_density_score": 0.4,
            "quality_score": 1.0,
            "new_cluster_id": 3,
            "new_cluster_size": 1,
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_corruption_candidate(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "worker_id",
        "source_sample_id",
        "split",
        "label",
        "is_corruption",
        "final_score",
        "old_novelty_score",
        "new_density_score",
        "quality_score",
        "new_cluster_id",
        "new_cluster_size",
    ]
    rows = [
        {
            "sample_id": "val-rare__corrupt_spike_0001",
            "worker_id": "val-rare__corrupt_spike_0001",
            "source_sample_id": "val-rare",
            "split": "corruption",
            "label": 0,
            "is_corruption": True,
            "final_score": 0.01,
            "old_novelty_score": 0.99,
            "new_density_score": 0.1,
            "quality_score": 0.05,
            "new_cluster_id": 3,
            "new_cluster_size": 1,
        }
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _config(root: Path) -> dict:
    return {
        "execution": {
            "provider": "modal",
            "artifacts_volume": "activelearning-imu-rebuild-cache",
            "timeout_seconds": 600,
        },
        "artifacts": {
            "ranking_dir": str(root / "ranking"),
            "tokens_dir": str(root / "tokens"),
            "output_dir": str(root / "ablation"),
        },
        "grammar": {
            "fit_split": "pretrain",
            "order": 3,
            "smoothing": 0.1,
            "rare_threshold": 0,
        },
        "eval": {
            "k_values": [1, 2, 4],
            "low_quality_threshold": 0.45,
        },
    }


if __name__ == "__main__":
    unittest.main()
