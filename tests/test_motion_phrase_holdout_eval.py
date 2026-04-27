import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.eval.motion_phrase_holdout_eval import (
    SUPPORT_FEATURE_COLUMNS,
    build_phrase_holdout_rows,
    discover_phrase_families,
    run_motion_phrase_holdout_eval,
)
from marginal_value.tokenization.artifacts import TokenSequence, write_token_sequences_jsonl


class MotionPhraseHoldoutEvalTests(unittest.TestCase):
    def test_discover_phrase_families_returns_frequent_phrases_without_duplicates(self):
        sequences = [
            _sequence("a", ["A", "B", "C", "D"]),
            _sequence("b", ["A", "B", "E", "F"]),
            _sequence("c", ["A", "B", "C", "G"]),
            _sequence("d", ["X", "Y", "Z"]),
        ]

        families = discover_phrase_families(sequences, phrase_len=2, min_support=2, max_families=3)

        self.assertEqual(families[0]["phrase"], ("A", "B"))
        self.assertEqual(families[0]["support"], 3)
        self.assertEqual(len({family["phrase"] for family in families}), len(families))

    def test_build_phrase_holdout_rows_excludes_heldout_phrase_from_fit_and_adds_negative_types(self):
        sequences = [
            _sequence("pos-a", ["A", "B", "C", "D"], quality=0.98),
            _sequence("pos-b", ["X", "A", "B", "Y"], quality=0.97),
            _sequence("neg-a", ["C", "D", "E", "F"], quality=0.96),
            _sequence("neg-b", ["G", "H", "I", "J"], quality=0.95),
            _sequence("neg-c", ["K", "L", "M", "N"], quality=0.94),
        ]

        result = build_phrase_holdout_rows(
            sequences,
            phrase=("A", "B"),
            fold_index=1,
            grammar_order=3,
            smoothing=0.1,
            rare_threshold=0,
            negative_sample_size=2,
            artifact_negative_count=1,
            redundancy_negative_count=1,
            seed=5,
        )

        report = result["report"]
        rows = result["rows"]
        self.assertEqual(report["heldout_fit_overlap_count"], 0)
        self.assertEqual(report["phrase"], ["A", "B"])
        self.assertEqual(report["n_positive"], 2)
        self.assertEqual(report["n_artifact_negative"], 1)
        self.assertEqual(report["n_redundancy_negative"], 1)
        self.assertEqual({row["label"] for row in rows}, {0, 1})
        self.assertIn("artifact", {row["negative_type"] for row in rows})
        self.assertIn("redundancy", {row["negative_type"] for row in rows})
        for row in rows:
            if row["negative_type"] == "artifact":
                self.assertLess(row["quality_score"], 0.45)
                self.assertEqual(row["is_artifact"], 1)
            if row["negative_type"] == "redundancy":
                self.assertEqual(row["is_redundant"], 1)
                self.assertTrue(str(row["duplicate_group_id"]))
            self.assertNotIn("contains_heldout_phrase", row)

    def test_build_phrase_holdout_rows_adds_non_oracle_token_support_features(self):
        sequences = [
            _sequence("pos-a", ["A", "B", "C", "D"], quality=0.99),
            _sequence("neg-a", ["C", "D", "E", "F"], quality=0.96),
            _sequence("neg-b", ["G", "H", "I", "J"], quality=0.95),
            _sequence("neg-c", ["K", "L", "M", "N"], quality=0.94),
        ]

        result = build_phrase_holdout_rows(
            sequences,
            phrase=("A", "B"),
            fold_index=1,
            grammar_order=3,
            smoothing=0.1,
            rare_threshold=0,
            negative_sample_size=2,
            artifact_negative_count=0,
            redundancy_negative_count=2,
            seed=7,
        )

        rows = result["rows"]
        for row in rows:
            for feature in SUPPORT_FEATURE_COLUMNS:
                self.assertIn(feature, row)
                self.assertIsInstance(row[feature], float)
            self.assertNotIn("_primitive_tokens", row)

        repeated_rows = [row for row in rows if row["sample_id"] == "pos-a"]
        self.assertEqual(len(repeated_rows), 3)
        self.assertTrue(all(row["token_duplicate_count"] == 3.0 for row in repeated_rows))
        self.assertTrue(all(row["token_max_neighbor_similarity"] > 0.99 for row in repeated_rows))
        ordinary_rows = [row for row in rows if row["negative_type"] == "ordinary"]
        self.assertTrue(all(row["token_duplicate_count"] == 1.0 for row in ordinary_rows))

    def test_run_motion_phrase_holdout_eval_writes_report_scores_and_summary(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            token_path = root / "tokens.jsonl"
            output_dir = root / "out"
            sequences = [
                _sequence("p1", ["A", "B", "C", "D"]),
                _sequence("p2", ["X", "A", "B", "Y"]),
                _sequence("p3", ["Q", "A", "B", "R"]),
                _sequence("n1", ["C", "D", "E", "F"]),
                _sequence("n2", ["G", "H", "I", "J"]),
                _sequence("n3", ["K", "L", "M", "N"]),
                _sequence("n4", ["O", "P", "Q", "R"]),
            ]
            write_token_sequences_jsonl(token_path, sequences)
            config = {
                "execution": {"provider": "modal"},
                "artifacts": {
                    "tokens_dir": str(root),
                    "output_dir": str(output_dir),
                },
                "grammar": {
                    "fit_split": "pretrain",
                    "order": 3,
                    "smoothing": 0.1,
                    "rare_threshold": 0,
                },
                "phrase_holdout": {
                    "phrase_len": 2,
                    "min_support": 2,
                    "max_families": 2,
                    "negative_sample_size": 3,
                    "artifact_negative_count": 1,
                    "redundancy_negative_count": 1,
                    "seed": 13,
                },
                "eval": {
                    "k_values": [2, 4],
                    "low_quality_threshold": 0.45,
                },
            }

            result = run_motion_phrase_holdout_eval(
                config,
                token_sequence_path=token_path,
                output_dir=output_dir,
                allow_local_execution=True,
            )

            self.assertEqual(result["mode"], "full")
            self.assertTrue(Path(result["report_path"]).exists())
            self.assertTrue(Path(result["scores_path"]).exists())
            self.assertTrue(Path(result["summary_path"]).exists())
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(report["leakage_audit"]["heldout_fit_overlap_count"], 0)
            self.assertGreaterEqual(report["negative_type_counts"]["artifact"], 1)
            self.assertGreaterEqual(report["negative_type_counts"]["redundancy"], 1)
            self.assertIn("grammar_surprisal_quality_gated", report["variants"])
            with Path(result["scores_path"]).open(newline="", encoding="utf-8") as handle:
                score_rows = list(csv.DictReader(handle))
            self.assertNotIn("contains_heldout_phrase", score_rows[0])
            self.assertNotIn("_primitive_tokens", score_rows[0])
            self.assertIn("is_artifact", score_rows[0])
            self.assertIn("token_neighborhood_density", score_rows[0])


def _sequence(sample_id: str, tokens: list[str], *, quality: float = 1.0) -> TokenSequence:
    return TokenSequence(
        sample_id=sample_id,
        split="pretrain",
        base_token_ids=list(range(len(tokens))),
        primitive_token_ids=tokens,
        primitive_durations_sec=[0.5] * len(tokens),
        quality_score=quality,
    )


if __name__ == "__main__":
    unittest.main()
