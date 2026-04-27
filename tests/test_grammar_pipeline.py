import csv
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.tokenization.artifacts import TokenSequence, write_token_sequences_jsonl
from marginal_value.tokenization.modal_grammar import run_grammar_pipeline


class GrammarPipelineTests(unittest.TestCase):
    def test_grammar_pipeline_fits_pretrain_and_scores_val_only(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            token_path = root / "token_sequences_smoke.jsonl"
            write_token_sequences_jsonl(token_path, _sequences())
            output_dir = root / "grammar"

            result = run_grammar_pipeline(
                _grammar_config(root),
                token_sequence_path=token_path,
                output_dir=output_dir,
                smoke=True,
            )

            self.assertEqual(result["mode"], "smoke")
            self.assertEqual(result["fit_split"], "pretrain")
            self.assertEqual(result["score_split"], "val")
            self.assertEqual(result["fit_sequence_split_counts"], {"pretrain": 2})
            self.assertEqual(result["scored_sequence_split_counts"], {"val": 2})
            self.assertTrue(Path(result["feature_path"]).exists())
            self.assertTrue(Path(result["report_path"]).exists())

            rows = _read_csv(Path(result["feature_path"]))
            self.assertEqual([row["split"] for row in rows], ["val", "val"])
            self.assertEqual({row["sample_id"] for row in rows}, {"val-familiar", "val-rare"})
            for key in _expected_feature_columns():
                self.assertIn(key, rows[0])

            familiar = next(row for row in rows if row["sample_id"] == "val-familiar")
            rare = next(row for row in rows if row["sample_id"] == "val-rare")
            self.assertGreater(float(rare["token_nll_mean"]), float(familiar["token_nll_mean"]))

            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(report["leakage_audit"]["grammar_fit_splits"], ["pretrain"])
            self.assertEqual(report["leakage_audit"]["scored_splits"], ["val"])
            self.assertGreater(report["vocabulary_size"], 0)

    def test_grammar_pipeline_rejects_missing_fit_split_sequences(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            token_path = root / "token_sequences_smoke.jsonl"
            write_token_sequences_jsonl(token_path, [sequence for sequence in _sequences() if sequence.split == "val"])

            with self.assertRaises(ValueError):
                run_grammar_pipeline(
                    _grammar_config(root),
                    token_sequence_path=token_path,
                    output_dir=root / "grammar",
                    smoke=True,
                )

    def test_grammar_pipeline_can_write_to_overridden_output_dir(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            token_path = root / "token_sequences_smoke.jsonl"
            write_token_sequences_jsonl(token_path, _sequences())
            config = _grammar_config(root)

            result = run_grammar_pipeline(config, token_sequence_path=token_path, output_dir=root / "custom", smoke=True)

            self.assertTrue(str(result["feature_path"]).startswith(str(root / "custom")))

    def test_grammar_pipeline_can_score_pretrain_support_when_explicitly_allowed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            token_path = root / "token_sequences_smoke.jsonl"
            write_token_sequences_jsonl(token_path, _sequences())
            config = _grammar_config(root)
            config["splits"]["score_split"] = "pretrain"
            config["splits"]["allow_fit_split_scoring"] = True

            result = run_grammar_pipeline(config, token_sequence_path=token_path, output_dir=root / "grammar", smoke=True)

            self.assertEqual(result["score_split"], "pretrain")
            rows = _read_csv(Path(result["feature_path"]))
            self.assertEqual([row["split"] for row in rows], ["pretrain", "pretrain"])
            self.assertEqual(Path(result["feature_path"]).name, "grammar_features_pretrain_smoke.csv")

    def test_grammar_pipeline_emits_read_and_score_progress_logs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            token_path = root / "token_sequences_smoke.jsonl"
            write_token_sequences_jsonl(token_path, _sequences())
            buffer = io.StringIO()

            with redirect_stdout(buffer):
                run_grammar_pipeline(
                    _grammar_config(root),
                    token_sequence_path=token_path,
                    output_dir=root / "grammar",
                    smoke=True,
                )

            events = [json.loads(line)["event"] for line in buffer.getvalue().splitlines()]
            self.assertIn("sequence_read_start", events)
            self.assertIn("sequence_read_done", events)
            self.assertIn("score_sequence_progress", events)
            self.assertIn("feature_write_done", events)


def _sequences() -> list[TokenSequence]:
    return [
        TokenSequence(
            sample_id="pre-a",
            split="pretrain",
            base_token_ids=[1, 2, 3, 1],
            primitive_token_ids=["reach", "inspect", "return", "reach"],
            primitive_durations_sec=[0.5, 0.75, 0.5, 0.5],
        ),
        TokenSequence(
            sample_id="pre-b",
            split="pretrain",
            base_token_ids=[1, 2, 3, 1],
            primitive_token_ids=["reach", "inspect", "return", "reach"],
            primitive_durations_sec=[0.5, 0.75, 0.5, 0.5],
        ),
        TokenSequence(
            sample_id="val-familiar",
            split="val",
            base_token_ids=[1, 2, 3, 1],
            primitive_token_ids=["reach", "inspect", "return", "reach"],
            primitive_durations_sec=[0.5, 0.75, 0.5, 0.5],
        ),
        TokenSequence(
            sample_id="val-rare",
            split="val",
            base_token_ids=[9, 9, 8, 7],
            primitive_token_ids=["rare", "rare", "pause", "twist"],
            primitive_durations_sec=[0.25, 0.25, 1.0, 0.5],
        ),
    ]


def _grammar_config(root: Path) -> dict:
    return {
        "execution": {
            "provider": "modal",
            "artifacts_volume": "activelearning-imu-rebuild-cache",
            "timeout_seconds": 3600,
        },
        "tokens": {
            "input_dir": "/artifacts/tokens/window_mean_std_tokenizer",
        },
        "artifacts": {
            "root": "/artifacts",
            "output_dir": "/artifacts/tokens/window_mean_std_tokenizer",
        },
        "splits": {
            "fit_split": "pretrain",
            "score_split": "val",
        },
        "grammar": {
            "model": "ngram",
            "order": 3,
            "smoothing": 0.1,
            "rare_threshold": 0,
        },
    }


def _expected_feature_columns() -> list[str]:
    return [
        "token_nll_mean",
        "token_nll_p90",
        "token_nll_p95",
        "transition_nll_mean",
        "transition_nll_p95",
        "rare_bigram_fraction",
        "rare_trigram_fraction",
        "rare_phrase_fraction",
        "longest_unseen_phrase_len",
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
