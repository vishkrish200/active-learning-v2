import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.tokenization.config import (
    TokenizationLeakageError,
    build_modal_grammar_command,
    build_modal_tokenizer_command,
    load_grammar_config,
    load_tokenizer_config,
    validate_grammar_config,
    validate_tokenizer_config,
)


class TokenizationConfigTests(unittest.TestCase):
    def test_tokenizer_config_requires_modal_h100_and_pretrain_fit(self):
        config = load_tokenizer_config("configs/modal_tokenizer.json")

        validate_tokenizer_config(config)

        self.assertEqual(config["execution"]["provider"], "modal")
        self.assertEqual(config["execution"]["gpu"], "H100")
        self.assertEqual(config["splits"]["fit_split"], "pretrain")
        self.assertEqual(config["splits"]["transform_splits"], ["pretrain", "val"])
        self.assertEqual(config["patches"]["feature_source"], "npz_features")
        self.assertTrue(config["artifacts"]["output_dir"].startswith("/artifacts/tokens/"))

    def test_tokenizer_config_rejects_val_fit_leakage(self):
        config = _valid_tokenizer_config()
        config["splits"]["fit_split"] = "val"

        with self.assertRaises(TokenizationLeakageError):
            validate_tokenizer_config(config)

    def test_tokenizer_config_rejects_transform_splits_without_pretrain(self):
        config = _valid_tokenizer_config()
        config["splits"]["transform_splits"] = ["val"]

        with self.assertRaises(TokenizationLeakageError):
            validate_tokenizer_config(config)

    def test_tokenizer_config_rejects_local_provider(self):
        config = _valid_tokenizer_config()
        config["execution"]["provider"] = "local"

        with self.assertRaises(ValueError):
            validate_tokenizer_config(config)

    def test_tokenizer_modal_command_runs_smoke_by_default(self):
        command = build_modal_tokenizer_command("configs/modal_tokenizer.json", run_full=False)

        self.assertEqual(command[:3], ["modal", "run", "modal_tokenizer.py"])
        self.assertIn("--config-path", command)
        self.assertNotIn("--run-full", command)
        self.assertNotIn("python", command[0])

    def test_grammar_config_requires_pretrain_fit_and_val_score(self):
        config = load_grammar_config("configs/modal_grammar.json")

        validate_grammar_config(config)

        self.assertEqual(config["execution"]["provider"], "modal")
        self.assertEqual(config["splits"]["fit_split"], "pretrain")
        self.assertEqual(config["splits"]["score_split"], "val")
        self.assertTrue(config["tokens"]["input_dir"].startswith("/artifacts/tokens/"))
        self.assertEqual(config["grammar"]["model"], "ngram")

    def test_grammar_config_rejects_non_pretrain_fit(self):
        config = _valid_grammar_config()
        config["splits"]["fit_split"] = "val"

        with self.assertRaises(TokenizationLeakageError):
            validate_grammar_config(config)

    def test_grammar_config_rejects_scoring_the_fit_split(self):
        config = _valid_grammar_config()
        config["splits"]["score_split"] = "pretrain"

        with self.assertRaises(TokenizationLeakageError):
            validate_grammar_config(config)

    def test_grammar_config_allows_explicit_pretrain_support_scoring(self):
        config = _valid_grammar_config()
        config["splits"]["score_split"] = "pretrain"
        config["splits"]["allow_fit_split_scoring"] = True

        validate_grammar_config(config)

        self.assertTrue(config["splits"]["allow_fit_split_scoring"])

    def test_grammar_modal_command_requires_explicit_full_flag(self):
        command = build_modal_grammar_command("configs/modal_grammar.json", run_full=True)

        self.assertEqual(command[:3], ["modal", "run", "modal_grammar.py"])
        self.assertIn("--config-path", command)
        self.assertIn("--run-full", command)

    def test_config_loaders_read_json_files(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.json"
            path.write_text(json.dumps(_valid_tokenizer_config()), encoding="utf-8")

            loaded = load_tokenizer_config(path)

            self.assertEqual(loaded["splits"]["fit_split"], "pretrain")


def _valid_tokenizer_config() -> dict:
    return {
        "execution": {
            "provider": "modal",
            "gpu": "H100",
            "data_volume": "imu-novelty-subset-data",
            "artifacts_volume": "activelearning-imu-rebuild-cache",
            "timeout_seconds": 7200,
        },
        "data": {
            "root": "/data",
            "format": "npz_features",
            "feature_glob": "cache/features/*.npz",
            "raw_glob": "cache/raw/*.jsonl",
            "pretrain_manifest": "cache/manifests/pretrain_urls.txt",
            "val_manifest": "cache/manifests/val_urls.txt",
            "feature_dim": 75,
        },
        "artifacts": {
            "root": "/artifacts",
            "output_dir": "/artifacts/tokens/window_mean_std_tokenizer",
        },
        "splits": {
            "fit_split": "pretrain",
            "transform_splits": ["pretrain", "val"],
        },
        "patches": {
            "feature_source": "npz_features",
            "patch_len_sec": 0.5,
            "patch_stride_sec": 0.25,
            "sample_rate": 30.0,
            "smoke_fit_samples": 128,
            "smoke_transform_samples_per_split": 64,
            "full_fit_samples": 5000,
        },
        "vq": {
            "codebook_size": 256,
            "n_iter": 25,
            "seed": 7,
        },
        "bpe": {
            "num_merges": 2048,
            "min_count": 2,
            "base_step_sec": 0.25,
            "max_primitive_duration_sec": 12.0,
        },
        "diagnostics": {
            "top_k_primitives": 25,
        },
    }


def _valid_grammar_config() -> dict:
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


if __name__ == "__main__":
    unittest.main()
