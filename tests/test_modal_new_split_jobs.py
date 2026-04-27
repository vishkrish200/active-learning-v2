import unittest
from pathlib import Path

from marginal_value.ranking.config import load_ranking_config, validate_ranking_config
from marginal_value.tokenization.config import load_grammar_config, validate_grammar_config
from marginal_value.tokenization.config import load_tokenizer_config, validate_tokenizer_config


class ModalNewSplitJobTests(unittest.TestCase):
    def test_cache_new_split_dispatches_smoke_before_full(self):
        source = Path("modal_cache_new_split.py").read_text(encoding="utf-8")

        self.assertIn("remote_cache_new_split.remote(config, smoke=True)", source)
        self.assertIn("remote_cache_new_split.remote(config, smoke=False)", source)
        self.assertIn("target_volume.commit()", source)

    def test_cache_support_split_dispatches_smoke_before_full(self):
        source = Path("modal_cache_support_split.py").read_text(encoding="utf-8")

        self.assertIn("remote_cache_support_split.remote(config, smoke=True)", source)
        self.assertIn("remote_cache_support_split.remote(config, smoke=False)", source)
        self.assertIn("_run_sharded_full_cache", source)
        self.assertIn("num_shards", source)
        self.assertIn("target_volume.commit()", source)

    def test_tokenizer_transform_dispatches_smoke_before_full_without_refit_gpu(self):
        source = Path("modal_tokenizer_transform.py").read_text(encoding="utf-8")

        self.assertIn("remote_tokenizer_transform.remote(config, smoke=True)", source)
        self.assertIn("remote_tokenizer_transform.remote(config, smoke=False)", source)
        self.assertIn("artifacts_volume.commit()", source)
        self.assertNotIn('gpu="H100"', source)

    def test_new_grammar_config_scores_new_split(self):
        config = load_grammar_config("configs/modal_grammar_new.json")

        validate_grammar_config(config)

        self.assertEqual(config["splits"]["fit_split"], "pretrain")
        self.assertEqual(config["splits"]["score_split"], "new")
        self.assertIn("window_mean_std_tokenizer_new", config["tokens"]["input_dir"])

    def test_new_pretrain_grammar_config_explicitly_scores_support_split(self):
        config = load_grammar_config("configs/modal_grammar_new_pretrain_score.json")

        validate_grammar_config(config)

        self.assertEqual(config["splits"]["fit_split"], "pretrain")
        self.assertEqual(config["splits"]["score_split"], "pretrain")
        self.assertTrue(config["splits"]["allow_fit_split_scoring"])

    def test_worker_coverage_tokenizer_and_grammar_configs_validate(self):
        tokenizer_config = load_tokenizer_config("configs/modal_tokenizer_worker_coverage.json")
        new_grammar_config = load_grammar_config("configs/modal_grammar_worker_coverage_new.json")
        pretrain_grammar_config = load_grammar_config("configs/modal_grammar_worker_coverage_pretrain_score.json")

        validate_tokenizer_config(tokenizer_config)
        validate_grammar_config(new_grammar_config)
        validate_grammar_config(pretrain_grammar_config)

        self.assertIn("worker_coverage", tokenizer_config["artifacts"]["output_dir"])
        self.assertEqual(new_grammar_config["splits"]["score_split"], "new")
        self.assertEqual(pretrain_grammar_config["splits"]["score_split"], "pretrain")
        self.assertTrue(pretrain_grammar_config["splits"]["allow_fit_split_scoring"])

    def test_worker_coverage_ranking_config_validates(self):
        config = load_ranking_config("configs/baseline_ranking_new_quality_gated_grammar_worker_coverage.json")

        validate_ranking_config(config)

        self.assertEqual(config["splits"]["support_split"], "pretrain")
        self.assertEqual(config["splits"]["query_split"], "new")
        self.assertIn("worker_coverage", config["artifacts"]["output_dir"])

    def test_physical_source_ranking_config_uses_physical_pretrain_manifest(self):
        config = load_ranking_config("configs/baseline_ranking_new_quality_gated_grammar_physical_source_hybrid75.json")

        validate_ranking_config(config)

        self.assertEqual(config["data"]["pretrain_manifest"], "cache/manifests/pretrain_physical_source_urls.txt")
        self.assertEqual(config["splits"]["support_split"], "pretrain")
        self.assertEqual(config["splits"]["query_split"], "new")
        self.assertIn("physical_source", config["artifacts"]["output_dir"])


if __name__ == "__main__":
    unittest.main()
