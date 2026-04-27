import json
import unittest
from pathlib import Path


class ModalShadowRankingEntrypointTests(unittest.TestCase):
    def test_modal_shadow_ranking_is_cpu_only_and_artifact_driven(self):
        source = Path("modal_shadow_ranking.py").read_text(encoding="utf-8")
        config = json.loads(Path("configs/shadow_quality_gated_grammar.json").read_text(encoding="utf-8"))

        self.assertIn("marginal-value-shadow-ranking", source)
        self.assertIn("run_shadow_ranking_eval", source)
        self.assertIn("configs/shadow_quality_gated_grammar.json", source)
        self.assertIn("ARTIFACTS_VOLUME_NAME", source)
        self.assertNotIn('gpu="H100"', source)
        self.assertNotIn("gpu=", source)
        self.assertEqual(config["execution"]["provider"], "modal")
        self.assertEqual(config["execution"]["artifacts_volume"], "activelearning-imu-rebuild-cache")
        for key in ("diagnostics_path", "candidate_path", "quality_metadata_path", "output_dir"):
            self.assertTrue(config["artifacts"][key].startswith("/artifacts/"))

    def test_shadow_diversity_config_compares_cpu_variants_with_top200_gate(self):
        config = json.loads(Path("configs/shadow_quality_gated_grammar_diversity.json").read_text(encoding="utf-8"))

        variants = config["shadow"]["diversity_variants"]
        self.assertEqual(config["execution"]["provider"], "modal")
        self.assertEqual(config["artifacts"]["output_dir"], "/artifacts/eval/shadow_ranking/quality_gated_grammar_diversity")
        self.assertIn("cluster_bonus", [variant["diversity_method"] for variant in variants])
        self.assertIn("cluster_round_robin", [variant["diversity_method"] for variant in variants])
        top200_caps = [variant["cluster_max_per_cluster"] for variant in variants if variant.get("cluster_cap_top_k") == 200]
        self.assertEqual(top200_caps, [2, 3, 4])
        self.assertEqual(config["selection_criteria"]["candidate_top_k"], 200)
        self.assertEqual(config["selection_criteria"]["min_unique_clusters"], 30)
        self.assertGreaterEqual(config["selection_criteria"]["min_positive_fraction"], 0.9)


if __name__ == "__main__":
    unittest.main()
