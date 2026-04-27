import unittest
from pathlib import Path


class ModalGrammarAblationEntrypointTests(unittest.TestCase):
    def test_modal_grammar_ablation_is_cpu_only_and_artifact_driven(self):
        source = Path("modal_grammar_ablation.py").read_text(encoding="utf-8")

        self.assertIn("marginal-value-grammar-ablation", source)
        self.assertIn("run_grammar_ablation", source)
        self.assertIn("run_leave_cluster_out_ablation", source)
        self.assertIn("leave_cluster_out", source)
        self.assertIn("configs/grammar_ablation.json", source)
        self.assertIn("ARTIFACTS_VOLUME_NAME", source)
        self.assertNotIn('gpu="H100"', source)
        self.assertNotIn("data_volume", source)


if __name__ == "__main__":
    unittest.main()
