import json
import unittest
from pathlib import Path


class ModalRerankEvalEntrypointTests(unittest.TestCase):
    def test_modal_rerank_eval_is_cpu_only_and_artifact_driven(self):
        source = Path("modal_rerank_eval.py").read_text(encoding="utf-8")
        config = json.loads(Path("configs/rerank_eval_grammar_promoted.json").read_text(encoding="utf-8"))

        self.assertIn("marginal-value-rerank-eval", source)
        self.assertIn("run_rerank_eval", source)
        self.assertIn("configs/rerank_eval_grammar_promoted.json", source)
        self.assertIn("ARTIFACTS_VOLUME_NAME", source)
        self.assertNotIn('gpu="H100"', source)
        self.assertNotIn("gpu=", source)
        self.assertEqual(config["execution"]["provider"], "modal")
        self.assertEqual(config["execution"]["artifacts_volume"], "activelearning-imu-rebuild-cache")
        self.assertTrue(config["artifacts"]["candidate_path"].startswith("/artifacts/"))
        self.assertTrue(config["artifacts"]["output_dir"].startswith("/artifacts/"))


if __name__ == "__main__":
    unittest.main()
