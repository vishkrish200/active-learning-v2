import json
import unittest
from pathlib import Path


class ModalMotionPhraseHoldoutEntrypointTests(unittest.TestCase):
    def test_modal_motion_phrase_holdout_is_cpu_only_and_artifact_driven(self):
        source = Path("modal_motion_phrase_holdout.py").read_text(encoding="utf-8")
        config = json.loads(Path("configs/motion_phrase_holdout_eval.json").read_text(encoding="utf-8"))

        self.assertIn("marginal-value-motion-phrase-holdout", source)
        self.assertIn("run_motion_phrase_holdout_eval", source)
        self.assertIn("configs/motion_phrase_holdout_eval.json", source)
        self.assertIn("ARTIFACTS_VOLUME_NAME", source)
        self.assertNotIn('gpu="H100"', source)
        self.assertNotIn("data_volume", source)
        self.assertEqual(config["execution"]["provider"], "modal")
        self.assertEqual(config["execution"]["artifacts_volume"], "activelearning-imu-rebuild-cache")
        self.assertTrue(config["artifacts"]["tokens_dir"].startswith("/artifacts/"))
        self.assertTrue(config["artifacts"]["output_dir"].startswith("/artifacts/"))


if __name__ == "__main__":
    unittest.main()
