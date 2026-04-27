import json
import unittest
from pathlib import Path


class ModalScoreCalibrationEntrypointTests(unittest.TestCase):
    def test_modal_score_calibration_is_cpu_only_and_artifact_driven(self):
        source = Path("modal_score_calibration.py").read_text(encoding="utf-8")
        config = json.loads(Path("configs/score_calibration_eval.json").read_text(encoding="utf-8"))

        self.assertIn("marginal-value-score-calibration", source)
        self.assertIn("run_score_calibration_eval", source)
        self.assertIn("configs/score_calibration_eval.json", source)
        self.assertIn("ARTIFACTS_VOLUME_NAME", source)
        self.assertNotIn('gpu="H100"', source)
        self.assertNotIn("gpu=", source)
        self.assertEqual(config["execution"]["provider"], "modal")
        self.assertEqual(config["execution"]["artifacts_volume"], "activelearning-imu-rebuild-cache")
        self.assertTrue(config["artifacts"]["output_dir"].startswith("/artifacts/"))
        self.assertGreaterEqual(len(config["artifacts"]["datasets"]), 3)
        for dataset in config["artifacts"]["datasets"]:
            self.assertTrue(dataset["path"].startswith("/artifacts/"))


if __name__ == "__main__":
    unittest.main()
