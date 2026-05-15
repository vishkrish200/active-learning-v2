import unittest
import json
from pathlib import Path


class IMU2CLIPOptionBTrainingTests(unittest.TestCase):
    def test_modal_option_b_entrypoint_trains_and_audits_reproduced_checkpoint(self):
        source = Path("modal_train_imu2clip_style.py").read_text(encoding="utf-8")

        self.assertIn("imu2clip-style-training", source)
        self.assertIn('"torch==2.8.0"', source)
        self.assertIn('"scipy==1.14.1"', source)
        self.assertIn("train_imu2clip_style_encoder", source)
        self.assertIn("run_embedding_audit", source)
        self.assertIn("remote_train_and_audit_imu2clip_style", source)
        self.assertIn("remote_preload_imu2clip_windows", source)
        self.assertIn("preload_before_gpu", source)
        self.assertIn(".spawn(config, mode=", source)
        self.assertIn("effective_rank", source)
        self.assertIn('gpu="H100"', source)
        self.assertIn("run_id", source)
        self.assertIn("local_dispatch_start", source)
        self.assertIn("remote_audit_done", source)
        self.assertNotIn("run_id=run_id, **result", source)

        config = json.loads(Path("configs/imu2clip_style_training.json").read_text(encoding="utf-8"))
        self.assertEqual(config["execution"]["gpu"], "H100")
        self.assertTrue(config["data"]["preload_windows"])
        self.assertTrue(config["data"]["preload_before_gpu"])
        self.assertIn("preloaded_windows_path", config["data"])
        self.assertLessEqual(config["data"]["validation_max_files"], 4096)
        self.assertLessEqual(config["training"]["validation_steps"], 300)
        self.assertLessEqual(config["training"]["progress_every_steps"], 10)

    def test_training_module_keeps_torch_import_inside_modal_function_path(self):
        source = Path("marginal_value/training/imu2clip_style_train.py").read_text(encoding="utf-8")

        self.assertIn("def train_imu2clip_style_encoder", source)
        self.assertIn("def _nt_xent_loss", source)
        self.assertIn("IMU2CLIPStyleEncoder", source)
        self.assertIn("dataset_ready", source)
        self.assertIn("preload_progress", source)
        self.assertIn("preload_ready", source)
        self.assertIn("preload_imu2clip_style_windows", source)
        self.assertIn("preloaded_from_cache", source)
        self.assertIn("_sample_preloaded_batch", source)
        self.assertIn("checkpoint_write_start", source)
        self.assertIn("run_id", source)
        self.assertNotIn("import torch\n", source.split("def train_imu2clip_style_encoder", 1)[0])


if __name__ == "__main__":
    unittest.main()
