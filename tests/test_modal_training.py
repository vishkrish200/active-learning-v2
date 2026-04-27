import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.training.config import (
    LocalTrainingDisabledError,
    build_modal_run_command,
    load_training_config,
    refuse_local_training,
    validate_training_dispatch,
)


class ModalTrainingGuardrailTests(unittest.TestCase):
    def test_training_config_requires_modal_provider_and_fast_gpu(self):
        config = load_training_config("configs/modal_training.json")

        validate_training_dispatch(config)

        self.assertEqual(config["execution"]["provider"], "modal")
        self.assertEqual(config["execution"]["gpu"], "H100")
        self.assertGreater(config["training"]["validation_steps"], config["training"]["smoke_steps"])
        self.assertEqual(config["data"]["train_split"], "pretrain")
        self.assertEqual(config["data"]["holdout_split"], "val")
        self.assertEqual(config["training"]["validation_checkpoint_name"], "validation_encoder_pretrain_only.pt")
        self.assertEqual(config["training"]["full_checkpoint_name"], "ssl_encoder_pretrain_only.pt")

    def test_anticollapse_training_config_uses_distinct_pretrain_only_checkpoint(self):
        config = load_training_config("configs/modal_training_anticollapse.json")

        validate_training_dispatch(config)

        self.assertEqual(config["encoder"]["architecture"], "normalized_vicreg_mlp")
        self.assertIn("normalized_vicreg_pretrain_only", config["training"]["validation_checkpoint_name"])
        self.assertIn("normalized_vicreg_pretrain_only", config["training"]["full_checkpoint_name"])
        self.assertTrue(config["encoder"]["normalization"]["enabled"])
        self.assertGreaterEqual(config["training"]["validation_steps"], 100)

    def test_local_training_is_explicitly_disabled(self):
        with self.assertRaises(LocalTrainingDisabledError) as raised:
            refuse_local_training()

        self.assertIn("Local Mac training is disabled", str(raised.exception))

    def test_validate_rejects_local_provider(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(
                json.dumps(
                    {
                        "execution": {
                            "provider": "local",
                            "gpu": "cpu",
                            "data_volume": "imu-data",
                            "artifacts_volume": "imu-artifacts",
                        },
                        "data": {"root": "/data"},
                        "training": {"max_steps": 1, "batch_size": 2},
                    }
                )
            )

            with self.assertRaises(ValueError):
                validate_training_dispatch(load_training_config(path))

    def test_modal_command_runs_smoke_first_and_requires_explicit_full_flag(self):
        command = build_modal_run_command(
            "configs/modal_training.json",
            run_validation=True,
            run_full=False,
        )

        self.assertEqual(command[0:3], ["modal", "run", "modal_train.py"])
        self.assertIn("--config-path", command)
        self.assertIn("--run-validation", command)
        self.assertNotIn("--run-full", command)
        self.assertNotIn("python", command[0])

    def test_modal_app_requests_h100_and_never_invokes_remote_training_locally(self):
        source = Path("modal_train.py").read_text()

        self.assertIn('gpu="H100"', source)
        self.assertIn("remote_split_audit.remote", source)
        self.assertIn("remote_smoke_train.remote", source)
        self.assertIn("remote_validation_train.remote", source)
        self.assertIn("remote_full_train.remote", source)
        self.assertIn("checkpoint_read_ok", Path("marginal_value/training/torch_train.py").read_text())
        self.assertNotIn("remote_full_train.local", source)
        self.assertNotIn('load_training_config("configs/modal_training.json")', source)


if __name__ == "__main__":
    unittest.main()
