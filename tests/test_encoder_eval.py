import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.eval.encoder_eval import (
    EvalLeakageError,
    cosine_knn,
    embedding_diagnostics,
    evaluate_retrieval,
    load_eval_config,
    validate_eval_config,
)


class EncoderEvalTests(unittest.TestCase):
    def test_eval_config_uses_pretrain_support_and_val_query(self):
        config = load_eval_config("configs/eval_encoder.json")

        validate_eval_config(config)

        self.assertEqual(config["splits"]["support_split"], "pretrain")
        self.assertEqual(config["splits"]["query_split"], "val")
        self.assertIn("pretrain_only", config["checkpoint"]["path"])

    def test_anticollapse_eval_config_targets_validation_checkpoint_with_acceptance_gate(self):
        config = load_eval_config("configs/eval_encoder_anticollapse_validation.json")

        validate_eval_config(config)

        self.assertIn("validation_encoder_normalized_vicreg_pretrain_only.pt", config["checkpoint"]["path"])
        self.assertEqual(config["splits"]["support_split"], "pretrain")
        self.assertEqual(config["splits"]["query_split"], "val")
        self.assertGreater(config["acceptance"]["min_effective_rank"], 1.0)

    def test_eval_config_rejects_same_support_and_query_split(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_eval.json"
            path.write_text(
                json.dumps(
                    {
                        "execution": {"provider": "modal", "gpu": "H100"},
                        "data": {"root": "/data"},
                        "artifacts": {"root": "/artifacts"},
                        "checkpoint": {"path": "/artifacts/checkpoints/ssl_encoder_pretrain_only.pt"},
                        "splits": {"support_split": "val", "query_split": "val"},
                        "eval": {"k_values": [1]},
                    }
                )
            )

            with self.assertRaises(EvalLeakageError):
                validate_eval_config(load_eval_config(path))

    def test_cosine_knn_orders_nearest_support_items(self):
        support = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        query = np.array([[0.9, 0.1], [-0.8, 0.1]])

        distances, indices = cosine_knn(support, query, k=2)

        self.assertEqual(indices[0, 0], 0)
        self.assertEqual(indices[1, 0], 2)
        self.assertLess(distances[0, 0], distances[0, 1])

    def test_retrieval_eval_reports_encoder_vs_baseline(self):
        support_baseline = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        query_baseline = np.array([[0.9, 0.1], [-0.8, 0.1]])
        support_encoder = support_baseline.copy()
        query_encoder = query_baseline.copy()

        report = evaluate_retrieval(
            support_encoder=support_encoder,
            query_encoder=query_encoder,
            support_baseline=support_baseline,
            query_baseline=query_baseline,
            k_values=(1, 2),
        )

        self.assertEqual(report["n_support"], 3)
        self.assertEqual(report["n_query"], 2)
        self.assertIn("encoder", report)
        self.assertIn("baseline", report)
        self.assertIn("mean_knn_d1", report["encoder"])
        self.assertIn("diagnostics", report["encoder"])
        self.assertGreater(report["encoder"]["diagnostics"]["effective_rank"], 1.0)

    def test_embedding_diagnostics_detects_collapsed_embeddings(self):
        collapsed = np.ones((10, 4))
        spread = np.eye(4)

        collapsed_diag = embedding_diagnostics(collapsed)
        spread_diag = embedding_diagnostics(spread)

        self.assertLess(collapsed_diag["mean_dimension_std"], spread_diag["mean_dimension_std"])
        self.assertLess(collapsed_diag["effective_rank"], spread_diag["effective_rank"])

    def test_modal_eval_uses_remote_function_and_clean_checkpoint(self):
        source = Path("modal_eval.py").read_text()

        self.assertIn("remote_encoder_eval.remote", source)
        self.assertIn("ssl_encoder_pretrain_only.pt", Path("configs/eval_encoder.json").read_text())
        self.assertNotIn("ssl_encoder_baseline.pt", Path("configs/eval_encoder.json").read_text())

    def test_modal_ablation_uses_plan_phase_a_baselines(self):
        source = Path("marginal_value/eval/modal_ablation_eval.py").read_text()

        self.assertIn("clip_features_knn", source)
        self.assertIn("window_mean_pool_knn", source)
        self.assertIn("encoder_hidden_knn", source)
        self.assertIn("clip_knn_density_mmr", source)


if __name__ == "__main__":
    unittest.main()
