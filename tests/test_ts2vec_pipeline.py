import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from marginal_value.active.registry import ClipRecord


try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local project intentionally avoids torch.
    torch = None


class TS2VecPipelineTests(unittest.TestCase):
    def test_check_collapse_reports_rank_and_cosine(self):
        from marginal_value.training.train_ts2vec import check_collapse

        rng = np.random.default_rng(7)
        embeddings = rng.normal(size=(64, 16)).astype("float32")
        metrics = check_collapse(embeddings)

        self.assertGreater(metrics["effective_rank"], 5.0)
        self.assertLess(metrics["mean_pairwise_cosine"], 0.5)
        self.assertIn("std_per_dim_mean", metrics)

    def test_embedding_cache_registers_ts2vec_with_checkpoint_option(self):
        from marginal_value.active.embedding_cache import (
            SUPPORTED_REPRESENTATIONS,
            load_embedding_lookup,
        )

        self.assertIn("ts2vec", SUPPORTED_REPRESENTATIONS)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = _write_raw_clip(root, "clip-a")

            class FakeInference:
                def __init__(self, checkpoint_path, device="cpu"):
                    self.checkpoint_path = checkpoint_path
                    self.device = device

                def encode_clip(self, values):
                    if values.shape[1] != 6:
                        raise AssertionError(f"expected 6 channels, got {values.shape[1]}")
                    return np.asarray([1.0, 0.0, 0.0], dtype="float32")

            with patch("marginal_value.active.embedding_cache.TS2VecInference", FakeInference):
                result = load_embedding_lookup(
                    [clip],
                    representations=["ts2vec"],
                    sample_rate=30.0,
                    raw_shape_max_samples=90,
                    cache_dir=None,
                    component="test_ts2vec_cache",
                    mode="smoke",
                    representation_options={"ts2vec_checkpoint_path": "/tmp/ts2vec.pt"},
                )

        np.testing.assert_allclose(result.embeddings["ts2vec"]["clip-a"], np.asarray([1.0, 0.0, 0.0], dtype="float32"))

    def test_active_loop_config_accepts_ts2vec_representation(self):
        from marginal_value.active.evaluate_active_loop import validate_active_loop_eval_config

        config = {
            "execution": {"provider": "modal", "allow_local_paths_for_tests": True},
            "data": {
                "root": "/tmp/data",
                "manifests": {"pretrain": "cache/manifests/pretrain.txt"},
            },
            "artifacts": {"output_dir": "/tmp/artifacts"},
            "episodes": {"path": "/tmp/episodes.jsonl"},
            "evaluation": {
                "representations": ["ts2vec", "window_mean_std_pool"],
                "primary_representation": "ts2vec",
                "k_values": [5],
            },
        }

        validate_active_loop_eval_config(config)

    def test_append_training_log_writes_markdown_table_row(self):
        from marginal_value.training.train_ts2vec import append_training_log

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "docs" / "ts2vec_training_log.md"
            append_training_log(
                path,
                epoch=3,
                train_loss=1.25,
                collapse_metrics={
                    "effective_rank": 12.5,
                    "mean_pairwise_cosine": 0.22,
                },
                instance_loss=0.75,
                temporal_loss=1.5,
            )
            text = path.read_text(encoding="utf-8")

        self.assertIn("| epoch | effective_rank | mean_pairwise_cosine | train_loss | instance_loss | temporal_loss |", text)
        self.assertIn("| 3 | 12.5000 | 0.2200 | 1.250000 | 0.750000 | 1.500000 |", text)

    def test_alpha_schedule_supports_instance_warmup(self):
        from marginal_value.training.train_ts2vec import _alpha_for_epoch

        self.assertEqual(_alpha_for_epoch(1, alpha=0.1, instance_warmup_epochs=1), 0.0)
        self.assertEqual(_alpha_for_epoch(2, alpha=0.1, instance_warmup_epochs=1), 0.1)
        self.assertEqual(_alpha_for_epoch(1, alpha=0.8, instance_warmup_epochs=0), 0.8)

    def test_collapse_pooling_modes_are_validated(self):
        from marginal_value.training.train_ts2vec import _validate_collapse_pooling

        for mode in ("aggregation", "max", "mean", "mean_max"):
            self.assertEqual(_validate_collapse_pooling(mode), mode)
        with self.assertRaises(ValueError):
            _validate_collapse_pooling("median")

    def test_collapse_gate_allows_candidate_thresholds(self):
        from marginal_value.training.train_ts2vec import _is_collapsed

        metrics = {"effective_rank": 6.5, "mean_pairwise_cosine": 0.82}

        self.assertTrue(_is_collapsed(metrics))
        self.assertFalse(
            _is_collapsed(
                metrics,
                min_effective_rank=0.0,
                max_mean_pairwise_cosine=0.99,
            )
        )

    def test_training_defaults_are_short_crop_temporal_light(self):
        import inspect

        from marginal_value.training.train_ts2vec import train_ts2vec

        signature = inspect.signature(train_ts2vec)

        self.assertEqual(signature.parameters["crop_min_len"].default, 150)
        self.assertEqual(signature.parameters["crop_max_len"].default, 600)
        self.assertEqual(signature.parameters["alpha"].default, 0.1)
        self.assertEqual(signature.parameters["max_temporal_positions"].default, 64)
        self.assertEqual(signature.parameters["collapse_pooling"].default, "aggregation")
        self.assertEqual(signature.parameters["collapse_min_effective_rank"].default, 10.0)
        self.assertEqual(signature.parameters["collapse_max_mean_pairwise_cosine"].default, 0.9)

    def test_encoder_exports_aggregation_head(self):
        from marginal_value.models.ts2vec_encoder import AggregationHead, NonlinearInputProjection, expand_imu_features

        self.assertIsNotNone(AggregationHead)
        self.assertIsNotNone(NonlinearInputProjection)
        self.assertIsNotNone(expand_imu_features)

    def test_aggregation_head_avoids_output_layernorm(self):
        source = Path("marginal_value/models/ts2vec_encoder.py").read_text(encoding="utf-8")

        self.assertNotIn("self.norm = nn.LayerNorm(int(output_dims))", source)

    def test_encoder_uses_nonlinear_input_projection(self):
        source = Path("marginal_value/models/ts2vec_encoder.py").read_text(encoding="utf-8")

        self.assertIn("class NonlinearInputProjection", source)
        self.assertIn("nn.GELU()", source)
        self.assertIn("expand_imu_features(values)", source)
        self.assertIn("projection_channels = 14", source)
        self.assertNotIn("self.input_projection = nn.Linear(input_channels, input_dims)", source)

    @unittest.skipIf(torch is None, "PyTorch is only required inside Modal training/inference paths.")
    def test_imu_feature_expansion_adds_physical_channels(self):
        from marginal_value.models.ts2vec_encoder import expand_imu_features

        values = torch.tensor([[[3.0, 4.0, 0.0, 0.0, 0.0, 2.0]]])
        expanded = expand_imu_features(values)

        self.assertEqual(tuple(expanded.shape), (1, 1, 14))
        torch.testing.assert_close(expanded[:, :, :6], values)
        torch.testing.assert_close(expanded[0, 0, 6], torch.tensor(5.0))
        torch.testing.assert_close(expanded[0, 0, 7], torch.tensor(2.0))
        torch.testing.assert_close(expanded[0, 0, 8:11], torch.tensor([8.0, -6.0, 0.0]))
        torch.testing.assert_close(expanded[0, 0, 11:14], torch.zeros(3))

    @unittest.skipIf(torch is None, "PyTorch is only required inside Modal training/inference paths.")
    def test_encoder_is_causal_and_returns_clip_embedding(self):
        from marginal_value.models.ts2vec_encoder import TS2VecEncoder

        model = TS2VecEncoder(input_dims=8, hidden_dims=8, output_dims=12, n_layers=3)
        model.eval()
        x = torch.randn(2, 32, 6)
        y = x.clone()
        y[:, 20:] = y[:, 20:] + 100.0

        with torch.no_grad():
            x_seq = model.encode_sequence(x, return_layers=False)
            y_seq = model.encode_sequence(y, return_layers=False)
            clip = model(x)

        self.assertEqual(tuple(clip.shape), (2, 12))
        self.assertEqual(tuple(x_seq.shape), (2, 32, 12))
        torch.testing.assert_close(x_seq[:, :20], y_seq[:, :20], rtol=1.0e-5, atol=1.0e-5)

    @unittest.skipIf(torch is None, "PyTorch is only required inside Modal training/inference paths.")
    def test_hierarchical_loss_prefers_matching_views(self):
        from marginal_value.models.ts2vec_loss import hierarchical_contrastive_loss

        z1 = [torch.randn(4, 16, 8), torch.randn(4, 8, 8)]
        z2 = [values.clone() for values in z1]
        shuffled = [values.flip(0) for values in z1]

        matching = hierarchical_contrastive_loss(z1, z2)
        mismatched = hierarchical_contrastive_loss(z1, shuffled)

        self.assertTrue(torch.isfinite(matching))
        self.assertLess(float(matching), float(mismatched))

    @unittest.skipIf(torch is None, "PyTorch is only required inside Modal training/inference paths.")
    def test_temporal_loss_uses_explicit_overlap_alignment(self):
        from marginal_value.models.ts2vec_loss import temporal_contrastive_loss

        base = torch.eye(6, dtype=torch.float32)
        left = base[0:4].unsqueeze(0)
        right = base[2:6].unsqueeze(0)

        aligned = temporal_contrastive_loss(
            left,
            right,
            overlap_indices=[([2, 3], [0, 1])],
            temperature=0.1,
        )
        local_offset = temporal_contrastive_loss(left, right, temperature=0.1)

        self.assertLess(float(aligned), float(local_offset))

    def test_fixed_crop_views_return_true_overlap_indices(self):
        from marginal_value.training.train_ts2vec import _two_overlapping_fixed_crops

        values = np.arange(10, dtype=np.float32)[:, None].repeat(6, axis=1)
        rng = _FixedRng([0, 3])

        left, right, left_indices, right_indices = _two_overlapping_fixed_crops(values, crop_len=6, rng=rng)

        self.assertEqual(left[0, 0], 0.0)
        self.assertEqual(right[0, 0], 3.0)
        np.testing.assert_array_equal(left[left_indices, 0], right[right_indices, 0])
        np.testing.assert_array_equal(left_indices, np.asarray([3, 4, 5]))
        np.testing.assert_array_equal(right_indices, np.asarray([0, 1, 2]))

    def test_create_overlapping_crops_returns_distinct_aligned_views(self):
        from marginal_value.models.ts2vec_loss import create_overlapping_crops

        values = np.arange(10, dtype=np.float32)[:, None].repeat(6, axis=1)
        rng = _FixedRng([6, 3, 0, 0])

        left, right, left_indices, right_indices = create_overlapping_crops(
            values,
            min_overlap=0.5,
            crop_min_len=6,
            crop_max_len=6,
            rng=rng,
        )

        self.assertFalse(np.array_equal(left, right))
        self.assertEqual(left.shape, (6, 6))
        self.assertEqual(right.shape, (6, 6))
        np.testing.assert_array_equal(left[left_indices, 0], right[right_indices, 0])

    def test_overlap_subsampling_preserves_alignment_and_bounds(self):
        from marginal_value.models.ts2vec_loss import subsample_overlap_indices

        left = np.arange(1000, dtype=np.int64)
        right = left + 17

        sampled_left, sampled_right = subsample_overlap_indices(left, right, max_positions=128)

        self.assertEqual(len(sampled_left), 128)
        self.assertEqual(len(sampled_right), 128)
        np.testing.assert_array_equal(sampled_right - sampled_left, np.full(128, 17))
        self.assertEqual(sampled_left[0], 0)
        self.assertEqual(sampled_left[-1], 999)

    @unittest.skipIf(torch is None, "PyTorch is only required inside Modal training/inference paths.")
    def test_inference_l2_normalizes_checkpoint_embeddings(self):
        from marginal_value.models.ts2vec_encoder import TS2VecEncoder
        from marginal_value.models.ts2vec_inference import TS2VecInference

        with TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.pt"
            model = TS2VecEncoder(input_dims=8, hidden_dims=8, output_dims=10, n_layers=2)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "input_dims": 8,
                        "hidden_dims": 8,
                        "output_dims": 10,
                        "n_layers": 2,
                    },
                },
                checkpoint_path,
            )
            inference = TS2VecInference(str(checkpoint_path), device="cpu")
            clip = np.random.default_rng(3).normal(size=(96, 6)).astype("float32")
            embedding = inference.encode_clip(clip)

        self.assertEqual(embedding.shape, (10,))
        self.assertEqual(embedding.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=5)

    @unittest.skipIf(torch is None, "PyTorch is only required inside Modal training/inference paths.")
    def test_inference_batch_matches_individual_variable_length_clips(self):
        from marginal_value.models.ts2vec_encoder import TS2VecEncoder
        from marginal_value.models.ts2vec_inference import TS2VecInference

        with TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.pt"
            model = TS2VecEncoder(input_dims=8, hidden_dims=8, output_dims=10, n_layers=2)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "input_dims": 8,
                        "hidden_dims": 8,
                        "output_dims": 10,
                        "n_layers": 2,
                    },
                },
                checkpoint_path,
            )
            inference = TS2VecInference(str(checkpoint_path), device="cpu")
            rng = np.random.default_rng(4)
            clips = [
                rng.normal(size=(48, 6)).astype("float32"),
                rng.normal(size=(64, 6)).astype("float32"),
                rng.normal(size=(51, 6)).astype("float32"),
            ]
            individual = np.vstack([inference.encode_clip(clip) for clip in clips])
            batched = inference.encode_batch(clips, batch_size=3)

        np.testing.assert_allclose(batched, individual, rtol=1.0e-5, atol=1.0e-5)


def _write_raw_clip(root: Path, sample_id: str) -> ClipRecord:
    raw_dir = root / "cache" / "raw"
    feature_dir = root / "cache" / "features"
    raw_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{sample_id}.jsonl"
    feature_path = feature_dir / f"{sample_id}.npz"
    rows = []
    for idx in range(90):
        rows.append(
            json.dumps(
                {
                    "t_us": idx * 33333,
                    "acc": [float(np.sin(idx / 10.0)), 0.1, 9.81],
                    "gyro": [0.01, float(np.cos(idx / 9.0)), 0.02],
                }
            )
        )
    raw_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    np.savez(feature_path, window_features=np.zeros((2, 4), dtype="float32"))
    return ClipRecord(
        sample_id=sample_id,
        split="pretrain",
        url=f"https://storage.googleapis.com/unit/pretrain/worker00000/{sample_id}.jsonl",
        source_group_id="worker00000",
        worker_id="worker00000",
        raw_path=raw_path,
        feature_path=feature_path,
    )


class _FixedRng:
    def __init__(self, values):
        self.values = list(values)

    def integers(self, low, high=None, *args, **kwargs):
        value = self.values.pop(0)
        if high is None:
            high = low
            low = 0
        if value < low or value >= high:
            raise AssertionError(f"fixed RNG value {value} outside [{low}, {high})")
        return value


if __name__ == "__main__":
    unittest.main()
