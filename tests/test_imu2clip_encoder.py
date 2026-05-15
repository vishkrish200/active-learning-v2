import unittest

import numpy as np

from marginal_value.models.imu2clip_encoder import IMU2CLIPEncoder, option_b_checkpoint_metadata
from marginal_value.models.imu2clip_inference import IMU2CLIPInference


class IMU2CLIPEncoderTests(unittest.TestCase):
    def test_window_clip_drops_partial_tail_without_padding(self):
        encoder = IMU2CLIPEncoder.__new__(IMU2CLIPEncoder)
        clip = np.arange(11 * 6, dtype=np.float32).reshape(11, 6)

        windows = encoder._window_clip(clip, window_timesteps=4, stride_timesteps=2)

        self.assertEqual(windows.shape, (4, 4, 6))
        np.testing.assert_array_equal(windows[0], clip[0:4])
        np.testing.assert_array_equal(windows[-1], clip[6:10])

    def test_per_channel_normalization_uses_each_clip_channel_independently(self):
        encoder = IMU2CLIPEncoder.__new__(IMU2CLIPEncoder)
        clip = np.asarray(
            [
                [1.0, 10.0, -2.0, 0.5, 3.0, 100.0],
                [2.0, 12.0, -2.0, 0.5, 9.0, 200.0],
                [3.0, 14.0, -2.0, 0.5, 15.0, 300.0],
            ],
            dtype=np.float32,
        )

        normalized = encoder._normalize_clip(clip)

        np.testing.assert_allclose(np.mean(normalized[:, [0, 1, 4, 5]], axis=0), 0.0, atol=1.0e-6)
        np.testing.assert_allclose(np.std(normalized[:, [0, 1, 4, 5]], axis=0), 1.0, atol=1.0e-6)
        np.testing.assert_allclose(normalized[:, [2, 3]], 0.0, atol=1.0e-6)

    def test_infer_architecture_from_mw2_checkpoint_shapes(self):
        encoder = IMU2CLIPEncoder.__new__(IMU2CLIPEncoder)
        state_dict = {
            "net.0.weight": np.zeros((6,), dtype=np.float32),
            "net.0.bias": np.zeros((6,), dtype=np.float32),
            "net.1.net.0.weight": np.zeros((32, 6, 10), dtype=np.float32),
            "net.2.net.0.weight": np.zeros((32, 32, 5), dtype=np.float32),
            "net.3.net.0.weight": np.zeros((32, 32, 5), dtype=np.float32),
            "net.4.weight": np.zeros((32,), dtype=np.float32),
            "net.5.weight_ih_l0": np.zeros((1536, 32), dtype=np.float32),
            "net.5.weight_hh_l0": np.zeros((1536, 512), dtype=np.float32),
        }

        arch = encoder._infer_architecture_from_checkpoint(state_dict)

        self.assertEqual(arch["architecture"], "mw2_stack_rnn_pooling")
        self.assertEqual(arch["in_channels"], 6)
        self.assertEqual(arch["cnn_channels"], [32, 32, 32])
        self.assertEqual(arch["cnn_kernel_sizes"], [10, 5, 5])
        self.assertEqual(arch["gru_hidden_size"], 512)
        self.assertEqual(arch["output_dim"], 512)

    def test_inference_wrapper_batches_variable_length_clips_with_fake_encoder(self):
        class FakeEncoder:
            def encode_clip(self, clip):
                scale = float(len(clip))
                vector = np.zeros(512, dtype=np.float32)
                vector[0] = scale
                vector[1] = 1.0
                return vector / np.linalg.norm(vector)

            def encode_clip_multiscale(self, clip, window_sizes_s, stride_ratio, pool, combine="concat"):
                del window_sizes_s, stride_ratio, pool, combine
                return np.ones(1536, dtype=np.float32) / np.sqrt(1536.0)

        inference = IMU2CLIPInference(encoder=FakeEncoder())
        clips = [
            np.zeros((30, 6), dtype=np.float32),
            np.zeros((45, 6), dtype=np.float32),
        ]

        embeddings = inference.encode_batch(clips)

        self.assertEqual(embeddings.shape, (2, 512))
        self.assertEqual(embeddings.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1.0e-6)
        self.assertEqual(inference.encode_clip_multiscale(clips[0]).shape, (1536,))

    def test_option_b_source_documents_trainable_style_encoder(self):
        from pathlib import Path

        source = Path("marginal_value/models/imu2clip_encoder.py").read_text(encoding="utf-8")
        metadata = option_b_checkpoint_metadata()

        self.assertIn("class IMU2CLIPStyleEncoder", source)
        self.assertIn("SimCLR-style IMU-only contrastive", source)
        self.assertEqual(metadata["architecture"], "imu2clip_style_mw2")
        self.assertEqual(metadata["output_dim"], 512)
        self.assertEqual(metadata["input_format"], "B,C,T")


if __name__ == "__main__":
    unittest.main()
