import unittest

import numpy as np

from marginal_value.models.imagebind_imu_encoder import ImageBindIMUEncoder
from marginal_value.models.imagebind_imu_inference import ImageBindIMUInference


class FakeImageBindIMUBackend:
    def __init__(self):
        self.window_batches = []

    def encode_windows(self, windows):
        self.window_batches.append(np.asarray(windows, dtype=np.float32).copy())
        rows = []
        for index, window in enumerate(windows):
            vector = np.zeros(1024, dtype=np.float32)
            vector[0] = float(index + 1)
            vector[1] = float(np.mean(window))
            vector[2] = float(np.std(window))
            rows.append(vector)
        return np.vstack(rows).astype("float32")


class ImageBindIMUEncoderTests(unittest.TestCase):
    def test_encode_clip_windows_to_six_by_two_thousand_and_drops_partial_tail(self):
        backend = FakeImageBindIMUBackend()
        encoder = ImageBindIMUEncoder(
            backend=backend,
            target_hz=200,
            expected_hz=200,
            window_len_s=10.0,
            stride_ratio=0.5,
            batch_size=2,
        )
        t = np.linspace(0.0, 1.0, 4500, dtype=np.float32)[:, None]
        clip = np.hstack([t + channel for channel in range(6)]).astype("float32")

        embedding = encoder.encode_clip(clip)

        self.assertEqual(embedding.shape, (1024,))
        self.assertEqual(embedding.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(embedding), 1.0, atol=1.0e-6)
        seen_windows = np.concatenate(backend.window_batches, axis=0)
        self.assertEqual(seen_windows.shape, (3, 6, 2000))

    def test_inference_batch_returns_normalized_1024_dim_rows(self):
        backend = FakeImageBindIMUBackend()
        encoder = ImageBindIMUEncoder(
            backend=backend,
            target_hz=200,
            expected_hz=200,
            window_len_s=10.0,
            stride_ratio=1.0,
        )
        inference = ImageBindIMUInference(encoder=encoder)
        clips = [
            np.ones((2000, 6), dtype=np.float32),
            np.full((2000, 6), 2.0, dtype=np.float32),
        ]

        embeddings = inference.encode_batch(clips)

        self.assertEqual(embeddings.shape, (2, 1024))
        self.assertEqual(embeddings.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1.0e-6)

    def test_short_clip_without_complete_window_returns_zero_row(self):
        encoder = ImageBindIMUEncoder(
            backend=FakeImageBindIMUBackend(),
            target_hz=200,
            expected_hz=200,
            window_len_s=10.0,
        )

        embedding = encoder.encode_clip(np.ones((1000, 6), dtype=np.float32))

        self.assertEqual(embedding.shape, (1024,))
        self.assertEqual(embedding.dtype, np.float32)
        np.testing.assert_allclose(embedding, np.zeros(1024, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
