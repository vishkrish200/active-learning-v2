import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.training.feature_scaler import fit_feature_scaler, load_feature_scaler


class FeatureScalerTests(unittest.TestCase):
    def test_scaler_fits_pretrain_window_features_and_round_trips_checkpoint_payload(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path_a = root / "a.npz"
            path_b = root / "b.npz"
            np.savez(path_a, window_features=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype="float32"))
            np.savez(path_b, window_features=np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype="float32"))

            scaler = fit_feature_scaler([path_a, path_b], feature_dim=2)
            transformed = scaler.transform(np.asarray([[4.0, 5.0]], dtype="float32"))
            loaded = load_feature_scaler(scaler.to_checkpoint(), feature_dim=2)

            self.assertEqual(scaler.n_files, 2)
            self.assertEqual(scaler.n_windows, 4)
            np.testing.assert_allclose(scaler.mean, [4.0, 5.0])
            self.assertEqual(transformed.shape, (1, 2))
            self.assertIsNotNone(loaded)
            np.testing.assert_allclose(loaded.mean, scaler.mean)
            np.testing.assert_allclose(loaded.scale, scaler.scale)


if __name__ == "__main__":
    unittest.main()
