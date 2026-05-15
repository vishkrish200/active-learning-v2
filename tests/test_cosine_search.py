import unittest

import numpy as np

from marginal_value.indexing.cosine_search import cosine_knn, mean_nearest_cosine_distance


class CosineSearchTests(unittest.TestCase):
    def test_cosine_knn_matches_expected_numpy_ordering(self):
        support = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=float,
        )
        query = np.asarray(
            [
                [0.9, 0.1],
                [0.0, -2.0],
            ],
            dtype=float,
        )

        distances, indices = cosine_knn(support, query, k=2, backend="numpy")

        np.testing.assert_array_equal(indices[0], np.asarray([0, 1]))
        np.testing.assert_array_equal(indices[1], np.asarray([0, 2]))
        self.assertLess(distances[0, 0], distances[0, 1])
        self.assertAlmostEqual(float(distances[1, 0]), 1.0)
        self.assertAlmostEqual(float(distances[1, 1]), 1.0)

    def test_mean_nearest_cosine_distance_uses_nearest_support_neighbor(self):
        support = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=float,
        )
        query = np.asarray(
            [
                [1.0, 0.0],
                [0.0, -1.0],
            ],
            dtype=float,
        )

        distance = mean_nearest_cosine_distance(query, support, backend="numpy")

        self.assertAlmostEqual(distance, 0.5)

    def test_auto_backend_falls_back_to_numpy_without_changing_results(self):
        support = np.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=float)
        query = np.asarray([[1.0, 0.0]], dtype=float)

        auto_distances, auto_indices = cosine_knn(support, query, k=1, backend="auto")
        numpy_distances, numpy_indices = cosine_knn(support, query, k=1, backend="numpy")

        np.testing.assert_allclose(auto_distances, numpy_distances)
        np.testing.assert_array_equal(auto_indices, numpy_indices)


if __name__ == "__main__":
    unittest.main()
