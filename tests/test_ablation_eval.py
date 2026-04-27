import unittest

import numpy as np

from marginal_value.eval.ablation_eval import (
    average_precision_at_k,
    cluster_diversity_at_k,
    ndcg_at_k,
    precision_at_k,
    rank_by_score,
)


class AblationEvalTests(unittest.TestCase):
    def test_ranking_metrics_reward_positives_early(self):
        labels_good = np.array([1, 1, 0, 0])
        labels_bad = np.array([0, 0, 1, 1])

        self.assertGreater(ndcg_at_k(labels_good, 4), ndcg_at_k(labels_bad, 4))
        self.assertGreater(average_precision_at_k(labels_good, 4), average_precision_at_k(labels_bad, 4))
        self.assertEqual(precision_at_k(labels_good, 2), 1.0)

    def test_rank_by_score_sorts_descending_and_keeps_labels(self):
        scores = np.array([0.2, 0.9, 0.1])
        labels = np.array([0, 1, 0])

        ranked_scores, ranked_labels, order = rank_by_score(scores, labels)

        self.assertEqual(order.tolist(), [1, 0, 2])
        self.assertEqual(ranked_scores.tolist(), [0.9, 0.2, 0.1])
        self.assertEqual(ranked_labels.tolist(), [1, 0, 0])

    def test_cluster_diversity_counts_unique_clusters_in_top_k(self):
        labels = np.array([1, 1, 1, 0])
        clusters = np.array([2, 2, 3, 4])

        self.assertEqual(cluster_diversity_at_k(labels, clusters, 3), 2 / 3)


if __name__ == "__main__":
    unittest.main()
