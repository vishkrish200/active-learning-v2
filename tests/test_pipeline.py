import unittest

import numpy as np
import pandas as pd

from marginal_value.indexing.knn_features import ExactKNNIndex, build_old_support_features
from marginal_value.models.grammar_lm import NGramMotionGrammar
from marginal_value.models.ranker import assign_reason_code, score_candidates
from marginal_value.models.tokenizer import MotionBPE
from marginal_value.preprocessing.quality import compute_quality_features, score_quality
from marginal_value.submit.make_submission import build_submission_rows, diversity_rerank


def make_clean_imu(n_samples: int = 300, sample_rate: int = 30) -> np.ndarray:
    t = np.arange(n_samples, dtype=float) / sample_rate
    return np.column_stack(
        [
            np.sin(2 * np.pi * 0.7 * t),
            0.5 * np.cos(2 * np.pi * 0.7 * t),
            0.1 * np.sin(2 * np.pi * 1.3 * t) + 9.81,
            0.03 * np.cos(2 * np.pi * 0.2 * t),
            0.04 * np.sin(2 * np.pi * 0.4 * t),
            0.02 * np.cos(2 * np.pi * 0.3 * t),
        ]
    )


class QualityGateTests(unittest.TestCase):
    def test_flatline_and_spikes_score_lower_than_clean_motion(self):
        clean = make_clean_imu()
        flatline = clean.copy()
        flatline[:, 0] = flatline[0, 0]
        spiky = clean.copy()
        spiky[::20, 3] = 500.0

        clean_score = score_quality(compute_quality_features(clean, sample_rate=30))
        flatline_score = score_quality(compute_quality_features(flatline, sample_rate=30))
        spiky_score = score_quality(compute_quality_features(spiky, sample_rate=30))

        self.assertGreater(clean_score, 0.85)
        self.assertLess(flatline_score, clean_score - 0.15)
        self.assertLess(spiky_score, clean_score - 0.15)


class SupportAndRankingTests(unittest.TestCase):
    def test_supported_high_quality_novelty_beats_low_quality_singleton(self):
        old = np.array(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [0.9, -0.1],
                [0.0, 1.0],
                [0.05, 0.95],
            ]
        )
        new = np.array(
            [
                [0.0, -1.0],
                [0.02, -0.98],
                [-0.03, -0.97],
                [-1.0, 0.0],
            ]
        )
        worker_ids = ["cohesive_a", "cohesive_b", "cohesive_c", "bad_singleton"]
        support = build_old_support_features(ExactKNNIndex().fit(old), new, ks=(1, 3))
        feature_rows = []
        for worker_id, row in zip(worker_ids, support):
            row = dict(row)
            row["worker_id"] = worker_id
            row["quality_score"] = 0.96 if worker_id != "bad_singleton" else 0.22
            row["new_batch_density"] = 0.91 if worker_id.startswith("cohesive") else 0.05
            row["new_cluster_size"] = 3 if worker_id.startswith("cohesive") else 1
            row["token_nll_p95"] = 2.1
            row["rare_phrase_fraction"] = 0.25
            feature_rows.append(row)

        scored = score_candidates(feature_rows)
        by_id = {row["worker_id"]: row for row in scored}

        self.assertGreater(by_id["cohesive_a"]["final_score"], by_id["bad_singleton"]["final_score"])
        self.assertEqual(assign_reason_code(by_id["bad_singleton"]), "LOW_QUALITY")
        self.assertEqual(assign_reason_code(by_id["cohesive_a"]), "COHESIVE_NEW_WORKFLOW")

    def test_diversity_rerank_moves_near_duplicate_below_other_supported_mode(self):
        rows = [
            {"worker_id": "a1", "final_score": 0.95},
            {"worker_id": "a2", "final_score": 0.94},
            {"worker_id": "b1", "final_score": 0.90},
        ]
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
            ]
        )

        reranked = diversity_rerank(rows, embeddings, lambda_redundancy=0.20)

        self.assertEqual(reranked[0]["worker_id"], "a1")
        self.assertEqual(reranked[1]["worker_id"], "b1")
        self.assertEqual(reranked[2]["worker_id"], "a2")


class TokenGrammarTests(unittest.TestCase):
    def test_motion_bpe_discovers_variable_duration_repeated_phrase(self):
        sequences = [
            [17, 17, 42, 91, 8, 8, 8, 8],
            [17, 17, 42, 91, 8, 8, 8, 8],
            [17, 17, 42, 91, 31, 12],
        ]

        tokenizer = MotionBPE(num_merges=3, base_step_sec=0.25, max_primitive_duration_sec=2.0)
        tokenizer.fit(sequences)
        encoded = tokenizer.encode([17, 17, 42, 91, 8, 8])

        self.assertEqual(encoded[0].codes, (17, 17, 42, 91))
        self.assertAlmostEqual(encoded[0].duration_sec, 1.0)
        self.assertTrue(any(len(primitive.codes) > 1 for primitive in encoded))

    def test_grammar_surprisal_flags_rare_composition_not_known_repetition(self):
        known = [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 5],
            [7, 7, 7, 7],
        ]
        grammar = NGramMotionGrammar(order=3, smoothing=0.1).fit(known)

        familiar = grammar.sequence_features([1, 2, 3, 4])
        rare = grammar.sequence_features([1, 2, 8, 9])

        self.assertGreater(rare["token_nll_mean"], familiar["token_nll_mean"])
        self.assertGreater(rare["rare_phrase_fraction"], familiar["rare_phrase_fraction"])


class SubmissionTests(unittest.TestCase):
    def test_submission_rows_are_ranked_and_include_diagnostics(self):
        scored = [
            {
                "worker_id": "w2",
                "final_score": 0.5,
                "quality_score": 0.9,
                "reason_code": "RARE_TEMPORAL_COMPOSITION",
            },
            {
                "worker_id": "w1",
                "final_score": 0.9,
                "quality_score": 0.95,
                "reason_code": "COHESIVE_NEW_WORKFLOW",
            },
        ]

        rows = build_submission_rows(scored)
        frame = pd.DataFrame(rows)

        self.assertEqual(list(frame.columns), ["worker_id", "rank", "score", "quality_score", "reason_code"])
        self.assertEqual(frame.iloc[0]["worker_id"], "w1")
        self.assertEqual(frame.iloc[0]["rank"], 1)


if __name__ == "__main__":
    unittest.main()
