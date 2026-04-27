import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.data.split_manifest import SplitSample
from marginal_value.tokenization.artifacts import (
    TokenSequence,
    read_token_sequences_jsonl,
    write_token_sequences_jsonl,
)
from marginal_value.tokenization.patches import (
    extract_patch_vectors,
    load_window_features,
    patch_vectors_for_rows,
)


class TokenizationPatchTests(unittest.TestCase):
    def test_extract_patch_vectors_flattens_rolling_feature_windows(self):
        windows = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ]
        )

        patches = extract_patch_vectors(windows, patch_size_windows=2, stride_windows=1)

        self.assertEqual(patches.shape, (3, 4))
        np.testing.assert_allclose(patches[0], [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(patches[-1], [5.0, 6.0, 7.0, 8.0])

    def test_extract_patch_vectors_pads_short_sequences(self):
        windows = np.array([[1.0, 2.0], [3.0, 4.0]])

        patches = extract_patch_vectors(windows, patch_size_windows=4, stride_windows=2)

        self.assertEqual(patches.shape, (1, 8))
        np.testing.assert_allclose(patches[0], [1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0])

    def test_extract_patch_vectors_rejects_invalid_shape(self):
        with self.assertRaises(ValueError):
            extract_patch_vectors(np.ones((2, 3, 4)), patch_size_windows=2, stride_windows=1)

    def test_load_window_features_reads_modal_npz_key(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.npz"
            np.savez(path, window_features=np.arange(6, dtype=np.float32).reshape(3, 2))

            features = load_window_features(path)

            self.assertEqual(features.dtype, np.float32)
            self.assertEqual(features.shape, (3, 2))

    def test_patch_vectors_for_rows_preserves_split_provenance(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pretrain_path = root / "pretrain.npz"
            val_path = root / "val.npz"
            raw_path = root / "dummy.jsonl"
            raw_path.write_text("", encoding="utf-8")
            np.savez(pretrain_path, window_features=np.arange(8, dtype=np.float32).reshape(4, 2))
            np.savez(val_path, window_features=np.ones((4, 2), dtype=np.float32))
            rows = [
                SplitSample("pre", "pretrain", "url-pre", raw_path, pretrain_path),
                SplitSample("val", "val", "url-val", raw_path, val_path),
            ]

            patches, metadata = patch_vectors_for_rows(rows, patch_size_windows=2, stride_windows=2)

            self.assertEqual(patches.shape, (4, 4))
            self.assertEqual([item["sample_id"] for item in metadata], ["pre", "pre", "val", "val"])
            self.assertEqual([item["split"] for item in metadata], ["pretrain", "pretrain", "val", "val"])


class TokenizationArtifactTests(unittest.TestCase):
    def test_token_sequence_jsonl_round_trips(self):
        sequences = [
            TokenSequence(
                sample_id="worker-a",
                split="pretrain",
                base_token_ids=[1, 2, 3],
                primitive_token_ids=["1_2", "3"],
                primitive_durations_sec=[0.5, 0.25],
                quality_score=0.97,
                metadata={"feature_path": "cache/features/a.npz"},
            ),
            TokenSequence(
                sample_id="worker-b",
                split="val",
                base_token_ids=[4],
                primitive_token_ids=["4"],
                primitive_durations_sec=[0.25],
                quality_score=None,
                metadata={},
            ),
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokens" / "sequences.jsonl"

            write_token_sequences_jsonl(path, sequences)
            loaded = read_token_sequences_jsonl(path)

            self.assertEqual(loaded, sequences)

    def test_token_sequence_rejects_mismatched_primitive_lengths(self):
        with self.assertRaises(ValueError):
            TokenSequence(
                sample_id="bad",
                split="val",
                base_token_ids=[1],
                primitive_token_ids=["1", "2"],
                primitive_durations_sec=[0.25],
            )


if __name__ == "__main__":
    unittest.main()
