import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.data.split_manifest import hash_manifest_url
from marginal_value.tokenization.artifacts import TokenSequence, read_token_sequences_jsonl, write_token_sequences_jsonl
from marginal_value.tokenization.transform_existing import run_existing_tokenizer_transform


class TokenizerTransformTests(unittest.TestCase):
    def test_existing_tokenizer_transform_scores_new_without_fit_leakage(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            artifacts = root / "artifacts"
            token_in = artifacts / "tokens" / "base"
            token_out = artifacts / "tokens" / "new"
            (data / "cache" / "manifests").mkdir(parents=True)
            (data / "cache" / "features").mkdir(parents=True)
            (data / "cache" / "raw").mkdir(parents=True)
            token_in.mkdir(parents=True)

            pretrain_url = "https://example.com/pretrain/a.txt"
            new_url = "https://example.com/new/b.txt"
            new_id = hash_manifest_url(new_url)
            (data / "cache" / "manifests" / "pretrain_urls.txt").write_text(pretrain_url + "\n", encoding="utf-8")
            (data / "cache" / "manifests" / "val_empty.txt").write_text("", encoding="utf-8")
            (data / "cache" / "manifests" / "new_urls.txt").write_text(new_url + "\n", encoding="utf-8")
            np.savez(data / "cache" / "features" / f"{new_id}.npz", window_features=np.ones((3, 2), dtype=np.float32))
            (data / "cache" / "raw" / f"{new_id}.jsonl").write_text("{}\n", encoding="utf-8")
            pretrain_id = hash_manifest_url(pretrain_url)
            np.savez(data / "cache" / "features" / f"{pretrain_id}.npz", window_features=np.ones((3, 2), dtype=np.float32))
            (data / "cache" / "raw" / f"{pretrain_id}.jsonl").write_text("{}\n", encoding="utf-8")

            np.savez(token_in / "vq_codebook_full.npz", codebook=np.asarray([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.float32))
            (token_in / "bpe_merges_full.json").write_text("[]", encoding="utf-8")
            write_token_sequences_jsonl(
                token_in / "token_sequences_full.jsonl",
                [
                    TokenSequence(
                        sample_id=pretrain_id,
                        split="pretrain",
                        base_token_ids=[0, 1],
                        primitive_token_ids=["0", "1"],
                        primitive_durations_sec=[0.25, 0.25],
                    )
                ],
            )

            result = run_existing_tokenizer_transform(
                {
                    "data": {
                        "root": str(data),
                        "feature_glob": "cache/features/*.npz",
                        "raw_glob": "cache/raw/*.jsonl",
                        "pretrain_manifest": "cache/manifests/pretrain_urls.txt",
                        "val_manifest": "cache/manifests/val_empty.txt",
                        "new_manifest": "cache/manifests/new_urls.txt",
                    },
                    "tokens": {
                        "input_dir": str(token_in),
                        "output_dir": str(token_out),
                        "source_mode": "full",
                    },
                    "splits": {"fit_split": "pretrain", "transform_split": "new"},
                    "patches": {"patch_len_sec": 0.5, "patch_stride_sec": 0.25},
                    "bpe": {"base_step_sec": 0.25, "max_primitive_duration_sec": 12.0},
                },
                smoke=False,
            )

            sequences = read_token_sequences_jsonl(result["sequence_path"])
            self.assertEqual(result["n_fit_sequences"], 1)
            self.assertEqual(result["n_transformed_sequences"], 1)
            self.assertEqual([sequence.split for sequence in sequences], ["pretrain", "new"])


if __name__ == "__main__":
    unittest.main()
