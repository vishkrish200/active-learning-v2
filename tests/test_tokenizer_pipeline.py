import json
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marginal_value.data.split_manifest import SplitSample
from marginal_value.tokenization.artifacts import read_token_sequences_jsonl
from marginal_value.tokenization.modal_tokenizer import run_tokenizer_pipeline


class TokenizerPipelineTests(unittest.TestCase):
    def test_tokenizer_pipeline_fits_only_pretrain_and_transforms_val(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = _make_rows(root)
            config = _pipeline_config()
            output_dir = root / "artifacts" / "tokens"

            result = run_tokenizer_pipeline(config, rows, output_dir=output_dir, smoke=False)

            self.assertEqual(result["mode"], "full")
            self.assertEqual(result["fit_split"], "pretrain")
            self.assertEqual(result["transform_splits"], ["pretrain", "val"])
            self.assertEqual(result["fit_patch_split_counts"], {"pretrain": 6})
            self.assertEqual(result["bpe_fit_sequence_split_counts"], {"pretrain": 2})
            self.assertEqual(result["sequence_split_counts"], {"pretrain": 2, "val": 1})
            self.assertTrue(Path(result["codebook_path"]).exists())
            self.assertTrue(Path(result["bpe_merges_path"]).exists())
            self.assertTrue(Path(result["sequence_path"]).exists())
            self.assertTrue(Path(result["report_path"]).exists())

            sequences = read_token_sequences_jsonl(result["sequence_path"])
            self.assertEqual([sequence.split for sequence in sequences], ["pretrain", "pretrain", "val"])
            self.assertTrue(all(sequence.base_token_ids for sequence in sequences))
            self.assertTrue(all(sequence.primitive_token_ids for sequence in sequences))

            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(report["leakage_audit"]["vq_fit_splits"], ["pretrain"])
            self.assertEqual(report["leakage_audit"]["bpe_fit_splits"], ["pretrain"])
            self.assertEqual(report["diagnostics"]["dead_code_fraction"], 0.0)

    def test_tokenizer_pipeline_smoke_bounds_sample_counts(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = _make_rows(root)
            config = _pipeline_config()
            config["patches"]["smoke_fit_samples"] = 1
            config["patches"]["smoke_transform_samples_per_split"] = 1

            result = run_tokenizer_pipeline(config, rows, output_dir=root / "tokens", smoke=True)

            self.assertEqual(result["mode"], "smoke")
            self.assertEqual(result["fit_patch_split_counts"], {"pretrain": 3})
            self.assertEqual(result["sequence_split_counts"], {"pretrain": 1, "val": 1})

    def test_tokenizer_pipeline_rejects_manifest_without_pretrain_fit_rows(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [row for row in _make_rows(root) if row.split == "val"]

            with self.assertRaises(ValueError):
                run_tokenizer_pipeline(_pipeline_config(), rows, output_dir=root / "tokens", smoke=False)

    def test_tokenizer_pipeline_emits_full_progress_logs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = _make_rows(root)
            config = _pipeline_config()
            buffer = io.StringIO()

            with redirect_stdout(buffer):
                run_tokenizer_pipeline(config, rows, output_dir=root / "tokens", smoke=False)

            events = [json.loads(line)["event"] for line in buffer.getvalue().splitlines()]
            self.assertIn("fit_rows_selected", events)
            self.assertIn("fit_patch_extract_progress", events)
            self.assertIn("vq_fit_iteration_progress", events)
            self.assertIn("transform_split_start", events)
            self.assertIn("transform_split_progress", events)
            self.assertIn("transform_split_done", events)


def _make_rows(root: Path) -> list[SplitSample]:
    raw_path = root / "dummy.jsonl"
    raw_path.write_text("", encoding="utf-8")
    specs = [
        ("pre-a", "pretrain", 0.0),
        ("pre-b", "pretrain", 10.0),
        ("val-a", "val", 100.0),
    ]
    rows = []
    for sample_id, split, offset in specs:
        feature_path = root / f"{sample_id}.npz"
        windows = (np.arange(8, dtype=np.float32).reshape(4, 2) + offset).astype(np.float32)
        np.savez(feature_path, window_features=windows)
        rows.append(SplitSample(sample_id, split, f"url-{sample_id}", raw_path, feature_path))
    return rows


def _pipeline_config() -> dict:
    return {
        "execution": {
            "provider": "modal",
            "gpu": "H100",
            "data_volume": "imu-novelty-subset-data",
            "artifacts_volume": "activelearning-imu-rebuild-cache",
            "timeout_seconds": 7200,
        },
        "data": {
            "root": "/data",
            "format": "npz_features",
            "feature_glob": "cache/features/*.npz",
            "raw_glob": "cache/raw/*.jsonl",
            "pretrain_manifest": "cache/manifests/pretrain_urls.txt",
            "val_manifest": "cache/manifests/val_urls.txt",
            "feature_dim": 75,
        },
        "artifacts": {
            "root": "/artifacts",
            "output_dir": "/artifacts/tokens/window_mean_std_tokenizer",
        },
        "splits": {
            "fit_split": "pretrain",
            "transform_splits": ["pretrain", "val"],
        },
        "patches": {
            "feature_source": "npz_features",
            "patch_len_sec": 0.5,
            "patch_stride_sec": 0.25,
            "sample_rate": 30.0,
            "smoke_fit_samples": 128,
            "smoke_transform_samples_per_split": 64,
            "full_fit_samples": 5000,
        },
        "vq": {
            "codebook_size": 4,
            "n_iter": 3,
            "seed": 7,
        },
        "bpe": {
            "num_merges": 8,
            "min_count": 2,
            "base_step_sec": 0.25,
            "max_primitive_duration_sec": 2.0,
        },
        "diagnostics": {
            "top_k_primitives": 5,
        },
    }


if __name__ == "__main__":
    unittest.main()
