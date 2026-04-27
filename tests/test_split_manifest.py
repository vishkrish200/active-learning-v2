import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.data.split_manifest import (
    LeakageError,
    build_split_manifest,
    hash_manifest_url,
    split_counts,
)


class SplitManifestTests(unittest.TestCase):
    def test_hash_manifest_url_matches_cached_filename_convention(self):
        url = "https://storage.googleapis.com/buildai-imu-benchmark-v1-new/val/sample000001.txt"

        self.assertEqual(len(hash_manifest_url(url)), 64)
        self.assertEqual(hash_manifest_url(url), hash_manifest_url(url + "\n"))

    def test_build_manifest_maps_pretrain_and_val_without_overlap(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cache" / "features").mkdir(parents=True)
            (root / "cache" / "raw").mkdir(parents=True)
            (root / "cache" / "manifests").mkdir(parents=True)

            train_url = "https://example.com/pretrain/sample001.txt"
            val_url = "https://example.com/val/sample001.txt"
            (root / "cache" / "manifests" / "pretrain_urls.txt").write_text(train_url + "\n")
            (root / "cache" / "manifests" / "val_urls.txt").write_text(val_url + "\n")
            for url in (train_url, val_url):
                sample_id = hash_manifest_url(url)
                (root / "cache" / "features" / f"{sample_id}.npz").write_text("feature")
                (root / "cache" / "raw" / f"{sample_id}.jsonl").write_text("{}\n")

            manifest = build_split_manifest(
                root,
                pretrain_manifest="cache/manifests/pretrain_urls.txt",
                val_manifest="cache/manifests/val_urls.txt",
                feature_glob="cache/features/*.npz",
                raw_glob="cache/raw/*.jsonl",
            )

            self.assertEqual(split_counts(manifest), {"pretrain": 1, "val": 1})
            self.assertTrue(all(row.feature_path.exists() for row in manifest))
            self.assertTrue(all(row.raw_path.exists() for row in manifest))

    def test_overlap_between_train_and_val_raises_leakage_error(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cache" / "features").mkdir(parents=True)
            (root / "cache" / "raw").mkdir(parents=True)
            (root / "cache" / "manifests").mkdir(parents=True)

            url = "https://example.com/shared/sample001.txt"
            (root / "cache" / "manifests" / "pretrain_urls.txt").write_text(url + "\n")
            (root / "cache" / "manifests" / "val_urls.txt").write_text(url + "\n")

            with self.assertRaises(LeakageError):
                build_split_manifest(
                    root,
                    pretrain_manifest="cache/manifests/pretrain_urls.txt",
                    val_manifest="cache/manifests/val_urls.txt",
                    feature_glob="cache/features/*.npz",
                    raw_glob="cache/raw/*.jsonl",
                )

    def test_build_manifest_can_include_new_split_without_pretrain_overlap(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cache" / "features").mkdir(parents=True)
            (root / "cache" / "raw").mkdir(parents=True)
            (root / "cache" / "manifests").mkdir(parents=True)

            train_url = "https://example.com/pretrain/sample001.txt"
            val_url = "https://example.com/val/sample001.txt"
            new_url = "https://example.com/new/sample001.txt"
            (root / "cache" / "manifests" / "pretrain_urls.txt").write_text(train_url + "\n")
            (root / "cache" / "manifests" / "val_urls.txt").write_text(val_url + "\n")
            (root / "cache" / "manifests" / "new_urls.txt").write_text(new_url + "\n")
            for url in (train_url, val_url, new_url):
                sample_id = hash_manifest_url(url)
                (root / "cache" / "features" / f"{sample_id}.npz").write_text("feature")
                (root / "cache" / "raw" / f"{sample_id}.jsonl").write_text("{}\n")

            manifest = build_split_manifest(
                root,
                pretrain_manifest="cache/manifests/pretrain_urls.txt",
                val_manifest="cache/manifests/val_urls.txt",
                extra_manifests={"new": "cache/manifests/new_urls.txt"},
                feature_glob="cache/features/*.npz",
                raw_glob="cache/raw/*.jsonl",
            )

            self.assertEqual(split_counts(manifest), {"pretrain": 1, "val": 1, "new": 1})

    def test_new_split_overlapping_pretrain_raises_leakage_error(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cache" / "features").mkdir(parents=True)
            (root / "cache" / "raw").mkdir(parents=True)
            (root / "cache" / "manifests").mkdir(parents=True)

            train_url = "https://example.com/shared/sample001.txt"
            val_url = "https://example.com/val/sample001.txt"
            (root / "cache" / "manifests" / "pretrain_urls.txt").write_text(train_url + "\n")
            (root / "cache" / "manifests" / "val_urls.txt").write_text(val_url + "\n")
            (root / "cache" / "manifests" / "new_urls.txt").write_text(train_url + "\n")

            with self.assertRaises(LeakageError):
                build_split_manifest(
                    root,
                    pretrain_manifest="cache/manifests/pretrain_urls.txt",
                    val_manifest="cache/manifests/val_urls.txt",
                    extra_manifests={"new": "cache/manifests/new_urls.txt"},
                    feature_glob="cache/features/*.npz",
                    raw_glob="cache/raw/*.jsonl",
                )


if __name__ == "__main__":
    unittest.main()
