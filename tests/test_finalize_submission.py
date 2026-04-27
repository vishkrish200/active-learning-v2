import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.data.split_manifest import hash_manifest_url
from marginal_value.submit.finalize_submission import finalize_submission_ids, manifest_hash_to_url_stem


class FinalizeSubmissionTests(unittest.TestCase):
    def test_manifest_hash_to_url_stem_maps_hashes_to_manifest_ids(self):
        url = "https://storage.googleapis.com/buildai-imu-benchmark-v1-new/val/sample000042.txt"

        mapping = manifest_hash_to_url_stem([url])

        self.assertEqual(mapping[hash_manifest_url(url)], "sample000042")

    def test_finalize_submission_ids_rewrites_worker_id(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            url = "https://storage.googleapis.com/buildai-imu-benchmark-v1-new/val/sample000001.txt"
            submission_path = root / "internal.csv"
            manifest_path = root / "new_urls.txt"
            output_path = root / "final.csv"
            manifest_path.write_text(url + "\n", encoding="utf-8")
            _write_rows(
                submission_path,
                ["worker_id", "rank", "score", "quality_score", "reason_code"],
                [
                    {
                        "worker_id": hash_manifest_url(url),
                        "rank": "1",
                        "score": "0.9",
                        "quality_score": "1.0",
                        "reason_code": "RARE_TEMPORAL_COMPOSITION",
                    }
                ],
            )

            result = finalize_submission_ids(
                submission_path=submission_path,
                manifest_path=manifest_path,
                output_path=output_path,
            )

            with output_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(result["output_rows"], 1)
            self.assertEqual(rows[0]["worker_id"], "sample000001")
            self.assertEqual(rows[0]["rank"], "1")

    def test_finalize_submission_ids_can_rename_id_column(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            url = "https://storage.googleapis.com/buildai-imu-benchmark-v1-new/val/sample000001.txt"
            submission_path = root / "internal.csv"
            manifest_path = root / "new_urls.txt"
            output_path = root / "final.csv"
            manifest_path.write_text(url + "\n", encoding="utf-8")
            _write_rows(
                submission_path,
                ["worker_id", "rank", "score", "quality_score", "reason_code"],
                [
                    {
                        "worker_id": hash_manifest_url(url),
                        "rank": "1",
                        "score": "0.9",
                        "quality_score": "1.0",
                        "reason_code": "RARE_TEMPORAL_COMPOSITION",
                    }
                ],
            )

            finalize_submission_ids(
                submission_path=submission_path,
                manifest_path=manifest_path,
                output_path=output_path,
                output_id_column="new_worker_id",
            )

            with output_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(list(rows[0].keys()), ["new_worker_id", "rank", "score", "quality_score", "reason_code"])
            self.assertEqual(rows[0]["new_worker_id"], "sample000001")

    def test_finalize_submission_ids_raises_for_unmapped_hash(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            submission_path = root / "internal.csv"
            manifest_path = root / "new_urls.txt"
            output_path = root / "final.csv"
            manifest_path.write_text(
                "https://storage.googleapis.com/buildai-imu-benchmark-v1-new/val/sample000001.txt\n",
                encoding="utf-8",
            )
            _write_rows(
                submission_path,
                ["worker_id", "rank", "score"],
                [{"worker_id": "not-in-manifest", "rank": "1", "score": "0.9"}],
            )

            with self.assertRaises(ValueError):
                finalize_submission_ids(
                    submission_path=submission_path,
                    manifest_path=manifest_path,
                    output_path=output_path,
                )


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
