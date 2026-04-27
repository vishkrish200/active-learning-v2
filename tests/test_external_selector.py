import csv
import json
import tempfile
import unittest
from pathlib import Path

from marginal_value.select import run_external_selector


class ExternalSelectorTests(unittest.TestCase):
    def test_selector_ranks_candidates_without_target_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support_dir = root / "support"
            candidate_dir = root / "candidates"
            support_dir.mkdir()
            candidate_dir.mkdir()

            _write_jsonl_imu(support_dir / "old_a.jsonl", _wave(scale=1.0, bias=0.0))
            _write_jsonl_imu(support_dir / "old_b.jsonl", _wave(scale=1.1, bias=0.1))
            _write_jsonl_imu(candidate_dir / "known.jsonl", _wave(scale=1.05, bias=0.05))
            _write_jsonl_imu(candidate_dir / "novel.jsonl", _wave(scale=3.0, bias=1.5))
            _write_jsonl_imu(candidate_dir / "bad_abs.jsonl", _wave(scale=2.0, bias=80.0))

            old_manifest = root / "old_support.csv"
            candidate_manifest = root / "candidate_pool.csv"
            _write_manifest(
                old_manifest,
                [
                    ("old-a", "support/old_a.jsonl"),
                    ("old-b", "support/old_b.jsonl"),
                ],
            )
            _write_manifest(
                candidate_manifest,
                [
                    ("known", "candidates/known.jsonl"),
                    ("novel", "candidates/novel.jsonl"),
                    ("bad-abs", "candidates/bad_abs.jsonl"),
                ],
            )
            output = root / "ranked.csv"

            result = run_external_selector(
                old_support_path=old_manifest,
                candidate_pool_path=candidate_manifest,
                output_path=output,
                quality_threshold=0.0,
                max_stationary_fraction=1.0,
                max_abs_value=60.0,
                source_cap=None,
            )

            with output.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(result["n_support"], 2)
            self.assertEqual(result["n_candidates"], 3)
            self.assertEqual([row["sample_id"] for row in rows], ["novel", "known", "bad-abs"])
            self.assertEqual(rows[0]["rank"], "1")
            self.assertEqual(rows[0]["quality_gate_pass"], "True")
            self.assertEqual(rows[-1]["physical_validity_pass"], "False")
            self.assertEqual(rows[-1]["physical_validity_failure_reasons"], "max_abs_value")


def _write_manifest(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "raw_path"])
        writer.writeheader()
        for sample_id, raw_path in rows:
            writer.writerow({"sample_id": sample_id, "raw_path": raw_path})


def _write_jsonl_imu(path: Path, samples: list[list[float]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples):
            handle.write(
                json.dumps(
                    {
                        "timestamp": index / 30.0,
                        "acc": sample[:3],
                        "gyro": sample[3:],
                    }
                )
                + "\n"
            )


def _wave(*, scale: float, bias: float) -> list[list[float]]:
    rows = []
    for index in range(90):
        phase = index / 9.0
        rows.append(
            [
                bias + scale * phase,
                scale * ((index % 5) - 2),
                scale * ((index % 7) - 3),
                0.1 * scale * ((index % 3) - 1),
                0.2 * scale * ((index % 4) - 2),
                0.3 * scale * ((index % 6) - 3),
            ]
        )
    return rows


if __name__ == "__main__":
    unittest.main()
