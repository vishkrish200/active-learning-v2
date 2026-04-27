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

    def test_challenge_mode_segments_old_trace_and_uses_window_shape_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support_dir = root / "support"
            candidate_dir = root / "candidates"
            support_dir.mkdir()
            candidate_dir.mkdir()

            old_trace = [
                *_wave(scale=1.0, bias=0.0, n_rows=90),
                *_wave(scale=1.2, bias=0.2, n_rows=90),
            ]
            _write_jsonl_imu(support_dir / "old_worker.jsonl", old_trace)
            _write_jsonl_imu(candidate_dir / "known.jsonl", _wave(scale=1.1, bias=0.1, n_rows=90))
            _write_jsonl_imu(candidate_dir / "novel.jsonl", _wave(scale=2.8, bias=1.4, n_rows=90))

            old_manifest = root / "old_workers.csv"
            candidate_manifest = root / "new_workers.csv"
            _write_manifest(old_manifest, [("old-worker", "support/old_worker.jsonl")])
            _write_manifest(
                candidate_manifest,
                [
                    ("known", "candidates/known.jsonl"),
                    ("novel", "candidates/novel.jsonl"),
                ],
            )
            output = root / "ranked.csv"

            result = run_external_selector(
                old_support_path=old_manifest,
                candidate_pool_path=candidate_manifest,
                output_path=output,
                representation="window_shape_stats",
                segment_old_support=True,
                support_clip_seconds=3.0,
                quality_threshold=0.0,
                max_stationary_fraction=1.0,
                max_abs_value=60.0,
                source_cap=None,
            )

            with output.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(result["selector"], "qgate_oldnovelty_window_shape_stats")
            self.assertEqual(result["n_support"], 1)
            self.assertEqual(result["n_support_segments"], 2)
            self.assertEqual([row["sample_id"] for row in rows], ["novel", "known"])

    def test_selector_accepts_one_path_per_line_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support_dir = root / "support"
            candidate_dir = root / "candidates"
            support_dir.mkdir()
            candidate_dir.mkdir()

            _write_jsonl_imu(support_dir / "old_worker.jsonl", _wave(scale=1.0, bias=0.0, n_rows=90))
            _write_jsonl_imu(candidate_dir / "known.jsonl", _wave(scale=1.0, bias=0.0, n_rows=90))
            _write_jsonl_imu(candidate_dir / "novel.jsonl", _wave(scale=2.5, bias=1.0, n_rows=90))

            old_manifest = root / "old_urls.txt"
            candidate_manifest = root / "new_urls.txt"
            old_manifest.write_text("support/old_worker.jsonl\n", encoding="utf-8")
            candidate_manifest.write_text("candidates/known.jsonl\ncandidates/novel.jsonl\n", encoding="utf-8")
            output = root / "ranked.csv"

            run_external_selector(
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
            self.assertEqual([row["sample_id"] for row in rows], ["novel", "known"])

    def test_selector_accepts_url_line_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support_dir = root / "support"
            candidate_dir = root / "candidates"
            support_dir.mkdir()
            candidate_dir.mkdir()

            old_path = support_dir / "old_worker.jsonl"
            known_path = candidate_dir / "known.jsonl"
            novel_path = candidate_dir / "novel.jsonl"
            _write_jsonl_imu(old_path, _wave(scale=1.0, bias=0.0, n_rows=90))
            _write_jsonl_imu(known_path, _wave(scale=1.0, bias=0.0, n_rows=90))
            _write_jsonl_imu(novel_path, _wave(scale=2.5, bias=1.0, n_rows=90))

            old_manifest = root / "old_urls.txt"
            candidate_manifest = root / "new_urls.txt"
            old_manifest.write_text(old_path.as_uri() + "\n", encoding="utf-8")
            candidate_manifest.write_text(f"{known_path.as_uri()}\n{novel_path.as_uri()}\n", encoding="utf-8")
            output = root / "ranked.csv"

            run_external_selector(
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
            self.assertEqual([row["sample_id"] for row in rows], ["novel", "known"])


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


def _wave(*, scale: float, bias: float, n_rows: int = 90) -> list[list[float]]:
    rows = []
    for index in range(n_rows):
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
