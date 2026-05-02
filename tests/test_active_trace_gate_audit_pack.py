import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.active.trace_gate_audit_pack import (
    run_trace_gate_audit_pack,
    select_trace_gate_audit_candidates,
    validate_trace_gate_audit_pack_config,
)


class ActiveTraceGateAuditPackTests(unittest.TestCase):
    def test_select_trace_gate_audit_candidates_targets_top_added_and_removed(self):
        current = [_row("a", 1), _row("b", 2), _row("c", 3), _row("d", 4), _row("e", 5)]
        trace = [_row("a", 1), _row("c", 2), _row("d", 3), _row("f", 4), _row("g", 5)]

        selected = select_trace_gate_audit_candidates(current, trace, top_n=2, compare_k=5)

        by_id = {row["sample_id"]: row for row in selected}
        self.assertIn("a", by_id)
        self.assertIn("c", by_id)
        self.assertIn("b", by_id)
        self.assertIn("e", by_id)
        self.assertIn("f", by_id)
        self.assertIn("g", by_id)
        self.assertIn("trace_gate:top", by_id["a"]["audit_groups"])
        self.assertIn("current:removed_top50", by_id["b"]["audit_groups"])
        self.assertIn("trace_gate:added_top50", by_id["f"]["audit_groups"])

    def test_runner_writes_targeted_report_without_plotting(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_a = root / "a.jsonl"
            raw_b = root / "b.jsonl"
            raw_c = root / "c.jsonl"
            _write_raw(raw_a)
            _write_raw(raw_b)
            _write_raw(raw_c)
            current_path = root / "current.csv"
            trace_path = root / "trace.csv"
            _write_csv(current_path, [_row("a", 1, raw_path=str(raw_a)), _row("b", 2, raw_path=str(raw_b))])
            _write_csv(trace_path, [_row("a", 1, raw_path=str(raw_a)), _row("c", 2, raw_path=str(raw_c))])
            output_dir = root / "audit"
            config = {
                "execution": {"provider": "modal", "allow_local_paths_for_tests": True},
                "data": {"root": str(root)},
                "quality": {"sample_rate": 30.0, "max_samples_per_clip": 90},
                "audit": {"top_n": 1, "compare_k": 2, "generate_plots": False},
                "artifacts": {
                    "current_diagnostics_path": str(current_path),
                    "trace_gate_diagnostics_path": str(trace_path),
                    "output_dir": str(output_dir),
                },
            }

            result = run_trace_gate_audit_pack(config, smoke=False)

            self.assertEqual(result["n_selected"], 3)
            self.assertEqual(result["n_plots_written"], 0)
            self.assertTrue((output_dir / "trace_gate_audit_report_full.json").exists())
            self.assertTrue((output_dir / "trace_gate_audit_plot_index_full.csv").exists())

    def test_validate_rejects_non_artifact_paths(self):
        with self.assertRaises(ValueError):
            validate_trace_gate_audit_pack_config(
                {
                    "execution": {"provider": "modal"},
                    "data": {"root": "/data"},
                    "artifacts": {
                        "current_diagnostics_path": "/tmp/current.csv",
                        "trace_gate_diagnostics_path": "/artifacts/trace.csv",
                        "output_dir": "/artifacts/out",
                    },
                }
            )


def _row(sample_id: str, rank: int, *, raw_path: str = "") -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "worker_id": sample_id,
        "rank": rank,
        "original_rank": rank,
        "raw_path": raw_path,
        "final_score": 1.0 / rank,
        "blend_old_novelty_score": 1.0 / rank,
        "quality_score": 0.99,
        "stationary_fraction": 0.1,
        "max_abs_value": 12.0,
        "new_cluster_id": f"cluster-{sample_id}",
        "new_cluster_size": 1,
        "physical_validity_pass": True,
        "physical_validity_failure_reasons": "",
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_raw(path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for idx in range(90):
            handle.write(
                json.dumps(
                    {
                        "t_us": int(idx * 1_000_000 / 30),
                        "acc": [0.1 * (idx % 10), 0.2, 9.81],
                        "gyro": [0.01 * (idx % 5), 0.02, 0.03],
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    unittest.main()
