import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.active.topk_audit_pack import (
    run_topk_audit_pack,
    select_audit_candidates,
    trace_verdict,
    validate_topk_audit_pack_config,
)


class ActiveTopKAuditPackTests(unittest.TestCase):
    def test_select_audit_candidates_combines_top_disagreements_and_rejected(self):
        exact = [
            _row("a", 1, blend=0.9),
            _row("b", 2, blend=0.8),
            _row("c", 3, blend=0.7),
            _row("d", 4, blend=0.99, quality=0.4),
        ]
        consensus = [
            _row("c", 1, blend=0.7),
            _row("a", 2, blend=0.9),
            _row("b", 3, blend=0.8),
            _row("d", 4, blend=0.99, quality=0.4),
        ]

        selected = select_audit_candidates(
            exact,
            consensus,
            top_n=2,
            disagreement_n=1,
            rejected_n=1,
            quality_threshold=0.85,
            max_stationary_fraction=0.90,
            max_abs_value=60.0,
        )

        by_id = {row["sample_id"]: row for row in selected}
        self.assertIn("a", by_id)
        self.assertIn("c", by_id)
        self.assertIn("d", by_id)
        self.assertIn("exact:top", by_id["a"]["audit_groups"])
        self.assertIn("consensus:top", by_id["c"]["audit_groups"])
        self.assertIn("exact:high_novelty_rejected", by_id["d"]["audit_groups"])

    def test_topk_audit_runner_writes_report_without_plotting(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_a = root / "a.jsonl"
            raw_b = root / "b.jsonl"
            _write_raw(raw_a, stationary=False)
            _write_raw(raw_b, stationary=True)
            exact_path = root / "exact.csv"
            consensus_path = root / "consensus.csv"
            _write_csv(
                exact_path,
                [
                    _row("a", 1, raw_path=str(raw_a), quality=0.99, stationary=0.1),
                    _row("b", 2, raw_path=str(raw_b), quality=0.99, stationary=0.95),
                ],
            )
            _write_csv(
                consensus_path,
                [
                    _row("b", 1, raw_path=str(raw_b), quality=0.99, stationary=0.95),
                    _row("a", 2, raw_path=str(raw_a), quality=0.99, stationary=0.1),
                ],
            )
            output_dir = root / "audit"
            config = {
                "execution": {"provider": "modal", "allow_local_paths_for_tests": True},
                "artifacts": {
                    "exact_diagnostics_path": str(exact_path),
                    "consensus_diagnostics_path": str(consensus_path),
                    "output_dir": str(output_dir),
                },
                "quality": {"sample_rate": 30.0, "max_samples_per_clip": 90},
                "audit": {
                    "top_n": 1,
                    "disagreement_n": 1,
                    "rejected_n": 1,
                    "top_ks": [1, 2],
                    "generate_plots": False,
                },
                "ranking": {
                    "quality_threshold": 0.85,
                    "max_stationary_fraction": 0.90,
                    "max_abs_value": 60.0,
                },
            }

            result = run_topk_audit_pack(config, smoke=False)

            self.assertEqual(result["n_plots_written"], 0)
            report = json.loads((output_dir / "topk_audit_report_full.json").read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["topk"]["1"]["exact_consensus_overlap"], 0.0)
            self.assertTrue((output_dir / "topk_audit_plot_index_full.csv").exists())
            self.assertTrue((output_dir / "topk_audit_report_full.md").exists())

    def test_trace_verdict_prioritizes_artifacts(self):
        self.assertEqual(trace_verdict(["spiky_or_extreme", "mostly_stationary"]), "likely_artifact")
        self.assertEqual(trace_verdict(["mostly_stationary"]), "mostly_stationary")
        self.assertEqual(trace_verdict(["clean_motion"]), "plausible_motion")

    def test_validate_rejects_non_artifact_paths(self):
        with self.assertRaises(ValueError):
            validate_topk_audit_pack_config(
                {
                    "execution": {"provider": "modal"},
                    "data": {"root": "/data"},
                    "artifacts": {
                        "exact_diagnostics_path": "/tmp/exact.csv",
                        "consensus_diagnostics_path": "/artifacts/consensus.csv",
                        "output_dir": "/artifacts/out",
                    },
                }
            )


def _row(
    sample_id: str,
    rank: int,
    *,
    blend: float = 0.5,
    quality: float = 0.99,
    stationary: float = 0.1,
    max_abs: float = 12.0,
    raw_path: str = "",
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "worker_id": sample_id,
        "rank": rank,
        "raw_path": raw_path,
        "final_score": blend * quality,
        "blend_old_novelty_score": blend,
        "quality_score": quality,
        "stationary_fraction": stationary,
        "max_abs_value": max_abs,
        "quality_gate_pass": quality >= 0.85,
        "physical_validity_pass": stationary <= 0.90 and max_abs <= 60.0,
        "physical_validity_failure_reasons": "" if stationary <= 0.90 and max_abs <= 60.0 else "stationary",
        "new_cluster_id": f"cluster-{sample_id}",
        "new_cluster_size": 1,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_raw(path: Path, *, stationary: bool) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for idx in range(90):
            if stationary:
                acc = [0.0, 0.0, 9.81]
                gyro = [0.0, 0.0, 0.0]
            else:
                acc = [0.1 * idx, 0.2, 9.81 + 0.1 * ((idx % 10) - 5)]
                gyro = [0.01 * (idx % 5), 0.02, 0.03]
            handle.write(json.dumps({"t_us": int(idx * 1_000_000 / 30), "acc": acc, "gyro": gyro}) + "\n")


if __name__ == "__main__":
    unittest.main()
