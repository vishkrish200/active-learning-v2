import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.active.spike_hygiene_ablation import (
    build_spike_ablation_rows,
    run_spike_hygiene_ablation,
    validate_spike_hygiene_ablation_config,
)


class ActiveSpikeHygieneAblationTests(unittest.TestCase):
    def test_build_spike_ablation_rows_demotes_hard_gate_failures(self):
        rows = [
            _row("a", 1, spike=0.03, cluster="c1"),
            _row("b", 2, spike=0.01, cluster="c2"),
            _row("c", 3, spike=0.02, cluster="c3"),
        ]

        hard = build_spike_ablation_rows(rows, mode="hard_gate", spike_rate_threshold=0.025)
        soft = build_spike_ablation_rows(rows, mode="soft_penalty", spike_rate_threshold=0.025)

        self.assertEqual([row["sample_id"] for row in hard], ["b", "c", "a"])
        self.assertEqual(hard[-1]["spike_hygiene_pass"], False)
        self.assertEqual([row["sample_id"] for row in soft][:2], ["b", "c"])
        self.assertLess(float(soft[-1]["spike_penalty_multiplier"]), 1.0)

    def test_trace_gate_demotes_likely_artifacts_below_clean_rows(self):
        rows = [
            _row("a", 1, spike=0.01, cluster="c1", trace_verdict="likely_artifact"),
            _row("b", 2, spike=0.01, cluster="c2", trace_verdict="plausible_motion"),
            _row("c", 3, spike=0.04, cluster="c3", trace_verdict="plausible_motion"),
            _row("d", 4, spike=0.01, cluster="c4", trace_verdict="mostly_stationary"),
        ]

        trace_gate = build_spike_ablation_rows(rows, mode="trace_gate", spike_rate_threshold=0.025)

        self.assertEqual([row["sample_id"] for row in trace_gate], ["b", "a", "c", "d"])
        self.assertEqual(trace_gate[0]["trace_hygiene_pass"], True)
        self.assertEqual(trace_gate[1]["trace_hygiene_pass"], False)

    def test_spike_hygiene_ablation_runner_writes_reports(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            diagnostics_path = root / "exact.csv"
            _write_csv(
                diagnostics_path,
                [
                    _row("a", 1, raw_path=str(root / "a.jsonl"), spike=0.03, cluster="c1"),
                    _row("b", 2, raw_path=str(root / "b.jsonl"), spike=0.01, cluster="c2"),
                    _row("c", 3, raw_path=str(root / "c.jsonl"), spike=0.02, cluster="c3"),
                ],
            )
            _write_raw(root / "a.jsonl", spike=True)
            _write_raw(root / "b.jsonl", spike=False)
            _write_raw(root / "c.jsonl", spike=False)
            output_dir = root / "out"
            config = {
                "execution": {"provider": "modal", "allow_local_paths_for_tests": True},
                "artifacts": {
                    "exact_diagnostics_path": str(diagnostics_path),
                    "output_dir": str(output_dir),
                },
                "quality": {"sample_rate": 30.0, "max_samples_per_clip": 90},
                "ablation": {
                    "spike_rate_threshold": 0.025,
                    "top_k_values": [1, 2, 3],
                    "generate_submissions": False,
                },
            }

            result = run_spike_hygiene_ablation(config, smoke=False)

            self.assertEqual(result["mode"], "full")
            self.assertTrue((output_dir / "spike_hygiene_ablation_report_full.json").exists())
            self.assertTrue((output_dir / "spike_hygiene_ablation_hard_gate_diagnostics_full.csv").exists())
            report = json.loads((output_dir / "spike_hygiene_ablation_report_full.json").read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["hard_gate"]["topk"]["1"]["spike_fail_rate"], 0.0)

    def test_validate_rejects_non_artifact_paths(self):
        with self.assertRaises(ValueError):
            validate_spike_hygiene_ablation_config(
                {
                    "execution": {"provider": "modal"},
                    "artifacts": {
                        "exact_diagnostics_path": "/tmp/exact.csv",
                        "output_dir": "/artifacts/out",
                    },
                }
            )


def _row(
    sample_id: str,
    rank: int,
    *,
    spike: float,
    cluster: str,
    raw_path: str = "",
    trace_verdict: str = "plausible_motion",
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "worker_id": sample_id,
        "rank": rank,
        "raw_path": raw_path,
        "final_score": 1.0 / rank,
        "score": 1.0 / rank,
        "rerank_score": 1.0 / rank,
        "blend_old_novelty_score": 1.0 / rank,
        "quality_score": 0.99,
        "stationary_fraction": 0.1,
        "max_abs_value": 12.0,
        "quality__spike_rate": spike,
        "new_cluster_id": cluster,
        "physical_validity_pass": True,
        "trace__verdict": trace_verdict,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_raw(path: Path, *, spike: bool) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for idx in range(90):
            acc_z = 9.81
            if spike and idx in {10, 11, 12}:
                acc_z = 30.0 if idx == 11 else -10.0
            handle.write(
                json.dumps(
                    {
                        "t_us": int(idx * 1_000_000 / 30),
                        "acc": [0.1, 0.2, acc_z],
                        "gyro": [0.01, 0.02, 0.03],
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    unittest.main()
