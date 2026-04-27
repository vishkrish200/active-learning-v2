import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from marginal_value.logging_utils import log_event, log_progress


class ModalLoggingTests(unittest.TestCase):
    def test_log_event_writes_structured_json_line(self):
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            log_event("unit", "started", mode="smoke", count=3)

        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["component"], "unit")
        self.assertEqual(payload["event"], "started")
        self.assertEqual(payload["mode"], "smoke")
        self.assertEqual(payload["count"], 3)
        self.assertIn("ts", payload)

    def test_log_progress_throttles_by_interval_and_final_item(self):
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            emitted = [
                log_progress("unit", "loop", index=index, total=5, every=2)
                for index in range(1, 6)
            ]

        self.assertEqual(emitted, [False, True, False, True, True])
        payloads = [json.loads(line) for line in buffer.getvalue().splitlines()]
        self.assertEqual([payload["index"] for payload in payloads], [2, 4, 5])

    def test_modal_workloads_use_structured_logging(self):
        paths = [
            "modal_train.py",
            "modal_eval.py",
            "modal_ablation.py",
            "modal_rank.py",
            "modal_audit.py",
            "modal_top_clip_visual_audit.py",
            "modal_source_blocked_eval.py",
            "modal_tokenizer.py",
            "modal_grammar.py",
            "modal_grammar_ablation.py",
            "modal_physical_leave_cluster_eval.py",
            "marginal_value/training/torch_train.py",
            "marginal_value/eval/modal_encoder_eval.py",
            "marginal_value/eval/modal_ablation_eval.py",
            "marginal_value/eval/grammar_ablation_eval.py",
            "marginal_value/eval/physical_leave_cluster_eval.py",
            "marginal_value/eval/source_blocked_eval.py",
            "marginal_value/eval/top_clip_visual_audit.py",
            "marginal_value/ranking/modal_baseline_rank.py",
            "marginal_value/eval/ranking_audit.py",
            "marginal_value/tokenization/modal_tokenizer.py",
            "marginal_value/tokenization/modal_grammar.py",
        ]

        for path in paths:
            with self.subTest(path=path):
                source = Path(path).read_text(encoding="utf-8")
                self.assertIn("log_event", source)


if __name__ == "__main__":
    unittest.main()
