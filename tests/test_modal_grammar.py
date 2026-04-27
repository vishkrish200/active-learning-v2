import unittest
from pathlib import Path


class ModalGrammarEntrypointTests(unittest.TestCase):
    def test_modal_grammar_dispatches_smoke_before_optional_full(self):
        source = Path("modal_grammar.py").read_text(encoding="utf-8")

        self.assertIn("remote_grammar_smoke.remote", source)
        self.assertIn("remote_grammar_full.remote", source)
        self.assertIn("run_grammar_pipeline", source)
        self.assertIn("configs/modal_grammar.json", source)
        self.assertIn("if not run_full", source)
        self.assertNotIn("remote_grammar_full.local", source)

    def test_modal_grammar_requests_h100_and_structured_logs(self):
        source = Path("modal_grammar.py").read_text(encoding="utf-8")

        self.assertIn('gpu="H100"', source)
        self.assertIn("log_event", source)
        self.assertIn("artifacts_volume.commit", source)
        self.assertIn("ARTIFACTS_VOLUME_NAME", source)

    def test_modal_grammar_can_skip_redundant_smoke_for_full_runs(self):
        source = Path("modal_grammar.py").read_text(encoding="utf-8")

        self.assertIn("skip_smoke", source)
        self.assertIn("local_smoke_skipped", source)


if __name__ == "__main__":
    unittest.main()
