import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marginal_value.active.run_hidden_test import (
    prepare_hidden_test_run,
    validate_hidden_test_run_package,
)


class ActiveRunHiddenTestTests(unittest.TestCase):
    def test_prepare_hidden_test_run_writes_manifest_bound_configs_and_commands(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_manifest = root / "old_urls.txt"
            new_manifest = root / "new_urls.txt"
            old_manifest.write_text(
                "\n".join(
                    [
                        "https://storage.googleapis.com/unit/pretrain/worker00000/clip-a.jsonl",
                        "https://storage.googleapis.com/unit/pretrain/worker00001/clip-b.jsonl",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            new_manifest.write_text(
                "\n".join(
                    [
                        "https://storage.googleapis.com/unit/new/worker10000/clip-a.jsonl",
                        "https://storage.googleapis.com/unit/new/worker10001/clip-b.jsonl",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            run_dir = root / "hidden_run"

            result = prepare_hidden_test_run(
                {
                    "inputs": {
                        "old_manifest": str(old_manifest),
                        "new_manifest": str(new_manifest),
                    },
                    "artifacts": {"run_dir": str(run_dir)},
                    "modal": {
                        "data_volume": "imu-novelty-subset-data",
                        "artifacts_volume": "activelearning-imu-rebuild-cache",
                        "remote_manifest_dir": "cache/manifests/hidden_test/unit_run",
                        "remote_artifact_dir": "/artifacts/active/hidden_test/unit_run",
                    },
                    "method": {
                        "left_support_shard_dir": "/artifacts/frozen/ts2vec_support_shards",
                        "min_left_support_clips": 2,
                        "ts2vec_checkpoint_path": "/artifacts/checkpoints/ts2vec_best.pt",
                    },
                }
            )
            validation = validate_hidden_test_run_package(run_dir)
            exact_config = json.loads((run_dir / "configs" / "active_exact_window_blend_rank.json").read_text())
            precompute_config = json.loads(
                (run_dir / "configs" / "active_embedding_precompute_ts2vec_new.json").read_text()
            )
            package_config = json.loads((run_dir / "configs" / "final_package_artifact_gate.json").read_text())
            commands = (run_dir / "commands.sh").read_text(encoding="utf-8")
            readme = (run_dir / "README_hidden_test.md").read_text(encoding="utf-8")
            copied_old_manifest_exists = (run_dir / "manifests" / "pretrain_urls.txt").exists()
            copied_new_manifest_exists = (run_dir / "manifests" / "new_urls.txt").exists()

        self.assertEqual(result["status"], "prepared")
        self.assertEqual(result["old_manifest_count"], 2)
        self.assertEqual(result["new_manifest_count"], 2)
        self.assertEqual(validation["status"], "prepared")
        self.assertEqual(validation["old_manifest_count"], 2)
        self.assertEqual(validation["new_manifest_count"], 2)
        self.assertTrue(copied_old_manifest_exists)
        self.assertTrue(copied_new_manifest_exists)
        self.assertEqual(exact_config["data"]["manifests"]["pretrain"], "cache/manifests/hidden_test/unit_run/pretrain_urls.txt")
        self.assertEqual(exact_config["data"]["manifests"]["new"], "cache/manifests/hidden_test/unit_run/new_urls.txt")
        self.assertEqual(exact_config["ranking"]["left_support_shard_dir"], "/artifacts/frozen/ts2vec_support_shards")
        self.assertIn("/artifacts/active/hidden_test/unit_run/query_ts2vec/embeddings_", exact_config["ranking"]["left_query_shard_dir"])
        self.assertTrue(exact_config["ranking"]["left_query_shard_dir"].endswith("_shards"))
        self.assertEqual(precompute_config["embeddings"]["cache_dir"], "/artifacts/active/hidden_test/unit_run/query_ts2vec")
        self.assertEqual(
            package_config["source_artifacts"]["source_dir"],
            "/artifacts/active/hidden_test/unit_run/artifact_hygiene_ablation",
        )
        self.assertIn("modal volume put imu-novelty-subset-data", commands)
        self.assertIn("modal_build_full_support_shards.py", commands)
        self.assertIn("modal_active_embedding_precompute.py", commands)
        self.assertIn("modal_active_exact_window_blend_rank.py", commands)
        self.assertIn("modal_active_spike_hygiene_ablation.py", commands)
        self.assertIn("does not use hidden targets", readme)

    def test_prepare_hidden_test_run_rejects_duplicate_manifest_urls(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_manifest = root / "old_urls.txt"
            new_manifest = root / "new_urls.txt"
            old_manifest.write_text("https://storage.googleapis.com/unit/pretrain/a.jsonl\n", encoding="utf-8")
            new_manifest.write_text(
                "https://storage.googleapis.com/unit/new/a.jsonl\n"
                "https://storage.googleapis.com/unit/new/a.jsonl\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate"):
                prepare_hidden_test_run(
                    {
                        "inputs": {
                            "old_manifest": str(old_manifest),
                            "new_manifest": str(new_manifest),
                        },
                        "artifacts": {"run_dir": str(root / "hidden_run")},
                    }
                )


if __name__ == "__main__":
    unittest.main()
