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
            cache_old_config = json.loads((run_dir / "configs" / "cache_old_manifest_urls.json").read_text())
            cache_new_config = json.loads((run_dir / "configs" / "cache_new_manifest_urls.json").read_text())
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
        self.assertEqual(validation["stage_config_validation"]["status"], "valid")
        self.assertIn("cache_old_manifest_urls", validation["stage_config_validation"]["validated_configs"])
        self.assertIn("cache_new_manifest_urls", validation["stage_config_validation"]["validated_configs"])
        self.assertTrue(copied_old_manifest_exists)
        self.assertTrue(copied_new_manifest_exists)
        self.assertEqual(cache_old_config["target"]["source_manifest"], "cache/manifests/hidden_test/unit_run/pretrain_urls.txt")
        self.assertEqual(cache_old_config["target"]["cached_manifest"], "cache/manifests/hidden_test/unit_run/pretrain_cached_urls.txt")
        self.assertTrue(cache_old_config["execution"]["fail_if_incomplete"])
        self.assertEqual(cache_new_config["target"]["source_manifest"], "cache/manifests/hidden_test/unit_run/new_urls.txt")
        self.assertEqual(cache_new_config["target"]["cached_manifest"], "cache/manifests/hidden_test/unit_run/new_cached_urls.txt")
        self.assertTrue(cache_new_config["execution"]["fail_if_incomplete"])
        self.assertEqual(
            exact_config["data"]["manifests"]["pretrain"],
            "cache/manifests/hidden_test/unit_run/pretrain_cached_urls.txt",
        )
        self.assertEqual(exact_config["data"]["manifests"]["new"], "cache/manifests/hidden_test/unit_run/new_cached_urls.txt")
        self.assertEqual(
            precompute_config["data"]["manifests"]["new"],
            "cache/manifests/hidden_test/unit_run/new_cached_urls.txt",
        )
        self.assertEqual(exact_config["ranking"]["left_support_shard_dir"], "/artifacts/frozen/ts2vec_support_shards")
        self.assertIn("/artifacts/active/hidden_test/unit_run/query_ts2vec/embeddings_", exact_config["ranking"]["left_query_shard_dir"])
        self.assertTrue(exact_config["ranking"]["left_query_shard_dir"].endswith("_shards"))
        self.assertEqual(precompute_config["embeddings"]["cache_dir"], "/artifacts/active/hidden_test/unit_run/query_ts2vec")
        self.assertEqual(
            package_config["source_artifacts"]["source_dir"],
            "/artifacts/active/hidden_test/unit_run/artifact_hygiene_ablation",
        )
        self.assertIn('export MV_DATA_VOLUME="imu-novelty-subset-data"', commands)
        self.assertIn('export MV_ARTIFACTS_VOLUME="activelearning-imu-rebuild-cache"', commands)
        self.assertIn('modal volume put "$MV_DATA_VOLUME"', commands)
        self.assertIn('modal_cache_manifest_urls.py --config-path "$RUN_DIR/configs/cache_old_manifest_urls.json" --run-full', commands)
        self.assertIn('modal_cache_manifest_urls.py --config-path "$RUN_DIR/configs/cache_new_manifest_urls.json" --run-full', commands)
        self.assertLess(
            commands.index("modal_cache_manifest_urls.py --config-path"),
            commands.index("modal_build_full_support_shards.py"),
        )
        self.assertIn("modal_build_full_support_shards.py", commands)
        self.assertIn("modal_active_embedding_precompute.py", commands)
        self.assertIn(
            'modal_active_embedding_precompute.py --config-path "$RUN_DIR/configs/active_embedding_precompute_ts2vec_new.json" --run-full --skip-smoke --wait-full',
            commands,
        )
        self.assertNotIn("spawned by its Modal wrapper", commands)
        self.assertIn("modal_active_exact_window_blend_rank.py", commands)
        self.assertIn("modal_active_spike_hygiene_ablation.py", commands)
        self.assertIn('mkdir -p "$RUN_DIR/source_artifacts/artifact_hygiene_ablation"', commands)
        self.assertIn(
            "spike_hygiene_ablation_artifact_gate_submission_full_new_worker_id.csv",
            commands,
        )
        self.assertIn(
            "spike_hygiene_ablation_artifact_gate_submission_full_worker_id.csv",
            commands,
        )
        self.assertIn("spike_hygiene_ablation_artifact_gate_diagnostics_full.csv", commands)
        self.assertIn("spike_hygiene_ablation_report_full.json", commands)
        self.assertNotIn(
            'modal volume get activelearning-imu-rebuild-cache active/hidden_test/unit_run/artifact_hygiene_ablation "$RUN_DIR/source_artifacts/artifact_hygiene_ablation"',
            commands,
        )
        self.assertIn("Caches raw JSONL and feature NPZ files", readme)
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

    def test_validate_hidden_test_run_package_rejects_cross_stage_query_shard_mismatch(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_manifest = root / "old_urls.txt"
            new_manifest = root / "new_urls.txt"
            old_manifest.write_text("https://storage.googleapis.com/unit/pretrain/a.jsonl\n", encoding="utf-8")
            new_manifest.write_text("https://storage.googleapis.com/unit/new/a.jsonl\n", encoding="utf-8")
            run_dir = root / "hidden_run"
            prepare_hidden_test_run(
                {
                    "inputs": {
                        "old_manifest": str(old_manifest),
                        "new_manifest": str(new_manifest),
                    },
                    "artifacts": {"run_dir": str(run_dir)},
                }
            )
            exact_config_path = run_dir / "configs" / "active_exact_window_blend_rank.json"
            exact_config = json.loads(exact_config_path.read_text(encoding="utf-8"))
            exact_config["ranking"]["left_query_shard_dir"] = "/artifacts/active/hidden_test/bad/query_shards"
            exact_config_path.write_text(json.dumps(exact_config, indent=2, sort_keys=True), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "left_query_shard_dir"):
                validate_hidden_test_run_package(run_dir)


if __name__ == "__main__":
    unittest.main()
