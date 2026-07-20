"""CLI regression tests for training resume support."""

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import train as train_module


class TrainResumeCliTest(unittest.TestCase):
    @staticmethod
    def _run_config(observation_schema_version=3):
        return {
            "training_data_schema_version": 2,
            "observation_schema_version": observation_schema_version,
            "reward_schema_version": 2,
            "extractor": "candidate-cnn",
            "state_context": "full",
            "grid_size": 64,
            "ordered_future_count": 16,
            "pending_queue_slots": 32,
            "future_day_windows": [[0, 5], [6, 20], [21, 60]],
            "observation_scales": {
                "max_length": 100.0,
                "max_breadth": 50.0,
                "max_duration": 60,
                "base_date": "2025-12-01",
                "date_span_workdays": 150,
                "max_workspace_area": 10_000.0,
                "total_workspace_area": 80_000.0,
                "max_workspace_length": 200.0,
                "max_workspace_breadth": 100.0,
                "dropout_threshold": 7,
            },
            "features_dim": 256,
            "active_workspace_codes": [
                "PE049", "PE050", "PE055", "PE054", "PE056",
                "PE048", "PE044", "PE059", "PE060", "PE061",
            ],
            "data_split_seed": 20260716,
            "source_sha256": "abc123",
            "episode_block_count": 913,
            "target_month_counts": {"2026-01": 913},
            "excluded_start_months": [7, 11],
            "monthly_jitter": 20,
            "empirical_profile_probability": 0.2,
        }

    def test_primary_model_filename_avoids_security_filtered_zip_suffix(self):
        self.assertEqual(
            "block_placement_ppo.sb3", train_module.MODEL_FILENAME
        )

    def test_resumable_model_prefers_sb3_and_supports_legacy_zip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            legacy = output_dir / "block_placement_ppo.zip"
            with zipfile.ZipFile(legacy, "w") as archive:
                archive.writestr("probe", "legacy")
            self.assertEqual(
                legacy, train_module.find_resumable_model(output_dir)
            )

            preferred = output_dir / "block_placement_ppo.sb3"
            with zipfile.ZipFile(preferred, "w") as archive:
                archive.writestr("probe", "preferred")
            self.assertEqual(
                preferred, train_module.find_resumable_model(output_dir)
            )

    def test_checkpoint_callback_uses_sb3_extension(self):
        callback_class = getattr(
            train_module, "Sb3CheckpointCallback", None
        )
        self.assertIsNotNone(callback_class)
        callback = callback_class(
            save_freq=10,
            save_path="checkpoints",
            name_prefix="block_placement_ppo",
        )
        callback.num_timesteps = 20

        self.assertTrue(
            callback._checkpoint_path(extension="zip").endswith(
                "block_placement_ppo_20_steps.sb3"
            )
        )

    def test_model_archive_resolution_accepts_suffixless_sb3_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.sb3"
            with zipfile.ZipFile(model_path, "w") as archive:
                archive.writestr("probe", "model")

            resolved = train_module.resolve_model_archive_path(
                model_path.with_suffix("")
            )

        self.assertEqual(model_path.resolve(), resolved)

    def test_model_archive_resolution_rejects_filtered_archive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.zip"
            model_path.write_bytes(b"HHIDfiltered")

            with self.assertRaisesRegex(ValueError, "not a readable SB3"):
                train_module.resolve_model_archive_path(model_path)

    def test_auto_resume_does_not_silently_ignore_filtered_final_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "block_placement_ppo.zip"
            model_path.write_bytes(b"HHIDfiltered")

            with self.assertRaisesRegex(ValueError, "not a readable SB3"):
                train_module.find_resumable_model(tmpdir)

    def test_resume_from_argument_is_accepted(self):
        captured = {}

        def fake_train(args):
            captured["resume_from"] = args.resume_from
            captured["extractor"] = args.extractor
            captured["state_context"] = args.state_context
            captured["gae_lambda"] = args.gae_lambda
            captured["seed"] = args.seed
            captured["eval_scenarios"] = args.eval_scenarios

        argv = [
            "train.py",
            "--resume-from",
            ".\\output\\block_placement_ppo.zip",
            "--eval-scenarios",
            ".\\data\\fixed_eval_scenarios.json",
            "--no-export-onnx",
        ]

        with patch.object(sys, "argv", argv), patch.object(train_module, "train", fake_train):
            train_module.main()

        self.assertEqual(captured["resume_from"], ".\\output\\block_placement_ppo.zip")
        self.assertEqual("candidate-cnn", captured["extractor"])
        self.assertEqual("full", captured["state_context"])
        self.assertEqual(0.98, captured["gae_lambda"])
        self.assertEqual(0, captured["seed"])
        self.assertEqual(
            ".\\data\\fixed_eval_scenarios.json",
            captured["eval_scenarios"],
        )

    def test_obsolete_future_cli_argument_is_rejected(self):
        with (
            patch.object(
                sys,
                "argv",
                ["train.py", "--n-future-blocks", "4"],
            ),
            patch.object(train_module, "train") as train,
            self.assertRaises(SystemExit),
        ):
            train_module.main()

        train.assert_not_called()

    def test_explicit_resume_rejects_incompatible_run_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "block_placement_ppo.sb3"
            with zipfile.ZipFile(model_path, "w") as archive:
                archive.writestr("probe", "model")
            train_module.write_run_config(
                tmpdir, self._run_config(observation_schema_version=1)
            )
            args = SimpleNamespace(
                resume_from=str(model_path), auto_resume=False
            )

            with self.assertRaisesRegex(
                ValueError, "observation_schema_version"
            ):
                train_module.resolve_resume_path(
                    args, tmpdir, self._run_config()
                )

    def test_explicit_resume_requires_recorded_run_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "block_placement_ppo.sb3"
            with zipfile.ZipFile(model_path, "w") as archive:
                archive.writestr("probe", "model")
            args = SimpleNamespace(
                resume_from=str(model_path), auto_resume=False
            )

            with self.assertRaisesRegex(FileNotFoundError, "run_config.json"):
                train_module.resolve_resume_path(
                    args, tmpdir, self._run_config()
                )

    def test_explicit_resume_accepts_compatible_run_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "block_placement_ppo.sb3"
            with zipfile.ZipFile(model_path, "w") as archive:
                archive.writestr("probe", "model")
            train_module.write_run_config(tmpdir, self._run_config())
            args = SimpleNamespace(
                resume_from=str(model_path), auto_resume=False
            )

            resolved = train_module.resolve_resume_path(
                args, tmpdir, self._run_config()
            )

        self.assertEqual(model_path.resolve(), resolved)

    def test_resume_rejects_changed_monthly_training_profile(self):
        saved = self._run_config()
        current = self._run_config()
        saved["monthly_jitter"] = 10

        compatible, bad_key = train_module.configs_compatible(
            saved, current
        )

        self.assertFalse(compatible)
        self.assertEqual("monthly_jitter", bad_key)

    def test_model_tools_reject_legacy_training_data_schema(self):
        validator = getattr(
            train_module, "require_current_training_data_schema", None
        )
        self.assertIsNotNone(validator)

        with self.assertRaisesRegex(ValueError, "training_data_schema_version"):
            validator({}, source="test")
        with self.assertRaisesRegex(ValueError, "training_data_schema_version"):
            validator({"training_data_schema_version": 0}, source="test")

        validator(
            {"training_data_schema_version": 2}, source="test"
        )

    def test_model_tools_require_observation_schema3(self):
        validator = getattr(
            train_module, "require_current_observation_schema", None
        )
        self.assertIsNotNone(validator)

        with self.assertRaisesRegex(ValueError, "observation_schema_version"):
            validator({}, source="test")
        with self.assertRaisesRegex(ValueError, "schema-3"):
            validator(
                {"observation_schema_version": 2}, source="test"
            )

        validator({"observation_schema_version": 3}, source="test")

    def test_model_contract_reconstructs_saved_schema3_observation_values(self):
        parser = getattr(
            train_module, "observation_contract_from_run_config", None
        )
        self.assertIsNotNone(parser)

        workspace_codes, state_context, scales = parser(
            self._run_config(), source="test"
        )

        self.assertEqual(
            self._run_config()["active_workspace_codes"], workspace_codes
        )
        self.assertEqual("full", state_context)
        self.assertEqual(
            self._run_config()["observation_scales"], scales.to_dict()
        )

    def test_model_contract_rejects_changed_fixed_grid_constant(self):
        parser = getattr(
            train_module, "observation_contract_from_run_config", None
        )
        self.assertIsNotNone(parser)
        config = self._run_config()
        config["grid_size"] = 32

        with self.assertRaisesRegex(ValueError, "grid_size"):
            parser(config, source="test")


if __name__ == "__main__":
    unittest.main()
