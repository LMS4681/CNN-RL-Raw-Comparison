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
    def _run_config(observation_schema_version=2):
        return {
            "training_data_schema_version": 2,
            "observation_schema_version": observation_schema_version,
            "reward_schema_version": 2,
            "extractor": "candidate-cnn",
            "n_future_blocks": 4,
            "grid_size": 32,
            "features_dim": 256,
            "active_workspace_codes": ["PE001"],
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
            captured["n_future_blocks"] = args.n_future_blocks
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
        self.assertEqual(4, captured["n_future_blocks"])
        self.assertEqual(0.98, captured["gae_lambda"])
        self.assertEqual(0, captured["seed"])
        self.assertEqual(
            ".\\data\\fixed_eval_scenarios.json",
            captured["eval_scenarios"],
        )

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


if __name__ == "__main__":
    unittest.main()
