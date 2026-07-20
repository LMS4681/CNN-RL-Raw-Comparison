"""CLI regression tests for training resume support."""

import os
import sys
import tempfile
import unittest
import zipfile
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import train as train_module
from alloc_env.observation_state import ObservationScales


WORKSPACE_CODES = [
    "PE049", "PE050", "PE055", "PE054", "PE056",
    "PE048", "PE044", "PE059", "PE060", "PE061",
]

REQUIRED_COMPATIBILITY_KEYS = {
    "training_data_schema_version",
    "observation_schema_version",
    "reward_schema_version",
    "extractor",
    "features_dim",
    "active_workspace_codes",
    "state_context",
    "grid_size",
    "ordered_future_count",
    "pending_queue_slots",
    "future_day_windows",
    "observation_scales",
    "data_split_seed",
    "source_sha256",
    "episode_block_count",
    "target_month_counts",
    "excluded_start_months",
    "monthly_jitter",
    "empirical_profile_probability",
    "learning_rate",
    "n_steps",
    "batch_size",
    "n_epochs",
    "gamma",
    "gae_lambda",
}


def complete_config(observation_schema_version=3):
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
        "active_workspace_codes": list(WORKSPACE_CODES),
        "data_split_seed": 20260716,
        "source_sha256": "abc123",
        "episode_block_count": 913,
        "target_month_counts": {"2026-01": 913},
        "excluded_start_months": [7, 11],
        "monthly_jitter": 20,
        "empirical_profile_probability": 0.2,
        "learning_rate": 3e-4,
        "n_steps": 960,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 1.0,
        "gae_lambda": 0.98,
        "seed": 0,
        "eval_scenarios": "./data/fixed_eval_scenarios.json",
    }


def make_args():
    return SimpleNamespace(
        extractor="candidate-cnn",
        features_dim=256,
        state_context="full",
        monthly_jitter=20,
        empirical_profile_probability=0.2,
        lr=3e-4,
        n_steps=960,
        batch_size=64,
        n_epochs=10,
        gamma=1.0,
        gae_lambda=0.98,
        seed=0,
        eval_scenarios="./data/fixed_eval_scenarios.json",
    )


def source_manifest():
    return {
        "split_seed": 20260716,
        "source_sha256": "abc123",
        "source_row_count": 913,
        "source_month_counts": {"2026-01": 913},
    }


def different_value(value):
    if isinstance(value, str):
        return f"{value}-changed"
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.125
    if isinstance(value, list):
        return [*value, "changed"]
    if isinstance(value, dict):
        return {**value, "changed": True}
    raise TypeError(f"No changed value for {type(value).__name__}")


def touch_model(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"model archive placeholder")
    return path


def fake_loader(timesteps_by_path, unreadable=()):
    normalized = {
        Path(path).resolve(): value for path, value in timesteps_by_path.items()
    }
    unreadable_paths = {Path(path).resolve() for path in unreadable}

    def load(path, *, device):
        assert device == "cpu"
        resolved = Path(path).resolve()
        if resolved in unreadable_paths:
            raise ValueError("unreadable archive")
        return SimpleNamespace(num_timesteps=normalized[resolved])

    load.calls = []

    def recording_load(path, *, device):
        load.calls.append(Path(path).resolve())
        return load(path, device=device)

    recording_load.calls = load.calls
    return recording_load


def test_run_config_contains_every_compatibility_key():
    config = train_module.current_run_config(
        make_args(),
        WORKSPACE_CODES,
        source_manifest(),
        ObservationScales.from_dict(complete_config()["observation_scales"]),
    )

    assert REQUIRED_COMPATIBILITY_KEYS <= config.keys()
    assert train_module.CONFIG_COMPATIBILITY_KEYS == tuple(
        sorted(REQUIRED_COMPATIBILITY_KEYS)
    )


@pytest.mark.parametrize("key", sorted(REQUIRED_COMPATIBILITY_KEYS))
def test_resume_rejects_each_changed_compatibility_value(key):
    saved = complete_config()
    current = complete_config()
    current[key] = different_value(current[key])

    assert not train_module.configs_compatible(saved, current)


@pytest.mark.parametrize("missing_key", ["split_seed", "source_sha256"])
def test_run_config_requires_manifest_compatibility_fields(missing_key):
    manifest = source_manifest()
    del manifest[missing_key]

    with pytest.raises(ValueError, match=missing_key):
        train_module.current_run_config(
            make_args(),
            WORKSPACE_CODES,
            manifest,
            ObservationScales.from_dict(
                complete_config()["observation_scales"]
            ),
        )


def test_config_mismatch_diagnostics_report_one_line_per_key():
    saved = complete_config()
    current = complete_config()
    current["batch_size"] = 128
    current["source_sha256"] = "changed"

    with pytest.raises(ValueError) as error:
        train_module.require_compatible_run_config(
            saved, current, source="test"
        )

    lines = str(error.value).splitlines()
    assert len([line for line in lines if "batch_size" in line]) == 1
    assert len([line for line in lines if "source_sha256" in line]) == 1
    assert train_module.config_mismatches(saved, current) == {
        "batch_size": (64, 128),
        "source_sha256": ("abc123", "changed"),
    }


def test_incompatible_resume_is_rejected_before_training_environment(tmp_path):
    args = make_args()
    args.final_holdout_report = False
    args.data_dir = str(tmp_path / "data")
    args.output_dir = str(tmp_path / "output")
    args.active_workspace_codes = train_module.DEFAULT_ACTIVE_WORKSPACE_CODES
    args.grid_size = 64
    args.n_envs = 1
    args.vec_env = "dummy"
    args.device = "cpu"
    blocks = [SimpleNamespace(in_date=date(2026, 1, 5))]
    workspaces = [SimpleNamespace(code=code) for code in WORKSPACE_CODES]
    source_split = SimpleNamespace(
        training_blocks=blocks,
        manifest=source_manifest(),
    )
    training_env = MagicMock()
    training_env.observation_space = "observation-space"
    training_env.action_space = "action-space"

    with (
        patch.object(
            train_module,
            "load_requested_evaluation_scenarios",
            return_value=None,
        ),
        patch.object(train_module, "set_global_seed"),
        patch.object(
            train_module,
            "load_allocation_scenario",
            return_value=(blocks, workspaces),
        ),
        patch(
            "alloc_env.observation_state.build_observation_scales",
            return_value=ObservationScales.from_dict(
                complete_config()["observation_scales"]
            ),
        ),
        patch(
            "alloc_env.data_split.split_blocks_by_ship",
            return_value=source_split,
        ),
        patch(
            "alloc_env.block_generator.SyntheticBlockGenerator.from_blocks",
            return_value=object(),
        ),
        patch.object(
            train_module,
            "current_run_config",
            return_value=complete_config(),
        ),
        patch.object(
            train_module,
            "resolve_resume_path",
            side_effect=ValueError("incompatible source_sha256"),
        ),
        patch.object(
            train_module,
            "create_training_env",
            return_value=training_env,
        ) as create_training_env,
        patch.object(train_module, "resolve_vec_env_type", return_value="single"),
        patch.object(train_module, "estimate_rollout_buffer_mb", return_value=1.0),
        patch.object(train_module, "build_policy_kwargs", return_value={}),
        pytest.raises(ValueError, match="source_sha256"),
    ):
        train_module.train(args)

    create_training_env.assert_not_called()


def test_newer_checkpoint_beats_stale_final_model(tmp_path):
    final = touch_model(tmp_path / train_module.MODEL_FILENAME)
    newer = touch_model(
        tmp_path / "checkpoints" / "model_150000_steps.sb3"
    )
    loader = fake_loader({final: 100_000, newer: 150_000})

    assert train_module.find_resumable_model(tmp_path, loader=loader) == newer


def test_unreadable_high_named_checkpoint_is_ignored(tmp_path):
    valid = touch_model(
        tmp_path / "checkpoints" / "model_100000_steps.sb3"
    )
    broken = touch_model(
        tmp_path / "checkpoints" / "model_999999_steps.sb3"
    )
    loader = fake_loader({valid: 100_000}, unreadable={broken})

    assert train_module.find_resumable_model(tmp_path, loader=loader) == valid


def test_final_model_wins_only_when_stored_timesteps_tie(tmp_path):
    final = touch_model(tmp_path / train_module.MODEL_FILENAME)
    checkpoint = touch_model(
        tmp_path / "checkpoints" / "model_100000_steps.sb3"
    )
    os.utime(checkpoint, ns=(2_000_000_000, 2_000_000_000))
    loader = fake_loader({final: 100_000, checkpoint: 100_000})

    assert train_module.find_resumable_model(tmp_path, loader=loader) == final


def test_equal_checkpoint_timesteps_use_mtime_then_name(tmp_path):
    older = touch_model(
        tmp_path / "checkpoints" / "zeta_100000_steps.sb3"
    )
    newer = touch_model(
        tmp_path / "checkpoints" / "alpha_100000_steps.sb3"
    )
    os.utime(older, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer, ns=(2_000_000_000, 2_000_000_000))
    loader = fake_loader({older: 100_000, newer: 100_000})

    assert train_module.find_resumable_model(tmp_path, loader=loader) == newer

    os.utime(older, ns=(2_000_000_000, 2_000_000_000))
    assert train_module.find_resumable_model(tmp_path, loader=loader) == older


def test_actual_checkpoint_filename_is_resumable(tmp_path):
    callback = train_module.Sb3CheckpointCallback(
        save_freq=10,
        save_path=str(tmp_path / "checkpoints"),
        name_prefix="block_placement_ppo",
    )
    callback.num_timesteps = 20
    checkpoint = touch_model(Path(callback._checkpoint_path(extension="zip")))
    loader = fake_loader({checkpoint: 20})

    assert train_module.find_resumable_model(tmp_path, loader=loader) == checkpoint


def test_best_model_selection_artifact_is_excluded(tmp_path):
    checkpoint = touch_model(
        tmp_path / "checkpoints" / "model_100000_steps.sb3"
    )
    root_best = touch_model(tmp_path / "best_model.sb3")
    checkpoint_best = touch_model(tmp_path / "checkpoints" / "best_model.sb3")
    loader = fake_loader(
        {checkpoint: 100_000, root_best: 999_999, checkpoint_best: 999_999}
    )

    assert train_module.find_resumable_model(tmp_path, loader=loader) == checkpoint
    assert root_best.resolve() not in loader.calls
    assert checkpoint_best.resolve() not in loader.calls


@pytest.mark.parametrize(
    "error_type",
    [EOFError, OSError, RuntimeError, ValueError, zipfile.BadZipFile],
)
def test_model_num_timesteps_ignores_supported_archive_errors(
    tmp_path, error_type
):
    model_path = touch_model(tmp_path / "model.sb3")

    def loader(path, *, device):
        assert path == str(model_path)
        assert device == "cpu"
        raise error_type("broken")

    assert train_module.model_num_timesteps(model_path, loader=loader) is None


class TrainResumeCliTest(unittest.TestCase):
    @staticmethod
    def _run_config(observation_schema_version=3):
        return complete_config(observation_schema_version)

    def test_primary_model_filename_avoids_security_filtered_zip_suffix(self):
        self.assertEqual(
            "block_placement_ppo.sb3", train_module.MODEL_FILENAME
        )

    def test_resumable_model_supports_legacy_final_archive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            legacy = touch_model(output_dir / "block_placement_ppo.zip")
            self.assertEqual(
                legacy,
                train_module.find_resumable_model(
                    output_dir, loader=fake_loader({legacy: 100})
                ),
            )

            preferred = touch_model(output_dir / "block_placement_ppo.sb3")
            self.assertEqual(
                preferred,
                train_module.find_resumable_model(
                    output_dir,
                    loader=fake_loader({legacy: 100, preferred: 200}),
                ),
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

    def test_auto_resume_ignores_unreadable_filtered_final_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "block_placement_ppo.zip"
            model_path.write_bytes(b"HHIDfiltered")

            self.assertIsNone(train_module.find_resumable_model(tmpdir))

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

        self.assertFalse(train_module.configs_compatible(saved, current))
        self.assertEqual(
            {"monthly_jitter": (10, 20)},
            train_module.config_mismatches(saved, current),
        )

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
