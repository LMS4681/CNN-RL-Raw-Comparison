"""CLI regression tests for training resume support."""

import os
import pickle
import sys
import tempfile
import unittest
import zipfile
from dataclasses import asdict
from datetime import date
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from stable_baselines3.common import save_util

import train as train_module
from alloc_env.observation_state import ObservationScales
from comparison.wall_clock_callback import WallClockState, atomic_write_json


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
    "extractor_output_dim",
    "policy_net_arch",
    "policy_activation",
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
        "extractor_output_dim": 256,
        "policy_net_arch": {"pi": [64, 64], "vf": [64, 64]},
        "policy_activation": "ReLU",
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


def save_real_model(path: Path, timesteps: int) -> Path:
    model = train_module.MaskablePPO(
        "MlpPolicy",
        "CartPole-v1",
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        device="cpu",
        verbose=0,
    )
    try:
        model.num_timesteps = timesteps
        model.save(path)
    finally:
        env = model.get_env()
        if env is not None:
            env.close()
    return path


def write_wall_clock_state(
    output_dir: Path,
    checkpoint: Path,
    *,
    timestep: int,
    state_path: Path | None = None,
) -> Path:
    target = state_path or output_dir / "run_state.json"
    state = WallClockState(
        schema_version=1,
        target_training_seconds=10_800.0,
        completed_training_seconds=600.0,
        last_checkpoint_timestep=timestep,
        last_regular_checkpoint_timestep=0,
        last_checkpoint_file=checkpoint.name,
        last_checkpoint_sha256=sha256(checkpoint.read_bytes()).hexdigest(),
        config_sha256="a" * 64,
        generation=1,
        restart_count=0,
        max_unrecorded_seconds=300.0,
        status="running",
        started_at_utc="2026-07-21T00:00:00+00:00",
        updated_at_utc="2026-07-21T00:10:00+00:00",
        completed_at_utc=None,
    )
    atomic_write_json(target, asdict(state))
    return target


def wall_clock_args(checkpoint: Path | None, **overrides):
    values = {
        "resume_from": str(checkpoint) if checkpoint is not None else None,
        "auto_resume": False,
        "max_training_seconds": 10_800.0,
        "wall_clock_state": None,
        "comparison_config_sha256": "a" * 64,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def corrupt_policy_member(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    replaced = False
    with (
        zipfile.ZipFile(source, "r") as source_archive,
        zipfile.ZipFile(destination, "w") as destination_archive,
    ):
        for info in source_archive.infolist():
            content = source_archive.read(info.filename)
            if info.filename == "policy.pth":
                content = b"not a torch pickle"
                replaced = True
            destination_archive.writestr(info, content)
    assert replaced


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


def test_raw_direct_run_config_records_fixed_output_dimension():
    args = make_args()
    args.extractor = "raw-direct"
    config = train_module.current_run_config(
        args,
        WORKSPACE_CODES,
        source_manifest(),
        ObservationScales.from_dict(complete_config()["observation_scales"]),
    )
    assert config["extractor_output_dim"] == 2772
    assert config["policy_net_arch"] == {"pi": [64, 64], "vf": [64, 64]}
    assert config["policy_activation"] == "ReLU"


@pytest.mark.parametrize("key", sorted(REQUIRED_COMPATIBILITY_KEYS))
def test_resume_rejects_each_changed_compatibility_value(key):
    saved = complete_config()
    current = complete_config()
    current[key] = different_value(current[key])

    assert train_module.configs_compatible(saved, current) is False


def test_configs_compatible_returns_an_explicit_bool():
    assert train_module.configs_compatible(
        complete_config(), complete_config()
    ) is True


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


@pytest.mark.parametrize(
    ("corruption", "expected_error"),
    [
        ("policy", pickle.UnpicklingError),
        ("empty", AssertionError),
    ],
    ids=["corrupt-policy-pth", "empty-zip"],
)
def test_real_sb3_corrupt_candidate_is_ignored_without_leaking_handles(
    tmp_path,
    monkeypatch,
    corruption,
    expected_error,
):
    valid = save_real_model(tmp_path / train_module.MODEL_FILENAME, 100_000)
    corrupt = tmp_path / "checkpoints" / "model_999999_steps.sb3"
    if corruption == "policy":
        corrupt_policy_member(valid, corrupt)
    else:
        corrupt.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(corrupt, "w"):
            pass

    opened_files = []
    original_open_path = save_util.open_path

    def tracking_open_path(*args, **kwargs):
        file = original_open_path(*args, **kwargs)
        opened_files.append(file)
        return file

    monkeypatch.setattr(save_util, "open_path", tracking_open_path)
    with pytest.raises(expected_error):
        train_module.MaskablePPO.load(str(corrupt), device="cpu")

    assert train_module.find_resumable_model(tmp_path) == valid
    assert opened_files
    assert all(file.closed for file in opened_files)

    corrupt.unlink()
    valid.unlink()


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


@pytest.mark.parametrize(
    "error",
    [AssertionError("unexpected assertion"), LookupError("unexpected loader")],
)
def test_model_num_timesteps_does_not_swallow_unexpected_loader_errors(
    tmp_path, error
):
    model_path = touch_model(tmp_path / "model.sb3")

    def loader(path, *, device):
        raise error

    with pytest.raises(type(error), match="unexpected"):
        train_module.model_num_timesteps(model_path, loader=loader)


def test_wall_clock_resume_accepts_only_state_named_verified_archive(tmp_path):
    output_dir = tmp_path / "output"
    checkpoint = save_real_model(
        output_dir / "checkpoints" / "model_100_g1.sb3", 100
    )
    write_wall_clock_state(output_dir, checkpoint, timestep=100)
    train_module.write_run_config(output_dir, complete_config())

    resolved = train_module.resolve_resume_path(
        wall_clock_args(checkpoint), output_dir, complete_config()
    )

    assert resolved == checkpoint.resolve()


def test_wall_clock_resume_rejects_archive_not_named_by_state(tmp_path):
    output_dir = tmp_path / "output"
    named = save_real_model(
        output_dir / "checkpoints" / "model_100_g1.sb3", 100
    )
    other = save_real_model(
        output_dir / "checkpoints" / "model_100_g2.sb3", 100
    )
    write_wall_clock_state(output_dir, named, timestep=100)
    train_module.write_run_config(output_dir, complete_config())

    with pytest.raises(ValueError, match="exact checkpoint named by run_state"):
        train_module.resolve_resume_path(
            wall_clock_args(other), output_dir, complete_config()
        )


def test_wall_clock_resume_rejects_checkpoint_hash_mismatch(tmp_path):
    output_dir = tmp_path / "output"
    checkpoint = save_real_model(
        output_dir / "checkpoints" / "model_100_g1.sb3", 100
    )
    write_wall_clock_state(output_dir, checkpoint, timestep=100)
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    train_module.write_run_config(output_dir, complete_config())

    with pytest.raises(ValueError, match="SHA256"):
        train_module.resolve_resume_path(
            wall_clock_args(checkpoint), output_dir, complete_config()
        )


def test_wall_clock_resume_rejects_stored_timestep_mismatch(tmp_path):
    output_dir = tmp_path / "output"
    checkpoint = save_real_model(
        output_dir / "checkpoints" / "model_100_g1.sb3", 100
    )
    write_wall_clock_state(output_dir, checkpoint, timestep=99)
    train_module.write_run_config(output_dir, complete_config())

    with pytest.raises(ValueError, match="state checkpoint timestep 99.*archive 100"):
        train_module.resolve_resume_path(
            wall_clock_args(checkpoint), output_dir, complete_config()
        )


def test_wall_clock_mode_rejects_broad_auto_resume(tmp_path):
    with pytest.raises(ValueError, match="auto-resume.*wall-clock"):
        train_module.resolve_resume_path(
            wall_clock_args(None, auto_resume=True),
            tmp_path,
            complete_config(),
        )


def test_wall_clock_state_requires_explicit_exact_resume_archive(tmp_path):
    output_dir = tmp_path / "output"
    checkpoint = save_real_model(
        output_dir / "checkpoints" / "model_100_g1.sb3", 100
    )
    write_wall_clock_state(output_dir, checkpoint, timestep=100)
    train_module.write_run_config(output_dir, complete_config())

    with pytest.raises(ValueError, match="--resume-from"):
        train_module.resolve_resume_path(
            wall_clock_args(None), output_dir, complete_config()
        )


class TrainResumeCliTest(unittest.TestCase):
    @staticmethod
    def _run_config(observation_schema_version=3):
        return complete_config(observation_schema_version)

    def test_raw_direct_extractor_argument_is_accepted(self):
        captured = {}

        def fake_train(args):
            captured["extractor"] = args.extractor

        with (
            patch.object(
                sys, "argv", ["train.py", "--extractor", "raw-direct"]
            ),
            patch.object(train_module, "train", fake_train),
        ):
            train_module.main()

        self.assertEqual("raw-direct", captured["extractor"])

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

    def test_wall_clock_arguments_are_accepted(self):
        captured = {}

        def fake_train(args):
            captured.update(vars(args))

        argv = [
            "train.py",
            "--max-training-seconds",
            "10800",
            "--wall-clock-state",
            ".\\output\\run_state.json",
            "--wall-clock-heartbeat-seconds",
            "300",
            "--comparison-config-sha256",
            "a" * 64,
        ]

        with patch.object(sys, "argv", argv), patch.object(
            train_module, "train", fake_train
        ):
            train_module.main()

        self.assertEqual(10_800, captured["max_training_seconds"])
        self.assertEqual(
            ".\\output\\run_state.json", captured["wall_clock_state"]
        )
        self.assertEqual(300, captured["wall_clock_heartbeat_seconds"])
        self.assertEqual("a" * 64, captured["comparison_config_sha256"])

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
