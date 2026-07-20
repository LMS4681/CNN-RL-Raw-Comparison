"""Regression tests for GPU/device and vectorized training configuration."""

import sys
import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import cloudpickle
from sb3_contrib.common.maskable.utils import get_action_masks

import train as train_module
from alloc_env.alloc_env import DROPOUT_THRESHOLD
from alloc_env.block import Block
from alloc_env.cnn_extractor import (
    CandidateCnnExtractor,
    FixedGridExtractor,
    StructuredExtractor,
)
from alloc_env.strategy import BaseGridStrategy
from alloc_env.observation_state import (
    GRID_SIZE,
    N_WORKSPACES,
    ObservationScales,
    build_observation_space,
)
from alloc_env.workspace import Workspace


WORKSPACE_CODES = [
    "PE049", "PE050", "PE055", "PE054", "PE056",
    "PE048", "PE044", "PE059", "PE060", "PE061",
]


def source_manifest() -> dict:
    return {
        "split_seed": 20260716,
        "source_sha256": "abc123",
        "source_row_count": 913,
        "source_month_counts": {
            "2025-12": 64,
            "2026-01": 122,
            "2026-02": 106,
            "2026-03": 142,
            "2026-04": 153,
            "2026-05": 151,
            "2026-06": 175,
        },
    }


def full_source_scales() -> ObservationScales:
    return ObservationScales(
        max_length=100.0,
        max_breadth=50.0,
        max_duration=60,
        base_date=date(2025, 12, 1),
        date_span_workdays=150,
        max_workspace_area=10_000.0,
        total_workspace_area=80_000.0,
        max_workspace_length=200.0,
        max_workspace_breadth=100.0,
        dropout_threshold=DROPOUT_THRESHOLD,
    )


def make_args() -> SimpleNamespace:
    return SimpleNamespace(
        extractor="candidate-cnn",
        features_dim=256,
        state_context="full",
        monthly_jitter=20,
        empirical_profile_probability=0.2,
        seed=0,
        eval_scenarios="./data/fixed_eval_scenarios.json",
    )


def make_block(name: str) -> Block:
    return Block(
        name=name,
        ship_no="T001",
        block_type="BUILD",
        length=10.0,
        breadth=10.0,
        height=5.0,
        weight=10.0,
        in_date=date(2026, 1, 5),
        out_date=date(2026, 1, 30),
    )


def make_workspace(code: str) -> Workspace:
    return Workspace(
        code=code,
        origin_x=0.0,
        origin_y=0.0,
        breadth=80.0,
        length=100.0,
        strategy=BaseGridStrategy(step=10.0),
    )


class ParallelTrainingConfigTests(unittest.TestCase):
    def test_policy_kwargs_select_only_approved_extractors(self):
        expected_extractors = {
            "structured": StructuredExtractor,
            "fixed-grid": FixedGridExtractor,
            "candidate-cnn": CandidateCnnExtractor,
        }
        for name, expected in expected_extractors.items():
            with self.subTest(extractor=name):
                kwargs = train_module.build_policy_kwargs(
                    name, features_dim=128
                )
                self.assertIs(
                    expected, kwargs["features_extractor_class"]
                )
                self.assertEqual(
                    {"features_dim": 128},
                    kwargs["features_extractor_kwargs"],
                )
                self.assertTrue(kwargs["share_features_extractor"])

        with self.assertRaises(ValueError):
            train_module.build_policy_kwargs("block-attn")

    def test_parallel_cli_arguments_are_accepted(self):
        captured = {}

        def fake_train(args):
            captured["device"] = args.device
            captured["n_envs"] = args.n_envs
            captured["vec_env"] = args.vec_env
            captured["active_workspace_codes"] = args.active_workspace_codes
            captured["seed"] = args.seed
            captured["monthly_jitter"] = args.monthly_jitter
            captured["empirical_profile_probability"] = (
                args.empirical_profile_probability
            )
            captured["state_context"] = args.state_context
            captured["grid_size"] = args.grid_size
            captured["holdout_eval_freq"] = args.holdout_eval_freq
            captured["holdout_selection_count"] = (
                args.holdout_selection_count
            )
            captured["final_holdout_report"] = args.final_holdout_report

        argv = [
            "train.py",
            "--device",
            "cuda",
            "--n-envs",
            "4",
            "--vec-env",
            "subproc",
            "--active-workspace-codes",
            "PE001,PE002",
            "--seed",
            "41",
            "--monthly-jitter",
            "12",
            "--empirical-profile-probability",
            "0.35",
            "--state-context",
            "current",
            "--grid-size",
            "64",
            "--holdout-eval-freq",
            "25000",
            "--holdout-selection-count",
            "5",
            "--final-holdout-report",
            "--no-export-onnx",
        ]

        with patch.object(sys, "argv", argv), patch.object(train_module, "train", fake_train):
            train_module.main()

        self.assertEqual("cuda", captured["device"])
        self.assertEqual(4, captured["n_envs"])
        self.assertEqual("subproc", captured["vec_env"])
        self.assertEqual("PE001,PE002", captured["active_workspace_codes"])
        self.assertEqual(41, captured["seed"])
        self.assertEqual(12, captured["monthly_jitter"])
        self.assertEqual(0.35, captured["empirical_profile_probability"])
        self.assertEqual("current", captured["state_context"])
        self.assertEqual(64, captured["grid_size"])
        self.assertEqual(25_000, captured["holdout_eval_freq"])
        self.assertEqual(5, captured["holdout_selection_count"])
        self.assertTrue(captured["final_holdout_report"])

    def test_cli_defaults_match_ten_workspace_episode_shape(self):
        captured = {}

        def fake_train(args):
            captured["n_steps"] = args.n_steps
            captured["active_workspace_codes"] = args.active_workspace_codes
            captured["monthly_jitter"] = args.monthly_jitter
            captured["empirical_profile_probability"] = (
                args.empirical_profile_probability
            )
            captured["holdout_eval_freq"] = args.holdout_eval_freq
            captured["holdout_selection_count"] = (
                args.holdout_selection_count
            )
            captured["final_holdout_report"] = args.final_holdout_report

        with (
            patch.object(sys, "argv", ["train.py", "--no-export-onnx"]),
            patch.object(train_module, "train", fake_train),
        ):
            train_module.main()

        self.assertEqual(960, captured["n_steps"])
        self.assertEqual(
            train_module.DEFAULT_ACTIVE_WORKSPACE_CODES,
            captured["active_workspace_codes"],
        )
        self.assertEqual(20, captured["monthly_jitter"])
        self.assertEqual(0.2, captured["empirical_profile_probability"])
        self.assertEqual(50_000, captured["holdout_eval_freq"])
        self.assertEqual(5, captured["holdout_selection_count"])
        self.assertFalse(captured["final_holdout_report"])

    def test_holdout_selection_count_cli_rejects_values_other_than_five(self):
        with (
            patch.object(
                sys,
                "argv",
                ["train.py", "--holdout-selection-count", "4"],
            ),
            self.assertRaises(SystemExit),
        ):
            train_module.main()

    def test_holdout_callback_wiring_honors_disabled_and_enabled_modes(self):
        scenarios = [{"seed": seed} for seed in range(1000, 1020)]
        evaluate_fn = lambda policy_factory, selected: []

        self.assertIsNone(train_module.create_holdout_eval_callback(
            None,
            evaluate_fn,
            "./output",
            eval_freq=50_000,
            selection_count=5,
        ))
        self.assertIsNone(train_module.create_holdout_eval_callback(
            scenarios,
            evaluate_fn,
            "./output",
            eval_freq=0,
            selection_count=5,
        ))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            train_module.create_holdout_eval_callback(
                scenarios,
                evaluate_fn,
                "./output",
                eval_freq=-1,
                selection_count=5,
            )

        callback = train_module.create_holdout_eval_callback(
            scenarios,
            evaluate_fn,
            "./output",
            eval_freq=25_000,
            selection_count=5,
        )
        self.assertEqual("FixedHoldoutEvalCallback", type(callback).__name__)

    def test_auto_vec_env_selection_is_platform_aware(self):
        with patch.object(train_module.sys, "platform", "win32"):
            self.assertEqual(
                "dummy",
                train_module.resolve_vec_env_type("auto", n_envs=2),
            )

        with patch.object(train_module.sys, "platform", "linux"):
            self.assertEqual(
                "subproc",
                train_module.resolve_vec_env_type("auto", n_envs=2),
            )

        self.assertEqual(
            "single",
            train_module.resolve_vec_env_type("auto", n_envs=1),
        )

    def test_rollout_memory_estimate_scales_with_env_count(self):
        observation_space = gym.spaces.Dict({
            "first": gym.spaces.Box(0, 1, (2, 3), dtype=np.float32),
            "second": gym.spaces.Box(0, 1, (5,), dtype=np.float32),
        })
        single_env_mb = train_module.estimate_rollout_buffer_mb(
            observation_space,
            n_steps=8,
            n_envs=1,
        )
        four_env_mb = train_module.estimate_rollout_buffer_mb(
            observation_space,
            n_steps=8,
            n_envs=4,
        )

        self.assertAlmostEqual(single_env_mb * 4, four_env_mb)

    def test_run_config_records_schema3_observation_constants(self):
        scales = full_source_scales()

        config = train_module.current_run_config(
            make_args(), WORKSPACE_CODES, source_manifest(), scales
        )

        self.assertEqual(3, config["observation_schema_version"])
        self.assertEqual(64, config["grid_size"])
        self.assertEqual(16, config["ordered_future_count"])
        self.assertEqual(32, config["pending_queue_slots"])
        self.assertEqual(
            [[0, 5], [6, 20], [21, 60]],
            config["future_day_windows"],
        )
        self.assertEqual("full", config["state_context"])
        self.assertEqual(scales.to_dict(), config["observation_scales"])
        self.assertEqual(20260716, config["data_split_seed"])
        self.assertEqual("abc123", config["source_sha256"])
        self.assertEqual(913, config["episode_block_count"])
        self.assertEqual(
            source_manifest()["source_month_counts"],
            config["target_month_counts"],
        )

    def test_run_config_rejects_non_ten_workspace_order(self):
        with self.assertRaisesRegex(ValueError, "exactly 10"):
            train_module.current_run_config(
                make_args(),
                WORKSPACE_CODES[:-1],
                source_manifest(),
                full_source_scales(),
            )

    def test_rollout_estimate_counts_every_schema3_float(self):
        floats_per_observation = (
            8
            + 10 * 4 * 64 * 64
            + 10 * 4
            + 16 * 6
            + 16
            + 3 * 4
            + 10 * 32 * 7
            + 10 * 32
            + 10 * 4
        )
        expected_mb = floats_per_observation * 4 * 960 / 1024 / 1024

        actual_mb = train_module.estimate_rollout_buffer_mb(
            build_observation_space(), n_steps=960, n_envs=1
        )

        self.assertAlmostEqual(expected_mb, actual_mb)

    def test_production_factories_have_no_obsolete_future_parameter(self):
        import inspect

        for factory in (
            train_module.make_env,
            train_module.create_training_env,
            train_module.create_evaluation_env,
        ):
            with self.subTest(factory=factory.__name__):
                self.assertNotIn(
                    "n_future_blocks", inspect.signature(factory).parameters
                )

    def test_evaluation_factory_requires_ten_workspaces_and_grid_64(self):
        scales = full_source_scales()
        strategy = BaseGridStrategy(step=10.0)
        workspaces = [make_workspace(code) for code in WORKSPACE_CODES]

        with self.assertRaisesRegex(ValueError, "exactly 10 workspaces"):
            train_module.create_evaluation_env(
                [make_block("A001")],
                workspaces[:-1],
                strategy,
                observation_scales=scales,
            )
        with self.assertRaisesRegex(ValueError, "grid_size must be 64"):
            train_module.create_evaluation_env(
                [make_block("A001")],
                workspaces,
                strategy,
                observation_scales=scales,
                grid_size=32,
            )

    def test_training_and_worker_factories_reject_nonproduction_shapes(self):
        scales = full_source_scales()
        strategy = BaseGridStrategy(step=10.0)
        workspaces = [make_workspace(code) for code in WORKSPACE_CODES]
        common = {
            "blocks": [make_block("A001"), make_block("B001")],
            "workspaces": workspaces,
            "strategy": strategy,
            "observation_scales": scales,
        }

        with self.assertRaisesRegex(ValueError, "exactly 10 workspaces"):
            train_module.make_env(
                **{**common, "workspaces": workspaces[:-1]}
            )
        with self.assertRaisesRegex(ValueError, "grid_size must be 64"):
            train_module.make_env(**common, grid_size=32)
        with self.assertRaisesRegex(ValueError, "exactly 10 workspaces"):
            train_module.create_training_env(
                **{**common, "workspaces": workspaces[:-1]},
                generator=None,
            )
        with self.assertRaisesRegex(ValueError, "grid_size must be 64"):
            train_module.create_training_env(
                **common, generator=None, grid_size=32
            )

    def test_vector_worker_serialization_preserves_observation_scales(self):
        scales = full_source_scales()
        factory = train_module.make_env(
            blocks=[make_block("A001"), make_block("B001")],
            workspaces=[make_workspace(code) for code in WORKSPACE_CODES],
            strategy=BaseGridStrategy(step=10.0),
            observation_scales=scales,
            state_context_mode="full",
        )

        worker_env = cloudpickle.loads(cloudpickle.dumps(factory))()
        try:
            observation, _ = worker_env.reset(seed=9)
        finally:
            worker_env.close()

        self.assertEqual(
            scales.to_dict(), worker_env._observation_scales.to_dict()
        )
        self.assertTrue(worker_env.observation_space.contains(observation))

    def test_factories_reuse_the_supplied_observation_scales_object(self):
        scales = full_source_scales()
        strategy = BaseGridStrategy(step=10.0)
        workspaces = [make_workspace(code) for code in WORKSPACE_CODES]
        env = train_module.create_evaluation_env(
            [make_block("A001"), make_block("B001")],
            workspaces,
            strategy,
            observation_scales=scales,
            state_context_mode="current",
        )
        try:
            observation, _ = env.reset(seed=5)
        finally:
            env.close()

        self.assertIs(scales, env.unwrapped._observation_scales)
        self.assertEqual("current", env.unwrapped._state_context_mode)
        self.assertTrue(env.observation_space.contains(observation))

    def test_dummy_vector_env_exposes_action_masks(self):
        scales = full_source_scales()
        env = train_module.create_training_env(
            blocks=[make_block("A001"), make_block("B001")],
            workspaces=[make_workspace(code) for code in WORKSPACE_CODES],
            strategy=BaseGridStrategy(step=10.0),
            generator=None,
            observation_scales=scales,
            grid_size=GRID_SIZE,
            n_envs=2,
            vec_env="dummy",
            seed=5,
        )
        try:
            env.reset()
            masks = get_action_masks(env)
        finally:
            env.close()

        self.assertEqual((2, N_WORKSPACES), masks.shape)
        self.assertTrue(masks.all())


if __name__ == "__main__":
    unittest.main()
