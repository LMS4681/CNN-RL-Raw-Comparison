"""Regression tests for GPU/device and vectorized training configuration."""

import sys
import unittest
from datetime import date
from unittest.mock import patch

from sb3_contrib.common.maskable.utils import get_action_masks

import train as train_module
from alloc_env.block import Block
from alloc_env.cnn_extractor import (
    CandidateCnnExtractor,
    FixedGridExtractor,
    StructuredExtractor,
)
from alloc_env.strategy import BaseGridStrategy
from alloc_env.workspace import Workspace


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
        breadth=100.0,
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

    def test_cli_defaults_match_ten_workspace_episode_shape(self):
        captured = {}

        def fake_train(args):
            captured["n_steps"] = args.n_steps
            captured["active_workspace_codes"] = args.active_workspace_codes
            captured["monthly_jitter"] = args.monthly_jitter
            captured["empirical_profile_probability"] = (
                args.empirical_profile_probability
            )

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
        single_env_mb = train_module.estimate_rollout_buffer_mb(
            n_workspaces=2,
            grid_size=16,
            n_steps=8,
            n_envs=1,
        )
        four_env_mb = train_module.estimate_rollout_buffer_mb(
            n_workspaces=2,
            grid_size=16,
            n_steps=8,
            n_envs=4,
        )

        self.assertAlmostEqual(single_env_mb * 4, four_env_mb)

    def test_dummy_vector_env_exposes_action_masks(self):
        env = train_module.create_training_env(
            blocks=[make_block("A001"), make_block("B001")],
            workspaces=[make_workspace("PE001"), make_workspace("PE002")],
            strategy=BaseGridStrategy(step=10.0),
            generator=None,
            grid_size=16,
            n_envs=2,
            vec_env="dummy",
            seed=5,
        )
        try:
            env.reset()
            masks = get_action_masks(env)
        finally:
            env.close()

        self.assertEqual((2, 2), masks.shape)
        self.assertTrue(masks.all())


if __name__ == "__main__":
    unittest.main()
