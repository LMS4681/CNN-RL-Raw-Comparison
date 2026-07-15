import random
import unittest
from datetime import date

import numpy as np
import torch

from alloc_env.block import Block
from alloc_env.block_generator import SyntheticBlockGenerator
from alloc_env.data_loader import select_workspaces
from alloc_env.strategy import BaseGridStrategy
from alloc_env.workspace import Workspace
from train import create_evaluation_env, create_training_env, set_global_seed


def workspace(code: str) -> Workspace:
    return Workspace(
        code=code,
        origin_x=0.0,
        origin_y=0.0,
        length=500.0,
        breadth=500.0,
        strategy=BaseGridStrategy(step=10.0),
    )


def blocks() -> list[Block]:
    return [
        Block(
            name=f"B{index}",
            ship_no="T001",
            block_type="BUILD",
            length=10.0,
            breadth=10.0,
            height=5.0,
            weight=10.0,
            in_date=date(2026, 1, 5 + index),
            out_date=date(2026, 2, 20),
        )
        for index in range(3)
    ]


def make_seeded_training_env(seed: int):
    return create_training_env(
        blocks(),
        [workspace("PE001"), workspace("PE002")],
        BaseGridStrategy(step=10.0),
        SyntheticBlockGenerator.from_defaults(seed=999),
        grid_size=32,
        n_envs=1,
        vec_env="auto",
        n_future_blocks=4,
        seed=seed,
    )


class WorkspaceFilteringAndSeedTests(unittest.TestCase):
    def test_select_workspaces_preserves_source_order(self):
        source = [workspace("PE003"), workspace("PE001"), workspace("PE002")]

        selected = select_workspaces(source, ["PE002", "PE003"])

        self.assertEqual(["PE003", "PE002"], [ws.code for ws in selected])

    def test_select_workspaces_rejects_unknown_codes(self):
        with self.assertRaisesRegex(ValueError, "Unknown active workspace"):
            select_workspaces([workspace("PE001")], ["PE999"])

    def test_filtered_environment_has_filtered_shapes(self):
        selected = select_workspaces(
            [workspace("PE001"), workspace("PE002")],
            ["PE002"],
        )
        env = create_evaluation_env(
            blocks(),
            selected,
            BaseGridStrategy(step=10.0),
            grid_size=32,
            n_future_blocks=4,
            seed=11,
        )
        try:
            obs, _ = env.reset(seed=11)
        finally:
            env.close()

        self.assertEqual(1, env.action_space.n)
        self.assertEqual(1, obs["grids"].shape[0])
        self.assertEqual(1, obs["ws_meta"].shape[0])

    def test_same_seed_repeats_synthetic_initial_observation(self):
        first = make_seeded_training_env(seed=123)
        second = make_seeded_training_env(seed=123)
        try:
            obs_a = first.reset()[0]
            obs_b = second.reset()[0]
        finally:
            first.close()
            second.close()

        for key in obs_a:
            np.testing.assert_array_equal(obs_a[key], obs_b[key])

    def test_global_seed_repeats_python_numpy_and_torch(self):
        set_global_seed(17)
        first = (random.random(), np.random.random(), torch.rand(1).item())
        set_global_seed(17)
        second = (random.random(), np.random.random(), torch.rand(1).item())

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
