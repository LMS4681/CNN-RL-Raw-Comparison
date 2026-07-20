import random
import unittest
from datetime import date

import numpy as np
import torch

from alloc_env.block import Block
from alloc_env.block_generator import SyntheticBlockGenerator
from alloc_env.data_loader import (
    select_workspaces,
    select_workspaces_in_order,
)
from alloc_env.observation_state import ObservationScales
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


def ten_workspaces() -> list[Workspace]:
    return [workspace(f"PE{index:03d}") for index in range(1, 11)]


def observation_scales() -> ObservationScales:
    return ObservationScales(
        max_length=500.0,
        max_breadth=500.0,
        max_duration=365,
        base_date=date(2026, 1, 1),
        date_span_workdays=365,
        max_workspace_area=250_000.0,
        total_workspace_area=2_500_000.0,
        max_workspace_length=500.0,
        max_workspace_breadth=500.0,
        dropout_threshold=7,
    )


def make_seeded_training_env(seed: int):
    return create_training_env(
        blocks(),
        ten_workspaces(),
        BaseGridStrategy(step=10.0),
        SyntheticBlockGenerator.from_defaults(seed=999),
        observation_scales=observation_scales(),
        grid_size=64,
        n_envs=1,
        vec_env="auto",
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

    def test_select_workspaces_in_order_restores_recorded_action_order(self):
        source = [workspace("PE003"), workspace("PE001"), workspace("PE002")]

        selected = select_workspaces_in_order(
            source, ["pe002", "PE003"]
        )

        self.assertEqual(["PE002", "PE003"], [ws.code for ws in selected])

    def test_production_environment_rejects_filtered_workspace_count(self):
        selected = select_workspaces(
            [workspace("PE001"), workspace("PE002")],
            ["PE002"],
        )
        with self.assertRaisesRegex(ValueError, "exactly 10 workspaces"):
            create_evaluation_env(
                blocks(),
                selected,
                BaseGridStrategy(step=10.0),
                observation_scales=observation_scales(),
                seed=11,
            )

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
