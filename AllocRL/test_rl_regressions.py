import unittest
from datetime import date
from unittest.mock import patch

import numpy as np

import alloc_env.alloc_env as alloc_env_module
from alloc_env.alloc_env import BlockPlacementEnv
from alloc_env.block import Block
from alloc_env.data_loader import apply_allowable_block_patterns
from alloc_env.constraints import (
    BlockPatternConstraint,
    DimensionConstraint,
    ValidWorkspacePicker,
)
from alloc_env.incremental_simulator import IncrementalPlacementSimulator
from alloc_env.simulator import PlacementSimulator, SimulationResult
from alloc_env.strategy import BaseGridStrategy
from alloc_env.workspace import Workspace
from train import create_evaluation_env


def make_block(name: str, in_date: date) -> Block:
    return Block(
        name=name,
        ship_no="T001",
        block_type="BUILD",
        length=10.0,
        breadth=10.0,
        height=5.0,
        weight=10.0,
        in_date=in_date,
        out_date=date(2026, 1, 30),
    )


def make_sized_block(
    name: str,
    in_date: date,
    out_date: date,
    length: float,
    breadth: float,
) -> Block:
    return Block(
        name=name,
        ship_no="T001",
        block_type="BUILD",
        length=length,
        breadth=breadth,
        height=5.0,
        weight=10.0,
        in_date=in_date,
        out_date=out_date,
    )


def make_workspace() -> Workspace:
    return Workspace(
        code="PE001",
        origin_x=0.0,
        origin_y=0.0,
        breadth=100.0,
        length=100.0,
        strategy=BaseGridStrategy(step=10.0),
    )


def make_sized_workspace(length: float, breadth: float) -> Workspace:
    return Workspace(
        code="PE001",
        origin_x=0.0,
        origin_y=0.0,
        breadth=breadth,
        length=length,
        strategy=BaseGridStrategy(step=10.0),
    )


class RlRegressionTests(unittest.TestCase):
    def test_incremental_simulator_matches_batch_replay(self):
        blocks = [
            make_block("A001", date(2026, 1, 5)),
            make_block("A002", date(2026, 1, 6)),
            make_block("A003", date(2026, 1, 7)),
        ]
        workspaces = [make_workspace()]
        assignments = [0, 0, 0]
        batch = PlacementSimulator().replay(
            blocks, workspaces, assignments, dropout_threshold=7
        )
        incremental = IncrementalPlacementSimulator(
            blocks, workspaces, dropout_threshold=7
        )

        while not incremental.is_done:
            idx = incremental.current_block_index
            incremental.assign_current(assignments[idx])

        result = incremental.result()

        self.assertEqual(batch.delay_days, result.delay_days)
        self.assertEqual(
            [b.workspace_code for b in batch.blocks],
            [b.workspace_code for b in result.blocks],
        )

    def test_infeasible_block_is_auto_dropped_and_never_presented(self):
        """어느 작업장에도 배치 불가한 블록은 agent에 제시되지 않고 즉시 탈락.

        이 가드가 없으면 action_masks()가 전부 False가 되어 MaskablePPO가
        불안정해진다(전-마스킹). infeasible 블록은 도착일에 DROPOUT 처리되며,
        agent에게 제시되는 모든 블록은 유효 작업장이 최소 1개 있어야 한다.
        """
        # 유일한 작업장은 30x30, OVERSIZE 블록은 50x50 → 회전해도 배치 불가.
        workspaces = [make_sized_workspace(30.0, 30.0)]
        blocks = [
            make_sized_block("OK1", date(2026, 1, 5), date(2026, 1, 30), 10.0, 10.0),
            make_sized_block("OVERSIZE", date(2026, 1, 6), date(2026, 1, 30), 50.0, 50.0),
            make_sized_block("OK2", date(2026, 1, 7), date(2026, 1, 30), 10.0, 10.0),
        ]
        picker = ValidWorkspacePicker(
            blocks, workspaces,
            [DimensionConstraint(), BlockPatternConstraint()],
        )
        infeasible = set(picker.get_infeasible_blocks())
        self.assertEqual(infeasible, {1})

        assignments = [0, 0, 0]
        sim = IncrementalPlacementSimulator(
            blocks, workspaces, dropout_threshold=7,
            infeasible_indices=infeasible,
        )
        presented = []
        while not sim.is_done:
            idx = sim.current_block_index
            presented.append(idx)
            # 제시되는 모든 블록은 유효 작업장이 최소 1개 있어야 한다.
            self.assertTrue(any(picker.get_action_mask(idx, len(workspaces))))
            sim.assign_current(assignments[idx])

        # 배치 불가 블록은 결코 결정 지점으로 제시되지 않는다.
        self.assertNotIn(1, presented)
        self.assertEqual(presented, [0, 2])

        result = sim.result()
        self.assertEqual(result.delay_days[1], SimulationResult.DROPOUT)

        # incremental과 batch replay가 동일 assignment에서 같은 결과(둘 다 DROPOUT).
        batch = PlacementSimulator().replay(
            blocks, workspaces, assignments, dropout_threshold=7
        )
        self.assertEqual(batch.delay_days, result.delay_days)

    def test_env_requests_blocks_in_simulator_due_order(self):
        later = make_sized_block(
            "LATER",
            date(2026, 1, 10),
            date(2026, 1, 20),
            length=20.0,
            breadth=10.0,
        )
        earlier = make_sized_block(
            "EARLIER",
            date(2026, 1, 5),
            date(2026, 1, 20),
            length=10.0,
            breadth=10.0,
        )
        env = BlockPlacementEnv(
            [later, earlier],
            [make_workspace()],
            BaseGridStrategy(step=10.0),
            grid_size=32,
        )

        obs, _ = env.reset()

        self.assertEqual(1, env._current_block_index)
        self.assertEqual(0.5, obs["block"][0])

    def test_ws_meta_exposes_placeability_column(self):
        # B2: ws_meta는 (N, 3) = [scale, occupancy_ratio, placeable_now].
        blocks = [
            make_block("A001", date(2026, 1, 5)),
            make_block("A002", date(2026, 1, 6)),
        ]
        env = BlockPlacementEnv(
            blocks,
            [make_workspace()],
            BaseGridStrategy(step=10.0),
            grid_size=32,
        )
        obs, _ = env.reset()

        self.assertEqual(obs["ws_meta"].shape, (1, 3))
        placeable = obs["ws_meta"][:, 2]
        # 이진 신호(0/1)여야 한다.
        self.assertTrue(bool(((placeable == 0.0) | (placeable == 1.0)).all()))
        # 빈 100x100 작업장에 10x10 블록은 즉시 배치 가능 → 1.
        self.assertEqual(obs["ws_meta"][0, 2], 1.0)

    def test_step_updates_simulator_workspace_and_cnn_grid(self):
        blocks = [
            make_block("A001", date(2026, 1, 5)),
            make_block("A002", date(2026, 1, 6)),
        ]
        env = BlockPlacementEnv(
            blocks,
            [make_workspace()],
            BaseGridStrategy(step=10.0),
            grid_size=32,
        )

        obs, _ = env.reset()
        before = obs["grids"][0, 0].sum()
        obs, _, terminated, _, _ = env.step(0)

        self.assertFalse(terminated)
        self.assertEqual(1, len(env._workspaces[0].blocks))
        self.assertGreater(obs["grids"][0, 0].sum(), before)

    def test_step_rerenders_only_changed_workspace_when_date_is_unchanged(self):
        blocks = [
            make_block("A001", date(2026, 1, 5)),
            make_block("A002", date(2026, 1, 5)),
        ]
        ws_a = make_workspace()
        ws_b = make_workspace()
        ws_b.code = "PE002"
        env = BlockPlacementEnv(
            blocks,
            [ws_a, ws_b],
            BaseGridStrategy(step=10.0),
            grid_size=32,
        )

        env.reset()
        render_calls = []
        original_render = env._renderer.render

        def counting_render(ws, env_date, max_remaining_days=60):
            render_calls.append(ws.code)
            return original_render(ws, env_date, max_remaining_days)

        env._renderer.render = counting_render

        obs, _, terminated, _, _ = env.step(0)

        self.assertFalse(terminated)
        self.assertEqual(date(2026, 1, 5), env._env_date)
        self.assertEqual(["PE001"], render_calls)
        expected_grids = np.stack([
            original_render(ws, env._env_date) for ws in env._workspaces
        ], axis=0)
        np.testing.assert_array_equal(obs["grids"], expected_grids)

    def test_step_moves_observation_date_to_next_block(self):
        blocks = [
            make_block("A001", date(2026, 1, 5)),
            make_block("A002", date(2026, 1, 12)),
        ]
        env = BlockPlacementEnv(
            blocks,
            [make_workspace()],
            BaseGridStrategy(step=10.0),
            grid_size=32,
        )

        env.reset()
        self.assertEqual(date(2026, 1, 5), env._env_date)

        env.step(0)

        self.assertEqual(date(2026, 1, 12), env._env_date)

    def test_reset_discards_simulator_blocks(self):
        blocks = [
            make_block("A001", date(2026, 1, 5)),
            make_block("A002", date(2026, 1, 6)),
        ]
        env = BlockPlacementEnv(
            blocks,
            [make_workspace()],
            BaseGridStrategy(step=10.0),
            grid_size=32,
        )

        env.reset()
        env.step(0)
        self.assertEqual(1, len(env._workspaces[0].blocks))

        env.reset()

        self.assertEqual(0, len(env._workspaces[0].blocks))

    def test_apply_allowable_block_patterns_sets_workspace_rules(self):
        workspaces = [make_workspace()]

        apply_allowable_block_patterns(workspaces, {"PE001": ["A*", "B*"]})

        self.assertEqual(["A*", "B*"], workspaces[0].allowable_block_patterns)

    def test_action_mask_honors_workspace_block_patterns(self):
        ws_a = make_workspace()
        ws_b = make_workspace()
        ws_b.code = "PE002"
        apply_allowable_block_patterns(
            [ws_a, ws_b],
            {"PE001": ["A*"], "PE002": ["B*"]},
        )
        env = BlockPlacementEnv(
            [make_block("A001", date(2026, 1, 5))],
            [ws_a, ws_b],
            BaseGridStrategy(step=10.0),
            grid_size=32,
        )

        env.reset()

        self.assertEqual([True, False], env.action_masks().tolist())

    def test_evaluation_env_uses_original_blocks(self):
        blocks = [
            make_block("CSV-001", date(2026, 1, 5)),
            make_block("CSV-002", date(2026, 1, 6)),
        ]

        env = create_evaluation_env(
            blocks,
            [make_workspace()],
            BaseGridStrategy(step=10.0),
            grid_size=32,
        )
        env.reset()

        self.assertFalse(env.unwrapped._use_synthetic)
        self.assertEqual("CSV-001", env.unwrapped._blocks[0].name)

    def test_partial_replay_shaping_adds_intermediate_reward(self):
        blocks = [
            make_block("A001", date(2026, 1, 5)),
            make_block("A002", date(2026, 1, 6)),
        ]
        env = BlockPlacementEnv(
            blocks,
            [make_workspace()],
            BaseGridStrategy(step=10.0),
            grid_size=32,
        )

        with patch.object(
            alloc_env_module,
            "PARTIAL_REPLAY_INTERVAL",
            1,
            create=True,
        ):
            env.reset()
            _, reward, terminated, _, _ = env.step(0)

        self.assertFalse(terminated)
        self.assertGreater(reward, 0.05)

    def test_terminal_reward_uses_incremental_simulator_state(self):
        block = make_sized_block(
            "A001",
            date(2026, 1, 5),
            date(2026, 1, 20),
            length=10.0,
            breadth=10.0,
        )
        env = BlockPlacementEnv(
            [block],
            [make_sized_workspace(length=10.0, breadth=10.0)],
            BaseGridStrategy(step=10.0),
            grid_size=32,
        )

        env.reset()
        _, _, terminated, _, info = env.step(0)

        self.assertTrue(terminated)
        self.assertEqual(1.0, info["terminal_reward"])
        self.assertEqual(0, info["raw_result"].delay_days[0])


if __name__ == "__main__":
    unittest.main()
