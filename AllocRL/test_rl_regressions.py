import unittest
from datetime import date

import numpy as np

from alloc_env.alloc_env import BlockPlacementEnv
from alloc_env.block import Block
from alloc_env.block_generator import SyntheticBlockGenerator
from alloc_env.data_loader import select_workspaces
from alloc_env.data_loader import apply_allowable_block_patterns
from alloc_env.constraints import (
    BlockPatternConstraint,
    DimensionConstraint,
    ValidWorkspacePicker,
)
from alloc_env.incremental_simulator import IncrementalPlacementSimulator
from alloc_env.observation_state import ObservationScales
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


def fixture_observation_scales() -> ObservationScales:
    return ObservationScales(
        max_length=200.0,
        max_breadth=200.0,
        max_duration=365,
        base_date=date(2026, 1, 1),
        date_span_workdays=365,
        max_workspace_area=40000.0,
        total_workspace_area=80000.0,
        max_workspace_length=200.0,
        max_workspace_breadth=200.0,
        dropout_threshold=7,
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
        # Schema 3: [length, breadth, placed-area ratio, placeability].
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

        self.assertEqual(obs["ws_meta"].shape, (1, 4))
        placeable = obs["ws_meta"][:, 3]
        # 이진 신호(0/1)여야 한다.
        self.assertTrue(bool(((placeable == 0.0) | (placeable == 1.0)).all()))
        # 빈 100x100 작업장에 10x10 블록은 즉시 배치 가능 → 1.
        self.assertEqual(obs["ws_meta"][0, 3], 1.0)

    def test_synthetic_env_uses_csv_like_spread_range(self):
        blocks = [
            make_block("A001", date(2026, 1, 5)),
            make_block("A002", date(2026, 7, 3)),
        ]
        generator = SyntheticBlockGenerator.from_defaults(seed=0)
        original_generate = generator.generate
        captured = {}

        def capture_generate(n_blocks, base_date, spread_days=90):
            captured["spread_days"] = spread_days
            return original_generate(n_blocks, base_date, spread_days=spread_days)

        generator.generate = capture_generate

        env = BlockPlacementEnv(
            blocks,
            [make_workspace()],
            BaseGridStrategy(step=10.0),
            observation_scales=fixture_observation_scales(),
            use_synthetic=True,
            generator=generator,
            synthetic_n_blocks=len(blocks),
            grid_size=32,
        )
        fixed_scales = env._observation_scales
        env.reset()

        csv_spread = (blocks[1].in_date - blocks[0].in_date).days
        expected = (
            max(30, int(round(csv_spread * 0.5))),
            max(30, int(round(csv_spread * 1.2))),
        )
        self.assertEqual(expected, captured["spread_days"])
        self.assertIs(fixed_scales, env._observation_scales)

    def test_filtered_workspaces_define_action_and_observation_shape(self):
        ws_a = make_workspace()
        ws_b = make_workspace()
        ws_b.code = "PE002"
        selected = select_workspaces([ws_a, ws_b], ["PE001"])
        env = BlockPlacementEnv(
            [make_block("A001", date(2026, 1, 5))],
            selected,
            BaseGridStrategy(step=10.0),
            observation_scales=fixture_observation_scales(),
            grid_size=32,
        )

        obs, _ = env.reset()

        self.assertEqual((1, 4, 32, 32), obs["grids"].shape)
        self.assertEqual((1, 4), obs["ws_meta"].shape)
        self.assertEqual([True], env.action_masks().tolist())
        self.assertGreater(float(obs["grids"][0].sum()), 0.0)

    def test_blocks_without_filtered_valid_workspace_are_not_presented(self):
        small = make_sized_workspace(10.0, 10.0)
        large = make_sized_workspace(100.0, 100.0)
        large.code = "PE002"
        blocks = [
            make_sized_block("TOO_BIG", date(2026, 1, 5), date(2026, 1, 20), 20.0, 20.0),
            make_sized_block("OK", date(2026, 1, 6), date(2026, 1, 20), 5.0, 5.0),
        ]
        env = BlockPlacementEnv(
            blocks,
            select_workspaces([small, large], ["PE001"]),
            BaseGridStrategy(step=5.0),
            grid_size=32,
        )

        env.reset()

        self.assertEqual(1, env._current_block_index)
        self.assertEqual([True], env.action_masks().tolist())

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
            observation_scales=fixture_observation_scales(),
            grid_size=32,
        )

        env.reset()
        render_calls = []
        original_render_base = env._renderer.render_base

        def counting_render_base(ws, env_date):
            render_calls.append(ws.code)
            return original_render_base(ws, env_date)

        env._renderer.render_base = counting_render_base

        obs, _, terminated, _, _ = env.step(0)

        self.assertFalse(terminated)
        self.assertEqual(date(2026, 1, 5), env._env_date)
        self.assertEqual(["PE001"], render_calls)
        expected_base_grids = np.stack([
            original_render_base(ws, env._env_date) for ws in env._workspaces
        ], axis=0)
        np.testing.assert_array_equal(obs["grids"][:, :2], expected_base_grids)
        self.assertGreater(float(obs["grids"][:, 2].sum()), 0.0)
        self.assertGreater(float(obs["grids"][:, 3].sum()), 0.0)

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
            observation_scales=fixture_observation_scales(),
            grid_size=32,
        )

        env.reset()

        self.assertEqual([True, False], env.action_masks().tolist())

    def test_evaluation_env_uses_original_blocks(self):
        blocks = [
            make_block("CSV-001", date(2026, 1, 5)),
            make_block("CSV-002", date(2026, 1, 6)),
        ]
        workspaces = [make_sized_workspace(100.0, 80.0) for _ in range(10)]
        for index, workspace in enumerate(workspaces):
            workspace.code = f"PE{index + 1:03d}"

        env = create_evaluation_env(
            blocks,
            workspaces,
            BaseGridStrategy(step=10.0),
            observation_scales=fixture_observation_scales(),
            grid_size=64,
        )
        try:
            env.reset()

            self.assertFalse(env.unwrapped._use_synthetic)
            self.assertEqual("CSV-001", env.unwrapped._blocks[0].name)
        finally:
            env.close()

    def test_resolved_rewards_conserve_terminal_score(self):
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
        _, first_reward, terminated, _, first_info = env.step(0)

        self.assertFalse(terminated)
        self.assertEqual([0], first_info["newly_resolved_indices"])
        self.assertAlmostEqual(0.5, first_reward)

        _, final_reward, terminated, _, final_info = env.step(0)

        self.assertTrue(terminated)
        self.assertEqual([1], final_info["newly_resolved_indices"])
        self.assertAlmostEqual(
            final_info["terminal_score"], first_reward + final_reward
        )

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
            observation_scales=fixture_observation_scales(),
            grid_size=32,
        )

        env.reset()
        _, _, terminated, _, info = env.step(0)

        self.assertTrue(terminated)
        self.assertEqual(1.0, info["terminal_reward"])
        self.assertEqual(0, info["raw_result"].delay_days[0])


if __name__ == "__main__":
    unittest.main()
