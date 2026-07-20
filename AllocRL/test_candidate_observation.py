import unittest
from datetime import date

import numpy as np
import pytest

from alloc_env.alloc_env import BlockPlacementEnv
from alloc_env.block import SAFETY_DISTANCE, Block, PrePlacedBlock
from alloc_env.incremental_simulator import IncrementalPlacementSimulator
from alloc_env.observation_state import ObservationScales
from alloc_env.occupancy_grid import CandidatePlacement, OccupancyGridRenderer
from alloc_env.strategy import BaseGridStrategy
from alloc_env.workspace import LotRegion, Workspace


def fixture_scales(workspaces) -> ObservationScales:
    areas = [workspace.length * workspace.breadth for workspace in workspaces]
    return ObservationScales(
        max_length=20.0,
        max_breadth=10.0,
        max_duration=30,
        base_date=date(2026, 1, 5),
        date_span_workdays=10,
        max_workspace_area=max(areas),
        total_workspace_area=sum(areas),
        max_workspace_length=max(workspace.length for workspace in workspaces),
        max_workspace_breadth=max(workspace.breadth for workspace in workspaces),
        dropout_threshold=7,
    )


class StatefulPlacementStrategy(BaseGridStrategy):
    def __init__(self, responses):
        super().__init__(step=1.0)
        self.responses = list(responses)
        self.calls = []

    def determine_position(self, workspace, block, env_date):
        del workspace, env_date
        self.calls.append(block.name)
        return self.responses[len(self.calls) - 1]


def make_env(
    *,
    block_length: float,
    block_breadth: float,
    workspace_length: float = 100.0,
    workspace_breadth: float = 100.0,
    fill_workspace_with_preplacement: bool = False,
) -> BlockPlacementEnv:
    strategy = BaseGridStrategy(step=5.0)
    workspace = Workspace(
        code="PE001",
        origin_x=0.0,
        origin_y=0.0,
        length=workspace_length,
        breadth=workspace_breadth,
        strategy=strategy,
    )
    if fill_workspace_with_preplacement:
        workspace.add_pre_placement(
            PrePlacedBlock(
                label="FULL",
                pos_x=workspace_length / 2.0,
                pos_y=workspace_breadth / 2.0,
                length=workspace_length,
                breadth=workspace_breadth,
                start_date=date(2026, 1, 1),
                end_date=date(2026, 2, 28),
            )
        )
    block = Block(
        name="A",
        ship_no="T001",
        block_type="BUILD",
        length=block_length,
        breadth=block_breadth,
        height=5.0,
        weight=10.0,
        in_date=date(2026, 1, 5),
        out_date=date(2026, 1, 30),
    )
    return BlockPlacementEnv(
        [block],
        [workspace],
        strategy,
        grid_size=32,
        observation_scales=fixture_scales([workspace]),
    )


def decision_block(name: str, day: int = 5) -> Block:
    return Block(
        name=name,
        ship_no="T001",
        block_type="BUILD",
        length=4.0,
        breadth=2.0,
        height=5.0,
        weight=10.0,
        in_date=date(2026, 1, day),
        out_date=date(2026, 1, 30),
    )


def stateful_workspace(strategy) -> Workspace:
    return Workspace(
        code="STATEFUL",
        origin_x=0.0,
        origin_y=0.0,
        length=100.0,
        breadth=100.0,
        strategy=strategy,
    )


def test_simulator_omitted_override_keeps_direct_position_lookup():
    strategy = StatefulPlacementStrategy([(10.0, 5.0)])
    workspace = stateful_workspace(strategy)
    simulator = IncrementalPlacementSimulator(
        [decision_block("CURRENT")], [workspace], dropout_threshold=7
    )

    result = simulator.assign_current(0)

    assert result.placed
    assert simulator.workspaces[0].strategy.calls == ["CURRENT"]
    assert (simulator.blocks[0].ref_x, simulator.blocks[0].ref_y) == (10.0, 5.0)


def test_simulator_tuple_override_uses_exact_center_without_strategy_call():
    strategy = StatefulPlacementStrategy([(90.0, 90.0)])
    workspace = stateful_workspace(strategy)
    simulator = IncrementalPlacementSimulator(
        [decision_block("CURRENT")], [workspace], dropout_threshold=7
    )

    result = simulator.assign_current(0, placement_override=(10.0, 5.0))

    assert result.placed
    assert simulator.workspaces[0].strategy.calls == []
    assert (simulator.blocks[0].ref_x, simulator.blocks[0].ref_y) == (10.0, 5.0)


def test_simulator_none_override_is_exact_initial_miss_without_strategy_call():
    strategy = StatefulPlacementStrategy([(90.0, 90.0)])
    workspace = stateful_workspace(strategy)
    simulator = IncrementalPlacementSimulator(
        [decision_block("FIRST"), decision_block("SECOND")],
        [workspace],
        dropout_threshold=7,
    )

    result = simulator.assign_current(0, placement_override=None)

    assert result.delayed
    assert simulator.workspaces[0].strategy.calls == []
    assert simulator.current_block_index == 1
    assert 0 in simulator.pending
    assert len(simulator.workspaces[0].blocks) == 0


def test_simulator_clears_none_override_before_delayed_retry():
    strategy = StatefulPlacementStrategy([(20.0, 5.0)])
    workspace = stateful_workspace(strategy)
    simulator = IncrementalPlacementSimulator(
        [decision_block("FIRST"), decision_block("SECOND")],
        [workspace],
        dropout_threshold=7,
    )

    simulator.assign_current(0, placement_override=None)
    simulator.assign_current(0, placement_override=(10.0, 5.0))

    assert simulator.is_done
    assert simulator.workspaces[0].strategy.calls == ["FIRST"]
    assert (simulator.blocks[0].ref_x, simulator.blocks[0].ref_y) == (20.0, 5.0)


@pytest.mark.parametrize(
    "placement_override",
    [(1.0,), (1.0, 2.0, 3.0), (float("nan"), 2.0), (1.0, float("inf"))],
)
def test_simulator_rejects_invalid_placement_override(placement_override):
    simulator = IncrementalPlacementSimulator(
        [decision_block("CURRENT")],
        [stateful_workspace(StatefulPlacementStrategy([(10.0, 5.0)]))],
        dropout_threshold=7,
    )

    with pytest.raises(ValueError, match="placement_override"):
        simulator.assign_current(0, placement_override=placement_override)

    assert simulator.assignments == [None]


def test_environment_consumes_decision_time_candidate_without_second_query():
    strategy = StatefulPlacementStrategy([(10.0, 5.0), (90.0, 90.0)])
    workspace = stateful_workspace(strategy)
    env = BlockPlacementEnv(
        [decision_block("CURRENT")],
        [workspace],
        strategy,
        grid_size=32,
        observation_scales=fixture_scales([workspace]),
    )
    env.reset(seed=0)
    decision_strategy = env._workspaces[0].strategy
    candidate_position = env._candidate_placements[0].position

    env.step(0)

    placed = env._placement_simulator.blocks[0]
    assert decision_strategy.calls == ["CURRENT"]
    assert candidate_position == (10.0, 5.0)
    assert (placed.ref_x, placed.ref_y) == candidate_position


def test_environment_consumes_observed_none_as_initial_miss():
    strategy = StatefulPlacementStrategy([None, (90.0, 90.0), (20.0, 5.0)])
    workspace = stateful_workspace(strategy)
    blocks = [decision_block("FIRST"), decision_block("SECOND")]
    env = BlockPlacementEnv(
        blocks,
        [workspace],
        strategy,
        grid_size=32,
        observation_scales=fixture_scales([workspace]),
    )
    env.reset(seed=0)
    decision_strategy = env._workspaces[0].strategy
    assert env._candidate_placements[0].position is None

    _observation, _reward, terminated, _truncated, _info = env.step(0)

    assert not terminated
    assert decision_strategy.calls == ["FIRST", "SECOND"]
    assert env._placement_simulator.blocks[0].workspace_code is None
    assert 0 in env._placement_simulator.pending
    assert len(env._workspaces[0].blocks) == 0


class CandidateObservationTests(unittest.TestCase):
    def make_lot_workspace(self) -> Workspace:
        workspace = Workspace(
            code="LOTS",
            origin_x=0.0,
            origin_y=0.0,
            length=20.0,
            breadth=10.0,
            strategy=BaseGridStrategy(step=1.0),
        )
        workspace.add_lot(LotRegion("LEFT", 0.0, 0.0, 10.0, 10.0))
        workspace.add_lot(LotRegion("RIGHT", 10.0, 0.0, 10.0, 10.0))
        return workspace

    def make_current_block(self) -> Block:
        return Block(
            name="CURRENT",
            ship_no="T001",
            block_type="BUILD",
            length=4.0,
            breadth=2.0,
            height=5.0,
            weight=10.0,
            in_date=date(2026, 1, 5),
            out_date=date(2026, 1, 30),
        )

    def test_plain_workspace_lot_context_is_available_everywhere(self):
        workspace = Workspace(
            code="PLAIN",
            origin_x=0.0,
            origin_y=0.0,
            length=20.0,
            breadth=10.0,
            strategy=BaseGridStrategy(step=1.0),
        )

        context = OccupancyGridRenderer(64).render_candidate_context(
            workspace,
            CandidatePlacement(None, 4.0, 2.0),
            self.make_current_block(),
            date(2026, 1, 5),
        )

        self.assertEqual((2, 64, 64), context.shape)
        np.testing.assert_array_equal(context[0], np.full((64, 64), 0.25))
        self.assertEqual(0.0, float(context[1].sum()))

    def test_lot_context_distinguishes_unavailable_and_available_lots(self):
        workspace = self.make_lot_workspace()
        existing = self.make_current_block()
        existing.move(5.0, 5.0)
        workspace.add_block(existing, date(2026, 1, 5))

        context = OccupancyGridRenderer(64).render_candidate_context(
            workspace,
            CandidatePlacement(None, 4.0, 2.0),
            self.make_current_block(),
            date(2026, 1, 5),
        )

        self.assertEqual(1.0, float(context[0, 32, 16]))
        self.assertEqual(0.25, float(context[0, 32, 48]))

    def test_placeable_candidate_updates_lot_and_renders_safety_footprint(self):
        workspace = self.make_lot_workspace()
        current = self.make_current_block()
        current.move(2.0, 3.0)
        original_block_state = (
            current.ref_x,
            current.ref_y,
            current.workspace_code,
        )
        renderer = OccupancyGridRenderer(64)

        current_context = renderer.render_candidate_context(
            workspace,
            CandidatePlacement(None, current.length, current.breadth),
            current,
            date(2026, 1, 5),
        )
        post_context = renderer.render_candidate_context(
            workspace,
            CandidatePlacement((15.0, 5.0), current.length, current.breadth),
            current,
            date(2026, 1, 5),
        )

        self.assertEqual(0.25, float(current_context[0, 32, 48]))
        self.assertEqual(1.0, float(post_context[0, 32, 48]))
        ys, xs = np.nonzero(post_context[1])
        self.assertEqual(SAFETY_DISTANCE, 1.0)
        self.assertEqual((38, 57, 19, 44), (xs.min(), xs.max(), ys.min(), ys.max()))
        self.assertEqual(0, len(workspace.blocks))
        self.assertEqual(
            original_block_state,
            (current.ref_x, current.ref_y, current.workspace_code),
        )

    def test_candidate_lot_state_uses_current_delayed_dates(self):
        workspace = self.make_lot_workspace()
        workspace.add_pre_placement(
            PrePlacedBlock(
                label="TEMP",
                pos_x=5.0,
                pos_y=5.0,
                length=4.0,
                breadth=2.0,
                start_date=date(2026, 1, 10),
                end_date=date(2026, 1, 15),
            )
        )
        current = self.make_current_block()
        current.in_date = date(2026, 1, 20)
        current.out_date = date(2026, 1, 21)
        renderer = OccupancyGridRenderer(64)

        delayed = renderer.render_candidate_context(
            workspace,
            CandidatePlacement(None, current.length, current.breadth),
            current,
            date(2026, 1, 20),
        )
        current.in_date = date(2026, 1, 12)
        current.out_date = date(2026, 1, 13)
        overlapping = renderer.render_candidate_context(
            workspace,
            CandidatePlacement(None, current.length, current.breadth),
            current,
            date(2026, 1, 12),
        )

        self.assertEqual(0.25, float(delayed[0, 32, 16]))
        self.assertEqual(1.0, float(overlapping[0, 32, 16]))

    def test_overlapping_lot_pixels_keep_strongest_unavailable_value(self):
        workspace = Workspace(
            code="OVERLAP",
            origin_x=0.0,
            origin_y=0.0,
            length=20.0,
            breadth=10.0,
            strategy=BaseGridStrategy(step=1.0),
        )
        workspace.add_lot(LotRegion("OCCUPIED", 0.0, 0.0, 10.0, 10.0))
        workspace.add_lot(LotRegion("AVAILABLE", 9.0, 0.0, 10.0, 10.0))
        existing = self.make_current_block()
        existing.move(3.0, 5.0)
        workspace.add_block(existing, date(2026, 1, 5))

        context = OccupancyGridRenderer(64).render_candidate_context(
            workspace,
            CandidatePlacement(None, 4.0, 2.0),
            self.make_current_block(),
            date(2026, 1, 5),
        )

        self.assertEqual(1.0, float(context[0, 32, 30]))

    def test_candidate_channel_marks_strategy_position(self):
        env = make_env(block_length=20.0, block_breadth=10.0)
        obs, _ = env.reset(seed=3)
        candidate = env._candidate_placements[0]

        self.assertTrue(candidate.placeable)
        self.assertEqual((10.0, 5.0), candidate.position)
        self.assertEqual((1, 32, 32), obs["grids"][0, 3:4].shape)
        self.assertGreater(float(obs["grids"][0, 3].sum()), 0.0)

    def test_candidate_position_matches_the_applied_placement(self):
        env = make_env(block_length=20.0, block_breadth=10.0)
        env.reset(seed=3)
        candidate_position = env._candidate_placements[0].position

        env.step(0)
        placed = env._placement_simulator.blocks[0]

        self.assertEqual(candidate_position, (placed.ref_x, placed.ref_y))

    def test_observation_does_not_mutate_current_block(self):
        env = make_env(block_length=20.0, block_breadth=10.0)
        env.reset(seed=3)
        current = env._placement_simulator.current_block
        before = (
            current.length,
            current.breadth,
            current.ref_x,
            current.ref_y,
            current.angle,
        )

        env._get_obs()

        after = (
            current.length,
            current.breadth,
            current.ref_x,
            current.ref_y,
            current.angle,
        )
        self.assertEqual(before, after)

    def test_unplaceable_candidate_channel_is_zero(self):
        env = make_env(
            block_length=10.0,
            block_breadth=10.0,
            fill_workspace_with_preplacement=True,
        )
        obs, _ = env.reset(seed=3)

        self.assertEqual(0.0, float(obs["grids"][0, 3].sum()))
        self.assertEqual(0.0, float(obs["ws_meta"][0, 3]))

    def test_candidate_contract_has_only_original_dimensions(self):
        env = make_env(block_length=20.0, block_breadth=10.0)
        obs, _ = env.reset(seed=3)
        candidate = env._candidate_placements[0]
        mask = obs["grids"][0, 3]
        rows, columns = np.where(mask > 0.0)

        self.assertFalse(hasattr(candidate, "rotated"))
        self.assertEqual((20.0, 10.0), (candidate.length, candidate.breadth))
        self.assertGreater(np.unique(columns).size, np.unique(rows).size)

    def test_action_preview_does_not_turn_current_block(self):
        strategy = BaseGridStrategy(step=1.0)
        workspace = Workspace(
            code="NARROW_GAP",
            origin_x=0.0,
            origin_y=0.0,
            length=10.0,
            breadth=10.0,
            strategy=strategy,
        )
        workspace.add_pre_placement(
            PrePlacedBlock(
                label="RIGHT_STRIP",
                pos_x=8.0,
                pos_y=5.0,
                length=4.0,
                breadth=10.0,
                start_date=date(2026, 1, 1),
                end_date=date(2026, 2, 28),
            )
        )
        current = Block(
            name="CURRENT",
            ship_no="T001",
            block_type="BUILD",
            length=8.0,
            breadth=4.0,
            height=5.0,
            weight=10.0,
            in_date=date(2026, 1, 5),
            out_date=date(2026, 1, 30),
        )
        future = Block(
            name="FUTURE",
            ship_no="T001",
            block_type="BUILD",
            length=4.0,
            breadth=8.0,
            height=5.0,
            weight=10.0,
            in_date=date(2026, 1, 6),
            out_date=date(2026, 1, 30),
        )
        env = BlockPlacementEnv([current, future], [workspace], strategy)
        env.reset(seed=0)
        future_indices = env.future_workspace_choice_indices()
        current_state = (
            env._placement_simulator.current_block.length,
            env._placement_simulator.current_block.breadth,
            env._placement_simulator.current_block.angle,
        )

        after = env.future_workspace_choice_count_after_action(
            0, future_indices
        )

        self.assertEqual(1, after)
        self.assertEqual(
            current_state,
            (
                env._placement_simulator.current_block.length,
                env._placement_simulator.current_block.breadth,
                env._placement_simulator.current_block.angle,
            ),
        )
        self.assertEqual(0, len(env._workspaces[0].blocks))

    def test_action_preview_does_not_turn_future_block(self):
        strategy = BaseGridStrategy(step=1.0)
        current = Block(
            name="CURRENT",
            ship_no="T001",
            block_type="BUILD",
            length=2.0,
            breadth=2.0,
            height=5.0,
            weight=10.0,
            in_date=date(2026, 1, 5),
            out_date=date(2026, 1, 30),
        )
        future = Block(
            name="FUTURE",
            ship_no="T001",
            block_type="BUILD",
            length=8.0,
            breadth=4.0,
            height=5.0,
            weight=10.0,
            in_date=date(2026, 1, 6),
            out_date=date(2026, 1, 30),
        )
        rotation_only_workspace = Workspace(
            code="ROTATION_ONLY",
            origin_x=0.0,
            origin_y=0.0,
            length=10.0,
            breadth=10.0,
            strategy=strategy,
        )
        rotation_only_workspace.add_pre_placement(
            PrePlacedBlock(
                label="RIGHT_STRIP",
                pos_x=8.0,
                pos_y=5.0,
                length=4.0,
                breadth=10.0,
                start_date=date(2026, 1, 1),
                end_date=date(2026, 2, 28),
            )
        )
        workspaces = [
            Workspace(
                code="CURRENT_ONLY",
                origin_x=0.0,
                origin_y=0.0,
                length=2.0,
                breadth=2.0,
                strategy=strategy,
            ),
            rotation_only_workspace,
        ]
        env = BlockPlacementEnv([current, future], workspaces, strategy)
        env.reset(seed=0)
        future_indices = env.future_workspace_choice_indices()
        future_state = (
            env._blocks[future_indices[0]].length,
            env._blocks[future_indices[0]].breadth,
            env._blocks[future_indices[0]].angle,
        )

        after = env.future_workspace_choice_count_after_action(
            0, future_indices
        )

        self.assertEqual(0, after)
        self.assertEqual(
            future_state,
            (
                env._blocks[future_indices[0]].length,
                env._blocks[future_indices[0]].breadth,
                env._blocks[future_indices[0]].angle,
            ),
        )


if __name__ == "__main__":
    unittest.main()
