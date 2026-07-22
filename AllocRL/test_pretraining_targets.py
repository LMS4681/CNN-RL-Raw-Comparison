from __future__ import annotations

from copy import deepcopy
from datetime import date

import numpy as np

from alloc_env.alloc_env import BlockPlacementEnv
from alloc_env.block import Block
from alloc_env.incremental_simulator import IncrementalPlacementSimulator
from alloc_env.strategy import BaseGridStrategy
from alloc_env.workspace import Workspace
from pretraining.targets import (
    AuxiliaryTargets,
    build_auxiliary_targets,
    grid_geometry_features,
)


def make_block(index: int) -> Block:
    return Block(
        name=f"B-{index}",
        ship_no="S-1",
        block_type="BUILD",
        length=4.0,
        breadth=2.0,
        height=1.0,
        weight=1.0,
        in_date=date(2026, 1, 5 + index),
        out_date=date(2026, 2, 20),
    )


def make_workspaces(count: int = 2) -> list[Workspace]:
    workspaces = [
        Workspace(
            code=f"WS-{index}",
            origin_x=0.0,
            origin_y=0.0,
            length=30.0,
            breadth=20.0,
            strategy=BaseGridStrategy(step=1.0),
        )
        for index in range(count)
    ]
    return workspaces


def make_noninitial_env() -> BlockPlacementEnv:
    strategy = BaseGridStrategy(step=1.0)
    env = BlockPlacementEnv(
        [make_block(index) for index in range(4)],
        make_workspaces(),
        strategy,
        grid_size=16,
    )
    env.reset(seed=123)
    env.step(0)
    assert env._current_block_index is not None
    return env


def observation_snapshot(env: BlockPlacementEnv) -> dict[str, np.ndarray]:
    return {
        key: value.copy()
        for key, value in env._get_obs().items()
    }


def assert_observations_equal(
    expected: dict[str, np.ndarray], actual: dict[str, np.ndarray]
) -> None:
    assert expected.keys() == actual.keys()
    for key in expected:
        np.testing.assert_array_equal(expected[key], actual[key])


def test_incremental_simulator_clone_preserves_links_without_sharing_state():
    simulator = IncrementalPlacementSimulator(
        [make_block(index) for index in range(3)],
        make_workspaces(),
        dropout_threshold=7,
    )
    simulator.assign_current(0)

    clone = simulator.clone_for_diagnostics()

    assert clone is not simulator
    assert clone.blocks[0] is not simulator.blocks[0]
    assert clone.workspaces[0] is not simulator.workspaces[0]
    assert clone.workspaces[0].blocks[0] is clone.blocks[0]
    assert clone.pending == simulator.pending
    assert clone.pending is not simulator.pending
    assert clone.assignments == simulator.assignments
    assert clone.assignments is not simulator.assignments
    assert clone.delay_days == simulator.delay_days
    assert clone.delay_days is not simulator.delay_days

    original_pending = set(simulator.pending)
    original_assignments = list(simulator.assignments)
    clone.assign_current(1)

    assert simulator.pending == original_pending
    assert simulator.assignments == original_assignments
    assert len(simulator.workspaces[1].blocks) == 0


def test_environment_diagnostic_clone_is_byte_equal_and_independent():
    env = make_noninitial_env()
    before = observation_snapshot(env)
    mask_before = env.action_masks().copy()
    rng_before = deepcopy(env.np_random.bit_generator.state)

    clone = env.clone_for_diagnostics()

    assert_observations_equal(before, clone._get_obs())
    np.testing.assert_array_equal(mask_before, clone.action_masks())
    assert clone._grid_cache is not env._grid_cache
    assert clone._placement_simulator is not env._placement_simulator
    assert clone._blocks[0] is not env._blocks[0]
    assert clone._workspaces[0] is not env._workspaces[0]
    assert clone._candidate_placements is not env._candidate_placements

    clone.np_random.random()
    assert env.np_random.bit_generator.state == rng_before

    clone.step(1)
    assert_observations_equal(before, env._get_obs())
    np.testing.assert_array_equal(mask_before, env.action_masks())

    clone_before_original_step = observation_snapshot(clone)
    env.step(0)
    assert_observations_equal(
        clone_before_original_step, clone._get_obs()
    )


def make_target_env() -> BlockPlacementEnv:
    workspaces = make_workspaces(10)
    workspaces[-1].max_weight = 0.5
    env = BlockPlacementEnv(
        [make_block(index) for index in range(4)],
        workspaces,
        BaseGridStrategy(step=1.0),
        grid_size=16,
    )
    env.reset(seed=321)
    return env


def test_exact_auxiliary_targets_match_environment_diagnostics_and_masks():
    env = make_target_env()
    before = observation_snapshot(env)
    simulator_before = (
        list(env._placement_simulator.assignments),
        list(env._placement_simulator.delay_days),
        set(env._placement_simulator.pending),
        env._placement_simulator.current_block_index,
        env._placement_simulator.env_date,
    )
    future_indices = env.future_workspace_choice_indices()

    targets = build_auxiliary_targets(env, include_replay=False)

    assert isinstance(targets, AuxiliaryTargets)
    assert targets.action_mask.shape == (10,)
    assert targets.current_placeable.shape == (10,)
    assert targets.future_fit.shape == (10, 16)
    assert targets.future_optionality_after.shape == (10,)
    assert targets.future_optionality_delta.shape == (10,)
    assert targets.largest_free_rectangle_ratio.shape == (10,)
    assert targets.free_component_count_normalized.shape == (10,)
    assert targets.replay_success_rate.shape == (10,)
    assert targets.replay_dropout_rate.shape == (10,)
    assert targets.replay_delay_ratio.shape == (10,)
    assert targets.replay_mask.shape == (10,)
    assert targets.action_mask.dtype == targets.replay_mask.dtype == np.bool_
    assert all(
        value.dtype == np.float32
        for value in (
            targets.current_placeable,
            targets.future_fit,
            targets.future_optionality_after,
            targets.future_optionality_delta,
            targets.largest_free_rectangle_ratio,
            targets.free_component_count_normalized,
            targets.replay_success_rate,
            targets.replay_dropout_rate,
            targets.replay_delay_ratio,
        )
    )

    denominator = 16 * 10
    baseline = env.future_workspace_choice_count(future_indices)
    for action in np.flatnonzero(targets.action_mask):
        expected_after = env.future_workspace_choice_count_after_action(
            int(action), future_indices
        )
        assert targets.future_optionality_after[action] == np.float32(
            expected_after / denominator
        )
        assert targets.future_optionality_delta[action] == np.float32(
            (expected_after - baseline) / denominator
        )
    invalid = np.flatnonzero(~targets.action_mask)
    assert invalid.tolist() == [9]
    assert np.count_nonzero(targets.future_fit[invalid]) == 0
    assert np.count_nonzero(targets.replay_mask) == 0
    assert all(
        np.all(np.isfinite(value))
        for value in vars(targets).values()
        if isinstance(value, np.ndarray)
    )
    assert_observations_equal(before, env._get_obs())
    assert simulator_before == (
        list(env._placement_simulator.assignments),
        list(env._placement_simulator.delay_days),
        set(env._placement_simulator.pending),
        env._placement_simulator.current_block_index,
        env._placement_simulator.env_date,
    )


def test_grid_geometry_distinguishes_equal_area_fragmentation():
    compact = np.zeros((8, 8), dtype=bool)
    compact[0, :] = True
    divider = np.zeros((8, 8), dtype=bool)
    divider[:, 3] = True

    compact_features = grid_geometry_features(compact)
    divider_features = grid_geometry_features(divider)

    assert np.count_nonzero(compact) == np.count_nonzero(divider)
    assert compact_features != divider_features
    assert compact_features[0] > divider_features[0]
    assert compact_features[1] < divider_features[1]


def test_bounded_replay_is_deterministic_and_handles_partial_horizon():
    env = make_target_env()
    before = observation_snapshot(env)

    first = build_auxiliary_targets(env, include_replay=True)
    second = build_auxiliary_targets(env, include_replay=True)

    np.testing.assert_array_equal(first.replay_mask, first.action_mask)
    np.testing.assert_array_equal(first.replay_mask, second.replay_mask)
    for name in (
        "replay_success_rate",
        "replay_dropout_rate",
        "replay_delay_ratio",
    ):
        first_values = getattr(first, name)
        second_values = getattr(second, name)
        np.testing.assert_array_equal(first_values, second_values)
        assert np.all((0.0 <= first_values) & (first_values <= 1.0))
    assert_observations_equal(before, env._get_obs())
