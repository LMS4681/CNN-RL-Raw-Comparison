import json
import unittest
from datetime import date, datetime

import numpy as np
import pytest

from alloc_env import calendar as cal
from alloc_env.alloc_env import BlockPlacementEnv
from alloc_env.block import Block, PrePlacedBlock
from alloc_env.incremental_simulator import IncrementalPlacementSimulator
from alloc_env.observation_state import (
    ObservationScales,
    build_observation_scales,
    build_observation_space,
    encode_current_block,
    encode_future_blocks,
    encode_future_demand,
    encode_pending_queues,
    working_day_position,
    working_days_until,
)
from alloc_env.simulator import SimulationResult
from alloc_env.strategy import BaseGridStrategy
from alloc_env.workspace import Workspace


def add_workdays(start: date, count: int) -> date:
    value = start
    for _index in range(count):
        value = cal.next_working_day(value)
    return value


def make_scales(**overrides) -> ObservationScales:
    values = {
        "max_length": 10.0,
        "max_breadth": 5.0,
        "max_duration": 10,
        "base_date": date(2026, 1, 5),
        "date_span_workdays": 100,
        "max_workspace_area": 5_000.0,
        "total_workspace_area": 50_000.0,
        "max_workspace_length": 100.0,
        "max_workspace_breadth": 50.0,
        "dropout_threshold": 7,
    }
    values.update(overrides)
    return ObservationScales(**values)


def make_observation_fixture(block_count: int = 40) -> dict:
    base = date(2026, 1, 5)
    blocks = [
        Block(
            name=f"B-{index}", ship_no="S-1", block_type="BUILD",
            length=10.0, breadth=5.0, height=1.0, weight=1.0,
            in_date=add_workdays(base, index),
            out_date=add_workdays(base, index + 9),
        )
        for index in range(block_count)
    ]
    workspaces = [
        Workspace(
            code=f"W-{index}", origin_x=0.0, origin_y=0.0,
            length=100.0, breadth=50.0,
            strategy=BaseGridStrategy(step=1.0),
        )
        for index in range(10)
    ]
    simulator = IncrementalPlacementSimulator(blocks, workspaces, 7)
    return {
        "blocks": simulator.blocks,
        "workspaces": simulator.workspaces,
        "simulator": simulator,
        "env_date": simulator.env_date,
        "scales": make_scales(),
    }


EXPECTED_SCHEMA3_SHAPES = {
    "block": (8,),
    "future_blocks": (16, 6),
    "future_demand": (3, 4),
    "future_mask": (16,),
    "grids": (10, 4, 64, 64),
    "pending_blocks": (10, 32, 7),
    "pending_mask": (10, 32),
    "pending_summary": (10, 4),
    "ws_meta": (10, 4),
}


def make_ten_workspace_env(
    state_context_mode: str = "full",
    *,
    observation_scales: ObservationScales | None = None,
) -> BlockPlacementEnv:
    fixture = make_observation_fixture(block_count=40)
    return BlockPlacementEnv(
        fixture["blocks"],
        fixture["workspaces"],
        BaseGridStrategy(step=1.0),
        use_synthetic=False,
        grid_size=64,
        state_context_mode=state_context_mode,
        observation_scales=observation_scales,
    )


class FixedEpisodeGenerator:
    def __init__(self, blocks, workspaces=None):
        self._blocks = blocks
        self._workspaces = workspaces

    def generate(self, n_blocks, base_date, spread_days=90):
        del n_blocks, base_date, spread_days
        return [block.clone() for block in self._blocks]

    def generate_workspaces(self, original_workspaces):
        if self._workspaces is None:
            return Workspace.deep_copy_list(original_workspaces)
        return Workspace.deep_copy_list(self._workspaces)


def build_structured_state(fixture: dict) -> dict[str, np.ndarray]:
    simulator = fixture["simulator"]
    indices = simulator.unassigned_block_indices()
    future_blocks, future_mask = encode_future_blocks(
        fixture["blocks"], indices, fixture["env_date"], fixture["scales"]
    )
    pending_blocks, pending_mask, pending_summary = encode_pending_queues(
        fixture["blocks"], fixture["workspaces"], simulator,
        fixture["scales"],
    )
    return {
        "block": encode_current_block(
            simulator.current_block, fixture["env_date"],
            simulator.assigned_count, fixture["scales"],
        ),
        "future_blocks": future_blocks,
        "future_mask": future_mask,
        "future_demand": encode_future_demand(
            fixture["blocks"], indices, fixture["env_date"], fixture["scales"]
        ),
        "pending_blocks": pending_blocks,
        "pending_mask": pending_mask,
        "pending_summary": pending_summary,
    }


def snapshot_workspace_state(workspaces) -> list[dict]:
    snapshots = []
    for workspace in workspaces:
        state = {
            key: value
            for key, value in vars(workspace).items()
            if key not in {
                "allowable_block_patterns",
                "lots",
                "blocks",
                "pre_placements",
                "strategy",
            }
        }
        state["allowable_block_patterns"] = (
            None
            if workspace.allowable_block_patterns is None
            else tuple(workspace.allowable_block_patterns)
        )
        state["lots"] = tuple(tuple(vars(lot).items()) for lot in workspace.lots)
        state["blocks"] = tuple(
            tuple(vars(block).items()) for block in workspace.blocks
        )
        state["pre_placements"] = tuple(
            tuple(vars(item).items()) for item in workspace.pre_placements
        )
        state["strategy"] = (
            None
            if workspace.strategy is None
            else (type(workspace.strategy), tuple(vars(workspace.strategy).items()))
        )
        snapshots.append(state)
    return snapshots


def make_queue_simulator(block_count: int = 4) -> IncrementalPlacementSimulator:
    blocks = [
        Block(
            name=f"B-{index}",
            ship_no="S-1",
            block_type="BUILD",
            length=5.0,
            breadth=5.0,
            height=1.0,
            weight=1.0,
            in_date=date(2026, 1, 5),
            out_date=date(2026, 1, 20),
        )
        for index in range(block_count)
    ]
    workspaces = [
        Workspace(
            code=f"W-{index}",
            origin_x=0.0,
            origin_y=0.0,
            length=100.0,
            breadth=100.0,
            strategy=BaseGridStrategy(step=1.0),
        )
        for index in range(2)
    ]
    return IncrementalPlacementSimulator(blocks, workspaces, 7)


class PendingAssignmentIndicesTests(unittest.TestCase):
    def test_pending_assignments_are_grouped_and_retry_sorted(self):
        simulator = make_queue_simulator()
        simulator.assignments[:] = [1, 0, 1, None]
        simulator.pending = {0, 1, 2, 3}
        simulator.blocks[0].delay_placement(2)
        simulator.blocks[2].delay_placement(2)

        self.assertEqual(simulator.pending_assignment_indices(1), [0, 2])
        self.assertEqual(simulator.pending_assignment_indices(0), [1])
        self.assertEqual(simulator.current_delay_workdays(0), 2)

    def test_pending_assignments_exclude_resolved_blocks(self):
        simulator = make_queue_simulator()
        simulator.assignments[:] = [1, 1, 1, None]
        simulator.delay_days[1] = 0

        self.assertEqual(simulator.pending_assignment_indices(), [0, 2])

    def test_current_delay_never_returns_negative(self):
        simulator = make_queue_simulator()
        simulator.blocks[0].in_date = date(2026, 1, 2)

        self.assertEqual(simulator.current_delay_workdays(0), 0)

    def test_current_delay_rejects_negative_block_index(self):
        simulator = make_queue_simulator()

        with self.assertRaisesRegex(IndexError, "block index"):
            simulator.current_delay_workdays(-1)

    def test_current_delay_rejects_upper_out_of_range_block_index(self):
        simulator = make_queue_simulator()

        with self.assertRaisesRegex(IndexError, "block index"):
            simulator.current_delay_workdays(len(simulator.blocks))

    def test_pending_assignments_exclude_non_pending_and_resolved_blocks(self):
        simulator = make_queue_simulator()
        simulator.assignments[:] = [1, 1, 1, 1]
        simulator.pending = {0, 1, 2}
        simulator.delay_days[1] = 0
        simulator.delay_days[2] = SimulationResult.DROPOUT

        self.assertEqual(simulator.pending_assignment_indices(), [0])

    def test_unknown_workspace_filter_returns_empty_list(self):
        simulator = make_queue_simulator()
        simulator.assignments[0] = 1

        self.assertEqual(simulator.pending_assignment_indices(99), [])


def assert_schema3_observation(env, observation):
    assert list(observation) == list(env.observation_space.spaces)
    assert {
        key: value.shape for key, value in observation.items()
    } == EXPECTED_SCHEMA3_SHAPES
    assert all(value.dtype == np.float32 for value in observation.values())
    assert env.observation_space.contains(observation)


def test_reset_step_and_terminal_observations_match_schema3():
    env = make_ten_workspace_env()

    reset_observation, _ = env.reset(seed=0)
    step_observation, _, terminated, _, _ = env.step(0)
    terminal_observation = env.unwrapped._get_terminal_obs()

    assert not terminated
    assert_schema3_observation(env, reset_observation)
    assert_schema3_observation(env, step_observation)
    assert_schema3_observation(env, terminal_observation)


def test_current_mode_zeros_only_structured_context():
    full_env = make_ten_workspace_env(state_context_mode="full")
    current_env = make_ten_workspace_env(state_context_mode="current")
    full_observation, _ = full_env.reset(seed=0)
    current_observation, _ = current_env.reset(seed=0)

    for key in (
        "future_blocks",
        "future_mask",
        "future_demand",
        "pending_blocks",
        "pending_mask",
        "pending_summary",
    ):
        assert np.count_nonzero(current_observation[key]) == 0
    for key in ("block", "grids", "ws_meta"):
        np.testing.assert_array_equal(current_observation[key], full_observation[key])
    assert_schema3_observation(current_env, current_observation)


def test_current_mode_context_stays_zero_after_step_and_at_terminal():
    env = make_ten_workspace_env(state_context_mode="current")
    env.reset(seed=0)

    observation, _reward, terminated, _truncated, _info = env.step(0)

    assert not terminated
    for key in (
        "future_blocks",
        "future_mask",
        "future_demand",
        "pending_blocks",
        "pending_mask",
        "pending_summary",
    ):
        assert np.count_nonzero(observation[key]) == 0

    while not terminated:
        observation, _reward, terminated, _truncated, _info = env.step(0)

    assert_schema3_observation(env, observation)
    for key in (
        "future_blocks",
        "future_mask",
        "future_demand",
        "pending_blocks",
        "pending_mask",
        "pending_summary",
    ):
        assert np.count_nonzero(observation[key]) == 0


@pytest.mark.parametrize("mode", ["future", None, []])
def test_constructor_rejects_invalid_state_context_mode(mode):
    with pytest.raises(ValueError, match="state_context_mode"):
        make_ten_workspace_env(state_context_mode=mode)


def test_constructor_rejects_environment_without_workspaces():
    fixture = make_observation_fixture(block_count=2)

    with pytest.raises(ValueError, match="workspace"):
        BlockPlacementEnv(fixture["blocks"], [], grid_size=64)


def test_constructor_rejects_non_scale_observation_scales():
    with pytest.raises(TypeError, match="ObservationScales"):
        make_ten_workspace_env(observation_scales={})


def test_fixture_scale_fallback_propagates_zero_working_day_span():
    fixture = make_observation_fixture(block_count=2)
    for block in fixture["blocks"]:
        block.in_date = date(2026, 1, 5)

    with pytest.raises(ValueError, match="date_span_workdays"):
        BlockPlacementEnv(
            fixture["blocks"], fixture["workspaces"], grid_size=64
        )


@pytest.mark.parametrize(
    ("scales", "message"),
    [
        (make_scales(max_length=9.0), "max_length"),
        (make_scales(max_breadth=4.0), "max_breadth"),
        (make_scales(max_duration=9), "max_duration"),
        (make_scales(date_span_workdays=1), "date_span_workdays"),
        (make_scales(max_workspace_length=99.0), "max_workspace_length"),
        (make_scales(max_workspace_breadth=49.0), "max_workspace_breadth"),
        (make_scales(max_workspace_area=4_999.0), "max_workspace_area"),
        (make_scales(total_workspace_area=49_999.0), "total_workspace_area"),
        (make_scales(dropout_threshold=6), "dropout_threshold"),
    ],
)
def test_constructor_rejects_incompatible_source_scales(scales, message):
    with pytest.raises(ValueError, match=message):
        make_ten_workspace_env(observation_scales=scales)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda blocks: setattr(blocks[0], "length", 11.0), "max_length"),
        (lambda blocks: setattr(blocks[0], "breadth", 6.0), "max_breadth"),
        (
            lambda blocks: setattr(blocks[0], "original_duration", 11),
            "max_duration",
        ),
        (
            lambda blocks: setattr(
                blocks[-1], "in_date", add_workdays(date(2026, 1, 5), 101)
            ),
            "date_span_workdays",
        ),
    ],
)
def test_synthetic_reset_rejects_generated_blocks_outside_scales(
    mutation, message
):
    fixture = make_observation_fixture(block_count=3)
    generated = [block.clone() for block in fixture["blocks"]]
    mutation(generated)
    env = BlockPlacementEnv(
        fixture["blocks"],
        fixture["workspaces"],
        BaseGridStrategy(step=1.0),
        use_synthetic=True,
        generator=FixedEpisodeGenerator(generated),
        synthetic_n_blocks=len(generated),
        vary_layout=False,
        grid_size=64,
        observation_scales=make_scales(),
    )

    with pytest.raises(ValueError, match=message):
        env.reset(seed=0)


@pytest.mark.parametrize(
    ("mutation", "scales", "message"),
    [
        (
            lambda workspaces: setattr(workspaces[0], "length", 101.0),
            make_scales(),
            "max_workspace_length",
        ),
        (
            lambda workspaces: setattr(workspaces[0], "breadth", 51.0),
            make_scales(),
            "max_workspace_breadth",
        ),
        (
            lambda workspaces: setattr(workspaces[0], "breadth", 51.0),
            make_scales(max_workspace_breadth=60.0),
            "max_workspace_area",
        ),
        (
            lambda workspaces: setattr(workspaces[0], "breadth", 51.0),
            make_scales(
                max_workspace_breadth=60.0,
                max_workspace_area=6_000.0,
            ),
            "total_workspace_area",
        ),
    ],
)
def test_synthetic_reset_rejects_generated_workspace_outside_scales(
    mutation, scales, message
):
    fixture = make_observation_fixture(block_count=3)
    generated_workspaces = Workspace.deep_copy_list(fixture["workspaces"])
    mutation(generated_workspaces)
    env = BlockPlacementEnv(
        fixture["blocks"],
        fixture["workspaces"],
        BaseGridStrategy(step=1.0),
        use_synthetic=True,
        generator=FixedEpisodeGenerator(
            fixture["blocks"], generated_workspaces
        ),
        synthetic_n_blocks=len(fixture["blocks"]),
        vary_layout=True,
        grid_size=64,
        observation_scales=scales,
    )

    with pytest.raises(ValueError, match=message):
        env.reset(seed=0)


def test_obsolete_n_future_blocks_constructor_keyword_is_rejected():
    fixture = make_observation_fixture(block_count=40)

    with pytest.raises(TypeError, match="n_future_blocks"):
        BlockPlacementEnv(
            fixture["blocks"],
            fixture["workspaces"],
            BaseGridStrategy(step=1.0),
            n_future_blocks=1,
        )


def test_delayed_assignment_uses_exact_pending_workspace_slot_and_delay():
    fixture = make_observation_fixture(block_count=3)
    fixture["workspaces"][0].add_pre_placement(
        PrePlacedBlock(
            label="FULL",
            pos_x=50.0,
            pos_y=25.0,
            length=100.0,
            breadth=50.0,
            start_date=date(2026, 1, 1),
            end_date=date(2026, 12, 31),
        )
    )
    env = BlockPlacementEnv(
        fixture["blocks"],
        fixture["workspaces"],
        BaseGridStrategy(step=1.0),
        grid_size=64,
    )
    before, _ = env.reset(seed=0)

    after, _, terminated, _, _ = env.step(0)

    assert not terminated
    assert np.count_nonzero(before["pending_blocks"]) == 0
    assert env._placement_simulator.pending_assignment_indices(0) == [0]
    assert after["pending_mask"][0, 0] == 1.0
    assert np.count_nonzero(after["pending_mask"]) == 1
    assert np.count_nonzero(after["pending_mask"][1:]) == 0
    assert after["pending_blocks"][0, 0, 3] == pytest.approx(2 / 7)
    assert after["pending_summary"][0, 2] == pytest.approx(2 / 7)
    assert np.count_nonzero(after["pending_blocks"][1:]) == 0
    assert np.count_nonzero(after["pending_summary"][1:]) == 0
    np.testing.assert_array_equal(after["grids"][1], before["grids"][1])
    current = env._placement_simulator.current_block
    current_state = vars(current).copy()
    repeated = env._get_obs()
    assert vars(current) == current_state
    np.testing.assert_array_equal(repeated["grids"][1], after["grids"][1])


def test_schema3_observation_space_has_complete_exact_contract():
    space = build_observation_space(n_workspaces=3, grid_size=8)

    assert list(space.spaces) == [
        "block",
        "future_blocks",
        "future_demand",
        "future_mask",
        "grids",
        "pending_blocks",
        "pending_mask",
        "pending_summary",
        "ws_meta",
    ]
    expected_shapes = {
        "block": (8,),
        "grids": (3, 4, 8, 8),
        "ws_meta": (3, 4),
        "future_blocks": (16, 6),
        "future_mask": (16,),
        "future_demand": (3, 4),
        "pending_blocks": (3, 32, 7),
        "pending_mask": (3, 32),
        "pending_summary": (3, 4),
    }
    assert {key: value.shape for key, value in space.spaces.items()} == expected_shapes
    assert all(value.dtype == np.dtype(np.float32) for value in space.spaces.values())
    assert all(np.all(value.low == 0.0) for value in space.spaces.values())
    assert all(np.all(value.high == 1.0) for value in space.spaces.values())


@pytest.mark.parametrize(
    ("n_workspaces", "grid_size", "message"),
    [(0, 64, "workspace"), (10, 0, "grid_size")],
)
def test_schema3_observation_space_rejects_nonpositive_dimensions(
    n_workspaces, grid_size, message
):
    with pytest.raises(ValueError, match=message):
        build_observation_space(n_workspaces=n_workspaces, grid_size=grid_size)


def test_working_day_helpers_exclude_weekends_and_count_positions():
    friday = date(2026, 1, 9)
    saturday = date(2026, 1, 10)
    monday = date(2026, 1, 12)

    assert working_days_until(friday, saturday) == 0
    assert working_days_until(friday, monday) == 1
    assert working_day_position(friday, monday) == 1
    assert working_days_until(monday, friday) == 0


def test_working_days_until_handles_weekend_starts_by_fixed_formula():
    saturday = date(2026, 1, 10)
    sunday = date(2026, 1, 11)
    monday = date(2026, 1, 12)
    tuesday = date(2026, 1, 13)

    assert working_days_until(saturday, sunday) == 0
    assert working_days_until(saturday, monday) == 0
    assert working_days_until(saturday, tuesday) == 1
    assert working_days_until(sunday, monday) == 0


def test_observation_scales_round_trip_is_deterministic_and_does_not_mutate_input():
    scales = make_scales()
    expected_keys = [
        "max_length",
        "max_breadth",
        "max_duration",
        "base_date",
        "date_span_workdays",
        "max_workspace_area",
        "total_workspace_area",
        "max_workspace_length",
        "max_workspace_breadth",
        "dropout_threshold",
    ]

    serialized = scales.to_dict()
    original = dict(serialized)

    assert list(serialized) == expected_keys
    assert serialized["base_date"] == "2026-01-05"
    assert ObservationScales.from_dict(serialized) == scales
    assert serialized == original
    assert scales.to_dict() == serialized


def test_observation_scales_accepts_real_and_integral_numpy_values_json_safely():
    scales = make_scales(
        max_length=np.float32(10.5),
        max_duration=np.int64(11),
        date_span_workdays=np.int32(101),
        dropout_threshold=np.int64(8),
    )

    serialized = scales.to_dict()

    assert isinstance(scales.max_length, float)
    assert isinstance(scales.max_duration, int)
    assert isinstance(scales.date_span_workdays, int)
    assert isinstance(scales.dropout_threshold, int)
    assert ObservationScales.from_dict(
        json.loads(json.dumps(serialized))
    ) == scales


@pytest.mark.parametrize(
    "field",
    [
        "max_length",
        "max_breadth",
        "max_workspace_area",
        "total_workspace_area",
        "max_workspace_length",
        "max_workspace_breadth",
    ],
)
@pytest.mark.parametrize("invalid", [True, "1.0", object()])
def test_observation_scales_rejects_non_real_float_denominators(field, invalid):
    with pytest.raises(TypeError, match=field):
        make_scales(**{field: invalid})


@pytest.mark.parametrize(
    "field", ["max_duration", "date_span_workdays", "dropout_threshold"]
)
@pytest.mark.parametrize("invalid", [True, 1.5, "1"])
def test_observation_scales_rejects_non_integral_integer_fields(field, invalid):
    with pytest.raises(TypeError, match=field):
        make_scales(**{field: invalid})


def test_observation_scales_rejects_datetime_as_base_date():
    with pytest.raises(TypeError, match="base_date"):
        make_scales(base_date=datetime(2026, 1, 5))


@pytest.mark.parametrize(
    ("mutation", "error", "message"),
    [
        (lambda values: values.pop("max_length"), ValueError, "missing.*max_length"),
        (lambda values: values.update(extra=1), ValueError, "unexpected.*extra"),
        (lambda values: values.update({1: "extra"}), ValueError, "unexpected.*1"),
        (lambda values: values.update(base_date=date(2026, 1, 5)), TypeError,
         "base_date.*ISO"),
        (lambda values: values.update(base_date="not-a-date"), ValueError,
         "base_date.*ISO"),
        (lambda values: values.update(max_duration=1.5), TypeError,
         "max_duration"),
    ],
)
def test_observation_scales_from_dict_rejects_invalid_payloads(
    mutation, error, message
):
    values = make_scales().to_dict()
    mutation(values)

    with pytest.raises(error, match=message):
        ObservationScales.from_dict(values)


def test_observation_scales_from_dict_requires_mapping():
    with pytest.raises(TypeError, match="mapping"):
        ObservationScales.from_dict([])


@pytest.mark.parametrize(
    "field",
    [
        "max_length",
        "max_breadth",
        "max_duration",
        "date_span_workdays",
        "max_workspace_area",
        "total_workspace_area",
        "max_workspace_length",
        "max_workspace_breadth",
        "dropout_threshold",
    ],
)
def test_observation_scales_reject_zero_normalization_denominators(field):
    with pytest.raises(ValueError, match=field):
        make_scales(**{field: 0})


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0])
def test_observation_scales_reject_nonfinite_or_negative_scales(invalid):
    with pytest.raises(ValueError, match="max_length"):
        make_scales(max_length=invalid)


def test_build_observation_scales_uses_complete_source_and_working_days():
    base = date(2026, 1, 5)
    small = Block(
        name="small", ship_no="S-1", block_type="BUILD",
        length=4.0, breadth=3.0, height=1.0, weight=1.0,
        in_date=base, out_date=add_workdays(base, 4),
    )
    late = Block(
        name="late", ship_no="S-1", block_type="BUILD",
        length=12.0, breadth=7.0, height=1.0, weight=1.0,
        in_date=add_workdays(base, 5), out_date=add_workdays(base, 14),
    )
    source_blocks = [small] * 912 + [late]
    workspaces = [
        Workspace(
            code=f"W-{index}", origin_x=0.0, origin_y=0.0,
            length=10.0 * (index + 1), breadth=20.0 * (index + 1),
            strategy=BaseGridStrategy(step=1.0),
        )
        for index in range(10)
    ]

    scales = build_observation_scales(source_blocks, workspaces, 7)

    assert scales.max_length == 12.0
    assert scales.max_breadth == 7.0
    assert scales.max_duration == 10
    assert scales.base_date == base
    assert scales.date_span_workdays == 5
    assert scales.max_workspace_area == 20_000.0
    assert scales.total_workspace_area == 77_000.0
    assert scales.max_workspace_length == 100.0
    assert scales.max_workspace_breadth == 200.0
    assert scales.dropout_threshold == 7


@pytest.mark.parametrize("block_count", [912, 914])
def test_build_observation_scales_requires_exact_block_cardinality(block_count):
    simulator = make_queue_simulator()

    with pytest.raises(ValueError, match="913"):
        build_observation_scales(
            [simulator.blocks[0]] * block_count,
            [simulator.workspaces[0]] * 10,
            7,
        )


@pytest.mark.parametrize("workspace_count", [9, 11])
def test_build_observation_scales_requires_exact_workspace_cardinality(
    workspace_count,
):
    simulator = make_queue_simulator()

    with pytest.raises(ValueError, match="10"):
        build_observation_scales(
            [simulator.blocks[0]] * 913,
            [simulator.workspaces[0]] * workspace_count,
            7,
        )


def test_build_observation_scales_allows_explicit_focused_fixture_opt_out():
    simulator = make_queue_simulator()
    simulator.blocks[1].in_date = cal.next_working_day(
        simulator.blocks[0].in_date
    )

    scales = build_observation_scales(
        simulator.blocks,
        simulator.workspaces,
        7,
        require_full_source=False,
    )

    assert scales.max_length == 5.0
    assert scales.max_workspace_area == 10_000.0


@pytest.mark.parametrize("require_full_source", [False, True])
def test_build_observation_scales_rejects_zero_working_day_span(
    require_full_source,
):
    simulator = make_queue_simulator()
    block_count = 913 if require_full_source else 2
    workspace_count = 10 if require_full_source else 2

    with pytest.raises(ValueError, match="date_span_workdays"):
        build_observation_scales(
            [simulator.blocks[0]] * block_count,
            [simulator.workspaces[0]] * workspace_count,
            7,
            require_full_source=require_full_source,
        )


def test_schema3_structured_shapes_ranges_and_initial_zero_pending_state():
    state = build_structured_state(make_observation_fixture())

    assert state["block"].shape == (8,)
    assert state["future_blocks"].shape == (16, 6)
    assert state["future_mask"].shape == (16,)
    assert state["future_demand"].shape == (3, 4)
    assert state["pending_blocks"].shape == (10, 32, 7)
    assert state["pending_mask"].shape == (10, 32)
    assert state["pending_summary"].shape == (10, 4)
    assert all(value.dtype == np.float32 for value in state.values())
    assert all(
        np.all((0.0 <= value) & (value <= 1.0))
        for value in state.values()
    )
    assert np.count_nonzero(state["pending_blocks"]) == 0
    assert np.count_nonzero(state["pending_mask"]) == 0
    assert np.count_nonzero(state["pending_summary"]) == 0


def test_pending_rows_follow_simulator_queue_order_and_exact_feature_order():
    simulator = make_queue_simulator()
    simulator.current_block_index = None
    simulator.assignments[:] = [1, 0, 1, None]
    simulator.blocks[0].delay_placement(2)
    simulator.blocks[2].delay_placement(1)
    before_blocks = [vars(block).copy() for block in simulator.blocks]
    before_workspaces = snapshot_workspace_state(simulator.workspaces)
    before_assignments = list(simulator.assignments)
    before_delay_days = list(simulator.delay_days)
    before_pending = set(simulator.pending)
    before_env_date = simulator.env_date
    before_current_block_index = simulator.current_block_index

    blocks, mask, summary = encode_pending_queues(
        simulator.blocks, simulator.workspaces, simulator, make_scales()
    )

    assert simulator.pending_assignment_indices(1) == [0, 2]
    np.testing.assert_allclose(
        blocks[1, 0],
        np.array([0.5, 1.0, 1.0, 2 / 7, 1.0, 0.0025, 0.05], np.float32),
    )
    np.testing.assert_allclose(
        blocks[1, 1],
        np.array([0.5, 1.0, 1.0, 1 / 7, 1.0, 0.0025, 0.05], np.float32),
    )
    np.testing.assert_array_equal(mask[1, :2], np.ones(2, np.float32))
    assert np.count_nonzero(blocks[1, 2:]) == 0
    assert np.count_nonzero(mask[1, 2:]) == 0
    np.testing.assert_allclose(
        summary[1],
        np.array([2 / 913, 50 / 40_000, 2 / 7, 0.0], np.float32),
    )
    assert [vars(block) for block in simulator.blocks] == before_blocks
    after_workspaces = snapshot_workspace_state(simulator.workspaces)
    assert after_workspaces == before_workspaces
    assert simulator.assignments == before_assignments
    assert simulator.delay_days == before_delay_days
    assert simulator.pending == before_pending
    assert simulator.env_date == before_env_date
    assert simulator.current_block_index == before_current_block_index


def test_pending_overflow_keeps_first_32_and_summarizes_complete_queue():
    fixture = make_observation_fixture(block_count=35)
    simulator = fixture["simulator"]
    simulator.current_block_index = None
    simulator.assignments = [0] * 35
    for index, block in enumerate(fixture["blocks"]):
        block.length = float(index + 1)
        block.breadth = 1.0
    scales = make_scales(max_length=40.0, max_breadth=1.0)

    blocks, mask, summary = encode_pending_queues(
        fixture["blocks"], fixture["workspaces"], simulator,
        scales,
    )

    assert mask[0].sum() == 32
    np.testing.assert_allclose(
        blocks[0, :, 0], np.arange(1, 33, dtype=np.float32) / 40
    )
    assert summary[0, 0] == pytest.approx(35 / 913)
    assert summary[0, 1] == pytest.approx(630 / 20_000)
    assert summary[0, 2] == 0.0
    assert summary[0, 3] == pytest.approx(3 / 913)
    assert np.count_nonzero(blocks[1:]) == 0
    assert np.count_nonzero(mask[1:]) == 0
    assert np.count_nonzero(summary[1:]) == 0


def test_delayed_assigned_block_updates_pending_features_before_resolution():
    fixture = make_observation_fixture(block_count=2)
    simulator = fixture["simulator"]
    simulator.assignments[0] = 0
    simulator.current_block_index = 1

    before, before_mask, _summary = encode_pending_queues(
        fixture["blocks"], fixture["workspaces"], simulator,
        fixture["scales"],
    )
    fixture["blocks"][0].delay_placement(1)
    after, after_mask, _summary = encode_pending_queues(
        fixture["blocks"], fixture["workspaces"], simulator,
        fixture["scales"],
    )

    assert before_mask[0, 0] == after_mask[0, 0] == 1.0
    assert before[0, 0, 3] == 0.0
    assert after[0, 0, 3] == pytest.approx(1 / 7)
    assert simulator.delay_days[0] is None
    assert 0 in simulator.pending


@pytest.mark.parametrize("delay", [7, 8])
def test_pending_delay_at_and_above_dropout_threshold_clips_to_one(delay):
    fixture = make_observation_fixture(block_count=2)
    simulator = fixture["simulator"]
    simulator.assignments[0] = 0
    simulator.current_block_index = 1
    fixture["blocks"][0].delay_placement(delay)

    blocks, mask, summary = encode_pending_queues(
        fixture["blocks"], fixture["workspaces"], simulator,
        fixture["scales"],
    )

    assert mask[0, 0] == 1.0
    assert blocks[0, 0, 3] == 1.0
    assert summary[0, 2] == 1.0
    assert simulator.delay_days[0] is None
    assert 0 in simulator.pending


def test_pending_encoder_rejects_zero_workspace_denominator():
    fixture = make_observation_fixture(block_count=2)
    simulator = fixture["simulator"]
    simulator.assignments[0] = 0
    simulator.current_block_index = 1
    fixture["workspaces"][0].length = 0.0

    with pytest.raises(ValueError, match="workspace dimensions"):
        encode_pending_queues(
            fixture["blocks"], fixture["workspaces"], simulator,
            fixture["scales"],
        )


def test_pending_encoder_rejects_nonfinite_features():
    fixture = make_observation_fixture(block_count=2)
    simulator = fixture["simulator"]
    simulator.assignments[0] = 0
    simulator.current_block_index = 1
    fixture["blocks"][0].original_duration = float("nan")

    with pytest.raises(ValueError, match="finite"):
        encode_pending_queues(
            fixture["blocks"], fixture["workspaces"], simulator,
            fixture["scales"],
        )


if __name__ == "__main__":
    unittest.main()
