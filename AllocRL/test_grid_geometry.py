from datetime import date

import numpy as np
import pytest

from alloc_env.block import SAFETY_DISTANCE, Block, PrePlacedBlock
from alloc_env.observation_state import working_days_until
from alloc_env.occupancy_grid import BaseGridCache, OccupancyGridRenderer
from alloc_env.strategy import BaseGridStrategy
from alloc_env.workspace import Workspace


TEST_DATE = date(2026, 1, 5)


def make_grid_workspace(
    length: float,
    breadth: float,
    *,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> Workspace:
    return Workspace(
        code="W-1",
        origin_x=origin_x,
        origin_y=origin_y,
        length=length,
        breadth=breadth,
        strategy=BaseGridStrategy(step=1.0),
    )


def make_grid_block(
    length: float,
    breadth: float,
    *,
    out_date: date = date(2026, 1, 20),
) -> Block:
    return Block(
        name="B-1",
        ship_no="S-1",
        block_type="BUILD",
        length=length,
        breadth=breadth,
        height=1.0,
        weight=1.0,
        in_date=TEST_DATE,
        out_date=out_date,
    )


def test_axes_fill_full_grid_independently_with_workspace_origin():
    renderer = OccupancyGridRenderer(64)
    workspace = make_grid_workspace(
        length=200.0,
        breadth=20.0,
        origin_x=10.0,
        origin_y=-20.0,
    )

    info = renderer.coordinate_map(workspace)

    assert info.x_px_per_m == pytest.approx(64 / 200.0)
    assert info.y_px_per_m == pytest.approx(64 / 20.0)
    assert renderer.rectangle_bounds(
        workspace,
        center_x=110.0,
        center_y=-10.0,
        length=200.0,
        breadth=20.0,
    ) == (0, 0, 64, 64)


def test_positive_in_bounds_rectangle_gets_at_least_one_pixel_per_axis():
    renderer = OccupancyGridRenderer(64)
    workspace = make_grid_workspace(length=1000.0, breadth=1000.0)

    bounds = renderer.rectangle_bounds(
        workspace,
        center_x=0.1,
        center_y=0.1,
        length=0.01,
        breadth=0.01,
    )

    assert bounds[2] - bounds[0] >= 1
    assert bounds[3] - bounds[1] >= 1


def test_outside_rectangle_clamps_to_an_empty_in_range_axis():
    renderer = OccupancyGridRenderer(64)
    workspace = make_grid_workspace(length=100.0, breadth=100.0)

    bounds = renderer.rectangle_bounds(
        workspace,
        center_x=-5.0,
        center_y=50.0,
        length=2.0,
        breadth=10.0,
    )

    assert bounds == (0, 28, 0, 36)
    assert all(0 <= bound <= 64 for bound in bounds)


@pytest.mark.parametrize("grid_size", [0, -1, 1.5, float("inf")])
def test_renderer_rejects_invalid_grid_size(grid_size):
    with pytest.raises(ValueError, match="grid_size"):
        OccupancyGridRenderer(grid_size)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("length", 0.0),
        ("breadth", -1.0),
        ("length", float("nan")),
        ("origin_x", float("inf")),
    ],
)
def test_coordinate_map_rejects_invalid_workspace_geometry(field, value):
    workspace = make_grid_workspace(length=100.0, breadth=100.0)
    setattr(workspace, field, value)

    with pytest.raises(ValueError, match="workspace"):
        OccupancyGridRenderer(64).coordinate_map(workspace)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("center_x", float("nan")),
        ("center_y", float("inf")),
        ("length", 0.0),
        ("breadth", -1.0),
    ],
)
def test_rectangle_bounds_rejects_invalid_rectangle_geometry(field, value):
    values = {
        "center_x": 50.0,
        "center_y": 50.0,
        "length": 10.0,
        "breadth": 10.0,
    }
    values[field] = value

    with pytest.raises(ValueError, match="rectangle"):
        OccupancyGridRenderer(64).rectangle_bounds(
            make_grid_workspace(length=100.0, breadth=100.0),
            **values,
        )


def test_collision_channel_expands_by_exact_safety_distance():
    workspace = make_grid_workspace(length=200.0, breadth=20.0)
    block = make_grid_block(length=10.0, breadth=6.0)
    block.move(100.0, 10.0)
    workspace.add_block(block, TEST_DATE)

    grid = OccupancyGridRenderer(64).render_base(workspace, TEST_DATE)

    ys, xs = np.nonzero(grid[0])
    assert SAFETY_DISTANCE == 1.0
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (30, 33, 19, 44)


def test_remaining_lifetime_uses_working_days_clips_and_keeps_overlap_maximum():
    workspace = make_grid_workspace(length=100.0, breadth=100.0)
    short = make_grid_block(
        length=10.0,
        breadth=10.0,
        out_date=date(2026, 1, 9),
    )
    short.move(20.0, 20.0)
    long = make_grid_block(
        length=10.0,
        breadth=10.0,
        out_date=date(2026, 6, 1),
    )
    long.move(50.0, 50.0)
    overlap = make_grid_block(
        length=4.0,
        breadth=4.0,
        out_date=date(2026, 1, 6),
    )
    overlap.move(20.0, 20.0)
    workspace.add_block(short, TEST_DATE)
    workspace.add_block(long, TEST_DATE)
    workspace.add_block(overlap, TEST_DATE)

    grid = OccupancyGridRenderer(64).render_base(workspace, TEST_DATE)

    short_px = (12, 12)
    long_px = (32, 32)
    assert working_days_until(TEST_DATE, short.out_date) == 4
    assert grid[1, short_px[1], short_px[0]] == pytest.approx(4 / 60)
    assert grid[1, long_px[1], long_px[0]] == pytest.approx(1.0)


def test_only_active_preplacements_are_rendered_with_safety_expansion():
    workspace = make_grid_workspace(length=100.0, breadth=100.0)
    workspace.add_pre_placement(
        PrePlacedBlock(
            label="ACTIVE",
            pos_x=20.0,
            pos_y=20.0,
            length=10.0,
            breadth=10.0,
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 9),
        )
    )
    workspace.add_pre_placement(
        PrePlacedBlock(
            label="INACTIVE",
            pos_x=80.0,
            pos_y=80.0,
            length=10.0,
            breadth=10.0,
            start_date=date(2026, 1, 6),
            end_date=date(2026, 1, 9),
        )
    )

    grid = OccupancyGridRenderer(64).render_base(workspace, TEST_DATE)

    assert grid[0, 12, 12] == 1.0
    assert grid[1, 12, 12] == pytest.approx(4 / 60)
    assert grid[:, 51, 51].sum() == 0.0


def test_base_grid_cache_has_two_channels_and_respects_invalidation():
    workspace = make_grid_workspace(length=100.0, breadth=100.0)
    block = make_grid_block(length=10.0, breadth=10.0)
    block.move(20.0, 20.0)
    workspace.add_block(block, TEST_DATE)
    cache = BaseGridCache(OccupancyGridRenderer(64), n_workspaces=1)

    initial = cache.get_base_grids([workspace], TEST_DATE)
    block.move(50.0, 0.0)
    stale = cache.get_base_grids([workspace], TEST_DATE)
    cache.invalidate(0)
    refreshed = cache.get_base_grids([workspace], TEST_DATE)

    assert initial.shape == (1, 2, 64, 64)
    np.testing.assert_array_equal(stale, initial)
    assert not np.array_equal(refreshed, initial)


def test_base_grid_cache_refreshes_lifetime_when_environment_date_changes():
    workspace = make_grid_workspace(length=100.0, breadth=100.0)
    block = make_grid_block(
        length=10.0,
        breadth=10.0,
        out_date=date(2026, 1, 9),
    )
    block.move(20.0, 20.0)
    workspace.add_block(block, TEST_DATE)
    cache = BaseGridCache(OccupancyGridRenderer(64), n_workspaces=1)

    monday = cache.get_base_grids([workspace], TEST_DATE)
    tuesday = cache.get_base_grids([workspace], date(2026, 1, 6))

    assert monday[0, 1, 12, 12] == pytest.approx(4 / 60)
    assert tuesday[0, 1, 12, 12] == pytest.approx(3 / 60)
