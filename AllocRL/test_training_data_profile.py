from __future__ import annotations

from collections import Counter
from datetime import date
from functools import lru_cache
from pathlib import Path

import pytest

from alloc_env import data_loader
from alloc_env.block import Block, PrePlacedBlock
from alloc_env.block_generator import SyntheticBlockGenerator
from alloc_env.data_split import split_blocks_by_ship
from alloc_env.strategy import BaseGridStrategy
import train as train_module


DATA_DIR = Path(__file__).parent / "data"
WORKSPACE_CSV = DATA_DIR / "선행건조 작업장 기준정보.csv"
LOT_CSV = DATA_DIR / "선행건조 지번 기준정보.csv"
BLOCK_CSV = DATA_DIR / "블록데이터.csv"

TARGET_WORKSPACE_CODES = [
    "PE049",
    "PE050",
    "PE055",
    "PE054",
    "PE056",
    "PE048",
    "PE044",
    "PE059",
    "PE060",
    "PE061",
]
SUPPLEMENTAL_WORKSPACES = {
    "PE054": ("500-B", 51.0, 31.0),
}
EXPECTED_MONTH_COUNTS = {
    (2025, 12): 64,
    (2026, 1): 122,
    (2026, 2): 106,
    (2026, 3): 142,
    (2026, 4): 153,
    (2026, 5): 151,
    (2026, 6): 175,
}


@lru_cache(maxsize=1)
def target_blocks() -> tuple[Block, ...]:
    return tuple(data_loader.load_target_blocks(
        str(BLOCK_CSV), excluded_start_months=(7, 11)
    ))


def count_start_months(blocks: list[Block] | tuple[Block, ...]):
    return Counter((block.in_date.year, block.in_date.month) for block in blocks)


def block_signature(block: Block):
    return (
        block.length,
        block.breadth,
        block.height,
        block.weight,
        block.original_duration,
    )


def test_target_loader_includes_all_eligible_rows_without_obstacles():
    strategy = BaseGridStrategy(step=5.0)
    workspaces = data_loader.load_workspaces(
        str(WORKSPACE_CSV),
        str(LOT_CSV),
        strategy,
        supplemental_workspaces=SUPPLEMENTAL_WORKSPACES,
    )

    targets = data_loader.load_target_blocks(
        str(BLOCK_CSV), excluded_start_months=(7, 11)
    )

    assert len(targets) == 913
    assert {block.in_date.month for block in targets}.isdisjoint({7, 11})
    assert count_start_months(targets) == EXPECTED_MONTH_COUNTS
    assert all(not workspace.pre_placements for workspace in workspaces)
    assert all(not workspace.blocks for workspace in workspaces)


def test_supplemented_workspace_uses_lot_derived_500b_geometry():
    workspaces = data_loader.load_workspaces(
        str(WORKSPACE_CSV),
        str(LOT_CSV),
        BaseGridStrategy(step=5.0),
        supplemental_workspaces=SUPPLEMENTAL_WORKSPACES,
    )

    selected = data_loader.select_workspaces_in_order(
        workspaces, TARGET_WORKSPACE_CODES
    )
    workspace = selected[3]

    assert [item.code for item in selected] == TARGET_WORKSPACE_CODES
    assert workspace.code == "PE054"
    assert workspace.name == "500-B"
    assert workspace.length == 51.0
    assert workspace.breadth == 31.0
    assert len(workspace.lots) == 3


def test_empty_workspace_clone_removes_dynamic_placement_state():
    workspaces = data_loader.load_workspaces(
        str(WORKSPACE_CSV),
        str(LOT_CSV),
        BaseGridStrategy(step=5.0),
        supplemental_workspaces=SUPPLEMENTAL_WORKSPACES,
    )
    source = data_loader.select_workspaces_in_order(
        workspaces, TARGET_WORKSPACE_CODES
    )[:1]
    source[0].add_pre_placement(
        PrePlacedBlock(
            label="existing",
            pos_x=10.0,
            pos_y=10.0,
            length=5.0,
            breadth=5.0,
            start_date=date(2025, 12, 1),
            end_date=date(2025, 12, 31),
        )
    )
    source[0].blocks.append(
        Block(
            name="assigned",
            ship_no="S1",
            block_type="BUILD",
            length=5.0,
            breadth=5.0,
            height=2.0,
            weight=10.0,
            in_date=date(2025, 12, 1),
            out_date=date(2025, 12, 5),
        )
    )

    empty = data_loader.clone_empty_workspaces(source)

    assert len(source[0].pre_placements) == 1
    assert len(source[0].blocks) == 1
    assert empty[0] is not source[0]
    assert empty[0].pre_placements == []
    assert empty[0].blocks == []
    assert empty[0].lots == source[0].lots


def test_balanced_profile_keeps_fixed_total_and_bounded_monthly_jitter():
    source = list(target_blocks())
    generator = SyntheticBlockGenerator.from_blocks(
        source,
        seed=17,
        monthly_jitter=20,
        empirical_profile_probability=0.0,
    )

    generated = generator.generate(
        n_blocks=913, base_date=min(block.in_date for block in source)
    )
    counts = count_start_months(generated)

    assert len(generated) == 913
    assert set(counts) == set(EXPECTED_MONTH_COUNTS)
    assert sum(counts.values()) == 913
    assert all(110 <= count <= 151 for count in counts.values())


def test_empirical_profile_preserves_source_month_counts():
    source = list(target_blocks())
    generator = SyntheticBlockGenerator.from_blocks(
        source,
        seed=23,
        monthly_jitter=20,
        empirical_profile_probability=1.0,
    )

    generated = generator.generate(
        n_blocks=913, base_date=min(block.in_date for block in source)
    )

    assert count_start_months(generated) == EXPECTED_MONTH_COUNTS


def test_split_sources_generate_full_empirical_target_profile():
    blocks, _ = train_module.load_allocation_scenario(
        DATA_DIR,
        BaseGridStrategy(step=5.0),
        train_module.parse_workspace_codes(
            train_module.DEFAULT_ACTIVE_WORKSPACE_CODES
        ),
    )
    split = split_blocks_by_ship(blocks, BLOCK_CSV)
    target_counts = count_start_months(blocks)

    for source in (split.training_blocks, split.holdout_blocks):
        generator = SyntheticBlockGenerator.from_blocks(
            source,
            seed=3,
            empirical_profile_probability=1.0,
            target_month_counts=target_counts,
        )
        generated = generator.generate(913, min(block.in_date for block in blocks))

        assert len(generated) == 913
        assert count_start_months(generated) == target_counts


def test_target_profile_rejects_month_without_source_templates():
    source = list(target_blocks())
    missing_month_source = [
        block for block in source if block.in_date.month != 12
    ]

    with pytest.raises(ValueError, match="no source templates"):
        SyntheticBlockGenerator.from_blocks(
            missing_month_source,
            target_month_counts=EXPECTED_MONTH_COUNTS,
        )


def test_monthly_bootstrap_is_seeded_and_preserves_row_correlations():
    source = list(target_blocks())
    kwargs = {
        "seed": 31,
        "monthly_jitter": 20,
        "empirical_profile_probability": 0.0,
    }
    first = SyntheticBlockGenerator.from_blocks(source, **kwargs).generate(
        913, min(block.in_date for block in source)
    )
    second = SyntheticBlockGenerator.from_blocks(source, **kwargs).generate(
        913, min(block.in_date for block in source)
    )

    first_records = [
        (block_signature(block), block.in_date, block.out_date)
        for block in first
    ]
    second_records = [
        (block_signature(block), block.in_date, block.out_date)
        for block in second
    ]
    source_signatures = {block_signature(block) for block in source}

    assert first_records == second_records
    assert all(block_signature(block) in source_signatures for block in first)
    assert all(block.in_date.weekday() < 5 for block in first)


def test_training_defaults_use_approved_ten_workspace_scenario():
    assert train_module.parse_workspace_codes(
        train_module.DEFAULT_ACTIVE_WORKSPACE_CODES
    ) == TARGET_WORKSPACE_CODES
    assert train_module.DEFAULT_MONTHLY_JITTER == 20
    assert train_module.DEFAULT_EMPIRICAL_PROFILE_PROBABILITY == 0.2
    assert train_module.DEFAULT_EXCLUDED_START_MONTHS == (7, 11)
    assert train_module.DEFAULT_SUPPLEMENTAL_WORKSPACES == SUPPLEMENTAL_WORKSPACES


def test_training_env_keeps_real_geometry_empty_and_generates_913_targets():
    split = split_blocks_by_ship(target_blocks(), BLOCK_CSV)
    source = split.training_blocks
    assert len(source) == 673
    strategy = BaseGridStrategy(step=5.0)
    all_workspaces = data_loader.load_workspaces(
        str(WORKSPACE_CSV),
        str(LOT_CSV),
        strategy,
        supplemental_workspaces=SUPPLEMENTAL_WORKSPACES,
    )
    workspaces = data_loader.clone_empty_workspaces(
        data_loader.select_workspaces_in_order(
            all_workspaces, TARGET_WORKSPACE_CODES
        )
    )
    dimensions = [
        (workspace.code, workspace.length, workspace.breadth)
        for workspace in workspaces
    ]
    generator = SyntheticBlockGenerator.from_blocks(
        source,
        seed=41,
        target_month_counts=EXPECTED_MONTH_COUNTS,
    )

    env = train_module.create_training_env(
        source,
        workspaces,
        strategy,
        generator,
        episode_n_blocks=913,
        grid_size=16,
        n_envs=1,
        seed=41,
    )
    try:
        env.reset()
        base_env = env.unwrapped
        generated_dimensions = [
            (workspace.code, workspace.length, workspace.breadth)
            for workspace in base_env._workspaces
        ]
        assert base_env._vary_layout is False
        assert len(base_env._blocks) == 913
        assert generated_dimensions == dimensions
        assert all(not item.pre_placements for item in base_env._workspaces)
        assert max(
            block.original_duration for block in base_env._blocks
        ) <= base_env._max_duration
    finally:
        env.close()


def test_shared_allocation_scenario_loader_recreates_model_inputs():
    blocks, workspaces = train_module.load_allocation_scenario(
        DATA_DIR,
        BaseGridStrategy(step=5.0),
        TARGET_WORKSPACE_CODES,
    )

    assert len(blocks) == 913
    assert [workspace.code for workspace in workspaces] == TARGET_WORKSPACE_CODES
    assert all(not workspace.blocks for workspace in workspaces)
    assert all(not workspace.pre_placements for workspace in workspaces)
