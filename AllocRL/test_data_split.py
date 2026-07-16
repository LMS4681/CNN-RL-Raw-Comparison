from pathlib import Path

import pytest

from alloc_env.data_split import (
    DEFAULT_HOLDOUT_FRACTION,
    DEFAULT_SPLIT_SEED,
    sha256_file,
    split_blocks_by_ship,
    write_split_manifest,
)
from alloc_env.strategy import BaseGridStrategy
from train import (
    DEFAULT_ACTIVE_WORKSPACE_CODES,
    load_allocation_scenario,
    parse_workspace_codes,
)


DATA_DIR = Path(__file__).parent / "data"
BLOCK_CSV = DATA_DIR / "블록데이터.csv"


def load_targets():
    blocks, _ = load_allocation_scenario(
        DATA_DIR,
        BaseGridStrategy(step=5.0),
        parse_workspace_codes(DEFAULT_ACTIVE_WORKSPACE_CODES),
    )
    return blocks


def test_current_source_split_is_pinned_and_group_disjoint():
    split = split_blocks_by_ship(load_targets(), BLOCK_CSV)

    assert DEFAULT_SPLIT_SEED == 20260716
    assert DEFAULT_HOLDOUT_FRACTION == 0.20
    assert len(split.training_blocks) == 673
    assert len(split.holdout_blocks) == 240
    training_ships = {block.ship_no for block in split.training_blocks}
    holdout_ships = {block.ship_no for block in split.holdout_blocks}
    assert len(training_ships) == 29
    assert len(holdout_ships) == 11
    assert training_ships.isdisjoint(holdout_ships)
    assert split.manifest["source_row_count"] == 913
    assert sum(split.manifest["source_month_counts"].values()) == 913
    assert split.manifest["source_sha256"] == sha256_file(BLOCK_CSV)


def test_split_is_deterministic_under_input_reordering():
    blocks = load_targets()
    normal = split_blocks_by_ship(blocks, BLOCK_CSV)
    reversed_split = split_blocks_by_ship(list(reversed(blocks)), BLOCK_CSV)

    assert normal.manifest["training_ship_nos"] == reversed_split.manifest[
        "training_ship_nos"
    ]
    assert normal.manifest["holdout_ship_nos"] == reversed_split.manifest[
        "holdout_ship_nos"
    ]


def test_split_preserves_source_order_and_returns_clones():
    blocks = load_targets()
    split = split_blocks_by_ship(blocks, BLOCK_CSV)
    training_ships = set(split.manifest["training_ship_nos"])
    holdout_ships = set(split.manifest["holdout_ship_nos"])
    training_sources = [
        block for block in blocks if block.ship_no in training_ships
    ]
    holdout_sources = [
        block for block in blocks if block.ship_no in holdout_ships
    ]

    assert [block.name for block in split.training_blocks] == [
        block.name for block in training_sources
    ]
    assert [block.name for block in split.holdout_blocks] == [
        block.name for block in holdout_sources
    ]
    assert all(
        split_block is not source_block
        for split_block, source_block in zip(
            split.training_blocks, training_sources
        )
    )
    assert all(
        split_block is not source_block
        for split_block, source_block in zip(
            split.holdout_blocks, holdout_sources
        )
    )


def test_split_rejects_empty_ship_number():
    block = load_targets()[0].clone()
    block.ship_no = ""

    with pytest.raises(ValueError, match="non-empty ship_no"):
        split_blocks_by_ship([block], BLOCK_CSV)


def test_write_split_manifest_creates_json_output(tmp_path):
    manifest_path = tmp_path / "manifests" / "split.json"
    manifest = {"source_row_count": 913, "source_sha256": "abc"}

    write_split_manifest(manifest_path, manifest)

    assert manifest_path.read_text(encoding="utf-8") == (
        '{\n  "source_row_count": 913,\n  "source_sha256": "abc"\n}\n'
    )
