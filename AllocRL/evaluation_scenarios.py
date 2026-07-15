"""JSON-safe fixed evaluation scenarios and evaluation-only metrics."""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

from alloc_env.block import Block, PrePlacedBlock
from alloc_env.block_generator import BlockDistribution, SyntheticBlockGenerator
from alloc_env.strategy import BaseGridStrategy
from alloc_env.workspace import LotRegion, Workspace


SCENARIO_SCHEMA_VERSION = 1


def _block_record(block: Block) -> dict:
    return {
        "name": block.name,
        "ship_no": block.ship_no,
        "block_type": block.block_type,
        "length": block.length,
        "breadth": block.breadth,
        "height": block.height,
        "weight": block.weight,
        "in_date": block.in_date.isoformat(),
        "out_date": block.out_date.isoformat(),
    }


def _workspace_record(workspace: Workspace) -> dict:
    def finite(value: float) -> float | None:
        return value if math.isfinite(value) else None

    return {
        "code": workspace.code,
        "origin_x": workspace.origin_x,
        "origin_y": workspace.origin_y,
        "length": workspace.length,
        "breadth": workspace.breadth,
        "max_weight": finite(workspace.max_weight),
        "max_breadth": finite(workspace.max_breadth),
        "max_height": finite(workspace.max_height),
        "name": workspace.name,
        "allowable_block_patterns": (
            list(workspace.allowable_block_patterns)
            if workspace.allowable_block_patterns
            else None
        ),
        "lots": [
            {
                "lot_id": lot.lot_id,
                "origin_x": lot.origin_x,
                "origin_y": lot.origin_y,
                "length": lot.length,
                "breadth": lot.breadth,
            }
            for lot in workspace.lots
        ],
        "pre_placements": [
            {
                "label": item.label,
                "pos_x": item.pos_x,
                "pos_y": item.pos_y,
                "length": item.length,
                "breadth": item.breadth,
                "start_date": item.start_date.isoformat(),
                "end_date": item.end_date.isoformat(),
            }
            for item in workspace.pre_placements
        ],
    }


def generate_scenarios(
    distribution: BlockDistribution,
    workspaces: list[Workspace],
    seeds: list[int],
    n_blocks: int,
    base_date: date,
    spread_days: int,
) -> list[dict]:
    scenarios = []
    for seed in seeds:
        generator = SyntheticBlockGenerator(dist=distribution, seed=seed)
        blocks = generator.generate(
            n_blocks=n_blocks,
            base_date=base_date,
            spread_days=spread_days,
        )
        scenario_workspaces = generator.generate_workspaces(workspaces)
        scenarios.append(
            {
                "seed": int(seed),
                "blocks": [_block_record(block) for block in blocks],
                "workspaces": [
                    _workspace_record(workspace)
                    for workspace in scenario_workspaces
                ],
            }
        )
    return scenarios


def write_scenarios(path: str | Path, scenarios: list[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCENARIO_SCHEMA_VERSION,
        "scenarios": scenarios,
    }
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_scenarios(path: str | Path) -> list[dict]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCENARIO_SCHEMA_VERSION:
        raise ValueError("Unsupported evaluation scenario schema")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("Evaluation scenario payload must contain a list")
    return scenarios


def materialize_scenario(
    scenario: dict,
    strategy: BaseGridStrategy,
) -> tuple[list[Block], list[Workspace]]:
    blocks = [
        Block(
            name=item["name"],
            ship_no=item["ship_no"],
            block_type=item["block_type"],
            length=item["length"],
            breadth=item["breadth"],
            height=item["height"],
            weight=item["weight"],
            in_date=date.fromisoformat(item["in_date"]),
            out_date=date.fromisoformat(item["out_date"]),
        )
        for item in scenario["blocks"]
    ]

    workspaces = []
    for item in scenario["workspaces"]:
        workspace = Workspace(
            code=item["code"],
            origin_x=item["origin_x"],
            origin_y=item["origin_y"],
            length=item["length"],
            breadth=item["breadth"],
            max_weight=(
                item["max_weight"]
                if item["max_weight"] is not None
                else float("inf")
            ),
            max_breadth=(
                item["max_breadth"]
                if item["max_breadth"] is not None
                else float("inf")
            ),
            max_height=(
                item["max_height"]
                if item["max_height"] is not None
                else float("inf")
            ),
            name=item["name"],
            allowable_block_patterns=item["allowable_block_patterns"],
            strategy=strategy,
        )
        for lot in item["lots"]:
            workspace.add_lot(LotRegion(**lot))
        for pre in item["pre_placements"]:
            workspace.add_pre_placement(
                PrePlacedBlock(
                    label=pre["label"],
                    pos_x=pre["pos_x"],
                    pos_y=pre["pos_y"],
                    length=pre["length"],
                    breadth=pre["breadth"],
                    start_date=date.fromisoformat(pre["start_date"]),
                    end_date=date.fromisoformat(pre["end_date"]),
                )
            )
        workspaces.append(workspace)

    return blocks, workspaces


def compute_retained_choice_ratio(before: int, after: int) -> float:
    if before <= 0:
        return 1.0 if after <= 0 else float(after)
    return float(after) / float(before)
