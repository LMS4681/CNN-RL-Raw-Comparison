"""JSON-safe fixed evaluation scenarios and evaluation-only metrics."""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Mapping

from alloc_env.block import Block, PrePlacedBlock
from alloc_env.block_generator import BlockDistribution, SyntheticBlockGenerator
from alloc_env.data_loader import clone_empty_workspaces
from alloc_env.strategy import BaseGridStrategy
from alloc_env.workspace import LotRegion, Workspace


SCENARIO_SCHEMA_VERSION = 3
SCENARIO_REQUIRED_KEYS = frozenset({
    "seed", "source", "blocks", "workspaces"
})


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
    source_blocks: list[Block] | None = None,
    vary_layout: bool = True,
    empirical_profile_probability: float = 0.2,
    target_month_counts: Mapping[tuple[int, int], int] | None = None,
    source_name: str = "holdout_fixed",
) -> list[dict]:
    scenarios = []
    for seed in seeds:
        generator = SyntheticBlockGenerator(
            dist=distribution,
            seed=seed,
            source_blocks=source_blocks,
            empirical_profile_probability=empirical_profile_probability,
            target_month_counts=target_month_counts,
        )
        blocks = generator.generate(
            n_blocks=n_blocks,
            base_date=base_date,
            spread_days=spread_days,
        )
        scenario_workspaces = (
            generator.generate_workspaces(workspaces)
            if vary_layout
            else clone_empty_workspaces(workspaces)
        )
        scenarios.append(
            {
                "seed": int(seed),
                "source": source_name,
                "blocks": [_block_record(block) for block in blocks],
                "workspaces": [
                    _workspace_record(workspace)
                    for workspace in scenario_workspaces
                ],
            }
        )
    return scenarios


def write_scenarios(
    path: str | Path,
    scenarios: list[dict],
    metadata: dict,
) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("Evaluation scenario metadata must be a dictionary")
    for scenario in scenarios:
        _validate_scenario(scenario)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCENARIO_SCHEMA_VERSION,
        "metadata": dict(metadata),
        "scenarios": scenarios,
    }
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


def _read_payload(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Evaluation scenario payload must be an object")
    if payload.get("schema_version") != SCENARIO_SCHEMA_VERSION:
        raise ValueError("Unsupported evaluation scenario schema")
    if not isinstance(payload.get("metadata"), dict):
        raise ValueError("Evaluation scenario payload must contain metadata")
    if not isinstance(payload.get("scenarios"), list):
        raise ValueError("Evaluation scenario payload must contain a list")
    for scenario in payload["scenarios"]:
        _validate_scenario(scenario)
    return payload


def _validate_scenario(scenario: object) -> None:
    if not isinstance(scenario, dict) or not SCENARIO_REQUIRED_KEYS.issubset(
        scenario
    ):
        raise ValueError(
            "Evaluation scenarios must include seed, source, blocks, and "
            "workspaces"
        )


def read_scenarios(path: str | Path) -> list[dict]:
    return _read_payload(path)["scenarios"]


def read_scenario_metadata(path: str | Path) -> dict:
    return dict(_read_payload(path)["metadata"])


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
