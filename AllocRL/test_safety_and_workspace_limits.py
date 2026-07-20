import csv
import importlib
import sys
import tempfile
import types
import unittest
from datetime import date
from pathlib import Path

import train as train_module
from alloc_env.block import Block, PrePlacedBlock
from alloc_env.data_loader import load_workspaces
from alloc_env.simulator import SimulationResult
from alloc_env.simulator import PlacementSimulator
from alloc_env.strategy import BaseGridStrategy
from alloc_env.workspace import Workspace


def make_block(name: str, x: float, y: float) -> Block:
    block = Block(
        name=name,
        ship_no="T001",
        block_type="BUILD",
        length=10.0,
        breadth=10.0,
        height=5.0,
        weight=10.0,
        in_date=date(2026, 1, 5),
        out_date=date(2026, 1, 20),
    )
    block.ref_x = x
    block.ref_y = y
    block.workspace_code = "PE001"
    return block


class SafetyAndWorkspaceLimitTests(unittest.TestCase):
    def test_checkpoint_schema_versions_must_match(self):
        saved = {
            key: None for key in train_module.CONFIG_COMPATIBILITY_KEYS
        }
        current = dict(saved)
        saved["observation_schema_version"] = 1
        current["observation_schema_version"] = 2

        self.assertIs(
            False,
            train_module.configs_compatible(saved, current),
        )
        self.assertEqual(
            {"observation_schema_version": (1, 2)},
            train_module.config_mismatches(saved, current),
        )

    def test_blocks_closer_than_safety_distance_intersect(self):
        left = make_block("A001", x=5.0, y=5.0)
        right = make_block("A002", x=15.5, y=5.0)

        self.assertTrue(left.intersects(right))

    def test_preplaced_block_closer_than_safety_distance_intersects(self):
        block = make_block("A001", x=15.5, y=5.0)
        preplaced = PrePlacedBlock(
            label="PRE-1",
            pos_x=5.0,
            pos_y=5.0,
            length=10.0,
            breadth=10.0,
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 30),
        )

        self.assertTrue(preplaced.intersects_block(block))

    def test_violation_report_flags_safety_distance_without_physical_overlap(self):
        workspace = Workspace(
            code="PE001",
            origin_x=0.0,
            origin_y=0.0,
            breadth=100.0,
            length=100.0,
            strategy=BaseGridStrategy(step=10.0),
        )
        result = SimulationResult(
            workspaces=[workspace],
            blocks=[
                make_block("A001", x=5.0, y=5.0),
                make_block("A002", x=15.5, y=5.0),
            ],
            delay_days=[0, 0],
        )

        violations = _find_placement_violations_without_matplotlib(result)

        self.assertIn("safety_distance", {item["type"] for item in violations})

    def test_clear_outgoing_blocks_keeps_blocks_through_out_date(self):
        workspace = Workspace(
            code="PE001",
            origin_x=0.0,
            origin_y=0.0,
            breadth=100.0,
            length=100.0,
            strategy=BaseGridStrategy(step=10.0),
        )
        block = make_block("A001", x=5.0, y=5.0)
        block.out_date = date(2026, 10, 1)
        workspace.add_block(block, date(2026, 9, 30))

        workspace.clear_outgoing_blocks(date(2026, 10, 1))

        self.assertEqual([block], workspace.blocks)

    def test_same_day_incoming_block_cannot_reuse_outgoing_position(self):
        workspace = Workspace(
            code="PE001",
            origin_x=0.0,
            origin_y=0.0,
            breadth=10.0,
            length=10.0,
            strategy=BaseGridStrategy(step=10.0),
        )
        outgoing = Block(
            name="OUT",
            ship_no="T001",
            block_type="BUILD",
            length=10.0,
            breadth=10.0,
            height=5.0,
            weight=10.0,
            in_date=date(2026, 9, 30),
            out_date=date(2026, 10, 1),
        )
        incoming = Block(
            name="IN",
            ship_no="T001",
            block_type="BUILD",
            length=10.0,
            breadth=10.0,
            height=5.0,
            weight=10.0,
            in_date=date(2026, 10, 1),
            out_date=date(2026, 10, 2),
        )

        result = PlacementSimulator().replay(
            [outgoing, incoming],
            [workspace],
            [0, 0],
            dropout_threshold=2,
        )

        self.assertEqual(1, result.delay_days[1])
        self.assertEqual(date(2026, 10, 2), result.blocks[1].in_date)

    def test_violation_report_flags_same_day_same_position_as_overlap(self):
        workspace = Workspace(
            code="PE001",
            origin_x=0.0,
            origin_y=0.0,
            breadth=100.0,
            length=100.0,
            strategy=BaseGridStrategy(step=10.0),
        )
        outgoing = make_block("OUT", x=5.0, y=5.0)
        outgoing.out_date = date(2026, 10, 1)
        incoming = make_block("IN", x=5.0, y=5.0)
        incoming.in_date = date(2026, 10, 1)
        incoming.out_date = date(2026, 10, 2)
        result = SimulationResult(
            workspaces=[workspace],
            blocks=[outgoing, incoming],
            delay_days=[0, 0],
        )

        violations = _find_placement_violations_without_matplotlib(result)

        self.assertIn("overlap", {item["type"] for item in violations})

    def test_visualization_prefers_csv_workspace_name_over_code(self):
        workspace = Workspace(
            code="PE001",
            origin_x=0.0,
            origin_y=0.0,
            breadth=100.0,
            length=100.0,
            name="Original-1",
            strategy=BaseGridStrategy(step=10.0),
        )
        module = _load_visualize_eval_without_matplotlib()

        self.assertEqual("Original-1", module._workspace_display_name(workspace))

    def test_load_workspaces_preserves_csv_workspaces_and_names_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            workspace_csv = tmp / "workspaces.csv"
            lot_csv = tmp / "lots.csv"

            with open(workspace_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["loc", "loc_code", "code", "name", "width", "height"])
                for idx in range(9):
                    writer.writerow([
                        "HQ",
                        "G02",
                        f"PE{idx + 1:03d}",
                        f"Original-{idx + 1}",
                        100 + idx,
                        50 + idx,
                    ])

            with open(lot_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["code", "lot", "x", "y", "width", "height"])
                writer.writerow(["PE001", "L1", 0, 0, 10, 10])

            workspaces = load_workspaces(
                str(workspace_csv),
                str(lot_csv),
                BaseGridStrategy(step=10.0),
            )

        self.assertEqual(9, len(workspaces))
        self.assertEqual(
            [f"Original-{i}" for i in range(1, 10)],
            [ws.name for ws in workspaces],
        )
        self.assertEqual("PE001", workspaces[0].code)
        self.assertEqual(1, len(workspaces[0].lots))


def _find_placement_violations_without_matplotlib(result: SimulationResult):
    module = _load_visualize_eval_without_matplotlib()
    return module.find_placement_violations(result)


def _load_visualize_eval_without_matplotlib():
    old_modules = {
        "matplotlib": sys.modules.get("matplotlib"),
        "matplotlib.pyplot": sys.modules.get("matplotlib.pyplot"),
        "numpy": sys.modules.get("numpy"),
        "visualize_eval_placement": sys.modules.get("visualize_eval_placement"),
    }
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_pyplot = types.ModuleType("matplotlib.pyplot")
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.ndarray = object
    sys.modules["matplotlib"] = fake_matplotlib
    sys.modules["matplotlib.pyplot"] = fake_pyplot
    sys.modules["numpy"] = fake_numpy
    sys.modules.pop("visualize_eval_placement", None)
    try:
        return importlib.import_module("visualize_eval_placement")
    finally:
        sys.modules.pop("visualize_eval_placement", None)
        for name, module in old_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


if __name__ == "__main__":
    unittest.main()
