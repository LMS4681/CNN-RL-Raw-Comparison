"""Tests for synthetic-data diagnostics and placement visualization exports."""

import csv
import tempfile
import unittest
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from alloc_env.block import Block, PrePlacedBlock
from alloc_env.simulator import SimulationResult
from alloc_env.strategy import BaseGridStrategy
from alloc_env.workspace import Workspace
from diagnose_synthetic_data import build_diagnostic_report, write_diagnostic_report
from visualize_eval_placement import (
    block_plot_label,
    export_evaluation_visualization,
    find_placement_violations,
)


def make_block(
    name: str,
    workspace_code: str | None = None,
    ref_x: float = 0.0,
    ref_y: float = 0.0,
    length: float = 10.0,
    breadth: float = 10.0,
) -> Block:
    block = Block(
        name=name,
        ship_no="T001",
        block_type="BUILD",
        length=length,
        breadth=breadth,
        height=5.0,
        weight=10.0,
        in_date=date(2026, 1, 5),
        out_date=date(2026, 1, 10),
    )
    block.workspace_code = workspace_code
    block.ref_x = ref_x
    block.ref_y = ref_y
    return block


def make_workspace() -> Workspace:
    return Workspace(
        code="PE001",
        origin_x=0.0,
        origin_y=0.0,
        breadth=100.0,
        length=100.0,
        strategy=BaseGridStrategy(step=10.0),
    )


class DiagnosticsAndPlacementVisualizationTests(unittest.TestCase):
    def test_synthetic_diagnostic_report_flags_zero_valid_synthetic_blocks(self):
        workspace = make_workspace()
        workspace.set_allowable_block_patterns(["A*"])

        csv_blocks = [make_block("A001")]
        synthetic_blocks = [make_block("SYN-00001")]
        report = build_diagnostic_report(csv_blocks, synthetic_blocks, [workspace])

        self.assertEqual(0, report["csv"]["zero_valid_workspace_count"])
        self.assertEqual(1, report["synthetic"]["zero_valid_workspace_count"])
        self.assertGreater(report["synthetic"]["zero_valid_workspace_ratio"], 0.0)

    def test_write_diagnostic_report_creates_csv_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = {
                "csv": {
                    "count": 1,
                    "zero_valid_workspace_count": 0,
                    "zero_valid_workspace_ratio": 0.0,
                    "valid_workspace_mean": 1.0,
                    "valid_workspace_min": 1,
                    "valid_workspace_max": 1,
                    "length_mean": 10.0,
                },
                "synthetic": {
                    "count": 1,
                    "zero_valid_workspace_count": 1,
                    "zero_valid_workspace_ratio": 1.0,
                    "valid_workspace_mean": 0.0,
                    "valid_workspace_min": 0,
                    "valid_workspace_max": 0,
                    "length_mean": 10.0,
                },
            }

            write_diagnostic_report(report, tmpdir)

            summary_path = Path(tmpdir) / "synthetic_diagnostic_summary.csv"
            self.assertTrue(summary_path.exists())
            with open(summary_path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual({"csv", "synthetic"}, {row["dataset"] for row in rows})

    def test_block_plot_label_includes_placement_dates(self):
        block = make_block("A001")

        self.assertEqual(
            "A001\n2026-01-05~2026-01-10",
            block_plot_label(block),
        )

    def test_find_placement_violations_detects_bounds_and_overlap(self):
        workspace = make_workspace()
        blocks = [
            make_block("A001", "PE001", ref_x=5.0, ref_y=5.0),
            make_block("A002", "PE001", ref_x=5.0, ref_y=5.0),
            make_block("A003", "PE001", ref_x=120.0, ref_y=5.0),
            make_block("A004", None),
        ]
        result = SimulationResult(
            workspaces=[workspace],
            blocks=blocks,
            delay_days=[0, 0, 0, SimulationResult.DROPOUT],
        )

        violations = find_placement_violations(result)
        violation_types = {item["type"] for item in violations}

        self.assertIn("overlap", violation_types)
        self.assertIn("out_of_bounds", violation_types)
        self.assertIn("dropout", violation_types)

    def test_find_placement_violations_flags_same_day_same_position(self):
        workspace = make_workspace()
        first = make_block("A001", "PE001", ref_x=5.0, ref_y=5.0)
        first.out_date = date(2026, 1, 10)
        second = make_block("A002", "PE001", ref_x=5.0, ref_y=5.0)
        second.in_date = date(2026, 1, 10)
        second.out_date = date(2026, 1, 15)
        result = SimulationResult(
            workspaces=[workspace],
            blocks=[first, second],
            delay_days=[0, 0],
        )

        violations = find_placement_violations(result)

        self.assertIn("overlap", {item["type"] for item in violations})

    def test_export_evaluation_visualization_writes_assignments_violations_and_frames(self):
        workspace = make_workspace()
        workspace.add_pre_placement(
            PrePlacedBlock(
                label="PRE-1",
                pos_x=70.0,
                pos_y=70.0,
                length=10.0,
                breadth=10.0,
                start_date=date(2026, 1, 1),
                end_date=date(2026, 1, 20),
            )
        )
        result = SimulationResult(
            workspaces=[workspace],
            blocks=[make_block("A001", "PE001", ref_x=10.0, ref_y=10.0)],
            delay_days=[0],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            export_evaluation_visualization(result, [0], tmpdir)

            output_dir = Path(tmpdir)
            frame_files = list((output_dir / "placement_frames").glob("*.png"))
            self.assertTrue((output_dir / "assignments.csv").exists())
            self.assertTrue((output_dir / "placement_violations.csv").exists())
            self.assertGreaterEqual(len(frame_files), 1)


if __name__ == "__main__":
    unittest.main()
