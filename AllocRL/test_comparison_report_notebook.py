"""Contract tests for the comparison figures notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "comparison_report.ipynb"
FIGURES = (
    "comparison_training_curves.png",
    "comparison_failure_modes.png",
    "comparison_paired_seeds.png",
    "comparison_trend.png",
)


def notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def code_cells() -> list[str]:
    return [
        "".join(cell["source"])
        for cell in notebook()["cells"]
        if cell["cell_type"] == "code"
    ]


def test_notebook_is_clean_and_every_code_cell_parses():
    data = notebook()
    assert data["nbformat"] == 4
    assert all(cell.get("outputs", []) == [] for cell in data["cells"])
    assert all(cell.get("execution_count") is None for cell in data["cells"])
    for source in code_cells():
        ast.parse(source)


def test_notebook_discovers_its_inputs_from_drive():
    source = "\n".join(code_cells())
    assert "raw-direct-830k-seed0" in source and "scale-aware-cnn-6h-seed0" in source
    assert 'COMPARISON_DIR.glob("arm_comparison_summary_*.json")' in source
    assert "training_log.csv" in source and "holdout_selection.csv" in source
    assert "NOTHING FOUND" in source, "missing inputs must be reported, not guessed"


def test_every_figure_is_skipped_rather_than_drawn_from_partial_data():
    source = "\n".join(code_cells())
    assert source.count('print("skipped:') == 4
    assert "if len(training) < 2:" in source
    assert "if not comparisons:" in source
    assert "if len(comparisons) < 2:" in source


def test_each_arm_keeps_one_colour_across_every_figure():
    source = "\n".join(code_cells())
    assert 'ARM_COLOR = {"raw_direct": "#eb6834", "candidate_cnn": "#2a78d6"}' in source
    assert source.count("ARM_COLOR = {") == 1


def test_notebook_saves_the_four_named_figures():
    source = "\n".join(code_cells())
    for name in FIGURES:
        assert f'FIGURE_DIR / "{name}"' in source
    assert source.count("figure.savefig(") == len(FIGURES)


def test_paired_figure_headlines_the_primary_test_seeds_with_its_interval():
    source = "\n".join(code_cells())
    assert '["partitions"]["primary_test"]' in source
    assert "bootstrap_ci_95" in source
    assert "seeds_favouring_candidate" in source
    assert "primary-test seeds at timestep" in source


def test_timestep_axes_use_compact_tick_labels():
    source = "\n".join(code_cells())
    assert "STEP_TICKS = FuncFormatter" in source
    assert source.count("axis.xaxis.set_major_formatter(STEP_TICKS)") == 3
