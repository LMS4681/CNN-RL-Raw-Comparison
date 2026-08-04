"""Contract tests for the arm comparison notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "arm_comparison.ipynb"


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


def test_notebook_runs_the_module_against_both_arm_roots():
    source = "\n".join(code_cells())
    assert '"-m", "comparison.arm_comparison"' in source
    assert '"--raw-dir", str(RAW_DIR)' in source
    assert '"--cnn-dir", str(CNN_DIR)' in source
    assert '"--scenarios", "./data/fixed_eval_scenarios.json"' in source
    assert "raw-direct-830k-seed0/ppo" in source
    assert "scale-aware-cnn-6h-seed0/ppo" in source


def test_notebook_uses_a_separate_main_checkout_with_the_locked_dependencies():
    source = "\n".join(code_cells())
    assert "https://github.com/LMS4681/CNN-RL-Raw-Comparison.git" in source
    assert 'Path("/content/compare")' in source
    assert '"reset", "--hard", "origin/main"' in source
    assert '"--no-deps", "--require-hashes"' in source
    assert "requirements-comparison.txt" in source
    assert "/content/CNN-RL-Raw-Comparison" not in source


def test_notebook_reports_the_primary_test_partition_as_the_headline():
    source = "\n".join(code_cells())
    assert 'summary["partitions"]["primary_test"]' in source
    assert "bootstrap_ci_95" in source
    assert "seeds_favouring_candidate" in source
    assert "excludes 0" in source
    assert "best_model" not in "\n".join(
        cell for cell in code_cells()
    ), "the comparison must not load the selection-chosen model"


def test_notebook_fails_fast_when_an_arm_has_no_checkpoints():
    source = "\n".join(code_cells())
    assert 'raise RuntimeError(f"{label} has no checkpoints under {path}")' in source
