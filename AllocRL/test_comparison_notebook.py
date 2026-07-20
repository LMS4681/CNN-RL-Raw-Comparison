"""Structural contract tests for the operator-facing overnight Colab notebook."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPOSITORY_ROOT / "notebooks" / "overnight_compare.ipynb"
ALLOC_RL = REPOSITORY_ROOT / "AllocRL"


def load_notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def code_cells(notebook: dict[str, object]) -> list[str]:
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def test_notebook_is_clean_and_has_the_required_semantic_cell_order():
    notebook = load_notebook()
    cells = notebook["cells"]
    assert notebook["nbformat"] == 4
    assert all(cell.get("outputs", []) == [] for cell in cells)
    assert all(cell.get("execution_count") is None for cell in cells)
    assert [cell["cell_type"] for cell in cells] == [
        "markdown", "code", "code", "code", "code", "code", "code", "code"
    ]
    markdown = "".join(cells[0]["source"]).lower()
    assert all(term in markdown for term in ("single-seed", "raw-direct", "candidate cnn", "3 hours", "6 hours", "not statistically conclusive"))


def test_notebook_mounts_drive_and_gates_on_cuda_with_runtime_facts():
    mount, cuda, *_ = code_cells(load_notebook())
    assert "drive.mount" in mount and "/content/drive" in mount
    assert "torch.cuda.is_available()" in cuda
    assert "raise RuntimeError" in cuda
    assert "nvidia-smi" in cuda
    assert all(term in cuda for term in ("torch.__version__", "torch.__file__", "torch.version.cuda", "torch.backends.cudnn", "get_device_name", "RAM"))


def test_notebook_clones_only_the_pinned_public_tag_into_the_exact_safe_path():
    *_, clone, _, _, _, _ = code_cells(load_notebook())
    assert "https://github.com/LMS4681/CNN-RL-Raw-Comparison.git" in clone
    assert "overnight-v1" in clone and '"--depth", "1"' in clone
    assert "/content/CNN-RL-Raw-Comparison" in clone
    assert '"rev-parse", "HEAD"' in clone and '"status", "--porcelain"' in clone
    assert "raise RuntimeError" in clone
    assert "shutil.rmtree" not in clone
    assert "target.resolve()" in clone and "target.parent" in clone


def test_notebook_installs_hashed_lock_without_mutating_colab_torch_stack():
    *_, install, _, _, _ = code_cells(load_notebook())
    assert "requirements-comparison.txt" in install and "--require-hashes" in install
    assert '"pip", "check"' in install
    assert all(term in install for term in ("before_torch", "torch.__version__", "torch.__file__", "torch.cuda.is_available()", "torch.version.cuda"))
    assert "assert after_torch == before_torch" in install
    assert "check=True" in install


def test_notebook_fails_closed_on_fixed_provenance_and_lock_hashes():
    *_, verify, _, _ = code_cells(load_notebook())
    for value in (
        "cd4e14fc1725a4ff159e59d6874d3602f3b65a06",
        "6125f53939a1b8eef8662b2628c0da2f1d0f26b5b541a99252858326b38cd814",
        "d3df1d0076248b4bcbddb4c910a3cb81481da65c7415c6b3cacf9e055cc3f9df",
        "UPSTREAM_BASELINE.md",
        "sha256",
        "%cd /content/CNN-RL-Raw-Comparison/AllocRL",
    ):
        assert value in verify
    assert "raise RuntimeError" in verify


def test_notebook_runs_one_exact_runner_command_and_honestly_reports_completion():
    *_, runner, status = code_cells(load_notebook())
    assert "python -m comparison.experiment_runner" in runner
    assert "--config ./configs/overnight_seed0.json" in runner
    assert "--output-root /content/drive/MyDrive/CNN-RL-comparison/overnight-20260721" in runner
    assert "check=True" in runner
    assert "COMPLETE.json" in status and "preliminary_comparison_ko.md" in status
    assert "PARTIAL_REPORT.md" in status
    assert "python -m comparison.experiment_runner --config ./configs/overnight_seed0.json --output-root /content/drive/MyDrive/CNN-RL-comparison/overnight-20260721" in status


def test_direct_requirements_are_exact_and_lock_is_hashed_without_colab_gpu_packages():
    direct = (ALLOC_RL / "requirements-comparison.in").read_text(encoding="utf-8")
    direct_lines = [line.strip() for line in direct.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    assert direct_lines == [
        "gymnasium==1.3.0", "stable-baselines3==2.9.0", "sb3-contrib==2.9.0",
        "matplotlib", "numpy", "pandas", "tensorboard", "tqdm", "rich",
    ]
    lock = (ALLOC_RL / "requirements-comparison.txt").read_text(encoding="utf-8")
    blocks = re.findall(r"(?ms)^[a-z0-9][a-z0-9-]+==.*?(?=^[a-z0-9][a-z0-9-]+==|\Z)", lock)
    assert blocks and all("--hash=sha256:" in block for block in blocks)
    assert not re.search(r"^(?:torch|triton|nvidia-[a-z0-9-]+)==", lock, re.MULTILINE)
    for pin in ("gymnasium==1.3.0", "stable-baselines3==2.9.0", "sb3-contrib==2.9.0"):
        assert pin in lock


def test_operator_docs_bind_the_lock_and_disclose_colab_limitations():
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert all(term in readme for term in (
        "one gpu colab notebook", "run all once", "keep the browser tab/runtime active",
        "6 hours plus setup/eval", "drive is authoritative", "rerun all to resume",
        "300 seconds plus the current callback interval", "cannot guarantee uninterrupted completion",
    ))
    provenance = (REPOSITORY_ROOT / "UPSTREAM_BASELINE.md").read_text(encoding="utf-8")
    lock = (ALLOC_RL / "requirements-comparison.txt").read_bytes()
    assert hashlib.sha256(lock).hexdigest() in provenance
    for value in (
        "cd4e14fc1725a4ff159e59d6874d3602f3b65a06",
        "6125f53939a1b8eef8662b2628c0da2f1d0f26b5b541a99252858326b38cd814",
        "d3df1d0076248b4bcbddb4c910a3cb81481da65c7415c6b3cacf9e055cc3f9df",
        "overnight-v1",
        "must never be placed onto the original upstream main/history",
    ):
        assert value in provenance
