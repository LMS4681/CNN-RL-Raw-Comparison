"""Structural contract tests for the operator-facing overnight Colab notebook."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPOSITORY_ROOT / "notebooks" / "overnight_compare.ipynb"
ALLOC_RL = REPOSITORY_ROOT / "AllocRL"
PLAN_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "superpowers"
    / "plans"
    / "2026-07-21-overnight-raw-cnn-comparison-implementation.md"
)
COLAB_BADGE = (
    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
    "(https://colab.research.google.com/github/LMS4681/"
    "CNN-RL-Raw-Comparison/blob/overnight-v1/notebooks/overnight_compare.ipynb)"
)
DATA_TREE_OID = "0140dfe704c607045da2f20faa32a0141e7bcc9b"
LOCK_SHA256 = "2098a1d132dde6e3255b0e7be6193edb3b09f758565aa319837afd53dbdf4bd7"


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
    assert install.startswith("import json\n")
    assert "requirements-comparison.txt" in install and "--require-hashes" in install
    assert '"--no-deps"' in install
    assert '"pip", "check"' in install
    assert "def child_torch_snapshot" in install
    assert "sys.executable" in install and '"-c"' in install and "json.loads" in install
    assert all(term in install for term in (
        "torch.__version__", "torch.__file__", "torch.cuda.is_available()",
        "torch.version.cuda", "torch.backends.cudnn.version()", "importlib.metadata",
        "before_torch", "after_torch", "before_distributions", "after_distributions",
    ))
    assert "import torch as torch_after" not in install
    assert "assert after_torch == before_torch" in install
    assert "assert after_distributions == before_distributions" in install
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
    assert "--take-over-stale-lease" in runner
    assert "check=True" in runner
    assert "runner_error = None" in runner
    assert "except subprocess.CalledProcessError as error" in runner
    assert "runner_error = error" in runner
    assert "COMPLETE.json" in status and "preliminary_comparison_ko.md" in status
    assert "PARTIAL_REPORT.md" in status
    assert "--take-over-stale-lease" in status
    assert "runner_error is not None" in status and "raise RuntimeError" in status
    assert status.index("display(") < status.index("if runner_error is not None")


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
    assert not re.search(
        r"^(?:torch|triton|nvidia-[a-z0-9-]+|cuda-[a-z0-9-]+|filelock|fsspec|jinja2|networkx|sympy|mpmath)==",
        lock,
        re.MULTILINE,
    )
    for pin in ("gymnasium==1.3.0", "stable-baselines3==2.9.0", "sb3-contrib==2.9.0"):
        assert pin in lock


def test_readme_has_exact_pinned_colab_badge():
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    assert COLAB_BADGE in readme


def test_data_provenance_records_unchanged_upstream_tree():
    provenance = (REPOSITORY_ROOT / "UPSTREAM_BASELINE.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(provenance.split())
    assert DATA_TREE_OID in provenance
    assert (
        "The tracked `AllocRL/data` tree is inherited unchanged from the same "
        "owner's public baseline at the approved commit; this comparison adds "
        "or modifies no files under that directory."
    ) in normalized


def test_publication_secret_gate_does_not_match_itself_or_print_secret_values():
    plan = PLAN_PATH.read_text(encoding="utf-8")
    secret_pattern = re.compile(
        "(" + "|".join((
            "gh" + r"p_[A-Za-z0-9]{20,}",
            "github" + r"_pat_",
            "AI" + r"za[0-9A-Za-z_-]{20,}",
            "-----BEGIN " + r"(RSA|OPENSSH|EC) PRIVATE KEY-----",
        )) + ")"
    )
    assert secret_pattern.search(plan) is None
    assert "$secretPatternParts = @(" in plan
    assert "$pattern = '(' + ($secretPatternParts -join '|') + ')'" in plan
    assert "git grep -l -I -E $pattern HEAD" in plan
    assert "git grep -n -I -E" not in plan


def test_publication_secret_gate_exits_zero_in_a_clean_repository(tmp_path):
    powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
    if powershell is None:
        import pytest

        pytest.skip("PowerShell is not installed")

    plan = PLAN_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"Run this tracked-content secret gate before any public push:\s*"
        r"```powershell\r?\n(?P<source>.*?)\r?\n```",
        plan,
        re.DOTALL,
    )
    assert match is not None

    repository = tmp_path / "clean-repository"
    repository.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.name", "Publication Test"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "publication-test@example.invalid"],
        cwd=repository,
        check=True,
    )
    (repository / "README.md").write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "test fixture"],
        cwd=repository,
        check=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""

    result = subprocess.run(
        [
            powershell,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            match["source"],
        ],
        cwd=repository,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_comparison_lock_is_checked_out_as_canonical_lf_bytes():
    lock = (ALLOC_RL / "requirements-comparison.txt").read_bytes()
    assert b"\r\n" not in lock
    assert hashlib.sha256(lock).hexdigest() == LOCK_SHA256
    attributes = (REPOSITORY_ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "AllocRL/requirements-comparison.txt text eol=lf" in attributes.splitlines()


def test_operator_docs_bind_the_lock_and_disclose_colab_limitations():
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert all(term in readme for term in (
        "one gpu colab notebook", "run all once", "keep the browser tab/runtime active",
        "6 hours plus setup/eval", "drive is authoritative", "rerun all to resume",
        "300 seconds plus the current callback interval", "cannot guarantee uninterrupted completion",
        "more than 15 minutes old", "guarded stale takeover",
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
