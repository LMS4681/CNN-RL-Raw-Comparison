"""Contract tests for the raw-direct step-matched comparison notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "raw_direct_830k.ipynb"
RELEASE_TAG = "raw-direct-830k-v1"


def notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def code_cells() -> list[str]:
    return [
        "".join(cell["source"])
        for cell in notebook()["cells"]
        if cell["cell_type"] == "code"
    ]


def training_cell() -> str:
    matches = [source for source in code_cells() if "ppo_command = [" in source]
    assert len(matches) == 1
    return matches[0]


def test_notebook_is_clean_and_every_code_cell_parses():
    data = notebook()
    assert data["nbformat"] == 4
    assert all(cell.get("outputs", []) == [] for cell in data["cells"])
    assert all(cell.get("execution_count") is None for cell in data["cells"])
    for source in code_cells():
        ast.parse(source)


def test_notebook_pins_its_own_tag_and_verifies_the_checkout():
    source = "\n".join(code_cells())
    assert "https://github.com/LMS4681/CNN-RL-Raw-Comparison.git" in source
    assert f'RELEASE_TAG = "{RELEASE_TAG}"' in source
    assert '"--depth", "1"' in source
    assert '"status", "--porcelain"' in source
    assert '"--no-deps"' in source and '"--require-hashes"' in source
    assert "child_torch_snapshot" in source


def test_notebook_requires_l4_and_verifies_immutable_inputs():
    source = "\n".join(code_cells())
    assert 'ACCEPTED_GPU_NAMES = {"NVIDIA L4"}' in source
    assert "psutil.virtual_memory" in source
    for value in (
        "913cac9046dec8164ef65da60275522f7127de5ea775b1c5a6b6aac255716271",
        "601bd6143ed8890577e5ff34921241d36fd6a0e99c4bdab4e26152ab168178f8",
        "37634576e34043d169cf24bfc0cc2261818dc65b9358d4b9b2e46ab614d0bdda",
    ):
        assert value in source


def test_notebook_uses_a_separate_drive_root_and_step_target():
    source = "\n".join(code_cells())
    assert (
        '/content/drive/MyDrive/CNN-RL-improved/raw-direct-830k-seed0' in source
    )
    assert "TARGET_TIMESTEPS = 830_000" in source
    assert "scale-aware-cnn-6h-seed0" not in source


def test_training_command_is_the_v7_command_without_the_extractor_transfer():
    source = training_cell()
    for term in (
        '"--extractor", "raw-direct"',
        '"--state-context", "full"',
        '"--seed", "0"',
        '"--timesteps", str(remaining)',
        '"--lr", "0.0001"',
        '"--lr-schedule", "linear"',
        '"--lr-final", "0.00001"',
        '"--lr-decay-steps", "1000000"',
        '"--n-envs", "8"',
        '"--vec-env", "subproc"',
        '"--n-steps", "120"',
        '"--batch-size", "64"',
        '"--n-epochs", "5"',
        '"--gamma", "1.0"',
        '"--gae-lambda", "0.98"',
        '"--checkpoint-freq", "10000"',
        '"--holdout-eval-freq", "50000"',
        '"--holdout-selection-count", "5"',
        '"--monthly-jitter", "20"',
        '"--empirical-profile-probability", "0.2"',
        '"--eval-scenarios", "./data/fixed_eval_scenarios.json"',
        '"--final-holdout-report"',
        '"--auto-resume"',
        '"--no-export-onnx"',
    ):
        assert term in source, term
    for forbidden in (
        "--max-training-seconds",
        "--wall-clock-heartbeat-seconds",
        "--pretrained-extractor",
        "--pretraining-complete",
        "--require-pretrained-extractor",
        "--freeze-extractor-steps",
        "--extractor-lr-scale",
        "--comparison-config-sha256",
        "--comparison-baseline-sha256",
        "--export-onnx",
    ):
        assert forbidden not in source, forbidden


def test_training_cell_trains_only_the_remaining_budget():
    source = training_cell()
    assert "from train import find_resumable_model, model_num_timesteps" in source
    assert "remaining = max(TARGET_TIMESTEPS - start_timestep, 0)" in source
    assert "if remaining == 0:" in source
    assert "arm_timing.json" in source
    assert "steps_per_second" in source


def test_training_cell_streams_logs_and_reports_checkpoint_progress():
    source = training_cell()
    for term in (
        "PPO_LOG_INTERVAL_SECONDS = 30",
        "subprocess.Popen(",
        "stdout=subprocess.PIPE",
        "stderr=subprocess.STDOUT",
        "bufsize=1",
        "threading.Thread",
        "_stream_child_output",
        ".wait(timeout=PPO_LOG_INTERVAL_SECONDS)",
        "durable_timestep",
        "signal.SIGINT",
        "raise subprocess.CalledProcessError",
    ):
        assert term in source, term
    assert "run_state.json" not in source


def test_notebook_prunes_rolled_back_curve_rows_before_training():
    source = "\n".join(code_cells())
    assert "prune_rolled_back_rows" in source
    assert "pre_prune_backup" in source
    assert source.index("prune_rolled_back_rows") < source.index("ppo_command = [")


def test_readme_links_the_pinned_raw_direct_notebook():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"{RELEASE_TAG}/notebooks/raw_direct_830k.ipynb" in readme
    assert "830,000" in readme or "830000" in readme
