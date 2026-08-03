"""Contract tests for the raw-direct six-hour comparison notebook."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "raw_direct_6h.ipynb"
CONFIG_PATH = ROOT / "AllocRL" / "configs" / "raw_direct_6h_seed0.json"
IMPROVED_CONFIG_PATH = ROOT / "AllocRL" / "configs" / "improved_cnn_6h_seed0.json"
RELEASE_TAG = "raw-direct-6h-v1"

SHARED_BUDGET_KEYS = (
    "state_context",
    "seed",
    "timesteps_ceiling",
    "max_training_seconds",
    "learning_rate",
    "learning_rate_schedule",
    "final_learning_rate",
    "learning_rate_decay_steps",
    "n_envs",
    "vec_env",
    "n_steps",
    "batch_size",
    "n_epochs",
    "gamma",
    "gae_lambda",
    "checkpoint_freq",
    "wall_clock_heartbeat_seconds",
    "holdout_eval_freq",
    "holdout_selection_count",
    "monthly_jitter",
    "empirical_profile_probability",
    "device",
    "export_onnx",
)


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


def test_arm_config_differs_from_the_cnn_arm_only_by_the_extractor():
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    improved = json.loads(IMPROVED_CONFIG_PATH.read_text(encoding="utf-8"))

    assert raw["extractor"] == "raw-direct"
    assert raw["require_pretrained_extractor"] is False
    assert "freeze_extractor_steps" not in raw
    assert "extractor_learning_rate_scale" not in raw
    for key in SHARED_BUDGET_KEYS:
        assert raw[key] == improved[key], key


def test_notebook_pins_the_hash_of_its_own_arm_config():
    source = "\n".join(code_cells())
    digest = hashlib.sha256(
        CONFIG_PATH.read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()

    assert f'"configs/raw_direct_6h_seed0.json": "{digest}"' in source
    assert "PPO_CONFIG_SHA256 = expected_hashes" in source
    assert "improved_cnn_6h_seed0.json" not in source


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


def test_notebook_uses_a_separate_drive_root():
    source = "\n".join(code_cells())
    assert "/content/drive/MyDrive/CNN-RL-improved/raw-direct-6h-seed0" in source
    assert "scale-aware-cnn-6h-seed0" not in source
    assert "raw-direct-830k-seed0" not in source


def test_training_command_matches_the_cnn_budget_without_the_transfer():
    source = training_cell()
    for term in (
        '"--extractor", "raw-direct"',
        '"--state-context", "full"',
        '"--seed", "0"',
        '"--timesteps", "2000000000"',
        '"--max-training-seconds", "21600"',
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
        '"--wall-clock-heartbeat-seconds", "300"',
        '"--holdout-eval-freq", "50000"',
        '"--holdout-selection-count", "5"',
        '"--eval-scenarios", "./data/fixed_eval_scenarios.json"',
        '"--final-holdout-report"',
        '"--auto-resume"',
        '"--no-export-onnx"',
    ):
        assert term in source, term
    for forbidden in (
        "--pretrained-extractor",
        "--pretraining-complete",
        "--require-pretrained-extractor",
        "--freeze-extractor-steps",
        "--extractor-lr-scale",
        "--export-onnx",
        "--timesteps\", str(",
    ):
        assert forbidden not in source, forbidden


def test_training_command_carries_every_provenance_hash():
    source = training_cell()
    for term in (
        '"--comparison-config-sha256", PPO_CONFIG_SHA256',
        '"--comparison-baseline-sha256", PPO_BASELINE_SHA',
        '"--comparison-scenario-sha256", PPO_SCENARIO_SHA256',
        '"--comparison-split-sha256", PPO_SPLIT_SHA256',
        '"--comparison-lock-sha256", PPO_LOCK_SHA256',
    ):
        assert term in source, term


def test_training_cell_keeps_the_verified_resume_and_monitor_contract():
    source = training_cell()
    for term in (
        'state_path = PPO_ROOT / "run_state.json"',
        '"--finalize-complete-state"',
        'if state.get("status") == "complete"',
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


def test_notebook_prunes_rolled_back_curve_rows_before_training():
    source = "\n".join(code_cells())
    assert "prune_rolled_back_rows" in source
    assert "pre_prune_backup" in source
    assert source.index("prune_rolled_back_rows") < source.index("ppo_command = [")


def test_readme_links_the_pinned_raw_direct_notebook():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"{RELEASE_TAG}/notebooks/raw_direct_6h.ipynb" in readme
    assert "largest common regular checkpoint" in readme
