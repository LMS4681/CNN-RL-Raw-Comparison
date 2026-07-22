from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "AllocRL" / "configs" / "improved_cnn_6h_seed0.json"
NOTEBOOK_PATH = ROOT / "notebooks" / "improved_cnn_6h.ipynb"

EXPECTED_CONFIG = {
    "extractor": "candidate-cnn",
    "state_context": "full",
    "seed": 0,
    "timesteps_ceiling": 2_000_000_000,
    "max_training_seconds": 21_600,
    "learning_rate": 0.0001,
    "learning_rate_schedule": "linear",
    "final_learning_rate": 0.00001,
    "learning_rate_decay_steps": 1_000_000,
    "require_pretrained_extractor": True,
    "freeze_extractor_steps": 50_000,
    "extractor_learning_rate_scale": 0.1,
    "n_envs": 8,
    "vec_env": "subproc",
    "n_steps": 120,
    "batch_size": 64,
    "n_epochs": 5,
    "gamma": 1.0,
    "gae_lambda": 0.98,
    "checkpoint_freq": 10_000,
    "wall_clock_heartbeat_seconds": 300,
    "holdout_eval_freq": 50_000,
    "holdout_selection_count": 5,
    "monthly_jitter": 20,
    "empirical_profile_probability": 0.2,
    "device": "cuda",
    "export_onnx": False,
}


def notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def code_cells() -> list[str]:
    return [
        "".join(cell["source"])
        for cell in notebook()["cells"]
        if cell["cell_type"] == "code"
    ]


def test_frozen_six_hour_config_is_exact_and_self_hashable():
    assert json.loads(CONFIG_PATH.read_text(encoding="utf-8")) == EXPECTED_CONFIG
    assert len(hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()) == 64
    assert EXPECTED_CONFIG["n_envs"] * EXPECTED_CONFIG["n_steps"] == 960
    assert 960 % EXPECTED_CONFIG["batch_size"] == 0


def test_notebook_is_clean_and_orders_both_stages_before_ppo():
    data = notebook()
    assert data["nbformat"] == 4
    assert all(cell.get("outputs", []) == [] for cell in data["cells"])
    assert all(cell.get("execution_count") is None for cell in data["cells"])
    source = "\n".join(code_cells())
    assert source.index("drive.mount") < source.index("NVIDIA L4")
    assert source.index("pretraining.dataset") < source.index(
        "pretraining.train_encoder"
    )
    assert source.index("verify_pretraining_artifacts") < source.index(
        "pretraining.two_stage_smoke"
    )
    assert source.index("pretraining.two_stage_smoke") < source.index(
        "train.py"
    )


def test_notebook_requires_l4_and_sufficient_cpu_ram_and_drive():
    source = "\n".join(code_cells())
    assert "torch.cuda.is_available()" in source
    assert "torch.cuda.get_device_name(0)" in source
    assert 'ACCEPTED_GPU_NAMES = {"NVIDIA L4"}' in source
    assert "os.cpu_count()" in source and ">= 2" in source
    assert "psutil.virtual_memory" in source
    assert "shutil.disk_usage" in source
    assert "raise RuntimeError" in source
    assert "nvidia-smi" in source


def test_notebook_uses_clean_immutable_checkout_and_preserves_torch():
    source = "\n".join(code_cells())
    assert "https://github.com/LMS4681/CNN-RL-Raw-Comparison.git" in source
    assert "scale-aware-cnn-6h-v1" in source
    assert '"--depth", "1"' in source
    assert '"status", "--porcelain"' in source
    assert "requirements-comparison.txt" in source
    assert '"--no-deps"' in source and '"--require-hashes"' in source
    assert "child_torch_snapshot" in source
    assert "new_pip_conflicts" in source


def test_notebook_verifies_inputs_and_durable_stage1_artifacts():
    source = "\n".join(code_cells())
    for value in (
        "913cac9046dec8164ef65da60275522f7127de5ea775b1c5a6b6aac255716271",
        "601bd6143ed8890577e5ff34921241d36fd6a0e99c4bdab4e26152ab168178f8",
        "37634576e34043d169cf24bfc0cc2261818dc65b9358d4b9b2e46ab614d0bdda",
        "candidate_pretrain_seed0.json",
        "improved_cnn_6h_seed0.json",
    ):
        assert value in source
    assert "/content/pretraining-dataset" in source
    assert "dataset_manifest.json" in source
    assert '"--progress-mirror-dir"' in source
    assert "PRETRAINING_COMPLETE.json" in source
    assert "candidate_encoder_pretrained.pt" in source
    assert "verify_pretraining_artifacts" in source
    assert "if PRETRAIN_COMPLETE.is_file()" in source
    assert "verified_stage1.receipt.config_sha256" in source
    assert "verified_stage1.receipt.manifest_sha256" in source
    assert "copytree" in source or "copy2" in source


def test_notebook_runs_32_step_smoke_and_exact_stage2_command():
    source = "\n".join(code_cells())
    assert "sys.executable" in source and '"-m"' in source
    assert "pretraining.two_stage_smoke" in source
    assert '"--timesteps", "32"' in source
    for term in (
        '"--timesteps", "2000000000"',
        '"--max-training-seconds", "21600"',
        '"--lr-schedule", "linear"',
        '"--lr-final", "0.00001"',
        '"--lr-decay-steps", "1000000"',
        '"--n-envs", "8"',
        '"--vec-env", "subproc"',
        '"--n-steps", "120"',
        '"--batch-size", "64"',
        '"--n-epochs", "5"',
        '"--require-pretrained-extractor"',
        '"--freeze-extractor-steps", "50000"',
        '"--extractor-lr-scale", "0.1"',
        '"--auto-resume"',
        '"--no-export-onnx"',
    ):
        assert term in source
    assert '"--export-onnx"' not in source
    assert "PYTHONUNBUFFERED" in source


def test_notebook_uses_separate_drive_roots_and_displays_receipts():
    source = "\n".join(code_cells())
    experiment = (
        "/content/drive/MyDrive/CNN-RL-improved/"
        "scale-aware-cnn-6h-seed0"
    )
    assert experiment in source
    assert 'PRETRAINING_ROOT = EXPERIMENT_ROOT / "pretraining"' in source
    assert 'PPO_ROOT = EXPERIMENT_ROOT / "ppo"' in source
    for name in (
        "pretraining_metrics.json",
        "run_state.json",
        "progress_timing.csv",
        "training_completion.json",
        "evaluation_csv.csv",
        "evaluation_scenarios.csv",
    ):
        assert name in source
    assert "resume command" in source.lower()


def test_readme_documents_l4_and_new_pinned_notebook_url():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "GPU type: L4" in readme
    assert "six PPO hours" in readme
    assert "scale-aware-cnn-6h-v1/notebooks/improved_cnn_6h.ipynb" in readme
