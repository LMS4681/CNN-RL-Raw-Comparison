from __future__ import annotations

import ast
import hashlib
import io
import json
import signal
import subprocess
import types
from pathlib import Path

import pytest


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


def stage2_cell() -> str:
    matches = [
        source for source in code_cells()
        if "ppo_command = [" in source
    ]
    assert len(matches) == 1
    return matches[0]


def live_log_helpers() -> types.SimpleNamespace:
    tree = ast.parse(stage2_cell())
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            selected.append(node)
        elif (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id in {
                    "PPO_LOG_INTERVAL_SECONDS",
                    "CHILD_OUTPUT_TAIL_LINES",
                }
                for target in node.targets
            )
        ):
            selected.append(node)
        elif (
            isinstance(node, ast.FunctionDef)
            and node.name in {
                "_durable_monitor_fields",
                "_stream_child_output",
                "_stop_training_process",
                "_run_training_with_live_logs",
            }
        ):
            selected.append(node)
    namespace: dict = {}
    module = ast.fix_missing_locations(
        ast.Module(body=selected, type_ignores=[])
    )
    exec(compile(module, "<stage2-live-log-helpers>", "exec"), namespace)
    return types.SimpleNamespace(**namespace)


class FakeProcess:
    def __init__(self, outcomes, *, poll_result=None, stdout_text=""):
        self.outcomes = list(outcomes)
        self.poll_result = poll_result
        self.stdout = io.StringIO(stdout_text)
        self.wait_timeouts = []
        self.signals = []
        self.terminated = False
        self.killed = False

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def poll(self):
        return self.poll_result

    def send_signal(self, value):
        self.signals.append(value)

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


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
    assert "scale-aware-cnn-6h-v7" in source
    assert '"--depth", "1"' in source
    assert '"status", "--porcelain"' in source
    assert "requirements-comparison.txt" in source
    assert '"--no-deps"' in source and '"--require-hashes"' in source
    assert "child_torch_snapshot" in source
    assert "new_pip_conflicts" in source


def test_notebook_verifies_inputs_and_durable_stage1_artifacts():
    source = "\n".join(code_cells())
    for value in (
        "cd4e14fc1725a4ff159e59d6874d3602f3b65a06",
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


def test_notebook_resyncs_the_dataset_when_drive_is_more_advanced():
    source = "\n".join(code_cells())
    assert "def dataset_progress_rank(root):" in source
    assert "dataset_progress.json" in source and "dataset_manifest.json" in source
    assert "if drive_rank > local_rank:" in source
    assert "shutil.rmtree(LOCAL_DATASET, ignore_errors=True)" in source
    assert "shutil.copytree(PRETRAINING_DATA_ROOT, LOCAL_DATASET)" in source
    assert "not LOCAL_DATASET.exists()" not in source


def test_notebook_discards_rolled_back_curve_rows_before_stage2():
    source = "\n".join(code_cells())
    assert "prune_rolled_back_rows" in source
    assert "pre_prune_backup" in source
    assert source.index("prune_rolled_back_rows") < source.index("ppo_command = [")
    assert source.index("pretraining.two_stage_smoke") < source.index(
        "prune_rolled_back_rows"
    )


def test_readme_documents_l4_and_new_pinned_notebook_url():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "GPU type: L4" in readme
    assert "six PPO hours" in readme
    assert "scale-aware-cnn-6h-v7/notebooks/improved_cnn_6h.ipynb" in readme


def test_stage2_cell_streams_logs_and_reports_durable_progress():
    source = stage2_cell()

    for term in (
        "PPO_LOG_INTERVAL_SECONDS = 30",
        "subprocess.Popen(",
        "stdout=subprocess.PIPE",
        "stderr=subprocess.STDOUT",
        "text=True",
        "bufsize=1",
        "threading.Thread",
        "_stream_child_output",
        ".wait(timeout=PPO_LOG_INTERVAL_SECONDS)",
        "except subprocess.TimeoutExpired:",
        "run_state.json",
        "durable_timestep",
        "flush=True",
        "signal.SIGINT",
        "raise subprocess.CalledProcessError",
        "_run_training_with_live_logs(",
    ):
        assert term in source

    for forbidden in (
        "subprocess.run(command_to_run",
        "stdout=None",
        "stderr=None",
        "capture_output=True",
    ):
        assert forbidden not in source


def test_stage2_command_includes_complete_immutable_provenance():
    source = stage2_cell()
    for term in (
        '"--comparison-baseline-sha256", PPO_BASELINE_SHA',
        '"--comparison-scenario-sha256", PPO_SCENARIO_SHA256',
        '"--comparison-split-sha256", PPO_SPLIT_SHA256',
        '"--comparison-lock-sha256", PPO_LOCK_SHA256',
    ):
        assert term in source


def test_stage2_resume_selection_matches_v2_contract():
    lines = stage2_cell().splitlines(keepends=True)
    resume_start = next(
        index for index, line in enumerate(lines)
        if line.startswith('state_path = PPO_ROOT / "run_state.json"')
    )
    resume_end = next(
        index for index, line in enumerate(lines)
        if line.startswith("PPO_LOG_INTERVAL_SECONDS = ")
    )
    resume_source = "".join(lines[resume_start:resume_end])

    assert hashlib.sha256(resume_source.encode()).hexdigest() == (
        "9f284be6e10ac774be2b82eaa4525b26eff2d86505961649e38cc04531c920c4"
    )


def test_live_log_helper_inherits_streams_and_reports_durable_state(
    monkeypatch, tmp_path, capsys
):
    helpers = live_log_helpers()
    state_path = tmp_path / "run_state.json"
    state_path.write_text(json.dumps({
        "last_checkpoint_timestep": 12345,
        "completed_training_seconds": 90.5,
        "target_training_seconds": 21600,
        "last_checkpoint_file": "checkpoint.zip",
        "updated_at_utc": "2026-07-30T00:00:00Z",
    }), encoding="utf-8")
    process = FakeProcess(
        [subprocess.TimeoutExpired(["python"], 30), 0],
        stdout_text="native progress line\nnative traceback line\n",
    )
    observed = {}

    def fake_popen(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return process

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    command = ["python", "-u", "train.py"]
    helpers._run_training_with_live_logs(
        command,
        cwd=tmp_path,
        env={"PYTHONUNBUFFERED": "1"},
        state_path=state_path,
    )

    output = capsys.readouterr().out
    assert "native progress line" in output
    assert "native traceback line" in output
    assert "durable_timestep=12345" in output
    assert "recorded_training_seconds=90.5" in output
    assert "return_code=0" in output
    assert observed["command"] is command
    assert observed["kwargs"]["stdout"] is subprocess.PIPE
    assert observed["kwargs"]["stderr"] is subprocess.STDOUT
    assert observed["kwargs"]["text"] is True
    assert observed["kwargs"]["bufsize"] == 1
    assert process.stdout.closed
    assert process.wait_timeouts == [30, 30]


def test_live_log_helper_preserves_output_tail_on_nonzero_exit(
    monkeypatch, tmp_path, capsys
):
    helpers = live_log_helpers()
    process = FakeProcess([7], stdout_text="actual child failure\n")
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    command = ["python", "-u", "train.py"]

    with pytest.raises(subprocess.CalledProcessError) as captured:
        helpers._run_training_with_live_logs(
            command,
            cwd=tmp_path,
            env={"PYTHONUNBUFFERED": "1"},
            state_path=tmp_path / "run_state.json",
        )

    assert captured.value.returncode == 7
    assert captured.value.cmd is command
    assert captured.value.output == "actual child failure\n"
    assert "actual child failure" in capsys.readouterr().out


def test_live_log_helper_repolls_before_reporting_running(
    monkeypatch, tmp_path, capsys
):
    helpers = live_log_helpers()
    process = FakeProcess(
        [subprocess.TimeoutExpired(["python"], 30)],
        poll_result=7,
    )
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)

    with pytest.raises(subprocess.CalledProcessError):
        helpers._run_training_with_live_logs(
            ["python", "-u", "train.py"],
            cwd=tmp_path,
            env={"PYTHONUNBUFFERED": "1"},
            state_path=tmp_path / "run_state.json",
        )

    output = capsys.readouterr().out
    assert "process=running" not in output
    assert "return_code=7" in output
    assert process.wait_timeouts == [30]


def test_live_log_helper_cleans_up_interrupted_child(monkeypatch, tmp_path):
    helpers = live_log_helpers()
    process = FakeProcess([
        KeyboardInterrupt(),
        subprocess.TimeoutExpired(["python"], 30),
        subprocess.TimeoutExpired(["python"], 10),
        -9,
    ])
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)

    with pytest.raises(KeyboardInterrupt):
        helpers._run_training_with_live_logs(
            ["python", "-u", "train.py"],
            cwd=tmp_path,
            env={"PYTHONUNBUFFERED": "1"},
            state_path=tmp_path / "run_state.json",
        )

    assert process.signals == [signal.SIGINT]
    assert process.terminated
    assert process.killed
    assert process.wait_timeouts == [30, 30, 10, None]


def test_durable_monitor_tolerates_missing_and_malformed_state(tmp_path):
    helpers = live_log_helpers()
    state_path = tmp_path / "run_state.json"
    assert helpers._durable_monitor_fields(state_path) == [
        "durable_state=pending"
    ]

    state_path.write_text("{", encoding="utf-8")
    fields = helpers._durable_monitor_fields(state_path)
    assert len(fields) == 1
    assert fields[0].startswith("durable_state=read_error:")


def test_final_cell_is_standalone_read_only_resume_diagnostic():
    final_cell = notebook()["cells"][-1]
    source = "".join(final_cell["source"])

    assert final_cell["cell_type"] == "code"
    assert "=== PPO RESUME DIAGNOSTIC V7 ===" in source
    for term in (
        'Path("/content/CNN-RL-Raw-Comparison")',
        'Path("/content/drive/MyDrive/CNN-RL-improved/scale-aware-cnn-6h-seed0")',
        "run_state.json",
        "run_config.json",
        "last_checkpoint_file",
        "hashlib.sha256",
        "zipfile.is_zipfile",
        'ScaleAwareMaskablePPO.load(checkpoint_path, device="cpu")',
        "model.num_timesteps",
        "verify_pretraining_artifacts",
        "progress_timing.csv",
        "DIAGNOSTIC_STATUS",
    ):
        assert term in source

    for forbidden in (
        "subprocess.run",
        "train.py",
        "pretraining.train_encoder",
        ".write_text(",
        ".write_bytes(",
        ".unlink(",
        ".rename(",
        ".replace(",
        "shutil.rmtree",
    ):
        assert forbidden not in source
