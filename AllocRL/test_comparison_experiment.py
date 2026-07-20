"""Contract tests for the overnight comparison orchestrator."""

from __future__ import annotations

import json
import math
import hashlib
from pathlib import Path

import pytest


def test_production_config_is_strict_and_commands_are_safe(tmp_path: Path):
    from comparison.experiment_runner import (
        ExperimentConfig,
        build_smoke_command,
        build_train_command,
    )

    config = ExperimentConfig.for_test(target_training_seconds_per_arm=1)
    raw = build_train_command("raw_direct", config, output_root=tmp_path, lock_sha256="a" * 64)
    cnn = build_train_command("candidate_cnn", config, output_root=tmp_path, lock_sha256="a" * 64)
    assert raw[0]
    assert raw[raw.index("--extractor") + 1] == "raw-direct"
    assert cnn[cnn.index("--extractor") + 1] == "candidate-cnn"
    assert "--auto-resume" not in raw
    assert build_smoke_command("raw_direct", config, output_root=tmp_path)[1:5] == [
        "smoke_test.py", "--extractor", "raw-direct", "--timesteps"
    ]


def test_loader_rejects_unknown_production_key(tmp_path: Path):
    from comparison.experiment_runner import load_experiment_config

    source = Path(__file__).with_name("configs") / "overnight_seed0.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["surprise"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="keys"):
        load_experiment_config(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_training_seconds_per_arm", 1),
        ("timesteps_ceiling", 1),
        ("checkpoint_freq", 1),
        ("checkpoint_heartbeat_seconds", 1),
        ("holdout_eval_freq", 0),
        ("smoke_timesteps", 1),
        ("seed", True),
        ("learning_rate", math.nan),
        ("batch_size", 64.0),
    ],
)
def test_production_loader_rejects_every_altered_fixed_value(tmp_path: Path, field: str, value: object):
    from comparison.experiment_runner import load_experiment_config

    source = Path(__file__).with_name("configs") / "overnight_seed0.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload[field] = value
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="immutable"):
        load_experiment_config(path)


def test_for_test_only_allows_operational_overrides_and_zero_holdout_frequency():
    from comparison.experiment_runner import ExperimentConfig

    config = ExperimentConfig.for_test(holdout_eval_freq=0, smoke_timesteps=1)
    assert config.holdout_eval_freq == 0
    assert config.smoke_timesteps == 1
    with pytest.raises(ValueError, match="operational"):
        ExperimentConfig.for_test(seed=1)
    with pytest.raises(ValueError, match="non-negative"):
        ExperimentConfig.for_test(holdout_eval_freq=-1)


def test_lease_released_is_reacquirable_but_live_owner_is_untouched(tmp_path: Path):
    from comparison.experiment_runner import LeaseError, _Lease

    now = [100.0]
    first = _Lease(tmp_path, stale_takeover=False, clock=lambda: now[0], interval=999)
    with first:
        original = (tmp_path / "lease.json").read_bytes()
        with pytest.raises(LeaseError):
            with _Lease(tmp_path, stale_takeover=False, clock=lambda: now[0], interval=999):
                pass
        assert (tmp_path / "lease.json").read_bytes() == original
    with _Lease(tmp_path, stale_takeover=False, clock=lambda: now[0], interval=999) as second:
        assert second.acquired


def test_lease_stale_takeover_requires_flag_and_old_token_cannot_release(tmp_path: Path):
    from comparison.experiment_runner import LeaseError, _Lease, atomic_write_json

    atomic_write_json(tmp_path / "lease.json", {
        "token": "old", "pid": 1, "boot_id": "same", "heartbeat_utc": "x",
        "heartbeat_monotonic": 0.0, "status": "active",
    })
    (tmp_path / ".lease.acquire").write_text("old\n", encoding="utf-8")
    with pytest.raises(LeaseError, match="stale"):
        with _Lease(tmp_path, stale_takeover=False, clock=lambda: 1000.0, interval=999):
            pass
    new = _Lease(tmp_path, stale_takeover=True, clock=lambda: 1000.0, interval=999)
    with new:
        old = _Lease(tmp_path, stale_takeover=False, clock=lambda: 1000.0, interval=999)
        old.token = "old"
        with pytest.raises(LeaseError, match="token"):
            old._write("released")
        assert json.loads((tmp_path / "lease.json").read_text(encoding="utf-8"))["token"] == new.token


def test_semantic_preflight_hash_ignores_later_checkpoint_manifest_updates(tmp_path: Path):
    from comparison.artifact_manifest import REQUIRED_ENVIRONMENT_KEYS
    from comparison.experiment_runner import ExperimentConfig, _Runner, atomic_write_json

    runner = _Runner(ExperimentConfig.for_test(), tmp_path, subprocess_runner=lambda *a, **k: None,
                     clock=lambda: 0.0, python_executable=None, archive_timestep_reader=None)
    calls = []
    def preflight_output():
        calls.append("preflight")
        atomic_write_json(tmp_path / "environment.json", {key: None for key in REQUIRED_ENVIRONMENT_KEYS})
        atomic_write_json(tmp_path / "manifest.json", {
            "schema_version": 1, "baseline_sha256": "b", "config_sha256": "c",
            "scenario_sha256": "s", "split_sha256": "p", "lock_sha256": "l", "checkpoints": {},
        })
    runner.run_stage("preflight", preflight_output)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["checkpoints"] = {"raw_direct": {"selected": {"path": "raw_direct/x.sb3"}}}
    atomic_write_json(tmp_path / "manifest.json", manifest)
    runner.run_stage("preflight", lambda: pytest.fail("preflight must remain complete"))
    assert calls == ["preflight"]


def test_malformed_completed_journal_is_rejected_before_it_is_trusted(tmp_path: Path):
    from comparison.experiment_runner import ExperimentConfig, ExperimentIntegrityError, JOURNAL_STAGES, _Runner

    entry = {"status": "complete", "input_sha256": "a" * 64, "output_sha256": None,
             "started_at_utc": "2026-01-01T00:00:00Z", "completed_at_utc": None, "error": None}
    (tmp_path / "stage_journal.json").write_text(json.dumps({stage: entry for stage in JOURNAL_STAGES}), encoding="utf-8")
    runner = _Runner(ExperimentConfig.for_test(), tmp_path, subprocess_runner=lambda *a, **k: None,
                     clock=lambda: 0.0, python_executable=None, archive_timestep_reader=None)
    with pytest.raises(ExperimentIntegrityError, match="invalid stage journal"):
        runner.journal()


@pytest.mark.parametrize("timestep", [None, 1])
def test_smoke_zero_exit_without_a_valid_requested_archive_fails(tmp_path: Path, timestep: int | None):
    from comparison.experiment_runner import ExperimentConfig, ExperimentIntegrityError, ExperimentStageError, _Runner

    config = ExperimentConfig.for_test(smoke_timesteps=2)
    def process(argv, **_kwargs):
        path = tmp_path / "smoke" / "raw_direct" / "raw-direct.sb3"
        path.parent.mkdir(parents=True, exist_ok=True)
        if timestep is not None:
            path.write_bytes(b"archive")
    runner = _Runner(config, tmp_path, subprocess_runner=process, clock=lambda: 0,
                     python_executable="python", archive_timestep_reader=lambda _: timestep)
    with pytest.raises(ExperimentStageError, match="readable"):
        runner.smoke("raw_direct")


def test_smoke_marker_records_verified_archive_identity(tmp_path: Path):
    from comparison.artifact_manifest import sha256_file
    from comparison.experiment_runner import ExperimentConfig, _Runner

    config = ExperimentConfig.for_test(smoke_timesteps=2)
    def process(argv, **_kwargs):
        path = tmp_path / "smoke" / "raw_direct" / "raw-direct.sb3"
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"archive")
    runner = _Runner(config, tmp_path, subprocess_runner=process, clock=lambda: 0,
                     python_executable="python", archive_timestep_reader=lambda _: 2)
    runner.smoke("raw_direct")
    marker = json.loads((tmp_path / "smoke" / "raw_direct" / "runner_verified.json").read_text())
    assert marker == {"arm": "raw_direct", "config_sha256": config.config_sha256,
                      "path": "raw-direct.sb3", "sha256": sha256_file(tmp_path / "smoke" / "raw_direct" / "raw-direct.sb3"), "timestep": 2}


def test_run_entrypoint_orders_both_smokes_before_raw_training(tmp_path: Path, monkeypatch):
    """Cheap harness: exercise the public entrypoint while model work is injected."""
    from comparison import experiment_runner as runner_module

    config = runner_module.ExperimentConfig.for_test()
    observed: list[str] = []
    monkeypatch.setattr(runner_module._Runner, "run_stage", lambda self, name, action: (observed.append(name), action()))
    monkeypatch.setattr(runner_module._Runner, "preflight", lambda self: None)
    monkeypatch.setattr(runner_module._Runner, "smoke", lambda self, arm: None)
    monkeypatch.setattr(runner_module._Runner, "train", lambda self, arm: None)
    monkeypatch.setattr(runner_module._Runner, "evaluate_arm", lambda self, arm: None)
    monkeypatch.setattr(runner_module._Runner, "common_evaluation", lambda self: None)
    monkeypatch.setattr(runner_module, "write_complete_report", lambda root: None)
    monkeypatch.setattr(runner_module._Runner, "integrity", lambda self: None)
    monkeypatch.setattr(runner_module._Runner, "provenance", lambda self: {"lock_sha256": "a" * 64})
    runner_module.run_overnight_experiment(config, tmp_path, lease_interval_seconds=999)
    assert observed == list(runner_module.JOURNAL_STAGES)
    assert observed.index("smoke_candidate_cnn") < observed.index("train_raw_direct")


def _write_runner_state(root: Path, config, *, status="complete", target=10, seconds=10,
                        config_sha=None, checkpoint_bytes=b"state", timestep=7) -> Path:
    checkpoint = root / "checkpoints" / "model_7_g1.sb3"
    checkpoint.parent.mkdir(parents=True, exist_ok=True); checkpoint.write_bytes(checkpoint_bytes)
    payload = {
        "schema_version": 1, "target_training_seconds": target,
        "completed_training_seconds": seconds, "last_checkpoint_timestep": 7,
        "last_regular_checkpoint_timestep": 0, "last_checkpoint_file": checkpoint.name,
        "last_checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "config_sha256": config_sha or config.config_sha256, "generation": 1,
        "restart_count": 0, "max_unrecorded_seconds": 0.0, "status": status,
        "started_at_utc": "2026-01-01T00:00:00Z", "updated_at_utc": "2026-01-01T00:00:10Z",
        "completed_at_utc": "2026-01-01T00:00:10Z" if status == "complete" else None,
    }
    (root / "run_state.json").write_text(json.dumps(payload), encoding="utf-8")
    return checkpoint


@pytest.mark.parametrize("kind", ["missing", "running", "wrong_target", "short", "wrong_config", "wrong_sha", "wrong_timestep"])
def test_train_rejects_every_unverified_or_incomplete_completion_state(tmp_path: Path, kind: str, monkeypatch):
    from comparison.experiment_runner import ExperimentConfig, ExperimentIntegrityError, ExperimentStageError, _Runner

    config = ExperimentConfig.for_test(target_training_seconds_per_arm=10)
    root = tmp_path / "raw_direct"; calls = []
    runner = _Runner(config, tmp_path, subprocess_runner=lambda argv, **kwargs: calls.append(argv), clock=lambda: 0,
                     python_executable="python", archive_timestep_reader=lambda _: 7)
    monkeypatch.setattr(runner, "provenance", lambda: {"lock_sha256": "a" * 64})
    if kind != "missing":
        checkpoint = _write_runner_state(root, config,
            status="running" if kind == "running" else "complete",
            target=9 if kind == "wrong_target" else 10,
            seconds=9 if kind == "short" else 10,
            config_sha="b" * 64 if kind == "wrong_config" else None)
        if kind == "wrong_sha": checkpoint.write_bytes(b"tampered")
        if kind == "wrong_timestep": runner.archive_reader = lambda _: 8
    with pytest.raises((ExperimentStageError, ExperimentIntegrityError, ValueError, FileNotFoundError)):
        runner.train("raw_direct")


def test_train_valid_complete_state_skips_subprocess(tmp_path: Path, monkeypatch):
    from comparison.experiment_runner import ExperimentConfig, _Runner
    config = ExperimentConfig.for_test(target_training_seconds_per_arm=10)
    _write_runner_state(tmp_path / "raw_direct", config)
    calls = []
    runner = _Runner(config, tmp_path, subprocess_runner=lambda argv, **kwargs: calls.append(argv), clock=lambda: 0,
                     python_executable="python", archive_timestep_reader=lambda _: 7)
    monkeypatch.setattr(runner, "provenance", lambda: {"lock_sha256": "a" * 64})
    runner.train("raw_direct")
    assert calls == []


def test_train_resume_uses_only_state_named_checkpoint_not_higher_orphan(tmp_path: Path, monkeypatch):
    from comparison.experiment_runner import ExperimentConfig, _Runner
    config = ExperimentConfig.for_test(target_training_seconds_per_arm=10)
    named = _write_runner_state(tmp_path / "raw_direct", config, status="running", seconds=1)
    orphan = tmp_path / "raw_direct" / "checkpoints" / "model_999_g9.sb3"; orphan.write_bytes(b"orphan")
    commands = []
    def process(argv, **_kwargs):
        commands.append(argv)
        _write_runner_state(tmp_path / "raw_direct", config)
    runner = _Runner(config, tmp_path, subprocess_runner=process, clock=lambda: 0,
                     python_executable="python", archive_timestep_reader=lambda path: 7 if path == named else 999)
    monkeypatch.setattr(runner, "provenance", lambda: {"lock_sha256": "a" * 64})
    runner.train("raw_direct")
    assert commands[0][commands[0].index("--resume-from") + 1] == str(named)
    assert str(orphan) not in commands[0]
    assert "--auto-resume" not in commands[0] and "--final-holdout-report" not in commands[0]


@pytest.mark.parametrize("stage", ["train_raw_direct", "evaluate_raw_direct", "train_candidate_cnn", "evaluate_candidate_cnn", "evaluate_common_step", "build_report", "integrity_verification"])
def test_run_stage_failed_entry_retries_without_changing_prior_verified_hashes(tmp_path: Path, stage: str, monkeypatch):
    from comparison.experiment_runner import ExperimentConfig, ExperimentStageError, JOURNAL_STAGES, _Runner, _journal_entry
    hashes = {name: "a" * 64 for name in JOURNAL_STAGES}
    runner = _Runner(ExperimentConfig.for_test(), tmp_path, subprocess_runner=lambda *a, **k: None, clock=lambda: 0,
                     python_executable=None, archive_timestep_reader=None, output_hasher=lambda name: hashes[name])
    prior = JOURNAL_STAGES[:JOURNAL_STAGES.index(stage)]
    journal = {name: _journal_entry("complete", input_sha256="b" * 64, output_sha256=hashes[name], started_at_utc="2026-01-01T00:00:00Z", completed_at_utc="2026-01-01T00:00:01Z") if name in prior else _journal_entry() for name in JOURNAL_STAGES}
    runner.save_journal(journal)
    monkeypatch.setattr(runner, "input_hash", lambda name, data: "b" * 64)
    monkeypatch.setattr(runner, "stage_path", lambda name: tmp_path)
    calls = []
    def action():
        calls.append(1)
        if len(calls) == 1: raise RuntimeError("once")
    with pytest.raises(ExperimentStageError): runner.run_stage(stage, action)
    failed = runner.journal(); assert failed[stage]["status"] == "failed"
    assert all(failed[name]["output_sha256"] == hashes[name] for name in prior)
    runner.run_stage(stage, action)
    assert runner.journal()[stage]["status"] == "complete" and len(calls) == 2


@pytest.mark.parametrize("stage", ["train_raw_direct", "build_report"])
def test_run_stage_keyboard_interrupt_is_interrupted_then_retries(tmp_path: Path, stage: str, monkeypatch):
    from comparison.experiment_runner import ExperimentConfig, JOURNAL_STAGES, _Runner
    runner = _Runner(ExperimentConfig.for_test(), tmp_path, subprocess_runner=lambda *a, **k: None, clock=lambda: 0,
                     python_executable=None, archive_timestep_reader=None, output_hasher=lambda _: "a" * 64)
    monkeypatch.setattr(runner, "input_hash", lambda *args: "b" * 64); monkeypatch.setattr(runner, "stage_path", lambda _: tmp_path)
    with pytest.raises(KeyboardInterrupt): runner.run_stage(stage, lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
    entry = runner.journal()[stage]; assert entry["status"] == "interrupted" and entry["completed_at_utc"] and entry["error"] == "interrupted"
    runner.run_stage(stage, lambda: None); assert runner.journal()[stage]["status"] == "complete"


def test_stale_valid_in_progress_becomes_interrupted_and_retries(tmp_path: Path, monkeypatch):
    from comparison.experiment_runner import ExperimentConfig, JOURNAL_STAGES, _Runner, _journal_entry
    runner = _Runner(ExperimentConfig.for_test(), tmp_path, subprocess_runner=lambda *a, **k: None, clock=lambda: 0,
                     python_executable=None, archive_timestep_reader=None, output_hasher=lambda _: "a" * 64)
    stale = _journal_entry("in_progress", input_sha256="a" * 64, started_at_utc="2026-01-01T00:00:00Z")
    runner.save_journal({name: stale if name == "evaluate_raw_direct" else _journal_entry() for name in JOURNAL_STAGES})
    normalized = runner.journal()["evaluate_raw_direct"]
    assert normalized["status"] == "interrupted" and normalized["input_sha256"] == "a" * 64 and normalized["started_at_utc"] == "2026-01-01T00:00:00Z" and normalized["completed_at_utc"]
    monkeypatch.setattr(runner, "input_hash", lambda *args: "a" * 64); monkeypatch.setattr(runner, "stage_path", lambda _: tmp_path)
    runner.run_stage("evaluate_raw_direct", lambda: None)
    assert runner.journal()["evaluate_raw_direct"]["status"] == "complete"


@pytest.mark.parametrize("field,value", [("input_sha256", None), ("input_sha256", "bad"), ("started_at_utc", None), ("started_at_utc", "bad"), ("output_sha256", "a" * 64), ("completed_at_utc", "2026-01-01T00:00:01Z"), ("error", "premature")])
def test_malformed_in_progress_is_rejected_not_normalized(tmp_path: Path, field: str, value: object):
    from comparison.experiment_runner import ExperimentConfig, ExperimentIntegrityError, JOURNAL_STAGES, _Runner, _journal_entry
    entry = _journal_entry("in_progress", input_sha256="a" * 64, started_at_utc="2026-01-01T00:00:00Z"); entry[field] = value
    runner = _Runner(ExperimentConfig.for_test(), tmp_path, subprocess_runner=lambda *a, **k: None, clock=lambda: 0, python_executable=None, archive_timestep_reader=None)
    runner.save_journal({name: entry if name == "evaluate_raw_direct" else _journal_entry() for name in JOURNAL_STAGES})
    with pytest.raises(ExperimentIntegrityError, match="invalid stage journal"): runner.journal()
