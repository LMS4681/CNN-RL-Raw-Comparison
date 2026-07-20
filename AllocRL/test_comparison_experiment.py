"""Contract tests for the overnight comparison orchestrator."""

from __future__ import annotations

import json
import math
import hashlib
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest


def test_production_config_is_strict_and_commands_are_safe(tmp_path: Path):
    from comparison.experiment_runner import (
        ExperimentConfig,
        build_finalize_command,
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
    finalize = build_finalize_command(
        "raw_direct", config, tmp_path / "state.sb3",
        output_root=tmp_path, lock_sha256="a" * 64,
    )
    assert "--finalize-complete-state" in finalize
    assert finalize[finalize.index("--resume-from") + 1] == str(tmp_path / "state.sb3")
    assert "--auto-resume" not in finalize
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
        "token": "old", "pid": 1, "boot_id": "same", "heartbeat_utc": "2026-01-01T00:00:00Z",
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


def test_fresh_orphan_sentinel_refuses_even_explicit_takeover_without_mutation(tmp_path: Path):
    from comparison.experiment_runner import LeaseError, _Lease

    sentinel = tmp_path / ".lease.acquire"; sentinel.write_text("orphan\n", encoding="utf-8")
    before = sentinel.read_bytes()
    with pytest.raises(LeaseError, match="orphan"):
        with _Lease(tmp_path, stale_takeover=True, clock=lambda: 0.0, wall_time=lambda: 100.0, stale_after=10, interval=999):
            pass
    assert sentinel.read_bytes() == before and not (tmp_path / "lease.json").exists()


def test_stale_orphan_requires_explicit_takeover_then_leaves_released_audit(tmp_path: Path):
    from comparison.experiment_runner import LeaseError, _Lease

    sentinel = tmp_path / ".lease.acquire"; sentinel.write_text("orphan\n", encoding="utf-8")
    import os
    os.utime(sentinel, (0, 0))
    with pytest.raises(LeaseError, match="stale"):
        with _Lease(tmp_path, stale_takeover=False, clock=lambda: 0.0, wall_time=lambda: 100.0, stale_after=10, interval=999):
            pass
    with _Lease(tmp_path, stale_takeover=True, clock=lambda: 0.0, wall_time=lambda: 100.0, stale_after=10, interval=999):
        assert (tmp_path / "lease.json").is_file()
    released = json.loads((tmp_path / "lease.json").read_text(encoding="utf-8"))
    assert released["status"] == "released" and not sentinel.exists()


def test_stale_orphan_toctou_change_refuses_without_deleting_new_sentinel(tmp_path: Path, monkeypatch):
    from comparison.experiment_runner import LeaseError, _Lease

    sentinel = tmp_path / ".lease.acquire"; sentinel.write_text("old\n", encoding="utf-8")
    import os
    os.utime(sentinel, (0, 0))
    lease = _Lease(tmp_path, stale_takeover=True, clock=lambda: 0.0, wall_time=lambda: 100.0, stale_after=10, interval=999)
    original = lease._snapshot_sentinel; calls = [0]
    def changed_snapshot():
        calls[0] += 1
        if calls[0] == 2:
            sentinel.write_text("new-owner\n", encoding="utf-8")
            os.utime(sentinel, (1, 1))
        return original()
    monkeypatch.setattr(lease, "_snapshot_sentinel", changed_snapshot)
    with pytest.raises(LeaseError, match="changed"):
        with lease: pass
    assert sentinel.read_text(encoding="utf-8") == "new-owner\n" and not (tmp_path / "lease.json").exists()


@pytest.mark.parametrize("payload", [
    "[]", "{\"token\":\"x\",\"token\":\"y\"}", "{}",
    '{"token":"","pid":1,"boot_id":"boot","heartbeat_utc":"2026-01-01T00:00:00Z","heartbeat_monotonic":0,"status":"active"}',
    '{"token":true,"pid":1,"boot_id":"boot","heartbeat_utc":"2026-01-01T00:00:00Z","heartbeat_monotonic":0,"status":"active"}',
    '{"token":"x","pid":true,"boot_id":"boot","heartbeat_utc":"2026-01-01T00:00:00Z","heartbeat_monotonic":0,"status":"active"}',
    '{"token":"x","pid":"1","boot_id":"boot","heartbeat_utc":"2026-01-01T00:00:00Z","heartbeat_monotonic":0,"status":"active"}',
    '{"token":"x","pid":1,"boot_id":"","heartbeat_utc":"2026-01-01T00:00:00Z","heartbeat_monotonic":0,"status":"active"}',
    '{"token":"x","pid":1,"boot_id":1,"heartbeat_utc":"2026-01-01T00:00:00Z","heartbeat_monotonic":0,"status":"active"}',
    '{"token":"x","pid":1,"boot_id":"boot","heartbeat_utc":"bad","heartbeat_monotonic":0,"status":"active"}',
    '{"token":"x","pid":1,"boot_id":"boot","heartbeat_utc":0,"heartbeat_monotonic":0,"status":"active"}',
    '{"token":"x","pid":1,"boot_id":"boot","heartbeat_utc":"2026-01-01T00:00:00Z","heartbeat_monotonic":"0","status":"active"}',
    '{"token":"x","pid":1,"boot_id":"boot","heartbeat_utc":"2026-01-01T00:00:00Z","heartbeat_monotonic":-1,"status":"active"}',
    '{"token":"x","pid":1,"boot_id":"boot","heartbeat_utc":"2026-01-01T00:00:00Z","heartbeat_monotonic":0,"status":"unknown"}',
    '{"token":"x","pid":1,"boot_id":"boot","heartbeat_utc":"2026-01-01T00:00:00Z","heartbeat_monotonic":0,"status":true}',
])
def test_malformed_lease_json_fails_closed_without_root_mutation(tmp_path: Path, payload: str):
    from comparison.experiment_runner import LeaseError, _Lease

    path = tmp_path / "lease.json"; path.write_text(payload, encoding="utf-8"); before = path.read_bytes()
    with pytest.raises(LeaseError, match="lease"):
        with _Lease(tmp_path, stale_takeover=True, clock=lambda: 10.0, wall_time=lambda: 10.0, interval=999):
            pass
    assert path.read_bytes() == before and not (tmp_path / ".lease.acquire").exists()


def test_valid_released_lease_reacquires_immediately_and_active_foreign_boot_needs_takeover(tmp_path: Path):
    from comparison.experiment_runner import LeaseError, _Lease, atomic_write_json

    payload = {"token": "released-token", "pid": 1, "boot_id": "other-boot", "heartbeat_utc": "1970-01-01T00:00:00Z", "heartbeat_monotonic": 0.0, "status": "released"}
    atomic_write_json(tmp_path / "lease.json", payload); (tmp_path / ".lease.acquire").write_text("released-token\n", encoding="utf-8")
    with _Lease(tmp_path, stale_takeover=False, clock=lambda: 0.0, wall_time=lambda: 0.0, interval=999) as lease:
        assert lease.acquired
    payload["status"] = "active"; atomic_write_json(tmp_path / "lease.json", payload); (tmp_path / ".lease.acquire").write_text("released-token\n", encoding="utf-8")
    with pytest.raises(LeaseError, match="stale"):
        with _Lease(tmp_path, stale_takeover=False, clock=lambda: 1000.0, wall_time=lambda: 1000.0, interval=999): pass


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
            "scenario_sha256": "s", "split_sha256": "p", "lock_sha256": "l",
            "comparison_git_sha": "a" * 40, "comparison_git_dirty": False,
            "checkpoints": {},
        })
    runner.run_stage("preflight", preflight_output)
    before = runner.output_hash("preflight")
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["checkpoints"] = {"raw_direct": {"selected": {"path": "raw_direct/x.sb3"}}}
    atomic_write_json(tmp_path / "manifest.json", manifest)
    runner.run_stage("preflight", lambda: pytest.fail("preflight must remain complete"))
    assert calls == ["preflight"] and runner.output_hash("preflight") == before
    manifest["comparison_git_sha"] = "b" * 40
    atomic_write_json(tmp_path / "manifest.json", manifest)
    assert runner.output_hash("preflight") != before


def _scenario_document(seeds=range(1000, 1020)):
    return {
        "schema_version": 3,
        "metadata": {},
        "scenarios": [
            {"seed": seed, "source": "test", "blocks": [], "workspaces": []}
            for seed in seeds
        ],
    }


def _preflight_runner(tmp_path: Path, monkeypatch, *, scenario_payload=None):
    from comparison import experiment_runner as runner_module
    from comparison.artifact_manifest import sha256_file
    from comparison.experiment_runner import ExperimentConfig, _Runner

    base = tmp_path / "allocrl"
    data = base / "data"
    data.mkdir(parents=True)
    scenario = data / "fixed_eval_scenarios.json"
    scenario.write_text(json.dumps(_scenario_document() if scenario_payload is None else scenario_payload), encoding="utf-8")
    split = data / "data_split_manifest.json"
    split.write_text("split", encoding="utf-8")
    lock = base / "requirements-comparison.txt"
    lock.write_text("lock", encoding="utf-8")
    config = replace(
        ExperimentConfig.for_test(),
        fixed_scenarios_sha256=sha256_file(scenario),
        split_manifest_sha256=sha256_file(split),
    )
    monkeypatch.setattr(runner_module, "_allocrl_dir", lambda: base)
    return runner_module, _Runner(
        config, tmp_path / "output", subprocess_runner=lambda *args, **kwargs: None,
        clock=lambda: 0.0, python_executable=None, archive_timestep_reader=None,
        runner_command=["C:/Program Files/Python/python.exe", "runner.py", "--output-root", "C:/Drive Folder/output root"],
    )


def _valid_root_environment(provenance, *, command=None):
    from comparison.artifact_manifest import REQUIRED_ENVIRONMENT_KEYS

    environment = {key: None for key in REQUIRED_ENVIRONMENT_KEYS}
    environment.update({
        "captured_at_utc": "2026-07-21T00:00:00Z",
        "command": list(command or ["python", "runner.py"]),
        "python_version": "3.12",
        "platform": "test-platform",
        "comparison_git_sha": "a" * 40,
        "comparison_git_dirty": False,
        "vm_boot_id": "boot-id",
        "torch_version": "2.0",
        "cuda_version": None,
        "cudnn_version": None,
        "resolved_device": "cpu",
        "gpu_name": None,
        "gpu_uuid": None,
        "gpu_total_memory_bytes": None,
        "cpu_count": 1,
        "process_id": 1,
        "pip_freeze": [],
    })
    environment.update(provenance)
    return environment


@pytest.mark.parametrize("case", ["missing", "hash_mismatch", "malformed", "nineteen", "duplicate", "out_of_order"])
def test_preflight_rejects_invalid_fixed_scenarios_before_environment_capture(tmp_path: Path, monkeypatch, case: str):
    from comparison.experiment_runner import ExperimentIntegrityError
    from comparison.artifact_manifest import sha256_file

    payloads = {
        "malformed": "{not JSON",
        "nineteen": _scenario_document(range(1000, 1019)),
        "duplicate": _scenario_document([*range(1000, 1019), 1018]),
        "out_of_order": _scenario_document([1001, 1000, *range(1002, 1020)]),
    }
    runner_module, runner = _preflight_runner(tmp_path, monkeypatch, scenario_payload=payloads.get(case))
    scenario = runner_module._allocrl_dir() / "data" / "fixed_eval_scenarios.json"
    if case == "missing":
        scenario.unlink()
    elif case == "hash_mismatch":
        runner.config = replace(runner.config, fixed_scenarios_sha256="f" * 64)
    elif case == "malformed":
        runner.config = replace(runner.config, fixed_scenarios_sha256=sha256_file(scenario))
    monkeypatch.setattr(runner_module, "collect_environment", lambda *args, **kwargs: pytest.fail("environment capture must follow scenario validation"))
    with pytest.raises(ExperimentIntegrityError):
        runner.preflight()


def test_preflight_records_actual_runner_command_and_strict_manifest_provenance(tmp_path: Path, monkeypatch):
    runner_module, runner = _preflight_runner(tmp_path, monkeypatch)
    provenance = runner.provenance()
    seen = []
    monkeypatch.setattr(runner_module, "collect_environment", lambda command, supplied: (seen.append((command, supplied)), _valid_root_environment(supplied, command=command))[1])

    runner.preflight()

    environment = json.loads((runner.root / "environment.json").read_text(encoding="utf-8"))
    manifest = json.loads((runner.root / "manifest.json").read_text(encoding="utf-8"))
    assert seen == [(runner.runner_command, provenance)]
    assert environment["command"] == ["C:/Program Files/Python/python.exe", "runner.py", "--output-root", "C:/Drive Folder/output root"]
    assert manifest == {
        "schema_version": 1, **provenance, "comparison_git_sha": "a" * 40,
        "comparison_git_dirty": False, "checkpoints": {},
    }


@pytest.mark.parametrize("mutation", [
    lambda environment: environment.pop("command"),
    lambda environment: environment.update(comparison_git_sha="A" * 40),
    lambda environment: environment.update(comparison_git_dirty=True),
    lambda environment: environment.update(baseline_sha256="b" * 40),
    lambda environment: environment.update(command="python runner.py"),
    lambda environment: environment.update(resolved_device="cuda:0", gpu_name=None, gpu_uuid="GPU-a", gpu_total_memory_bytes=1),
])
def test_preflight_rejects_root_environment_schema_types_and_provenance(tmp_path: Path, monkeypatch, mutation):
    from comparison.experiment_runner import ExperimentIntegrityError

    runner_module, runner = _preflight_runner(tmp_path, monkeypatch)
    monkeypatch.setattr(runner_module, "collect_environment", lambda command, provenance: (mutation(environment := _valid_root_environment(provenance, command=command)), environment)[1])
    with pytest.raises(ExperimentIntegrityError):
        runner.preflight()


@pytest.mark.parametrize("environment", [
    {"resolved_device": "cpu", "gpu_name": None, "gpu_uuid": None, "gpu_total_memory_bytes": None},
    {"resolved_device": "cuda:0", "gpu_name": "GPU", "gpu_uuid": None, "gpu_total_memory_bytes": 1},
])
def test_production_preflight_requires_real_cuda_identity(tmp_path: Path, monkeypatch, environment):
    from comparison.experiment_runner import ExperimentIntegrityError

    runner_module, runner = _preflight_runner(tmp_path, monkeypatch)
    runner.config = replace(runner.config, production_loaded=True)
    monkeypatch.setattr(runner_module, "collect_environment", lambda command, provenance: _valid_root_environment(provenance, command=command) | environment)
    with pytest.raises(ExperimentIntegrityError, match="production preflight requires CUDA"):
        runner.preflight()


def _integrity_fixture(tmp_path: Path, monkeypatch):
    from comparison.artifact_manifest import sha256_file
    from comparison.wall_clock_callback import atomic_write_json

    runner_module, runner = _preflight_runner(tmp_path, monkeypatch)
    root = runner.root
    root.mkdir(parents=True)
    provenance = runner.provenance()
    environment = _valid_root_environment(provenance, command=runner.runner_command)
    refs = {}
    timesteps = {}
    for arm in ("raw_direct", "candidate_cnn"):
        final = _write_runner_state(root / arm, runner.config)
        selected = root / arm / "best_model.sb3"
        selected.write_bytes(f"{arm}-selected".encode())
        common = root / arm / "checkpoints" / "model_8_g1.sb3"
        common.write_bytes(f"{arm}-common".encode())
        refs[arm] = {
            "selected": {"path": selected.relative_to(root).as_posix(), "label": "best_model", "sha256": sha256_file(selected), "timestep": 9},
            "final": {"path": final.relative_to(root).as_posix(), "label": "final", "sha256": sha256_file(final), "timestep": 7},
            "common": {"path": common.relative_to(root).as_posix(), "label": "common_step", "sha256": sha256_file(common), "timestep": 8},
        }
        timesteps.update({str(final.resolve()): 7, str(selected.resolve()): 9, str(common.resolve()): 8})
        (root / arm / "environment_segments.jsonl").write_text(json.dumps(environment) + "\n", encoding="utf-8")
    manifest = {"schema_version": 1, **provenance, "comparison_git_sha": environment["comparison_git_sha"], "comparison_git_dirty": False, "checkpoints": refs}
    atomic_write_json(root / "manifest.json", manifest)
    atomic_write_json(root / "environment.json", environment)
    runner.archive_reader = lambda path: timesteps.get(str(path.resolve()))
    return runner_module, runner, manifest, environment


def test_checkpoint_manifest_validator_accepts_exact_honest_refs(tmp_path: Path, monkeypatch):
    runner_module, runner, manifest, _ = _integrity_fixture(tmp_path, monkeypatch)
    runner_module._validate_checkpoint_manifest(runner.root, manifest, runner.archive_reader)


@pytest.mark.parametrize("mutation", [
    lambda manifest, root: manifest["checkpoints"].pop("raw_direct"),
    lambda manifest, root: manifest["checkpoints"].update(extra_arm={}),
    lambda manifest, root: manifest["checkpoints"]["raw_direct"].pop("selected"),
    lambda manifest, root: manifest["checkpoints"]["raw_direct"].update(extra_ref={}),
    lambda manifest, root: manifest["checkpoints"]["raw_direct"]["selected"].update(extra=True),
    lambda manifest, root: manifest["checkpoints"]["raw_direct"]["selected"].update(label="final"),
    lambda manifest, root: manifest["checkpoints"]["raw_direct"]["selected"].update(path="C:/outside.sb3"),
    lambda manifest, root: manifest["checkpoints"]["raw_direct"]["selected"].update(path="../outside.sb3"),
    lambda manifest, root: manifest["checkpoints"]["raw_direct"]["selected"].update(path="raw_direct/missing.sb3"),
    lambda manifest, root: manifest["checkpoints"]["raw_direct"]["selected"].update(sha256="f" * 64),
    lambda manifest, root: manifest["checkpoints"]["raw_direct"]["selected"].update(timestep=10),
    lambda manifest, root: manifest["checkpoints"]["raw_direct"]["final"].update(path=manifest["checkpoints"]["raw_direct"]["common"]["path"], sha256=manifest["checkpoints"]["raw_direct"]["common"]["sha256"], timestep=8),
    lambda manifest, root: manifest["checkpoints"]["raw_direct"]["selected"].update(timestep=True),
    lambda manifest, root: manifest["checkpoints"]["raw_direct"]["selected"].update(timestep=9.0),
])
def test_checkpoint_manifest_validator_rejects_malformed_or_dishonest_refs(tmp_path: Path, monkeypatch, mutation):
    from comparison.experiment_runner import ExperimentIntegrityError

    runner_module, runner, manifest, _ = _integrity_fixture(tmp_path, monkeypatch)
    mutation(manifest, runner.root)
    with pytest.raises(ExperimentIntegrityError):
        runner_module._validate_checkpoint_manifest(runner.root, manifest, runner.archive_reader)


def test_checkpoint_manifest_validator_rejects_symlink_escape(tmp_path: Path, monkeypatch):
    from comparison.experiment_runner import ExperimentIntegrityError

    runner_module, runner, manifest, _ = _integrity_fixture(tmp_path, monkeypatch)
    outside = tmp_path / "outside.sb3"
    outside.write_bytes(b"outside")
    escaped = runner.root / "raw_direct" / "escaped.sb3"
    try:
        escaped.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows host")
    manifest["checkpoints"]["raw_direct"]["selected"].update(
        path="raw_direct/escaped.sb3", sha256=hashlib.sha256(b"outside").hexdigest(), timestep=9,
    )
    with pytest.raises(ExperimentIntegrityError):
        runner_module._validate_checkpoint_manifest(runner.root, manifest, runner.archive_reader)


def test_environment_segment_validator_accepts_every_restart_and_arm(tmp_path: Path, monkeypatch):
    runner_module, runner, _, environment = _integrity_fixture(tmp_path, monkeypatch)
    for arm in ("raw_direct", "candidate_cnn"):
        path = runner.root / arm / "environment_segments.jsonl"
        path.write_text(json.dumps(environment) + "\n" + json.dumps(environment) + "\n", encoding="utf-8")
    runner_module._validate_environment_segments(runner.root, environment, runner.provenance(), production_loaded=False)


@pytest.mark.parametrize("field,value", [
    ("torch_version", "other-torch"), ("gpu_name", "other-gpu"),
    ("gpu_total_memory_bytes", 999),
    ("cuda_version", "12.0"), ("cudnn_version", 9),
    ("lock_sha256", "f" * 64), ("comparison_git_sha", "b" * 40),
])
def test_environment_segment_validator_rejects_cross_arm_vm_library_or_hash_mismatch(tmp_path: Path, monkeypatch, field, value):
    from comparison.experiment_runner import ExperimentIntegrityError

    runner_module, runner, _, environment = _integrity_fixture(tmp_path, monkeypatch)
    changed = dict(environment); changed[field] = value
    (runner.root / "candidate_cnn" / "environment_segments.jsonl").write_text(json.dumps(changed) + "\n", encoding="utf-8")
    with pytest.raises(ExperimentIntegrityError):
        runner_module._validate_environment_segments(runner.root, environment, runner.provenance(), production_loaded=False)


def test_environment_segments_accept_new_boot_and_same_gpu_model_runtime(tmp_path: Path, monkeypatch):
    runner_module, runner, _, environment = _integrity_fixture(tmp_path, monkeypatch)
    environment = environment | {
        "resolved_device": "cuda:0", "gpu_name": "Same GPU",
        "gpu_uuid": "GPU-old-instance", "gpu_total_memory_bytes": 1024,
    }
    changed = dict(environment)
    changed.update(vm_boot_id="new-boot", gpu_uuid="GPU-new-instance", process_id=999)
    for arm in ("raw_direct", "candidate_cnn"):
        (runner.root / arm / "environment_segments.jsonl").write_text(
            json.dumps(changed) + "\n", encoding="utf-8"
        )
    runner_module._validate_environment_segments(
        runner.root, environment, runner.provenance(), production_loaded=False
    )


@pytest.mark.parametrize("payload", ["{bad json}\n", "{\"command\":[],\"command\":[]}\n", "\n"])
def test_environment_segment_validator_rejects_malformed_duplicate_or_empty_arm(tmp_path: Path, monkeypatch, payload):
    from comparison.experiment_runner import ExperimentIntegrityError

    runner_module, runner, _, environment = _integrity_fixture(tmp_path, monkeypatch)
    (runner.root / "raw_direct" / "environment_segments.jsonl").write_text(payload, encoding="utf-8")
    with pytest.raises(ExperimentIntegrityError):
        runner_module._validate_environment_segments(runner.root, environment, runner.provenance(), production_loaded=False)


def test_environment_segment_validator_rejects_missing_arm_segments(tmp_path: Path, monkeypatch):
    from comparison.experiment_runner import ExperimentIntegrityError

    runner_module, runner, _, environment = _integrity_fixture(tmp_path, monkeypatch)
    (runner.root / "raw_direct" / "environment_segments.jsonl").unlink()
    with pytest.raises(ExperimentIntegrityError):
        runner_module._validate_environment_segments(runner.root, environment, runner.provenance(), production_loaded=False)


def test_environment_segment_validator_rejects_cross_device_or_gpu_identity(tmp_path: Path, monkeypatch):
    from comparison.experiment_runner import ExperimentIntegrityError

    runner_module, runner, _, environment = _integrity_fixture(tmp_path, monkeypatch)
    gpu = environment | {"resolved_device": "cuda:0", "gpu_name": "GPU", "gpu_uuid": "GPU-a", "gpu_total_memory_bytes": 1}
    for arm in ("raw_direct", "candidate_cnn"):
        (runner.root / arm / "environment_segments.jsonl").write_text(json.dumps(gpu) + "\n", encoding="utf-8")
    with pytest.raises(ExperimentIntegrityError):
        runner_module._validate_environment_segments(runner.root, environment, runner.provenance(), production_loaded=False)


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
    monkeypatch.setattr(runner_module._Runner, "_complete_marker", lambda self: {"status": "test"})
    runner_module.run_overnight_experiment(config, tmp_path, lease_interval_seconds=999)
    assert observed == list(runner_module.JOURNAL_STAGES)
    assert observed.index("smoke_candidate_cnn") < observed.index("train_raw_direct")


def test_common_evaluation_uses_configured_checkpoint_interval(tmp_path: Path, monkeypatch):
    from comparison import experiment_runner as runner_module

    config = runner_module.ExperimentConfig.for_test(checkpoint_freq=128)
    records = [{"seed": 1000 + index} for index in range(20)]
    scenario_path = tmp_path / config.scenario_path
    scenario_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_path.write_text(json.dumps({"scenarios": records}), encoding="utf-8")
    for arm in ("raw_direct", "candidate_cnn"):
        directory = tmp_path / arm
        directory.mkdir()
        (directory / "run_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner_module, "_allocrl_dir", lambda: tmp_path)
    captured = {}
    def evaluate(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
    monkeypatch.setattr(runner_module, "evaluate_comparison_artifacts", evaluate)
    runner = runner_module._Runner(
        config,
        tmp_path,
        subprocess_runner=lambda *_args, **_kwargs: None,
        clock=lambda: 0,
        python_executable="python",
        archive_timestep_reader=None,
    )
    runner.common_evaluation()
    assert captured["args"][:3] == (
        tmp_path,
        tmp_path / "raw_direct",
        tmp_path / "candidate_cnn",
    )
    assert captured["args"][3] == records
    assert captured["kwargs"]["regular_interval"] == config.checkpoint_freq


def test_runner_boot_id_is_the_manifest_boot_id_source():
    from comparison import artifact_manifest as manifest_module
    from comparison import experiment_runner as runner_module

    assert runner_module._boot_id is manifest_module._boot_id


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


def test_train_complete_state_without_receipt_runs_finalize_only(tmp_path: Path, monkeypatch):
    from comparison.experiment_runner import ExperimentConfig, _Runner
    config = ExperimentConfig.for_test(target_training_seconds_per_arm=10)
    _write_runner_state(tmp_path / "raw_direct", config)
    calls = []
    runner = _Runner(config, tmp_path, subprocess_runner=lambda argv, **kwargs: calls.append(argv), clock=lambda: 0,
                     python_executable="python", archive_timestep_reader=lambda _: 7)
    monkeypatch.setattr(runner, "provenance", lambda: {"lock_sha256": "a" * 64})
    monkeypatch.setattr(runner, "_training_completion", lambda _root: {"valid": True})
    monkeypatch.setattr(
        runner, "_validate_current_environment", lambda *_args: None
    )
    runner.train("raw_direct")
    assert len(calls) == 1
    assert "--finalize-complete-state" in calls[0]


def test_train_valid_receipt_skips_subprocess(tmp_path: Path, monkeypatch):
    from comparison.experiment_runner import ExperimentConfig, _Runner
    config = ExperimentConfig.for_test(target_training_seconds_per_arm=10)
    root = tmp_path / "raw_direct"; _write_runner_state(root, config)
    (root / "training_completion.json").write_text("{}", encoding="utf-8")
    calls = []
    runner = _Runner(config, tmp_path, subprocess_runner=lambda argv, **kwargs: calls.append(argv), clock=lambda: 0,
                     python_executable="python", archive_timestep_reader=lambda _: 7)
    monkeypatch.setattr(runner, "provenance", lambda: {"lock_sha256": "a" * 64})
    monkeypatch.setattr(runner, "_training_completion", lambda _root: {"valid": True})
    monkeypatch.setattr(
        runner, "_validate_current_environment", lambda *_args: None
    )
    runner.train("raw_direct")
    assert calls == []


def test_train_invalid_present_receipt_fails_closed_without_subprocess(tmp_path: Path, monkeypatch):
    from comparison.experiment_runner import ExperimentConfig, _Runner
    config = ExperimentConfig.for_test(target_training_seconds_per_arm=10)
    root = tmp_path / "raw_direct"; _write_runner_state(root, config)
    (root / "training_completion.json").write_text('{"corrupt":true}', encoding="utf-8")
    calls = []
    runner = _Runner(config, tmp_path, subprocess_runner=lambda argv, **kwargs: calls.append(argv), clock=lambda: 0,
                     python_executable="python", archive_timestep_reader=lambda _: 7)
    monkeypatch.setattr(runner, "provenance", lambda: {"lock_sha256": "a" * 64})
    monkeypatch.setattr(runner, "_training_completion", lambda _root: (_ for _ in ()).throw(ValueError("invalid receipt")))
    with pytest.raises(ValueError, match="invalid receipt"):
        runner.train("raw_direct")
    assert calls == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("resolved_device", "cuda:0"),
        ("gpu_name", "different-gpu"),
        ("gpu_total_memory_bytes", 2048),
        ("torch_version", "different-torch"),
        ("cuda_version", "different-cuda"),
        ("cudnn_version", "different-cudnn"),
        ("lock_sha256", "f" * 64),
        ("comparison_git_sha", "b" * 40),
    ],
)
def test_train_rejects_current_environment_mismatch_before_subprocess(
    tmp_path: Path, monkeypatch, field: str, value: object
):
    from comparison.experiment_runner import ExperimentIntegrityError
    from comparison.wall_clock_callback import atomic_write_json

    runner_module, runner = _preflight_runner(tmp_path, monkeypatch)
    provenance = runner.provenance()
    root_environment = _valid_root_environment(
        provenance, command=runner.runner_command
    )
    atomic_write_json(runner.root / "environment.json", root_environment)
    current = dict(root_environment)
    current[field] = value
    if field == "resolved_device":
        current.update(
            gpu_name="GPU", gpu_uuid="GPU-current", gpu_total_memory_bytes=1024
        )
    elif field in {"gpu_name", "gpu_total_memory_bytes"}:
        root_environment.update(
            resolved_device="cuda:0",
            gpu_name="GPU",
            gpu_uuid="GPU-root",
            gpu_total_memory_bytes=1024,
        )
        atomic_write_json(runner.root / "environment.json", root_environment)
        current = dict(root_environment)
        current[field] = value
    seen: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setattr(
        runner_module,
        "collect_environment",
        lambda command, supplied: (
            seen.append((list(command), dict(supplied))),
            current,
        )[1],
    )
    calls: list[list[str]] = []
    runner.subprocess_runner = lambda argv, **_kwargs: calls.append(list(argv))

    with pytest.raises(ExperimentIntegrityError, match="comparable"):
        runner.train("raw_direct")
    assert calls == []
    assert len(seen) == 1


def test_train_current_environment_allows_new_boot_gpu_uuid_and_process(
    tmp_path: Path, monkeypatch
):
    from comparison.wall_clock_callback import atomic_write_json

    runner_module, runner = _preflight_runner(tmp_path, monkeypatch)
    provenance = runner.provenance()
    root_environment = _valid_root_environment(
        provenance, command=runner.runner_command
    ) | {
        "resolved_device": "cuda:0",
        "gpu_name": "Same GPU",
        "gpu_uuid": "GPU-root-instance",
        "gpu_total_memory_bytes": 1024,
    }
    atomic_write_json(runner.root / "environment.json", root_environment)
    current = dict(root_environment)
    current.update(
        vm_boot_id="new-boot",
        gpu_uuid="GPU-new-instance",
        process_id=999,
        python_version="different-python",
        platform="different-vm-platform",
        cpu_count=999,
        pip_freeze=["different-instance-package==1"],
    )
    seen: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setattr(
        runner_module,
        "collect_environment",
        lambda command, supplied: (
            seen.append((list(command), dict(supplied))),
            current,
        )[1],
    )
    calls: list[list[str]] = []
    runner.subprocess_runner = lambda argv, **_kwargs: calls.append(list(argv))
    monkeypatch.setattr(runner, "_training_completion", lambda _root: {"valid": True})

    runner.train("raw_direct")
    assert len(calls) == 1
    assert len(seen) == 1
    assert seen[0][0] == calls[0]
    assert seen[0][1] == provenance


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
    monkeypatch.setattr(runner, "_training_completion", lambda _root: {"valid": True})
    monkeypatch.setattr(
        runner, "_validate_current_environment", lambda *_args: None
    )
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


def test_public_stage_harness_completes_once_then_skips_every_action(tmp_path: Path):
    from comparison.experiment_runner import ExperimentConfig, JOURNAL_STAGES, run_overnight_experiment
    calls = []
    actions = {stage: (lambda stage=stage: calls.append(stage)) for stage in JOURNAL_STAGES}
    hashers = {stage: (lambda stage=stage: f"{JOURNAL_STAGES.index(stage):064x}") for stage in JOURNAL_STAGES}
    config = ExperimentConfig.for_test()
    run_overnight_experiment(config, tmp_path, stage_actions=actions, stage_output_hashers=hashers, lease_interval_seconds=999)
    assert calls == list(JOURNAL_STAGES)
    journal = json.loads((tmp_path / "stage_journal.json").read_text())
    assert all(journal[stage]["status"] == "complete" and journal[stage]["output_sha256"] == hashers[stage]() for stage in JOURNAL_STAGES)
    assert (tmp_path / "COMPLETE.json").is_file()
    before = list(calls); run_overnight_experiment(config, tmp_path, stage_actions=actions, stage_output_hashers=hashers, lease_interval_seconds=999)
    assert calls == before and json.loads((tmp_path / "lease.json").read_text())["status"] == "released"


def _public_harness():
    from comparison.experiment_runner import JOURNAL_STAGES
    calls = []
    return calls, {stage: (lambda stage=stage: calls.append(stage)) for stage in JOURNAL_STAGES}, {stage: (lambda stage=stage: f"{JOURNAL_STAGES.index(stage):064x}") for stage in JOURNAL_STAGES}


def _capture(errors: list[BaseException], action) -> None:
    try: action()
    except BaseException as error: errors.append(error)


def test_public_harness_interrupt_retries_only_interrupted_and_downstream(tmp_path: Path):
    from comparison.experiment_runner import ExperimentConfig, JOURNAL_STAGES, run_overnight_experiment
    calls, actions, hashers = _public_harness(); actions["train_raw_direct"] = lambda: (_ for _ in ()).throw(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt): run_overnight_experiment(ExperimentConfig.for_test(), tmp_path, stage_actions=actions, stage_output_hashers=hashers, lease_interval_seconds=999)
    journal = json.loads((tmp_path / "stage_journal.json").read_text()); assert journal["train_raw_direct"]["status"] == "interrupted" and not (tmp_path / "COMPLETE.json").exists() and (tmp_path / "comparison" / "PARTIAL_REPORT.md").is_file()
    before = {stage: journal[stage]["output_sha256"] for stage in JOURNAL_STAGES[:3]}; calls.clear(); actions = {stage: (lambda stage=stage: calls.append(stage)) for stage in JOURNAL_STAGES}
    run_overnight_experiment(ExperimentConfig.for_test(), tmp_path, stage_actions=actions, stage_output_hashers=hashers, lease_interval_seconds=999)
    assert calls == list(JOURNAL_STAGES[3:]) and all(json.loads((tmp_path / "stage_journal.json").read_text())[stage]["output_sha256"] == value for stage, value in before.items()) and (tmp_path / "COMPLETE.json").exists()


def test_public_harness_live_lease_refusal_does_not_mutate_root(tmp_path: Path):
    from comparison.experiment_runner import ExperimentConfig, LeaseError, run_overnight_experiment
    (tmp_path / "lease.json").write_text(json.dumps({"token":"owner","pid":1,"boot_id":"process-1","heartbeat_utc":"x","heartbeat_monotonic":0,"status":"active"}), encoding="utf-8")
    (tmp_path / ".lease.acquire").write_text("owner\n", encoding="utf-8"); before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    calls, actions, hashers = _public_harness()
    with pytest.raises(LeaseError): run_overnight_experiment(ExperimentConfig.for_test(), tmp_path, stage_actions=actions, stage_output_hashers=hashers, lease_interval_seconds=999)
    assert calls == [] and {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_public_harness_uses_injected_wall_clock_for_fresh_orphan_refusal(tmp_path: Path):
    from comparison.experiment_runner import ExperimentConfig, LeaseError, run_overnight_experiment

    sentinel = tmp_path / ".lease.acquire"; sentinel.write_text("orphan\n", encoding="utf-8"); before = sentinel.read_bytes()
    calls, actions, hashers = _public_harness()
    with pytest.raises(LeaseError, match="fresh orphan"):
        run_overnight_experiment(ExperimentConfig.for_test(), tmp_path, stage_actions=actions, stage_output_hashers=hashers, clock=lambda: 0.0, lease_wall_time=lambda: 0.0, lease_stale_seconds=10, lease_interval_seconds=999)
    assert calls == [] and sentinel.read_bytes() == before and not (tmp_path / "lease.json").exists()


@pytest.mark.parametrize("kwargs", [{"lease_interval_seconds": 0}, {"lease_stale_seconds": float("nan")}])
def test_public_runner_rejects_invalid_lease_timing(tmp_path: Path, kwargs: dict):
    from comparison.experiment_runner import ExperimentConfig, run_overnight_experiment
    with pytest.raises(ValueError, match="positive finite"):
        run_overnight_experiment(ExperimentConfig.for_test(), tmp_path, **kwargs)


def test_public_harness_refreshes_lease_while_smoke_action_blocks(tmp_path: Path):
    from comparison.experiment_runner import ExperimentConfig, run_overnight_experiment
    calls, actions, hashers = _public_harness(); entered = threading.Event(); release = threading.Event(); errors = []
    def block():
        calls.append("smoke_raw_direct"); entered.set(); assert release.wait(2)
    actions["smoke_raw_direct"] = block
    worker = threading.Thread(target=lambda: (run_overnight_experiment(ExperimentConfig.for_test(), tmp_path, stage_actions=actions, stage_output_hashers=hashers, lease_interval_seconds=.01),), daemon=True)
    worker.start(); assert entered.wait(1)
    seen = set(); deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline and len(seen) < 2:
        try:
            payload = json.loads((tmp_path / "lease.json").read_text());
            if payload.get("status") == "active": seen.add(payload["heartbeat_utc"])
        except (OSError, json.JSONDecodeError): pass
        time.sleep(.01)
    release.set(); worker.join(2)
    assert len(seen) >= 2 and not worker.is_alive() and json.loads((tmp_path / "lease.json").read_text())["status"] == "released"


def test_blocked_owner_cannot_overwrite_or_delete_stolen_lease(tmp_path: Path):
    from comparison.experiment_runner import ExperimentConfig, LeaseError, run_overnight_experiment
    calls, actions, hashers = _public_harness(); entered = threading.Event(); release = threading.Event(); errors = []
    def block():
        entered.set(); assert release.wait(2)
    actions["smoke_raw_direct"] = block
    def invoke():
        try: run_overnight_experiment(ExperimentConfig.for_test(), tmp_path, stage_actions=actions, stage_output_hashers=hashers, lease_interval_seconds=.01)
        except BaseException as error: errors.append(error)
    worker = threading.Thread(target=invoke, daemon=True); worker.start(); assert entered.wait(1)
    old = json.loads((tmp_path / "lease.json").read_text()); replacement = dict(old); replacement["token"] = "new-token"
    from comparison.experiment_runner import atomic_write_json
    atomic_write_json(tmp_path / "lease.json", replacement); (tmp_path / ".lease.acquire").write_text("new-token\n", encoding="utf-8")
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline and not errors: time.sleep(.01)
    release.set(); worker.join(2)
    assert errors and isinstance(errors[0], LeaseError) and not (tmp_path / "COMPLETE.json").exists()
    assert json.loads((tmp_path / "lease.json").read_text())["token"] == "new-token" and (tmp_path / ".lease.acquire").read_text(encoding="utf-8").strip() == "new-token"


def _report_integrity_fixture(tmp_path: Path):
    from test_comparison_report import write_complete_fixture
    from comparison.report_builder import write_complete_report
    write_complete_fixture(tmp_path); write_complete_report(tmp_path)
    return tmp_path / "comparison"


def test_report_integrity_validator_requires_current_canonical_summary_and_pairs(tmp_path: Path):
    from comparison import experiment_runner as runner_module
    comparison = _report_integrity_fixture(tmp_path)
    hashes = runner_module._validate_report_artifacts(tmp_path)
    assert set(hashes) == {"summary.json", "scenario_paired_differences.csv", "learning_curves.png", "holdout_comparison.png", "preliminary_comparison_ko.md"}
    assert all(len(value) == 64 for value in hashes.values())
    (comparison / "summary.json").write_text('{"stale":true}\n', encoding="utf-8")
    with pytest.raises(runner_module.ExperimentIntegrityError): runner_module._validate_report_artifacts(tmp_path)


@pytest.mark.parametrize("mutation", [
    lambda path: path.write_text("wrong,header\n1005,0\n", encoding="utf-8"),
    lambda path: path.write_text(path.read_text(encoding="utf-8").replace("1005", "999", 1), encoding="utf-8"),
    lambda path: path.write_text(path.read_text(encoding="utf-8").replace("0.09999999999999998", "nan"), encoding="utf-8"),
])
def test_report_integrity_validator_rejects_paired_csv_tampering(tmp_path: Path, mutation):
    from comparison import experiment_runner as runner_module
    comparison = _report_integrity_fixture(tmp_path); mutation(comparison / "scenario_paired_differences.csv")
    with pytest.raises(runner_module.ExperimentIntegrityError): runner_module._validate_report_artifacts(tmp_path)


@pytest.mark.parametrize("name,payload", [("learning_curves.png", None), ("holdout_comparison.png", b""), ("preliminary_comparison_ko.md", b"\xff"), ("preliminary_comparison_ko.md", b"seed 0\n\xef\xbf\xbd")])
def test_report_integrity_validator_rejects_missing_or_invalid_rendered_artifacts(tmp_path: Path, name: str, payload: bytes | None):
    from comparison import experiment_runner as runner_module
    comparison = _report_integrity_fixture(tmp_path); path = comparison / name
    if payload is None: path.unlink()
    else: path.write_bytes(payload)
    with pytest.raises(runner_module.ExperimentIntegrityError): runner_module._validate_report_artifacts(tmp_path)


@pytest.mark.parametrize("relative", ["raw_direct/runtime_metrics.json", "raw_direct/evaluation_scenarios.csv", "manifest.json"])
def test_report_integrity_validator_detects_post_report_input_changes(tmp_path: Path, relative: str):
    from comparison import experiment_runner as runner_module
    _report_integrity_fixture(tmp_path); path = tmp_path / relative
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if relative.endswith("runtime_metrics.json"): payload["run_wall_span_seconds"] += 1
        else: payload["provenance"] = {"changed": True}
        path.write_text(json.dumps(payload), encoding="utf-8")
    else: path.write_text(path.read_text(encoding="utf-8").replace("0.4", "0.41", 1), encoding="utf-8")
    with pytest.raises(runner_module.ExperimentIntegrityError): runner_module._validate_report_artifacts(tmp_path)


def test_public_marker_has_exact_provenance_and_stage_hash_schema(tmp_path: Path):
    from comparison.experiment_runner import ExperimentConfig, JOURNAL_STAGES, REQUIRED_COMPLETE_STAGES, run_overnight_experiment
    calls, actions, hashers = _public_harness(); run_overnight_experiment(ExperimentConfig.for_test(), tmp_path, stage_actions=actions, stage_output_hashers=hashers, lease_interval_seconds=999)
    marker = json.loads((tmp_path / "COMPLETE.json").read_text(encoding="utf-8"))
    assert set(marker) == {"schema_version", "status", "stages", "config_sha256", "baseline_sha256", "scenario_sha256", "split_sha256", "lock_sha256", "comparison_git_sha", "manifest_sha256", "environment_sha256", "stage_output_sha256", "report_artifact_sha256", "completed_at_utc", "lease_token"}
    assert marker["schema_version"] == 1 and marker["status"] == "complete" and marker["stages"] == REQUIRED_COMPLETE_STAGES
    assert marker["stage_output_sha256"] == {stage: hashers[stage]() for stage in JOURNAL_STAGES}


def test_owned_failure_removes_an_old_complete_marker_before_stages(tmp_path: Path):
    from comparison.experiment_runner import ExperimentStageError, ExperimentConfig, run_overnight_experiment
    (tmp_path / "COMPLETE.json").write_text('{"old":true}', encoding="utf-8")
    calls, actions, hashers = _public_harness(); actions["smoke_raw_direct"] = lambda: (_ for _ in ()).throw(RuntimeError("stop"))
    with pytest.raises(ExperimentStageError): run_overnight_experiment(ExperimentConfig.for_test(), tmp_path, stage_actions=actions, stage_output_hashers=hashers, lease_interval_seconds=999)
    assert not (tmp_path / "COMPLETE.json").exists()


def test_complete_marker_write_failure_leaves_no_marker_and_writes_partial(tmp_path: Path, monkeypatch):
    from comparison import experiment_runner as runner_module

    original = runner_module.atomic_write_json
    def fail_marker(path, payload):
        if Path(path).name == "COMPLETE.json": raise OSError("marker write failed")
        return original(path, payload)
    monkeypatch.setattr(runner_module, "atomic_write_json", fail_marker)
    calls, actions, hashers = _public_harness()
    with pytest.raises(OSError, match="marker write failed"):
        runner_module.run_overnight_experiment(runner_module.ExperimentConfig.for_test(), tmp_path, stage_actions=actions, stage_output_hashers=hashers, lease_interval_seconds=999)
    assert not (tmp_path / "COMPLETE.json").exists() and (tmp_path / "comparison" / "PARTIAL_REPORT.md").is_file()


_SUBSTANTIVE_STAGES = ("train_raw_direct", "evaluate_raw_direct", "train_candidate_cnn", "evaluate_candidate_cnn", "evaluate_common_step", "build_report", "integrity_verification")


@pytest.mark.parametrize("failed_stage", _SUBSTANTIVE_STAGES)
def test_public_stage_failure_retries_exact_failed_stage_and_pending_downstream(tmp_path: Path, failed_stage: str):
    from comparison.experiment_runner import ExperimentConfig, ExperimentStageError, JOURNAL_STAGES, run_overnight_experiment

    counts = {stage: 0 for stage in JOURNAL_STAGES}
    hashes = {stage: f"{index:064x}" for index, stage in enumerate(JOURNAL_STAGES)}
    def action(stage):
        def invoke():
            counts[stage] += 1
            if stage == failed_stage and counts[stage] == 1: raise RuntimeError("once")
        return invoke
    actions = {stage: action(stage) for stage in JOURNAL_STAGES}
    hashers = {stage: (lambda stage=stage: hashes[stage]) for stage in JOURNAL_STAGES}
    config = ExperimentConfig.for_test()
    with pytest.raises(ExperimentStageError): run_overnight_experiment(config, tmp_path, stage_actions=actions, stage_output_hashers=hashers, lease_interval_seconds=999)
    journal = json.loads((tmp_path / "stage_journal.json").read_text(encoding="utf-8")); index = JOURNAL_STAGES.index(failed_stage)
    assert all(journal[stage]["status"] == "complete" for stage in JOURNAL_STAGES[:index])
    assert journal[failed_stage]["status"] == "failed" and all(journal[stage]["status"] == "pending" for stage in JOURNAL_STAGES[index + 1:])
    assert not (tmp_path / "COMPLETE.json").exists() and (tmp_path / "comparison" / "PARTIAL_REPORT.md").is_file() and json.loads((tmp_path / "lease.json").read_text())["status"] == "released"
    earlier = {stage: journal[stage]["output_sha256"] for stage in JOURNAL_STAGES[:index]}
    run_overnight_experiment(config, tmp_path, stage_actions=actions, stage_output_hashers=hashers, lease_interval_seconds=999)
    repaired = json.loads((tmp_path / "stage_journal.json").read_text(encoding="utf-8"))
    assert all(counts[stage] == 1 for stage in JOURNAL_STAGES[:index]) and counts[failed_stage] == 2 and all(counts[stage] == 1 for stage in JOURNAL_STAGES[index + 1:])
    assert {stage: repaired[stage]["output_sha256"] for stage in earlier} == earlier and all(repaired[stage]["status"] == "complete" for stage in JOURNAL_STAGES)
    assert json.loads((tmp_path / "COMPLETE.json").read_text(encoding="utf-8"))["stage_output_sha256"] == hashes


@pytest.mark.parametrize("tampered_stage", ["smoke_raw_direct", "smoke_candidate_cnn", "train_raw_direct", "evaluate_common_step", "build_report"])
def test_public_semantic_tamper_reruns_only_stage_and_downstream(tmp_path: Path, tampered_stage: str):
    from comparison.experiment_runner import ExperimentConfig, JOURNAL_STAGES, run_overnight_experiment

    calls = []; values = {stage: f"{index:064x}" for index, stage in enumerate(JOURNAL_STAGES)}
    actions = {stage: (lambda stage=stage: calls.append(stage)) for stage in JOURNAL_STAGES}
    hashers = {stage: (lambda stage=stage: values[stage]) for stage in JOURNAL_STAGES}
    config = ExperimentConfig.for_test(); run_overnight_experiment(config, tmp_path, stage_actions=actions, stage_output_hashers=hashers, lease_interval_seconds=999)
    before = list(calls); values[tampered_stage] = "f" * 64; calls.clear()
    run_overnight_experiment(config, tmp_path, stage_actions=actions, stage_output_hashers=hashers, lease_interval_seconds=999)
    index = JOURNAL_STAGES.index(tampered_stage)
    assert calls == list(JOURNAL_STAGES[index:]) and before == list(JOURNAL_STAGES)
    marker = json.loads((tmp_path / "COMPLETE.json").read_text(encoding="utf-8"))
    assert marker["stage_output_sha256"] == values


def test_public_tamper_restore_failure_removes_old_marker(tmp_path: Path):
    from comparison.experiment_runner import ExperimentConfig, ExperimentStageError, JOURNAL_STAGES, run_overnight_experiment

    calls, actions, hashers = _public_harness(); config = ExperimentConfig.for_test()
    run_overnight_experiment(config, tmp_path, stage_actions=actions, stage_output_hashers=hashers, lease_interval_seconds=999)
    actions["build_report"] = lambda: (_ for _ in ()).throw(RuntimeError("restore failed"))
    changed = {stage: hashers[stage] for stage in JOURNAL_STAGES}; changed["build_report"] = lambda: "f" * 64
    with pytest.raises(ExperimentStageError): run_overnight_experiment(config, tmp_path, stage_actions=actions, stage_output_hashers=changed, lease_interval_seconds=999)
    assert not (tmp_path / "COMPLETE.json").exists() and (tmp_path / "comparison" / "PARTIAL_REPORT.md").is_file()


def test_public_stale_in_progress_retries_that_stage_and_every_downstream(tmp_path: Path):
    from comparison.experiment_runner import ExperimentConfig, JOURNAL_STAGES, run_overnight_experiment

    calls, actions, hashers = _public_harness(); config = ExperimentConfig.for_test()
    run_overnight_experiment(config, tmp_path, stage_actions=actions, stage_output_hashers=hashers, lease_interval_seconds=999)
    journal_path = tmp_path / "stage_journal.json"; journal = json.loads(journal_path.read_text(encoding="utf-8")); target = "evaluate_raw_direct"; index = JOURNAL_STAGES.index(target)
    stale = dict(journal[target]); stale.update(status="in_progress", output_sha256=None, completed_at_utc=None, error=None); journal[target] = stale
    journal_path.write_text(json.dumps(journal), encoding="utf-8"); (tmp_path / "COMPLETE.json").unlink(); calls.clear()
    run_overnight_experiment(config, tmp_path, stage_actions=actions, stage_output_hashers=hashers, lease_interval_seconds=999)
    repaired = json.loads(journal_path.read_text(encoding="utf-8"))
    assert calls == list(JOURNAL_STAGES[index:]) and all(repaired[stage]["status"] == "complete" for stage in JOURNAL_STAGES)


def test_public_smoke_candidate_failure_blocks_raw_training_and_keeps_partial(tmp_path: Path):
    from comparison.experiment_runner import ExperimentConfig, ExperimentStageError, run_overnight_experiment

    calls, actions, hashers = _public_harness(); actions["smoke_candidate_cnn"] = lambda: (_ for _ in ()).throw(RuntimeError("smoke failed"))
    with pytest.raises(ExperimentStageError): run_overnight_experiment(ExperimentConfig.for_test(), tmp_path, stage_actions=actions, stage_output_hashers=hashers, lease_interval_seconds=999)
    assert "train_raw_direct" not in calls and not (tmp_path / "COMPLETE.json").exists() and (tmp_path / "comparison" / "PARTIAL_REPORT.md").is_file()


def test_live_lease_refusal_preserves_existing_complete_and_root_bytes(tmp_path: Path):
    from comparison.experiment_runner import ExperimentConfig, LeaseError, run_overnight_experiment

    calls, actions, hashers = _public_harness(); config = ExperimentConfig.for_test()
    run_overnight_experiment(config, tmp_path, stage_actions=actions, stage_output_hashers=hashers, lease_interval_seconds=999)
    (tmp_path / "lease.json").write_text(json.dumps({"token": "owner", "pid": 1, "boot_id": "process-1", "heartbeat_utc": "x", "heartbeat_monotonic": 0, "status": "active"}), encoding="utf-8")
    (tmp_path / ".lease.acquire").write_text("owner\n", encoding="utf-8"); before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(LeaseError): run_overnight_experiment(config, tmp_path, stage_actions=actions, stage_output_hashers=hashers, lease_interval_seconds=999)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_active_foreign_windows_fallback_lease_needs_proven_stale_wall_heartbeat(tmp_path: Path):
    from comparison.experiment_runner import LeaseError, _Lease, atomic_write_json

    payload = {"token": "owner", "pid": 2, "boot_id": "process-2", "heartbeat_utc": "1970-01-01T00:01:30Z", "heartbeat_monotonic": 0.0, "status": "active"}
    atomic_write_json(tmp_path / "lease.json", payload); (tmp_path / ".lease.acquire").write_text("owner\n", encoding="utf-8")
    with pytest.raises(LeaseError, match="live"):
        with _Lease(tmp_path, stale_takeover=True, clock=lambda: 10_000.0, wall_time=lambda: 100.0, stale_after=20, interval=999):
            pass


def test_rebooted_foreign_monotonic_clock_can_take_over_only_after_stale_utc(tmp_path: Path):
    from comparison.experiment_runner import _Lease, atomic_write_json

    payload = {"token": "owner", "pid": 2, "boot_id": "other-boot", "heartbeat_utc": "1970-01-01T00:00:00Z", "heartbeat_monotonic": 10_000.0, "status": "active"}
    atomic_write_json(tmp_path / "lease.json", payload); (tmp_path / ".lease.acquire").write_text("owner\n", encoding="utf-8")
    with _Lease(tmp_path, stale_takeover=True, clock=lambda: 10.0, wall_time=lambda: 1_000.0, stale_after=20, interval=999) as lease:
        assert lease.acquired


def test_foreign_reboot_record_with_fresh_utc_refuses_takeover_even_if_monotonic_is_larger(tmp_path: Path):
    from comparison.experiment_runner import LeaseError, _Lease, atomic_write_json

    payload = {"token": "owner", "pid": 2, "boot_id": "other-boot", "heartbeat_utc": "1970-01-01T00:16:39Z", "heartbeat_monotonic": 10_000.0, "status": "active"}
    atomic_write_json(tmp_path / "lease.json", payload); (tmp_path / ".lease.acquire").write_text("owner\n", encoding="utf-8")
    with pytest.raises(LeaseError, match="live"):
        with _Lease(tmp_path, stale_takeover=True, clock=lambda: 10.0, wall_time=lambda: 1_000.0, stale_after=20, interval=999):
            pass


def test_same_boot_future_monotonic_heartbeat_fails_closed(tmp_path: Path, monkeypatch):
    from comparison import experiment_runner as runner_module

    monkeypatch.setattr(runner_module, "_boot_id", lambda: "same-boot")
    payload = {"token": "owner", "pid": 2, "boot_id": "same-boot", "heartbeat_utc": "1970-01-01T00:00:00Z", "heartbeat_monotonic": 10_000.0, "status": "active"}
    runner_module.atomic_write_json(tmp_path / "lease.json", payload); (tmp_path / ".lease.acquire").write_text("owner\n", encoding="utf-8")
    with pytest.raises(runner_module.LeaseError, match="live"):
        with runner_module._Lease(tmp_path, stale_takeover=True, clock=lambda: 10.0, wall_time=lambda: 1_000.0, stale_after=20, interval=999):
            pass


def test_release_write_failure_keeps_owned_sentinel_unavailable(tmp_path: Path, monkeypatch):
    from comparison.experiment_runner import LeaseError, _Lease

    lease = _Lease(tmp_path, stale_takeover=False, clock=lambda: 0.0, wall_time=lambda: 0.0, interval=999)
    original = lease._write
    def fail_release(status):
        if status == "released": raise LeaseError("release write failed")
        return original(status)
    monkeypatch.setattr(lease, "_write", fail_release)
    with pytest.raises(LeaseError, match="release write failed"):
        with lease: pass
    assert (tmp_path / ".lease.acquire").read_text(encoding="utf-8").strip() == lease.token


def test_release_failure_after_marker_removes_only_its_owned_marker(tmp_path: Path, monkeypatch):
    from comparison import experiment_runner as runner_module

    original = runner_module._Lease._write
    def fail_release(self, status):
        if status == "released": raise runner_module.LeaseError("release write failed")
        return original(self, status)
    monkeypatch.setattr(runner_module._Lease, "_write", fail_release)
    calls, actions, hashers = _public_harness()
    with pytest.raises(runner_module.LeaseError, match="release write failed"):
        runner_module.run_overnight_experiment(runner_module.ExperimentConfig.for_test(), tmp_path, stage_actions=actions, stage_output_hashers=hashers, lease_interval_seconds=999)
    assert not (tmp_path / "COMPLETE.json").exists() and (tmp_path / ".lease.acquire").exists()


def test_release_failure_cleanup_never_deletes_replaced_marker(tmp_path: Path, monkeypatch):
    from comparison import experiment_runner as runner_module

    original = runner_module._Lease._write
    def replace_then_fail(self, status):
        if status == "released":
            runner_module.atomic_write_json(tmp_path / "COMPLETE.json", {"lease_token": "new-owner"})
            raise runner_module.LeaseError("release write failed")
        return original(self, status)
    monkeypatch.setattr(runner_module._Lease, "_write", replace_then_fail)
    calls, actions, hashers = _public_harness()
    with pytest.raises(runner_module.LeaseError, match="release write failed"):
        runner_module.run_overnight_experiment(runner_module.ExperimentConfig.for_test(), tmp_path, stage_actions=actions, stage_output_hashers=hashers, lease_interval_seconds=999)
    assert json.loads((tmp_path / "COMPLETE.json").read_text(encoding="utf-8"))["lease_token"] == "new-owner"


def test_lease_write_retries_transient_windows_permission_denial(tmp_path: Path, monkeypatch):
    from comparison import experiment_runner as runner_module

    lease = runner_module._Lease(tmp_path, stale_takeover=False, clock=lambda: 0.0, wall_time=lambda: 0.0, interval=999)
    original, calls = runner_module.atomic_write_json, [0]
    def fail_once(path, payload):
        if Path(path) == lease.path and calls[0] == 0:
            calls[0] += 1
            raise PermissionError("sharing violation")
        return original(path, payload)
    with lease:
        monkeypatch.setattr(runner_module, "atomic_write_json", fail_once)
        lease._write("active")
    assert calls == [1]


def test_task6_train_and_integrity_reject_duplicate_run_state(tmp_path: Path, monkeypatch):
    from comparison import experiment_runner as runner_module

    _, runner, _, _ = _integrity_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "_training_completion", lambda _root: {"valid": True})
    path = runner.root / "raw_direct" / "run_state.json"; raw = path.read_text(encoding="utf-8").rstrip()
    path.write_text(raw[:-1] + ',"generation":999}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        runner.train("raw_direct")
    with pytest.raises(ValueError, match="duplicate"):
        runner.integrity()


def test_task6_marker_validation_rejects_duplicate_runtime_metrics(tmp_path: Path):
    from comparison import experiment_runner as runner_module

    _report_integrity_fixture(tmp_path)
    path = tmp_path / "raw_direct" / "runtime_metrics.json"; raw = path.read_text(encoding="utf-8").rstrip()
    path.write_text(raw[:-1] + ',"restart_count":999}', encoding="utf-8")
    with pytest.raises(runner_module.ExperimentIntegrityError):
        runner_module._validate_report_artifacts(tmp_path)


def test_public_marker_is_published_while_first_runner_still_owns_lease(tmp_path: Path, monkeypatch):
    from comparison import experiment_runner as runner_module

    entered, release, errors = threading.Event(), threading.Event(), []
    original = runner_module._Runner._complete_marker
    def paused_marker(self):
        entered.set(); assert release.wait(2)
        return original(self)
    monkeypatch.setattr(runner_module._Runner, "_complete_marker", paused_marker)
    calls, actions, hashers = _public_harness()
    worker = threading.Thread(target=lambda: _capture(errors, lambda: runner_module.run_overnight_experiment(runner_module.ExperimentConfig.for_test(), tmp_path, stage_actions=actions, stage_output_hashers=hashers, lease_interval_seconds=999)), daemon=True)
    worker.start(); assert entered.wait(2)
    with pytest.raises(runner_module.LeaseError):
        with runner_module._Lease(tmp_path, stale_takeover=False, clock=time.monotonic, wall_time=time.time, interval=999):
            pass
    release.set(); worker.join(2)
    assert not errors and json.loads((tmp_path / "COMPLETE.json").read_text(encoding="utf-8"))["lease_token"] == json.loads((tmp_path / "lease.json").read_text(encoding="utf-8"))["token"]


def test_late_lease_heartbeat_failure_refuses_complete_marker(tmp_path: Path, monkeypatch):
    from comparison import experiment_runner as runner_module

    original = runner_module._Runner._complete_marker
    def fail_late(self):
        self.lease.failure = runner_module.LeaseError("late heartbeat")
        return original(self)
    monkeypatch.setattr(runner_module._Runner, "_complete_marker", fail_late)
    calls, actions, hashers = _public_harness()
    with pytest.raises(runner_module.LeaseError, match="late heartbeat"):
        runner_module.run_overnight_experiment(runner_module.ExperimentConfig.for_test(), tmp_path, stage_actions=actions, stage_output_hashers=hashers, lease_interval_seconds=999)
    assert not (tmp_path / "COMPLETE.json").exists()


def test_failure_cleanup_never_removes_newer_complete_marker(tmp_path: Path):
    from comparison.experiment_runner import _remove_owned_complete_marker

    marker = tmp_path / "COMPLETE.json"; marker.write_text('{"lease_token":"new-owner"}', encoding="utf-8")
    _remove_owned_complete_marker(marker, "old-owner")
    assert marker.read_text(encoding="utf-8") == '{"lease_token":"new-owner"}'


def test_run_stage_marks_blocking_action_failed_when_heartbeat_fails_before_completion(tmp_path: Path):
    from comparison.experiment_runner import ExperimentConfig, ExperimentStageError, _Runner

    runner = _Runner(ExperimentConfig.for_test(), tmp_path, subprocess_runner=lambda *args, **kwargs: None, clock=lambda: 0, python_executable=None, archive_timestep_reader=None, output_hasher=lambda _: "a" * 64)
    runner.lease = type("Lease", (), {"failure": None})()
    def block_then_fail(): runner.lease.failure = RuntimeError("refresh failed")
    with pytest.raises(ExperimentStageError, match="heartbeat"):
        runner.run_stage("smoke_raw_direct", block_then_fail)
    assert runner.journal()["smoke_raw_direct"]["status"] == "failed"


def test_integrity_output_hash_revalidates_environment_segments_before_skip(tmp_path: Path, monkeypatch):
    from comparison import experiment_runner as runner_module

    _, runner, _, environment = _integrity_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "_training_completion", lambda _root: {"valid": True})
    monkeypatch.setattr(runner, "_state_complete", lambda *args: True)
    monkeypatch.setattr(runner_module, "_validate_checkpoint_manifest", lambda *args: None)
    monkeypatch.setattr(runner_module, "_validate_report_artifacts", lambda *args: {"summary.json": "a" * 64})
    runner.integrity()
    (runner.root / "raw_direct" / "environment_segments.jsonl").write_text(json.dumps(environment | {"torch_version": "tampered"}) + "\n", encoding="utf-8")
    with pytest.raises(runner_module.ExperimentIntegrityError):
        runner.output_hash("integrity_verification")


def test_existing_journal_requires_every_stage_key(tmp_path: Path):
    from comparison.experiment_runner import ExperimentConfig, ExperimentIntegrityError, _Runner

    (tmp_path / "stage_journal.json").write_text("{}", encoding="utf-8")
    runner = _Runner(ExperimentConfig.for_test(), tmp_path, subprocess_runner=lambda *args, **kwargs: None, clock=lambda: 0, python_executable=None, archive_timestep_reader=None)
    with pytest.raises(ExperimentIntegrityError, match="exact"):
        runner.journal()


@pytest.mark.parametrize("relative", ["config.json", "environment.json", "manifest.json", "stage_journal.json", "COMPLETE.json"])
def test_json_loader_rejects_duplicate_keys_for_every_root_artifact(tmp_path: Path, relative: str):
    from comparison.experiment_runner import _json

    path = tmp_path / relative; path.write_text('{"key":1,"key":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        _json(path)


def test_train_load_run_config_rejects_duplicate_keys(tmp_path: Path):
    from train import load_run_config

    path = tmp_path / "run_config.json"; path.write_text('{"seed":0,"seed":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_run_config(path)


def test_root_manifest_rejects_extra_key_but_accepts_task4_atomic_format(tmp_path: Path, monkeypatch):
    from comparison.experiment_runner import ExperimentIntegrityError, _validate_root_manifest

    _, runner, manifest, environment = _integrity_fixture(tmp_path, monkeypatch)
    _validate_root_manifest(json.loads((runner.root / "manifest.json").read_text(encoding="utf-8")), runner.provenance(), environment)
    manifest["extra"] = True
    with pytest.raises(ExperimentIntegrityError, match="schema"):
        _validate_root_manifest(manifest, runner.provenance(), environment)


@pytest.mark.parametrize("timestamp", ["2026-01-01T00:00:00", "2026-01-01T00:00:00+09:00", "bad"])
def test_root_environment_rejects_naive_non_utc_or_invalid_capture_timestamp(tmp_path: Path, monkeypatch, timestamp: str):
    from comparison.experiment_runner import ExperimentIntegrityError, _validate_root_environment

    _, runner = _preflight_runner(tmp_path, monkeypatch); provenance = runner.provenance()
    with pytest.raises(ExperimentIntegrityError, match="captured_at_utc"):
        _validate_root_environment(_valid_root_environment(provenance, command=runner.runner_command) | {"captured_at_utc": timestamp}, provenance, production_loaded=False)


def test_root_environment_accepts_task3_plus_zero_zero_capture_timestamp(tmp_path: Path, monkeypatch):
    from comparison.experiment_runner import _validate_root_environment

    _, runner = _preflight_runner(tmp_path, monkeypatch); provenance = runner.provenance()
    _validate_root_environment(_valid_root_environment(provenance, command=runner.runner_command) | {"captured_at_utc": "2026-01-01T00:00:00+00:00"}, provenance, production_loaded=False)


@pytest.mark.parametrize("status,field,value", [
    ("pending", "error", "unexpected"),
    ("failed", "error", None),
    ("interrupted", "output_sha256", "a" * 64),
    ("complete", "completed_at_utc", "2026-01-01T00:00:00"),
    ("complete", "completed_at_utc", "2025-01-01T00:00:00Z"),
])
def test_journal_rejects_noncanonical_pending_terminal_or_timestamp_shapes(tmp_path: Path, status: str, field: str, value: object):
    from comparison.experiment_runner import ExperimentConfig, ExperimentIntegrityError, JOURNAL_STAGES, _Runner, _journal_entry

    entry = _journal_entry(status, input_sha256="a" * 64, output_sha256="a" * 64, started_at_utc="2026-01-01T00:00:00Z", completed_at_utc="2026-01-01T00:00:01Z", error=None)
    if status == "pending": entry = _journal_entry()
    if status in {"failed", "interrupted"}: entry = _journal_entry(status, input_sha256="a" * 64, started_at_utc="2026-01-01T00:00:00Z", completed_at_utc="2026-01-01T00:00:01Z", error="stop")
    entry[field] = value
    (tmp_path / "stage_journal.json").write_text(json.dumps({stage: dict(entry) for stage in JOURNAL_STAGES}), encoding="utf-8")
    runner = _Runner(ExperimentConfig.for_test(), tmp_path, subprocess_runner=lambda *args, **kwargs: None, clock=lambda: 0, python_executable=None, archive_timestep_reader=None)
    with pytest.raises(ExperimentIntegrityError, match="invalid stage journal"):
        runner.journal()


@pytest.mark.parametrize("mutation", ["tamper", "delete"])
def test_complete_marker_revalidates_environment_segments_after_integrity_stage(tmp_path: Path, monkeypatch, mutation: str):
    from comparison import experiment_runner as runner_module

    _, runner, _, environment = _integrity_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "_training_completion", lambda _root: {"valid": True})
    monkeypatch.setattr(runner, "_state_complete", lambda *args: True)
    monkeypatch.setattr(runner_module, "_validate_checkpoint_manifest", lambda *args: None)
    monkeypatch.setattr(runner_module, "_validate_report_artifacts", lambda *args: {"summary.json": "a" * 64})
    runner.integrity()
    runner.lease = type("Lease", (), {"token": "runner-token"})()
    runner._verify_completion_journal = lambda: {stage: {"output_sha256": "a" * 64} for stage in runner_module.JOURNAL_STAGES}
    segment = runner.root / "candidate_cnn" / "environment_segments.jsonl"
    if mutation == "tamper": segment.write_text(json.dumps(environment | {"torch_version": "tampered"}) + "\n", encoding="utf-8")
    else: segment.unlink()
    with pytest.raises(runner_module.ExperimentIntegrityError):
        runner._complete_marker()
