"""Contract tests for the overnight comparison orchestrator."""

from __future__ import annotations

import json
import math
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
