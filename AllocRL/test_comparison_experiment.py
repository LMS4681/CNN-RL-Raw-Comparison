"""Contract tests for the overnight comparison orchestrator."""

from __future__ import annotations

import json
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
