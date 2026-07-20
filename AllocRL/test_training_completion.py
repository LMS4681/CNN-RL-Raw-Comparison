"""Strict train-stage completion receipt regression tests."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch.nn as nn
from types import SimpleNamespace

from comparison.artifact_manifest import write_runtime_metrics
from comparison.training_completion import (
    read_training_completion,
    validate_training_completion,
    write_training_completion,
)
from comparison.wall_clock_callback import (
    ProgressTimingRow,
    WallClockState,
    atomic_write_json,
    atomic_write_progress_timing,
)


CONFIG_SHA = "a" * 64


def _read_timestep(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _bundle(root: Path) -> None:
    (root / "checkpoints").mkdir(parents=True)
    checkpoint = root / "checkpoints" / "model_120_g2.sb3"
    checkpoint.write_text("120", encoding="utf-8")
    conventional = root / "block_placement_ppo.sb3"
    conventional.write_text("120", encoding="utf-8")
    state = WallClockState(
        schema_version=1,
        target_training_seconds=10.0,
        completed_training_seconds=10.0,
        last_checkpoint_timestep=120,
        last_regular_checkpoint_timestep=100,
        last_checkpoint_file=checkpoint.name,
        last_checkpoint_sha256=__import__("hashlib").sha256(b"120").hexdigest(),
        config_sha256=CONFIG_SHA,
        generation=2,
        restart_count=1,
        max_unrecorded_seconds=3.0,
        status="complete",
        started_at_utc="2026-07-20T00:00:00+00:00",
        updated_at_utc="2026-07-20T00:00:10+00:00",
        completed_at_utc="2026-07-20T00:00:10+00:00",
    )
    atomic_write_json(root / "run_state.json", asdict(state))
    atomic_write_json(root / "run_origin.json", {
        "schema_version": 1,
        "config_sha256": CONFIG_SHA,
        "initial_timestep": 0,
        "source": "observed_before_first_learn",
            "created_at_utc": "2026-07-20T00:00:00+00:00",
    })
    atomic_write_json(root / "run_config.json", {"extractor": "raw-direct"})
    (root / "environment_segments.jsonl").write_text(
        '{"segment":1}\n', encoding="utf-8"
    )
    atomic_write_progress_timing(root / "progress_timing.csv", [
        ProgressTimingRow(
            generation=2,
            timestep=120,
            recorded_training_seconds=10.0,
            updated_at_utc=state.updated_at_utc,
            status="complete",
            checkpoint_file=checkpoint.name,
        )
    ])
    with (root / "evaluation_csv.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=("source", "policy", "score"))
        writer.writeheader()
        writer.writerow({"source": "original_csv", "policy": "model", "score": 1})
    write_runtime_metrics(root / "runtime_metrics.json", {
        "schema_version": 2,
        "target_training_seconds": 10.0,
        "recorded_training_seconds": 10.0,
        "run_wall_span_seconds": 12.0,
        "overrun_seconds": 0.0,
        "restart_count": 1,
        "max_unrecorded_seconds": 3.0,
        "start_timestep": 0,
        "start_timestep_source": "run_origin.initial_timestep",
        "end_timestep": 120,
        "steps_per_second": 12.0,
        "parameter_counts": {"total": 10, "feature_extractor": 1, "policy": 5, "value": 4},
        "peak_cuda_memory_bytes": None,
        "peak_cuda_memory_scope": "not_cuda",
        "evaluation_seconds": 2.0,
        "metrics_recorded_at_utc": "2026-07-20T00:00:12+00:00",
        "finalization_mode": "in_process",
        "selected_checkpoint_timestep": 120,
        "selection_count": 0,
        "selection_tuple": None,
        "selection_outcome": "fallback_final",
        "fallback_reason": "selection_not_run",
        "checkpoint_identity": {"filename": checkpoint.name, "sha256": state.last_checkpoint_sha256},
    })


def test_completion_receipt_is_written_last_and_strictly_validates_bundle(tmp_path):
    _bundle(tmp_path)
    receipt = write_training_completion(
        tmp_path,
        expected_config_sha256=CONFIG_SHA,
        expected_target_seconds=10.0,
        finalization_mode="in_process",
        archive_timestep_reader=_read_timestep,
        finalized_at_utc="2026-07-20T00:00:12+00:00",
    )
    assert read_training_completion(tmp_path / "training_completion.json") == receipt
    assert validate_training_completion(
        tmp_path,
        expected_config_sha256=CONFIG_SHA,
        expected_target_seconds=10.0,
        archive_timestep_reader=_read_timestep,
    ) == receipt
    assert receipt["artifact_sha256"]["holdout_selection.csv"] is None


@pytest.mark.parametrize(
    "relative",
    [
        "run_state.json",
        "run_origin.json",
        "runtime_metrics.json",
        "progress_timing.csv",
        "evaluation_csv.csv",
        "block_placement_ppo.sb3",
    ],
)
def test_completion_receipt_rejects_required_artifact_tampering(tmp_path, relative):
    _bundle(tmp_path)
    write_training_completion(
        tmp_path,
        expected_config_sha256=CONFIG_SHA,
        expected_target_seconds=10.0,
        finalization_mode="in_process",
        archive_timestep_reader=_read_timestep,
    )
    with (tmp_path / relative).open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="training completion"):
        validate_training_completion(
            tmp_path,
            expected_config_sha256=CONFIG_SHA,
            expected_target_seconds=10.0,
            archive_timestep_reader=_read_timestep,
        )


def test_completion_receipt_rejects_optional_presence_mismatch(tmp_path):
    _bundle(tmp_path)
    write_training_completion(
        tmp_path,
        expected_config_sha256=CONFIG_SHA,
        expected_target_seconds=10.0,
        finalization_mode="in_process",
        archive_timestep_reader=_read_timestep,
    )
    (tmp_path / "best_model.sb3").write_text("120", encoding="utf-8")
    with pytest.raises(ValueError, match="optional"):
        validate_training_completion(
            tmp_path,
            expected_config_sha256=CONFIG_SHA,
            expected_target_seconds=10.0,
            archive_timestep_reader=_read_timestep,
        )


def test_invalid_existing_receipt_is_not_overwritten(tmp_path):
    _bundle(tmp_path)
    path = tmp_path / "training_completion.json"
    path.write_text('{"corrupt":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="existing training completion"):
        write_training_completion(
            tmp_path,
            expected_config_sha256=CONFIG_SHA,
            expected_target_seconds=10.0,
            finalization_mode="in_process",
            archive_timestep_reader=_read_timestep,
        )
    assert json.loads(path.read_text("utf-8")) == {"corrupt": True}


def test_finalize_owner_never_learns_and_writes_receipt_last(
    tmp_path, monkeypatch
):
    import train as train_module

    _bundle(tmp_path)
    (tmp_path / "block_placement_ppo.sb3").unlink()
    (tmp_path / "runtime_metrics.json").unlink()

    class Model:
        num_timesteps = 120
        device = "cpu"
        policy = nn.Linear(2, 1)

        def save(self, path):
            Path(path).write_text("120", encoding="utf-8")

        def learn(self, *_args, **_kwargs):
            raise AssertionError("finalization must never call model.learn")

    class Env:
        def close(self):
            return None

    monkeypatch.setattr(train_module, "model_num_timesteps", _read_timestep)
    monkeypatch.setattr(train_module, "create_evaluation_env", lambda *_a, **_k: Env())
    monkeypatch.setattr(
        train_module,
        "evaluate_original_csv_row",
        lambda *_a, **_k: {"source": "original_csv", "policy": "model", "score": 1},
    )
    state = __import__(
        "comparison.wall_clock_callback", fromlist=["read_wall_clock_state"]
    ).read_wall_clock_state(tmp_path / "run_state.json")
    origin = __import__(
        "comparison.artifact_manifest", fromlist=["read_run_origin"]
    ).read_run_origin(tmp_path / "run_origin.json")
    receipt = train_module.finalize_complete_wall_clock_run(
        Model(),
        args=SimpleNamespace(
            grid_size=64,
            state_context="full",
            seed=0,
            n_eval=1,
            holdout_selection_count=5,
            comparison_config_sha256=CONFIG_SHA,
            max_training_seconds=10.0,
        ),
        output_dir=tmp_path,
        full_blocks=[],
        workspaces=[],
        strategy=object(),
        observation_scales=object(),
        wall_clock_state=state,
        run_origin=origin,
        finalization_mode="recovered_complete_state",
    )
    assert receipt["finalization_mode"] == "recovered_complete_state"
    runtime = json.loads((tmp_path / "runtime_metrics.json").read_text("utf-8"))
    assert runtime["peak_cuda_memory_scope"] == "not_cuda"
    assert (tmp_path / "training_completion.json").is_file()
