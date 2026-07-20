"""Atomic, strict receipt for one fully usable comparison training stage."""

from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from comparison.artifact_manifest import (
    read_json_object,
    read_run_origin,
    read_runtime_metrics,
    sha256_file,
)
from comparison.wall_clock_callback import (
    ProgressTimingRow,
    WallClockState,
    atomic_write_json,
    read_progress_timing,
    read_wall_clock_state,
    resolve_state_checkpoint,
)


REQUIRED_ARTIFACTS = (
    "run_state.json",
    "run_origin.json",
    "run_config.json",
    "environment_segments.jsonl",
    "runtime_metrics.json",
    "progress_timing.csv",
    "evaluation_csv.csv",
    "block_placement_ppo.sb3",
)
OPTIONAL_ARTIFACTS = (
    "holdout_selection.csv",
    "best_model.sb3",
    "training_log.csv",
    "loss_log.csv",
)
ARTIFACT_KEYS = frozenset((*REQUIRED_ARTIFACTS, *OPTIONAL_ARTIFACTS))
RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "config_sha256",
        "generation",
        "final_timestep",
        "checkpoint_file",
        "checkpoint_sha256",
        "recorded_training_seconds",
        "finalization_mode",
        "finalized_at_utc",
        "artifact_sha256",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _utc(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"training completion {field} must be UTC")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"training completion {field} must be UTC") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"training completion {field} must be UTC")
    return parsed


def _hash(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"training completion {field} must be SHA-256")
    return value


def _integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(
            f"training completion {field} must be a non-negative integer"
        )
    return value


def _number(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(
            f"training completion {field} must be finite and non-negative"
        )
    return float(value)


def _validated_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    receipt = dict(payload)
    if set(receipt) != set(RECEIPT_KEYS):
        raise ValueError("training completion receipt keys differ")
    if receipt["schema_version"] != 1:
        raise ValueError("training completion schema_version must be 1")
    _hash(receipt["config_sha256"], "config_sha256")
    _integer(receipt["generation"], "generation")
    _integer(receipt["final_timestep"], "final_timestep")
    _number(receipt["recorded_training_seconds"], "recorded_training_seconds")
    checkpoint_file = receipt["checkpoint_file"]
    if (
        not isinstance(checkpoint_file, str)
        or not checkpoint_file
        or Path(checkpoint_file).name != checkpoint_file
    ):
        raise ValueError("training completion checkpoint_file must be a basename")
    _hash(receipt["checkpoint_sha256"], "checkpoint_sha256")
    if receipt["finalization_mode"] not in {
        "in_process",
        "recovered_complete_state",
    }:
        raise ValueError("training completion finalization_mode is invalid")
    _utc(receipt["finalized_at_utc"], "finalized_at_utc")
    artifacts = receipt["artifact_sha256"]
    if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACT_KEYS):
        raise ValueError("training completion artifact hash keys differ")
    for name in REQUIRED_ARTIFACTS:
        _hash(artifacts[name], f"artifact_sha256.{name}")
    for name in OPTIONAL_ARTIFACTS:
        value = artifacts[name]
        if value is not None:
            _hash(value, f"artifact_sha256.{name}")
    return receipt


def read_training_completion(path: str | Path) -> dict[str, Any]:
    try:
        return _validated_receipt(read_json_object(path))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as error:
        if isinstance(error, ValueError) and str(error).startswith("training completion"):
            raise
        raise ValueError(f"training completion receipt is invalid: {error}") from error


def _regular_file(root: Path, name: str, *, required: bool) -> Path | None:
    path = root / name
    if path.is_symlink():
        raise ValueError(f"training completion artifact is a symlink: {name}")
    if not path.is_file():
        if required:
            raise ValueError(f"training completion required artifact is absent: {name}")
        return None
    return path


def _validate_environment_segments(path: Path) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("training completion environment segments are invalid") from error
    if not records or any(not isinstance(record, dict) for record in records):
        raise ValueError("training completion environment segments are empty or invalid")


def _validate_evaluation_csv(path: Path) -> None:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ValueError("training completion evaluation CSV is invalid") from error
    if not reader.fieldnames or len(rows) != 1:
        raise ValueError("training completion evaluation CSV must contain one row")
    row = rows[0]
    if row.get("source") != "original_csv" or row.get("policy") != "model":
        raise ValueError("training completion evaluation CSV identity is invalid")


def _expected_progress(state: WallClockState) -> ProgressTimingRow:
    return ProgressTimingRow(
        generation=state.generation,
        timestep=state.last_checkpoint_timestep,
        recorded_training_seconds=float(state.completed_training_seconds),
        updated_at_utc=state.updated_at_utc,
        status=state.status,
        checkpoint_file=state.last_checkpoint_file,
    )


def _bundle_receipt(
    root: Path,
    *,
    expected_config_sha256: str,
    expected_target_seconds: float,
    archive_timestep_reader: Callable[[Path], int | None],
    finalization_mode: str | None,
    finalized_at_utc: str,
) -> dict[str, Any]:
    _hash(expected_config_sha256, "expected config")
    expected_target = _number(expected_target_seconds, "expected target")
    required = {
        name: _regular_file(root, name, required=True)
        for name in REQUIRED_ARTIFACTS
    }
    assert all(path is not None for path in required.values())
    try:
        state = read_wall_clock_state(required["run_state.json"])
        origin = read_run_origin(required["run_origin.json"])
        runtime = read_runtime_metrics(required["runtime_metrics.json"])
        read_json_object(required["run_config.json"])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as error:
        raise ValueError(f"training completion core artifact is invalid: {error}") from error
    if (
        state.status != "complete"
        or state.config_sha256 != expected_config_sha256
        or state.target_training_seconds != expected_target
        or state.completed_training_seconds < expected_target
    ):
        raise ValueError("training completion state does not commit the expected budget")
    if origin["config_sha256"] != expected_config_sha256:
        raise ValueError("training completion origin config mismatch")
    checkpoint = resolve_state_checkpoint(root, state)
    try:
        checkpoint_timestep = archive_timestep_reader(checkpoint)
        conventional_timestep = archive_timestep_reader(
            required["block_placement_ppo.sb3"]
        )
    except (OSError, ValueError, TypeError) as error:
        raise ValueError("training completion model archive is unreadable") from error
    if checkpoint_timestep != state.last_checkpoint_timestep:
        raise ValueError("training completion checkpoint timestep mismatch")
    if conventional_timestep != state.last_checkpoint_timestep:
        raise ValueError("training completion conventional model timestep mismatch")
    try:
        progress = read_progress_timing(required["progress_timing.csv"])
    except (OSError, UnicodeDecodeError, csv.Error, ValueError, TypeError) as error:
        raise ValueError("training completion progress ledger is invalid") from error
    if not progress or progress[-1] != _expected_progress(state):
        raise ValueError("training completion final progress row differs from state")
    _validate_environment_segments(required["environment_segments.jsonl"])
    _validate_evaluation_csv(required["evaluation_csv.csv"])
    exact_runtime = {
        "target_training_seconds": float(state.target_training_seconds),
        "recorded_training_seconds": float(state.completed_training_seconds),
        "restart_count": state.restart_count,
        "max_unrecorded_seconds": float(state.max_unrecorded_seconds),
        "start_timestep": origin["initial_timestep"],
        "end_timestep": state.last_checkpoint_timestep,
    }
    if any(runtime[key] != value for key, value in exact_runtime.items()):
        raise ValueError("training completion runtime/state/origin arithmetic differs")
    trained = runtime["end_timestep"] - runtime["start_timestep"]
    expected_rate = (
        trained / runtime["recorded_training_seconds"]
        if runtime["recorded_training_seconds"] > 0
        else None
    )
    if runtime["steps_per_second"] != expected_rate:
        raise ValueError("training completion throughput arithmetic differs")
    recorded_at = _utc(runtime["metrics_recorded_at_utc"], "metrics_recorded_at_utc")
    started_at = _utc(state.started_at_utc, "state.started_at_utc")
    if runtime["run_wall_span_seconds"] != (recorded_at - started_at).total_seconds():
        raise ValueError("training completion wall span arithmetic differs")
    mode = runtime["finalization_mode"]
    if finalization_mode is not None and mode != finalization_mode:
        raise ValueError("training completion finalization mode differs")
    artifacts: dict[str, str | None] = {}
    for name, path in required.items():
        assert path is not None
        artifacts[name] = sha256_file(path)
    for name in OPTIONAL_ARTIFACTS:
        path = _regular_file(root, name, required=False)
        artifacts[name] = sha256_file(path) if path is not None else None
    return {
        "schema_version": 1,
        "config_sha256": expected_config_sha256,
        "generation": state.generation,
        "final_timestep": state.last_checkpoint_timestep,
        "checkpoint_file": state.last_checkpoint_file,
        "checkpoint_sha256": state.last_checkpoint_sha256,
        "recorded_training_seconds": float(state.completed_training_seconds),
        "finalization_mode": mode,
        "finalized_at_utc": finalized_at_utc,
        "artifact_sha256": artifacts,
    }


def validate_training_completion(
    root: str | Path,
    *,
    expected_config_sha256: str,
    expected_target_seconds: float,
    archive_timestep_reader: Callable[[Path], int | None],
) -> dict[str, Any]:
    base = Path(root).resolve()
    receipt = read_training_completion(base / "training_completion.json")
    expected = _bundle_receipt(
        base,
        expected_config_sha256=expected_config_sha256,
        expected_target_seconds=expected_target_seconds,
        archive_timestep_reader=archive_timestep_reader,
        finalization_mode=receipt["finalization_mode"],
        finalized_at_utc=receipt["finalized_at_utc"],
    )
    if receipt != expected:
        optional_mismatch = any(
            receipt["artifact_sha256"][name] != expected["artifact_sha256"][name]
            for name in OPTIONAL_ARTIFACTS
        )
        if optional_mismatch:
            raise ValueError("training completion optional artifact presence/hash differs")
        raise ValueError("training completion receipt differs from current bundle")
    return receipt


def write_training_completion(
    root: str | Path,
    *,
    expected_config_sha256: str,
    expected_target_seconds: float,
    finalization_mode: str,
    archive_timestep_reader: Callable[[Path], int | None],
    finalized_at_utc: str | None = None,
) -> dict[str, Any]:
    base = Path(root).resolve()
    path = base / "training_completion.json"
    if path.exists():
        try:
            return validate_training_completion(
                base,
                expected_config_sha256=expected_config_sha256,
                expected_target_seconds=expected_target_seconds,
                archive_timestep_reader=archive_timestep_reader,
            )
        except ValueError as error:
            raise ValueError(
                "existing training completion receipt is invalid; refusing overwrite"
            ) from error
    timestamp = finalized_at_utc or datetime.now(timezone.utc).isoformat()
    _utc(timestamp, "finalized_at_utc")
    receipt = _bundle_receipt(
        base,
        expected_config_sha256=expected_config_sha256,
        expected_target_seconds=expected_target_seconds,
        archive_timestep_reader=archive_timestep_reader,
        finalization_mode=finalization_mode,
        finalized_at_utc=timestamp,
    )
    _validated_receipt(receipt)
    atomic_write_json(path, receipt)
    return validate_training_completion(
        base,
        expected_config_sha256=expected_config_sha256,
        expected_target_seconds=expected_target_seconds,
        archive_timestep_reader=archive_timestep_reader,
    )
