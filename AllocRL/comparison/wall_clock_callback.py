"""Verified, resumable wall-clock training budgets for SB3 models.

The archive is copied, reopened, and hashed before ``run_state.json`` is
replaced.  This gives readers on the same mount a verified generation to
resume from; it does not claim POSIX crash durability for Google Drive FUSE.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from stable_baselines3.common.callbacks import BaseCallback
from comparison.artifact_manifest import read_json_object


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_STATE_KEYS = {
    "schema_version",
    "target_training_seconds",
    "completed_training_seconds",
    "last_checkpoint_timestep",
    "last_regular_checkpoint_timestep",
    "last_checkpoint_file",
    "last_checkpoint_sha256",
    "config_sha256",
    "generation",
    "restart_count",
    "max_unrecorded_seconds",
    "status",
    "started_at_utc",
    "updated_at_utc",
    "completed_at_utc",
}
_PROGRESS_FIELDS = (
    "generation",
    "timestep",
    "recorded_training_seconds",
    "updated_at_utc",
    "status",
    "checkpoint_file",
)


@dataclass(frozen=True)
class WallClockState:
    schema_version: int
    target_training_seconds: float
    completed_training_seconds: float
    last_checkpoint_timestep: int
    last_regular_checkpoint_timestep: int
    last_checkpoint_file: str
    last_checkpoint_sha256: str
    config_sha256: str
    generation: int
    restart_count: int
    max_unrecorded_seconds: float
    status: Literal["running", "complete"]
    started_at_utc: str
    updated_at_utc: str
    completed_at_utc: str | None


@dataclass(frozen=True)
class ProgressTimingRow:
    generation: int
    timestep: int
    recorded_training_seconds: float
    updated_at_utc: str
    status: Literal["running", "complete"]
    checkpoint_file: str


def _is_nonnegative_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def _is_nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validated_state(payload: Mapping[str, Any]) -> WallClockState:
    keys = set(payload)
    if keys != _STATE_KEYS:
        missing = sorted(_STATE_KEYS - keys)
        extra = sorted(keys - _STATE_KEYS)
        raise ValueError(
            f"wall-clock state keys differ: missing={missing}, extra={extra}"
        )
    if payload["schema_version"] != 1:
        raise ValueError("wall-clock state schema_version must be 1")
    for key in (
        "target_training_seconds",
        "completed_training_seconds",
        "max_unrecorded_seconds",
    ):
        if not _is_nonnegative_number(payload[key]):
            raise ValueError(f"wall-clock state {key} must be finite and non-negative")
    for key in (
        "last_checkpoint_timestep",
        "last_regular_checkpoint_timestep",
        "generation",
        "restart_count",
    ):
        if not _is_nonnegative_integer(payload[key]):
            raise ValueError(f"wall-clock state {key} must be a non-negative integer")
    for key in ("last_checkpoint_sha256", "config_sha256"):
        value = payload[key]
        if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(
                f"wall-clock state {key} must be lowercase hexadecimal SHA256"
            )
    checkpoint_file = payload["last_checkpoint_file"]
    if (
        not isinstance(checkpoint_file, str)
        or not checkpoint_file
        or Path(checkpoint_file).name != checkpoint_file
        or "/" in checkpoint_file
        or "\\" in checkpoint_file
    ):
        raise ValueError("wall-clock checkpoint filename must be a basename")
    if payload["status"] not in {"running", "complete"}:
        raise ValueError("wall-clock state status must be running or complete")
    for key in ("started_at_utc", "updated_at_utc"):
        if not isinstance(payload[key], str) or not payload[key]:
            raise ValueError(f"wall-clock state {key} must be a non-empty string")
        try:
            parsed = datetime.fromisoformat(payload[key].replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"wall-clock state {key} must be UTC") from error
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise ValueError(f"wall-clock state {key} must be UTC")
    completed_at = payload["completed_at_utc"]
    if completed_at is not None and (
        not isinstance(completed_at, str) or not completed_at
    ):
        raise ValueError(
            "wall-clock state completed_at_utc must be null or a non-empty string"
        )
    if payload["status"] == "complete" and completed_at is None:
        raise ValueError("complete wall-clock state requires completed_at_utc")
    if completed_at is not None:
        try:
            completed = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("wall-clock state completed_at_utc must be UTC") from error
        if completed.tzinfo is None or completed.utcoffset() != timezone.utc.utcoffset(completed):
            raise ValueError("wall-clock state completed_at_utc must be UTC")
    return WallClockState(**dict(payload))


def read_wall_clock_state(path: str | Path) -> WallClockState:
    """Read and strictly validate a persisted wall-clock generation."""
    state_path = Path(path)
    try: payload = read_json_object(state_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as error:
        raise ValueError(f"wall-clock state invalid JSON: {error}") from error
    return _validated_state(payload)


def _parse_utc(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"progress timing {field} must be UTC text")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"progress timing {field} must be UTC text") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"progress timing {field} must be UTC text")
    return value


def _progress_row(payload: Mapping[str, object]) -> ProgressTimingRow:
    if set(payload) != set(_PROGRESS_FIELDS):
        raise ValueError("progress timing row fields differ")
    try:
        generation = int(payload["generation"])
        timestep = int(payload["timestep"])
        seconds = float(payload["recorded_training_seconds"])
    except (TypeError, ValueError) as error:
        raise ValueError("progress timing numeric field is invalid") from error
    if str(generation) != payload["generation"] or generation <= 0:
        raise ValueError("progress timing generation must be a positive integer")
    if str(timestep) != payload["timestep"] or timestep < 0:
        raise ValueError("progress timing timestep must be a non-negative integer")
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("progress timing seconds must be finite and non-negative")
    status = payload["status"]
    if status not in {"running", "complete"}:
        raise ValueError("progress timing status is invalid")
    checkpoint = payload["checkpoint_file"]
    if (
        not isinstance(checkpoint, str)
        or not checkpoint
        or Path(checkpoint).name != checkpoint
        or "/" in checkpoint
        or "\\" in checkpoint
    ):
        raise ValueError("progress timing checkpoint must be a basename")
    return ProgressTimingRow(
        generation=generation,
        timestep=timestep,
        recorded_training_seconds=seconds,
        updated_at_utc=_parse_utc(payload["updated_at_utc"], "updated_at_utc"),
        status=status,
        checkpoint_file=checkpoint,
    )


def _validate_progress_rows(
    rows: list[ProgressTimingRow],
) -> list[ProgressTimingRow]:
    prior_generation = 0
    prior_timestep = 0
    prior_seconds = 0.0
    complete_seen = False
    for row in rows:
        if row.generation <= prior_generation:
            raise ValueError("progress timing generations must strictly increase")
        if row.timestep < prior_timestep:
            raise ValueError("progress timing timesteps must not decrease")
        if row.recorded_training_seconds < prior_seconds:
            raise ValueError("progress timing seconds must not decrease")
        if complete_seen:
            raise ValueError("progress timing cannot contain rows after complete")
        complete_seen = row.status == "complete"
        prior_generation = row.generation
        prior_timestep = row.timestep
        prior_seconds = row.recorded_training_seconds
    return rows


def read_progress_timing(path: str | Path) -> list[ProgressTimingRow]:
    """Read the exact progress CSV contract and validate its whole sequence."""
    source = Path(path)
    if not source.exists():
        return []
    try:
        with source.open(encoding="utf-8", newline="") as stream:
            physical_rows = list(csv.reader(stream, strict=True))
            if not physical_rows or tuple(physical_rows[0]) != _PROGRESS_FIELDS:
                raise ValueError("progress timing header differs")
            if any(
                len(row) != len(_PROGRESS_FIELDS)
                or not any(field != "" for field in row)
                for row in physical_rows[1:]
            ):
                raise ValueError("progress timing physical row is malformed")
            rows = [
                _progress_row(dict(zip(_PROGRESS_FIELDS, row)))
                for row in physical_rows[1:]
            ]
    except (OSError, UnicodeDecodeError, csv.Error, ValueError, TypeError) as error:
        if isinstance(error, ValueError) and str(error).startswith("progress timing"):
            raise
        raise ValueError(f"progress timing CSV is invalid: {error}") from error
    return _validate_progress_rows(rows)


def _state_progress_row(state: WallClockState) -> ProgressTimingRow:
    return ProgressTimingRow(
        generation=state.generation,
        timestep=state.last_checkpoint_timestep,
        recorded_training_seconds=float(state.completed_training_seconds),
        updated_at_utc=_parse_utc(state.updated_at_utc, "updated_at_utc"),
        status=state.status,
        checkpoint_file=state.last_checkpoint_file,
    )


def atomic_write_progress_timing(
    path: str | Path, rows: list[ProgressTimingRow]
) -> None:
    """Publish the entire validated ledger with one atomic replacement."""
    destination = Path(path)
    validated = _validate_progress_rows(list(rows))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=_PROGRESS_FIELDS)
        writer.writeheader()
        for row in validated:
            writer.writerow(asdict(row))
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        os.replace(temporary, destination)
        if read_progress_timing(destination) != validated:
            raise ValueError("progress timing reread differs from publication")
    finally:
        temporary.unlink(missing_ok=True)


def reconcile_progress_timing(
    path: str | Path, state: WallClockState | None
) -> list[ProgressTimingRow]:
    """Reconcile a valid crash tail against the authoritative state commit."""
    destination = Path(path)
    rows = read_progress_timing(destination)
    if state is None:
        if rows:
            atomic_write_progress_timing(destination, [])
        return []
    projection = _state_progress_row(state)
    matching = [row for row in rows if row.generation == state.generation]
    if matching and matching[0] != projection:
        raise ValueError("progress timing same-generation row conflicts with state")
    committed = [row for row in rows if row.generation < state.generation]
    if matching:
        committed.append(matching[0])
    else:
        committed.append(projection)
    committed = _validate_progress_rows(committed)
    if rows != committed:
        atomic_write_progress_timing(destination, committed)
    final = read_progress_timing(destination)
    if not final or final[-1] != projection:
        raise ValueError("progress timing final row does not match state")
    return final


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace JSON only after its temporary file is flushed to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
        json.loads(path.read_text(encoding="utf-8"))
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_state_checkpoint(
    output_dir: str | Path, state: WallClockState
) -> Path:
    """Return the exact state-named checkpoint after verifying its SHA256."""
    validated = _validated_state(asdict(state))
    checkpoint = (
        Path(output_dir) / "checkpoints" / validated.last_checkpoint_file
    ).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"wall-clock state checkpoint is absent: {checkpoint}"
        )
    actual_sha256 = _file_sha256(checkpoint)
    if actual_sha256 != validated.last_checkpoint_sha256:
        raise ValueError(
            "wall-clock state checkpoint SHA256 mismatch: "
            f"expected {validated.last_checkpoint_sha256}, got {actual_sha256}"
        )
    return checkpoint


def _default_archive_timestep_reader(path: Path) -> int | None:
    # Lazy bridge avoids importing train.py while train.py imports this module.
    from train import model_num_timesteps

    return model_num_timesteps(path)


class WallClockBudgetCallback(BaseCallback):
    """Stop at a cumulative wall-clock budget with verified checkpoints."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        target_seconds: float,
        checkpoint_freq: int,
        heartbeat_seconds: float,
        config_sha256: str,
        monotonic: Callable[[], float] = time.monotonic,
        utc_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        archive_timestep_reader: Callable[
            [Path], int | None
        ] = _default_archive_timestep_reader,
        state_path: str | Path | None = None,
    ) -> None:
        super().__init__(verbose=0)
        if target_seconds <= 0 or checkpoint_freq <= 0 or heartbeat_seconds <= 0:
            raise ValueError(
                "wall-clock limits and persistence intervals must be positive"
            )
        if (
            not isinstance(config_sha256, str)
            or _SHA256_PATTERN.fullmatch(config_sha256) is None
        ):
            raise ValueError(
                "comparison config SHA256 must be lowercase hexadecimal"
            )
        self._output_dir = Path(output_dir)
        self._state_path = (
            Path(state_path)
            if state_path is not None
            else self._output_dir / "run_state.json"
        )
        self._target_seconds = float(target_seconds)
        self._checkpoint_freq = int(checkpoint_freq)
        self._heartbeat_seconds = float(heartbeat_seconds)
        self._config_sha256 = config_sha256
        self._monotonic = monotonic
        self._utc_now = utc_now
        self._archive_timestep_reader = archive_timestep_reader
        self._completed_before_segment = 0.0
        self._last_checkpoint_timestep = 0
        self._last_regular_checkpoint_timestep = 0
        self._last_persisted_monotonic = 0.0
        self._segment_started = 0.0
        self._generation = 0
        self._restart_count = 0
        self._max_unrecorded_seconds = 0.0
        self._started_at_utc = ""
        self._initialized = False

    def _on_training_start(self) -> None:
        self._initialize_segment()

    def _initialize_segment(self) -> None:
        if self._initialized:
            return
        now = self._monotonic()
        started_at = self._utc_now().isoformat()
        if self._state_path.exists():
            state = read_wall_clock_state(self._state_path)
            if state.target_training_seconds != self._target_seconds:
                raise ValueError(
                    "wall-clock target changed on resume: "
                    f"state {state.target_training_seconds}, "
                    f"requested {self._target_seconds}"
                )
            if state.config_sha256 != self._config_sha256:
                raise ValueError("wall-clock config SHA256 changed on resume")
            checkpoint = resolve_state_checkpoint(self._output_dir, state)
            stored_timestep = self._archive_timestep_reader(checkpoint)
            if stored_timestep != state.last_checkpoint_timestep:
                raise ValueError(
                    "state checkpoint timestep "
                    f"{state.last_checkpoint_timestep} does not match archive "
                    f"{stored_timestep}"
                )
            model_timestep = int(self.model.num_timesteps)
            if model_timestep != state.last_checkpoint_timestep:
                raise ValueError(
                    "state checkpoint timestep "
                    f"{state.last_checkpoint_timestep} does not match model "
                    f"{model_timestep}"
                )
            self._completed_before_segment = state.completed_training_seconds
            self._last_checkpoint_timestep = state.last_checkpoint_timestep
            self._last_regular_checkpoint_timestep = (
                state.last_regular_checkpoint_timestep
            )
            self._generation = state.generation
            self._restart_count = state.restart_count + 1
            self._max_unrecorded_seconds = state.max_unrecorded_seconds
            self._started_at_utc = state.started_at_utc
            reconcile_progress_timing(
                self._output_dir / "progress_timing.csv", state
            )
        else:
            reconcile_progress_timing(
                self._output_dir / "progress_timing.csv", None
            )
            self._started_at_utc = started_at
        self._segment_started = now
        self._last_persisted_monotonic = now
        self._initialized = True

    def _next_generation(self, timestep: int) -> tuple[int, Path, Path]:
        checkpoints = self._output_dir / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        generation = self._generation + 1
        while True:
            final = checkpoints / f"model_{timestep}_g{generation}.sb3"
            partial = Path(f"{final}.partial")
            if not final.exists() and not partial.exists() and not any(
                checkpoints.glob(f"model_*_g{generation}.sb3*")
            ):
                return generation, partial, final
            generation += 1

    def _adopt_newer_durable_state(self) -> WallClockState | None:
        """Recover a state-first commit whose progress publication raised."""
        if not self._state_path.exists():
            return None
        durable = read_wall_clock_state(self._state_path)
        if durable.generation <= self._generation:
            return None
        if (
            durable.target_training_seconds != self._target_seconds
            or durable.config_sha256 != self._config_sha256
        ):
            raise ValueError("newer durable wall-clock state is incompatible")
        checkpoint = resolve_state_checkpoint(self._output_dir, durable)
        stored_timestep = self._archive_timestep_reader(checkpoint)
        if (
            stored_timestep != durable.last_checkpoint_timestep
            or int(self.model.num_timesteps) != durable.last_checkpoint_timestep
        ):
            raise ValueError(
                "newer durable wall-clock state does not match current model"
            )
        reconcile_progress_timing(
            self._output_dir / "progress_timing.csv", durable
        )
        now = self._monotonic()
        self._completed_before_segment = durable.completed_training_seconds
        self._segment_started = now
        self._last_persisted_monotonic = now
        self._generation = durable.generation
        self._last_checkpoint_timestep = durable.last_checkpoint_timestep
        self._last_regular_checkpoint_timestep = (
            durable.last_regular_checkpoint_timestep
        )
        self._restart_count = durable.restart_count
        self._max_unrecorded_seconds = durable.max_unrecorded_seconds
        return durable

    @staticmethod
    def _flush_file(path: Path) -> None:
        with path.open("rb+") as stream:
            stream.flush()
            os.fsync(stream.fileno())

    def persist_checkpoint(
        self, *, status: Literal["running", "complete"]
    ) -> WallClockState:
        """Persist one archive generation before advancing durable state."""
        if status not in {"running", "complete"}:
            raise ValueError("checkpoint status must be running or complete")
        self._initialize_segment()
        adopted = self._adopt_newer_durable_state()
        if adopted is not None and adopted.status == "complete":
            return adopted
        timestep = int(self.model.num_timesteps)
        generation, partial, final = self._next_generation(timestep)

        with tempfile.TemporaryDirectory(prefix="wall-clock-checkpoint-") as tmp:
            local_archive = Path(tmp) / "model.sb3"
            self.model.save(local_archive)
            if not local_archive.is_file():
                zip_archive = Path(f"{local_archive}.zip")
                if zip_archive.is_file():
                    local_archive = zip_archive
                else:
                    raise FileNotFoundError(
                        "model.save did not create the requested local archive"
                    )
            self._flush_file(local_archive)
            local_sha256 = _file_sha256(local_archive)
            shutil.copyfile(local_archive, partial)
            self._flush_file(partial)
            if final.exists():
                raise FileExistsError(
                    f"refusing to overwrite checkpoint generation: {final}"
                )
            os.rename(partial, final)

        stored_timestep = self._archive_timestep_reader(final)
        if stored_timestep != timestep:
            raise ValueError(
                f"saved archive timestep {stored_timestep} does not match model "
                f"{timestep}"
            )
        checkpoint_sha256 = _file_sha256(final)
        if checkpoint_sha256 != local_sha256:
            raise ValueError(
                "checkpoint SHA256 differs from the locally saved archive: "
                f"expected {local_sha256}, got {checkpoint_sha256}"
            )
        persisted_at = self._monotonic()
        completed_seconds = self._completed_before_segment + (
            persisted_at - self._segment_started
        )
        effective_status: Literal["running", "complete"] = (
            "complete"
            if status == "complete" or completed_seconds >= self._target_seconds
            else "running"
        )
        updated_at_utc = self._utc_now().isoformat()
        regular_boundary = (
            timestep // self._checkpoint_freq
        ) * self._checkpoint_freq
        last_regular = max(
            self._last_regular_checkpoint_timestep, regular_boundary
        )
        max_unrecorded = max(
            self._max_unrecorded_seconds,
            persisted_at - self._last_persisted_monotonic,
        )
        state = WallClockState(
            schema_version=1,
            target_training_seconds=self._target_seconds,
            completed_training_seconds=completed_seconds,
            last_checkpoint_timestep=timestep,
            last_regular_checkpoint_timestep=last_regular,
            last_checkpoint_file=final.name,
            last_checkpoint_sha256=checkpoint_sha256,
            config_sha256=self._config_sha256,
            generation=generation,
            restart_count=self._restart_count,
            max_unrecorded_seconds=max_unrecorded,
            status=effective_status,
            started_at_utc=self._started_at_utc,
            updated_at_utc=updated_at_utc,
            completed_at_utc=(
                updated_at_utc if effective_status == "complete" else None
            ),
        )
        atomic_write_json(self._state_path, asdict(state))
        verified_state = read_wall_clock_state(self._state_path)
        if verified_state != state:
            raise ValueError("wall-clock state reread differs from written state")
        progress_path = self._output_dir / "progress_timing.csv"
        rows = read_progress_timing(progress_path)
        atomic_write_progress_timing(
            progress_path, [*rows, _state_progress_row(verified_state)]
        )

        self._generation = generation
        self._last_checkpoint_timestep = timestep
        self._last_regular_checkpoint_timestep = last_regular
        self._last_persisted_monotonic = persisted_at
        self._max_unrecorded_seconds = max_unrecorded
        return verified_state

    def _on_step(self) -> bool:
        self._initialize_segment()
        now = self._monotonic()
        elapsed = self._completed_before_segment + (
            now - self._segment_started
        )
        current_timestep = int(self.model.num_timesteps)
        current_regular_boundary = (
            current_timestep // self._checkpoint_freq
        ) * self._checkpoint_freq
        should_checkpoint = (
            current_regular_boundary > self._last_regular_checkpoint_timestep
        )
        should_heartbeat = (
            now - self._last_persisted_monotonic >= self._heartbeat_seconds
        )
        should_stop = elapsed >= self._target_seconds
        if not (should_checkpoint or should_heartbeat or should_stop):
            return True

        state = self.persist_checkpoint(
            status="complete" if should_stop else "running"
        )
        if state.status != "complete":
            return True
        reread = read_wall_clock_state(self._state_path)
        if reread != state:
            raise ValueError("completed wall-clock state changed after persistence")
        resolve_state_checkpoint(self._output_dir, reread)
        return False
