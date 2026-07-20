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
    completed_at = payload["completed_at_utc"]
    if completed_at is not None and (
        not isinstance(completed_at, str) or not completed_at
    ):
        raise ValueError(
            "wall-clock state completed_at_utc must be null or a non-empty string"
        )
    if payload["status"] == "complete" and completed_at is None:
        raise ValueError("complete wall-clock state requires completed_at_utc")
    return WallClockState(**dict(payload))


def read_wall_clock_state(path: str | Path) -> WallClockState:
    """Read and strictly validate a persisted wall-clock generation."""
    state_path = Path(path)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("wall-clock state must be a JSON object")
    return _validated_state(payload)


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
        else:
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

    @staticmethod
    def _flush_file(path: Path) -> None:
        with path.open("rb+") as stream:
            stream.flush()
            os.fsync(stream.fileno())

    def _append_progress(self, state: WallClockState) -> None:
        path = self._output_dir / "progress_timing.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=_PROGRESS_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "generation": state.generation,
                    "timestep": state.last_checkpoint_timestep,
                    "recorded_training_seconds": (
                        state.completed_training_seconds
                    ),
                    "updated_at_utc": state.updated_at_utc,
                    "status": state.status,
                    "checkpoint_file": state.last_checkpoint_file,
                }
            )
            stream.flush()
            os.fsync(stream.fileno())

    def persist_checkpoint(
        self, *, status: Literal["running", "complete"]
    ) -> WallClockState:
        """Persist one archive generation before advancing durable state."""
        if status not in {"running", "complete"}:
            raise ValueError("checkpoint status must be running or complete")
        self._initialize_segment()
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
        self._append_progress(state)
        atomic_write_json(self._state_path, asdict(state))
        verified_state = read_wall_clock_state(self._state_path)
        if verified_state != state:
            raise ValueError("wall-clock state reread differs from written state")

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
