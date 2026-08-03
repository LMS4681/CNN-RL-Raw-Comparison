"""Strict shared validation and narrowly safe recovery for curve CSV logs."""

from __future__ import annotations

import csv
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Literal

from comparison.path_integrity import resolve_direct_regular_file


TRAINING_LOG_COLUMNS = (
    "episode",
    "timestep",
    "resolved_reward",
    "terminal_residual",
    "terminal_score",
    "episode_reward",
    "delayed_count",
    "dropout_count",
    "total_delay_days",
    "success_rate",
)
LOSS_LOG_COLUMNS = (
    "timestep",
    "policy_gradient_loss",
    "value_loss",
    "entropy_loss",
    "approx_kl",
    "clip_fraction",
    "loss",
    "explained_variance",
    "cnn_gradient_norm",
    "cnn_weight_change",
    "workspace_feature_variance",
    "candidate_channel_sensitivity",
)
CurveLogKind = Literal["training_log", "loss_log"]
_COLUMNS = {
    "training_log": TRAINING_LOG_COLUMNS,
    "loss_log": LOSS_LOG_COLUMNS,
}
_TRAINING_INTEGER_FIELDS = {
    "episode",
    "timestep",
    "delayed_count",
    "dropout_count",
    "total_delay_days",
}
_CANONICAL_INTEGER = re.compile(r"(?:0|[1-9][0-9]*)\Z")


def _finite(value: str, field: str, kind: CurveLogKind) -> None:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{kind}.{field} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{kind}.{field} must be a finite number")


def _physical_record(line: str, kind: CurveLogKind) -> list[str]:
    if line.endswith("\r"):
        line = line[:-1]
    if "\r" in line:
        raise ValueError(f"{kind} contains an invalid physical record")
    try:
        parsed = list(csv.reader([line], strict=True))
    except csv.Error as error:
        raise ValueError(f"{kind} contains malformed CSV") from error
    if len(parsed) != 1:
        raise ValueError(f"{kind} contains malformed CSV")
    return parsed[0]


def _physical_lines(raw: bytes, kind: CurveLogKind) -> list[str]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{kind} must be valid UTF-8") from error
    physical = text.split("\n")
    if physical and physical[-1] == "":
        physical.pop()
    if not physical or any(line == "" for line in physical):
        raise ValueError(f"{kind} contains a blank or missing physical record")
    return physical


def _physical_records(raw: bytes, kind: CurveLogKind) -> list[list[str]]:
    return [
        _physical_record(line, kind) for line in _physical_lines(raw, kind)
    ]


def _parse(raw: bytes, kind: CurveLogKind) -> list[dict[str, str]]:
    columns = _COLUMNS[kind]
    records = _physical_records(raw, kind)
    if tuple(records[0]) != columns:
        raise ValueError(f"{kind} has incompatible header")
    rows: list[dict[str, str]] = []
    prior_timestep = -1
    prior_episode = -1
    for values in records[1:]:
        if len(values) != len(columns):
            raise ValueError(f"{kind} has malformed row")
        row = dict(zip(columns, values, strict=True))
        timestep_text = row["timestep"]
        if _CANONICAL_INTEGER.fullmatch(timestep_text) is None:
            raise ValueError(f"{kind}.timestep must be a canonical integer")
        timestep = int(timestep_text)
        if timestep < prior_timestep:
            raise ValueError(f"{kind} timestep regresses")
        prior_timestep = timestep
        if kind == "training_log":
            for field, value in row.items():
                if value == "":
                    raise ValueError(f"{kind}.{field} must not be empty")
                if field in _TRAINING_INTEGER_FIELDS:
                    if _CANONICAL_INTEGER.fullmatch(value) is None:
                        raise ValueError(
                            f"{kind}.{field} must be a canonical integer"
                        )
                else:
                    _finite(value, field, kind)
            episode = int(row["episode"])
            if episode <= prior_episode:
                raise ValueError(f"{kind} episode does not increase")
            prior_episode = episode
        else:
            for field, value in row.items():
                if field != "timestep" and value != "":
                    _finite(value, field, kind)
        rows.append(row)
    return rows


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
        if path.read_bytes() != payload:
            raise ValueError("curve log reread differs from repaired bytes")
    finally:
        temporary.unlink(missing_ok=True)


def read_curve_log(
    path: str | Path,
    kind: CurveLogKind,
    *,
    repair_trailing_partial: bool = False,
) -> list[dict[str, str]]:
    """Read one exact curve log, optionally discarding one corrupt tail."""
    if kind not in _COLUMNS:
        raise ValueError(f"unknown curve log kind: {kind}")
    requested = Path(path)
    try:
        source = resolve_direct_regular_file(
            requested.parent,
            requested,
            label=kind,
        )
        raw = source.read_bytes()
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(kind):
            raise
        raise ValueError(f"{kind} cannot be read") from error
    if raw.endswith(b"\n"):
        return _parse(raw, kind)

    try:
        complete_rows = _parse(raw, kind)
    except ValueError as full_error:
        if not repair_trailing_partial:
            raise
        split = raw.rfind(b"\n")
        if split < 0:
            raise ValueError(f"{kind} has no exact-valid repair prefix") from full_error
        prefix = raw[: split + 1]
        suffix = raw[split + 1 :]
        if not suffix or b"\r" in suffix or b"\n" in suffix:
            raise ValueError(f"{kind} trailing record is not safely repairable") from full_error
        rows = _parse(prefix, kind)
        _atomic_replace_bytes(source, prefix)
        if _parse(source.read_bytes(), kind) != rows:
            raise ValueError(f"{kind} repaired reread differs")
        return rows
    raise ValueError(
        f"{kind} contains a valid but physically unterminated final record"
    )


def _timestep_of(record: list[str], columns: tuple[str, ...], kind: CurveLogKind) -> int:
    if len(record) != len(columns):
        raise ValueError(f"{kind} has malformed row")
    text = record[columns.index("timestep")]
    if _CANONICAL_INTEGER.fullmatch(text) is None:
        raise ValueError(f"{kind}.timestep must be a canonical integer")
    return int(text)


def prune_rolled_back_rows(
    path: str | Path,
    kind: CurveLogKind,
    *,
    resume_timestep: int | None = None,
    repair_trailing_partial: bool = True,
) -> int:
    """Drop rolled-back rows so an append after resume stays monotonic.

    A resumed run restarts from its last checkpoint, so rows already logged
    past that timestep describe steps the resume discarded; appending after
    them would make ``timestep`` regress. Rows above ``resume_timestep`` and
    any row a later row undercuts are removed. Retained records keep their
    original bytes, and the rewritten log is validated before replacement.
    Returns the number of discarded rows.
    """
    if kind not in _COLUMNS:
        raise ValueError(f"unknown curve log kind: {kind}")
    columns = _COLUMNS[kind]
    requested = Path(path)
    try:
        source = resolve_direct_regular_file(
            requested.parent,
            requested,
            label=kind,
        )
        raw = source.read_bytes()
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(kind):
            raise
        raise ValueError(f"{kind} cannot be read") from error

    body = raw
    discarded_partial = 0
    if not raw.endswith(b"\n"):
        split = raw.rfind(b"\n")
        suffix = raw[split + 1 :]
        complete_final = False
        if split >= 0 and suffix and b"\r" not in suffix:
            try:
                _timestep_of(
                    _physical_record(suffix.decode("utf-8"), kind),
                    columns,
                    kind,
                )
                complete_final = True
            except (UnicodeDecodeError, ValueError):
                complete_final = False
        if complete_final:
            body = raw + b"\n"
        elif repair_trailing_partial and split >= 0:
            body = raw[: split + 1]
            discarded_partial = 1
        else:
            raise ValueError(f"{kind} has no exact-valid repair prefix")

    lines = _physical_lines(body, kind)
    if tuple(_physical_record(lines[0], kind)) != columns:
        raise ValueError(f"{kind} has incompatible header")
    timesteps = [
        _timestep_of(_physical_record(line, kind), columns, kind)
        for line in lines[1:]
    ]

    ceiling = float("inf")
    if resume_timestep is not None and resume_timestep > 0:
        ceiling = int(resume_timestep)
    retained = [False] * len(timesteps)
    minimum_right: float = ceiling
    for index in reversed(range(len(timesteps))):
        if timesteps[index] <= minimum_right:
            retained[index] = True
            minimum_right = timesteps[index]

    payload = "\n".join(
        [lines[0], *[line for line, keep in zip(lines[1:], retained) if keep]]
    ).encode("utf-8") + b"\n"
    if payload == raw:
        return 0
    _parse(payload, kind)
    _atomic_replace_bytes(source, payload)
    return retained.count(False) + discarded_partial
