"""Canonical, credential-safe runtime metadata for comparison artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import torch

from comparison.path_integrity import resolve_direct_regular_file


FALLBACK_REASON_CODES = frozenset({
    "selection_not_run",
    "selection_has_no_best",
    "selection_metadata_invalid",
    "best_model_missing",
    "best_model_unreadable",
    "best_model_timestep_mismatch",
})
CANONICAL_SELECTION_COUNT = 5


REQUIRED_ENVIRONMENT_KEYS = frozenset(
    {
        "captured_at_utc",
        "command",
        "python_version",
        "platform",
        "comparison_git_sha",
        "comparison_git_dirty",
        "baseline_sha256",
        "config_sha256",
        "scenario_sha256",
        "split_sha256",
        "lock_sha256",
        "vm_boot_id",
        "torch_version",
        "cuda_version",
        "cudnn_version",
        "resolved_device",
        "gpu_name",
        "gpu_uuid",
        "gpu_total_memory_bytes",
        "cpu_count",
        "process_id",
        "pip_freeze",
    }
)

RUN_ORIGIN_KEYS = frozenset(
    {
        "schema_version",
        "config_sha256",
        "initial_timestep",
        "source",
        "created_at_utc",
    }
)
RUNTIME_METRICS_KEYS = frozenset(
    {
        "schema_version",
        "target_training_seconds",
        "recorded_training_seconds",
        "run_wall_span_seconds",
        "overrun_seconds",
        "restart_count",
        "max_unrecorded_seconds",
        "start_timestep",
        "start_timestep_source",
        "end_timestep",
        "steps_per_second",
        "parameter_counts",
        "peak_cuda_memory_bytes",
        "peak_cuda_memory_scope",
        "evaluation_seconds",
        "metrics_recorded_at_utc",
        "finalization_mode",
        "selected_checkpoint_timestep",
        "selection_count",
        "selection_tuple",
        "selection_outcome",
        "fallback_reason",
        "checkpoint_identity",
    }
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


_URL_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s]+")
_WINDOWS_BOOT_ID: str | None = None


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a mapping using the same canonical representation written to disk."""
    encoded = _canonical_json(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse one JSON object while rejecting duplicate keys at every depth."""
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result: raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    value = json.loads(text, object_pairs_hook=reject_duplicate_keys)
    if not isinstance(value, dict): raise ValueError("JSON object required")
    return value


def read_json_object(path: str | Path) -> dict[str, Any]:
    return parse_json_object(Path(path).read_text(encoding="utf-8"))


def _utc_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be a UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{field} must be a UTC timestamp")
    return value


def _finite_nonnegative(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{field} must be finite and non-negative")
    return float(value)


def _nonnegative_integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be lowercase SHA-256")
    return value


def read_run_origin(path: str | Path) -> dict[str, Any]:
    """Read the durable, observed-before-first-learn timestep origin."""
    payload = read_json_object(path)
    if set(payload) != set(RUN_ORIGIN_KEYS):
        raise ValueError("run origin keys differ")
    if payload["schema_version"] != 1:
        raise ValueError("run origin schema_version must be 1")
    _sha256(payload["config_sha256"], "run origin config_sha256")
    _nonnegative_integer(payload["initial_timestep"], "run origin initial_timestep")
    if payload["source"] != "observed_before_first_learn":
        raise ValueError("run origin source is invalid")
    _utc_text(payload["created_at_utc"], "run origin created_at_utc")
    return payload


def write_run_origin(
    path: str | Path,
    *,
    config_sha256: str,
    initial_timestep: int,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    _sha256(config_sha256, "run origin config_sha256")
    _nonnegative_integer(initial_timestep, "run origin initial_timestep")
    destination = Path(path)
    if destination.exists():
        existing = read_run_origin(destination)
        if (
            existing["config_sha256"] != config_sha256
            or existing["initial_timestep"] != initial_timestep
            or existing["source"] != "observed_before_first_learn"
        ):
            raise ValueError("existing run origin differs from observed origin")
        return existing
    payload = {
        "schema_version": 1,
        "config_sha256": config_sha256,
        "initial_timestep": initial_timestep,
        "source": "observed_before_first_learn",
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
    }
    _validate_run_origin_payload(payload)
    _write_json(destination, payload)
    reread = read_run_origin(destination)
    if reread != payload:
        raise ValueError("run origin reread differs from written origin")
    return reread


def _validate_run_origin_payload(payload: Mapping[str, Any]) -> None:
    # Keep construction and disk validation on one exact contract.
    if set(payload) != set(RUN_ORIGIN_KEYS):
        raise ValueError("run origin keys differ")
    if payload["schema_version"] != 1:
        raise ValueError("run origin schema_version must be 1")
    _sha256(payload["config_sha256"], "run origin config_sha256")
    _nonnegative_integer(payload["initial_timestep"], "run origin initial_timestep")
    if payload["source"] != "observed_before_first_learn":
        raise ValueError("run origin source is invalid")
    _utc_text(payload["created_at_utc"], "run origin created_at_utc")


def _validate_runtime_metrics_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(payload)
    if set(payload) != set(RUNTIME_METRICS_KEYS):
        raise ValueError("runtime metrics keys differ")
    if payload["schema_version"] != 2:
        raise ValueError("runtime metrics schema_version must be 2")
    for field in (
        "target_training_seconds",
        "recorded_training_seconds",
        "run_wall_span_seconds",
        "overrun_seconds",
        "max_unrecorded_seconds",
        "evaluation_seconds",
    ):
        _finite_nonnegative(payload[field], f"runtime metrics {field}")
    for field in (
        "restart_count",
        "start_timestep",
        "end_timestep",
        "selected_checkpoint_timestep",
        "selection_count",
    ):
        _nonnegative_integer(payload[field], f"runtime metrics {field}")
    if payload["start_timestep_source"] != "run_origin.initial_timestep":
        raise ValueError("runtime metrics start_timestep_source is invalid")
    if payload["finalization_mode"] not in {
        "in_process",
        "recovered_complete_state",
    }:
        raise ValueError("runtime metrics finalization_mode is invalid")
    _utc_text(payload["metrics_recorded_at_utc"], "metrics_recorded_at_utc")
    steps = payload["steps_per_second"]
    if steps is not None:
        _finite_nonnegative(steps, "runtime metrics steps_per_second")
    start = payload["start_timestep"]
    end = payload["end_timestep"]
    recorded = float(payload["recorded_training_seconds"])
    if end < start:
        raise ValueError("runtime metrics end_timestep precedes start_timestep")
    if recorded == 0:
        if steps is not None:
            raise ValueError("zero recorded runtime requires null throughput")
    else:
        expected_steps = (end - start) / recorded
        if steps is None or not math.isclose(
            float(steps), expected_steps, rel_tol=1e-12, abs_tol=0.0
        ):
            raise ValueError("runtime metrics throughput arithmetic differs")
    expected_overrun = max(
        0.0,
        recorded - float(payload["target_training_seconds"]),
    )
    if not math.isclose(
        float(payload["overrun_seconds"]),
        expected_overrun,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise ValueError("runtime metrics overrun arithmetic differs")
    peak = payload["peak_cuda_memory_bytes"]
    if peak is not None:
        _nonnegative_integer(peak, "runtime metrics peak_cuda_memory_bytes")
    scope = payload["peak_cuda_memory_scope"]
    if scope not in {
        "training_process",
        "unavailable_after_training_process",
        "not_cuda",
    }:
        raise ValueError("runtime metrics peak_cuda_memory_scope is invalid")
    if scope == "unavailable_after_training_process" and peak is not None:
        raise ValueError("recovered runtime peak CUDA memory must be null")
    if scope == "not_cuda" and peak is not None:
        raise ValueError("CPU runtime peak CUDA memory must be null")
    mode = payload["finalization_mode"]
    if mode == "recovered_complete_state":
        if scope not in {"unavailable_after_training_process", "not_cuda"} or peak is not None:
            raise ValueError("recovered runtime cannot claim training-process CUDA peak")
    elif scope not in {"training_process", "not_cuda"}:
        raise ValueError("in-process runtime peak CUDA scope is invalid")
    counts = payload["parameter_counts"]
    if not isinstance(counts, dict) or set(counts) != {
        "total", "feature_extractor", "policy", "value"
    }:
        raise ValueError("runtime metrics parameter_counts schema differs")
    for field, value in counts.items():
        _nonnegative_integer(value, f"parameter_counts.{field}")
    if counts["total"] != sum(counts[key] for key in ("feature_extractor", "policy", "value")):
        raise ValueError("runtime metrics parameter counts do not reconcile")
    identity = payload["checkpoint_identity"]
    if (
        not isinstance(identity, dict)
        or set(identity) != {"filename", "sha256"}
        or not isinstance(identity["filename"], str)
        or not identity["filename"]
        or Path(identity["filename"]).name != identity["filename"]
    ):
        raise ValueError("runtime metrics checkpoint_identity is invalid")
    _sha256(identity["sha256"], "runtime checkpoint identity")
    selection_tuple = payload["selection_tuple"]
    outcome = payload["selection_outcome"]
    fallback_reason = payload["fallback_reason"]
    if outcome == "fallback_final":
        if (
            payload["selection_count"] != 0
            or selection_tuple is not None
            or fallback_reason not in FALLBACK_REASON_CODES
            or payload["selected_checkpoint_timestep"] != end
        ):
            raise ValueError("fallback selection provenance is invalid")
    elif outcome != "best_model":
        raise ValueError("runtime metrics selection_outcome is invalid")
    elif (
        fallback_reason is not None
        or payload["selection_count"] != CANONICAL_SELECTION_COUNT
    ):
        raise ValueError(
            "best-model selection_count/fallback provenance is invalid"
        )
    if payload["selected_checkpoint_timestep"] > end:
        raise ValueError(
            "runtime metrics selected checkpoint exceeds final timestep"
        )
    if outcome == "best_model" and (
        not isinstance(selection_tuple, list)
        or len(selection_tuple) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in selection_tuple
        )
    ):
        raise ValueError("runtime metrics selection_tuple is invalid")
    return payload


def read_runtime_metrics(path: str | Path) -> dict[str, Any]:
    return _validate_runtime_metrics_payload(read_json_object(path))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_requirement_line(line: str) -> str:
    """Remove URL credentials and transient query/fragment credentials."""
    def redact(match: re.Match[str]) -> str:
        parsed = urlsplit(match.group(0))
        hostname = parsed.hostname or ""
        host = hostname if parsed.port is None else f"{hostname}:{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))

    return _URL_PATTERN.sub(redact, line.strip())


def _run_text(command: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _pip_freeze() -> list[str]:
    output = _run_text([sys.executable, "-m", "pip", "freeze"])
    if output is not None:
        return sorted(
            sanitize_requirement_line(line)
            for line in output.splitlines()
            if line.strip()
        )
    # Some narrowly provisioned test/runtime environments intentionally omit pip.
    return sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )


def _boot_id() -> str:
    boot_file = Path("/proc/sys/kernel/random/boot_id")
    if platform.system() == "Linux" and boot_file.is_file():
        value = boot_file.read_text(encoding="utf-8").strip()
        if value:
            return value
    if platform.system() == "Windows":
        global _WINDOWS_BOOT_ID
        if _WINDOWS_BOOT_ID is not None:
            return _WINDOWS_BOOT_ID
        output = _run_text(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "(Get-CimInstance -ClassName Win32_OperatingSystem -ErrorAction Stop).LastBootUpTime.ToUniversalTime().ToString('o')",
            ]
        )
        timestamp = output.strip() if isinstance(output, str) else ""
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as error:
            raise RuntimeError("Windows boot timestamp is unavailable or invalid") from error
        if not timestamp or parsed.tzinfo is None:
            raise RuntimeError("Windows boot timestamp is unavailable or invalid")
        normalized = parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")
        _WINDOWS_BOOT_ID = "windows-" + hashlib.sha256(
            f"windows-{normalized}".encode("utf-8")
        ).hexdigest()
        return _WINDOWS_BOOT_ID
    return f"process-{os.getpid()}"


def _git_metadata() -> tuple[str | None, bool | None]:
    sha = _run_text(["git", "rev-parse", "HEAD"])
    dirty = _run_text(["git", "status", "--porcelain"])
    return (sha.strip() if sha else None, bool(dirty) if dirty is not None else None)


def _physical_gpu_identifier(index: int) -> str:
    """Map a logical CUDA index through CUDA_VISIBLE_DEVICES when present."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return str(index)
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    return tokens[index] if index < len(tokens) else str(index)


def _gpu_uuid(index: int) -> str | None:
    output = _run_text(
        [
            "nvidia-smi",
            f"--id={_physical_gpu_identifier(index)}",
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ]
    )
    return output.splitlines()[0].strip() if output and output.splitlines() else None


def _resolved_cuda_index(device: str, cuda_available: bool) -> int | None:
    if not cuda_available or device == "cpu":
        return None
    if device == "cuda":
        return torch.cuda.current_device()
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    return None


def collect_environment(
    command: Sequence[str], provenance: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Capture safe, reproducible process/runtime facts without environment data."""
    supplied = dict(provenance or {})
    git_sha, git_dirty = _git_metadata()
    cuda_available = torch.cuda.is_available()
    requested_device = supplied.get(
        "resolved_device", "cuda:0" if cuda_available else "cpu"
    )
    cuda_index = _resolved_cuda_index(requested_device, cuda_available)
    if cuda_index is not None:
        properties = torch.cuda.get_device_properties(cuda_index)
        gpu_name: str | None = torch.cuda.get_device_name(cuda_index)
        gpu_memory: int | None = int(properties.total_memory)
        resolved_device = f"cuda:{cuda_index}"
        gpu_uuid = _gpu_uuid(cuda_index)
    else:
        gpu_name = None
        gpu_memory = None
        resolved_device = "cpu"
        gpu_uuid = None
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command),
        "python_version": sys.version,
        "platform": platform.platform(),
        "comparison_git_sha": supplied.get("comparison_git_sha", git_sha),
        "comparison_git_dirty": supplied.get("comparison_git_dirty", git_dirty),
        "baseline_sha256": supplied.get("baseline_sha256"),
        "config_sha256": supplied.get("config_sha256"),
        "scenario_sha256": supplied.get("scenario_sha256"),
        "split_sha256": supplied.get("split_sha256"),
        "lock_sha256": supplied.get("lock_sha256"),
        "vm_boot_id": _boot_id(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "resolved_device": resolved_device,
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "gpu_total_memory_bytes": gpu_memory,
        "cpu_count": os.cpu_count(),
        "process_id": os.getpid(),
        "pip_freeze": _pip_freeze(),
    }


def _parameter_ids(module: Any) -> set[int]:
    if module is None or not hasattr(module, "parameters"):
        return set()
    return {id(parameter) for parameter in module.parameters() if parameter.requires_grad}


def count_trainable_parameters(model: Any) -> dict[str, int]:
    """Partition each trainable parameter identity once across policy components."""
    all_parameters = {
        id(parameter): parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    feature_ids = _parameter_ids(getattr(model, "features_extractor", None))
    mlp = getattr(model, "mlp_extractor", None)
    value_ids = _parameter_ids(getattr(mlp, "value_net", None))
    value_ids |= _parameter_ids(getattr(model, "value_net", None))
    value_ids |= _parameter_ids(getattr(model, "value_head", None))
    value_ids -= feature_ids
    policy_ids = set(all_parameters) - feature_ids - value_ids
    assert feature_ids.isdisjoint(value_ids)
    assert feature_ids.isdisjoint(policy_ids)
    assert value_ids.isdisjoint(policy_ids)
    assert feature_ids | value_ids | policy_ids == set(all_parameters)

    def count(ids: set[int]) -> int:
        return sum(all_parameters[parameter_id].numel() for parameter_id in ids)

    return {
        "total": count(set(all_parameters)),
        "feature_extractor": count(feature_ids),
        "policy": count(policy_ids),
        "value": count(value_ids),
    }


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        stream.write(_canonical_json(payload) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        os.replace(temporary, destination)
        read_json_object(destination)
    finally:
        temporary.unlink(missing_ok=True)


def append_environment_segment(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically append one record to a strictly valid JSON-lines ledger."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        resolve_direct_regular_file(
            destination.parent,
            destination,
            label="environment segment ledger",
        )
    except FileNotFoundError:
        existing = []
    else:
        existing = read_environment_segments(destination)
    record = parse_json_object(_canonical_json(dict(payload)))
    expected = [*existing, record]
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        for item in expected:
            stream.write(_canonical_json(item) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        os.replace(temporary, destination)
        if read_environment_segments(destination) != expected:
            raise ValueError(
                "environment segment reread differs from written payload"
            )
    finally:
        temporary.unlink(missing_ok=True)


def read_environment_segments(path: str | Path) -> list[dict[str, Any]]:
    """Read a complete JSON-lines ledger and reject all partial/corrupt bytes."""
    requested = Path(path)
    try:
        source = resolve_direct_regular_file(
            requested.parent,
            requested,
            label="environment segment ledger",
        )
        raw = source.read_bytes()
        if not raw or not raw.endswith(b"\n"):
            raise ValueError("environment segment file must end with a newline")
        text = raw.decode("utf-8")
        lines = text.splitlines()
        if not lines or any(not line for line in lines):
            raise ValueError("environment segment file contains a blank record")
        return [parse_json_object(line) for line in lines]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as error:
        if isinstance(error, ValueError) and str(error).startswith(
            "environment segment"
        ):
            raise
        raise ValueError(f"invalid environment segment file: {requested}") from error


def write_runtime_metrics(path: str | Path, metrics: Mapping[str, Any]) -> None:
    payload = dict(metrics)
    # Validate the exact schema before publication and again after reread.
    _validate_runtime_metrics_payload(payload)
    destination = Path(path)
    _write_json(destination, payload)
    if read_runtime_metrics(destination) != payload:
        raise ValueError("runtime metrics reread differs from written payload")


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    _write_json(path, manifest)
