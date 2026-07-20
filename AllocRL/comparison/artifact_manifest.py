"""Canonical, credential-safe runtime metadata for comparison artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import torch


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


_URL_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s]+")


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
    if boot_file.is_file():
        value = boot_file.read_text(encoding="utf-8").strip()
        if value:
            return value
    # Windows lacks a Linux boot-id API.  The process-local fallback is explicit
    # and stable throughout a training subprocess without exposing host secrets.
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
    destination.write_text(_canonical_json(payload) + "\n", encoding="utf-8")


def append_environment_segment(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(_canonical_json(payload) + "\n")


def write_runtime_metrics(path: str | Path, metrics: Mapping[str, Any]) -> None:
    _write_json(path, metrics)


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    _write_json(path, manifest)
