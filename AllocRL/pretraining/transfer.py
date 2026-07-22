"""Strict transfer of production-eligible Stage 1 extractor weights."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from alloc_env.cnn_extractor import CandidateCnnExtractor


_MARKER_KEYS = {
    "schema_version",
    "observation_schema_version",
    "checkpoint_filename",
    "checkpoint_sha256",
    "metrics_filename",
    "metrics_sha256",
    "config_sha256",
    "dataset_manifest_sha256",
    "gates",
    "smoke_mode",
    "production_eligible",
}
_CHECKPOINT_KEYS = {
    "checkpoint_schema_version",
    "observation_schema_version",
    "config_sha256",
    "dataset_manifest_sha256",
    "best_epoch",
    "extractor_state_dict",
}


@dataclass(frozen=True)
class PretrainingReceipt:
    checkpoint_sha256: str
    manifest_sha256: str
    complete_sha256: str
    config_sha256: str
    metrics_sha256: str


@dataclass(frozen=True)
class VerifiedPretrainingArtifacts:
    receipt: PretrainingReceipt
    extractor_state_dict: Mapping[str, torch.Tensor]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _require_exact_keys(
    values: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(values)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{label} keys differ: missing={missing}, unknown={unknown}"
        )


def verify_pretraining_artifacts(
    checkpoint_path: Path,
    complete_path: Path,
) -> VerifiedPretrainingArtifacts:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    complete_path = Path(complete_path).expanduser().resolve()
    if not complete_path.is_file():
        raise FileNotFoundError(
            f"PRETRAINING_COMPLETE marker not found: {complete_path}"
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"pretraining checkpoint not found: {checkpoint_path}"
        )

    marker = _read_json(complete_path, "pretraining completion marker")
    _require_exact_keys(marker, _MARKER_KEYS, "completion marker")
    if marker["schema_version"] != 1:
        raise ValueError("completion marker schema must be 1")
    if marker["observation_schema_version"] != 4:
        raise ValueError("completion marker observation schema must be 4")
    if marker["smoke_mode"] is not False:
        raise ValueError("smoke pretraining is not production eligible")
    if marker["production_eligible"] is not True:
        raise ValueError("pretraining marker is not production eligible")
    gates = marker["gates"]
    if (
        not isinstance(gates, Mapping)
        or not gates
        or any(value is not True for value in gates.values())
    ):
        raise ValueError("all pretraining gates must pass")
    if marker["checkpoint_filename"] != checkpoint_path.name:
        raise ValueError("completion marker checkpoint filename differs")
    if (complete_path.parent / checkpoint_path.name).resolve() != checkpoint_path:
        raise ValueError("checkpoint must be beside PRETRAINING_COMPLETE")

    checkpoint_sha = _sha256_file(checkpoint_path)
    if marker["checkpoint_sha256"] != checkpoint_sha:
        raise ValueError("pretraining checkpoint SHA256 mismatch")
    metrics_path = complete_path.parent / str(marker["metrics_filename"])
    if not metrics_path.is_file():
        raise FileNotFoundError(f"pretraining metrics not found: {metrics_path}")
    metrics_sha = _sha256_file(metrics_path)
    if marker["metrics_sha256"] != metrics_sha:
        raise ValueError("pretraining metrics SHA256 mismatch")
    metrics = _read_json(metrics_path, "pretraining metrics")
    if metrics.get("gates") != gates:
        raise ValueError("pretraining metrics gates differ from marker")
    if metrics.get("smoke_mode") is not False:
        raise ValueError("pretraining metrics describe a smoke run")

    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise ValueError("invalid pretraining checkpoint payload") from error
    if not isinstance(payload, Mapping):
        raise ValueError("pretraining checkpoint payload must be a mapping")
    _require_exact_keys(payload, _CHECKPOINT_KEYS, "checkpoint payload")
    if payload["checkpoint_schema_version"] != 1:
        raise ValueError("checkpoint schema must be 1")
    if payload["observation_schema_version"] != 4:
        raise ValueError("checkpoint observation schema must be 4")
    if payload["config_sha256"] != marker["config_sha256"]:
        raise ValueError("pretraining config SHA256 mismatch")
    if (
        payload["dataset_manifest_sha256"]
        != marker["dataset_manifest_sha256"]
    ):
        raise ValueError("pretraining dataset manifest SHA256 mismatch")
    state_dict = payload["extractor_state_dict"]
    if not isinstance(state_dict, Mapping) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in state_dict.items()
    ):
        raise ValueError("extractor state dict must contain only named tensors")

    return VerifiedPretrainingArtifacts(
        receipt=PretrainingReceipt(
            checkpoint_sha256=checkpoint_sha,
            manifest_sha256=str(marker["dataset_manifest_sha256"]),
            complete_sha256=_sha256_file(complete_path),
            config_sha256=str(marker["config_sha256"]),
            metrics_sha256=metrics_sha,
        ),
        extractor_state_dict=state_dict,
    )


def load_verified_pretrained_extractor(
    model: Any,
    checkpoint_path: Path,
    complete_path: Path,
) -> PretrainingReceipt:
    verified = verify_pretraining_artifacts(checkpoint_path, complete_path)
    extractor = getattr(model.policy, "features_extractor", None)
    if not isinstance(extractor, CandidateCnnExtractor):
        raise TypeError(
            "pretrained transfer requires CandidateCnnExtractor"
        )
    expected = set(extractor.state_dict())
    actual = set(verified.extractor_state_dict)
    if actual != expected:
        raise ValueError(
            "extractor state keys differ: "
            f"missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    try:
        extractor.load_state_dict(verified.extractor_state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError("extractor tensors are incompatible") from error
    return verified.receipt

