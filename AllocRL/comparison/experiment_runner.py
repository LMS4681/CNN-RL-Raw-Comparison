"""Fail-closed, resumable orchestration for the overnight comparison.

This module intentionally owns only root-level orchestration artifacts.  Model
training, checkpoint persistence, evaluation and reporting remain in their
specialist modules so a restart can verify rather than infer their state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
import uuid
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Mapping, Sequence

from comparison.artifact_manifest import (
    REQUIRED_ENVIRONMENT_KEYS,
    canonical_json_sha256,
    collect_environment,
    sha256_file,
)
from comparison.checkpoint_evaluator import evaluate_comparison_artifacts
from comparison.report_builder import PAIR_COLUMNS, JOURNAL_STAGES, JOURNAL_STATUSES, build_comparison_summary, build_paired_differences, write_complete_report, write_partial_report
from comparison.wall_clock_callback import atomic_write_json, read_wall_clock_state, resolve_state_checkpoint
from evaluation_scenarios import read_scenarios
from holdout_model_selection import validate_fixed_holdout_scenarios


PRODUCTION_CONFIG = {
    "schema_version": 1, "baseline_commit": "cd4e14fc1725a4ff159e59d6874d3602f3b65a06",
    "fixed_scenarios_sha256": "6125f53939a1b8eef8662b2628c0da2f1d0f26b5b541a99252858326b38cd814",
    "split_manifest_path": "data/data_split_manifest.json",
    "split_manifest_sha256": "d3df1d0076248b4bcbddb4c910a3cb81481da65c7415c6b3cacf9e055cc3f9df",
    "seed": 0, "state_context": "full", "target_training_seconds_per_arm": 10800,
    "timesteps_ceiling": 2_000_000_000, "learning_rate": 0.0003, "n_steps": 960,
    "batch_size": 64, "n_epochs": 10, "gamma": 1.0, "gae_lambda": 0.98,
    "n_envs": 1, "vec_env": "auto", "device": "auto", "checkpoint_freq": 10_000,
    "checkpoint_heartbeat_seconds": 300, "holdout_eval_freq": 50_000,
    "holdout_selection_count": 5, "smoke_timesteps": 1024,
    "scenario_path": "data/fixed_eval_scenarios.json", "dependency_lock_path": "requirements-comparison.txt",
}
_OPERATING_OVERRIDES = frozenset({"target_training_seconds_per_arm", "timesteps_ceiling", "checkpoint_freq", "checkpoint_heartbeat_seconds", "holdout_eval_freq", "smoke_timesteps"})
_SHA256 = __import__("re").compile(r"[0-9a-f]{64}\Z")
_SHA1 = __import__("re").compile(r"[0-9a-f]{40}\Z")
_CUDA_DEVICE = __import__("re").compile(r"cuda:[0-9]+\Z")
REQUIRED_COMPLETE_STAGES = list(JOURNAL_STAGES)


class ExperimentStageError(RuntimeError): pass
class ExperimentIntegrityError(RuntimeError): pass
class LeaseError(RuntimeError): pass


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict): raise ValueError(f"JSON object required: {path}")
    return value


def _valid_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_root_environment(
    environment: Mapping[str, Any],
    provenance: Mapping[str, str],
    *,
    production_loaded: bool,
) -> None:
    """Reject a root capture that cannot prove its checkout and hardware facts."""
    if set(environment) != set(REQUIRED_ENVIRONMENT_KEYS):
        raise ExperimentIntegrityError("root environment schema differs")
    command = environment["command"]
    if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
        raise ExperimentIntegrityError("root environment command must be a list of strings")
    if not isinstance(environment["comparison_git_sha"], str) or _SHA1.fullmatch(environment["comparison_git_sha"]) is None:
        raise ExperimentIntegrityError("comparison_git_sha must be a lowercase commit SHA")
    if environment["comparison_git_dirty"] is not False:
        raise ExperimentIntegrityError("comparison checkout must identify a clean commit")
    for key, pattern in (
        ("baseline_sha256", _SHA1),
        ("config_sha256", _SHA256),
        ("scenario_sha256", _SHA256),
        ("split_sha256", _SHA256),
        ("lock_sha256", _SHA256),
    ):
        value = environment[key]
        if not isinstance(value, str) or pattern.fullmatch(value) is None or value != provenance.get(key):
            raise ExperimentIntegrityError(f"root environment provenance mismatch: {key}")
    for key in ("captured_at_utc", "python_version", "platform", "vm_boot_id", "torch_version"):
        if not _nonempty_string(environment[key]):
            raise ExperimentIntegrityError(f"root environment type mismatch: {key}")
    if environment["cuda_version"] is not None and not _nonempty_string(environment["cuda_version"]):
        raise ExperimentIntegrityError("root environment type mismatch: cuda_version")
    if (environment["cudnn_version"] is not None
            and (not isinstance(environment["cudnn_version"], (str, int)) or isinstance(environment["cudnn_version"], bool))):
        raise ExperimentIntegrityError("root environment type mismatch: cudnn_version")
    if not isinstance(environment["pip_freeze"], list) or not all(isinstance(item, str) for item in environment["pip_freeze"]):
        raise ExperimentIntegrityError("root environment type mismatch: pip_freeze")
    if (not isinstance(environment["cpu_count"], int) or isinstance(environment["cpu_count"], bool) or environment["cpu_count"] <= 0
            or not isinstance(environment["process_id"], int) or isinstance(environment["process_id"], bool) or environment["process_id"] <= 0):
        raise ExperimentIntegrityError("root environment type mismatch: process metadata")
    device = environment["resolved_device"]
    is_cuda = isinstance(device, str) and _CUDA_DEVICE.fullmatch(device) is not None
    gpu_values = (environment["gpu_name"], environment["gpu_uuid"], environment["gpu_total_memory_bytes"])
    has_gpu_identity = (_nonempty_string(gpu_values[0]) and _nonempty_string(gpu_values[1])
                        and isinstance(gpu_values[2], int) and not isinstance(gpu_values[2], bool) and gpu_values[2] > 0)
    if production_loaded and (not is_cuda or not has_gpu_identity):
        raise ExperimentIntegrityError("production preflight requires CUDA resolved device with GPU identity")
    if device == "cpu":
        if gpu_values != (None, None, None):
            raise ExperimentIntegrityError("CPU environment must not claim GPU metadata")
    elif not is_cuda or not has_gpu_identity:
        raise ExperimentIntegrityError("CUDA environment metadata is incoherent")


def _validate_root_manifest(
    manifest: Mapping[str, Any],
    provenance: Mapping[str, str],
    environment: Mapping[str, Any],
) -> None:
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != 1:
        raise ExperimentIntegrityError("root manifest schema differs")
    for key, value in provenance.items():
        if manifest.get(key) != value:
            raise ExperimentIntegrityError(f"root manifest provenance mismatch: {key}")
    if manifest.get("comparison_git_sha") != environment["comparison_git_sha"] or manifest.get("comparison_git_dirty") is not False:
        raise ExperimentIntegrityError("root manifest comparison checkout mismatch")
    if not isinstance(manifest.get("checkpoints"), dict):
        raise ExperimentIntegrityError("root manifest checkpoints must be an object")


_COMPARISON_ARMS = ("raw_direct", "candidate_cnn")
_CHECKPOINT_KINDS = ("selected", "final", "common")
_CHECKPOINT_REF_KEYS = frozenset({"path", "label", "sha256", "timestep"})


def _safe_checkpoint_path(root: Path, arm: str, value: Any) -> Path:
    if not _nonempty_string(value) or "\\" in value:
        raise ExperimentIntegrityError("checkpoint path must be a nonempty root-relative POSIX path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or ".." in posix.parts:
        raise ExperimentIntegrityError("checkpoint path escapes the output root")
    raw_path = root / posix
    try:
        resolved = raw_path.resolve(strict=True)
        resolved.relative_to(root)
        resolved.relative_to((root / arm).resolve())
    except (OSError, RuntimeError, ValueError, FileNotFoundError) as error:
        raise ExperimentIntegrityError("checkpoint path escapes or is absent from its arm directory") from error
    if raw_path.is_symlink() or not resolved.is_file():
        raise ExperimentIntegrityError("checkpoint reference must name a regular in-root file")
    return resolved


def _validate_checkpoint_manifest(
    root: Path,
    manifest: Mapping[str, Any],
    archive_timestep_reader: Callable[[Path], int | None] | None,
) -> None:
    """Verify every manifest reference against the archive and complete state."""
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != set(_COMPARISON_ARMS):
        raise ExperimentIntegrityError("missing paired checkpoint manifest")
    from train import model_num_timesteps
    reader = archive_timestep_reader or model_num_timesteps
    root = root.resolve()
    for arm in _COMPARISON_ARMS:
        refs = checkpoints[arm]
        if not isinstance(refs, Mapping) or set(refs) != set(_CHECKPOINT_KINDS):
            raise ExperimentIntegrityError("checkpoint refs incomplete")
        verified: dict[str, tuple[dict[str, Any], Path]] = {}
        for kind in _CHECKPOINT_KINDS:
            ref = refs[kind]
            if not isinstance(ref, dict) or set(ref) != _CHECKPOINT_REF_KEYS:
                raise ExperimentIntegrityError("checkpoint reference has invalid schema")
            label = ref["label"]
            allowed_labels = {
                "selected": {"best_model", "fallback_final"},
                "final": {"final"},
                "common": {"common_step"},
            }[kind]
            if not isinstance(label, str) or label not in allowed_labels:
                raise ExperimentIntegrityError("checkpoint reference label is dishonest")
            timestep = ref["timestep"]
            if not isinstance(timestep, int) or isinstance(timestep, bool) or timestep < 0:
                raise ExperimentIntegrityError("checkpoint reference timestep must be a non-negative integer")
            digest = ref["sha256"]
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise ExperimentIntegrityError("checkpoint reference SHA256 is invalid")
            path = _safe_checkpoint_path(root, arm, ref["path"])
            try:
                actual_digest = sha256_file(path)
                actual_timestep = reader(path)
            except (OSError, ValueError, TypeError) as error:
                raise ExperimentIntegrityError("checkpoint reference cannot be read") from error
            if actual_digest != digest:
                raise ExperimentIntegrityError("tampered checkpoint reference")
            if not isinstance(actual_timestep, int) or isinstance(actual_timestep, bool) or actual_timestep != timestep:
                raise ExperimentIntegrityError("checkpoint archive timestep differs from manifest")
            verified[kind] = (ref, path)

        final_ref, final_path = verified["final"]
        try:
            state = read_wall_clock_state(root / arm / "run_state.json")
            state_path = resolve_state_checkpoint(root / arm, state)
            state_timestep = reader(state_path)
        except (OSError, ValueError, TypeError, KeyError, FileNotFoundError) as error:
            raise ExperimentIntegrityError("final reference lacks a complete verified state checkpoint") from error
        if (state.status != "complete" or not isinstance(state_timestep, int) or isinstance(state_timestep, bool)
                or state_timestep != state.last_checkpoint_timestep
                or final_path != state_path.resolve()
                or final_ref["sha256"] != state.last_checkpoint_sha256
                or final_ref["timestep"] != state.last_checkpoint_timestep):
            raise ExperimentIntegrityError("final reference is not the exact complete state checkpoint")

        selected_ref, selected_path = verified["selected"]
        if selected_ref["label"] == "fallback_final":
            if (selected_path != final_path or selected_ref["sha256"] != final_ref["sha256"]
                    or selected_ref["timestep"] != final_ref["timestep"]):
                raise ExperimentIntegrityError("fallback selected reference must equal final state checkpoint")
        elif selected_ref["path"] != f"{arm}/best_model.sb3" or selected_path == final_path:
            raise ExperimentIntegrityError("best-model selected reference is dishonest")

        common_ref, _ = verified["common"]
        if not common_ref["path"].startswith(f"{arm}/checkpoints/"):
            raise ExperimentIntegrityError("common checkpoint must be stored beneath its arm checkpoint directory")


def _strict_json_line(line: str) -> dict[str, Any]:
    def object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result
    value = json.loads(line, object_pairs_hook=object_without_duplicate_keys)
    if not isinstance(value, dict):
        raise ValueError("environment segment must be a JSON object")
    return value


_REPORT_ARTIFACTS = (
    "summary.json", "scenario_paired_differences.csv", "learning_curves.png",
    "holdout_comparison.png", "preliminary_comparison_ko.md",
)


def _validate_report_artifacts(root: Path) -> dict[str, str]:
    """Rebuild report inputs and require byte-identical deterministic outputs."""
    base = Path(root)
    comparison = base / "comparison"
    try:
        summary = build_comparison_summary(base)
        pairs = build_paired_differences(base)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError, KeyError) as error:
        raise ExperimentIntegrityError("report inputs no longer reconcile") from error
    summary_path = comparison / "summary.json"
    try:
        summary_bytes = summary_path.read_bytes()
        decoded_summary = summary_bytes.decode("utf-8")
        persisted_summary = _strict_json_line(decoded_summary)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as error:
        raise ExperimentIntegrityError("summary.json is not strict canonical JSON") from error
    expected_summary = _strict_json_line(_canonical(summary).decode("utf-8"))
    if persisted_summary != expected_summary or summary_bytes != _canonical(summary):
        raise ExperimentIntegrityError("summary.json differs from current canonical report inputs")

    pairs_path = comparison / "scenario_paired_differences.csv"
    try:
        with pairs_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != PAIR_COLUMNS:
                raise ValueError("paired CSV header differs")
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error, ValueError) as error:
        raise ExperimentIntegrityError("paired-difference CSV is invalid") from error
    if len(rows) != 15:
        raise ExperimentIntegrityError("paired-difference CSV must have exactly 15 rows")
    normalized: list[dict[str, Any]] = []
    try:
        for expected_seed, row in zip(range(1005, 1020), rows):
            if set(row) != set(PAIR_COLUMNS) or row["seed"] != str(expected_seed):
                raise ValueError("paired CSV seed order differs")
            parsed = {"seed": expected_seed}
            for field in PAIR_COLUMNS[1:]:
                value = float(row[field])
                if not math.isfinite(value):
                    raise ValueError("paired CSV value is non-finite")
                parsed[field] = value
            normalized.append(parsed)
    except (KeyError, TypeError, ValueError) as error:
        raise ExperimentIntegrityError("paired-difference CSV rows are invalid") from error
    if normalized != pairs:
        raise ExperimentIntegrityError("paired-difference CSV differs from current report inputs")

    for name in ("learning_curves.png", "holdout_comparison.png"):
        path = comparison / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise ExperimentIntegrityError(f"missing report artifact: {name}")
    markdown = comparison / "preliminary_comparison_ko.md"
    try:
        text = markdown.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ExperimentIntegrityError("Korean report is not valid UTF-8") from error
    if not text.strip() or "\ufffd" in text or "seed 0" not in text or "primary_test" not in text:
        raise ExperimentIntegrityError("Korean report lacks required preliminary limitations")
    try:
        return {name: sha256_file(comparison / name) for name in _REPORT_ARTIFACTS}
    except OSError as error:
        raise ExperimentIntegrityError("report artifact disappeared during verification") from error


def _validate_environment_segments(
    root: Path,
    environment: Mapping[str, Any],
    provenance: Mapping[str, str],
    *,
    production_loaded: bool,
) -> None:
    """Require every persisted arm/restart environment to equal the root facts."""
    _validate_root_environment(environment, provenance, production_loaded=production_loaded)
    comparison_keys = (
        "vm_boot_id", "resolved_device", "gpu_name", "gpu_uuid",
        "gpu_total_memory_bytes", "torch_version", "cuda_version",
        "cudnn_version", "lock_sha256", "comparison_git_sha", "comparison_git_dirty",
    )
    expected = tuple(environment[key] for key in comparison_keys)
    for arm in _COMPARISON_ARMS:
        path = root / arm / "environment_segments.jsonl"
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as error:
            raise ExperimentIntegrityError(f"missing environment segments for {arm}") from error
        records: list[dict[str, Any]] = []
        try:
            for line in lines:
                if line.strip():
                    records.append(_strict_json_line(line))
        except (json.JSONDecodeError, ValueError, TypeError) as error:
            raise ExperimentIntegrityError(f"invalid environment segment for {arm}") from error
        if not records:
            raise ExperimentIntegrityError(f"empty environment segments for {arm}")
        for record in records:
            _validate_root_environment(record, provenance, production_loaded=production_loaded)
            if tuple(record[key] for key in comparison_keys) != expected:
                raise ExperimentIntegrityError("arms must use the same Colab VM/GPU/library environment")


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int; baseline_commit: str; fixed_scenarios_sha256: str
    split_manifest_path: str; split_manifest_sha256: str; seed: int; state_context: str
    target_training_seconds_per_arm: float; timesteps_ceiling: int; learning_rate: float
    n_steps: int; batch_size: int; n_epochs: int; gamma: float; gae_lambda: float
    n_envs: int; vec_env: str; device: str; checkpoint_freq: int
    checkpoint_heartbeat_seconds: float; holdout_eval_freq: int; holdout_selection_count: int
    smoke_timesteps: int; scenario_path: str; dependency_lock_path: str
    config_sha256: str = field(default="", compare=True)
    path: Path | None = field(default=None, compare=False, repr=False)
    production_loaded: bool = field(default=False, compare=False, repr=False)

    @classmethod
    def for_test(cls, **operational_overrides: int | float) -> "ExperimentConfig":
        unknown = set(operational_overrides) - _OPERATING_OVERRIDES
        if unknown: raise ValueError(f"test overrides are operational only: {sorted(unknown)}")
        payload = dict(PRODUCTION_CONFIG); payload.update(operational_overrides)
        _validate_config_payload(payload, allow_operational_overrides=True)
        return cls(**payload, config_sha256=canonical_json_sha256(payload), production_loaded=False)


def _validate_config_payload(payload: Mapping[str, Any], *, allow_operational_overrides: bool = False) -> None:
    expected = set(PRODUCTION_CONFIG)
    if set(payload) != expected:
        raise ValueError(f"config keys differ: missing={sorted(expected-set(payload))}, extra={sorted(set(payload)-expected)}")
    for key, expected_value in PRODUCTION_CONFIG.items():
        value = payload[key]
        if key in _OPERATING_OVERRIDES and allow_operational_overrides:
            if not _valid_number(value) or float(value) < 0: raise ValueError(f"{key} must be a non-negative finite number")
            if key != "holdout_eval_freq" and float(value) <= 0: raise ValueError(f"{key} must be a positive finite number")
            if key in {"timesteps_ceiling", "checkpoint_freq", "holdout_eval_freq", "smoke_timesteps"} and (not isinstance(value, int) or isinstance(value, bool)): raise ValueError(f"{key} must be an integer")
            continue
        if isinstance(expected_value, float):
            if not _valid_number(value) or float(value) != expected_value: raise ValueError(f"immutable config value differs: {key}")
        elif type(value) is not type(expected_value) or value != expected_value:
            raise ValueError(f"immutable config value differs: {key}")
    if _SHA1.fullmatch(payload["baseline_commit"]) is None or _SHA256.fullmatch(payload["fixed_scenarios_sha256"]) is None or _SHA256.fullmatch(payload["split_manifest_sha256"]) is None:
        raise ValueError("invalid immutable provenance digest")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    source = Path(path).resolve(); payload = _json(source); _validate_config_payload(payload)
    return ExperimentConfig(**payload, config_sha256=canonical_json_sha256(payload), path=source, production_loaded=True)


def _arm(arm: str) -> tuple[str, str]:
    if arm == "raw_direct": return arm, "raw-direct"
    if arm == "candidate_cnn": return arm, "candidate-cnn"
    raise ValueError("arm must be raw_direct or candidate_cnn")


def _allocrl_dir() -> Path: return Path(__file__).resolve().parents[1]


def build_smoke_command(arm: str, config: ExperimentConfig, *, output_root: str | Path = "output", python_executable: str | None = None) -> list[str]:
    name, extractor = _arm(arm); root = Path(output_root)
    return [python_executable or sys.executable, "smoke_test.py", "--extractor", extractor, "--timesteps", str(config.smoke_timesteps), "--device", "cuda", "--output-dir", str(root / "smoke" / name)]


def build_train_command(arm: str, config: ExperimentConfig, resume_path: Path | None = None, *, output_root: str | Path = "output", python_executable: str | None = None, lock_sha256: str | None = None) -> list[str]:
    name, extractor = _arm(arm); root = Path(output_root); arm_root = root / name
    if not isinstance(lock_sha256, str) or _SHA256.fullmatch(lock_sha256) is None:
        raise ExperimentIntegrityError("build_train_command requires a real preflight lock SHA-256")
    command = [python_executable or sys.executable, "train.py", "--output-dir", str(arm_root), "--timesteps", str(config.timesteps_ceiling), "--lr", str(config.learning_rate), "--n-steps", str(config.n_steps), "--batch-size", str(config.batch_size), "--n-epochs", str(config.n_epochs), "--gamma", str(config.gamma), "--gae-lambda", str(config.gae_lambda), "--n-envs", str(config.n_envs), "--vec-env", config.vec_env, "--device", config.device, "--seed", str(config.seed), "--extractor", extractor, "--state-context", config.state_context, "--eval-scenarios", config.scenario_path, "--max-training-seconds", str(config.target_training_seconds_per_arm), "--wall-clock-heartbeat-seconds", str(config.checkpoint_heartbeat_seconds), "--comparison-config-sha256", config.config_sha256, "--comparison-baseline-sha256", config.baseline_commit, "--comparison-scenario-sha256", config.fixed_scenarios_sha256, "--comparison-split-sha256", config.split_manifest_sha256, "--comparison-lock-sha256", lock_sha256, "--checkpoint-freq", str(config.checkpoint_freq), "--holdout-eval-freq", str(config.holdout_eval_freq), "--holdout-selection-count", str(config.holdout_selection_count), "--no-export-onnx"]
    if resume_path is not None: command += ["--resume-from", str(Path(resume_path))]
    return command


def _tree_sha(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        digest.update(b"absent"); return digest.hexdigest()
    if path.is_file(): return sha256_file(path)
    for item in sorted(path.rglob("*"), key=lambda value: value.as_posix()):
        if item.is_file(): digest.update(item.relative_to(path).as_posix().encode()); digest.update(sha256_file(item).encode())
    return digest.hexdigest()


def _journal_entry(status: str = "pending", *, input_sha256: str | None = None, output_sha256: str | None = None, started_at_utc: str | None = None, completed_at_utc: str | None = None, error: str | None = None) -> dict[str, Any]:
    return {"status": status, "input_sha256": input_sha256, "output_sha256": output_sha256, "started_at_utc": started_at_utc, "completed_at_utc": completed_at_utc, "error": error}


class _Lease(AbstractContextManager["_Lease"]):
    def __init__(self, root: Path, *, stale_takeover: bool, clock: Callable[[], float], interval: float = 60, stale_after: float = 900) -> None:
        self.path=root/"lease.json"; self.sentinel=root/".lease.acquire"; self.stale_takeover=stale_takeover; self.clock=clock; self.interval=interval; self.stale_after=stale_after; self.token=uuid.uuid4().hex; self.stop=threading.Event(); self.thread: threading.Thread | None=None; self.acquired=False; self.failure: BaseException | None=None
    def _payload(self, status: str) -> dict[str, Any]: return {"token": self.token, "pid": os.getpid(), "boot_id": _boot_id(), "heartbeat_utc": _utc(), "heartbeat_monotonic": self.clock(), "status": status}
    def _write(self, status: str) -> None:
        if self.path.exists() and _json(self.path).get("token") != self.token:
            raise LeaseError("lease ownership token changed")
        atomic_write_json(self.path, self._payload(status))
    def _claim_sentinel(self) -> None:
        try:
            handle=os.open(self.sentinel, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise LeaseError("another live comparison runner owns this output root")
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(self.token + "\n"); stream.flush(); os.fsync(stream.fileno())
    def __enter__(self):
        if self.path.exists():
            prior=_json(self.path); status=prior.get("status"); age=self.clock()-float(prior.get("heartbeat_monotonic", float("-inf"))); foreign_boot=prior.get("boot_id") != _boot_id()
            if status != "released" and not foreign_boot and age < self.stale_after:
                raise LeaseError("another live comparison runner owns this output root")
            if status != "released" and not self.stale_takeover:
                raise LeaseError("stale lease requires explicit --take-over-stale-lease")
            if self.sentinel.exists():
                # Recheck the payload before stealing a stale sentinel so a live
                # owner cannot be overwritten between the first observation and unlink.
                current=_json(self.path)
                if current.get("token") != prior.get("token"):
                    raise LeaseError("lease changed while attempting stale takeover")
                self.sentinel.unlink()
        self._claim_sentinel(); self.acquired=True
        atomic_write_json(self.path, self._payload("active"))
        def refresh() -> None:
            while not self.stop.wait(self.interval):
                try: self._write("active")
                except BaseException as error: self.failure=error; self.stop.set(); return
        self.thread=threading.Thread(target=refresh, daemon=True, name="comparison-lease"); self.thread.start(); return self
    def __exit__(self, *exc: object) -> None:
        self.stop.set()
        if self.thread: self.thread.join(timeout=max(1, self.interval + 1))
        try: self._write("released")
        finally:
            try:
                if self.sentinel.exists() and self.sentinel.read_text(encoding="utf-8").strip() == self.token: self.sentinel.unlink()
            except OSError: pass
        if self.failure is not None and exc[0] is None: raise self.failure


def _boot_id() -> str:
    path=Path("/proc/sys/kernel/random/boot_id")
    return path.read_text().strip() if path.is_file() else f"process-{os.getpid()}"


class _Runner:
    def __init__(self, config: ExperimentConfig, root: Path, *, subprocess_runner: Callable[..., Any], clock: Callable[[], float], python_executable: str | None, archive_timestep_reader: Callable[[Path], int | None] | None, output_hasher: Callable[[str], str] | None = None, runner_command: Sequence[str] | None = None) -> None:
        command = list(sys.argv if runner_command is None else runner_command)
        if not all(isinstance(part, str) for part in command):
            raise ValueError("runner_command must contain only strings")
        self.config=config; self.root=root.resolve(); self.subprocess_runner=subprocess_runner; self.clock=clock; self.python=python_executable or sys.executable; self.archive_reader=archive_timestep_reader; self.journal_path=self.root/"stage_journal.json"; self.lock_sha=""; self._injected_output_hasher=output_hasher; self.runner_command=command
    def journal(self) -> dict[str, dict[str, Any]]:
        if not self.journal_path.exists(): return {name:_journal_entry() for name in JOURNAL_STAGES}
        data=_json(self.journal_path)
        if set(data)-set(JOURNAL_STAGES): raise ExperimentIntegrityError("unknown stage in journal")
        result={name:data.get(name,_journal_entry()) for name in JOURNAL_STAGES}
        for entry in result.values():
            if set(entry)!={"status","input_sha256","output_sha256","started_at_utc","completed_at_utc","error"} or entry["status"] not in JOURNAL_STATUSES: raise ExperimentIntegrityError("invalid stage journal")
            for key in ("input_sha256", "output_sha256"):
                if entry[key] is not None and (not isinstance(entry[key], str) or _SHA256.fullmatch(entry[key]) is None): raise ExperimentIntegrityError("invalid stage journal")
            for key in ("started_at_utc", "completed_at_utc"):
                if entry[key] is not None:
                    try: valid_time = isinstance(entry[key], str) and datetime.fromisoformat(entry[key].replace("Z", "+00:00")).tzinfo is not None
                    except ValueError: valid_time = False
                    if not valid_time: raise ExperimentIntegrityError("invalid stage journal")
            if entry["error"] is not None and (not isinstance(entry["error"], str) or "\ufffd" in entry["error"]): raise ExperimentIntegrityError("invalid stage journal")
            if entry["status"] == "complete" and (entry["input_sha256"] is None or entry["output_sha256"] is None or entry["started_at_utc"] is None or entry["completed_at_utc"] is None or entry["error"] is not None): raise ExperimentIntegrityError("invalid stage journal")
            if entry["status"] == "in_progress" and (entry["input_sha256"] is None or entry["started_at_utc"] is None or entry["output_sha256"] is not None or entry["completed_at_utc"] is not None or entry["error"] is not None): raise ExperimentIntegrityError("invalid stage journal")
            if entry["status"] in {"failed", "interrupted"} and entry["completed_at_utc"] is None: raise ExperimentIntegrityError("invalid stage journal")
            if entry["status"]=="in_progress": entry.update(_journal_entry("interrupted", input_sha256=entry["input_sha256"], output_sha256=entry["output_sha256"], started_at_utc=entry["started_at_utc"], completed_at_utc=_utc(), error="previous runner interrupted"))
        self.save_journal(result); return result
    def save_journal(self, data: Mapping[str, Any]) -> None: atomic_write_json(self.journal_path, dict(data))
    def stage_path(self, name: str) -> Path:
        return {"preflight":self.root/"manifest.json", "smoke_raw_direct":self.root/"smoke"/"raw_direct"/"runner_verified.json", "smoke_candidate_cnn":self.root/"smoke"/"candidate_cnn"/"runner_verified.json", "train_raw_direct":self.root/"raw_direct"/"run_state.json", "evaluate_raw_direct":self.root/"raw_direct"/"evaluation_stage.json", "train_candidate_cnn":self.root/"candidate_cnn"/"run_state.json", "evaluate_candidate_cnn":self.root/"candidate_cnn"/"evaluation_stage.json", "evaluate_common_step":self.root/"comparison"/"common_step_evaluation.csv", "build_report":self.root/"comparison"/"preliminary_comparison_ko.md", "integrity_verification":self.root/"integrity_verification.json"}[name]
    def output_hash(self, name: str) -> str:
        """Hash only artifacts owned by this stage, never mutable descendants."""
        if self._injected_output_hasher is not None: return self._injected_output_hasher(name)
        if name == "preflight":
            manifest = _json(self.root / "manifest.json")
            environment = _json(self.root / "environment.json")
            stable_manifest = {key: manifest.get(key) for key in ("schema_version", "baseline_sha256", "config_sha256", "scenario_sha256", "split_sha256", "lock_sha256", "comparison_git_sha", "comparison_git_dirty")}
            return canonical_json_sha256({"manifest": stable_manifest, "environment": environment})
        if name.startswith("smoke_"):
            return sha256_file(self.stage_path(name))
        if name.startswith("train_"):
            arm = name.removeprefix("train_")
            root = self.root / arm
            state = read_wall_clock_state(root / "run_state.json")
            checkpoint = resolve_state_checkpoint(root, state)
            owned = {"state": asdict(state), "checkpoint_sha256": sha256_file(checkpoint)}
            for filename in ("run_config.json", "runtime_metrics.json", "progress_timing.csv"):
                path = root / filename
                if path.is_file(): owned[filename] = sha256_file(path)
            return canonical_json_sha256(owned)
        if name.startswith("evaluate_") and name != "evaluate_common_step":
            return sha256_file(self.stage_path(name))
        if name == "evaluate_common_step":
            manifest = _json(self.root / "manifest.json")
            return canonical_json_sha256({"common_csv": sha256_file(self.stage_path(name)), "checkpoints": manifest.get("checkpoints")})
        if name == "build_report":
            base = self.root / "comparison"
            required = ("summary.json", "scenario_paired_differences.csv", "learning_curves.png", "holdout_comparison.png", "preliminary_comparison_ko.md")
            return canonical_json_sha256({item: sha256_file(base / item) for item in required})
        return sha256_file(self.stage_path(name))
    def input_hash(self, name: str, journal: Mapping[str, Mapping[str, Any]]) -> str:
        lock = _allocrl_dir()/self.config.dependency_lock_path
        observed_lock = sha256_file(lock) if lock.is_file() else "missing"
        previous = {stage: journal[stage]["output_sha256"] for stage in JOURNAL_STAGES[:JOURNAL_STAGES.index(name)]}
        return canonical_json_sha256({"stage":name,"config":self.config.config_sha256,"lock":observed_lock,"previous":previous})
    def run_stage(self, name: str, action: Callable[[], None]) -> None:
        # The daemon cannot throw on the worker thread; surface a refresh failure
        # before issuing another stage/subprocess.
        if getattr(self, "lease", None) is not None and self.lease.failure is not None: raise LeaseError(f"lease heartbeat failed: {self.lease.failure}")
        journal=self.journal(); entry=journal[name]; incoming=self.input_hash(name, journal); output=self.stage_path(name)
        try: current_output = self.output_hash(name)
        except (OSError, ValueError, KeyError, TypeError): current_output = None
        if entry["status"]=="complete" and entry["input_sha256"]==incoming and entry["output_sha256"]==current_output: return
        journal[name]=_journal_entry("in_progress", input_sha256=incoming, started_at_utc=_utc()); self.save_journal(journal)
        try:
            action(); output_hash=self.output_hash(name)
            if self._injected_output_hasher is None and not output.exists(): raise ExperimentStageError(f"stage produced no output: {name}")
        except KeyboardInterrupt:
            journal[name]=_journal_entry("interrupted", input_sha256=incoming, started_at_utc=journal[name]["started_at_utc"], completed_at_utc=_utc(), error="interrupted"); self.save_journal(journal); raise
        except BaseException as error:
            journal[name]=_journal_entry("failed", input_sha256=incoming, started_at_utc=journal[name]["started_at_utc"], completed_at_utc=_utc(), error=f"{type(error).__name__}: {error}"); self.save_journal(journal); raise ExperimentStageError(f"{name} failed: {error}") from error
        journal[name]=_journal_entry("complete", input_sha256=incoming, output_sha256=output_hash, started_at_utc=journal[name]["started_at_utc"], completed_at_utc=_utc()); self.save_journal(journal)
    def command(self, stage: str, argv: Sequence[str]) -> None:
        logs=self.root/"logs"; logs.mkdir(parents=True, exist_ok=True); log=logs/f"{stage}.log"
        with log.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(list(argv))+"\n")
            self.subprocess_runner(list(argv), check=True, cwd=str(_allocrl_dir()), stdout=stream, stderr=subprocess.STDOUT, text=True)
    def provenance(self) -> dict[str, str]:
        base=_allocrl_dir(); paths={"scenario":base/self.config.scenario_path,"split":base/self.config.split_manifest_path,"lock":base/self.config.dependency_lock_path}
        for key,path in paths.items():
            if not path.is_file(): raise ExperimentIntegrityError(f"required {key} input is absent: {path}")
        scenario,split,lock=(sha256_file(paths[key]) for key in ("scenario","split","lock"))
        if scenario!=self.config.fixed_scenarios_sha256 or split!=self.config.split_manifest_sha256: raise ExperimentIntegrityError("immutable input hash mismatch")
        try:
            scenarios = read_scenarios(paths["scenario"])
            if any(type(item["seed"]) is not int for item in scenarios):
                raise ValueError("fixed holdout seeds must be JSON integers")
            validate_fixed_holdout_scenarios(scenarios)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError, KeyError) as error:
            raise ExperimentIntegrityError("fixed scenario bundle is malformed or violates the exact holdout seed protocol") from error
        self.lock_sha=lock
        return {"baseline_sha256":self.config.baseline_commit,"config_sha256":self.config.config_sha256,"scenario_sha256":scenario,"split_sha256":split,"lock_sha256":lock}
    def preflight(self) -> None:
        provenance=self.provenance(); environment=collect_environment(self.runner_command, provenance)
        _validate_root_environment(environment, provenance, production_loaded=self.config.production_loaded)
        manifest={"schema_version":1, **provenance, "comparison_git_sha":environment["comparison_git_sha"], "comparison_git_dirty":environment["comparison_git_dirty"], "checkpoints":{}}
        _validate_root_manifest(manifest, provenance, environment)
        atomic_write_json(self.root/"environment.json",environment)
        atomic_write_json(self.root/"manifest.json",manifest)
    def smoke(self, arm: str) -> None:
        self.command(f"smoke_{arm}",build_smoke_command(arm,self.config,output_root=self.root,python_executable=self.python))
        marker=self.root/"smoke"/arm/"runner_verified.json"
        _, extractor = _arm(arm)
        archive = marker.parent / f"{extractor}.sb3"
        from train import model_num_timesteps
        timestep = (self.archive_reader or model_num_timesteps)(archive)
        if not archive.is_file() or timestep is None or timestep < self.config.smoke_timesteps:
            raise ExperimentStageError("smoke subprocess did not produce a readable requested-timestep archive")
        atomic_write_json(marker,{"arm":arm,"config_sha256":self.config.config_sha256,"path":archive.name,"sha256":sha256_file(archive),"timestep":timestep})
    def evaluate_arm(self, arm: str) -> None:
        # The paired evaluator is the sole authority for selected/final CSVs;
        # this ordered stage proves the arm's complete state before CNN starts.
        state=read_wall_clock_state(self.root/arm/"run_state.json")
        if not self._state_complete(self.root/arm,state): raise ExperimentIntegrityError("arm cannot be evaluated before a complete verified state")
        atomic_write_json(self.root/arm/"evaluation_stage.json",{"arm":arm,"checkpoint":state.last_checkpoint_file,"sha256":state.last_checkpoint_sha256,"timestep":state.last_checkpoint_timestep})
    def train(self, arm: str) -> None:
        root=self.root/arm; root.mkdir(parents=True,exist_ok=True); resume=None
        # A restarted process may legitimately skip the preflight journal entry;
        # recompute provenance here rather than retaining an invented/empty lock.
        self.lock_sha = self.provenance()["lock_sha256"]
        state_path=root/"run_state.json"
        if state_path.exists():
            state=read_wall_clock_state(state_path)
            checkpoint=resolve_state_checkpoint(root,state)
            if self._state_complete(root, state): return
            resume=checkpoint
        argv=build_train_command(arm,self.config,resume,output_root=self.root,python_executable=self.python,lock_sha256=self.lock_sha)
        self.command(f"train_{arm}",argv)
        state=read_wall_clock_state(state_path)
        if not self._state_complete(root,state): raise ExperimentStageError("training exited without a complete verified wall-clock state")
    def _state_complete(self, root: Path, state: Any) -> bool:
        if state.status!="complete" or state.config_sha256!=self.config.config_sha256 or state.target_training_seconds != self.config.target_training_seconds_per_arm or state.completed_training_seconds < self.config.target_training_seconds_per_arm: return False
        checkpoint=resolve_state_checkpoint(root,state)
        from train import model_num_timesteps
        timestep=(self.archive_reader or model_num_timesteps)(checkpoint)
        if timestep != state.last_checkpoint_timestep: raise ExperimentIntegrityError("state checkpoint stored timestep mismatch")
        return True
    def common_evaluation(self) -> None:
        scenarios=_json(_allocrl_dir()/self.config.scenario_path)
        records=scenarios.get("scenarios",scenarios) if isinstance(scenarios,dict) else scenarios
        if not isinstance(records,list): raise ExperimentIntegrityError("fixed scenarios must be a list")
        raw_config=_json(self.root/"raw_direct"/"run_config.json"); cnn_config=_json(self.root/"candidate_cnn"/"run_config.json")
        evaluate_comparison_artifacts(self.root,self.root/"raw_direct",self.root/"candidate_cnn",records,raw_config,cnn_config)
    def integrity(self) -> None:
        provenance=self.provenance(); manifest=_json(self.root/"manifest.json"); environment=_json(self.root/"environment.json")
        _validate_root_environment(environment, provenance, production_loaded=self.config.production_loaded)
        _validate_root_manifest(manifest, provenance, environment)
        for arm in ("raw_direct","candidate_cnn"):
            state=read_wall_clock_state(self.root/arm/"run_state.json")
            if not self._state_complete(self.root/arm,state): raise ExperimentIntegrityError("incomplete arm")
        _validate_environment_segments(self.root, environment, provenance, production_loaded=self.config.production_loaded)
        _validate_checkpoint_manifest(self.root, manifest, self.archive_reader)
        reports = _validate_report_artifacts(self.root)
        atomic_write_json(self.root/"integrity_verification.json", {
            "schema_version": 1, "manifest_sha256": sha256_file(self.root / "manifest.json"),
            "environment_sha256": sha256_file(self.root / "environment.json"),
            "report_artifact_sha256": reports, "verified_at_utc": _utc(),
        })

    def _verify_completion_journal(self) -> dict[str, dict[str, Any]]:
        raw = _json(self.journal_path)
        if set(raw) != set(JOURNAL_STAGES):
            raise ExperimentIntegrityError("completion requires an exact complete stage journal")
        journal = self.journal()
        for stage in JOURNAL_STAGES:
            entry = journal[stage]
            if entry["status"] != "complete" or entry["output_sha256"] != self.output_hash(stage):
                raise ExperimentIntegrityError("completion journal output hash is stale")
            if entry["input_sha256"] != self.input_hash(stage, journal):
                raise ExperimentIntegrityError("completion journal input chain is stale")
        return journal

    def _complete_marker(self) -> dict[str, Any]:
        journal = self._verify_completion_journal()
        stage_hashes = {stage: journal[stage]["output_sha256"] for stage in JOURNAL_STAGES}
        if self._injected_output_hasher is not None:
            provenance = {
                "config_sha256": self.config.config_sha256, "baseline_sha256": self.config.baseline_commit,
                "scenario_sha256": self.config.fixed_scenarios_sha256, "split_sha256": self.config.split_manifest_sha256,
                "lock_sha256": "0" * 64, "comparison_git_sha": "0" * 40,
            }
            report_hashes = {name: canonical_json_sha256({"test_artifact": name}) for name in _REPORT_ARTIFACTS}
            manifest_hash = canonical_json_sha256({"test": "manifest"})
            environment_hash = canonical_json_sha256({"test": "environment"})
        else:
            provenance = self.provenance()
            manifest = _json(self.root / "manifest.json"); environment = _json(self.root / "environment.json")
            _validate_root_environment(environment, provenance, production_loaded=self.config.production_loaded)
            _validate_root_manifest(manifest, provenance, environment)
            report_hashes = _validate_report_artifacts(self.root)
            manifest_hash, environment_hash = sha256_file(self.root / "manifest.json"), sha256_file(self.root / "environment.json")
            provenance = {**provenance, "comparison_git_sha": environment["comparison_git_sha"]}
        return {
            "schema_version": 1, "status": "complete", "stages": REQUIRED_COMPLETE_STAGES,
            "config_sha256": provenance["config_sha256"], "baseline_sha256": provenance["baseline_sha256"],
            "scenario_sha256": provenance["scenario_sha256"], "split_sha256": provenance["split_sha256"],
            "lock_sha256": provenance["lock_sha256"], "comparison_git_sha": provenance["comparison_git_sha"],
            "manifest_sha256": manifest_hash, "environment_sha256": environment_hash,
            "stage_output_sha256": stage_hashes, "report_artifact_sha256": report_hashes,
            "completed_at_utc": _utc(),
        }


def run_overnight_experiment(config_path: str | Path | ExperimentConfig, output_root: str | Path, *, subprocess_runner: Callable[..., Any] = subprocess.run, clock: Callable[[], float] = time.monotonic, python_executable: str | None = None, archive_timestep_reader: Callable[[Path], int | None] | None = None, runner_command: Sequence[str] | None = None, stale_takeover: bool = False, lease_interval_seconds: float = 60, lease_stale_seconds: float = 900, stage_actions: Mapping[str, Callable[[], None]] | None = None, stage_output_hashers: Mapping[str, Callable[[], str]] | None = None) -> None:
    if not _valid_number(lease_interval_seconds) or float(lease_interval_seconds) <= 0 or not _valid_number(lease_stale_seconds) or float(lease_stale_seconds) <= 0: raise ValueError("lease intervals must be positive finite numbers")
    if (stage_actions is None) != (stage_output_hashers is None): raise ValueError("stage actions and output hashers must be supplied together")
    if stage_actions is not None and (set(stage_actions) != set(JOURNAL_STAGES) or set(stage_output_hashers or ()) != set(JOURNAL_STAGES) or not all(callable(value) for value in stage_actions.values()) or not all(callable(value) for value in (stage_output_hashers or {}).values())): raise ValueError("test stage mappings must have exact callable journal-stage keys")
    config=config_path if isinstance(config_path,ExperimentConfig) else load_experiment_config(config_path)
    root=Path(output_root).resolve(); root.mkdir(parents=True,exist_ok=True); runner=_Runner(config,root,subprocess_runner=subprocess_runner,clock=clock,python_executable=python_executable,archive_timestep_reader=archive_timestep_reader,output_hasher=(lambda name: stage_output_hashers[name]()) if stage_output_hashers else None,runner_command=runner_command)
    lease = _Lease(root,stale_takeover=stale_takeover,clock=clock,interval=lease_interval_seconds,stale_after=lease_stale_seconds)
    runner.lease = lease
    complete_path = root / "COMPLETE.json"
    try:
        with lease:
            complete_path.unlink(missing_ok=True)
            actions = stage_actions or {"preflight":runner.preflight,"smoke_raw_direct":lambda: runner.smoke("raw_direct"),"smoke_candidate_cnn":lambda: runner.smoke("candidate_cnn"),"train_raw_direct":lambda: runner.train("raw_direct"),"evaluate_raw_direct":lambda: runner.evaluate_arm("raw_direct"),"train_candidate_cnn":lambda: runner.train("candidate_cnn"),"evaluate_candidate_cnn":lambda: runner.evaluate_arm("candidate_cnn"),"evaluate_common_step":runner.common_evaluation,"build_report":lambda: write_complete_report(root),"integrity_verification":runner.integrity}
            for stage in JOURNAL_STAGES: runner.run_stage(stage, actions[stage])
        # A refresh error raised by __exit__ prevents publication.  The marker
        # is deliberately outside the lease context so it never claims a run
        # complete while its ownership heartbeat is uncertain.
        atomic_write_json(complete_path, runner._complete_marker())
    except BaseException as error:
        if lease.acquired:
            try: complete_path.unlink(missing_ok=True)
            except OSError: pass
            try: write_partial_report(root,f"{type(error).__name__}: {error}")
            except BaseException: pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description="Run the verified overnight raw/CNN comparison")
    parser.add_argument("--config",required=True); parser.add_argument("--output-root",required=True); parser.add_argument("--take-over-stale-lease",action="store_true")
    args=parser.parse_args(argv); run_overnight_experiment(args.config,args.output_root,stale_takeover=args.take_over_stale_lease); return 0


if __name__ == "__main__": raise SystemExit(main())
