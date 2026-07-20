"""Fail-closed, resumable orchestration for the overnight comparison.

This module intentionally owns only root-level orchestration artifacts.  Model
training, checkpoint persistence, evaluation and reporting remain in their
specialist modules so a restart can verify rather than infer their state.
"""

from __future__ import annotations

import argparse
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
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from comparison.artifact_manifest import (
    REQUIRED_ENVIRONMENT_KEYS,
    canonical_json_sha256,
    collect_environment,
    sha256_file,
)
from comparison.checkpoint_evaluator import evaluate_comparison_artifacts
from comparison.report_builder import JOURNAL_STAGES, JOURNAL_STATUSES, write_complete_report, write_partial_report
from comparison.wall_clock_callback import atomic_write_json, read_wall_clock_state, resolve_state_checkpoint


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
    def __init__(self, config: ExperimentConfig, root: Path, *, subprocess_runner: Callable[..., Any], clock: Callable[[], float], python_executable: str | None, archive_timestep_reader: Callable[[Path], int | None] | None, output_hasher: Callable[[str], str] | None = None) -> None:
        self.config=config; self.root=root.resolve(); self.subprocess_runner=subprocess_runner; self.clock=clock; self.python=python_executable or sys.executable; self.archive_reader=archive_timestep_reader; self.journal_path=self.root/"stage_journal.json"; self.lock_sha=""; self._injected_output_hasher=output_hasher
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
            stable_manifest = {key: manifest.get(key) for key in ("schema_version", "baseline_sha256", "config_sha256", "scenario_sha256", "split_sha256", "lock_sha256")}
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
        self.lock_sha=lock
        return {"baseline_sha256":self.config.baseline_commit,"config_sha256":self.config.config_sha256,"scenario_sha256":scenario,"split_sha256":split,"lock_sha256":lock}
    def preflight(self) -> None:
        provenance=self.provenance(); environment=collect_environment([], provenance)
        if set(environment)!=set(REQUIRED_ENVIRONMENT_KEYS): raise ExperimentIntegrityError("root environment schema differs")
        atomic_write_json(self.root/"environment.json",environment)
        atomic_write_json(self.root/"manifest.json",{"schema_version":1, **provenance, "checkpoints":{}})
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
        if any(manifest.get(key)!=value for key,value in provenance.items()) or set(environment)!=set(REQUIRED_ENVIRONMENT_KEYS): raise ExperimentIntegrityError("root provenance mismatch")
        for arm in ("raw_direct","candidate_cnn"):
            state=read_wall_clock_state(self.root/arm/"run_state.json")
            if not self._state_complete(self.root/arm,state): raise ExperimentIntegrityError("incomplete arm")
        segments=[]
        for arm in ("raw_direct","candidate_cnn"):
            p=self.root/arm/"environment_segments.jsonl"
            if not p.is_file(): raise ExperimentIntegrityError("missing environment segments")
            segments += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]
        keys=("vm_boot_id","gpu_uuid","torch_version","cuda_version","cudnn_version","lock_sha256")
        if not segments or any(set(item)!=set(REQUIRED_ENVIRONMENT_KEYS) for item in segments) or any(tuple(item.get(key) for key in keys)!=tuple(segments[0].get(key) for key in keys) for item in segments): raise ExperimentIntegrityError("arms must use the same Colab VM/GPU/library environment")
        if any(environment.get(key) != segments[0].get(key) for key in keys):
            raise ExperimentIntegrityError("root environment differs from arm environment segments")
        if not isinstance(environment.get("comparison_git_sha"), str) or environment.get("comparison_git_dirty") is not False:
            raise ExperimentIntegrityError("comparison checkout must identify a clean commit")
        checkpoints=manifest.get("checkpoints")
        if not isinstance(checkpoints,dict) or set(checkpoints)!={"raw_direct","candidate_cnn"}: raise ExperimentIntegrityError("missing paired checkpoint manifest")
        for arm,refs in checkpoints.items():
            if set(refs)!={"selected","final","common"}: raise ExperimentIntegrityError("checkpoint refs incomplete")
            for ref in refs.values():
                path=(self.root/ref["path"]).resolve()
                if self.root not in path.parents or not path.is_file() or sha256_file(path)!=ref["sha256"]: raise ExperimentIntegrityError("tampered checkpoint reference")
        for path in (self.root/"comparison"/"summary.json", self.root/"comparison"/"scenario_paired_differences.csv", self.root/"comparison"/"learning_curves.png", self.root/"comparison"/"holdout_comparison.png", self.root/"comparison"/"preliminary_comparison_ko.md"):
            if not path.is_file() or path.stat().st_size == 0: raise ExperimentIntegrityError(f"missing report artifact: {path.name}")
        atomic_write_json(self.root/"integrity_verification.json", {"manifest_sha256": sha256_file(self.root / "manifest.json"), "verified_at_utc": _utc()})


def run_overnight_experiment(config_path: str | Path | ExperimentConfig, output_root: str | Path, *, subprocess_runner: Callable[..., Any] = subprocess.run, clock: Callable[[], float] = time.monotonic, python_executable: str | None = None, archive_timestep_reader: Callable[[Path], int | None] | None = None, stale_takeover: bool = False, lease_interval_seconds: float = 60, lease_stale_seconds: float = 900, stage_actions: Mapping[str, Callable[[], None]] | None = None, stage_output_hashers: Mapping[str, Callable[[], str]] | None = None) -> None:
    if not _valid_number(lease_interval_seconds) or float(lease_interval_seconds) <= 0 or not _valid_number(lease_stale_seconds) or float(lease_stale_seconds) <= 0: raise ValueError("lease intervals must be positive finite numbers")
    if (stage_actions is None) != (stage_output_hashers is None): raise ValueError("stage actions and output hashers must be supplied together")
    if stage_actions is not None and (set(stage_actions) != set(JOURNAL_STAGES) or set(stage_output_hashers or ()) != set(JOURNAL_STAGES) or not all(callable(value) for value in stage_actions.values()) or not all(callable(value) for value in (stage_output_hashers or {}).values())): raise ValueError("test stage mappings must have exact callable journal-stage keys")
    config=config_path if isinstance(config_path,ExperimentConfig) else load_experiment_config(config_path)
    root=Path(output_root).resolve(); root.mkdir(parents=True,exist_ok=True); runner=_Runner(config,root,subprocess_runner=subprocess_runner,clock=clock,python_executable=python_executable,archive_timestep_reader=archive_timestep_reader,output_hasher=(lambda name: stage_output_hashers[name]()) if stage_output_hashers else None)
    lease = _Lease(root,stale_takeover=stale_takeover,clock=clock,interval=lease_interval_seconds,stale_after=lease_stale_seconds)
    runner.lease = lease
    try:
        with lease:
            actions = stage_actions or {"preflight":runner.preflight,"smoke_raw_direct":lambda: runner.smoke("raw_direct"),"smoke_candidate_cnn":lambda: runner.smoke("candidate_cnn"),"train_raw_direct":lambda: runner.train("raw_direct"),"evaluate_raw_direct":lambda: runner.evaluate_arm("raw_direct"),"train_candidate_cnn":lambda: runner.train("candidate_cnn"),"evaluate_candidate_cnn":lambda: runner.evaluate_arm("candidate_cnn"),"evaluate_common_step":runner.common_evaluation,"build_report":lambda: write_complete_report(root),"integrity_verification":runner.integrity}
            for stage in JOURNAL_STAGES: runner.run_stage(stage, actions[stage])
        # A refresh error raised by __exit__ prevents publication.  The marker
        # is deliberately outside the lease context so it never claims a run
        # complete while its ownership heartbeat is uncertain.
        atomic_write_json(root/"COMPLETE.json",{"status":"complete","stages":REQUIRED_COMPLETE_STAGES})
    except BaseException as error:
        if lease.acquired:
            try: write_partial_report(root,f"{type(error).__name__}: {error}")
            except BaseException: pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description="Run the verified overnight raw/CNN comparison")
    parser.add_argument("--config",required=True); parser.add_argument("--output-root",required=True); parser.add_argument("--take-over-stale-lease",action="store_true")
    args=parser.parse_args(argv); run_overnight_experiment(args.config,args.output_root,stale_takeover=args.take_over_stale_lease); return 0


if __name__ == "__main__": raise SystemExit(main())
