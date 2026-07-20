"""Build deterministic, single-seed preliminary comparison report artifacts."""

from __future__ import annotations

import csv
import html
import json
import math
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from comparison.artifact_manifest import REQUIRED_ENVIRONMENT_KEYS
from comparison.checkpoint_evaluator import ARMS, EVALUATION_COLUMNS, PRIMARY_TEST_SEEDS, SELECTION_SEEDS


HEADLINE_COLUMNS = (
    "mean_terminal_score", "mean_dropout_rate", "mean_delay_days", "mean_delayed_count",
)
PAIR_COLUMNS = (
    "seed", "terminal_score_delta_cnn_minus_raw", "dropout_rate_delta_cnn_minus_raw",
    "mean_delay_days_delta_cnn_minus_raw", "delayed_count_delta_cnn_minus_raw",
)
TRAINING_LOG_COLUMNS = ("episode", "timestep", "resolved_reward", "terminal_residual", "terminal_score", "episode_reward", "delayed_count", "dropout_count", "total_delay_days", "success_rate")
LOSS_LOG_COLUMNS = ("timestep", "policy_gradient_loss", "value_loss", "entropy_loss", "approx_kl", "clip_fraction", "loss", "explained_variance", "cnn_gradient_norm", "cnn_weight_change", "workspace_feature_variance", "candidate_channel_sensitivity")
PROGRESS_TIMING_COLUMNS = ("generation", "timestep", "recorded_training_seconds", "updated_at_utc", "status", "checkpoint_file")
JOURNAL_STAGES = ("preflight", "smoke_raw_direct", "smoke_candidate_cnn", "train_raw_direct", "evaluate_raw_direct", "train_candidate_cnn", "evaluate_candidate_cnn", "evaluate_common_step", "build_report", "integrity_verification")
JOURNAL_STATUSES = frozenset({"pending", "in_progress", "interrupted", "failed", "complete"})
MISSING = "자료 없음"
RUNTIME_FIELDS = (
    "target_training_seconds", "recorded_training_seconds", "end_to_end_training_seconds",
    "overrun_seconds", "restart_count", "max_unrecorded_seconds", "start_timestep",
    "end_timestep", "steps_per_second", "parameter_counts", "peak_cuda_memory_bytes",
    "evaluation_seconds", "selected_checkpoint_timestep", "selection_count",
    "selection_tuple", "checkpoint_identity",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise ValueError(f"missing canonical artifact: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite JSON number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _csv_number(value: Any, field: str) -> float:
    if not isinstance(value, str): raise ValueError(f"{field} must be a CSV number")
    try: number = float(value)
    except ValueError as error: raise ValueError(f"{field} must be a finite CSV number") from error
    if not math.isfinite(number): raise ValueError(f"{field} must be a finite CSV number")
    return number


def _integer(value: Any, field: str, *, nonnegative: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if nonnegative and value < 0:
        raise ValueError(f"{field} must be nonnegative")
    return value


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be SHA-256")
    return value


def _read_evaluation(path: Path, arm: str, seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != EVALUATION_COLUMNS:
                raise ValueError(f"evaluation CSV has incompatible header: {path}")
            rows = list(reader)
    except OSError as error:
        raise ValueError(f"missing evaluation CSV: {path}") from error
    if len(rows) != len(seeds):
        raise ValueError(f"evaluation CSV must contain exact {len(seeds)} rows: {path}")
    expected_partition = "selection" if seeds == SELECTION_SEEDS else "primary_test"
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if set(row) != set(EVALUATION_COLUMNS) or row["arm"] != arm:
            raise ValueError(f"evaluation CSV has invalid arm/schema: {path}")
        try:
            seed = int(row["seed"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"evaluation seed must be an integer: {path}") from error
        row_partition = "selection" if seed in SELECTION_SEEDS else "primary_test"
        if row["evaluation_partition"] != row_partition:
            raise ValueError(f"evaluation partition is invalid: {path}")
        if row["source"] != "holdout_fixed20" or row["policy"] != arm or not row["checkpoint"]:
            raise ValueError(f"evaluation CSV has invalid source/policy/checkpoint: {path}")
        if row["checkpoint"] not in {"best_model", "fallback_final", "common_step"}:
            raise ValueError(f"evaluation CSV has invalid checkpoint label: {path}")
        converted = dict(row); converted["seed"] = seed
        for column in ("mean_reward", *HEADLINE_COLUMNS, "mean_retained_choice_ratio"):
            converted[column] = _csv_number(row[column], column)
        converted["checkpoint_timestep"] = _integer(int(row["checkpoint_timestep"]) if row["checkpoint_timestep"].isdigit() else None, "checkpoint_timestep")
        converted["checkpoint_sha256"] = _sha(row["checkpoint_sha256"], "checkpoint_sha256")
        normalized.append(converted)
    if tuple(sorted(row["seed"] for row in normalized)) != seeds:
        raise ValueError(f"evaluation CSV must contain exact unique {expected_partition} seeds: {path}")
    provenance = {(row["checkpoint"], row["checkpoint_timestep"], row["checkpoint_sha256"]) for row in normalized}
    if len(provenance) != 1:
        raise ValueError(f"evaluation CSV must have one provenance tuple: {path}")
    return sorted(normalized, key=lambda row: row["seed"])


def _load_arm(root: Path, arm: str) -> dict[str, Any]:
    arm_root = root / arm
    all_rows = _read_evaluation(arm_root / "evaluation_scenarios.csv", arm, tuple(range(1000, 1020)))
    selection = [row for row in all_rows if row["seed"] in SELECTION_SEEDS]
    primary = _read_evaluation(arm_root / "evaluation_primary_test.csv", arm, PRIMARY_TEST_SEEDS)
    all_primary = [row for row in all_rows if row["seed"] in PRIMARY_TEST_SEEDS]
    if any(row != corresponding for row, corresponding in zip(primary, all_primary)):
        raise ValueError(f"primary CSV disagrees with all-scenarios CSV: {arm}")
    return {"selection_rows": selection, "primary_rows": primary, "all_rows": all_rows, "runtime": _runtime(arm_root / "runtime_metrics.json")}


def _runtime(path: Path) -> dict[str, Any]:
    data = _read_json(path)
    if set(data) != set(RUNTIME_FIELDS):
        raise ValueError("runtime_metrics must contain exactly required fields")
    for key in ("target_training_seconds", "recorded_training_seconds", "end_to_end_training_seconds", "overrun_seconds", "max_unrecorded_seconds", "steps_per_second", "evaluation_seconds"):
        _number(data[key], key)
        if data[key] < 0: raise ValueError(f"{key} must be nonnegative")
    for key in ("restart_count", "start_timestep", "end_timestep", "selected_checkpoint_timestep"):
        _integer(data[key], key)
    peak = data["peak_cuda_memory_bytes"]
    if peak is not None: _integer(peak, "peak_cuda_memory_bytes")
    counts = data["parameter_counts"]
    if not isinstance(counts, Mapping) or set(counts) != {"total", "feature_extractor", "policy", "value"}:
        raise ValueError("parameter_counts must contain exactly total, feature_extractor, policy, value")
    parsed = {key: _integer(value, f"parameter_counts.{key}") for key, value in counts.items()}
    if parsed["total"] != parsed["feature_extractor"] + parsed["policy"] + parsed["value"]:
        raise ValueError("disjoint parameter counts do not reconcile")
    identity = data["checkpoint_identity"]
    if not isinstance(identity, Mapping) or set(identity) != {"filename", "sha256"} or not isinstance(identity["filename"], str) or not identity["filename"]:
        raise ValueError("checkpoint_identity is invalid")
    _sha(identity["sha256"], "checkpoint_identity.sha256")
    count = _integer(data["selection_count"], "selection_count")
    tuple_value = data["selection_tuple"]
    if count == 0:
        if tuple_value is not None: raise ValueError("fallback selection_tuple must be null")
    else:
        if not isinstance(tuple_value, list) or len(tuple_value) != 3: raise ValueError("selection_tuple must contain three numbers")
        for value in tuple_value: _number(value, "selection_tuple")
    data = dict(data); data["parameter_counts"] = parsed
    return data


def _environment(root: Path) -> dict[str, Any]:
    env = _read_json(root / "environment.json")
    if set(env) != set(REQUIRED_ENVIRONMENT_KEYS): raise ValueError("environment has incomplete schema")
    for key in ("baseline_sha256",):
        if not isinstance(env[key], str) or _SHA1.fullmatch(env[key]) is None: raise ValueError(f"environment {key} invalid")
    for key in ("config_sha256", "scenario_sha256", "split_sha256", "lock_sha256"): _sha(env[key], f"environment {key}")
    if not isinstance(env["comparison_git_sha"], str) or _SHA1.fullmatch(env["comparison_git_sha"]) is None: raise ValueError("environment comparison_git_sha invalid")
    if not isinstance(env["comparison_git_dirty"], bool) or not isinstance(env["command"], list) or not isinstance(env["pip_freeze"], list): raise ValueError("environment type invalid")
    for key in ("captured_at_utc", "python_version", "platform", "vm_boot_id", "torch_version", "resolved_device"):
        if not isinstance(env[key], str) or not env[key]: raise ValueError(f"environment {key} invalid")
    cpu = env["resolved_device"] == "cpu"
    for key in ("gpu_name", "gpu_uuid", "gpu_total_memory_bytes"):
        if cpu:
            if env[key] is not None: raise ValueError("CPU environment GPU fields must be null")
        elif env[key] is None: raise ValueError("GPU environment fields are required")
    for key in ("cpu_count", "process_id"):
        _integer(env[key], f"environment {key}")
    if env["gpu_total_memory_bytes"] is not None: _integer(env["gpu_total_memory_bytes"], "environment gpu_total_memory_bytes")
    return env


def _manifest(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "manifest.json")
    if set(manifest) != {"schema_version", "checkpoints"} or manifest["schema_version"] != 1 or not isinstance(manifest["checkpoints"], Mapping) or set(manifest["checkpoints"]) != set(ARMS):
        raise ValueError("manifest has incomplete schema")
    labels = {"selected": {"best_model", "fallback_final"}, "final": {"final"}, "common": {"common_step"}}
    for arm in ARMS:
        refs = manifest["checkpoints"][arm]
        if not isinstance(refs, Mapping) or set(refs) != set(labels): raise ValueError("manifest checkpoint refs incomplete")
        for kind, allowed in labels.items():
            ref = refs[kind]
            if not isinstance(ref, Mapping) or set(ref) != {"path", "label", "sha256", "timestep"} or not isinstance(ref["path"], str) or not ref["path"].startswith(f"{arm}/") or ".." in Path(ref["path"]).parts or ref["label"] not in allowed:
                raise ValueError("manifest checkpoint ref invalid")
            _sha(ref["sha256"], "manifest checkpoint sha256"); _integer(ref["timestep"], "manifest checkpoint timestep")
    return manifest


def _means(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {f"mean_{column.removeprefix('mean_')}": sum(row[column] for row in rows) / len(rows) for column in HEADLINE_COLUMNS}


def build_paired_differences(root: str | Path) -> list[dict]:
    """Return primary-test scenario deltas, always candidate CNN minus raw direct."""
    base = Path(root)
    raw = _load_arm(base, "raw_direct")["primary_rows"]
    cnn = _load_arm(base, "candidate_cnn")["primary_rows"]
    by_raw = {row["seed"]: row for row in raw}; by_cnn = {row["seed"]: row for row in cnn}
    if set(by_raw) != set(PRIMARY_TEST_SEEDS) or set(by_cnn) != set(PRIMARY_TEST_SEEDS):
        raise ValueError("primary-test pairing requires exact matching seeds")
    return [{
        "seed": seed,
        "terminal_score_delta_cnn_minus_raw": by_cnn[seed]["mean_terminal_score"] - by_raw[seed]["mean_terminal_score"],
        "dropout_rate_delta_cnn_minus_raw": by_cnn[seed]["mean_dropout_rate"] - by_raw[seed]["mean_dropout_rate"],
        "mean_delay_days_delta_cnn_minus_raw": by_cnn[seed]["mean_delay_days"] - by_raw[seed]["mean_delay_days"],
        "delayed_count_delta_cnn_minus_raw": by_cnn[seed]["mean_delayed_count"] - by_raw[seed]["mean_delayed_count"],
    } for seed in PRIMARY_TEST_SEEDS]


def _common_summary(root: Path) -> dict[str, Any]:
    path = root / "comparison" / "common_step_evaluation.csv"
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != EVALUATION_COLUMNS:
                raise ValueError("common-step CSV has incompatible header")
            rows = list(reader)
    except OSError as error:
        raise ValueError("missing common-step evaluation CSV") from error
    if len(rows) != 40:
        raise ValueError("common-step CSV must have exactly 40 rows")
    grouped = {arm: [] for arm in ARMS}
    for row in rows:
        arm = row.get("arm")
        if set(row) != set(EVALUATION_COLUMNS) or arm not in grouped or row.get("checkpoint") != "common_step":
            raise ValueError("common-step CSV has invalid arm/checkpoint")
        if not row["seed"].isdigit() or not row["checkpoint_timestep"].isdigit() or row["source"] != "holdout_fixed20" or row["policy"] != arm:
            raise ValueError("common-step CSV has invalid scalar fields")
        normalized = dict(row); normalized["seed"] = int(row["seed"]); normalized["checkpoint_timestep"] = int(row["checkpoint_timestep"]); normalized["checkpoint_sha256"] = _sha(row["checkpoint_sha256"], "common checkpoint_sha256")
        if normalized["evaluation_partition"] != ("selection" if normalized["seed"] in SELECTION_SEEDS else "primary_test"):
            raise ValueError("common-step CSV has invalid partition")
        for column in HEADLINE_COLUMNS:
            normalized[column] = _csv_number(row[column], column)
        grouped[arm].append(normalized)
    if any(tuple(sorted(row["seed"] for row in values)) != tuple(range(1000, 1020)) for values in grouped.values()):
        raise ValueError("common-step CSV must have exact unique seeds for both arms")
    provenance = {arm: {(row["checkpoint"], row["checkpoint_timestep"], row["checkpoint_sha256"]) for row in values} for arm, values in grouped.items()}
    if any(len(value) != 1 for value in provenance.values()): raise ValueError("common-step CSV has inconsistent provenance")
    timesteps = {row["checkpoint_timestep"] for values in grouped.values() for row in values}
    if len(timesteps) != 1:
        raise ValueError("common-step rows must share one timestep")
    return {"timestep": int(next(iter(timesteps))), "provenance": {arm: next(iter(value)) for arm, value in provenance.items()}, **{arm: {"primary_test": _means([row for row in rows if row["seed"] in PRIMARY_TEST_SEEDS]), "all_holdout": _means(rows)} for arm, rows in grouped.items()}}


def _reconcile(arms: Mapping[str, dict[str, Any]], manifest: Mapping[str, Any], common: Mapping[str, Any]) -> None:
    for arm, data in arms.items():
        selected = manifest["checkpoints"][arm]["selected"]
        runtime = data["runtime"]
        evidence = data["all_rows"][0]
        expected = (selected["label"], selected["timestep"], selected["sha256"])
        actual = (evidence["checkpoint"], evidence["checkpoint_timestep"], evidence["checkpoint_sha256"])
        if actual != expected or runtime["selected_checkpoint_timestep"] != selected["timestep"] or runtime["checkpoint_identity"]["sha256"] != selected["sha256"] or runtime["checkpoint_identity"]["filename"] != Path(selected["path"]).name:
            raise ValueError("selected checkpoint reconciliation failed")
        if (runtime["selection_count"] > 0) != (selected["label"] == "best_model"):
            raise ValueError("selection_count semantics do not match selected checkpoint")
        common_ref = manifest["checkpoints"][arm]["common"]
        label, timestep, digest = common["provenance"][arm]
        if (label, timestep, digest) != (common_ref["label"], common_ref["timestep"], common_ref["sha256"]):
            raise ValueError("common checkpoint reconciliation failed")


def build_comparison_summary(root: str | Path) -> dict[str, Any]:
    """Read only canonical artifacts and return a JSON-safe, finite comparison summary."""
    base = Path(root)
    arms = {arm: _load_arm(base, arm) for arm in ARMS}
    environment = _environment(base); manifest = _manifest(base); common = _common_summary(base)
    _reconcile(arms, manifest, common)
    result: dict[str, Any] = {
        "schema_version": 1,
        "primary_test_definition": {"seeds": list(PRIMARY_TEST_SEEDS), "training_runs": 1, "seed": 0},
        "selection_definition": {"seeds": list(SELECTION_SEEDS), "role": "checkpoint_selection_only"},
        "environment": environment,
        "manifest": manifest,
        "common_step": common,
    }
    for arm, data in arms.items():
        result[arm] = {"selection": _means(data["selection_rows"]), "primary_test": _means(data["primary_rows"]), "all_holdout": _means(data["all_rows"]), "runtime_metrics": data["runtime"]}
    return result


def _canonical_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _write_pairs(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        writer = csv.DictWriter(stream, fieldnames=PAIR_COLUMNS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _cell(value: Any) -> str:
    return MISSING if value is None else str(value)


def _curve_rows(path: Path, columns: tuple[str, ...], name: str) -> list[dict[str, str]] | None:
    if not path.is_file(): return None
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != columns: raise ValueError(f"{name} has incompatible header")
        rows = list(reader)
    prior = -1
    for row in rows:
        if set(row) != set(columns): raise ValueError(f"{name} has malformed row")
        if not row["timestep"].isdigit() or int(row["timestep"]) < prior: raise ValueError(f"{name} timestep is invalid")
        prior = int(row["timestep"])
        for key, value in row.items():
            if key in {"updated_at_utc", "status", "checkpoint_file"} or value == "": continue
            _csv_number(value, f"{name}.{key}")
        if name == "progress_timing" and (not row["generation"].isdigit() or row["status"] not in {"running", "complete"}): raise ValueError("progress_timing row is invalid")
    return rows


def _stage_journal(data: Mapping[str, Any]) -> str:
    """Validate Task 6's direct stage-name-to-entry journal object."""
    groups = {status: [] for status in JOURNAL_STATUSES}
    keys = {"status", "input_sha256", "output_sha256", "started_at_utc", "completed_at_utc", "error"}
    for name, entry in data.items():
        if name not in JOURNAL_STAGES or not isinstance(entry, Mapping) or set(entry) != keys or entry["status"] not in JOURNAL_STATUSES:
            raise ValueError("invalid stage journal")
        for sha_key in ("input_sha256", "output_sha256"):
            value = entry[sha_key]
            if value is not None: _sha(value, f"stage {sha_key}")
        for timestamp in ("started_at_utc", "completed_at_utc"):
            value = entry[timestamp]
            if value is not None:
                if not isinstance(value, str) or datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is None: raise ValueError("invalid stage timestamp")
        if entry["error"] is not None and (not isinstance(entry["error"], str) or "\ufffd" in entry["error"]): raise ValueError("invalid stage error")
        if entry["status"] == "complete" and (entry["output_sha256"] is None or entry["completed_at_utc"] is None): raise ValueError("complete stage lacks outputs")
        if entry["status"] in {"failed", "interrupted"} and entry["completed_at_utc"] is None: raise ValueError("terminal stage lacks completion")
        groups[entry["status"]].append(name)
    return "; ".join(f"{label}: {', '.join(sorted(groups[status])) or MISSING}" for status, label in (("complete", "completed"), ("failed", "failed"), ("interrupted", "interrupted"), ("in_progress", "in-progress"), ("pending", "missing")))


def _learning_plot(root: Path, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    try:
        for arm, label in (("raw_direct", "raw-direct"), ("candidate_cnn", "candidate-CNN")):
            training = _curve_rows(root / arm / "training_log.csv", TRAINING_LOG_COLUMNS, "training_log")
            progress = _curve_rows(root / arm / "progress_timing.csv", PROGRESS_TIMING_COLUMNS, "progress_timing")
            loss = _curve_rows(root / arm / "loss_log.csv", LOSS_LOG_COLUMNS, "loss_log")
            if training: axes[0].plot([float(r["timestep"]) for r in training], [float(r["terminal_score"]) for r in training], label=label)
            if progress: axes[1].plot([float(r["recorded_training_seconds"]) for r in progress], [float(r["timestep"]) for r in progress], label=label)
            if loss: axes[2].plot([float(r["timestep"]) for r in loss if r["loss"]], [float(r["loss"]) for r in loss if r["loss"]], label=label)
        for axis, title, x, y in zip(axes, ("Episode terminal score", "Checkpoint progress", "Training loss"), ("timestep", "recorded subprocess seconds", "timestep"), ("score", "timestep", "loss")):
            axis.set(title=title, xlabel=x, ylabel=y)
            if axis.lines: axis.legend()
            else: axis.text(.5, .5, MISSING, ha="center", va="center", transform=axis.transAxes)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Glyph .* missing from font")
            figure.tight_layout(); figure.savefig(output, dpi=150)
    finally:
        plt.close(figure)


def _holdout_plot(summary: Mapping[str, Any], pairs: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    metric = "mean_terminal_score"
    axes[0].bar(["raw-direct", "candidate-CNN"], [summary[arm]["primary_test"][metric] for arm in ARMS])
    axes[0].set(title="Primary test terminal score (15 paired scenarios)", ylabel="score")
    axes[1].bar([str(row["seed"]) for row in pairs], [row["terminal_score_delta_cnn_minus_raw"] for row in pairs])
    axes[1].axhline(0, color="black", linewidth=.8); axes[1].set(title="CNN minus raw paired terminal-score difference", xlabel="scenario seed", ylabel="delta")
    try: figure.tight_layout(); figure.savefig(output, dpi=150)
    finally: plt.close(figure)


def _report_text(summary: Mapping[str, Any], pairs: list[dict]) -> str:
    env = summary["environment"]
    raw, cnn = summary["raw_direct"], summary["candidate_cnn"]
    def budget(arm: Mapping[str, Any]) -> str:
        runtime = arm["runtime_metrics"]; count = runtime.get("selection_count")
        if int(count) > 0:
            return f"label=best_model, timestep {_cell(runtime.get('selected_checkpoint_timestep'))}, SHA={_cell(runtime.get('checkpoint_identity', {}).get('sha256'))}, selection count {count}, tuple {_cell(runtime.get('selection_tuple'))}"
        return f"label=fallback_final, timestep {_cell(runtime.get('selected_checkpoint_timestep'))}, SHA={_cell(runtime.get('checkpoint_identity', {}).get('sha256'))}, selection count 0, fallback 사유: 자료 없음"
    pair_mean = {key: sum(row[key] for row in pairs) / len(pairs) for key in PAIR_COLUMNS[1:]}
    return f"""# Raw-direct vs Candidate-CNN 예비 결과

## 목적과 모델

이 문서는 seed 0 단일 학습 실행의 예비 결과이다. raw-direct는 grid를 읽지 않고 2772개 정규화 scalar를 직접 입력하며, candidate-CNN은 10개 작업장의 4채널 64×64 grid와 기존 구조화 pipeline을 사용한다. 두 arm은 PPO actor/critic MLP `pi=[64,64]`, `vf=[64,64]`, ReLU와 shared extractor를 사용한다.

## 통제와 실행 환경

913-block episode, deterministic ship-disjoint split, 고정 10-workspace 순서, action mask, no-rotation, reward/normalization, full context, PPO 설정(learning_rate=3e-4, n_steps=960, batch_size=64, n_epochs=10, gamma=1.0, gae_lambda=0.98), seed 0, n_envs=1, selection seed 1000..1004와 primary seed 1005..1019 분리를 고정했다. 실제 환경은 device={_cell(env.get('resolved_device'))}, GPU={_cell(env.get('gpu_name'))}, GPU UUID={_cell(env.get('gpu_uuid'))}, Torch={_cell(env.get('torch_version'))}, CUDA={_cell(env.get('cuda_version'))}이다.

## 3시간 budget 및 주 비교

raw-direct: target={_cell(raw['runtime_metrics']['target_training_seconds'])}, recorded={_cell(raw['runtime_metrics']['recorded_training_seconds'])}, end-to-end={_cell(raw['runtime_metrics']['end_to_end_training_seconds'])}, overrun={_cell(raw['runtime_metrics']['overrun_seconds'])}, restarts={_cell(raw['runtime_metrics']['restart_count'])}, max-unrecorded={_cell(raw['runtime_metrics']['max_unrecorded_seconds'])}, timestep={_cell(raw['runtime_metrics']['start_timestep'])}->{_cell(raw['runtime_metrics']['end_timestep'])}, evaluation seconds={_cell(raw['runtime_metrics']['evaluation_seconds'])}; {budget(raw)}. candidate-CNN: target={_cell(cnn['runtime_metrics']['target_training_seconds'])}, recorded={_cell(cnn['runtime_metrics']['recorded_training_seconds'])}, end-to-end={_cell(cnn['runtime_metrics']['end_to_end_training_seconds'])}, overrun={_cell(cnn['runtime_metrics']['overrun_seconds'])}, restarts={_cell(cnn['runtime_metrics']['restart_count'])}, max-unrecorded={_cell(cnn['runtime_metrics']['max_unrecorded_seconds'])}, timestep={_cell(cnn['runtime_metrics']['start_timestep'])}->{_cell(cnn['runtime_metrics']['end_timestep'])}, evaluation seconds={_cell(cnn['runtime_metrics']['evaluation_seconds'])}; {budget(cnn)}. 주 성능은 checkpoint 선택에 쓰지 않은 primary_test seed 1005..1019의 15개 scenario 평균이며, 이는 15개의 독립 학습 실행이 아니라 하나의 seed 0 실행에서 짝지은 평가다. raw terminal score={raw['primary_test']['mean_terminal_score']:.6g}, CNN terminal score={cnn['primary_test']['mean_terminal_score']:.6g}; 이 수치만으로 우열 또는 통계적 유의성을 결론내리지 않는다.

## 공통 timestep과 효율

공통 timestep={summary['common_step']['timestep']}; primary terminal score는 raw={summary['common_step']['raw_direct']['primary_test']['mean_terminal_score']:.6g}, CNN={summary['common_step']['candidate_cnn']['primary_test']['mean_terminal_score']:.6g}이다. raw-direct steps/s={_cell(raw['runtime_metrics'].get('steps_per_second'))}, CNN steps/s={_cell(cnn['runtime_metrics'].get('steps_per_second'))}; parameter counts는 raw={_cell(raw['runtime_metrics'].get('parameter_counts'))}, CNN={_cell(cnn['runtime_metrics'].get('parameter_counts'))}; peak GPU memory는 raw={_cell(raw['runtime_metrics'].get('peak_cuda_memory_bytes'))}, CNN={_cell(cnn['runtime_metrics'].get('peak_cuda_memory_bytes'))}. raw arm도 동일 schema-3 환경의 shared grid construction 비용을 지불하므로 이 throughput은 grid 생성을 제거한 환경 최적화 효과를 뜻하지 않는다.

## 15개 paired scenario와 20개 보조 보기

`scenario_paired_differences.csv`는 primary_test 15개 seed의 CNN-minus-raw terminal score, dropout, mean delay days, delayed count 차이를 담는다. paired 평균은 terminal={pair_mean['terminal_score_delta_cnn_minus_raw']:.6g}, dropout={pair_mean['dropout_rate_delta_cnn_minus_raw']:.6g}, delay={pair_mean['mean_delay_days_delta_cnn_minus_raw']:.6g}, delayed count={pair_mean['delayed_count_delta_cnn_minus_raw']:.6g}이다. all_holdout 20개 평균 terminal score는 raw={raw['all_holdout']['mean_terminal_score']:.6g}, CNN={cnn['all_holdout']['mean_terminal_score']:.6g}이며 selection seed 1000..1004를 포함하는 보조 보기로서 주 일반화 성능으로 해석하지 않는다.

## 제한과 다음 단계

단일 seed 0, 순차 실행 순서, 동적 Colab 자원, parameter mismatch 때문에 통계적 유의성이나 일반적인 CNN 우월성을 주장하지 않는다. 다음 단계는 paired seed 1..4와 parameter-matched control을 추가하는 것이다. 선택적 curve 입력이 없는 경우 그래프에는 `{MISSING}`로 표시했으며 값을 0으로 만들지 않았다.
"""


def write_complete_report(root: str | Path) -> Path:
    """Write complete report artifacts; integrity marker creation belongs to Task 6."""
    base = Path(root); comparison = base / "comparison"
    summary = build_comparison_summary(base); pairs = build_paired_differences(base)
    _canonical_write_json(comparison / "summary.json", summary); _write_pairs(comparison / "scenario_paired_differences.csv", pairs)
    _learning_plot(base, comparison / "learning_curves.png"); _holdout_plot(summary, pairs, comparison / "holdout_comparison.png")
    report = comparison / "preliminary_comparison_ko.md"; text = _report_text(summary, pairs)
    if "\ufffd" in text: raise ValueError("report contains replacement character")
    report.write_text(text, encoding="utf-8", newline="\n")
    return report


def write_partial_report(root: str | Path, failure: str) -> Path:
    """Document a failed stage without fabricating report data or COMPLETE.json."""
    if "\ufffd" in failure: raise ValueError("failure text contains replacement character")
    base = Path(root)
    def valid_arm(arm: str) -> bool:
        try:
            _runtime(base / arm / "runtime_metrics.json")
            _read_evaluation(base / arm / "evaluation_scenarios.csv", arm, tuple(range(1000, 1020)))
            _read_evaluation(base / arm / "evaluation_primary_test.csv", arm, PRIMARY_TEST_SEEDS)
            return True
        except ValueError:
            return False
    raw_ok, cnn_ok = valid_arm("raw_direct"), valid_arm("candidate_cnn")
    journal_text = "stage metadata: absent"
    journal = base / "stage_journal.json"
    if journal.is_file():
        try:
            journal_text = _stage_journal(_read_json(journal))
        except ValueError:
            journal_text = "stage metadata: invalid metadata"
    state = "후보 CNN 결과가 없어 우열을 결론내리지 않음" if raw_ok and not cnn_ok else ("raw-direct 결과가 없어 비교를 결론내리지 않음" if cnn_ok and not raw_ok else ("두 arm 모두 없어서 비교를 결론내리지 않음" if not raw_ok and not cnn_ok else "두 arm 자료는 있으나 무결성 또는 보고 단계가 불완전하여 결론내리지 않음"))
    text = f"# 부분 비교 보고서\n\n실패 원인: {html.escape(failure)}\n\n사용 가능 단계: raw-direct runtime={'있음' if raw_ok else '없음'}, candidate-CNN runtime={'있음' if cnn_ok else '없음'}.\n{journal_text}\n\n{state}. 누락 수치는 {MISSING}이며 0 또는 추정값으로 대체하지 않는다. 같은 output_root로 같은 experiment runner/notebook을 다시 실행하여 검증 완료 stage를 건너뛰고 재개한다.\n"
    path = base / "comparison" / "PARTIAL_REPORT.md"; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text, encoding="utf-8", newline="\n")
    return path
