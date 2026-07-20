"""Honest selected, final, and paired common-step checkpoint evaluation."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from sb3_contrib import MaskablePPO

import evaluation_runner
from alloc_env.observation_state import ObservationScales
from comparison.artifact_manifest import read_json_object, sha256_file
from comparison.wall_clock_callback import atomic_write_json, read_wall_clock_state, resolve_state_checkpoint
from evaluation_runner import ModelActionPolicy


REGULAR_INTERVAL = 10_000
EXPECTED_HOLDOUT_SEEDS = tuple(range(1000, 1020))
SELECTION_SEEDS = tuple(range(1000, 1005))
PRIMARY_TEST_SEEDS = tuple(range(1005, 1020))
ARMS = ("raw_direct", "candidate_cnn")
EVALUATION_COLUMNS = (
    "source", "policy", "seed", "mean_reward", "mean_terminal_score",
    "mean_dropout_rate", "mean_delay_days", "mean_delayed_count",
    "mean_retained_choice_ratio", "arm", "checkpoint",
    "checkpoint_timestep", "checkpoint_sha256", "evaluation_partition",
)


class PartialResultError(RuntimeError):
    """Raised when a comparison artifact cannot honestly be produced."""


@dataclass(frozen=True)
class CheckpointRef:
    path: Path
    label: Literal["best_model", "fallback_final", "final", "common_step"]
    timestep: int
    sha256: str


def _archive_timestep(path: Path, loader=MaskablePPO.load) -> int | None:
    """Use train's canonical, corruption-tolerant archive reader lazily."""
    from train import model_num_timesteps
    return model_num_timesteps(path, loader=loader)


def _archive_candidates(output_dir: Path) -> list[Path]:
    root = output_dir / "checkpoints"
    return sorted(root.glob("*.sb3"), key=lambda path: path.as_posix()) if root.is_dir() else []


def readable_checkpoint_inventory(
    output_dir: Path,
    regular_interval: int = REGULAR_INTERVAL,
    *,
    model_loader=MaskablePPO.load,
) -> dict[int, Path]:
    """Return newest readable archive at each regular stored timestep."""
    if regular_interval <= 0:
        raise ValueError("regular_interval must be positive")
    inventory: dict[int, Path] = {}
    metadata: dict[int, tuple[int, str]] = {}
    for path in _archive_candidates(Path(output_dir)):
        try:
            timestep = _archive_timestep(path, model_loader)
            digest = sha256_file(path)
            mtime = path.stat().st_mtime_ns
        except (OSError, FileNotFoundError):
            continue
        if timestep is None or timestep < 0 or timestep % regular_interval:
            continue
        prior = inventory.get(timestep)
        if prior is None or (mtime, path.as_posix()) > metadata[timestep]:
            inventory[timestep] = path
            metadata[timestep] = (mtime, path.as_posix())
    return inventory


def select_common_timestep(
    raw_dir: Path,
    cnn_dir: Path,
    regular_interval: int = REGULAR_INTERVAL, *, model_loader=MaskablePPO.load,
) -> int:
    raw_steps = set(readable_checkpoint_inventory(raw_dir, regular_interval, model_loader=model_loader))
    cnn_steps = set(readable_checkpoint_inventory(cnn_dir, regular_interval, model_loader=model_loader))
    shared = raw_steps & cnn_steps
    if not shared:
        raise PartialResultError("no common readable regular checkpoint timestep")
    return max(shared)


def _selected_timestep(selection_path: Path) -> int | None:
    required = (
        "timestep", "mean_terminal_score", "mean_dropout_rate",
        "mean_delay_days", "is_best",
    )
    try:
        with selection_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != required:
                return None
            best = [row for row in reader if row.get("is_best") == "1"]
        return int(best[-1]["timestep"]) if best else None
    except (OSError, ValueError, KeyError):
        return None


def _verified_ref(path: Path, label: Literal["best_model", "fallback_final", "final", "common_step"], timestep: int, loader=MaskablePPO.load) -> CheckpointRef | None:
    try:
        if not path.is_file() or _archive_timestep(path, loader) != timestep:
            return None
        return CheckpointRef(path=path, label=label, timestep=timestep, sha256=sha256_file(path))
    except (OSError, FileNotFoundError):
        return None


def resolve_selected_or_fallback(
    output_dir: Path, *, model_loader=MaskablePPO.load
) -> CheckpointRef:
    """Use a proven selected model or only the exact complete state checkpoint."""
    root = Path(output_dir)
    best = root / "best_model.sb3"
    selected_timestep = _selected_timestep(root / "holdout_selection.csv")
    if selected_timestep is not None:
        selected = _verified_ref(best, "best_model", selected_timestep, model_loader)
        if selected is not None:
            return selected

    final = resolve_final_checkpoint(root, model_loader=model_loader)
    return CheckpointRef(final.path, "fallback_final", final.timestep, final.sha256)


def resolve_final_checkpoint(
    output_dir: Path, *, model_loader=MaskablePPO.load
) -> CheckpointRef:
    """Return only the readable archive named and verified by complete state."""
    root = Path(output_dir)
    try:
        state = read_wall_clock_state(root / "run_state.json")
        if state.status != "complete":
            raise ValueError("run_state is not complete")
        checkpoint = resolve_state_checkpoint(root, state)
        final = _verified_ref(checkpoint, "final", state.last_checkpoint_timestep, model_loader)
        if final is None or final.sha256 != state.last_checkpoint_sha256:
            raise ValueError("run_state checkpoint is unreadable or does not match")
        return final
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise PartialResultError("no exact complete final checkpoint") from error


def split_holdout_records(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict], list[dict]]:
    normalized = [dict(record) for record in records]
    seeds = [int(record["seed"]) for record in normalized]
    if len(seeds) != len(set(seeds)) or set(seeds) != set(EXPECTED_HOLDOUT_SEEDS):
        raise ValueError("holdout scenarios must have each seed 1000 through 1019 exactly once")
    by_seed = {int(record["seed"]): record for record in normalized}
    return ([by_seed[seed] for seed in SELECTION_SEEDS], [by_seed[seed] for seed in PRIMARY_TEST_SEEDS])


def evaluate_checkpoint(
    model_path: Path,
    run_config: Mapping[str, Any],
    scenarios: Sequence[dict],
    checkpoint_label: str,
    arm: str,
    model_loader=MaskablePPO.load,
) -> list[dict]:
    """Evaluate one archive and attach truthful checkpoint provenance."""
    selection, primary = split_holdout_records(scenarios)
    ordered = [*selection, *primary]
    if arm not in ARMS:
        raise ValueError("unknown comparison arm")
    try:
        from train import load_model_run_config
        adjacent_config = load_model_run_config(model_path)
        if dict(adjacent_config) != dict(run_config):
            raise ValueError("provided run_config does not match adjacent run_config.json")
        scales = ObservationScales.from_dict(adjacent_config["observation_scales"])
        workspace_codes = list(adjacent_config["active_workspace_codes"])
        state_context = adjacent_config["state_context"]
        model = model_loader(str(model_path), device="cpu")
        timestep = int(getattr(model, "num_timesteps"))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise PartialResultError(f"checkpoint/config is not evaluable: {model_path}") from error
    try:
        digest = sha256_file(model_path)
    except (OSError, FileNotFoundError) as error:
        raise PartialResultError(f"checkpoint disappeared during verification: {model_path}") from error
    base_rows = evaluation_runner.evaluate_scenarios(
        lambda _seed: ModelActionPolicy(model, name=arm), list(ordered),
        workspace_codes=workspace_codes, observation_scales=scales,
        state_context_mode=state_context,
    )
    if [int(row["seed"]) for row in base_rows] != list(EXPECTED_HOLDOUT_SEEDS):
        raise PartialResultError("evaluation runner did not return the fixed holdout seeds")
    rows: list[dict] = []
    for base in base_rows:
        row = dict(base)
        seed = int(row["seed"])
        row.update({
            "arm": arm, "checkpoint": checkpoint_label,
            "checkpoint_timestep": timestep, "checkpoint_sha256": digest,
            "evaluation_partition": "selection" if seed in SELECTION_SEEDS else "primary_test",
        })
        rows.append(row)
    return rows


def _validated_rows(rows: Sequence[Mapping[str, Any]], *, arm: str | None = None, common: bool = False) -> list[dict]:
    if arm is not None and arm not in ARMS:
        raise ValueError("unknown comparison arm")
    normalized = [dict(row) for row in rows]
    if not normalized or any(set(row) != set(EVALUATION_COLUMNS) for row in normalized):
        raise ValueError("evaluation rows must have the exact stable columns")
    by_arm = {row["arm"] for row in normalized}
    expected_arms = set(ARMS) if common else {arm}
    if by_arm != expected_arms:
        raise ValueError("evaluation rows have wrong arm identifiers")
    for arm_name in by_arm:
        group = [row for row in normalized if row["arm"] == arm_name]
        split_holdout_records(group)
        if any(row["evaluation_partition"] != ("selection" if int(row["seed"]) in SELECTION_SEEDS else "primary_test") for row in group):
            raise ValueError("evaluation partition is inconsistent")
        if common and any(row["checkpoint"] != "common_step" for row in group):
            raise ValueError("common rows must be labelled common_step")
        if len({(row["checkpoint"], row["checkpoint_timestep"], row["checkpoint_sha256"]) for row in group}) != 1:
            raise ValueError("checkpoint provenance is inconsistent")
    if common and len({row["checkpoint_timestep"] for row in normalized}) != 1:
        raise ValueError("common rows must share a timestep")
    ordered = [{field: row[field] for field in EVALUATION_COLUMNS} for row in normalized]
    return sorted(ordered, key=lambda row: ((ARMS.index(row["arm"]) if common else 0), int(row["seed"])))


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    if not rows:
        raise ValueError("evaluation rows are required")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EVALUATION_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_arm_evaluations(root: Path, arm: str, rows: Sequence[Mapping[str, Any]]) -> tuple[Path, Path]:
    rows = _validated_rows(rows, arm=arm)
    arm_dir = Path(root) / arm
    all_path = _write_rows(arm_dir / "evaluation_scenarios.csv", rows)
    primary = [row for row in rows if int(row["seed"]) in PRIMARY_TEST_SEEDS]
    return all_path, _write_rows(arm_dir / "evaluation_primary_test.csv", primary)


def write_common_step_evaluation(root: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    return _write_rows(Path(root) / "comparison" / "common_step_evaluation.csv", _validated_rows(rows, common=True))


def merge_checkpoint_manifest(
    manifest: Mapping[str, Any], arm: str, checkpoints: Mapping[str, CheckpointRef]
) -> dict[str, Any]:
    """Copy and update one arm's known checkpoint references only."""
    merged = dict(manifest)
    all_checkpoints = dict(merged.get("checkpoints", {}))
    arm_checkpoints = dict(all_checkpoints.get(arm, {}))
    for name, reference in checkpoints.items():
        arm_checkpoints[name] = {
            "path": reference.path.as_posix(), "label": reference.label,
            "sha256": reference.sha256, "timestep": reference.timestep,
        }
    all_checkpoints[arm] = arm_checkpoints
    merged["checkpoints"] = all_checkpoints
    return merged


def update_checkpoint_manifest(path: Path, arm: str, checkpoints: Mapping[str, CheckpointRef]) -> dict[str, Any]:
    manifest_path = Path(path)
    try: existing = read_json_object(manifest_path)
    except (OSError, UnicodeDecodeError, ValueError) as error: raise ValueError("manifest must be a JSON object") from error
    merged = merge_checkpoint_manifest(existing, arm, checkpoints)
    atomic_write_json(manifest_path, merged)
    return merged


def evaluate_comparison_artifacts(
    root: Path, raw_dir: Path, cnn_dir: Path, scenarios: Sequence[dict],
    raw_config: Mapping[str, Any], cnn_config: Mapping[str, Any], *,
    regular_interval: int = REGULAR_INTERVAL, model_loader=MaskablePPO.load,
) -> dict[str, dict[str, CheckpointRef]]:
    """Evaluate both arms only when an existing root manifest is present."""
    root = Path(root).resolve()
    raw_dir, cnn_dir = Path(raw_dir).resolve(), Path(cnn_dir).resolve()
    for expected, directory in (("raw_direct", raw_dir), ("candidate_cnn", cnn_dir)):
        if directory.parent != root or directory.name != expected:
            raise PartialResultError("comparison arm directory escapes root")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise PartialResultError("root manifest.json must already exist")
    common_step = select_common_timestep(raw_dir, cnn_dir, regular_interval, model_loader=model_loader)
    inventories = {
        "raw_direct": readable_checkpoint_inventory(raw_dir, regular_interval, model_loader=model_loader),
        "candidate_cnn": readable_checkpoint_inventory(cnn_dir, regular_interval, model_loader=model_loader),
    }
    refs: dict[str, dict[str, CheckpointRef]] = {}
    manifest_updates: dict[str, dict[str, CheckpointRef]] = {}
    configs = {"raw_direct": raw_config, "candidate_cnn": cnn_config}
    directories = {"raw_direct": raw_dir, "candidate_cnn": cnn_dir}
    common_rows: list[dict] = []
    for arm in ARMS:
        final = resolve_final_checkpoint(directories[arm], model_loader=model_loader)
        selected = resolve_selected_or_fallback(directories[arm], model_loader=model_loader)
        common_path = inventories[arm][common_step]
        common = _verified_ref(common_path, "common_step", common_step, model_loader)
        if common is None:
            raise PartialResultError("common checkpoint became unreadable")
        refs[arm] = {"selected": selected, "final": final, "common": common}
        selected_rows = evaluate_checkpoint(selected.path, configs[arm], scenarios, selected.label, arm, model_loader)
        write_arm_evaluations(root, arm, selected_rows)
        common_rows.extend(evaluate_checkpoint(common.path, configs[arm], scenarios, "common_step", arm, model_loader))
        try:
            relative_refs = {name: CheckpointRef(ref.path.resolve().relative_to(root), ref.label, ref.timestep, ref.sha256) for name, ref in refs[arm].items()}
        except ValueError as error:
            raise PartialResultError("checkpoint reference escapes root") from error
        manifest_updates[arm] = relative_refs
    write_common_step_evaluation(root, common_rows)
    try: manifest = read_json_object(manifest_path)
    except (OSError, UnicodeDecodeError, ValueError) as error: raise PartialResultError("root manifest.json is invalid") from error
    for arm in ARMS:
        manifest = merge_checkpoint_manifest(manifest, arm, manifest_updates[arm])
    atomic_write_json(manifest_path, manifest)
    return refs
