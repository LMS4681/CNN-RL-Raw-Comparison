"""Honest selected, final, and paired common-step checkpoint evaluation."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from sb3_contrib import MaskablePPO

import evaluation_runner
from alloc_env.observation_state import ObservationScales
from comparison.artifact_manifest import sha256_file
from evaluation_runner import ModelActionPolicy


REGULAR_INTERVAL = 10_000
EXPECTED_HOLDOUT_SEEDS = tuple(range(1000, 1020))
SELECTION_SEEDS = tuple(range(1000, 1005))
PRIMARY_TEST_SEEDS = tuple(range(1005, 1020))
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
    """Return a readable archive's stored timestep, without a training env."""
    try:
        model = loader(str(path), device="cpu")
        timestep = getattr(model, "num_timesteps", None)
        return int(timestep) if timestep is not None else None
    except (EOFError, OSError, RuntimeError, ValueError, AssertionError):
        return None


def _archive_candidates(output_dir: Path) -> list[Path]:
    roots = [output_dir / "checkpoints", output_dir]
    candidates: set[Path] = set()
    for root in roots:
        if root.is_dir():
            candidates.update(
                path for path in root.iterdir()
                if path.is_file() and path.suffix.lower() in {".sb3", ".zip"}
            )
    return sorted(candidates, key=lambda path: path.as_posix())


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
    for path in _archive_candidates(Path(output_dir)):
        timestep = _archive_timestep(path, model_loader)
        if timestep is None or timestep < 0 or timestep % regular_interval:
            continue
        prior = inventory.get(timestep)
        if prior is None or (path.stat().st_mtime_ns, path.as_posix()) > (
            prior.stat().st_mtime_ns, prior.as_posix()
        ):
            inventory[timestep] = path
    return inventory


def select_common_timestep(
    raw_dir: Path,
    cnn_dir: Path,
    regular_interval: int = REGULAR_INTERVAL,
) -> int:
    raw_steps = set(readable_checkpoint_inventory(raw_dir, regular_interval))
    cnn_steps = set(readable_checkpoint_inventory(cnn_dir, regular_interval))
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
    if not path.is_file() or _archive_timestep(path, loader) != timestep:
        return None
    return CheckpointRef(path=path, label=label, timestep=timestep, sha256=sha256_file(path))


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
        state = json.loads((root / "run_state.json").read_text(encoding="utf-8"))
        if not isinstance(state, dict) or state.get("status") != "complete":
            raise ValueError("run_state is not complete")
        filename = state["last_checkpoint_file"]
        timestep = int(state["last_checkpoint_timestep"])
        expected_sha = state["last_checkpoint_sha256"]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("run_state checkpoint filename is unsafe")
        checkpoint = root / "checkpoints" / filename
        final = _verified_ref(checkpoint, "final", timestep, model_loader)
        if final is None or final.sha256 != expected_sha:
            raise ValueError("run_state checkpoint is unreadable or does not match")
        return final
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
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
    scales = ObservationScales.from_dict(run_config["observation_scales"])
    workspace_codes = list(run_config["active_workspace_codes"])
    state_context = run_config["state_context"]
    model = model_loader(str(model_path), device="cpu")
    timestep = _archive_timestep(Path(model_path), model_loader)
    if timestep is None:
        raise PartialResultError(f"checkpoint is unreadable: {model_path}")
    digest = sha256_file(model_path)
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
    split_holdout_records(rows)
    arm_dir = Path(root) / arm
    all_path = _write_rows(arm_dir / "evaluation_scenarios.csv", rows)
    primary = [row for row in rows if int(row["seed"]) in PRIMARY_TEST_SEEDS]
    return all_path, _write_rows(arm_dir / "evaluation_primary_test.csv", primary)


def write_common_step_evaluation(root: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    return _write_rows(Path(root) / "comparison" / "common_step_evaluation.csv", rows)


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
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(existing, dict):
        raise ValueError("manifest must be a JSON object")
    merged = merge_checkpoint_manifest(existing, arm, checkpoints)
    manifest_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return merged
