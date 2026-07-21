"""Honest selected, final, and paired common-step checkpoint evaluation."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Literal

from sb3_contrib import MaskablePPO

import evaluation_runner
from alloc_env.observation_state import ObservationScales
from comparison.artifact_manifest import (
    CANONICAL_SELECTION_COUNT,
    FALLBACK_REASON_CODES,
    canonical_json_sha256,
    read_json_object,
    read_runtime_metrics,
    sha256_file,
)
from comparison.path_integrity import (
    resolve_direct_directory,
    resolve_direct_regular_file,
)
from comparison.training_completion import read_training_completion
from comparison.wall_clock_callback import atomic_write_json, read_wall_clock_state, resolve_state_checkpoint
from evaluation_runner import ModelActionPolicy


REGULAR_INTERVAL = 10_000
EXPECTED_HOLDOUT_SEEDS = tuple(range(1000, 1020))
SELECTION_SEEDS = tuple(range(1000, 1005))
PRIMARY_TEST_SEEDS = tuple(range(1005, 1020))
ARMS = ("raw_direct", "candidate_cnn")
if len(SELECTION_SEEDS) != CANONICAL_SELECTION_COUNT:
    raise RuntimeError("selection protocol count differs from canonical metadata")
EVALUATION_COLUMNS = (
    "source", "policy", "seed", "mean_reward", "mean_terminal_score",
    "mean_dropout_rate", "mean_delay_days", "mean_delayed_count",
    "mean_retained_choice_ratio", "arm", "checkpoint",
    "checkpoint_timestep", "checkpoint_sha256", "evaluation_partition",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ARM_EVALUATION_MARKER_KEYS = frozenset(
    {
        "schema_version",
        "arm",
        "config_sha256",
        "scenario_sha256",
        "checkpoints",
        "artifacts",
        "evaluation_seed_count",
        "primary_test_seed_count",
        "selection_outcome",
        "fallback_reason",
    }
)
_ARM_EVALUATION_ARTIFACTS = (
    "evaluation_scenarios.csv",
    "evaluation_primary_test.csv",
    "training_completion.json",
    "runtime_metrics.json",
)
_COMMON_CACHE_NAMES = {
    arm: f"common_step_{arm}.cache.json" for arm in ARMS
}
_COMMON_COMBINED_NAME = "common_step_evaluation.csv"
_COMMON_MARKER_NAME = "common_step_stage.json"
_COMMON_CACHE_KEYS = frozenset(
    {
        "schema_version",
        "arm",
        "config_sha256",
        "run_config_sha256",
        "scenario_sha256",
        "checkpoint",
        "rows",
    }
)
_COMMON_MARKER_KEYS = frozenset(
    {
        "schema_version",
        "config_sha256",
        "run_config_sha256",
        "scenario_sha256",
        "common_timestep",
        "checkpoints",
        "artifacts",
        "evaluation_seed_count_per_arm",
    }
)


class PartialResultError(RuntimeError):
    """Raised when a comparison artifact cannot honestly be produced."""


@dataclass(frozen=True)
class CheckpointRef:
    path: Path
    label: Literal["best_model", "fallback_final", "final", "common_step"]
    timestep: int
    sha256: str


@dataclass(frozen=True)
class SelectionDecision:
    reference: CheckpointRef
    selection_outcome: Literal["best_model", "fallback_final"]
    fallback_reason: str | None
    selection_count: int
    selection_tuple: list[float] | None

    def runtime_fields(self) -> dict[str, Any]:
        return {
            "selected_checkpoint_timestep": self.reference.timestep,
            "selection_count": self.selection_count,
            "selection_tuple": self.selection_tuple,
            "selection_outcome": self.selection_outcome,
            "fallback_reason": self.fallback_reason,
            "checkpoint_identity": {
                "filename": self.reference.path.name,
                "sha256": self.reference.sha256,
            },
        }


def _archive_timestep(path: Path, loader=MaskablePPO.load) -> int | None:
    """Use train's canonical, corruption-tolerant archive reader lazily."""
    from train import model_num_timesteps
    return model_num_timesteps(path, loader=loader)


def _archive_candidates(output_dir: Path) -> list[Path]:
    try:
        root = resolve_direct_directory(
            output_dir,
            output_dir / "checkpoints",
            label="checkpoint inventory directory",
        )
    except (FileNotFoundError, ValueError):
        return []
    candidates: list[Path] = []
    try:
        paths = list(root.glob("*.sb3"))
    except OSError:
        return []
    for path in paths:
        try:
            candidates.append(
                resolve_direct_regular_file(
                    root, path, label="checkpoint inventory candidate"
                )
            )
        except (FileNotFoundError, ValueError):
            continue
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
    metadata: dict[int, tuple[int, str]] = {}
    for path in _archive_candidates(Path(output_dir)):
        try:
            timestep = _archive_timestep(path, model_loader)
            digest = sha256_file(path)
            # The candidate was already lstat-verified as a direct regular
            # file.  Do not restat a prior duplicate through Path.stat: it may
            # disappear independently while another candidate remains valid.
            mtime = os.lstat(path).st_mtime_ns
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


def _selection_best_row(
    selection_path: Path,
) -> tuple[tuple[int, list[float]] | None, str | None]:
    required = (
        "timestep", "mean_terminal_score", "mean_dropout_rate",
        "mean_delay_days", "is_best",
    )
    try:
        selected_file = resolve_direct_regular_file(
            selection_path.parent,
            selection_path,
            label="holdout selection metadata",
        )
    except FileNotFoundError:
        return None, "selection_not_run"
    except (OSError, ValueError):
        return None, "selection_metadata_invalid"
    try:
        if selected_file.stat().st_size == 0:
            return None, "selection_not_run"
    except OSError:
        return None, "selection_metadata_invalid"
    try:
        with selected_file.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != required:
                return None, "selection_metadata_invalid"
            rows = list(reader)
        normalized: list[tuple[int, list[float], str]] = []
        for row in rows:
            if set(row) != set(required) or row["is_best"] not in {"0", "1"}:
                return None, "selection_metadata_invalid"
            timestep = int(row["timestep"])
            values = [
                float(row["mean_terminal_score"]),
                -float(row["mean_dropout_rate"]),
                -float(row["mean_delay_days"]),
            ]
            if timestep < 0 or any(not math.isfinite(value) for value in values):
                return None, "selection_metadata_invalid"
            normalized.append((timestep, values, row["is_best"]))
        best = [(timestep, values) for timestep, values, flag in normalized if flag == "1"]
        return (best[-1], None) if best else (None, "selection_has_no_best")
    except (OSError, UnicodeDecodeError, csv.Error, ValueError, KeyError, TypeError):
        return None, "selection_metadata_invalid"


def _selected_timestep(selection_path: Path) -> int | None:
    best, _reason = _selection_best_row(selection_path)
    return best[0] if best is not None else None


def _verified_ref(path: Path, label: Literal["best_model", "fallback_final", "final", "common_step"], timestep: int, loader=MaskablePPO.load) -> CheckpointRef | None:
    try:
        verified_path = resolve_direct_regular_file(
            path.parent, path, label=f"{label} checkpoint"
        )
        if _archive_timestep(verified_path, loader) != timestep:
            return None
        return CheckpointRef(path=verified_path, label=label, timestep=timestep, sha256=sha256_file(verified_path))
    except (OSError, FileNotFoundError, ValueError):
        return None


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


def resolve_selection_decision(
    output_dir: Path,
    *,
    model_loader=MaskablePPO.load,
    archive_timestep_reader=None,
    final_reference: CheckpointRef | None = None,
) -> SelectionDecision:
    """Resolve one selected checkpoint and preserve the exact fallback cause."""
    root = Path(output_dir)
    final = final_reference or resolve_final_checkpoint(
        root, model_loader=model_loader
    )
    metadata, reason = _selection_best_row(root / "holdout_selection.csv")
    if metadata is not None:
        selected_timestep, ranking = metadata
        if selected_timestep > final.timestep:
            metadata = None
            reason = "selection_metadata_invalid"
    if metadata is not None:
        selected_timestep, ranking = metadata
        best = root / "best_model.sb3"
        try:
            verified_best = resolve_direct_regular_file(
                root, best, label="best model"
            )
        except FileNotFoundError:
            reason = "best_model_missing"
        except (OSError, ValueError):
            reason = "best_model_unreadable"
        else:
            try:
                digest_before = sha256_file(verified_best)
                actual_timestep = (
                    archive_timestep_reader(verified_best)
                    if archive_timestep_reader is not None
                    else _archive_timestep(verified_best, model_loader)
                )
                digest_after = sha256_file(verified_best)
                stable_best = resolve_direct_regular_file(
                    root, best, label="best model"
                )
                digest_stable = sha256_file(stable_best)
            except FileNotFoundError:
                reason = "best_model_missing"
            except (OSError, RuntimeError, ValueError, TypeError):
                actual_timestep = None
                reason = "best_model_unreadable"
            if reason in {"best_model_missing", "best_model_unreadable"}:
                pass
            elif (
                stable_best != verified_best
                or digest_before != digest_after
                or digest_after != digest_stable
            ):
                reason = "best_model_unreadable"
            elif actual_timestep is None:
                reason = "best_model_unreadable"
            elif actual_timestep != selected_timestep:
                reason = "best_model_timestep_mismatch"
            else:
                reference = CheckpointRef(
                    verified_best,
                    "best_model",
                    selected_timestep,
                    digest_stable,
                )
                return SelectionDecision(
                    reference=reference,
                    selection_outcome="best_model",
                    fallback_reason=None,
                    selection_count=len(SELECTION_SEEDS),
                    selection_tuple=ranking,
                )
    if reason not in FALLBACK_REASON_CODES:
        raise PartialResultError("selection fallback reason is not canonical")
    fallback = CheckpointRef(
        final.path, "fallback_final", final.timestep, final.sha256
    )
    return SelectionDecision(
        reference=fallback,
        selection_outcome="fallback_final",
        fallback_reason=reason,
        selection_count=0,
        selection_tuple=None,
    )


def resolve_selected_or_fallback(
    output_dir: Path, *, model_loader=MaskablePPO.load
) -> CheckpointRef:
    """Compatibility wrapper returning only the canonical selected reference."""
    return resolve_selection_decision(
        output_dir, model_loader=model_loader
    ).reference


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


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Replace one artifact only after flushing, then verify exact bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
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
            raise OSError(f"atomic artifact verification failed: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _rows_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("evaluation rows are required")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=EVALUATION_COLUMNS,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    _atomic_write_bytes(path, _rows_bytes(rows))
    return path


def write_arm_evaluations(root: Path, arm: str, rows: Sequence[Mapping[str, Any]]) -> tuple[Path, Path]:
    rows = _validated_rows(rows, arm=arm)
    arm_dir = Path(root) / arm
    all_path = _write_rows(arm_dir / "evaluation_scenarios.csv", rows)
    primary = [row for row in rows if int(row["seed"]) in PRIMARY_TEST_SEEDS]
    return all_path, _write_rows(arm_dir / "evaluation_primary_test.csv", primary)


def write_common_step_evaluation(root: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    return _write_rows(Path(root) / "comparison" / "common_step_evaluation.csv", _validated_rows(rows, common=True))


def _sha256_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be SHA-256")
    return value


def _exact_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _reference_path_within_arm(reference: CheckpointRef, arm_root: Path) -> Path:
    if reference.label == "best_model":
        directory = arm_root
        if reference.path.name != "best_model.sb3":
            raise PartialResultError("best checkpoint is not the direct canonical file")
    elif reference.label in {"final", "fallback_final", "common_step"}:
        directory = arm_root / "checkpoints"
        is_junction = getattr(directory, "is_junction", None)
        if directory.is_symlink() or (
            is_junction is not None and is_junction()
        ):
            raise PartialResultError("checkpoint directory is not a regular directory")
    else:
        raise PartialResultError("checkpoint reference label is invalid")
    try:
        return resolve_direct_regular_file(
            directory,
            reference.path,
            label=f"{reference.label} checkpoint",
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise PartialResultError(
            "checkpoint reference is not a direct regular file"
        ) from error


def _stable_checkpoint_reference(
    reference: CheckpointRef,
    arm_root: Path,
    *,
    model_loader=MaskablePPO.load,
) -> None:
    """Recheck identity after evaluation so publication cannot bind a raced file."""
    path = _reference_path_within_arm(reference, arm_root)
    try:
        digest_before = sha256_file(path)
        timestep = _archive_timestep(path, model_loader)
        digest_after = sha256_file(path)
        stable_path = _reference_path_within_arm(reference, arm_root)
        digest_stable = sha256_file(stable_path)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise PartialResultError("checkpoint changed during arm evaluation") from error
    if (
        stable_path != path
        or digest_before != digest_after
        or digest_after != digest_stable
        or digest_stable != reference.sha256
        or timestep != reference.timestep
    ):
        raise PartialResultError("checkpoint changed during arm evaluation")


def _root_relative_reference(
    root: Path, arm_root: Path, reference: CheckpointRef
) -> CheckpointRef:
    path = _reference_path_within_arm(reference, arm_root)
    try:
        relative = path.relative_to(root.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as error:
        raise PartialResultError("checkpoint reference escapes comparison root") from error
    return CheckpointRef(relative, reference.label, reference.timestep, reference.sha256)


def _reference_payload(reference: CheckpointRef) -> dict[str, Any]:
    return {
        "path": reference.path.as_posix(),
        "label": reference.label,
        "sha256": reference.sha256,
        "timestep": reference.timestep,
    }


def _validate_marker_reference(value: Any, arm: str, kind: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "label",
        "sha256",
        "timestep",
    }:
        raise ValueError("evaluation checkpoint reference has invalid schema")
    result = dict(value)
    path = result["path"]
    if (
        not isinstance(path, str)
        or not path
        or "\\" in path
        or Path(path).is_absolute()
        or Path(path).parts[0] != arm
        or any(part in {"", ".", ".."} for part in Path(path).parts)
    ):
        raise ValueError("evaluation checkpoint path is invalid")
    allowed = {"best_model", "fallback_final"} if kind == "selected" else {"final"}
    if result["label"] not in allowed:
        raise ValueError("evaluation checkpoint label is invalid")
    expected_prefix = (
        f"{arm}/best_model.sb3"
        if result["label"] == "best_model"
        else f"{arm}/checkpoints/"
    )
    if (
        result["label"] == "best_model"
        and path != expected_prefix
    ) or (
        result["label"] != "best_model"
        and not path.startswith(expected_prefix)
    ):
        raise ValueError("evaluation checkpoint path does not match its label")
    _sha256_text(result["sha256"], "evaluation checkpoint sha256")
    _exact_nonnegative_int(result["timestep"], "evaluation checkpoint timestep")
    return result


def _read_published_evaluation(
    arm_root: Path,
    name: str,
    arm: str,
    expected_seeds: tuple[int, ...],
) -> list[dict[str, str]]:
    try:
        path = resolve_direct_regular_file(
            arm_root,
            arm_root / name,
            label=f"{arm} {name}",
        )
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != EVALUATION_COLUMNS:
                raise ValueError("evaluation CSV has incompatible header")
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ValueError("evaluation CSV is unreadable") from error
    if len(rows) != len(expected_seeds):
        raise ValueError("evaluation CSV has wrong row count")
    observed_seeds: list[int] = []
    for row in rows:
        if set(row) != set(EVALUATION_COLUMNS):
            raise ValueError("evaluation CSV has invalid columns")
        seed_text = row["seed"]
        timestep_text = row["checkpoint_timestep"]
        if not seed_text.isdigit() or not timestep_text.isdigit():
            raise ValueError("evaluation CSV has invalid integer fields")
        seed = int(seed_text)
        observed_seeds.append(seed)
        if (
            row["source"] != "holdout_fixed20"
            or row["policy"] != arm
            or row["arm"] != arm
            or row["checkpoint"] not in {"best_model", "fallback_final"}
            or row["evaluation_partition"]
            != ("selection" if seed in SELECTION_SEEDS else "primary_test")
        ):
            raise ValueError("evaluation CSV has invalid fixed provenance")
        _sha256_text(row["checkpoint_sha256"], "evaluation CSV checkpoint sha256")
        for column in (
            "mean_reward",
            "mean_terminal_score",
            "mean_dropout_rate",
            "mean_delay_days",
            "mean_delayed_count",
            "mean_retained_choice_ratio",
        ):
            try:
                number = float(row[column])
            except (TypeError, ValueError) as error:
                raise ValueError("evaluation CSV has invalid numeric fields") from error
            if not math.isfinite(number):
                raise ValueError("evaluation CSV has invalid numeric fields")
    if tuple(sorted(observed_seeds)) != expected_seeds:
        raise ValueError("evaluation CSV has wrong fixed seeds")
    return sorted(rows, key=lambda row: int(row["seed"]))


def validate_arm_evaluation_stage(
    root: str | Path,
    arm: str,
    *,
    expected_config_sha256: str | None = None,
    expected_scenario_sha256: str | None = None,
    archive_timestep_reader: Callable[[Path], int | None] | None = None,
) -> dict[str, Any]:
    """Validate the per-arm marker as the commit point for its two CSVs."""
    if arm not in ARMS:
        raise ValueError("unknown comparison arm")
    base = Path(root).resolve(strict=True)
    arm_root = (base / arm).resolve(strict=True)
    if arm_root.parent != base or arm_root.name != arm:
        raise ValueError("comparison arm directory escapes root")
    marker_path = resolve_direct_regular_file(
        arm_root,
        arm_root / "evaluation_stage.json",
        label=f"{arm} evaluation stage marker",
    )
    try:
        marker = read_json_object(marker_path)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ValueError("evaluation stage marker is invalid") from error
    if set(marker) != _ARM_EVALUATION_MARKER_KEYS:
        raise ValueError("evaluation stage marker has invalid schema")
    if marker["schema_version"] != 1 or marker["arm"] != arm:
        raise ValueError("evaluation stage marker identity is invalid")
    config_sha = _sha256_text(marker["config_sha256"], "evaluation config hash")
    scenario_sha = _sha256_text(marker["scenario_sha256"], "evaluation scenario hash")
    if expected_config_sha256 is not None and config_sha != expected_config_sha256:
        raise ValueError("evaluation config hash mismatch")
    if expected_scenario_sha256 is not None and scenario_sha != expected_scenario_sha256:
        raise ValueError("evaluation scenario hash mismatch")
    if marker["evaluation_seed_count"] != len(EXPECTED_HOLDOUT_SEEDS):
        raise ValueError("evaluation seed count is invalid")
    if marker["primary_test_seed_count"] != len(PRIMARY_TEST_SEEDS):
        raise ValueError("primary-test seed count is invalid")
    checkpoints = marker["checkpoints"]
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != {"selected", "final"}:
        raise ValueError("evaluation checkpoints have invalid schema")
    selected = _validate_marker_reference(checkpoints["selected"], arm, "selected")
    final = _validate_marker_reference(checkpoints["final"], arm, "final")
    outcome = marker["selection_outcome"]
    fallback_reason = marker["fallback_reason"]
    if outcome == "best_model":
        if selected["label"] != "best_model" or fallback_reason is not None:
            raise ValueError("evaluation selection outcome is inconsistent")
    elif outcome == "fallback_final":
        if (
            selected["label"] != "fallback_final"
            or fallback_reason not in FALLBACK_REASON_CODES
        ):
            raise ValueError("evaluation selection outcome is inconsistent")
    else:
        raise ValueError("evaluation selection outcome is invalid")
    artifacts = marker["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        _ARM_EVALUATION_ARTIFACTS
    ):
        raise ValueError("evaluation artifacts have invalid schema")
    for name in _ARM_EVALUATION_ARTIFACTS:
        expected_digest = _sha256_text(
            artifacts[name], f"{name} evaluation artifact hash"
        )
        try:
            artifact = resolve_direct_regular_file(
                arm_root, arm_root / name, label=f"{arm} {name}"
            )
            actual_digest = sha256_file(artifact)
        except (OSError, ValueError) as error:
            raise ValueError("evaluation artifact is not a direct regular file") from error
        if actual_digest != expected_digest:
            raise ValueError("evaluation artifact hash mismatch")
    try:
        runtime = read_runtime_metrics(arm_root / "runtime_metrics.json")
        receipt = read_training_completion(arm_root / "training_completion.json")
    except (OSError, UnicodeDecodeError, TypeError, ValueError) as error:
        raise ValueError("evaluation training evidence is invalid") from error
    if (
        runtime["selected_checkpoint_timestep"] != selected["timestep"]
        or runtime["selection_outcome"] != outcome
        or runtime["fallback_reason"] != fallback_reason
        or runtime["checkpoint_identity"]
        != {
            "filename": Path(selected["path"]).name,
            "sha256": selected["sha256"],
        }
        or runtime["selection_count"]
        != (len(SELECTION_SEEDS) if outcome == "best_model" else 0)
        or (
            (runtime["selection_tuple"] is None)
            != (outcome == "fallback_final")
        )
    ):
        raise ValueError("evaluation runtime selected fields do not reconcile")
    if (
        receipt["config_sha256"] != config_sha
        or receipt["final_timestep"] != final["timestep"]
        or receipt["checkpoint_file"] != Path(final["path"]).name
        or receipt["checkpoint_sha256"] != final["sha256"]
        or receipt["artifact_sha256"]["runtime_metrics.json"]
        != artifacts["runtime_metrics.json"]
        or (
            outcome == "best_model"
            and receipt["artifact_sha256"]["best_model.sb3"]
            != selected["sha256"]
        )
    ):
        raise ValueError("evaluation training receipt does not reconcile")
    if outcome == "fallback_final" and (
        selected["path"], selected["timestep"], selected["sha256"]
    ) != (final["path"], final["timestep"], final["sha256"]):
        raise ValueError("evaluation fallback does not identify final checkpoint")
    all_rows = _read_published_evaluation(
        arm_root, "evaluation_scenarios.csv", arm, EXPECTED_HOLDOUT_SEEDS
    )
    primary_rows = _read_published_evaluation(
        arm_root, "evaluation_primary_test.csv", arm, PRIMARY_TEST_SEEDS
    )
    if primary_rows != [
        row for row in all_rows if int(row["seed"]) in PRIMARY_TEST_SEEDS
    ]:
        raise ValueError("primary-test evaluation is not the exact all-row subset")
    provenance = {
        (
            row["checkpoint"],
            int(row["checkpoint_timestep"]),
            row["checkpoint_sha256"],
        )
        for row in all_rows
    }
    if provenance != {
        (selected["label"], selected["timestep"], selected["sha256"])
    }:
        raise ValueError("evaluation CSV provenance differs from selected checkpoint")
    try:
        manifest_path = resolve_direct_regular_file(
            base, base / "manifest.json", label="root manifest"
        )
        manifest = read_json_object(manifest_path)
        manifest_checkpoints = manifest["checkpoints"][arm]
    except (OSError, UnicodeDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("evaluation checkpoint manifest is invalid") from error
    if not isinstance(manifest_checkpoints, Mapping) or {
        "selected": manifest_checkpoints.get("selected"),
        "final": manifest_checkpoints.get("final"),
    } != {"selected": selected, "final": final}:
        raise ValueError("evaluation checkpoint manifest mismatch")
    if manifest.get("config_sha256") != config_sha:
        raise ValueError("evaluation config hash differs from root manifest")
    if manifest.get("scenario_sha256") != scenario_sha:
        raise ValueError("evaluation scenario hash differs from root manifest")
    selected_reference = CheckpointRef(
        base / selected["path"],
        selected["label"],
        selected["timestep"],
        selected["sha256"],
    )
    final_reference = CheckpointRef(
        base / final["path"],
        "final",
        final["timestep"],
        final["sha256"],
    )
    try:
        for reference in (selected_reference, final_reference):
            checkpoint = _reference_path_within_arm(reference, arm_root)
            if sha256_file(checkpoint) != reference.sha256:
                raise ValueError("evaluation checkpoint hash mismatch")
            if (
                archive_timestep_reader is not None
                and archive_timestep_reader(checkpoint) != reference.timestep
            ):
                raise ValueError("evaluation checkpoint timestep mismatch")
    except PartialResultError as error:
        raise ValueError("evaluation checkpoint is not direct regular") from error
    return marker


def _validate_common_reference(value: Any, arm: str) -> dict[str, Any]:
    """Validate the exact, direct checkpoint reference owned by this stage."""
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "label",
        "sha256",
        "timestep",
    }:
        raise ValueError("common checkpoint reference has invalid schema")
    reference = dict(value)
    path = reference["path"]
    parts = PurePosixPath(path).parts if isinstance(path, str) else ()
    if (
        not isinstance(path, str)
        or not path
        or "\\" in path
        or PurePosixPath(path).is_absolute()
        or parts[:2] != (arm, "checkpoints")
        or len(parts) != 3
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError("common checkpoint path is invalid")
    if reference["label"] != "common_step":
        raise ValueError("common checkpoint label is invalid")
    _sha256_text(reference["sha256"], "common checkpoint sha256")
    _exact_nonnegative_int(reference["timestep"], "common checkpoint timestep")
    return reference


def _normalized_common_rows(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
    checkpoint: Mapping[str, Any],
) -> list[dict[str, Any]]:
    try:
        validated = _validated_rows(rows, arm=arm)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("common cache rows are invalid") from error
    if len(validated) != len(EXPECTED_HOLDOUT_SEEDS):
        raise ValueError("common cache must contain exact fixed holdout rows")
    normalized: list[dict[str, Any]] = []
    for row in validated:
        if (
            row["source"] != "holdout_fixed20"
            or row["policy"] != arm
            or row["checkpoint"] != "common_step"
        ):
            raise ValueError("common cache row identity is invalid")
        try:
            seed = int(row["seed"])
            timestep = int(row["checkpoint_timestep"])
        except (TypeError, ValueError) as error:
            raise ValueError("common cache integer field is invalid") from error
        if isinstance(row["seed"], bool) or isinstance(
            row["checkpoint_timestep"], bool
        ):
            raise ValueError("common cache integer field is invalid")
        converted = dict(row)
        converted["seed"] = seed
        converted["checkpoint_timestep"] = timestep
        for column in (
            "mean_reward",
            "mean_terminal_score",
            "mean_dropout_rate",
            "mean_delay_days",
            "mean_delayed_count",
            "mean_retained_choice_ratio",
        ):
            try:
                number = float(row[column])
            except (TypeError, ValueError) as error:
                raise ValueError("common cache numeric field is invalid") from error
            if not math.isfinite(number):
                raise ValueError("common cache numeric field is invalid")
            converted[column] = number
        normalized.append(
            {field: converted[field] for field in EVALUATION_COLUMNS}
        )
    expected_provenance = {
        (
            "common_step",
            checkpoint["timestep"],
            checkpoint["sha256"],
        )
    }
    provenance = {
        (
            row["checkpoint"],
            row["checkpoint_timestep"],
            row["checkpoint_sha256"],
        )
        for row in normalized
    }
    if provenance != expected_provenance:
        raise ValueError("common cache checkpoint provenance mismatch")
    return sorted(normalized, key=lambda row: row["seed"])


def _common_cache_payload(
    arm: str,
    *,
    config_sha256: str,
    run_config_sha256: str,
    scenario_sha256: str,
    checkpoint: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "arm": arm,
        "config_sha256": config_sha256,
        "run_config_sha256": run_config_sha256,
        "scenario_sha256": scenario_sha256,
        "checkpoint": dict(checkpoint),
        "rows": _normalized_common_rows(rows, arm, checkpoint),
    }


def _read_common_cache(
    comparison_root: Path,
    arm: str,
    *,
    config_sha256: str,
    run_config_sha256: str,
    scenario_sha256: str,
    checkpoint: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    name = _COMMON_CACHE_NAMES[arm]
    try:
        path = resolve_direct_regular_file(
            comparison_root,
            comparison_root / name,
            label=f"{arm} common cache",
        )
        payload = read_json_object(path)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"{arm} common cache is invalid") from error
    if set(payload) != _COMMON_CACHE_KEYS:
        raise ValueError(f"{arm} common cache schema is invalid")
    if (
        payload["schema_version"] != 1
        or payload["arm"] != arm
        or payload["config_sha256"] != config_sha256
        or payload["run_config_sha256"] != run_config_sha256
        or payload["scenario_sha256"] != scenario_sha256
        or payload["checkpoint"] != dict(checkpoint)
        or not isinstance(payload["rows"], list)
    ):
        raise ValueError(f"{arm} common cache identity is invalid")
    _sha256_text(payload["config_sha256"], "common cache config hash")
    _sha256_text(payload["run_config_sha256"], "common cache run config hash")
    _sha256_text(payload["scenario_sha256"], "common cache scenario hash")
    return payload, _normalized_common_rows(
        payload["rows"], arm, checkpoint
    )


def _read_combined_common_rows(
    comparison_root: Path,
    checkpoints: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    try:
        path = resolve_direct_regular_file(
            comparison_root,
            comparison_root / _COMMON_COMBINED_NAME,
            label="combined common evaluation",
        )
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != EVALUATION_COLUMNS:
                raise ValueError("combined common CSV header is invalid")
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ValueError("combined common CSV is invalid") from error
    expected_order = [
        (arm, seed) for arm in ARMS for seed in EXPECTED_HOLDOUT_SEEDS
    ]
    try:
        observed_order = [(row["arm"], int(row["seed"])) for row in rows]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("combined common CSV order is invalid") from error
    if observed_order != expected_order:
        raise ValueError("combined common CSV order is invalid")
    normalized: list[dict[str, Any]] = []
    for arm in ARMS:
        normalized.extend(
            _normalized_common_rows(
                [row for row in rows if row.get("arm") == arm],
                arm,
                checkpoints[arm],
            )
        )
    return normalized


def validate_common_step_stage(
    root: str | Path,
    *,
    expected_config_sha256: str | None = None,
    expected_run_config_sha256: Mapping[str, str] | None = None,
    expected_scenario_sha256: str | None = None,
    archive_timestep_reader: Callable[[Path], int | None] | None = None,
) -> dict[str, Any]:
    """Validate only artifacts and manifest refs owned by common evaluation."""
    root_path = Path(root)
    root_is_junction = getattr(root_path, "is_junction", None)
    if root_path.is_symlink() or (
        root_is_junction is not None and root_is_junction()
    ):
        raise ValueError("comparison root must be a regular directory")
    try:
        base = root_path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("comparison root is invalid") from error
    if not base.is_dir():
        raise ValueError("comparison root must be a regular directory")
    comparison_root = resolve_direct_directory(
        base, base / "comparison", label="common comparison directory"
    )
    arm_roots = {
        arm: resolve_direct_directory(
            base, base / arm, label=f"{arm} comparison directory"
        )
        for arm in ARMS
    }
    marker_path = resolve_direct_regular_file(
        comparison_root,
        comparison_root / _COMMON_MARKER_NAME,
        label="common stage marker",
    )
    try:
        marker = read_json_object(marker_path)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ValueError("common stage marker is invalid") from error
    if set(marker) != _COMMON_MARKER_KEYS or marker["schema_version"] != 1:
        raise ValueError("common stage marker schema is invalid")
    config_sha = _sha256_text(marker["config_sha256"], "common config hash")
    scenario_sha = _sha256_text(marker["scenario_sha256"], "common scenario hash")
    if expected_config_sha256 is not None and config_sha != expected_config_sha256:
        raise ValueError("common config hash mismatch")
    if expected_scenario_sha256 is not None and scenario_sha != expected_scenario_sha256:
        raise ValueError("common scenario hash mismatch")
    run_config_sha256 = marker["run_config_sha256"]
    if not isinstance(run_config_sha256, Mapping) or set(
        run_config_sha256
    ) != set(ARMS):
        raise ValueError("common run config hash schema is invalid")
    normalized_run_config_sha256 = {
        arm: _sha256_text(
            run_config_sha256[arm], f"{arm} common run config hash"
        )
        for arm in ARMS
    }
    if expected_run_config_sha256 is not None:
        if not isinstance(expected_run_config_sha256, Mapping) or set(
            expected_run_config_sha256
        ) != set(ARMS):
            raise ValueError("expected common run config hash schema is invalid")
        if normalized_run_config_sha256 != dict(expected_run_config_sha256):
            raise ValueError("common run config hash mismatch")
    common_timestep = _exact_nonnegative_int(
        marker["common_timestep"], "common timestep"
    )
    if marker["evaluation_seed_count_per_arm"] != len(EXPECTED_HOLDOUT_SEEDS):
        raise ValueError("common evaluation seed count is invalid")
    checkpoints = marker["checkpoints"]
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != set(ARMS):
        raise ValueError("common checkpoint schema is invalid")
    normalized_checkpoints = {
        arm: _validate_common_reference(checkpoints[arm], arm) for arm in ARMS
    }
    if {ref["timestep"] for ref in normalized_checkpoints.values()} != {
        common_timestep
    }:
        raise ValueError("common checkpoint timesteps differ")
    expected_artifacts = {
        *(_COMMON_CACHE_NAMES[arm] for arm in ARMS),
        _COMMON_COMBINED_NAME,
    }
    artifacts = marker["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_artifacts:
        raise ValueError("common artifact schema is invalid")
    for name in expected_artifacts:
        expected_digest = _sha256_text(
            artifacts[name], f"{name} common artifact hash"
        )
        try:
            artifact = resolve_direct_regular_file(
                comparison_root,
                comparison_root / name,
                label=f"common artifact {name}",
            )
        except (OSError, ValueError) as error:
            raise ValueError("common artifact is not direct regular") from error
        if sha256_file(artifact) != expected_digest:
            raise ValueError("common artifact hash mismatch")
    cache_rows: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        _payload, cache_rows[arm] = _read_common_cache(
            comparison_root,
            arm,
            config_sha256=config_sha,
            run_config_sha256=normalized_run_config_sha256[arm],
            scenario_sha256=scenario_sha,
            checkpoint=normalized_checkpoints[arm],
        )
    combined_rows = _read_combined_common_rows(
        comparison_root, normalized_checkpoints
    )
    expected_rows = [row for arm in ARMS for row in cache_rows[arm]]
    if combined_rows != expected_rows:
        raise ValueError("combined common CSV differs from per-arm caches")
    try:
        manifest_path = resolve_direct_regular_file(
            base, base / "manifest.json", label="root manifest"
        )
        manifest = read_json_object(manifest_path)
        manifest_checkpoints = manifest["checkpoints"]
    except (OSError, UnicodeDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("common checkpoint manifest is invalid") from error
    if manifest.get("config_sha256") != config_sha:
        raise ValueError("common config differs from root manifest")
    if manifest.get("scenario_sha256") != scenario_sha:
        raise ValueError("common scenario differs from root manifest")
    for arm in ARMS:
        try:
            current_common = manifest_checkpoints[arm]["common"]
        except (KeyError, TypeError) as error:
            raise ValueError("common checkpoint manifest is invalid") from error
        if current_common != normalized_checkpoints[arm]:
            raise ValueError("common checkpoint manifest mismatch")
        reference = CheckpointRef(
            base / normalized_checkpoints[arm]["path"],
            "common_step",
            normalized_checkpoints[arm]["timestep"],
            normalized_checkpoints[arm]["sha256"],
        )
        try:
            checkpoint_path = _reference_path_within_arm(
                reference, arm_roots[arm]
            )
        except PartialResultError as error:
            raise ValueError("common checkpoint is not direct regular") from error
        if sha256_file(checkpoint_path) != reference.sha256:
            raise ValueError("common checkpoint hash mismatch")
        if archive_timestep_reader is not None:
            try:
                stored_timestep = archive_timestep_reader(checkpoint_path)
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                raise ValueError("common checkpoint is unreadable") from error
            if stored_timestep != reference.timestep:
                raise ValueError("common checkpoint stored timestep mismatch")
    return marker


def _pretty_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _unlink_if_exact(path: Path, expected: bytes) -> None:
    try:
        if path.read_bytes() == expected:
            path.unlink()
    except (FileNotFoundError, OSError):
        return


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


def evaluate_arm_artifacts(
    root: str | Path,
    arm: str,
    scenarios: Sequence[dict],
    run_config: Mapping[str, Any],
    *,
    config_sha256: str,
    scenario_sha256: str,
    model_loader=MaskablePPO.load,
) -> dict[str, Any]:
    """Evaluate one selected checkpoint and commit its artifacts marker-last."""
    if arm not in ARMS:
        raise ValueError("unknown comparison arm")
    _sha256_text(config_sha256, "evaluation config hash")
    _sha256_text(scenario_sha256, "evaluation scenario hash")
    try:
        base = Path(root).resolve(strict=True)
        arm_root = (base / arm).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise PartialResultError("comparison root or arm is absent") from error
    if arm_root.parent != base or arm_root.name != arm:
        raise PartialResultError("comparison arm directory escapes root")
    marker_path = arm_root / "evaluation_stage.json"
    archive_timestep_reader = lambda path: _archive_timestep(path, model_loader)
    try:
        return validate_arm_evaluation_stage(
            base,
            arm,
            expected_config_sha256=config_sha256,
            expected_scenario_sha256=scenario_sha256,
            archive_timestep_reader=archive_timestep_reader,
        )
    except (OSError, ValueError):
        pass
    if marker_path.exists() or marker_path.is_symlink():
        marker_path.unlink()
    try:
        manifest_path = resolve_direct_regular_file(
            base, base / "manifest.json", label="root manifest"
        )
        manifest = read_json_object(manifest_path)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise PartialResultError("root manifest.json must already be valid") from error
    if manifest.get("config_sha256") != config_sha256:
        raise PartialResultError("root manifest config hash mismatch")
    if manifest.get("scenario_sha256") != scenario_sha256:
        raise PartialResultError("root manifest scenario hash mismatch")

    final = resolve_final_checkpoint(arm_root, model_loader=model_loader)
    decision = resolve_selection_decision(
        arm_root,
        model_loader=model_loader,
        final_reference=final,
    )
    selected = decision.reference
    _reference_path_within_arm(selected, arm_root)
    _reference_path_within_arm(final, arm_root)
    rows = evaluate_checkpoint(
        selected.path,
        run_config,
        scenarios,
        selected.label,
        arm,
        model_loader,
    )
    try:
        validated_rows = _validated_rows(rows, arm=arm)
    except (KeyError, TypeError, ValueError) as error:
        raise PartialResultError("arm evaluation rows are invalid") from error
    provenance = {
        (
            row["checkpoint"],
            int(row["checkpoint_timestep"]),
            row["checkpoint_sha256"],
        )
        for row in validated_rows
    }
    if provenance != {(selected.label, selected.timestep, selected.sha256)}:
        raise PartialResultError(
            "evaluation rows do not match the resolved selected checkpoint"
        )
    _stable_checkpoint_reference(selected, arm_root, model_loader=model_loader)
    _stable_checkpoint_reference(final, arm_root, model_loader=model_loader)

    all_path, primary_path = write_arm_evaluations(base, arm, validated_rows)
    relative_refs = {
        "selected": _root_relative_reference(base, arm_root, selected),
        "final": _root_relative_reference(base, arm_root, final),
    }
    updated_manifest = update_checkpoint_manifest(
        manifest_path, arm, relative_refs
    )
    marker = {
        "schema_version": 1,
        "arm": arm,
        "config_sha256": config_sha256,
        "scenario_sha256": scenario_sha256,
        "checkpoints": {
            key: updated_manifest["checkpoints"][arm][key]
            for key in ("selected", "final")
        },
        "artifacts": {
            "evaluation_scenarios.csv": sha256_file(all_path),
            "evaluation_primary_test.csv": sha256_file(primary_path),
            "training_completion.json": sha256_file(
                arm_root / "training_completion.json"
            ),
            "runtime_metrics.json": sha256_file(
                arm_root / "runtime_metrics.json"
            ),
        },
        "evaluation_seed_count": len(EXPECTED_HOLDOUT_SEEDS),
        "primary_test_seed_count": len(PRIMARY_TEST_SEEDS),
        "selection_outcome": decision.selection_outcome,
        "fallback_reason": decision.fallback_reason,
    }
    expected_marker_bytes = _pretty_json_bytes(marker)
    try:
        atomic_write_json(marker_path, marker)
        validated_marker = validate_arm_evaluation_stage(
            base,
            arm,
            expected_config_sha256=config_sha256,
            expected_scenario_sha256=scenario_sha256,
            archive_timestep_reader=archive_timestep_reader,
        )
    except BaseException:
        _unlink_if_exact(marker_path, expected_marker_bytes)
        raise
    return validated_marker


def evaluate_common_step_artifacts(
    root: str | Path,
    scenarios: Sequence[dict],
    arm_configs: Mapping[str, Mapping[str, Any]],
    *,
    config_sha256: str,
    scenario_sha256: str,
    regular_interval: int = REGULAR_INTERVAL,
    model_loader=MaskablePPO.load,
) -> dict[str, Any]:
    """Evaluate only the shared regular checkpoint and commit marker-last.

    Each arm cache is its own durable commit.  A restart can therefore reuse a
    completed arm after the other arm crashed, while the combined CSV and stage
    marker are published only after both exact caches have been validated.
    """
    _sha256_text(config_sha256, "common evaluation config hash")
    _sha256_text(scenario_sha256, "common evaluation scenario hash")
    if regular_interval <= 0:
        raise ValueError("regular_interval must be positive")
    if not isinstance(arm_configs, Mapping) or set(arm_configs) != set(ARMS):
        raise ValueError("common evaluation requires exact arm configs")
    if any(not isinstance(arm_configs[arm], Mapping) for arm in ARMS):
        raise ValueError("common evaluation arm config must be a mapping")
    run_config_sha256 = {
        arm: canonical_json_sha256(arm_configs[arm]) for arm in ARMS
    }
    try:
        split_holdout_records(scenarios)
    except (KeyError, TypeError, ValueError) as error:
        raise PartialResultError(
            "common evaluation requires the exact fixed holdout scenarios"
        ) from error

    root_path = Path(root)
    root_is_junction = getattr(root_path, "is_junction", None)
    if root_path.is_symlink() or (
        root_is_junction is not None and root_is_junction()
    ):
        raise PartialResultError("comparison root must be a regular directory")
    try:
        base = root_path.resolve(strict=True)
        if not base.is_dir():
            raise ValueError("comparison root is not a directory")
        arm_roots = {
            arm: resolve_direct_directory(
                base, base / arm, label=f"{arm} comparison directory"
            )
            for arm in ARMS
        }
        comparison_candidate = base / "comparison"
        comparison_is_junction = getattr(
            comparison_candidate, "is_junction", None
        )
        if comparison_candidate.is_symlink() or (
            comparison_is_junction is not None and comparison_is_junction()
        ):
            raise ValueError("comparison artifact directory is linked")
        if not comparison_candidate.exists():
            comparison_candidate.mkdir()
        comparison_root = resolve_direct_directory(
            base,
            comparison_candidate,
            label="common comparison directory",
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise PartialResultError("comparison directory layout is invalid") from error

    archive_timestep_reader = lambda path: _archive_timestep(path, model_loader)
    try:
        return validate_common_step_stage(
            base,
            expected_config_sha256=config_sha256,
            expected_run_config_sha256=run_config_sha256,
            expected_scenario_sha256=scenario_sha256,
            archive_timestep_reader=archive_timestep_reader,
        )
    except (OSError, ValueError):
        pass

    marker_path = comparison_root / _COMMON_MARKER_NAME
    try:
        if marker_path.exists() or marker_path.is_symlink():
            marker_path.unlink()
    except OSError as error:
        raise PartialResultError("invalid common stage marker cannot be removed") from error

    manifest_path = base / "manifest.json"
    try:
        manifest_file = resolve_direct_regular_file(
            base, manifest_path, label="root manifest"
        )
        manifest = read_json_object(manifest_file)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise PartialResultError("root manifest.json must already be valid") from error
    if manifest.get("config_sha256") != config_sha256:
        raise PartialResultError("root manifest config hash mismatch")
    if manifest.get("scenario_sha256") != scenario_sha256:
        raise PartialResultError("root manifest scenario hash mismatch")

    inventories = {
        arm: readable_checkpoint_inventory(
            arm_roots[arm],
            regular_interval,
            model_loader=model_loader,
        )
        for arm in ARMS
    }
    shared_steps = set(inventories[ARMS[0]]) & set(inventories[ARMS[1]])
    if not shared_steps:
        raise PartialResultError("no common readable regular checkpoint timestep")
    common_step = max(shared_steps)

    absolute_refs: dict[str, CheckpointRef] = {}
    relative_refs: dict[str, CheckpointRef] = {}
    checkpoint_payloads: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        reference = _verified_ref(
            inventories[arm][common_step],
            "common_step",
            common_step,
            model_loader,
        )
        if reference is None:
            raise PartialResultError("common checkpoint became unreadable")
        _reference_path_within_arm(reference, arm_roots[arm])
        relative = _root_relative_reference(base, arm_roots[arm], reference)
        absolute_refs[arm] = reference
        relative_refs[arm] = relative
        checkpoint_payloads[arm] = _reference_payload(relative)

    cached_rows: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        try:
            _payload, rows = _read_common_cache(
                comparison_root,
                arm,
                config_sha256=config_sha256,
                run_config_sha256=run_config_sha256[arm],
                scenario_sha256=scenario_sha256,
                checkpoint=checkpoint_payloads[arm],
            )
        except (KeyError, OSError, TypeError, ValueError):
            rows = evaluate_checkpoint(
                absolute_refs[arm].path,
                arm_configs[arm],
                scenarios,
                "common_step",
                arm,
                model_loader,
            )
            _stable_checkpoint_reference(
                absolute_refs[arm],
                arm_roots[arm],
                model_loader=model_loader,
            )
            cache = _common_cache_payload(
                arm,
                config_sha256=config_sha256,
                run_config_sha256=run_config_sha256[arm],
                scenario_sha256=scenario_sha256,
                checkpoint=checkpoint_payloads[arm],
                rows=rows,
            )
            atomic_write_json(
                comparison_root / _COMMON_CACHE_NAMES[arm], cache
            )
            _payload, rows = _read_common_cache(
                comparison_root,
                arm,
                config_sha256=config_sha256,
                run_config_sha256=run_config_sha256[arm],
                scenario_sha256=scenario_sha256,
                checkpoint=checkpoint_payloads[arm],
            )
        cached_rows[arm] = rows

    for arm in ARMS:
        _stable_checkpoint_reference(
            absolute_refs[arm],
            arm_roots[arm],
            model_loader=model_loader,
        )

    combined_rows = [row for arm in ARMS for row in cached_rows[arm]]
    combined_path = write_common_step_evaluation(base, combined_rows)

    try:
        current_manifest = read_json_object(
            resolve_direct_regular_file(
                base, manifest_path, label="root manifest"
            )
        )
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise PartialResultError("root manifest.json became invalid") from error
    if current_manifest.get("config_sha256") != config_sha256:
        raise PartialResultError("root manifest config hash changed")
    if current_manifest.get("scenario_sha256") != scenario_sha256:
        raise PartialResultError("root manifest scenario hash changed")
    try:
        updated_manifest = current_manifest
        for arm in ARMS:
            updated_manifest = merge_checkpoint_manifest(
                updated_manifest, arm, {"common": relative_refs[arm]}
            )
        atomic_write_json(manifest_path, updated_manifest)
    except (OSError, TypeError, ValueError) as error:
        raise PartialResultError("common checkpoint manifest publication failed") from error

    marker = {
        "schema_version": 1,
        "config_sha256": config_sha256,
        "run_config_sha256": run_config_sha256,
        "scenario_sha256": scenario_sha256,
        "common_timestep": common_step,
        "checkpoints": {
            arm: checkpoint_payloads[arm] for arm in ARMS
        },
        "artifacts": {
            _COMMON_CACHE_NAMES[arm]: sha256_file(
                comparison_root / _COMMON_CACHE_NAMES[arm]
            )
            for arm in ARMS
        }
        | {_COMMON_COMBINED_NAME: sha256_file(combined_path)},
        "evaluation_seed_count_per_arm": len(EXPECTED_HOLDOUT_SEEDS),
    }
    expected_marker_bytes = _pretty_json_bytes(marker)
    try:
        atomic_write_json(marker_path, marker)
        validated_marker = validate_common_step_stage(
            base,
            expected_config_sha256=config_sha256,
            expected_run_config_sha256=run_config_sha256,
            expected_scenario_sha256=scenario_sha256,
            archive_timestep_reader=archive_timestep_reader,
        )
    except BaseException:
        _unlink_if_exact(marker_path, expected_marker_bytes)
        raise
    return validated_marker
