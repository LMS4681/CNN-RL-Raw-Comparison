from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from baseline_policies import GreedyImmediateAreaPolicy, RandomValidPolicy
from pretraining.targets import AuxiliaryTargets, build_auxiliary_targets


DATASET_SCHEMA_VERSION = 1
OBSERVATION_SCHEMA_VERSION = 4
DEFAULT_WORKSPACE_CODES = (
    "PE049",
    "PE050",
    "PE055",
    "PE054",
    "PE056",
    "PE048",
    "PE044",
    "PE059",
    "PE060",
    "PE061",
)
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _seed_values(bounds: tuple[int, int]) -> tuple[int, ...]:
    start, end = bounds
    return tuple(range(start, end + 1))


@dataclass(frozen=True)
class PretrainingDataConfig:
    train_state_count: int = 5_000
    validation_state_count: int = 1_000
    train_episode_seeds: tuple[int, int] = (20_000, 20_039)
    validation_episode_seeds: tuple[int, int] = (30_000, 30_009)
    states_per_shard: int = 100
    replay_every_n_states: int = 4
    replay_resolved_blocks: int = 8
    replay_max_decisions: int = 32
    source_manifest_sha256: str | None = None
    data_dir: str = "data"
    split_manifest_path: str = "data/data_split_manifest.json"
    active_workspace_codes: tuple[str, ...] = DEFAULT_WORKSPACE_CODES
    episode_n_blocks: int = 913
    grid_size: int = 64
    monthly_jitter: int = 20
    empirical_profile_probability: float = 0.2

    def __post_init__(self) -> None:
        for name in (
            "train_state_count",
            "validation_state_count",
            "states_per_shard",
            "replay_every_n_states",
            "replay_resolved_blocks",
            "replay_max_decisions",
            "episode_n_blocks",
            "grid_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("train_episode_seeds", "validation_episode_seeds"):
            bounds = getattr(self, name)
            if len(bounds) != 2 or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in bounds
            ):
                raise ValueError(f"{name} must be an inclusive integer pair")
            if bounds[0] > bounds[1]:
                raise ValueError(f"{name} start must not exceed end")
        if set(_seed_values(self.train_episode_seeds)) & set(
            _seed_values(self.validation_episode_seeds)
        ):
            raise ValueError("training and validation episode seeds must be disjoint")
        if self.replay_resolved_blocks != 8:
            raise ValueError("replay_resolved_blocks must be exactly 8")
        if self.replay_max_decisions != 32:
            raise ValueError("replay_max_decisions must be exactly 32")
        if len(self.active_workspace_codes) != 10:
            raise ValueError("active_workspace_codes must contain exactly 10 codes")
        if self.source_manifest_sha256 is not None and (
            len(self.source_manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.source_manifest_sha256.lower()
            )
        ):
            raise ValueError("source_manifest_sha256 must be a SHA256 hex digest")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> PretrainingDataConfig:
        known = {field.name for field in fields(cls)}
        selected = {key: value for key, value in values.items() if key in known}
        for name in (
            "train_episode_seeds",
            "validation_episode_seeds",
            "active_workspace_codes",
        ):
            if name in selected:
                selected[name] = tuple(selected[name])
        return cls(**selected)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(values: Mapping[str, object]) -> bytes:
    return json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _config_dict(config: PretrainingDataConfig) -> dict[str, object]:
    values = asdict(config)
    for key in (
        "train_episode_seeds",
        "validation_episode_seeds",
        "active_workspace_codes",
    ):
        values[key] = list(values[key])
    return values


def _config_sha256(config: PretrainingDataConfig) -> str:
    return hashlib.sha256(_canonical_json(_config_dict(config))).hexdigest()


def _atomic_json(path: Path, values: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="\n") as destination:
        json.dump(values, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)


def _atomic_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with source.open("rb") as input_file, temporary.open("wb") as output_file:
        shutil.copyfileobj(input_file, output_file)
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary, destination)


def _resolve_input_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    working_directory_path = Path.cwd() / path
    if working_directory_path.exists():
        return working_directory_path
    return PACKAGE_ROOT / path


def _resolved_config(config: PretrainingDataConfig) -> PretrainingDataConfig:
    if config.source_manifest_sha256 is not None:
        return config
    manifest_path = _resolve_input_path(config.split_manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"split manifest not found: {manifest_path}")
    return replace(config, source_manifest_sha256=_sha256_file(manifest_path))


def _environment_factory(
    config: PretrainingDataConfig,
) -> Callable[[int], object]:
    from alloc_env.alloc_env import BlockPlacementEnv, DROPOUT_THRESHOLD
    from alloc_env.block_generator import SyntheticBlockGenerator
    from alloc_env.observation_state import build_observation_scales
    from alloc_env.strategy import BaseGridStrategy
    from train import load_allocation_scenario

    data_dir = _resolve_input_path(config.data_dir)
    manifest_path = _resolve_input_path(config.split_manifest_path)
    split_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _sha256_file(manifest_path) != config.source_manifest_sha256:
        raise ValueError("source split manifest SHA256 mismatch")

    strategy = BaseGridStrategy(step=5.0)
    full_blocks, workspaces = load_allocation_scenario(
        data_dir, strategy, list(config.active_workspace_codes)
    )
    training_ships = set(split_manifest["training_ship_nos"])
    training_blocks = [
        block.clone()
        for block in full_blocks
        if block.ship_no in training_ships
    ]
    if not training_blocks:
        raise ValueError("source split contains no training blocks")
    scales = build_observation_scales(
        full_blocks, workspaces, DROPOUT_THRESHOLD
    )
    target_month_counts = Counter(
        (block.in_date.year, block.in_date.month) for block in full_blocks
    )

    def create(seed: int) -> BlockPlacementEnv:
        generator = SyntheticBlockGenerator.from_blocks(
            training_blocks,
            seed=seed,
            monthly_jitter=config.monthly_jitter,
            empirical_profile_probability=(
                config.empirical_profile_probability
            ),
            target_month_counts=target_month_counts,
        )
        env = BlockPlacementEnv(
            training_blocks,
            workspaces,
            strategy,
            use_synthetic=True,
            generator=generator,
            synthetic_n_blocks=config.episode_n_blocks,
            vary_layout=False,
            grid_size=config.grid_size,
            state_context_mode="full",
            observation_scales=scales,
        )
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.reset(seed=seed)
        return env

    return create


def _stack_buffer(
    observations: list[dict[str, np.ndarray]],
    targets: list[AuxiliaryTargets],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key in observations[0]:
        values = np.stack([observation[key] for observation in observations])
        arrays[f"obs__{key}"] = values.astype(
            np.float16 if key == "grids" else np.float32,
            copy=False,
        )
    for field in fields(AuxiliaryTargets):
        values = np.stack([getattr(target, field.name) for target in targets])
        arrays[f"target__{field.name}"] = values.astype(
            np.bool_ if field.name in {"action_mask", "replay_mask"} else np.float32,
            copy=False,
        )
    return arrays


def _write_shard(
    root: Path,
    split_name: str,
    shard_index: int,
    start_index: int,
    observations: list[dict[str, np.ndarray]],
    targets: list[AuxiliaryTargets],
) -> dict[str, object]:
    relative = Path(split_name) / f"shard-{shard_index:05d}.npz"
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    arrays = _stack_buffer(observations, targets)
    with temporary.open("wb") as output:
        np.savez_compressed(output, **arrays)
        output.flush()
        os.fsync(output.fileno())
    digest = _sha256_file(temporary)
    os.replace(temporary, destination)
    return {
        "path": relative.as_posix(),
        "sha256": digest,
        "state_count": len(observations),
        "start_index": start_index,
    }


def _episode_quotas(state_count: int, seeds: tuple[int, ...]) -> list[int]:
    base, remainder = divmod(state_count, len(seeds))
    return [base + int(index < remainder) for index in range(len(seeds))]


def _collect_split(
    root: Path,
    split_name: str,
    state_count: int,
    seeds: tuple[int, ...],
    config: PretrainingDataConfig,
    create_environment: Callable[[int], object],
    existing_shards: list[dict[str, object]] | None = None,
    on_shard_complete: Callable[
        [str, list[dict[str, object]]], None
    ] | None = None,
) -> dict[str, object]:
    shards = [dict(entry) for entry in (existing_shards or [])]
    episodes: list[dict[str, object]] = []
    observations: list[dict[str, np.ndarray]] = []
    targets: list[AuxiliaryTargets] = []
    state_index = 0
    completed_states = sum(int(entry["state_count"]) for entry in shards)
    shard_start = completed_states

    def flush() -> None:
        nonlocal shard_start
        if not observations:
            return
        shards.append(
            _write_shard(
                root,
                split_name,
                len(shards),
                shard_start,
                observations,
                targets,
            )
        )
        shard_start += len(observations)
        observations.clear()
        targets.clear()
        if on_shard_complete is not None:
            on_shard_complete(split_name, shards)

    for episode_index, (seed, quota) in enumerate(
        zip(seeds, _episode_quotas(state_count, seeds))
    ):
        if quota == 0:
            continue
        env = create_environment(seed)
        policy = (
            RandomValidPolicy(seed)
            if episode_index % 2 == 0
            else GreedyImmediateAreaPolicy()
        )
        collected = 0
        terminated = False
        while collected < quota:
            if terminated:
                raise RuntimeError(
                    f"episode seed {seed} ended before its state quota"
                )
            current_observation = env._get_obs()
            observation = {
                key: value.copy()
                for key, value in current_observation.items()
            }
            if state_index >= completed_states:
                include_replay = (
                    state_index % config.replay_every_n_states == 0
                )
                target = build_auxiliary_targets(
                    env, include_replay=include_replay
                )
                observations.append(observation)
                targets.append(target)
            collected += 1
            state_index += 1
            action = policy.select_action(env, observation)
            _, _, terminated, _, _ = env.step(action)
            if len(observations) == config.states_per_shard:
                flush()
        episodes.append({
            "seed": seed,
            "collector_policy": policy.name,
            "state_count": collected,
        })
        close = getattr(env, "close", None)
        if close is not None:
            close()
    flush()
    if state_index != state_count:
        raise RuntimeError(
            f"collected {state_index} {split_name} states, expected {state_count}"
        )
    return {
        "state_count": state_count,
        "episode_seeds": list(seeds),
        "episodes": episodes,
        "shards": shards,
    }


def read_dataset_manifest(path: str | Path) -> dict[str, object]:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if values.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported pretraining dataset schema version")
    return values


def load_pretraining_shard(
    root: str | Path,
    manifest: Mapping[str, object],
    entry: Mapping[str, object],
) -> dict[str, dict[str, np.ndarray]]:
    if manifest.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported pretraining dataset schema version")
    root_path = Path(root).resolve()
    path = (root_path / str(entry["path"])).resolve()
    if root_path not in path.parents:
        raise ValueError("dataset shard path escapes dataset root")
    actual = _sha256_file(path)
    if actual != entry["sha256"]:
        raise ValueError(
            f"dataset shard SHA256 mismatch for {entry['path']}"
        )
    observations: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as stored:
        for key in stored.files:
            if key.startswith("obs__"):
                observations[key[5:]] = stored[key].astype(np.float32)
            elif key.startswith("target__"):
                name = key[8:]
                targets[name] = stored[key].astype(
                    np.bool_
                    if name in {"action_mask", "replay_mask"}
                    else np.float32
                )
            else:
                raise ValueError(f"unexpected dataset array {key}")
    return {"observations": observations, "targets": targets}


def _verify_existing_dataset(
    root: Path,
    manifest: Mapping[str, object],
    config_sha256: str,
) -> bool:
    if manifest.get("config_sha256") != config_sha256:
        return False
    for split in manifest["splits"].values():
        for entry in split["shards"]:
            load_pretraining_shard(root, manifest, entry)
    return True


def _verified_progress_shards(
    root: Path,
    progress: Mapping[str, object],
    config_sha256: str,
    split_name: str,
    state_count: int,
    states_per_shard: int,
) -> list[dict[str, object]]:
    if progress.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported dataset progress schema version")
    if progress.get("config_sha256") != config_sha256:
        raise ValueError("dataset progress configuration mismatch")
    splits = progress.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("dataset progress is missing splits")
    entries = splits.get(split_name, [])
    if not isinstance(entries, list):
        raise ValueError("dataset progress split must be a list")
    verified: list[dict[str, object]] = []
    expected_start = 0
    minimal_manifest = {"dataset_schema_version": DATASET_SCHEMA_VERSION}
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, Mapping):
            raise ValueError("dataset progress shard must be an object")
        entry = dict(raw_entry)
        count = int(entry.get("state_count", 0))
        if count < 1 or count > states_per_shard:
            raise ValueError("dataset progress shard state_count is invalid")
        if int(entry.get("start_index", -1)) != expected_start:
            raise ValueError("dataset progress shard indices are not contiguous")
        expected_path = f"{split_name}/shard-{index:05d}.npz"
        if entry.get("path") != expected_path:
            raise ValueError("dataset progress shard path is not canonical")
        load_pretraining_shard(root, minimal_manifest, entry)
        verified.append(entry)
        expected_start += count
    if expected_start > state_count:
        raise ValueError("dataset progress exceeds configured state count")
    if (
        expected_start < state_count
        and verified
        and int(verified[-1]["state_count"]) != states_per_shard
    ):
        raise ValueError("only a final dataset shard may be partial")
    return verified


def collect_pretraining_dataset(
    config: PretrainingDataConfig,
    output_dir: Path,
    *,
    progress_mirror_dir: Path | None = None,
) -> Path:
    config = _resolved_config(config)
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "dataset_manifest.json"
    progress_path = root / "dataset_progress.json"
    mirror_root = (
        None
        if progress_mirror_dir is None
        else Path(progress_mirror_dir).resolve()
    )
    if mirror_root == root:
        raise ValueError("dataset progress mirror must differ from output_dir")
    if mirror_root is not None:
        mirror_root.mkdir(parents=True, exist_ok=True)
    config_sha256 = _config_sha256(config)
    if manifest_path.is_file():
        existing = read_dataset_manifest(manifest_path)
        if _verify_existing_dataset(root, existing, config_sha256):
            if mirror_root is not None:
                for split in existing["splits"].values():
                    for entry in split["shards"]:
                        relative = Path(str(entry["path"]))
                        _atomic_copy_file(
                            root / relative, mirror_root / relative
                        )
                _atomic_copy_file(
                    manifest_path, mirror_root / manifest_path.name
                )
                (mirror_root / progress_path.name).unlink(missing_ok=True)
            return manifest_path
        raise ValueError("existing dataset manifest does not match configuration")

    progress: dict[str, object] = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "config_sha256": config_sha256,
        "splits": {"train": [], "validation": []},
    }
    if progress_path.is_file():
        loaded_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_progress, dict):
            raise ValueError("dataset progress must contain a JSON object")
        progress = loaded_progress

    existing = {
        split_name: _verified_progress_shards(
            root,
            progress,
            config_sha256,
            split_name,
            state_count,
            config.states_per_shard,
        )
        for split_name, state_count in (
            ("train", config.train_state_count),
            ("validation", config.validation_state_count),
        )
    }

    def record_progress(
        split_name: str, shards: list[dict[str, object]]
    ) -> None:
        progress_splits = progress["splits"]
        if not isinstance(progress_splits, dict):
            raise ValueError("dataset progress splits must be mutable")
        progress_splits[split_name] = [dict(entry) for entry in shards]
        _atomic_json(progress_path, progress)
        if mirror_root is not None:
            latest = shards[-1]
            relative = Path(str(latest["path"]))
            _atomic_copy_file(root / relative, mirror_root / relative)
            _atomic_copy_file(
                progress_path, mirror_root / progress_path.name
            )

    create_environment = _environment_factory(config)
    train_seeds = _seed_values(config.train_episode_seeds)
    validation_seeds = _seed_values(config.validation_episode_seeds)
    splits = {}
    splits["train"] = _collect_split(
        root,
        "train",
        config.train_state_count,
        train_seeds,
        config,
        create_environment,
        existing["train"],
        record_progress,
    )
    splits["validation"] = _collect_split(
        root,
        "validation",
        config.validation_state_count,
        validation_seeds,
        config,
        create_environment,
        existing["validation"],
        record_progress,
    )
    manifest = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "source_manifest_sha256": config.source_manifest_sha256,
        "config_sha256": config_sha256,
        "config": _config_dict(config),
        "target_normalizers": {
            "future_optionality": "16 * 10",
            "replay_delay": "8 * dropout_threshold",
            "geometry": "grid_cell_count",
        },
        "splits": splits,
    }
    _atomic_json(manifest_path, manifest)
    if mirror_root is not None:
        _atomic_copy_file(manifest_path, mirror_root / manifest_path.name)
        (mirror_root / progress_path.name).unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate simulator-supervised pretraining shards"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-mirror-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    values = json.loads(args.config.read_text(encoding="utf-8"))
    path = collect_pretraining_dataset(
        PretrainingDataConfig.from_mapping(values),
        args.output_dir,
        progress_mirror_dir=args.progress_mirror_dir,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
