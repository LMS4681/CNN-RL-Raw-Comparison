from __future__ import annotations

import shutil
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

import pretraining.dataset as dataset_module
from alloc_env.alloc_env import BlockPlacementEnv
from alloc_env.block import Block
from alloc_env.strategy import BaseGridStrategy
from alloc_env.workspace import Workspace
from pretraining.dataset import (
    PretrainingDataConfig,
    collect_pretraining_dataset,
    load_pretraining_shard,
    read_dataset_manifest,
)


def tiny_environment(seed: int) -> BlockPlacementEnv:
    strategy = BaseGridStrategy(step=1.0)
    blocks = [
        Block(
            name=f"B-{index}",
            ship_no="S-1",
            block_type="BUILD",
            length=4.0,
            breadth=2.0,
            height=1.0,
            weight=1.0,
            in_date=date(2026, 1, 5) + timedelta(days=index),
            out_date=date(2026, 3, 1),
        )
        for index in range(8)
    ]
    workspaces = [
        Workspace(
            code=f"WS-{index}",
            origin_x=0.0,
            origin_y=0.0,
            length=30.0,
            breadth=20.0,
            strategy=strategy,
        )
        for index in range(10)
    ]
    env = BlockPlacementEnv(
        blocks, workspaces, strategy, grid_size=16
    )
    env.reset(seed=seed)
    return env


def tiny_config(**overrides) -> PretrainingDataConfig:
    values = {
        "train_state_count": 8,
        "validation_state_count": 4,
        "train_episode_seeds": (10, 11),
        "validation_episode_seeds": (20, 21),
        "states_per_shard": 3,
        "replay_every_n_states": 4,
        "source_manifest_sha256": "a" * 64,
    }
    values.update(overrides)
    return PretrainingDataConfig(**values)


def test_config_rejects_overlapping_train_and_validation_seeds():
    with pytest.raises(ValueError, match="disjoint"):
        tiny_config(validation_episode_seeds=(11, 12))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"replay_resolved_blocks": 7}, "exactly 8"),
        ({"replay_max_decisions": 31}, "exactly 32"),
    ],
)
def test_config_rejects_noncontract_replay_horizons(override, message):
    with pytest.raises(ValueError, match=message):
        tiny_config(**override)


def test_collector_writes_split_shards_dtypes_policies_and_replay_mask(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        dataset_module,
        "_environment_factory",
        lambda _config: tiny_environment,
    )

    manifest_path = collect_pretraining_dataset(tiny_config(), tmp_path)
    manifest = read_dataset_manifest(manifest_path)

    assert manifest_path == tmp_path / "dataset_manifest.json"
    assert manifest["dataset_schema_version"] == 1
    assert manifest["observation_schema_version"] == 4
    assert manifest["source_manifest_sha256"] == "a" * 64
    assert manifest["splits"]["train"]["state_count"] == 8
    assert manifest["splits"]["validation"]["state_count"] == 4
    assert manifest["splits"]["train"]["episode_seeds"] == [10, 11]
    assert manifest["splits"]["validation"]["episode_seeds"] == [20, 21]
    assert [
        episode["collector_policy"]
        for episode in manifest["splits"]["train"]["episodes"]
    ] == ["random_valid", "greedy_immediate_area"]
    assert len(manifest["splits"]["train"]["shards"]) == 3
    assert len(manifest["splits"]["validation"]["shards"]) == 2
    assert not list(tmp_path.rglob("*.tmp"))

    replay_masks = []
    for split_name in ("train", "validation"):
        for entry in manifest["splits"][split_name]["shards"]:
            path = tmp_path / entry["path"]
            with np.load(path, allow_pickle=False) as stored:
                assert stored["obs__grids"].dtype == np.float16
                assert stored["obs__ws_meta"].dtype == np.float32
            loaded = load_pretraining_shard(tmp_path, manifest, entry)
            assert loaded["observations"]["grids"].dtype == np.float32
            assert loaded["observations"]["ws_meta"].dtype == np.float32
            assert loaded["targets"]["future_fit"].dtype == np.float32
            assert loaded["targets"]["action_mask"].dtype == np.bool_
            if split_name == "train":
                replay_masks.extend(
                    loaded["targets"]["replay_mask"].any(axis=1).tolist()
                )
    assert replay_masks == [True, False, False, False, True, False, False, False]


def test_loader_rejects_modified_shard_before_numpy_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        dataset_module,
        "_environment_factory",
        lambda _config: tiny_environment,
    )
    manifest_path = collect_pretraining_dataset(tiny_config(), tmp_path)
    manifest = read_dataset_manifest(manifest_path)
    entry = manifest["splits"]["train"]["shards"][0]
    shard_path = tmp_path / entry["path"]
    with shard_path.open("ab") as destination:
        destination.write(b"modified")

    load_called = False

    def fail_if_loaded(*_args, **_kwargs):
        nonlocal load_called
        load_called = True
        raise AssertionError("np.load must not run before hash verification")

    monkeypatch.setattr(np, "load", fail_if_loaded)
    with pytest.raises(ValueError, match="SHA256"):
        load_pretraining_shard(tmp_path, manifest, entry)
    assert not load_called


def test_interrupted_collection_resumes_after_verified_shards(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        dataset_module,
        "_environment_factory",
        lambda _config: tiny_environment,
    )
    config = tiny_config(
        train_state_count=4,
        validation_state_count=2,
        train_episode_seeds=(10, 10),
        validation_episode_seeds=(20, 20),
        states_per_shard=2,
    )
    original_write = dataset_module._write_shard
    write_count = 0

    def interrupt_second_shard(*args, **kwargs):
        nonlocal write_count
        write_count += 1
        if write_count == 2:
            raise RuntimeError("injected shard interruption")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(
        dataset_module, "_write_shard", interrupt_second_shard
    )
    with pytest.raises(RuntimeError, match="shard interruption"):
        collect_pretraining_dataset(config, tmp_path)

    progress = tmp_path / "dataset_progress.json"
    assert progress.is_file()
    first_shard = tmp_path / "train" / "shard-00000.npz"
    first_bytes = first_shard.read_bytes()

    target_calls = 0
    original_targets = dataset_module.build_auxiliary_targets

    def count_targets(*args, **kwargs):
        nonlocal target_calls
        target_calls += 1
        return original_targets(*args, **kwargs)

    monkeypatch.setattr(dataset_module, "_write_shard", original_write)
    monkeypatch.setattr(
        dataset_module, "build_auxiliary_targets", count_targets
    )
    manifest = collect_pretraining_dataset(config, tmp_path)

    assert manifest.is_file()
    assert target_calls == 4
    assert first_shard.read_bytes() == first_bytes
    assert not progress.exists()


def test_progress_mirror_survives_loss_of_local_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        dataset_module,
        "_environment_factory",
        lambda _config: tiny_environment,
    )
    config = tiny_config(
        train_state_count=4,
        validation_state_count=2,
        train_episode_seeds=(10, 10),
        validation_episode_seeds=(20, 20),
        states_per_shard=2,
    )
    local = tmp_path / "local"
    mirror = tmp_path / "drive"
    original_write = dataset_module._write_shard
    write_count = 0

    def interrupt_second_shard(*args, **kwargs):
        nonlocal write_count
        write_count += 1
        if write_count == 2:
            raise RuntimeError("injected shard interruption")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(
        dataset_module, "_write_shard", interrupt_second_shard
    )
    with pytest.raises(RuntimeError, match="shard interruption"):
        collect_pretraining_dataset(
            config, local, progress_mirror_dir=mirror
        )

    assert (mirror / "dataset_progress.json").is_file()
    mirrored_shard = mirror / "train" / "shard-00000.npz"
    assert mirrored_shard.read_bytes() == (
        local / "train" / "shard-00000.npz"
    ).read_bytes()

    resumed = tmp_path / "resumed"
    shutil.copytree(mirror, resumed)
    monkeypatch.setattr(dataset_module, "_write_shard", original_write)
    manifest_path = collect_pretraining_dataset(
        config, resumed, progress_mirror_dir=mirror
    )

    assert manifest_path.is_file()
    assert (mirror / "dataset_manifest.json").is_file()
    assert not (mirror / "dataset_progress.json").exists()
