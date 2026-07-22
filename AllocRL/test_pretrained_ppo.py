from __future__ import annotations

import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from alloc_env.cnn_extractor import CandidateCnnExtractor
from learning_rate_schedule import (
    AbsoluteLearningRateSchedule,
    AbsoluteScheduleCallback,
)
from pretraining.ppo import (
    ExtractorFineTuneCallback,
    ScaleAwareMaskablePPO,
)
from pretraining.transfer import load_verified_pretrained_extractor
from evaluation_runner import model_class_from_run_config


def observation_space(grid_size: int = 8) -> gym.spaces.Dict:
    return gym.spaces.Dict({
        "block": gym.spaces.Box(0, 1, (8,), np.float32),
        "grids": gym.spaces.Box(
            0, 1, (10, 4, grid_size, grid_size), np.float32
        ),
        "ws_meta": gym.spaces.Box(0, 1, (10, 8), np.float32),
        "future_blocks": gym.spaces.Box(0, 1, (16, 6), np.float32),
        "future_mask": gym.spaces.Box(0, 1, (16,), np.float32),
        "future_demand": gym.spaces.Box(0, 1, (3, 6), np.float32),
        "pending_blocks": gym.spaces.Box(
            0, 1, (10, 32, 7), np.float32
        ),
        "pending_mask": gym.spaces.Box(0, 1, (10, 32), np.float32),
        "pending_summary": gym.spaces.Box(0, 1, (10, 4), np.float32),
    })


def numpy_observation(space: gym.spaces.Dict, value: float = 0.25) -> dict:
    return {
        key: np.full(item.shape, value, dtype=np.float32)
        for key, item in space.spaces.items()
    }


class TinyCandidateEnv(gym.Env):
    def __init__(self):
        self.observation_space = observation_space()
        self.action_space = gym.spaces.Discrete(10)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return numpy_observation(self.observation_space), {}

    def action_masks(self):
        return np.ones(10, dtype=bool)

    def step(self, action):
        self.steps += 1
        value = 0.25 + 0.01 * self.steps
        return (
            numpy_observation(self.observation_space, value),
            1.0 if int(action) == 0 else -0.1,
            self.steps >= 3,
            False,
            {},
        )


def make_model(*, seed: int = 0) -> ScaleAwareMaskablePPO:
    schedule = AbsoluteLearningRateSchedule(
        mode="constant",
        initial_rate=1e-3,
        final_rate=1e-3,
        decay_steps=0,
    )
    return ScaleAwareMaskablePPO(
        "MultiInputPolicy",
        TinyCandidateEnv(),
        learning_rate=schedule,
        extractor_lr_scale=0.1,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        gamma=0.9,
        seed=seed,
        device="cpu",
        verbose=0,
        policy_kwargs={
            "features_extractor_class": CandidateCnnExtractor,
            "features_extractor_kwargs": {"features_dim": 32},
            "share_features_extractor": True,
            "net_arch": {"pi": [16], "vf": [16]},
        },
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_artifacts(
    root: Path,
    state_dict: dict[str, torch.Tensor],
    *,
    marker_overrides: dict | None = None,
    payload_overrides: dict | None = None,
) -> tuple[Path, Path]:
    config_sha = "a" * 64
    manifest_sha = "b" * 64
    checkpoint = root / "candidate_encoder_pretrained.pt"
    payload = {
        "checkpoint_schema_version": 1,
        "observation_schema_version": 4,
        "config_sha256": config_sha,
        "dataset_manifest_sha256": manifest_sha,
        "best_epoch": 3,
        "extractor_state_dict": state_dict,
    }
    payload.update(payload_overrides or {})
    torch.save(payload, checkpoint)

    gates = {
        "finite": True,
        "future_fit": True,
        "optionality": True,
        "grid_dependence": True,
        "counterfactual_geometry": True,
    }
    metrics = root / "pretraining_metrics.json"
    metrics.write_text(
        json.dumps({"gates": gates, "smoke_mode": False}) + "\n",
        encoding="utf-8",
    )
    marker = root / "PRETRAINING_COMPLETE.json"
    receipt = {
        "schema_version": 1,
        "observation_schema_version": 4,
        "checkpoint_filename": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "metrics_filename": metrics.name,
        "metrics_sha256": sha256_file(metrics),
        "config_sha256": config_sha,
        "dataset_manifest_sha256": manifest_sha,
        "gates": gates,
        "smoke_mode": False,
        "production_eligible": True,
    }
    receipt.update(marker_overrides or {})
    marker.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    return checkpoint, marker


def clone_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
    }


def any_tensor_changed(
    before: dict[str, torch.Tensor], module: torch.nn.Module
) -> bool:
    return any(
        not torch.equal(before[name], value.detach())
        for name, value in module.state_dict().items()
    )


def test_verified_transfer_is_exact_and_does_not_touch_policy_heads(tmp_path):
    model = make_model(seed=11)
    extractor = model.policy.features_extractor
    source = CandidateCnnExtractor(observation_space(), features_dim=32)
    with torch.no_grad():
        for index, parameter in enumerate(source.parameters()):
            parameter.fill_(0.001 * (index + 1))
    checkpoint, marker = write_artifacts(tmp_path, source.state_dict())
    heads_before = {
        name: value.detach().clone()
        for name, value in model.policy.state_dict().items()
        if "features_extractor" not in name
    }

    receipt = load_verified_pretrained_extractor(
        model, checkpoint, marker
    )

    for name, value in source.state_dict().items():
        assert torch.equal(extractor.state_dict()[name], value)
    for name, value in heads_before.items():
        assert torch.equal(model.policy.state_dict()[name], value)
    assert receipt.checkpoint_sha256 == sha256_file(checkpoint)
    assert receipt.manifest_sha256 == "b" * 64
    assert receipt.complete_sha256 == sha256_file(marker)


def test_transfer_rejects_missing_completion_marker(tmp_path):
    model = make_model()
    checkpoint, marker = write_artifacts(
        tmp_path, model.policy.features_extractor.state_dict()
    )
    marker.unlink()
    with pytest.raises(FileNotFoundError, match="PRETRAINING_COMPLETE"):
        load_verified_pretrained_extractor(model, checkpoint, marker)


@pytest.mark.parametrize(
    "marker_overrides, payload_overrides, message",
    [
        ({"production_eligible": False}, {}, "production eligible"),
        ({"gates": {"future_fit": False}}, {}, "gates"),
        ({"observation_schema_version": 3}, {}, "schema"),
        ({"checkpoint_sha256": "0" * 64}, {}, "checkpoint SHA256"),
        ({}, {"observation_schema_version": 3}, "schema"),
        ({}, {"config_sha256": "c" * 64}, "config SHA256"),
    ],
)
def test_transfer_rejects_invalid_receipts(
    tmp_path, marker_overrides, payload_overrides, message
):
    model = make_model()
    checkpoint, marker = write_artifacts(
        tmp_path,
        model.policy.features_extractor.state_dict(),
        marker_overrides=marker_overrides,
        payload_overrides=payload_overrides,
    )
    with pytest.raises(ValueError, match=message):
        load_verified_pretrained_extractor(model, checkpoint, marker)


@pytest.mark.parametrize("mutation", ["unknown", "missing", "auxiliary"])
def test_transfer_rejects_non_exact_extractor_keys(tmp_path, mutation):
    model = make_model()
    state = clone_state(model.policy.features_extractor)
    if mutation == "unknown":
        state["unknown.weight"] = torch.zeros(1)
    elif mutation == "missing":
        state.pop(next(iter(state)))
    else:
        state["heads.future_fit.weight"] = torch.zeros(1)
    checkpoint, marker = write_artifacts(tmp_path, state)

    with pytest.raises(ValueError, match="extractor state keys"):
        load_verified_pretrained_extractor(model, checkpoint, marker)


def test_optimizer_groups_preserve_scaled_learning_rates():
    model = make_model()
    groups = {group["name"]: group for group in model.policy.optimizer.param_groups}

    assert set(groups) == {"policy", "extractor"}
    assert groups["policy"]["lr"] == pytest.approx(1e-3)
    assert groups["extractor"]["lr"] == pytest.approx(1e-4)
    assert groups["policy"]["lr_scale"] == pytest.approx(1.0)
    assert groups["extractor"]["lr_scale"] == pytest.approx(0.1)


def test_schema4_candidate_loader_requires_scale_aware_model_class():
    config = {
        "observation_schema_version": 4,
        "extractor": "candidate-cnn",
        "model_class": "ScaleAwareMaskablePPO",
    }
    assert model_class_from_run_config(config) is ScaleAwareMaskablePPO

    config["model_class"] = "MaskablePPO"
    with pytest.raises(ValueError, match="model_class"):
        model_class_from_run_config(config)


def test_schema3_loader_remains_maskable_ppo():
    from sb3_contrib import MaskablePPO

    assert model_class_from_run_config({
        "observation_schema_version": 3,
        "extractor": "candidate-cnn",
    }) is MaskablePPO


def test_checkpoint_evaluator_selects_loader_from_run_config(
    tmp_path, monkeypatch
):
    from comparison import checkpoint_evaluator

    class SelectedLoader:
        calls = []

        @classmethod
        def load(cls, path, **kwargs):
            cls.calls.append((path, kwargs))
            return "loaded"

    archive = tmp_path / "model.sb3"
    archive.write_bytes(b"archive")
    monkeypatch.setattr(
        "train.load_model_run_config",
        lambda path: {
            "observation_schema_version": 4,
            "extractor": "candidate-cnn",
            "model_class": "ScaleAwareMaskablePPO",
        },
    )
    monkeypatch.setattr(
        checkpoint_evaluator.evaluation_runner,
        "model_class_from_run_config",
        lambda config: SelectedLoader,
    )

    loaded = checkpoint_evaluator.load_model_archive(
        str(archive), device="cpu"
    )

    assert loaded == "loaded"
    assert SelectedLoader.calls == [(str(archive), {"device": "cpu"})]


def test_freeze_boundary_uses_absolute_timesteps_across_resume(tmp_path):
    model = make_model(seed=7)
    extractor = model.policy.features_extractor
    extractor_before = clone_state(extractor)
    policy_before = {
        name: value.detach().clone()
        for name, value in model.policy.state_dict().items()
        if "features_extractor" not in name
    }
    callbacks = [
        AbsoluteScheduleCallback(),
        ExtractorFineTuneCallback(freeze_until_timestep=5),
    ]
    model.learn(total_timesteps=4, callback=callbacks, progress_bar=False)

    assert model.num_timesteps == 4
    assert not any_tensor_changed(extractor_before, extractor)
    assert all(parameter.grad is None for parameter in extractor.parameters())
    assert any(
        not torch.equal(policy_before[name], value.detach())
        for name, value in model.policy.state_dict().items()
        if name in policy_before
    )
    archive = tmp_path / "before_boundary.sb3"
    model.save(archive)

    loaded = ScaleAwareMaskablePPO.load(
        archive, env=TinyCandidateEnv(), device="cpu"
    )
    loaded_extractor = loaded.policy.features_extractor
    before_resume = clone_state(loaded_extractor)
    loaded.learn(
        total_timesteps=4,
        callback=[
            AbsoluteScheduleCallback(),
            ExtractorFineTuneCallback(freeze_until_timestep=5),
        ],
        reset_num_timesteps=False,
        progress_bar=False,
    )

    assert loaded.num_timesteps == 8
    assert any_tensor_changed(before_resume, loaded_extractor)
    assert any(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in loaded_extractor.parameters()
    )
    groups = {
        group["name"]: group
        for group in loaded.policy.optimizer.param_groups
    }
    assert groups["extractor"]["lr"] == pytest.approx(
        groups["policy"]["lr"] * 0.1
    )
