from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

import pretraining.dataset as dataset_module
from pretraining.model import CandidatePretrainingModel
from pretraining.dataset import collect_pretraining_dataset
from pretraining.train_encoder import (
    PretrainingGateError,
    publish_pretraining_completion,
    train_candidate_encoder,
    validate_dataset_contract,
    write_extractor_checkpoint,
)
from test_pretraining_dataset import tiny_config, tiny_environment


def observation_space(grid_size: int = 16) -> gym.spaces.Dict:
    return gym.spaces.Dict({
        "block": gym.spaces.Box(0, 1, (8,), np.float32),
        "grids": gym.spaces.Box(
            0, 1, (10, 4, grid_size, grid_size), np.float32
        ),
        "ws_meta": gym.spaces.Box(0, 1, (10, 8), np.float32),
        "future_blocks": gym.spaces.Box(0, 1, (16, 6), np.float32),
        "future_mask": gym.spaces.Box(0, 1, (16,), np.float32),
        "future_demand": gym.spaces.Box(0, 1, (3, 6), np.float32),
        "pending_blocks": gym.spaces.Box(0, 1, (10, 32, 7), np.float32),
        "pending_mask": gym.spaces.Box(0, 1, (10, 32), np.float32),
        "pending_summary": gym.spaces.Box(0, 1, (10, 4), np.float32),
    })


def observation_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    values = {}
    for key, space in observation_space().spaces.items():
        values[key] = torch.rand((batch_size, *space.shape))
    values["future_mask"].fill_(1.0)
    values["pending_mask"].fill_(1.0)
    return values


def target_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    action_mask = torch.ones(batch_size, 10, dtype=torch.bool)
    action_mask[:, 9] = False
    replay_mask = torch.zeros(batch_size, 10, dtype=torch.bool)
    replay_mask[:, 0] = True
    return {
        "action_mask": action_mask,
        "current_placeable": torch.rand(batch_size, 10).round(),
        "future_fit": torch.rand(batch_size, 10, 16).round(),
        "future_optionality_after": torch.rand(batch_size, 10),
        "future_optionality_delta": torch.rand(batch_size, 10) * 2 - 1,
        "largest_free_rectangle_ratio": torch.rand(batch_size, 10),
        "free_component_count_normalized": torch.rand(batch_size, 10),
        "replay_success_rate": torch.rand(batch_size, 10),
        "replay_dropout_rate": torch.rand(batch_size, 10),
        "replay_delay_ratio": torch.rand(batch_size, 10),
        "replay_mask": replay_mask,
    }


def test_auxiliary_heads_emit_finite_per_action_predictions():
    model = CandidatePretrainingModel(observation_space(), features_dim=256)
    predictions = model(observation_batch())

    assert predictions.current_placeable.shape == (2, 10)
    assert predictions.future_fit.shape == (2, 10, 16)
    for name in (
        "future_optionality_after",
        "future_optionality_delta",
        "largest_free_rectangle_ratio",
        "free_component_count_normalized",
        "replay_success_rate",
        "replay_dropout_rate",
        "replay_delay_ratio",
    ):
        assert getattr(predictions, name).shape == (2, 10)
    assert all(
        torch.isfinite(value).all()
        for value in vars(predictions).values()
    )
    assert all(
        not key.startswith("heads.")
        for key in model.extractor.state_dict()
    )
    assert any(key.startswith("heads.") for key in model.state_dict())


def test_masked_loss_ignores_invalid_actions_and_unavailable_replay():
    torch.manual_seed(7)
    model = CandidatePretrainingModel(observation_space(), features_dim=256)
    observations = observation_batch()
    targets = target_batch()
    changed = {key: value.clone() for key, value in targets.items()}
    changed["current_placeable"][:, 9] = 1 - changed["current_placeable"][:, 9]
    changed["future_fit"][:, 9] = 1 - changed["future_fit"][:, 9]
    changed["future_optionality_after"][:, 9] = 100.0
    changed["future_optionality_delta"][:, 9] = -100.0
    changed["largest_free_rectangle_ratio"][:, 9] = 100.0
    changed["free_component_count_normalized"][:, 9] = 100.0
    for key in (
        "replay_success_rate",
        "replay_dropout_rate",
        "replay_delay_ratio",
    ):
        changed[key][:, 1:] = 1 - changed[key][:, 1:]

    predictions = model(observations)
    expected = model.loss(predictions, targets)
    actual = model.loss(predictions, changed)

    torch.testing.assert_close(expected, actual)


def test_auxiliary_loss_trains_cnn_structured_workspace_and_global_layers():
    torch.manual_seed(11)
    model = CandidatePretrainingModel(observation_space(), features_dim=256)
    loss = model.loss(model(observation_batch()), target_batch())

    loss.backward()

    assert torch.isfinite(loss)
    assert model.extractor.image_encoder[0].weight.grad.norm() > 0
    assert model.extractor.structured_encoder.current[0].weight.grad.norm() > 0
    assert model.extractor.workspace_fusion[0].weight.grad.norm() > 0
    assert model.extractor.global_fusion[0].weight.grad.norm() > 0


def test_production_pretraining_config_is_exact():
    path = Path(__file__).with_name("configs") / "candidate_pretrain_seed0.json"
    values = json.loads(path.read_text(encoding="utf-8"))

    assert values == {
        "schema_version": 1,
        "seed": 0,
        "train_state_count": 5000,
        "validation_state_count": 1000,
        "train_episode_seeds": [20000, 20039],
        "validation_episode_seeds": [30000, 30009],
        "states_per_shard": 100,
        "replay_every_n_states": 4,
        "replay_resolved_blocks": 8,
        "replay_max_decisions": 32,
        "optimizer": "AdamW",
        "learning_rate": 0.0001,
        "batch_size": 8,
        "max_epochs": 30,
        "early_stopping_patience": 5,
        "minimum_relative_baseline_improvement": 0.05,
        "minimum_shuffled_grid_degradation": 0.05,
    }


def test_checkpoint_contains_only_extractor_and_reproducibility_metadata(
    tmp_path: Path,
):
    model = CandidatePretrainingModel(observation_space())
    path = write_extractor_checkpoint(
        model,
        tmp_path / "candidate_encoder_pretrained.pt",
        observation_schema_version=4,
        config_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        best_epoch=3,
    )

    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert set(payload) == {
        "checkpoint_schema_version",
        "observation_schema_version",
        "config_sha256",
        "dataset_manifest_sha256",
        "best_epoch",
        "extractor_state_dict",
    }
    assert set(payload["extractor_state_dict"]) == set(
        model.extractor.state_dict()
    )
    assert all(
        not key.startswith("heads.")
        for key in payload["extractor_state_dict"]
    )


def test_completion_marker_is_omitted_on_failed_gate_and_hashed_on_success(
    tmp_path: Path,
):
    checkpoint = tmp_path / "candidate_encoder_pretrained.pt"
    checkpoint.write_bytes(b"extractor")
    metrics = tmp_path / "pretraining_metrics.json"
    metrics.write_text('{"validation_total_loss": 0.5}\n', encoding="utf-8")
    marker = tmp_path / "PRETRAINING_COMPLETE.json"

    with pytest.raises(PretrainingGateError, match="future_fit"):
        publish_pretraining_completion(
            checkpoint,
            metrics,
            marker,
            config_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
            gates={"future_fit": False, "grid_dependence": True},
            smoke_mode=False,
        )
    assert not marker.exists()

    publish_pretraining_completion(
        checkpoint,
        metrics,
        marker,
        config_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        gates={"future_fit": True, "grid_dependence": True},
        smoke_mode=False,
    )
    receipt = json.loads(marker.read_text(encoding="utf-8"))
    assert receipt["checkpoint_sha256"] == sha256(b"extractor").hexdigest()
    assert receipt["metrics_sha256"] == sha256(metrics.read_bytes()).hexdigest()
    assert receipt["production_eligible"] is True


def test_production_training_rejects_dataset_config_mismatch():
    configured = {
        "train_state_count": 5000,
        "validation_state_count": 1000,
        "train_episode_seeds": [20000, 20039],
        "validation_episode_seeds": [30000, 30009],
        "states_per_shard": 100,
        "replay_every_n_states": 4,
        "replay_resolved_blocks": 8,
        "replay_max_decisions": 32,
    }
    manifest = {"config": {**configured, "train_state_count": 32}}

    with pytest.raises(ValueError, match="train_state_count"):
        validate_dataset_contract(configured, manifest, smoke_mode=False)
    validate_dataset_contract(configured, manifest, smoke_mode=True)


def test_tiny_training_publishes_finite_smoke_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        dataset_module,
        "_environment_factory",
        lambda _config: tiny_environment,
    )
    dataset_root = tmp_path / "dataset"
    data_config = tiny_config(grid_size=16)
    collect_pretraining_dataset(data_config, dataset_root)
    config_values = {
        **data_config.__dict__,
        "schema_version": 1,
        "seed": 0,
        "optimizer": "AdamW",
        "learning_rate": 0.0001,
        "batch_size": 2,
        "max_epochs": 2,
        "early_stopping_patience": 1,
        "minimum_relative_baseline_improvement": 0.05,
        "minimum_shuffled_grid_degradation": 0.05,
    }
    for key, value in tuple(config_values.items()):
        if isinstance(value, tuple):
            config_values[key] = list(value)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(config_values), encoding="utf-8"
    )
    output = tmp_path / "output"

    marker = train_candidate_encoder(
        config_path,
        dataset_root,
        output,
        smoke_state_count=4,
        max_epochs=1,
        device="cpu",
    )

    assert marker == output / "PRETRAINING_COMPLETE.json"
    receipt = json.loads(marker.read_text(encoding="utf-8"))
    metrics = json.loads(
        (output / "pretraining_metrics.json").read_text(encoding="utf-8")
    )
    assert receipt["production_eligible"] is False
    assert receipt["smoke_mode"] is True
    assert np.isfinite(metrics["validation_total_loss"])
    assert (output / "candidate_encoder_pretrained.pt").is_file()
