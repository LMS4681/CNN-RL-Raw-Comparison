import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest
import torch

from alloc_env.callbacks import AllocationCallback, CnnDiagnosticTracker
from alloc_env.cnn_extractor import (
    CandidateCnnExtractor,
    FixedGridExtractor,
    StructuredExtractor,
)


N_WORKSPACES = 10
GRID_SIZE = 16


def space() -> gym.spaces.Dict:
    return gym.spaces.Dict({
        "block": gym.spaces.Box(0, 1, (8,), np.float32),
        "grids": gym.spaces.Box(
            0, 1, (N_WORKSPACES, 4, GRID_SIZE, GRID_SIZE), np.float32
        ),
        "ws_meta": gym.spaces.Box(0, 1, (N_WORKSPACES, 8), np.float32),
        "future_blocks": gym.spaces.Box(0, 1, (16, 6), np.float32),
        "future_mask": gym.spaces.Box(0, 1, (16,), np.float32),
        "future_demand": gym.spaces.Box(0, 1, (3, 6), np.float32),
        "pending_blocks": gym.spaces.Box(
            0, 1, (N_WORKSPACES, 32, 7), np.float32
        ),
        "pending_mask": gym.spaces.Box(
            0, 1, (N_WORKSPACES, 32), np.float32
        ),
        "pending_summary": gym.spaces.Box(
            0, 1, (N_WORKSPACES, 4), np.float32
        ),
    })


def observation() -> dict[str, torch.Tensor]:
    return {
        "block": torch.rand(2, 8),
        "grids": torch.rand(
            2, N_WORKSPACES, 4, GRID_SIZE, GRID_SIZE
        ),
        "ws_meta": torch.rand(2, N_WORKSPACES, 8),
        "future_blocks": torch.rand(2, 16, 6),
        "future_mask": torch.ones(2, 16),
        "future_demand": torch.rand(2, 3, 6),
        "pending_blocks": torch.rand(2, N_WORKSPACES, 32, 7),
        "pending_mask": torch.ones(2, N_WORKSPACES, 32),
        "pending_summary": torch.rand(2, N_WORKSPACES, 4),
    }


def numpy_observation() -> dict[str, np.ndarray]:
    return {
        key: value.numpy()
        for key, value in observation().items()
    }


class RecordingLogger:
    def __init__(self):
        self.records: dict[str, float] = {}

    def record(self, key: str, value: float) -> None:
        self.records[key] = value


class TensorPolicy:
    def __init__(self):
        self.observations: dict[str, np.ndarray] | None = None

    def obs_to_tensor(self, observations: dict[str, np.ndarray]):
        self.observations = observations
        return {
            key: torch.as_tensor(value)
            for key, value in observations.items()
        }, False


def prepared_allocation_callback(
    tmp_path,
    extractor_class=CandidateCnnExtractor,
) -> tuple[AllocationCallback, TensorPolicy, RecordingLogger]:
    callback = AllocationCallback(tmp_path, verbose=0)
    policy = TensorPolicy()
    logger = RecordingLogger()
    callback.model = SimpleNamespace(
        policy=policy,
        _last_obs=None,
        logger=logger,
    )
    callback._diagnostic_tracker = CnnDiagnosticTracker(
        extractor_class(space(), features_dim=32)
    )
    return callback, policy, logger


def diagnostic_copy_calls(array_mock):
    return [
        call
        for call in array_mock.call_args_list
        if call.kwargs == {"copy": True}
    ]


def diagnostic_copy_count(array_mock) -> int:
    return len(
        diagnostic_copy_calls(array_mock)
    )


class FailingArraySource:
    def __array__(self, *args, **kwargs):
        raise RuntimeError("diagnostic array copy failed")


def copied_snapshot_matches_sources(
    latest: dict[str, np.ndarray],
    snapshot: dict[str, np.ndarray],
) -> None:
    expected = {
        key: value.copy()
        for key, value in latest.items()
    }
    for key, source in latest.items():
        saved = snapshot[key]
        assert saved is not source
        assert not np.shares_memory(saved, source)
        np.testing.assert_array_equal(saved, expected[key])
        source.fill(-1.0)
        np.testing.assert_array_equal(saved, expected[key])


def assert_exact_source_copies(
    array_mock,
    latest: dict[str, np.ndarray],
) -> None:
    assert len(array_mock.call_args_list) == len(latest)
    assert all(
        call.kwargs == {"copy": True}
        for call in array_mock.call_args_list
    )
    copy_calls = diagnostic_copy_calls(array_mock)
    assert len(copy_calls) == len(latest)
    for source in latest.values():
        assert sum(
            call.args[0] is source
            for call in copy_calls
        ) == 1
    assert all(
        any(call.args[0] is source for source in latest.values())
        for call in copy_calls
    )


def test_diagnostic_observation_is_not_copied_on_steps(tmp_path):
    callback, _, _ = prepared_allocation_callback(tmp_path)

    with patch.object(np, "array", wraps=np.array) as array_mock:
        for _ in range(10):
            callback.locals = {
                "new_obs": numpy_observation(),
                "dones": [],
                "infos": [],
            }
            callback._on_step()

    assert diagnostic_copy_calls(array_mock) == []


def test_diagnostic_observation_copies_each_rollout_source_once(tmp_path):
    callback, _, _ = prepared_allocation_callback(tmp_path)
    latest = numpy_observation()
    callback.model._last_obs = latest

    with patch.object(np, "array", wraps=np.array) as array_mock:
        callback._on_rollout_end()

    assert_exact_source_copies(array_mock, latest)
    assert callback._diagnostic_observation is not None
    copied_snapshot_matches_sources(latest, callback._diagnostic_observation)


@pytest.mark.parametrize("extractor_class", (StructuredExtractor, FixedGridExtractor))
def test_non_cnn_extractors_do_not_copy_or_retain_diagnostic_observations(
    tmp_path,
    extractor_class,
):
    callback, _, _ = prepared_allocation_callback(tmp_path, extractor_class)
    callback._diagnostic_observation = numpy_observation()
    callback.model._last_obs = numpy_observation()

    with patch.object(np, "array", wraps=np.array) as array_mock:
        callback._on_rollout_end()

    assert diagnostic_copy_count(array_mock) == 0
    assert callback._diagnostic_observation is None


def test_rollout_end_clears_stale_observation_before_copy_failure(tmp_path):
    callback, _, _ = prepared_allocation_callback(tmp_path)
    callback._diagnostic_observation = numpy_observation()
    callback.model._last_obs = {
        "block": np.zeros((2, 8), dtype=np.float32),
        "grids": FailingArraySource(),
    }

    with pytest.raises(RuntimeError, match="diagnostic array copy failed"):
        callback._on_rollout_end()

    assert callback._diagnostic_observation is None


def test_rollout_start_records_all_cnn_diagnostics_from_rollout_copy(tmp_path):
    callback, policy, logger = prepared_allocation_callback(tmp_path)
    latest = numpy_observation()
    callback.model._last_obs = latest
    callback._rollout_count = 1
    callback._on_rollout_end()
    latest["grids"].fill(0.0)

    callback._on_rollout_start()

    assert set(logger.records) == EXPECTED_CNN_DIAGNOSTIC_LOGS
    assert policy.observations is not None
    assert np.any(policy.observations["grids"])
    assert callback._diagnostic_observation is None


@pytest.mark.parametrize("extractor_class", (StructuredExtractor, FixedGridExtractor))
def test_rollout_start_emits_exactly_no_cnn_diagnostics_for_non_cnn_extractors(
    tmp_path,
    extractor_class,
):
    callback, _, logger = prepared_allocation_callback(tmp_path, extractor_class)
    callback.model._last_obs = numpy_observation()
    callback._rollout_count = 1
    callback._on_rollout_end()

    callback._on_rollout_start()

    assert set(logger.records) == set()


def test_rollout_end_clears_stale_observation_without_a_dict_last_obs(tmp_path):
    callback, _, _ = prepared_allocation_callback(tmp_path)
    callback._diagnostic_observation = numpy_observation()
    callback.model._last_obs = np.zeros(1, dtype=np.float32)

    callback._on_rollout_end()

    assert callback._diagnostic_observation is None


def test_rollout_start_clears_observation_when_feature_measurement_fails(tmp_path):
    callback, _, _ = prepared_allocation_callback(tmp_path)

    class FailingPolicy:
        def obs_to_tensor(self, observations):
            raise RuntimeError("diagnostic conversion failed")

    callback.model.policy = FailingPolicy()
    callback._diagnostic_observation = numpy_observation()
    callback._rollout_count = 1

    with pytest.raises(RuntimeError, match="diagnostic conversion failed"):
        callback._on_rollout_start()

    assert callback._diagnostic_observation is None


EXPECTED_CNN_DIAGNOSTICS = {
    "cnn_gradient_norm",
    "cnn_weight_change",
    "workspace_feature_variance",
    "candidate_channel_sensitivity",
}
EXPECTED_CNN_DIAGNOSTIC_LOGS = {
    f"diagnostics/{key}"
    for key in EXPECTED_CNN_DIAGNOSTICS
}


class CnnDiagnosticTrackerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)

    def test_tracker_records_gradient_and_weight_change(self):
        extractor = CandidateCnnExtractor(space(), features_dim=32)
        tracker = CnnDiagnosticTracker(extractor)
        tracker.attach()
        optimizer = torch.optim.Adam(extractor.parameters(), lr=1e-3)

        output = extractor(observation())
        output.square().mean().backward()
        optimizer.step()
        metrics = tracker.record_update()
        tracker.close()

        self.assertGreater(metrics["cnn_gradient_norm"], 0.0)
        self.assertGreater(metrics["cnn_weight_change"], 0.0)

    def test_candidate_sensitivity_uses_candidate_channel(self):
        extractor = CandidateCnnExtractor(
            space(), features_dim=32
        ).eval()

        metrics = CnnDiagnosticTracker(extractor).measure_features(
            observation()
        )

        self.assertGreaterEqual(metrics["workspace_feature_variance"], 0.0)
        self.assertGreater(metrics["candidate_channel_sensitivity"], 0.0)

    def test_workspace_feature_variance_uses_only_workspace_axis(self):
        extractor = CandidateCnnExtractor(
            space(), features_dim=32
        ).eval()
        observations = observation()
        observations["grids"][:, 1:] = observations["grids"][:, :1]

        metrics = CnnDiagnosticTracker(extractor).measure_features(
            observations
        )

        self.assertAlmostEqual(
            0.0, metrics["workspace_feature_variance"], places=7
        )

    def test_non_cnn_extractors_emit_no_cnn_metrics(self):
        for extractor_class in (StructuredExtractor, FixedGridExtractor):
            with self.subTest(extractor=extractor_class.__name__):
                tracker = CnnDiagnosticTracker(extractor_class(space()))
                tracker.attach()
                self.assertEqual({}, tracker.record_update())
                self.assertEqual({}, tracker.measure_features(observation()))
                tracker.close()


if __name__ == "__main__":
    unittest.main()
