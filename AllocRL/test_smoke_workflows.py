from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

import smoke_test
from alloc_env.observation_state import build_observation_space


EXPECTED_SCHEMA3_SHAPES = {
    "block": (8,),
    "grids": (10, 4, 64, 64),
    "ws_meta": (10, 4),
    "future_blocks": (16, 6),
    "future_mask": (16,),
    "future_demand": (3, 4),
    "pending_blocks": (10, 32, 7),
    "pending_mask": (10, 32),
    "pending_summary": (10, 4),
}


class FakeModel:
    def __init__(self) -> None:
        self.saved_paths: list[Path] = []

    def save(self, path: Path) -> None:
        self.saved_paths.append(path)


def test_smoke_contract_lists_all_extractors_and_schema3_shapes():
    assert smoke_test.EXTRACTORS == (
        "structured",
        "fixed-grid",
        "candidate-cnn",
    )
    assert smoke_test.SCHEMA3_OBSERVATION_SHAPES == EXPECTED_SCHEMA3_SHAPES
    smoke_test.validate_schema3_observation_space(build_observation_space())


def test_schema3_validation_rejects_legacy_grid_shape():
    spaces = dict(build_observation_space().spaces)
    spaces["grids"] = gym.spaces.Box(
        0, 1, shape=(10, 3, 128, 128), dtype=np.float32
    )

    with pytest.raises(AssertionError, match="grids"):
        smoke_test.validate_schema3_observation_space(
            gym.spaces.Dict(spaces)
        )


def test_run_extractor_smoke_saves_loads_and_evaluates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    model = FakeModel()
    environment = object()
    loaded = object()
    calls: dict[str, object] = {}

    def fake_train(*, extractor: str, timesteps: int):
        calls["train"] = (extractor, timesteps)
        return model, environment

    def fake_load(path: Path, *, env):
        calls["load"] = (path, env)
        return loaded

    def fake_evaluate(loaded_model, env, *, n_eval, return_metrics):
        calls["evaluate"] = (
            loaded_model,
            env,
            n_eval,
            return_metrics,
        )
        return {"mean_terminal_score": -12.5}

    monkeypatch.setattr(smoke_test, "train_tiny_model", fake_train)
    monkeypatch.setattr(smoke_test.MaskablePPO, "load", fake_load)
    monkeypatch.setattr(smoke_test, "evaluate", fake_evaluate)

    metrics = smoke_test.run_extractor_smoke(
        "structured", tmp_path, timesteps=1_024
    )

    path = tmp_path / "structured.sb3"
    assert calls["train"] == ("structured", 1_024)
    assert model.saved_paths == [path]
    assert calls["load"] == (path, environment)
    assert calls["evaluate"] == (loaded, environment, 1, True)
    assert metrics["mean_terminal_score"] == -12.5


def test_run_extractor_smoke_rejects_nonfinite_terminal_score(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        smoke_test,
        "train_tiny_model",
        lambda **_kwargs: (FakeModel(), object()),
    )
    monkeypatch.setattr(
        smoke_test.MaskablePPO,
        "load",
        lambda _path, *, env: object(),
    )
    monkeypatch.setattr(
        smoke_test,
        "evaluate",
        lambda *_args, **_kwargs: {"mean_terminal_score": math.nan},
    )

    with pytest.raises(AssertionError, match="finite terminal score"):
        smoke_test.run_extractor_smoke(
            "fixed-grid", tmp_path, timesteps=1_024
        )


@pytest.mark.parametrize(
    ("diagnostics", "message"),
    [
        (
            {"cnn_gradient_norm": 0.0, "cnn_weight_change": 1.0},
            "gradient",
        ),
        (
            {"cnn_gradient_norm": 1.0, "cnn_weight_change": 0.0},
            "weight",
        ),
    ],
)
def test_cnn_diagnostics_require_gradient_and_weight_change(
    diagnostics: dict[str, float],
    message: str,
):
    with pytest.raises(AssertionError, match=message):
        smoke_test.validate_cnn_diagnostics(diagnostics)


def test_all_extractors_cli_uses_temporary_output_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[str, Path, int]] = []

    def fake_run(extractor: str, output_dir: Path, *, timesteps: int):
        assert output_dir.is_dir()
        calls.append((extractor, output_dir, timesteps))
        return {"mean_terminal_score": 0.0}

    monkeypatch.setattr(smoke_test, "run_extractor_smoke", fake_run)

    assert smoke_test.main(
        ["--all-extractors", "--timesteps", "1024"]
    ) == 0

    assert [extractor for extractor, _path, _steps in calls] == list(
        smoke_test.EXTRACTORS
    )
    assert all(steps == 1_024 for _extractor, _path, steps in calls)
    temporary_path = calls[0][1]
    assert all(path == temporary_path for _extractor, path, _steps in calls)
    assert not temporary_path.exists()


def test_single_extractor_cli_preserves_explicit_output_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    calls: list[tuple[str, Path, int]] = []

    def fake_run(extractor: str, output_dir: Path, *, timesteps: int):
        calls.append((extractor, output_dir, timesteps))
        return {"mean_terminal_score": 0.0}

    monkeypatch.setattr(smoke_test, "run_extractor_smoke", fake_run)

    assert smoke_test.main([
        "--extractor",
        "fixed-grid",
        "--timesteps",
        "12",
        "--output-dir",
        str(tmp_path),
    ]) == 0

    assert calls == [("fixed-grid", tmp_path.resolve(), 12)]
    assert tmp_path.is_dir()
