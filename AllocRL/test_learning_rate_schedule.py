from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from sb3_contrib import MaskablePPO

from learning_rate_schedule import (
    AbsoluteLearningRateSchedule,
    AbsoluteScheduleCallback,
    require_model_schedule,
)


class TinyMaskedEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(
            -1.0, 1.0, shape=(2,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(2)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(2, dtype=np.float32), {}

    def action_masks(self):
        return np.ones(2, dtype=bool)

    def step(self, action):
        self.steps += 1
        terminated = self.steps >= 5
        return (
            np.full(2, float(action), dtype=np.float32),
            0.0,
            terminated,
            False,
            {},
        )


def test_absolute_linear_schedule_values_and_floor():
    schedule = AbsoluteLearningRateSchedule(
        mode="linear",
        initial_rate=1e-4,
        final_rate=1e-5,
        decay_steps=1_000_000,
    )

    assert schedule.at_step(0) == pytest.approx(1e-4)
    assert schedule.at_step(500_000) == pytest.approx(5.5e-5)
    assert schedule.at_step(1_000_000) == pytest.approx(1e-5)
    assert schedule.at_step(2_000_000) == pytest.approx(1e-5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_rate": float("nan")},
        {"initial_rate": 0.0},
        {"final_rate": float("inf")},
        {"final_rate": 0.0},
        {"decay_steps": 0},
        {"final_rate": 2e-4},
        {"mode": "cosine"},
    ],
)
def test_absolute_schedule_rejects_invalid_contract(kwargs):
    values = {
        "mode": "linear",
        "initial_rate": 1e-4,
        "final_rate": 1e-5,
        "decay_steps": 100,
    }
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        AbsoluteLearningRateSchedule(**values)


def test_maskable_ppo_schedule_continues_after_save_load_resume(tmp_path: Path):
    env = TinyMaskedEnv()
    schedule = AbsoluteLearningRateSchedule(
        mode="linear",
        initial_rate=1e-4,
        final_rate=1e-5,
        decay_steps=64,
    )
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=schedule,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        seed=0,
        device="cpu",
        verbose=0,
    )
    model.learn(
        total_timesteps=16,
        callback=AbsoluteScheduleCallback(),
        progress_bar=False,
    )
    before_step = model.num_timesteps
    before_rate = model.policy.optimizer.param_groups[0]["lr"]
    path = tmp_path / "model.sb3"
    model.save(path)

    loaded = MaskablePPO.load(path, env=TinyMaskedEnv(), device="cpu")
    require_model_schedule(loaded, schedule.spec())
    loaded.learn(
        total_timesteps=16,
        callback=AbsoluteScheduleCallback(),
        reset_num_timesteps=False,
        progress_bar=False,
    )
    after_rate = loaded.policy.optimizer.param_groups[0]["lr"]

    assert before_step == 16
    assert loaded.num_timesteps == 32
    assert before_rate == pytest.approx(schedule.at_step(16))
    assert after_rate == pytest.approx(schedule.at_step(32))
    assert after_rate < before_rate < 1e-4

