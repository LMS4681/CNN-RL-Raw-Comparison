"""Learning-rate schedules keyed to cumulative SB3 timesteps."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback


class AbsoluteLearningRateSchedule:
    """A picklable SB3 schedule that ignores per-call progress resets."""

    def __init__(
        self,
        *,
        mode: str,
        initial_rate: float,
        final_rate: float,
        decay_steps: int,
    ) -> None:
        if mode not in {"constant", "linear"}:
            raise ValueError(f"unsupported learning-rate schedule: {mode!r}")
        for name, value in (
            ("initial_rate", initial_rate),
            ("final_rate", final_rate),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise TypeError(f"{name} must be a finite number")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if isinstance(decay_steps, bool) or not isinstance(decay_steps, int):
            raise TypeError("decay_steps must be an integer")
        if mode == "linear" and decay_steps <= 0:
            raise ValueError("linear decay_steps must be positive")
        if mode == "constant" and decay_steps != 0:
            raise ValueError("constant decay_steps must be zero")
        if mode == "linear" and final_rate > initial_rate:
            raise ValueError("linear final_rate cannot exceed initial_rate")
        if mode == "constant" and final_rate != initial_rate:
            raise ValueError("constant final_rate must equal initial_rate")

        self.mode = mode
        self.initial_rate = float(initial_rate)
        self.final_rate = float(final_rate)
        self.decay_steps = int(decay_steps)
        self.current_step = 0

    def at_step(self, timestep: int) -> float:
        if isinstance(timestep, bool) or not isinstance(timestep, int):
            raise TypeError("timestep must be an integer")
        if timestep < 0:
            raise ValueError("timestep must be non-negative")
        if self.mode == "constant":
            return self.initial_rate
        fraction = min(timestep / self.decay_steps, 1.0)
        return self.initial_rate + fraction * (
            self.final_rate - self.initial_rate
        )

    def sync(self, timestep: int) -> float:
        rate = self.at_step(timestep)
        self.current_step = timestep
        return rate

    def __call__(self, _progress_remaining: float) -> float:
        return self.at_step(self.current_step)

    def spec(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "initial_rate": self.initial_rate,
            "final_rate": self.final_rate,
            "decay_steps": self.decay_steps,
        }


def build_absolute_schedule(
    *,
    mode: str,
    initial_rate: float,
    final_rate: float | None,
    decay_steps: int | None,
) -> AbsoluteLearningRateSchedule:
    """Validate CLI-style values and construct the persisted schedule."""
    if mode == "constant":
        if final_rate is not None and float(final_rate) != float(initial_rate):
            raise ValueError("--lr-final is only configurable in linear mode")
        if decay_steps not in (None, 0):
            raise ValueError("--lr-decay-steps is only configurable in linear mode")
        return AbsoluteLearningRateSchedule(
            mode=mode,
            initial_rate=initial_rate,
            final_rate=initial_rate,
            decay_steps=0,
        )
    if mode == "linear":
        if final_rate is None:
            raise ValueError("linear mode requires --lr-final")
        if decay_steps is None:
            raise ValueError("linear mode requires --lr-decay-steps")
        return AbsoluteLearningRateSchedule(
            mode=mode,
            initial_rate=initial_rate,
            final_rate=final_rate,
            decay_steps=decay_steps,
        )
    raise ValueError(f"unsupported learning-rate schedule: {mode!r}")


def schedule_from_args(args: Any) -> AbsoluteLearningRateSchedule:
    return build_absolute_schedule(
        mode=str(getattr(args, "lr_schedule", "constant")),
        initial_rate=float(args.lr),
        final_rate=getattr(args, "lr_final", None),
        decay_steps=getattr(args, "lr_decay_steps", None),
    )


def _model_schedules(model: Any) -> list[AbsoluteLearningRateSchedule]:
    schedules: list[AbsoluteLearningRateSchedule] = []
    for name in ("learning_rate", "lr_schedule"):
        candidate = getattr(model, name, None)
        if isinstance(candidate, AbsoluteLearningRateSchedule) and all(
            candidate is not existing for existing in schedules
        ):
            schedules.append(candidate)
    return schedules


def require_model_schedule(
    model: Any,
    expected_spec: Mapping[str, object],
) -> AbsoluteLearningRateSchedule:
    """Require the loaded archive to contain the exact schedule contract."""
    schedules = _model_schedules(model)
    if not schedules:
        raise ValueError(
            "loaded model does not contain an AbsoluteLearningRateSchedule"
        )
    expected = dict(expected_spec)
    for schedule in schedules:
        if schedule.spec() != expected:
            raise ValueError(
                "loaded model learning-rate schedule differs from run config: "
                f"saved={schedule.spec()!r}, expected={expected!r}"
            )
    return schedules[0]


class AbsoluteScheduleCallback(BaseCallback):
    """Synchronize a stateful schedule before each PPO optimizer update."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)

    def _sync(self) -> None:
        schedules = _model_schedules(self.model)
        if not schedules:
            raise ValueError(
                "model does not contain an AbsoluteLearningRateSchedule"
            )
        for schedule in schedules:
            schedule.sync(int(self.model.num_timesteps))
        self.model._update_learning_rate(self.model.policy.optimizer)

    def _on_training_start(self) -> None:
        self._sync()

    def _on_rollout_end(self) -> None:
        self._sync()

    def _on_step(self) -> bool:
        return True
