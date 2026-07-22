"""Real 32-step transfer/freeze/resume smoke used by the Colab gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
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


def _observation_space() -> gym.spaces.Dict:
    return gym.spaces.Dict({
        "block": gym.spaces.Box(0, 1, (8,), np.float32),
        "grids": gym.spaces.Box(0, 1, (10, 4, 8, 8), np.float32),
        "ws_meta": gym.spaces.Box(0, 1, (10, 8), np.float32),
        "future_blocks": gym.spaces.Box(0, 1, (16, 6), np.float32),
        "future_mask": gym.spaces.Box(0, 1, (16,), np.float32),
        "future_demand": gym.spaces.Box(0, 1, (3, 6), np.float32),
        "pending_blocks": gym.spaces.Box(0, 1, (10, 32, 7), np.float32),
        "pending_mask": gym.spaces.Box(0, 1, (10, 32), np.float32),
        "pending_summary": gym.spaces.Box(0, 1, (10, 4), np.float32),
    })


class _SmokeEnv(gym.Env):
    def __init__(self) -> None:
        self.observation_space = _observation_space()
        self.action_space = gym.spaces.Discrete(10)
        self.steps = 0

    def _observation(self) -> dict[str, np.ndarray]:
        value = 0.2 + 0.01 * self.steps
        return {
            key: np.full(space.shape, value, dtype=np.float32)
            for key, space in self.observation_space.spaces.items()
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self._observation(), {}

    def action_masks(self) -> np.ndarray:
        return np.ones(10, dtype=bool)

    def step(self, action):
        self.steps += 1
        return (
            self._observation(),
            1.0 if int(action) == self.steps % 10 else -0.1,
            self.steps >= 5,
            False,
            {},
        )


def _clone_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _changed(
    before: dict[str, torch.Tensor], module: torch.nn.Module
) -> bool:
    return any(
        not torch.equal(before[name], value.detach().cpu())
        for name, value in module.state_dict().items()
    )


def _callbacks(freeze_until: int):
    return [
        AbsoluteScheduleCallback(),
        ExtractorFineTuneCallback(freeze_until),
    ]


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def run_two_stage_smoke(
    checkpoint_path: Path,
    complete_path: Path,
    output_dir: Path,
    *,
    timesteps: int = 32,
    device: str = "cpu",
) -> Path:
    if timesteps != 32:
        raise ValueError("the production transfer smoke must use exactly 32 steps")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    freeze_until = 16
    schedule = AbsoluteLearningRateSchedule(
        mode="linear",
        initial_rate=1e-3,
        final_rate=1e-4,
        decay_steps=timesteps,
    )
    model = ScaleAwareMaskablePPO(
        "MultiInputPolicy",
        _SmokeEnv(),
        learning_rate=schedule,
        extractor_lr_scale=0.1,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        gamma=0.9,
        seed=0,
        device=device,
        verbose=0,
        policy_kwargs={
            "features_extractor_class": CandidateCnnExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
            "share_features_extractor": True,
            "net_arch": {"pi": [32], "vf": [32]},
        },
    )
    receipt = load_verified_pretrained_extractor(
        model, Path(checkpoint_path), Path(complete_path)
    )
    extractor = model.policy.features_extractor
    transferred = _clone_state(extractor)
    model.learn(
        total_timesteps=8,
        callback=_callbacks(freeze_until),
        progress_bar=False,
    )
    unchanged_while_frozen = not _changed(transferred, extractor)
    before_path = output_dir / "before_boundary.sb3"
    model.save(before_path)

    loaded = ScaleAwareMaskablePPO.load(
        before_path, env=_SmokeEnv(), device=device
    )
    loaded_extractor = loaded.policy.features_extractor
    before_unfreeze = _clone_state(loaded_extractor)
    loaded.learn(
        total_timesteps=24,
        callback=_callbacks(freeze_until),
        reset_num_timesteps=False,
        progress_bar=False,
    )
    changed_after = _changed(before_unfreeze, loaded_extractor)
    nonzero_gradient = any(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in loaded_extractor.parameters()
    )
    groups = {
        str(group["name"]): group
        for group in loaded.policy.optimizer.param_groups
    }
    lr_ratio = float(groups["extractor"]["lr"]) / float(
        groups["policy"]["lr"]
    )
    if not unchanged_while_frozen:
        raise AssertionError("extractor changed during the frozen smoke phase")
    if not changed_after or not nonzero_gradient:
        raise AssertionError("extractor did not fine-tune after unfreezing")
    if not np.isclose(lr_ratio, 0.1):
        raise AssertionError("extractor learning-rate ratio differs from 0.1")
    after_path = output_dir / "after_boundary.sb3"
    loaded.save(after_path)
    result_path = output_dir / "SMOKE_COMPLETE.json"
    _atomic_json(result_path, {
        "schema_version": 1,
        "total_timesteps": int(loaded.num_timesteps),
        "extractor_unchanged_while_frozen": unchanged_while_frozen,
        "extractor_changed_after_unfreeze": changed_after,
        "nonzero_extractor_gradient": nonzero_gradient,
        "extractor_lr_ratio": lr_ratio,
        "pretraining_checkpoint_sha256": receipt.checkpoint_sha256,
        "pretraining_complete_sha256": receipt.complete_sha256,
    })
    return result_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--complete", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    print(run_two_stage_smoke(
        args.checkpoint,
        args.complete,
        args.output_dir,
        timesteps=args.timesteps,
        device=args.device,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

