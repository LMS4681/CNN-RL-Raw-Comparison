"""Eight-subprocess release rehearsal for transfer, freeze, and resume."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv

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
from pretraining.two_stage_smoke import _SmokeEnv


N_ENVS = 8
N_STEPS = 120
ROLLOUT_TRANSITIONS = N_ENVS * N_STEPS
FREEZE_UNTIL = 1_440
TOTAL_TIMESTEPS = 3 * ROLLOUT_TRANSITIONS


def _make_vector_env() -> SubprocVecEnv:
    return SubprocVecEnv([_SmokeEnv for _ in range(N_ENVS)])


def _clone_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _state_changed(
    before: dict[str, torch.Tensor], module: torch.nn.Module
) -> bool:
    return any(
        not torch.equal(before[name], value.detach().cpu())
        for name, value in module.state_dict().items()
    )


def _callbacks() -> list:
    return [
        AbsoluteScheduleCallback(),
        ExtractorFineTuneCallback(FREEZE_UNTIL),
    ]


def _policy_rate(model: ScaleAwareMaskablePPO) -> float:
    groups = {
        str(group["name"]): group
        for group in model.policy.optimizer.param_groups
    }
    return float(groups["policy"]["lr"])


def _extractor_rate_ratio(model: ScaleAwareMaskablePPO) -> float:
    groups = {
        str(group["name"]): group
        for group in model.policy.optimizer.param_groups
    }
    return float(groups["extractor"]["lr"]) / float(
        groups["policy"]["lr"]
    )


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def run_release_rehearsal(
    checkpoint_path: Path,
    complete_path: Path,
    output_dir: Path,
    *,
    device: str = "cpu",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule = AbsoluteLearningRateSchedule(
        mode="linear",
        initial_rate=1e-3,
        final_rate=1e-4,
        decay_steps=TOTAL_TIMESTEPS,
    )
    env = _make_vector_env()
    model = ScaleAwareMaskablePPO(
        "MultiInputPolicy",
        env,
        learning_rate=schedule,
        extractor_lr_scale=0.1,
        n_steps=N_STEPS,
        batch_size=64,
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
        total_timesteps=ROLLOUT_TRANSITIONS,
        callback=_callbacks(),
        progress_bar=False,
    )
    first_timestep = int(model.num_timesteps)
    unchanged_while_frozen = not _state_changed(transferred, extractor)
    frozen_gradients_clear = all(
        parameter.grad is None for parameter in extractor.parameters()
    )
    learning_rates = [_policy_rate(model)]
    before_path = output_dir / "before_boundary.sb3"
    model.save(before_path)
    model.get_env().close()

    resumed_env = _make_vector_env()
    resumed = ScaleAwareMaskablePPO.load(
        before_path, env=resumed_env, device=device
    )
    exact_before_resume = int(resumed.num_timesteps) == first_timestep
    before_unfreeze = _clone_state(resumed.policy.features_extractor)
    resumed.learn(
        total_timesteps=ROLLOUT_TRANSITIONS,
        callback=_callbacks(),
        reset_num_timesteps=False,
        progress_bar=False,
    )
    second_timestep = int(resumed.num_timesteps)
    changed_after_unfreeze = _state_changed(
        before_unfreeze, resumed.policy.features_extractor
    )
    nonzero_gradient = any(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in resumed.policy.features_extractor.parameters()
    )
    learning_rates.append(_policy_rate(resumed))
    lr_ratio = _extractor_rate_ratio(resumed)
    after_state = _clone_state(resumed.policy.features_extractor)
    after_path = output_dir / "after_boundary.sb3"
    resumed.save(after_path)
    resumed.get_env().close()

    final_env = _make_vector_env()
    final_model = ScaleAwareMaskablePPO.load(
        after_path, env=final_env, device=device
    )
    exact_after_resume = (
        int(final_model.num_timesteps) == second_timestep
        and not _state_changed(after_state, final_model.policy.features_extractor)
    )
    final_model.learn(
        total_timesteps=ROLLOUT_TRANSITIONS,
        callback=_callbacks(),
        reset_num_timesteps=False,
        progress_bar=False,
    )
    third_timestep = int(final_model.num_timesteps)
    learning_rates.append(_policy_rate(final_model))
    final_path = output_dir / "after_second_resume.sb3"
    final_model.save(final_path)
    final_model.get_env().close()

    checks = {
        "extractor_unchanged_while_frozen": unchanged_while_frozen,
        "frozen_gradients_clear": frozen_gradients_clear,
        "extractor_changed_after_unfreeze": changed_after_unfreeze,
        "nonzero_extractor_gradient": nonzero_gradient,
        "exact_before_boundary_resume": exact_before_resume,
        "exact_after_boundary_resume": exact_after_resume,
    }
    if not all(checks.values()):
        raise AssertionError(f"release rehearsal checks failed: {checks}")
    if not np.isclose(lr_ratio, 0.1):
        raise AssertionError("extractor learning-rate ratio differs from 0.1")
    if learning_rates != sorted(learning_rates, reverse=True):
        raise AssertionError("absolute learning rates are not monotonic")

    result_path = output_dir / "REHEARSAL_COMPLETE.json"
    _atomic_json(result_path, {
        "schema_version": 1,
        "n_envs": N_ENVS,
        "vec_env": "subproc",
        "n_steps": N_STEPS,
        "rollout_transitions": ROLLOUT_TRANSITIONS,
        "freeze_until_timestep": FREEZE_UNTIL,
        "timesteps": [first_timestep, second_timestep, third_timestep],
        "learning_rates": learning_rates,
        "extractor_lr_ratio": lr_ratio,
        "strict_pretraining_transfer": bool(receipt.checkpoint_sha256),
        **checks,
    })
    return result_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--complete", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    print(run_release_rehearsal(
        args.checkpoint,
        args.complete,
        args.output_dir,
        device=args.device,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
