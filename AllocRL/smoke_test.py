"""Short schema-4 train/save/load/evaluate workflows for every extractor."""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO

from alloc_env.alloc_env import DROPOUT_THRESHOLD
from alloc_env.callbacks import CnnDiagnosticTracker
from alloc_env.observation_state import (
    CURRENT_BLOCK_FEATURE_DIM,
    FUTURE_BLOCK_FEATURE_DIM,
    FUTURE_DEMAND_FEATURE_DIM,
    GRID_SIZE,
    N_WORKSPACES,
    ORDERED_FUTURE_COUNT,
    PENDING_BLOCK_FEATURE_DIM,
    PENDING_QUEUE_SLOTS,
    PENDING_SUMMARY_FEATURE_DIM,
    WORKSPACE_META_FEATURE_DIM,
    build_observation_scales,
)
from alloc_env.strategy import BaseGridStrategy
from train import (
    DEFAULT_ACTIVE_WORKSPACE_CODES,
    build_policy_kwargs,
    create_evaluation_env,
    evaluate,
    load_allocation_scenario,
    parse_workspace_codes,
)


BASE_DIR = Path(__file__).resolve().parent
EXTRACTORS = ("structured", "fixed-grid", "candidate-cnn", "raw-direct")
SMOKE_ROLLOUT_STEPS = 32
SCHEMA4_OBSERVATION_SHAPES = {
    "block": (CURRENT_BLOCK_FEATURE_DIM,),
    "grids": (N_WORKSPACES, 4, GRID_SIZE, GRID_SIZE),
    "ws_meta": (N_WORKSPACES, WORKSPACE_META_FEATURE_DIM),
    "future_blocks": (
        ORDERED_FUTURE_COUNT,
        FUTURE_BLOCK_FEATURE_DIM,
    ),
    "future_mask": (ORDERED_FUTURE_COUNT,),
    "future_demand": (3, FUTURE_DEMAND_FEATURE_DIM),
    "pending_blocks": (
        N_WORKSPACES,
        PENDING_QUEUE_SLOTS,
        PENDING_BLOCK_FEATURE_DIM,
    ),
    "pending_mask": (N_WORKSPACES, PENDING_QUEUE_SLOTS),
    "pending_summary": (
        N_WORKSPACES,
        PENDING_SUMMARY_FEATURE_DIM,
    ),
}


def validate_schema4_observation_space(
    observation_space: gym.spaces.Dict,
) -> None:
    """Fail early if a smoke environment does not expose exact schema 4."""
    assert isinstance(observation_space, gym.spaces.Dict), (
        "schema-4 observation space must be gym.spaces.Dict"
    )
    actual_keys = set(observation_space.spaces)
    expected_keys = set(SCHEMA4_OBSERVATION_SHAPES)
    assert actual_keys == expected_keys, (
        "schema-4 keys differ: "
        f"expected {sorted(expected_keys)}, got {sorted(actual_keys)}"
    )
    for key, expected_shape in SCHEMA4_OBSERVATION_SHAPES.items():
        space = observation_space.spaces[key]
        assert space.shape == expected_shape, (
            f"{key} must have shape {expected_shape}, got {space.shape}"
        )
        assert space.dtype == np.dtype(np.float32), (
            f"{key} must use np.float32, got {space.dtype}"
        )


def validate_cnn_diagnostics(diagnostics: dict[str, float]) -> None:
    """Require evidence that PPO updated candidate-CNN parameters."""
    gradient_norm = float(diagnostics.get("cnn_gradient_norm", 0.0))
    weight_change = float(diagnostics.get("cnn_weight_change", 0.0))
    assert math.isfinite(gradient_norm) and gradient_norm > 0.0, (
        f"candidate CNN gradient norm must be positive, got {gradient_norm}"
    )
    assert math.isfinite(weight_change) and weight_change > 0.0, (
        f"candidate CNN weight change must be positive, got {weight_change}"
    )


def _build_smoke_environment(seed: int = 0):
    strategy = BaseGridStrategy(step=5.0)
    blocks, workspaces = load_allocation_scenario(
        BASE_DIR / "data",
        strategy,
        parse_workspace_codes(DEFAULT_ACTIVE_WORKSPACE_CODES),
    )
    observation_scales = build_observation_scales(
        blocks,
        workspaces,
        DROPOUT_THRESHOLD,
    )
    env = create_evaluation_env(
        blocks=blocks,
        workspaces=workspaces,
        strategy=strategy,
        observation_scales=observation_scales,
        grid_size=GRID_SIZE,
        state_context_mode="full",
        seed=seed,
    )
    validate_schema4_observation_space(env.observation_space)
    return env


def train_tiny_model(
    *,
    extractor: str,
    timesteps: int,
    device: str,
):
    """Train one extractor through real MaskablePPO actor/critic losses."""
    if extractor not in EXTRACTORS:
        raise ValueError(
            f"unknown extractor {extractor!r}; choose one of {EXTRACTORS}"
        )
    if timesteps < 2:
        raise ValueError("timesteps must be at least 2")

    env = _build_smoke_environment(seed=0)
    n_steps = min(SMOKE_ROLLOUT_STEPS, timesteps)
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=build_policy_kwargs(
            extractor=extractor,
            features_dim=256,
        ),
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=n_steps,
        n_epochs=1,
        seed=0,
        device=device,
        verbose=0,
    )

    tracker = None
    if extractor == "candidate-cnn":
        tracker = CnnDiagnosticTracker(model.policy.features_extractor)
        tracker.attach()
    try:
        model.learn(total_timesteps=timesteps, progress_bar=False)
        if tracker is not None:
            diagnostics = tracker.record_update()
            validate_cnn_diagnostics(diagnostics)
            print(
                "[candidate-cnn] "
                f"cnn_gradient_norm={diagnostics['cnn_gradient_norm']:.9f} "
                f"cnn_weight_change={diagnostics['cnn_weight_change']:.9f}"
            )
    finally:
        if tracker is not None:
            tracker.close()

    return model, env


def run_extractor_smoke(
    extractor: str,
    output_dir: Path,
    *,
    timesteps: int = 1_024,
    device: str = "cpu",
) -> dict[str, float]:
    """Train, save, load, and complete one evaluation for an extractor."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, env = train_tiny_model(
        extractor=extractor,
        timesteps=timesteps,
        device=device,
    )
    path = output_dir / f"{extractor}.sb3"
    try:
        model.save(path)
        loaded = MaskablePPO.load(path, env=env)
        metrics = evaluate(loaded, env, n_eval=1, return_metrics=True)
        terminal_score = float(metrics["mean_terminal_score"])
        assert math.isfinite(terminal_score), (
            f"{extractor} must produce a finite terminal score, "
            f"got {terminal_score}"
        )
        print(
            f"[{extractor}] trained={timesteps} saved={path.name} "
            f"terminal_score={terminal_score:.6f}"
        )
        return metrics
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()


def _positive_timesteps(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("timesteps must be at least 2")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run schema-4 extractor train/save/load/evaluate smoke checks"
    )
    parser.add_argument(
        "--extractor",
        choices=EXTRACTORS,
        default="candidate-cnn",
    )
    parser.add_argument(
        "--all-extractors",
        action="store_true",
        help="run structured, fixed-grid, candidate-cnn, and raw-direct",
    )
    parser.add_argument(
        "--timesteps",
        type=_positive_timesteps,
        default=1_024,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def _run_selected_extractors(
    extractors: Sequence[str],
    output_dir: Path,
    timesteps: int,
    device: str,
) -> None:
    for extractor in extractors:
        run_extractor_smoke(
            extractor,
            output_dir,
            timesteps=timesteps,
            device=device,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    extractors = EXTRACTORS if args.all_extractors else (args.extractor,)

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        _run_selected_extractors(
            extractors, output_dir, args.timesteps, args.device
        )
        return 0

    with tempfile.TemporaryDirectory(prefix="allocrl-schema4-smoke-") as tmp:
        _run_selected_extractors(
            extractors, Path(tmp), args.timesteps, args.device
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
