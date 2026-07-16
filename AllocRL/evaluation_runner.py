"""Shared evaluation loop for model and heuristic action policies."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import numpy as np

from alloc_env.alloc_env import DELAY_THRESHOLD
from alloc_env.data_loader import select_workspaces_in_order
from alloc_env.simulator import SimulationResult
from alloc_env.strategy import BaseGridStrategy
from baseline_policies import ActionPolicy
from evaluation_scenarios import (
    compute_retained_choice_ratio,
    materialize_scenario,
)


class ModelActionPolicy:
    def __init__(self, model, name: str = "model"):
        self.model = model
        self.name = name

    def select_action(self, env, observation) -> int:
        action, _ = self.model.predict(
            observation,
            action_masks=env.action_masks(),
            deterministic=True,
        )
        return int(np.asarray(action).item())


def evaluate_policy(
    policy: ActionPolicy,
    env,
    episodes: int = 1,
    collect_retained_choice: bool = True,
) -> dict[str, float]:
    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    values = {
        "reward": [],
        "terminal": [],
        "dropout": [],
        "delay": [],
        "delayed": [],
        "retained": [],
    }
    for _episode in range(episodes):
        observation, _ = env.reset()
        diagnostic = getattr(env, "unwrapped", env)
        total_reward = 0.0
        ratios = []
        done = False
        while not done:
            indices = (
                diagnostic.future_workspace_choice_indices()
                if collect_retained_choice
                and hasattr(diagnostic, "future_workspace_choice_indices")
                else []
            )
            before = (
                diagnostic.future_workspace_choice_count(indices)
                if indices
                else 0
            )
            action = policy.select_action(diagnostic, observation)
            after = (
                diagnostic.future_workspace_choice_count_after_action(
                    action, indices
                )
                if indices
                and hasattr(
                    diagnostic,
                    "future_workspace_choice_count_after_action",
                )
                else 0
            )
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if collect_retained_choice:
                ratios.append(compute_retained_choice_ratio(before, after))
            done = bool(terminated or truncated)

        result = info.get("raw_result")
        delay_days = list(result.delay_days) if result is not None else []
        dropout_count = sum(
            value == SimulationResult.DROPOUT for value in delay_days
        )
        placed = [
            value for value in delay_days
            if value != SimulationResult.DROPOUT
        ]
        values["reward"].append(total_reward)
        values["terminal"].append(float(info.get(
            "terminal_score", info.get("terminal_reward", total_reward)
        )))
        values["dropout"].append(
            dropout_count / len(delay_days) if delay_days else 0.0
        )
        values["delay"].append(float(np.mean(placed)) if placed else 0.0)
        values["delayed"].append(
            float(sum(value > DELAY_THRESHOLD for value in placed))
        )
        values["retained"].append(
            float(np.mean(ratios)) if ratios else 1.0
        )

    return {
        "mean_reward": float(np.mean(values["reward"])),
        "mean_terminal_score": float(np.mean(values["terminal"])),
        "mean_dropout_rate": float(np.mean(values["dropout"])),
        "mean_delay_days": float(np.mean(values["delay"])),
        "mean_delayed_count": float(np.mean(values["delayed"])),
        "mean_retained_choice_ratio": float(np.mean(values["retained"])),
    }


def evaluate_scenarios(
    policy_factory: Callable[[int], ActionPolicy],
    scenarios: list[dict],
    grid_size: int,
    n_future_blocks: int,
    workspace_codes: list[str] | None,
) -> list[dict]:
    from train import create_evaluation_env

    rows = []
    for scenario in scenarios:
        seed = int(scenario["seed"])
        strategy = BaseGridStrategy(step=5.0)
        blocks, workspaces = materialize_scenario(scenario, strategy)
        ordered = select_workspaces_in_order(workspaces, workspace_codes)
        env = create_evaluation_env(
            blocks=blocks,
            workspaces=ordered,
            strategy=strategy,
            grid_size=grid_size,
            n_future_blocks=n_future_blocks,
            seed=seed,
        )
        try:
            policy = policy_factory(seed)
            metrics = evaluate_policy(policy, env, episodes=1)
            rows.append({
                "source": "holdout_fixed20",
                "policy": policy.name,
                "seed": seed,
                **metrics,
            })
        finally:
            env.close()
    return rows


def write_evaluation_metrics(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("At least one evaluation metric row is required")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
