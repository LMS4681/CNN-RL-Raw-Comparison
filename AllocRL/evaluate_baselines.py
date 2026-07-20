"""Evaluate the required heuristic baselines on fixed holdout scenarios."""

from __future__ import annotations

import argparse

import numpy as np

from alloc_env.alloc_env import DROPOUT_THRESHOLD
from alloc_env.observation_state import build_observation_scales
from alloc_env.strategy import BaseGridStrategy
from baseline_policies import GreedyImmediateAreaPolicy, RandomValidPolicy
from evaluation_runner import evaluate_scenarios, write_evaluation_metrics
from evaluation_scenarios import read_scenarios
from train import (
    DEFAULT_ACTIVE_WORKSPACE_CODES,
    load_allocation_scenario,
    parse_workspace_codes,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate heuristic policies on fixed holdout scenarios"
    )
    parser.add_argument(
        "--scenarios", default="./data/fixed_eval_scenarios.json"
    )
    parser.add_argument(
        "--output",
        default="./output_ablation/baselines/evaluation_scenarios.csv",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument(
        "--active-workspace-codes",
        default=DEFAULT_ACTIVE_WORKSPACE_CODES,
    )
    args = parser.parse_args()

    scenarios = read_scenarios(args.scenarios)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        scenarios = scenarios[:args.limit]

    workspace_codes = parse_workspace_codes(args.active_workspace_codes)
    strategy = BaseGridStrategy(step=5.0)
    full_blocks, workspaces = load_allocation_scenario(
        args.data_dir, strategy, workspace_codes
    )
    observation_scales = build_observation_scales(
        full_blocks,
        workspaces,
        DROPOUT_THRESHOLD,
    )
    factories = (
        lambda seed: RandomValidPolicy(seed),
        lambda _seed: GreedyImmediateAreaPolicy(),
    )
    rows = []
    for factory in factories:
        rows.extend(
            evaluate_scenarios(
                factory,
                scenarios,
                workspace_codes=workspace_codes,
                observation_scales=observation_scales,
                state_context_mode="full",
            )
        )

    write_evaluation_metrics(args.output, rows)
    for policy_name in sorted({row["policy"] for row in rows}):
        selected = [row for row in rows if row["policy"] == policy_name]
        print(
            policy_name,
            "score=",
            np.mean([
                float(row["mean_terminal_score"]) for row in selected
            ]),
            "dropout=",
            np.mean([
                float(row["mean_dropout_rate"]) for row in selected
            ]),
            "delay=",
            np.mean([
                float(row["mean_delay_days"]) for row in selected
            ]),
        )


if __name__ == "__main__":
    main()
