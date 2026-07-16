"""Prepare fixed scenarios and run the approved A-E ablation matrix."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ABLATIONS = {
    "A": ("structured", 0),
    "B": ("structured", 4),
    "C": ("fixed-grid", 4),
    "D": ("candidate-cnn", 0),
    "E": ("candidate-cnn", 4),
}


def build_ablation_commands(
    mode: str,
    seeds: list[int],
    common_args: list[str],
) -> list[list[str]]:
    if mode not in {"screening", "final"}:
        raise ValueError("mode must be 'screening' or 'final'")
    timesteps = 20_000 if mode == "screening" else 100_000
    commands = []
    for seed in seeds:
        for label, (extractor, horizon) in ABLATIONS.items():
            output = f"./output_ablation/{mode}/{label}/seed_{seed}"
            commands.append(
                [
                    sys.executable,
                    "train.py",
                    *common_args,
                    "--timesteps",
                    str(timesteps),
                    "--extractor",
                    extractor,
                    "--n-future-blocks",
                    str(horizon),
                    "--seed",
                    str(seed),
                    "--output-dir",
                    output,
                    "--eval-scenarios",
                    "./data/fixed_eval_scenarios.json",
                    "--no-export-onnx",
                ]
            )
    return commands


def prepare_evaluation_file(data_dir: Path, output_path: Path) -> None:
    from alloc_env.block_generator import BlockDistribution
    from alloc_env.strategy import BaseGridStrategy
    from evaluation_scenarios import generate_scenarios, write_scenarios
    from train import (
        DEFAULT_ACTIVE_WORKSPACE_CODES,
        load_allocation_scenario,
        parse_workspace_codes,
    )

    strategy = BaseGridStrategy(step=5.0)
    csv_blocks, active = load_allocation_scenario(
        data_dir,
        strategy,
        parse_workspace_codes(DEFAULT_ACTIVE_WORKSPACE_CODES),
    )
    distribution = BlockDistribution.from_blocks(csv_blocks)
    base_date = min(block.in_date for block in csv_blocks)
    spread_days = max(
        (max(block.in_date for block in csv_blocks) - base_date).days,
        1,
    )
    scenarios = generate_scenarios(
        distribution=distribution,
        workspaces=active,
        seeds=list(range(1000, 1020)),
        n_blocks=len(csv_blocks),
        base_date=base_date,
        spread_days=spread_days,
        source_blocks=csv_blocks,
        vary_layout=False,
        empirical_profile_probability=1.0,
    )
    write_scenarios(output_path, scenarios)


def _parse_seeds(value: str | None, mode: str) -> list[int]:
    if value is None:
        return [0, 1, 2] if mode == "screening" else [0, 1, 2, 3, 4]
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AllocRL A-E ablations")
    parser.add_argument(
        "--mode", choices=["screening", "final"], default="screening"
    )
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-eval-scenarios", action="store_true")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument(
        "--scenario-path", default="./data/fixed_eval_scenarios.json"
    )
    args, extra_args = parser.parse_known_args()

    if args.prepare_eval_scenarios:
        output_path = Path(args.scenario_path)
        prepare_evaluation_file(Path(args.data_dir), output_path)
        print(f"Fixed evaluation scenarios saved to: {output_path.resolve()}")
        return

    seeds = _parse_seeds(args.seeds, args.mode)
    common_args = ["--data-dir", args.data_dir, *extra_args]
    commands = build_ablation_commands(args.mode, seeds, common_args)
    for command in commands:
        if args.dry_run:
            print(subprocess.list2cmdline(command))
        else:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
