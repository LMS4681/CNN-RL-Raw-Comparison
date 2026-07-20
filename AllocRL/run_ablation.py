"""Prepare fixed scenarios and run the approved A-E ablation matrix."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import cast


@dataclass(frozen=True, eq=False)
class ExperimentSpec:
    extractor: str
    state_context: str

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ExperimentSpec):
            return (
                self.extractor == other.extractor
                and self.state_context == other.state_context
            )
        if isinstance(other, tuple):
            return (self.extractor, self.state_context) == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.extractor, self.state_context))


@dataclass(frozen=True)
class HyperparameterSpec:
    gae_lambda: float
    n_steps: int


ABLATIONS = {
    "A": ExperimentSpec("structured", "current"),
    "B": ExperimentSpec("structured", "full"),
    "C": ExperimentSpec("fixed-grid", "full"),
    "D": ExperimentSpec("candidate-cnn", "current"),
    "E": ExperimentSpec("candidate-cnn", "full"),
}
STAGES = {
    "smoke": (20_000, (0,)),
    "screening": (300_000, (0, 1, 2)),
    "final": (1_000_000, (0, 1, 2, 3, 4)),
}
SMOKE_HYPERPARAMETERS = (HyperparameterSpec(0.98, 960),)
SCREENING_HYPERPARAMETERS = (
    HyperparameterSpec(0.98, 512),
    HyperparameterSpec(0.98, 960),
    HyperparameterSpec(0.995, 512),
    HyperparameterSpec(0.995, 960),
)
FIXED_SCENARIO_PATH = "./data/fixed_eval_scenarios.json"
CONTROLLED_TRAIN_FLAGS = frozenset({
    "--extractor",
    "--state-context",
    "--timesteps",
    "--seed",
    "--gae-lambda",
    "--n-steps",
    "--eval-scenarios",
    "--output-dir",
    "--auto-resume",
    "--checkpoint-freq",
    "--holdout-eval-freq",
    "--export-onnx",
    "--no-export-onnx",
    "--final-holdout-report",
})
BLOCK_SOURCE_FILENAME = "\ube14\ub85d\ub370\uc774\ud130.csv"


def _block_source_path(data_dir: Path) -> Path:
    source_path = data_dir / BLOCK_SOURCE_FILENAME
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Evaluation source file does not exist: {source_path}"
        )
    lines = source_path.read_bytes().splitlines()
    if not lines or b"STAGE" not in lines[0]:
        raise ValueError(
            f"Evaluation source file is invalid: {source_path}"
        )
    return source_path


def build_ablation_commands(
    stage: str,
    seeds: list[int],
    hyperparameters: list[HyperparameterSpec] | list[str],
    common_args: list[str] | None = None,
) -> list[list[str]]:
    if common_args is None:
        if not all(isinstance(argument, str) for argument in hyperparameters):
            raise TypeError(
                "common_args is required by the staged ablation contract"
            )
        return _build_legacy_ablation_commands(
            stage, seeds, cast(list[str], hyperparameters)
        )
    if stage not in STAGES:
        raise ValueError("stage must be 'smoke', 'screening', or 'final'")
    _validate_common_args(common_args)
    selected = tuple(cast(list[HyperparameterSpec], hyperparameters))
    if stage == "smoke" and selected != SMOKE_HYPERPARAMETERS:
        raise ValueError("smoke requires the default (0.98, 960) pair")
    if stage == "screening" and selected != SCREENING_HYPERPARAMETERS:
        raise ValueError("screening requires all four approved pairs")
    if stage == "final" and len(selected) != 1:
        raise ValueError("final requires exactly one hyperparameter pair")
    if stage == "final" and selected[0] not in SCREENING_HYPERPARAMETERS:
        raise ValueError("final hyperparameters must be selected from screening")
    if not seeds:
        raise ValueError("At least one seed is required")

    timesteps = STAGES[stage][0]
    commands: list[list[str]] = []
    for label, experiment in ABLATIONS.items():
        for hyperparameter in selected:
            lambda_value = str(hyperparameter.gae_lambda)
            for seed in seeds:
                output = (
                    f"output_ablation/{stage}/{label}/"
                    f"lambda_{lambda_value}/nsteps_{hyperparameter.n_steps}/"
                    f"seed_{seed}"
                )
                command = [
                    sys.executable,
                    "train.py",
                    *common_args,
                    "--extractor",
                    experiment.extractor,
                    "--state-context",
                    experiment.state_context,
                    "--timesteps",
                    str(timesteps),
                    "--seed",
                    str(seed),
                    "--gae-lambda",
                    lambda_value,
                    "--n-steps",
                    str(hyperparameter.n_steps),
                    "--eval-scenarios",
                    FIXED_SCENARIO_PATH,
                    "--output-dir",
                    output,
                    "--auto-resume",
                    "--checkpoint-freq",
                    "50000",
                    "--holdout-eval-freq",
                    "50000",
                    "--no-export-onnx",
                ]
                if stage == "final":
                    command.append("--final-holdout-report")
                commands.append(command)
    return commands


def _validate_common_args(common_args: list[str]) -> None:
    conflicts = sorted({
        argument.split("=", 1)[0]
        for argument in common_args
        if argument.split("=", 1)[0] in CONTROLLED_TRAIN_FLAGS
    })
    if conflicts:
        raise ValueError(
            "common_args contains matrix-controlled options: "
            + ", ".join(conflicts)
        )


def _build_legacy_ablation_commands(
    mode: str, seeds: list[int], common_args: list[str]
) -> list[list[str]]:
    """Keep the Stage-A helper contract; the CLI no longer uses this path."""
    if mode not in {"screening", "final"}:
        raise ValueError("mode must be 'screening' or 'final'")
    timesteps = 20_000 if mode == "screening" else 100_000
    commands: list[list[str]] = []
    for seed in seeds:
        for label, experiment in ABLATIONS.items():
            output = f"./output_ablation/{mode}/{label}/seed_{seed}"
            commands.append(
                [
                    sys.executable,
                    "train.py",
                    *common_args,
                    "--timesteps",
                    str(timesteps),
                    "--extractor",
                    experiment.extractor,
                    "--state-context",
                    experiment.state_context,
                    "--seed",
                    str(seed),
                    "--output-dir",
                    output,
                    "--eval-scenarios",
                    FIXED_SCENARIO_PATH,
                    "--no-export-onnx",
                ]
            )
    return commands


def build_baseline_command(
    scenario_path: str, output_path: str
) -> list[str]:
    return [
        sys.executable,
        "evaluate_baselines.py",
        "--scenarios",
        scenario_path,
        "--output",
        output_path,
    ]


def prepare_evaluation_file(data_dir: Path, output_path: Path) -> None:
    from alloc_env.block_generator import BlockDistribution
    from alloc_env.data_split import split_blocks_by_ship, write_split_manifest
    from alloc_env.strategy import BaseGridStrategy
    from evaluation_scenarios import generate_scenarios, write_scenarios
    from train import (
        DEFAULT_ACTIVE_WORKSPACE_CODES,
        load_allocation_scenario,
        parse_workspace_codes,
    )

    source_path = _block_source_path(data_dir)
    strategy = BaseGridStrategy(step=5.0)
    csv_blocks, active = load_allocation_scenario(
        data_dir,
        strategy,
        parse_workspace_codes(DEFAULT_ACTIVE_WORKSPACE_CODES),
    )
    split = split_blocks_by_ship(csv_blocks, source_path)
    target_month_counts = Counter(
        (block.in_date.year, block.in_date.month) for block in csv_blocks
    )
    base_date = min(block.in_date for block in csv_blocks)
    spread_days = max(
        (max(block.in_date for block in csv_blocks) - base_date).days,
        1,
    )
    scenarios = generate_scenarios(
        distribution=BlockDistribution.from_blocks(split.holdout_blocks),
        workspaces=active,
        seeds=list(range(1000, 1020)),
        n_blocks=len(csv_blocks),
        base_date=base_date,
        spread_days=spread_days,
        source_blocks=list(split.holdout_blocks),
        target_month_counts=target_month_counts,
        vary_layout=False,
        empirical_profile_probability=1.0,
        source_name="holdout_fixed",
    )
    metadata = {
        **split.manifest,
        "source": "holdout_fixed",
        "source_ship_count": len({block.ship_no for block in csv_blocks}),
        "training_ship_count": len(split.manifest["training_ship_nos"]),
        "holdout_ship_count": len(split.manifest["holdout_ship_nos"]),
        "target_month_counts": {
            f"{year:04d}-{month:02d}": count
            for (year, month), count in sorted(target_month_counts.items())
        },
        "provenance": {
            "template_source": "holdout_ship_split",
            "vary_layout": False,
            "empirical_profile_probability": 1.0,
            "scenario_count": len(scenarios),
            "scenario_block_count": len(csv_blocks),
            "workspace_count": len(active),
        },
    }
    manifest_path = output_path.with_name("data_split_manifest.json")
    write_scenarios(output_path, scenarios, metadata)
    write_split_manifest(manifest_path, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AllocRL A-E ablations")
    parser.add_argument(
        "--stage", choices=tuple(STAGES), default=None
    )
    parser.add_argument(
        "--selected-gae-lambda",
        type=float,
        choices=[0.98, 0.995],
        default=None,
    )
    parser.add_argument(
        "--selected-n-steps",
        type=int,
        choices=[512, 960],
        default=None,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-eval-scenarios", action="store_true")
    parser.add_argument("--evaluate-baselines", action="store_true")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument(
        "--scenario-path", default="./data/fixed_eval_scenarios.json"
    )
    args, extra_args = parser.parse_known_args()

    if args.prepare_eval_scenarios:
        output_path = Path(args.scenario_path)
        prepare_evaluation_file(Path(args.data_dir), output_path)
        print(f"Fixed evaluation scenarios saved to: {output_path}")
        print(
            "Data split manifest saved to: "
            f"{output_path.with_name('data_split_manifest.json')}"
        )
        return

    if args.evaluate_baselines:
        command = build_baseline_command(
            args.scenario_path,
            "./output_ablation/baselines/evaluation_scenarios.csv",
        )
        subprocess.run(command, check=True)
        return

    if args.stage is None:
        parser.error(
            "--stage is required unless preparing scenarios or evaluating baselines"
        )
    selected_values = (
        args.selected_gae_lambda,
        args.selected_n_steps,
    )
    if args.stage != "final" and any(
        value is not None for value in selected_values
    ):
        parser.error("selected hyperparameters are valid only for final stage")
    if args.stage == "final" and any(
        value is None for value in selected_values
    ):
        parser.error(
            "final stage requires --selected-gae-lambda and --selected-n-steps"
        )

    if args.stage == "smoke":
        hyperparameters = list(SMOKE_HYPERPARAMETERS)
    elif args.stage == "screening":
        hyperparameters = list(SCREENING_HYPERPARAMETERS)
    else:
        hyperparameters = [HyperparameterSpec(*selected_values)]

    seeds = list(STAGES[args.stage][1])
    common_args = ["--data-dir", args.data_dir, *extra_args]
    try:
        commands = build_ablation_commands(
            args.stage, seeds, hyperparameters, common_args
        )
    except ValueError as error:
        parser.error(str(error))
    for command in commands:
        print(subprocess.list2cmdline(command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
