"""Exact command and CLI contract for staged schema-3 ablations."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from run_ablation import (
    ABLATIONS,
    CONTROLLED_TRAIN_FLAGS,
    STAGES,
    ExperimentSpec,
    HyperparameterSpec,
    build_ablation_commands,
    main,
)


SCENARIO_PATH = "./data/fixed_eval_scenarios.json"
SCREENING_GRID = [
    HyperparameterSpec(0.98, 512),
    HyperparameterSpec(0.98, 960),
    HyperparameterSpec(0.995, 512),
    HyperparameterSpec(0.995, 960),
]


def value_after(command: list[str], flag: str) -> str:
    index = command.index(flag)
    return command[index + 1]


def assert_exactly_once(command: list[str], flag: str) -> None:
    assert command.count(flag) == 1


def test_ablation_rows_match_approved_state_and_grid_matrix():
    assert ABLATIONS == {
        "A": ExperimentSpec("structured", "current"),
        "B": ExperimentSpec("structured", "full"),
        "C": ExperimentSpec("fixed-grid", "full"),
        "D": ExperimentSpec("candidate-cnn", "current"),
        "E": ExperimentSpec("candidate-cnn", "full"),
    }


def test_experiment_and_hyperparameter_specs_are_frozen():
    experiment = ExperimentSpec("structured", "current")
    hyperparameters = HyperparameterSpec(0.98, 960)

    with pytest.raises(dataclasses.FrozenInstanceError):
        experiment.extractor = "candidate-cnn"
    with pytest.raises(dataclasses.FrozenInstanceError):
        hyperparameters.n_steps = 512


def test_stage_budgets_and_default_seeds_are_exact():
    assert STAGES == {
        "smoke": (20_000, (0,)),
        "screening": (300_000, (0, 1, 2)),
        "final": (1_000_000, (0, 1, 2, 3, 4)),
    }


def test_smoke_builds_exactly_five_default_commands():
    commands = build_ablation_commands(
        stage="smoke",
        seeds=[0],
        hyperparameters=[HyperparameterSpec(0.98, 960)],
        common_args=[],
    )

    assert len(commands) == 5
    assert all(value_after(cmd, "--timesteps") == "20000" for cmd in commands)
    assert all(value_after(cmd, "--seed") == "0" for cmd in commands)
    assert all(value_after(cmd, "--gae-lambda") == "0.98" for cmd in commands)
    assert all(value_after(cmd, "--n-steps") == "960" for cmd in commands)


def test_screening_builds_sixty_equal_budget_commands():
    commands = build_ablation_commands(
        stage="screening",
        seeds=[0, 1, 2],
        hyperparameters=SCREENING_GRID,
        common_args=[],
    )

    assert len(commands) == 5 * 3 * 4
    assert all(value_after(cmd, "--timesteps") == "300000" for cmd in commands)
    assert len({value_after(cmd, "--output-dir") for cmd in commands}) == 60


def test_final_builds_twenty_five_full_budget_commands():
    selected = HyperparameterSpec(0.995, 512)
    commands = build_ablation_commands(
        stage="final",
        seeds=[0, 1, 2, 3, 4],
        hyperparameters=[selected],
        common_args=[],
    )

    assert len(commands) == 25
    assert all(value_after(cmd, "--timesteps") == "1000000" for cmd in commands)
    assert {int(value_after(cmd, "--seed")) for cmd in commands} == set(range(5))
    assert all("--final-holdout-report" in cmd for cmd in commands)


@pytest.mark.parametrize("hyperparameters", [[], SCREENING_GRID[:2]])
def test_final_rejects_anything_other_than_one_hyperparameter_pair(
    hyperparameters: list[HyperparameterSpec],
):
    with pytest.raises(ValueError, match="exactly one"):
        build_ablation_commands(
            "final", [0, 1, 2, 3, 4], hyperparameters, []
        )


def test_new_contract_rejects_omitted_common_args():
    with pytest.raises(TypeError, match="common_args is required"):
        build_ablation_commands(
            "smoke", [0], [HyperparameterSpec(0.98, 960)]
        )


def test_commands_have_deterministic_order_and_collision_free_exact_paths():
    commands = build_ablation_commands(
        "screening", [0, 1, 2], SCREENING_GRID, []
    )

    assert value_after(commands[0], "--output-dir") == (
        "output_ablation/screening/A/lambda_0.98/nsteps_512/seed_0"
    )
    assert value_after(commands[1], "--output-dir") == (
        "output_ablation/screening/A/lambda_0.98/nsteps_512/seed_1"
    )
    assert value_after(commands[-1], "--output-dir") == (
        "output_ablation/screening/E/lambda_0.995/nsteps_960/seed_2"
    )
    assert len({value_after(cmd, "--output-dir") for cmd in commands}) == len(
        commands
    )


def test_every_command_has_one_exact_controlled_contract():
    common_args = ["--data-dir", "./data", "--device", "cpu"]
    commands = build_ablation_commands(
        "screening", [0, 1, 2], SCREENING_GRID, common_args
    )
    value_flags = (
        "--extractor",
        "--state-context",
        "--timesteps",
        "--seed",
        "--gae-lambda",
        "--n-steps",
        "--eval-scenarios",
        "--output-dir",
        "--checkpoint-freq",
        "--holdout-eval-freq",
    )

    for command in commands:
        assert command[:2] == [sys.executable, "train.py"]
        assert command[2:6] == common_args
        for flag in value_flags:
            assert_exactly_once(command, flag)
        assert value_after(command, "--eval-scenarios") == SCENARIO_PATH
        assert value_after(command, "--checkpoint-freq") == "50000"
        assert value_after(command, "--holdout-eval-freq") == "50000"
        assert_exactly_once(command, "--auto-resume")
        assert_exactly_once(command, "--no-export-onnx")
        assert "--final-holdout-report" not in command

    assert {value_after(cmd, "--data-dir") for cmd in commands} == {"./data"}
    assert {value_after(cmd, "--eval-scenarios") for cmd in commands} == {
        SCENARIO_PATH
    }


@pytest.mark.parametrize("flag", sorted(CONTROLLED_TRAIN_FLAGS))
def test_common_args_cannot_override_matrix_controlled_options(flag):
    with pytest.raises(ValueError, match="controlled"):
        build_ablation_commands(
            "smoke",
            [0],
            [HyperparameterSpec(0.98, 960)],
            [f"{flag}=override"],
        )


def test_dry_run_prints_five_shell_usable_smoke_commands(capsys):
    with (
        patch.object(
            sys,
            "argv",
            ["run_ablation.py", "--stage", "smoke", "--dry-run"],
        ),
        patch("run_ablation.subprocess.run") as run,
    ):
        main()

    output = capsys.readouterr().out.splitlines()
    assert len(output) == 5
    assert all(
        "train.py" in line and "--timesteps 20000" in line
        for line in output
    )
    run.assert_not_called()


@pytest.mark.parametrize(
    "argv",
    [
        ["run_ablation.py", "--stage", "final"],
        [
            "run_ablation.py",
            "--stage",
            "final",
            "--selected-gae-lambda",
            "0.98",
        ],
        [
            "run_ablation.py",
            "--stage",
            "smoke",
            "--selected-gae-lambda",
            "0.98",
            "--selected-n-steps",
            "960",
        ],
        ["run_ablation.py"],
        [
            "run_ablation.py",
            "--stage",
            "final",
            "--selected-gae-lambda",
            "0.5",
            "--selected-n-steps",
            "960",
        ],
        [
            "run_ablation.py",
            "--stage",
            "final",
            "--selected-gae-lambda",
            "0.98",
            "--selected-n-steps",
            "128",
        ],
    ],
)
def test_cli_rejects_invalid_stage_selection_combinations(argv):
    with patch.object(sys, "argv", argv), pytest.raises(SystemExit):
        main()


def test_prepare_only_remains_useful_without_a_stage(tmp_path):
    scenario_path = tmp_path / "fixed.json"
    with (
        patch.object(
            sys,
            "argv",
            [
                "run_ablation.py",
                "--prepare-eval-scenarios",
                "--scenario-path",
                str(scenario_path),
            ],
        ),
        patch("run_ablation.prepare_evaluation_file") as prepare,
    ):
        main()

    prepare.assert_called_once_with(Path("./data"), scenario_path)


def test_non_dry_run_executes_with_current_python_interpreter():
    with (
        patch.object(sys, "argv", ["run_ablation.py", "--stage", "smoke"]),
        patch("run_ablation.subprocess.run") as run,
    ):
        main()

    assert run.call_count == 5
    for call in run.call_args_list:
        command = call.args[0]
        assert command[0] == sys.executable
        assert call.kwargs == {"check": True}


def test_printed_command_round_trips_as_windows_command_line():
    command = build_ablation_commands(
        "smoke", [0], [HyperparameterSpec(0.98, 960)], []
    )[0]

    rendered = subprocess.list2cmdline(command)

    assert sys.executable in rendered
    assert "--output-dir" in rendered
