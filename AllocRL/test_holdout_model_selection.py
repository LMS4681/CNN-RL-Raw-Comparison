"""Tests for deterministic fixed-holdout model selection."""

from __future__ import annotations

import csv
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from holdout_model_selection import (
    FixedHoldoutEvalCallback,
    SelectionMetric,
    append_selection_row,
    is_better_metric,
    read_best_metric,
)


def metric(score: float, dropout: float, delay: float) -> SelectionMetric:
    return SelectionMetric(
        mean_terminal_score=score,
        mean_dropout_rate=dropout,
        mean_delay_days=delay,
    )


def evaluation_rows(value: SelectionMetric, scenarios) -> list[dict]:
    return [
        {
            "seed": int(scenario["seed"]),
            "mean_terminal_score": value.mean_terminal_score,
            "mean_dropout_rate": value.mean_dropout_rate,
            "mean_delay_days": value.mean_delay_days,
        }
        for scenario in scenarios
    ]


class FakeModel:
    def __init__(self, num_timesteps: int = 0):
        self.num_timesteps = num_timesteps
        self.saved_paths: list[Path] = []

    def save(self, path) -> None:
        self.saved_paths.append(Path(path))

    def get_env(self):
        raise AssertionError("holdout selection inspected the training env")


def fixed_scenarios() -> list[dict]:
    return [{"seed": seed} for seed in range(1000, 1020)]


def make_callback(
    tmp_path: Path,
    *,
    start_timestep: int = 0,
    values: list[SelectionMetric] | None = None,
):
    calls: list[list[int]] = []
    model = FakeModel(start_timestep)
    remaining = list(values or [metric(0.5, 0.2, 2.0)])

    def evaluate_fn(policy_factory, scenarios):
        calls.append([int(item["seed"]) for item in scenarios])
        policy = policy_factory(int(scenarios[0]["seed"]))
        assert policy.model is model
        value = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return evaluation_rows(value, scenarios)

    callback = FixedHoldoutEvalCallback(
        fixed_scenarios(),
        evaluate_fn,
        tmp_path,
        eval_freq=50_000,
        selection_count=5,
    )
    callback.model = model
    callback._on_training_start()
    return callback, calls, model


def test_selection_metric_is_immutable():
    value = metric(0.5, 0.2, 2.0)

    with pytest.raises(FrozenInstanceError):
        value.mean_terminal_score = 0.6


def test_selection_order_is_score_then_dropout_then_delay():
    incumbent = metric(0.50, 0.20, 2.0)
    assert is_better_metric(metric(0.51, 0.90, 9.0), incumbent)
    assert is_better_metric(metric(0.50, 0.19, 9.0), incumbent)
    assert is_better_metric(metric(0.50, 0.20, 1.9), incumbent)
    assert not is_better_metric(metric(0.50, 0.20, 2.0), incumbent)


def test_selection_comparison_does_not_round_floats():
    incumbent = metric(0.5, 0.2, 2.0)
    next_score = math.nextafter(incumbent.mean_terminal_score, math.inf)

    assert is_better_metric(metric(next_score, 1.0, 100.0), incumbent)
    assert not is_better_metric(metric(0.5, 0.2, math.nextafter(2.0, math.inf)), incumbent)


def test_selection_metric_aggregates_only_holdout_rows():
    rows = [
        {
            "mean_terminal_score": 0.25,
            "mean_dropout_rate": 0.375,
            "mean_delay_days": 3.0,
        },
        {
            "mean_terminal_score": 0.75,
            "mean_dropout_rate": 0.125,
            "mean_delay_days": 1.0,
        },
    ]

    assert SelectionMetric.from_rows(rows) == metric(0.5, 0.25, 2.0)
    with pytest.raises(ValueError, match="at least one row"):
        SelectionMetric.from_rows([])


def test_training_environment_metrics_cannot_enter_comparator():
    training_metrics = {
        "mean_terminal_score": 0.9,
        "mean_dropout_rate": 0.0,
        "mean_delay_days": 0.0,
    }

    with pytest.raises(TypeError, match="SelectionMetric"):
        is_better_metric(training_metrics, metric(0.5, 0.2, 2.0))


def test_callback_validates_fixed_seed_order_and_configuration(tmp_path):
    evaluator = lambda policy_factory, scenarios: []
    invalid = fixed_scenarios()
    invalid[0], invalid[1] = invalid[1], invalid[0]

    with pytest.raises(ValueError, match="1000 through 1019"):
        FixedHoldoutEvalCallback(invalid, evaluator, tmp_path)
    with pytest.raises(ValueError, match="positive"):
        FixedHoldoutEvalCallback(
            fixed_scenarios(), evaluator, tmp_path, eval_freq=0
        )
    with pytest.raises(ValueError, match="five"):
        FixedHoldoutEvalCallback(
            fixed_scenarios(), evaluator, tmp_path, selection_count=4
        )


def test_callback_evaluates_first_five_every_50000_steps(tmp_path):
    callback, calls, model = make_callback(tmp_path, start_timestep=0)

    for timestep in (49_999, 50_000, 99_999, 100_000):
        model.num_timesteps = timestep
        assert callback._on_step()

    assert calls == [
        [1000, 1001, 1002, 1003, 1004],
        [1000, 1001, 1002, 1003, 1004],
    ]


def test_resume_schedules_next_strict_multiple(tmp_path):
    callback, calls, model = make_callback(tmp_path, start_timestep=120_000)

    model.num_timesteps = 149_999
    callback._on_step()
    model.num_timesteps = 150_000
    callback._on_step()

    assert len(calls) == 1


def test_starting_on_frequency_boundary_schedules_following_boundary(tmp_path):
    callback, calls, model = make_callback(tmp_path, start_timestep=50_000)

    model.num_timesteps = 50_000
    callback._on_step()
    model.num_timesteps = 100_000
    callback._on_step()

    assert len(calls) == 1


def test_better_result_saves_exact_sb3_path_only_once(tmp_path):
    callback, _calls, model = make_callback(
        tmp_path,
        values=[metric(0.5, 0.2, 2.0), metric(0.5, 0.2, 2.0)],
    )

    for timestep in (50_000, 100_000):
        model.num_timesteps = timestep
        callback._on_step()

    assert model.saved_paths == [tmp_path / "best_model.sb3"]


def test_callback_never_resets_or_reads_training_environment_metrics(tmp_path):
    callback, calls, model = make_callback(tmp_path)

    model.num_timesteps = 50_000
    callback._on_step()

    assert len(calls) == 1


def test_selection_csv_appends_exact_header_values_and_best_flag(tmp_path):
    path = tmp_path / "holdout_selection.csv"
    append_selection_row(path, 50_000, metric(0.5, 0.2, 2.0), True)
    append_selection_row(path, 100_000, metric(0.4, 0.3, 3.0), False)

    assert path.read_bytes() == (
        b"timestep,mean_terminal_score,mean_dropout_rate,"
        b"mean_delay_days,is_best\r\n"
        b"50000,0.5,0.2,2.0,1\r\n"
        b"100000,0.4,0.3,3.0,0\r\n"
    )


def test_read_best_metric_is_strict_and_returns_last_best_row(tmp_path):
    path = tmp_path / "holdout_selection.csv"
    assert read_best_metric(path) is None

    append_selection_row(path, 50_000, metric(0.5, 0.2, 2.0), True)
    append_selection_row(path, 100_000, metric(0.4, 0.3, 3.0), False)
    append_selection_row(path, 150_000, metric(0.6, 0.1, 1.0), True)
    assert read_best_metric(path) == metric(0.6, 0.1, 1.0)

    bad_header = tmp_path / "bad_header.csv"
    bad_header.write_text("timestep,score\n1,0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header is incompatible"):
        read_best_metric(bad_header)

    no_best = tmp_path / "no_best.csv"
    with no_best.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=FixedHoldoutEvalCallback.CSV_FIELDS
        )
        writer.writeheader()
        writer.writerow({
            "timestep": 50_000,
            "mean_terminal_score": 0.5,
            "mean_dropout_rate": 0.2,
            "mean_delay_days": 2.0,
            "is_best": 0,
        })
    with pytest.raises(ValueError, match="no best row"):
        read_best_metric(no_best)


def test_resume_preserves_better_existing_metric(tmp_path):
    append_selection_row(
        tmp_path / "holdout_selection.csv",
        50_000,
        metric(0.9, 0.1, 1.0),
        True,
    )
    callback, _calls, model = make_callback(
        tmp_path,
        start_timestep=120_000,
        values=[metric(0.8, 0.1, 1.0)],
    )

    model.num_timesteps = 150_000
    callback._on_step()

    assert model.saved_paths == []
    assert read_best_metric(tmp_path / "holdout_selection.csv") == metric(
        0.9, 0.1, 1.0
    )
    with (tmp_path / "holdout_selection.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        rows = list(csv.DictReader(source))
    assert rows[-1]["is_best"] == "0"
