"""Tests for deterministic fixed-holdout model selection."""

from __future__ import annotations

import csv
import gc
import math
import weakref
import zipfile
from contextlib import ExitStack
from dataclasses import FrozenInstanceError
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import train as train_module
from alloc_env.observation_state import ObservationScales
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


class CloseCountingEnv:
    def __init__(self, name: str, events: list[str]):
        self.name = name
        self.events = events
        self.close_count = 0
        self.observation_space = object()
        self.action_space = object()

    def close(self) -> None:
        self.close_count += 1
        self.events.append(f"close:{self.name}")
        if self.close_count > 1:
            raise AssertionError(f"{self.name} closed more than once")


def full_source_scales() -> ObservationScales:
    return ObservationScales(
        max_length=100.0,
        max_breadth=50.0,
        max_duration=60,
        base_date=date(2025, 12, 1),
        date_span_workdays=150,
        max_workspace_area=10_000.0,
        total_workspace_area=80_000.0,
        max_workspace_length=200.0,
        max_workspace_breadth=100.0,
        dropout_threshold=7,
    )


def orchestration_args(
    tmp_path: Path,
    *,
    holdout_eval_freq: int,
    final_holdout_report: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "output"),
        timesteps=1,
        lr=3e-4,
        n_steps=1,
        grid_size=64,
        batch_size=1,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=0.98,
        n_eval=1,
        eval_scenarios=str(tmp_path / "fixed_eval_scenarios.json"),
        holdout_eval_freq=holdout_eval_freq,
        holdout_selection_count=5,
        final_holdout_report=final_holdout_report,
        extractor="candidate-cnn",
        state_context="current",
        features_dim=256,
        device="cpu",
        seed=41,
        monthly_jitter=20,
        empirical_profile_probability=0.2,
        n_envs=1,
        vec_env="dummy",
        active_workspace_codes=train_module.DEFAULT_ACTIVE_WORKSPACE_CODES,
        resume_from=None,
        auto_resume=False,
        checkpoint_freq=0,
        export_onnx=True,
    )


def run_train_orchestration(
    tmp_path: Path,
    *,
    holdout_eval_freq: int = 50_000,
    final_holdout_report: bool = True,
    fail_at: str | None = None,
):
    events: list[str] = []
    scenarios = fixed_scenarios()
    scales = full_source_scales()
    blocks = [SimpleNamespace(in_date=date(2026, 1, 5))]
    workspaces = [
        SimpleNamespace(code=code)
        for code in train_module.DEFAULT_ACTIVE_WORKSPACE_CODES.split(",")
    ]
    training_env = CloseCountingEnv("training", events)
    original_env = CloseCountingEnv("original", events)
    selected_env = CloseCountingEnv("selected", events)
    evaluation_envs = [original_env, selected_env]
    callback_refs: list[weakref.ReferenceType] = []
    state = {
        "events": events,
        "scenarios": scenarios,
        "scales": scales,
        "training_env": training_env,
        "original_env": original_env,
        "selected_env": selected_env,
        "callback_refs": callback_refs,
        "writes": [],
    }

    class ExistingAllocationCallback:
        def __init__(self, **kwargs):
            self.model = None
            self.locals = {}
            self.globals = {}
            self.parent = None

    class ExistingTrainingMetricsCallback(ExistingAllocationCallback):
        pass

    class SelectedModel:
        def __init__(self, env):
            self.env = env

        def get_env(self):
            return self.env

    class FakeMaskablePPO:
        final_model_ref = None

        def __init__(self, policy, env, **kwargs):
            self.env = env
            self._vec_normalize_env = env
            type(self).final_model_ref = weakref.ref(self)
            events.append("create:final-model")

        def learn(self, *, callback, **kwargs):
            state["callback_types"] = [type(item) for item in callback]
            for item in callback:
                item.model = self
                item.locals["algorithm"] = self
                item.globals["algorithm"] = self
                callback_refs.append(weakref.ref(item))
            events.append("learn")
            if fail_at == "learn":
                raise RuntimeError("injected learn failure")
            return self

        def save(self, path):
            events.append(f"save:{Path(path).name}")

        @classmethod
        def load(cls, path, *, env, device, **kwargs):
            gc.collect()
            final_released = cls.final_model_ref() is None
            callbacks_released = all(
                callback_ref() is None for callback_ref in callback_refs
            )
            state["released_before_selected"] = (
                final_released and callbacks_released
            )
            if not state["released_before_selected"]:
                raise AssertionError(
                    "final model or callbacks survived selected model load"
                )
            events.append("load:selected-model")
            if fail_at == "selected-load":
                raise RuntimeError("injected selected-load failure")
            return SelectedModel(env)

    def create_evaluation_env(*args, **kwargs):
        state.setdefault("evaluation_env_calls", []).append((args, kwargs))
        return evaluation_envs.pop(0)

    def evaluate_original_csv_row(model, env, n_eval):
        assert model is FakeMaskablePPO.final_model_ref()
        assert env is original_env
        state["original_received_final_model"] = True
        events.append("evaluate:original")
        if fail_at == "original":
            raise RuntimeError("injected original failure")
        return {"source": "original_csv", "policy": "model"}

    def export_onnx(model, env, path):
        assert model is FakeMaskablePPO.final_model_ref()
        assert env is training_env
        events.append("export:onnx")
        return True

    def evaluate_scenarios(
        policy_factory,
        selected_scenarios,
        *,
        workspace_codes,
        observation_scales,
        state_context_mode,
    ):
        state["runner_scenarios"] = selected_scenarios
        state["runner_workspace_codes"] = workspace_codes
        state["runner_scales"] = observation_scales
        state["runner_state_context"] = state_context_mode
        state["selected_policy_model"] = weakref.ref(
            policy_factory(1000).model
        )
        events.append("evaluate:selected")
        return [{"seed": item["seed"]} for item in selected_scenarios]

    def write_metrics(path, rows):
        state["writes"].append((Path(path).name, [dict(row) for row in rows]))

    source_split = SimpleNamespace(
        training_blocks=blocks,
        manifest={"source": "bounded-test"},
    )
    output_dir = Path(orchestration_args(
        tmp_path,
        holdout_eval_freq=holdout_eval_freq,
        final_holdout_report=final_holdout_report,
    ).output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if final_holdout_report:
        with zipfile.ZipFile(output_dir / "best_model.sb3", "w") as archive:
            archive.writestr("probe", "selected")

    args = orchestration_args(
        tmp_path,
        holdout_eval_freq=holdout_eval_freq,
        final_holdout_report=final_holdout_report,
    )
    with ExitStack() as stack:
        stack.enter_context(patch.object(
            train_module,
            "load_requested_evaluation_scenarios",
            return_value=scenarios,
        ))
        stack.enter_context(patch.object(train_module, "set_global_seed"))
        stack.enter_context(patch.object(
            train_module,
            "load_allocation_scenario",
            return_value=(blocks, workspaces),
        ))
        build_scales = stack.enter_context(patch(
            "alloc_env.observation_state.build_observation_scales",
            return_value=scales,
        ))
        stack.enter_context(patch(
            "alloc_env.data_split.split_blocks_by_ship",
            return_value=source_split,
        ))
        stack.enter_context(patch(
            "alloc_env.block_generator.SyntheticBlockGenerator.from_blocks",
            return_value=object(),
        ))
        create_training = stack.enter_context(patch.object(
            train_module,
            "create_training_env",
            return_value=training_env,
        ))
        stack.enter_context(patch.object(
            train_module, "resolve_vec_env_type", return_value="single"
        ))
        stack.enter_context(patch.object(
            train_module, "estimate_rollout_buffer_mb", return_value=1.0
        ))
        stack.enter_context(patch.object(
            train_module, "build_policy_kwargs", return_value={}
        ))
        stack.enter_context(patch.object(
            train_module, "current_run_config", return_value={}
        ))
        stack.enter_context(patch.object(
            train_module, "resolve_resume_path", return_value=None
        ))
        stack.enter_context(patch.object(train_module, "write_run_config"))
        stack.enter_context(patch(
            "alloc_env.callbacks.AllocationCallback",
            ExistingAllocationCallback,
        ))
        stack.enter_context(patch(
            "alloc_env.callbacks.TrainingMetricsCallback",
            ExistingTrainingMetricsCallback,
        ))
        stack.enter_context(patch("sb3_contrib.MaskablePPO", FakeMaskablePPO))
        stack.enter_context(patch.object(
            train_module, "create_evaluation_env", new=create_evaluation_env
        ))
        stack.enter_context(patch.object(
            train_module,
            "evaluate_original_csv_row",
            new=evaluate_original_csv_row,
        ))
        stack.enter_context(patch.object(
            train_module, "try_export_to_onnx", new=export_onnx
        ))
        stack.enter_context(patch.object(
            train_module, "write_evaluation_metrics", new=write_metrics
        ))
        stack.enter_context(patch(
            "evaluation_runner.evaluate_scenarios",
            new=evaluate_scenarios,
        ))

        if fail_at is None:
            train_module.train(args)
        else:
            with pytest.raises(RuntimeError, match=f"injected {fail_at}"):
                train_module.train(args)

    state["build_scales_calls"] = build_scales.call_args_list
    state["create_training_call"] = create_training.call_args
    return state


def test_train_orchestrates_holdout_selection_and_releases_before_report(
    tmp_path,
):
    state = run_train_orchestration(tmp_path)

    callback_types = state["callback_types"]
    assert [item.__name__ for item in callback_types[:2]] == [
        "ExistingAllocationCallback",
        "ExistingTrainingMetricsCallback",
    ]
    assert callback_types[2] is FixedHoldoutEvalCallback
    assert state["original_received_final_model"]
    assert state["released_before_selected"]
    assert state["build_scales_calls"][0].args[:2] == (
        [SimpleNamespace(in_date=date(2026, 1, 5))],
        [
            SimpleNamespace(code=code)
            for code in train_module.DEFAULT_ACTIVE_WORKSPACE_CODES.split(",")
        ],
    )
    assert state["create_training_call"].kwargs["observation_scales"] is state[
        "scales"
    ]
    assert state["create_training_call"].kwargs["state_context_mode"] == (
        "current"
    )
    assert state["runner_scales"] is state["scales"]
    assert state["runner_state_context"] == "current"
    assert state["runner_workspace_codes"] == [
        code for code in train_module.DEFAULT_ACTIVE_WORKSPACE_CODES.split(",")
    ]
    assert len(state["evaluation_env_calls"]) == 2
    for _args, kwargs in state["evaluation_env_calls"]:
        assert kwargs["observation_scales"] is state["scales"]
        assert kwargs["state_context_mode"] == "current"
    assert [item["seed"] for item in state["runner_scenarios"]] == list(
        range(1000, 1020)
    )
    assert all(
        actual is expected
        for actual, expected in zip(
            state["runner_scenarios"], state["scenarios"]
        )
    )
    scenario_write = next(
        rows for name, rows in state["writes"]
        if name == "evaluation_scenarios.csv"
    )
    assert len(scenario_write) == 20
    assert all(row["checkpoint"] == "best_model" for row in scenario_write)
    assert state["selected_policy_model"]() is None
    assert state["training_env"].close_count == 1
    assert state["original_env"].close_count == 1
    assert state["selected_env"].close_count == 1
    assert state["events"].index("evaluate:original") < state["events"].index(
        "close:training"
    )
    assert state["events"].index("save:block_placement_ppo.sb3") < state[
        "events"
    ].index("evaluate:original")
    assert state["events"].index("export:onnx") < state["events"].index(
        "evaluate:original"
    )
    assert state["events"].index("close:training") < state["events"].index(
        "load:selected-model"
    )


def test_train_frequency_zero_disables_holdout_callback(tmp_path):
    state = run_train_orchestration(
        tmp_path,
        holdout_eval_freq=0,
        final_holdout_report=False,
    )

    callback_names = [item.__name__ for item in state["callback_types"]]
    assert callback_names[:2] == [
        "ExistingAllocationCallback",
        "ExistingTrainingMetricsCallback",
    ]
    assert "FixedHoldoutEvalCallback" not in callback_names
    assert "AbsoluteScheduleCallback" in callback_names
    assert "ExtractorFineTuneCallback" in callback_names
    assert "load:selected-model" not in state["events"]
    assert state["training_env"].close_count == 1
    assert state["original_env"].close_count == 1


@pytest.mark.parametrize(
    ("fail_at", "expected_original_closes", "expected_selected_closes"),
    [
        ("learn", 0, 0),
        ("original", 1, 0),
        ("selected-load", 1, 1),
    ],
)
def test_train_closes_created_envs_on_failure(
    tmp_path,
    fail_at,
    expected_original_closes,
    expected_selected_closes,
):
    state = run_train_orchestration(tmp_path, fail_at=fail_at)

    assert state["training_env"].close_count == 1
    assert state["original_env"].close_count == expected_original_closes
    assert state["selected_env"].close_count == expected_selected_closes
