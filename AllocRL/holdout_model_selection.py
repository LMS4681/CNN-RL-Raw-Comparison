"""Deterministic model selection on the fixed holdout scenarios."""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from evaluation_runner import ModelActionPolicy


BEST_MODEL_FILENAME = "best_model.sb3"
SELECTION_FILENAME = "holdout_selection.csv"
FIXED_HOLDOUT_SEEDS = tuple(range(1000, 1020))


@dataclass(frozen=True)
class SelectionMetric:
    mean_terminal_score: float
    mean_dropout_rate: float
    mean_delay_days: float

    @classmethod
    def from_rows(
        cls, rows: Sequence[Mapping[str, Any]]
    ) -> "SelectionMetric":
        if not rows:
            raise ValueError("holdout selection requires at least one row")
        return cls(
            mean_terminal_score=float(np.mean([
                float(row["mean_terminal_score"]) for row in rows
            ])),
            mean_dropout_rate=float(np.mean([
                float(row["mean_dropout_rate"]) for row in rows
            ])),
            mean_delay_days=float(np.mean([
                float(row["mean_delay_days"]) for row in rows
            ])),
        )


def is_better_metric(
    candidate: SelectionMetric,
    incumbent: SelectionMetric | None,
) -> bool:
    if not isinstance(candidate, SelectionMetric):
        raise TypeError("candidate must be a SelectionMetric")
    if incumbent is not None and not isinstance(incumbent, SelectionMetric):
        raise TypeError("incumbent must be a SelectionMetric or None")
    if incumbent is None:
        return True
    candidate_key = (
        candidate.mean_terminal_score,
        -candidate.mean_dropout_rate,
        -candidate.mean_delay_days,
    )
    incumbent_key = (
        incumbent.mean_terminal_score,
        -incumbent.mean_dropout_rate,
        -incumbent.mean_delay_days,
    )
    return candidate_key > incumbent_key


def validate_fixed_holdout_scenarios(scenarios: Sequence[Mapping]) -> None:
    seeds = [int(item["seed"]) for item in scenarios]
    if seeds != list(FIXED_HOLDOUT_SEEDS):
        raise ValueError("fixed holdout seeds must be 1000 through 1019")


class FixedHoldoutEvalCallback(BaseCallback):
    CSV_FIELDS = (
        "timestep",
        "mean_terminal_score",
        "mean_dropout_rate",
        "mean_delay_days",
        "is_best",
    )

    def __init__(
        self,
        scenarios: Sequence[dict],
        evaluate_fn: Callable[[Callable, Sequence[dict]], list[dict]],
        output_dir: str | Path,
        eval_freq: int = 50_000,
        selection_count: int = 5,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        validate_fixed_holdout_scenarios(scenarios)
        if eval_freq <= 0:
            raise ValueError("eval_freq must be positive")
        if selection_count != 5:
            raise ValueError("selection_count must be five")
        self._selection_scenarios = list(scenarios[:selection_count])
        self._evaluate_fn = evaluate_fn
        self._output_dir = Path(output_dir)
        self._eval_freq = int(eval_freq)
        self._next_eval_timestep = self._eval_freq
        self._best_metric: SelectionMetric | None = None

    def _on_training_start(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        current = int(self.model.num_timesteps)
        self._next_eval_timestep = (
            (current // self._eval_freq) + 1
        ) * self._eval_freq
        self._best_metric = read_best_metric(
            self._output_dir / SELECTION_FILENAME
        )

    def _on_step(self) -> bool:
        current = int(self.model.num_timesteps)
        if current < self._next_eval_timestep:
            return True

        rows = self._evaluate_fn(
            lambda _seed: ModelActionPolicy(self.model, name="model"),
            self._selection_scenarios,
        )
        metric = SelectionMetric.from_rows(rows)
        is_best = is_better_metric(metric, self._best_metric)
        if is_best:
            self.model.save(self._output_dir / BEST_MODEL_FILENAME)
            self._best_metric = metric
        append_selection_row(
            self._output_dir / SELECTION_FILENAME,
            current,
            metric,
            is_best,
        )
        while self._next_eval_timestep <= current:
            self._next_eval_timestep += self._eval_freq
        return True


def append_selection_row(
    path: Path,
    timestep: int,
    metric: SelectionMetric,
    is_best: bool,
) -> None:
    new_file = not path.is_file() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=FixedHoldoutEvalCallback.CSV_FIELDS
        )
        if new_file:
            writer.writeheader()
        writer.writerow({
            "timestep": int(timestep),
            "mean_terminal_score": metric.mean_terminal_score,
            "mean_dropout_rate": metric.mean_dropout_rate,
            "mean_delay_days": metric.mean_delay_days,
            "is_best": int(is_best),
        })
        output.flush()


def read_best_metric(path: Path) -> SelectionMetric | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != FixedHoldoutEvalCallback.CSV_FIELDS:
            raise ValueError("holdout selection CSV header is incompatible")
        best_rows = [row for row in reader if row["is_best"] == "1"]
    if not best_rows:
        raise ValueError("holdout selection CSV has no best row")
    row = best_rows[-1]
    return SelectionMetric(
        mean_terminal_score=float(row["mean_terminal_score"]),
        mean_dropout_rate=float(row["mean_dropout_rate"]),
        mean_delay_days=float(row["mean_delay_days"]),
    )
