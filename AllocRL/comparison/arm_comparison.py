"""Compare two trained arms at their largest common regular checkpoint.

The arms are evaluated on the twenty fixed holdout scenarios and reported as
paired per-seed differences. The headline is the fifteen primary-test seeds
that model selection never saw; the five selection seeds are a secondary view.

This module only orchestrates `checkpoint_evaluator`, which needs nothing but
the checkpoint archives and the `run_config.json` beside them. A step-bounded
arm has no wall-clock receipts, so the marker-validated staged pipeline does not
apply to it, but this comparison does.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path
from typing import Callable, Mapping, Sequence

from comparison.checkpoint_evaluator import (
    EVALUATION_COLUMNS,
    EXPECTED_HOLDOUT_SEEDS,
    PRIMARY_TEST_SEEDS,
    SELECTION_SEEDS,
    PartialResultError,
    evaluate_checkpoint,
    readable_checkpoint_inventory,
    select_common_timestep,
)


METRIC_KEYS = ("mean_terminal_score", "mean_dropout_rate", "mean_delay_days")
LOWER_IS_BETTER = frozenset({"mean_dropout_rate", "mean_delay_days"})
PARTITIONS = {
    "primary_test": PRIMARY_TEST_SEEDS,
    "selection": SELECTION_SEEDS,
    "all_holdout": EXPECTED_HOLDOUT_SEEDS,
}
BASELINE_ARM = "raw_direct"
CANDIDATE_ARM = "candidate_cnn"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260803


def resolve_common_checkpoints(
    raw_dir: str | Path,
    cnn_dir: str | Path,
    *,
    timestep: int | None = None,
    model_loader=None,
) -> tuple[int, dict[str, Path]]:
    """Fix the shared checkpoint both arms actually reached."""
    kwargs = {} if model_loader is None else {"model_loader": model_loader}
    inventories = {
        BASELINE_ARM: readable_checkpoint_inventory(Path(raw_dir), **kwargs),
        CANDIDATE_ARM: readable_checkpoint_inventory(Path(cnn_dir), **kwargs),
    }
    if timestep is None:
        common = select_common_timestep(Path(raw_dir), Path(cnn_dir), **kwargs)
    else:
        common = int(timestep)
        missing = [arm for arm, found in inventories.items() if common not in found]
        if missing:
            raise PartialResultError(
                f"requested timestep {common} is missing for: {', '.join(missing)}"
            )
    return common, {arm: found[common] for arm, found in inventories.items()}


def bootstrap_ci(
    values: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean paired difference.

    `scipy` is deliberately absent from the dependency lock, and with fifteen
    paired seeds an interval is a more honest summary than a p-value.
    """
    if not values:
        raise ValueError("bootstrap requires at least one value")
    rng = random.Random(seed)
    count = len(values)
    means = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(count):
            total += values[rng.randrange(count)]
        means.append(total / count)
    means.sort()
    return means[int(0.025 * (resamples - 1))], means[int(0.975 * (resamples - 1))]


def _by_seed(rows: Sequence[Mapping], arm: str) -> dict[int, Mapping]:
    selected = {}
    for row in rows:
        if row["arm"] != arm:
            continue
        seed = int(row["seed"])
        if seed in selected:
            raise ValueError(f"arm {arm} has duplicate seed {seed}")
        selected[seed] = row
    if sorted(selected) != sorted(EXPECTED_HOLDOUT_SEEDS):
        raise ValueError(f"arm {arm} did not evaluate every fixed holdout seed")
    return selected


def summarize(rows: Sequence[Mapping]) -> dict:
    """Pair the candidate arm against the baseline arm, seed by seed."""
    indexed = {arm: _by_seed(rows, arm) for arm in (BASELINE_ARM, CANDIDATE_ARM)}
    timesteps = {
        arm: int(next(iter(seeds.values()))["checkpoint_timestep"])
        for arm, seeds in indexed.items()
    }
    if len(set(timesteps.values())) != 1:
        raise ValueError("arms were evaluated at different checkpoint timesteps")
    summary = {
        "common_timestep": next(iter(timesteps.values())),
        "baseline_arm": BASELINE_ARM,
        "candidate_arm": CANDIDATE_ARM,
        "checkpoints": {
            arm: {
                "file": next(iter(seeds.values()))["checkpoint"],
                "sha256": next(iter(seeds.values()))["checkpoint_sha256"],
            }
            for arm, seeds in indexed.items()
        },
        "partitions": {},
    }
    for partition, seeds in PARTITIONS.items():
        entry = {"seeds": list(seeds), "arms": {}, "paired": {}}
        for arm in (BASELINE_ARM, CANDIDATE_ARM):
            entry["arms"][arm] = {
                key: statistics.fmean(float(indexed[arm][seed][key]) for seed in seeds)
                for key in METRIC_KEYS
            }
        for key in METRIC_KEYS:
            differences = [
                float(indexed[CANDIDATE_ARM][seed][key])
                - float(indexed[BASELINE_ARM][seed][key])
                for seed in seeds
            ]
            low, high = bootstrap_ci(differences)
            favouring = sum(
                1
                for value in differences
                if (value < 0 if key in LOWER_IS_BETTER else value > 0)
            )
            entry["paired"][key] = {
                "per_seed_difference": differences,
                "mean_difference": statistics.fmean(differences),
                "stdev_difference": (
                    statistics.stdev(differences) if len(differences) > 1 else 0.0
                ),
                "bootstrap_ci_95": [low, high],
                "excludes_zero": low > 0 or high < 0,
                "seeds_favouring_candidate": favouring,
                "seed_count": len(differences),
                "lower_is_better": key in LOWER_IS_BETTER,
            }
        summary["partitions"][partition] = entry
    return summary


def write_rows(path: str | Path, rows: Sequence[Mapping]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(EVALUATION_COLUMNS))
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["arm"], int(item["seed"]))):
            writer.writerow({key: row[key] for key in EVALUATION_COLUMNS})


def format_report(summary: Mapping) -> str:
    lines = [
        f"common checkpoint timestep: {summary['common_timestep']}",
        f"  {BASELINE_ARM:<14} {summary['checkpoints'][BASELINE_ARM]['file']}",
        f"  {CANDIDATE_ARM:<14} {summary['checkpoints'][CANDIDATE_ARM]['file']}",
        "",
        "candidate_cnn minus raw_direct, paired by scenario seed",
    ]
    for partition in ("primary_test", "selection", "all_holdout"):
        entry = summary["partitions"][partition]
        lines.append(
            f"\n[{partition}] {len(entry['seeds'])} seeds"
            + ("  <- headline" if partition == "primary_test" else "")
        )
        for key in METRIC_KEYS:
            values = entry["paired"][key]
            low, high = values["bootstrap_ci_95"]
            lines.append(
                f"  {key:<20} raw {entry['arms'][BASELINE_ARM][key]:+.4f}"
                f"  cnn {entry['arms'][CANDIDATE_ARM][key]:+.4f}"
                f"  diff {values['mean_difference']:+.4f}"
                f"  CI [{low:+.4f}, {high:+.4f}]"
                f"  {'excludes 0' if values['excludes_zero'] else 'includes 0'}"
                f"  cnn better on {values['seeds_favouring_candidate']}/{values['seed_count']}"
                + ("  (lower is better)" if values["lower_is_better"] else "")
            )
    return "\n".join(lines)


def compare_arms(
    raw_dir: str | Path,
    cnn_dir: str | Path,
    scenarios: Sequence[dict],
    *,
    timestep: int | None = None,
    evaluate: Callable[..., list[dict]] = evaluate_checkpoint,
    model_loader=None,
) -> tuple[list[dict], dict]:
    """Evaluate both arms at the common checkpoint and pair the results."""
    common, checkpoints = resolve_common_checkpoints(
        raw_dir, cnn_dir, timestep=timestep, model_loader=model_loader
    )
    from train import load_model_run_config

    rows: list[dict] = []
    for arm, checkpoint in checkpoints.items():
        run_config = load_model_run_config(checkpoint)
        rows.extend(
            evaluate(checkpoint, run_config, list(scenarios), "common_step", arm)
        )
    return rows, summarize(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Paired comparison of two arms at their common checkpoint"
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--cnn-dir", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timestep", type=int, default=None)
    args = parser.parse_args(argv)

    from evaluation_scenarios import read_scenarios
    from holdout_model_selection import validate_fixed_holdout_scenarios

    scenarios = read_scenarios(args.scenarios)
    validate_fixed_holdout_scenarios(scenarios)

    rows, summary = compare_arms(
        args.raw_dir, args.cnn_dir, scenarios, timestep=args.timestep
    )
    output_dir = Path(args.output_dir)
    write_rows(output_dir / "arm_comparison_rows.csv", rows)
    (output_dir / "arm_comparison_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(format_report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
