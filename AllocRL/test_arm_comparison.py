"""Tests for the paired common-checkpoint arm comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from comparison.arm_comparison import (
    BASELINE_ARM,
    CANDIDATE_ARM,
    bootstrap_ci,
    compare_arms,
    format_report,
    format_trend,
    resolve_targets,
    summarize,
    write_rows,
)
from comparison.checkpoint_evaluator import (
    EVALUATION_COLUMNS,
    EXPECTED_HOLDOUT_SEEDS,
    PRIMARY_TEST_SEEDS,
    PartialResultError,
)


def _row(arm, seed, score, dropout=0.2, delay=1.0, timestep=830_000):
    return {
        "source": "holdout_fixed20",
        "policy": arm,
        "seed": seed,
        "mean_reward": score,
        "mean_terminal_score": score,
        "mean_dropout_rate": dropout,
        "mean_delay_days": delay,
        "mean_delayed_count": 80.0,
        "mean_retained_choice_ratio": 0.98,
        "arm": arm,
        "checkpoint": "common_step",
        "checkpoint_timestep": timestep,
        "checkpoint_sha256": "a" * 64,
        "evaluation_partition": "selection" if seed < 1005 else "primary_test",
    }


def _rows(raw_score=0.10, cnn_score=0.30, **kwargs):
    return [
        *[_row(BASELINE_ARM, seed, raw_score, **kwargs) for seed in EXPECTED_HOLDOUT_SEEDS],
        *[_row(CANDIDATE_ARM, seed, cnn_score, **kwargs) for seed in EXPECTED_HOLDOUT_SEEDS],
    ]


def _stub_inventory(steps):
    return lambda directory, *args, **kwargs: {
        step: directory / "checkpoints" / f"model_{step}_g1.sb3"
        for step in steps[directory.name]
    }


def test_bootstrap_interval_is_deterministic_and_brackets_the_mean():
    values = [0.1, 0.2, -0.05, 0.3, 0.15]

    low, high = bootstrap_ci(values, resamples=500, seed=7)

    assert (low, high) == bootstrap_ci(values, resamples=500, seed=7)
    assert low < sum(values) / len(values) < high


def test_summary_reports_the_primary_test_partition_separately():
    rows = _rows(raw_score=0.10, cnn_score=0.30)

    summary = summarize(rows)

    assert summary["common_timestep"] == 830_000
    primary = summary["partitions"]["primary_test"]
    assert primary["seeds"] == list(PRIMARY_TEST_SEEDS)
    assert len(primary["seeds"]) == 15
    assert primary["arms"][BASELINE_ARM]["mean_terminal_score"] == pytest.approx(0.10)
    assert primary["arms"][CANDIDATE_ARM]["mean_terminal_score"] == pytest.approx(0.30)
    paired = primary["paired"]["mean_terminal_score"]
    assert paired["mean_difference"] == pytest.approx(0.20)
    assert paired["seeds_favouring_candidate"] == 15
    assert paired["excludes_zero"] is True
    assert len(summary["partitions"]["selection"]["seeds"]) == 5
    assert len(summary["partitions"]["all_holdout"]["seeds"]) == 20


def test_lower_is_better_metrics_count_the_candidate_as_favoured_when_smaller():
    rows = [
        *[_row(BASELINE_ARM, seed, 0.1, dropout=0.30) for seed in EXPECTED_HOLDOUT_SEEDS],
        *[_row(CANDIDATE_ARM, seed, 0.1, dropout=0.20) for seed in EXPECTED_HOLDOUT_SEEDS],
    ]

    paired = summarize(rows)["partitions"]["primary_test"]["paired"]

    assert paired["mean_dropout_rate"]["lower_is_better"] is True
    assert paired["mean_dropout_rate"]["mean_difference"] == pytest.approx(-0.10)
    assert paired["mean_dropout_rate"]["seeds_favouring_candidate"] == 15
    assert paired["mean_terminal_score"]["seeds_favouring_candidate"] == 0


def test_summary_rejects_incomplete_seed_sets_and_mismatched_timesteps():
    rows = _rows()
    with pytest.raises(ValueError, match="every fixed holdout seed"):
        summarize(rows[:-1])

    mixed = [
        *[_row(BASELINE_ARM, seed, 0.1, timestep=820_000) for seed in EXPECTED_HOLDOUT_SEEDS],
        *[_row(CANDIDATE_ARM, seed, 0.1, timestep=830_000) for seed in EXPECTED_HOLDOUT_SEEDS],
    ]
    with pytest.raises(ValueError, match="different checkpoint timesteps"):
        summarize(mixed)


def _inventory(steps):
    return {
        BASELINE_ARM: {step: Path(f"raw/model_{step}_g1.sb3") for step in steps["raw"]},
        CANDIDATE_ARM: {step: Path(f"cnn/model_{step}_g1.sb3") for step in steps["cnn"]},
    }


def test_default_target_is_the_largest_shared_regular_timestep():
    inventories = _inventory(
        {"raw": [810_000, 820_000, 830_000, 840_000], "cnn": [820_000, 830_000]}
    )

    targets = resolve_targets(inventories)

    assert [timestep for timestep, _ in targets] == [830_000]
    assert targets[0][1][BASELINE_ARM].name == "model_830000_g1.sb3"


def test_requested_timesteps_are_sorted_deduplicated_and_present_in_both_arms():
    inventories = _inventory(
        {"raw": [200_000, 400_000], "cnn": [200_000, 400_000, 600_000]}
    )

    targets = resolve_targets(inventories, [400_000, 200_000, 400_000])
    assert [timestep for timestep, _ in targets] == [200_000, 400_000]

    with pytest.raises(PartialResultError, match="raw_direct"):
        resolve_targets(inventories, [600_000])


def test_no_shared_checkpoint_is_reported_as_a_partial_result():
    with pytest.raises(PartialResultError, match="no common readable"):
        resolve_targets(_inventory({"raw": [100_000], "cnn": [200_000]}))


def test_compare_arms_reuses_one_inventory_pass_for_every_timestep(tmp_path, monkeypatch):
    inventory_calls = []

    def fake_inventory(directory, *args, **kwargs):
        inventory_calls.append(Path(directory).name)
        return {
            step: Path(directory) / f"model_{step}_g1.sb3"
            for step in (400_000, 830_000)
        }

    monkeypatch.setattr(
        "comparison.arm_comparison.readable_checkpoint_inventory", fake_inventory
    )
    import train

    monkeypatch.setattr(
        train, "load_model_run_config", lambda path: {"extractor": path.name}
    )
    observed = []

    def fake_evaluate(checkpoint, run_config, scenarios, label, arm):
        observed.append((arm, label, checkpoint.name))
        score = 0.30 if arm == CANDIDATE_ARM else 0.10
        return [
            _row(arm, seed, score, timestep=int(checkpoint.name.split("_")[1]))
            for seed in EXPECTED_HOLDOUT_SEEDS
        ]

    results = compare_arms(
        tmp_path / "raw",
        tmp_path / "cnn",
        [{"seed": seed} for seed in EXPECTED_HOLDOUT_SEEDS],
        timesteps=[830_000, 400_000],
        evaluate=fake_evaluate,
    )

    assert inventory_calls == ["raw", "cnn"], "the slow inventory pass runs once per arm"
    assert [result["timestep"] for result in results] == [400_000, 830_000]
    assert [entry[0] for entry in observed] == [BASELINE_ARM, CANDIDATE_ARM] * 2
    assert len(results[0]["rows"]) == 40
    assert (
        results[0]["summary"]["checkpoints"][BASELINE_ARM]["file"]
        == "model_400000_g1.sb3"
    )
    assert results[0]["summary"]["checkpoints"][BASELINE_ARM]["label"] == "common_step"
    trend = format_trend(results)
    assert "400000" in trend and "830000" in trend
    assert trend.count("mean_terminal_score") == 2


def test_report_marks_the_headline_partition_and_serializes(tmp_path):
    summary = summarize(_rows())

    report = format_report(summary)

    assert "<- headline" in report
    assert "common checkpoint timestep: 830000" in report
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["common_timestep"] == 830_000


def test_written_rows_keep_the_stable_column_order(tmp_path):
    write_rows(tmp_path / "rows.csv", _rows())

    text = (tmp_path / "rows.csv").read_text(encoding="utf-8")

    assert text.splitlines()[0] == ",".join(EVALUATION_COLUMNS)
    assert len(text.strip().splitlines()) == 41
    assert text.splitlines()[1].startswith("holdout_fixed20,candidate_cnn,1000")
