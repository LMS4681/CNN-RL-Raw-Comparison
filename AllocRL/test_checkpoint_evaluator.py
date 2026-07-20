from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from types import SimpleNamespace
from pathlib import Path

import pytest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint(root: Path, name: str, timestep: int) -> Path:
    path = root / "checkpoints" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"checkpoint:" + str(timestep).encode("ascii"))
    return path


def test_common_step_uses_largest_verified_regular_intersection(tmp_path, monkeypatch):
    from comparison import checkpoint_evaluator as evaluator

    raw = tmp_path / "raw"
    cnn = tmp_path / "cnn"
    raw_paths = [_checkpoint(raw, f"raw_{step}.sb3", step) for step in (10_000, 20_000, 30_000)]
    cnn_paths = [_checkpoint(cnn, f"cnn_{step}.sb3", step) for step in (10_000, 20_000)]
    timesteps = {path: int(path.stem.rsplit("_", 1)[1]) for path in raw_paths + cnn_paths}
    monkeypatch.setattr(evaluator, "_archive_timestep", lambda path, loader: timesteps[path])

    assert evaluator.select_common_timestep(raw, cnn) == 20_000


def test_common_step_uses_newest_verified_duplicate_and_rejects_partial(tmp_path, monkeypatch):
    from comparison import checkpoint_evaluator as evaluator

    raw = tmp_path / "raw"
    cnn = tmp_path / "cnn"
    older = _checkpoint(raw, "older.sb3", 10_000)
    newer = _checkpoint(raw, "newer.zip", 10_000)
    _checkpoint(cnn, "cnn.sb3", 20_000)
    _checkpoint(raw, "raw_20000.sb3", 20_000)
    timesteps = {older: 10_000, newer: 10_000}
    for path in (raw / "checkpoints").glob("*"):
        timesteps.setdefault(path, 20_000)
    for path in (cnn / "checkpoints").glob("*"):
        timesteps.setdefault(path, 20_000)
    monkeypatch.setattr(evaluator, "_archive_timestep", lambda path, loader: timesteps[path])
    now = time.time()
    os.utime(older, (now - 60, now - 60))
    os.utime(newer, (now + 60, now + 60))

    assert evaluator.readable_checkpoint_inventory(raw, 10_000)[10_000] == older
    assert newer not in evaluator.readable_checkpoint_inventory(raw, 10_000).values()
    assert evaluator.select_common_timestep(raw, cnn, 10_000) == 20_000

    (cnn / "checkpoints" / "cnn.sb3").unlink()
    with pytest.raises(evaluator.PartialResultError, match="no common"):
        evaluator.select_common_timestep(raw, cnn, 10_000)


def test_final_test_excludes_selection_scenarios():
    from comparison.checkpoint_evaluator import split_holdout_records

    records = [{"seed": seed} for seed in range(1000, 1020)]
    selection, primary = split_holdout_records(records)
    assert [row["seed"] for row in selection] == list(range(1000, 1005))
    assert [row["seed"] for row in primary] == list(range(1005, 1020))


def test_missing_best_uses_exact_complete_state_checkpoint(tmp_path, monkeypatch):
    from comparison import checkpoint_evaluator as evaluator

    checkpoint = _checkpoint(tmp_path, "generation_12345.sb3", 12_345)
    (tmp_path / "run_state.json").write_text(json.dumps({
        "status": "complete",
        "last_checkpoint_file": checkpoint.name,
        "last_checkpoint_timestep": 12_345,
        "last_checkpoint_sha256": _sha256(checkpoint),
    }), encoding="utf-8")
    monkeypatch.setattr(evaluator, "_archive_timestep", lambda path, loader: 12_345)
    monkeypatch.setattr(evaluator, "read_wall_clock_state", lambda _path: SimpleNamespace(status="complete", last_checkpoint_timestep=12_345, last_checkpoint_sha256=_sha256(checkpoint)))
    monkeypatch.setattr(evaluator, "resolve_state_checkpoint", lambda _root, _state: checkpoint)

    selected = evaluator.resolve_selected_or_fallback(tmp_path)
    assert selected.path == checkpoint
    assert selected.label == "fallback_final"
    assert selected.timestep == 12_345


def test_best_requires_selection_proof(tmp_path, monkeypatch):
    from comparison import checkpoint_evaluator as evaluator

    best = tmp_path / "best_model.sb3"
    best.write_bytes(b"best")
    final = _checkpoint(tmp_path, "generation_60000.sb3", 60_000)
    (tmp_path / "run_state.json").write_text(json.dumps({
        "status": "complete", "last_checkpoint_file": final.name,
        "last_checkpoint_timestep": 60_000, "last_checkpoint_sha256": _sha256(final),
    }), encoding="utf-8")
    (tmp_path / "holdout_selection.csv").write_text(
        "timestep,mean_terminal_score,mean_dropout_rate,mean_delay_days,is_best\n"
        "50000,1,0,0,1\n", encoding="utf-8"
    )
    monkeypatch.setattr(evaluator, "_archive_timestep", lambda path, loader: 50_000 if path == best else 60_000)
    monkeypatch.setattr(evaluator, "read_wall_clock_state", lambda _path: SimpleNamespace(status="complete", last_checkpoint_timestep=60_000, last_checkpoint_sha256=_sha256(final)))
    monkeypatch.setattr(evaluator, "resolve_state_checkpoint", lambda _root, _state: final)

    assert evaluator.resolve_selected_or_fallback(tmp_path).label == "best_model"
    (tmp_path / "holdout_selection.csv").write_text("timestep,is_best\n50000,1\n", encoding="utf-8")
    assert evaluator.resolve_selected_or_fallback(tmp_path).path == final


def test_final_reference_uses_only_the_complete_state_checkpoint(tmp_path, monkeypatch):
    from comparison import checkpoint_evaluator as evaluator

    final = _checkpoint(tmp_path, "generation_60000.sb3", 60_000)
    (tmp_path / "run_state.json").write_text(json.dumps({
        "status": "complete", "last_checkpoint_file": final.name,
        "last_checkpoint_timestep": 60_000, "last_checkpoint_sha256": _sha256(final),
    }), encoding="utf-8")
    monkeypatch.setattr(evaluator, "_archive_timestep", lambda path, loader: 60_000)
    monkeypatch.setattr(evaluator, "read_wall_clock_state", lambda _path: SimpleNamespace(status="complete", last_checkpoint_timestep=60_000, last_checkpoint_sha256=_sha256(final)))
    monkeypatch.setattr(evaluator, "resolve_state_checkpoint", lambda _root, _state: final)

    reference = evaluator.resolve_final_checkpoint(tmp_path)
    assert (reference.path, reference.label, reference.timestep) == (final, "final", 60_000)


def test_malformed_state_is_a_partial_result(tmp_path):
    from comparison import checkpoint_evaluator as evaluator

    (tmp_path / "run_state.json").write_text('{"status":"complete"}', encoding="utf-8")
    with pytest.raises(evaluator.PartialResultError, match="final checkpoint"):
        evaluator.resolve_final_checkpoint(tmp_path)


def test_evaluate_checkpoint_labels_rows_and_writes_partitions(tmp_path, monkeypatch):
    from comparison import checkpoint_evaluator as evaluator

    model_path = tmp_path / "model_20000_steps.sb3"
    model_path.write_bytes(b"model")
    config = {
        "observation_scales": {
            "max_length": 1.0, "max_breadth": 1.0, "max_duration": 1,
            "base_date": "2020-01-01", "date_span_workdays": 1,
            "max_workspace_area": 1.0, "total_workspace_area": 1.0,
            "max_workspace_length": 1.0, "max_workspace_breadth": 1.0,
            "dropout_threshold": 1,
        },
        "active_workspace_codes": ["PE001"], "state_context": "full",
    }
    scenarios = [{"seed": seed} for seed in range(1000, 1020)]
    (tmp_path / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    fake_model = SimpleNamespace(num_timesteps=20_000)
    captured = {}
    monkeypatch.setattr(evaluator, "_archive_timestep", lambda path, loader: 20_000)
    monkeypatch.setattr(evaluator.evaluation_runner, "evaluate_scenarios", lambda policy_factory, received, **kwargs: [
            {"source": "holdout_fixed20", "policy": policy_factory(row["seed"]).name, "seed": row["seed"], "mean_reward": 1.0, "mean_terminal_score": 1.0, "mean_dropout_rate": 0.0, "mean_delay_days": 0.0, "mean_delayed_count": 0.0, "mean_retained_choice_ratio": 1.0}
        for row in received
    ])

    rows = evaluator.evaluate_checkpoint(model_path, config, scenarios, "common_step", "raw_direct", model_loader=lambda *_args, **_kwargs: fake_model)
    assert len(rows) == 20
    assert {row["checkpoint"] for row in rows} == {"common_step"}
    assert {row["checkpoint_timestep"] for row in rows} == {20_000}
    assert {row["arm"] for row in rows} == {"raw_direct"}
    assert {row["evaluation_partition"] for row in rows if row["seed"] == 1000} == {"selection"}
    assert {row["evaluation_partition"] for row in rows if row["seed"] == 1005} == {"primary_test"}

    all_path, primary_path = evaluator.write_arm_evaluations(tmp_path, "raw_direct", rows)
    assert len(list(csv.DictReader(all_path.open(encoding="utf-8")))) == 20
    assert [int(row["seed"]) for row in csv.DictReader(primary_path.open(encoding="utf-8"))] == list(range(1005, 1020))


def test_manifest_checkpoint_entries_merge_without_fabricating_metadata(tmp_path):
    from comparison.checkpoint_evaluator import CheckpointRef, merge_checkpoint_manifest

    manifest = {"schema_version": 1, "unchanged": {"value": True}}
    result = merge_checkpoint_manifest(manifest, "raw_direct", {
        "selected": CheckpointRef(Path("raw/best_model.sb3"), "best_model", 50_000, "a" * 64),
        "final": CheckpointRef(Path("raw/final.sb3"), "final", 60_000, "b" * 64),
    })
    assert result["unchanged"] == {"value": True}
    assert result["checkpoints"]["raw_direct"]["selected"] == {
        "path": "raw/best_model.sb3", "label": "best_model", "sha256": "a" * 64, "timestep": 50_000,
    }


def test_common_writer_rejects_incomplete_or_mismatched_pair(tmp_path):
    from comparison import checkpoint_evaluator as evaluator

    rows = []
    for arm in evaluator.ARMS:
        for seed in range(1000, 1020):
            rows.append(dict(zip(evaluator.EVALUATION_COLUMNS, (
                "holdout_fixed20", arm, seed, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                arm, "common_step", 20_000, arm[0] * 64,
                "selection" if seed < 1005 else "primary_test",
            ))))
    evaluator.write_common_step_evaluation(tmp_path, list(reversed(rows)))
    written = list(csv.DictReader((tmp_path / "comparison" / "common_step_evaluation.csv").open(encoding="utf-8")))
    assert [(row["arm"], int(row["seed"])) for row in written] == [(arm, seed) for arm in evaluator.ARMS for seed in range(1000, 1020)]
    rows.pop()
    with pytest.raises(ValueError, match="holdout"):
        evaluator.write_common_step_evaluation(tmp_path, rows)


def test_inventory_does_not_restat_disappearing_prior_duplicate(tmp_path, monkeypatch):
    from comparison import checkpoint_evaluator as evaluator

    older = _checkpoint(tmp_path, "older.sb3", 10_000)
    newer = _checkpoint(tmp_path, "newer.sb3", 10_000)
    now = time.time(); os.utime(older, (now - 60, now - 60)); os.utime(newer, (now + 60, now + 60))
    monkeypatch.setattr(evaluator, "_archive_timestep", lambda *_: 10_000)
    monkeypatch.setattr(evaluator, "sha256_file", lambda path: "a" * 64)
    original_stat = Path.stat
    calls = {older: 0}
    def stat(path, *args, **kwargs):
        if path == older:
            calls[older] += 1
            if calls[older] > 1:
                raise FileNotFoundError(path)
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", stat)
    assert evaluator.readable_checkpoint_inventory(tmp_path)[10_000] == newer
    assert calls[older] == 1


def test_best_hash_race_falls_back_to_verified_state(tmp_path, monkeypatch):
    from comparison import checkpoint_evaluator as evaluator

    best = tmp_path / "best_model.sb3"; best.write_bytes(b"best")
    final = _checkpoint(tmp_path, "final.sb3", 60_000)
    (tmp_path / "holdout_selection.csv").write_text("timestep,mean_terminal_score,mean_dropout_rate,mean_delay_days,is_best\n50000,1,0,0,1\n", encoding="utf-8")
    monkeypatch.setattr(evaluator, "_archive_timestep", lambda path, loader: 50_000 if path == best else 60_000)
    monkeypatch.setattr(evaluator, "read_wall_clock_state", lambda _: SimpleNamespace(status="complete", last_checkpoint_timestep=60_000, last_checkpoint_sha256=_sha256(final)))
    monkeypatch.setattr(evaluator, "resolve_state_checkpoint", lambda *_: final)
    original_hash = evaluator.sha256_file
    monkeypatch.setattr(evaluator, "sha256_file", lambda path: (_ for _ in ()).throw(FileNotFoundError(path)) if path == best else original_hash(path))
    assert evaluator.resolve_selected_or_fallback(tmp_path).path == final


def test_common_selection_passes_custom_loader_to_inventory(tmp_path, monkeypatch):
    from comparison import checkpoint_evaluator as evaluator
    raw = _checkpoint(tmp_path / "raw", "raw.sb3", 10_000).parents[1]
    cnn = _checkpoint(tmp_path / "cnn", "cnn.sb3", 10_000).parents[1]
    seen = []
    def reader(path, loader):
        seen.append(loader); return 10_000
    marker = object()
    monkeypatch.setattr(evaluator, "_archive_timestep", reader)
    assert evaluator.select_common_timestep(raw, cnn, model_loader=marker) == 10_000
    assert seen == [marker, marker]
