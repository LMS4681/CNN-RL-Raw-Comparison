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


def _selection_fixture(root: Path, *, final_timestep: int = 60_000) -> Path:
    final = _checkpoint(root, f"final_{final_timestep}.sb3", final_timestep)
    (root / "run_state.json").write_text(json.dumps({
        "schema_version": 1,
        "target_training_seconds": 10.0,
        "completed_training_seconds": 10.0,
        "last_checkpoint_timestep": final_timestep,
        "last_regular_checkpoint_timestep": 50_000,
        "last_checkpoint_file": final.name,
        "last_checkpoint_sha256": _sha256(final),
        "config_sha256": "a" * 64,
        "generation": 1,
        "restart_count": 0,
        "max_unrecorded_seconds": 1.0,
        "status": "complete",
        "started_at_utc": "2026-01-01T00:00:00+00:00",
        "updated_at_utc": "2026-01-01T00:00:10+00:00",
        "completed_at_utc": "2026-01-01T00:00:10+00:00",
    }), encoding="utf-8")
    return final


def _text_timestep(path: Path, *_args) -> int | None:
    try:
        return int(path.read_bytes().split(b":")[-1])
    except (OSError, ValueError):
        return None


def _text_loader(path, **_kwargs):
    return SimpleNamespace(num_timesteps=_text_timestep(Path(path)))


def test_selection_decision_records_verified_best_provenance(tmp_path):
    from comparison import checkpoint_evaluator as evaluator

    _selection_fixture(tmp_path)
    best = tmp_path / "best_model.sb3"
    best.write_bytes(b"checkpoint:50000")
    (tmp_path / "holdout_selection.csv").write_text(
        "timestep,mean_terminal_score,mean_dropout_rate,mean_delay_days,is_best\n"
        "50000,1.25,0.2,3.0,1\n",
        encoding="utf-8",
    )

    decision = evaluator.resolve_selection_decision(
        tmp_path, model_loader=_text_loader
    )

    assert decision.reference == evaluator.CheckpointRef(
        best, "best_model", 50_000, _sha256(best)
    )
    assert decision.selection_outcome == "best_model"
    assert decision.fallback_reason is None
    assert decision.selection_count == 5
    assert decision.selection_tuple == [1.25, -0.2, -3.0]


@pytest.mark.parametrize(
    "case,expected_reason",
    [
        ("not_run", "selection_not_run"),
        ("no_best", "selection_has_no_best"),
        ("invalid_metadata", "selection_metadata_invalid"),
        ("missing_best", "best_model_missing"),
        ("unreadable_best", "best_model_unreadable"),
        ("wrong_timestep", "best_model_timestep_mismatch"),
    ],
)
def test_selection_decision_uses_canonical_fallback_reason(
    tmp_path, case, expected_reason
):
    from comparison import checkpoint_evaluator as evaluator

    final = _selection_fixture(tmp_path)
    selection = tmp_path / "holdout_selection.csv"
    best = tmp_path / "best_model.sb3"
    if case == "no_best":
        selection.write_text(
            "timestep,mean_terminal_score,mean_dropout_rate,mean_delay_days,is_best\n"
            "50000,1,0.2,3,0\n",
            encoding="utf-8",
        )
    elif case == "invalid_metadata":
        selection.write_text("timestep,is_best\n50000,1\n", encoding="utf-8")
    elif case != "not_run":
        selection.write_text(
            "timestep,mean_terminal_score,mean_dropout_rate,mean_delay_days,is_best\n"
            "50000,1,0.2,3,1\n",
            encoding="utf-8",
        )
        if case == "unreadable_best":
            best.write_bytes(b"not-an-archive")
        elif case == "wrong_timestep":
            best.write_bytes(b"checkpoint:49999")
        elif case != "missing_best":
            best.write_bytes(b"checkpoint:50000")

    decision = evaluator.resolve_selection_decision(
        tmp_path, model_loader=_text_loader
    )

    assert decision.reference == evaluator.CheckpointRef(
        final, "fallback_final", 60_000, _sha256(final)
    )
    assert decision.selection_outcome == "fallback_final"
    assert decision.fallback_reason == expected_reason
    assert decision.selection_count == 0
    assert decision.selection_tuple is None


def test_selection_after_exact_final_budget_is_invalid_metadata(tmp_path):
    from comparison import checkpoint_evaluator as evaluator

    final = _selection_fixture(tmp_path, final_timestep=60_000)
    (tmp_path / "holdout_selection.csv").write_text(
        "timestep,mean_terminal_score,mean_dropout_rate,mean_delay_days,is_best\n"
        "70000,1.25,0.2,3.0,1\n",
        encoding="utf-8",
    )
    (tmp_path / "best_model.sb3").write_bytes(b"checkpoint:70000")

    decision = evaluator.resolve_selection_decision(
        tmp_path, model_loader=_text_loader
    )

    assert decision.reference == evaluator.CheckpointRef(
        final, "fallback_final", 60_000, _sha256(final)
    )
    assert decision.fallback_reason == "selection_metadata_invalid"


def test_nonregular_selection_metadata_is_invalid_not_absent(tmp_path):
    from comparison import checkpoint_evaluator as evaluator

    _selection_fixture(tmp_path)
    (tmp_path / "holdout_selection.csv").mkdir()

    decision = evaluator.resolve_selection_decision(
        tmp_path, model_loader=_text_loader
    )

    assert decision.selection_outcome == "fallback_final"
    assert decision.fallback_reason == "selection_metadata_invalid"


def test_symlinked_selection_metadata_is_never_followed(tmp_path):
    from comparison import checkpoint_evaluator as evaluator

    _selection_fixture(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-selection.csv"
    outside.write_text(
        "timestep,mean_terminal_score,mean_dropout_rate,mean_delay_days,is_best\n"
        "50000,1.25,0.2,3.0,1\n",
        encoding="utf-8",
    )
    selection = tmp_path / "holdout_selection.csv"
    try:
        selection.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows host")
    (tmp_path / "best_model.sb3").write_bytes(b"checkpoint:50000")

    decision = evaluator.resolve_selection_decision(
        tmp_path, model_loader=_text_loader
    )

    assert decision.selection_outcome == "fallback_final"
    assert decision.fallback_reason == "selection_metadata_invalid"


def test_best_model_mutation_during_read_falls_back_as_unreadable(tmp_path):
    from comparison import checkpoint_evaluator as evaluator

    _selection_fixture(tmp_path)
    best = tmp_path / "best_model.sb3"
    best.write_bytes(b"checkpoint:50000")
    (tmp_path / "holdout_selection.csv").write_text(
        "timestep,mean_terminal_score,mean_dropout_rate,mean_delay_days,is_best\n"
        "50000,1.25,0.2,3.0,1\n",
        encoding="utf-8",
    )

    def mutate_after_read(path):
        timestep = _text_timestep(Path(path))
        Path(path).write_bytes(b"checkpoint:50000-mutated")
        return timestep

    decision = evaluator.resolve_selection_decision(
        tmp_path,
        model_loader=_text_loader,
        archive_timestep_reader=mutate_after_read,
    )

    assert decision.selection_outcome == "fallback_final"
    assert decision.fallback_reason == "best_model_unreadable"


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
    with all_path.open(encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 20
    with primary_path.open(encoding="utf-8") as stream:
        assert [int(row["seed"]) for row in csv.DictReader(stream)] == list(range(1005, 1020))


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
    with (tmp_path / "comparison" / "common_step_evaluation.csv").open(encoding="utf-8") as stream:
        written = list(csv.DictReader(stream))
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


def test_evaluate_checkpoint_hash_race_is_partial_result(tmp_path, monkeypatch):
    from comparison import checkpoint_evaluator as evaluator
    model = tmp_path / "model.sb3"; model.write_bytes(b"x")
    config = {"observation_scales": {"max_length": 1., "max_breadth": 1., "max_duration": 1, "base_date": "2020-01-01", "date_span_workdays": 1, "max_workspace_area": 1., "total_workspace_area": 1., "max_workspace_length": 1., "max_workspace_breadth": 1., "dropout_threshold": 1}, "active_workspace_codes": ["PE001"], "state_context": "full"}
    (tmp_path / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(evaluator, "sha256_file", lambda _: (_ for _ in ()).throw(FileNotFoundError()))
    with pytest.raises(evaluator.PartialResultError):
        evaluator.evaluate_checkpoint(model, config, [{"seed": s} for s in range(1000, 1020)], "common_step", "raw_direct", model_loader=lambda *_args, **_kwargs: SimpleNamespace(num_timesteps=10_000))


def test_validated_rows_accepts_different_mapping_insertion_order(tmp_path):
    from comparison import checkpoint_evaluator as evaluator
    rows = []
    for seed in range(1000, 1020):
        values = {"source":"x","policy":"raw_direct","seed":seed,"mean_reward":1.,"mean_terminal_score":1.,"mean_dropout_rate":0.,"mean_delay_days":0.,"mean_delayed_count":0.,"mean_retained_choice_ratio":1.,"arm":"raw_direct","checkpoint":"best_model","checkpoint_timestep":1,"checkpoint_sha256":"a"*64,"evaluation_partition":"selection" if seed < 1005 else "primary_test"}
        rows.append({key: values[key] for key in reversed(evaluator.EVALUATION_COLUMNS)})
    all_path, _ = evaluator.write_arm_evaluations(tmp_path, "raw_direct", list(reversed(rows)))
    with all_path.open(encoding="utf-8") as stream:
        loaded = list(csv.DictReader(stream))
    assert list(loaded[0]) == list(evaluator.EVALUATION_COLUMNS)
    assert [int(row["seed"]) for row in loaded] == list(range(1000, 1020))


def test_evaluate_comparison_artifacts_writes_complete_paired_outputs_and_manifest(tmp_path, monkeypatch):
    from comparison import checkpoint_evaluator as evaluator
    root = tmp_path; (root / "manifest.json").write_text('{"sentinel":true}', encoding="utf-8")
    refs = {arm: {name: evaluator.CheckpointRef(root / arm / f"{name}.sb3", "final" if name == "final" else ("common_step" if name == "common" else "best_model"), 10_000, arm[0]*64) for name in ("selected","final","common")} for arm in evaluator.ARMS}
    monkeypatch.setattr(evaluator, "select_common_timestep", lambda *a, **k: 10_000)
    monkeypatch.setattr(evaluator, "readable_checkpoint_inventory", lambda directory, *a, **k: {10_000: Path(directory) / "common.sb3"})
    monkeypatch.setattr(evaluator, "resolve_final_checkpoint", lambda d, **k: refs["raw_direct" if "raw" in str(d) else "candidate_cnn"]["final"])
    monkeypatch.setattr(evaluator, "resolve_selected_or_fallback", lambda d, **k: refs["raw_direct" if "raw" in str(d) else "candidate_cnn"]["selected"])
    monkeypatch.setattr(evaluator, "_verified_ref", lambda p, *a, **k: refs["raw_direct" if "raw" in str(p) else "candidate_cnn"]["common"])
    def fake_eval(_path, _cfg, scenarios, label, arm, _loader):
        return [{key: value for key, value in zip(evaluator.EVALUATION_COLUMNS, ("x",arm,s,1.,1.,0.,0.,0.,1.,arm,label,10_000,arm[0]*64,"selection" if s<1005 else "primary_test"))} for s in range(1000,1020)]
    monkeypatch.setattr(evaluator, "evaluate_checkpoint", fake_eval)
    evaluator.evaluate_comparison_artifacts(root, root / "raw_direct", root / "candidate_cnn", [{"seed":s} for s in range(1000,1020)], {}, {}, model_loader=object())
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8")); assert manifest["sentinel"] is True
    assert set(manifest["checkpoints"]) == set(evaluator.ARMS)
    expected = {arm: {name: {"path": f"{arm}/{name}.sb3", "label": ("final" if name == "final" else ("common_step" if name == "common" else "best_model")), "sha256": arm[0] * 64, "timestep": 10_000} for name in ("selected", "final", "common")} for arm in evaluator.ARMS}
    assert manifest["checkpoints"] == expected
    with (root / "comparison" / "common_step_evaluation.csv").open(encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 40


def test_evaluate_comparison_artifacts_failure_does_not_partially_update_manifest(tmp_path, monkeypatch):
    from comparison import checkpoint_evaluator as evaluator
    original = b'{"sentinel":true}'; (tmp_path / "manifest.json").write_bytes(original)
    monkeypatch.setattr(evaluator, "select_common_timestep", lambda *a, **k: (_ for _ in ()).throw(evaluator.PartialResultError("fail")))
    with pytest.raises(evaluator.PartialResultError):
        evaluator.evaluate_comparison_artifacts(tmp_path, tmp_path / "raw_direct", tmp_path / "candidate_cnn", [], {}, {})
    assert (tmp_path / "manifest.json").read_bytes() == original


def test_manifest_update_uses_atomic_publication(tmp_path, monkeypatch):
    from comparison import checkpoint_evaluator as evaluator
    path = tmp_path / "manifest.json"; path.write_text('{"sentinel":true}', encoding="utf-8")
    calls = []
    monkeypatch.setattr(evaluator, "atomic_write_json", lambda target, payload: calls.append((target, payload)))
    evaluator.update_checkpoint_manifest(path, "raw_direct", {"final": evaluator.CheckpointRef(Path("raw/final.sb3"), "final", 1, "a" * 64)})
    assert calls and path.read_text(encoding="utf-8") == '{"sentinel":true}'


def test_manifest_update_rejects_duplicate_json_keys(tmp_path):
    from comparison import checkpoint_evaluator as evaluator

    path = tmp_path / "manifest.json"; path.write_text('{"checkpoints":{},"checkpoints":{}}', encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        evaluator.update_checkpoint_manifest(path, "raw_direct", {"final": evaluator.CheckpointRef(Path("raw/final.sb3"), "final", 1, "a" * 64)})


def test_evaluate_comparison_artifacts_rejects_outside_arm_directory_without_manifest_change(tmp_path):
    from comparison import checkpoint_evaluator as evaluator
    original = b'{"sentinel":true}'; (tmp_path / "manifest.json").write_bytes(original)
    outside = tmp_path.parent / "outside"; outside.mkdir(exist_ok=True)
    with pytest.raises(evaluator.PartialResultError):
        evaluator.evaluate_comparison_artifacts(tmp_path, tmp_path / "sub" / ".." / ".." / "outside", tmp_path / "candidate_cnn", [], {}, {})
    assert (tmp_path / "manifest.json").read_bytes() == original


def test_evaluate_comparison_artifacts_rejects_symlink_escape_without_manifest_change(tmp_path):
    from comparison import checkpoint_evaluator as evaluator
    import os
    original = b'{"sentinel":true}'; (tmp_path / "manifest.json").write_bytes(original)
    outside = tmp_path.parent / "outside_arm"; outside.mkdir(exist_ok=True)
    link = tmp_path / "raw_direct"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink unavailable: {error}")
    with pytest.raises(evaluator.PartialResultError):
        evaluator.evaluate_comparison_artifacts(tmp_path, link, tmp_path / "candidate_cnn", [], {}, {})
    assert (tmp_path / "manifest.json").read_bytes() == original


def test_atomic_publication_failure_preserves_manifest_bytes(tmp_path, monkeypatch):
    from comparison import checkpoint_evaluator as evaluator
    path = tmp_path / "manifest.json"; original = b'{"sentinel":true}'; path.write_bytes(original)
    monkeypatch.setattr(evaluator, "atomic_write_json", lambda *_: (_ for _ in ()).throw(OSError("replace failed")))
    with pytest.raises(OSError, match="replace failed"):
        evaluator.update_checkpoint_manifest(path, "raw_direct", {"final": evaluator.CheckpointRef(Path("raw/final.sb3"), "final", 1, "a" * 64)})
    assert path.read_bytes() == original
