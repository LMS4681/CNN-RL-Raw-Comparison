from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
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


def _evaluation_rows(
    arm: str,
    *,
    checkpoint: str = "best_model",
    timestep: int = 50_000,
    digest: str | None = None,
) -> list[dict]:
    from comparison.checkpoint_evaluator import EVALUATION_COLUMNS

    checkpoint_digest = digest or (("a" if arm == "raw_direct" else "b") * 64)
    return [
        dict(
            zip(
                EVALUATION_COLUMNS,
                (
                    "holdout_fixed20",
                    arm,
                    seed,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    arm,
                    checkpoint,
                    timestep,
                    checkpoint_digest,
                    "selection" if seed < 1005 else "primary_test",
                ),
            )
        )
        for seed in range(1000, 1020)
    ]


def _stub_training_evidence(
    monkeypatch,
    evaluator,
    arm_root: Path,
    selected,
    final,
    *,
    selection_outcome: str = "best_model",
    fallback_reason: str | None = None,
):
    runtime_path = arm_root / "runtime_metrics.json"
    receipt_path = arm_root / "training_completion.json"
    runtime = {
        "selected_checkpoint_timestep": selected.timestep,
        "selection_count": 5 if selection_outcome == "best_model" else 0,
        "selection_tuple": [1.0, 0.0, 0.0]
        if selection_outcome == "best_model"
        else None,
        "selection_outcome": selection_outcome,
        "fallback_reason": fallback_reason,
        "checkpoint_identity": {
            "filename": selected.path.name,
            "sha256": selected.sha256,
        },
    }
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    receipt = {
        "config_sha256": "c" * 64,
        "final_timestep": final.timestep,
        "checkpoint_file": final.path.name,
        "checkpoint_sha256": final.sha256,
        "artifact_sha256": {
            "runtime_metrics.json": _sha256(runtime_path),
            "best_model.sb3": (
                selected.sha256 if selection_outcome == "best_model" else None
            ),
        },
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(
        evaluator,
        "read_runtime_metrics",
        lambda _path: runtime,
        raising=False,
    )
    monkeypatch.setattr(
        evaluator,
        "read_training_completion",
        lambda _path: receipt,
        raising=False,
    )
    return runtime, receipt


def _common_artifact_fixture(tmp_path, evaluator):
    root = tmp_path.resolve()
    checkpoint_refs = {}
    manifest_checkpoints = {}
    protected_bytes = {}
    for arm in evaluator.ARMS:
        arm_root = root / arm
        checkpoint_root = arm_root / "checkpoints"
        checkpoint_root.mkdir(parents=True)
        best = arm_root / "best_model.sb3"
        final = checkpoint_root / "final.sb3"
        common = checkpoint_root / "common_10000.sb3"
        best.write_bytes(f"{arm}:best".encode())
        final.write_bytes(f"{arm}:final".encode())
        common.write_bytes(f"{arm}:common".encode())
        per_arm_csv = arm_root / "evaluation_scenarios.csv"
        primary_csv = arm_root / "evaluation_primary_test.csv"
        per_arm_csv.write_bytes(f"{arm}:selected-csv".encode())
        primary_csv.write_bytes(f"{arm}:primary-csv".encode())
        protected_bytes[best] = best.read_bytes()
        protected_bytes[final] = final.read_bytes()
        protected_bytes[per_arm_csv] = per_arm_csv.read_bytes()
        protected_bytes[primary_csv] = primary_csv.read_bytes()
        checkpoint_refs[arm] = evaluator.CheckpointRef(
            common,
            "common_step",
            10_000,
            _sha256(common),
        )
        manifest_checkpoints[arm] = {
            "selected": {
                "path": f"{arm}/best_model.sb3",
                "label": "best_model",
                "sha256": _sha256(best),
                "timestep": 9_000,
            },
            "final": {
                "path": f"{arm}/checkpoints/final.sb3",
                "label": "final",
                "sha256": _sha256(final),
                "timestep": 11_000,
            },
        }
        (arm_root / "run_config.json").write_text("{}", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "config_sha256": "c" * 64,
        "scenario_sha256": "d" * 64,
        "sentinel": True,
        "checkpoints": manifest_checkpoints,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, checkpoint_refs, protected_bytes, manifest


def _arm_artifact_fixture(tmp_path, monkeypatch, evaluator):
    root = tmp_path.resolve()
    arm = "raw_direct"
    arm_root = root / arm
    arm_root.mkdir()
    selected_path = arm_root / "best_model.sb3"
    final_path = arm_root / "checkpoints" / "final.sb3"
    selected_path.write_bytes(b"selected")
    final_path.parent.mkdir()
    final_path.write_bytes(b"final")
    selected = evaluator.CheckpointRef(
        selected_path, "best_model", 50_000, _sha256(selected_path)
    )
    final = evaluator.CheckpointRef(
        final_path, "final", 60_000, _sha256(final_path)
    )
    monkeypatch.setattr(
        evaluator,
        "_archive_timestep",
        lambda path, _loader: (
            selected.timestep if Path(path) == selected_path else final.timestep
        ),
    )
    manifest = {
        "schema_version": 1,
        "config_sha256": "c" * 64,
        "scenario_sha256": "d" * 64,
        "checkpoints": {},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        evaluator, "resolve_final_checkpoint", lambda *_args, **_kwargs: final
    )
    monkeypatch.setattr(
        evaluator,
        "resolve_selection_decision",
        lambda *_args, **_kwargs: evaluator.SelectionDecision(
            selected, "best_model", None, 5, [1.0, 0.0, 0.0]
        ),
    )
    _stub_training_evidence(monkeypatch, evaluator, arm_root, selected, final)
    calls = []

    def evaluate(*_args, **_kwargs):
        calls.append(arm)
        return _evaluation_rows(
            arm,
            checkpoint="best_model",
            timestep=selected.timestep,
            digest=selected.sha256,
        )

    monkeypatch.setattr(evaluator, "evaluate_checkpoint", evaluate)
    scenarios = [{"seed": seed} for seed in range(1000, 1020)]
    return root, arm, selected, final, scenarios, calls


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


def test_evaluate_arm_artifacts_publishes_csv_manifest_then_exact_marker(
    tmp_path, monkeypatch
):
    from comparison import checkpoint_evaluator as evaluator

    root = tmp_path.resolve()
    arm = "raw_direct"
    arm_root = root / arm
    arm_root.mkdir()
    selected_path = arm_root / "best_model.sb3"
    final_path = arm_root / "checkpoints" / "final.sb3"
    selected_path.write_bytes(b"selected")
    final_path.parent.mkdir()
    final_path.write_bytes(b"final")
    selected = evaluator.CheckpointRef(
        selected_path, "best_model", 50_000, _sha256(selected_path)
    )
    final = evaluator.CheckpointRef(
        final_path, "final", 60_000, _sha256(final_path)
    )
    monkeypatch.setattr(
        evaluator,
        "_archive_timestep",
        lambda path, _loader: (
            selected.timestep if Path(path) == selected_path else final.timestep
        ),
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config_sha256": "c" * 64,
                "scenario_sha256": "d" * 64,
                "sentinel": True,
                "checkpoints": {"candidate_cnn": {"sentinel": True}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        evaluator, "resolve_final_checkpoint", lambda *_args, **_kwargs: final
    )
    monkeypatch.setattr(
        evaluator,
        "resolve_selection_decision",
        lambda *_args, **_kwargs: evaluator.SelectionDecision(
            selected, "best_model", None, 5, [1.0, 0.0, 0.0]
        ),
    )
    runtime, receipt = _stub_training_evidence(
        monkeypatch, evaluator, arm_root, selected, final
    )
    calls = []

    def fake_evaluate(path, config, scenarios, label, received_arm, model_loader):
        calls.append(
            (path, dict(config), list(scenarios), label, received_arm, model_loader)
        )
        return _evaluation_rows(
            arm,
            checkpoint="best_model",
            timestep=selected.timestep,
            digest=selected.sha256,
        )

    monkeypatch.setattr(evaluator, "evaluate_checkpoint", fake_evaluate)
    scenarios = [{"seed": seed} for seed in range(1000, 1020)]
    marker = evaluator.evaluate_arm_artifacts(
        root,
        arm,
        scenarios,
        {"run": "config"},
        config_sha256="c" * 64,
        scenario_sha256="d" * 64,
        model_loader="loader",
    )

    assert calls == [
        (
            selected_path,
            {"run": "config"},
            scenarios,
            "best_model",
            arm,
            "loader",
        )
    ]
    with (arm_root / "evaluation_scenarios.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        assert len(list(csv.DictReader(stream))) == 20
    with (arm_root / "evaluation_primary_test.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        assert [int(row["seed"]) for row in csv.DictReader(stream)] == list(
            range(1005, 1020)
        )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sentinel"] is True
    assert manifest["checkpoints"]["candidate_cnn"] == {"sentinel": True}
    assert set(manifest["checkpoints"][arm]) == {"selected", "final"}
    expected_marker = {
        "schema_version": 1,
        "arm": arm,
        "config_sha256": "c" * 64,
        "scenario_sha256": "d" * 64,
        "checkpoints": manifest["checkpoints"][arm],
        "artifacts": {
            "evaluation_scenarios.csv": _sha256(
                arm_root / "evaluation_scenarios.csv"
            ),
            "evaluation_primary_test.csv": _sha256(
                arm_root / "evaluation_primary_test.csv"
            ),
            "training_completion.json": _sha256(
                arm_root / "training_completion.json"
            ),
            "runtime_metrics.json": _sha256(
                arm_root / "runtime_metrics.json"
            ),
        },
        "evaluation_seed_count": 20,
        "primary_test_seed_count": 15,
        "selection_outcome": "best_model",
        "fallback_reason": None,
    }
    assert marker == expected_marker
    assert json.loads(
        (arm_root / "evaluation_stage.json").read_text(encoding="utf-8")
    ) == expected_marker
    assert evaluator.validate_arm_evaluation_stage(
        root,
        arm,
        expected_config_sha256="c" * 64,
        expected_scenario_sha256="d" * 64,
    ) == expected_marker
    calls.clear()
    assert evaluator.evaluate_arm_artifacts(
        root,
        arm,
        scenarios,
        {"run": "config"},
        config_sha256="c" * 64,
        scenario_sha256="d" * 64,
        model_loader="loader",
    ) == expected_marker
    assert calls == []
    runtime["selected_checkpoint_timestep"] += 1
    runtime_path = arm_root / "runtime_metrics.json"
    receipt_path = arm_root / "training_completion.json"
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    receipt["artifact_sha256"]["runtime_metrics.json"] = _sha256(runtime_path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    forged_marker = json.loads(json.dumps(expected_marker))
    forged_marker["artifacts"]["runtime_metrics.json"] = _sha256(runtime_path)
    forged_marker["artifacts"]["training_completion.json"] = _sha256(
        receipt_path
    )
    (arm_root / "evaluation_stage.json").write_text(
        json.dumps(forged_marker), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="runtime selected fields"):
        evaluator.validate_arm_evaluation_stage(root, arm)


def test_evaluate_arm_artifacts_publishes_real_fallback_final(
    tmp_path, monkeypatch
):
    from comparison import checkpoint_evaluator as evaluator

    root = tmp_path.resolve()
    arm = "raw_direct"
    arm_root = root / arm
    arm_root.mkdir()
    final_path = _selection_fixture(arm_root)
    final = evaluator.CheckpointRef(
        final_path, "final", 60_000, _sha256(final_path)
    )
    selected = evaluator.CheckpointRef(
        final_path, "fallback_final", 60_000, final.sha256
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config_sha256": "c" * 64,
                "scenario_sha256": "d" * 64,
                "checkpoints": {},
            }
        ),
        encoding="utf-8",
    )
    _stub_training_evidence(
        monkeypatch,
        evaluator,
        arm_root,
        selected,
        final,
        selection_outcome="fallback_final",
        fallback_reason="selection_not_run",
    )
    calls = []

    def evaluate(path, _config, _scenarios, label, received_arm, _loader):
        calls.append((Path(path), label, received_arm))
        return _evaluation_rows(
            arm,
            checkpoint="fallback_final",
            timestep=selected.timestep,
            digest=selected.sha256,
        )

    monkeypatch.setattr(evaluator, "evaluate_checkpoint", evaluate)
    marker = evaluator.evaluate_arm_artifacts(
        root,
        arm,
        [{"seed": seed} for seed in range(1000, 1020)],
        {"run": "config"},
        config_sha256="c" * 64,
        scenario_sha256="d" * 64,
        model_loader=_text_loader,
    )

    assert calls == [(final_path, "fallback_final", arm)]
    assert marker["selection_outcome"] == "fallback_final"
    assert marker["fallback_reason"] == "selection_not_run"
    assert marker["checkpoints"]["selected"] == {
        "path": "raw_direct/checkpoints/final_60000.sb3",
        "label": "fallback_final",
        "sha256": final.sha256,
        "timestep": 60_000,
    }
    assert marker["checkpoints"]["final"] == {
        **marker["checkpoints"]["selected"],
        "label": "final",
    }


@pytest.mark.parametrize("missing_key", ["config_sha256", "scenario_sha256"])
def test_evaluate_arm_requires_exact_root_manifest_hashes(
    tmp_path, monkeypatch, missing_key
):
    from comparison import checkpoint_evaluator as evaluator

    root, arm, _selected, _final, scenarios, calls = _arm_artifact_fixture(
        tmp_path, monkeypatch, evaluator
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop(missing_key)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(evaluator.PartialResultError, match="root manifest"):
        evaluator.evaluate_arm_artifacts(
            root,
            arm,
            scenarios,
            {"run": "config"},
            config_sha256="c" * 64,
            scenario_sha256="d" * 64,
        )
    assert calls == []


@pytest.mark.parametrize("missing_key", ["config_sha256", "scenario_sha256"])
def test_arm_marker_validation_requires_exact_root_manifest_hashes(
    tmp_path, monkeypatch, missing_key
):
    from comparison import checkpoint_evaluator as evaluator

    root, arm, _selected, _final, scenarios, _calls = _arm_artifact_fixture(
        tmp_path, monkeypatch, evaluator
    )
    evaluator.evaluate_arm_artifacts(
        root,
        arm,
        scenarios,
        {"run": "config"},
        config_sha256="c" * 64,
        scenario_sha256="d" * 64,
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop(missing_key)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="root manifest"):
        evaluator.validate_arm_evaluation_stage(root, arm)


def test_evaluate_arm_artifacts_failure_invalidates_old_marker(tmp_path, monkeypatch):
    from comparison import checkpoint_evaluator as evaluator

    arm_root = tmp_path / "raw_direct"
    arm_root.mkdir()
    marker = arm_root / "evaluation_stage.json"
    marker.write_text('{"stale":true}', encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config_sha256": "c" * 64,
                "scenario_sha256": "d" * 64,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        evaluator,
        "resolve_final_checkpoint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            evaluator.PartialResultError("evaluation failed")
        ),
    )

    with pytest.raises(evaluator.PartialResultError, match="evaluation failed"):
        evaluator.evaluate_arm_artifacts(
            tmp_path,
            "raw_direct",
            [{"seed": seed} for seed in range(1000, 1020)],
            {},
            config_sha256="c" * 64,
            scenario_sha256="d" * 64,
        )
    assert not marker.exists()


def test_evaluate_arm_marker_is_not_trusted_after_csv_or_manifest_tamper(
    tmp_path, monkeypatch
):
    from comparison import checkpoint_evaluator as evaluator

    root = tmp_path.resolve()
    arm_root = root / "raw_direct"
    arm_root.mkdir()
    selected_path = arm_root / "best_model.sb3"
    final_path = arm_root / "checkpoints" / "final.sb3"
    selected_path.write_bytes(b"selected")
    final_path.parent.mkdir()
    final_path.write_bytes(b"final")
    selected = evaluator.CheckpointRef(
        selected_path, "best_model", 50_000, _sha256(selected_path)
    )
    final = evaluator.CheckpointRef(final_path, "final", 60_000, _sha256(final_path))
    monkeypatch.setattr(
        evaluator,
        "_archive_timestep",
        lambda path, _loader: (
            selected.timestep if Path(path) == selected_path else final.timestep
        ),
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config_sha256": "c" * 64,
                "scenario_sha256": "d" * 64,
                "checkpoints": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluator, "resolve_final_checkpoint", lambda *_a, **_k: final)
    monkeypatch.setattr(
        evaluator,
        "resolve_selection_decision",
        lambda *_a, **_k: evaluator.SelectionDecision(
            selected, "best_model", None, 5, [1.0, 0.0, 0.0]
        ),
    )
    _stub_training_evidence(monkeypatch, evaluator, arm_root, selected, final)
    monkeypatch.setattr(
        evaluator,
        "evaluate_checkpoint",
        lambda *_a, **_k: _evaluation_rows(
            "raw_direct",
            checkpoint="best_model",
            timestep=selected.timestep,
            digest=selected.sha256,
        ),
    )
    evaluator.evaluate_arm_artifacts(
        root,
        "raw_direct",
        [{"seed": seed} for seed in range(1000, 1020)],
        {},
        config_sha256="c" * 64,
        scenario_sha256="d" * 64,
    )

    csv_path = arm_root / "evaluation_scenarios.csv"
    original_csv = csv_path.read_bytes()
    csv_path.write_bytes(original_csv + b"tamper")
    with pytest.raises(ValueError, match="evaluation artifact hash"):
        evaluator.validate_arm_evaluation_stage(root, "raw_direct")
    csv_path.write_bytes(original_csv)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoints"]["raw_direct"]["common"] = {
        "path": "raw_direct/checkpoints/common.sb3",
        "label": "common_step",
        "sha256": "f" * 64,
        "timestep": 40_000,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert evaluator.validate_arm_evaluation_stage(root, "raw_direct")["arm"] == (
        "raw_direct"
    )
    manifest["checkpoints"]["raw_direct"]["selected"]["timestep"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation checkpoint manifest"):
        evaluator.validate_arm_evaluation_stage(root, "raw_direct")


@pytest.mark.parametrize(
    "artifact_name",
    [
        "evaluation_stage.json",
        "evaluation_scenarios.csv",
        "runtime_metrics.json",
        "training_completion.json",
    ],
)
def test_arm_validator_rejects_symlinked_stage_artifacts(
    tmp_path, monkeypatch, artifact_name
):
    from comparison import checkpoint_evaluator as evaluator

    root, arm, _selected, _final, scenarios, _calls = _arm_artifact_fixture(
        tmp_path, monkeypatch, evaluator
    )
    evaluator.evaluate_arm_artifacts(
        root,
        arm,
        scenarios,
        {"run": "config"},
        config_sha256="c" * 64,
        scenario_sha256="d" * 64,
    )
    artifact = root / arm / artifact_name
    outside = root / f"outside-{artifact_name}"
    outside.write_bytes(artifact.read_bytes())
    artifact.unlink()
    try:
        os.symlink(outside, artifact)
    except OSError as error:
        pytest.skip(f"symlink creation was rejected by the OS: {error}")

    with pytest.raises(ValueError):
        evaluator.validate_arm_evaluation_stage(root, arm)


def test_evaluate_arm_marker_write_failure_leaves_no_trusted_marker(
    tmp_path, monkeypatch
):
    from comparison import checkpoint_evaluator as evaluator

    root = tmp_path.resolve()
    arm_root = root / "raw_direct"
    arm_root.mkdir()
    selected_path = arm_root / "best_model.sb3"
    final_path = arm_root / "checkpoints" / "final.sb3"
    selected_path.write_bytes(b"selected")
    final_path.parent.mkdir()
    final_path.write_bytes(b"final")
    selected = evaluator.CheckpointRef(
        selected_path, "best_model", 50_000, _sha256(selected_path)
    )
    final = evaluator.CheckpointRef(final_path, "final", 60_000, _sha256(final_path))
    monkeypatch.setattr(
        evaluator,
        "_archive_timestep",
        lambda path, _loader: (
            selected.timestep if Path(path) == selected_path else final.timestep
        ),
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config_sha256": "c" * 64,
                "scenario_sha256": "d" * 64,
                "checkpoints": {},
            }
        ),
        encoding="utf-8",
    )
    (arm_root / "evaluation_stage.json").write_text("stale", encoding="utf-8")
    monkeypatch.setattr(evaluator, "resolve_final_checkpoint", lambda *_a, **_k: final)
    monkeypatch.setattr(
        evaluator,
        "resolve_selection_decision",
        lambda *_a, **_k: evaluator.SelectionDecision(
            selected, "best_model", None, 5, [1.0, 0.0, 0.0]
        ),
    )
    _stub_training_evidence(monkeypatch, evaluator, arm_root, selected, final)
    monkeypatch.setattr(
        evaluator,
        "evaluate_checkpoint",
        lambda *_a, **_k: _evaluation_rows(
            "raw_direct",
            checkpoint="best_model",
            timestep=selected.timestep,
            digest=selected.sha256,
        ),
    )
    original_atomic_write_json = evaluator.atomic_write_json

    def fail_marker(path, payload):
        if Path(path).name == "evaluation_stage.json":
            raise OSError("marker replace failed")
        return original_atomic_write_json(path, payload)

    monkeypatch.setattr(evaluator, "atomic_write_json", fail_marker)
    with pytest.raises(OSError, match="marker replace failed"):
        evaluator.evaluate_arm_artifacts(
            root,
            "raw_direct",
            [{"seed": seed} for seed in range(1000, 1020)],
            {},
            config_sha256="c" * 64,
            scenario_sha256="d" * 64,
        )
    assert not (arm_root / "evaluation_stage.json").exists()


def test_evaluate_arm_manifest_failure_leaves_no_marker_and_retry_is_coherent(
    tmp_path, monkeypatch
):
    from comparison import checkpoint_evaluator as evaluator

    root, arm, selected, _final, scenarios, calls = _arm_artifact_fixture(
        tmp_path, monkeypatch, evaluator
    )
    marker_path = root / arm / "evaluation_stage.json"
    marker_path.write_text('{"stale":true}', encoding="utf-8")
    original_atomic_write_json = evaluator.atomic_write_json
    fail_manifest = {"value": True}

    def fail_manifest_once(path, payload):
        if Path(path).name == "manifest.json" and fail_manifest["value"]:
            fail_manifest["value"] = False
            assert (root / arm / "evaluation_scenarios.csv").is_file()
            assert (root / arm / "evaluation_primary_test.csv").is_file()
            raise OSError("manifest replace failed")
        return original_atomic_write_json(path, payload)

    monkeypatch.setattr(evaluator, "atomic_write_json", fail_manifest_once)
    arguments = (root, arm, scenarios, {"run": "config"})
    keywords = {
        "config_sha256": "c" * 64,
        "scenario_sha256": "d" * 64,
    }
    with pytest.raises(OSError, match="manifest replace failed"):
        evaluator.evaluate_arm_artifacts(*arguments, **keywords)
    assert not marker_path.exists()
    assert calls == [arm]

    marker = evaluator.evaluate_arm_artifacts(*arguments, **keywords)
    assert calls == [arm, arm]
    assert marker_path.is_file()
    assert marker == evaluator.validate_arm_evaluation_stage(root, arm)
    with (root / arm / "evaluation_scenarios.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert {
        (
            row["checkpoint"],
            int(row["checkpoint_timestep"]),
            row["checkpoint_sha256"],
        )
        for row in rows
    } == {("best_model", selected.timestep, selected.sha256)}


def test_evaluate_arm_rejects_rows_not_bound_to_resolved_selected_checkpoint(
    tmp_path, monkeypatch
):
    from comparison import checkpoint_evaluator as evaluator

    root = tmp_path.resolve()
    arm_root = root / "raw_direct"
    arm_root.mkdir()
    selected_path = arm_root / "best_model.sb3"
    final_path = arm_root / "checkpoints" / "final.sb3"
    selected_path.write_bytes(b"selected")
    final_path.parent.mkdir()
    final_path.write_bytes(b"final")
    selected = evaluator.CheckpointRef(
        selected_path, "best_model", 50_000, _sha256(selected_path)
    )
    final = evaluator.CheckpointRef(final_path, "final", 60_000, _sha256(final_path))
    monkeypatch.setattr(
        evaluator,
        "_archive_timestep",
        lambda path, _loader: (
            selected.timestep if Path(path) == selected_path else final.timestep
        ),
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config_sha256": "c" * 64,
                "scenario_sha256": "d" * 64,
                "checkpoints": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluator, "resolve_final_checkpoint", lambda *_a, **_k: final)
    monkeypatch.setattr(
        evaluator,
        "resolve_selection_decision",
        lambda *_a, **_k: evaluator.SelectionDecision(
            selected, "best_model", None, 5, [1.0, 0.0, 0.0]
        ),
    )
    monkeypatch.setattr(
        evaluator,
        "evaluate_checkpoint",
        lambda *_a, **_k: _evaluation_rows(
            "raw_direct",
            checkpoint="best_model",
            timestep=selected.timestep,
            digest="0" * 64,
        ),
    )

    with pytest.raises(evaluator.PartialResultError, match="selected checkpoint"):
        evaluator.evaluate_arm_artifacts(
            root,
            "raw_direct",
            [{"seed": seed} for seed in range(1000, 1020)],
            {},
            config_sha256="c" * 64,
            scenario_sha256="d" * 64,
        )
    assert not (arm_root / "evaluation_stage.json").exists()
    assert not (arm_root / "evaluation_scenarios.csv").exists()


def test_second_atomic_csv_replace_failure_preserves_complete_files_without_marker(
    tmp_path, monkeypatch
):
    from comparison import checkpoint_evaluator as evaluator

    root = tmp_path.resolve()
    arm_root = root / "raw_direct"
    arm_root.mkdir()
    selected_path = arm_root / "best_model.sb3"
    final_path = arm_root / "checkpoints" / "final.sb3"
    selected_path.write_bytes(b"selected")
    final_path.parent.mkdir()
    final_path.write_bytes(b"final")
    selected = evaluator.CheckpointRef(
        selected_path, "best_model", 50_000, _sha256(selected_path)
    )
    final = evaluator.CheckpointRef(final_path, "final", 60_000, _sha256(final_path))
    monkeypatch.setattr(
        evaluator,
        "_archive_timestep",
        lambda path, _loader: (
            selected.timestep if Path(path) == selected_path else final.timestep
        ),
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config_sha256": "c" * 64,
                "scenario_sha256": "d" * 64,
                "checkpoints": {},
            }
        ),
        encoding="utf-8",
    )
    old_rows = _evaluation_rows(
        "raw_direct",
        checkpoint="best_model",
        timestep=selected.timestep,
        digest=selected.sha256,
    )
    for row in old_rows:
        row["mean_reward"] = 0.0
    evaluator.write_arm_evaluations(root, "raw_direct", old_rows)
    old_primary = (arm_root / "evaluation_primary_test.csv").read_bytes()
    (arm_root / "evaluation_stage.json").write_text("stale", encoding="utf-8")
    monkeypatch.setattr(evaluator, "resolve_final_checkpoint", lambda *_a, **_k: final)
    monkeypatch.setattr(
        evaluator,
        "resolve_selection_decision",
        lambda *_a, **_k: evaluator.SelectionDecision(
            selected, "best_model", None, 5, [1.0, 0.0, 0.0]
        ),
    )
    _stub_training_evidence(monkeypatch, evaluator, arm_root, selected, final)
    monkeypatch.setattr(
        evaluator,
        "evaluate_checkpoint",
        lambda *_a, **_k: _evaluation_rows(
            "raw_direct",
            checkpoint="best_model",
            timestep=selected.timestep,
            digest=selected.sha256,
        ),
    )
    original_replace = evaluator.os.replace

    def fail_primary(source, destination):
        if Path(destination).name == "evaluation_primary_test.csv":
            raise OSError("second CSV replace failed")
        return original_replace(source, destination)

    monkeypatch.setattr(evaluator.os, "replace", fail_primary)
    with pytest.raises(OSError, match="second CSV replace failed"):
        evaluator.evaluate_arm_artifacts(
            root,
            "raw_direct",
            [{"seed": seed} for seed in range(1000, 1020)],
            {},
            config_sha256="c" * 64,
            scenario_sha256="d" * 64,
        )

    assert not (arm_root / "evaluation_stage.json").exists()
    assert (arm_root / "evaluation_primary_test.csv").read_bytes() == old_primary
    for name, count in (
        ("evaluation_scenarios.csv", 20),
        ("evaluation_primary_test.csv", 15),
    ):
        with (arm_root / name).open(encoding="utf-8", newline="") as stream:
            assert len(list(csv.DictReader(stream))) == count
    assert not list(arm_root.glob(".*.tmp"))


def test_selected_checkpoint_symlink_inside_arm_is_rejected_before_model_load(
    tmp_path,
):
    from comparison import checkpoint_evaluator as evaluator

    arm_root = tmp_path / "raw_direct"
    arm_root.mkdir()
    target = arm_root / "real_best.sb3"
    target.write_bytes(b"checkpoint:50000")
    link = arm_root / "best_model.sb3"
    try:
        os.symlink(target, link)
    except OSError as error:
        pytest.skip(f"symlink unavailable: {error}")
    reference = evaluator.CheckpointRef(
        link, "best_model", 50_000, _sha256(target)
    )
    loads = []

    def loader(*args, **kwargs):
        loads.append((args, kwargs))
        return SimpleNamespace(num_timesteps=50_000)

    with pytest.raises(evaluator.PartialResultError, match="regular"):
        evaluator._stable_checkpoint_reference(
            reference, arm_root, model_loader=loader
        )
    assert loads == []


def test_arm_marker_validator_checks_actual_checkpoint_timestep(
    tmp_path, monkeypatch
):
    from comparison import checkpoint_evaluator as evaluator

    root = tmp_path.resolve()
    arm_root = root / "raw_direct"
    checkpoints = arm_root / "checkpoints"
    checkpoints.mkdir(parents=True)
    selected_path = arm_root / "best_model.sb3"
    final_path = checkpoints / "final.sb3"
    selected_path.write_bytes(b"selected")
    final_path.write_bytes(b"final")
    selected = evaluator.CheckpointRef(
        selected_path, "best_model", 50_000, _sha256(selected_path)
    )
    final = evaluator.CheckpointRef(final_path, "final", 60_000, _sha256(final_path))
    monkeypatch.setattr(
        evaluator,
        "_archive_timestep",
        lambda path, _loader: (
            selected.timestep if Path(path) == selected_path else final.timestep
        ),
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config_sha256": "c" * 64,
                "scenario_sha256": "d" * 64,
                "checkpoints": {},
            }
        ),
        encoding="utf-8",
    )
    _stub_training_evidence(monkeypatch, evaluator, arm_root, selected, final)
    monkeypatch.setattr(evaluator, "resolve_final_checkpoint", lambda *_a, **_k: final)
    monkeypatch.setattr(
        evaluator,
        "resolve_selection_decision",
        lambda *_a, **_k: evaluator.SelectionDecision(
            selected, "best_model", None, 5, [1.0, 0.0, 0.0]
        ),
    )
    monkeypatch.setattr(
        evaluator,
        "evaluate_checkpoint",
        lambda *_a, **_k: _evaluation_rows(
            "raw_direct",
            checkpoint="best_model",
            timestep=selected.timestep,
            digest=selected.sha256,
        ),
    )
    evaluator.evaluate_arm_artifacts(
        root,
        "raw_direct",
        [{"seed": seed} for seed in range(1000, 1020)],
        {},
        config_sha256="c" * 64,
        scenario_sha256="d" * 64,
    )

    with pytest.raises(ValueError, match="checkpoint timestep"):
        evaluator.validate_arm_evaluation_stage(
            root,
            "raw_direct",
            archive_timestep_reader=lambda path: (
                49_999 if Path(path).name == "best_model.sb3" else 60_000
            ),
        )


def test_common_only_evaluation_publishes_caches_combined_refs_and_marker(
    tmp_path, monkeypatch
):
    from comparison import checkpoint_evaluator as evaluator

    root, refs, protected_bytes, original_manifest = _common_artifact_fixture(
        tmp_path, evaluator
    )
    monkeypatch.setattr(evaluator, "select_common_timestep", lambda *_a, **_k: 10_000)
    monkeypatch.setattr(
        evaluator,
        "readable_checkpoint_inventory",
        lambda directory, *_a, **_k: {
            10_000: refs[Path(directory).name].path
        },
    )
    monkeypatch.setattr(
        evaluator,
        "_archive_timestep",
        lambda _path, _loader: 10_000,
    )
    monkeypatch.setattr(
        evaluator,
        "resolve_selected_or_fallback",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("common stage must not resolve selected checkpoints")
        ),
    )
    monkeypatch.setattr(
        evaluator,
        "write_arm_evaluations",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("common stage must not rewrite arm evaluations")
        ),
    )
    calls = []

    def evaluate(_path, _config, _scenarios, label, arm, _loader):
        calls.append(arm)
        return _evaluation_rows(
            arm,
            checkpoint=label,
            timestep=10_000,
            digest=refs[arm].sha256,
        )

    monkeypatch.setattr(evaluator, "evaluate_checkpoint", evaluate)
    marker = evaluator.evaluate_common_step_artifacts(
        root,
        [{"seed": seed} for seed in range(1000, 1020)],
        {arm: {} for arm in evaluator.ARMS},
        config_sha256="c" * 64,
        scenario_sha256="d" * 64,
        model_loader="loader",
    )

    assert calls == list(evaluator.ARMS)
    assert set(marker) == {
        "schema_version",
        "config_sha256",
        "run_config_sha256",
        "scenario_sha256",
        "common_timestep",
        "checkpoints",
        "artifacts",
        "evaluation_seed_count_per_arm",
    }
    assert marker["common_timestep"] == 10_000
    assert set(marker["checkpoints"]) == set(evaluator.ARMS)
    assert set(marker["artifacts"]) == {
        "common_step_raw_direct.cache.json",
        "common_step_candidate_cnn.cache.json",
        "common_step_evaluation.csv",
    }
    comparison = root / "comparison"
    with (comparison / "common_step_evaluation.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert [(row["arm"], int(row["seed"])) for row in rows] == [
        (arm, seed) for arm in evaluator.ARMS for seed in range(1000, 1020)
    ]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sentinel"] is True
    for arm in evaluator.ARMS:
        assert manifest["checkpoints"][arm]["selected"] == (
            original_manifest["checkpoints"][arm]["selected"]
        )
        assert manifest["checkpoints"][arm]["final"] == (
            original_manifest["checkpoints"][arm]["final"]
        )
        assert manifest["checkpoints"][arm]["common"] == marker["checkpoints"][arm]
    assert {path: path.read_bytes() for path in protected_bytes} == protected_bytes
    assert evaluator.validate_common_step_stage(
        root,
        expected_config_sha256="c" * 64,
        expected_scenario_sha256="d" * 64,
        archive_timestep_reader=lambda _path: 10_000,
    ) == marker
    calls.clear()
    assert evaluator.evaluate_common_step_artifacts(
        root,
        [{"seed": seed} for seed in range(1000, 1020)],
        {arm: {} for arm in evaluator.ARMS},
        config_sha256="c" * 64,
        scenario_sha256="d" * 64,
        model_loader="loader",
    ) == marker
    assert calls == []
    manifest_path = root / "manifest.json"
    changed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed_manifest["checkpoints"]["raw_direct"]["selected"]["timestep"] += 1
    manifest_path.write_text(json.dumps(changed_manifest), encoding="utf-8")
    assert evaluator.validate_common_step_stage(
        root, archive_timestep_reader=lambda _path: 10_000
    ) == marker
    changed_manifest["checkpoints"]["raw_direct"]["common"]["timestep"] += 1
    manifest_path.write_text(json.dumps(changed_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="common checkpoint manifest"):
        evaluator.validate_common_step_stage(root)


def test_common_first_arm_cache_survives_crash_and_is_reused(
    tmp_path, monkeypatch
):
    from comparison import checkpoint_evaluator as evaluator

    root, refs, _protected, _manifest = _common_artifact_fixture(tmp_path, evaluator)
    monkeypatch.setattr(evaluator, "select_common_timestep", lambda *_a, **_k: 10_000)
    monkeypatch.setattr(
        evaluator,
        "readable_checkpoint_inventory",
        lambda directory, *_a, **_k: {10_000: refs[Path(directory).name].path},
    )
    monkeypatch.setattr(evaluator, "_archive_timestep", lambda *_a, **_k: 10_000)
    calls = []
    fail_candidate = {"value": True}

    def evaluate(_path, _config, _scenarios, label, arm, _loader):
        calls.append(arm)
        if arm == "candidate_cnn" and fail_candidate["value"]:
            raise RuntimeError("candidate evaluation crash")
        return _evaluation_rows(
            arm,
            checkpoint=label,
            timestep=10_000,
            digest=refs[arm].sha256,
        )

    monkeypatch.setattr(evaluator, "evaluate_checkpoint", evaluate)
    arguments = (
        root,
        [{"seed": seed} for seed in range(1000, 1020)],
        {arm: {} for arm in evaluator.ARMS},
    )
    keywords = {
        "config_sha256": "c" * 64,
        "scenario_sha256": "d" * 64,
        "model_loader": "loader",
    }
    with pytest.raises(RuntimeError, match="candidate evaluation crash"):
        evaluator.evaluate_common_step_artifacts(*arguments, **keywords)
    raw_cache = root / "comparison" / "common_step_raw_direct.cache.json"
    assert raw_cache.is_file()
    assert not (root / "comparison" / "common_step_stage.json").exists()
    assert not (root / "comparison" / "common_step_evaluation.csv").exists()

    fail_candidate["value"] = False
    evaluator.evaluate_common_step_artifacts(*arguments, **keywords)
    assert calls == ["raw_direct", "candidate_cnn", "candidate_cnn"]

    (root / "comparison" / "common_step_candidate_cnn.cache.json").write_text(
        "{malformed", encoding="utf-8"
    )
    (root / "comparison" / "common_step_stage.json").unlink()
    calls.clear()
    evaluator.evaluate_common_step_artifacts(*arguments, **keywords)
    assert calls == ["candidate_cnn"]


def test_common_manifest_failure_preserves_both_caches_for_retry(
    tmp_path, monkeypatch
):
    from comparison import checkpoint_evaluator as evaluator

    root, refs, _protected, original_manifest = _common_artifact_fixture(
        tmp_path, evaluator
    )
    monkeypatch.setattr(
        evaluator,
        "readable_checkpoint_inventory",
        lambda directory, *_a, **_k: {10_000: refs[Path(directory).name].path},
    )
    monkeypatch.setattr(evaluator, "_archive_timestep", lambda *_a, **_k: 10_000)
    calls = []

    def evaluate(_path, _config, _scenarios, label, arm, _loader):
        calls.append(arm)
        return _evaluation_rows(
            arm,
            checkpoint=label,
            timestep=10_000,
            digest=refs[arm].sha256,
        )

    monkeypatch.setattr(evaluator, "evaluate_checkpoint", evaluate)
    original_atomic_write_json = evaluator.atomic_write_json
    fail_manifest = {"value": True}

    def fail_manifest_once(path, payload):
        if Path(path).name == "manifest.json" and fail_manifest["value"]:
            fail_manifest["value"] = False
            for arm in evaluator.ARMS:
                assert (
                    root / "comparison" / f"common_step_{arm}.cache.json"
                ).is_file()
            raise OSError("common manifest replace failed")
        return original_atomic_write_json(path, payload)

    monkeypatch.setattr(evaluator, "atomic_write_json", fail_manifest_once)
    arguments = (
        root,
        [{"seed": seed} for seed in range(1000, 1020)],
        {arm: {} for arm in evaluator.ARMS},
    )
    keywords = {
        "config_sha256": "c" * 64,
        "scenario_sha256": "d" * 64,
        "model_loader": "loader",
    }
    with pytest.raises(
        evaluator.PartialResultError, match="manifest publication failed"
    ):
        evaluator.evaluate_common_step_artifacts(*arguments, **keywords)

    comparison = root / "comparison"
    marker_path = comparison / "common_step_stage.json"
    assert not marker_path.exists()
    cache_bytes = {
        arm: (comparison / f"common_step_{arm}.cache.json").read_bytes()
        for arm in evaluator.ARMS
    }
    assert calls == list(evaluator.ARMS)
    assert json.loads((root / "manifest.json").read_text(encoding="utf-8")) == (
        original_manifest
    )

    marker = evaluator.evaluate_common_step_artifacts(*arguments, **keywords)
    assert calls == list(evaluator.ARMS)
    assert {
        arm: (comparison / f"common_step_{arm}.cache.json").read_bytes()
        for arm in evaluator.ARMS
    } == cache_bytes
    assert marker_path.is_file()
    assert marker == evaluator.validate_common_step_stage(
        root, archive_timestep_reader=lambda _path: 10_000
    )


def test_common_cache_and_marker_are_bound_to_each_arm_run_config(
    tmp_path, monkeypatch
):
    from comparison import checkpoint_evaluator as evaluator
    from comparison.artifact_manifest import canonical_json_sha256

    root, refs, _protected, _manifest = _common_artifact_fixture(tmp_path, evaluator)
    monkeypatch.setattr(
        evaluator,
        "readable_checkpoint_inventory",
        lambda directory, *_a, **_k: {10_000: refs[Path(directory).name].path},
    )
    monkeypatch.setattr(evaluator, "_archive_timestep", lambda *_a, **_k: 10_000)
    calls = []

    def evaluate(_path, config, _scenarios, label, arm, _loader):
        calls.append((arm, dict(config)))
        return _evaluation_rows(
            arm,
            checkpoint=label,
            timestep=10_000,
            digest=refs[arm].sha256,
        )

    monkeypatch.setattr(evaluator, "evaluate_checkpoint", evaluate)
    scenarios = [{"seed": seed} for seed in range(1000, 1020)]
    configs = {
        "raw_direct": {"extractor": "raw", "scale": 1},
        "candidate_cnn": {"extractor": "cnn", "scale": 2},
    }
    keywords = {
        "config_sha256": "c" * 64,
        "scenario_sha256": "d" * 64,
        "model_loader": "loader",
    }

    marker = evaluator.evaluate_common_step_artifacts(
        root, scenarios, configs, **keywords
    )
    expected_hashes = {
        arm: canonical_json_sha256(config) for arm, config in configs.items()
    }
    assert marker["run_config_sha256"] == expected_hashes
    for arm in evaluator.ARMS:
        cache = json.loads(
            (
                root
                / "comparison"
                / f"common_step_{arm}.cache.json"
            ).read_text(encoding="utf-8")
        )
        assert cache["run_config_sha256"] == expected_hashes[arm]

    changed_configs = {
        **configs,
        "raw_direct": {"extractor": "raw", "scale": 3},
    }
    changed_hashes = {
        arm: canonical_json_sha256(config)
        for arm, config in changed_configs.items()
    }
    with pytest.raises(ValueError, match="run config hash mismatch"):
        evaluator.validate_common_step_stage(
            root,
            expected_run_config_sha256=changed_hashes,
            archive_timestep_reader=lambda _path: 10_000,
        )

    calls.clear()
    marker = evaluator.evaluate_common_step_artifacts(
        root, scenarios, changed_configs, **keywords
    )
    assert calls == [("raw_direct", changed_configs["raw_direct"])]
    assert marker["run_config_sha256"] == changed_hashes


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


def test_inventory_rejects_linked_checkpoint_candidate_before_model_load(
    tmp_path,
):
    from comparison import checkpoint_evaluator as evaluator

    arm_root = tmp_path / "raw_direct"
    checkpoint_root = arm_root / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    real = checkpoint_root / "real.txt"
    real.write_bytes(b"checkpoint")
    linked = checkpoint_root / "linked.sb3"
    try:
        linked.symlink_to(real)
    except OSError:
        pytest.skip("file symlink creation is unavailable on this host")
    loads = []

    def loader(*args, **kwargs):
        loads.append((args, kwargs))
        return SimpleNamespace(num_timesteps=10_000)

    assert evaluator.readable_checkpoint_inventory(
        arm_root, model_loader=loader
    ) == {}
    assert loads == []


def test_inventory_rejects_checkpoint_directory_junction_before_model_load(
    tmp_path,
):
    from comparison import checkpoint_evaluator as evaluator

    if platform.system() != "Windows" or not hasattr(type(tmp_path), "is_junction"):
        pytest.skip("directory junctions are unavailable on this host")
    arm_root = tmp_path / "raw_direct"
    arm_root.mkdir()
    target = tmp_path / "real_checkpoints"
    target.mkdir()
    (target / "model.sb3").write_bytes(b"checkpoint")
    checkpoint_root = arm_root / "checkpoints"
    created = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(checkpoint_root), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if created.returncode != 0:
        pytest.skip("directory junction creation is unavailable on this host")
    loads = []
    try:
        assert checkpoint_root.is_junction()
        assert evaluator.readable_checkpoint_inventory(
            arm_root,
            model_loader=lambda *_a, **_k: loads.append(1),
        ) == {}
        assert loads == []
    finally:
        checkpoint_root.rmdir()


def test_atomic_publication_failure_preserves_manifest_bytes(tmp_path, monkeypatch):
    from comparison import checkpoint_evaluator as evaluator
    path = tmp_path / "manifest.json"; original = b'{"sentinel":true}'; path.write_bytes(original)
    monkeypatch.setattr(evaluator, "atomic_write_json", lambda *_: (_ for _ in ()).throw(OSError("replace failed")))
    with pytest.raises(OSError, match="replace failed"):
        evaluator.update_checkpoint_manifest(path, "raw_direct", {"final": evaluator.CheckpointRef(Path("raw/final.sb3"), "final", 1, "a" * 64)})
    assert path.read_bytes() == original
