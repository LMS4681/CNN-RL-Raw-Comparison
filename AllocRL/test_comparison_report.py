"""Contracts for the preliminary raw-direct versus CNN report."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from comparison.artifact_manifest import (
    REQUIRED_ENVIRONMENT_KEYS,
    canonical_json_sha256,
)
from comparison.checkpoint_evaluator import EVALUATION_COLUMNS


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _rows(
    arm: str,
    scores: dict[int, tuple[float, float, float, float]],
    checkpoint: str = "best_model",
    checkpoint_sha256: str | None = None,
) -> list[dict]:
    rows = []
    for seed in range(1000, 1020):
        score, dropout, delay, delayed = scores.get(seed, (0.1, 0.2, 5.0, 4.0))
        rows.append(dict(zip(EVALUATION_COLUMNS, (
            "holdout_fixed20", arm, seed, score, score, dropout, delay, delayed,
            0.5, arm, checkpoint, 50_000,
            checkpoint_sha256 or (("a" if arm == "raw_direct" else "b") * 64),
            "selection" if seed < 1005 else "primary_test",
        ))))
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EVALUATION_COLUMNS)
        writer.writeheader(); writer.writerows(rows)


def _write_evaluation_marker(root: Path, arm: str) -> None:
    arm_root = root / arm
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    runtime = json.loads(
        (arm_root / "runtime_metrics.json").read_text(encoding="utf-8")
    )
    _json(
        arm_root / "evaluation_stage.json",
        {
            "schema_version": 1,
            "arm": arm,
            "config_sha256": "b" * 64,
            "scenario_sha256": "c" * 64,
            "checkpoints": {
                key: manifest["checkpoints"][arm][key]
                for key in ("selected", "final")
            },
            "artifacts": {
                name: hashlib.sha256((arm_root / name).read_bytes()).hexdigest()
                for name in (
                    "evaluation_scenarios.csv",
                    "evaluation_primary_test.csv",
                    "training_completion.json",
                    "runtime_metrics.json",
                )
            },
            "evaluation_seed_count": 20,
            "primary_test_seed_count": 15,
            "selection_outcome": runtime["selection_outcome"],
            "fallback_reason": runtime["fallback_reason"],
        },
    )


def _set_fallback_final(root: Path, arm: str, reason: str) -> None:
    arm_root = root / arm
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final = manifest["checkpoints"][arm]["final"]
    manifest["checkpoints"][arm]["selected"] = {
        **final,
        "label": "fallback_final",
    }
    _json(manifest_path, manifest)

    runtime_path = arm_root / "runtime_metrics.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime.update(
        {
            "selected_checkpoint_timestep": final["timestep"],
            "selection_count": 0,
            "selection_tuple": None,
            "selection_outcome": "fallback_final",
            "fallback_reason": reason,
            "checkpoint_identity": {
                "filename": Path(final["path"]).name,
                "sha256": final["sha256"],
            },
        }
    )
    _json(runtime_path, runtime)
    for name in ("evaluation_scenarios.csv", "evaluation_primary_test.csv"):
        path = arm_root / name
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row["checkpoint"] = "fallback_final"
            row["checkpoint_timestep"] = str(final["timestep"])
            row["checkpoint_sha256"] = final["sha256"]
        _write_csv(path, rows)

    _refresh_evaluation_contract(root, arm)


def _refresh_evaluation_contract(root: Path, arm: str) -> None:
    arm_root = root / arm
    runtime_path = arm_root / "runtime_metrics.json"
    receipt_path = arm_root / "training_completion.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifact_sha256"]["runtime_metrics.json"] = hashlib.sha256(
        runtime_path.read_bytes()
    ).hexdigest()
    _json(receipt_path, receipt)
    _write_evaluation_marker(root, arm)


def write_complete_fixture(root: Path, *, raw_primary: float = 0.4, cnn_primary: float = 0.5) -> None:
    selected_sha: dict[str, str] = {}
    final_sha: dict[str, str] = {}
    common_sha: dict[str, str] = {}
    run_config_sha: dict[str, str] = {}
    for arm in ("raw_direct", "candidate_cnn"):
        arm_root = root / arm
        arm_root.mkdir(parents=True, exist_ok=True)
        selected_path = arm_root / "best_model.sb3"
        final_path = arm_root / "checkpoints" / "model_50000_g1.sb3"
        common_path = arm_root / "checkpoints" / "common_50000.sb3"
        selected_path.write_bytes(f"{arm}:selected".encode("utf-8"))
        final_path.parent.mkdir(exist_ok=True)
        final_path.write_bytes(f"{arm}:final".encode("utf-8"))
        common_path.write_bytes(f"{arm}:common".encode("utf-8"))
        run_config = {"arm": arm, "fixture": True}
        _json(arm_root / "run_config.json", run_config)
        selected_sha[arm] = hashlib.sha256(selected_path.read_bytes()).hexdigest()
        final_sha[arm] = hashlib.sha256(final_path.read_bytes()).hexdigest()
        common_sha[arm] = hashlib.sha256(common_path.read_bytes()).hexdigest()
        run_config_sha[arm] = canonical_json_sha256(run_config)
    raw = {seed: (raw_primary, 0.10, 4.0, 3.0) for seed in range(1005, 1020)}
    cnn = {seed: (cnn_primary, 0.04, 2.5, 1.0) for seed in range(1005, 1020)}
    raw.update({seed: (0.99, 0.01, 1.0, 0.0) for seed in range(1000, 1005)})
    cnn.update({seed: (0.98, 0.02, 1.2, 0.1) for seed in range(1000, 1005)})
    _write_csv(root / "raw_direct" / "evaluation_scenarios.csv", _rows("raw_direct", raw, checkpoint_sha256=selected_sha["raw_direct"]))
    _write_csv(root / "raw_direct" / "evaluation_primary_test.csv", _rows("raw_direct", raw, checkpoint_sha256=selected_sha["raw_direct"])[5:])
    _write_csv(root / "candidate_cnn" / "evaluation_scenarios.csv", _rows("candidate_cnn", cnn, checkpoint_sha256=selected_sha["candidate_cnn"]))
    _write_csv(root / "candidate_cnn" / "evaluation_primary_test.csv", _rows("candidate_cnn", cnn, checkpoint_sha256=selected_sha["candidate_cnn"])[5:])
    common_rows = {
        "raw_direct": _rows(
            "raw_direct", raw, "common_step", common_sha["raw_direct"]
        ),
        "candidate_cnn": _rows(
            "candidate_cnn", cnn, "common_step", common_sha["candidate_cnn"]
        ),
    }
    common = common_rows["raw_direct"] + common_rows["candidate_cnn"]
    _write_csv(root / "comparison" / "common_step_evaluation.csv", common)
    for arm, total, feature in (("raw_direct", 100, 0), ("candidate_cnn", 200, 80)):
        _json(root / arm / "runtime_metrics.json", {
            "schema_version": 2,
            "target_training_seconds": 10800.0, "recorded_training_seconds": 10800.0,
            "run_wall_span_seconds": 10900.0, "overrun_seconds": 0.0,
            "restart_count": 1, "max_unrecorded_seconds": 300.0, "start_timestep": 0,
            "start_timestep_source": "run_origin.initial_timestep",
            "end_timestep": 50000, "steps_per_second": 50000 / 10800,
            "parameter_counts": {"total": total, "feature_extractor": feature, "policy": 60, "value": total-feature-60},
            "peak_cuda_memory_bytes": 1234, "peak_cuda_memory_scope": "training_process",
            "evaluation_seconds": 12.0,
            "metrics_recorded_at_utc": "2026-07-21T03:01:40+00:00",
            "finalization_mode": "in_process",
            "selected_checkpoint_timestep": 50000, "selection_count": 5,
            "selection_tuple": [0.9, -0.1, -4.0],
            "selection_outcome": "best_model", "fallback_reason": None,
            "checkpoint_identity": {"filename": "best_model.sb3", "sha256": selected_sha[arm]},
        })
        _json(root / arm / "run_origin.json", {
            "schema_version": 1,
            "config_sha256": "b" * 64,
            "initial_timestep": 0,
            "source": "observed_before_first_learn",
            "created_at_utc": "2026-07-21T00:00:00+00:00",
        })
        _json(root / arm / "run_state.json", {
            "schema_version": 1,
            "target_training_seconds": 10800.0,
            "completed_training_seconds": 10800.0,
            "last_checkpoint_timestep": 50000,
            "last_regular_checkpoint_timestep": 50000,
            "last_checkpoint_file": "model_50000_g1.sb3",
            "last_checkpoint_sha256": final_sha[arm],
            "config_sha256": "b" * 64,
            "generation": 1,
            "restart_count": 1,
            "max_unrecorded_seconds": 300.0,
            "status": "complete",
            "started_at_utc": "2026-07-21T00:00:00+00:00",
            "updated_at_utc": "2026-07-21T03:00:00+00:00",
            "completed_at_utc": "2026-07-21T03:00:00+00:00",
        })
    environment = {key: None for key in REQUIRED_ENVIRONMENT_KEYS}
    environment.update({"captured_at_utc": "2026-07-21T00:00:00+00:00", "command": ["python", "train.py"], "python_version": "3.12", "platform": "Linux", "comparison_git_sha": "d" * 40, "comparison_git_dirty": False, "baseline_sha256": "a" * 40, "config_sha256": "b" * 64, "scenario_sha256": "c" * 64, "split_sha256": "d" * 64, "lock_sha256": "e" * 64, "vm_boot_id": "boot", "torch_version": "2.0", "cuda_version": "12", "cudnn_version": 1, "resolved_device": "cuda:0", "gpu_name": "Test GPU", "gpu_uuid": "GPU-test", "gpu_total_memory_bytes": 99, "cpu_count": 2, "process_id": 1, "pip_freeze": ["pytest==1"]})
    _json(root / "environment.json", environment)
    checkpoints = {}
    for arm in ("raw_direct", "candidate_cnn"):
        checkpoints[arm] = {
            "selected": {"path": f"{arm}/best_model.sb3", "label": "best_model", "sha256": selected_sha[arm], "timestep": 50000},
            "final": {"path": f"{arm}/checkpoints/model_50000_g1.sb3", "label": "final", "sha256": final_sha[arm], "timestep": 50000},
            "common": {"path": f"{arm}/checkpoints/common_50000.sb3", "label": "common_step", "sha256": common_sha[arm], "timestep": 50000},
        }
    _json(root / "manifest.json", {
        "schema_version": 1,
        "config_sha256": "b" * 64,
        "scenario_sha256": "c" * 64,
        "checkpoints": checkpoints,
    })
    from comparison.training_completion import ARTIFACT_KEYS
    for arm in ("raw_direct", "candidate_cnn"):
        artifacts = {name: "0" * 64 for name in ARTIFACT_KEYS}
        artifacts["runtime_metrics.json"] = hashlib.sha256(
            (root / arm / "runtime_metrics.json").read_bytes()
        ).hexdigest()
        artifacts["best_model.sb3"] = selected_sha[arm]
        _json(
            root / arm / "training_completion.json",
            {
                "schema_version": 1,
                "config_sha256": "b" * 64,
                "generation": 1,
                "final_timestep": 50000,
                "checkpoint_file": "model_50000_g1.sb3",
                "checkpoint_sha256": final_sha[arm],
                "recorded_training_seconds": 10800.0,
                "finalization_mode": "in_process",
                "finalized_at_utc": "2026-07-21T03:01:40+00:00",
                "artifact_sha256": artifacts,
            },
        )
    for arm in ("raw_direct", "candidate_cnn"):
        _write_evaluation_marker(root, arm)
    common_artifacts = {}
    for arm in ("raw_direct", "candidate_cnn"):
        name = f"common_step_{arm}.cache.json"
        _json(root / "comparison" / name, {
            "schema_version": 1,
            "arm": arm,
            "config_sha256": "b" * 64,
            "run_config_sha256": run_config_sha[arm],
            "scenario_sha256": "c" * 64,
            "checkpoint": checkpoints[arm]["common"],
            "rows": common_rows[arm],
        })
        common_artifacts[name] = hashlib.sha256(
            (root / "comparison" / name).read_bytes()
        ).hexdigest()
    common_artifacts["common_step_evaluation.csv"] = hashlib.sha256(
        (root / "comparison" / "common_step_evaluation.csv").read_bytes()
    ).hexdigest()
    _json(root / "comparison" / "common_step_stage.json", {
        "schema_version": 1,
        "config_sha256": "b" * 64,
        "run_config_sha256": run_config_sha,
        "scenario_sha256": "c" * 64,
        "common_timestep": 50000,
        "checkpoints": {
            arm: checkpoints[arm]["common"]
            for arm in ("raw_direct", "candidate_cnn")
        },
        "artifacts": common_artifacts,
        "evaluation_seed_count_per_arm": 20,
    })
    for arm in ("raw_direct", "candidate_cnn"):
        arm_root = root / arm
        (arm_root / "progress_timing.csv").write_text("generation,timestep,recorded_training_seconds,updated_at_utc,status,checkpoint_file\n1,50000,10800,2026-07-21T03:00:00+00:00,complete,model_50000_g1.sb3\n", encoding="utf-8")
        (arm_root / "training_log.csv").write_text("episode,timestep,resolved_reward,terminal_residual,terminal_score,episode_reward,delayed_count,dropout_count,total_delay_days,success_rate\n1,10000,0,0,0.2,0,3,1,4,0.5\n", encoding="utf-8")


def test_summary_uses_only_primary_test_for_primary_means(tmp_path):
    from comparison.report_builder import build_comparison_summary
    write_complete_fixture(tmp_path)
    summary = build_comparison_summary(tmp_path)
    assert summary["raw_direct"]["primary_test"]["mean_terminal_score"] == 0.4
    assert summary["raw_direct"]["selection"]["mean_terminal_score"] == 0.99


def test_paired_differences_match_scenario_seeds(tmp_path):
    from comparison.report_builder import build_paired_differences
    write_complete_fixture(tmp_path, raw_primary=0.2, cnn_primary=0.5)
    rows = build_paired_differences(tmp_path)
    assert rows[0] == {"seed": 1005, "terminal_score_delta_cnn_minus_raw": pytest.approx(0.3), "dropout_rate_delta_cnn_minus_raw": pytest.approx(-0.06), "mean_delay_days_delta_cnn_minus_raw": pytest.approx(-1.5), "delayed_count_delta_cnn_minus_raw": pytest.approx(-2.0)}
    assert [row["seed"] for row in rows] == list(range(1005, 1020))


def test_rejects_malformed_or_unpaired_evaluation_schema(tmp_path):
    from comparison.report_builder import build_paired_differences
    write_complete_fixture(tmp_path)
    path = tmp_path / "candidate_cnn" / "evaluation_primary_test.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))[:-1]
    _write_csv(path, rows)
    with pytest.raises(ValueError, match="primary"):
        build_paired_differences(tmp_path)


def test_complete_report_is_deterministic_and_creates_nonempty_closed_plots(tmp_path):
    import matplotlib.pyplot as plt
    from comparison.report_builder import write_complete_report
    write_complete_fixture(tmp_path)
    report = write_complete_report(tmp_path)
    first_summary = (tmp_path / "comparison" / "summary.json").read_bytes()
    first_csv = (tmp_path / "comparison" / "scenario_paired_differences.csv").read_bytes()
    write_complete_report(tmp_path)
    assert report.name == "preliminary_comparison_ko.md"
    assert first_summary == (tmp_path / "comparison" / "summary.json").read_bytes()
    assert first_csv == (tmp_path / "comparison" / "scenario_paired_differences.csv").read_bytes()
    assert all((tmp_path / "comparison" / name).stat().st_size > 0 for name in ("learning_curves.png", "holdout_comparison.png"))
    assert plt.get_fignums() == []


def test_missing_arm_creates_partial_not_complete_report(tmp_path):
    from comparison.report_builder import write_partial_report
    write_complete_fixture(tmp_path)
    raw_artifacts = {
        name: (tmp_path / "raw_direct" / name).read_bytes()
        for name in (
            "evaluation_scenarios.csv",
            "evaluation_primary_test.csv",
            "evaluation_stage.json",
        )
    }
    (tmp_path / "candidate_cnn" / "evaluation_stage.json").unlink()
    path = write_partial_report(tmp_path, failure="candidate runtime stopped")
    assert path.name == "PARTIAL_REPORT.md"
    assert not (tmp_path / "COMPLETE.json").exists()
    assert "후보 CNN 결과가 없어 우열을 결론내리지 않음" in path.read_text("utf-8")


    assert {
        name: (tmp_path / "raw_direct" / name).read_bytes()
        for name in raw_artifacts
    } == raw_artifacts


def test_partial_report_rejects_orphan_evaluation_csvs_without_stage_marker(
    tmp_path,
):
    from comparison.report_builder import write_partial_report

    write_complete_fixture(tmp_path)
    (tmp_path / "raw_direct" / "evaluation_stage.json").unlink()
    (tmp_path / "candidate_cnn" / "evaluation_stage.json").unlink()

    text = write_partial_report(tmp_path, "candidate training failed").read_text(
        encoding="utf-8"
    )

    assert "raw-direct runtime=없음" in text


@pytest.mark.parametrize(
    ("relative", "mutation"),
    [
        ("raw_direct/evaluation_stage.json", "missing"),
        ("candidate_cnn/evaluation_stage.json", "malformed"),
        ("raw_direct/evaluation_stage.json", "provenance"),
        ("comparison/common_step_stage.json", "missing"),
        ("comparison/common_step_stage.json", "malformed"),
        ("comparison/common_step_stage.json", "provenance"),
    ],
)
def test_complete_report_requires_valid_stage_markers(
    tmp_path: Path, relative: str, mutation: str
):
    from comparison.report_builder import write_complete_report

    write_complete_fixture(tmp_path)
    marker = tmp_path / relative
    if mutation == "missing":
        marker.unlink()
    elif mutation == "malformed":
        marker.write_text("not-json", encoding="utf-8")
    else:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        payload["config_sha256"] = "f" * 64
        _json(marker, payload)

    with pytest.raises(ValueError):
        write_complete_report(tmp_path)
    assert not (tmp_path / "comparison" / "preliminary_comparison_ko.md").exists()


@pytest.mark.parametrize(
    "relative",
    ["raw_direct/evaluation_stage.json", "comparison/common_step_stage.json"],
)
def test_complete_report_rejects_symlinked_stage_markers(
    tmp_path: Path, relative: str
):
    from comparison.report_builder import write_complete_report

    write_complete_fixture(tmp_path)
    marker = tmp_path / relative
    outside = tmp_path.parent / f"{tmp_path.name}-{marker.name}"
    outside.write_bytes(marker.read_bytes())
    marker.unlink()
    try:
        marker.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows host")

    with pytest.raises(ValueError):
        write_complete_report(tmp_path)
    assert not (tmp_path / "comparison" / "preliminary_comparison_ko.md").exists()


def test_korean_report_is_utf8_and_has_no_replacement_character(tmp_path):
    from comparison.report_builder import write_complete_report
    write_complete_fixture(tmp_path)
    text = write_complete_report(tmp_path).read_text(encoding="utf-8")
    for phrase in ("예비 결과", "seed 0", "통계적 유의성", "자료 없음"):
        assert phrase in text
    assert "\ufffd" not in text
    assert "wall span" in text
    assert "end-to-end" not in text
    assert "scope=training_process" in text


@pytest.mark.parametrize("reason", ["selection_not_run", "best_model_missing"])
def test_fallback_final_report_prints_exact_canonical_reason(
    tmp_path: Path, reason: str
):
    from comparison.report_builder import write_complete_report

    write_complete_fixture(tmp_path)
    _set_fallback_final(tmp_path, "candidate_cnn", reason)

    text = write_complete_report(tmp_path).read_text(encoding="utf-8")

    assert f"fallback 사유: {reason}" in text
    assert "fallback 사유: 자료 없음" not in text


def test_missing_runtime_value_is_json_null_not_a_guessed_zero(tmp_path):
    from comparison.report_builder import build_comparison_summary
    write_complete_fixture(tmp_path)
    path = tmp_path / "raw_direct" / "runtime_metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8")); del payload["peak_cuda_memory_bytes"]
    _json(path, payload)
    with pytest.raises(ValueError, match="runtime"):
        build_comparison_summary(tmp_path)


def test_report_summary_rejects_duplicate_runtime_metrics_field(tmp_path):
    from comparison.report_builder import build_comparison_summary

    write_complete_fixture(tmp_path)
    path = tmp_path / "raw_direct" / "runtime_metrics.json"; raw = path.read_text(encoding="utf-8").rstrip()
    path.write_text(raw[:-1] + ',"restart_count":999}', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        build_comparison_summary(tmp_path)


def test_rejects_coercive_runtime_values_and_reconciles_selected_provenance(tmp_path):
    from comparison.report_builder import build_comparison_summary
    write_complete_fixture(tmp_path)
    path = tmp_path / "raw_direct" / "runtime_metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8")); payload["restart_count"] = "1"
    _json(path, payload)
    with pytest.raises(ValueError, match="restart_count"):
        build_comparison_summary(tmp_path)
    payload["restart_count"] = 1; payload["checkpoint_identity"]["sha256"] = "0" * 64
    _json(path, payload)
    with pytest.raises(ValueError, match="selected"):
        build_comparison_summary(tmp_path)


def test_rejects_incomplete_environment_and_common_sha_mismatch(tmp_path):
    from comparison.report_builder import build_comparison_summary
    write_complete_fixture(tmp_path)
    _json(tmp_path / "environment.json", {})
    with pytest.raises(ValueError, match="environment"):
        build_comparison_summary(tmp_path)
    write_complete_fixture(tmp_path)
    common = tmp_path / "comparison" / "common_step_evaluation.csv"
    with common.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[-1]["checkpoint_sha256"] = "0" * 64; _write_csv(common, rows)
    with pytest.raises(ValueError, match="common"):
        build_comparison_summary(tmp_path)


def test_dynamic_timings_fallback_and_partial_failure_safety(tmp_path):
    from comparison.report_builder import write_complete_report, write_partial_report
    write_complete_fixture(tmp_path)
    for arm in ("raw_direct", "candidate_cnn"):
        path = tmp_path / arm / "runtime_metrics.json"; payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update({"target_training_seconds": 15.0, "recorded_training_seconds": 16.0, "run_wall_span_seconds": 18.0, "metrics_recorded_at_utc": "2026-07-21T00:00:18+00:00", "steps_per_second": 50000 / 16, "overrun_seconds": 1.0})
        _json(path, payload)
        state_path = tmp_path / arm / "run_state.json"; state = json.loads(state_path.read_text("utf-8")); state.update({"target_training_seconds": 15.0, "completed_training_seconds": 16.0}); _json(state_path, state)
        _set_fallback_final(tmp_path, arm, "selection_not_run")
    text = write_complete_report(tmp_path).read_text(encoding="utf-8")
    assert "15.0" in text and "fallback 사유: selection_not_run" in text and "10,800" not in text
    with pytest.raises(ValueError, match="replacement"):
        write_partial_report(tmp_path, "bad\ufffdfailure")


def test_optional_curve_logs_are_strict_and_loss_is_rendered(tmp_path):
    from comparison.report_builder import write_complete_report
    write_complete_fixture(tmp_path)
    (tmp_path / "raw_direct" / "loss_log.csv").write_text("bad\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="loss_log"):
        write_complete_report(tmp_path)
    (tmp_path / "raw_direct" / "loss_log.csv").write_text("timestep,policy_gradient_loss,value_loss,entropy_loss,approx_kl,clip_fraction,loss,explained_variance,cnn_gradient_norm,cnn_weight_change,workspace_feature_variance,candidate_channel_sensitivity\n10000,,,,,,0.1,,,,,\n", encoding="utf-8")
    write_complete_report(tmp_path)
    assert (tmp_path / "comparison" / "learning_curves.png").stat().st_size > 0


def test_report_uses_shared_curve_validator_without_repair(tmp_path, monkeypatch):
    from comparison import report_builder, training_log_validation

    write_complete_fixture(tmp_path)
    (tmp_path / "raw_direct" / "loss_log.csv").write_text(
        "timestep,policy_gradient_loss,value_loss,entropy_loss,approx_kl,"
        "clip_fraction,loss,explained_variance,cnn_gradient_norm,"
        "cnn_weight_change,workspace_feature_variance,"
        "candidate_channel_sensitivity\n10000,,,,,,0.1,,,,,\n",
        encoding="utf-8",
    )
    calls = []
    original = training_log_validation.read_curve_log

    def spy(path, kind, *, repair_trailing_partial=False):
        calls.append((Path(path).name, kind, repair_trailing_partial))
        return original(
            path,
            kind,
            repair_trailing_partial=repair_trailing_partial,
        )

    monkeypatch.setattr(training_log_validation, "read_curve_log", spy)
    report_builder.write_complete_report(tmp_path)

    assert ("loss_log.csv", "loss_log", False) in calls
    assert calls.count(("training_log.csv", "training_log", False)) == 2


def test_plots_close_figures_when_save_fails(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from comparison import report_builder
    write_complete_fixture(tmp_path)
    summary = report_builder.build_comparison_summary(tmp_path); pairs = report_builder.build_paired_differences(tmp_path)
    before = set(plt.get_fignums())
    monkeypatch.setattr(report_builder.Figure, "savefig", lambda *_a, **_k: (_ for _ in ()).throw(OSError("save")))
    with pytest.raises(OSError): report_builder._learning_plot(tmp_path, tmp_path / "x.png")
    with pytest.raises(OSError): report_builder._holdout_plot(summary, pairs, tmp_path / "x.png")
    assert set(plt.get_fignums()) == before


@pytest.mark.parametrize("present, phrase", [(("raw_direct",), "후보 CNN 결과가 없어 우열을 결론내리지 않음"), (("candidate_cnn",), "raw-direct 결과가 없어 비교를 결론내리지 않음"), (("raw_direct", "candidate_cnn"), "무결성 또는 보고 단계가 불완전"), ((), "두 arm 모두 없어서")])
def test_partial_report_uses_valid_artifact_groups_and_stage_journal(tmp_path, present, phrase):
    from comparison.report_builder import write_partial_report
    write_complete_fixture(tmp_path)
    for arm in {"raw_direct", "candidate_cnn"} - set(present):
        (tmp_path / arm / "evaluation_stage.json").unlink()
    _json(tmp_path / "stage_journal.json", {"preflight": {"status": "complete", "input_sha256": "a" * 64, "output_sha256": "b" * 64, "started_at_utc": "2026-01-01T00:00:00+00:00", "completed_at_utc": "2026-01-01T00:01:00+00:00", "error": None}, "train_candidate_cnn": {"status": "failed", "input_sha256": "a" * 64, "output_sha256": None, "started_at_utc": "2026-01-01T00:00:00+00:00", "completed_at_utc": "2026-01-01T00:01:00+00:00", "error": "bad"}})
    complete = tmp_path / "COMPLETE.json"; complete.write_text("sentinel", encoding="utf-8")
    text = write_partial_report(tmp_path, "<b>bad</b>\n[link]").read_text(encoding="utf-8")
    assert phrase in text and "completed: preflight" in text and "failed: train_candidate_cnn" in text
    assert "&lt;b&gt;bad&lt;/b&gt;" in text and complete.read_text(encoding="utf-8") == "sentinel"


def test_partial_report_handles_invalid_journal_and_integrity_matrix(tmp_path):
    from comparison.report_builder import build_comparison_summary, write_partial_report
    write_complete_fixture(tmp_path)
    _json(tmp_path / "stage_journal.json", {"bad": {"status": "unknown"}})
    assert "invalid metadata" in write_partial_report(tmp_path, "failed").read_text(encoding="utf-8")
    path = tmp_path / "raw_direct" / "runtime_metrics.json"; payload = json.loads(path.read_text(encoding="utf-8"))
    for value in ("NaN", True, 1.5):
        payload["parameter_counts"]["total"] = value; _json(path, payload)
        with pytest.raises(ValueError): build_comparison_summary(tmp_path)


@pytest.mark.parametrize("mutate", [
    lambda e: e.update(input_sha256="bad"), lambda e: e.update(started_at_utc="nope"),
    lambda e: e.update(error=[]), lambda e: e.update(status="unknown"),
    lambda e: e.pop("error"), lambda e: e.update(extra=True),
])
def test_stage_journal_entry_schema_is_strict(tmp_path, mutate):
    from comparison.report_builder import write_partial_report
    entry = {"status":"failed", "input_sha256":"a" * 64, "output_sha256":None, "started_at_utc":"2026-01-01T00:00:00+00:00", "completed_at_utc":"2026-01-01T00:01:00+00:00", "error":"x"}
    mutate(entry); _json(tmp_path / "stage_journal.json", {"preflight": entry})
    assert "invalid metadata" in write_partial_report(tmp_path, "failed").read_text(encoding="utf-8")


def test_valid_stage_journal_and_no_replacement_source():
    from comparison.report_builder import _stage_journal
    from pathlib import Path
    assert _stage_journal({"preflight": {"status":"complete", "input_sha256":"a" * 64, "output_sha256":"b" * 64, "started_at_utc":"2026-01-01T00:00:00+00:00", "completed_at_utc":"2026-01-01T00:01:00+00:00", "error":None}}).startswith("completed: preflight")
    root = Path(__file__).resolve().parents[1]
    assert all("\ufffd" not in path.read_text(encoding="utf-8") for path in (root / "AllocRL" / "comparison").glob("*.py"))


@pytest.mark.parametrize("field,value", [("selection_tuple", [float("nan"), 0.0, 0.0]), ("parameter_counts.policy", True), ("parameter_counts.policy", 1.5), ("selected_checkpoint_timestep", 99)])
def test_runtime_integrity_matrix(tmp_path, field, value):
    from comparison.report_builder import build_comparison_summary
    write_complete_fixture(tmp_path); path = tmp_path / "raw_direct" / "runtime_metrics.json"; payload = json.loads(path.read_text("utf-8"))
    if "." in field: outer, inner = field.split("."); payload[outer][inner] = value
    else: payload[field] = value
    _json(path, payload)
    with pytest.raises(ValueError): build_comparison_summary(tmp_path)


@pytest.mark.parametrize("field,value", [("checkpoint", "fallback_final"), ("checkpoint_timestep", "49999"), ("checkpoint_sha256", "0" * 64), ("mean_reward", "9.9")])
def test_primary_all_provenance_and_metric_matrix(tmp_path, field, value):
    from comparison.report_builder import build_comparison_summary
    write_complete_fixture(tmp_path); path = tmp_path / "raw_direct" / "evaluation_primary_test.csv"
    with path.open(encoding="utf-8", newline="") as stream: rows = list(csv.DictReader(stream))
    rows[0][field] = value; _write_csv(path, rows)
    with pytest.raises(ValueError): build_comparison_summary(tmp_path)


@pytest.mark.parametrize("arm,kind,field,value", [(arm, kind, field, value) for arm in ("raw_direct", "candidate_cnn") for kind in ("selected", "final", "common") for field, value in (("label", "wrong"), ("timestep", "x"), ("sha256", "x"), ("path", 1))])
def test_manifest_ref_matrix(tmp_path, arm, kind, field, value):
    from comparison.report_builder import build_comparison_summary
    write_complete_fixture(tmp_path); path = tmp_path / "manifest.json"; payload = json.loads(path.read_text("utf-8")); payload["checkpoints"][arm][kind][field] = value; _json(path, payload)
    with pytest.raises(ValueError): build_comparison_summary(tmp_path)


@pytest.mark.parametrize("key,value", [("baseline_sha256", "x"), ("config_sha256", "x"), ("scenario_sha256", "x"), ("split_sha256", "x"), ("lock_sha256", "x"), ("resolved_device", 3), ("command", "x"), ("pip_freeze", "x")])
def test_environment_integrity_matrix(tmp_path, key, value):
    from comparison.report_builder import build_comparison_summary
    write_complete_fixture(tmp_path); path = tmp_path / "environment.json"; payload = json.loads(path.read_text("utf-8")); payload[key] = value; _json(path, payload)
    with pytest.raises(ValueError): build_comparison_summary(tmp_path)


@pytest.mark.parametrize("arm,kind", [(arm, kind) for arm in ("raw_direct", "candidate_cnn") for kind in ("selected", "final", "common")])
def test_manifest_missing_ref_matrix(tmp_path, arm, kind):
    from comparison.report_builder import build_comparison_summary
    write_complete_fixture(tmp_path); path = tmp_path / "manifest.json"; payload = json.loads(path.read_text("utf-8")); del payload["checkpoints"][arm][kind]; _json(path, payload)
    with pytest.raises(ValueError): build_comparison_summary(tmp_path)


@pytest.mark.parametrize("arm,digest", [("raw_direct", "1" * 64), ("candidate_cnn", "2" * 64)])
def test_common_arm_provenance_must_match_manifest(tmp_path, arm, digest):
    from comparison.report_builder import build_comparison_summary
    write_complete_fixture(tmp_path); path = tmp_path / "comparison" / "common_step_evaluation.csv"
    with path.open(encoding="utf-8", newline="") as stream: rows = list(csv.DictReader(stream))
    for row in rows:
        if row["arm"] == arm: row["checkpoint_sha256"] = digest
    _write_csv(path, rows)
    with pytest.raises(ValueError, match="common checkpoint"):
        build_comparison_summary(tmp_path)


def test_report_renders_all_actual_runtime_and_selection_fields(tmp_path):
    from comparison.report_builder import write_complete_report
    write_complete_fixture(tmp_path)
    for arm, tag, fallback in (("raw_direct", "11", False), ("candidate_cnn", "22", True)):
        manifest_path = tmp_path / "manifest.json"; manifest=json.loads(manifest_path.read_text("utf-8")); digest = manifest["checkpoints"][arm]["selected"]["sha256"]
        path = tmp_path / arm / "runtime_metrics.json"; p = json.loads(path.read_text("utf-8"))
        selected_step = 50000 if fallback else int(tag) * 100 + 1
        p.update({"target_training_seconds": float(tag), "recorded_training_seconds": float(tag)+.1, "run_wall_span_seconds":float(tag)+.2, "metrics_recorded_at_utc":f"2026-07-21T00:00:{float(tag)+.2:04.1f}+00:00", "overrun_seconds":(float(tag)+.1)-float(tag), "restart_count":int(tag), "max_unrecorded_seconds":float(tag)+.4, "start_timestep":int(tag)*100, "end_timestep":selected_step, "steps_per_second":(selected_step-int(tag)*100)/(float(tag)+.1), "evaluation_seconds":float(tag)+.6, "parameter_counts":{"total":int(tag)*10,"feature_extractor":int(tag),"policy":int(tag)*2,"value":int(tag)*7}, "peak_cuda_memory_bytes":int(tag)*1000, "selection_count":5, "selection_tuple":[1.1,2.2,3.3], "selection_outcome":"best_model", "fallback_reason":None, "selected_checkpoint_timestep":selected_step, "checkpoint_identity":{"filename":"best_model.sb3","sha256":digest}}); _json(path,p)
        origin_path=tmp_path/arm/"run_origin.json"; origin=json.loads(origin_path.read_text("utf-8")); origin["initial_timestep"]=int(tag)*100; _json(origin_path,origin)
        state_path=tmp_path/arm/"run_state.json"; state=json.loads(state_path.read_text("utf-8")); state.update({"target_training_seconds":float(tag),"completed_training_seconds":float(tag)+.1,"restart_count":int(tag),"max_unrecorded_seconds":float(tag)+.4,"last_checkpoint_timestep":selected_step}); _json(state_path,state)
        manifest["checkpoints"][arm]["selected"].update({"timestep":selected_step}); _json(manifest_path,manifest)
        for name in ("evaluation_scenarios.csv","evaluation_primary_test.csv"):
            file=tmp_path/arm/name
            with file.open(encoding="utf-8",newline="") as s: rows=list(csv.DictReader(s))
            for row in rows:
                row["checkpoint_timestep"] = str(selected_step)
            _write_csv(file,rows)
        if fallback:
            _set_fallback_final(tmp_path, arm, "selection_not_run")
        else:
            _refresh_evaluation_contract(tmp_path, arm)
    text=write_complete_report(tmp_path).read_text("utf-8")
    for value in ("11.0","11.1","11.2","11.4","11.6","22.0","22.1","22.2","22.4","22.6","1100","1101","2200","11000","22000","selection count 5","selection count 0","[1.1, 2.2, 3.3]","best_model","fallback_final","fallback 사유: selection_not_run"):
        assert value in text
    assert "10,800" not in text


def test_zero_recorded_runtime_allows_null_steps_per_second_and_extra_manifest_metadata(tmp_path):
    from comparison.report_builder import build_comparison_summary, write_complete_report
    write_complete_fixture(tmp_path)
    for arm in ("raw_direct", "candidate_cnn"):
        path=tmp_path/arm/"runtime_metrics.json"; payload=json.loads(path.read_text("utf-8")); payload.update({"target_training_seconds":0.0,"recorded_training_seconds":0.0,"overrun_seconds":0.0,"start_timestep":7,"end_timestep":7,"selected_checkpoint_timestep":7,"steps_per_second":None}); _json(path,payload)
        origin_path=tmp_path/arm/"run_origin.json"; origin=json.loads(origin_path.read_text("utf-8")); origin["initial_timestep"]=7; _json(origin_path,origin)
        state_path=tmp_path/arm/"run_state.json"; state=json.loads(state_path.read_text("utf-8")); state.update({"target_training_seconds":0.0,"completed_training_seconds":0.0,"last_checkpoint_timestep":7}); _json(state_path,state)
        manifest_path=tmp_path/"manifest.json"; manifest_payload=json.loads(manifest_path.read_text("utf-8")); manifest_payload["checkpoints"][arm]["selected"]["timestep"]=7; _json(manifest_path,manifest_payload)
        for name in ("evaluation_scenarios.csv", "evaluation_primary_test.csv"):
            file = tmp_path / arm / name
            with file.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
                for row in rows:
                    row["checkpoint_timestep"] = "7"
                _write_csv(file, rows)
        _refresh_evaluation_contract(tmp_path, arm)
    manifest=tmp_path/"manifest.json"; payload=json.loads(manifest.read_text("utf-8")); payload["provenance"]={"run":"x"}; _json(manifest,payload)
    summary=build_comparison_summary(tmp_path); assert summary["raw_direct"]["runtime_metrics"]["steps_per_second"] is None and summary["manifest"]["provenance"] == {"run":"x"}
    assert "자료 없음" in write_complete_report(tmp_path).read_text("utf-8")


@pytest.mark.parametrize("field,value", [("mean_reward", "NaN"), ("mean_terminal_score", "inf"), ("mean_dropout_rate", "no"), ("mean_delay_days", "NaN"), ("mean_delayed_count", "inf"), ("mean_retained_choice_ratio", "no")])
def test_common_numeric_fields_are_finite(tmp_path, field, value):
    from comparison.report_builder import build_comparison_summary
    write_complete_fixture(tmp_path); path=tmp_path/"comparison"/"common_step_evaluation.csv"
    with path.open(encoding="utf-8",newline="") as s: rows=list(csv.DictReader(s))
    rows[0][field]=value; _write_csv(path,rows)
    with pytest.raises(ValueError): build_comparison_summary(tmp_path)


def test_partial_failure_text_cannot_inject_markdown_or_html(tmp_path):
    from comparison.report_builder import write_partial_report
    failure = "[click](https://example.invalid) ![img](https://example.invalid/x) # heading **bold** ``` <script>alert(1)</script>\nnext"
    text = write_partial_report(tmp_path, failure).read_text("utf-8")
    assert "<pre><code>" in text and "</code></pre>" in text and "&lt;script&gt;alert" in text and "&lt;/script&gt;" in text
    for unsafe in ("[click](https://example.invalid)", "![img](https://example.invalid/x)", "# heading", "```", "<script>"):
        assert unsafe not in text
    assert "click" in text and "heading" in text and "\ufffd" not in text


def test_holdout_plot_closes_figure_when_first_bar_raises(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from comparison import report_builder
    write_complete_fixture(tmp_path)
    summary = report_builder.build_comparison_summary(tmp_path); pairs = report_builder.build_paired_differences(tmp_path)
    before = set(plt.get_fignums())
    monkeypatch.setattr(Axes, "bar", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("bar")))
    with pytest.raises(RuntimeError, match="bar"):
        report_builder._holdout_plot(summary, pairs, tmp_path / "x.png")
    assert set(plt.get_fignums()) == before
