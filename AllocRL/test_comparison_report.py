"""Contracts for the preliminary raw-direct versus CNN report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from comparison.artifact_manifest import REQUIRED_ENVIRONMENT_KEYS
from comparison.checkpoint_evaluator import EVALUATION_COLUMNS


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _rows(arm: str, scores: dict[int, tuple[float, float, float, float]], checkpoint: str = "best_model") -> list[dict]:
    rows = []
    for seed in range(1000, 1020):
        score, dropout, delay, delayed = scores.get(seed, (0.1, 0.2, 5.0, 4.0))
        rows.append(dict(zip(EVALUATION_COLUMNS, (
            "holdout_fixed20", arm, seed, score, score, dropout, delay, delayed,
            0.5, arm, checkpoint, 50_000, ("a" if arm == "raw_direct" else "b") * 64,
            "selection" if seed < 1005 else "primary_test",
        ))))
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EVALUATION_COLUMNS)
        writer.writeheader(); writer.writerows(rows)


def write_complete_fixture(root: Path, *, raw_primary: float = 0.4, cnn_primary: float = 0.5) -> None:
    raw = {seed: (raw_primary, 0.10, 4.0, 3.0) for seed in range(1005, 1020)}
    cnn = {seed: (cnn_primary, 0.04, 2.5, 1.0) for seed in range(1005, 1020)}
    raw.update({seed: (0.99, 0.01, 1.0, 0.0) for seed in range(1000, 1005)})
    cnn.update({seed: (0.98, 0.02, 1.2, 0.1) for seed in range(1000, 1005)})
    _write_csv(root / "raw_direct" / "evaluation_scenarios.csv", _rows("raw_direct", raw))
    _write_csv(root / "raw_direct" / "evaluation_primary_test.csv", _rows("raw_direct", raw)[5:])
    _write_csv(root / "candidate_cnn" / "evaluation_scenarios.csv", _rows("candidate_cnn", cnn))
    _write_csv(root / "candidate_cnn" / "evaluation_primary_test.csv", _rows("candidate_cnn", cnn)[5:])
    common = _rows("raw_direct", raw, "common_step") + _rows("candidate_cnn", cnn, "common_step")
    _write_csv(root / "comparison" / "common_step_evaluation.csv", common)
    for arm, total, feature in (("raw_direct", 100, 0), ("candidate_cnn", 200, 80)):
        _json(root / arm / "runtime_metrics.json", {
            "target_training_seconds": 10800.0, "recorded_training_seconds": 10800.0,
            "end_to_end_training_seconds": 10900.0, "overrun_seconds": 0.0,
            "restart_count": 1, "max_unrecorded_seconds": 300.0, "start_timestep": 0,
            "end_timestep": 50000, "steps_per_second": 4.63,
            "parameter_counts": {"total": total, "feature_extractor": feature, "policy": 60, "value": total-feature-60},
            "peak_cuda_memory_bytes": 1234, "evaluation_seconds": 12.0,
            "selected_checkpoint_timestep": 50000, "selection_count": 5,
            "selection_tuple": [0.9, -0.1, -4.0], "checkpoint_identity": {"filename": "best_model.sb3", "sha256": ("a" if arm == "raw_direct" else "b") * 64},
        })
    environment = {key: None for key in REQUIRED_ENVIRONMENT_KEYS}
    environment.update({"captured_at_utc": "2026-07-21T00:00:00+00:00", "command": ["python", "train.py"], "python_version": "3.12", "platform": "Linux", "comparison_git_sha": "d" * 40, "comparison_git_dirty": False, "baseline_sha256": "a" * 40, "config_sha256": "b" * 64, "scenario_sha256": "c" * 64, "split_sha256": "d" * 64, "lock_sha256": "e" * 64, "vm_boot_id": "boot", "torch_version": "2.0", "cuda_version": "12", "cudnn_version": 1, "resolved_device": "cuda:0", "gpu_name": "Test GPU", "gpu_uuid": "GPU-test", "gpu_total_memory_bytes": 99, "cpu_count": 2, "process_id": 1, "pip_freeze": ["pytest==1"]})
    _json(root / "environment.json", environment)
    checkpoints = {}
    for arm in ("raw_direct", "candidate_cnn"):
        checkpoints[arm] = {
            "selected": {"path": f"{arm}/best_model.sb3", "label": "best_model", "sha256": ("a" if arm == "raw_direct" else "b") * 64, "timestep": 50000},
            "final": {"path": f"{arm}/final.sb3", "label": "final", "sha256": "f" * 64, "timestep": 50000},
            "common": {"path": f"{arm}/common.sb3", "label": "common_step", "sha256": ("a" if arm == "raw_direct" else "b") * 64, "timestep": 50000},
        }
    _json(root / "manifest.json", {"schema_version": 1, "checkpoints": checkpoints})
    for arm in ("raw_direct", "candidate_cnn"):
        arm_root = root / arm
        (arm_root / "progress_timing.csv").write_text("generation,timestep,recorded_training_seconds,updated_at_utc,status,checkpoint_file\n1,50000,10800,now,complete,best_model.sb3\n", encoding="utf-8")
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
    for path in (tmp_path / "candidate_cnn").glob("*"):
        path.unlink()
    path = write_partial_report(tmp_path, failure="candidate runtime stopped")
    assert path.name == "PARTIAL_REPORT.md"
    assert not (tmp_path / "COMPLETE.json").exists()
    assert "후보 CNN 결과가 없어 우열을 결론내리지 않음" in path.read_text("utf-8")


def test_korean_report_is_utf8_and_has_no_replacement_character(tmp_path):
    from comparison.report_builder import write_complete_report
    write_complete_fixture(tmp_path)
    text = write_complete_report(tmp_path).read_text(encoding="utf-8")
    for phrase in ("예비 결과", "seed 0", "통계적 유의성", "자료 없음"):
        assert phrase in text
    assert "\ufffd" not in text


def test_missing_runtime_value_is_json_null_not_a_guessed_zero(tmp_path):
    from comparison.report_builder import build_comparison_summary
    write_complete_fixture(tmp_path)
    path = tmp_path / "raw_direct" / "runtime_metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8")); del payload["peak_cuda_memory_bytes"]
    _json(path, payload)
    with pytest.raises(ValueError, match="runtime"):
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
        payload.update({"target_training_seconds": 17.0, "recorded_training_seconds": 16.0, "end_to_end_training_seconds": 18.0, "overrun_seconds": 1.0, "selection_count": 0, "selection_tuple": None, "checkpoint_identity": {"filename": "fallback.sb3", "sha256": ("a" if arm == "raw_direct" else "b") * 64}})
        _json(path, payload)
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")); manifest["checkpoints"][arm]["selected"].update({"path": f"{arm}/fallback.sb3", "label": "fallback_final"}); _json(tmp_path / "manifest.json", manifest)
        for name in ("evaluation_scenarios.csv", "evaluation_primary_test.csv"):
            file = tmp_path / arm / name
            with file.open(encoding="utf-8", newline="") as stream: rows = list(csv.DictReader(stream))
            for row in rows: row["checkpoint"] = "fallback_final"
            _write_csv(file, rows)
    text = write_complete_report(tmp_path).read_text(encoding="utf-8")
    assert "17.0" in text and "fallback 사유: 자료 없음" in text and "10,800" not in text
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
