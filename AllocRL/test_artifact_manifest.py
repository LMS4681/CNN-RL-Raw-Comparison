import hashlib
import json
import platform

import pytest
import torch.nn as nn
from types import SimpleNamespace

from comparison import artifact_manifest as artifact_manifest_module
from comparison.artifact_manifest import (
    REQUIRED_ENVIRONMENT_KEYS,
    append_environment_segment,
    canonical_json_sha256,
    collect_environment,
    count_trainable_parameters,
    read_run_origin,
    read_runtime_metrics,
    sanitize_requirement_line,
    sha256_file,
    write_manifest,
    write_run_origin,
    write_runtime_metrics,
)
import train as train_module


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.features_extractor = nn.Linear(2, 3)
        self.mlp_extractor = nn.Module()
        self.mlp_extractor.policy_net = nn.Linear(3, 4)
        self.mlp_extractor.value_net = nn.Linear(3, 4)
        self.value_net = self.mlp_extractor.value_net
        self.value_head = nn.Linear(4, 1)


@pytest.fixture
def tiny_policy():
    return TinyPolicy()


def test_parameter_counts_are_split_by_model_component(tiny_policy):
    counts = count_trainable_parameters(tiny_policy)
    assert counts["total"] == sum(
        parameter.numel()
        for parameter in tiny_policy.parameters()
        if parameter.requires_grad
    )
    assert set(counts) == {"total", "feature_extractor", "policy", "value"}
    assert counts["total"] == (
        counts["feature_extractor"] + counts["policy"] + counts["value"]
    )


def test_environment_manifest_contains_required_provenance(monkeypatch):
    monkeypatch.setattr(platform, "platform", lambda: "test-platform")
    manifest = collect_environment(command=["python", "train.py"])
    assert REQUIRED_ENVIRONMENT_KEYS <= manifest.keys()
    assert manifest["command"] == ["python", "train.py"]
    assert manifest["vm_boot_id"]
    assert "gpu_uuid" in manifest


def test_windows_boot_id_is_cached_and_independent_of_process_id(monkeypatch):
    timestamp = "2026-07-21T01:02:03.4567890+00:00"
    calls: list[list[str]] = []
    monkeypatch.setattr(artifact_manifest_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(artifact_manifest_module, "_WINDOWS_BOOT_ID", None, raising=False)
    monkeypatch.setattr(artifact_manifest_module.Path, "is_file", lambda _path: True)
    monkeypatch.setattr(artifact_manifest_module.Path, "read_text", lambda *_args, **_kwargs: "linux-boot-id")
    monkeypatch.setattr(
        artifact_manifest_module,
        "_run_text",
        lambda command: (calls.append(list(command)), timestamp)[1],
    )
    monkeypatch.setattr(artifact_manifest_module.os, "getpid", lambda: 101)
    first = artifact_manifest_module._boot_id()
    monkeypatch.setattr(artifact_manifest_module.os, "getpid", lambda: 202)
    second = artifact_manifest_module._boot_id()
    assert first == second
    assert first.startswith("windows-")
    assert timestamp not in first
    assert len(calls) == 1
    assert calls[0][:3] == ["powershell.exe", "-NoProfile", "-NonInteractive"]


@pytest.mark.parametrize("output", [None, "", "not-a-timestamp"])
def test_windows_boot_id_fails_closed_when_boot_timestamp_is_unavailable_or_invalid(
    monkeypatch, output
):
    monkeypatch.setattr(artifact_manifest_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(artifact_manifest_module, "_WINDOWS_BOOT_ID", None, raising=False)
    monkeypatch.setattr(artifact_manifest_module, "_run_text", lambda _command: output)
    with pytest.raises(RuntimeError, match="Windows boot timestamp"):
        artifact_manifest_module._boot_id()


def test_sha256_file_is_content_based(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == hashlib.sha256(b"abc").hexdigest()


def test_canonical_json_hash_is_independent_of_mapping_order():
    assert canonical_json_sha256({"b": 2, "a": 1}) == canonical_json_sha256(
        {"a": 1, "b": 2}
    )


def test_manifest_has_one_canonical_checkpoint_inventory(tmp_path):
    manifest = {
        "checkpoints": {
            "raw_direct": {
                "selected": {
                    "path": "best_model.sb3",
                    "sha256": "a" * 64,
                    "timestep": 50_000,
                }
            },
            "candidate_cnn": {
                "selected": {
                    "path": "best_model.sb3",
                    "sha256": "b" * 64,
                    "timestep": 60_000,
                }
            },
        }
    }
    write_manifest(tmp_path / "manifest.json", manifest)
    loaded = json.loads((tmp_path / "manifest.json").read_text("utf-8"))
    assert loaded["checkpoints"]["raw_direct"]["selected"]["timestep"] == 50_000


def test_environment_capture_redacts_requirement_credentials():
    assert sanitize_requirement_line(
        "pkg @ https://user:secret@example.test/pkg.whl?token=abc"
    ) == "pkg @ https://example.test/pkg.whl"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "-e git+https://user:token@example.test/repo.git?branch=main#egg=pkg",
            "-e git+https://example.test/repo.git",
        ),
        (
            "https://user:secret@example.test/pkg.whl?token=abc#fragment",
            "https://example.test/pkg.whl",
        ),
        (
            "pkg @ git+https://user:secret@example.test/repo.git@v1#egg=pkg",
            "pkg @ git+https://example.test/repo.git@v1",
        ),
    ],
)
def test_environment_capture_redacts_all_url_requirement_forms(source, expected):
    sanitized = sanitize_requirement_line(source)
    assert sanitized == expected
    assert "user" not in sanitized
    assert "secret" not in sanitized
    assert "token" not in sanitized


def test_runtime_writers_use_real_json_files(tmp_path):
    segment = tmp_path / "environment_segments.jsonl"
    append_environment_segment(segment, {"z": 1, "a": 2})
    append_environment_segment(segment, {"a": 3})
    assert [json.loads(line) for line in segment.read_text("utf-8").splitlines()] == [
        {"a": 2, "z": 1},
        {"a": 3},
    ]
    metrics = tmp_path / "runtime_metrics.json"
    payload = {
        "schema_version": 2,
        "target_training_seconds": 10.0,
        "recorded_training_seconds": 10.0,
        "run_wall_span_seconds": 12.0,
        "overrun_seconds": 0.0,
        "restart_count": 0,
        "max_unrecorded_seconds": 1.0,
        "start_timestep": 0,
        "start_timestep_source": "run_origin.initial_timestep",
        "end_timestep": 120,
        "steps_per_second": 12.0,
        "parameter_counts": {"total": 10, "feature_extractor": 1, "policy": 5, "value": 4},
        "peak_cuda_memory_bytes": None,
        "peak_cuda_memory_scope": "not_cuda",
        "evaluation_seconds": 2.0,
        "metrics_recorded_at_utc": "2026-07-21T00:00:12+00:00",
        "finalization_mode": "in_process",
        "selected_checkpoint_timestep": 120,
        "selection_count": 0,
        "selection_tuple": None,
        "selection_outcome": "fallback_final",
        "fallback_reason": "selection_not_run",
        "checkpoint_identity": {"filename": "model.sb3", "sha256": "a" * 64},
    }
    write_runtime_metrics(metrics, payload)
    assert read_runtime_metrics(metrics) == payload


def test_run_origin_is_atomic_exact_and_never_inferred(tmp_path):
    path = tmp_path / "run_origin.json"
    origin = write_run_origin(
        path,
        config_sha256="a" * 64,
        initial_timestep=0,
        created_at_utc="2026-07-21T00:00:00+00:00",
    )
    assert read_run_origin(path) == origin
    with pytest.raises(ValueError, match="keys differ"):
        path.write_text(
            json.dumps({**origin, "legacy_inference": True}), encoding="utf-8"
        )
        read_run_origin(path)


@pytest.mark.parametrize(
    "updates",
    [
        {"start_timestep": 121, "end_timestep": 120},
        {"recorded_training_seconds": 0.0, "steps_per_second": 1.0},
        {"recorded_training_seconds": 10.0, "steps_per_second": 99.0},
        {
            "finalization_mode": "recovered_complete_state",
            "peak_cuda_memory_scope": "training_process",
            "peak_cuda_memory_bytes": 1,
        },
        {
            "finalization_mode": "in_process",
            "peak_cuda_memory_scope": "unavailable_after_training_process",
            "peak_cuda_memory_bytes": None,
        },
    ],
)
def test_runtime_v2_rejects_impossible_arithmetic_and_peak_scope(
    tmp_path, updates
):
    path = tmp_path / "runtime.json"
    payload = {
        "schema_version": 2, "target_training_seconds": 10.0,
        "recorded_training_seconds": 10.0, "run_wall_span_seconds": 12.0,
        "overrun_seconds": 0.0, "restart_count": 0,
        "max_unrecorded_seconds": 1.0, "start_timestep": 0,
        "start_timestep_source": "run_origin.initial_timestep",
        "end_timestep": 120, "steps_per_second": 12.0,
        "parameter_counts": {"total": 10, "feature_extractor": 1, "policy": 5, "value": 4},
        "peak_cuda_memory_bytes": None, "peak_cuda_memory_scope": "not_cuda",
        "evaluation_seconds": 2.0,
        "metrics_recorded_at_utc": "2026-07-21T00:00:12+00:00",
        "finalization_mode": "in_process", "selected_checkpoint_timestep": 120,
        "selection_count": 0, "selection_tuple": None,
        "selection_outcome": "fallback_final", "fallback_reason": "selection_not_run",
        "checkpoint_identity": {"filename": "model.sb3", "sha256": "a" * 64},
    }
    payload.update(updates)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        read_runtime_metrics(path)


def test_jsonl_writer_uses_compact_canonical_json_and_rejects_nan(tmp_path):
    segment = tmp_path / "segment.jsonl"
    append_environment_segment(segment, {"b": 1, "a": 2})
    assert segment.read_text("utf-8") == '{"a":2,"b":1}\n'
    with pytest.raises(ValueError):
        append_environment_segment(segment, {"not_a_number": float("nan")})


def test_cpu_request_on_cuda_host_does_not_record_gpu(monkeypatch):
    monkeypatch.setattr("comparison.artifact_manifest.torch.cuda.is_available", lambda: True)
    manifest = collect_environment(
        ["python", "train.py"], provenance={"resolved_device": "cpu"}
    )
    assert manifest["resolved_device"] == "cpu"
    assert manifest["gpu_name"] is None
    assert manifest["gpu_uuid"] is None
    assert manifest["gpu_total_memory_bytes"] is None


def test_selected_cuda_index_is_used_for_gpu_metadata(monkeypatch):
    calls = []
    monkeypatch.setattr("comparison.artifact_manifest.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "comparison.artifact_manifest.torch.cuda.get_device_name",
        lambda index: calls.append(("name", index)) or "GPU two",
    )
    monkeypatch.setattr(
        "comparison.artifact_manifest.torch.cuda.get_device_properties",
        lambda index: calls.append(("properties", index)) or SimpleNamespace(total_memory=42),
    )
    monkeypatch.setattr(
        "comparison.artifact_manifest._gpu_uuid",
        lambda index: calls.append(("uuid", index)) or "GPU-2",
    )
    manifest = collect_environment(
        ["python"], provenance={"resolved_device": "cuda:2"}
    )
    assert manifest["gpu_name"] == "GPU two"
    assert manifest["gpu_uuid"] == "GPU-2"
    assert manifest["gpu_total_memory_bytes"] == 42
    assert {index for _, index in calls} == {2}


def test_visible_cuda_devices_maps_logical_index_to_physical_smi_id(monkeypatch):
    commands = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-visible-uuid,3")
    monkeypatch.setattr("comparison.artifact_manifest.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "comparison.artifact_manifest.torch.cuda.get_device_name", lambda _: "GPU"
    )
    monkeypatch.setattr(
        "comparison.artifact_manifest.torch.cuda.get_device_properties",
        lambda _: SimpleNamespace(total_memory=42),
    )
    monkeypatch.setattr(
        "comparison.artifact_manifest._run_text",
        lambda command: commands.append(command) or "GPU-physical-3\n",
    )
    manifest = collect_environment(["python"], {"resolved_device": "cuda:1"})
    assert manifest["gpu_uuid"] == "GPU-physical-3"
    smi_command = next(command for command in commands if command[0] == "nvidia-smi")
    assert "--id=3" in smi_command


@pytest.mark.parametrize(
    ("device", "expected_peak_call"),
    [("cpu", None), ("cuda:2", 2)],
)
def test_runtime_peak_memory_uses_model_device(
    tiny_policy, monkeypatch, device, expected_peak_call
):
    calls = []
    monkeypatch.setattr(train_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        train_module.torch.cuda,
        "max_memory_allocated",
        lambda index: calls.append(index) or 123,
    )
    model = SimpleNamespace(policy=tiny_policy, num_timesteps=120, device=device)
    state = SimpleNamespace(
        target_training_seconds=10.0,
        completed_training_seconds=8.0,
        restart_count=0,
        max_unrecorded_seconds=1.0,
        last_checkpoint_timestep=120,
        last_checkpoint_file="model.sb3",
        last_checkpoint_sha256="a" * 64,
        started_at_utc="2026-07-21T00:00:00+00:00",
    )
    origin = {"initial_timestep": 0}
    metrics = train_module.comparison_runtime_metrics(
        model,
        state,
        origin=origin,
        metrics_recorded_at_utc="2026-07-21T00:00:08+00:00",
        evaluation_seconds=0.0,
        finalization_mode="in_process",
    )
    assert metrics["peak_cuda_memory_bytes"] == (123 if expected_peak_call is not None else None)
    assert calls == ([] if expected_peak_call is None else [expected_peak_call])


def test_invalid_comparison_provenance_fails_before_segment_write(tmp_path):
    args = SimpleNamespace(
        comparison_baseline_sha256="not-a-commit",
        comparison_config_sha256="a" * 64,
        comparison_scenario_sha256="b" * 64,
        comparison_split_sha256="c" * 64,
        comparison_lock_sha256="d" * 63,
    )
    segment = tmp_path / "environment_segments.jsonl"
    with pytest.raises(ValueError, match="provenance"):
        train_module.comparison_runtime_provenance(args)
    assert not segment.exists()


def test_comparison_runtime_metrics_use_wall_clock_state_and_model_counts(
    tiny_policy,
):
    model = SimpleNamespace(policy=tiny_policy, num_timesteps=120)
    state = SimpleNamespace(
        target_training_seconds=10.0,
        completed_training_seconds=8.0,
        restart_count=1,
        max_unrecorded_seconds=2.0,
        last_checkpoint_timestep=120,
        last_checkpoint_file="model_120_g2.sb3",
        last_checkpoint_sha256="a" * 64,
        started_at_utc="2026-07-21T00:00:00+00:00",
    )
    metrics = train_module.comparison_runtime_metrics(
        model,
        state,
        origin={"initial_timestep": 20},
        metrics_recorded_at_utc="2026-07-21T00:00:12+00:00",
        evaluation_seconds=3.0,
        finalization_mode="in_process",
    )
    assert metrics["target_training_seconds"] == 10.0
    assert metrics["recorded_training_seconds"] == 8.0
    assert metrics["start_timestep"] == 20
    assert metrics["end_timestep"] == 120
    assert metrics["schema_version"] == 2
    assert metrics["run_wall_span_seconds"] == 12.0
    assert metrics["start_timestep_source"] == "run_origin.initial_timestep"
    assert metrics["parameter_counts"]["total"] == sum(
        parameter.numel() for parameter in tiny_policy.parameters()
    )


def test_runtime_metrics_uses_verified_holdout_best_checkpoint(tmp_path, tiny_policy, monkeypatch):
    (tmp_path / "best_model.sb3").write_bytes(b"best")
    state_checkpoint = tmp_path / "checkpoints" / "model_40_g1.sb3"
    state_checkpoint.parent.mkdir()
    state_checkpoint.write_bytes(b"state")
    (tmp_path / "holdout_selection.csv").write_text(
        "timestep,mean_terminal_score,mean_dropout_rate,mean_delay_days,is_best\n"
        "50,10.0,0.2,3.0,1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(train_module, "model_num_timesteps", lambda path: 50)
    selected = train_module.runtime_selected_checkpoint(
        tmp_path,
        SimpleNamespace(
            last_checkpoint_timestep=40,
            last_checkpoint_file=state_checkpoint.name,
            last_checkpoint_sha256=hashlib.sha256(b"state").hexdigest(),
        ),
    )
    assert selected["selected_checkpoint_timestep"] == 50
    assert selected["checkpoint_identity"]["filename"] == "best_model.sb3"
    assert selected["selection_count"] == 1
    assert selected["selection_tuple"] == [10.0, -0.2, -3.0]
    assert selected["selection_outcome"] == "best_model"
    assert selected["fallback_reason"] is None


def test_runtime_metrics_falls_back_to_verified_complete_state_checkpoint(tmp_path):
    state_checkpoint = tmp_path / "checkpoints" / "model_40_g1.sb3"
    state_checkpoint.parent.mkdir()
    state_checkpoint.write_bytes(b"state")
    digest = hashlib.sha256(b"state").hexdigest()
    selected = train_module.runtime_selected_checkpoint(
        tmp_path,
        SimpleNamespace(
            last_checkpoint_timestep=40,
            last_checkpoint_file=state_checkpoint.name,
            last_checkpoint_sha256=digest,
        ),
    )
    assert selected["selected_checkpoint_timestep"] == 40
    assert selected["checkpoint_identity"] == {
        "filename": state_checkpoint.name,
        "sha256": digest,
    }
    assert selected["selection_count"] == 0
    assert selected["selection_tuple"] is None
    assert selected["selection_outcome"] == "fallback_final"
    assert selected["fallback_reason"] == "selection_not_run"


def test_runtime_metrics_ignores_malformed_holdout_selection_and_falls_back(tmp_path):
    (tmp_path / "best_model.sb3").write_bytes(b"not-used")
    (tmp_path / "holdout_selection.csv").write_text(
        "timestep,is_best\n50,1\n", encoding="utf-8"
    )
    checkpoint = tmp_path / "checkpoints" / "model_40_g1.sb3"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"state")
    digest = hashlib.sha256(b"state").hexdigest()
    selected = train_module.runtime_selected_checkpoint(
        tmp_path,
        SimpleNamespace(
            last_checkpoint_timestep=40,
            last_checkpoint_file=checkpoint.name,
            last_checkpoint_sha256=digest,
        ),
    )
    assert selected["selected_checkpoint_timestep"] == 40
    assert selected["fallback_reason"] == "selection_metadata_invalid"


@pytest.mark.parametrize(
    "selection_csv",
    [
        "timestep,is_best\n50,1\n",
        (
            "timestep,mean_terminal_score,mean_dropout_rate,mean_delay_days,is_best\n"
            "50,not-a-number,0.2,3.0,1\n"
        ),
        (
            "timestep,mean_terminal_score,mean_dropout_rate,mean_delay_days,is_best\n"
            "not-an-int,10.0,0.2,3.0,1\n"
        ),
    ],
)
def test_runtime_metrics_rejects_malformed_readable_selected_checkpoint(
    tmp_path, monkeypatch, selection_csv
):
    (tmp_path / "best_model.sb3").write_bytes(b"readable-best")
    (tmp_path / "holdout_selection.csv").write_text(selection_csv, encoding="utf-8")
    monkeypatch.setattr(train_module, "model_num_timesteps", lambda path: 50)
    checkpoint = tmp_path / "checkpoints" / "model_40_g1.sb3"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"state")
    digest = hashlib.sha256(b"state").hexdigest()
    selected = train_module.runtime_selected_checkpoint(
        tmp_path,
        SimpleNamespace(
            last_checkpoint_timestep=40,
            last_checkpoint_file=checkpoint.name,
            last_checkpoint_sha256=digest,
        ),
    )
    assert selected["selected_checkpoint_timestep"] == 40
