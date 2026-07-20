import hashlib
import json
import platform

import pytest
import torch.nn as nn
from types import SimpleNamespace

from comparison.artifact_manifest import (
    REQUIRED_ENVIRONMENT_KEYS,
    append_environment_segment,
    canonical_json_sha256,
    collect_environment,
    count_trainable_parameters,
    sanitize_requirement_line,
    sha256_file,
    write_manifest,
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


def test_runtime_writers_use_real_json_files(tmp_path):
    segment = tmp_path / "environment_segments.jsonl"
    append_environment_segment(segment, {"z": 1, "a": 2})
    append_environment_segment(segment, {"a": 3})
    assert [json.loads(line) for line in segment.read_text("utf-8").splitlines()] == [
        {"a": 2, "z": 1},
        {"a": 3},
    ]
    metrics = tmp_path / "runtime_metrics.json"
    write_runtime_metrics(metrics, {"steps_per_second": 4.0})
    assert json.loads(metrics.read_text("utf-8")) == {"steps_per_second": 4.0}


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
    )
    metrics = train_module.comparison_runtime_metrics(
        model, state, start_timestep=20, end_to_end_seconds=12.0, evaluation_seconds=3.0
    )
    assert metrics["target_training_seconds"] == 10.0
    assert metrics["recorded_training_seconds"] == 8.0
    assert metrics["start_timestep"] == 20
    assert metrics["end_timestep"] == 120
    assert metrics["parameter_counts"]["total"] == sum(
        parameter.numel() for parameter in tiny_policy.parameters()
    )
