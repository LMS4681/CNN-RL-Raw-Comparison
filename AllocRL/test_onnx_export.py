import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

import train as train_module
from alloc_env.observation_state import build_observation_space


class ActorHead(torch.nn.Module):
    def __init__(self, finite: bool = True):
        super().__init__()
        self.finite = finite

    def forward(self, features):
        value = 0.0 if self.finite else float("nan")
        return torch.full(
            (features.shape[0], 2),
            value,
            dtype=features.dtype,
            device=features.device,
        )


class ActorMlp(torch.nn.Module):
    @staticmethod
    def forward_actor(features):
        return features


class MinimalPolicy(torch.nn.Module):
    def __init__(self, finite: bool = True):
        super().__init__()
        self.device = torch.device("cpu")
        self.pi_features_extractor = torch.nn.Identity()
        self.mlp_extractor = ActorMlp()
        self.action_net = ActorHead(finite=finite)

    def extract_features(self, observation, _extractor):
        return torch.cat(
            [observation[key].flatten(start_dim=1) for key in sorted(observation)],
            dim=1,
        )


class BlockOnlyPolicy(MinimalPolicy):
    def extract_features(self, observation, _extractor):
        return observation["block"]

def minimal_model_and_env():
    observation_space = gym.spaces.Dict(
        {
            "block": gym.spaces.Box(
                0.0, 1.0, shape=(10,), dtype=np.float32
            )
        }
    )
    policy = MinimalPolicy()
    return (
        SimpleNamespace(policy=policy),
        SimpleNamespace(observation_space=observation_space),
    )


def checked_onnx_model(input_names=()):
    inputs = [SimpleNamespace(name=name) for name in input_names]
    return SimpleNamespace(graph=SimpleNamespace(input=inputs))


class OnnxExportCompatibilityTests(unittest.TestCase):
    def test_export_forces_legacy_path_when_dynamo_keyword_is_supported(self):
        captured = {}

        def modern_export(
            model,
            args,
            output_path,
            *,
            input_names,
            output_names,
            dynamic_axes,
            opset_version,
            dynamo=True,
        ):
            captured["dynamo"] = dynamo

        model, env = minimal_model_and_env()
        with (
            patch("torch.onnx.export", new=modern_export),
            patch("onnx.load", return_value=checked_onnx_model()),
            patch("onnx.checker.check_model"),
        ):
            train_module.export_to_onnx(model, env, "unused.onnx")

        self.assertIs(False, captured["dynamo"])

    def test_export_omits_dynamo_for_legacy_pytorch_signature(self):
        captured = {}

        def legacy_export(
            model,
            args,
            output_path,
            *,
            input_names,
            output_names,
            dynamic_axes,
            opset_version,
        ):
            captured["called"] = True

        model, env = minimal_model_and_env()
        with (
            patch("torch.onnx.export", new=legacy_export),
            patch("onnx.load", return_value=checked_onnx_model()),
            patch("onnx.checker.check_model"),
        ):
            train_module.export_to_onnx(model, env, "unused.onnx")

        self.assertTrue(captured["called"])

    def test_optional_export_failure_is_nonfatal_and_removes_partial_file(self):
        self.assertTrue(
            hasattr(train_module, "try_export_to_onnx"),
            "optional ONNX export needs an isolated failure boundary",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            onnx_path = Path(temp_dir) / "partial.onnx"
            onnx_path.write_bytes(b"partial")
            output = io.StringIO()

            with (
                patch.object(
                    train_module,
                    "export_to_onnx",
                    side_effect=ModuleNotFoundError("No module named 'onnxscript'"),
                ),
                contextlib.redirect_stdout(output),
            ):
                exported = train_module.try_export_to_onnx(
                    object(), object(), onnx_path
                )

            self.assertFalse(exported)
            self.assertFalse(onnx_path.exists())
            self.assertIn("ONNX", output.getvalue())
            self.assertIn("SB3", output.getvalue())
            self.assertIn("평가는 계속", output.getvalue())

    def test_schema3_export_uses_all_sorted_inputs_with_dynamic_batches(self):
        observation_space = build_observation_space()
        expected_keys = sorted(observation_space.spaces)
        captured = {}

        def capture_export(
            model,
            args,
            output_path,
            *,
            input_names,
            output_names,
            dynamic_axes,
            opset_version,
            **kwargs,
        ):
            captured["input_names"] = input_names
            captured["dynamic_axes"] = dynamic_axes

        env = SimpleNamespace(observation_space=observation_space)
        model = SimpleNamespace(policy=MinimalPolicy())
        with (
            patch("torch.onnx.export", new=capture_export),
            patch(
                "onnx.load",
                return_value=checked_onnx_model(expected_keys),
            ),
            patch("onnx.checker.check_model"),
        ):
            train_module.export_to_onnx(model, env, "unused.onnx")

        self.assertEqual(expected_keys, captured["input_names"])
        self.assertEqual(9, len(captured["input_names"]))
        for key in [*expected_keys, "action_logits"]:
            self.assertEqual({0: "batch"}, captured["dynamic_axes"][key])

    def test_export_rejects_nonfinite_actor_output(self):
        observation_space = build_observation_space()
        env = SimpleNamespace(observation_space=observation_space)
        model = SimpleNamespace(policy=MinimalPolicy(finite=False))

        with (
            patch("torch.onnx.export"),
            patch("onnx.load", return_value=checked_onnx_model()),
            patch("onnx.checker.check_model"),
            self.assertRaisesRegex(ValueError, "finite"),
        ):
            train_module.export_to_onnx(model, env, "unused.onnx")

    def test_actual_graph_keeps_all_inputs_for_block_only_extractor(self):
        import onnx

        observation_space = build_observation_space()
        expected_keys = sorted(observation_space.spaces)
        env = SimpleNamespace(observation_space=observation_space)
        model = SimpleNamespace(policy=BlockOnlyPolicy())

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "block-only.onnx"
            train_module.export_to_onnx(model, env, str(path))
            exported = onnx.load(path)

        self.assertEqual(
            expected_keys,
            [value.name for value in exported.graph.input],
        )
        self.assertTrue(all(
            value.type.tensor_type.shape.dim[0].dim_param == "batch"
            for value in exported.graph.input
        ))
        self.assertEqual(
            "batch",
            exported.graph.output[0].type.tensor_type.shape.dim[0].dim_param,
        )


if __name__ == "__main__":
    unittest.main()
