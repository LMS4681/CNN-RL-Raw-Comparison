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


def minimal_model_and_env():
    observation_space = gym.spaces.Dict(
        {
            "block": gym.spaces.Box(
                0.0, 1.0, shape=(10,), dtype=np.float32
            )
        }
    )
    policy = SimpleNamespace(device=torch.device("cpu"))
    return (
        SimpleNamespace(policy=policy),
        SimpleNamespace(observation_space=observation_space),
    )


def checked_onnx_model():
    return SimpleNamespace(graph=SimpleNamespace(input=[]))


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


if __name__ == "__main__":
    unittest.main()
