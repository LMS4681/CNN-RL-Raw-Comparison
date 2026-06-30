"""CLI regression tests for training resume support."""

import sys
import unittest
from unittest.mock import patch

import train as train_module


class TrainResumeCliTest(unittest.TestCase):
    def test_resume_from_argument_is_accepted(self):
        captured = {}

        def fake_train(args):
            captured["resume_from"] = args.resume_from

        argv = [
            "train.py",
            "--resume-from",
            ".\\output\\block_placement_ppo.zip",
            "--no-export-onnx",
        ]

        with patch.object(sys, "argv", argv), patch.object(train_module, "train", fake_train):
            train_module.main()

        self.assertEqual(captured["resume_from"], ".\\output\\block_placement_ppo.zip")


if __name__ == "__main__":
    unittest.main()
