"""Tests for CSV loss logging and notebook-friendly training plots."""

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import train as train_module
from alloc_env.block import Block
from alloc_env.callbacks import AllocationCallback, TrainingMetricsCsvWriter
from alloc_env.observation_state import ObservationScales
from alloc_env.strategy import BaseGridStrategy
from alloc_env.workspace import Workspace
from plot_training_curves import plot_training_curves
from visualize_grids import visualize_grids


class TrainingVisualizationTests(unittest.TestCase):
    def test_run_config_json_round_trip(self):
        config = {
            "observation_schema_version": 3,
            "extractor": "candidate-cnn",
            "state_context": "full",
            "grid_size": 64,
            "active_workspace_codes": ["PE001"],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run_config.json"
            train_module.write_run_config(path.parent, config)
            loader = getattr(train_module, "load_run_config", None)
            self.assertIsNotNone(loader)
            loaded = loader(path)

        self.assertEqual(config, loaded)

    def test_colab_defaults_use_candidate_cnn_without_attention_options(self):
        notebook = json.loads(
            Path(__file__).with_name("Colab_train.ipynb").read_text(
                encoding="utf-8"
            )
        )
        notebook_text = "\n".join(
            "".join(cell.get("source", ""))
            if isinstance(cell.get("source", ""), list)
            else cell.get("source", "")
            for cell in notebook["cells"]
        )

        self.assertIn('EXTRACTOR       = "candidate-cnn"', notebook_text)
        self.assertIn("OBSERVATION_SCHEMA_VERSION = 3", notebook_text)
        self.assertIn('STATE_CONTEXT   = "full"', notebook_text)
        self.assertIn("GAE_LAMBDA", notebook_text)
        self.assertIn("SEED", notebook_text)
        self.assertIn("N_STEPS     = 960", notebook_text)
        self.assertIn("MONTHLY_JITTER = 20", notebook_text)
        self.assertIn(
            "EMPIRICAL_PROFILE_PROBABILITY = 0.2", notebook_text
        )
        self.assertIn(
            'ACTIVE_WORKSPACE_CODES = "PE049,PE050,PE055,PE054,'
            'PE056,PE048,PE044,PE059,PE060,PE061"',
            notebook_text,
        )
        self.assertIn(
            "/content/drive/MyDrive/CNN-RL-outputs/"
            "candidate_cnn_10ws_empty_v1",
            notebook_text,
        )
        self.assertIn("--monthly-jitter", notebook_text)
        self.assertIn("--empirical-profile-probability", notebook_text)
        self.assertIn("--state-context", notebook_text)
        self.assertNotIn("N_FUTURE_BLOCKS", notebook_text)
        self.assertNotIn("--n-future-blocks", notebook_text)
        self.assertNotIn("EMBED_DIM", notebook_text)
        self.assertNotIn("NUM_HEADS", notebook_text)

    def test_grid_workflow_reports_independent_coordinate_maps(self):
        strategy = BaseGridStrategy(step=1.0)
        workspace = Workspace(
            code="RECTANGULAR",
            name="Rectangular",
            origin_x=0.0,
            origin_y=0.0,
            length=128.0,
            breadth=32.0,
            strategy=strategy,
        )
        block = Block(
            name="CURRENT",
            ship_no="T001",
            block_type="BUILD",
            length=4.0,
            breadth=4.0,
            height=1.0,
            weight=1.0,
            in_date=date(2026, 1, 5),
            out_date=date(2026, 1, 20),
        )
        scales = ObservationScales(
            max_length=4.0,
            max_breadth=4.0,
            max_duration=20,
            base_date=date(2026, 1, 5),
            date_span_workdays=1,
            max_workspace_area=4096.0,
            total_workspace_area=4096.0,
            max_workspace_length=128.0,
            max_workspace_breadth=32.0,
            dropout_threshold=7,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                data_dir="unused",
                output_dir=tmpdir,
                grid_size=64,
                synthetic=False,
                date=None,
                seed=0,
            )
            output = io.StringIO()
            with (
                patch.object(
                    train_module,
                    "load_allocation_scenario",
                    return_value=([block], [workspace]),
                ),
                patch(
                    "alloc_env.observation_state.build_observation_scales",
                    return_value=scales,
                ),
                redirect_stdout(output),
            ):
                visualize_grids(args)

        report = output.getvalue()
        self.assertIn("x=0.500 px/m", report)
        self.assertIn("y=2.000 px/m", report)

    def test_allocation_log_uses_resolved_reward_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = AllocationCallback(tmpdir, verbose=0)
            callback._on_training_start()
            callback._on_training_end()

            with open(
                Path(tmpdir) / "training_log.csv", encoding="utf-8"
            ) as file:
                header = next(csv.reader(file))

        self.assertIn("resolved_reward", header)
        self.assertIn("terminal_residual", header)
        self.assertIn("terminal_score", header)
        self.assertNotIn("shaped_reward", header)

    def test_training_metrics_writer_records_train_losses(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TrainingMetricsCsvWriter(tmpdir)
            writer.write(
                {
                    "train/policy_gradient_loss": -0.12,
                    "train/value_loss": 1.5,
                    "train/entropy_loss": -2.0,
                    "train/approx_kl": 0.01,
                    "train/clip_fraction": 0.2,
                    "train/loss": 0.9,
                    "train/explained_variance": 0.4,
                    "diagnostics/cnn_gradient_norm": 2.5,
                    "diagnostics/cnn_weight_change": 0.03,
                    "diagnostics/workspace_feature_variance": 0.4,
                    "diagnostics/candidate_channel_sensitivity": 0.2,
                },
                {},
                step=128,
            )
            writer.close()

            with open(Path(tmpdir) / "loss_log.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(1, len(rows))
        self.assertEqual("128", rows[0]["timestep"])
        self.assertEqual("-0.120000", rows[0]["policy_gradient_loss"])
        self.assertEqual("1.500000", rows[0]["value_loss"])
        self.assertEqual("2.500000", rows[0]["cnn_gradient_norm"])
        self.assertEqual("0.200000", rows[0]["candidate_channel_sensitivity"])

    def test_training_metrics_writer_skips_non_train_dumps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TrainingMetricsCsvWriter(tmpdir)
            writer.write({"time/fps": 100}, {}, step=64)
            writer.close()

            with open(Path(tmpdir) / "loss_log.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual([], rows)

    def test_training_metrics_resume_appends_existing_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = TrainingMetricsCsvWriter(tmpdir)
            first.write({"train/loss": 1.0}, {}, step=10)
            first.close()

            resumed = TrainingMetricsCsvWriter(tmpdir, append=True)
            resumed.write({"train/loss": 0.5}, {}, step=20)
            resumed.close()

            with open(Path(tmpdir) / "loss_log.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(["10", "20"], [row["timestep"] for row in rows])
        self.assertEqual(["1.000000", "0.500000"], [row["loss"] for row in rows])

    def test_allocation_log_resume_preserves_rows_and_episode_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "training_log.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(AllocationCallback.CSV_COLUMNS)
                writer.writerow([7, 700, 0.1, 0.0, 0.1, 0.1, 0, 0, 0, 1.0])

            callback = AllocationCallback(
                tmpdir, verbose=0, append=True
            )
            callback._on_training_start()
            episode_count = callback._episode_count
            callback._on_training_end()

            with path.open(encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(7, episode_count)
        self.assertEqual(1, len(rows))
        self.assertEqual("7", rows[0]["episode"])

    def test_plot_training_curves_creates_quality_axes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with open(output_dir / "training_log.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode", "timestep", "resolved_reward", "terminal_residual",
                    "terminal_score", "episode_reward",
                    "delayed_count", "dropout_count", "total_delay_days", "success_rate",
                ])
                writer.writerow([1, 100, -0.8, 0.0, -0.8, -0.8, 3, 1, 7, 0.5])
                writer.writerow([2, 200, -0.2, 0.0, -0.2, -0.2, 2, 0, 3, 0.7])

            with open(output_dir / "loss_log.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestep", "policy_gradient_loss", "value_loss", "entropy_loss",
                    "approx_kl", "clip_fraction", "loss", "explained_variance",
                ])
                writer.writerow([100, -0.1, 1.0, -2.0, 0.01, 0.1, 0.8, 0.2])
                writer.writerow([200, -0.05, 0.7, -1.8, 0.02, 0.2, 0.5, 0.3])

            fig = plot_training_curves(output_dir, show=False)

        self.assertEqual(4, len(fig.axes))
        self.assertEqual("Reward", fig.axes[0].get_title())
        self.assertEqual("Training Loss", fig.axes[1].get_title())
        self.assertEqual("Success Rate", fig.axes[2].get_title())
        self.assertEqual("Delay and Dropout", fig.axes[3].get_title())
        self.assertGreaterEqual(len(fig.axes[0].lines), 1)
        self.assertGreaterEqual(len(fig.axes[1].lines), 1)
        self.assertGreaterEqual(len(fig.axes[2].lines), 1)
        self.assertGreaterEqual(len(fig.axes[3].lines), 2)

    def test_plot_training_curves_explains_empty_reward_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with open(output_dir / "training_log.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode", "timestep", "resolved_reward", "terminal_residual",
                    "terminal_score", "episode_reward",
                    "delayed_count", "dropout_count", "total_delay_days", "success_rate",
                ])

            with open(output_dir / "loss_log.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestep", "policy_gradient_loss", "value_loss", "entropy_loss",
                    "approx_kl", "clip_fraction", "loss", "explained_variance",
                ])
                writer.writerow([100, -0.1, 1.0, -2.0, 0.01, 0.1, 0.8, 0.2])

            fig = plot_training_curves(output_dir, show=False)

        reward_texts = [text.get_text() for text in fig.axes[0].texts]
        self.assertIn("No completed episodes logged yet", reward_texts)


if __name__ == "__main__":
    unittest.main()
