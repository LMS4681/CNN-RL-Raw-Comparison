"""Tests for CSV loss logging and notebook-friendly training plots."""

import csv
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from alloc_env.callbacks import TrainingMetricsCsvWriter
from plot_training_curves import plot_training_curves


class TrainingVisualizationTests(unittest.TestCase):
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

    def test_training_metrics_writer_skips_non_train_dumps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TrainingMetricsCsvWriter(tmpdir)
            writer.write({"time/fps": 100}, {}, step=64)
            writer.close()

            with open(Path(tmpdir) / "loss_log.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual([], rows)

    def test_plot_training_curves_creates_quality_axes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with open(output_dir / "training_log.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode", "timestep", "reward", "shaped_reward", "episode_reward",
                    "delayed_count", "dropout_count", "total_delay_days", "success_rate",
                ])
                writer.writerow([1, 100, -1.0, 0.2, -0.8, 3, 1, 7, 0.5])
                writer.writerow([2, 200, -0.5, 0.3, -0.2, 2, 0, 3, 0.7])

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
                    "episode", "timestep", "reward", "shaped_reward", "episode_reward",
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
