"""Plot training reward and loss curves from CSV logs.

Colab usage:

    from plot_training_curves import plot_training_curves
    plot_training_curves("/content/drive/MyDrive/CNN_RL_output")
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def _read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"CSV log not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _to_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def _plot_reward(ax, reward_df: pd.DataFrame, smooth_window: int) -> None:
    if reward_df.empty:
        ax.text(
            0.5,
            0.5,
            "No completed episodes logged yet",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Cumulative Reward")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)
        return

    x_col = "timestep" if "timestep" in reward_df.columns else "episode"
    y_col = "episode_reward" if "episode_reward" in reward_df.columns else "reward"

    x = _to_numeric(reward_df, x_col)
    y = _to_numeric(reward_df, y_col)
    ax.plot(x, y, label=y_col, linewidth=1.5)

    if smooth_window > 1 and len(y) >= smooth_window:
        smoothed = y.rolling(smooth_window, min_periods=1).mean()
        ax.plot(x, smoothed, label=f"{smooth_window}-episode mean", linewidth=2.0)

    ax.set_title("Cumulative Reward")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()


def _plot_loss(ax, loss_df: pd.DataFrame, smooth_window: int) -> None:
    if loss_df.empty:
        ax.text(
            0.5,
            0.5,
            "loss_log.csv is not available yet",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Training Loss")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        return

    x = _to_numeric(loss_df, "timestep")
    plotted = False
    for column in ["loss", "policy_gradient_loss", "value_loss"]:
        if column not in loss_df.columns:
            continue
        y = _to_numeric(loss_df, column)
        if y.notna().sum() == 0:
            continue
        if smooth_window > 1 and len(y) >= smooth_window:
            y = y.rolling(smooth_window, min_periods=1).mean()
        ax.plot(x, y, label=column, linewidth=1.5)
        plotted = True

    if not plotted:
        ax.text(
            0.5,
            0.5,
            "No train loss columns found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_title("Training Loss")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend()


def plot_training_curves(
    output_dir: str | Path = "./output",
    reward_smooth_window: int = 10,
    loss_smooth_window: int = 1,
    show: bool = True,
    save_path: Optional[str | Path] = None,
):
    """Draw cumulative reward and training loss graphs from output CSV files."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    reward_df = _read_csv(output_dir / "training_log.csv", required=True)
    loss_df = _read_csv(output_dir / "loss_log.csv", required=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    _plot_reward(axes[0], reward_df, reward_smooth_window)
    _plot_loss(axes[1], loss_df, loss_smooth_window)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RL training reward/loss curves")
    parser.add_argument("--output-dir", default="./output", help="training output directory")
    parser.add_argument("--save-path", default=None, help="optional image output path")
    parser.add_argument("--reward-smooth-window", type=int, default=10)
    parser.add_argument("--loss-smooth-window", type=int, default=1)
    args = parser.parse_args()

    plot_training_curves(
        args.output_dir,
        reward_smooth_window=args.reward_smooth_window,
        loss_smooth_window=args.loss_smooth_window,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
