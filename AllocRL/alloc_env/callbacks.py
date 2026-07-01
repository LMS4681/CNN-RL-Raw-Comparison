"""
학습 콜백 - 에피소드별 상세 지표 로깅.

TensorBoard + CSV 로그 동시 기록.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter

from .simulator import SimulationResult


# ── 전역 상수 (AllocConst 대응) ──────────────────────────────────

DELAY_THRESHOLD = 2
DROPOUT_THRESHOLD = 7


class TrainingMetricsCsvWriter(KVWriter):
    """Write SB3 train/* loss metrics to loss_log.csv."""

    METRIC_COLUMNS = [
        ("train/policy_gradient_loss", "policy_gradient_loss"),
        ("train/value_loss", "value_loss"),
        ("train/entropy_loss", "entropy_loss"),
        ("train/approx_kl", "approx_kl"),
        ("train/clip_fraction", "clip_fraction"),
        ("train/loss", "loss"),
        ("train/explained_variance", "explained_variance"),
    ]

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.log_dir / "loss_log.csv"
        self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(
            ["timestep"] + [column for _, column in self.METRIC_COLUMNS]
        )

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, tuple[str, ...]],
        step: int = 0,
    ) -> None:
        if not any(key in key_values for key, _ in self.METRIC_COLUMNS):
            return

        row = [int(step)]
        for key, _ in self.METRIC_COLUMNS:
            value = key_values.get(key, "")
            if isinstance(value, (int, float, np.number)):
                row.append(f"{float(value):.6f}")
            else:
                row.append(value)
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def close(self) -> None:
        if not self._csv_file.closed:
            self._csv_file.close()


class TrainingMetricsCallback(BaseCallback):
    """Attach TrainingMetricsCsvWriter after SB3 configures its logger."""

    def __init__(self, log_dir: str = "./output", verbose: int = 1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self._writer: Optional[TrainingMetricsCsvWriter] = None

    def _on_training_start(self) -> None:
        self._writer = TrainingMetricsCsvWriter(self.log_dir)
        self.model.logger.output_formats.append(self._writer)
        if self.verbose:
            print(f"[Callback] Loss CSV 로그: {Path(self.log_dir) / 'loss_log.csv'}")

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        if self._writer is None:
            return
        self.model.logger.dump(step=self.num_timesteps)
        if self._writer in self.model.logger.output_formats:
            self.model.logger.output_formats.remove(self._writer)
        self._writer.close()


class AllocationCallback(BaseCallback):
    """
    에피소드 종료 시 상세 배치 지표를 로깅하는 콜백.

    기록 지표:
    - terminal_reward: 최종 배치 품질 보상
    - shaped_reward: 중간 shaping 보상 합
    - episode_reward: terminal_reward + shaped_reward
    - delayed_count: 지연 블록 수
    - dropout_count: 탈락 블록 수
    - total_delay_days: 총 지연 일수 합
    - success_rate: 정상 배치(준수) 비율
    """

    def __init__(
        self,
        log_dir: str = "./output",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # CSV 로그
        self._csv_path = self.log_dir / "training_log.csv"
        self._csv_file = None
        self._csv_writer = None

        # 에피소드 카운터
        self._episode_count = 0

        # 지표 히스토리 (학습 후 시각화용)
        self.history: Dict[str, List[float]] = {
            "episode": [],
            "reward": [],
            "shaped_reward": [],
            "episode_reward": [],
            "delayed_count": [],
            "dropout_count": [],
            "total_delay_days": [],
            "success_rate": [],
            "timestep": [],
        }

    def _on_training_start(self):
        self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "episode", "timestep", "reward", "shaped_reward", "episode_reward",
            "delayed_count", "dropout_count",
            "total_delay_days", "success_rate",
        ])
        if self.verbose:
            print(f"[Callback] CSV 로그: {self._csv_path}")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        if dones is None:
            return True

        for i, done in enumerate(dones):
            if not done:
                continue

            info = infos[i] if i < len(infos) else {}
            raw_result = info.get("raw_result", None)

            if raw_result is None:
                continue

            self._episode_count += 1
            terminal_reward = info.get("terminal_reward", 0.0)
            shaped_reward = info.get("step_reward_sum", 0.0)
            episode_reward = info.get(
                "episode_reward", terminal_reward + shaped_reward
            )
            self._log_episode(raw_result, terminal_reward,
                              shaped_reward, episode_reward)

        return True

    def _log_episode(
        self,
        result: SimulationResult,
        terminal_reward: float,
        shaped_reward: float,
        episode_reward: float,
    ):
        """에피소드 종료 시 지표를 계산하고 로깅."""
        delay_days = result.delay_days
        n = len(delay_days)

        dropout_count = sum(1 for d in delay_days if d == SimulationResult.DROPOUT)
        delayed_count = sum(1 for d in delay_days
                           if d != SimulationResult.DROPOUT and d > DELAY_THRESHOLD)
        normal_count = n - dropout_count - delayed_count
        success_rate = normal_count / n if n > 0 else 0.0

        total_delay = sum(d for d in delay_days
                         if d != SimulationResult.DROPOUT and d > 0)

        # TensorBoard 로깅
        self.logger.record("alloc/reward", terminal_reward)
        self.logger.record("alloc/terminal_reward", terminal_reward)
        self.logger.record("alloc/shaped_reward", shaped_reward)
        self.logger.record("alloc/episode_reward", episode_reward)
        self.logger.record("alloc/delayed_count", delayed_count)
        self.logger.record("alloc/dropout_count", dropout_count)
        self.logger.record("alloc/total_delay_days", total_delay)
        self.logger.record("alloc/success_rate", success_rate)
        self.logger.record("alloc/episode", self._episode_count)

        # 히스토리 저장
        self.history["episode"].append(self._episode_count)
        self.history["timestep"].append(self.num_timesteps)
        self.history["reward"].append(terminal_reward)
        self.history["shaped_reward"].append(shaped_reward)
        self.history["episode_reward"].append(episode_reward)
        self.history["delayed_count"].append(delayed_count)
        self.history["dropout_count"].append(dropout_count)
        self.history["total_delay_days"].append(total_delay)
        self.history["success_rate"].append(success_rate)

        # CSV 기록
        if self._csv_writer:
            self._csv_writer.writerow([
                self._episode_count, self.num_timesteps, f"{terminal_reward:.4f}",
                f"{shaped_reward:.4f}", f"{episode_reward:.4f}",
                delayed_count, dropout_count, total_delay, f"{success_rate:.4f}",
            ])
            self._csv_file.flush()

        # 콘솔 출력 (10 에피소드마다)
        if self.verbose and self._episode_count % 10 == 0:
            print(f"  [EP {self._episode_count:>4}] "
                  f"reward={terminal_reward:>+7.3f}  "
                  f"episode={episode_reward:>+7.3f}  "
                  f"delay={delayed_count}  dropout={dropout_count}  "
                  f"success={success_rate:.1%}")

    def _on_training_end(self):
        if self._csv_file:
            self._csv_file.close()
        if self.verbose:
            print(f"[Callback] 총 {self._episode_count} 에피소드 학습 완료")
            print(f"[Callback] CSV 로그 저장: {self._csv_path}")
