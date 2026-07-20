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
from comparison.training_log_validation import read_curve_log
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter

from .simulator import SimulationResult


# ── 전역 상수 (AllocConst 대응) ──────────────────────────────────

DELAY_THRESHOLD = 2
DROPOUT_THRESHOLD = 7


class CnnDiagnosticTracker:
    """Measure whether the candidate CNN receives and uses learning signals."""

    def __init__(self, extractor):
        self.extractor = extractor
        self.module = getattr(extractor, "image_encoder", None)
        self._hooks = []
        self._grad_sq = 0.0
        self._snapshot = None

    def attach(self) -> None:
        if self.module is None:
            return
        self.close()
        self._snapshot = [
            parameter.detach().clone()
            for parameter in self.module.parameters()
        ]
        self._hooks = [
            parameter.register_hook(self._capture_gradient)
            for parameter in self.module.parameters()
            if parameter.requires_grad
        ]

    def _capture_gradient(self, gradient: torch.Tensor) -> None:
        self._grad_sq += float(
            gradient.detach().square().sum().item()
        )

    def record_update(self) -> Dict[str, float]:
        if self.module is None:
            return {}
        current = [
            parameter.detach() for parameter in self.module.parameters()
        ]
        if self._snapshot is None:
            self._snapshot = [parameter.clone() for parameter in current]
        delta_sq = sum(
            float((now - old).square().sum().item())
            for now, old in zip(current, self._snapshot)
        )
        metrics = {
            "cnn_gradient_norm": self._grad_sq ** 0.5,
            "cnn_weight_change": delta_sq ** 0.5,
        }
        self._snapshot = [parameter.clone() for parameter in current]
        self._grad_sq = 0.0
        return metrics

    def measure_features(
        self,
        observations: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        if self.module is None:
            return {}
        grids = observations["grids"]
        without_candidate = grids.clone()
        without_candidate[:, :, 3] = 0.0
        with torch.no_grad():
            normal = self.extractor.encode_grids(grids)
            baseline = self.extractor.encode_grids(without_candidate)
        sensitivity = torch.linalg.vector_norm(
            normal - baseline, dim=-1
        ).mean()
        return {
            "workspace_feature_variance": float(
                normal.var(dim=1, unbiased=False).mean().item()
            ),
            "candidate_channel_sensitivity": float(sensitivity.item()),
        }

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


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
        ("diagnostics/cnn_gradient_norm", "cnn_gradient_norm"),
        ("diagnostics/cnn_weight_change", "cnn_weight_change"),
        (
            "diagnostics/workspace_feature_variance",
            "workspace_feature_variance",
        ),
        (
            "diagnostics/candidate_channel_sensitivity",
            "candidate_channel_sensitivity",
        ),
    ]

    def __init__(self, log_dir: str, append: bool = False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.log_dir / "loss_log.csv"
        append_existing = (
            append
            and self._csv_path.is_file()
            and self._csv_path.stat().st_size > 0
        )
        if append_existing:
            read_curve_log(
                self._csv_path,
                "loss_log",
                repair_trailing_partial=True,
            )
        expected_header = [
            "timestep", *[column for _, column in self.METRIC_COLUMNS]
        ]
        if append_existing:
            with self._csv_path.open(
                encoding="utf-8", newline=""
            ) as existing_file:
                existing_header = next(csv.reader(existing_file), [])
            if existing_header != expected_header:
                raise ValueError(
                    "Cannot append loss_log.csv with an incompatible header."
                )

        mode = "a" if append_existing else "w"
        self._csv_file = open(
            self._csv_path, mode, newline="", encoding="utf-8"
        )
        self._csv_writer = csv.writer(self._csv_file)
        if not append_existing:
            self._csv_writer.writerow(expected_header)

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

    def __init__(
        self,
        log_dir: str = "./output",
        verbose: int = 1,
        append: bool = False,
    ):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.append = append
        self._writer: Optional[TrainingMetricsCsvWriter] = None

    def _on_training_start(self) -> None:
        self._writer = TrainingMetricsCsvWriter(
            self.log_dir, append=self.append
        )
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
    - resolved_reward: 블록별 확정 결과를 exactly-once 합산한 보상
    - terminal_residual: 확정 보상 합을 최종 점수에 맞추는 잔차
    - terminal_score: 최종 배치 품질 점수
    - episode_reward: 실제 에피소드 보상 합
    - delayed_count: 지연 블록 수
    - dropout_count: 탈락 블록 수
    - total_delay_days: 총 지연 일수 합
    - success_rate: 정상 배치(준수) 비율
    """

    CSV_COLUMNS = [
        "episode",
        "timestep",
        "resolved_reward",
        "terminal_residual",
        "terminal_score",
        "episode_reward",
        "delayed_count",
        "dropout_count",
        "total_delay_days",
        "success_rate",
    ]

    def __init__(
        self,
        log_dir: str = "./output",
        verbose: int = 1,
        append: bool = False,
    ):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # CSV 로그
        self._csv_path = self.log_dir / "training_log.csv"
        self._csv_file = None
        self._csv_writer = None
        self._append = append

        # 에피소드 카운터
        self._episode_count = 0
        self._diagnostic_tracker: Optional[CnnDiagnosticTracker] = None
        self._diagnostic_observation: Optional[Dict[str, np.ndarray]] = None
        self._rollout_count = 0

        # 지표 히스토리 (학습 후 시각화용)
        self.history: Dict[str, List[float]] = {
            "episode": [],
            "resolved_reward": [],
            "terminal_residual": [],
            "terminal_score": [],
            "episode_reward": [],
            "delayed_count": [],
            "dropout_count": [],
            "total_delay_days": [],
            "success_rate": [],
            "timestep": [],
        }

    def _on_training_start(self):
        append_existing = (
            self._append
            and self._csv_path.is_file()
            and self._csv_path.stat().st_size > 0
        )
        if append_existing:
            read_curve_log(
                self._csv_path,
                "training_log",
                repair_trailing_partial=True,
            )
        if append_existing:
            with self._csv_path.open(
                encoding="utf-8", newline=""
            ) as existing_file:
                reader = csv.DictReader(existing_file)
                if reader.fieldnames != self.CSV_COLUMNS:
                    raise ValueError(
                        "Cannot append training_log.csv with an "
                        "incompatible header."
                    )
                for row in reader:
                    try:
                        self._episode_count = max(
                            self._episode_count, int(row["episode"])
                        )
                    except (KeyError, TypeError, ValueError):
                        continue

        mode = "a" if append_existing else "w"
        self._csv_file = open(
            self._csv_path, mode, newline="", encoding="utf-8"
        )
        self._csv_writer = csv.writer(self._csv_file)
        if not append_existing:
            self._csv_writer.writerow(self.CSV_COLUMNS)

        model = getattr(self, "model", None)
        policy = getattr(model, "policy", None)
        extractor = getattr(policy, "features_extractor", None)
        if extractor is None:
            extractor = getattr(policy, "pi_features_extractor", None)
        if extractor is not None:
            self._diagnostic_tracker = CnnDiagnosticTracker(extractor)
            self._diagnostic_tracker.attach()
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
            resolved_reward = info.get("resolved_reward", 0.0)
            terminal_residual = info.get("terminal_residual", 0.0)
            terminal_score = info.get(
                "terminal_score", info.get("terminal_reward", 0.0)
            )
            episode_reward = info.get(
                "episode_reward", terminal_score
            )
            self._log_episode(
                raw_result,
                resolved_reward,
                terminal_residual,
                terminal_score,
                episode_reward,
            )

        return True

    def _on_rollout_end(self) -> None:
        self._diagnostic_observation = None
        tracker = self._diagnostic_tracker
        if tracker is None or tracker.module is None:
            return
        latest_observation = getattr(self.model, "_last_obs", None)
        if not isinstance(latest_observation, dict):
            return
        self._diagnostic_observation = {
            key: np.array(value, copy=True)
            for key, value in latest_observation.items()
        }

    def _on_rollout_start(self) -> None:
        tracker = self._diagnostic_tracker
        if tracker is None:
            self._diagnostic_observation = None
            return

        try:
            if self._rollout_count > 0:
                for key, value in tracker.record_update().items():
                    self.logger.record(f"diagnostics/{key}", value)

                if self._diagnostic_observation is not None:
                    obs_tensor, _ = self.model.policy.obs_to_tensor(
                        self._diagnostic_observation
                    )
                    for key, value in tracker.measure_features(
                        obs_tensor
                    ).items():
                        self.logger.record(f"diagnostics/{key}", value)
        finally:
            self._diagnostic_observation = None
            self._rollout_count += 1

    def _log_episode(
        self,
        result: SimulationResult,
        resolved_reward: float,
        terminal_residual: float,
        terminal_score: float,
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
        self.logger.record("alloc/resolved_reward", resolved_reward)
        self.logger.record("alloc/terminal_residual", terminal_residual)
        self.logger.record("alloc/terminal_score", terminal_score)
        self.logger.record("alloc/episode_reward", episode_reward)
        self.logger.record("alloc/delayed_count", delayed_count)
        self.logger.record("alloc/dropout_count", dropout_count)
        self.logger.record("alloc/total_delay_days", total_delay)
        self.logger.record("alloc/success_rate", success_rate)
        self.logger.record("alloc/episode", self._episode_count)

        # 히스토리 저장
        self.history["episode"].append(self._episode_count)
        self.history["timestep"].append(self.num_timesteps)
        self.history["resolved_reward"].append(resolved_reward)
        self.history["terminal_residual"].append(terminal_residual)
        self.history["terminal_score"].append(terminal_score)
        self.history["episode_reward"].append(episode_reward)
        self.history["delayed_count"].append(delayed_count)
        self.history["dropout_count"].append(dropout_count)
        self.history["total_delay_days"].append(total_delay)
        self.history["success_rate"].append(success_rate)

        # CSV 기록
        if self._csv_writer:
            self._csv_writer.writerow([
                self._episode_count,
                self.num_timesteps,
                f"{resolved_reward:.4f}",
                f"{terminal_residual:.4f}",
                f"{terminal_score:.4f}",
                f"{episode_reward:.4f}",
                delayed_count, dropout_count, total_delay, f"{success_rate:.4f}",
            ])
            self._csv_file.flush()

        # 콘솔 출력 (10 에피소드마다)
        if self.verbose and self._episode_count % 10 == 0:
            print(f"  [EP {self._episode_count:>4}] "
                  f"score={terminal_score:>+7.3f}  "
                  f"episode={episode_reward:>+7.3f}  "
                  f"delay={delayed_count}  dropout={dropout_count}  "
                  f"success={success_rate:.1%}")

    def _on_training_end(self):
        self._diagnostic_observation = None
        if self._diagnostic_tracker is not None:
            self._diagnostic_tracker.close()
        if self._csv_file:
            self._csv_file.close()
        if self.verbose:
            print(f"[Callback] 총 {self._episode_count} 에피소드 학습 완료")
            print(f"[Callback] CSV 로그 저장: {self._csv_path}")
