"""MaskablePPO variant for frozen then low-rate extractor fine-tuning."""

from __future__ import annotations

import math
from typing import Any

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from alloc_env.cnn_extractor import CandidateCnnExtractor


class ScaleAwareMaskablePPO(MaskablePPO):
    def __init__(
        self,
        *args: Any,
        extractor_lr_scale: float = 0.1,
        **kwargs: Any,
    ) -> None:
        if (
            not math.isfinite(extractor_lr_scale)
            or not 0.0 < extractor_lr_scale <= 1.0
        ):
            raise ValueError("extractor_lr_scale must be in (0, 1]")
        self.extractor_lr_scale = float(extractor_lr_scale)
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        extractor = self.policy.features_extractor
        if not isinstance(extractor, CandidateCnnExtractor):
            raise TypeError(
                "ScaleAwareMaskablePPO requires CandidateCnnExtractor"
            )
        extractor_ids = {id(parameter) for parameter in extractor.parameters()}
        extractor_parameters = list(extractor.parameters())
        policy_parameters = [
            parameter
            for parameter in self.policy.parameters()
            if id(parameter) not in extractor_ids
        ]
        if not extractor_parameters or not policy_parameters:
            raise ValueError("policy and extractor optimizer groups must be nonempty")
        base_rate = float(self.lr_schedule(1.0))
        self.policy.optimizer = self.policy.optimizer_class(
            [
                {
                    "params": policy_parameters,
                    "name": "policy",
                    "lr_scale": 1.0,
                    "lr": base_rate,
                },
                {
                    "params": extractor_parameters,
                    "name": "extractor",
                    "lr_scale": self.extractor_lr_scale,
                    "lr": base_rate * self.extractor_lr_scale,
                },
            ],
            lr=base_rate,
            **self.policy.optimizer_kwargs,
        )

    def _update_learning_rate(
        self,
        optimizers: list[torch.optim.Optimizer] | torch.optim.Optimizer,
    ) -> None:
        base_rate = float(
            self.lr_schedule(self._current_progress_remaining)
        )
        self.logger.record("train/learning_rate", base_rate)
        self.logger.record("train/policy_learning_rate", base_rate)
        self.logger.record(
            "train/extractor_learning_rate",
            base_rate * self.extractor_lr_scale,
        )
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                scale = float(group.get("lr_scale", 1.0))
                group["lr"] = base_rate * scale


class ExtractorFineTuneCallback(BaseCallback):
    def __init__(self, freeze_until_timestep: int, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        if (
            isinstance(freeze_until_timestep, bool)
            or not isinstance(freeze_until_timestep, int)
            or freeze_until_timestep < 0
        ):
            raise ValueError("freeze_until_timestep must be non-negative")
        self.freeze_until_timestep = freeze_until_timestep
        self._frozen: bool | None = None

    def _extractor(self) -> CandidateCnnExtractor:
        extractor = self.model.policy.features_extractor
        if not isinstance(extractor, CandidateCnnExtractor):
            raise TypeError(
                "extractor fine-tuning requires CandidateCnnExtractor"
            )
        return extractor

    def _apply_state(self) -> None:
        frozen = int(self.model.num_timesteps) < self.freeze_until_timestep
        if frozen != self._frozen:
            extractor = self._extractor()
            extractor.requires_grad_(not frozen)
            if frozen:
                for parameter in extractor.parameters():
                    parameter.grad = None
            self._frozen = frozen
        self.logger.record("diagnostics/extractor_frozen", float(frozen))

    def _on_training_start(self) -> None:
        self._apply_state()

    def _on_step(self) -> bool:
        self._apply_state()
        return True

    def _on_rollout_end(self) -> None:
        self._apply_state()

