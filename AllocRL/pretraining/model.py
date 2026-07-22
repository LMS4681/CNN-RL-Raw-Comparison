from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from alloc_env.cnn_extractor import CandidateCnnExtractor


@dataclass(frozen=True)
class AuxiliaryPredictions:
    current_placeable: torch.Tensor
    future_fit: torch.Tensor
    future_optionality_after: torch.Tensor
    future_optionality_delta: torch.Tensor
    largest_free_rectangle_ratio: torch.Tensor
    free_component_count_normalized: torch.Tensor
    replay_success_rate: torch.Tensor
    replay_dropout_rate: torch.Tensor
    replay_delay_ratio: torch.Tensor


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = mask.to(dtype=values.dtype)
    while expanded_mask.ndim < values.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    expanded_mask = expanded_mask.expand_as(values)
    denominator = expanded_mask.sum()
    if denominator.item() == 0:
        return values.sum() * 0.0
    return (values * expanded_mask).sum() / denominator


class CandidatePretrainingModel(nn.Module):
    """Temporary supervised heads over the deployable candidate extractor."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
    ) -> None:
        super().__init__()
        self.extractor = CandidateCnnExtractor(
            observation_space, features_dim=features_dim
        )
        context_dim = 64 + features_dim
        self.heads = nn.ModuleDict({
            "current_placeable": nn.Linear(context_dim, 1),
            "future_fit": nn.Linear(context_dim, 16),
            "future_optionality_after": nn.Linear(context_dim, 1),
            "future_optionality_delta": nn.Linear(context_dim, 1),
            "largest_free_rectangle_ratio": nn.Linear(context_dim, 1),
            "free_component_count_normalized": nn.Linear(context_dim, 1),
            "replay_success_rate": nn.Linear(context_dim, 1),
            "replay_dropout_rate": nn.Linear(context_dim, 1),
            "replay_delay_ratio": nn.Linear(context_dim, 1),
        })

    def forward(
        self,
        observations: dict[str, torch.Tensor],
    ) -> AuxiliaryPredictions:
        workspace_features = self.extractor.encode_workspace_features(
            observations
        )
        global_features = self.extractor.global_fusion(
            workspace_features.flatten(1)
        )
        expanded_global = global_features.unsqueeze(1).expand(
            -1, self.extractor.n_workspaces, -1
        )
        context = torch.cat(
            [workspace_features, expanded_global], dim=-1
        )

        def sigmoid(name: str) -> torch.Tensor:
            return torch.sigmoid(self.heads[name](context).squeeze(-1))

        return AuxiliaryPredictions(
            current_placeable=sigmoid("current_placeable"),
            future_fit=torch.sigmoid(self.heads["future_fit"](context)),
            future_optionality_after=sigmoid(
                "future_optionality_after"
            ),
            future_optionality_delta=torch.tanh(
                self.heads["future_optionality_delta"](context).squeeze(-1)
            ),
            largest_free_rectangle_ratio=sigmoid(
                "largest_free_rectangle_ratio"
            ),
            free_component_count_normalized=sigmoid(
                "free_component_count_normalized"
            ),
            replay_success_rate=sigmoid("replay_success_rate"),
            replay_dropout_rate=sigmoid("replay_dropout_rate"),
            replay_delay_ratio=sigmoid("replay_delay_ratio"),
        )

    def loss_components(
        self,
        predictions: AuxiliaryPredictions,
        targets: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        action_mask = targets["action_mask"].bool()
        replay_mask = targets["replay_mask"].bool() & action_mask

        current = _masked_mean(
            F.binary_cross_entropy(
                predictions.current_placeable,
                targets["current_placeable"],
                reduction="none",
            ),
            action_mask,
        )
        future_fit = _masked_mean(
            F.binary_cross_entropy(
                predictions.future_fit,
                targets["future_fit"],
                reduction="none",
            ),
            action_mask,
        )

        def regression(name: str, mask: torch.Tensor) -> torch.Tensor:
            return _masked_mean(
                F.smooth_l1_loss(
                    getattr(predictions, name),
                    targets[name],
                    reduction="none",
                ),
                mask,
            )

        optionality = (
            regression("future_optionality_after", action_mask)
            + regression("future_optionality_delta", action_mask)
        ) / 2.0
        geometry = (
            regression("largest_free_rectangle_ratio", action_mask)
            + regression("free_component_count_normalized", action_mask)
        ) / 2.0
        replay = (
            regression("replay_success_rate", replay_mask)
            + regression("replay_dropout_rate", replay_mask)
            + regression("replay_delay_ratio", replay_mask)
        ) / 3.0
        return {
            "current_placeable": current,
            "future_fit": future_fit,
            "future_optionality": optionality,
            "geometry": geometry,
            "replay": replay,
        }

    def loss(
        self,
        predictions: AuxiliaryPredictions,
        targets: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        components = self.loss_components(predictions, targets)
        return (
            0.25 * components["current_placeable"]
            + components["future_fit"]
            + components["future_optionality"]
            + 0.5 * components["geometry"]
            + components["replay"]
        )
