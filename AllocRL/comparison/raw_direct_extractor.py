from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from alloc_env.cnn_extractor import validate_observation_space


RAW_DIRECT_KEYS = (
    "block",
    "ws_meta",
    "future_blocks",
    "future_mask",
    "future_demand",
    "pending_blocks",
    "pending_mask",
    "pending_summary",
)
RAW_DIRECT_FEATURE_DIM = 2818


class RawDirectExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
    ):
        validate_observation_space(observation_space)
        _ = features_dim
        super().__init__(observation_space, RAW_DIRECT_FEATURE_DIM)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        future_mask = observations["future_mask"].to(
            dtype=observations["future_blocks"].dtype
        )
        pending_mask = observations["pending_mask"].to(
            dtype=observations["pending_blocks"].dtype
        )
        parts = (
            observations["block"],
            observations["ws_meta"].flatten(1),
            (observations["future_blocks"] * future_mask.unsqueeze(-1)).flatten(1),
            future_mask.flatten(1),
            observations["future_demand"].flatten(1),
            (observations["pending_blocks"] * pending_mask.unsqueeze(-1)).flatten(1),
            pending_mask.flatten(1),
            observations["pending_summary"].flatten(1),
        )
        output = torch.cat(parts, dim=1)
        if output.shape[1] != RAW_DIRECT_FEATURE_DIM:
            raise RuntimeError(
                f"raw-direct feature width must be {RAW_DIRECT_FEATURE_DIM}, "
                f"got {output.shape[1]}"
            )
        return output
