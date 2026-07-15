"""Feature extractors for ordered block context and candidate grids."""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class OrderedBlockEncoder(nn.Module):
    """Encode the current block and the exact ordered future sequence."""

    output_dim = 96

    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__()
        block_dim = observation_space["block"].shape[0]
        self.current = nn.Sequential(
            nn.Linear(block_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.k = (
            observation_space["future_blocks"].shape[0]
            if "future_blocks" in observation_space.spaces
            else 0
        )
        if self.k:
            future_dim = observation_space["future_blocks"].shape[1]
            self.future: Optional[nn.Module] = nn.Sequential(
                nn.Linear(self.k * (future_dim + 1), 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
        else:
            self.future = None

    def forward(
        self,
        observations: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        current = self.current(observations["block"])
        if self.future is None:
            future = current.new_zeros((current.shape[0], 64))
        else:
            mask = observations["future_mask"]
            masked = observations["future_blocks"] * mask.unsqueeze(-1)
            future_input = torch.cat(
                [masked.flatten(1), mask.flatten(1)], dim=1
            )
            future = self.future(future_input)
        return torch.cat([current, future], dim=1)


class _WorkspaceExtractor(BaseFeaturesExtractor):
    """Fuse ordered block context with one feature vector per workspace."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int,
        grid_feature_dim: int,
    ):
        super().__init__(observation_space, features_dim)
        self.n_workspaces = observation_space["grids"].shape[0]
        ws_meta_dim = observation_space["ws_meta"].shape[1]
        self.block_encoder = OrderedBlockEncoder(observation_space)

        workspace_input_dim = (
            OrderedBlockEncoder.output_dim
            + ws_meta_dim
            + grid_feature_dim
        )
        self.workspace_fusion = nn.Sequential(
            nn.Linear(workspace_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.global_fusion = nn.Sequential(
            nn.Linear(self.n_workspaces * 64, features_dim),
            nn.ReLU(),
        )

    def _grid_features(
        self,
        observations: dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        return None

    def forward(
        self,
        observations: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        context = self.block_encoder(observations)
        workspace_context = context.unsqueeze(1).expand(
            -1, self.n_workspaces, -1
        )
        inputs = [workspace_context, observations["ws_meta"]]
        grid_features = self._grid_features(observations)
        if grid_features is not None:
            inputs.append(grid_features)

        workspace_features = self.workspace_fusion(
            torch.cat(inputs, dim=-1)
        )
        return self.global_fusion(workspace_features.flatten(1))


class StructuredExtractor(_WorkspaceExtractor):
    """Structured-only baseline that deliberately ignores all grid pixels."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim, grid_feature_dim=0)


class FixedGridExtractor(_WorkspaceExtractor):
    """Non-learned fixed-grid baseline with no convolutional parameters."""

    pooled_size = 8

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
    ):
        grid_channels = observation_space["grids"].shape[1]
        if grid_channels != 4:
            raise ValueError(
                f"FixedGridExtractor requires 4 grid channels, got {grid_channels}."
            )
        grid_feature_dim = grid_channels * self.pooled_size * self.pooled_size
        super().__init__(
            observation_space,
            features_dim,
            grid_feature_dim=grid_feature_dim,
        )

    def _grid_features(
        self,
        observations: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        grids = observations["grids"]
        batch_size, n_workspaces, channels, height, width = grids.shape
        pooled = F.adaptive_avg_pool2d(
            grids.reshape(
                batch_size * n_workspaces, channels, height, width
            ),
            (self.pooled_size, self.pooled_size),
        )
        return pooled.reshape(batch_size, n_workspaces, -1)


class CandidateCnnExtractor(_WorkspaceExtractor):
    """Learned shared CNN over occupancy, boundary, and candidate channels."""

    image_feature_dim = 128

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
    ):
        grid_shape = observation_space["grids"].shape
        grid_channels = grid_shape[1]
        if grid_channels != 4:
            raise ValueError(
                f"CandidateCnnExtractor requires 4 grid channels, got {grid_channels}."
            )

        # Integral fixed-grid resize matches adaptive 8x8 pooling while
        # remaining exportable with a dynamic ONNX batch axis.
        conv_height = (grid_shape[-2] + 3) // 4
        conv_width = (grid_shape[-1] + 3) // 4
        if (
            conv_height >= 8
            and conv_width >= 8
            and conv_height % 8 == 0
            and conv_width % 8 == 0
        ):
            pool_kernel = (conv_height // 8, conv_width // 8)
            spatial_pool = nn.AvgPool2d(
                kernel_size=pool_kernel, stride=pool_kernel
            )
        elif (
            conv_height <= 8
            and conv_width <= 8
            and 8 % conv_height == 0
            and 8 % conv_width == 0
        ):
            spatial_pool = nn.Upsample(size=(8, 8), mode="nearest")
        else:
            raise ValueError(
                "CandidateCnnExtractor grid dimensions must produce CNN "
                "feature maps that divide or are divisible by 8x8."
            )

        super().__init__(
            observation_space,
            features_dim,
            grid_feature_dim=self.image_feature_dim,
        )
        self.image_encoder = nn.Sequential(
            nn.Conv2d(4, 32, 5, stride=1, padding=2),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            spatial_pool,
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, self.image_feature_dim),
            nn.ReLU(),
        )

    def encode_grids(self, grids: torch.Tensor) -> torch.Tensor:
        batch_size, n_workspaces, channels, height, width = grids.shape
        encoded = self.image_encoder(
            grids.reshape(
                batch_size * n_workspaces, channels, height, width
            )
        )
        return encoded.reshape(
            batch_size, n_workspaces, self.image_feature_dim
        )

    def _grid_features(
        self,
        observations: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.encode_grids(observations["grids"])
