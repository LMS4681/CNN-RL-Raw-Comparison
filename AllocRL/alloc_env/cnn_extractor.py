"""Feature extractors for corrected structured state and candidate grids."""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


N_WORKSPACES = 10
EXPECTED_OBSERVATION_SHAPES = {
    "block": (8,),
    "ws_meta": (N_WORKSPACES, 4),
    "future_blocks": (16, 6),
    "future_mask": (16,),
    "future_demand": (3, 4),
    "pending_blocks": (N_WORKSPACES, 32, 7),
    "pending_mask": (N_WORKSPACES, 32),
    "pending_summary": (N_WORKSPACES, 4),
}
EXPECTED_OBSERVATION_KEYS = {
    *EXPECTED_OBSERVATION_SHAPES,
    "grids",
}


def _validate_observation_space(
    observation_space: gym.spaces.Dict,
) -> None:
    actual_keys = set(observation_space.spaces)
    missing = sorted(EXPECTED_OBSERVATION_KEYS - actual_keys)
    unexpected = sorted(actual_keys - EXPECTED_OBSERVATION_KEYS)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected {', '.join(unexpected)}")
        raise ValueError(
            "Schema-3 extractors require exactly nine keys: "
            + "; ".join(details)
        )

    for key, expected_shape in EXPECTED_OBSERVATION_SHAPES.items():
        actual_shape = observation_space[key].shape
        if actual_shape != expected_shape:
            raise ValueError(
                f"Schema-3 {key} must have shape {expected_shape}, "
                f"got {actual_shape}."
            )

    grid_shape = observation_space["grids"].shape
    if (
        len(grid_shape) != 4
        or grid_shape[:2] != (N_WORKSPACES, 4)
        or grid_shape[2] < 1
        or grid_shape[3] < 1
    ):
        raise ValueError(
            "Schema-3 grids must have shape "
            f"({N_WORKSPACES}, 4, height, width) with positive spatial "
            f"dimensions, got {grid_shape}."
        )


class StructuredStateEncoder(nn.Module):
    """Encode every corrected structured observation without pooling order."""

    def __init__(self):
        super().__init__()
        self.current = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.future = nn.Sequential(
            nn.Linear(16 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.demand = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.pending = nn.Sequential(
            nn.Linear(32 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(
        self,
        observations: dict[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        current = self.current(observations["block"])

        future_blocks = observations["future_blocks"]
        future_mask = observations["future_mask"].to(
            dtype=future_blocks.dtype
        )
        future_input = torch.cat(
            [
                future_blocks * future_mask.unsqueeze(-1),
                future_mask.unsqueeze(-1),
            ],
            dim=-1,
        ).flatten(1)
        future = self.future(future_input)

        demand = self.demand(observations["future_demand"].flatten(1))

        pending_blocks = observations["pending_blocks"]
        pending_mask = observations["pending_mask"].to(
            dtype=pending_blocks.dtype
        )
        pending_input = torch.cat(
            [
                pending_blocks * pending_mask.unsqueeze(-1),
                pending_mask.unsqueeze(-1),
            ],
            dim=-1,
        ).flatten(2)
        pending = self.pending(pending_input)
        return current, future, demand, pending


class _WorkspaceExtractor(BaseFeaturesExtractor):
    """Fuse corrected state with one feature vector per action workspace."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int,
        grid_feature_dim: int,
    ):
        _validate_observation_space(observation_space)
        super().__init__(observation_space, features_dim)
        self.n_workspaces = N_WORKSPACES
        self.structured_encoder = StructuredStateEncoder()

        workspace_input_dim = (
            32
            + 64
            + 32
            + 64
            + 4
            + 4
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
        current, future, demand, pending = self.structured_encoder(
            observations
        )
        inputs = [
            current.unsqueeze(1).expand(-1, self.n_workspaces, -1),
            future.unsqueeze(1).expand(-1, self.n_workspaces, -1),
            demand.unsqueeze(1).expand(-1, self.n_workspaces, -1),
            pending,
            observations["pending_summary"],
            observations["ws_meta"],
        ]
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
