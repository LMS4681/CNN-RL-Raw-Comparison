"""
CNN+MLP 하이브리드 Feature Extractor.

SB3 MultiInputPolicy와 호환되는 커스텀 Feature Extractor.

구조:
  obs["block"]   → MLP      → block_feat   (64,)
  obs["grids"]   → SharedCNN → ws_cnn_feats (N × cnn_out_dim)
  obs["ws_meta"] → Flatten   → ws_meta_flat (N × 3)

  concat → FusionLinear → features_dim (256)

핵심:
  - CNN 가중치 공유: 모든 작업장 그리드에 동일 CNN 적용
  - Skip Connection: 다중 스케일 패턴 인식 (극단 종횡비 대응)
  - 현재 블록 스케일 인식: block 피처에 scale 정보 포함
"""

from __future__ import annotations

import math

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SharedCNN(nn.Module):
    """
    단일 작업장 그리드 → 특징 벡터 추출 CNN.

    Skip Connection 기반 Multi-Scale 인식:
      - conv1 (큰 수용장): 전체 배치 패턴
      - conv2 (중간):     블록 군집
      - conv3 (작은):     개별 빈 공간
      → skip connection으로 3개 스케일 특징 결합

    Input:  (batch, 3, G, G)  where G = grid_size (default: 64)
    Output: (batch, cnn_out_dim)
    """

    def __init__(self, in_channels: int = 3, cnn_out_dim: int = 64):
        super().__init__()

        # ── 인코더 ────────────────────────────────────────────────
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )  # (32, 64, 64)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )  # (64, 32, 32)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )  # (64, 16, 16)

        # ── Skip Connection 기반 Multi-Scale 풀링 ─────────────────
        self.pool1 = nn.AdaptiveAvgPool2d((4, 4))  # conv1 → (32, 4, 4)
        self.pool2 = nn.AdaptiveAvgPool2d((4, 4))  # conv2 → (64, 4, 4)
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))  # conv3 → (64, 4, 4)

        # 32*4*4 + 64*4*4 + 64*4*4 = 512 + 1024 + 1024 = 2560
        skip_dim = (32 + 64 + 64) * 4 * 4  # 2560

        # ── 출력 FC ───────────────────────────────────────────────
        self.fc = nn.Sequential(
            nn.Linear(skip_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, cnn_out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, 128, 128)
        Returns:
            (batch, cnn_out_dim)
        """
        f1 = self.conv1(x)   # (batch, 32, 64, 64)
        f2 = self.conv2(f1)  # (batch, 64, 32, 32)
        f3 = self.conv3(f2)  # (batch, 64, 16, 16)

        # Skip connection: 3개 스케일 특징 concat
        p1 = self.pool1(f1).flatten(1)  # (batch, 512)
        p2 = self.pool2(f2).flatten(1)  # (batch, 1024)
        p3 = self.pool3(f3).flatten(1)  # (batch, 1024)

        combined = torch.cat([p1, p2, p3], dim=1)  # (batch, 2560)
        return self.fc(combined)  # (batch, cnn_out_dim)


class OccupancyCnnExtractor(BaseFeaturesExtractor):
    """
    Dict 관측 공간용 CNN+MLP 하이브리드 Feature Extractor.

    SB3 MultiInputPolicy에서 policy_kwargs를 통해 주입.

    관측 공간 구조:
      "block"   : Box(block_dim,)            ← 블록 속성 + 시간 + 스케일
      "grids"   : Box(N, 3, 128, 128)        ← 작업장별 3채널 그리드
      "ws_meta" : Box(N, 3)                  ← 작업장별 (scale, occupancy_ratio, placeable_now)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        cnn_out_dim: int = 64,
    ):
        # BaseFeaturesExtractor는 features_dim을 받아야 함
        super().__init__(observation_space, features_dim)

        block_dim = observation_space["block"].shape[0]
        n_workspaces = observation_space["grids"].shape[0]
        ws_meta_dim = observation_space["ws_meta"].shape[0] * observation_space["ws_meta"].shape[1]

        self._n_workspaces = n_workspaces
        self._cnn_out_dim = cnn_out_dim

        # ── Block MLP ─────────────────────────────────────────────
        self.block_mlp = nn.Sequential(
            nn.Linear(block_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        # ── Shared CNN (가중치 공유) ──────────────────────────────
        self.shared_cnn = SharedCNN(
            in_channels=3,
            cnn_out_dim=cnn_out_dim,
        )

        # ── Fusion Layer ──────────────────────────────────────────
        # block_feat(64) + ws_cnn(N*cnn_out_dim) + ws_meta(N*3)
        fusion_in = 64 + n_workspaces * cnn_out_dim + ws_meta_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        """
        Args:
            observations: Dict with "block", "grids", "ws_meta"
        Returns:
            (batch, features_dim) 특징 벡터
        """
        batch_size = observations["block"].shape[0]

        # 1. Block features → MLP
        block_feat = self.block_mlp(observations["block"])  # (B, 64)

        # 2. Workspace grids → Shared CNN
        grids = observations["grids"]  # (B, N, 3, 128, 128)
        N = self._n_workspaces

        # N개 작업장을 batch 차원으로 reshape하여 한 번에 처리
        grids_flat = grids.reshape(batch_size * N, 3,
                                   grids.shape[3], grids.shape[4])
        cnn_out = self.shared_cnn(grids_flat)  # (B*N, cnn_out_dim)
        ws_cnn_feats = cnn_out.reshape(batch_size, N * self._cnn_out_dim)

        # 3. Workspace metadata → flatten
        ws_meta_flat = observations["ws_meta"].reshape(batch_size, -1)

        # 4. Fusion
        combined = torch.cat([block_feat, ws_cnn_feats, ws_meta_flat], dim=1)
        return self.fusion(combined)  # (B, features_dim)


class PointerAttentionCnnExtractor(BaseFeaturesExtractor):
    """
    CNN workspace encoder with workspace self-attention and pointer-style scores.

    The observation contract is unchanged:
      - block: current block features
      - grids: workspace occupancy grids
      - ws_meta: workspace metadata

    This extractor keeps one token per workspace, lets workspaces attend to each
    other, then scores each workspace token against the current block embedding.
    The final vector is still a standard SB3 feature vector, so MaskablePPO and
    action masking continue to work without a custom policy class.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        cnn_out_dim: int = 64,
        embed_dim: int = 64,
        num_heads: int = 4,
    ):
        super().__init__(observation_space, features_dim)

        block_dim = observation_space["block"].shape[0]
        n_workspaces = observation_space["grids"].shape[0]
        ws_meta_dim = observation_space["ws_meta"].shape[1]

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self._n_workspaces = n_workspaces
        self._embed_dim = embed_dim

        self.block_encoder = nn.Sequential(
            nn.Linear(block_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        self.shared_cnn = SharedCNN(
            in_channels=3,
            cnn_out_dim=cnn_out_dim,
        )
        self.workspace_encoder = nn.Sequential(
            nn.Linear(cnn_out_dim + ws_meta_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        self.workspace_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.context_layer = nn.Sequential(
            nn.Linear(embed_dim * 3 + n_workspaces, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        batch_size = observations["block"].shape[0]
        grids = observations["grids"]
        ws_meta = observations["ws_meta"]
        n_workspaces = self._n_workspaces

        block_query = self.block_encoder(observations["block"])

        grids_flat = grids.reshape(
            batch_size * n_workspaces,
            3,
            grids.shape[3],
            grids.shape[4],
        )
        cnn_out = self.shared_cnn(grids_flat)
        cnn_tokens = cnn_out.reshape(batch_size, n_workspaces, -1)
        workspace_tokens = self.workspace_encoder(
            torch.cat([cnn_tokens, ws_meta], dim=-1)
        )

        attended, _ = self.workspace_attention(
            workspace_tokens,
            workspace_tokens,
            workspace_tokens,
            need_weights=False,
        )
        workspace_tokens = self.attention_norm(workspace_tokens + attended)

        keys = self.key_layer(workspace_tokens)
        pointer_scores = (
            keys * block_query.unsqueeze(1)
        ).sum(dim=-1) / math.sqrt(self._embed_dim)
        pointer_weights = torch.softmax(pointer_scores, dim=1).unsqueeze(-1)
        pointer_context = (workspace_tokens * pointer_weights).sum(dim=1)
        pooled_context = workspace_tokens.mean(dim=1)

        combined = torch.cat(
            [block_query, pointer_context, pooled_context, pointer_scores],
            dim=1,
        )
        return self.context_layer(combined)
