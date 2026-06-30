"""
블록 배치 강화학습 Gymnasium 환경 (CNN 관측 버전).

- Action:  Discrete(num_workspaces) - 현재 블록을 어느 작업장에 배정
- Obs:     Dict {
              "block":   블록 속성 + 시간 + 스케일 정보,
              "grids":   작업장별 3채널 점유 그리드 (N, 3, 128, 128),
              "ws_meta": 작업장별 메타데이터 (N, 2),
           }
- Reward:  Terminal + shaped reward (즉시 배치 가능성 + 부분 replay)
- Mask:    하드 제약 위반 작업장 마스킹 (sb3-contrib MaskablePPO 호환)

CNN 관측 핵심:
  - 3채널: 점유 마스크 / 잔여 출고 공기 / 작업장 경계
  - 비율 유지 리사이즈: 모든 작업장이 128×128 그리드에 수용
  - 그리드 캐싱: 변경된 작업장만 재렌더링
  - 작업장 레이아웃 합성: 매 에피소드마다 크기 변형
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from datetime import date
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple

from .block import Block
from .workspace import Workspace
from .constraints import (
    BlockPatternConstraint,
    DimensionConstraint,
    ValidWorkspacePicker,
)
from .incremental_simulator import (
    IncrementalPlacementSimulator,
    PlacementStepResult,
)
from .simulator import SimulationResult
from .strategy import BaseGridStrategy
from .occupancy_grid import OccupancyGridRenderer, GridCache, GRID_SIZE

# C# AllocConst 대응
DELAY_THRESHOLD = 2       # 준수(compliance) 기준: 지연 <= 2일
DROPOUT_THRESHOLD = 7     # 탈락(dropout) 기준: 지연 > 7일

# ── Reward 설정 ──────────────────────────────────────────────────
REWARD_COMPLIANT = 1.0     # 준수(지연 <= 2일)
REWARD_DROPOUT = -2.0      # 탈락(지연 > 7일)
SHAPING_PLACEMENT_SUCCESS = 0.0002
SHAPING_PLACEMENT_FAILURE = -0.002
PARTIAL_REPLAY_WEIGHT = 0.1
PARTIAL_REPLAY_INTERVAL = 8


class BlockPlacementEnv(gym.Env):
    """
    블록 배치 강화학습 환경 (CNN 관측 버전).

    에이전트는 각 블록을 하나씩 순차적으로 받고,
    3채널 점유 그리드를 통해 작업장 공간 상태를 인식한 후
    어느 작업장에 배정할지 결정합니다.

    Reward:
      - 전체 블록 배정 완료 후 시뮬레이션 실행
      - 블록별 준수(+1) / 지연(비례 감점) / 탈락(-2)
      - 합산 후 블록 수로 정규화 → [-2.0, +1.0]
      - 중간 단계는 즉시 배치 가능성 및 simulator potential delta로 shaping

    Observation: Dict
      "block"  : (10,)           - 블록 물리 속성 5 + 시간 3 + 진행률 + 블록 스케일
      "grids"  : (N, 3, 128, 128) - 작업장별 3채널 점유 그리드
      "ws_meta": (N, 2)          - 작업장별 (scale, occupancy_ratio)

    use_synthetic=True일 때 매 에피소드마다 랜덤 블록 + 레이아웃 변형.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        blocks: List[Block],
        workspaces: List[Workspace],
        strategy: Optional[BaseGridStrategy] = None,
        use_synthetic: bool = False,
        generator=None,
        synthetic_n_blocks: Optional[int] = None,
        synthetic_base_date: Optional[date] = None,
        synthetic_n_preplaced: int = 0,
        vary_layout: bool = True,
        grid_size: int = GRID_SIZE,
    ):
        super().__init__()

        self._original_blocks = blocks
        self._original_workspaces = workspaces
        self._strategy = strategy or BaseGridStrategy()

        # ── Synthetic 설정 ────────────────────────────────────────
        self._use_synthetic = use_synthetic
        self._generator = generator
        self._synthetic_n_blocks = synthetic_n_blocks or len(blocks)
        self._synthetic_base_date = synthetic_base_date or (
            min(b.in_date for b in blocks) if blocks else date(2026, 4, 1)
        )
        self._synthetic_n_preplaced = synthetic_n_preplaced
        self._vary_layout = vary_layout and use_synthetic

        # 현재 에피소드의 블록 (synthetic이면 reset에서 갱신)
        self._blocks = blocks
        self._num_blocks = len(blocks)
        self._num_workspaces = len(workspaces)
        self._workspaces = workspaces

        # 전략 주입 (원본에)
        for ws in self._original_workspaces:
            if ws.strategy is None:
                ws.strategy = self._strategy

        # 하드 제약
        self._constraints = [
            DimensionConstraint(),
            BlockPatternConstraint(),
        ]
        self._picker = ValidWorkspacePicker(
            blocks, workspaces, self._constraints
        )

        # ── 점유 그리드 렌더러 + 캐시 ─────────────────────────────
        self._grid_size = grid_size
        self._renderer = OccupancyGridRenderer(grid_size)
        self._grid_cache = GridCache(self._renderer, self._num_workspaces)

        # ── Action Space ──────────────────────────────────────────
        self.action_space = spaces.Discrete(self._num_workspaces)

        # ── Observation Space (Dict) ──────────────────────────────
        N = self._num_workspaces
        G = self._grid_size
        self.observation_space = spaces.Dict({
            "block": spaces.Box(
                low=0.0, high=1.0, shape=(10,), dtype=np.float32
            ),
            "grids": spaces.Box(
                low=0.0, high=1.0,
                shape=(N, 3, G, G), dtype=np.float32
            ),
            "ws_meta": spaces.Box(
                low=0.0, high=1.0,
                shape=(N, 2), dtype=np.float32
            ),
        })

        # ── 정규화 상수 ───────────────────────────────────────────
        self._update_norm_constants()
        self._update_ws_areas()

        # ── 에피소드 상태 ─────────────────────────────────────────
        self._current_step = 0
        self._assignments: List[int] = []
        self._ws_used_area: np.ndarray = np.zeros(
            self._num_workspaces, dtype=np.float32
        )
        self._env_date: date = self._base_date
        self._step_reward_sum = 0.0
        self._last_replay_potential = 0.0
        self._last_result: Optional[SimulationResult] = None
        self._placement_simulator: Optional[IncrementalPlacementSimulator] = None
        self._current_block_index: Optional[int] = None

    # ── 정규화 상수 갱신 ──────────────────────────────────────────

    def _update_norm_constants(self):
        """블록 속성 및 시간 정규화 상수 갱신."""
        blocks = self._blocks
        self._max_length  = max((b.length  for b in blocks), default=1.0) or 1.0
        self._max_breadth = max((b.breadth for b in blocks), default=1.0) or 1.0
        self._max_height  = max((b.height  for b in blocks), default=1.0) or 1.0
        self._max_weight  = max((b.weight  for b in blocks), default=1.0) or 1.0
        self._max_duration = max(
            (b.original_duration for b in blocks), default=1
        ) or 1

        # 입고일 분산 범위 (긴급도 정규화에 사용)
        if blocks:
            min_in = min(b.in_date for b in blocks)
            max_in = max(b.in_date for b in blocks)
            self._base_date = min_in
            self._date_spread = max((max_in - min_in).days, 1)
        else:
            self._base_date = date(2026, 4, 1)
            self._date_spread = 1

    def _update_ws_areas(self):
        """작업장 면적 및 스케일 정보 갱신."""
        self._ws_areas = np.array(
            [ws.length * ws.breadth for ws in self._workspaces],
            dtype=np.float32,
        )
        self._ws_areas = np.maximum(self._ws_areas, 1.0)

        # 스케일 정보 (1px당 미터, 정규화용)
        scales = np.array(
            [self._renderer.compute_scale_value(ws) for ws in self._workspaces],
            dtype=np.float32,
        )
        self._max_scale = max(scales.max(), 1e-6)
        self._ws_scales = scales / self._max_scale  # 정규화 [0, 1]

    # ── 작업장 재빌드 ─────────────────────────────────────────────

    def _rebuild_workspaces(self):
        """작업장 레이아웃 변형 + 기배치 블록 재생성."""
        import copy

        # 레이아웃 변형 (학습 시)
        if self._vary_layout and self._generator:
            self._workspaces = self._generator.generate_workspaces(
                self._original_workspaces
            )
        else:
            self._workspaces = copy.deepcopy(self._original_workspaces)

        for ws in self._workspaces:
            if ws.strategy is None:
                ws.strategy = self._strategy

        # 기배치 블록 합성 생성
        if self._generator and self._synthetic_n_preplaced > 0:
            preplaced = self._generator.generate_preplaced(
                self._synthetic_n_preplaced,
                self._workspaces,
                self._synthetic_base_date,
            )
            ws_map = {ws.code: ws for ws in self._workspaces}
            for ws_code, pp in preplaced:
                if ws_code in ws_map:
                    ws_map[ws_code].add_pre_placement(pp)

        # 면적·스케일 재계산
        self._update_ws_areas()

    def _sync_from_simulator(self, invalidate_grids: bool = True) -> None:
        """시뮬레이터의 실제 상태를 Env 관측 상태로 동기화합니다."""
        if self._placement_simulator is None:
            return

        sim = self._placement_simulator
        self._blocks = sim.blocks
        self._workspaces = sim.workspaces
        self._assignments = [
            int(a) if a is not None else -1
            for a in sim.assignments
        ]
        self._current_step = sim.assigned_count
        self._current_block_index = sim.current_block_index
        self._env_date = sim.env_date
        self._ws_used_area = np.array([
            sum(b.length * b.breadth for b in ws.blocks)
            for ws in self._workspaces
        ], dtype=np.float32)
        if invalidate_grids:
            self._grid_cache.invalidate_all()

    def _workspace_grid_signature(self, ws: Workspace, env_date: date) -> Tuple:
        """Return the workspace state that affects occupancy-grid rendering."""
        block_sig = tuple(
            (
                b.ref_x,
                b.ref_y,
                b.length,
                b.breadth,
                b.out_date,
            )
            for b in ws.blocks
        )
        preplaced_sig = tuple(
            (
                pp.label,
                pp.pos_x,
                pp.pos_y,
                pp.length,
                pp.breadth,
                pp.end_date,
            )
            for pp in ws.get_active_pre_placements(env_date)
        )
        return (
            ws.origin_x,
            ws.origin_y,
            ws.length,
            ws.breadth,
            block_sig,
            preplaced_sig,
        )

    def _workspace_grid_signatures(self, env_date: date) -> List[Tuple]:
        return [
            self._workspace_grid_signature(ws, env_date)
            for ws in self._workspaces
        ]

    # ── reset / step ──────────────────────────────────────────────

    def reset(
        self, *, seed=None, options=None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)

        # ── Synthetic 모드: 매 에피소드마다 새 블록 + 레이아웃 ────
        if self._use_synthetic and self._generator:
            self._blocks = self._generator.generate(
                self._synthetic_n_blocks,
                self._synthetic_base_date,
            )
            self._num_blocks = len(self._blocks)
            self._update_norm_constants()
            self._rebuild_workspaces()
            self._picker = ValidWorkspacePicker(
                self._blocks, self._workspaces, self._constraints
            )
        else:
            self._blocks = [b.clone() for b in self._original_blocks]
            self._num_blocks = len(self._blocks)
            self._update_norm_constants()
            self._rebuild_workspaces()
            self._picker = ValidWorkspacePicker(
                self._blocks, self._workspaces, self._constraints
            )

        self._current_step = 0
        self._assignments = []
        self._ws_used_area = np.zeros(self._num_workspaces, dtype=np.float32)
        self._env_date = self._base_date
        self._step_reward_sum = 0.0
        self._last_replay_potential = 0.0
        self._last_result = None
        self._placement_simulator = IncrementalPlacementSimulator(
            self._blocks,
            self._workspaces,
            DROPOUT_THRESHOLD,
        )
        self._sync_from_simulator()

        # 그리드 캐시 초기화 (전체 재렌더링)
        self._grid_cache.invalidate_all()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        action = int(action)
        if self._placement_simulator is None:
            raise RuntimeError("Environment must be reset before step().")

        prev_env_date = self._env_date
        prev_grid_signatures = self._workspace_grid_signatures(prev_env_date)

        step_result = self._placement_simulator.assign_current(action)
        self._sync_from_simulator(invalidate_grids=False)

        if self._env_date != prev_env_date:
            self._grid_cache.invalidate_all()
        else:
            next_grid_signatures = self._workspace_grid_signatures(self._env_date)
            for i, (before, after) in enumerate(
                zip(prev_grid_signatures, next_grid_signatures)
            ):
                if before != after:
                    self._grid_cache.invalidate(i)

        terminated = self._placement_simulator.is_done
        truncated = False
        step_reward = self._compute_intermediate_reward(step_result)
        self._step_reward_sum += step_reward
        reward = step_reward

        if terminated:
            terminal_reward = self._compute_terminal_reward()
            reward += terminal_reward

        obs = self._get_obs() if not terminated else self._get_terminal_obs()
        info = self._get_info()

        if terminated:
            info["assignments"] = [
                int(a) for a in self._placement_simulator.assignments
            ]
            info["raw_result"] = self._last_result
            info["terminal_reward"] = terminal_reward
            info["step_reward_sum"] = self._step_reward_sum
            info["episode_reward"] = self._step_reward_sum + terminal_reward

        return obs, reward, terminated, truncated, info

    # ── Action Masking (sb3-contrib 호환) ─────────────────────────

    def action_masks(self) -> np.ndarray:
        """현재 블록에 대한 유효 작업장 마스크."""
        if self._current_block_index is None:
            return np.ones(self._num_workspaces, dtype=bool)

        mask = self._picker.get_action_mask(
            self._current_block_index, self._num_workspaces
        )
        return np.array(mask, dtype=bool)

    # ── 보상 계산 ─────────────────────────────────────────────────

    def _score_delay_day(self, delay_days: int) -> float:
        if delay_days == SimulationResult.DROPOUT:
            return REWARD_DROPOUT
        if delay_days <= DELAY_THRESHOLD:
            return REWARD_COMPLIANT
        return -(
            (delay_days - DELAY_THRESHOLD)
            / (DROPOUT_THRESHOLD - DELAY_THRESHOLD)
        )

    def _score_delay_days(
        self,
        delay_days: List[int],
        divisor: Optional[int] = None,
    ) -> float:
        if not delay_days:
            return 0.0
        score_divisor = divisor if divisor is not None else len(delay_days)
        if score_divisor <= 0:
            return 0.0
        return sum(self._score_delay_day(dd) for dd in delay_days) / score_divisor

    def _compute_intermediate_reward(
        self,
        step_result: PlacementStepResult,
    ) -> float:
        reward = (
            SHAPING_PLACEMENT_SUCCESS
            if step_result.placed
            else SHAPING_PLACEMENT_FAILURE
        )
        reward += self._compute_partial_replay_reward()
        return reward

    def _compute_partial_replay_reward(self) -> float:
        if self._placement_simulator is None or self._current_step <= 0:
            return 0.0

        should_replay = (
            self._placement_simulator.is_done
            or self._current_step % PARTIAL_REPLAY_INTERVAL == 0
        )
        if not should_replay:
            return 0.0

        potential = self._score_delay_days(
            self._placement_simulator.resolved_delay_days(),
            divisor=max(self._num_blocks, 1),
        )
        reward = PARTIAL_REPLAY_WEIGHT * (
            potential - self._last_replay_potential
        )
        self._last_replay_potential = potential
        return reward

    def _compute_terminal_reward(self) -> float:
        """
        에피소드 종료 시 최종 보상.

        시뮬레이션 실행 후 블록별 결과 합산:
          준수(지연 <= 2일):    +1.0
          지연(2 < d <= 7일):   -(d-2)/5  → [-0.2, -1.0]
          탈락(d > 7일):        -2.0

        최종 보상 = sum / n  → [-2.0, +1.0]
        """
        if self._placement_simulator is None:
            return 0.0

        result = self._placement_simulator.result()
        self._last_result = result

        return self._score_delay_days(result.delay_days)

    # ── 관측 헬퍼 ────────────────────────────────────────────────

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        Dict 관측 구성.

        "block" (10,):
          [0]  길이 / max_length         블록 물리 속성
          [1]  폭 / max_breadth
          [2]  높이 / max_height
          [3]  중량 / max_weight
          [4]  공기(근무일) / max_duration
          [5]  입고 긴급도               시간/진행 정보
          [6]  공기 여유도
          [7]  진행률
          [8]  현재 블록 면적 스케일      (블록 면적 / 최대 작업장 면적)
          [9]  현재 블록 최대 축 비율     (max(L,B) / 최대 작업장 축)

        "grids" (N, 3, 128, 128):
          Ch0: 블록 점유 마스크
          Ch1: 잔여 출고 공기 (정규화)
          Ch2: 작업장 경계 마스크

        "ws_meta" (N, 2):
          [0] scale (1px당 m, 정규화)
          [1] occupancy_ratio (면적 점유율)
        """
        if self._current_block_index is None:
            return self._get_terminal_obs()

        blk = self._blocks[self._current_block_index]

        # ── Block features (10,) ──────────────────────────────────
        block_features = np.array([
            blk.length / self._max_length,
            blk.breadth / self._max_breadth,
            blk.height / self._max_height,
            blk.weight / self._max_weight,
            blk.original_duration / self._max_duration,
            # 입고 긴급도: 기준일에 가까울수록 0 (급함)
            np.clip(
                (blk.in_date - self._base_date).days / self._date_spread,
                0.0, 1.0
            ),
            # 공기 여유도
            blk.original_duration / self._max_duration,
            # 진행률
            self._current_step / max(self._num_blocks - 1, 1),
            # 블록 면적 스케일 (최대 작업장 대비)
            np.clip(
                (blk.length * blk.breadth) / self._ws_areas.max(),
                0.0, 1.0
            ),
            # 블록 최대 축 비율
            np.clip(
                max(blk.length, blk.breadth)
                / max(
                    max(ws.length for ws in self._workspaces),
                    max(ws.breadth for ws in self._workspaces),
                    1.0
                ),
                0.0, 1.0
            ),
        ], dtype=np.float32)

        # ── Workspace grids (N, 3, 128, 128) ─────────────────────
        grids = self._grid_cache.get_grids(
            self._workspaces, self._env_date
        )

        # ── Workspace meta (N, 2) ────────────────────────────────
        occupancy = np.clip(
            self._ws_used_area / self._ws_areas, 0.0, 1.0
        )
        ws_meta = np.stack([self._ws_scales, occupancy], axis=1)

        return {
            "block": block_features,
            "grids": grids,
            "ws_meta": ws_meta.astype(np.float32),
        }

    def _get_terminal_obs(self) -> Dict[str, np.ndarray]:
        """에피소드 종료 시 더미 관측."""
        N = self._num_workspaces
        G = self._grid_size
        return {
            "block": np.zeros(10, dtype=np.float32),
            "grids": np.zeros((N, 3, G, G), dtype=np.float32),
            "ws_meta": np.zeros((N, 2), dtype=np.float32),
        }

    def _get_info(self) -> Dict[str, Any]:
        return {
            "current_step": self._current_step,
            "current_block_index": self._current_block_index,
            "total_blocks": self._num_blocks,
        }
