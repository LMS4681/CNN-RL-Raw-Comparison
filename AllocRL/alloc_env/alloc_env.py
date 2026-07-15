"""
블록 배치 강화학습 Gymnasium 환경 (CNN 관측 버전).

- Action:  Discrete(num_workspaces) - 현재 블록을 어느 작업장에 배정
- Obs:     Dict {
              "block":         블록 속성 + 시간 + 스케일 정보,
              "grids":         작업장별 3채널 점유 그리드 (N, 3, G, G),
              "ws_meta":       작업장별 메타데이터 (N, 3),
              # 아래는 n_future_blocks > 0일 때만 추가되는 미래 lookahead:
              "future_blocks": 다음 k개 블록 피처 (k, FUTURE_BLOCK_FEATURE_DIM),
              "future_mask":   미래 블록 유효 마스크 (k,), 1=유효/0=패딩,
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
)
from .simulator import SimulationResult
from .strategy import BaseGridStrategy
from .occupancy_grid import OccupancyGridRenderer, GridCache, GRID_SIZE

# C# AllocConst 대응
DELAY_THRESHOLD = 2       # 준수(compliance) 기준: 지연 <= 2일
DROPOUT_THRESHOLD = 7     # 탈락(dropout) 기준: 지연 > 7일

# 미래 블록 lookahead 관측 (n_future_blocks > 0일 때만 obs에 포함).
# 각 미래 블록당 피처 차원. _future_block_features 참고.
FUTURE_BLOCK_FEATURE_DIM = 8

# ── Reward 설정 ──────────────────────────────────────────────────
REWARD_COMPLIANT = 1.0     # 준수(지연 <= 2일)
REWARD_DROPOUT = -2.0      # 탈락(지연 > 7일)

# 원본 착수일 폭이 없을 때 쓰는 synthetic 날짜 분산 fallback.
# 실제 학습 synthetic은 원본 블록 착수일 폭의 0.5~1.2배 범위를 사용합니다.
DEFAULT_SYNTHETIC_SPREAD_DAYS = 90
MIN_SYNTHETIC_SPREAD_DAYS = 30
SYNTHETIC_SPREAD_MIN_RATIO = 0.5
SYNTHETIC_SPREAD_MAX_RATIO = 1.2


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
      "ws_meta": (N, 3)          - 작업장별 (scale, occupancy_ratio, placeable_now)

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
        active_workspace_codes: Optional[List[str]] = None,
        vary_layout: bool = True,
        grid_size: int = GRID_SIZE,
        n_future_blocks: int = 0,
    ):
        super().__init__()

        self._original_blocks = blocks
        self._original_workspaces = workspaces
        self._strategy = strategy or BaseGridStrategy()

        # 미래 블록 lookahead 관측 개수 (0이면 관측에 포함하지 않음 → 기존 계약 유지)
        self._n_future_blocks = max(int(n_future_blocks), 0)

        # ── Synthetic 설정 ────────────────────────────────────────
        self._use_synthetic = use_synthetic
        self._generator = generator
        self._synthetic_n_blocks = synthetic_n_blocks or len(blocks)
        self._synthetic_base_date = synthetic_base_date or (
            min(b.in_date for b in blocks) if blocks else date(2026, 4, 1)
        )
        self._synthetic_spread_range = self._compute_synthetic_spread_range(blocks)
        self._synthetic_n_preplaced = synthetic_n_preplaced
        self._vary_layout = vary_layout and use_synthetic

        # 현재 에피소드의 블록 (synthetic이면 reset에서 갱신)
        self._blocks = blocks
        self._num_blocks = len(blocks)
        self._num_workspaces = len(workspaces)
        self._workspaces = workspaces
        self._active_workspace_codes = (
            {code.strip().upper() for code in active_workspace_codes if code.strip()}
            if active_workspace_codes else None
        )
        if self._active_workspace_codes:
            workspace_codes = {ws.code.upper() for ws in workspaces}
            unknown_codes = sorted(self._active_workspace_codes - workspace_codes)
            if unknown_codes:
                raise ValueError(
                    "Unknown active workspace code(s): "
                    + ", ".join(unknown_codes)
                )
            self._active_workspace_mask = np.array(
                [ws.code.upper() in self._active_workspace_codes for ws in workspaces],
                dtype=bool,
            )
            if not self._active_workspace_mask.any():
                raise ValueError("At least one active workspace is required.")
        else:
            self._active_workspace_mask = np.ones(
                self._num_workspaces, dtype=bool
            )

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
        if (
            not self._use_synthetic
            and self._num_blocks > 0
            and len(self._get_active_infeasible_blocks()) == self._num_blocks
        ):
            raise ValueError(
                "Environment has no agent decision: all blocks are infeasible."
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
        obs_spaces = {
            "block": spaces.Box(
                low=0.0, high=1.0, shape=(10,), dtype=np.float32
            ),
            "grids": spaces.Box(
                low=0.0, high=1.0,
                shape=(N, 3, G, G), dtype=np.float32
            ),
            "ws_meta": spaces.Box(
                low=0.0, high=1.0,
                shape=(N, 3), dtype=np.float32
            ),
        }
        # 미래 블록 lookahead: 옵트인. n_future_blocks=0이면 키를 추가하지 않아
        # 기존 관측 계약({block, grids, ws_meta})을 그대로 유지한다.
        if self._n_future_blocks > 0:
            obs_spaces["future_blocks"] = spaces.Box(
                low=0.0, high=1.0,
                shape=(self._n_future_blocks, FUTURE_BLOCK_FEATURE_DIM),
                dtype=np.float32,
            )
            obs_spaces["future_mask"] = spaces.Box(
                low=0.0, high=1.0,
                shape=(self._n_future_blocks,), dtype=np.float32,
            )
        self.observation_space = spaces.Dict(obs_spaces)

        # ── 정규화 상수 ───────────────────────────────────────────
        self._init_norm_constants()
        self._update_ws_areas()

        # ── 에피소드 상태 ─────────────────────────────────────────
        self._current_step = 0
        self._assignments: List[int] = []
        self._ws_used_area: np.ndarray = np.zeros(
            self._num_workspaces, dtype=np.float32
        )
        self._env_date: date = self._base_date
        self._emitted_resolved_indices: set[int] = set()
        self._resolved_reward_sum = 0.0
        self._last_result: Optional[SimulationResult] = None
        self._placement_simulator: Optional[IncrementalPlacementSimulator] = None
        self._current_block_index: Optional[int] = None

    # ── 정규화 상수 갱신 ──────────────────────────────────────────

    @staticmethod
    def _compute_synthetic_spread_range(
        blocks: List[Block],
    ) -> Tuple[int, int]:
        if not blocks:
            return (MIN_SYNTHETIC_SPREAD_DAYS, DEFAULT_SYNTHETIC_SPREAD_DAYS)

        min_in = min(b.in_date for b in blocks)
        max_in = max(b.in_date for b in blocks)
        reference_spread = max((max_in - min_in).days, 1)
        spread_min = max(
            MIN_SYNTHETIC_SPREAD_DAYS,
            int(round(reference_spread * SYNTHETIC_SPREAD_MIN_RATIO)),
        )
        spread_max = max(
            spread_min,
            int(round(reference_spread * SYNTHETIC_SPREAD_MAX_RATIO)),
        )
        return (spread_min, spread_max)

    def _get_active_infeasible_blocks(self) -> List[int]:
        infeasible = set(self._picker.get_infeasible_blocks())
        if self._active_workspace_mask.all():
            return sorted(infeasible)

        for block_index in range(self._num_blocks):
            valid_workspaces = self._picker.get_valid_workspaces(block_index)
            if not any(self._active_workspace_mask[i] for i in valid_workspaces):
                infeasible.add(block_index)
        return sorted(infeasible)

    def _init_norm_constants(self):
        """
        블록 속성·시간 정규화 상수를 '고정값'으로 1회 계산합니다.

        synthetic 모드에서 매 에피소드 max(...)를 재계산하면, 동일한 물리
        블록이 에피소드마다 다른 정규화 값을 갖게 되어(관측 semantics 드리프트)
        정책·가치 함수 학습이 불안정해집니다. 따라서 안정적인 기준
        (생성기 분포 상한 또는 원본 블록 최댓값)에서 상수를 한 번만 구합니다.
        """
        gen = self._generator
        if self._use_synthetic and gen is not None:
            dist = gen.dist
            self._max_length  = max(float(dist.length.high), 1.0)
            self._max_breadth = max(float(dist.breadth.high), 1.0)
            self._max_height  = max(float(dist.height.high), 1.0)
            self._max_weight  = max(float(dist.weight.high), 1.0)
            # duration_days는 달력일 분포. 생성기는 근무일 공기를
            # working_dur ≈ dur*0.7 (최소 2)로 만든다(block_generator.generate).
            self._max_duration = max(int(dist.duration_days.high * 0.7), 1)
            # 입고 긴급도 기준: synthetic base_date와 생성기 분산 범위
            self._base_date = self._synthetic_base_date
            self._date_spread = max(self._synthetic_spread_range[1], 1)
        else:
            blocks = self._original_blocks
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
        # 정규화 상수는 __init__에서 고정 계산됨(에피소드 간 관측 일관성).
        if self._use_synthetic and self._generator:
            for _ in range(10):
                self._blocks = self._generator.generate(
                    self._synthetic_n_blocks,
                    self._synthetic_base_date,
                    spread_days=self._synthetic_spread_range,
                )
                self._num_blocks = len(self._blocks)
                self._rebuild_workspaces()
                self._picker = ValidWorkspacePicker(
                    self._blocks, self._workspaces, self._constraints
                )
                if len(self._get_active_infeasible_blocks()) < self._num_blocks:
                    break
            else:
                raise RuntimeError(
                    "Synthetic environment has no agent decision after 10 attempts."
                )
        else:
            self._blocks = [b.clone() for b in self._original_blocks]
            self._num_blocks = len(self._blocks)
            self._rebuild_workspaces()
            self._picker = ValidWorkspacePicker(
                self._blocks, self._workspaces, self._constraints
            )

        self._current_step = 0
        self._assignments = []
        self._ws_used_area = np.zeros(self._num_workspaces, dtype=np.float32)
        self._env_date = self._base_date
        self._emitted_resolved_indices = set()
        self._resolved_reward_sum = 0.0
        self._last_result = None
        self._placement_simulator = IncrementalPlacementSimulator(
            self._blocks,
            self._workspaces,
            DROPOUT_THRESHOLD,
            infeasible_indices=self._get_active_infeasible_blocks(),
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

        self._placement_simulator.assign_current(action)
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
        resolved_reward, newly_resolved = self._collect_resolved_reward()
        reward = resolved_reward

        if terminated:
            terminal_score = self._compute_terminal_reward()
            terminal_residual = terminal_score - self._resolved_reward_sum
            reward += terminal_residual

        obs = self._get_obs() if not terminated else self._get_terminal_obs()
        info = self._get_info()
        info["newly_resolved_indices"] = newly_resolved
        info["resolved_step_reward"] = resolved_reward

        if terminated:
            info["assignments"] = [
                int(a) for a in self._placement_simulator.assignments
            ]
            info["raw_result"] = self._last_result
            info["resolved_reward"] = self._resolved_reward_sum
            info["terminal_residual"] = terminal_residual
            info["terminal_score"] = terminal_score
            info["terminal_reward"] = terminal_score
            info["episode_reward"] = (
                self._resolved_reward_sum + terminal_residual
            )

        return obs, reward, terminated, truncated, info

    # ── Action Masking (sb3-contrib 호환) ─────────────────────────

    def action_masks(self) -> np.ndarray:
        """현재 블록에 대한 유효 작업장 마스크."""
        if self._current_block_index is None:
            return self._active_workspace_mask.copy()

        mask = self._picker.get_action_mask(
            self._current_block_index, self._num_workspaces
        )
        return np.array(mask, dtype=bool) & self._active_workspace_mask

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

    def _collect_resolved_reward(self) -> Tuple[float, List[int]]:
        if self._placement_simulator is None:
            return 0.0, []

        score = 0.0
        newly_resolved: List[int] = []
        for result in self._placement_simulator.drain_transition_results():
            block_index = result.block_index
            if (
                result.delay_days is None
                or block_index in self._emitted_resolved_indices
            ):
                continue
            self._emitted_resolved_indices.add(block_index)
            newly_resolved.append(block_index)
            score += self._score_delay_day(result.delay_days) / max(
                self._num_blocks, 1
            )

        self._resolved_reward_sum += score
        return score, newly_resolved

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

    def _compute_placeability(self, blk: Block) -> np.ndarray:
        """
        현재 블록을 각 작업장에 'env_date 기준 지금' 즉시 배치할 수 있는지 여부.

        시뮬레이터가 실제 배치에 쓰는 strategy.determine_position을 그대로 호출하므로
        즉시 배치 성공을 정확히 예측한다(90° 회전 재시도 포함). 하드 제약(치수/패턴)에
        걸리는 작업장은 계산을 건너뛰고 0으로 둔다.

        이 신호는 '피처'이지 '마스크'가 아니다 — 지금 꽉 차 있어도 나중에 블록이
        출고되어 자리가 나길 기다리는 전략이 유효하므로 하드 마스킹하지 않는다.

        성능: 결정마다 최대 N번의 determine_position 탐색이 든다. 빈 작업장은 즉시
        반환되어 저렴하지만 꽉 찬 작업장은 탐색을 소진한다(백로그 D의 캐싱 대상).
        """
        n = self._num_workspaces
        placeable = np.zeros(n, dtype=np.float32)
        if self._current_block_index is None:
            return placeable

        mask = self.action_masks()  # 하드 제약(치수/패턴) 통과 여부
        env_date = self._env_date
        for i, ws in enumerate(self._workspaces):
            if not mask[i]:
                continue  # 치수/패턴상 애초에 배치 불가 → 0
            pos = ws.determine_placement_position(blk, env_date)
            if pos is None:
                # 시뮬레이터와 동일하게 90° 회전 후 재시도
                blk.turn()
                pos = ws.determine_placement_position(blk, env_date)
                blk.turn()  # 원래 방향 복원 (turn은 자기 역연산)
            if pos is not None:
                placeable[i] = 1.0
        return placeable

    def _future_block_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        다음 k개 pending 블록의 피처와 유효 마스크 (미래 lookahead 관측).

        반환:
          features (k, FUTURE_BLOCK_FEATURE_DIM):
            [0] 길이 / max_length
            [1] 폭 / max_breadth
            [2] 높이 / max_height
            [3] 중량 / max_weight
            [4] 공기(근무일) / max_duration
            [5] 도착 임박도: (in_date - 현재 env_date) / date_spread, [0,1] 클립
                (0 = 지금 도착 대기, 1 = 먼 미래)
            [6] 종횡비 (min/max)
            [7] 면적 스케일 (블록 면적 / 최대 작업장 면적)
          mask (k,): 유효 슬롯 1.0, 패딩 슬롯 0.0.

        블록 정규화 상수는 현재 블록 피처와 동일하게 사용해 의미를 일치시킨다.
        패딩(남은 블록 < k) 슬롯은 0으로 채운다.
        """
        k = self._n_future_blocks
        features = np.zeros((k, FUTURE_BLOCK_FEATURE_DIM), dtype=np.float32)
        mask = np.zeros(k, dtype=np.float32)
        if k == 0 or self._placement_simulator is None:
            return features, mask

        upcoming = self._placement_simulator.upcoming_block_indices(k)
        max_ws_area = float(self._ws_areas.max())
        for slot, idx in enumerate(upcoming):
            blk = self._blocks[idx]
            features[slot] = (
                blk.length / self._max_length,
                blk.breadth / self._max_breadth,
                blk.height / self._max_height,
                blk.weight / self._max_weight,
                blk.original_duration / self._max_duration,
                np.clip(
                    (blk.in_date - self._env_date).days / self._date_spread,
                    0.0, 1.0,
                ),
                min(blk.length, blk.breadth)
                / max(blk.length, blk.breadth, 1e-6),
                np.clip(
                    (blk.length * blk.breadth) / max_ws_area, 0.0, 1.0
                ),
            )
            mask[slot] = 1.0
        return features, mask

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
          [6]  블록 종횡비 (min/max)      형상 정보 (회전 적합성)
          [7]  진행률
          [8]  현재 블록 면적 스케일      (블록 면적 / 최대 작업장 면적)
          [9]  현재 블록 최대 축 비율     (max(L,B) / 최대 작업장 축)

        "grids" (N, 3, 128, 128):
          Ch0: 블록 점유 마스크
          Ch1: 잔여 출고 공기 (정규화)
          Ch2: 작업장 경계 마스크

        "ws_meta" (N, 3):
          [0] scale (1px당 m, 정규화)
          [1] occupancy_ratio (면적 점유율)
          [2] placeable_now (현재 env_date에 즉시 배치 가능하면 1, 아니면 0)
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
            # 블록 종횡비 (min/max) — 정사각(≈1)인지 길쭉한지, 회전 적합성 판단
            min(blk.length, blk.breadth) / max(blk.length, blk.breadth, 1e-6),
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
        if not self._active_workspace_mask.all():
            grids = grids.copy()
            grids[~self._active_workspace_mask] = 0.0

        # ── Workspace meta (N, 3) ────────────────────────────────
        occupancy = np.clip(
            self._ws_used_area / self._ws_areas, 0.0, 1.0
        )
        placeable = self._compute_placeability(blk)
        ws_meta = np.stack([self._ws_scales, occupancy, placeable], axis=1)
        if not self._active_workspace_mask.all():
            ws_meta[~self._active_workspace_mask] = 0.0

        obs = {
            "block": block_features,
            "grids": grids,
            "ws_meta": ws_meta.astype(np.float32),
        }
        if self._n_future_blocks > 0:
            future_blocks, future_mask = self._future_block_features()
            obs["future_blocks"] = future_blocks
            obs["future_mask"] = future_mask
        return obs

    def _get_terminal_obs(self) -> Dict[str, np.ndarray]:
        """에피소드 종료 시 더미 관측."""
        N = self._num_workspaces
        G = self._grid_size
        obs = {
            "block": np.zeros(10, dtype=np.float32),
            "grids": np.zeros((N, 3, G, G), dtype=np.float32),
            "ws_meta": np.zeros((N, 3), dtype=np.float32),
        }
        if self._n_future_blocks > 0:
            obs["future_blocks"] = np.zeros(
                (self._n_future_blocks, FUTURE_BLOCK_FEATURE_DIM),
                dtype=np.float32,
            )
            obs["future_mask"] = np.zeros(
                self._n_future_blocks, dtype=np.float32
            )
        return obs

    def _get_info(self) -> Dict[str, Any]:
        return {
            "current_step": self._current_step,
            "current_block_index": self._current_block_index,
            "total_blocks": self._num_blocks,
        }
