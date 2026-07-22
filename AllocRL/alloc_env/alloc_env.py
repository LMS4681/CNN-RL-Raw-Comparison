"""Gymnasium environment exposing the fixed schema-4 allocation state."""

from __future__ import annotations

import copy
import gymnasium as gym
import numpy as np
from datetime import date
from numbers import Integral
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
from .occupancy_grid import (
    BaseGridCache,
    CandidatePlacement,
    GRID_SIZE,
    OccupancyGridRenderer,
)
from .observation_state import (
    ORDERED_FUTURE_COUNT,
    ObservationScales,
    build_observation_scales,
    build_observation_space,
    encode_current_block,
    encode_future_blocks,
    encode_future_demand,
    encode_pending_queues,
    working_day_position,
)

# C# AllocConst 대응
DELAY_THRESHOLD = 2       # 준수(compliance) 기준: 지연 <= 2일
DROPOUT_THRESHOLD = 7     # 탈락(dropout) 기준: 지연 > 7일

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
    """Sequential block-allocation environment with fixed schema-4 state."""

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
        state_context_mode: str = "full",
        observation_scales: ObservationScales | None = None,
    ):
        super().__init__()

        self._original_blocks = blocks
        self._original_workspaces = workspaces
        self._strategy = strategy or BaseGridStrategy()

        if (
            isinstance(grid_size, bool)
            or not isinstance(grid_size, Integral)
            or grid_size < 1
        ):
            raise ValueError("grid_size must be a positive integer")
        if (
            not isinstance(state_context_mode, str)
            or state_context_mode not in {"full", "current"}
        ):
            raise ValueError(
                "state_context_mode must be 'full' or 'current'"
            )
        if observation_scales is not None and not isinstance(
            observation_scales, ObservationScales
        ):
            raise TypeError("observation_scales must be an ObservationScales")
        self._state_context_mode = state_context_mode
        if not workspaces:
            raise ValueError("at least one workspace is required")

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
            and len(self._get_infeasible_blocks()) == self._num_blocks
        ):
            raise ValueError(
                "Environment has no agent decision: all blocks are infeasible."
            )

        # ── 점유 그리드 렌더러 + 캐시 ─────────────────────────────
        self._grid_size = int(grid_size)
        self._renderer = OccupancyGridRenderer(grid_size)
        self._grid_cache = BaseGridCache(self._renderer, self._num_workspaces)

        # ── Action Space ──────────────────────────────────────────
        self.action_space = gym.spaces.Discrete(self._num_workspaces)

        # ── Observation Space (Dict) ──────────────────────────────
        self.observation_space = build_observation_space(
            self._num_workspaces, self._grid_size
        )
        # ── 정규화 상수 ───────────────────────────────────────────
        self._observation_scales = observation_scales or build_observation_scales(
            self._original_blocks,
            self._original_workspaces,
            DROPOUT_THRESHOLD,
            require_full_source=False,
        )
        self._validate_observation_scales(self._observation_scales)
        self._base_date = self._observation_scales.base_date
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
        self._candidate_placements: List[CandidatePlacement] = []

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

    def _get_infeasible_blocks(self) -> List[int]:
        return self._picker.get_infeasible_blocks()

    def _validate_observation_scales(
        self,
        scales: ObservationScales,
        blocks: Optional[List[Block]] = None,
        workspaces: Optional[List[Workspace]] = None,
    ) -> None:
        source_blocks = self._original_blocks if blocks is None else blocks
        source_workspaces = (
            self._original_workspaces if workspaces is None else workspaces
        )
        if scales.dropout_threshold != DROPOUT_THRESHOLD:
            raise ValueError(
                "observation scales dropout_threshold does not match environment"
            )
        if source_blocks:
            if max(block.length for block in source_blocks) > scales.max_length:
                raise ValueError(
                    "source blocks exceed observation scales max_length"
                )
            if max(block.breadth for block in source_blocks) > scales.max_breadth:
                raise ValueError(
                    "source blocks exceed observation scales max_breadth"
                )
            if (
                max(block.original_duration for block in source_blocks)
                > scales.max_duration
            ):
                raise ValueError(
                    "source blocks exceed observation scales max_duration"
                )
            earliest_start = min(block.in_date for block in source_blocks)
            if scales.base_date > earliest_start:
                raise ValueError(
                    "observation scales base_date is later than the earliest "
                    "source block in_date"
                )
            latest_start = max(block.in_date for block in source_blocks)
            if (
                working_day_position(scales.base_date, latest_start)
                > scales.date_span_workdays
            ):
                raise ValueError(
                    "source blocks exceed observation scales date_span_workdays"
                )

        workspace_areas = []
        for workspace in source_workspaces:
            if (
                not np.isfinite(workspace.length)
                or not np.isfinite(workspace.breadth)
                or workspace.length <= 0
                or workspace.breadth <= 0
            ):
                raise ValueError(
                    "workspace geometry must be finite and positive"
                )
            workspace_areas.append(workspace.length * workspace.breadth)
        if any(
            workspace.length > scales.max_workspace_length
            for workspace in source_workspaces
        ):
            raise ValueError(
                "workspaces exceed observation scales max_workspace_length"
            )
        if any(
            workspace.breadth > scales.max_workspace_breadth
            for workspace in source_workspaces
        ):
            raise ValueError(
                "workspaces exceed observation scales max_workspace_breadth"
            )
        if any(area > scales.max_workspace_area for area in workspace_areas):
            raise ValueError(
                "workspaces exceed observation scales max_workspace_area"
            )
        if sum(workspace_areas) > scales.total_workspace_area:
            raise ValueError(
                "workspaces exceed observation scales total_workspace_area"
            )

    def _update_ws_areas(self):
        """작업장 면적 및 스케일 정보 갱신."""
        self._ws_areas = np.array(
            [ws.length * ws.breadth for ws in self._workspaces],
            dtype=np.float32,
        )
        self._ws_areas = np.maximum(self._ws_areas, 1.0)

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

    def clone_for_diagnostics(self) -> BlockPlacementEnv:
        """Clone the current environment state for side-effect-free previews."""
        if self._placement_simulator is None:
            raise RuntimeError(
                "Environment must be reset before diagnostic cloning."
            )

        clone = copy.copy(self)
        clone._original_blocks = [
            block.clone() for block in self._original_blocks
        ]
        clone._original_workspaces = Workspace.deep_copy_list(
            self._original_workspaces
        )
        clone._strategy = copy.deepcopy(self._strategy)
        clone._generator = copy.deepcopy(self._generator)
        clone._constraints = copy.deepcopy(self._constraints)
        clone._placement_simulator = (
            self._placement_simulator.clone_for_diagnostics()
        )
        clone._blocks = clone._placement_simulator.blocks
        clone._workspaces = clone._placement_simulator.workspaces
        clone._picker = ValidWorkspacePicker(
            clone._blocks, clone._workspaces, clone._constraints
        )
        clone._assignments = list(self._assignments)
        clone._ws_areas = self._ws_areas.copy()
        clone._ws_used_area = self._ws_used_area.copy()
        clone._emitted_resolved_indices = set(
            self._emitted_resolved_indices
        )
        clone._last_result = copy.deepcopy(self._last_result)
        clone._candidate_placements = copy.deepcopy(
            self._candidate_placements
        )
        clone._renderer = OccupancyGridRenderer(self._grid_size)
        clone._grid_cache = BaseGridCache(
            clone._renderer, clone._num_workspaces
        )
        clone.action_space = copy.deepcopy(self.action_space)
        clone.observation_space = copy.deepcopy(self.observation_space)
        clone._np_random = copy.deepcopy(self.np_random)
        if hasattr(self, "_np_random_seed"):
            clone._np_random_seed = self._np_random_seed
        return clone

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
                self._validate_observation_scales(
                    self._observation_scales,
                    self._blocks,
                    self._workspaces,
                )
                self._picker = ValidWorkspacePicker(
                    self._blocks, self._workspaces, self._constraints
                )
                if len(self._get_infeasible_blocks()) < self._num_blocks:
                    break
            else:
                raise RuntimeError(
                    "Synthetic environment has no agent decision after 10 attempts."
                )
        else:
            self._blocks = [b.clone() for b in self._original_blocks]
            self._num_blocks = len(self._blocks)
            self._rebuild_workspaces()
            self._validate_observation_scales(
                self._observation_scales,
                self._blocks,
                self._workspaces,
            )
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
            infeasible_indices=self._get_infeasible_blocks(),
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

        candidate_position = self._candidate_placements[action].position
        self._placement_simulator.assign_current(
            action, placement_override=candidate_position
        )
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
            return np.ones(self._num_workspaces, dtype=bool)

        mask = self._picker.get_action_mask(
            self._current_block_index, self._num_workspaces
        )
        return np.array(mask, dtype=bool)

    # ── Evaluation-only future optionality diagnostics ────────────

    def immediate_placeability(self) -> np.ndarray:
        """Return current geometric placeability without changing state."""
        return np.array(
            [candidate.placeable for candidate in self._candidate_placements],
            dtype=bool,
        )

    def workspace_free_areas(self) -> np.ndarray:
        """Return non-negative workspace free areas without changing state."""
        return np.maximum(
            self._ws_areas - self._ws_used_area, 0.0
        ).astype(np.float32)

    def future_workspace_choice_indices(self) -> List[int]:
        """Return the exact future block set used by the current observation."""
        if self._placement_simulator is None:
            return []
        return self._placement_simulator.upcoming_block_indices(
            ORDERED_FUTURE_COUNT
        )

    def future_workspace_choice_count(
        self,
        block_indices: Optional[Iterable[int]] = None,
    ) -> int:
        """Count immediately usable workspaces for a fixed future block set."""
        simulator = self._placement_simulator
        if simulator is None:
            return 0
        indices = (
            self.future_workspace_choice_indices()
            if block_indices is None
            else list(block_indices)
        )

        return self._future_workspace_choice_count_on(
            indices, self._workspaces
        )

    def future_workspace_choice_count_after_action(
        self,
        action: int,
        block_indices: Optional[Iterable[int]] = None,
    ) -> int:
        """Preview future choices immediately after one candidate placement."""
        simulator = self._placement_simulator
        if simulator is None or simulator.current_block is None:
            return 0

        action = int(action)
        if not 0 <= action < self._num_workspaces:
            raise IndexError(f"Workspace action is out of range: {action}")
        if not self.action_masks()[action]:
            raise ValueError(f"Workspace action is masked: {action}")

        indices = (
            self.future_workspace_choice_indices()
            if block_indices is None
            else list(block_indices)
        )
        preview_workspaces = list(self._workspaces)
        preview_workspaces[action] = self._workspaces[action].deep_copy()
        placed_block = simulator.current_block.clone()
        position = self._candidate_placements[action].position
        if position is not None:
            center_x, center_y = position
            placed_block.move(
                center_x - placed_block.ref_x,
                center_y - placed_block.ref_y,
            )
            preview_workspaces[action].add_block(
                placed_block, self._env_date
            )

        return self._future_workspace_choice_count_on(
            indices, preview_workspaces
        )

    def _future_workspace_choice_count_on(
        self,
        block_indices: Iterable[int],
        workspaces: List[Workspace],
    ) -> int:
        simulator = self._placement_simulator
        if simulator is None:
            return 0

        total = 0
        for block_index in block_indices:
            if not 0 <= block_index < len(self._blocks):
                continue
            if (
                block_index not in simulator.pending
                or simulator.delay_days[block_index] is not None
            ):
                continue
            block = self._blocks[block_index]
            for workspace in workspaces:
                if not all(
                    constraint.is_feasible(block, workspace)
                    for constraint in self._constraints
                ):
                    continue
                trial = block.clone()
                position = workspace.determine_placement_position(
                    trial, self._env_date
                )
                total += int(position is not None)
        return total

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

    def _compute_candidate_placements(
        self,
        blk: Block,
    ) -> List[CandidatePlacement]:
        candidates: List[CandidatePlacement] = []
        hard_mask = self.action_masks()
        for allowed, workspace in zip(hard_mask, self._workspaces):
            if not allowed:
                candidates.append(
                    CandidatePlacement(None, blk.length, blk.breadth)
                )
                continue

            trial = blk.clone()
            position = workspace.determine_placement_position(
                trial, self._env_date
            )

            candidates.append(
                CandidatePlacement(
                    position,
                    blk.length,
                    blk.breadth,
                )
            )
        return candidates

    def _get_obs(self) -> Dict[str, np.ndarray]:
        if self._current_block_index is None:
            return self._get_terminal_obs()

        simulator = self._placement_simulator
        if simulator is None or simulator.current_block is None:
            return self._get_terminal_obs()
        block = simulator.current_block
        block_features = encode_current_block(
            block,
            self._env_date,
            simulator.assigned_count,
            self._observation_scales,
        )

        base_grids = self._grid_cache.get_base_grids(
            self._workspaces, self._env_date
        )
        self._candidate_placements = self._compute_candidate_placements(block)
        candidate_context = np.stack([
            self._renderer.render_candidate_context(
                workspace, candidate, block, self._env_date
            )
            for workspace, candidate in zip(
                self._workspaces, self._candidate_placements
            )
        ])
        grids = np.concatenate([base_grids, candidate_context], axis=1)

        workspace_lengths = np.array(
            [workspace.length for workspace in self._workspaces],
            dtype=np.float32,
        )
        workspace_breadths = np.array(
            [workspace.breadth for workspace in self._workspaces],
            dtype=np.float32,
        )
        if (
            not np.all(np.isfinite(workspace_lengths))
            or not np.all(np.isfinite(workspace_breadths))
            or np.any(workspace_lengths <= 0.0)
            or np.any(workspace_breadths <= 0.0)
        ):
            raise ValueError(
                "workspace dimensions must be finite and positive"
            )
        placed_area_ratio = np.clip(
            self._ws_used_area / self._ws_areas, 0.0, 1.0
        )
        placeable = np.array(
            [candidate.placeable for candidate in self._candidate_placements],
            dtype=np.float32,
        )
        ws_meta = np.stack([
            np.clip(
                workspace_lengths
                / self._observation_scales.max_workspace_length,
                0.0,
                1.0,
            ),
            np.clip(
                workspace_breadths
                / self._observation_scales.max_workspace_breadth,
                0.0,
                1.0,
            ),
            placed_area_ratio,
            placeable,
            np.clip(block.length / workspace_lengths, 0.0, 1.0),
            np.clip(block.breadth / workspace_breadths, 0.0, 1.0),
            np.clip(
                (block.length * block.breadth)
                / (workspace_lengths * workspace_breadths),
                0.0,
                1.0,
            ),
            np.minimum(workspace_lengths, workspace_breadths)
            / np.maximum(workspace_lengths, workspace_breadths),
        ], axis=1).astype(np.float32)

        future_indices = simulator.unassigned_block_indices()
        future_blocks, future_mask = encode_future_blocks(
            self._blocks,
            future_indices,
            self._env_date,
            self._observation_scales,
        )
        future_demand = encode_future_demand(
            self._blocks,
            future_indices,
            self._env_date,
            self._observation_scales,
        )
        pending_blocks, pending_mask, pending_summary = encode_pending_queues(
            self._blocks,
            self._workspaces,
            simulator,
            self._observation_scales,
        )
        if self._state_context_mode == "current":
            future_blocks = np.zeros_like(future_blocks)
            future_mask = np.zeros_like(future_mask)
            future_demand = np.zeros_like(future_demand)
            pending_blocks = np.zeros_like(pending_blocks)
            pending_mask = np.zeros_like(pending_mask)
            pending_summary = np.zeros_like(pending_summary)

        arrays = {
            "block": block_features,
            "grids": grids,
            "ws_meta": ws_meta,
            "future_blocks": future_blocks,
            "future_mask": future_mask,
            "future_demand": future_demand,
            "pending_blocks": pending_blocks,
            "pending_mask": pending_mask,
            "pending_summary": pending_summary,
        }
        return {
            key: arrays[key]
            for key in self.observation_space.spaces
        }

    def _get_terminal_obs(self) -> Dict[str, np.ndarray]:
        """Return a schema-4 zero observation after the episode ends."""
        return {
            key: np.zeros(space.shape, dtype=space.dtype)
            for key, space in self.observation_space.spaces.items()
        }

    def _get_info(self) -> Dict[str, Any]:
        return {
            "current_step": self._current_step,
            "current_block_index": self._current_block_index,
            "total_blocks": self._num_blocks,
        }
