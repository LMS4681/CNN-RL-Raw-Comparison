"""Incremental placement simulator for RL environment state transitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Tuple

from . import calendar as cal
from .block import Block
from .simulator import SimulationResult
from .workspace import Workspace


@dataclass
class PlacementStepResult:
    """Result of applying one RL workspace assignment."""

    block_index: int
    workspace_index: int
    placed: bool
    delayed: bool
    dropped: bool
    delay_days: Optional[int]


class IncrementalPlacementSimulator:
    """
    Date-driven simulator that requests workspace assignments just-in-time.

    It uses the same placement, delay, and dropout rules as
    PlacementSimulator.replay(), but exposes one decision point at a time.
    """

    def __init__(
        self,
        original_blocks: List[Block],
        original_workspaces: List[Workspace],
        dropout_threshold: int,
        infeasible_indices: Optional[set] = None,
    ):
        self._original_blocks = original_blocks
        self._original_workspaces = original_workspaces
        self._dropout_threshold = dropout_threshold
        # 어느 작업장에도 하드 제약상 배치 불가한 블록 인덱스.
        # 이런 블록은 agent에 묻지 않고(action mask가 전부 False가 되는 것을
        # 방지) 도착일에 즉시 탈락 처리한다.
        self._infeasible: set = set(infeasible_indices or ())

        self.blocks: List[Block] = []
        self.workspaces: List[Workspace] = []
        self.assignments: List[Optional[int]] = []
        self.delay_days: List[Optional[int]] = []
        self.pending: set[int] = set()
        self.env_date: date = date(2026, 4, 1)
        self.current_block_index: Optional[int] = None
        self._transition_results: List[PlacementStepResult] = []

        self.reset(original_blocks, original_workspaces)

    def reset(
        self,
        original_blocks: Optional[List[Block]] = None,
        original_workspaces: Optional[List[Workspace]] = None,
        infeasible_indices: Optional[set] = None,
    ) -> None:
        if original_blocks is not None:
            self._original_blocks = original_blocks
        if original_workspaces is not None:
            self._original_workspaces = original_workspaces
        if infeasible_indices is not None:
            self._infeasible = set(infeasible_indices)

        self.blocks = [b.clone() for b in self._original_blocks]
        self.workspaces = Workspace.deep_copy_list(self._original_workspaces)
        self.assignments = [None] * len(self.blocks)
        self.delay_days = [None] * len(self.blocks)
        self.pending = set(range(len(self.blocks)))
        self.current_block_index = None
        self._transition_results = []

        if self.blocks:
            earliest = min(b.in_date for b in self.blocks)
            self.env_date = cal.adjust_to_working_day(earliest, forward=True)
        else:
            self.env_date = date(2026, 4, 1)

        self._advance_to_next_decision()

    @property
    def is_done(self) -> bool:
        return self.current_block_index is None and not self.pending

    @property
    def assigned_count(self) -> int:
        return sum(1 for a in self.assignments if a is not None)

    @property
    def resolved_count(self) -> int:
        return len(self.blocks) - len(self.pending)

    @property
    def current_block(self) -> Optional[Block]:
        if self.current_block_index is None:
            return None
        return self.blocks[self.current_block_index]

    def assign_current(self, workspace_index: int) -> PlacementStepResult:
        if self.current_block_index is None:
            raise RuntimeError("No block is waiting for assignment.")

        block_index = self.current_block_index
        self.assignments[block_index] = int(workspace_index)
        result = self._process_assigned_block(block_index)
        self._record_result(result)
        self._advance_to_next_decision()
        return result

    def _record_result(self, result: PlacementStepResult) -> None:
        self._transition_results.append(result)

    def drain_transition_results(self) -> List[PlacementStepResult]:
        results = self._transition_results
        self._transition_results = []
        return results

    def result(self) -> SimulationResult:
        return SimulationResult(
            self.workspaces,
            self.blocks,
            [d if d is not None else SimulationResult.DROPOUT
             for d in self.delay_days],
        )

    def resolved_delay_days(self) -> List[int]:
        return [d for d in self.delay_days if d is not None]

    def current_delay_workdays(self, block_index: int) -> int:
        return max(
            cal.get_working_days_between(
                self._original_blocks[block_index].in_date,
                self.blocks[block_index].in_date,
            ) - 1,
            0,
        )

    def unassigned_block_indices(self) -> List[int]:
        indices = [
            idx for idx in self.pending
            if idx != self.current_block_index
            and idx not in self._infeasible
            and self.assignments[idx] is None
        ]
        return sorted(indices, key=self._sort_key)

    def upcoming_block_indices(self, k: int) -> List[int]:
        """다음에 '에이전트에게 물어볼' pending 블록 인덱스를 결정 순서로 최대 k개 반환.

        미래 블록 lookahead 관측용. 다음은 제외한다:
          - 현재 결정 대상 블록(current_block_index)
          - 하드 제약상 배치 불가로 자동 탈락하는 infeasible 블록
          - 이미 배정되었으나 배치 실패로 지연 대기 중인 블록
            (agent 결정 지점이 아니라 매일 자동 재시도되는 블록)

        미결정 pending 블록은 아직 지연되지 않아 _sort_key가 사실상 원래 착수일
        순서를 주므로, 실제 시뮬레이터가 블록을 제시하는 순서와 일치하는 근사
        lookahead를 제공한다. 지연으로 인한 순서 변동은 근사로 감수한다.
        """
        return self.unassigned_block_indices()[:max(k, 0)]

    def pending_assignment_indices(
        self, workspace_index: Optional[int] = None
    ) -> List[int]:
        indices = [
            idx for idx in self.pending
            if self.assignments[idx] is not None
            and self.delay_days[idx] is None
            and (
                workspace_index is None
                or self.assignments[idx] == workspace_index
            )
        ]
        return sorted(indices, key=self._sort_key)

    def _advance_to_next_decision(self) -> None:
        self.current_block_index = None

        while self.pending:
            for ws in self.workspaces:
                ws.clear_outgoing_blocks(self.env_date)

            today_targets = [
                idx for idx in self.pending
                if self.blocks[idx].in_date <= self.env_date
            ]

            if not today_targets:
                self.env_date = cal.next_working_day(self.env_date)
                continue

            today_targets.sort(key=self._sort_key)

            for idx in today_targets:
                if idx not in self.pending:
                    continue

                if self.assignments[idx] is None:
                    # 배치 가능한 작업장이 하나도 없는 블록은 agent에 묻지 않고
                    # 즉시 탈락 처리한다(action mask가 전부 False가 되는 것을 방지).
                    if idx in self._infeasible:
                        self.delay_days[idx] = SimulationResult.DROPOUT
                        self.pending.discard(idx)
                        self._record_result(
                            PlacementStepResult(
                                idx, -1, placed=False, delayed=False,
                                dropped=True, delay_days=SimulationResult.DROPOUT,
                            )
                        )
                        continue
                    self.current_block_index = idx
                    return

                result = self._process_assigned_block(idx)
                self._record_result(result)

            self.env_date = cal.next_working_day(self.env_date)

    def _sort_key(self, idx: int) -> Tuple[int, date, int]:
        delay = self.current_delay_workdays(idx)
        return (-delay, self._original_blocks[idx].in_date, idx)

    def _process_assigned_block(self, idx: int) -> PlacementStepResult:
        assignment = self.assignments[idx]
        if assignment is None:
            raise RuntimeError("Cannot process a block without assignment.")

        block = self.blocks[idx]
        cur_delay = self.current_delay_workdays(idx)

        if cur_delay > self._dropout_threshold:
            self.delay_days[idx] = SimulationResult.DROPOUT
            self.pending.discard(idx)
            return PlacementStepResult(
                idx, assignment, placed=False, delayed=False,
                dropped=True, delay_days=SimulationResult.DROPOUT,
            )

        workspace = self.workspaces[assignment]
        trial = block.clone()
        pos = workspace.determine_placement_position(trial, self.env_date)

        if pos is not None:
            cx, cy = pos
            block.move(cx - block.ref_x, cy - block.ref_y)
            workspace.add_block(block, self.env_date)
            self.delay_days[idx] = cur_delay
            self.pending.discard(idx)
            return PlacementStepResult(
                idx, assignment, placed=True, delayed=False,
                dropped=False, delay_days=cur_delay,
            )

        block.delay_placement(1)
        return PlacementStepResult(
            idx, assignment, placed=False, delayed=True,
            dropped=False, delay_days=None,
        )
