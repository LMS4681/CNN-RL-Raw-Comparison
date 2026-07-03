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

    It uses the same placement, rotation, delay, and dropout rules as
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
        self.last_internal_results: List[PlacementStepResult] = []

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
        self.last_internal_results = []

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
        self.last_internal_results = [result]
        self._advance_to_next_decision()
        return result

    def result(self) -> SimulationResult:
        return SimulationResult(
            self.workspaces,
            self.blocks,
            [d if d is not None else SimulationResult.DROPOUT
             for d in self.delay_days],
        )

    def resolved_delay_days(self) -> List[int]:
        return [d for d in self.delay_days if d is not None]

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
                        self.last_internal_results.append(
                            PlacementStepResult(
                                idx, -1, placed=False, delayed=False,
                                dropped=True, delay_days=SimulationResult.DROPOUT,
                            )
                        )
                        continue
                    self.current_block_index = idx
                    return

                result = self._process_assigned_block(idx)
                self.last_internal_results.append(result)

            self.env_date = cal.next_working_day(self.env_date)

    def _sort_key(self, idx: int) -> Tuple[int, date]:
        delay = cal.get_working_days_between(
            self._original_blocks[idx].in_date,
            self.blocks[idx].in_date,
        ) - 1
        return (-delay, self._original_blocks[idx].in_date)

    def _process_assigned_block(self, idx: int) -> PlacementStepResult:
        assignment = self.assignments[idx]
        if assignment is None:
            raise RuntimeError("Cannot process a block without assignment.")

        block = self.blocks[idx]
        cur_delay = cal.get_working_days_between(
            self._original_blocks[idx].in_date,
            block.in_date,
        ) - 1

        if cur_delay > self._dropout_threshold:
            self.delay_days[idx] = SimulationResult.DROPOUT
            self.pending.discard(idx)
            return PlacementStepResult(
                idx, assignment, placed=False, delayed=False,
                dropped=True, delay_days=SimulationResult.DROPOUT,
            )

        workspace = self.workspaces[assignment]
        pos = workspace.determine_placement_position(block, self.env_date)

        if pos is None:
            block.turn()
            pos = workspace.determine_placement_position(block, self.env_date)
            if pos is None:
                block.turn()

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
