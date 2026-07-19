"""미래 블록 lookahead (upcoming_block_indices) 순수 로직 테스트.

RL 의존성(torch/gym/numpy) 없이 base python에서 실행 가능:
    py test_future_block_lookahead.py

관측/extractor 레벨 테스트는 test_candidate_observation.py와
test_feature_extractors.py (torch/gym 필요) 참고.
"""

import unittest
from datetime import date

from alloc_env.block import Block
from alloc_env.workspace import Workspace
from alloc_env.strategy import BaseGridStrategy
from alloc_env.incremental_simulator import IncrementalPlacementSimulator


def make_block(name: str, in_date: date) -> Block:
    return Block(
        name=name,
        ship_no="T001",
        block_type="BUILD",
        length=10.0,
        breadth=10.0,
        height=5.0,
        weight=10.0,
        in_date=in_date,
        out_date=date(2026, 5, 30),
    )


def make_workspace(code: str = "PE001") -> Workspace:
    return Workspace(
        code=code,
        origin_x=0.0,
        origin_y=0.0,
        breadth=100.0,
        length=100.0,
        strategy=BaseGridStrategy(step=10.0),
    )


class UpcomingBlockIndicesTests(unittest.TestCase):
    def _make_sim(self, in_dates, infeasible=None):
        blocks = [make_block(f"B{i}", d) for i, d in enumerate(in_dates)]
        workspaces = [make_workspace()]
        return IncrementalPlacementSimulator(
            blocks,
            workspaces,
            dropout_threshold=7,
            infeasible_indices=infeasible,
        )

    def test_k_zero_or_negative_returns_empty(self):
        sim = self._make_sim([date(2026, 4, 6)] * 3)
        self.assertEqual(sim.upcoming_block_indices(0), [])
        self.assertEqual(sim.upcoming_block_indices(-1), [])

    def test_excludes_current_and_orders_by_in_date(self):
        in_dates = [
            date(2026, 4, 6),
            date(2026, 4, 7),
            date(2026, 4, 8),
            date(2026, 4, 9),
            date(2026, 4, 10),
        ]
        sim = self._make_sim(in_dates)
        current = sim.current_block_index
        self.assertIsNotNone(current)

        upcoming = sim.upcoming_block_indices(3)
        self.assertEqual(len(upcoming), 3)
        self.assertNotIn(current, upcoming)
        # 결정 순서 = in_date 비내림차순
        dates = [sim.blocks[i].in_date for i in upcoming]
        self.assertEqual(dates, sorted(dates))
        # 중복 없음
        self.assertEqual(len(set(upcoming)), len(upcoming))

    def test_k_larger_than_remaining_clamps(self):
        # 2블록 중 1개가 current → 남은 후보 1개, 예외 없이 clamp
        sim = self._make_sim([date(2026, 4, 6), date(2026, 4, 7)])
        upcoming = sim.upcoming_block_indices(10)
        self.assertEqual(len(upcoming), 1)
        self.assertNotIn(sim.current_block_index, upcoming)

    def test_excludes_infeasible(self):
        in_dates = [
            date(2026, 4, 6),
            date(2026, 4, 7),
            date(2026, 4, 8),
            date(2026, 4, 9),
        ]
        # 인덱스 2를 하드 제약상 배치 불가로 표시 (자동 탈락 대상)
        sim = self._make_sim(in_dates, infeasible={2})
        upcoming = sim.upcoming_block_indices(5)
        self.assertNotIn(2, upcoming)

    def test_excludes_already_assigned_pending(self):
        in_dates = [
            date(2026, 4, 6),
            date(2026, 4, 7),
            date(2026, 4, 8),
            date(2026, 4, 9),
        ]
        sim = self._make_sim(in_dates)
        current = sim.current_block_index
        # current가 아닌 pending 하나를 '배정됨(지연 대기)' 상태로 모사
        others = [i for i in sim.pending if i != current]
        assigned_idx = others[0]
        sim.assignments[assigned_idx] = 0
        upcoming = sim.upcoming_block_indices(5)
        self.assertNotIn(assigned_idx, upcoming)

    def test_unassigned_ties_use_block_index(self):
        sim = self._make_sim([date(2026, 4, 6)] * 4)
        sim.current_block_index = 0
        sim.pending = {3, 2, 1, 0}

        self.assertEqual(sim.unassigned_block_indices(), [1, 2, 3])
        self.assertEqual(sim.upcoming_block_indices(2), [1, 2])

    def test_next_decision_uses_unassigned_queue_order(self):
        sim = self._make_sim([date(2026, 4, 6)] * 3)
        sim.blocks[1].delay_placement(1)
        sim.blocks[2].delay_placement(1)

        self.assertEqual(sim.unassigned_block_indices(), [1, 2])
        sim.assign_current(0)
        self.assertEqual(sim.current_block_index, 1)


if __name__ == "__main__":
    unittest.main()
