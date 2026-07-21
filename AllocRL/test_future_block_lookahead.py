"""미래 블록 lookahead (upcoming_block_indices) 순수 로직 테스트.

RL 의존성(torch/gym/numpy) 없이 base python에서 실행 가능:
    py test_future_block_lookahead.py

관측/extractor 레벨 테스트는 test_candidate_observation.py와
test_feature_extractors.py (torch/gym 필요) 참고.
"""

import unittest
from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from alloc_env import calendar as cal
from alloc_env.block import Block
from alloc_env.incremental_simulator import IncrementalPlacementSimulator
from alloc_env.observation_state import (
    ObservationScales,
    encode_current_block,
    encode_future_blocks,
    encode_future_demand,
)
from alloc_env.strategy import BaseGridStrategy
from alloc_env.workspace import Workspace


def add_workdays(start: date, count: int) -> date:
    value = start
    for _index in range(count):
        value = cal.next_working_day(value)
    return value


def make_encoder_scales() -> ObservationScales:
    return ObservationScales(
        max_length=10.0,
        max_breadth=5.0,
        max_duration=10,
        base_date=date(2026, 1, 5),
        date_span_workdays=100,
        max_workspace_area=5_000.0,
        total_workspace_area=50_000.0,
        max_workspace_length=100.0,
        max_workspace_breadth=50.0,
        dropout_threshold=7,
    )


def make_encoder_block(
    index: int,
    in_date: date,
    *,
    length: float = 10.0,
    breadth: float = 5.0,
) -> Block:
    return Block(
        name=f"E-{index}",
        ship_no="S-1",
        block_type="BUILD",
        length=length,
        breadth=breadth,
        height=1.0,
        weight=1.0,
        in_date=in_date,
        out_date=add_workdays(in_date, 9),
    )


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

    def test_equal_delays_order_by_original_in_date_then_block_index(self):
        sim = self._make_sim([
            date(2026, 4, 8),
            date(2026, 4, 6),
            date(2026, 4, 6),
        ])
        sim.current_block_index = None
        sim.pending = {0, 1, 2}
        for index in sim.pending:
            sim.blocks[index].delay_placement(1)

        self.assertEqual(sim.unassigned_block_indices(), [1, 2, 0])

    def test_greater_delay_outranks_earlier_original_in_date(self):
        sim = self._make_sim([
            date(2026, 4, 6),
            date(2026, 4, 8),
        ])
        sim.current_block_index = None
        sim.blocks[1].delay_placement(2)

        self.assertEqual(sim.unassigned_block_indices(), [1, 0])

    def test_none_current_block_index_excludes_no_pending_block(self):
        sim = self._make_sim([date(2026, 4, 6)] * 3)
        sim.current_block_index = None

        self.assertEqual(sim.unassigned_block_indices(), [0, 1, 2])


def test_current_block_feature_order_and_normalizers_are_exact():
    scales = make_encoder_scales()
    block = make_encoder_block(0, scales.base_date)
    env_date = add_workdays(scales.base_date, 50)

    encoded = encode_current_block(block, env_date, 456, scales)

    np.testing.assert_allclose(
        encoded,
        np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.01, 0.1], np.float32),
    )
    assert encoded.shape == (8,)
    assert encoded.dtype == np.float32


def test_current_block_clips_every_feature_and_does_not_mutate_block():
    scales = make_encoder_scales()
    block = make_encoder_block(
        0, scales.base_date, length=20.0, breadth=0.0
    )
    before = vars(block).copy()

    encoded = encode_current_block(block, scales.base_date, -1, scales)

    assert np.all((0.0 <= encoded) & (encoded <= 1.0))
    assert encoded[0] == 1.0
    assert encoded[1] == 0.0
    assert encoded[4] == 0.0
    assert encoded[5] == 0.0
    assert vars(block) == before


@pytest.mark.parametrize(
    ("mutation", "assigned_count"),
    [
        (lambda block: setattr(block, "length", float("nan")), 0),
        (lambda _block: None, float("inf")),
    ],
)
def test_current_block_rejects_nonfinite_features(mutation, assigned_count):
    block = make_encoder_block(0, date(2026, 1, 5))
    mutation(block)

    with pytest.raises(ValueError, match="finite"):
        encode_current_block(
            block, date(2026, 1, 5), assigned_count, make_encoder_scales()
        )


def test_future_blocks_preserve_passed_order_mask_and_zero_padding():
    base = date(2026, 1, 5)
    scales = make_encoder_scales()
    blocks = [
        make_encoder_block(0, add_workdays(base, 1), length=2.0, breadth=1.0),
        make_encoder_block(1, add_workdays(base, 8), length=4.0, breadth=2.0),
        make_encoder_block(2, add_workdays(base, 15), length=6.0, breadth=3.0),
    ]
    indices = [2, 0]
    before_blocks = [vars(block).copy() for block in blocks]
    before_indices = list(indices)

    encoded, mask = encode_future_blocks(blocks, indices, base, scales)

    np.testing.assert_allclose(
        encoded[0],
        np.array([0.6, 0.6, 1.0, 0.5, 0.5, 0.0036], np.float32),
    )
    np.testing.assert_allclose(
        encoded[1],
        np.array([0.2, 0.2, 1.0, 1 / 30, 0.5, 0.0004], np.float32),
    )
    np.testing.assert_array_equal(mask[:2], np.ones(2, np.float32))
    assert np.count_nonzero(encoded[2:]) == 0
    assert np.count_nonzero(mask[2:]) == 0
    assert encoded.shape == (16, 6)
    assert mask.shape == (16,)
    assert encoded.dtype == mask.dtype == np.float32
    assert [vars(block) for block in blocks] == before_blocks
    assert indices == before_indices


def test_future_blocks_truncate_to_first_16_indices():
    base = date(2026, 1, 5)
    blocks = [make_encoder_block(i, base) for i in range(20)]

    encoded, mask = encode_future_blocks(
        blocks, list(range(19, -1, -1)), base, make_encoder_scales()
    )

    assert mask.sum() == 16
    assert encoded[0, 0] == pytest.approx(blocks[19].length / 10.0)


@pytest.mark.parametrize("encoder", [encode_future_blocks, encode_future_demand])
@pytest.mark.parametrize(
    ("indices", "error", "message"),
    [
        ([-1], IndexError, "out of range"),
        ([2], IndexError, "out of range"),
        ([True], TypeError, "integer"),
        ([0.5], TypeError, "integer"),
        ([0, 0], ValueError, "duplicate"),
    ],
)
def test_future_encoders_reject_invalid_indices(
    encoder, indices, error, message
):
    base = date(2026, 1, 5)
    blocks = [make_encoder_block(index, base) for index in range(2)]

    with pytest.raises(error, match=message):
        encoder(blocks, indices, base, make_encoder_scales())


def test_future_blocks_validates_indices_beyond_encoded_limit():
    base = date(2026, 1, 5)
    blocks = [make_encoder_block(index, base) for index in range(16)]

    with pytest.raises(IndexError, match="out of range"):
        encode_future_blocks(
            blocks, list(range(16)) + [16], base, make_encoder_scales()
        )


def test_future_encoders_accept_numpy_integral_indices_in_caller_order():
    base = date(2026, 1, 5)
    blocks = [
        make_encoder_block(0, base, length=2.0),
        make_encoder_block(1, base, length=6.0),
    ]
    indices = [np.int64(1), np.int32(0)]

    encoded, mask = encode_future_blocks(
        blocks, indices, base, make_encoder_scales()
    )
    demand = encode_future_demand(
        blocks, indices, base, make_encoder_scales()
    )

    np.testing.assert_allclose(encoded[:2, 0], [0.6, 0.2])
    np.testing.assert_array_equal(mask[:2], np.ones(2, np.float32))
    assert demand[0, 0] == pytest.approx(2 / 913)
    np.testing.assert_array_equal(indices, [np.int64(1), np.int32(0)])


def test_future_blocks_reject_nonfinite_features():
    base = date(2026, 1, 5)
    block = make_encoder_block(0, base)
    block.breadth = float("inf")

    with pytest.raises(ValueError, match="finite"):
        encode_future_blocks([block], [0], base, make_encoder_scales())


def test_future_working_day_windows_include_exact_boundaries():
    base = date(2026, 1, 5)
    arrivals = [0, 5, 6, 20, 21, 60, 61]
    blocks = [
        make_encoder_block(index, add_workdays(base, offset))
        for index, offset in enumerate(arrivals)
    ]
    before = [vars(block).copy() for block in blocks]

    demand = encode_future_demand(
        blocks, list(range(len(blocks))), base, make_encoder_scales()
    )

    expected_row = np.array(
        [2 / 913, 100 / 200_000, 1.0, 0.01, 1.0, 1.0],
        np.float32,
    )
    np.testing.assert_allclose(demand, np.tile(expected_row, (3, 1)))
    assert demand.shape == (3, 6)
    assert demand.dtype == np.float32
    assert [vars(block) for block in blocks] == before


def test_future_demand_preserves_length_and_breadth_for_equal_area_blocks():
    base = date(2026, 1, 5)
    scales = replace(
        make_encoder_scales(), max_length=20.0, max_breadth=20.0
    )
    horizontal = make_encoder_block(
        0, base, length=10.0, breadth=5.0
    )
    vertical = make_encoder_block(
        1, base, length=5.0, breadth=10.0
    )
    horizontal_before = vars(horizontal).copy()
    vertical_before = vars(vertical).copy()

    horizontal_demand = encode_future_demand(
        [horizontal], [0], base, scales
    )
    vertical_demand = encode_future_demand([vertical], [0], base, scales)

    assert horizontal_demand[0, 3] == pytest.approx(vertical_demand[0, 3])
    np.testing.assert_allclose(horizontal_demand[0, 4:], [0.5, 0.25])
    np.testing.assert_allclose(vertical_demand[0, 4:], [0.25, 0.5])
    assert vars(horizontal) == horizontal_before
    assert vars(vertical) == vertical_before


def test_future_demand_empty_windows_are_zero_and_uses_only_passed_indices():
    base = date(2026, 1, 5)
    blocks = [
        make_encoder_block(0, add_workdays(base, 2)),
        make_encoder_block(1, add_workdays(base, 10)),
        make_encoder_block(2, add_workdays(base, 30)),
    ]

    demand = encode_future_demand(blocks, [1], base, make_encoder_scales())

    assert np.count_nonzero(demand[0]) == 0
    assert demand[1, 0] == pytest.approx(1 / 913)
    assert np.count_nonzero(demand[2]) == 0
    assert np.all((0.0 <= demand) & (demand <= 1.0))


def test_future_demand_rejects_nonfinite_features():
    base = date(2026, 1, 5)
    block = make_encoder_block(0, base)
    block.length = float("inf")

    with pytest.raises(ValueError, match="finite"):
        encode_future_demand([block], [0], base, make_encoder_scales())


if __name__ == "__main__":
    unittest.main()
