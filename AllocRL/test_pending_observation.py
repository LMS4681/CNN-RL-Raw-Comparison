import unittest
from datetime import date

from alloc_env.block import Block
from alloc_env.incremental_simulator import IncrementalPlacementSimulator
from alloc_env.strategy import BaseGridStrategy
from alloc_env.workspace import Workspace


def make_queue_simulator(block_count: int = 4) -> IncrementalPlacementSimulator:
    blocks = [
        Block(
            name=f"B-{index}",
            ship_no="S-1",
            block_type="BUILD",
            length=5.0,
            breadth=5.0,
            height=1.0,
            weight=1.0,
            in_date=date(2026, 1, 5),
            out_date=date(2026, 1, 20),
        )
        for index in range(block_count)
    ]
    workspaces = [
        Workspace(
            code=f"W-{index}",
            origin_x=0.0,
            origin_y=0.0,
            length=100.0,
            breadth=100.0,
            strategy=BaseGridStrategy(step=1.0),
        )
        for index in range(2)
    ]
    return IncrementalPlacementSimulator(blocks, workspaces, 7)


class PendingAssignmentIndicesTests(unittest.TestCase):
    def test_pending_assignments_are_grouped_and_retry_sorted(self):
        simulator = make_queue_simulator()
        simulator.assignments[:] = [1, 0, 1, None]
        simulator.pending = {0, 1, 2, 3}
        simulator.blocks[0].delay_placement(2)
        simulator.blocks[2].delay_placement(2)

        self.assertEqual(simulator.pending_assignment_indices(1), [0, 2])
        self.assertEqual(simulator.pending_assignment_indices(0), [1])
        self.assertEqual(simulator.current_delay_workdays(0), 2)

    def test_pending_assignments_exclude_resolved_blocks(self):
        simulator = make_queue_simulator()
        simulator.assignments[:] = [1, 1, 1, None]
        simulator.delay_days[1] = 0

        self.assertEqual(simulator.pending_assignment_indices(), [0, 2])

    def test_current_delay_never_returns_negative(self):
        simulator = make_queue_simulator()
        simulator.blocks[0].in_date = date(2026, 1, 2)

        self.assertEqual(simulator.current_delay_workdays(0), 0)


if __name__ == "__main__":
    unittest.main()
