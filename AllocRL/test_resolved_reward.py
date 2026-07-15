import unittest
from datetime import date

from alloc_env.alloc_env import BlockPlacementEnv
from alloc_env.block import Block
from alloc_env.strategy import BaseGridStrategy
from alloc_env.workspace import Workspace


def block(name: str, day: int, out_day: int = 30) -> Block:
    return Block(
        name=name,
        ship_no="T001",
        block_type="BUILD",
        length=10.0,
        breadth=10.0,
        height=5.0,
        weight=10.0,
        in_date=date(2026, 1, day),
        out_date=date(2026, 1, out_day),
    )


def workspace() -> Workspace:
    return Workspace(
        code="PE001",
        origin_x=0.0,
        origin_y=0.0,
        length=100.0,
        breadth=100.0,
        strategy=BaseGridStrategy(step=10.0),
    )


class ResolvedRewardTests(unittest.TestCase):
    def test_static_environment_rejects_no_agent_decisions(self):
        oversized = block("OVERSIZE", 5)
        oversized.length = 500.0
        oversized.breadth = 500.0

        with self.assertRaisesRegex(ValueError, "no agent decision"):
            BlockPlacementEnv([oversized], [workspace()], grid_size=32)

    def test_episode_rewards_sum_to_terminal_score(self):
        env = BlockPlacementEnv(
            [block("A", 5), block("B", 6), block("C", 7)],
            [workspace()],
            grid_size=32,
        )
        env.reset(seed=7)
        total = 0.0
        done = False
        while not done:
            _, reward, done, _, info = env.step(0)
            total += reward

        self.assertAlmostEqual(info["terminal_score"], total, places=6)
        self.assertAlmostEqual(info["episode_reward"], total, places=6)
        self.assertAlmostEqual(
            info["terminal_score"],
            info["resolved_reward"] + info["terminal_residual"],
            places=6,
        )

    def test_resolved_block_is_emitted_once(self):
        env = BlockPlacementEnv(
            [block("A", 5), block("B", 6)],
            [workspace()],
            grid_size=32,
        )
        env.reset(seed=9)
        emitted = set()
        done = False
        while not done:
            _, _, done, _, info = env.step(0)
            current = set(info.get("newly_resolved_indices", []))
            self.assertTrue(emitted.isdisjoint(current))
            emitted.update(current)

        self.assertEqual({0, 1}, emitted)


if __name__ == "__main__":
    unittest.main()
