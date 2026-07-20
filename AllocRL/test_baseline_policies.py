import unittest
from datetime import date

import numpy as np

from alloc_env.alloc_env import BlockPlacementEnv
from alloc_env.observation_state import ObservationScales
from alloc_env.strategy import BaseGridStrategy
from baseline_policies import GreedyImmediateAreaPolicy, RandomValidPolicy
from test_evaluation_scenarios import make_blocks, make_plain_workspace


class StubEnv:
    def __init__(self, hard_valid, immediate, free_areas):
        self._hard_valid = np.asarray(hard_valid, dtype=bool)
        self._immediate = np.asarray(immediate, dtype=bool)
        self._free_areas = np.asarray(free_areas, dtype=np.float32)

    def action_masks(self):
        return self._hard_valid.copy()

    def immediate_placeability(self):
        return self._immediate.copy()

    def workspace_free_areas(self):
        return self._free_areas.copy()


class BaselinePolicyTests(unittest.TestCase):
    def test_random_policy_is_seeded_and_never_selects_masked_actions(self):
        env = StubEnv(
            hard_valid=[True, False, True],
            immediate=[False, False, True],
            free_areas=[100.0, 999.0, 50.0],
        )
        first = RandomValidPolicy(seed=7)
        second = RandomValidPolicy(seed=7)

        first_actions = [first.select_action(env, {}) for _ in range(50)]
        second_actions = [second.select_action(env, {}) for _ in range(50)]

        self.assertEqual(first_actions, second_actions)
        self.assertLessEqual(set(first_actions), {0, 2})

    def test_random_policy_rejects_a_state_without_hard_valid_actions(self):
        env = StubEnv([False, False], [False, False], [10.0, 20.0])

        with self.assertRaisesRegex(RuntimeError, "no hard-valid action"):
            RandomValidPolicy(seed=0).select_action(env, {})

    def test_greedy_prefers_immediate_candidate_over_larger_waiting_space(self):
        env = StubEnv(
            hard_valid=[True, False, True],
            immediate=[False, False, True],
            free_areas=[100.0, 999.0, 50.0],
        )

        action = GreedyImmediateAreaPolicy().select_action(env, {})

        self.assertEqual(2, action)

    def test_greedy_chooses_largest_immediately_placeable_hard_valid_area(self):
        env = StubEnv(
            hard_valid=[True, False, True, True],
            immediate=[True, True, True, False],
            free_areas=[40.0, 999.0, 80.0, 500.0],
        )

        action = GreedyImmediateAreaPolicy().select_action(env, {})

        self.assertEqual(2, action)

    def test_greedy_falls_back_to_largest_hard_valid_area(self):
        env = StubEnv(
            hard_valid=[True, False, True],
            immediate=[False, True, False],
            free_areas=[40.0, 999.0, 80.0],
        )

        action = GreedyImmediateAreaPolicy().select_action(env, {})

        self.assertEqual(2, action)

    def test_greedy_rejects_a_state_without_hard_valid_actions(self):
        env = StubEnv([False, False], [True, True], [10.0, 20.0])

        with self.assertRaisesRegex(RuntimeError, "no hard-valid action"):
            GreedyImmediateAreaPolicy().select_action(env, {})

    def test_environment_diagnostics_do_not_mutate_or_extend_observation(self):
        env = BlockPlacementEnv(
            make_blocks(),
            [
                make_plain_workspace("PE001", 12.0, 12.0),
                make_plain_workspace("PE002", 100.0, 100.0),
            ],
            BaseGridStrategy(step=1.0),
            use_synthetic=False,
            grid_size=32,
            observation_scales=ObservationScales(
                max_length=100.0,
                max_breadth=100.0,
                max_duration=365,
                base_date=date(2026, 1, 1),
                date_span_workdays=365,
                max_workspace_area=10_000.0,
                total_workspace_area=20_000.0,
                max_workspace_length=100.0,
                max_workspace_breadth=100.0,
                dropout_threshold=7,
            ),
        )
        try:
            observation, _ = env.reset(seed=0)
            observation_keys = set(observation)
            state_before = (
                env._current_step,
                list(env._assignments),
                env._ws_used_area.copy(),
                [len(workspace.blocks) for workspace in env._workspaces],
            )

            placeability = env.immediate_placeability()
            free_areas = env.workspace_free_areas()

            self.assertEqual(observation_keys, set(env.observation_space.spaces))
            self.assertEqual(observation_keys, set(observation))
            np.testing.assert_array_equal(
                placeability, observation["ws_meta"][:, 3].astype(bool)
            )
            np.testing.assert_allclose(
                free_areas,
                np.maximum(env._ws_areas - env._ws_used_area, 0.0),
            )
            self.assertEqual(state_before[0], env._current_step)
            self.assertEqual(state_before[1], env._assignments)
            np.testing.assert_array_equal(state_before[2], env._ws_used_area)
            self.assertEqual(
                state_before[3],
                [len(workspace.blocks) for workspace in env._workspaces],
            )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
