"""Evaluation policies that do not depend on a trained model."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class ActionPolicy(Protocol):
    name: str

    def select_action(self, env, observation) -> int:
        """Select one workspace action for the current evaluation state."""
        raise NotImplementedError


class RandomValidPolicy:
    name = "random_valid"

    def __init__(self, seed: int):
        self._rng = np.random.default_rng(seed)

    def select_action(self, env, observation) -> int:
        valid = np.flatnonzero(env.action_masks())
        if not len(valid):
            raise RuntimeError("Evaluation state has no hard-valid action")
        return int(self._rng.choice(valid))


class GreedyImmediateAreaPolicy:
    name = "greedy_immediate_area"

    def select_action(self, env, observation) -> int:
        hard_valid = np.asarray(env.action_masks(), dtype=bool)
        immediate = np.asarray(env.immediate_placeability(), dtype=bool)
        free_area = np.asarray(env.workspace_free_areas(), dtype=np.float64)
        preferred = np.flatnonzero(hard_valid & immediate)
        candidates = preferred if len(preferred) else np.flatnonzero(hard_valid)
        if not len(candidates):
            raise RuntimeError("Evaluation state has no hard-valid action")
        return int(candidates[np.argmax(free_area[candidates])])
