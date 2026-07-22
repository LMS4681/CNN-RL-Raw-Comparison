from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from alloc_env.alloc_env import BlockPlacementEnv, DROPOUT_THRESHOLD
from alloc_env.observation_state import N_WORKSPACES, ORDERED_FUTURE_COUNT
from alloc_env.simulator import SimulationResult


@dataclass(frozen=True)
class AuxiliaryTargets:
    action_mask: np.ndarray
    current_placeable: np.ndarray
    future_fit: np.ndarray
    future_optionality_after: np.ndarray
    future_optionality_delta: np.ndarray
    largest_free_rectangle_ratio: np.ndarray
    free_component_count_normalized: np.ndarray
    replay_success_rate: np.ndarray
    replay_dropout_rate: np.ndarray
    replay_delay_ratio: np.ndarray
    replay_mask: np.ndarray


def _largest_free_rectangle(free: np.ndarray) -> int:
    heights = np.zeros(free.shape[1], dtype=np.int32)
    largest = 0
    for row in free:
        heights = np.where(row, heights + 1, 0)
        stack: list[int] = []
        for column in range(len(heights) + 1):
            height = int(heights[column]) if column < len(heights) else 0
            while stack and int(heights[stack[-1]]) > height:
                top = stack.pop()
                width = column if not stack else column - stack[-1] - 1
                largest = max(largest, int(heights[top]) * width)
            stack.append(column)
    return largest


def _free_component_count(free: np.ndarray) -> int:
    visited = np.zeros_like(free, dtype=bool)
    components = 0
    rows, columns = free.shape
    for row in range(rows):
        for column in range(columns):
            if not free[row, column] or visited[row, column]:
                continue
            components += 1
            visited[row, column] = True
            queue = deque([(row, column)])
            while queue:
                current_row, current_column = queue.popleft()
                for next_row, next_column in (
                    (current_row - 1, current_column),
                    (current_row + 1, current_column),
                    (current_row, current_column - 1),
                    (current_row, current_column + 1),
                ):
                    if (
                        0 <= next_row < rows
                        and 0 <= next_column < columns
                        and free[next_row, next_column]
                        and not visited[next_row, next_column]
                    ):
                        visited[next_row, next_column] = True
                        queue.append((next_row, next_column))
    return components


def grid_geometry_features(occupied: np.ndarray) -> tuple[float, float]:
    occupied_array = np.asarray(occupied, dtype=bool)
    if occupied_array.ndim != 2 or 0 in occupied_array.shape:
        raise ValueError("occupied grid must be a non-empty rank-two array")
    free = ~occupied_array
    cell_count = int(free.size)
    return (
        _largest_free_rectangle(free) / cell_count,
        _free_component_count(free) / cell_count,
    )


def _preview_workspace(env: BlockPlacementEnv, action: int):
    workspace = env._workspaces[action].deep_copy()
    candidate = env._candidate_placements[action]
    current = env._placement_simulator.current_block
    if current is not None and candidate.position is not None:
        block = current.clone()
        center_x, center_y = candidate.position
        block.move(center_x - block.ref_x, center_y - block.ref_y)
        workspace.add_block(block, env._env_date)
    return workspace


def _future_fit(
    env: BlockPlacementEnv,
    action: int,
    future_indices: list[int],
) -> np.ndarray:
    values = np.zeros(ORDERED_FUTURE_COUNT, dtype=np.float32)
    workspace = _preview_workspace(env, action)
    for slot, block_index in enumerate(
        future_indices[:ORDERED_FUTURE_COUNT]
    ):
        block = env._blocks[block_index]
        if not all(
            constraint.is_feasible(block, workspace)
            for constraint in env._constraints
        ):
            continue
        trial = block.clone()
        values[slot] = float(
            workspace.determine_placement_position(
                trial, env._env_date
            )
            is not None
        )
    return values


def _teacher_action(env: BlockPlacementEnv) -> int:
    valid_actions = np.flatnonzero(env.action_masks())
    if len(valid_actions) == 0:
        raise RuntimeError("diagnostic replay reached a decision without actions")
    free_areas = env.workspace_free_areas()
    placeable = env.immediate_placeability()
    current = env._placement_simulator.current_block
    current_area = (
        current.length * current.breadth if current is not None else 0.0
    )
    ranked = []
    for action_value in valid_actions:
        action = int(action_value)
        optionality = env.future_workspace_choice_count_after_action(action)
        free_after = float(free_areas[action])
        if placeable[action]:
            free_after = max(free_after - current_area, 0.0)
        ranked.append((optionality, free_after, -action, action))
    return max(ranked)[-1]


def _bounded_replay(
    env: BlockPlacementEnv,
    action: int,
) -> tuple[float, float, float]:
    clone = env.clone_for_diagnostics()
    initial_index = clone._placement_simulator.current_block_index
    _, _, terminated, _, info = clone.step(action)
    resolved: list[int] = []
    seen: set[int] = set()

    def record(indices) -> None:
        for block_index in indices:
            if block_index == initial_index or block_index in seen:
                continue
            if clone._placement_simulator.delay_days[block_index] is None:
                continue
            seen.add(block_index)
            resolved.append(block_index)
            if len(resolved) == 8:
                return

    record(info.get("newly_resolved_indices", ()))
    decisions = 0
    while not terminated and len(resolved) < 8 and decisions < 32:
        teacher_action = _teacher_action(clone)
        _, _, terminated, _, info = clone.step(teacher_action)
        decisions += 1
        record(info.get("newly_resolved_indices", ()))

    delays = [
        clone._placement_simulator.delay_days[index]
        for index in resolved
    ]
    if not delays:
        return 0.0, 0.0, 0.0
    dropouts = sum(delay == SimulationResult.DROPOUT for delay in delays)
    successes = len(delays) - dropouts
    delay_total = sum(
        max(int(delay), 0)
        for delay in delays
        if delay != SimulationResult.DROPOUT
    )
    return (
        successes / len(delays),
        dropouts / len(delays),
        min(delay_total / (8 * DROPOUT_THRESHOLD), 1.0),
    )


def build_auxiliary_targets(
    env: BlockPlacementEnv,
    *,
    include_replay: bool,
) -> AuxiliaryTargets:
    simulator = env._placement_simulator
    if simulator is None or simulator.current_block is None:
        raise RuntimeError("environment must be at an active decision")
    if env.action_space.n != N_WORKSPACES:
        raise ValueError(
            f"pretraining targets require {N_WORKSPACES} workspaces"
        )

    observation = env._get_obs()
    action_mask = env.action_masks().astype(bool, copy=True)
    current_placeable = np.zeros(N_WORKSPACES, dtype=np.float32)
    future_fit = np.zeros(
        (N_WORKSPACES, ORDERED_FUTURE_COUNT), dtype=np.float32
    )
    future_optionality_after = np.zeros(N_WORKSPACES, dtype=np.float32)
    future_optionality_delta = np.zeros(N_WORKSPACES, dtype=np.float32)
    largest_rectangle = np.zeros(N_WORKSPACES, dtype=np.float32)
    component_count = np.zeros(N_WORKSPACES, dtype=np.float32)
    replay_success = np.zeros(N_WORKSPACES, dtype=np.float32)
    replay_dropout = np.zeros(N_WORKSPACES, dtype=np.float32)
    replay_delay = np.zeros(N_WORKSPACES, dtype=np.float32)
    replay_mask = np.zeros(N_WORKSPACES, dtype=bool)

    future_indices = env.future_workspace_choice_indices()
    denominator = ORDERED_FUTURE_COUNT * N_WORKSPACES
    baseline_optionality = env.future_workspace_choice_count(future_indices)
    placeability = env.immediate_placeability()

    for action_value in np.flatnonzero(action_mask):
        action = int(action_value)
        current_placeable[action] = float(placeability[action])
        future_fit[action] = _future_fit(env, action, future_indices)
        after = env.future_workspace_choice_count_after_action(
            action, future_indices
        )
        future_optionality_after[action] = after / denominator
        future_optionality_delta[action] = (
            after - baseline_optionality
        ) / denominator

        occupied = np.logical_or(
            observation["grids"][action, 0] > 0.0,
            observation["grids"][action, 3] > 0.0,
        )
        rectangle, components = grid_geometry_features(occupied)
        largest_rectangle[action] = rectangle
        component_count[action] = components

        if include_replay:
            (
                replay_success[action],
                replay_dropout[action],
                replay_delay[action],
            ) = _bounded_replay(env, action)
            replay_mask[action] = True

    return AuxiliaryTargets(
        action_mask=action_mask,
        current_placeable=current_placeable,
        future_fit=future_fit,
        future_optionality_after=future_optionality_after,
        future_optionality_delta=future_optionality_delta,
        largest_free_rectangle_ratio=largest_rectangle,
        free_component_count_normalized=component_count,
        replay_success_rate=replay_success,
        replay_dropout_rate=replay_dropout,
        replay_delay_ratio=replay_delay,
        replay_mask=replay_mask,
    )
