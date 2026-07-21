from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date, datetime
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np

from . import calendar as cal
from .block import Block
from .incremental_simulator import IncrementalPlacementSimulator
from .workspace import Workspace


N_WORKSPACES = 10
GRID_SIZE = 64
EPISODE_BLOCK_COUNT = 913
ORDERED_FUTURE_COUNT = 16
PENDING_QUEUE_SLOTS = 32
FUTURE_DAY_WINDOWS = ((0, 5), (6, 20), (21, 60))
FUTURE_DAY_NORMALIZER = 30
GRID_LIFETIME_NORMALIZER = 60

CURRENT_BLOCK_FEATURE_DIM = 8
FUTURE_BLOCK_FEATURE_DIM = 6
FUTURE_DEMAND_FEATURE_DIM = 6
PENDING_BLOCK_FEATURE_DIM = 7
PENDING_SUMMARY_FEATURE_DIM = 4
WORKSPACE_META_FEATURE_DIM = 8

_FLOAT_SCALE_FIELDS = (
    "max_length",
    "max_breadth",
    "max_workspace_area",
    "total_workspace_area",
    "max_workspace_length",
    "max_workspace_breadth",
)
_INTEGER_SCALE_FIELDS = (
    "max_duration",
    "date_span_workdays",
    "dropout_threshold",
)
_OBSERVATION_SCALE_FIELDS = (
    "max_length",
    "max_breadth",
    "max_duration",
    "base_date",
    "date_span_workdays",
    "max_workspace_area",
    "total_workspace_area",
    "max_workspace_length",
    "max_workspace_breadth",
    "dropout_threshold",
)


@dataclass(frozen=True)
class ObservationScales:
    max_length: float
    max_breadth: float
    max_duration: int
    base_date: date
    date_span_workdays: int
    max_workspace_area: float
    total_workspace_area: float
    max_workspace_length: float
    max_workspace_breadth: float
    dropout_threshold: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.base_date, date)
            or isinstance(self.base_date, datetime)
        ):
            raise TypeError("base_date must be a date, not datetime")
        for field_name in _FLOAT_SCALE_FIELDS:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{field_name} must be a real number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be finite and positive")
            object.__setattr__(self, field_name, float(value))
        for field_name in _INTEGER_SCALE_FIELDS:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{field_name} must be an integer")
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
            object.__setattr__(self, field_name, int(value))

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["base_date"] = self.base_date.isoformat()
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ObservationScales":
        if not isinstance(values, Mapping):
            raise TypeError("ObservationScales values must be a mapping")
        expected = set(_OBSERVATION_SCALE_FIELDS)
        actual = set(values)
        missing = sorted(expected - actual)
        if missing:
            raise ValueError(
                f"missing ObservationScales fields: {', '.join(missing)}"
            )
        unexpected = sorted(str(field_name) for field_name in actual - expected)
        if unexpected:
            raise ValueError(
                f"unexpected ObservationScales fields: {', '.join(unexpected)}"
            )

        serialized_date = values["base_date"]
        if not isinstance(serialized_date, str):
            raise TypeError("base_date must be an ISO date string")
        try:
            parsed_date = date.fromisoformat(serialized_date)
        except ValueError as error:
            raise ValueError("base_date must be an ISO date string") from error
        parsed = dict(values)
        parsed["base_date"] = parsed_date
        return cls(**parsed)


def _clip01(value: float) -> np.float32:
    if not math.isfinite(value):
        raise ValueError("observation feature values must be finite")
    return np.float32(np.clip(value, 0.0, 1.0))


def working_days_until(start: date, end: date) -> int:
    if end <= start:
        return 0
    return max(cal.get_working_days_between(start, end) - 1, 0)


def working_day_position(start: date, current: date) -> int:
    return working_days_until(start, current)


def _feature_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray([_clip01(value) for value in values], dtype=np.float32)


def _validate_block_indices(
    blocks: Sequence[Block], indices: Sequence[int]
) -> tuple[int, ...]:
    validated = []
    seen = set()
    for position, index in enumerate(indices):
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError(f"indices[{position}] must be an integer")
        normalized = int(index)
        if not 0 <= normalized < len(blocks):
            raise IndexError(
                f"indices[{position}]={normalized} is out of range for "
                f"{len(blocks)} blocks"
            )
        if normalized in seen:
            raise ValueError(f"duplicate block index {normalized}")
        validated.append(normalized)
        seen.add(normalized)
    return tuple(validated)


def build_observation_scales(
    source_blocks: Sequence[Block],
    workspaces: Sequence[Workspace],
    dropout_threshold: int,
    require_full_source: bool = True,
) -> ObservationScales:
    blocks = tuple(source_blocks)
    workspace_values = tuple(workspaces)
    if require_full_source and len(blocks) != EPISODE_BLOCK_COUNT:
        raise ValueError(
            f"full normalization source must contain exactly "
            f"{EPISODE_BLOCK_COUNT} blocks"
        )
    if require_full_source and len(workspace_values) != N_WORKSPACES:
        raise ValueError(
            f"full normalization source must contain exactly "
            f"{N_WORKSPACES} workspaces"
        )
    if not blocks:
        raise ValueError("at least one source block is required")
    if not workspace_values:
        raise ValueError("at least one workspace is required")
    if any(ws.length <= 0 or ws.breadth <= 0 for ws in workspace_values):
        raise ValueError("workspace dimensions must be positive")

    minimum_start = min(block.in_date for block in blocks)
    maximum_start = max(block.in_date for block in blocks)
    workspace_areas = [ws.length * ws.breadth for ws in workspace_values]
    return ObservationScales(
        max_length=max(block.length for block in blocks),
        max_breadth=max(block.breadth for block in blocks),
        max_duration=max(block.original_duration for block in blocks),
        base_date=minimum_start,
        date_span_workdays=working_days_until(
            minimum_start, maximum_start
        ),
        max_workspace_area=max(workspace_areas),
        total_workspace_area=sum(workspace_areas),
        max_workspace_length=max(ws.length for ws in workspace_values),
        max_workspace_breadth=max(ws.breadth for ws in workspace_values),
        dropout_threshold=dropout_threshold,
    )


def encode_current_block(
    block: Block,
    env_date: date,
    assigned_count: int,
    scales: ObservationScales,
) -> np.ndarray:
    return _feature_array([
        block.length / scales.max_length,
        block.breadth / scales.max_breadth,
        block.original_duration / scales.max_duration,
        working_day_position(scales.base_date, env_date)
        / scales.date_span_workdays,
        min(block.length, block.breadth)
        / max(block.length, block.breadth, 1e-6),
        assigned_count / (EPISODE_BLOCK_COUNT - 1),
        block.length * block.breadth / scales.max_workspace_area,
        max(block.length, block.breadth)
        / max(scales.max_workspace_length, scales.max_workspace_breadth),
    ])


def encode_future_blocks(
    blocks: Sequence[Block],
    indices: Sequence[int],
    env_date: date,
    scales: ObservationScales,
) -> tuple[np.ndarray, np.ndarray]:
    validated_indices = _validate_block_indices(blocks, indices)
    features = np.zeros(
        (ORDERED_FUTURE_COUNT, FUTURE_BLOCK_FEATURE_DIM), dtype=np.float32
    )
    mask = np.zeros(ORDERED_FUTURE_COUNT, dtype=np.float32)
    for slot, index in enumerate(validated_indices[:ORDERED_FUTURE_COUNT]):
        block = blocks[index]
        features[slot] = _feature_array([
            block.length / scales.max_length,
            block.breadth / scales.max_breadth,
            block.original_duration / scales.max_duration,
            working_days_until(env_date, block.in_date)
            / FUTURE_DAY_NORMALIZER,
            min(block.length, block.breadth)
            / max(block.length, block.breadth, 1e-6),
            block.length * block.breadth / scales.max_workspace_area,
        ])
        mask[slot] = 1.0
    return features, mask


def encode_future_demand(
    blocks: Sequence[Block],
    indices: Sequence[int],
    env_date: date,
    scales: ObservationScales,
) -> np.ndarray:
    validated_indices = _validate_block_indices(blocks, indices)
    demand = np.zeros(
        (len(FUTURE_DAY_WINDOWS), FUTURE_DEMAND_FEATURE_DIM),
        dtype=np.float32,
    )
    indexed_offsets = [
        (blocks[index], working_days_until(env_date, blocks[index].in_date))
        for index in validated_indices
    ]
    for row, (window_start, window_end) in enumerate(FUTURE_DAY_WINDOWS):
        window_blocks = [
            block
            for block, offset in indexed_offsets
            if window_start <= offset <= window_end
        ]
        if not window_blocks:
            continue
        areas = [block.length * block.breadth for block in window_blocks]
        demand[row] = _feature_array([
            len(window_blocks) / EPISODE_BLOCK_COUNT,
            sum(areas) / (4 * scales.total_workspace_area),
            sum(block.original_duration for block in window_blocks)
            / (len(window_blocks) * scales.max_duration),
            max(areas) / scales.max_workspace_area,
            max(block.length for block in window_blocks)
            / scales.max_length,
            max(block.breadth for block in window_blocks)
            / scales.max_breadth,
        ])
    return demand


def encode_pending_queues(
    blocks: Sequence[Block],
    workspaces: Sequence[Workspace],
    simulator: IncrementalPlacementSimulator,
    scales: ObservationScales,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_workspaces = len(workspaces)
    features = np.zeros(
        (n_workspaces, PENDING_QUEUE_SLOTS, PENDING_BLOCK_FEATURE_DIM),
        dtype=np.float32,
    )
    mask = np.zeros(
        (n_workspaces, PENDING_QUEUE_SLOTS), dtype=np.float32
    )
    summary = np.zeros(
        (n_workspaces, PENDING_SUMMARY_FEATURE_DIM), dtype=np.float32
    )

    for workspace_index, workspace in enumerate(workspaces):
        if (
            not math.isfinite(float(workspace.length))
            or not math.isfinite(float(workspace.breadth))
            or workspace.length <= 0
            or workspace.breadth <= 0
        ):
            raise ValueError("workspace dimensions must be finite and positive")

        queue = simulator.pending_assignment_indices(workspace_index)
        workspace_area = workspace.length * workspace.breadth
        workspace_axis = max(workspace.length, workspace.breadth)
        for slot, block_index in enumerate(queue[:PENDING_QUEUE_SLOTS]):
            block = blocks[block_index]
            features[workspace_index, slot] = _feature_array([
                block.length / scales.max_length,
                block.breadth / scales.max_breadth,
                block.original_duration / scales.max_duration,
                simulator.current_delay_workdays(block_index)
                / scales.dropout_threshold,
                min(block.length, block.breadth)
                / max(block.length, block.breadth, 1e-6),
                block.length * block.breadth / workspace_area,
                max(block.length, block.breadth) / workspace_axis,
            ])
            mask[workspace_index, slot] = 1.0

        if queue:
            areas = [blocks[index].length * blocks[index].breadth for index in queue]
            maximum_delay = max(
                simulator.current_delay_workdays(index) for index in queue
            )
            summary[workspace_index] = _feature_array([
                len(queue) / EPISODE_BLOCK_COUNT,
                sum(areas) / (4 * workspace_area),
                maximum_delay / scales.dropout_threshold,
                max(len(queue) - PENDING_QUEUE_SLOTS, 0)
                / EPISODE_BLOCK_COUNT,
            ])

    return features, mask, summary


def build_observation_space(
    n_workspaces: int = N_WORKSPACES,
    grid_size: int = GRID_SIZE,
) -> gym.spaces.Dict:
    if n_workspaces < 1:
        raise ValueError("at least one workspace is required")
    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    return gym.spaces.Dict({
        "block": gym.spaces.Box(
            0, 1, (CURRENT_BLOCK_FEATURE_DIM,), np.float32
        ),
        "grids": gym.spaces.Box(
            0, 1, (n_workspaces, 4, grid_size, grid_size), np.float32
        ),
        "ws_meta": gym.spaces.Box(
            0, 1, (n_workspaces, WORKSPACE_META_FEATURE_DIM), np.float32
        ),
        "future_blocks": gym.spaces.Box(
            0, 1, (ORDERED_FUTURE_COUNT, FUTURE_BLOCK_FEATURE_DIM), np.float32
        ),
        "future_mask": gym.spaces.Box(
            0, 1, (ORDERED_FUTURE_COUNT,), np.float32
        ),
        "future_demand": gym.spaces.Box(
            0,
            1,
            (len(FUTURE_DAY_WINDOWS), FUTURE_DEMAND_FEATURE_DIM),
            np.float32,
        ),
        "pending_blocks": gym.spaces.Box(
            0,
            1,
            (n_workspaces, PENDING_QUEUE_SLOTS, PENDING_BLOCK_FEATURE_DIM),
            np.float32,
        ),
        "pending_mask": gym.spaces.Box(
            0, 1, (n_workspaces, PENDING_QUEUE_SLOTS), np.float32
        ),
        "pending_summary": gym.spaces.Box(
            0,
            1,
            (n_workspaces, PENDING_SUMMARY_FEATURE_DIM),
            np.float32,
        ),
    })
