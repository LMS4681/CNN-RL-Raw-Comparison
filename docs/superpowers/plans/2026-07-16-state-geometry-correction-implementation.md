# State and Geometry Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every allocation path orientation-preserving and replace the policy input with the approved schema-3 current, future, pending, and candidate-conditioned workspace state.

**Architecture:** `IncrementalPlacementSimulator` remains the source of decision and retry order, while a new pure observation module converts simulator state into bounded fixed-shape arrays. `OccupancyGridRenderer` owns physical-to-pixel geometry and candidate previews, and the three feature extractors share one structured encoder so their only experimental difference is the grid path.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, PyTorch, Stable-Baselines3, sb3-contrib MaskablePPO, pytest/unittest, ONNX.

## Global Constraints

- Run commands from `D:\Sub\Allocation\CNN-RL\AllocRL` unless a step says otherwise.
- Complete the Evaluation Foundation plan before starting this plan.
- The action remains one of the ten fixed workspaces; the policy never selects coordinates.
- Rotation is physically prohibited. Keep `Block.turn()` only as an unused domain utility.
- Keep the ten-workspace order `PE049,PE050,PE055,PE054,PE056,PE048,PE044,PE059,PE060,PE061`.
- Keep exact 913-block empty-workspace episodes and the existing final delay/dropout reward.
- Do not add reward shaping, attention, Pointer Network, or PointNet code.
- Set `OBSERVATION_SCHEMA_VERSION = 3`; reject schema-2 checkpoints.
- Fix `GRID_SIZE = 64`, `ORDERED_FUTURE_COUNT = 16`, and `PENDING_QUEUE_SLOTS = 32`.
- Use future working-day windows `(0, 5)`, `(6, 20)`, and `(21, 60)`.
- Normalize ordered-future arrival by 30 working days and remaining grid lifetime by 60 working days.
- All observation arrays are `np.float32`, have fixed shapes, and remain in `[0, 1]`.
- Preserve reward conservation within `1e-6`.
- Use `apply_patch` for manual source edits.
- Each task ends with focused tests and a separate commit.

---

## File Structure

- Create `AllocRL/alloc_env/observation_state.py`: schema-3 constants, normalization scales, and pure current/future/demand/pending encoders.
- Create `AllocRL/test_no_rotation.py`: hard-mask, candidate, incremental, replay, and preview no-rotation regressions.
- Create `AllocRL/test_pending_observation.py`: deterministic queue order, masks, overflow summaries, demand windows, and state sensitivity.
- Create `AllocRL/test_grid_geometry.py`: independent-axis mapping, one-pixel coverage, exclusion zones, working-day lifetime, and lot-preview values.
- Modify `AllocRL/alloc_env/constraints.py`: orientation-preserving dimension feasibility.
- Modify `AllocRL/alloc_env/incremental_simulator.py`: one-orientation placement and deterministic public queue queries.
- Modify `AllocRL/alloc_env/simulator.py`: one-orientation replay placement.
- Modify `AllocRL/alloc_env/strategy.py`: public occupied-lot query used by rendering.
- Modify `AllocRL/alloc_env/occupancy_grid.py`: four-channel candidate-conditioned renderer and two-channel base cache.
- Modify `AllocRL/alloc_env/alloc_env.py`: schema-3 space, candidate construction, observations, preview, and terminal arrays.
- Modify `AllocRL/alloc_env/cnn_extractor.py`: shared schema-3 structured encoder and three grid ablation paths.
- Modify `AllocRL/train.py`: schema version, fixed observation constants, memory estimate, ONNX input contract, and legacy rejection.
- Modify `AllocRL/evaluation_runner.py`: fixed schema-3 scenario environment signature.
- Modify `AllocRL/evaluate_baselines.py`: remove schema-2 future-count options.
- Modify `AllocRL/run_ablation.py`: replace future-count ablation with fixed-shape state context.
- Modify `AllocRL/export_onnx.py`: restore schema-3 environment configuration.
- Modify `AllocRL/visualize.py`: schema-3 model loading metadata.
- Modify `AllocRL/visualize_grids.py`: four corrected channel labels and candidate preview display.
- Modify `AllocRL/visualize_eval_placement.py`: schema-3 environment reconstruction without rotation.
- Modify `AllocRL/test_candidate_observation.py`: candidate/simulator agreement under fixed orientation.
- Modify `AllocRL/test_feature_extractors.py`: full schema-3 shape, mask, order, and gradient behavior.
- Modify `AllocRL/test_future_block_lookahead.py`: 16-slot deterministic order and working-day normalization.
- Modify `AllocRL/test_cnn_diagnostics.py`: schema-3 diagnostic fixtures.
- Modify `AllocRL/test_onnx_export.py`: all schema-3 dictionary inputs.
- Modify `AllocRL/test_parallel_training_config.py`: corrected rollout-memory calculation and fixed constants.
- Modify `AllocRL/test_train_resume_cli.py`: schema-2 rejection and schema-3 acceptance fixtures.
- Modify `AllocRL/test_resolved_reward.py`: conservation regression with pending observations enabled.

---

### Task 1: Remove Rotation from Every Allocation Path

**Files:**
- Create: `test_no_rotation.py`
- Modify: `alloc_env/constraints.py:25-51`
- Modify: `alloc_env/occupancy_grid.py:34-41`
- Modify: `alloc_env/incremental_simulator.py:27-33,212-255`
- Modify: `alloc_env/simulator.py:82-116`
- Modify: `alloc_env/alloc_env.py:558-636,705-740`
- Modify: `test_candidate_observation.py`

**Interfaces:**
- Consumes: `Block.length`, `Block.breadth`, `Workspace.determine_placement_position(block, env_date)`.
- Produces: `CandidatePlacement(position: Optional[tuple[float, float]], length: float, breadth: float)` with no `rotated` field; every simulator makes exactly one placement attempt in original orientation.

- [ ] **Step 1: Write failing fixed-orientation tests**

Add tests with a `Block(length=8, breadth=4)` and a `Workspace(length=5, breadth=10)` so only the forbidden 90-degree orientation would fit:

Import `date`, all four allocation classes used below, both simulators, and
`SimulationResult` at the top of `test_no_rotation.py`.

```python
def make_block(length: float, breadth: float) -> Block:
    return Block(
        name="B-1", ship_no="S-1", block_type="BUILD",
        length=length, breadth=breadth, height=1.0, weight=1.0,
        in_date=date(2026, 1, 5), out_date=date(2026, 1, 20),
    )

def make_workspace(code: str, length: float, breadth: float) -> Workspace:
    return Workspace(
        code=code, origin_x=0.0, origin_y=0.0,
        length=length, breadth=breadth,
        strategy=BaseGridStrategy(step=1.0),
    )

def test_dimension_constraint_rejects_rotation_only_fit():
    block = make_block(length=8.0, breadth=4.0)
    workspace = make_workspace("ROTATION_ONLY", length=5.0, breadth=10.0)
    assert not DimensionConstraint().is_feasible(block, workspace)


def test_candidate_does_not_rotate_to_find_a_position():
    block = make_block(length=8.0, breadth=4.0)
    strategy = BaseGridStrategy(step=1.0)
    env = BlockPlacementEnv(
        [block],
        [
            make_workspace("ROTATION_ONLY", 5.0, 10.0),
            make_workspace("VALID", 20.0, 20.0),
        ],
        strategy,
        grid_size=64,
    )
    env.reset(seed=0)
    candidate = env.unwrapped._compute_candidate_placements(
        env.unwrapped._placement_simulator.current_block
    )[0]
    assert candidate.position is None
    assert candidate.length == 8.0
    assert candidate.breadth == 4.0


def test_incremental_and_replay_keep_original_orientation():
    block = make_block(length=8.0, breadth=4.0)
    workspace = make_workspace("ROTATION_ONLY", length=5.0, breadth=10.0)
    incremental = IncrementalPlacementSimulator(
        [block], [workspace], dropout_threshold=0
    )
    incremental.assign_current(0)
    assert incremental.result().delay_days == [SimulationResult.DROPOUT]
    replay = PlacementSimulator().replay([block], [workspace], [0], 0)
    assert replay.delay_days == [SimulationResult.DROPOUT]
    assert replay.blocks[0].length == 8.0
    assert replay.blocks[0].breadth == 4.0
```

Also update the preview regression so `future_workspace_choice_count_after_action()` cannot gain a placement by turning either the current or future block.

- [ ] **Step 2: Run the focused tests and verify the forbidden fallback is exposed**

Run: `python -m pytest test_no_rotation.py test_candidate_observation.py -q`

Expected: FAIL because `DimensionConstraint`, candidate generation, incremental simulation, replay, and preview currently call the rotation path.

- [ ] **Step 3: Make dimension and candidate semantics orientation-preserving**

Use this dimension gate and candidate data contract:

```python
class DimensionConstraint:
    def is_feasible(self, block: Block, ws: Workspace) -> bool:
        if block.length > ws.length or block.breadth > ws.breadth:
            return False
        if block.breadth > ws.max_breadth:
            return False
        if block.weight > ws.max_weight:
            return False
        if block.height > ws.max_height:
            return False
        return True


@dataclass(frozen=True)
class CandidatePlacement:
    position: Optional[Tuple[float, float]]
    length: float
    breadth: float

    @property
    def placeable(self) -> bool:
        return self.position is not None
```

In `_compute_candidate_placements`, incremental `_process_assigned_block`, replay, `_future_workspace_choice_count_on`, and `future_workspace_choice_count_after_action`, clone once, call `determine_placement_position()` once, and never call `turn()`.

- [ ] **Step 4: Prove there are no allocation-side rotation references**

Run: `rg -n "\.turn\(|rotated|rot90|rotation" alloc_env --glob "!block.py"`

Expected: no matches. `alloc_env/block.py` may still define `Block.turn()` but no allocation module calls it.

- [ ] **Step 5: Run no-rotation and safety regressions**

Run: `python -m pytest test_no_rotation.py test_candidate_observation.py test_safety_and_workspace_limits.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the no-rotation behavior**

```bash
git add AllocRL/alloc_env/constraints.py AllocRL/alloc_env/occupancy_grid.py AllocRL/alloc_env/incremental_simulator.py AllocRL/alloc_env/simulator.py AllocRL/alloc_env/alloc_env.py AllocRL/test_no_rotation.py AllocRL/test_candidate_observation.py
git commit -m "fix: prohibit block rotation in allocation paths"
```

---

### Task 2: Expose Deterministic Unassigned and Pending Queues

**Files:**
- Modify: `alloc_env/incremental_simulator.py:135-210`
- Test: `test_pending_observation.py`
- Test: `test_future_block_lookahead.py`

**Interfaces:**
- Consumes: simulator `pending`, `assignments`, `delay_days`, current block, original dates, and delayed dates.
- Produces: `unassigned_block_indices() -> list[int]`, `upcoming_block_indices(k: int) -> list[int]`, `pending_assignment_indices(workspace_index: int | None = None) -> list[int]`, and `current_delay_workdays(block_index: int) -> int`.

- [ ] **Step 1: Write failing queue-order tests**

```python
def make_queue_simulator(block_count: int = 4) -> IncrementalPlacementSimulator:
    blocks = [
        Block(
            name=f"B-{index}", ship_no="S-1", block_type="BUILD",
            length=5.0, breadth=5.0, height=1.0, weight=1.0,
            in_date=date(2026, 1, 5), out_date=date(2026, 1, 20),
        )
        for index in range(block_count)
    ]
    workspaces = [
        Workspace(
            code=f"W-{index}", origin_x=0.0, origin_y=0.0,
            length=100.0, breadth=100.0,
            strategy=BaseGridStrategy(step=1.0),
        )
        for index in range(2)
    ]
    return IncrementalPlacementSimulator(blocks, workspaces, 7)

def test_pending_assignments_are_grouped_and_retry_sorted():
    simulator = make_queue_simulator()
    simulator.assignments[:] = [1, 0, 1, None]
    simulator.pending = {0, 1, 2, 3}
    simulator.blocks[0].delay_placement(2)
    simulator.blocks[2].delay_placement(2)
    assert simulator.pending_assignment_indices(1) == [0, 2]
    assert simulator.pending_assignment_indices(0) == [1]
    assert simulator.current_delay_workdays(0) == 2


def test_unassigned_ties_use_block_index():
    simulator = make_queue_simulator(block_count=4)
    simulator.current_block_index = 0
    simulator.pending = {3, 2, 1, 0}
    assert simulator.unassigned_block_indices() == [1, 2, 3]
    assert simulator.upcoming_block_indices(2) == [1, 2]
```

- [ ] **Step 2: Run tests and verify the public methods are absent**

Run: `python -m pytest test_pending_observation.py test_future_block_lookahead.py -q`

Expected: FAIL with missing queue methods or nondeterministic tie order.

- [ ] **Step 3: Implement one deterministic order for decisions and retries**

Use block index as the final tie breaker everywhere:

```python
def _sort_key(self, idx: int) -> Tuple[int, date, int]:
    delay = self.current_delay_workdays(idx)
    return (-delay, self._original_blocks[idx].in_date, idx)

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
```

Use `_sort_key` for `today_targets` too, so observation order and actual next-decision order cannot diverge.

- [ ] **Step 4: Run queue and existing simulator tests**

Run: `python -m pytest test_pending_observation.py test_future_block_lookahead.py test_resolved_reward.py test_rl_regressions.py -q`

Expected: PASS.

- [ ] **Step 5: Commit deterministic simulator queries**

```bash
git add AllocRL/alloc_env/incremental_simulator.py AllocRL/test_pending_observation.py AllocRL/test_future_block_lookahead.py
git commit -m "feat: expose deterministic placement queues"
```

---

### Task 3: Build Pure Schema-3 Structured Encoders

**Files:**
- Create: `alloc_env/observation_state.py`
- Modify: `test_pending_observation.py`
- Modify: `test_future_block_lookahead.py`

**Interfaces:**
- Consumes: `Block`, `Workspace`, `IncrementalPlacementSimulator`, current environment date, decision count, and fixed normalization maxima.
- Produces: `encode_current_block(block, env_date, assigned_count, scales) -> np.ndarray`.
- Produces: `encode_future_blocks(blocks, indices, env_date, scales) -> tuple[np.ndarray, np.ndarray]`.
- Produces: `encode_future_demand(blocks, indices, env_date, scales) -> np.ndarray`.
- Produces: `encode_pending_queues(blocks, workspaces, simulator, scales) -> tuple[np.ndarray, np.ndarray, np.ndarray]`.
- Produces: `build_observation_space(n_workspaces=10, grid_size=64) -> gym.spaces.Dict`.
- Produces: `build_observation_scales(source_blocks, workspaces, dropout_threshold, require_full_source=True) -> ObservationScales` from the complete pre-split source.

- [ ] **Step 1: Write failing shape, boundary, mask, and overflow tests**

Cover all exact contracts:

Import `date`, `numpy as np`, `pytest`, `alloc_env.calendar as cal`, the block,
workspace, strategy, simulator, and all five `observation_state` interfaces used
below at the top of `test_pending_observation.py`.

```python
def add_workdays(start: date, count: int) -> date:
    value = start
    for _index in range(count):
        value = cal.next_working_day(value)
    return value

def make_observation_fixture(block_count: int = 40) -> dict:
    base = date(2026, 1, 5)
    blocks = [
        Block(
            name=f"B-{index}", ship_no="S-1", block_type="BUILD",
            length=10.0, breadth=5.0, height=1.0, weight=1.0,
            in_date=add_workdays(base, index),
            out_date=add_workdays(base, index + 10),
        )
        for index in range(block_count)
    ]
    workspaces = [
        Workspace(
            code=f"W-{index}", origin_x=0.0, origin_y=0.0,
            length=100.0, breadth=50.0,
            strategy=BaseGridStrategy(step=1.0),
        )
        for index in range(10)
    ]
    simulator = IncrementalPlacementSimulator(blocks, workspaces, 7)
    scales = ObservationScales(
        max_length=10.0, max_breadth=5.0, max_duration=10,
        base_date=base, date_span_workdays=100,
        max_workspace_area=5_000.0,
        total_workspace_area=50_000.0,
        max_workspace_length=100.0,
        max_workspace_breadth=50.0,
        dropout_threshold=7,
    )
    return {
        "blocks": simulator.blocks,
        "workspaces": simulator.workspaces,
        "simulator": simulator,
        "env_date": simulator.env_date,
        "scales": scales,
    }

def build_structured_state(fixture: dict) -> dict[str, np.ndarray]:
    simulator = fixture["simulator"]
    indices = simulator.unassigned_block_indices()
    future_blocks, future_mask = encode_future_blocks(
        fixture["blocks"], indices, fixture["env_date"], fixture["scales"]
    )
    pending_blocks, pending_mask, pending_summary = encode_pending_queues(
        fixture["blocks"], fixture["workspaces"], simulator,
        fixture["scales"],
    )
    return {
        "block": encode_current_block(
            simulator.current_block, fixture["env_date"],
            simulator.assigned_count, fixture["scales"],
        ),
        "future_blocks": future_blocks,
        "future_mask": future_mask,
        "future_demand": encode_future_demand(
            fixture["blocks"], indices, fixture["env_date"], fixture["scales"]
        ),
        "pending_blocks": pending_blocks,
        "pending_mask": pending_mask,
        "pending_summary": pending_summary,
    }

def test_schema3_structured_shapes_and_ranges():
    state = build_structured_state(make_observation_fixture())
    assert state["block"].shape == (8,)
    assert state["future_blocks"].shape == (16, 6)
    assert state["future_mask"].shape == (16,)
    assert state["future_demand"].shape == (3, 4)
    assert state["pending_blocks"].shape == (10, 32, 7)
    assert state["pending_mask"].shape == (10, 32)
    assert state["pending_summary"].shape == (10, 4)
    assert all(value.dtype == np.float32 for value in state.values())
    assert all(np.all((0.0 <= value) & (value <= 1.0)) for value in state.values())


def test_future_working_day_windows_include_exact_boundaries():
    fixture = make_observation_fixture(block_count=7)
    arrivals = [0, 5, 6, 20, 21, 60, 61]
    for block, offset in zip(fixture["blocks"], arrivals):
        block.in_date = add_workdays(fixture["env_date"], offset)
    demand = encode_future_demand(
        fixture["blocks"], list(range(7)),
        fixture["env_date"], fixture["scales"],
    )
    assert np.rint(demand[:, 0] * 913).astype(int).tolist() == [2, 2, 2]


def test_pending_overflow_keeps_first_32_and_summarizes_all_35():
    fixture = make_observation_fixture(block_count=35)
    simulator = fixture["simulator"]
    simulator.current_block_index = None
    simulator.assignments = [0] * 35
    blocks, mask, summary = encode_pending_queues(
        fixture["blocks"], fixture["workspaces"], simulator,
        fixture["scales"],
    )
    assert mask[0].sum() == 32
    assert summary[0, 0] == pytest.approx(35 / 913)
    assert summary[0, 3] == pytest.approx(3 / 913)
    assert np.count_nonzero(blocks[1:]) == 0
```

Also assert that a future arrival 15 working days away encodes as `0.5`, calendar weekends do not increase the count, masked rows are zero, and a delayed assigned block changes `pending_blocks` before resolution.

- [ ] **Step 2: Run focused observation tests**

Run: `python -m pytest test_pending_observation.py test_future_block_lookahead.py -q`

Expected: FAIL because `observation_state.py` does not exist.

- [ ] **Step 3: Define constants and normalization scales**

Create the module with these fixed values and immutable scale object:

```python
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
FUTURE_DEMAND_FEATURE_DIM = 4
PENDING_BLOCK_FEATURE_DIM = 7
PENDING_SUMMARY_FEATURE_DIM = 4
WORKSPACE_META_FEATURE_DIM = 4

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

    def to_dict(self) -> dict:
        values = asdict(self)
        values["base_date"] = self.base_date.isoformat()
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ObservationScales":
        parsed = dict(values)
        parsed["base_date"] = date.fromisoformat(str(parsed["base_date"]))
        return cls(**parsed)
```

Implement `build_observation_scales` with max length, breadth, and original
duration from all 913 pre-split blocks; working-day span from their minimum and
maximum start dates; area and axis values from the fixed ten workspaces; and the
provided dropout threshold. When `require_full_source=True`, validate exactly 913
blocks and ten workspaces so a split subset or one scenario cannot silently become
the normalization source. Unit tests with reduced fixtures pass
`require_full_source=False` explicitly.

Add `_clip01(value: float) -> np.float32` and
`working_days_until(start: date, end: date) -> int` helpers so all public arrays
are clipped and cast in one place.

Name the working-day helper `working_days_until` and define date progress as:

```python
def working_days_until(start: date, end: date) -> int:
    if end <= start:
        return 0
    return max(cal.get_working_days_between(start, end) - 1, 0)

def working_day_position(start: date, current: date) -> int:
    return working_days_until(start, current)

def build_observation_space(
    n_workspaces: int = N_WORKSPACES,
    grid_size: int = GRID_SIZE,
) -> gym.spaces.Dict:
    if n_workspaces < 1:
        raise ValueError("at least one workspace is required")
    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    return gym.spaces.Dict({
        "block": gym.spaces.Box(0, 1, (8,), np.float32),
        "grids": gym.spaces.Box(
            0, 1, (n_workspaces, 4, grid_size, grid_size), np.float32
        ),
        "ws_meta": gym.spaces.Box(0, 1, (n_workspaces, 4), np.float32),
        "future_blocks": gym.spaces.Box(0, 1, (16, 6), np.float32),
        "future_mask": gym.spaces.Box(0, 1, (16,), np.float32),
        "future_demand": gym.spaces.Box(0, 1, (3, 4), np.float32),
        "pending_blocks": gym.spaces.Box(
            0, 1, (n_workspaces, 32, 7), np.float32
        ),
        "pending_mask": gym.spaces.Box(
            0, 1, (n_workspaces, 32), np.float32
        ),
        "pending_summary": gym.spaces.Box(
            0, 1, (n_workspaces, 4), np.float32
        ),
    })
```

Import `gymnasium as gym` and `alloc_env.calendar as cal` in this module.
The reusable environment may use smaller `N` and grid sizes in focused tests.
`train.py`, the scenario runner, export, and visualization factories validate
`len(workspaces) == N_WORKSPACES` and `grid_size == GRID_SIZE` before creating any
model-facing environment.

- [ ] **Step 4: Implement current and ordered-future encoders**

Implement the approved feature order exactly:

```python
def encode_current_block(
    block: Block,
    env_date: date,
    assigned_count: int,
    scales: ObservationScales,
) -> np.ndarray:
    return np.asarray([
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
    ], dtype=np.float32).clip(0.0, 1.0)

def encode_future_blocks(
    blocks: Sequence[Block],
    indices: Sequence[int],
    env_date: date,
    scales: ObservationScales,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.zeros((ORDERED_FUTURE_COUNT, 6), dtype=np.float32)
    mask = np.zeros(ORDERED_FUTURE_COUNT, dtype=np.float32)
    for slot, index in enumerate(indices[:ORDERED_FUTURE_COUNT]):
        block = blocks[index]
        features[slot] = np.asarray([
            block.length / scales.max_length,
            block.breadth / scales.max_breadth,
            block.original_duration / scales.max_duration,
            working_days_until(env_date, block.in_date)
            / FUTURE_DAY_NORMALIZER,
            min(block.length, block.breadth)
            / max(block.length, block.breadth, 1e-6),
            block.length * block.breadth / scales.max_workspace_area,
        ], dtype=np.float32).clip(0.0, 1.0)
        mask[slot] = 1.0
    return features, mask
```

- [ ] **Step 5: Implement demand and pending encoders**

For each demand window, use all `simulator.unassigned_block_indices()`, working days from `env_date` to current `block.in_date`, count divided by 913, total area divided by `4 * total_workspace_area`, mean duration divided by source maximum, and max area divided by maximum workspace area.

For each workspace, use `pending_assignment_indices(ws_index)`. Fill the first 32 rows with normalized length, breadth, original duration, current delay/dropout threshold, aspect ratio, area/workspace area, and max block axis/max workspace axis. Compute summary count, total area, max delay, and overflow from the complete queue. Return zero arrays when no item exists.

- [ ] **Step 6: Run pure encoder tests**

Run: `python -m pytest test_pending_observation.py test_future_block_lookahead.py -q`

Expected: PASS.

- [ ] **Step 7: Commit schema-3 state encoders**

```bash
git add AllocRL/alloc_env/observation_state.py AllocRL/test_pending_observation.py AllocRL/test_future_block_lookahead.py
git commit -m "feat: encode corrected future and pending state"
```

---

### Task 4: Correct Candidate-Conditioned Grid Geometry

**Files:**
- Create: `test_grid_geometry.py`
- Modify: `alloc_env/strategy.py:118-170`
- Modify: `alloc_env/occupancy_grid.py:1-305`
- Modify: `test_candidate_observation.py`

**Interfaces:**
- Consumes: physical workspace/block rectangles, `SAFETY_DISTANCE`, active preplacements, lots, candidate position, and environment date.
- Produces: `BaseGridCache.get_base_grids(workspaces, env_date) -> np.ndarray` with shape `(N, 2, 64, 64)`.
- Produces: `OccupancyGridRenderer.render_candidate_context(workspace, candidate, current_block, env_date) -> np.ndarray` with shape `(2, 64, 64)`.

- [ ] **Step 1: Write failing physical-to-pixel tests**

```python
TEST_DATE = date(2026, 1, 5)

def make_grid_workspace(length: float, breadth: float) -> Workspace:
    return Workspace(
        code="W-1", origin_x=0.0, origin_y=0.0,
        length=length, breadth=breadth,
        strategy=BaseGridStrategy(step=1.0),
    )

def make_grid_block(length: float, breadth: float) -> Block:
    return Block(
        name="B-1", ship_no="S-1", block_type="BUILD",
        length=length, breadth=breadth, height=1.0, weight=1.0,
        in_date=TEST_DATE, out_date=date(2026, 1, 20),
    )

def test_axes_fill_full_grid_independently():
    renderer = OccupancyGridRenderer(64)
    workspace = make_grid_workspace(length=200.0, breadth=20.0)
    info = renderer.coordinate_map(workspace)
    assert info.x_px_per_m == pytest.approx(64 / 200.0)
    assert info.y_px_per_m == pytest.approx(64 / 20.0)
    assert renderer.rectangle_bounds(
        workspace, center_x=100.0, center_y=10.0,
        length=200.0, breadth=20.0,
    ) == (0, 0, 64, 64)


def test_positive_rectangle_gets_at_least_one_pixel_per_axis():
    renderer = OccupancyGridRenderer(64)
    workspace = make_grid_workspace(length=1000.0, breadth=1000.0)
    bounds = renderer.rectangle_bounds(
        workspace, center_x=0.1, center_y=0.1,
        length=0.01, breadth=0.01,
    )
    assert bounds[2] - bounds[0] >= 1
    assert bounds[3] - bounds[1] >= 1


def test_collision_channel_expands_by_safety_distance():
    workspace = make_grid_workspace(length=200.0, breadth=20.0)
    block = make_grid_block(length=10.0, breadth=6.0)
    block.move(100.0, 10.0)
    workspace.add_block(block, TEST_DATE)
    grid = OccupancyGridRenderer(64).render_base(workspace, TEST_DATE)
    ys, xs = np.nonzero(grid[0])
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (30, 33, 19, 44)
```

Import `date`, `numpy as np`, `pytest`, `SAFETY_DISTANCE`, `Block`, `Workspace`,
`BaseGridStrategy`, and renderer classes in `test_grid_geometry.py`.

Add tests showing remaining lifetime uses working days and clips at 60, an empty plain workspace has channel-2 value `0.25` everywhere, unavailable lots are `1.0`, available lots are `0.25`, and a placeable preview can change a lot from `0.25` to `1.0` without mutating the real workspace.

- [ ] **Step 2: Run geometry tests and verify old aspect-preserving rendering fails**

Run: `python -m pytest test_grid_geometry.py test_candidate_observation.py -q`

Expected: FAIL on independent-axis mapping, minimum pixel coverage, exclusion expansion, or lot-state channels.

- [ ] **Step 3: Publish occupied lot IDs without duplicating strategy rules**

Add this method to `BaseGridStrategy` and use it from the renderer:

```python
def occupied_lot_ids(
    self,
    workspace: Workspace,
    block_in: date,
    block_out: date,
) -> set[str]:
    return set(self._build_occupied_lot_set(workspace, block_in, block_out))
```

Do not copy the one-third lot occupation rule into `occupancy_grid.py`.

- [ ] **Step 4: Replace scalar scaling with independent-axis floor/ceil bounds**

Define an immutable mapping and clamp rectangles to `[0, 64]`:

```python
@dataclass(frozen=True)
class CoordinateMap:
    x_px_per_m: float
    y_px_per_m: float

def coordinate_map(self, ws: Workspace) -> CoordinateMap:
    return CoordinateMap(
        x_px_per_m=self.grid_size / max(ws.length, 1e-6),
        y_px_per_m=self.grid_size / max(ws.breadth, 1e-6),
    )

def rectangle_bounds(
    self,
    ws: Workspace,
    center_x: float,
    center_y: float,
    length: float,
    breadth: float,
) -> tuple[int, int, int, int]:
    x0 = math.floor((left - ws.origin_x) * mapping.x_px_per_m)
    y0 = math.floor((bottom - ws.origin_y) * mapping.y_px_per_m)
    x1 = math.ceil((right - ws.origin_x) * mapping.x_px_per_m)
    y1 = math.ceil((top - ws.origin_y) * mapping.y_px_per_m)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(self.grid_size, x1), min(self.grid_size, y1)
    if right > left and x1 <= x0:
        x1 = min(self.grid_size, x0 + 1)
    if top > bottom and y1 <= y0:
        y1 = min(self.grid_size, y0 + 1)
    return x0, y0, x1, y1
```

- [ ] **Step 5: Implement the four exact channels and two-channel cache**

Use these APIs:

```python
def render_base(self, ws: Workspace, env_date: date) -> np.ndarray:
    grid = np.zeros((2, self.grid_size, self.grid_size), dtype=np.float32)
    for block in ws.blocks:
        self._render_existing_exclusion(
            grid, ws, block.ref_x, block.ref_y,
            block.length, block.breadth, block.out_date, env_date,
        )
    for placed in ws.get_active_pre_placements(env_date):
        self._render_existing_exclusion(
            grid, ws, placed.pos_x, placed.pos_y,
            placed.length, placed.breadth, placed.end_date, env_date,
        )
    return grid

def render_candidate_context(
    self,
    ws: Workspace,
    candidate: CandidatePlacement,
    current_block: Block,
    env_date: date,
) -> np.ndarray:
    context = np.zeros((2, self.grid_size, self.grid_size), dtype=np.float32)
    preview = ws.deep_copy()
    if candidate.position is not None:
        block = current_block.clone()
        center_x, center_y = candidate.position
        block.move(center_x - block.ref_x, center_y - block.ref_y)
        preview.add_block(block, env_date)
        self._render_rectangle(
            context[1], ws, center_x, center_y,
            block.length + 2 * SAFETY_DISTANCE,
            block.breadth + 2 * SAFETY_DISTANCE,
            value=1.0,
        )
    if not preview.has_lots:
        context[0].fill(0.25)
        return context
    strategy = preview.strategy or BaseGridStrategy()
    occupied = strategy.occupied_lot_ids(
        preview, current_block.in_date, current_block.out_date
    )
    for lot in preview.lots:
        self._render_rectangle(
            context[0], preview,
            lot.origin_x + lot.length / 2,
            lot.origin_y + lot.breadth / 2,
            lot.length, lot.breadth,
            value=1.0 if lot.lot_id in occupied else 0.25,
        )
    return context

def _render_existing_exclusion(
    self, grid: np.ndarray, ws: Workspace,
    center_x: float, center_y: float,
    length: float, breadth: float,
    out_date: date, env_date: date,
) -> None:
    bounds = self.rectangle_bounds(
        ws, center_x, center_y,
        length + 2 * SAFETY_DISTANCE,
        breadth + 2 * SAFETY_DISTANCE,
    )
    x0, y0, x1, y1 = bounds
    if x1 <= x0 or y1 <= y0:
        return
    grid[0, y0:y1, x0:x1] = 1.0
    lifetime = min(working_days_until(env_date, out_date) / 60, 1.0)
    grid[1, y0:y1, x0:x1] = np.maximum(
        grid[1, y0:y1, x0:x1], lifetime
    )
```

Render existing physical blocks and active preplacements expanded by `SAFETY_DISTANCE` into Ch0. Fill Ch1 over the same rectangles with remaining working days divided by 60. For Ch2, start from current lot state, clone the workspace and current block for a placeable candidate, add only to the clone, and recalculate occupied lots; use `0.25` for available and `1.0` for unavailable. A workspace without lots is all `0.25`. Render the candidate expanded by `SAFETY_DISTANCE` into Ch3. Cache only Ch0 and Ch1.
Make `_render_rectangle` assign `np.maximum(existing_slice, value)` so overlapping
lot and exclusion values retain the strongest state.

Rename the existing `GridCache` class to `BaseGridCache`, make its internal array
shape `(n_workspaces, 2, 64, 64)`, and rename `get_grids` to `get_base_grids`.
Update the import and constructor call in `alloc_env.py` in Task 5.

- [ ] **Step 6: Run geometry, candidate, and safety tests**

Run: `python -m pytest test_grid_geometry.py test_candidate_observation.py test_safety_and_workspace_limits.py -q`

Expected: PASS and the non-mutation assertions leave workspace block counts unchanged.

- [ ] **Step 7: Commit corrected raster semantics**

```bash
git add AllocRL/alloc_env/strategy.py AllocRL/alloc_env/occupancy_grid.py AllocRL/test_grid_geometry.py AllocRL/test_candidate_observation.py
git commit -m "feat: render physical candidate workspace state"
```

---

### Task 5: Integrate the Fixed Schema-3 Environment Contract

**Files:**
- Modify: `alloc_env/alloc_env.py:50-210,237-313,348-405,520-636,703-907`
- Modify: `test_pending_observation.py`
- Modify: `test_future_block_lookahead.py`
- Modify: `test_candidate_observation.py`
- Modify: `test_resolved_reward.py`

**Interfaces:**
- Consumes: Task 2 simulator queries, Task 3 pure encoders, Task 4 base/candidate grid APIs.
- Produces: `BlockPlacementEnv` observations with exactly nine keys and fixed schema-3 shapes; optional `state_context_mode` values `"full"` and `"current"` keep the same shapes.

- [ ] **Step 1: Write failing environment-contract tests**

```python
def make_ten_workspace_env(
    state_context_mode: str = "full",
) -> BlockPlacementEnv:
    fixture = make_observation_fixture(block_count=40)
    return BlockPlacementEnv(
        fixture["blocks"],
        fixture["workspaces"],
        BaseGridStrategy(step=1.0),
        use_synthetic=False,
        grid_size=64,
        state_context_mode=state_context_mode,
    )

EXPECTED_SHAPES = {
    "block": (8,),
    "grids": (10, 4, 64, 64),
    "ws_meta": (10, 4),
    "future_blocks": (16, 6),
    "future_mask": (16,),
    "future_demand": (3, 4),
    "pending_blocks": (10, 32, 7),
    "pending_mask": (10, 32),
    "pending_summary": (10, 4),
}

def test_reset_step_and_terminal_observations_match_schema3():
    env = make_ten_workspace_env()
    observation, _ = env.reset(seed=0)
    assert {key: value.shape for key, value in observation.items()} == EXPECTED_SHAPES
    assert env.observation_space.contains(observation)
    terminal = env.unwrapped._get_terminal_obs()
    assert {key: value.shape for key, value in terminal.items()} == EXPECTED_SHAPES
    assert env.observation_space.contains(terminal)


def test_current_mode_zeros_context_without_changing_shapes():
    env = make_ten_workspace_env(state_context_mode="current")
    observation, _ = env.reset(seed=0)
    for key in (
        "future_blocks", "future_mask", "future_demand",
        "pending_blocks", "pending_mask", "pending_summary",
    ):
        assert np.count_nonzero(observation[key]) == 0
```

Add a state-sensitivity test: assign a block to a full workspace so it remains delayed, then assert the next observation's pending arrays differ while current block and untouched workspace grids do not spuriously mutate. Keep the existing `sum(rewards) == terminal_score` assertion at tolerance `1e-6`.

- [ ] **Step 2: Run environment tests and verify schema mismatch**

Run: `python -m pytest test_pending_observation.py test_future_block_lookahead.py test_candidate_observation.py test_resolved_reward.py -q`

Expected: FAIL because the environment still emits schema 2.

- [ ] **Step 3: Fix constructor parameters and observation space**

Replace the final constructor parameter `n_future_blocks` with `state_context_mode`
and `observation_scales`, while retaining `grid_size` for reduced unit fixtures:

```python
grid_size: int = GRID_SIZE,
state_context_mode: str = "full",
observation_scales: ObservationScales | None = None,
```

```python
if grid_size < 1:
    raise ValueError("grid_size must be positive")
if state_context_mode not in {"full", "current"}:
    raise ValueError("state_context_mode must be 'full' or 'current'")
self._observation_scales = observation_scales or build_observation_scales(
    self._original_blocks,
    self._original_workspaces,
    DROPOUT_THRESHOLD,
    require_full_source=False,
)
```

The fallback exists for direct deterministic unit-test environments. Every
training, fixed-holdout, original-CSV, export, and visualization factory must pass
the one scale record built from the complete 913-row source.

Assign
`self.observation_space = build_observation_space(self._num_workspaces, self._grid_size)`
so all nine entries are declared unconditionally. Keep array bounds `[0, 1]` and
dtype `np.float32`. Production factories reject values other than 64.

- [ ] **Step 4: Build and reuse fixed normalization scales**

Replace height/weight policy maxima and calendar-day span with
`ObservationScales`. In `train(args)`, compute it once from `full_blocks` before
the ship split and from the fixed ten empty workspaces. Pass that same object to
the synthetic training environment, original-CSV environment, and fixed-scenario
runner. Never derive production scales from the 673 training rows, 240 holdout
rows, one generated episode, or one scenario.

- [ ] **Step 5: Assemble grids, metadata, and structured arrays**

In `_get_obs`:

```python
base_grids = self._grid_cache.get_base_grids(
    self._workspaces, self._env_date
)
self._candidate_placements = self._compute_candidate_placements(block)
candidate_context = np.stack([
    self._renderer.render_candidate_context(
        workspace, candidate, block, self._env_date
    )
    for workspace, candidate in zip(
        self._workspaces, self._candidate_placements
    )
])
grids = np.concatenate([base_grids, candidate_context], axis=1)
ws_meta = np.stack([
    workspace_lengths / max_workspace_length,
    workspace_breadths / max_workspace_breadth,
    placed_areas / workspace_areas,
    candidate_placeability,
], axis=1).astype(np.float32)
```

Call the Task 3 encoders with `unassigned_block_indices()` and `pending_assignment_indices()`. In `current` mode, replace only future, demand, and pending outputs with zeros; current block, grids, and workspace metadata remain real. Build terminal observations from the same space-shape constants.

- [ ] **Step 6: Keep preview and candidate placement in exact agreement**

Use the already-computed `CandidatePlacement` orientation and position for preview. Add a regression that compares the candidate position at decision time with the actual block coordinates after `step(action)` whenever immediate placement succeeds.

- [ ] **Step 7: Run environment and reward tests**

Run: `python -m pytest test_pending_observation.py test_future_block_lookahead.py test_candidate_observation.py test_resolved_reward.py test_reward.py test_rl_regressions.py -q`

Expected: PASS, including reward conservation within `1e-6`.

- [ ] **Step 8: Commit the schema-3 environment**

```bash
git add AllocRL/alloc_env/alloc_env.py AllocRL/test_pending_observation.py AllocRL/test_future_block_lookahead.py AllocRL/test_candidate_observation.py AllocRL/test_resolved_reward.py
git commit -m "feat: expose schema3 allocation observations"
```

---

### Task 6: Share Corrected Structured State Across Extractors

**Files:**
- Modify: `alloc_env/cnn_extractor.py:1-245`
- Modify: `test_feature_extractors.py`
- Modify: `test_cnn_diagnostics.py`

**Interfaces:**
- Consumes: all schema-3 observation keys from Task 5.
- Produces: `StructuredStateEncoder.forward(observations) -> tuple[current_context, future_context, demand_context, pending_context]`; all three extractors return `(batch, features_dim)` and differ only by grid encoding.

- [ ] **Step 1: Replace test fixtures with full schema-3 tensors**

Use ten workspaces by default and include every key:

```python
def observation_space() -> gym.spaces.Dict:
    return gym.spaces.Dict({
        "block": gym.spaces.Box(0, 1, (8,), np.float32),
        "grids": gym.spaces.Box(0, 1, (10, 4, 64, 64), np.float32),
        "ws_meta": gym.spaces.Box(0, 1, (10, 4), np.float32),
        "future_blocks": gym.spaces.Box(0, 1, (16, 6), np.float32),
        "future_mask": gym.spaces.Box(0, 1, (16,), np.float32),
        "future_demand": gym.spaces.Box(0, 1, (3, 4), np.float32),
        "pending_blocks": gym.spaces.Box(0, 1, (10, 32, 7), np.float32),
        "pending_mask": gym.spaces.Box(0, 1, (10, 32), np.float32),
        "pending_summary": gym.spaces.Box(0, 1, (10, 4), np.float32),
    })
```

Assert output shape `(2, 256)` for each extractor, future and pending order sensitivity, masked-padding invariance for both sequences, demand sensitivity, and finite values.

- [ ] **Step 2: Write a failing end-to-end CNN update test**

```python
def test_candidate_cnn_weights_receive_nonzero_update():
    extractor = CandidateCnnExtractor(observation_space())
    optimizer = torch.optim.Adam(extractor.parameters(), lr=1e-3)
    before = extractor.image_encoder[0].weight.detach().clone()
    loss = extractor(observation()).square().mean()
    optimizer.zero_grad()
    loss.backward()
    assert extractor.image_encoder[0].weight.grad.norm().item() > 0.0
    optimizer.step()
    assert not torch.equal(before, extractor.image_encoder[0].weight)
```

- [ ] **Step 3: Run extractor tests and verify old dimensions fail**

Run: `python -m pytest test_feature_extractors.py test_cnn_diagnostics.py -q`

Expected: FAIL on missing pending/demand encoders or old `(10,)`, `(N,3)`, and `(K,8)` dimensions.

- [ ] **Step 4: Implement the exact shared structured encoder**

Use these networks:

```python
self.current = nn.Sequential(
    nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU()
)
self.future = nn.Sequential(
    nn.Linear(16 * 7, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU()
)
self.demand = nn.Sequential(
    nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU()
)
self.pending = nn.Sequential(
    nn.Linear(32 * 8, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU()
)
```

For future, multiply six features by the mask and append one mask value per slot before flattening. For pending, do the same with seven features and one mask value independently for each workspace. Flattening preserves order; do not pool, sort, or add attention.

- [ ] **Step 5: Fuse one corrected feature vector per workspace**

For each workspace concatenate 32 current, 64 future, 32 demand, 64 pending, 4 pending-summary, 4 workspace-meta, and the extractor-specific grid context. Use 0 grid values for `StructuredExtractor`, fixed adaptive `8x8` pooling for `FixedGridExtractor` (`4 * 8 * 8 = 256` values), and the existing shared GroupNorm CNN for `CandidateCnnExtractor` (128 values). Keep workspace fusion `Linear(input,128) -> ReLU -> Linear(128,64) -> ReLU`, flatten ten workspaces in action order, and project to 256 policy features.

- [ ] **Step 6: Run extractor and diagnostic tests**

Run: `python -m pytest test_feature_extractors.py test_cnn_diagnostics.py -q`

Expected: PASS. `StructuredExtractor` and `FixedGridExtractor` contain no `nn.Conv2d`; `CandidateCnnExtractor` has nonzero image gradients and preserves GroupNorm.

- [ ] **Step 7: Commit shared extractor state**

```bash
git add AllocRL/alloc_env/cnn_extractor.py AllocRL/test_feature_extractors.py AllocRL/test_cnn_diagnostics.py
git commit -m "feat: fuse corrected state in all extractors"
```

---

### Task 7: Version Training, ONNX, and Visualization Contracts

**Files:**
- Modify: `train.py:37-39,137-156,171-309,319-510,916-989,1009-1090`
- Modify: `evaluation_runner.py`
- Modify: `evaluate_baselines.py`
- Modify: `run_ablation.py`
- Modify: `export_onnx.py`
- Modify: `visualize.py`
- Modify: `visualize_grids.py`
- Modify: `visualize_eval_placement.py`
- Modify: `test_onnx_export.py`
- Modify: `test_parallel_training_config.py`
- Modify: `test_train_resume_cli.py`
- Test: `test_diagnostics_and_placement_visualization.py`

**Interfaces:**
- Consumes: Task 5 environment arguments and Task 6 extractor contract.
- Produces: schema-3 `run_config.json`, fixed memory estimate, schema-aware model utilities, ONNX graph with nine dictionary inputs, and corrected channel visualizations.

- [ ] **Step 1: Write failing schema and memory tests**

Import `date`, `SimpleNamespace`, `ObservationScales`,
`build_observation_space`, and the schema constants in the affected test modules.

```python
WORKSPACE_CODES = [
    "PE049", "PE050", "PE055", "PE054", "PE056",
    "PE048", "PE044", "PE059", "PE060", "PE061",
]

def source_manifest() -> dict:
    return {
        "split_seed": 20260716,
        "source_sha256": "abc123",
        "source_row_count": 913,
        "source_month_counts": {
            "2025-12": 64, "2026-01": 122, "2026-02": 106,
            "2026-03": 142, "2026-04": 153, "2026-05": 151,
            "2026-06": 175,
        },
    }

def full_source_scales() -> ObservationScales:
    return ObservationScales(
        max_length=100.0, max_breadth=50.0, max_duration=60,
        base_date=date(2025, 12, 1), date_span_workdays=150,
        max_workspace_area=10_000.0,
        total_workspace_area=80_000.0,
        max_workspace_length=200.0,
        max_workspace_breadth=100.0,
        dropout_threshold=7,
    )

def make_args() -> SimpleNamespace:
    return SimpleNamespace(
        extractor="candidate-cnn", features_dim=256,
        state_context="full", monthly_jitter=20,
        empirical_profile_probability=0.2, seed=0,
        eval_scenarios="./data/fixed_eval_scenarios.json",
    )

def test_run_config_records_schema3_observation_constants():
    scales = full_source_scales()
    config = current_run_config(
        make_args(), WORKSPACE_CODES, source_manifest(), scales
    )
    assert config["observation_schema_version"] == 3
    assert config["grid_size"] == 64
    assert config["ordered_future_count"] == 16
    assert config["pending_queue_slots"] == 32
    assert config["future_day_windows"] == [[0, 5], [6, 20], [21, 60]]
    assert config["observation_scales"] == scales.to_dict()


def test_rollout_estimate_counts_every_schema3_float():
    floats_per_observation = (
        8 + 10 * 4 * 64 * 64 + 10 * 4 + 16 * 6 + 16
        + 3 * 4 + 10 * 32 * 7 + 10 * 32 + 10 * 4
    )
    expected_mb = floats_per_observation * 4 * 960 / 1024 / 1024
    observation_space = build_observation_space()
    assert estimate_rollout_buffer_mb(
        observation_space, n_steps=960, n_envs=1
    ) == pytest.approx(expected_mb)
```

Add tests that schema-2 configs raise a clear incompatibility error, schema-3 configs load, and ONNX input names equal the sorted nine observation keys.

- [ ] **Step 2: Run tooling tests and verify schema-2 assumptions fail**

Run: `python -m pytest test_onnx_export.py test_parallel_training_config.py test_train_resume_cli.py test_diagnostics_and_placement_visualization.py -q`

Expected: FAIL on schema version, old optional future args, old memory count, or channel labels.

- [ ] **Step 3: Pin schema and environment creation arguments**

Set `OBSERVATION_SCHEMA_VERSION = 3`. Remove `n_future_blocks` from `make_env`, `create_training_env`, `create_evaluation_env`, evaluation helpers, and model tools. Keep `--grid-size` only as `choices=[64]` for command compatibility. Add `--state-context {full,current}` with default `full`, pass it to each environment, and record it in run config.

Change the run-config signature to:

```python
def current_run_config(
    args,
    active_workspace_codes: Sequence[str],
    source_manifest: Mapping[str, Any],
    observation_scales: ObservationScales,
) -> dict:
    return {
        "training_data_schema_version": TRAINING_DATA_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "extractor": args.extractor,
        "features_dim": args.features_dim,
        "active_workspace_codes": list(active_workspace_codes),
        "state_context": args.state_context,
        "grid_size": GRID_SIZE,
        "ordered_future_count": ORDERED_FUTURE_COUNT,
        "pending_queue_slots": PENDING_QUEUE_SLOTS,
        "future_day_windows": [list(item) for item in FUTURE_DAY_WINDOWS],
        "observation_scales": observation_scales.to_dict(),
        "data_split_seed": int(source_manifest["split_seed"]),
        "source_sha256": str(source_manifest["source_sha256"]),
        "episode_block_count": int(source_manifest["source_row_count"]),
        "target_month_counts": dict(source_manifest["source_month_counts"]),
        "excluded_start_months": list(DEFAULT_EXCLUDED_START_MONTHS),
        "monthly_jitter": int(args.monthly_jitter),
        "empirical_profile_probability": float(
            args.empirical_profile_probability
        ),
        "seed": int(args.seed),
        "eval_scenarios": args.eval_scenarios,
    }
```

Stage C extends this dictionary with every PPO hyperparameter before resume is
enabled for full experiments.

Change the Stage-A scenario runner to this fixed contract:

```python
def evaluate_scenarios(
    policy_factory: Callable[[int], ActionPolicy],
    scenarios: list[dict],
    workspace_codes: list[str] | None,
    observation_scales: ObservationScales,
    state_context_mode: str = "full",
) -> list[dict]:
    from train import create_evaluation_env

    rows = []
    for scenario in scenarios:
        seed = int(scenario["seed"])
        strategy = BaseGridStrategy(step=5.0)
        blocks, workspaces = materialize_scenario(scenario, strategy)
        ordered = select_workspaces_in_order(workspaces, workspace_codes)
        env = create_evaluation_env(
            blocks=blocks,
            workspaces=ordered,
            strategy=strategy,
            grid_size=GRID_SIZE,
            state_context_mode=state_context_mode,
            observation_scales=observation_scales,
            seed=seed,
        )
        policy = policy_factory(seed)
        try:
            metrics = evaluate_policy(policy, env, episodes=1)
        finally:
            env.close()
        rows.append({
            "source": "holdout_fixed20",
            "policy": policy.name,
            "seed": seed,
            **metrics,
        })
    return rows
```

It must create each evaluation environment with `grid_size=64` and the supplied
state context and pre-split normalization record. Remove `--n-future-blocks` and
`--grid-size` from `evaluate_baselines.py`; load the full source CSV to build its
normalization record and call scenarios with `state_context_mode="full"`.
Replace A-E definitions in `run_ablation.py` temporarily with
`(extractor, state_context)` pairs; Stage C adds budgets and hyperparameter grids.

- [ ] **Step 4: Calculate memory from actual observation-space shapes**

Replace hand-maintained dimensions with:

```python
def observation_float_count(observation_space: gym.spaces.Dict) -> int:
    return sum(
        int(np.prod(space.shape))
        for space in observation_space.spaces.values()
    )

def estimate_rollout_buffer_mb(
    observation_space: gym.spaces.Dict,
    n_steps: int,
    n_envs: int,
) -> float:
    return observation_float_count(observation_space) * 4 * n_steps * n_envs / 1024 / 1024
```

Build one environment before printing the estimate and pass its real observation space.

- [ ] **Step 5: Export all schema-3 dictionary inputs to ONNX**

Keep the existing generic wrapper, derive `obs_keys` from `env.observation_space.spaces`, and create one dummy tensor per key with a dynamic batch axis. Assert output is finite after `onnx.checker.check_model`. Preserve nonfatal optional export behavior when `onnxscript` or ONNX is unavailable.

- [ ] **Step 6: Update standalone tools and channel labels**

Require `observation_schema_version == 3` before model reconstruction. Read
`state_context`, workspace codes, fixed grid constants, and
`ObservationScales.from_dict(config["observation_scales"])` from run config. Label
grid channels exactly:

```python
CHANNEL_LABELS = (
    "collision exclusion",
    "remaining working days",
    "post-candidate lot state",
    "candidate exclusion",
)
```

Remove rotation text and rotated dimensions from visualization annotations. Ensure placement visualization still uses simulator-produced coordinates.

- [ ] **Step 7: Run tooling and visualization tests**

Run: `python -m pytest test_onnx_export.py test_parallel_training_config.py test_train_resume_cli.py test_diagnostics_and_placement_visualization.py -q`

Expected: PASS.

- [ ] **Step 8: Run all Stage-B focused tests**

Run: `python -m pytest test_no_rotation.py test_pending_observation.py test_grid_geometry.py test_candidate_observation.py test_future_block_lookahead.py test_feature_extractors.py test_cnn_diagnostics.py test_onnx_export.py test_resolved_reward.py -q`

Expected: PASS.

- [ ] **Step 9: Commit versioned model tools**

```bash
git add AllocRL/train.py AllocRL/evaluation_runner.py AllocRL/evaluate_baselines.py AllocRL/run_ablation.py AllocRL/export_onnx.py AllocRL/visualize.py AllocRL/visualize_grids.py AllocRL/visualize_eval_placement.py AllocRL/test_onnx_export.py AllocRL/test_parallel_training_config.py AllocRL/test_train_resume_cli.py AllocRL/test_diagnostics_and_placement_visualization.py
git commit -m "feat: version schema3 training and model tools"
```

---

### Task 8: Verify Stage B as an Independently Working Revision

**Files:**
- Modify: `ABLATION.md`
- Modify: `smoke_test.py`
- Test: all `test_*.py`

**Interfaces:**
- Consumes: complete schema-3 environment and extractors.
- Produces: documented no-rotation/schema-3 contract and a short save/load/evaluate smoke command for each extractor.

- [ ] **Step 1: Update the technical contract documentation**

Document the nine observation keys and dimensions, original-orientation-only placement, fixed 16 future rows, 32 pending rows per workspace, candidate grid channels, and the fact that the CNN is trained end-to-end by actor/critic losses rather than as a feasibility classifier. State explicitly that reward shaping remains absent.

- [ ] **Step 2: Add an all-extractor smoke mode**

Extend `smoke_test.py` with:

```python
EXTRACTORS = ("structured", "fixed-grid", "candidate-cnn")

def run_extractor_smoke(extractor: str, output_dir: Path) -> None:
    model, env = train_tiny_model(extractor=extractor, timesteps=1_024)
    path = output_dir / f"{extractor}.sb3"
    model.save(path)
    loaded = MaskablePPO.load(path, env=env)
    metrics = evaluate(loaded, env, n_eval=1, return_metrics=True)
    assert math.isfinite(metrics["mean_terminal_score"])
```

Expose it as `python smoke_test.py --all-extractors --timesteps 1024` and use temporary output directories unless `--output-dir` is passed.

- [ ] **Step 3: Run the complete regression suite**

Run: `python -m pytest -q`

Expected: all tests PASS with no unexpected skips. Optional ONNX dependency skips must retain their explicit reason.

- [ ] **Step 4: Run all three save/load/evaluate smoke checks**

Run: `python smoke_test.py --all-extractors --timesteps 1024`

Expected: `structured`, `fixed-grid`, and `candidate-cnn` each train, save, load, and finish one evaluation episode; the CNN reports a nonzero gradient norm and weight change.

- [ ] **Step 5: Audit source blocks under no-rotation constraints**

Run: `python -c "from pathlib import Path; from alloc_env.strategy import BaseGridStrategy; from train import DEFAULT_ACTIVE_WORKSPACE_CODES,load_allocation_scenario,parse_workspace_codes; from alloc_env.constraints import DimensionConstraint,BlockPatternConstraint; b,w=load_allocation_scenario(Path('data'),BaseGridStrategy(),parse_workspace_codes(DEFAULT_ACTIVE_WORKSPACE_CODES)); c=(DimensionConstraint(),BlockPatternConstraint()); bad=[x.name for x in b if not any(all(k.is_feasible(x,y) for k in c) for y in w)]; print(len(bad),bad[:10]); assert not bad"`

Expected: prints `0 []`.

- [ ] **Step 6: Commit Stage-B documentation and smoke coverage**

```bash
git add AllocRL/ABLATION.md AllocRL/smoke_test.py
git commit -m "test: verify schema3 extractor workflows"
```
