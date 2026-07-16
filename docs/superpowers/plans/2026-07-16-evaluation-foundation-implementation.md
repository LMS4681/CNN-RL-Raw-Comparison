# Evaluation Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a ship-group-disjoint training/holdout split, preserve 913-block monthly episode profiles independently of bootstrap source size, and evaluate learned and heuristic policies on the same reproducible scenarios.

**Architecture:** A new data-split module owns deterministic ship grouping and manifest metadata. `SyntheticBlockGenerator` receives an explicit target month profile so both the 673-row training source and 240-row holdout source can generate 913-block episodes. Scenario serialization carries provenance, while a separate evaluation runner applies model adapters and non-learning baselines without changing the model observation schema.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, sb3-contrib MaskablePPO, pytest/unittest, JSON, CSV, SHA256.

## Global Constraints

- Run commands from `D:\Sub\Allocation\CNN-RL\AllocRL` unless a step says otherwise.
- Do not change observation shapes, CNN architecture, reward values, action count, simulator placement rules, or rotation behavior in Stage A.
- Keep the ten fixed empty workspaces and exact 913-block episode length.
- Keep the approved 80 percent balanced and 20 percent empirical training profile.
- Use source month counts `64, 122, 106, 142, 153, 151, 175` as the explicit empirical target.
- Split by non-empty `ship_no`; no ship group may occur in both sources.
- Use split seed `20260716`, SHA256 threshold `0.20`, and holdout scenario seeds `1000..1019`.
- The current CSV must split to 673 training rows and 240 holdout rows.
- The deterministic full CSV business reference runs exactly once.
- Do not select a model from training-source diagnostic episodes.
- Use `apply_patch` for manual source edits.
- Each task ends with focused tests and a separate commit.

---

## File Structure

- Create `AllocRL/alloc_env/data_split.py`: deterministic ship split, file hashing, manifest construction and serialization.
- Create `AllocRL/evaluation_runner.py`: policy adapters, common episode evaluation, fixed scenario evaluation, and baseline result rows.
- Create `AllocRL/baseline_policies.py`: seeded random and immediate-free-area policies.
- Create `AllocRL/evaluate_baselines.py`: baseline evaluation CLI over the fixed holdout bundle.
- Create `AllocRL/test_data_split.py`: split counts, disjointness, determinism, validation, and manifest tests.
- Create `AllocRL/test_baseline_policies.py`: action selection and shared-scenario evaluation tests.
- Modify `AllocRL/alloc_env/block_generator.py`: explicit target month counts independent of source templates.
- Modify `AllocRL/train.py`: split-aware training source, fixed 913 episode count, one-shot original evaluation, and evaluation-runner integration.
- Modify `AllocRL/evaluation_scenarios.py`: schema 3 provenance metadata and explicit month-profile generation.
- Modify `AllocRL/run_ablation.py`: holdout scenario preparation and baseline command.
- Modify `AllocRL/test_training_data_profile.py`: split-source 913-block generation tests.
- Modify `AllocRL/test_evaluation_scenarios.py`: schema 3, provenance, one-shot CSV, and 20-scenario assertions.
- Modify `AllocRL/ABLATION.md`: document holdout and baseline protocol.
- Regenerate `AllocRL/data/fixed_eval_scenarios.json`: 20 holdout-only schema-3 scenarios.
- Create `AllocRL/data/data_split_manifest.json`: pinned source and split provenance.

---

### Task 1: Deterministic Ship-Group Split

**Files:**
- Create: `AllocRL/alloc_env/data_split.py`
- Create: `AllocRL/test_data_split.py`

**Interfaces:**
- Consumes: `Sequence[Block]`, source CSV `Path`, split seed `int`, holdout fraction `float`.
- Produces: `BlockSourceSplit(training_blocks, holdout_blocks, manifest)`.
- Produces: `sha256_file(path: str | Path) -> str`.
- Produces: `write_split_manifest(path: str | Path, manifest: Mapping) -> None`.

- [ ] **Step 1: Write failing split tests**

Create `test_data_split.py` with the real-data contract and focused synthetic edge cases:

```python
from pathlib import Path

import pytest

from alloc_env.block import Block
from alloc_env.data_split import (
    DEFAULT_HOLDOUT_FRACTION,
    DEFAULT_SPLIT_SEED,
    split_blocks_by_ship,
)
from alloc_env.strategy import BaseGridStrategy
from train import (
    DEFAULT_ACTIVE_WORKSPACE_CODES,
    load_allocation_scenario,
    parse_workspace_codes,
)


DATA_DIR = Path(__file__).parent / "data"
BLOCK_CSV = DATA_DIR / "블록데이터.csv"


def load_targets():
    blocks, _ = load_allocation_scenario(
        DATA_DIR,
        BaseGridStrategy(step=5.0),
        parse_workspace_codes(DEFAULT_ACTIVE_WORKSPACE_CODES),
    )
    return blocks


def test_current_source_split_is_pinned_and_group_disjoint():
    split = split_blocks_by_ship(load_targets(), BLOCK_CSV)
    assert DEFAULT_SPLIT_SEED == 20260716
    assert DEFAULT_HOLDOUT_FRACTION == 0.20
    assert len(split.training_blocks) == 673
    assert len(split.holdout_blocks) == 240
    training_ships = {block.ship_no for block in split.training_blocks}
    holdout_ships = {block.ship_no for block in split.holdout_blocks}
    assert len(training_ships) == 29
    assert len(holdout_ships) == 11
    assert training_ships.isdisjoint(holdout_ships)
    assert split.manifest["source_row_count"] == 913
    assert sum(split.manifest["source_month_counts"].values()) == 913
    assert split.manifest["source_sha256"]


def test_split_is_deterministic_under_input_reordering():
    blocks = load_targets()
    normal = split_blocks_by_ship(blocks, BLOCK_CSV)
    reversed_split = split_blocks_by_ship(list(reversed(blocks)), BLOCK_CSV)
    assert normal.manifest["training_ship_nos"] == reversed_split.manifest[
        "training_ship_nos"
    ]
    assert normal.manifest["holdout_ship_nos"] == reversed_split.manifest[
        "holdout_ship_nos"
    ]


def test_split_rejects_empty_ship_number():
    block = load_targets()[0].clone()
    block.ship_no = ""
    with pytest.raises(ValueError, match="non-empty ship_no"):
        split_blocks_by_ship([block], BLOCK_CSV)
```

- [ ] **Step 2: Run the split tests and verify failure**

Run:

```powershell
py -B -m pytest test_data_split.py -q
```

Expected: collection fails because `alloc_env.data_split` does not exist.

- [ ] **Step 3: Implement the split module**

Create `alloc_env/data_split.py` with these exact public types and functions:

```python
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .block import Block


DEFAULT_SPLIT_SEED = 20260716
DEFAULT_HOLDOUT_FRACTION = 0.20


@dataclass(frozen=True)
class BlockSourceSplit:
    training_blocks: Sequence[Block]
    holdout_blocks: Sequence[Block]
    manifest: dict


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _holdout_fraction(ship_no: str, split_seed: int) -> float:
    payload = f"{split_seed}:{ship_no}".encode("utf-8")
    prefix = hashlib.sha256(payload).digest()[:8]
    return int.from_bytes(prefix, "big", signed=False) / float(2**64)


def split_blocks_by_ship(
    blocks: Sequence[Block],
    source_path: str | Path,
    split_seed: int = DEFAULT_SPLIT_SEED,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
) -> BlockSourceSplit:
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")
    if any(not block.ship_no.strip() for block in blocks):
        raise ValueError("Every target must have a non-empty ship_no")

    holdout_ships = {
        ship_no
        for ship_no in {block.ship_no for block in blocks}
        if _holdout_fraction(ship_no, split_seed) < holdout_fraction
    }
    training = tuple(block.clone() for block in blocks if block.ship_no not in holdout_ships)
    holdout = tuple(block.clone() for block in blocks if block.ship_no in holdout_ships)
    if not training or not holdout:
        raise ValueError("Ship split must produce non-empty training and holdout sources")

    def month_counts(items: Sequence[Block]) -> dict[str, int]:
        counts = Counter(f"{b.in_date.year:04d}-{b.in_date.month:02d}" for b in items)
        return dict(sorted(counts.items()))

    training_ships = sorted({block.ship_no for block in training})
    holdout_ship_list = sorted(holdout_ships)
    manifest = {
        "split_seed": int(split_seed),
        "holdout_fraction": float(holdout_fraction),
        "source_sha256": sha256_file(source_path),
        "source_row_count": len(blocks),
        "source_month_counts": month_counts(blocks),
        "training_row_count": len(training),
        "holdout_row_count": len(holdout),
        "training_ship_nos": training_ships,
        "holdout_ship_nos": holdout_ship_list,
        "training_month_counts": month_counts(training),
        "holdout_month_counts": month_counts(holdout),
    }
    return BlockSourceSplit(training, holdout, manifest)


def write_split_manifest(path: str | Path, manifest: Mapping) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
py -B -m pytest test_data_split.py -q
```

Expected: all split tests pass with `673` training rows and `240` holdout rows.

- [ ] **Step 5: Commit the split module**

```powershell
git add AllocRL/alloc_env/data_split.py AllocRL/test_data_split.py
git commit -m "feat: add ship-group training holdout split"
```

---

### Task 2: Explicit Month Profiles and Fixed Episode Length

**Files:**
- Modify: `AllocRL/alloc_env/block_generator.py:124`
- Modify: `AllocRL/train.py:171`
- Modify: `AllocRL/train.py:221`
- Modify: `AllocRL/train.py:582`
- Modify: `AllocRL/test_training_data_profile.py:153`
- Modify: `AllocRL/test_train_resume_cli.py:16`

**Interfaces:**
- Extends: `SyntheticBlockGenerator.from_blocks(blocks, seed=None, monthly_jitter=20, empirical_profile_probability=0.2, target_month_counts=None)`.
- Extends: `create_training_env(blocks, workspaces, strategy, generator, grid_size=64, n_envs=1, vec_env="auto", n_future_blocks=4, seed=0, episode_n_blocks=913)`.
- Preserves: `SyntheticBlockGenerator.generate(n_blocks, base_date, spread_days)`.
- Sets: `TRAINING_DATA_SCHEMA_VERSION = 2` because split sources and explicit target profiles change episode semantics.

- [ ] **Step 1: Add failing explicit-profile tests**

Add these tests to `test_training_data_profile.py`:

Import `pytest` and `split_blocks_by_ship` at the top of the test module.

```python
def test_split_sources_generate_full_empirical_target_profile():
    blocks, _ = train_module.load_allocation_scenario(
        DATA_DIR,
        BaseGridStrategy(step=5.0),
        train_module.parse_workspace_codes(
            train_module.DEFAULT_ACTIVE_WORKSPACE_CODES
        ),
    )
    split = split_blocks_by_ship(blocks, BLOCK_CSV)
    target_counts = count_start_months(blocks)
    for source in (split.training_blocks, split.holdout_blocks):
        generator = SyntheticBlockGenerator.from_blocks(
            source,
            seed=3,
            empirical_profile_probability=1.0,
            target_month_counts=target_counts,
        )
        generated = generator.generate(913, min(block.in_date for block in blocks))
        assert len(generated) == 913
        assert count_start_months(generated) == target_counts


def test_target_profile_rejects_month_without_source_templates():
    source = list(target_blocks())
    missing_month_source = [
        block for block in source if block.in_date.month != 12
    ]
    with pytest.raises(ValueError, match="no source templates"):
        SyntheticBlockGenerator.from_blocks(
            missing_month_source,
            target_month_counts=EXPECTED_MONTH_COUNTS,
        )
```

Update the training-environment test to assert that a 673-row source still resets
to 913 targets through an explicit `episode_n_blocks=913` argument.

- [ ] **Step 2: Run the profile tests and verify failure**

```powershell
py -B -m pytest test_training_data_profile.py -q
```

Expected: failures report unknown `target_month_counts` and `episode_n_blocks`.

- [ ] **Step 3: Extend `SyntheticBlockGenerator`**

Add `target_month_counts` to `__init__` and `from_blocks`. Normalize the mapping to
sorted `(year, month) -> int` pairs, validate positive counts and template coverage,
and store it as `_target_month_counts`.

Use this implementation shape:

```python
self._target_month_counts = Counter(
    {
        (int(year), int(month)): int(count)
        for (year, month), count in (target_month_counts or self._source_month_counts).items()
    }
)
for key, count in self._target_month_counts.items():
    if count < 0:
        raise ValueError("target month counts must be non-negative")
    if count and key not in self._source_month_counts:
        raise ValueError(f"Target month {key} has no source templates")
```

Change `_generate_monthly_bootstrap` to use sorted target keys. Change
`_empirical_month_counts` to scale `_target_month_counts`, not
`_source_month_counts`. Keep `_balanced_month_counts` unchanged except for its
target-key input.

Expose a read-only copy:

```python
@property
def target_month_counts(self) -> dict[tuple[int, int], int]:
    return dict(self._target_month_counts)
```

- [ ] **Step 4: Separate source size from episode size in training factories**

Add `episode_n_blocks: int` to `create_training_env` and pass it through
`env_kwargs["synthetic_n_blocks"]`. Do not derive it from `len(blocks)`.

In `train(args)`, load the full source, split it, create the generator from
`split.training_blocks`, pass the full-source month counts, and pass
`episode_n_blocks=len(full_blocks)`:

```python
active_codes = parse_workspace_codes(args.active_workspace_codes)
full_blocks, workspaces = load_allocation_scenario(
    data_dir,
    strategy,
    active_codes,
)
source_split = split_blocks_by_ship(
    full_blocks,
    data_dir / "블록데이터.csv",
    split_seed=20260716,
    holdout_fraction=0.20,
)
target_month_counts = Counter(
    (block.in_date.year, block.in_date.month) for block in full_blocks
)
generator = SyntheticBlockGenerator.from_blocks(
    source_split.training_blocks,
    seed=args.seed,
    monthly_jitter=args.monthly_jitter,
    empirical_profile_probability=args.empirical_profile_probability,
    target_month_counts=target_month_counts,
)
env = create_training_env(
    source_split.training_blocks,
    workspaces,
    strategy,
    generator,
    episode_n_blocks=len(full_blocks),
    grid_size=args.grid_size,
    n_envs=args.n_envs,
    vec_env=args.vec_env,
    n_future_blocks=args.n_future_blocks,
    seed=args.seed,
)
```

Set `TRAINING_DATA_SCHEMA_VERSION = 2` in `train.py` and update the run-config
fixture in `test_train_resume_cli.py` in the same change. A schema-1 model must be
rejected because it was trained before the group-disjoint source contract.

Keep evaluation construction on `full_blocks` until Task 4 replaces evaluation
dispatch.

- [ ] **Step 5: Run data-profile and environment tests**

```powershell
py -B -m pytest test_training_data_profile.py test_synthetic.py test_parallel_training_config.py -q
```

Expected: all pass; every training reset contains exactly 913 blocks.

- [ ] **Step 6: Commit profile separation**

```powershell
git add AllocRL/alloc_env/block_generator.py AllocRL/train.py AllocRL/test_training_data_profile.py AllocRL/test_train_resume_cli.py
git commit -m "feat: separate bootstrap sources from episode profiles"
```

---

### Task 3: Provenance-Aware Holdout Scenarios

**Files:**
- Modify: `AllocRL/evaluation_scenarios.py:17`
- Modify: `AllocRL/run_ablation.py:55`
- Modify: `AllocRL/test_evaluation_scenarios.py:100`
- Create: `AllocRL/data/data_split_manifest.json` during Step 5.
- Regenerate: `AllocRL/data/fixed_eval_scenarios.json` during Step 5.

**Interfaces:**
- Sets: `SCENARIO_SCHEMA_VERSION = 3`.
- Extends: `generate_scenarios(distribution, workspaces, seeds, n_blocks, base_date, spread_days, source_blocks=None, vary_layout=True, empirical_profile_probability=0.2, target_month_counts=None, source_name="holdout_fixed")`.
- Extends: `write_scenarios(path, scenarios, metadata)`.
- Produces: `read_scenario_metadata(path) -> dict`.
- Preserves: `read_scenarios(path) -> list[dict]` for training callers.

- [ ] **Step 1: Write failing schema-3 provenance tests**

Add tests that assert:

Add `Counter` to the collections imports and import
`read_scenario_metadata` in the existing scenario test module.

```python
def test_scenario_bundle_round_trips_provenance():
    blocks = make_blocks()
    workspaces = [make_workspace()]
    metadata = {
        "source": "holdout_fixed",
        "split_seed": 20260716,
        "source_sha256": "abc123",
    }
    scenarios = generate_scenarios(
        distribution=BlockDistribution.from_blocks(blocks),
        workspaces=workspaces,
        seeds=[1000],
        n_blocks=3,
        base_date=date(2026, 1, 5),
        spread_days=30,
        source_blocks=blocks,
        target_month_counts=Counter(
            (block.in_date.year, block.in_date.month) for block in blocks
        ),
        vary_layout=False,
        empirical_profile_probability=1.0,
        source_name="holdout_fixed",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "scenarios.json"
        write_scenarios(path, scenarios, metadata)
        assert read_scenarios(path) == scenarios
        assert read_scenario_metadata(path) == metadata
```

Add a real-data preparation test using a temporary destination. Assert 20 scenarios,
seeds `1000..1019`, 913 blocks each, ten empty workspaces, exact empirical month
counts, and block ship numbers drawn only from the manifest holdout list.

- [ ] **Step 2: Run scenario tests and verify failure**

```powershell
py -B -m pytest test_evaluation_scenarios.py -q
```

Expected: schema and argument failures.

- [ ] **Step 3: Implement schema-3 metadata**

Change the payload format to:

```python
payload = {
    "schema_version": SCENARIO_SCHEMA_VERSION,
    "metadata": dict(metadata),
    "scenarios": scenarios,
}
```

Validate that metadata is a dictionary and each scenario has `seed`, `source`,
`blocks`, and `workspaces`. Add `source_name` to each scenario record. Pass
`target_month_counts` into the generator. Keep `materialize_scenario` unchanged.

Implement:

```python
def _read_payload(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCENARIO_SCHEMA_VERSION:
        raise ValueError("Unsupported evaluation scenario schema")
    if not isinstance(payload.get("metadata"), dict):
        raise ValueError("Evaluation scenario payload must contain metadata")
    if not isinstance(payload.get("scenarios"), list):
        raise ValueError("Evaluation scenario payload must contain a list")
    return payload


def read_scenarios(path: str | Path) -> list[dict]:
    return _read_payload(path)["scenarios"]


def read_scenario_metadata(path: str | Path) -> dict:
    return dict(_read_payload(path)["metadata"])
```

- [ ] **Step 4: Make scenario preparation split-aware**

In `run_ablation.prepare_evaluation_file`:

1. Load all 913 blocks and ten workspaces.
2. Build `BlockSourceSplit` from `블록데이터.csv`.
3. Build full-source target month counts.
4. Generate 20 holdout scenarios with exact empirical profile.
5. Write `data_split_manifest.json` next to the scenario file.
6. Embed split seed, source SHA256, row counts, and target profile in scenario
   metadata.

Use this call shape:

```python
scenarios = generate_scenarios(
    distribution=BlockDistribution.from_blocks(split.holdout_blocks),
    workspaces=active,
    seeds=list(range(1000, 1020)),
    n_blocks=len(csv_blocks),
    base_date=min(block.in_date for block in csv_blocks),
    spread_days=spread_days,
    source_blocks=list(split.holdout_blocks),
    target_month_counts=target_month_counts,
    vary_layout=False,
    empirical_profile_probability=1.0,
    source_name="holdout_fixed",
)
```

- [ ] **Step 5: Regenerate pinned scenario artifacts**

Run:

```powershell
py -B run_ablation.py --prepare-eval-scenarios
```

Expected:

```text
Fixed evaluation scenarios saved to: data\fixed_eval_scenarios.json
Data split manifest saved to: data\data_split_manifest.json
```

Then run:

```powershell
py -B -m pytest test_evaluation_scenarios.py test_data_split.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit scenario provenance**

```powershell
git add AllocRL/evaluation_scenarios.py AllocRL/run_ablation.py AllocRL/test_evaluation_scenarios.py AllocRL/data/fixed_eval_scenarios.json AllocRL/data/data_split_manifest.json
git commit -m "feat: create group-disjoint holdout scenarios"
```

---

### Task 4: Shared Evaluation Runner and Heuristic Baselines

**Files:**
- Create: `AllocRL/baseline_policies.py`
- Create: `AllocRL/evaluation_runner.py`
- Create: `AllocRL/test_baseline_policies.py`
- Modify: `AllocRL/train.py:530`
- Modify: `AllocRL/train.py:546`
- Modify: `AllocRL/train.py:753`
- Modify: `AllocRL/train.py:792`
- Modify: `AllocRL/test_evaluation_scenarios.py:253`

**Interfaces:**
- Produces: `ActionPolicy.select_action(env, observation) -> int` protocol.
- Produces: `ModelActionPolicy(model)` adapter.
- Produces: `RandomValidPolicy(seed)`.
- Produces: `GreedyImmediateAreaPolicy()`.
- Produces: `evaluate_policy(policy, env, episodes=1) -> dict[str, float]`.
- Produces: `evaluate_scenarios(policy_factory, scenarios, grid_size, n_future_blocks, workspace_codes) -> list[dict]`.

- [ ] **Step 1: Write failing policy-selection tests**

Create `test_baseline_policies.py`:

```python
import numpy as np

from baseline_policies import GreedyImmediateAreaPolicy, RandomValidPolicy


class StubEnv:
    def action_masks(self):
        return np.array([True, False, True], dtype=bool)

    def immediate_placeability(self):
        return np.array([False, False, True], dtype=bool)

    def workspace_free_areas(self):
        return np.array([100.0, 999.0, 50.0], dtype=np.float32)


def test_greedy_prefers_an_immediate_candidate_over_larger_waiting_space():
    assert GreedyImmediateAreaPolicy().select_action(StubEnv(), {}) == 2


def test_random_policy_never_selects_a_masked_action():
    policy = RandomValidPolicy(seed=7)
    actions = {policy.select_action(StubEnv(), {}) for _ in range(50)}
    assert actions <= {0, 2}
```

Add environment tests for public `immediate_placeability()` and
`workspace_free_areas()` methods without changing observation keys.

- [ ] **Step 2: Run baseline tests and verify failure**

```powershell
py -B -m pytest test_baseline_policies.py -q
```

Expected: imports or public environment methods are missing.

- [ ] **Step 3: Add public evaluation-state helpers to the environment**

In `BlockPlacementEnv`, add non-mutating methods:

```python
def immediate_placeability(self) -> np.ndarray:
    return np.array(
        [candidate.placeable for candidate in self._candidate_placements],
        dtype=bool,
    )


def workspace_free_areas(self) -> np.ndarray:
    return np.maximum(self._ws_areas - self._ws_used_area, 0.0).astype(np.float32)
```

These helpers are evaluation APIs only. Do not add their results to the Stage-A
observation.

- [ ] **Step 4: Implement policy classes**

Create `baseline_policies.py`:

```python
from __future__ import annotations

from typing import Protocol

import numpy as np


class ActionPolicy(Protocol):
    name: str

    def select_action(self, env, observation) -> int:
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
```

- [ ] **Step 5: Move common evaluation into `evaluation_runner.py`**

Create the model adapter and common episode loop. Keep the metric names unchanged:

```python
class ModelActionPolicy:
    def __init__(self, model, name: str = "model"):
        self.model = model
        self.name = name

    def select_action(self, env, observation) -> int:
        action, _ = self.model.predict(
            observation,
            action_masks=env.action_masks(),
            deterministic=True,
        )
        return int(np.asarray(action).item())


def evaluate_policy(
    policy: ActionPolicy,
    env,
    episodes: int = 1,
    collect_retained_choice: bool = True,
) -> dict[str, float]:
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    values = {
        "reward": [], "terminal": [], "dropout": [],
        "delay": [], "delayed": [], "retained": [],
    }
    for _episode in range(episodes):
        observation, _ = env.reset()
        diagnostic = getattr(env, "unwrapped", env)
        total_reward = 0.0
        ratios = []
        done = False
        while not done:
            indices = (
                diagnostic.future_workspace_choice_indices()
                if collect_retained_choice
                and hasattr(diagnostic, "future_workspace_choice_indices")
                else []
            )
            before = (
                diagnostic.future_workspace_choice_count(indices)
                if indices else 0
            )
            action = policy.select_action(diagnostic, observation)
            after = (
                diagnostic.future_workspace_choice_count_after_action(
                    action, indices
                )
                if indices
                and hasattr(
                    diagnostic,
                    "future_workspace_choice_count_after_action",
                )
                else 0
            )
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if collect_retained_choice:
                ratios.append(compute_retained_choice_ratio(before, after))
            done = bool(terminated or truncated)

        result = info.get("raw_result")
        delay_days = list(result.delay_days) if result is not None else []
        dropout_count = sum(
            value == SimulationResult.DROPOUT for value in delay_days
        )
        placed = [
            value for value in delay_days
            if value != SimulationResult.DROPOUT
        ]
        values["reward"].append(total_reward)
        values["terminal"].append(float(info.get(
            "terminal_score", info.get("terminal_reward", total_reward)
        )))
        values["dropout"].append(
            dropout_count / len(delay_days) if delay_days else 0.0
        )
        values["delay"].append(float(np.mean(placed)) if placed else 0.0)
        values["delayed"].append(
            float(sum(value > DELAY_THRESHOLD for value in placed))
        )
        values["retained"].append(
            float(np.mean(ratios)) if ratios else 1.0
        )

    return {
        "mean_reward": float(np.mean(values["reward"])),
        "mean_terminal_score": float(np.mean(values["terminal"])),
        "mean_dropout_rate": float(np.mean(values["dropout"])),
        "mean_delay_days": float(np.mean(values["delay"])),
        "mean_delayed_count": float(np.mean(values["delayed"])),
        "mean_retained_choice_ratio": float(np.mean(values["retained"])),
    }
```

Import `DELAY_THRESHOLD`, `SimulationResult`, and
`compute_retained_choice_ratio` at module scope. Materialize each fixed scenario
through one policy factory so the random baseline can be seeded per scenario:

```python
def evaluate_scenarios(
    policy_factory: Callable[[int], ActionPolicy],
    scenarios: list[dict],
    grid_size: int,
    n_future_blocks: int,
    workspace_codes: list[str] | None,
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
            grid_size=grid_size,
            n_future_blocks=n_future_blocks,
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


def write_evaluation_metrics(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("At least one evaluation metric row is required")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
```

Import `Callable`, `csv`, `Path`, `BaseGridStrategy`, `materialize_scenario`, and
`select_workspaces_in_order` at module scope. Keep thin compatibility wrappers in
`train.py` so existing callers do not break in the same commit.

- [ ] **Step 6: Make original CSV evaluation one-shot**

Replace the final original evaluation call with exactly one episode:

```python
csv_metrics = evaluate_policy(
    ModelActionPolicy(model),
    eval_env,
    episodes=1,
)
```

Write one `original_csv` row. Do not loop over `args.n_eval`. Keep `--n-eval`
temporarily accepted with a deprecation warning so existing notebooks still parse;
Stage C removes it from the notebook.

Add a test with a counting environment proving that reset is called once even when
the parsed CLI value is five.

- [ ] **Step 7: Run evaluation and baseline tests**

```powershell
py -B -m pytest test_baseline_policies.py test_evaluation_scenarios.py test_rl_regressions.py -q
```

Expected: all pass.

- [ ] **Step 8: Commit the common runner**

```powershell
git add AllocRL/baseline_policies.py AllocRL/evaluation_runner.py AllocRL/alloc_env/alloc_env.py AllocRL/train.py AllocRL/test_baseline_policies.py AllocRL/test_evaluation_scenarios.py
git commit -m "feat: evaluate models and heuristics on shared scenarios"
```

---

### Task 5: Baseline CLI, Reports, and Documentation

**Files:**
- Create: `AllocRL/evaluate_baselines.py`
- Modify: `AllocRL/run_ablation.py:100`
- Modify: `AllocRL/ABLATION.md`
- Modify: `AllocRL/test_evaluation_scenarios.py:178`

**Interfaces:**
- Adds: `run_ablation.py --evaluate-baselines`.
- Produces: `output_ablation/baselines/evaluation_scenarios.csv`.
- Produces: one row per `(policy, scenario_seed)`.

- [ ] **Step 1: Add failing baseline-command tests**

Extend `test_evaluation_scenarios.py`:

```python
def test_baseline_command_uses_the_fixed_holdout_bundle(tmp_path):
    output = tmp_path / "baselines.csv"
    command = run_ablation.build_baseline_command(
        scenario_path="./data/fixed_eval_scenarios.json",
        output_path=str(output),
    )
    joined = subprocess.list2cmdline(command)
    assert "evaluate_baselines.py" in joined
    assert "fixed_eval_scenarios.json" in joined
    assert str(output) in joined
```

- [ ] **Step 2: Run the command test and verify failure**

```powershell
py -B -m pytest test_evaluation_scenarios.py -q
```

Expected: `build_baseline_command` is missing.

- [ ] **Step 3: Add the baseline entry point**

Create `AllocRL/evaluate_baselines.py`. The entry point must:

1. read schema-3 scenarios;
2. evaluate `RandomValidPolicy(seed=scenario_seed)`;
3. evaluate `GreedyImmediateAreaPolicy()`;
4. write rows through `write_evaluation_metrics` moved to `evaluation_runner.py`;
5. print aggregate mean score, dropout, and delay per policy.

Use this complete dispatch shape:

```python
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios", default="./data/fixed_eval_scenarios.json"
    )
    parser.add_argument(
        "--output",
        default="./output_ablation/baselines/evaluation_scenarios.csv",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--n-future-blocks", type=int, default=4)
    parser.add_argument(
        "--active-workspace-codes",
        default=DEFAULT_ACTIVE_WORKSPACE_CODES,
    )
    args = parser.parse_args()
    scenarios = read_scenarios(args.scenarios)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        scenarios = scenarios[:args.limit]
    workspace_codes = parse_workspace_codes(args.active_workspace_codes)
    factories = (
        lambda seed: RandomValidPolicy(seed),
        lambda _seed: GreedyImmediateAreaPolicy(),
    )
    rows = []
    for factory in factories:
        rows.extend(evaluate_scenarios(
            factory,
            scenarios,
            grid_size=args.grid_size,
            n_future_blocks=args.n_future_blocks,
            workspace_codes=workspace_codes,
        ))
    write_evaluation_metrics(args.output, rows)
    for policy_name in sorted({row["policy"] for row in rows}):
        selected = [row for row in rows if row["policy"] == policy_name]
        print(
            policy_name,
            "score=", np.mean([
                float(row["mean_terminal_score"]) for row in selected
            ]),
            "dropout=", np.mean([
                float(row["mean_dropout_rate"]) for row in selected
            ]),
            "delay=", np.mean([
                float(row["mean_delay_days"]) for row in selected
            ]),
        )


if __name__ == "__main__":
    main()
```

Import `argparse`, `numpy`, both baseline classes, scenario reading, evaluation
runner functions, and the two workspace-code helpers from `train.py`.

Expose this command from `run_ablation.py`:

```python
def build_baseline_command(scenario_path: str, output_path: str) -> list[str]:
    return [
        sys.executable,
        "evaluate_baselines.py",
        "--scenarios",
        scenario_path,
        "--output",
        output_path,
    ]
```

`--evaluate-baselines` executes the command with
`subprocess.run(command, check=True)`.

- [ ] **Step 4: Document the Stage-A evaluation protocol**

Update `ABLATION.md` to state:

- training and holdout source row/ship counts;
- exact split seed and hash rule;
- original CSV is a one-shot business reference;
- fixed holdout reporting contains 20 scenarios; Stage A does not select a checkpoint;
- random and greedy policies are required baselines;
- no Stage-A result claims CNN improvement because observation correction is still
  pending in Stage B.

- [ ] **Step 5: Run a baseline smoke evaluation**

Run one scenario first:

```powershell
py -B evaluate_baselines.py --scenarios ./data/fixed_eval_scenarios.json --output ./output_ablation/baselines/smoke.csv --limit 1
```

Expected: two rows, one for each baseline, with the same scenario seed.

Then run all 20:

```powershell
py -B run_ablation.py --evaluate-baselines
```

Expected: 40 detail rows in
`output_ablation/baselines/evaluation_scenarios.csv`.

- [ ] **Step 6: Run Stage-A focused tests**

```powershell
py -B -m pytest test_data_split.py test_training_data_profile.py test_evaluation_scenarios.py test_baseline_policies.py -q
```

Expected: all pass.

- [ ] **Step 7: Commit the baseline workflow**

```powershell
git add AllocRL/run_ablation.py AllocRL/evaluate_baselines.py AllocRL/ABLATION.md AllocRL/test_evaluation_scenarios.py
git commit -m "feat: add reproducible heuristic evaluation workflow"
```

Do not commit generated `output_ablation` result files.

---

### Task 6: Stage-A Regression and Contract Audit

**Files:**
- Modify only files with failures attributable to Stage A.

**Interfaces:**
- Verifies every Stage-A output without changing Stage-B model contracts.

- [ ] **Step 1: Run static checks**

```powershell
git diff --check
py -B -m compileall -q alloc_env baseline_policies.py evaluation_runner.py evaluation_scenarios.py evaluate_baselines.py run_ablation.py train.py
```

Expected: both commands exit zero.

- [ ] **Step 2: Run the full regression suite**

```powershell
py -B -m pytest -q
```

Expected: all tests pass, including at least the previous `116` tests and all new
Stage-A tests.

- [ ] **Step 3: Verify pinned artifacts**

Run:

```powershell
py -B -c "from evaluation_scenarios import read_scenarios, read_scenario_metadata; p='data/fixed_eval_scenarios.json'; s=read_scenarios(p); m=read_scenario_metadata(p); assert len(s)==20; assert all(len(x['blocks'])==913 for x in s); assert m['split_seed']==20260716; print(len(s), m['training_row_count'], m['holdout_row_count'])"
```

Expected:

```text
20 673 240
```

- [ ] **Step 4: Review Stage-A scope**

Confirm from `git diff` that Stage A did not change:

- observation-space keys or shapes;
- CNN/MLP layer definitions;
- reward constants or reward timing;
- action-space size;
- simulator rotation or placement behavior.

- [ ] **Step 5: Commit audit-only corrections**

If no corrections were needed, do not create an empty commit. Otherwise:

```powershell
git add <only-the-corrected-stage-a-files>
git commit -m "test: complete evaluation foundation audit"
```

Stage A is complete when all commands pass and the split/scenario artifacts are
reproducible from source.
