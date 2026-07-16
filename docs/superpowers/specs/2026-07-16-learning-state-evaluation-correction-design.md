# Learning State and Evaluation Correction Design

Date: 2026-07-16
Status: Approved design

This document supersedes the affected observation, rotation, evaluation, rollout,
and model-compatibility sections of these earlier approved designs:

- `2026-07-15-candidate-cnn-ordered-future-design.md`;
- `2026-07-16-ten-workspace-training-data-design.md`.

Requirements not explicitly changed here remain in force.

## 1. Objective

Correct the state representation and evaluation methodology of AllocRL without
replacing the existing incremental simulator, ten-workspace assignment action,
MaskablePPO training flow, or final delay/dropout objective.

The corrected system must let the policy observe assigned-but-not-placed block
queues, expose useful medium-term demand, encode the geometry enforced by the
placement strategy, and measure generalization on block groups that never enter
the training bootstrap source.

## 2. Confirmed Constraints

The following constraints are fixed for this revision:

- An action selects exactly one of the ten configured workspaces.
- The policy does not select a coordinate.
- Block rotation is physically prohibited.
- No allocation path may rotate a block as a placement fallback.
- Workspace geometry and lot geometry remain fixed across episodes.
- Every training episode contains exactly 913 target blocks.
- Every episode starts from the ten empty workspaces.
- Historical placements are not obstacles.
- MaskablePPO remains the learning algorithm.
- Pointer networks, attention, PointNet, and learned world models remain absent.
- The final task score remains the existing delay/dropout score.
- Phase 1 does not add a fragmentation reward or other shaped reward.

The ten workspace action order remains:

1. `PE049`
2. `PE050`
3. `PE055`
4. `PE054`
5. `PE056`
6. `PE048`
7. `PE044`
8. `PE059`
9. `PE060`
10. `PE061`

## 3. Evidence Driving the Revision

The current code and source data establish these conditions:

- The source contains 913 eligible targets after excluding July and November.
- A placement-first heuristic encountered assigned-but-not-placed queues in 473
  of 913 decision states, with a maximum observed queue size of 17.
- A resolved outcome was observed as late as 66 decisions after its assignment.
- About 85 percent of consecutive source blocks have the same start date.
- The fourth following block is within three calendar days in 95 percent of the
  source sequence, so four ordered slots are not medium-term demand.
- `PE044` occupies only 64 by 5 pixels in the current aspect-preserving raster.
- In one complete source episode, 53 of 4,778 immediately placeable candidate
  masks rendered as zero pixels.
- The default deterministic CSV evaluation repeats the same episode five times.
- Training and evaluation currently draw physical block rows from the same 913
  source rows.
- Without rotation, every current target still has at least five dimension-valid
  workspaces. No current target becomes globally infeasible.

## 4. Placement Semantics: Rotation Is Removed

Rotation prohibition applies consistently at every boundary.

### 4.1 Hard constraint

`DimensionConstraint` accepts a block only when:

```text
block.length <= workspace.length
and block.breadth <= workspace.breadth
```

The swapped-dimension test is removed. Height and weight remain domain fields but
do not become active hard constraints until finite workspace limits are supplied.

### 4.2 Candidate calculation

Candidate generation calls the placement strategy exactly once with the original
length and breadth. A failed call produces an unplaceable candidate. It does not
call `Block.turn()`.

`CandidatePlacement` contains only:

```text
position: Optional[(float, float)]
length: float
breadth: float
```

The `rotated` field is removed so a later caller cannot silently reintroduce a
rotation path.

### 4.3 Simulation and diagnostics

Both `IncrementalPlacementSimulator` and `PlacementSimulator` attempt only the
original orientation. Evaluation preview helpers use the same single attempt.
`Block.turn()` may remain as a domain utility and unit-tested operation, but no
training, evaluation, replay, candidate, or visualization path may call it.

Tests must prove that a block which fits only after swapping dimensions is masked,
has an all-zero candidate channel, and remains unplaceable in both simulators.

## 5. Data Split and Evaluation

### 5.1 Group-disjoint split

The split unit is `ship_no`, not an individual block row. This prevents sibling
blocks from one ship appearing in both training and holdout sources.

Every eligible target must have a non-empty `ship_no`. Split preparation rejects
the dataset with a clear validation error if this invariant is violated.

For each unique ship number, calculate:

```text
digest = SHA256("20260716:" + ship_no)
fraction = unsigned_big_endian(digest[0:8]) / 2**64
holdout = fraction < 0.20
```

For the current CSV this produces:

```text
training source rows: 673
holdout source rows: 240
training ship groups: 29
holdout ship groups: 11
```

The holdout contains source templates in all seven included months. Tests pin the
current counts and prove that the ship-number sets are disjoint.

The split seed, source-file SHA256, selected ship groups, row counts, and month
counts are written to `data_split_manifest.json` beside generated scenarios.

### 5.2 Episode count and month profile

The bootstrap source row count is independent of episode length. Training still
generates exactly 913 blocks from the 673 training rows.

The full-data empirical month target remains:

```text
64, 122, 106, 142, 153, 151, 175
```

The generator accepts an explicit target month profile, separate from the source
template counts. This preserves the approved 80 percent balanced and 20 percent
empirical training policy after the source split.

Holdout evaluation scenarios contain 913 blocks and use the exact empirical target
profile. Each month samples only holdout templates from that month.

### 5.3 Evaluation sets

Evaluation has three named sources:

1. `original_csv`: the deterministic full 913-row business reference, run once.
2. `holdout_fixed`: 20 fixed 913-block scenarios with seeds 1000 through 1019.
3. `training_diagnostic`: synthetic training-source episodes used only for learning
   curves, never for model selection.

`original_csv` ignores `n_eval`; duplicate deterministic runs are rejected. Final
model comparison and best-model selection use `holdout_fixed`.

### 5.4 Non-learning baselines

Every fixed holdout scenario is evaluated with:

- a seeded random hard-valid workspace policy;
- `GreedyImmediateAreaPolicy`, which selects the immediately placeable workspace
  with the greatest current free rectangular area and otherwise selects the
  largest hard-valid workspace.

Baseline results use the same terminal score, dropout rate, delay metrics, and
scenario files as learned policies.

## 6. Observation Contract

The revised observation schema version is 3. Constants are:

```text
N_WORKSPACES = 10
GRID_SIZE = 64
ORDERED_FUTURE_COUNT = 16
PENDING_QUEUE_SLOTS = 32
FUTURE_DAY_WINDOWS = [(0, 5), (6, 20), (21, 60)]
FUTURE_DAY_NORMALIZER = 30 working days
```

All arrays are `float32` and remain in `[0, 1]`.

| Key | Shape | Meaning |
| --- | --- | --- |
| `block` | `(8,)` | Current block and decision context |
| `grids` | `(N, 4, 64, 64)` | Candidate-conditioned workspace geometry |
| `ws_meta` | `(N, 4)` | Physical dimensions, occupancy, placeability |
| `future_blocks` | `(16, 6)` | Next unassigned decisions in exact order |
| `future_mask` | `(16,)` | Valid future slots |
| `future_demand` | `(3, 4)` | Medium-term demand windows |
| `pending_blocks` | `(N, 32, 7)` | Assigned but unresolved queue slots |
| `pending_mask` | `(N, 32)` | Valid pending slots |
| `pending_summary` | `(N, 4)` | Total and overflow queue pressure |

### 6.1 Current block

`block` contains:

1. length divided by source maximum length;
2. breadth divided by source maximum breadth;
3. original working-day duration divided by source maximum duration;
4. current environment date position in the source date span;
5. `min(length, breadth) / max(length, breadth)`;
6. assigned-decision progress divided by 912;
7. block area divided by maximum workspace area;
8. maximum block axis divided by maximum workspace axis.

Height and weight remain in `Block` and generated records but are removed from the
policy observation because no current workspace limit or placement rule uses them.

### 6.2 Ordered future blocks

The next 16 unassigned decision blocks remain ordered and attention-free. Each row
contains:

1. normalized length;
2. normalized breadth;
3. normalized working-day duration;
4. working days until arrival divided by 30 and clipped to 1;
5. aspect ratio;
6. area divided by maximum workspace area.

Assigned pending blocks do not appear here because they have their own exact queue
section.

### 6.3 Medium-term future demand

For each working-day window, aggregate every unassigned block whose current start
date falls in the window. The four values are:

1. count divided by 913;
2. total area divided by four times the total workspace area and clipped to 1;
3. mean original duration divided by the source maximum duration;
4. maximum block area divided by the maximum workspace area.

Empty windows contain zeros.

### 6.4 Pending queues

A pending queue item is a block with a workspace assignment whose final delay is
not known and which has not been placed. Items are grouped by assigned workspace
and sorted by the simulator retry key, then by block index for deterministic ties.

Each of the first 32 rows contains:

1. normalized length;
2. normalized breadth;
3. normalized original duration;
4. current working-day delay divided by the dropout threshold and clipped to 1;
5. aspect ratio;
6. area divided by the assigned workspace area and clipped to 1;
7. maximum block axis divided by the assigned workspace maximum axis.

`pending_summary` contains, for the full queue including overflow:

1. total queue count divided by 913;
2. total queue area divided by four times workspace area and clipped to 1;
3. maximum current delay divided by the dropout threshold and clipped to 1;
4. overflow count beyond 32 divided by 913.

This bounded representation exposes every queue observed in the reviewed
placement-first episode and preserves aggregate pressure if a worse policy exceeds
32 items.

## 7. Candidate-Conditioned Grid

### 7.1 Coordinate mapping

Each workspace maps its X and Y axes independently to the full 64 by 64 grid:

```text
x_px_per_m = 64 / workspace.length
y_px_per_m = 64 / workspace.breadth
```

Workspace length and breadth remain available in `ws_meta`, so the shared CNN can
recover physical scale without compressing a narrow workspace to a few rows.

Rectangle lower bounds use `floor`; upper bounds use `ceil`. Any positive-area
rectangle intersecting the workspace receives at least one pixel on each axis.

### 7.2 Channel meanings

The four channels are:

1. existing-block collision exclusion zones;
2. normalized remaining working days over those exclusion zones;
3. post-candidate lot state;
4. the current candidate's resulting collision exclusion zone.

Existing-block exclusion rectangles expand each physical block by
`SAFETY_DISTANCE` on every side and clip at the workspace boundary. This directly
represents the region into which a new physical candidate may not extend.

The candidate channel expands the selected candidate by `SAFETY_DISTANCE` to show
the space unavailable to later blocks after the action. Exact immediate
placeability still comes from `ws_meta`, not from a CNN classification.

Channel 2 uses these values:

```text
0.00 = no lot region
0.25 = available lot region
1.00 = unavailable lot after previewing the current candidate
```

For a workspace without lots, the full interior uses 0.25. Overlapping lot values
take the maximum. If the current candidate is placeable, lot status is computed on
a non-mutating preview containing that candidate; otherwise it describes the
current workspace.

Remaining time uses working days and is clipped at 60 working days. The base grid
cache stores only state that is independent of the current candidate; lot preview
and candidate channels are rebuilt per decision.

### 7.3 Workspace metadata

`ws_meta` contains:

1. workspace length divided by maximum workspace length;
2. workspace breadth divided by maximum workspace breadth;
3. placed physical block area divided by workspace area;
4. exact immediate placeability of the current block.

## 8. Feature Extractors

The three existing ablation extractors remain:

- `structured`;
- `fixed-grid`;
- `candidate-cnn`.

All three consume current, future, demand, pending, and workspace metadata. Only
the grid path differs.

The structured encoder uses:

```text
current block:       Linear(8, 64) -> ReLU -> Linear(64, 32) -> ReLU
ordered future:      Linear(16 * 7, 128) -> ReLU -> Linear(128, 64) -> ReLU
future demand:       Linear(12, 64) -> ReLU -> Linear(64, 32) -> ReLU
workspace pending:   Linear(32 * 8, 128) -> ReLU -> Linear(128, 64) -> ReLU
```

The extra element per ordered or pending slot is its mask. Fixed flattening
preserves order without attention.

For each workspace, concatenate:

```text
current context:       32
future context:        64
demand context:        32
pending context:       64
pending summary:        4
workspace metadata:     4
grid context:            0, 256 fixed pooled values, or 128 CNN values
```

The shared workspace fusion remains `Linear(input, 128) -> ReLU -> Linear(128, 64)
-> ReLU`. The ten 64-value workspace embeddings are flattened in fixed action order
and projected to the configured 256 policy features.

The candidate CNN keeps shared weights and GroupNorm. It learns end-to-end through
the actor and critic losses; it is not trained as a feasibility classifier.

## 9. Reward and Credit Assignment

Phase 1 preserves the current scoring function:

```text
delay <= 2: +1.0
delay 3..7: -(delay - 2) / 5
dropout: -2.0
episode task score: mean block score
```

Newly resolved outcomes are emitted exactly once, and terminal residual correction
keeps:

```text
sum(environment rewards) == terminal task score
```

The queue observation and longer future context are evaluated before reward
shaping. Potential-based shaping becomes a separate revision only when the
corrected candidate CNN fails either of these gates:

- absolute holdout terminal-score improvement of 0.05 over the structured model;
- relative holdout dropout reduction of 10 percent over the structured model.

Any later shaping design must telescope to zero over an episode after terminal
correction so task-score conservation remains true.

## 10. Training and Model Selection

### 10.1 Staged budgets

Training is interpreted in stages:

```text
smoke:       20,000 timesteps, 1 seed
screening:  300,000 timesteps, seeds 0, 1, 2
final:    1,000,000 timesteps, seeds 0, 1, 2, 3, 4
```

The final stage runs only configurations that pass smoke and are competitive in
screening. A single 100,000-timestep run is not treated as convergence evidence.

Screening compares `gae_lambda` values 0.98 and 0.995 and rollout sizes 512 and
960 while keeping total timesteps, minibatch size, learning rate, and seed sets
equal. The selected pair is fixed before final comparison.

### 10.2 Periodic holdout evaluation

Every 50,000 training timesteps, evaluate seeds 1000 through 1004 from the fixed
holdout set. Save `best_model.sb3` by mean terminal score, breaking ties by lower
dropout rate and then lower mean delay. Final reporting evaluates all 20 holdout
scenarios from the selected checkpoint.

Training callbacks may record synthetic episode scores but may not select a model
from them.

### 10.3 Resume correctness

`run_config.json` records and compatibility-checks:

- observation, reward, and training-data schema versions;
- extractor and feature dimensions;
- workspace order;
- future and pending constants;
- data split seed and source SHA256;
- learning rate, rollout size, batch size, epoch count, gamma, and GAE lambda.

Auto-resume rejects incompatible settings. It selects the readable final model or
checkpoint with the greatest stored `num_timesteps`; a stale final model does not
override a newer interrupted-run checkpoint.

The revised observation is incompatible with schema-version-2 models. Colab uses
a new output directory:

```text
/content/drive/MyDrive/CNN-RL-outputs/candidate_cnn_state_v3
```

### 10.4 Diagnostic overhead

CNN diagnostic observations are copied once at rollout end, not on every
environment step. Gradient norm, weight change, workspace feature variance, and
candidate-channel sensitivity remain logged.

## 11. Ablation and Acceptance

The corrected comparison matrix is:

| ID | Extractor | Future/pending state | Purpose |
| --- | --- | --- | --- |
| A | `structured` | current only | Minimum learned baseline |
| B | `structured` | full corrected state | Value of future and queue state |
| C | `fixed-grid` | full corrected state | Direct raster baseline |
| D | `candidate-cnn` | current only | CNN without future state |
| E | `candidate-cnn` | full corrected state | Complete candidate model |

Every row uses the same split, fixed holdout scenarios, no-rotation semantics,
reward, training budget, and seed set.

The candidate CNN is accepted only when:

- E improves over B by at least 0.05 absolute terminal score or 10 percent relative
  dropout reduction;
- the improvement direction holds in at least four of five final seeds;
- E is not worse than `GreedyImmediateAreaPolicy` on mean holdout terminal score;
- CNN gradient norm and weight change remain nonzero during training;
- no increase in globally infeasible source blocks is introduced.

## 12. Required Tests

Focused tests must cover:

1. no-rotation dimension masks;
2. no rotation in candidate, incremental, replay, and preview paths;
3. group-disjoint deterministic split and pinned current counts;
4. fixed 913-block generation from 673 training or 240 holdout templates;
5. empirical and balanced monthly profile conservation;
6. pending queue ordering, masks, overflow summaries, and observation sensitivity;
7. 16-slot future ordering and 30-working-day arrival normalization;
8. three future-demand window boundaries;
9. independent-axis raster mapping and minimum one-pixel rectangles;
10. collision exclusion, working-day lifetime, and post-candidate lot channels;
11. candidate observation and simulator placement agreement;
12. extractor shape, mask invariance, order sensitivity, and nonzero CNN updates;
13. reward conservation within `1e-6`;
14. original evaluation executes once and holdout evaluation executes 20 scenarios;
15. learned and heuristic policies consume the same scenario files;
16. best-model tie breaking;
17. resume hyperparameter rejection and newest-timestep selection;
18. ONNX export and standalone visualization with schema version 3;
19. full regression suite;
20. short save/load/evaluate smoke training for all three extractors.

## 13. Implementation Boundaries

Implementation is split into three independently reviewable stages:

### Stage A: Evaluation foundation

Add the ship-group split, explicit month profiles, holdout scenarios, deterministic
original evaluation, heuristic baselines, and evaluation tests. No model shape
changes occur in this stage.

### Stage B: State and geometry correction

Remove rotation from every allocation path, add pending and future-demand state,
replace grid semantics, update extractors, increment schema versions, and update
visualization and ONNX. Existing schema-version-2 checkpoints are rejected.

### Stage C: Training operations and experiments

Add periodic fixed-holdout selection, complete resume compatibility, lower-cost CNN
diagnostics, staged ablation commands, and the new Colab output configuration.

Potential-based reward shaping is not part of Stages A through C and requires a
new approved design after corrected ablation evidence is available.

## 14. Completion Definition

The revision is complete only when:

- all allocation paths prohibit rotation;
- all current source blocks retain at least one hard-valid workspace;
- assigned pending queues change the observation before they resolve;
- every positive-area placeable candidate renders at least one pixel;
- train and holdout ship groups are disjoint;
- deterministic original evaluation is not duplicated;
- fixed holdout and heuristic results are written alongside learned results;
- reward conservation remains within `1e-6`;
- all focused tests and the full regression suite pass;
- all three extractors complete save/load/evaluate smoke training;
- the Stage C final report applies the acceptance rules in Section 11.
