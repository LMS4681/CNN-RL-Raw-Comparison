# Scale-Aware CNN and Six-Hour Colab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make workspace scale and non-rotatable block geometry easier to use, pretrain the candidate-CNN/structured feature extractor on simulator-derived future-quality targets, then run a restart-safe six-hour PPO fine-tuning stage with eight parallel environments and a monotonically decreasing learning rate.

**Architecture:** Keep the existing ten-workspace, 4-channel candidate-grid, action-mask, reward-schema-2, 913-block, no-rotation environment flow. Append scale-aware structured features and bump the observation contract to schema 4. Split learning into two persisted stages: first train the CNN plus structured/fusion MLPs with exact one-step geometry targets and bounded eight-block replay targets generated from cloned simulator states; then initialize PPO from that checkpoint, freeze the extractor for the first 50,000 cumulative timesteps, and unfreeze it for joint fine-tuning at one tenth of the policy learning rate. Run eight `SubprocVecEnv` workers with 120 steps each so every PPO update still contains 960 transitions.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, PyTorch, Stable-Baselines3, sb3-contrib MaskablePPO, pytest, Jupyter/Google Colab, Google Drive, Git/GitHub.

## Global Constraints

- Do not modify or resume `/content/drive/MyDrive/CNN-RL-comparison/overnight-20260721`.
- Do not move the immutable `overnight-v1` tag. The improved release uses a new tag and a new Drive output root.
- Do not modify reward schema 2, action masking, placement order, episode length, ten-workspace ordering, block generation, candidate placement, or evaluation scenarios.
- Block rotation remains forbidden. Length and breadth are orientation-specific; never sort or swap them while encoding ratios.
- Keep all four current grid channels in the executable model: occupancy, remaining lifetime, lot usage, and current candidate footprint. The already prepared three-channel explanatory report is a separate reporting artifact and is not rewritten by this work.
- Do not start PPO from random candidate-CNN/structured extractor weights. Stage 1 must publish a verified pretraining checkpoint before Stage 2 starts.
- During Stage 2, freeze the complete feature extractor only for the first 50,000 cumulative PPO timesteps, then unfreeze it and fine-tune it at `0.1 * policy_learning_rate`. Never freeze it permanently and never replace convolution kernels with all-one filters.
- The six-hour budget applies only to Stage 2 PPO training. Stage 1 dataset generation and pretraining complete before that timer starts.
- Do not add attention, pointer networks, coordinate channels, reward changes, or auxiliary losses to Stage 2 PPO. Stage 1 uses only the simulator-supervised prediction losses defined below.
- Existing schema-3 checkpoints are incompatible with schema 4 and must fail closed. The improved run starts from timestep zero in a new output root.
- Use TDD. Each implementation task begins with a failing focused test and ends with focused tests plus a commit.

---

## Frozen Observation Contract

Append fields to preserve the meaning and order of all existing positions.

`ws_meta` changes from `(10, 4)` to `(10, 8)`:

```text
0 workspace.length / max_workspace_length
1 workspace.breadth / max_workspace_breadth
2 placed_area / workspace_area
3 current_block_is_placeable
4 current_block.length / workspace.length
5 current_block.breadth / workspace.breadth
6 current_block.area / workspace.area
7 min(workspace.length, workspace.breadth) / max(workspace.length, workspace.breadth)
```

`future_demand` changes from `(3, 4)` to `(3, 6)` for each existing day window:

```text
0 block_count / 913
1 total_block_area / (4 * total_workspace_area)
2 mean_original_duration / max_duration
3 max_block_area / max_workspace_area
4 max_block_length / max_length
5 max_block_breadth / max_breadth
```

Every value remains finite `np.float32` clipped to `[0, 1]`. Empty future windows remain all zero. The new widths imply:

```text
StructuredStateEncoder.demand input: 3 * 6 = 18
workspace fusion input: previous width + 4
raw-direct output width: 2772 + 40 + 6 = 2818
OBSERVATION_SCHEMA_VERSION: 3 -> 4
```

The current workspace length/breadth metadata already lets the policy distinguish physical workspace scales after CNN fusion. The added current-block/workspace ratios remove the need for the fusion MLP to learn division indirectly, while the future maximum length and breadth preserve geometry that an area-only summary loses.

---

## Two-Stage Learning Contract

There is no unique correct 256-dimensional CNN feature vector. Stage 1 instead trains the feature extractor to predict deterministic or explicitly bounded simulator outcomes. Any internal representation is acceptable if it predicts held-out targets, depends on the grid rather than only scalar inputs, and improves the later PPO optimization gates.

### Stage 1: Simulator-Supervised Feature Pretraining

For every collected environment state, retain the full schema-4 observation and compute targets for every hard-valid workspace action:

```text
input
  10 x 4 x 64 x 64 candidate grids
  current block, future 16 blocks, future demand
  pending queues, pending summaries, workspace scale metadata

exact targets
  current_placeable[action]
  future_fit[action, future_slot]
  future_optionality_after[action]
  future_optionality_delta[action]
  largest_free_rectangle_ratio[action]
  free_component_count_normalized[action]

bounded-replay targets
  replay_success_rate[action]
  replay_dropout_rate[action]
  replay_delay_ratio[action]
```

`future_fit[action, future_slot]` means whether the selected workspace, after applying the observed current candidate without rotation, can place that future block. `future_optionality_after` is the total feasible future block/workspace pair count divided by `ORDERED_FUTURE_COUNT * N_WORKSPACES`. Geometry targets use the post-candidate union of occupancy channel 0 and candidate-footprint channel 3.

The bounded replay is exact for a fixed teacher, not a global optimal-planning label. Clone the complete environment state, apply the candidate action, then resolve the next eight future blocks with a deterministic future-optionality teacher. At each decision the teacher chooses the hard-valid action with the greatest `future_workspace_choice_count_after_action`; ties use greater free area and then lower workspace index. Stop after eight additional resolved blocks or 32 decisions. Record placed/resolved success, dropout, and normalized delay separately rather than combining them into one hand-weighted score.

Collect 5,000 training states from synthetic episode seeds `20000..20039` and 1,000 validation states from seeds `30000..30009`. Alternate `RandomValidPolicy` and `GreedyImmediateAreaPolicy` by episode. Use only the training source split; never collect pretraining labels from fixed holdout or primary-test scenarios. Compute bounded replay labels for every fourth collected state and mark the remaining replay rows unavailable so replay loss is masked.

Persist sharded, compressed `.npz` data with grids stored as `float16` and all structured arrays/targets restored to `float32` by the loader. Publish a manifest containing dataset schema version, source manifest hash, observation schema version, episode seeds, collector policy, target normalizers, shard hashes, and config hash.

Train `CandidateCnnExtractor` plus its structured/fusion MLPs and temporary auxiliary heads. The heads are discarded after pretraining; only the extractor state is transferred to PPO. Use AdamW, learning rate `1e-4`, batch size 8, at most 30 epochs, and early stopping after five validation epochs without improvement.

Stage 1 passes only when all conditions hold:

```text
all validation losses and outputs are finite
future-fit BCE improves at least 5% over a no-grid scalar baseline
optionality MAE improves at least 5% over a validation-mean baseline
shuffled-grid total loss is at least 5% worse than normal-grid loss
same-area/different-fragmentation counterfactuals produce different geometry predictions
```

Publish `PRETRAINING_COMPLETE.json` last. It records the best checkpoint SHA256 and every gate result. Stage 2 must reject a missing or failed marker.

### Stage 2: PPO Warm-Up and Joint Fine-Tuning

Initialize a new schema-4 `MaskablePPO` model, load the verified extractor weights, and discard the auxiliary heads:

```text
0 .. 49,999 cumulative PPO timesteps
  extractor requires_grad=False
  PPO actor/value networks train at the policy LR

50,000+ cumulative PPO timesteps
  extractor requires_grad=True
  PPO actor/value networks use the policy LR schedule
  extractor uses exactly 0.1 times the current policy LR
```

Freeze state is derived from absolute `model.num_timesteps`, so Colab resume cannot restart the warm-up. The pretraining checkpoint SHA256, manifest SHA256, gate receipt SHA256, freeze boundary, and LR multiplier are structural fields in `run_config.json`. A mismatch fails before model loading or learning.

The persisted artifacts are:

```text
pretraining/candidate_encoder_pretrained.pt
pretraining/pretraining_metrics.json
pretraining/PRETRAINING_COMPLETE.json
ppo/block_placement_ppo.sb3
ppo/run_config.json
ppo/run_state.json
```

The final deployable artifact is the SB3 PPO model. The standalone encoder checkpoint is an intermediate reproducibility artifact, not a second deployed placement model.

---

### Task 1: Lock Scale-Aware Observation Behavior in Tests

**Files:**
- Modify: `AllocRL/test_candidate_observation.py`
- Modify: `AllocRL/test_future_block_lookahead.py`
- Modify: `AllocRL/test_parallel_training_config.py`
- Modify: `AllocRL/test_rl_regressions.py`

**Interfaces:**
- `AllocEnv.reset()` and `AllocEnv.step()` must return schema-4 arrays.
- `encode_future_demand(...)` must return `(3, 6)` without mutating blocks.

- [ ] **Step 1: Add a failing workspace-scale test**

Build two workspaces with different physical dimensions and the same `64 x 64` normalized grid state. Use a non-square current block and assert exact orientation-specific fields:

```python
np.testing.assert_allclose(
    obs["ws_meta"][workspace_index, 4:],
    np.array([
        block.length / workspace.length,
        block.breadth / workspace.breadth,
        block.length * block.breadth / (workspace.length * workspace.breadth),
        min(workspace.length, workspace.breadth)
        / max(workspace.length, workspace.breadth),
    ], dtype=np.float32),
)
assert obs["ws_meta"].shape == (10, 8)
```

Add a second assertion proving that swapping the test block's length and breadth changes fields 4 and 5 rather than being treated as a legal rotation.

- [ ] **Step 2: Add failing future-geometry tests**

Use future blocks with equal area but different dimensions, such as `10 x 5` and `5 x 10`. Assert that area statistics can match while columns 4 and 5 preserve separate maximum length and breadth. Update boundary-window expectations from `(3, 4)` to `(3, 6)`.

- [ ] **Step 3: Add failing schema and memory tests**

Require:

```python
assert WORKSPACE_META_FEATURE_DIM == 8
assert FUTURE_DEMAND_FEATURE_DIM == 6
assert OBSERVATION_SCHEMA_VERSION == 4
```

Update the rollout float-count expectation to include `10 * 8` workspace metadata values and `3 * 6` future-demand values.

- [ ] **Step 4: Verify RED**

```powershell
cd D:\Sub\Allocation\CNN-RL-Raw-Comparison
python -m pytest AllocRL/test_candidate_observation.py AllocRL/test_future_block_lookahead.py AllocRL/test_parallel_training_config.py AllocRL/test_rl_regressions.py -q
```

Expected: failures report the old `(10, 4)`, `(3, 4)`, and schema-3 contracts.

---

### Task 2: Implement Schema-4 Encoders Without Changing the Environment Flow

**Files:**
- Modify: `AllocRL/alloc_env/observation_state.py`
- Modify: `AllocRL/alloc_env/alloc_env.py`
- Modify: `AllocRL/train.py`

**Interfaces:**
- `WORKSPACE_META_FEATURE_DIM = 8`
- `FUTURE_DEMAND_FEATURE_DIM = 6`
- `OBSERVATION_SCHEMA_VERSION = 4`

- [ ] **Step 1: Expand the constants and observation space**

Change only the two dimensions and schema version. Keep all keys and grid shapes unchanged.

- [ ] **Step 2: Append future maximum dimensions**

Append these values to each non-empty window in `encode_future_demand`:

```python
max(block.length for block in window_blocks) / scales.max_length,
max(block.breadth for block in window_blocks) / scales.max_breadth,
```

Pass the final list through the existing finite-value validation and clipping helper.

- [ ] **Step 3: Append per-workspace current-block ratios**

In `_get_obs`, validate positive finite workspace dimensions, derive the current block once, and append the four frozen fields after `placeable`. Use direct `length/length` and `breadth/breadth` ratios; do not use `max(axis)` for these orientation-specific fields.

- [ ] **Step 4: Update schema error text**

Replace hard-coded references to schema 3 in terminal-observation comments and compatibility errors. Error text must say that schema-3 models cannot be used with schema 4.

- [ ] **Step 5: Verify GREEN and commit**

```powershell
python -m pytest AllocRL/test_candidate_observation.py AllocRL/test_future_block_lookahead.py AllocRL/test_parallel_training_config.py AllocRL/test_rl_regressions.py -q
git add AllocRL/alloc_env/observation_state.py AllocRL/alloc_env/alloc_env.py AllocRL/train.py AllocRL/test_candidate_observation.py AllocRL/test_future_block_lookahead.py AllocRL/test_parallel_training_config.py AllocRL/test_rl_regressions.py
git commit -m "feat: expose workspace-relative block geometry"
```

Expected: focused tests pass and no reward/action code changes appear in `git diff`.

---

### Task 3: Align Every Feature Extractor and Resume Contract

**Files:**
- Modify: `AllocRL/alloc_env/cnn_extractor.py`
- Modify: `AllocRL/comparison/raw_direct_extractor.py`
- Modify: `AllocRL/train.py`
- Modify: `AllocRL/smoke_test.py`
- Modify: `AllocRL/test_feature_extractors.py`
- Modify: `AllocRL/test_raw_direct_extractor.py`
- Modify: `AllocRL/test_train_resume_cli.py`
- Modify: `AllocRL/test_smoke_workflows.py`
- Modify: `AllocRL/test_training_visualization.py`

**Interfaces:**
- `EXPECTED_OBSERVATION_SHAPES["ws_meta"] == (10, 8)`
- `EXPECTED_OBSERVATION_SHAPES["future_demand"] == (3, 6)`
- `RAW_DIRECT_FEATURE_DIM == 2818`

- [ ] **Step 1: Write failing dimension and incompatibility tests**

Require the structured demand encoder's first linear layer to accept 18 values. Increase each workspace-fusion input by four. Require raw-direct output `(batch, 2818)` with its existing concatenation order. Add a resume test in which a schema-3 sidecar/model is rejected before loading.

Expected fusion widths after the four appended `ws_meta` fields:

```text
StructuredExtractor: 204
FixedGridExtractor: 460
CandidateCnnExtractor: 332
```

- [ ] **Step 2: Verify RED**

```powershell
python -m pytest AllocRL/test_feature_extractors.py AllocRL/test_raw_direct_extractor.py AllocRL/test_train_resume_cli.py AllocRL/test_smoke_workflows.py AllocRL/test_training_visualization.py -q
```

- [ ] **Step 3: Update extractor widths and run config**

Change the demand linear input from 12 to 18, schema shape constants, fusion width calculation, raw-direct width, and `extractor_output_dim`. Import `RAW_DIRECT_FEATURE_DIM` where practical instead of duplicating `2818` in production code.

- [ ] **Step 4: Preserve the differentiable paths needed by pretraining and later fine-tuning**

Keep the existing candidate-CNN gradient/weight-change diagnostics passing. Add no permanent `requires_grad=False`, fixed kernels, or detached CNN/fusion outputs. Temporary PPO warm-up freezing is implemented only in Task 7 and is controlled by absolute timestep.

- [ ] **Step 5: Verify and commit**

```powershell
python -m pytest AllocRL/test_feature_extractors.py AllocRL/test_raw_direct_extractor.py AllocRL/test_train_resume_cli.py AllocRL/test_smoke_workflows.py AllocRL/test_training_visualization.py AllocRL/test_cnn_diagnostics.py -q
git add AllocRL/alloc_env/cnn_extractor.py AllocRL/comparison/raw_direct_extractor.py AllocRL/train.py AllocRL/smoke_test.py AllocRL/test_feature_extractors.py AllocRL/test_raw_direct_extractor.py AllocRL/test_train_resume_cli.py AllocRL/test_smoke_workflows.py AllocRL/test_training_visualization.py
git commit -m "feat: align extractors with observation schema 4"
```

---

### Task 4: Generate Simulator-Supervised Targets and Sharded Datasets

**Files:**
- Create: `AllocRL/pretraining/__init__.py`
- Create: `AllocRL/pretraining/targets.py`
- Create: `AllocRL/pretraining/dataset.py`
- Modify: `AllocRL/alloc_env/alloc_env.py`
- Modify: `AllocRL/alloc_env/incremental_simulator.py`
- Create: `AllocRL/test_pretraining_targets.py`
- Create: `AllocRL/test_pretraining_dataset.py`

**Interfaces:**
- Produces: `AllocEnv.clone_for_diagnostics() -> AllocEnv`
- Produces: `build_auxiliary_targets(env: AllocEnv, *, include_replay: bool) -> AuxiliaryTargets`
- Produces: `collect_pretraining_dataset(config: PretrainingDataConfig, output_dir: Path) -> Path`

- [ ] **Step 1: Write failing state-clone isolation tests**

Create an environment at a noninitial decision, clone it, apply different actions to the original and clone, and require independent blocks, workspaces, pending sets, delay arrays, candidate placements, dates, grid caches, and RNG state. The clone must initially produce byte-equal observations and action masks.

```python
clone = env.clone_for_diagnostics()
assert_observations_equal(env._get_obs(), clone._get_obs())
assert np.array_equal(env.action_masks(), clone.action_masks())
clone.step(int(np.flatnonzero(clone.action_masks())[0]))
assert_observations_equal(original_before, env._get_obs())
```

- [ ] **Step 2: Run the clone test and verify RED**

```powershell
python -m pytest AllocRL/test_pretraining_targets.py -k clone -q
```

Expected: failure because `clone_for_diagnostics` does not exist.

- [ ] **Step 3: Implement explicit diagnostic cloning**

Clone the incremental simulator's current blocks, workspaces, assignments, delay days, pending indices, date, current index, transition results, original inputs, and dropout settings. Rebuild environment candidate placements and grid cache from the cloned simulator. Do not use serialization or replay from episode start, and do not share mutable lists or NumPy arrays.

- [ ] **Step 4: Write exact target tests**

Define:

```python
@dataclass(frozen=True)
class AuxiliaryTargets:
    action_mask: np.ndarray                 # (10,) bool
    current_placeable: np.ndarray           # (10,) float32
    future_fit: np.ndarray                  # (10, 16) float32
    future_optionality_after: np.ndarray    # (10,) float32
    future_optionality_delta: np.ndarray    # (10,) float32 in [-1, 1]
    largest_free_rectangle_ratio: np.ndarray  # (10,) float32
    free_component_count_normalized: np.ndarray  # (10,) float32
    replay_success_rate: np.ndarray         # (10,) float32
    replay_dropout_rate: np.ndarray         # (10,) float32
    replay_delay_ratio: np.ndarray          # (10,) float32
    replay_mask: np.ndarray                 # (10,) bool
```

Require target generation to leave the source environment unchanged, keep invalid-action losses masked, preserve length/breadth orientation, and match `future_workspace_choice_count_after_action` for optionality. Use same-area grids with different fragmentation to prove that the largest-rectangle target differs.

- [ ] **Step 5: Implement exact and bounded-replay targets**

For each hard-valid action, create a fresh diagnostic clone. Apply the observed candidate without rotation. Compute `future_fit` for the selected workspace and the exact observed future indices. Compute largest empty axis-aligned rectangle and four-neighbor free-component count from occupancy channel 0 union candidate channel 3.

When `include_replay=True`, apply the action and run the deterministic teacher until eight additional blocks resolve or 32 decisions execute. At each decision, choose the action with maximum post-action future optionality; tie-break on greater free area and then lower action index. Normalize delay by `8 * DROPOUT_THRESHOLD` and clip target arrays to their documented ranges.

- [ ] **Step 6: Write dataset split, dtype, and integrity tests**

Require disjoint seed sets, exactly 5,000 training and 1,000 validation states, alternating collector policies, replay labels on state indices divisible by four, `float16` stored grids, `float32` loader outputs, atomic shard publication, and verified SHA256 values in `dataset_manifest.json`. A truncated or modified shard must fail before loading any samples.

- [ ] **Step 7: Implement the sharded collector**

Use `RandomValidPolicy(seed)` for even-numbered episodes and `GreedyImmediateAreaPolicy` for odd-numbered episodes. Save 100 states per compressed NPZ shard. Write all shards to temporary names, hash and rename them, then publish `dataset_manifest.json` last.

- [ ] **Step 8: Verify and commit**

```powershell
python -m pytest AllocRL/test_pretraining_targets.py AllocRL/test_pretraining_dataset.py AllocRL/test_candidate_observation.py AllocRL/test_no_rotation.py -q
git add AllocRL/pretraining AllocRL/alloc_env/alloc_env.py AllocRL/alloc_env/incremental_simulator.py AllocRL/test_pretraining_targets.py AllocRL/test_pretraining_dataset.py
git commit -m "feat: generate simulator-supervised CNN targets"
```

---

### Task 5: Pretrain and Verify the Candidate Feature Extractor

**Files:**
- Create: `AllocRL/pretraining/model.py`
- Create: `AllocRL/pretraining/train_encoder.py`
- Create: `AllocRL/configs/candidate_pretrain_seed0.json`
- Create: `AllocRL/test_encoder_pretraining.py`
- Modify: `AllocRL/alloc_env/cnn_extractor.py`

**Interfaces:**
- Produces: `CandidatePretrainingModel`
- Produces: `train_candidate_encoder(config_path: Path, dataset_root: Path, output_dir: Path) -> Path`
- Produces: `candidate_encoder_pretrained.pt`, `pretraining_metrics.json`, and `PRETRAINING_COMPLETE.json`

- [ ] **Step 1: Write failing auxiliary-head and masked-loss tests**

Require all per-action output shapes, finite values, invalid-action masking, replay masking, and gradient flow into both the CNN and structured/fusion MLPs. The auxiliary heads must not be part of `CandidateCnnExtractor.state_dict()`.

```python
predictions = model(observation_batch)
assert predictions.future_fit.shape == (8, 10, 16)
loss = model.loss(predictions, targets)
loss.backward()
assert cnn_gradient_norm(model.extractor) > 0
assert structured_gradient_norm(model.extractor) > 0
```

- [ ] **Step 2: Implement the pretraining model and losses**

Use the existing schema-4 `CandidateCnnExtractor` as the shared encoder and attach temporary linear heads. Use masked binary cross entropy for `current_placeable` and `future_fit`; use masked Smooth L1 for optionality, geometry, replay success, replay dropout, and replay delay. Weight losses as:

```text
current_placeable                 0.25
future_fit                       1.00
future optionality after/delta   1.00
largest rectangle/components     0.50
replay success/dropout/delay     1.00
```

- [ ] **Step 3: Add the exact pretraining config**

```json
{
  "schema_version": 1,
  "seed": 0,
  "train_state_count": 5000,
  "validation_state_count": 1000,
  "train_episode_seeds": [20000, 20039],
  "validation_episode_seeds": [30000, 30009],
  "states_per_shard": 100,
  "replay_every_n_states": 4,
  "replay_resolved_blocks": 8,
  "replay_max_decisions": 32,
  "optimizer": "AdamW",
  "learning_rate": 0.0001,
  "batch_size": 8,
  "max_epochs": 30,
  "early_stopping_patience": 5,
  "minimum_relative_baseline_improvement": 0.05,
  "minimum_shuffled_grid_degradation": 0.05
}
```

- [ ] **Step 4: Write checkpoint and gate tests**

Require the checkpoint to contain only extractor weights plus schema/config/dataset hashes and best epoch metadata. Require failed baseline or shuffled-grid gates to omit `PRETRAINING_COMPLETE.json`. Require successful publication to hash the checkpoint and metrics, then publish the complete marker last.

- [ ] **Step 5: Implement deterministic training and gate evaluation**

Seed Python, NumPy, Torch, CUDA, and DataLoader workers. Select the lowest finite validation total loss with atomic checkpoint replacement. Evaluate the no-grid scalar baseline, validation-mean optionality baseline, shuffled-grid test, and counterfactual geometry pairs after training. Abort before publication if any gate fails.

- [ ] **Step 6: Verify a tiny real pretraining run and commit**

```powershell
python -m pytest AllocRL/test_encoder_pretraining.py AllocRL/test_feature_extractors.py AllocRL/test_cnn_diagnostics.py -q
python -m pretraining.train_encoder --config configs/candidate_pretrain_seed0.json --dataset-root ./tmp/pretrain-smoke-data --output-dir ./tmp/pretrain-smoke --smoke-state-count 32 --max-epochs 2
git add AllocRL/pretraining/model.py AllocRL/pretraining/train_encoder.py AllocRL/configs/candidate_pretrain_seed0.json AllocRL/test_encoder_pretraining.py AllocRL/alloc_env/cnn_extractor.py
git commit -m "feat: pretrain candidate CNN feature extractor"
```

Expected: tests pass, the smoke command publishes a finite checkpoint and metrics, and production gate thresholds remain enabled outside smoke mode.

---

### Task 6: Add a Resume-Safe Absolute-Timestep Learning-Rate Schedule

**Files:**
- Create: `AllocRL/learning_rate_schedule.py`
- Create: `AllocRL/test_learning_rate_schedule.py`
- Modify: `AllocRL/train.py`
- Modify: `AllocRL/test_train_resume_cli.py`
- Modify: `AllocRL/test_parallel_training_config.py`

**Interfaces:**
- New CLI: `--lr-schedule {constant,linear}`; default `constant`.
- New CLI: `--lr-final FLOAT`; required for `linear`.
- New CLI: `--lr-decay-steps INTEGER`; positive for `linear`.
- Existing `--lr` remains the initial learning rate.

- [ ] **Step 1: Write schedule unit tests**

For `initial=1e-4`, `final=1e-5`, `decay_steps=1_000_000`, require:

```python
assert schedule.at_step(0) == pytest.approx(1e-4)
assert schedule.at_step(500_000) == pytest.approx(5.5e-5)
assert schedule.at_step(1_000_000) == pytest.approx(1e-5)
assert schedule.at_step(2_000_000) == pytest.approx(1e-5)
```

Reject non-finite rates, non-positive rates/decay steps, and `final > initial` for the `linear` mode.

- [ ] **Step 2: Write a real save/load/resume test**

Train a tiny MaskablePPO model, save it, load it with `reset_num_timesteps=False`, and verify that logged learning rates continue decreasing from absolute `model.num_timesteps`. They must not restart at `1e-4` after load.

- [ ] **Step 3: Implement an SB3-compatible stateful schedule**

The schedule callable must ignore SB3's per-`learn()` progress reset and calculate from absolute cumulative timesteps. A callback synchronizes the callable from `model.num_timesteps` at training start and rollout end, before PPO calls `_update_learning_rate`. Loading must verify that the saved model contains the same schedule type and parameters.

- [ ] **Step 4: Extend the exact run contract**

Record these fields in `run_config.json` and its structural compatibility set:

```text
learning_rate_schedule
final_learning_rate
learning_rate_decay_steps
```

Any change during `--auto-resume` must fail before learning. Constant mode remains byte-compatible in behavior for new schema-4 runs.

- [ ] **Step 5: Verify and commit**

```powershell
python -m pytest AllocRL/test_learning_rate_schedule.py AllocRL/test_train_resume_cli.py AllocRL/test_parallel_training_config.py -q
git add AllocRL/learning_rate_schedule.py AllocRL/test_learning_rate_schedule.py AllocRL/train.py AllocRL/test_train_resume_cli.py AllocRL/test_parallel_training_config.py
git commit -m "feat: add absolute timestep learning rate decay"
```

---

### Task 7: Transfer, Freeze, and Fine-Tune the Pretrained Extractor in PPO

**Files:**
- Create: `AllocRL/pretraining/transfer.py`
- Create: `AllocRL/pretraining/ppo.py`
- Create: `AllocRL/test_pretrained_ppo.py`
- Modify: `AllocRL/train.py`
- Modify: `AllocRL/test_train_resume_cli.py`
- Modify: `AllocRL/test_cnn_diagnostics.py`
- Modify: `AllocRL/holdout_model_selection.py`
- Modify: `AllocRL/evaluation_runner.py`

**Interfaces:**
- Produces: `load_verified_pretrained_extractor(model, checkpoint_path: Path, complete_path: Path) -> PretrainingReceipt`
- Produces: `ScaleAwareMaskablePPO`
- Produces: `ExtractorFineTuneCallback(freeze_until_timestep: int)`

- [ ] **Step 1: Write failing transfer-integrity tests**

Require a new PPO model to receive byte-equal extractor tensors from the verified pretraining checkpoint while actor/value parameters remain independently initialized. Reject missing markers, failed gates, SHA mismatches, schema mismatches, unknown keys, missing extractor keys, and any auxiliary-head keys in the transfer payload.

- [ ] **Step 2: Implement strict pretraining receipt and weight loading**

Parse `PRETRAINING_COMPLETE.json`, verify its config, dataset manifest, metrics, and checkpoint hashes, then load only `CandidateCnnExtractor.state_dict()` with `strict=True`. Return immutable hashes for inclusion in the PPO run configuration. Do not permit a warning-only or partial load.

- [ ] **Step 3: Write failing freeze/unfreeze and optimizer-group tests**

Run a tiny model across the 50,000 boundary with a reduced test threshold. Require zero extractor gradients and unchanged extractor tensors before the boundary, nonzero gradients and changed tensors after it, and uninterrupted actor/value updates in both phases. Require:

```python
assert optimizer_group("policy")["lr"] == pytest.approx(base_lr)
assert optimizer_group("extractor")["lr"] == pytest.approx(base_lr * 0.1)
```

Save before the boundary and resume after it; the warm-up must not restart.

- [ ] **Step 4: Implement the PPO subclass and absolute freeze callback**

`ScaleAwareMaskablePPO` must create named policy and extractor optimizer groups and override SB3 learning-rate updates so each group retains its fixed `lr_scale`. `ExtractorFineTuneCallback` sets extractor `requires_grad` from absolute `model.num_timesteps` at training start and on boundary crossing. Log `train/policy_learning_rate`, `train/extractor_learning_rate`, and `diagnostics/extractor_frozen`.

- [ ] **Step 5: Extend CLI and the exact run contract**

Add:

```text
--pretrained-extractor PATH
--pretraining-complete PATH
--require-pretrained-extractor
--freeze-extractor-steps 50000
--extractor-lr-scale 0.1
```

Record `pretraining_checkpoint_sha256`, `pretraining_manifest_sha256`, `pretraining_complete_sha256`, `freeze_extractor_steps`, and `extractor_lr_scale` as structural run fields. A new final candidate-CNN run must reject random initialization. On auto-resume, verify the receipt but load extractor weights from the SB3 checkpoint rather than reapplying the Stage 1 checkpoint.

- [ ] **Step 6: Update every schema-4 candidate model loader**

Use `ScaleAwareMaskablePPO.load` for new candidate training, holdout selection, fixed-scenario evaluation, checkpoint evaluation, and ONNX export. Keep the immutable schema-3 comparison artifacts and tag untouched; loader selection is driven by run-config model class and schema.

- [ ] **Step 7: Verify and commit**

```powershell
python -m pytest AllocRL/test_pretrained_ppo.py AllocRL/test_train_resume_cli.py AllocRL/test_cnn_diagnostics.py AllocRL/test_holdout_model_selection.py AllocRL/test_evaluation_scenarios.py AllocRL/test_onnx_export.py -q
git add AllocRL/pretraining/transfer.py AllocRL/pretraining/ppo.py AllocRL/test_pretrained_ppo.py AllocRL/train.py AllocRL/test_train_resume_cli.py AllocRL/test_cnn_diagnostics.py AllocRL/holdout_model_selection.py AllocRL/evaluation_runner.py
git commit -m "feat: warm start PPO from pretrained CNN features"
```

---

### Task 8: Create the Two-Stage Six-Hour Eight-Environment Colab Run

**Files:**
- Create: `AllocRL/configs/improved_cnn_6h_seed0.json`
- Create: `notebooks/improved_cnn_6h.ipynb`
- Create: `AllocRL/test_improved_cnn_notebook.py`
- Modify: `README.md`

**Runtime selection:** In Colab choose `Runtime -> Change runtime type -> Hardware accelerator: GPU -> GPU type: L4`. Enable the high-memory system profile if the account offers it. The final-run notebook must print and verify an NVIDIA L4 before starting the six-hour clock. A100 is an acceptable manual fallback only after changing the notebook's explicit accepted-device list; T4 is not the planned final runtime.

The L4 is the deliberate primary choice: this PPO job still spends substantial time in CPU environment simulation, while the 24 GB L4 has ample GPU memory for the small CNN and minibatches. Paying the higher A100 cost is unlikely to produce a proportional speedup for this workload. Colab itself states that accelerator availability and resource limits vary over time, so the notebook must fail visibly instead of silently using a different accelerator.

Runtime references:

- Colab resource and accelerator availability: `https://research.google.com/colaboratory/faq.html`
- NVIDIA L4 specifications, including 24 GB GPU memory: `https://www.nvidia.com/en-us/data-center/l4/`

**Frozen training configuration:**

```json
{
  "extractor": "candidate-cnn",
  "state_context": "full",
  "seed": 0,
  "timesteps_ceiling": 2000000000,
  "max_training_seconds": 21600,
  "learning_rate": 0.0001,
  "learning_rate_schedule": "linear",
  "final_learning_rate": 0.00001,
  "learning_rate_decay_steps": 1000000,
  "require_pretrained_extractor": true,
  "freeze_extractor_steps": 50000,
  "extractor_learning_rate_scale": 0.1,
  "n_envs": 8,
  "vec_env": "subproc",
  "n_steps": 120,
  "batch_size": 64,
  "n_epochs": 5,
  "gamma": 1.0,
  "gae_lambda": 0.98,
  "checkpoint_freq": 10000,
  "wall_clock_heartbeat_seconds": 300,
  "holdout_eval_freq": 50000,
  "holdout_selection_count": 5,
  "monthly_jitter": 20,
  "empirical_profile_probability": 0.2,
  "device": "cuda",
  "export_onnx": false
}
```

`8 * 120 = 960` transitions per PPO rollout, exactly matching the former one-environment `n_steps=960` update size. `960 / 64 = 15` complete minibatches. Do not set `n_steps=960` with eight environments: that would create a 7,680-transition rollout and roughly eight times the observation-buffer footprint.

The Drive experiment root is fixed and new. Stage 1 and Stage 2 never share mutable state files:

```text
experiment:  /content/drive/MyDrive/CNN-RL-improved/scale-aware-cnn-6h-seed0
pretraining: /content/drive/MyDrive/CNN-RL-improved/scale-aware-cnn-6h-seed0/pretraining
PPO:         /content/drive/MyDrive/CNN-RL-improved/scale-aware-cnn-6h-seed0/ppo
```

- [ ] **Step 1: Add failing notebook/config contract tests**

Require the exact values above, a clean pinned-tag checkout, Drive mounting, CUDA/L4 checks, no ONNX export in the critical path, verified Stage 1 completion before PPO, strict transfer hashes, 32-step pretrained candidate-CNN PPO smoke training in a temporary directory, and `--auto-resume` against the fixed PPO root.

- [ ] **Step 2: Build the notebook in these cells**

1. Mount Drive.
2. Verify L4, CUDA, system RAM, free Drive space, and at least two visible CPU cores.
3. Clone a future immutable tag `scale-aware-cnn-6h-v1` and verify clean HEAD/tag identity.
4. Install the existing hashed comparison dependency lock without replacing the Colab-provided Torch build; reject only newly introduced `pip check` conflicts.
5. Verify scenario, split, lock, and config hashes.
6. Verify existing dataset shards or generate the 5,000/1,000-state dataset on local Colab disk, then copy its verified manifest and shards to the pretraining root.
7. Resume or run `python -m pretraining.train_encoder` until `PRETRAINING_COMPLETE.json` passes every gate. Dataset generation and pretraining time do not count toward the PPO six-hour budget.
8. Run a 32-step PPO smoke with the verified extractor, reduced freeze boundary, and a temporary output root. Assert zero extractor change while frozen and nonzero change after unfreezing.
9. Launch `train.py` against the PPO root with `--require-pretrained-extractor`, exact Stage 1 artifact paths, `--freeze-extractor-steps 50000`, `--extractor-lr-scale 0.1`, `--auto-resume`, `--no-export-onnx`, and unbuffered stdout.
10. Display both stage receipts, pretraining metrics, `run_state.json`, `progress_timing.csv`, final model receipt, evaluation CSVs, and exact Stage 1/Stage 2 resume commands.

- [ ] **Step 3: Protect six-hour resume semantics**

Stage 1 resume verifies and skips complete dataset shards and restores `pretraining_last.pt` only when dataset/config hashes match. Stage 2 wall-clock state begins only after the Stage 1 gate marker is verified and records 21,600 cumulative PPO seconds. Reconnecting with the same immutable tag, configs, pretraining receipt, and Drive roots must continue the active stage. Changing `n_envs`, schedule parameters, freeze boundary, extractor LR scale, pretraining hashes, GPU acceptance, observation schema, or config hash must fail closed.

- [ ] **Step 4: Verify locally and commit**

```powershell
python -m pytest AllocRL/test_improved_cnn_notebook.py AllocRL/test_encoder_pretraining.py AllocRL/test_pretrained_ppo.py AllocRL/test_comparison_notebook.py AllocRL/test_smoke_workflows.py -q
git add AllocRL/configs/improved_cnn_6h_seed0.json notebooks/improved_cnn_6h.ipynb AllocRL/test_improved_cnn_notebook.py README.md
git commit -m "feat: add two-stage scale-aware CNN Colab run"
```

---

### Task 9: Full Verification, Release, and Handoff

**Files:**
- Modify: `WIP_HANDOFF.md`
- Inspect: all files changed since this planning commit

- [ ] **Step 1: Run all verification gates**

```powershell
python -m pytest -q
python -m compileall -q AllocRL
git diff --check
git status --short
```

Expected: all tests pass, compileall and diff checks exit zero, and only intentional files are modified.

- [ ] **Step 2: Run a real short two-stage rehearsal**

Use a new temporary output directory. Generate a tiny deterministic train/validation dataset, run feature pretraining through publication of `PRETRAINING_COMPLETE.json`, then start `n_envs=8`, `vec_env=subproc`, `n_steps=120` PPO with a reduced test-only freeze boundary. Save and resume once before and once after that boundary. Verify zero extractor gradients while frozen, non-zero extractor gradients after unfreezing, the `0.1` extractor-to-policy LR ratio, continuous absolute-timestep LR logs, a valid checkpoint, exact auto-resume, and no schema-3 or random-extractor load path.

- [ ] **Step 3: Review the behavioral diff**

Confirm that changes are limited to observation encoding/schema, extractor dimensions, isolated simulator cloning for diagnostic target generation, pretraining data/heads/trainer, checkpoint transfer, PPO freeze/fine-tune scheduling, LR scheduling/configuration, tests, and the new notebook. Reject any diff in reward constants/calculation, action mapping, rotation behavior, production block generator, episode order, or existing comparison output roots.

- [ ] **Step 4: Commit handoff, push, then create the immutable tag**

```powershell
git add WIP_HANDOFF.md
git commit -m "docs: hand off two-stage CNN Colab run"
git push origin HEAD:main
git tag -a scale-aware-cnn-6h-v1 -m "Scale-aware candidate CNN six-hour Colab release"
git push origin scale-aware-cnn-6h-v1
```

Do not create the tag until the full suite and short rehearsal pass. After the tag exists, the notebook URL is:

```text
https://colab.research.google.com/github/LMS4681/CNN-RL-Raw-Comparison/blob/scale-aware-cnn-6h-v1/notebooks/improved_cnn_6h.ipynb
```

---

## Acceptance Criteria

- A policy can distinguish the same normalized grid in differently sized workspaces through explicit physical-scale features.
- Current and future length/breadth remain orientation-specific under the no-rotation rule.
- Simulator target generation never mutates the source training environment or changes production episode state.
- Stage 1 passes held-out predictive, shuffled-grid dependence, and counterfactual-geometry gates before publishing `PRETRAINING_COMPLETE.json`.
- Stage 2 refuses to start from random extractor weights or from an absent, failed, or hash-mismatched Stage 1 completion marker.
- CNN and structured/fusion MLP gradients are zero during the 50,000-timestep warm-up and non-zero after the deterministic unfreeze boundary.
- The unfrozen extractor LR remains exactly `0.1 * policy_learning_rate` across save/load/resume.
- Schema-3 models cannot be resumed in schema 4.
- Linear LR is monotonic in absolute cumulative timesteps and survives save/load/resume.
- The production Colab uses exactly eight subprocess environments and 960 total transitions per PPO update.
- The six-hour clock applies only to Stage 2 PPO; Stage 1 dataset generation and pretraining finish before it starts.
- The production run records six cumulative PPO training hours, writes verified Drive artifacts for both stages, and can resume after interruption without resetting LR, freeze state, or elapsed time.
- The final notebook is pinned to an immutable release tag and explicitly verifies an L4 GPU before training.

## Expected Effect and Limits

This design separates spatial representation learning from PPO credit assignment. Stage 1 gives the CNN and structured/fusion MLPs direct simulator-derived supervision for scale, free-space geometry, immediate feasibility, future optionality, and bounded replay outcomes; Stage 2 then learns the placement policy from a verified representation instead of moving random extractor weights. Temporary freezing protects that representation while the policy/value heads establish a useful scale, and lower-LR joint fine-tuning still lets reward optimize features that the auxiliary targets do not capture.

The result is not guaranteed to outperform raw-direct or the prior joint-from-scratch candidate in a single six-hour seed. Replay labels are deterministic-teacher-relative approximations rather than globally optimal placement labels, auxiliary success does not prove policy improvement, and the simulator distribution can still differ from holdout scenarios. Final claims therefore require the unchanged holdout evaluation and must report throughput, stability, and placement reward separately.
