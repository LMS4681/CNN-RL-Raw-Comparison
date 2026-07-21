# Scale-Aware CNN and Six-Hour Colab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make workspace scale and non-rotatable block geometry easier for PPO to use, retain trainable CNN/MLP feature learning, and provide a restart-safe Colab notebook that trains one improved candidate-CNN model for six cumulative hours with eight parallel environments and a monotonically decreasing learning rate.

**Architecture:** Keep the existing ten-workspace, 4-channel candidate-grid, action-mask, reward-schema-2, 913-block, no-rotation training flow. Append scale-aware structured features beside the CNN output instead of changing the grid renderer or reward. Bump the observation contract to schema 4, update every extractor width, then add an absolute-timestep linear learning-rate schedule whose state survives SB3 save/load. Run eight `SubprocVecEnv` workers with 120 steps each so every PPO update still contains 960 transitions rather than growing to 7,680 transitions.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, PyTorch, Stable-Baselines3, sb3-contrib MaskablePPO, pytest, Jupyter/Google Colab, Google Drive, Git/GitHub.

## Global Constraints

- Do not modify or resume `/content/drive/MyDrive/CNN-RL-comparison/overnight-20260721`.
- Do not move the immutable `overnight-v1` tag. The improved release uses a new tag and a new Drive output root.
- Do not modify reward schema 2, action masking, placement order, episode length, ten-workspace ordering, block generation, candidate placement, or evaluation scenarios.
- Block rotation remains forbidden. Length and breadth are orientation-specific; never sort or swap them while encoding ratios.
- Keep all four current grid channels in the executable model: occupancy, remaining lifetime, lot usage, and current candidate footprint. The already prepared three-channel explanatory report is a separate reporting artifact and is not rewritten by this work.
- Keep the CNN and structured/fusion MLP trainable end to end through PPO. Do not freeze them and do not replace convolution kernels with all-one filters.
- Do not add attention, pointer networks, coordinate channels, auxiliary losses, or reward changes in this implementation.
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

- [ ] **Step 4: Prove trainability is retained**

Keep the existing candidate-CNN gradient/weight-change diagnostics passing. Add no `requires_grad=False`, fixed kernels, or detached CNN/fusion outputs.

- [ ] **Step 5: Verify and commit**

```powershell
python -m pytest AllocRL/test_feature_extractors.py AllocRL/test_raw_direct_extractor.py AllocRL/test_train_resume_cli.py AllocRL/test_smoke_workflows.py AllocRL/test_training_visualization.py AllocRL/test_cnn_diagnostics.py -q
git add AllocRL/alloc_env/cnn_extractor.py AllocRL/comparison/raw_direct_extractor.py AllocRL/train.py AllocRL/smoke_test.py AllocRL/test_feature_extractors.py AllocRL/test_raw_direct_extractor.py AllocRL/test_train_resume_cli.py AllocRL/test_smoke_workflows.py AllocRL/test_training_visualization.py
git commit -m "feat: align extractors with observation schema 4"
```

---

### Task 4: Add a Resume-Safe Absolute-Timestep Learning-Rate Schedule

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

### Task 5: Create the Six-Hour Eight-Environment Colab Run

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

The Drive output root is fixed and new:

```text
/content/drive/MyDrive/CNN-RL-improved/scale-aware-cnn-6h-seed0
```

- [ ] **Step 1: Add failing notebook/config contract tests**

Require the exact values above, a clean pinned-tag checkout, Drive mounting, CUDA/L4 checks, no ONNX export in the critical training path, 32-step candidate-CNN smoke training in a temporary directory, and `--auto-resume` against the fixed Drive root.

- [ ] **Step 2: Build the notebook in these cells**

1. Mount Drive.
2. Verify L4, CUDA, system RAM, free Drive space, and at least two visible CPU cores.
3. Clone a future immutable tag `scale-aware-cnn-6h-v1` and verify clean HEAD/tag identity.
4. Install the existing hashed comparison dependency lock without replacing the Colab-provided Torch build; reject only newly introduced `pip check` conflicts.
5. Verify scenario, split, lock, and config hashes.
6. Run `python smoke_test.py --extractor candidate-cnn --timesteps 32 --device cuda` outside the production output root.
7. Launch `train.py` with the frozen config, `--auto-resume`, `--no-export-onnx`, and an unbuffered subprocess so logs appear continuously.
8. Display `run_state.json`, `progress_timing.csv`, final model receipt, evaluation CSVs, and an exact resume command.

- [ ] **Step 3: Protect six-hour resume semantics**

The wall-clock callback already records cumulative active training seconds. Reconnecting with the same immutable tag, config, and Drive root must continue only the remaining duration. Changing `n_envs`, schedule parameters, GPU acceptance, observation schema, or config hash must fail closed.

- [ ] **Step 4: Verify locally and commit**

```powershell
python -m pytest AllocRL/test_improved_cnn_notebook.py AllocRL/test_comparison_notebook.py AllocRL/test_smoke_workflows.py -q
git add AllocRL/configs/improved_cnn_6h_seed0.json notebooks/improved_cnn_6h.ipynb AllocRL/test_improved_cnn_notebook.py README.md
git commit -m "feat: add six hour scale-aware CNN Colab run"
```

---

### Task 6: Full Verification, Release, and Handoff

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

- [ ] **Step 2: Run a real short eight-environment rehearsal**

Use a new temporary output directory, `n_envs=8`, `vec_env=subproc`, `n_steps=120`, CUDA if locally available, and a short wall-clock budget. Verify continuous LR logs, a valid checkpoint, exact auto-resume, and no schema-3 load path.

- [ ] **Step 3: Review the behavioral diff**

Confirm that changes are limited to observation encoding/schema, extractor dimensions, LR scheduling/configuration, tests, and the new notebook. Reject any diff in reward constants/calculation, action mapping, rotation behavior, block generator, or existing comparison output roots.

- [ ] **Step 4: Commit handoff, push, then create the immutable tag**

```powershell
git add WIP_HANDOFF.md
git commit -m "docs: hand off scale-aware CNN Colab run"
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
- CNN and structured/fusion MLP gradients remain non-zero in diagnostics.
- Schema-3 models cannot be resumed in schema 4.
- Linear LR is monotonic in absolute cumulative timesteps and survives save/load/resume.
- The production Colab uses exactly eight subprocess environments and 960 total transitions per PPO update.
- The production run records six cumulative training hours, writes verified Drive checkpoints, and can resume after interruption without resetting LR or elapsed time.
- The final notebook is pinned to an immutable release tag and explicitly verifies an L4 GPU before training.

## Expected Effect and Limits

This change should improve learnability more reliably than freezing CNN/MLP weights because it supplies the scale ratios and future geometry that the policy currently has to infer or cannot reconstruct from area summaries. It does not guarantee higher reward in a single six-hour seed. The strongest expected effects are faster separation of physically different workspaces, fewer scale-related representation errors, and more stable CNN/PPO optimization from the lower decaying LR. Final performance still requires holdout evaluation against the existing baselines.
