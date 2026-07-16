# Ten-Workspace Training Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the old 277-target, five-workspace synthetic setup with fixed-total 913-block episodes over the specified ten empty real workspaces without changing the CNN/MaskablePPO learning flow.

**Architecture:** A dedicated target loader converts placed and scheduled CSV rows into one target stream, while explicit workspace supplements provide the missing PE054 geometry. `SyntheticBlockGenerator` bootstraps complete target rows and chooses either a constrained balanced monthly profile or the empirical profile; the environment keeps the real workspace layout fixed and empty.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, Stable-Baselines3 2.8, sb3-contrib MaskablePPO, unittest/pytest, Jupyter notebook JSON.

## Global Constraints

- Keep the candidate-CNN observation, future-block context, action masking, simulator, and reward logic unchanged.
- Use exactly 913 targets after excluding July and November by start month.
- Keep total episode size fixed at 913.
- Use balanced/empirical monthly profiles with an 80:20 probability and balanced jitter of plus or minus 20.
- Use the ten workspaces in the approved action order and fixed physical geometry.
- Start training and CSV evaluation with no placed blocks or pre-placed obstacles.
- Preserve complete-row physical-property correlations during bootstrap sampling.
- Reject old checkpoints through the changed workspace configuration and use a new Colab output path.
- Follow TDD and observe each focused test fail before implementation.

---

### Task 1: All-Target CSV Loading and Workspace Supplement

**Files:**
- Create: `AllocRL/test_training_data_profile.py`
- Modify: `AllocRL/alloc_env/data_loader.py`

**Interfaces:**
- Produces: `load_target_blocks(block_csv, excluded_start_months=(7, 11)) -> list[Block]`.
- Produces: `load_workspaces(..., supplemental_workspaces=...) -> list[Workspace]`.
- Produces: `clone_empty_workspaces(workspaces) -> list[Workspace]`.

- [ ] **Step 1: Write failing loader tests**

  Assert that the repository CSV yields 913 targets, no July/November starts,
  no workspace mutation, and both placed and scheduled records.

- [ ] **Step 2: Run the focused tests and confirm the missing APIs fail**

  Run: `py -3 -m pytest test_training_data_profile.py -q`

- [ ] **Step 3: Implement target loading, PE054 supplement support, ordered selection, and empty cloning**

  Use construction start/end dates for every row, validate positive dimensions
  and a complete date pair, and preserve the existing `load_blocks`
  compatibility behavior.

- [ ] **Step 4: Run the focused tests until green**

  Run: `py -3 -m pytest test_training_data_profile.py -q`

### Task 2: Fixed-Total Monthly Bootstrap Generator

**Files:**
- Modify: `AllocRL/test_training_data_profile.py`
- Modify: `AllocRL/alloc_env/block_generator.py`

**Interfaces:**
- Produces: `SyntheticBlockGenerator.from_blocks(...)`.
- Produces deterministic 913-block balanced and empirical episodes from a seed.

- [ ] **Step 1: Add failing tests for total, month bounds, empirical counts, determinism, and row correlation**

- [ ] **Step 2: Run the generator tests and verify expected failures**

  Run: `py -3 -m pytest test_training_data_profile.py -q`

- [ ] **Step 3: Implement bounded zero-sum monthly allocation and row bootstrap**

  Balanced counts must remain within base quota plus or minus 20 and sum to
  913. Empirical counts are scaled only when requested total differs from the
  source total. Exact dates are sampled from working days in each month.

- [ ] **Step 4: Run the focused tests until green**

  Run: `py -3 -m pytest test_training_data_profile.py -q`

### Task 3: Wire Training and Evaluation to the New Scenario

**Files:**
- Modify: `AllocRL/test_training_data_profile.py`
- Modify: `AllocRL/test_parallel_training_config.py`
- Modify: `AllocRL/test_training_visualization.py`
- Modify: `AllocRL/alloc_env/alloc_env.py`
- Modify: `AllocRL/train.py`
- Modify: `AllocRL/export_onnx.py`
- Modify: `AllocRL/visualize_eval_placement.py`
- Modify: `AllocRL/run_ablation.py`

**Interfaces:**
- Default workspace order is the approved ten-code list.
- Training environment receives independent seeded source-bootstrap generators,
  913 blocks, fixed layouts, and no preplacements.
- CSV evaluation uses the original 913 targets against the same empty yards.

- [ ] **Step 1: Add failing wiring and CLI default tests**

- [ ] **Step 2: Run focused tests and verify old five-workspace/vary-layout defaults fail**

  Run: `py -3 -m pytest test_training_data_profile.py test_parallel_training_config.py test_training_visualization.py -q`

- [ ] **Step 3: Update training, evaluation, export, visualization, and ablation loaders**

- [ ] **Step 4: Run focused tests until green**

  Run: `py -3 -m pytest test_training_data_profile.py test_parallel_training_config.py test_training_visualization.py -q`

### Task 4: Colab Configuration and Verification

**Files:**
- Modify: `AllocRL/Colab_train.ipynb`
- Modify: `AllocRL/ABLATION.md`
- Regenerate: `AllocRL/data/fixed_eval_scenarios.json` if its workspace schema is retained by the ablation flow.

- [ ] **Step 1: Add notebook regression assertions before editing notebook JSON**

  Assert ten workspace codes, `N_STEPS = 960`, profile parameters, and the new
  `/content/drive/MyDrive/CNN-RL-outputs/candidate_cnn_10ws_empty_v1` path.

- [ ] **Step 2: Run the notebook test and observe failure**

  Run: `py -3 -m pytest test_training_visualization.py -q`

- [ ] **Step 3: Update notebook JSON through Python's JSON parser and document commands**

- [ ] **Step 4: Run all tests, data assertions, and a short environment/training smoke test**

  Run: `py -3 -m pytest -q`

- [ ] **Step 5: Review diff, commit, push `main`, and verify the remote commit**

  Commit message: `feat: train on 913 blocks across ten empty workspaces`
