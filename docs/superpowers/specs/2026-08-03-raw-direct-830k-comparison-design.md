# Raw-Direct 830k Step-Matched Comparison Design

Date: 2026-08-03

## Goal

Produce the comparison arm for the completed scale-aware candidate-CNN run
(`scale-aware-cnn-6h-v7`, final timestep 833,368). The arm trains PPO with no
learned feature extractor for a matched number of PPO timesteps, so the report
can state how much the MLP and CNN structure contributed relative to feeding
observations straight to the policy.

The comparison budget is **PPO timesteps only**. Stage 1 supervised
pretraining is not equalised and is not counted; the report states that the
candidate-CNN arm received supervised compute outside the matched budget.

## Scope

The change adds:

- `notebooks/raw_direct_830k.ipynb`;
- a notebook contract test;
- the README Colab link and an immutable release tag.

The environment, reward, observation schema, action masking, episode order,
block generation, `train.py`, and every existing extractor are unchanged. No
shared configuration file is modified.

## Arm Definition

`--extractor raw-direct` uses the existing `RawDirectExtractor`
(`comparison/raw_direct_extractor.py`): it concatenates `block`, `ws_meta`,
masked `future_blocks`, `future_mask`, `future_demand`, masked
`pending_blocks`, `pending_mask`, and `pending_summary` into a fixed 2,818-wide
vector with no trainable parameters. Grids are excluded. `run_config.json`
records `model_class: MaskablePPO`, so the scale-aware loader is not involved.

The environment still renders grids because the observation space is shared, so
per-step environment cost matches the candidate-CNN arm and only the extractor
forward pass disappears. Throughput is therefore expected to meet or exceed the
38.58 steps per second the v7 run recorded.

## Step Budget and Stopping

`TARGET_TIMESTEPS = 830_000`. The user's tolerance is one checkpoint interval
(10,000 steps), and 830,000 is the largest 10,000 multiple at or below the v7
final timestep of 833,368 — a 0.4 per cent difference.

The run stops on the timestep ceiling, not on a wall clock. `--timesteps` is
therefore the stopping condition and `--max-training-seconds` is absent.

This is a deliberate trade. `train.py` only produces `run_state.json`,
`progress_timing.csv`, `runtime_metrics.json`, and `training_completion.json`
when a wall-clock budget is set, and it explicitly rejects a run that reaches
the timestep ceiling while a wall-clock budget is active. Everything the
comparison needs is still produced: `block_placement_ppo.sb3`, `best_model.sb3`,
`checkpoints/`, `run_config.json`, `training_log.csv`, `loss_log.csv`,
`holdout_selection.csv`, `evaluation_csv.csv`, and `evaluation_scenarios.csv`.

Two numbers that `runtime_metrics.json` would have supplied are recovered
elsewhere: parameter counts come from loading the saved model offline, and
throughput comes from `arm_timing.json`, written by the notebook itself.

## Resume Contract

`--timesteps` is additive on resume, because `model.learn` is called with
`reset_num_timesteps=False` whenever a resumable archive is found. The notebook
therefore computes the remaining budget before every launch:

```python
from train import find_resumable_model, model_num_timesteps
resumable = find_resumable_model(PPO_ROOT)
done = model_num_timesteps(resumable) if resumable is not None else 0
remaining = max(TARGET_TIMESTEPS - done, 0)
```

When `remaining` is zero the notebook skips training and runs only the final
evaluation. Any number of interrupted sessions therefore converge on 830,000
total timesteps rather than adding 830,000 each time.

`--auto-resume` selects the readable archive with the greatest stored timestep,
and `run_config.json` compatibility validation still rejects a resume whose
observation or model configuration changed.

## Training Command

Identical to the v7 Stage 2 command except for the extractor and the budget:

```text
python -u train.py --data-dir ./data --output-dir <PPO_ROOT>
  --extractor raw-direct --state-context full --seed 0
  --timesteps <remaining>
  --lr 0.0001 --lr-schedule linear --lr-final 0.00001
  --lr-decay-steps 1000000
  --n-envs 8 --vec-env subproc --n-steps 120 --batch-size 64 --n-epochs 5
  --gamma 1.0 --gae-lambda 0.98
  --checkpoint-freq 10000
  --holdout-eval-freq 50000 --holdout-selection-count 5
  --monthly-jitter 20 --empirical-profile-probability 0.2
  --device cuda --eval-scenarios ./data/fixed_eval_scenarios.json
  --final-holdout-report --auto-resume --no-export-onnx
```

Removed relative to v7: `--max-training-seconds`,
`--wall-clock-heartbeat-seconds`, `--pretrained-extractor`,
`--pretraining-complete`, `--require-pretrained-extractor`,
`--freeze-extractor-steps`, `--extractor-lr-scale`, and every
`--comparison-*` provenance flag, which only the wall-clock runtime metrics
require.

`--holdout-selection-count` stays at 5 because `FixedHoldoutEvalCallback`
rejects any other value. The arm comparison does not use that selection.

## Notebook Structure

`notebooks/raw_direct_830k.ipynb` mirrors the v7 notebook with the four Stage 1
cells removed:

1. Drive mount, experiment roots under
   `/content/drive/MyDrive/CNN-RL-improved/raw-direct-830k-seed0`, and
   `TARGET_TIMESTEPS`.
2. Runtime guard: CUDA present, L4 GPU, at least two cores, 10 GiB RAM, 10 GiB
   free Drive space.
3. Clean checkout of the pinned tag, verified against `git rev-list` and
   `git status --porcelain`.
4. Hash-locked dependency install that must not change the Colab Torch stack.
5. Immutable input hashes for `data/fixed_eval_scenarios.json`,
   `data/data_split_manifest.json`, and `requirements-comparison.txt`.
6. Curve-log hygiene: `prune_rolled_back_rows` on `training_log.csv` and
   `loss_log.csv`, with a one-time backup, as in v7.
7. Training: remaining-budget computation, child process with streamed stdout
   and a 30-second monitor that reports elapsed seconds and the newest
   checkpoint timestep, then an appended session record in `arm_timing.json`.
8. Artifact listing.

The monitor reads progress from the checkpoint filenames
(`block_placement_ppo_<steps>_steps.sb3`) because this arm has no
`run_state.json`.

`arm_timing.json` accumulates one record per session with the UTC start and
end, elapsed seconds, and the start and end timesteps, plus totals for the arm.
It is written after the child exits successfully.

## Comparison Protocol

Reported separately from this run, using saved checkpoints:

1. Select the largest checkpoint present in both arms within one checkpoint
   interval of 830,000.
2. Evaluate both arms on all twenty scenarios in
   `data/fixed_eval_scenarios.json` and compare paired per-seed differences,
   which cancels scenario difficulty.
3. Overlay the two `training_log.csv` curves. Episodes are paired by
   construction: both arms use seed 0 with eight environments and a fixed
   913-decision episode, so the same block sequence appears at the same step
   position in both arms.
4. Never rank the arms by `best_model.sb3`, whose selection averages five
   scenarios and whose evaluation-to-evaluation standard deviation is 0.16.

## Testing

`AllocRL/test_raw_direct_notebook.py` asserts the notebook contract: cleanliness
and cell order, the pinned tag, the exact training command tokens, the absence
of every pretraining and wall-clock flag, the presence of the remaining-budget
computation and `arm_timing.json`, the separate Drive root, and the curve-log
prune step. The README must link the pinned notebook.

## Release

Tag `raw-direct-830k-v1` on `main` after the tests pass. The v1 through v7
`scale-aware-cnn-6h` tags remain immutable.
