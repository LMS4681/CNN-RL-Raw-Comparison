# Raw-Direct Comparison Arm Design

Date: 2026-08-03

## Goal

Produce the comparison arm for the completed scale-aware candidate-CNN run
(`scale-aware-cnn-6h-v7`, final timestep 833,368, 21,603.5 recorded seconds).
The arm trains PPO with no learned feature extractor so the report can state
how much the MLP and CNN structure contributed relative to feeding observations
straight to the policy.

Stage 1 supervised pretraining is not equalised and is not counted. The matched
budget is PPO only, and the report states that the candidate-CNN arm received
supervised compute outside it.

## Scope

The change adds `configs/raw_direct_6h_seed0.json`,
`notebooks/raw_direct_6h.ipynb`, a notebook contract test, the README link, and
an immutable release tag. The environment, reward, observation schema, action
masking, episode order, block generation, `train.py`, and every existing
extractor are unchanged.

## Arm Definition

`--extractor raw-direct` uses the existing `RawDirectExtractor`
(`comparison/raw_direct_extractor.py`): it concatenates `block`, `ws_meta`,
masked `future_blocks`, `future_mask`, `future_demand`, masked
`pending_blocks`, `pending_mask`, and `pending_summary` into a fixed 2,818-wide
vector with no trainable parameters. Grids are excluded. `run_config.json`
records `model_class: MaskablePPO`, so the scale-aware loader is not involved.

The environment still renders grids because the observation space is shared, so
per-step environment cost matches the candidate-CNN arm and only the extractor
forward pass disappears. A 15-minute pilot measured 44.4 steps per second
against the 38.58 the v7 run recorded, as expected.

## Budget: Wall Clock, Compared at a Common Checkpoint

The arm runs the same `--max-training-seconds 21600` budget as v7, with the
timestep ceiling left at 2,000,000,000 so the wall clock is the only stopping
condition.

An earlier revision of this design stopped on a timestep ceiling of 830,000
instead. That was abandoned because `train.py` writes `run_state.json`,
`progress_timing.csv`, `runtime_metrics.json`, and `training_completion.json`
only under a wall-clock budget, and the staged comparison pipeline in
`comparison/checkpoint_evaluator.py` requires those receipts
(`resolve_final_checkpoint` demands a complete `run_state.json`). The
`raw-direct-830k-v1` tag remains immutable but is superseded.

Step matching is preserved by comparing at a common checkpoint rather than by
stopping at one. `WallClockBudgetCallback._on_step` checkpoints when
`num_timesteps` crosses a `--checkpoint-freq` boundary, and with eight
environments `num_timesteps` advances in multiples of eight, so regular
checkpoints land on exact 10,000-step boundaries in both arms.
`readable_checkpoint_inventory` keeps only those exact multiples, and
`select_common_timestep` returns the largest boundary both arms reached. With
v7 finishing at 833,368 the common point is expected to be 830,000; if the raw
arm falls short, the largest shared boundary is used and reported.

## Training Command

Identical to the v7 Stage 2 command except for the extractor and the removed
transfer flags:

```text
python -u train.py --data-dir ./data --output-dir <PPO_ROOT>
  --extractor raw-direct --state-context full --seed 0
  --timesteps 2000000000 --max-training-seconds 21600
  --lr 0.0001 --lr-schedule linear --lr-final 0.00001
  --lr-decay-steps 1000000
  --n-envs 8 --vec-env subproc --n-steps 120 --batch-size 64 --n-epochs 5
  --gamma 1.0 --gae-lambda 0.98
  --checkpoint-freq 10000 --wall-clock-heartbeat-seconds 300
  --holdout-eval-freq 50000 --holdout-selection-count 5
  --monthly-jitter 20 --empirical-profile-probability 0.2
  --device cuda --eval-scenarios ./data/fixed_eval_scenarios.json
  --final-holdout-report
  --comparison-config-sha256 <raw_direct_6h_seed0.json>
  --comparison-baseline-sha256 cd4e14fc1725a4ff159e59d6874d3602f3b65a06
  --comparison-scenario-sha256 <fixed_eval_scenarios.json>
  --comparison-split-sha256 <data_split_manifest.json>
  --comparison-lock-sha256 <requirements-comparison.txt>
  --auto-resume --no-export-onnx
```

Removed relative to v7: `--pretrained-extractor`, `--pretraining-complete`,
`--require-pretrained-extractor`, `--freeze-extractor-steps`, and
`--extractor-lr-scale`. The wall-clock path requires every `--comparison-*`
hash, so this arm needs its own configuration file; the file differs from
`configs/improved_cnn_6h_seed0.json` only in the extractor and the pretraining
fields, which the contract test asserts key by key.

`--holdout-selection-count` stays at 5 because `FixedHoldoutEvalCallback`
rejects any other value. The arm comparison does not use that selection.

## Notebook

`notebooks/raw_direct_6h.ipynb` is derived from the v7 notebook so that the
resume selection, durable monitor, and curve-log hygiene stay byte-identical to
the verified implementation. The four Stage 1 cells and the resume diagnostic
are dropped, leaving nine cells: title, Drive roots under
`/content/drive/MyDrive/CNN-RL-improved/raw-direct-6h-seed0`, runtime guard,
pinned checkout, hash-locked dependency install, immutable input hashes,
curve-log prune, training, and artifacts.

The Drive root is new. The abandoned 830,000-step pilot under
`raw-direct-830k-seed0` is not resumed: it has no `run_state.json`, so a
wall-clock run there would start its budget at zero on top of 40,000 existing
timesteps and break the comparison with v7, which started from zero.

## Comparison Protocol

Both arms now produce the receipts the staged pipeline expects, so the
comparison reuses the existing implementation rather than a parallel one:

1. `select_common_timestep(raw_dir, cnn_dir)` fixes the shared checkpoint.
2. `evaluate_checkpoint` evaluates each arm on all twenty fixed scenarios,
   reading observation scales and workspace order from the `run_config.json`
   beside each archive.
3. Headline numbers are paired per-seed differences over `PRIMARY_TEST_SEEDS`
   (1005 through 1019), the fifteen seeds model selection never saw;
   `SELECTION_SEEDS` (1000 through 1004) are reported only as a secondary view.
4. Arms are never ranked by `best_model.sb3`, whose selection averages five
   scenarios and whose evaluation-to-evaluation standard deviation was 0.16 in
   the v7 run.
5. Training curves are overlaid. Episodes are paired by construction: both arms
   use seed 0 with eight environments and a fixed 913-decision episode, so the
   same block sequence appears at the same step position.

The staged wrappers additionally expect a comparison root holding `raw_direct/`
and `candidate_cnn/` beside a `manifest.json`. Assembling that root from the two
Drive experiment folders is a separate, later step and is not part of this
change.

## Testing

`AllocRL/test_raw_direct_notebook.py` asserts the arm configuration differs from
the candidate-CNN configuration only by the extractor and pretraining fields,
that the notebook pins the hash of its own configuration, the pinned tag, the
exact training command tokens, every provenance hash, the absence of all
transfer flags, the retained resume and monitor contract, the separate Drive
root, and the curve-log prune step. The README must link the pinned notebook.

## Release

Tag `raw-direct-6h-v1` on `main` after the tests pass. The `scale-aware-cnn-6h`
tags and `raw-direct-830k-v1` remain immutable.
