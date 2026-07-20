# AllocRL A-E Ablation

## Stage B Technical Contract

Stage B uses observation schema 3 for every extractor. The observation is a
`gym.spaces.Dict` containing exactly nine `np.float32` arrays in `[0, 1]`:

| Key | Fixed dimensions | Meaning |
| --- | --- | --- |
| `block` | `(8,)` | Current block, timing, and decision context |
| `grids` | `(10, 4, 64, 64)` | Four candidate-conditioned channels for each workspace |
| `ws_meta` | `(10, 4)` | Workspace dimensions, utilization, and candidate placeability |
| `future_blocks` | `(16, 6)` | Next 16 unassigned blocks in simulator decision order |
| `future_mask` | `(16,)` | Valid-row mask for `future_blocks` |
| `future_demand` | `(3, 4)` | Demand summaries for working-day windows 0-5, 6-20, and 21-60 |
| `pending_blocks` | `(10, 32, 7)` | First 32 retry-ordered pending blocks per workspace |
| `pending_mask` | `(10, 32)` | Valid-row mask for `pending_blocks` |
| `pending_summary` | `(10, 4)` | Queue size, area, delay, and overflow per workspace |

The ordered-future capacity is always 16. The pending queue capacity is always
32 independently for each of the ten workspaces. These dimensions are fixed;
they are not ablation parameters.

The four `grids` channels are, in order:

1. collision exclusion
2. remaining working days
3. post-candidate lot state
4. candidate exclusion

Blocks retain their original `length x breadth` orientation through dimension
checks, candidate generation, preview, incremental placement, and replay.
Rotation is prohibited; there is no rotated fallback or rotation-derived
candidate.

`CandidateCnnExtractor` is part of the shared `MaskablePPO` policy feature
path. Its weights are trained end to end by the actor and critic losses. It is
not a feasibility classifier, receives no feasibility labels, and is not
pretrained as one. The action mask remains a constraint mechanism outside the
CNN. The all-extractor smoke check proves that PPO gives the CNN a nonzero
gradient norm and changes its weights.

Reward shaping remains absent. Training uses only the existing exactly-once
resolved delay/dropout reward and terminal residual needed to conserve the
episode score; no occupancy, feasibility, future-choice, pending-queue, or CNN
auxiliary reward is added.

## A-E Matrix

All variants use the same schema-3 environment, normalization record, action
order, fixed 64 grid size, and reward contract. `current` state context keeps
the current block, workspace metadata, and grids real while zeroing future,
demand, and pending context.

| ID | Extractor | State context | Comparison role |
| --- | --- | --- | --- |
| A | `structured` | `current` | Current-state structured baseline |
| B | `structured` | `full` | Effect of future and pending structured state |
| C | `fixed-grid` | `full` | Effect of deterministic pooled grid pixels |
| D | `candidate-cnn` | `current` | Learned candidate-grid path without future/pending state |
| E | `candidate-cnn` | `full` | Complete recommended Stage B model |

## Stage A Fixed-Holdout Protocol

The fixed holdout bundle is `data/fixed_eval_scenarios.json`. Its source has
913 rows from 40 ships. The ship-level split has 673 training rows from 29
ships and 240 holdout rows from 11 ships. The split seed is `20260716`: a ship
is held out when the first eight bytes of
`SHA-256("20260716:<ship_no>")`, interpreted as an unsigned big-endian integer
and divided by `2**64`, are below `0.20`.

The original allocation CSV has a one-shot business-reference evaluation
role. Stage A reporting uses the 20 fixed holdout scenarios with seeds
`1000..1019`; those scenarios do not select checkpoints.

Prepare the fixed scenario file once:

```powershell
python -B run_ablation.py --prepare-eval-scenarios
```

Evaluate the required `RandomValidPolicy` and `GreedyImmediateAreaPolicy`
baselines through the shared evaluation runner:

```powershell
python -B run_ablation.py --evaluate-baselines
```

The result is `output_ablation/baselines/evaluation_scenarios.csv`, with one
detail row per policy and scenario.

## Training And Verification

Run the five 20,000-timestep smoke jobs first:

```powershell
py -B run_ablation.py --stage smoke
```

Run the complete 300,000-timestep screening matrix. This creates 60 jobs: five
A-E rows, seeds `0..2`, and all four `(gae_lambda, n_steps)` pairs from
`{0.98, 0.995} x {512, 960}`.

```powershell
py -B run_ablation.py --stage screening
```

After screening selection, run the 25 one-million-timestep final jobs with
exactly one selected pair:

```powershell
py -B run_ablation.py --stage final --selected-gae-lambda 0.995 --selected-n-steps 960
```

The final values `0.995` and `960` are illustrative. Replace both with the
winning screening pair recorded in `screening_selection.json`.

Use `--dry-run` on any invocation to print commands without executing them.
Every run uses the fixed holdout bundle, automatic resume, 50,000-timestep
checkpoints and holdout selection, and no ONNX export. Final runs additionally
produce the final holdout report. Outputs are isolated at
`output_ablation/<stage>/<ID>/lambda_<value>/nsteps_<value>/seed_<seed>`; SB3
models and checkpoints use the `.sb3` extension.

Verify all three schema-3 extractor workflows with temporary output:

```powershell
python smoke_test.py --all-extractors --timesteps 1024
```

Pass `--output-dir <path>` only when the smoke artifacts need to be retained
for inspection.

## Evaluation And Promotion

Rank runs by `mean_terminal_score`, then `mean_dropout_rate`, delay metrics,
seed stability, and `mean_retained_choice_ratio`. Retained-choice ratio is an
evaluation diagnostic only; it is not part of the reward or action mask.

Variant E must improve over B by at least one of these criteria:

- absolute mean terminal score improvement of at least `0.05`
- relative dropout-rate reduction of at least `10%`

The improvement direction must agree in at least four of the five final seeds.
Compare E with C to isolate learned convolution and E with D to isolate the
full future/pending state.
