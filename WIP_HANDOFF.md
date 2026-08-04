# Raw observation vs CNN comparison: code and experiment handoff

Updated: 2026-08-03 17:20 (Asia/Seoul)

## 2026-07-22 scale-aware two-stage CNN release

The focused implementation plan is complete in `feature/common-step-finish`
through implementation commit `7087558`. The working tree contains two
untracked report drafts under `reports/`; they are user artifacts and are not
part of this release. The code release adds:

- observation schema 4 with physical workspace scale and orientation-specific
  current/future block geometry;
- simulator-cloned auxiliary targets and sharded, resumable Stage 1 data;
- strict simulator-supervised CNN/structured-MLP pretraining publication;
- strict Stage 1 hash transfer into `ScaleAwareMaskablePPO`;
- a 50,000-absolute-timestep extractor freeze, then `0.1` LR fine-tuning;
- absolute cumulative linear LR decay that survives save/load/resume;
- a resumable L4 Colab workflow with eight subprocess environments and
  `n_steps=120`, preserving 960 transitions per PPO update.

No reward calculation or constant, action mapping, rotation behavior,
production block generator, episode ordering, existing comparison runner, or
existing comparison output root changed. Rotation remains forbidden.

Fresh local verification after the final implementation and release rehearsal:

```text
py -3.12 -m pytest -q
1097 passed, 87 warnings, 55 subtests passed in 606.44s

py -3.12 -m compileall -q AllocRL
exit 0

notebooks/improved_cnn_6h.ipynb code-cell compile
exit 0

git diff --check
exit 0
```

The warnings are existing dependency deprecations and missing local Hangul
plot glyphs. There were no test failures. The Windows CRLF-sensitive immutable
artifact test now compares the committed Git blob, matching its stated
contract and the existing canonical-LF test.

A new deterministic simulator rehearsal generated 48 training and 24
validation states with varied workspace and block geometry, trained Stage 1,
and published a production-eligible `PRETRAINING_COMPLETE.json`. All five
gates passed:

```text
validation_total_loss:          0.045702747690180935
future_fit_bce:                 0.04171265562375386
no_grid_future_fit_bce:         0.12697900484005611
optionality_mae:                0.04313018023967743
mean_baseline_optionality_mae:  0.06944443583488465
shuffled_grid_total_loss:       0.12836183793842793
counterfactual_geometry_delta:  0.004932411868746082
```

That real Stage 1 checkpoint then passed the eight-`SubprocVecEnv` release
rehearsal in `pretraining.release_rehearsal`: strict transfer, one frozen
rollout, resume before the boundary, unfreeze and update, resume after the
boundary, and a third update. The durable receipt reported:

```text
n_envs=8, n_steps=120, rollout_transitions=960
timesteps=[960, 1920, 2880]
policy_lr=[0.0007, 0.0004, 0.0001]
extractor_lr_ratio=0.1
extractor unchanged/gradients clear while frozen=true
extractor changed/nonzero gradient after unfreeze=true
exact before-boundary and after-boundary resume=true
```

The production notebook is `notebooks/improved_cnn_6h.ipynb`. Its immutable
release tag must be created only after this handoff commit is on `main`:

```text
scale-aware-cnn-6h-v1
```

Select an NVIDIA L4 runtime. The fixed Drive roots are:

```text
experiment:  /content/drive/MyDrive/CNN-RL-improved/scale-aware-cnn-6h-seed0
pretraining: /content/drive/MyDrive/CNN-RL-improved/scale-aware-cnn-6h-seed0/pretraining
PPO:         /content/drive/MyDrive/CNN-RL-improved/scale-aware-cnn-6h-seed0/ppo
```

After the tag is pushed, use:

```text
https://colab.research.google.com/github/LMS4681/CNN-RL-Raw-Comparison/blob/scale-aware-cnn-6h-v1/notebooks/improved_cnn_6h.ipynb
```

The notebook verifies L4, immutable tag identity, dependency and input hashes,
resumable Stage 1 shards/checkpoints, the 32-step CUDA transfer smoke, and the
exact state-named Stage 2 auto-resume before starting the six cumulative PPO
hours. Do not reuse the old overnight Drive root for this run.

The sections below preserve the earlier raw-direct versus joint-from-scratch
candidate-CNN experiment handoff and reporting constraints.

## Release state

- Final implementation branch: `feature/common-step-finish`.
- Reviewed implementation commits run through `17dea28`.
- The immutable execution release tag is `overnight-v1` at commit
  `4867f4e6e9f7425a8c438b6403493a097e2e58cb`.
- `main` may advance beyond that tag with handoff, plan, report, and future
  implementation commits. Never move the release tag to follow `main`.
- Repository: `https://github.com/LMS4681/CNN-RL-Raw-Comparison`.

The final comparison keeps the original sequential study design:

1. smoke-test raw-direct and candidate-CNN;
2. train and evaluate raw-direct;
3. publish an honest raw-only partial report;
4. train and evaluate candidate-CNN;
5. evaluate both arms at the greatest shared regular checkpoint timestep;
6. build the Korean report, verify every artifact, then publish
   `COMPLETE.json` last.

## Final verification

Fresh-checkout verification at `17dea28`:

```text
pytest -q
1020 passed, 87 warnings, 55 subtests passed
```

The warnings are existing dependency deprecations and missing local Hangul
plot glyphs. There were no test failures. `compileall`, `git diff --check`,
canonical LF checkout checks, and the tracked-content secret gate also passed.

Independent reviews approved:

- evaluator publication and retry integrity;
- report, runner, and `COMPLETE.json` integrity;
- the exact dependency-lock runtime pin;
- the temporary test-fixture update required by that pin.

The final immutable input hashes are:

```text
scenario  913cac9046dec8164ef65da60275522f7127de5ea775b1c5a6b6aac255716271
split     601bd6143ed8890577e5ff34921241d36fd6a0e99c4bdab4e26152ab168178f8
lock      37634576e34043d169cf24bfc0cc2261818dc65b9358d4b9b2e46ab614d0bdda
```

## Real local E2E

A real CPU E2E ran at `a053e41`, immediately before the lock-only runtime gate
and its test-fixture follow-up. Those two later commits do not alter model
training, evaluation, or report calculations; their behavior is covered by the
fresh full suite and focused negative lock tests at `17dea28`.

The E2E used 30 seconds of active training per arm, 16-step smokes, and an
8-step checkpoint interval. It exercised the real subprocess training path,
20 all-holdout rows and 15 primary-test rows per arm, a 40-row common-step CSV,
15 paired-difference rows, report generation, integrity verification, and final
publication. The greatest shared checkpoint timestep was 152.

The completed output root was rerun without deleting artifacts. Every verified
stage was skipped, output hashes remained valid, and a new trusted
`COMPLETE.json` was published. The complete marker records the expected
baseline/scenario/split/lock hashes and all ten stage output hashes.

## Running Colab r4

The currently running deadline notebook is intentionally pinned to preview tag
`deadline-preview-20260721-r4` at commit
`29528530f7aa26b6ae459fc1187349e8d0c53bec`. Do not interrupt it and do not run
another notebook against the same Drive root:

```text
/content/drive/MyDrive/CNN-RL-comparison/overnight-20260721
```

That preview predates the final dependency-lock config field, so its config SHA
is intentionally different from `overnight-v1`; the running log recorded
`ca0a8e4b0411551dd3330dd80fce34155fc680cabfd1cc0a208a4a2fa91b40dc`.
Let it finish under r4. Resume
that same Drive root only with the same r4 notebook/code. Do not attempt to
resume it with `overnight-v1`; use a new output root for any fresh final-release
run.

After r4 finishes, preserve `manifest.json`, `environment.json`,
`stage_journal.json`, both arm directories, `comparison/`,
`integrity_verification.json`, and `COMPLETE.json`. Validate/report those
artifacts against the recorded r4 commit and config rather than silently
rewriting them to final-release provenance.

## Resume on another PC

For new work from the final release:

```powershell
git clone https://github.com/LMS4681/CNN-RL-Raw-Comparison.git
cd CNN-RL-Raw-Comparison
git switch main
git rev-parse HEAD
git rev-parse "overnight-v1^{}"
git merge-base --is-ancestor overnight-v1 main
```

The tag command must print
`4867f4e6e9f7425a8c438b6403493a097e2e58cb`, and the ancestor check must exit
zero. `HEAD` is expected to be newer after this handoff commit.

Final Colab URL for a fresh output root:

```text
https://colab.research.google.com/github/LMS4681/CNN-RL-Raw-Comparison/blob/overnight-v1/notebooks/overnight_compare.ipynb
```

## Current live experiment snapshot

The r4 Colab job was still running when this handoff was written. The values
below are provisional stdout observations, not the final holdout result and not
an integrity-verified comparison:

```text
raw-direct:   199,680 timesteps, ep_rew_mean=0.358, about 18.99 steps/s
candidate-CNN  74,880 timesteps, ep_rew_mean=-0.166, about 12.50 steps/s
```

At the shared observed point near 49,920 timesteps, raw-direct had
`ep_rew_mean=0.175` and candidate-CNN had `ep_rew_mean=-0.176`. The candidate
diagnostics at 74,880 were `approx_kl=0.47678187`, `clip_fraction=0.671`, and
`explained_variance=-0.184`. These values indicate active but unstable joint
CNN/PPO learning; they do not support a current empirical CNN-superiority
claim. The candidate path is active: CNN gradient norm, weight change,
workspace-feature variance, and candidate-channel sensitivity are all
non-zero.

The downloaded candidate stdout snapshot was stored outside this repository at
`D:\Sub\Allocation\CNN-RL\docs\train_candidate_cnn.log`. The downloaded raw
stdout file was later removed; only sampled values and a local plot were
retained. Neither local log is authoritative or committed. The complete Drive
root, CSVs, model receipts, stage markers, and `COMPLETE.json` are the evidence
source for the report.

## Required report question and exact terminology

The report must address this request:

> CNN과 MLP를 거쳐서 강화학습을 하는 것이, CNN과 MLP를 거치지 않고
> 강화학습을 하는 것과 효과상으로 어떤 차이가 있을지에 대한 설명 또는
> 비교 데이터

The literal phrase "without MLP" is not the current implementation. Both arms
use the same trainable PPO actor/critic MLP:

```text
pi=[64,64], vf=[64,64], ReLU
```

The controlled comparison is therefore:

```text
raw-direct
  normalized non-grid observation
  -> mask + deterministic concatenation (no learned feature extractor)
  -> shared PPO actor/critic MLP
  -> workspace action

candidate-CNN
  10 x 4 x 64 x 64 workspace grids + structured/future observations
  -> shared CNN + structured/fusion MLPs -> 256 learned features
  -> the same PPO actor/critic MLP
  -> workspace action
```

Use the phrase **"학습형 CNN/MLP 특징추출 전처리 없음"** for raw-direct,
not **"신경망/MLP가 전혀 없음"**. A PPO neural policy cannot produce learned
actions in this project without a trainable policy/value mapping.

The expected effect must be explained separately from measured performance:

- raw-direct is faster and more sample-efficient early because it does not
  learn a spatial representation and ignores `grids`;
- candidate-CNN can distinguish layouts with identical scalar occupancy but
  different contiguous free space, lifetime geometry, lot use, and candidate
  footprint;
- the CNN advantage is indirect because reward schema 2 scores compliance,
  delay, and dropout, not fragmentation itself, so spatial credit may arrive
  many decisions later in a 913-step episode;
- more information and parameters do not guarantee better finite-budget PPO
  performance; throughput, optimization stability, and generalization must be
  reported independently.

## Reporting decision after Colab completion

Do not edit or restart the running r4 job. After it finishes:

1. Preserve the entire Drive output root before generating any new artifacts.
2. Require `COMPLETE.json` and validate all stage and model receipts. If it is
   partial, report only the verified subset and the exact missing stages.
3. If candidate-CNN is favorable on the unselected primary holdout and at the
   common timestep, publish the complete comparison with single-seed limits.
4. If candidate-CNN is unfavorable, an explanation-focused report may center
   on the information difference, active CNN diagnostics, computational cost,
   and identified optimization limits. It must say that performance superiority
   was not demonstrated. If the deliverable is explicitly a comparison report,
   adverse comparison data cannot be silently omitted; place it in the results
   or limitations/appendix rather than claiming CNN superiority.
5. CNN-only diagnostics may support the claim that the CNN path is active and
   responsive. They cannot by themselves support the claim that CNN placement
   performance is better than raw-direct.

## Next implementation plan

No source code was changed for the diagnostic conclusions above. The next-PC
work is defined in:

```text
docs/superpowers/plans/2026-07-21-cnn-feature-effect-and-stabilization-plan.md
```

The plan first finalizes the current report, then decomposes the professor's
question with four arms: `raw-direct`, `structured`, `fixed-grid`, and
`candidate-cnn`. This separates deterministic direct observation, learned
structured compression, grid-information value, and learned convolution. PPO
stabilization is attempted one variable at a time only after that diagnostic.

For another PC, clone `main`, read this handoff and the plan, and do not move the
immutable `overnight-v1` tag. The tag remains the released execution baseline;
new documentation and future implementation commits advance `main` only.

## Immediate next-PC implementation: scale-aware CNN and six-hour Colab

The user approved a focused improvement before the next candidate-CNN run.
This immediate implementation is specified in:

```text
docs/superpowers/plans/2026-07-21-scale-aware-cnn-six-hour-colab-implementation.md
```

Read that focused plan before the broader four-arm stabilization plan above.
It does not change reward, actions, episode order, production block generation,
workspace order, or the no-rotation rule. It appends four workspace-relative current
block fields to `ws_meta`, appends maximum future length and breadth to each
future-demand window, and bumps observation schema 3 to schema 4. Existing
schema-3 checkpoints must not be resumed.

The latest provisional candidate stdout extended to about 133,440 timesteps
with `ep_rew_mean=-0.123`, `approx_kl` near `0.955`, and
`explained_variance=-0.345`. CNN gradient, weight-change, feature-variance,
and candidate-channel sensitivity diagnostics remained non-zero. This shows
that the feature path is active, but joint PPO learning from random extractor
weights is unstable and does not establish candidate-CNN superiority.

The approved replacement is a strict two-stage learning flow. There is no
unique correct CNN feature vector, so Stage 1 predicts simulator-derived
future-quality targets instead of assigning labels to hidden features. Stage 2
then transfers the verified extractor into PPO:

```text
Stage 1: simulator-supervised feature pretraining
  schema-4 grids + structured/future observations
  -> candidate CNN + structured/fusion MLP
  -> temporary auxiliary prediction heads
  -> exact geometry/feasibility targets + bounded future replay targets
  -> best extractor checkpoint + PRETRAINING_COMPLETE.json

Stage 2: PPO warm-up and joint fine-tuning
  verified Stage 1 extractor
  -> freeze complete extractor for first 50,000 cumulative PPO timesteps
  -> train PPO policy/value heads
  -> unfreeze extractor at 0.1 * policy learning rate
  -> jointly fine-tune extractor and policy for six cumulative PPO hours
  -> final SB3 policy checkpoint
```

Stage 1 exact targets cover current placeability, future fit, optionality after
placement and its delta, largest free rectangle ratio, and normalized free
component count. A bounded eight-block replay additionally estimates replay
success, dropout, and delay under a deterministic future-optionality teacher.
These replay values are teacher-relative diagnostics, not globally optimal
placement labels. Target generation must run only on isolated simulator clones
and must never mutate the production episode.

## 2026-08-03 v7 release, measured six-hour outcome, and next work

`scale-aware-cnn-6h-v7` (commit `85b3175`) is released and immutable. It fixes
the two failures that surfaced while running v6:

1. Resumed runs appended to `training_log.csv` and `loss_log.csv` after the
   model restarted from an earlier checkpoint, so `timestep` regressed and
   `write_training_completion` refused the receipt after the wall-clock budget
   had already been spent. `prune_rolled_back_rows` now discards exactly the
   rolled-back rows before either log is reopened for append; retained records
   keep their original bytes and the rewritten log is validated before it
   replaces the original.
2. The notebook only copied the Drive dataset when the local directory did not
   exist, while itself creating that directory, so a rerun in the same runtime
   silently restarted shard generation at index 0. The dataset cell now ranks
   both copies and resyncs from Drive whenever Drive is more advanced.

### Measured outcome of the completed six-hour candidate-CNN run

```text
final_timestep                833368        checkpoint model_833368_g85.sb3
recorded_training_seconds     21603.5       target 21600, overrun 3.5
restart_count                 2             generation 85
steps_per_second              38.58         run_wall_span 271394 s
parameters                    910123        extractor 868192
final-model original CSV      terminal_score 0.3146, dropout 0.1763,
                              delayed 85, retained_choice 0.9820
```

Training-side performance never plateaued. Per 100k-timestep bin over the 904
logged episodes, `terminal_score` runs -0.082, 0.099, 0.307, 0.279, 0.267,
0.362, 0.353, 0.377, 0.408 and `success_rate` runs 0.541 to 0.754. The budget
stopped the run mid-learning, which is the strongest argument for the
throughput work listed below.

The remaining loss is dropout-dominated. At 800k and later the per-episode
decomposition is compliant +0.754, delay -0.081, dropout -0.280. Delayed count
halved from 196 to 97 across the run while dropout count stayed flat between
128 and 135 from 200k onward: PPO learned to shorten delays and did not learn
to avoid dropouts.

### Holdout selection signal is noise-dominated

`holdout_selection.csv` marks 600k as best with (0.2613, 0.1650, 1.0500). The
eleven evaluations from 300k onward have mean -0.02 and standard deviation
0.16, and adjacent checkpoints swing by up to 0.37 (600k to 650k). The best
pick sits about 1.7 sigma above that mean, which noise alone explains.

The cause is the sample size, not the metric. `FixedHoldoutEvalCallback` slices
`scenarios[:selection_count]` (`holdout_model_selection.py:100`) and the run
passed `--holdout-selection-count 5`, so every holdout point averaged five of
the twenty available scenarios. `data/fixed_eval_scenarios.json` holds twenty
913-block scenarios on seeds 1000-1019 built from the eleven holdout ships, so
fifteen scenarios of genuine generalization signal were discarded at every
evaluation.

### Next work, in order

1. `raw-direct` step-matched baseline. Run `--timesteps 833368` with no
   `--max-training-seconds`, every other hyperparameter identical to
   `configs/improved_cnn_6h_seed0.json`, and a separate Drive output root. The
   arms differ in per-step observation cost, so a wall-clock budget would
   confound algorithm quality with observation cost; report both axes.
2. Comparison protocol. Do not compare `best_model.sb3` across arms while it is
   selected from five scenarios. Evaluate each arm's final checkpoint on all
   twenty fixed holdout scenarios and compare paired per-seed differences, which
   cancels scenario difficulty. Optionally average each arm's last three
   checkpoints to damp checkpoint noise.
3. Raise `--holdout-selection-count` to 20 for future runs.
4. Training-configuration backlog, deferred until after the baseline. Env step
   throughput first: `_get_obs` runs `determine_placement_position` for all ten
   workspaces every step. Then `n_steps` 120 to 240-360 with `n_epochs` 5 to 3-4
   or an added `target_kl`, since a 913-step episode is currently split across
   eight rollouts under `gamma=1.0`. Then refresh the Stage 1 dataset from a
   mid-training policy, because it is collected only by random-valid and
   greedy-area policies. Supporting measurements: GPU and RAM sat near 10 per
   cent utilisation, one observation is 640 KB (`grids` is 10x4x64x64 float32),
   and the rollout buffer is about 614 MB at `n_steps=120`, so raising
   `n_steps` is affordable now while shrinking the grid dtype would change
   observation schema 4 and break checkpoint compatibility.
5. Dropout-focused investigation, since dropouts carry 78 per cent of the
   remaining score loss and PPO did not reduce them after 200k.

## 2026-08-03 next-PC handoff

Everything below is what a fresh assistant on another machine needs. The
conversation that produced it does not travel between machines; this section
does.

### Repository state

`main` is at `2c7a5c1`, pushed, with a clean working tree. Tags in play:

```text
scale-aware-cnn-6h-v7   candidate-CNN arm, complete, 833368 timesteps
raw-direct-830k-v1      raw-direct arm, STOPS AT 830000 STEPS - this is the arm being run
raw-direct-6h-v1        wall-clock variant of the same arm, not being run
```

The sibling repository `CNN-RL` (upstream baseline, commit
`cd4e14fc1725a4ff159e59d6874d3602f3b65a06`) is unchanged and is referenced only
as `--comparison-baseline-sha256`.

### What is running right now

The raw-direct arm trains in Colab from
`https://colab.research.google.com/github/LMS4681/CNN-RL-Raw-Comparison/blob/raw-direct-830k-v1/notebooks/raw_direct_830k.ipynb`
into Drive at `MyDrive/CNN-RL-improved/raw-direct-830k-seed0/ppo`. It had
40,000 timesteps when last observed and measured 44.4 steps per second. The
notebook launches only `830000 - done` each session, so any number of
interrupted sessions converges on 830,000. Roughly five hours remain.

The completed candidate-CNN arm lives at
`MyDrive/CNN-RL-improved/scale-aware-cnn-6h-seed0` with `pretraining/` and
`ppo/` children.

### Completed run: measured numbers

```text
final_timestep 833368     recorded_training_seconds 21603.5     restarts 2
steps_per_second 38.58    parameters 910123 (extractor 868192)
final model, original CSV: terminal_score 0.3146, dropout_rate 0.1763,
                           delayed 85, retained_choice 0.9820
holdout best pick: 600000 with (0.2613, 0.1650, 1.0500)
```

Training-side score improved monotonically to the end (0.36 to 0.41 over the
last 300k steps), so the budget stopped the run mid-learning. At 800k and later
the per-episode decomposition is compliant +0.754, delay -0.081, dropout
-0.280: dropouts carry 78 per cent of the remaining loss, and their count stayed
flat at 128 to 135 from 200k onward while delayed count halved from 196 to 97.

Holdout selection is noise-dominated. `FixedHoldoutEvalCallback` slices
`scenarios[:selection_count]` and the run passed 5, so every holdout point
averaged five of the twenty scenarios. The eleven evaluations after 300k have
mean -0.02 and standard deviation 0.16, and 600k sits about 1.7 sigma above
that mean. Never rank arms by `best_model.sb3`.

A published chart of both curves is at
`https://claude.ai/code/artifact/d21e3442-3332-4412-9192-7567ba2ebb75`.

### The comparison script

Written and unit-tested: `comparison/arm_comparison.py`, tests in
`AllocRL/test_arm_comparison.py` (9 pass). It resolves the largest common
regular checkpoint, evaluates both arms there on the twenty fixed scenarios,
and reports paired per-seed differences with a seeded percentile bootstrap
interval, headlining the fifteen primary-test seeds.

Run it in Colab once the raw-direct arm reaches 830,000. The training
notebook's checkout is pinned to a tag that predates this module, so clone
`main` separately:

```bash
git clone https://github.com/LMS4681/CNN-RL-Raw-Comparison.git /content/compare
cd /content/compare/AllocRL
python -m comparison.arm_comparison \
  --raw-dir /content/drive/MyDrive/CNN-RL-improved/raw-direct-830k-seed0/ppo \
  --cnn-dir /content/drive/MyDrive/CNN-RL-improved/scale-aware-cnn-6h-seed0/ppo \
  --scenarios ./data/fixed_eval_scenarios.json \
  --output-dir /content/drive/MyDrive/CNN-RL-improved/comparison-830k
```

It writes `arm_comparison_rows.csv` and `arm_comparison_summary.json` and
prints the paired report. Evaluation loads both models on CPU and runs forty
913-decision episodes, so expect a few minutes.

The rest of this subsection records why it is built this way.

`comparison/checkpoint_evaluator.py` already provides
`readable_checkpoint_inventory`, `select_common_timestep`, and
`evaluate_checkpoint`, and it defines `SELECTION_SEEDS` (1000-1004) and
`PRIMARY_TEST_SEEDS` (1005-1019). Those three functions read only the
checkpoint archives and the `run_config.json` beside them, so they work for the
step-bounded arm, which has no receipts.

The script should: select the largest common regular checkpoint between the two
Drive roots, expected 830,000; evaluate each arm there on all twenty fixed
scenarios; report paired per-seed differences over the fifteen primary-test
seeds as the headline, with the five selection seeds as a secondary view; and
give an interval rather than a p-value, since `scipy` is not in the lock file
and n is 15. A seeded percentile bootstrap over the paired differences is
enough. Also overlay the two `training_log.csv` curves: episodes are paired by
construction, because both arms use seed 0, eight environments, and a fixed
913-decision episode, so the same block sequence appears at the same step.

The marker-validated staged pipeline (`evaluate_arm_artifacts`,
`validate_arm_evaluation_stage`, `report_builder.write_complete_report`) is
unavailable for the step-bounded arm and must not be forced. `experiment_runner`
trains both arms from scratch with a candidate CNN that has no Stage 1
pretraining, which is a different experiment from v7; do not run it.

The report must state that Stage 1 supervised pretraining sits outside the
matched PPO budget, and that the comparison is step-for-step only because
equal-wall-clock efficiency was explicitly out of scope.

### Deferred backlog, in priority order

1. Profile and speed up the environment step. `_get_obs` calls
   `determine_placement_position` for all ten workspaces every step. GPU and RAM
   sat near 10 per cent during training, so the environment is the constraint.
2. `n_steps` 120 to 240-360 and `n_epochs` 5 to 3-4, or add `target_kl`. One
   913-step episode is split across eight rollouts under `gamma=1.0`. One
   observation is 640 KB (`grids` is 10x4x64x64 float32) and the rollout buffer
   is about 614 MB at `n_steps=120`, so the RAM headroom pays for this now.
   Shrinking the grid dtype would cut cost fourfold but changes observation
   schema 4 and breaks checkpoint compatibility, so it belongs to a new
   generation.
3. Refresh the Stage 1 dataset from a mid-training policy; it is currently
   collected only by random-valid and greedy-area policies.
4. Attack dropouts, per the decomposition above.
5. Raise `--holdout-selection-count` to 20 in future runs. The code currently
   rejects any value but 5, so that is a code change too.

Do not apply any of these before the raw-direct arm finishes: they would
invalidate the comparison.

### Local environment notes

Tests run with `py -3 -m pytest <file> -q` from `AllocRL`. Notebook contract
tests are `test_improved_cnn_notebook.py`, `test_raw_direct_830k_notebook.py`,
and `test_raw_direct_notebook.py`; 38 pass.

Two traps cost time on this machine:

- `test_pretrained_ppo.py::test_release_rehearsal_uses_eight_subprocess_envs_and_resumes_twice`
  deadlocks on Windows. Worker CPU time stops advancing and never recovers.
  Every other test in that file passes. Skip it here and rely on Colab.
- Pinned SHA-256 values in notebooks are of the LF content that Colab checks
  out. This checkout is CRLF, so hashing the working copy gives the wrong value.
  Compute them from the index instead: `git show :<path> | sha256sum`.

The Stage 1 acceptance gate requires all finite metrics, held-out improvement
over scalar-only/constant baselines, shuffled-grid degradation, and a
counterfactual geometry response. `PRETRAINING_COMPLETE.json` is published
last and records hashes for the accepted checkpoint, manifest, config, and
gate results. Stage 2 fails closed when this marker is missing, failed, or
hash-mismatched; random extractor initialization is not allowed for the new
production run.

The CNN and structured/fusion MLP are therefore learned twice: first through
direct simulator supervision, then through PPO reward after the temporary
freeze. Do not replace convolution kernels with fixed all-one filters and do
not freeze the extractor permanently. The executable candidate-CNN retains
its current four grid channels. The previously prepared report's three-channel
explanation is not a request to remove the candidate channel from new code.

The approved Stage 1 settings are:

```text
training states:         5,000 from seeds 20000..20039
validation states:       1,000 from seeds 30000..30009
bounded replay:          every fourth state
replay horizon:          next 8 blocks, at most 32 decisions
storage:                 sharded compressed NPZ, float16 grids
training dtype:          float32
optimizer:               AdamW
learning rate:           1e-4
batch size:              8
maximum epochs:          30
early-stop patience:     5
```

The approved Stage 2 production settings are:

```text
Colab GPU:             L4 (high-memory system profile if offered)
parallel environments: 8 with SubprocVecEnv
per-env n_steps:        120
rollout transitions:    8 * 120 = 960
batch size:             64
n_epochs:               5
initial LR:             1e-4
final LR:               1e-5
LR decay horizon:       1,000,000 cumulative timesteps
extractor warm-up:      frozen through cumulative PPO timestep 50,000
extractor LR scale:     0.1 after unfreezing
PPO training budget:    21,600 cumulative seconds (6 hours)
checkpoint interval:    10,000 timesteps
Drive experiment root:  /content/drive/MyDrive/CNN-RL-improved/scale-aware-cnn-6h-seed0
Stage 1 output:          <experiment-root>/pretraining
Stage 2 output:          <experiment-root>/ppo
```

The LR schedule must use absolute cumulative `model.num_timesteps`; SB3's
per-`learn()` progress value is insufficient because it can reset its horizon
on resume. The schedule and wall-clock budget must both continue from their
saved state after a Colab disconnect. The six-hour timer starts only after
Stage 1 passes and applies only to Stage 2 PPO.

Use `n_steps=120`, not 960, with eight environments. This preserves the old
960-transition PPO update size and prevents an eightfold rollout-buffer
increase. `batch_size=64` then produces exactly 15 minibatches per epoch.

For the final Colab UI, select:

```text
Runtime -> Change runtime type -> Hardware accelerator: GPU -> GPU type: L4
```

Colab resource availability is dynamic. The notebook must print and verify the
actual device before starting the six-hour timer. A100 is a manual fallback,
but L4 is the planned runtime because the environment simulation remains partly
CPU-bound and L4 has ample memory for this model. Do not silently run the final
job on T4.

Runtime references:

```text
https://research.google.com/colaboratory/faq.html
https://www.nvidia.com/en-us/data-center/l4/
```

Implementation must include a tiny real two-stage rehearsal: generate a small
dataset, pretrain and publish the marker, start eight-environment PPO with a
reduced test-only freeze boundary, then save/resume on both sides of that
boundary. It must prove zero extractor gradients while frozen, non-zero
gradients after unfreezing, and the exact `0.1` optimizer LR ratio after resume.

The future implementation creates a new immutable tag
`scale-aware-cnn-6h-v1`. Do not create or move that tag during planning; create
it only after schema-4 tests, Stage 1 gate tests, transfer/freeze tests, full
pytest, and the real two-stage resume rehearsal pass. Its eventual notebook
URL will be:

```text
https://colab.research.google.com/github/LMS4681/CNN-RL-Raw-Comparison/blob/scale-aware-cnn-6h-v1/notebooks/improved_cnn_6h.ipynb
```

That URL is not usable until the implementation is complete and the tag has
been pushed. The existing r4 run and `overnight-v1` release remain untouched.
