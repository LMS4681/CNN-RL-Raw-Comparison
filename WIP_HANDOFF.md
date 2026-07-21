# Raw observation vs CNN comparison: code and experiment handoff

Updated: 2026-07-21 15:00 (Asia/Seoul)

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
