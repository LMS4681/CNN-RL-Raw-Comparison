# CNN Feature Effect and Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Explain and measure the effect of learned CNN/MLP feature extraction before PPO versus parameter-free direct observation before the same PPO policy, then stabilize candidate-CNN learning without adding attention, pointer networks, block rotation, or changing the current completed experiment artifacts.

**Architecture:** Preserve the running r4 experiment as immutable evidence and finish its report first. Add a separate feature-effect study with four existing extractor families (`raw-direct`, `structured`, `fixed-grid`, and `candidate-cnn`) so learned structured compression, spatial-grid information, and learned convolution are not conflated. Stabilize candidate-CNN in short candidate-only trials by changing one PPO variable at a time; only then run matched common-timestep and holdout confirmation.

**Tech Stack:** Python 3.12, PyTorch, Gymnasium, Stable-Baselines3, sb3-contrib MaskablePPO, pytest, Matplotlib, Google Colab, Google Drive, Git/GitHub.

## Global Constraints

- Do not modify, delete, relabel, or resume `/content/drive/MyDrive/CNN-RL-comparison/overnight-20260721` with code other than preview tag `deadline-preview-20260721-r4`.
- Keep `overnight-v1` immutable. Future commits advance `main` and use new experiment tags/output roots.
- Both comparison arms retain the same PPO actor/critic MLP `pi=[64,64]`, `vf=[64,64]`, ReLU. The phrase "no MLP" means no learned feature-extractor preprocessing, not no neural policy.
- Preserve the 913-block episode, ten fixed workspaces, action masking, no rotation, reward schema 2, scenario split, seeds, normalization, and candidate placement semantics.
- Do not add attention or pointer-network components.
- Do not alter the current report to hide adverse results while claiming empirical superiority. Separate structural capability, active feature learning, training stability, throughput, and evaluated placement performance.
- Treat all seed-0 outcomes as preliminary. A general-superiority claim requires later multi-seed confirmation.
- Make one optimization change per diagnostic trial. Do not bundle learning-rate, epoch, normalization, and architecture changes.
- Use TDD for future source changes. Every implementation task ends with focused tests, the full suite, and a separate commit.

---

## Required Report Answer

The report must answer:

> CNN과 MLP를 거쳐서 강화학습을 하는 것이, CNN과 MLP를 거치지 않고
> 강화학습을 하는 것과 효과상으로 어떤 차이가 있을지에 대한 설명 또는
> 비교 데이터

Use this exact implementation-level comparison:

| Path | Learned feature preprocessing | Spatial grid | Shared PPO MLP |
|---|---|---|---|
| `raw-direct` | none; masks and concatenates 2772 normalized scalars | ignored | yes |
| `structured` | structured/fusion MLP | ignored | yes |
| `fixed-grid` | structured/fusion MLP plus deterministic 8x8 pooling | yes | yes |
| `candidate-cnn` | structured/fusion MLP plus learned shared CNN | yes | yes |

Interpret the contrasts as follows:

- `structured - raw-direct`: effect of learned structured compression/fusion;
- `fixed-grid - structured`: effect of supplying spatial-grid information without learned convolution;
- `candidate-cnn - fixed-grid`: effect of learned convolution versus fixed pooling;
- `candidate-cnn - raw-direct`: end-to-end combined effect requested by the report, with the confounding differences disclosed.

---

### Task 1: Freeze and Validate the Current r4 Evidence

**Files:**
- External input: Drive root `/content/drive/MyDrive/CNN-RL-comparison/overnight-20260721`
- Inspect: `WIP_HANDOFF.md`
- Generated report input only: complete Drive artifact tree

**Interfaces:**
- Consumes: the running r4 stage journal, receipts, CSVs, checkpoints, logs, and final marker.
- Produces: a read-only verified evidence snapshot and a decision between complete and partial reporting.

- [ ] **Step 1: Wait for the current runner to finish without changing code or output root**

Completion requires this file:

```text
/content/drive/MyDrive/CNN-RL-comparison/overnight-20260721/COMPLETE.json
```

If it is absent, inspect `stage_journal.json` and `comparison/PARTIAL_REPORT.md`; do not infer completion from stdout progress bars.

- [ ] **Step 2: Preserve the whole artifact tree**

Create a Drive copy or download archive before running any new command. Preserve `manifest.json`, `environment.json`, `stage_journal.json`, `raw_direct/`, `candidate_cnn/`, `comparison/`, `integrity_verification.json`, and `COMPLETE.json`.

- [ ] **Step 3: Re-run only the same r4 runner for idempotent validation if needed**

```bash
python -m comparison.experiment_runner \
  --config ./configs/overnight_seed0.json \
  --output-root /content/drive/MyDrive/CNN-RL-comparison/overnight-20260721 \
  --take-over-stale-lease
```

Expected: verified completed stages are skipped. Never issue this command from `overnight-v1` against the r4 root.

- [ ] **Step 4: Record the reporting status**

Record one exact status: `COMPLETE`, `PARTIAL`, or `INVALID`. Do not replace missing values with estimates from downloaded stdout.

- [ ] **Step 5: Commit only copied small report artifacts after validation**

Do not commit model archives, checkpoints, TensorBoard events, or full logs.

---

### Task 2: Add the Required Architecture and Effect Explanation to the Report

**Files:**
- Modify: `AllocRL/comparison/report_builder.py`
- Modify: `AllocRL/test_comparison_report.py`
- Modify: `reports/README.md`

**Interfaces:**
- Consumes: canonical comparison summary, runtime metrics, training diagnostics, and paired holdout rows.
- Produces: a Korean report section that accurately explains the CNN/MLP preprocessing contrast and its expected benefits/costs.

- [ ] **Step 1: Write failing report-text tests**

Add assertions requiring all of the following concepts:

```python
assert "양쪽 모두 동일한 PPO actor/critic MLP" in text
assert "raw-direct는 학습형 특징추출 전처리가 없다" in text
assert "raw-direct는 grid를 사용하지 않는다" in text
assert "candidate-CNN은 공간 형상과 파편화를 구분할 수 있다" in text
assert "표현력의 증가가 성능 우위를 보장하지 않는다" in text
```

Also assert that the report never contains the false claim `raw-direct는 MLP를 사용하지 않는다`.

- [ ] **Step 2: Run the focused test and verify RED**

```powershell
python -m pytest AllocRL/test_comparison_report.py -q
```

Expected: the new wording assertions fail before implementation.

- [ ] **Step 3: Add the architecture-flow and effect section**

The generated report must describe:

```text
raw-direct: normalized non-grid arrays -> deterministic mask/concat -> PPO MLP
candidate-CNN: 4-channel workspace grids + structured arrays
               -> CNN + fusion MLP -> PPO MLP
```

Explain information capacity, sample efficiency, throughput, delayed spatial credit, parameter count, and optimization stability as separate dimensions.

- [ ] **Step 4: Add report-scope rules**

If empirical CNN superiority is absent, use the exact conclusion class
`구조적 정보 이점은 있으나 본 실행에서 성능 우위는 입증되지 않음`.
CNN-only gradient, weight-change, variance, and sensitivity metrics may prove
that the CNN path is active, but the report must not convert them into a
placement-performance claim.

- [ ] **Step 5: Verify and commit**

```powershell
python -m pytest AllocRL/test_comparison_report.py -q
python -m pytest -q
git add AllocRL/comparison/report_builder.py AllocRL/test_comparison_report.py reports/README.md
git commit -m "docs: explain learned feature path comparison"
```

Expected: all tests pass.

---

### Task 3: Produce Counterfactual Representation Evidence

**Files:**
- Create: `AllocRL/comparison/representation_probe.py`
- Create: `AllocRL/test_representation_probe.py`
- Modify: `AllocRL/comparison/report_builder.py`

**Interfaces:**
- Produces: `probe_representation_pair(model, observation_a, observation_b) -> dict[str, float]` and a JSON/CSV row showing how each extractor reacts when only grid geometry changes.

- [ ] **Step 1: Write a failing counterfactual test**

Build two observations with identical `block`, future, pending, summary, and
workspace metadata but different `grids`. Require:

```python
raw_delta = probe_representation_pair(raw_model, obs_a, obs_b)
cnn_delta = probe_representation_pair(cnn_model, obs_a, obs_b)
assert raw_delta["feature_l2"] == 0.0
assert cnn_delta["feature_l2"] > 0.0
```

Add a second pair with equal occupied area but different connected free-space
geometry. This is the direct explanatory evidence that scalar-equivalent
states can be distinguishable to CNN and indistinguishable to raw-direct.

- [ ] **Step 2: Run and verify RED**

```powershell
python -m pytest AllocRL/test_representation_probe.py -q
```

- [ ] **Step 3: Implement inference-only probing**

Run models in `eval()` and `torch.no_grad()`. Record feature L2 difference,
action-probability L1 difference, and selected-action change. Reject mismatched
non-grid observations rather than silently accepting an invalid pair.

- [ ] **Step 4: Add the probe table to the report as mechanism evidence**

Label it `표현 민감도`, not placement quality. Missing probe artifacts must be
reported as `자료 없음`.

- [ ] **Step 5: Verify and commit**

```powershell
python -m pytest AllocRL/test_representation_probe.py AllocRL/test_comparison_report.py -q
python -m pytest -q
git add AllocRL/comparison/representation_probe.py AllocRL/test_representation_probe.py AllocRL/comparison/report_builder.py
git commit -m "feat: measure spatial representation sensitivity"
```

---

### Task 4: Add a Four-Arm Feature-Effect Study Without Changing the Existing Runner

**Files:**
- Create: `AllocRL/configs/feature_effect_seed0.json`
- Create: `AllocRL/comparison/feature_effect_runner.py`
- Create: `AllocRL/test_feature_effect_runner.py`
- Reuse unchanged: `AllocRL/train.py`, existing extractors, scenario evaluator

**Interfaces:**
- Consumes: existing `raw-direct`, `structured`, `fixed-grid`, and `candidate-cnn` extractor choices.
- Produces: four independent arm outputs, matched-timestep evaluation CSV, throughput table, and feature-effect contrast summary.

- [ ] **Step 1: Write failing exact-arm and order tests**

Require exact arms and labels:

```python
ARMS = ("raw-direct", "structured", "fixed-grid", "candidate-cnn")
CONTRASTS = {
    "learned_structured": ("structured", "raw-direct"),
    "grid_information": ("fixed-grid", "structured"),
    "learned_convolution": ("candidate-cnn", "fixed-grid"),
    "combined": ("candidate-cnn", "raw-direct"),
}
```

Assert that the existing overnight runner and `overnight_seed0.json` remain byte-unchanged.

- [ ] **Step 2: Define a short diagnostic budget**

Use the same seed/data/reward/PPO head and a fixed sample budget of 50,000
timesteps per arm. Record wall-clock time separately. The first study diagnoses
sample efficiency; it does not replace the three-hour wall-clock study.

- [ ] **Step 3: Implement the separate runner by composing existing train/evaluate commands**

Use a new output root and stage journal. Do not add the four arms to the current
overnight runner. Evaluate all arms on the same primary and all-holdout scenario
sets at exactly 50,000 timesteps.

- [ ] **Step 4: Write the contrast summary**

For each contrast, report terminal score, dropout rate, delayed count, mean
delay days, steps/s, parameter count, and peak CUDA memory. Use signed
`left-minus-right` columns and state that seed 0 is preliminary.

- [ ] **Step 5: Verify and commit**

```powershell
python -m pytest AllocRL/test_feature_effect_runner.py -q
python -m pytest -q
git add AllocRL/configs/feature_effect_seed0.json AllocRL/comparison/feature_effect_runner.py AllocRL/test_feature_effect_runner.py
git commit -m "feat: add feature effect ablation study"
```

---

### Task 5: Stabilize Candidate-CNN PPO One Variable at a Time

**Files:**
- Modify: `AllocRL/train.py`
- Create: `AllocRL/configs/candidate_stability_seed0.json`
- Create: `AllocRL/comparison/candidate_stability_runner.py`
- Create: `AllocRL/test_candidate_stability_runner.py`

**Interfaces:**
- Produces: short candidate-only trials with one explicit delta from the current PPO contract and identical 50,000-timestep diagnostics.

- [ ] **Step 1: Preserve the current candidate as trial A**

```text
A: learning_rate=3e-4, n_epochs=10, target_kl=None
```

- [ ] **Step 2: Test learning rate only**

```text
B: learning_rate=1e-4, n_epochs=10, target_kl=None
```

Keep every other field identical. Promote B only if it lowers mean late-window
KL and clip fraction without reducing holdout terminal score.

- [ ] **Step 3: Test epoch count only from the promoted learning rate**

```text
C: promoted learning_rate, n_epochs=5, target_kl=None
```

- [ ] **Step 4: Test the KL guard only after B/C selection**

```text
D: selected learning_rate/epochs, target_kl=0.03
```

Add a validated `--target-kl` option to `train.py`; default `None` preserves all
existing runs.

- [ ] **Step 5: Apply diagnostic gates**

Use these as engineering gates, not scientific claims:

```text
late-window approx_kl <= 0.05
late-window clip_fraction <= 0.30
explained_variance positive and improving
finite non-zero CNN gradient and weight change
no worse primary holdout terminal score than trial A
```

If no trial passes, stop before architecture changes and document the failure.

- [ ] **Step 6: Verify and commit**

```powershell
python -m pytest AllocRL/test_candidate_stability_runner.py AllocRL/test_parallel_training_config.py AllocRL/test_train_resume_cli.py -q
python -m pytest -q
git add AllocRL/train.py AllocRL/configs/candidate_stability_seed0.json AllocRL/comparison/candidate_stability_runner.py AllocRL/test_candidate_stability_runner.py
git commit -m "feat: add candidate PPO stability trials"
```

---

### Task 6: Normalize CNN Fusion Only If PPO Tuning Is Insufficient

**Files:**
- Modify: `AllocRL/alloc_env/cnn_extractor.py`
- Modify: `AllocRL/test_feature_extractors.py`
- Modify: `AllocRL/test_cnn_diagnostics.py`

**Interfaces:**
- Consumes: Task 5 evidence showing hyperparameter-only trials failed the gates.
- Produces: one normalization-only candidate variant with unchanged observation and reward semantics.

- [ ] **Step 1: Write failing scale-stability tests**

Require finite outputs and bounded batch feature variance for representative
observations before/after a grid-only perturbation. Preserve output shape 256
and positive candidate-channel sensitivity.

- [ ] **Step 2: Add `LayerNorm` after image projection and global fusion**

Do not alter convolution channels, grid resolution, future encoding, policy
head, or reward in this task.

- [ ] **Step 3: Repeat the selected 50,000-timestep stability trial**

Compare against the best Task 5 trial at the same timestep and scenarios.

- [ ] **Step 4: Verify and commit**

```powershell
python -m pytest AllocRL/test_feature_extractors.py AllocRL/test_cnn_diagnostics.py -q
python -m pytest -q
git add AllocRL/alloc_env/cnn_extractor.py AllocRL/test_feature_extractors.py AllocRL/test_cnn_diagnostics.py
git commit -m "fix: normalize candidate CNN fusion features"
```

---

### Task 7: Run Confirmation and Build the Final Report

**Files:**
- Generated externally: new Drive experiment roots
- Modify after validated results: `reports/preliminary_comparison_ko.md`
- Add small canonical artifacts only: summary JSON/CSV and plots

**Interfaces:**
- Consumes: the current r4 result plus Tasks 3-6 diagnostic/confirmation outputs.
- Produces: an evidence-scoped Korean report answering the required CNN/MLP effect question.

- [ ] **Step 1: Choose the final candidate before confirmation**

Selection uses only designated selection scenarios and stability gates. Do not
select using primary test seeds 1005..1019.

- [ ] **Step 2: Run common-timestep and equal-wall-clock views**

Report both because CNN may improve per sample while processing fewer samples
per hour. Keep throughput and performance conclusions separate.

- [ ] **Step 3: Use this report structure**

```text
1. 질문과 비교 모델의 정확한 정의
2. 관측 및 학습 데이터 흐름
3. CNN/MLP 특징추출의 기대 효과와 비용
4. 실험 통제와 한계
5. 표현 민감도 및 CNN 자체 진단
6. 동일 timestep 비교
7. 동일 wall-clock 비교
8. holdout 배치 성능
9. 결론 및 후속 개선
```

- [ ] **Step 4: Apply the honest result branch**

If CNN is favorable, state only that it was favorable in this seed/config and
avoid general superiority. If CNN is unfavorable, center the report on the
mechanism evidence and stabilization findings, but state that empirical
placement superiority was not demonstrated. If the submitted document is
explicitly a comparison report, include the comparison table or appendix even
when unfavorable.

- [ ] **Step 5: Verify report provenance and publish**

```powershell
python -m pytest AllocRL/test_comparison_report.py -q
python -m pytest -q
git diff --check
git status --short
git add reports docs WIP_HANDOFF.md
git commit -m "docs: publish CNN feature effect report"
git push origin HEAD:main
```

Expected: only small report/document artifacts are committed; no models,
checkpoints, TensorBoard logs, or full stdout logs are tracked.

---

## Execution Order on the Next PC

1. Clone `main` and read `WIP_HANDOFF.md` plus this plan.
2. Preserve and validate the current r4 Drive result.
3. Finish the current evidence-scoped report before changing training code.
4. Implement Tasks 2-3 for the required explanation and mechanism evidence.
5. Run the four-arm 50k feature-effect diagnostic.
6. Execute candidate stability trials one variable at a time.
7. Apply normalization only if hyperparameter changes are insufficient.
8. Run confirmation and publish the final report with explicit limitations.
