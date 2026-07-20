# Schema-3 State Correction Handoff

Updated: 2026-07-20

## Git Checkpoint

- Repository: `https://github.com/LMS4681/CNN-RL.git`
- Working branch: `feature/schema3-state-correction`
- Approved implementation checkpoint before this handoff: `ecd1ff1`
- Stage A base: `5aa39af`
- Stage A contains nine implementation commits.
- Stage B Task 1 was paused before any file edit, test run, or commit.

Do not continue this work from `main` until the feature branch has been reviewed
and merged. The feature branch is the source of truth for the remaining tasks.

## Completed Work

Stage A, Evaluation Foundation, passed its task reviews and final gate:

- Deterministic ship-group split using SHA256 of
  `20260716:{ship_no}` and a `0.20` holdout threshold.
- Pinned split: 913 source rows and 40 ships, 673 training rows and 29
  ships, 240 holdout rows and 11 ships.
- Training templates use only the training split while every episode keeps the
  full 913-block target workload and full-source monthly profile.
- Twenty fixed holdout scenarios use seeds `1000..1019`, 913 blocks, and the
  same ten empty workspaces.
- Scenario schema 3 records split provenance and validates loaded payloads.
- Model, random-valid, and greedy-immediate-area policies use one shared
  evaluation runner and metric contract.
- Original CSV evaluation is a one-shot business reference even when the
  deprecated `--n-eval 5` option is parsed.
- Baseline CLI and documentation are available through
  `run_ablation.py --evaluate-baselines`.

The complete Stage A commit sequence after the design/plans is:

1. `4766bae` - deterministic ship-group split
2. `45811b6` - explicit episode target profiles
3. `b4df296` - zero-profile validation
4. `a527d5c` - holdout scenario provenance
5. `52fb489` - hardened provenance and deterministic source selection
6. `5a4550a` - shared model and heuristic evaluation
7. `b4b3efb` - deprecated CLI regression coverage
8. `1ca1978` - shared original CSV result-row path
9. `ecd1ff1` - reproducible baseline evaluation workflow

## Verification At Checkpoint

Run from `AllocRL` unless the command is a Git command:

```powershell
py -B -m pytest -q
```

Result: `147 passed, 3 warnings, 14 subtests passed`.
The three warnings are pre-existing dependency deprecations.

```powershell
py -B -m compileall -q alloc_env baseline_policies.py evaluation_runner.py evaluation_scenarios.py evaluate_baselines.py run_ablation.py train.py
py -B -c "from evaluation_scenarios import read_scenarios, read_scenario_metadata; p='data/fixed_eval_scenarios.json'; s=read_scenarios(p); m=read_scenario_metadata(p); assert len(s)==20; assert all(len(x['blocks'])==913 for x in s); assert m['split_seed']==20260716; print(len(s), m['training_row_count'], m['holdout_row_count'])"
```

Artifact output: `20 673 240`.

Pinned artifact hashes:

- `data/fixed_eval_scenarios.json`:
  `6125F53939A1B8EEF8662B2628C0DA2F1D0F26B5B541A99252858326B38CD814`
- `data/data_split_manifest.json`:
  `D3DF1D0076248B4BCBDDB4C910A3CB81481DA65C7415C6B3CACF9E055CC3F9DF`

The scenario bundle was regenerated after enforcing the full-source
`2025-12-04` scale lower bound. The split manifest remained byte-identical.

Full baseline evaluation produced 40 ignored CSV rows, 20 seeds per policy.
The measured means were:

| Policy | Terminal score | Dropout rate | Delay days |
| --- | ---: | ---: | ---: |
| `greedy_immediate_area` | 0.8072179628 | 0.0483570646 | 0.1757393756 |
| `random_valid` | 0.0038006572 | 0.2305585980 | 1.4020701322 |

## Remaining Work

The approved design and detailed task plans are tracked under
`docs/superpowers`. Continue sequentially; each task should use focused TDD,
one full regression run before commit, and a task-scoped review.

Stage B, State and Geometry Correction:

- [ ] B1 Remove rotation from every allocation path.
- [ ] B2 Expose deterministic unassigned and pending queues.
- [ ] B3 Build pure schema-3 structured encoders.
- [ ] B4 Correct candidate-conditioned grid geometry.
- [ ] B5 Integrate the fixed schema-3 environment contract.
- [ ] B6 Share corrected structured state across all extractors.
- [ ] B7 Version training, ONNX, and visualization contracts.
- [ ] B8 Run the independent Stage B regression and contract audit.

Stage C, Training Operations and Ablation:

- [ ] C1 Add deterministic fixed-holdout model selection.
- [ ] C2 Complete run configuration and automatic resume selection.
- [ ] C3 Reduce CNN diagnostics to one observation copy per rollout.
- [ ] C4 Generate exact smoke, screening, and final commands.
- [ ] C5 Produce screening-selection and final-acceptance reports.
- [ ] C6 Update the Colab schema-3 training workflow.
- [ ] C7 Run end-to-end operational verification.

The next task is B1 in
`docs/superpowers/plans/2026-07-16-state-geometry-correction-implementation.md`.
Its non-negotiable contract is that blocks cannot rotate. `Block.turn()` may
remain defined for compatibility, but no allocation path may call it.

## Continue On Another PC

For a new clone:

```powershell
git clone https://github.com/LMS4681/CNN-RL.git
cd CNN-RL
git fetch origin
git switch --track origin/feature/schema3-state-correction
cd AllocRL
py -m pip install -r requirements.txt
py -B -m pytest -q
```

For an existing clone:

```powershell
git status
git fetch origin
git switch feature/schema3-state-correction
git pull --ff-only
cd AllocRL
py -B -m pytest -q
```

Do not run `git reset --hard` when the existing clone has local changes. Commit
or stash intentional local work before switching branches.

After completing a task:

```powershell
git status
git push origin feature/schema3-state-correction
```

Generated models, ONNX files, TensorBoard logs, and `output_ablation` results
remain ignored and are not transferred by Git. Copy those separately only when
they are needed.
