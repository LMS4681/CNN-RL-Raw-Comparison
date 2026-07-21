# Schema-3 State Correction Handoff

Updated: 2026-07-20

## Git Checkpoint

- Repository: `https://github.com/LMS4681/CNN-RL.git`
- Working branch: `feature/schema3-state-correction`
- Current handoff checkpoint: `26a5be3`
- Stage B integration-gate closure: `c7aced7`
- Stage C approved checkpoint through C3: `76d878c`
- C4 completion repair: `f55496e`
- Historical rejected C4 implementation: `2b5b66a`
- Stage A base: `5aa39af`
- Stage A contains nine implementation commits.
- Stage B Tasks B1-B8 are complete and independently verified.
- Stage C Tasks C1-C4 are complete and independently verified.

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

Stage B, State and Geometry Correction, is complete:

- Rotation is absent from every allocation path while `Block.turn()` remains
  available only for compatibility.
- Deterministic unassigned, future, and pending queues feed the fixed nine-key
  schema-3 observation.
- Candidate-conditioned grids use independent physical x/y coordinate maps,
  fixed `64 x 64` geometry, and cache identity tied to observable state.
- All extractors consume the corrected structured state under explicit
  `full` and `current` state-context modes.
- Training, resume, ONNX, fixed scenarios, and visualization enforce the
  schema-3 compatibility contract.
- The all-extractor smoke trained, saved, loaded, and evaluated `structured`,
  `fixed-grid`, and `candidate-cnn`; the candidate CNN had nonzero gradients
  and a nonzero parameter update.
- The integration-gate closure removes the stale Colab future-count option,
  repairs independent-axis grid reporting, and makes the future-choice preview
  reuse the exact candidate position observed for the current action.

The Stage B commit sequence after the handoff checkpoint is:

1. `0c749a9` - prohibit block rotation in allocation paths
2. `179f0e7`, `86c8f33` - deterministic placement queues and validation
3. `b668852`, `8633316` - schema-3 future/pending encoders and validation
4. `5cfe48b`, `7fa3956` - candidate grids, coordinate maps, and cache identity
5. `60e2c6f`, `e787651` - schema-3 environment and scale contracts
6. `d94d4ab`, `702ca26` - shared extractor state and strict validation
7. `ecaa5e1`, `9702a38` - schema-3 tools and fixed-scenario scale bounds
8. `d00e9ff` - independent Stage B extractor workflow verification
9. `c7aced7` - Stage B integration-gate closure

Stage C completed work:

1. `1840f65`, `cd481b6` - deterministic fixed-holdout model selection,
   selected-model reporting, and deterministic environment/model cleanup
2. `cfe2c60`, `4301aa6` - complete run compatibility and newest readable
   training-state resume selection, including real corrupt-archive handling
3. `1a28170`, `76d878c` - rollout-level candidate-CNN diagnostics with zero
   diagnostic observation copies for non-CNN extractors

4. `2b5b66a` was the rejected initial C4 implementation; `f55496e` repaired
   its five review findings: controlled-option abbreviation rejection, legacy
   builder removal, frozen stage seeds, repeated selected-option rejection,
   and accurate dry-run documentation. The follow-up review also confirmed
   duplicate-option validation runs before utility-mode dispatch.

C4 is complete. C5, screening-selection and final-acceptance reporting, is
the next task.

## Verification At Checkpoint

Run from `AllocRL` unless the command is a Git command:

```powershell
py -3.12 -m pytest -q
```

Current verification result: `519 passed, 2 warnings, 54 subtests passed`.
The two warnings are pre-existing ONNX-export dependency deprecations. The C4
focused command contract suite also passed with `99 passed, 6 subtests`.

```powershell
py -B -m compileall -q alloc_env baseline_policies.py evaluation_runner.py evaluation_scenarios.py evaluate_baselines.py run_ablation.py train.py
py -B -c "from evaluation_scenarios import read_scenarios, read_scenario_metadata; p='data/fixed_eval_scenarios.json'; s=read_scenarios(p); m=read_scenario_metadata(p); assert len(s)==20; assert all(len(x['blocks'])==913 for x in s); assert m['split_seed']==20260716; print(len(s), m['training_row_count'], m['holdout_row_count'])"
```

Artifact output: `20 673 240`.

Pinned artifact hashes:

- `data/fixed_eval_scenarios.json`:
  `913CAC9046DEC8164EF65DA60275522F7127DE5EA775B1C5A6B6AAC255716271`
- `data/data_split_manifest.json`:
  `601BD6143ED8890577E5FF34921241D36FD6A0E99C4BDAB4E26152AB168178F8`

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

- [x] B1 Remove rotation from every allocation path.
- [x] B2 Expose deterministic unassigned and pending queues.
- [x] B3 Build pure schema-3 structured encoders.
- [x] B4 Correct candidate-conditioned grid geometry.
- [x] B5 Integrate the fixed schema-3 environment contract.
- [x] B6 Share corrected structured state across all extractors.
- [x] B7 Version training, ONNX, and visualization contracts.
- [x] B8 Run the independent Stage B regression and contract audit.

Stage C, Training Operations and Ablation:

- [x] C1 Add deterministic fixed-holdout model selection.
- [x] C2 Complete run configuration and automatic resume selection.
- [x] C3 Reduce CNN diagnostics to one observation copy per rollout.
- [x] C4 Generate exact smoke, screening, and final commands (repaired in
  `f55496e`).
- [ ] C5 Produce screening-selection and final-acceptance reports.
- [ ] C6 Update the Colab schema-3 training workflow.
- [ ] C7 Run end-to-end operational verification.

C4 was initially implemented in rejected commit `2b5b66a`. Its independent
review identified five defects: forwarded controlled-option abbreviations, the
obsolete three-argument builder adapter, non-frozen builder seed sequences,
repeated selected-option handling, and inaccurate dry-run documentation. All
five were repaired in `f55496e`, including the review follow-up that moves
duplicate-option validation before utility-mode dispatch.

C4 verification is current: the focused command contract suite passed with
`99 passed, 6 subtests`, and the full suite passed with `519 passed, 2
warnings, 54 subtests`. The next task is C5: produce the screening-selection
and final-acceptance reports. Local `.superpowers/sdd` task reports are ignored
and are not transferred through Git; this tracked handoff and the tracked plan
above are the authoritative continuation record.

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
