# Raw observation vs CNN comparison: cross-PC WIP handoff

Updated: 2026-07-21 (Asia/Seoul)

## Git branches

- Stable handoff point for `main`: `04c2199e29a7029ae12db5089d68ffa6d6d22322`
  (`feat: evaluate and trust each comparison arm`)
- Common-step WIP code checkpoint: `d7c631c7df070eee578eb25171ee850dd6202681`
  (`wip: checkpoint common-step evaluation`)
- Continue from the remote branch `wip/common-step-evaluation`, not from the
  immutable `overnight-v1` tag. That release tag has intentionally not been
  created because final verification is incomplete.

After the repository is published, resume on another PC with:

```powershell
git clone https://github.com/LMS4681/CNN-RL-Raw-Comparison.git
cd CNN-RL-Raw-Comparison
git switch --track origin/wip/common-step-evaluation
```

## Read first

1. This file.
2. `docs/superpowers/specs/2026-07-21-raw-observation-cnn-comparison-design.md`.
3. `docs/superpowers/plans/2026-07-21-overnight-raw-cnn-comparison-implementation.md`.

Suggested instruction for the next coding session:

> `WIP_HANDOFF.md`를 읽고, common-step WIP 독립 검토와 아래 미완성
> 회귀 테스트부터 이어서 수정하라. 전체 검증 전에는 `overnight-v1` 태그를
> 만들거나 Colab 6시간 실행을 시작하지 마라.

## Completed and committed

- Separate raw-direct observation arm and the candidate CNN arm.
- Fixed scenario/split/config contracts and immutable input hashes.
- One-notebook sequential Colab workflow: raw 3 hours, then CNN 3 hours.
- Google Drive state, lease, stale takeover, restart and wall-clock checkpointing.
- Transactional `run_state.json` / `progress_timing.csv` publication.
- Durable `run_origin.json`, runtime schema v2 and `training_completion.json`.
- Learning-free finalize-only recovery after a completed-state crash.
- Comparable-environment check before a resumed training subprocess.
- Strict path, symlink/junction and curve-log validation.
- Canonical selected/fallback provenance with exact fallback reason codes.
- Real per-arm selected/fallback evaluation: exact 20 scenario rows and exact
  15 primary-test rows, atomic publication, manifest refs and marker-last.
- Raw-only `PARTIAL_REPORT.md` regeneration before CNN training, including a
  valid-marker resume path.
- Publication docs, exact Colab badge, LF-locked dependency file and a
  no-secret PowerShell gate.

Stable per-arm verification at commit `04c2199`:

- `321 passed, 4 skipped`
- compileall passed
- `git diff --check` passed

## Current common-step WIP

Commit `d7c631c` implements, but has not yet received independent review:

- model-load-before-link-check protection for checkpoint inventories;
- common-timestep-only evaluation (the legacy selected-model rewrite path is
  no longer used by the runner);
- one atomic durable cache per arm;
- reuse of a valid first-arm cache after a second-arm interruption;
- exact 40-row combined common-step CSV;
- common-only manifest reference updates;
- common stage marker-last validation and fast resume;
- runner stage path/output hash wired to the common marker.

Verification performed on the WIP after the last fix:

```text
python -m pytest test_checkpoint_evaluator.py test_comparison_experiment.py -q
237 passed, 4 skipped in 20.80s
```

Also passed:

- compileall for comparison/train/evaluation modules;
- `git diff --check`.

The four skips are Windows tests that require symlink privileges. Junction
coverage ran separately in the preceding persistence work.

## Required next work

1. Independently review `04c2199..d7c631c`, especially common-cache crash
   recovery, marker ownership, direct-regular paths and preservation of all
   per-arm CSV/selected/final refs.
2. Add or confirm these explicit regression tests:
   - real `fallback_final` `evaluate_arm_artifacts` publication;
   - manifest write failure after both per-arm CSVs, no trusted marker, then a
     coherent retry;
   - production (not injected test-hook) valid-marker skip -> raw partial
     report -> CNN subprocess ordering;
   - marker/CSV/runtime/receipt symlink rejection;
   - first common-arm cache survives a second-arm injected crash and only the
     second arm is evaluated on retry;
   - common manifest failure leaves no final marker and retries from caches;
   - selected/final per-arm CSV bytes and refs remain unchanged by common
     evaluation.
3. Reconcile report/integrity logic with all per-arm and common markers.
4. Print the canonical `fallback_reason` in the Korean report instead of the
   generic `자료 없음` text.
5. Run the complete pytest suite in a fresh checkout. The old Windows worktree
   still held a CRLF working copy of `requirements-comparison.txt`; a fresh
   checkout obeys `.gitattributes`, has LF bytes and SHA-256
   `37634576e34043d169cf24bfc0cc2261818dc65b9358d4b9b2e46ab614d0bdda`.
6. Run the real local end-to-end test with 30 seconds per arm and verify both
   model archives, receipts, exact row counts, common marker, report,
   integrity record and `COMPLETE.json`.
7. Only after clean review and full/E2E verification, create the immutable
   `overnight-v1` tag and release.

## Colab handoff

This Codex session had no connected in-app browser, so it could not choose a
Colab GPU or approve Google Drive OAuth. After the final tag exists, a signed-in
user must open the README badge, select a GPU runtime, choose **Run all once**,
and approve `drive.mount`. Everything after that is intended to run
sequentially and persist under:

```text
/content/drive/MyDrive/CNN-RL-comparison/overnight-20260721
```

Do not run two notebooks against that same Drive root.
