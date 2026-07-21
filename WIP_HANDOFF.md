# Raw observation vs CNN comparison: final handoff

Updated: 2026-07-21 (Asia/Seoul)

## Release state

- Final implementation branch: `feature/common-step-finish`.
- Reviewed implementation commits run through `17dea28`.
- The release commit adds only this final handoff update.
- The immutable release tag is `overnight-v1` and must resolve to the same
  commit as the published `main` branch.
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
is intentionally different from `overnight-v1`. Let it finish under r4. Resume
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
git describe --tags --exact-match
```

The last command must print `overnight-v1` when the checkout is the immutable
release commit.

Final Colab URL for a fresh output root:

```text
https://colab.research.google.com/github/LMS4681/CNN-RL-Raw-Comparison/blob/overnight-v1/notebooks/overnight_compare.ipynb
```
