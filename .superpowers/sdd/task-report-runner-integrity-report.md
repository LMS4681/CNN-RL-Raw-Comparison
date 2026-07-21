# Report and runner integrity task report

## Implementation commit

`c4c06a0705f6a936b163eaeb187700b96fcf5b3d`

## Red evidence

- Complete-report marker contract:
  `py -3.12 -m pytest test_comparison_report.py::test_complete_report_requires_valid_stage_markers test_comparison_report.py::test_complete_report_rejects_symlinked_stage_markers -q`
  failed `8` cases because `write_complete_report()` did not raise for missing,
  malformed, provenance-mismatched, or symlinked markers.
- Final-integrity marker contract:
  `py -3.12 -m pytest test_comparison_experiment.py::test_final_integrity_requires_valid_stage_markers test_comparison_experiment.py::test_final_integrity_rejects_symlinked_stage_markers -q`
  failed `8` cases because `integrity()` did not validate stage markers.
- Canonical fallback reason:
  `py -3.12 -m pytest test_comparison_report.py::test_fallback_final_report_prints_exact_canonical_reason -q`
  failed `2` cases because the report printed `자료 없음` instead of the
  validated reason codes.
- Common-step resume provenance:
  `py -3.12 -m pytest test_comparison_experiment.py::test_common_output_hash_validates_only_common_stage_marker -q`
  failed because `expected_run_config_sha256` was absent.
- Archive fallback:
  `py -3.12 -m pytest test_comparison_experiment.py::test_final_stage_marker_validation_uses_archive_reader_fallback -q`
  failed because all three validators received a non-callable reader.

The required production-path regression passed immediately (`1 passed`): the
existing non-injected runner path already skipped a valid raw marker, refreshed
`PARTIAL_REPORT.md`, and then launched the candidate-CNN subprocess. This was a
coverage gap rather than a production behavior defect.

## Green evidence

- Complete-report marker matrix: `8 passed`.
- Final-integrity marker matrix: `8 passed`.
- Canonical fallback reasons (`selection_not_run`, `best_model_missing`):
  `2 passed`.
- Common-step run-config provenance: `1 passed`.
- Production-path ordering: `1 passed`.
- Archive reader fallback: `1 passed`.
- Final focused suite:
  `py -3.12 -m pytest test_comparison_report.py test_comparison_experiment.py -q`
  returned `305 passed, 87 warnings in 375.74s`.
- `py -3.12 -m compileall -q comparison` returned exit code `0`.
- `git diff --check` returned exit code `0` with no findings.

## Changed files

- `AllocRL/comparison/report_builder.py`
- `AllocRL/comparison/experiment_runner.py`
- `AllocRL/test_comparison_report.py`
- `AllocRL/test_comparison_experiment.py`
- `.superpowers/sdd/task-report-runner-integrity-report.md`

## Concerns

- Pytest still reports pre-existing dependency deprecations and missing Hangul
  font glyph warnings; there are no test failures.
- Three lease-thread tests needed wider test-only waits because TensorFlow lazy
  import exceeded their original one- or two-second startup limits on this PC.
- No training, observation, reward, seed, PPO, dependency-lock, notebook, or
  deadline-preview tag files were changed.
