# Colab Resume Diagnostics V2 Design

Date: 2026-07-30

## Goal

Add one standalone diagnostic code cell to the end of
`notebooks/improved_cnn_6h.ipynb`. The cell must turn the opaque outer
`CalledProcessError` from the PPO resume cell into a compact report that can
be shared without copying a large traceback.

## Scope

The change is limited to:

- the final diagnostic cell in the improved six-hour Colab notebook;
- notebook contract tests;
- the README Colab link and immutable release tag.

The training command, reward system, checkpoint writer, resume selection,
pretraining workflow, and model implementation are unchanged.

## Diagnostic Contract

The final cell is independently runnable after Drive has been mounted. It
must initialize its own repository, PPO output, and pretraining paths rather
than depending on variables created by earlier notebook cells.

It reports:

1. `run_state.json` and `run_config.json` availability and selected fields;
2. the checkpoint selected by `last_checkpoint_file`;
3. checkpoint existence, size, ZIP validity, and SHA256 consistency;
4. checkpoint loadability through `ScaleAwareMaskablePPO.load(..., device="cpu")`;
5. model timestep versus the timestep recorded in resume state;
6. Stage 1 pretraining artifact verification;
7. the tail of `progress_timing.csv`;
8. a final `PASS`, `WARN`, or `FAIL` summary with concise reasons.

Missing or malformed artifacts must be reported in the cell output instead of
being hidden behind a second unhandled exception.

## Safety

The diagnostic cell is read-only. It must not:

- run training or evaluation;
- create, replace, rename, or delete artifacts;
- alter `run_state.json`;
- automatically repair hashes or checkpoint metadata;
- load the checkpoint onto the GPU.

## Release

The notebook pins `RELEASE_TAG = "scale-aware-cnn-6h-v2"`. The existing
`scale-aware-cnn-6h-v1` tag remains unchanged. After tests pass, the new
commit is published on `main` and tagged with the immutable annotated tag
`scale-aware-cnn-6h-v2`.

The resulting Colab URL is:

```text
https://colab.research.google.com/github/LMS4681/CNN-RL-Raw-Comparison/blob/scale-aware-cnn-6h-v2/notebooks/improved_cnn_6h.ipynb
```

## Verification

Automated tests must verify that:

- the notebook is clean and every code cell compiles;
- the last cell is a standalone diagnostic cell;
- all required artifact, hash, ZIP, model-load, timestep, Stage 1, and
  progress checks are present;
- forbidden write, delete, repair, and training operations are absent;
- the notebook and README use the V2 immutable release URL.

