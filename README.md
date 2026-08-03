# Raw Observation vs Candidate CNN: overnight Colab comparison

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LMS4681/CNN-RL-Raw-Comparison/blob/overnight-v1/notebooks/overnight_compare.ipynb)

Use the one GPU Colab notebook at `notebooks/overnight_compare.ipynb`. Select a GPU runtime, then **Run all once**. Keep the browser tab/runtime active: the comparison takes approximately 6 hours plus setup/eval.

Drive is authoritative. Rerun all to resume from the last verified generation after an interruption. After an abrupt VM termination, wait until the lease is more than 15 minutes old, then rerun all; the command performs guarded stale takeover. A VM kill can leave up to 300 seconds plus the current callback interval unrecorded. Colab cannot guarantee uninterrupted completion.

The notebook runs one single-seed preliminary comparison, not a statistically
conclusive result. It executes `raw-direct/full` first and then candidate CNN,
storing all durable artifacts under
`/content/drive/MyDrive/CNN-RL-comparison/overnight-20260721/`.

Local checks run from `AllocRL/`:

```powershell
python -m pytest test_comparison_notebook.py test_requirements.py -q
python -m pytest -q
```

The durable artifact root contains `manifest.json`, `environment.json`,
`stage_journal.json`, per-arm checkpoints and logs, `comparison/`, and either
`COMPLETE.json` or `comparison/PARTIAL_REPORT.md`.

## Scale-aware CNN two-stage run

[![Open improved CNN in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LMS4681/CNN-RL-Raw-Comparison/blob/scale-aware-cnn-6h-v7/notebooks/improved_cnn_6h.ipynb)

For the new candidate model, select `Runtime -> Change runtime type -> GPU ->
GPU type: L4` and run `notebooks/improved_cnn_6h.ipynb`. The notebook first
builds and verifies simulator-supervised Stage 1 features, then runs exactly
six PPO hours with eight subprocess environments. Stage 1 time is outside the
six-hour PPO budget. Rerunning all restores verified dataset shards,
`pretraining_last.pt`, and the exact checkpoint named by PPO `run_state.json`.

The durable root is
`/content/drive/MyDrive/CNN-RL-improved/scale-aware-cnn-6h-seed0`; Stage 1 uses
its `pretraining/` child and PPO uses its separate `ppo/` child. The pinned
`scale-aware-cnn-6h-v7` notebook supplies the immutable baseline, scenario,
split, and dependency-lock provenance required by wall-clock PPO training. It
also relays native PPO stdout and stderr into the cell, preserves the final
child-output tail on failure, and prints durable checkpoint progress every 30
seconds. It ends with a standalone, read-only resume diagnostic cell. V7
resyncs the local dataset copy whenever the Drive copy is more advanced, and
discards the rows a resume rolled back from `training_log.csv` and
`loss_log.csv` so the completion receipt stays reachable. The V1 through V6
tags remain immutable.

## Raw-direct comparison arm

[![Open raw-direct arm in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LMS4681/CNN-RL-Raw-Comparison/blob/raw-direct-830k-v1/notebooks/raw_direct_830k.ipynb)

`notebooks/raw_direct_830k.ipynb` trains the comparison arm that feeds
observations straight to the policy with no learned extractor
(`--extractor raw-direct`). It stops on a timestep ceiling of **830,000** PPO
steps rather than a wall clock, which matches the scale-aware run's 833,368
final timestep to within one checkpoint interval, and it trains only the
remaining budget on every resume. Because no wall-clock budget is set, this arm
writes no completion receipt; `arm_timing.json` records its throughput instead.
Stage 1 supervised pretraining is not part of this arm and is excluded from the
matched budget. The design is in
`docs/superpowers/specs/2026-08-03-raw-direct-830k-comparison-design.md`.
