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
