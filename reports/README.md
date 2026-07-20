# Generated comparison report contract

The experiment runtime writes generated artifacts beneath its Drive-root `comparison/` directory. Do not commit overnight models, checkpoints, TensorBoard data, raw logs, or generated results from this directory.

`summary.json` is canonical UTF-8 JSON, sorted by key. `scenario_paired_differences.csv` has a fixed column order and primary-test seeds 1005 through 1019 in increasing order. Rebuilding either from the same canonical artifacts must produce byte-identical output. `learning_curves.png` and `holdout_comparison.png` are visual artifacts and may differ in bytes while retaining the same data.

The headline is the one seed-0 training run evaluated on the 15 unselected primary scenarios. Those scenarios are paired evaluations, not 15 independent training runs. The five selection scenarios (1000–1004) are checkpoint-selection evidence only; the all-20 view is supporting context. A report must call the result preliminary, must not claim statistical significance or general CNN superiority, and must disclose missing values as `자료 없음` rather than zero or estimates.

`COMPLETE.json` is an integrity marker owned by the experiment orchestrator, not the report builder. If a required stage is missing, the report builder writes `PARTIAL_REPORT.md` describing the failure and available stages; it must not invent plots or data.

When present, `training_log.csv`, `loss_log.csv`, and `progress_timing.csv` are strict canonical inputs, not best-effort hints. Their headers and finite, monotonic rows are validated; the learning plot shows terminal score, checkpoint wall-time progress, and the recorded PPO loss trace. An absent optional curve is labelled `자료 없음`.

Partial reports validate `stage_journal.json` when present and describe completed, failed, interrupted, in-progress, and missing stages. Arm availability requires valid runtime and paired evaluation artifacts; a partial report never creates or changes `COMPLETE.json`.

Task 6 writes the journal as a direct object mapping the documented canonical stage names to entries with `status`, input/output SHA-256 values, UTC timestamps, and an optional error. Unknown stages, invalid hashes/timestamps, and unsafe error text are treated as invalid metadata.
