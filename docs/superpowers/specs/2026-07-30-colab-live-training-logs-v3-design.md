# Colab Live Training Logs V3 Design

Date: 2026-07-30

## Goal

Make Stage 2 PPO progress visible directly below the training cell in
`notebooks/improved_cnn_6h.ipynb`. Preserve the model, reward, observation,
pretraining, checkpoint, and resume behavior from V2.

## Approach

Replace the Stage 2 cell's blocking `subprocess.run(...)` wrapper with a
`subprocess.Popen(...)` polling wrapper.

The child process keeps inherited stdout and stderr. Existing `train.py`,
Stable-Baselines3, progress-bar, warning, and traceback output therefore
continues to appear directly in the Colab cell without being captured or
rewritten.

While the child is running, the parent prints one structured monitor line
every 30 seconds. The line contains:

- parent-observed elapsed wall time;
- process status;
- the most recently persisted timestep;
- persisted cumulative training seconds;
- target training seconds;
- checkpoint filename;
- durable state update time.

The monitor labels the timestep as `durable_timestep`. It must not describe
this value as the exact live timestep because `run_state.json` advances only
when the existing checkpoint/heartbeat callback persists a verified
generation.

## Data Flow

1. Build `ppo_command` and choose the exact resume command exactly as V2 does.
2. Print a concise process-start line.
3. Start the command in `ALLOC_RL` with the existing unbuffered Python
   environment and inherited stdout/stderr.
4. Wait for at most 30 seconds at a time.
5. On timeout, read `PPO_ROOT/run_state.json` if available and print a
   structured monitor line with `flush=True`.
6. On normal exit, print the return code.
7. On a nonzero exit, raise `subprocess.CalledProcessError` with the original
   command so failure semantics remain unchanged.

Malformed, missing, or temporarily unavailable state data is reported as
`durable_state=pending` or a concise read error. It does not terminate the
training process.

## Interruption

If the Colab cell receives `KeyboardInterrupt`, the wrapper forwards
`SIGINT` to a still-running child and waits briefly. It escalates to
termination only if the child does not exit. The interruption is then
re-raised to Colab. The wrapper does not create, alter, or repair checkpoint
artifacts.

## Scope

Changed:

- the Stage 2 execution wrapper in `notebooks/improved_cnn_6h.ipynb`;
- notebook contract tests;
- the README pinned Colab URL;
- the notebook immutable tag from V2 to V3.

Unchanged:

- every `ppo_command` argument;
- six-hour wall-clock budget;
- eight parallel environments;
- learning-rate schedule;
- reward and observation systems;
- model and feature-extractor code;
- checkpoint persistence and exact-resume selection;
- Stage 1 dataset generation and encoder pretraining;
- the final standalone diagnostic cell, except its visible V3 heading if
  needed for release consistency.

## Release

The existing `scale-aware-cnn-6h-v1` and `scale-aware-cnn-6h-v2` tags remain
immutable. The finished commit is published on `main` and tagged:

```text
scale-aware-cnn-6h-v3
```

Pinned Colab URL:

```text
https://colab.research.google.com/github/LMS4681/CNN-RL-Raw-Comparison/blob/scale-aware-cnn-6h-v3/notebooks/improved_cnn_6h.ipynb
```

## Verification

Tests must prove that:

- the Stage 2 cell uses `subprocess.Popen` with inherited stdout and stderr;
- the child still receives `PYTHONUNBUFFERED=1` and `python -u`;
- the monitor interval is exactly 30 seconds;
- progress reads are limited to `run_state.json`;
- monitor output uses the honest `durable_timestep` label;
- nonzero exits raise `subprocess.CalledProcessError`;
- the Stage 2 command arguments are unchanged from V2;
- no model, reward, resume, or training source file is modified;
- all notebook cells remain clean and compilable;
- README and notebook pin the immutable V3 URL.
