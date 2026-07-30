# Colab Live Training Logs V3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a V3 six-hour Colab notebook that shows native Stage 2 output and a stable durable-progress line every 30 seconds.

**Architecture:** Keep the exact V2 PPO command and resume selection, but replace the final blocking `subprocess.run` call in the Stage 2 cell with an inherited-stream `subprocess.Popen` polling loop. The loop reads only `run_state.json`, labels persisted progress honestly, preserves nonzero exit semantics, and forwards interactive interruption to the child.

**Tech Stack:** Jupyter Notebook JSON, Python 3.12, `subprocess`, pytest, Git/GitHub.

## Global Constraints

- Do not change `train.py`, model code, rewards, observations, pretraining, checkpoint persistence, or resume selection.
- Preserve every argument and argument order in `ppo_command`.
- Keep child stdout and stderr inherited; do not capture them through a pipe.
- Print monitor state every 30 seconds and label persisted progress `durable_timestep`.
- Keep `scale-aware-cnn-6h-v1` and `scale-aware-cnn-6h-v2` immutable.
- Publish the finished notebook as `scale-aware-cnn-6h-v3`.

---

### Task 1: Lock the V3 Live-Log Contract

**Files:**
- Modify: `AllocRL/test_improved_cnn_notebook.py`
- Test: `AllocRL/test_improved_cnn_notebook.py`

**Interfaces:**
- Consumes: the Stage 2 code cell containing the `ppo_command` assignment.
- Produces: source-level guarantees for inherited streams, polling, progress labels, exit handling, and V3 release references.

- [ ] **Step 1: Add a Stage 2 cell selector**

```python
def stage2_cell() -> str:
    matches = [
        source for source in code_cells()
        if "ppo_command = [" in source
    ]
    assert len(matches) == 1
    return matches[0]
```

- [ ] **Step 2: Write the failing live-log contract test**

```python
def test_stage2_cell_streams_logs_and_reports_durable_progress():
    source = stage2_cell()
    for term in (
        "PPO_LOG_INTERVAL_SECONDS = 30",
        "subprocess.Popen(",
        "stdout=None",
        "stderr=None",
        ".wait(timeout=PPO_LOG_INTERVAL_SECONDS)",
        "except subprocess.TimeoutExpired:",
        "run_state.json",
        "durable_timestep",
        "flush=True",
        "signal.SIGINT",
        "raise subprocess.CalledProcessError",
    ):
        assert term in source
    assert "subprocess.run(command_to_run" not in source
    assert "stdout=subprocess.PIPE" not in source
    assert "stderr=subprocess.PIPE" not in source
```

Update existing notebook and README release assertions from V2 to V3, and
require the final diagnostic heading to identify V3.

- [ ] **Step 3: Run the focused test and confirm failure**

Run:

```powershell
py -3.12 -m pytest -q AllocRL/test_improved_cnn_notebook.py
```

Expected: failures for the V3 tag and absent `Popen` monitor contract.

- [ ] **Step 4: Commit the failing contract**

```powershell
git add AllocRL/test_improved_cnn_notebook.py
git commit -m "test: require Colab live PPO logs"
```

### Task 2: Implement Inherited Output and Durable Monitoring

**Files:**
- Modify: `notebooks/improved_cnn_6h.ipynb`
- Modify: `README.md`
- Test: `AllocRL/test_improved_cnn_notebook.py`

**Interfaces:**
- Consumes: the unchanged `command_to_run`, `ALLOC_RL`, `PPO_ROOT`, and existing unbuffered environment.
- Produces: native child output plus `[colab-monitor]` lines every 30 seconds.

- [ ] **Step 1: Update immutable release references**

Change the notebook checkout tag and README improved-CNN URL to
`scale-aware-cnn-6h-v3`. Update the diagnostic heading text from V2 to V3
without changing diagnostic behavior.

- [ ] **Step 2: Replace only the Stage 2 process wrapper**

Use:

```python
PPO_LOG_INTERVAL_SECONDS = 30
training_process = subprocess.Popen(
    command_to_run,
    cwd=ALLOC_RL,
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
    stdout=None,
    stderr=None,
)
```

Poll with `training_process.wait(timeout=PPO_LOG_INTERVAL_SECONDS)`. On each
timeout, print elapsed seconds and the latest readable fields from
`PPO_ROOT/run_state.json`. Missing or malformed state produces a monitor
message but does not stop the child.

- [ ] **Step 3: Preserve interruption and failure semantics**

On `KeyboardInterrupt`, forward `signal.SIGINT`, wait briefly, then terminate
and kill only if required. Re-raise the interruption. After normal polling,
raise `subprocess.CalledProcessError(return_code, command_to_run)` when the
child return code is nonzero.

- [ ] **Step 4: Run focused tests**

Run:

```powershell
py -3.12 -m pytest -q AllocRL/test_improved_cnn_notebook.py
```

Expected: all tests pass.

- [ ] **Step 5: Compile notebook cells**

Run:

```powershell
py -3.12 -c "import json,pathlib; p=pathlib.Path('notebooks/improved_cnn_6h.ipynb'); n=json.loads(p.read_text(encoding='utf-8')); [compile(''.join(c['source']), f'{p}#cell-{i}', 'exec') for i,c in enumerate(n['cells']) if c['cell_type']=='code']; print('compiled')"
```

Expected: `compiled`.

- [ ] **Step 6: Verify V2 command identity**

Load the V2 notebook from tag `scale-aware-cnn-6h-v2`, extract the single
source line beginning with `ppo_command =`, and assert byte equality with the
V3 working notebook.

- [ ] **Step 7: Commit implementation**

```powershell
git add notebooks/improved_cnn_6h.ipynb README.md
git commit -m "fix: stream Colab PPO training progress"
```

### Task 3: Verify and Publish V3

**Files:**
- Verify: all changed files

**Interfaces:**
- Consumes: the tested feature branch.
- Produces: synchronized local/remote `main` and immutable tag `scale-aware-cnn-6h-v3`.

- [ ] **Step 1: Run relevant regression tests**

```powershell
py -3.12 -m pytest -q AllocRL/test_improved_cnn_notebook.py AllocRL/test_train_resume_cli.py AllocRL/test_parallel_training_config.py
git diff --check origin/main...HEAD
```

- [ ] **Step 2: Review scope**

Confirm that source changes are limited to the notebook, notebook tests,
README, and V3 design/plan documents. Confirm no training Python source is
modified.

- [ ] **Step 3: Publish main and V3**

After confirming that remote `main` has not moved:

```powershell
git push origin HEAD:main
git tag -a scale-aware-cnn-6h-v3 -m "Scale-aware CNN Colab with live PPO logs"
git push origin scale-aware-cnn-6h-v3
```

- [ ] **Step 4: Verify remote refs**

Confirm that remote `main` and peeled V3 tag resolve to the same commit and
that peeled V1/V2 tags retain their original targets.
