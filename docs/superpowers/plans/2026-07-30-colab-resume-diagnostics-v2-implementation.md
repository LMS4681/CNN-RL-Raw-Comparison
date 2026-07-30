# Colab Resume Diagnostics V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a V2 improved-CNN Colab notebook whose final cell diagnoses interrupted PPO resume artifacts without modifying them.

**Architecture:** Keep all training cells unchanged except for the immutable release tag. Append one self-contained, read-only diagnostic cell that reconstructs Drive paths, validates resume metadata and the referenced SB3 checkpoint, verifies Stage 1 artifacts, and prints a compact status summary. Lock the notebook contract with source-level tests before publishing a new immutable tag.

**Tech Stack:** Jupyter Notebook JSON, Python 3.12, pytest, Stable-Baselines3 ZIP checkpoints, SHA256, Git/GitHub.

## Global Constraints

- Do not change the reward system, training command, checkpoint writer, resume selection, pretraining workflow, or model implementation.
- The diagnostic cell must run independently after Drive mount and must use `device="cpu"` for checkpoint loading.
- The diagnostic cell must not write, repair, rename, replace, or delete any experiment artifact.
- Preserve the immutable `scale-aware-cnn-6h-v1` tag.
- Publish the finished notebook as `scale-aware-cnn-6h-v2`.

---

### Task 1: Lock the Diagnostic Notebook Contract

**Files:**
- Modify: `AllocRL/test_improved_cnn_notebook.py`
- Test: `AllocRL/test_improved_cnn_notebook.py`

**Interfaces:**
- Consumes: notebook JSON loaded by the existing `notebook()` helper.
- Produces: a test contract for the final cell and V2 release references.

- [ ] **Step 1: Write the failing final-cell contract test**

Add a test that selects `notebook()["cells"][-1]`, requires a code cell and
checks for the standalone path constants, state/config reads, checkpoint hash
and ZIP checks, CPU model loading, timestep comparison, Stage 1 verification,
progress tail, and final diagnostic status.

```python
def test_final_cell_is_standalone_read_only_resume_diagnostic():
    final_cell = notebook()["cells"][-1]
    source = "".join(final_cell["source"])
    assert final_cell["cell_type"] == "code"
    for term in (
        'Path("/content/CNN-RL-Raw-Comparison")',
        'Path("/content/drive/MyDrive/CNN-RL-improved/scale-aware-cnn-6h-seed0")',
        "run_state.json",
        "run_config.json",
        "last_checkpoint_file",
        "hashlib.sha256",
        "zipfile.is_zipfile",
        'ScaleAwareMaskablePPO.load(checkpoint_path, device="cpu")',
        "verify_pretraining_artifacts",
        "progress_timing.csv",
        "DIAGNOSTIC_STATUS",
    ):
        assert term in source
```

Also assert that the final cell does not contain `subprocess.run`, training
commands, filesystem delete calls, or text/file write calls. Update existing
release assertions from V1 to V2.

- [ ] **Step 2: Run the focused test and confirm failure**

Run:

```powershell
py -3.12 -m pytest -q AllocRL/test_improved_cnn_notebook.py
```

Expected: failure because the current final cell is only an artifact display
cell and the notebook still pins V1.

- [ ] **Step 3: Commit the failing contract**

```powershell
git add AllocRL/test_improved_cnn_notebook.py
git commit -m "test: require Colab resume diagnostics"
```

### Task 2: Add the Standalone Read-Only Diagnostic Cell

**Files:**
- Modify: `notebooks/improved_cnn_6h.ipynb`
- Modify: `README.md`
- Test: `AllocRL/test_improved_cnn_notebook.py`

**Interfaces:**
- Consumes: Drive artifacts beneath
  `/content/drive/MyDrive/CNN-RL-improved/scale-aware-cnn-6h-seed0`.
- Produces: compact notebook output ending in
  `DIAGNOSTIC_STATUS=PASS|WARN|FAIL`.

- [ ] **Step 1: Change the notebook release pin**

Change only:

```python
RELEASE_TAG = "scale-aware-cnn-6h-v2"
```

Do not change the Stage 1 or Stage 2 commands.

- [ ] **Step 2: Append the diagnostic cell**

The cell must initialize its own `Path` values, read JSON with exception
handling, locate the state-referenced checkpoint, compute SHA256 in chunks,
validate the ZIP, load the policy on CPU, compare timesteps, call
`verify_pretraining_artifacts`, show up to five progress rows, and aggregate
issues into a concise final status. Every artifact operation must be a read.

- [ ] **Step 3: Update the README release link**

Replace the improved notebook URL and release note with
`scale-aware-cnn-6h-v2`, while retaining no claim that V1 moved.

- [ ] **Step 4: Run the focused notebook tests**

Run:

```powershell
py -3.12 -m pytest -q AllocRL/test_improved_cnn_notebook.py
```

Expected: all tests pass.

- [ ] **Step 5: Compile every notebook code cell**

Run:

```powershell
py -3.12 -c "import json,pathlib; p=pathlib.Path('notebooks/improved_cnn_6h.ipynb'); n=json.loads(p.read_text(encoding='utf-8')); [compile(''.join(c['source']), f'{p}#cell-{i}', 'exec') for i,c in enumerate(n['cells']) if c['cell_type']=='code']; print('compiled')"
```

Expected: `compiled`.

- [ ] **Step 6: Commit the implementation**

```powershell
git add notebooks/improved_cnn_6h.ipynb README.md
git commit -m "fix: add Colab resume diagnostics"
```

### Task 3: Verify and Publish V2

**Files:**
- Verify: all changed files

**Interfaces:**
- Consumes: the tested feature branch commit.
- Produces: `main` and annotated tag `scale-aware-cnn-6h-v2` on GitHub.

- [ ] **Step 1: Run regression checks**

```powershell
py -3.12 -m pytest -q AllocRL/test_improved_cnn_notebook.py AllocRL/test_pretraining.py
git diff --check origin/main...HEAD
git status --short
```

Expected: tests pass, no whitespace errors, and only intended commits exist.

- [ ] **Step 2: Review the final diff**

Confirm that the Stage 2 `ppo_command` is byte-for-byte unchanged except for
surrounding notebook JSON offsets, and that the final cell contains no write
or training operation.

- [ ] **Step 3: Publish main**

```powershell
git push origin HEAD:main
```

- [ ] **Step 4: Create and publish the immutable tag**

```powershell
git tag -a scale-aware-cnn-6h-v2 -m "Scale-aware CNN six-hour Colab with resume diagnostics"
git push origin scale-aware-cnn-6h-v2
```

- [ ] **Step 5: Verify remote refs**

Confirm that `refs/heads/main` and `refs/tags/scale-aware-cnn-6h-v2^{}` point
to the implementation commit, while `scale-aware-cnn-6h-v1` still points to
its original commit.

