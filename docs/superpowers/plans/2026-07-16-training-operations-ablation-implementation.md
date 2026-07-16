# Training Operations and Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make schema-3 training reproducible and resumable, select checkpoints only on fixed holdout data, and run the approved staged A-E ablation with explicit acceptance results.

**Architecture:** A dedicated holdout callback evaluates the same first five fixed scenarios every 50,000 timesteps and owns deterministic best-model comparison. Training configuration becomes a complete compatibility contract, while experiment generation and reporting remain separate command-line tools so policy learning code stays unchanged.

**Tech Stack:** Python 3.12, Stable-Baselines3, sb3-contrib MaskablePPO, NumPy, pandas-free CSV/JSON utilities, pytest/unittest, Google Colab notebook JSON.

## Global Constraints

- Run commands from `D:\Sub\Allocation\CNN-RL\AllocRL` unless a step says otherwise.
- Complete the Evaluation Foundation and State and Geometry Correction plans first.
- Keep MaskablePPO, the ten-workspace action space, coordinate strategy, no-rotation semantics, and schema-3 observations unchanged.
- Keep the existing delay/dropout reward and exact reward conservation; do not add shaping.
- Do not add attention, Pointer Network, or PointNet code.
- Use only the ship-disjoint 673-row training source for synthetic training.
- Use holdout scenario seeds `1000..1004` for periodic selection and `1000..1019` for final reporting.
- Evaluate periodic holdout selection every 50,000 training timesteps.
- Select by higher mean terminal score, then lower dropout rate, then lower mean delay.
- Never select a checkpoint from training-source diagnostic episodes.
- Use budgets: smoke `20,000`/seed `0`; screening `300,000`/seeds `0,1,2`; final `1,000,000`/seeds `0,1,2,3,4`.
- Screening compares `gae_lambda` `0.98` and `0.995` with rollout sizes `512` and `960` at equal budgets and seeds.
- Final runs use one screening-selected `(gae_lambda, n_steps)` pair.
- Use Colab output `/content/drive/MyDrive/CNN-RL-outputs/candidate_cnn_state_v3`.
- Copy CNN diagnostic observation tensors once per rollout, not once per environment step.
- Use `apply_patch` for manual source edits.
- Each task ends with focused tests and a separate commit.

---

## File Structure

- Create `AllocRL/holdout_model_selection.py`: selection metric, deterministic comparator, fixed-holdout callback, and append-only selection CSV.
- Create `AllocRL/test_holdout_model_selection.py`: cadence, scenario limit, tie-break, save, and resume-cadence tests.
- Create `AllocRL/ablation_report.py`: aggregate final A-E results and evaluate all acceptance gates.
- Create `AllocRL/test_ablation_workflow.py`: exact stage commands, hyperparameter matrix, report math, and missing-input failures.
- Modify `AllocRL/train.py`: callback integration, complete run config, newest-timestep resume, selected-checkpoint final report, and CLI.
- Modify `AllocRL/alloc_env/callbacks.py`: one diagnostic observation copy at rollout end.
- Modify `AllocRL/run_ablation.py`: smoke/screening/final command matrix and selected hyperparameters.
- Modify `AllocRL/ABLATION.md`: exact staged commands, result locations, and decision rules.
- Modify `AllocRL/Colab_train.ipynb`: schema-3 output, fixed observation settings, holdout preparation, and million-step final defaults.
- Modify `AllocRL/test_train_resume_cli.py`: complete compatibility keys, unreadable candidate filtering, and greatest-timestep selection.
- Modify `AllocRL/test_cnn_diagnostics.py`: rollout-copy frequency and retained diagnostic metrics.
- Modify `AllocRL/test_parallel_training_config.py`: staged CLI defaults and callback arguments.
- Modify `AllocRL/test_evaluation_scenarios.py`: five-scenario selection and twenty-scenario selected-checkpoint report.
- Modify `AllocRL/test_requirements.py`: notebook and command dependencies remain installable.

---

### Task 1: Add Deterministic Fixed-Holdout Model Selection

**Files:**
- Create: `holdout_model_selection.py`
- Create: `test_holdout_model_selection.py`
- Modify: `evaluation_runner.py`
- Modify: `train.py:582-788`
- Modify: `test_evaluation_scenarios.py`

**Interfaces:**
- Consumes: Stage-A `ModelActionPolicy`, `evaluate_scenarios`, `write_evaluation_metrics`, schema-3 scenario bundle, and an SB3 model.
- Produces: `SelectionMetric`, `is_better_metric(candidate, incumbent) -> bool`, and `FixedHoldoutEvalCallback` saving `best_model.sb3` plus `holdout_selection.csv`.

- [ ] **Step 1: Write failing comparator tests**

```python
def metric(score, dropout, delay):
    return SelectionMetric(
        mean_terminal_score=score,
        mean_dropout_rate=dropout,
        mean_delay_days=delay,
    )

def test_selection_order_is_score_then_dropout_then_delay():
    incumbent = metric(0.50, 0.20, 2.0)
    assert is_better_metric(metric(0.51, 0.90, 9.0), incumbent)
    assert is_better_metric(metric(0.50, 0.19, 9.0), incumbent)
    assert is_better_metric(metric(0.50, 0.20, 1.9), incumbent)
    assert not is_better_metric(metric(0.50, 0.20, 2.0), incumbent)
```

Use exact float tuple comparison; do not round values before selecting.

- [ ] **Step 2: Write failing cadence and scenario-isolation tests**

Create a fake model with `num_timesteps`, `save()`, and a fake evaluator that records scenario seeds:

```python
def test_callback_evaluates_first_five_every_50000_steps(tmp_path):
    callback, calls, model = make_callback(tmp_path, start_timestep=0)
    for timestep in (49_999, 50_000, 99_999, 100_000):
        model.num_timesteps = timestep
        callback._on_step()
    assert calls == [
        [1000, 1001, 1002, 1003, 1004],
        [1000, 1001, 1002, 1003, 1004],
    ]

def test_resume_schedules_next_strict_multiple(tmp_path):
    callback, calls, model = make_callback(tmp_path, start_timestep=120_000)
    callback._on_training_start()
    model.num_timesteps = 149_999
    callback._on_step()
    model.num_timesteps = 150_000
    callback._on_step()
    assert len(calls) == 1
```

Also assert training-environment metrics cannot be passed into the comparator and a better result calls `model.save(output_dir / "best_model.sb3")` exactly once.

- [ ] **Step 3: Run tests and verify callback is absent**

Run: `python -m pytest test_holdout_model_selection.py test_evaluation_scenarios.py -q`

Expected: FAIL because `holdout_model_selection.py` does not exist.

- [ ] **Step 4: Implement immutable selection metrics and aggregation**

```python
@dataclass(frozen=True)
class SelectionMetric:
    mean_terminal_score: float
    mean_dropout_rate: float
    mean_delay_days: float

    @classmethod
    def from_rows(cls, rows: Sequence[Mapping[str, Any]]) -> "SelectionMetric":
        if not rows:
            raise ValueError("holdout selection requires at least one row")
        return cls(
            mean_terminal_score=float(np.mean([
                float(row["mean_terminal_score"]) for row in rows
            ])),
            mean_dropout_rate=float(np.mean([
                float(row["mean_dropout_rate"]) for row in rows
            ])),
            mean_delay_days=float(np.mean([
                float(row["mean_delay_days"]) for row in rows
            ])),
        )

def is_better_metric(
    candidate: SelectionMetric,
    incumbent: Optional[SelectionMetric],
) -> bool:
    if incumbent is None:
        return True
    candidate_key = (
        candidate.mean_terminal_score,
        -candidate.mean_dropout_rate,
        -candidate.mean_delay_days,
    )
    incumbent_key = (
        incumbent.mean_terminal_score,
        -incumbent.mean_dropout_rate,
        -incumbent.mean_delay_days,
    )
    return candidate_key > incumbent_key
```

- [ ] **Step 5: Implement the callback without touching training rewards**

`FixedHoldoutEvalCallback.__init__` receives the already-loaded 20 scenarios, an
`evaluate_fn`, output path, `eval_freq=50_000`, and `selection_count=5`. The
evaluation function has signature
`evaluate_fn(policy_factory, scenarios) -> list[dict]`. Validate seeds equal
`1000..1019`, slice the first five, and initialize `_next_eval_timestep` in
`_on_training_start` as the next strict frequency multiple above
`model.num_timesteps`.

Implement the callback state and persistence as follows:

```python
class FixedHoldoutEvalCallback(BaseCallback):
    CSV_FIELDS = (
        "timestep", "mean_terminal_score", "mean_dropout_rate",
        "mean_delay_days", "is_best",
    )

    def __init__(
        self,
        scenarios: Sequence[dict],
        evaluate_fn: Callable[[Callable, Sequence[dict]], list[dict]],
        output_dir: str | Path,
        eval_freq: int = 50_000,
        selection_count: int = 5,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        seeds = [int(item["seed"]) for item in scenarios]
        if seeds != list(range(1000, 1020)):
            raise ValueError("fixed holdout seeds must be 1000 through 1019")
        if eval_freq <= 0:
            raise ValueError("eval_freq must be positive")
        if selection_count != 5:
            raise ValueError("selection_count must be five")
        self._selection_scenarios = list(scenarios[:selection_count])
        self._evaluate_fn = evaluate_fn
        self._output_dir = Path(output_dir)
        self._eval_freq = int(eval_freq)
        self._next_eval_timestep = self._eval_freq
        self._best_metric: Optional[SelectionMetric] = None

    def _on_training_start(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        current = int(self.model.num_timesteps)
        self._next_eval_timestep = (
            (current // self._eval_freq) + 1
        ) * self._eval_freq
        self._best_metric = read_best_metric(
            self._output_dir / "holdout_selection.csv"
        )

    def _on_step(self) -> bool:
        current = int(self.model.num_timesteps)
        if current < self._next_eval_timestep:
            return True
        rows = self._evaluate_fn(
            lambda _seed: ModelActionPolicy(self.model, name="model"),
            self._selection_scenarios,
        )
        metric = SelectionMetric.from_rows(rows)
        is_best = is_better_metric(metric, self._best_metric)
        if is_best:
            self.model.save(self._output_dir / "best_model.sb3")
            self._best_metric = metric
        append_selection_row(
            self._output_dir / "holdout_selection.csv",
            current,
            metric,
            is_best,
        )
        while self._next_eval_timestep <= current:
            self._next_eval_timestep += self._eval_freq
        return True
```

`append_selection_row` writes `CSV_FIELDS`, writes a header only for a new file,
flushes before return, and serializes `is_best` as `0` or `1`.
`read_best_metric` returns `None` for a missing file, validates the exact header,
and returns the metric from the last row whose `is_best` is `1`; it raises
`ValueError` if a non-empty existing file has no best row. Add a resume test proving
that an existing better metric is not overwritten by the first post-resume eval.

```python
def append_selection_row(
    path: Path,
    timestep: int,
    metric: SelectionMetric,
    is_best: bool,
) -> None:
    new_file = not path.is_file() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FixedHoldoutEvalCallback.CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({
            "timestep": int(timestep),
            "mean_terminal_score": metric.mean_terminal_score,
            "mean_dropout_rate": metric.mean_dropout_rate,
            "mean_delay_days": metric.mean_delay_days,
            "is_best": int(is_best),
        })

def read_best_metric(path: Path) -> Optional[SelectionMetric]:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != FixedHoldoutEvalCallback.CSV_FIELDS:
            raise ValueError("holdout selection CSV header is incompatible")
        best_rows = [row for row in reader if row["is_best"] == "1"]
    if not best_rows:
        raise ValueError("holdout selection CSV has no best row")
    row = best_rows[-1]
    return SelectionMetric(
        mean_terminal_score=float(row["mean_terminal_score"]),
        mean_dropout_rate=float(row["mean_dropout_rate"]),
        mean_delay_days=float(row["mean_delay_days"]),
    )
```

Never call `self.training_env.reset()` from this callback.

- [ ] **Step 6: Integrate selection and selected-checkpoint reporting**

Add CLI arguments:

```python
--holdout-eval-freq       # int, default 50000; 0 disables callback
--holdout-selection-count # int, choices=[5], default 5
--final-holdout-report    # flag; report all 20 from best_model.sb3
```

When `--eval-scenarios` is present and frequency is positive, append the callback. After training, save the final model as before. If `--final-holdout-report` is set, require `best_model.sb3`, load it with the same environment contract, evaluate all 20 fixed scenarios, and write `evaluation_scenarios.csv` with `checkpoint=best_model`. Keep the one-shot `original_csv` evaluation separate.

Bind the shared scenario runner without recomputing normalization:

```python
def run_fixed_holdout(policy_factory, selected_scenarios):
    return evaluate_scenarios(
        policy_factory,
        list(selected_scenarios),
        workspace_codes=active_workspace_codes,
        observation_scales=observation_scales,
        state_context_mode=args.state_context,
    )
```

Pass `run_fixed_holdout` to the callback and reuse it for the all-20 final report.

- [ ] **Step 7: Run callback and evaluation tests**

Run: `python -m pytest test_holdout_model_selection.py test_evaluation_scenarios.py test_parallel_training_config.py -q`

Expected: PASS, with exactly five periodic seeds and twenty final-report seeds.

- [ ] **Step 8: Commit holdout model selection**

```bash
git add AllocRL/holdout_model_selection.py AllocRL/evaluation_runner.py AllocRL/train.py AllocRL/test_holdout_model_selection.py AllocRL/test_evaluation_scenarios.py AllocRL/test_parallel_training_config.py
git commit -m "feat: select models on fixed holdout scenarios"
```

---

### Task 2: Make Run Configuration and Resume Selection Complete

**Files:**
- Modify: `train.py:309-510`
- Modify: `test_train_resume_cli.py`

**Interfaces:**
- Consumes: Stage-A split manifest, Stage-B schema constants, MaskablePPO model archives, and parsed training arguments.
- Produces: complete `run_config.json`, `model_num_timesteps(path, loader) -> int | None`, and `find_resumable_model(output_dir, loader=MaskablePPO.load) -> Path | None` selecting the readable greatest-timestep training state.

- [ ] **Step 1: Write failing complete-config tests**

```python
REQUIRED_COMPATIBILITY_KEYS = {
    "training_data_schema_version",
    "observation_schema_version",
    "reward_schema_version",
    "extractor",
    "features_dim",
    "active_workspace_codes",
    "state_context",
    "grid_size",
    "ordered_future_count",
    "pending_queue_slots",
    "future_day_windows",
    "observation_scales",
    "data_split_seed",
    "source_sha256",
    "episode_block_count",
    "target_month_counts",
    "excluded_start_months",
    "monthly_jitter",
    "empirical_profile_probability",
    "learning_rate",
    "n_steps",
    "batch_size",
    "n_epochs",
    "gamma",
    "gae_lambda",
}

def test_run_config_contains_every_compatibility_key():
    config = current_run_config(
        make_args(), WORKSPACE_CODES, manifest(), full_source_scales()
    )
    assert REQUIRED_COMPATIBILITY_KEYS <= config.keys()

@pytest.mark.parametrize("key", sorted(REQUIRED_COMPATIBILITY_KEYS))
def test_resume_rejects_each_changed_compatibility_value(key):
    saved = complete_config()
    current = complete_config()
    current[key] = different_value_for(key, current[key])
    assert not configs_compatible(saved, current)
```

- [ ] **Step 2: Write failing greatest-timestep and unreadable-file tests**

```python
def test_newer_checkpoint_beats_stale_final_model(tmp_path):
    final = touch_model(tmp_path / "block_placement_ppo.sb3", timesteps=100_000)
    newer = touch_model(tmp_path / "checkpoints" / "model_150000_steps.sb3", timesteps=150_000)
    loader = fake_loader({final: 100_000, newer: 150_000})
    assert find_resumable_model(tmp_path, loader=loader) == newer

def test_unreadable_high_named_checkpoint_is_ignored(tmp_path):
    valid = touch_model(tmp_path / "checkpoints" / "model_100000_steps.sb3", timesteps=100_000)
    broken = touch_model(tmp_path / "checkpoints" / "model_999999_steps.sb3", timesteps=None)
    loader = fake_loader({valid: 100_000}, unreadable={broken})
    assert find_resumable_model(tmp_path, loader=loader) == valid
```

Add a tie test preferring final model over a checkpoint only when both store the same `num_timesteps`.

- [ ] **Step 3: Run resume tests and verify current final-first behavior fails**

Run: `python -m pytest test_train_resume_cli.py -q`

Expected: FAIL because current logic trusts filenames and final-file priority.

- [ ] **Step 4: Record and compare exact compatibility values**

Make `CONFIG_COMPATIBILITY_KEYS` the sorted tuple represented by the test set. Use JSON-safe lists for workspace order and future windows. Name the optimizer field `learning_rate` consistently in config even though CLI uses `args.lr`. Read `data_split_seed` and `source_sha256` from the Stage-A manifest passed to `current_run_config`; fail training if either is absent.

Report mismatches as one line per key:

```python
def config_mismatches(saved: Mapping, current: Mapping) -> dict[str, tuple[Any, Any]]:
    return {
        key: (saved.get(key), current.get(key))
        for key in CONFIG_COMPATIBILITY_KEYS
        if saved.get(key) != current.get(key)
    }
```

- [ ] **Step 5: Select resume archives by stored model state**

```python
def model_num_timesteps(path: Path, loader=MaskablePPO.load) -> Optional[int]:
    try:
        model = loader(str(path), device="cpu")
    except (EOFError, OSError, RuntimeError, ValueError, zipfile.BadZipFile):
        return None
    value = getattr(model, "num_timesteps", None)
    return int(value) if value is not None else None

def find_resumable_model(output_dir, loader=MaskablePPO.load):
    root = Path(output_dir)
    final = [root / MODEL_FILENAME, root / LEGACY_MODEL_FILENAME]
    checkpoints = list((root / "checkpoints").glob("*.sb3"))
    candidates = [path for path in final + checkpoints if path.is_file()]
    ranked = []
    for path in candidates:
        timesteps = model_num_timesteps(path, loader=loader)
        if timesteps is not None:
            ranked.append((
                timesteps,
                int(path in final),
                path.stat().st_mtime_ns,
                path.name,
                path,
            ))
    return max(ranked)[-1] if ranked else None
```

Do not include `best_model.sb3`: it is a model-selection artifact and can be older than the latest trainable state.

- [ ] **Step 6: Run resume and CLI regressions**

Run: `python -m pytest test_train_resume_cli.py test_parallel_training_config.py -q`

Expected: PASS. A schema-2 model, changed source hash, changed rollout length, or changed GAE lambda is rejected before loading into an environment.

- [ ] **Step 7: Commit complete resume compatibility**

```bash
git add AllocRL/train.py AllocRL/test_train_resume_cli.py AllocRL/test_parallel_training_config.py
git commit -m "fix: resume newest compatible training state"
```

---

### Task 3: Reduce CNN Diagnostics to One Observation Copy per Rollout

**Files:**
- Modify: `alloc_env/callbacks.py:244-384`
- Modify: `test_cnn_diagnostics.py`

**Interfaces:**
- Consumes: SB3 `model._last_obs` at rollout end and existing `CnnDiagnosticTracker` update hooks.
- Produces: unchanged diagnostic keys with one deep observation copy in `_on_rollout_end()`.

- [ ] **Step 1: Write failing copy-frequency tests**

```python
def test_diagnostic_observation_is_not_copied_on_steps(monkeypatch):
    callback = prepared_allocation_callback()
    copies = monkeypatch.spy(np, "array")
    for _ in range(100):
        callback.locals = {"new_obs": schema3_numpy_observation(), "dones": [], "infos": []}
        callback._on_step()
    assert diagnostic_copy_calls(copies) == 0

def test_diagnostic_observation_is_copied_once_at_rollout_end(monkeypatch):
    callback = prepared_allocation_callback()
    callback.model._last_obs = schema3_numpy_observation()
    copies = monkeypatch.spy(np, "array")
    callback._on_rollout_end()
    assert diagnostic_copy_calls(copies) == len(callback.model._last_obs)
```

The second count is one array copy per dictionary key, once per rollout, not one copy total for the entire dictionary.

- [ ] **Step 2: Run diagnostics tests and verify per-step copying**

Run: `python -m pytest test_cnn_diagnostics.py -q`

Expected: FAIL because `_on_step` currently deep-copies every new observation.

- [ ] **Step 3: Move capture to rollout end**

Delete the `new_obs` copy block from `_on_step`. Implement:

```python
def _on_rollout_end(self) -> None:
    latest = getattr(self.model, "_last_obs", None)
    if self._diagnostic_tracker is None or not isinstance(latest, dict):
        self._diagnostic_observation = None
        return
    self._diagnostic_observation = {
        key: np.array(value, copy=True)
        for key, value in latest.items()
    }
```

At the next `_on_rollout_start`, record post-update gradient and weight change, convert `_diagnostic_observation` through `policy.obs_to_tensor`, and log workspace feature variance plus candidate-channel sensitivity. Clear the stored observation after measurement.

- [ ] **Step 4: Preserve all diagnostic output fields**

Assert these keys still appear for `CandidateCnnExtractor` and remain absent for non-CNN extractors:

```python
EXPECTED_CNN_DIAGNOSTICS = {
    "cnn_gradient_norm",
    "cnn_weight_change",
    "workspace_feature_variance",
    "candidate_channel_sensitivity",
}
```

- [ ] **Step 5: Run callback and feature tests**

Run: `python -m pytest test_cnn_diagnostics.py test_feature_extractors.py -q`

Expected: PASS.

- [ ] **Step 6: Commit rollout-level diagnostics**

```bash
git add AllocRL/alloc_env/callbacks.py AllocRL/test_cnn_diagnostics.py
git commit -m "perf: sample cnn diagnostics once per rollout"
```

---

### Task 4: Generate Exact Smoke, Screening, and Final Commands

**Files:**
- Modify: `run_ablation.py:1-131`
- Create: `test_ablation_workflow.py`
- Modify: `ABLATION.md`

**Interfaces:**
- Consumes: fixed holdout scenario path and training CLI from Tasks 1-2.
- Produces: `ExperimentSpec`, `HyperparameterSpec`, `build_ablation_commands(stage, seeds, hyperparameters, common_args)` and non-overlapping output directories.

- [ ] **Step 1: Write failing matrix tests**

```python
def test_ablation_rows_match_approved_state_and_grid_matrix():
    assert ABLATIONS == {
        "A": ExperimentSpec("structured", "current"),
        "B": ExperimentSpec("structured", "full"),
        "C": ExperimentSpec("fixed-grid", "full"),
        "D": ExperimentSpec("candidate-cnn", "current"),
        "E": ExperimentSpec("candidate-cnn", "full"),
    }

def test_screening_builds_sixty_equal_budget_commands():
    commands = build_ablation_commands(
        stage="screening",
        seeds=[0, 1, 2],
        hyperparameters=[
            HyperparameterSpec(0.98, 512),
            HyperparameterSpec(0.98, 960),
            HyperparameterSpec(0.995, 512),
            HyperparameterSpec(0.995, 960),
        ],
        common_args=[],
    )
    assert len(commands) == 5 * 3 * 4
    assert all(value_after(cmd, "--timesteps") == "300000" for cmd in commands)
    assert len({value_after(cmd, "--output-dir") for cmd in commands}) == 60
```

Add tests for smoke exactly five commands at 20,000 and seed 0, final exactly 25 commands at 1,000,000 and seeds 0-4, all commands using the same scenario file, reward, split, and `--holdout-eval-freq 50000`. Final must reject zero or multiple hyperparameter pairs.

- [ ] **Step 2: Run workflow tests and verify old 20k/100k mode fails**

Run: `python -m pytest test_ablation_workflow.py test_parallel_training_config.py -q`

Expected: FAIL because current `run_ablation.py` has only screening/final and incorrect budgets/horizons.

- [ ] **Step 3: Define immutable experiment and hyperparameter records**

```python
@dataclass(frozen=True)
class ExperimentSpec:
    extractor: str
    state_context: str

@dataclass(frozen=True)
class HyperparameterSpec:
    gae_lambda: float
    n_steps: int

ABLATIONS = {
    "A": ExperimentSpec("structured", "current"),
    "B": ExperimentSpec("structured", "full"),
    "C": ExperimentSpec("fixed-grid", "full"),
    "D": ExperimentSpec("candidate-cnn", "current"),
    "E": ExperimentSpec("candidate-cnn", "full"),
}

STAGES = {
    "smoke": (20_000, (0,)),
    "screening": (300_000, (0, 1, 2)),
    "final": (1_000_000, (0, 1, 2, 3, 4)),
}
```

- [ ] **Step 4: Build collision-free commands**

Output paths must include stage, row, lambda, rollout, and seed:

```text
output_ablation/screening/E/lambda_0.995/nsteps_960/seed_2
```

Every command includes extractor, state context, total timesteps, seed, lambda, n-steps, fixed scenario path, output directory, `--auto-resume`, `--checkpoint-freq 50000`, `--holdout-eval-freq 50000`, and `--no-export-onnx`. Add `--final-holdout-report` only for final commands.

- [ ] **Step 5: Expose strict stage CLI rules**

Use `--stage {smoke,screening,final}`. Screening always creates all four hyperparameter pairs. Smoke defaults to `(0.98, 960)`. Final requires one `--selected-gae-lambda` in `{0.98,0.995}` and one `--selected-n-steps` in `{512,960}`. Keep `--dry-run` and `--prepare-eval-scenarios`.

- [ ] **Step 6: Document exact invocations**

Add these commands to `ABLATION.md`:

```powershell
py -B run_ablation.py --stage smoke
py -B run_ablation.py --stage screening
py -B run_ablation.py --stage final --selected-gae-lambda 0.995 --selected-n-steps 960
```

State that the final example values are illustrative and must be replaced by the winning screening pair recorded in `screening_selection.json`.

- [ ] **Step 7: Run command-generation tests**

Run: `python -m pytest test_ablation_workflow.py test_parallel_training_config.py -q`

Expected: PASS.

- [ ] **Step 8: Commit staged experiment commands**

```bash
git add AllocRL/run_ablation.py AllocRL/test_ablation_workflow.py AllocRL/ABLATION.md
git commit -m "feat: define staged schema3 ablations"
```

---

### Task 5: Produce Screening Selection and Final Acceptance Reports

**Files:**
- Create: `ablation_report.py`
- Modify: `test_ablation_workflow.py`
- Modify: `ABLATION.md`

**Interfaces:**
- Consumes: per-run holdout CSV files, final CNN diagnostics CSV files, Stage-A greedy baseline CSV, and no-rotation feasibility count.
- Produces: `screening_summary.csv`, `screening_selection.json`, `final_summary.csv`, and `acceptance_report.json`.

- [ ] **Step 1: Write failing screening aggregation tests**

```python
def test_screening_selects_hyperparameters_on_equal_seed_mean(tmp_path):
    write_screening_fixture(tmp_path, gae=0.98, n_steps=512, scores=[0.1, 0.2, 0.3])
    write_screening_fixture(tmp_path, gae=0.995, n_steps=960, scores=[0.4, 0.5, 0.6])
    selection = summarize_screening(tmp_path)
    assert selection["gae_lambda"] == 0.995
    assert selection["n_steps"] == 960
    assert selection["seeds"] == [0, 1, 2]
```

Require all A-E rows and all three seeds for each compared pair; reject incomplete groups instead of averaging unequal samples. Rank pairs by row-E mean score, then row-E dropout and delay.

- [ ] **Step 2: Write failing candidate acceptance tests**

```python
def test_candidate_acceptance_applies_every_gate():
    result = evaluate_acceptance(
        b_by_seed={seed: metrics(score=0.40, dropout=0.20) for seed in range(5)},
        e_by_seed={seed: metrics(score=0.46, dropout=0.17) for seed in range(5)},
        greedy_mean_score=0.45,
        cnn_diagnostics={seed: nonzero_diagnostics() for seed in range(5)},
        globally_infeasible_count=0,
    )
    assert result["aggregate_improvement_gate"] is True
    assert result["four_of_five_direction_gate"] is True
    assert result["greedy_gate"] is True
    assert result["cnn_learning_gate"] is True
    assert result["feasibility_gate"] is True
    assert result["accepted"] is True
```

Add boundary tests for exactly `0.05` score improvement, exactly `10%` relative dropout reduction, three-of-five direction failure, zero baseline dropout, E below greedy, zero gradient/weight change in one seed, and one infeasible source block.

- [ ] **Step 3: Run report tests and verify module is absent**

Run: `python -m pytest test_ablation_workflow.py -q`

Expected: FAIL because `ablation_report.py` does not exist.

- [ ] **Step 4: Implement strict CSV loading and screening selection**

Use `csv.DictReader` and validate required columns before conversion. Group by `(label, gae_lambda, n_steps, seed)`. `summarize_screening` accepts a pair only if every A-E label has seeds `{0,1,2}`. Write one summary row per pair/label and choose a pair using row E's `(score, -dropout, -delay)` key. Include source scenario SHA256 in `screening_selection.json`.

- [ ] **Step 5: Implement final acceptance math exactly**

For B and E aggregate across the same five seeds:

```python
score_gain = mean_e_score - mean_b_score
dropout_reduction = (
    (mean_b_dropout - mean_e_dropout) / mean_b_dropout
    if mean_b_dropout > 0.0
    else 0.0
)
aggregate_gate = score_gain >= 0.05 or dropout_reduction >= 0.10
primary_metric = "score" if score_gain >= 0.05 else "dropout"
direction_count = sum(
    (e[seed].score > b[seed].score)
    if primary_metric == "score"
    else (e[seed].dropout < b[seed].dropout)
    for seed in range(5)
)
```

Require `direction_count >= 4`, E mean score greater than or equal to greedy mean
score, and zero globally infeasible blocks. For every E seed, require at least two
post-update diagnostic rows and require every finite recorded
`cnn_gradient_norm` and `cnn_weight_change` value in those rows to be strictly
positive. `accepted` is `all(gates.values())`. Include raw deltas, per-seed
directions, and per-seed diagnostic minima in JSON.

- [ ] **Step 6: Add report CLI and documentation**

```powershell
py -B ablation_report.py screening --root ./output_ablation/screening
py -B ablation_report.py final --root ./output_ablation/final --baseline-csv ./output_ablation/baselines/evaluation_scenarios.csv --data-dir ./data
```

Document that potential reward shaping is considered only when corrected row E fails both the `0.05` score and `10%` dropout gates; it is not automatically enabled by this report.

- [ ] **Step 7: Run report tests**

Run: `python -m pytest test_ablation_workflow.py -q`

Expected: PASS.

- [ ] **Step 8: Commit deterministic ablation reporting**

```bash
git add AllocRL/ablation_report.py AllocRL/test_ablation_workflow.py AllocRL/ABLATION.md
git commit -m "feat: report cnn ablation acceptance"
```

---

### Task 6: Update the Colab Schema-3 Training Workflow

**Files:**
- Modify: `Colab_train.ipynb`
- Modify: `test_requirements.py`
- Modify: `test_parallel_training_config.py`

**Interfaces:**
- Consumes: Stage-C training and scenario-preparation CLI.
- Produces: a clean Colab notebook using Drive output `candidate_cnn_state_v3`, fixed schema-3 settings, auto-resume, fixed holdout selection, and final reporting.

- [ ] **Step 1: Write failing notebook contract tests**

Load the notebook as JSON and join code-cell sources:

```python
def test_colab_uses_new_schema3_output_and_budget():
    source = notebook_code("Colab_train.ipynb")
    assert 'candidate_cnn_state_v3' in source
    assert 'TIMESTEPS   = 1_000_000' in source
    assert 'N_FUTURE_BLOCKS' not in source
    assert 'HOLDOUT_EVAL_FREQ = 50_000' in source
    assert '--final-holdout-report' in source

def test_colab_prepares_and_uses_fixed_scenarios():
    source = notebook_code("Colab_train.ipynb")
    assert 'run_ablation.py --prepare-eval-scenarios' in source
    assert '--eval-scenarios ./data/fixed_eval_scenarios.json' in source
```

Also assert output cells are empty so stale execution results are not committed.

- [ ] **Step 2: Run notebook tests and verify old output path fails**

Run: `python -m pytest test_requirements.py test_parallel_training_config.py -q`

Expected: FAIL on old `candidate_cnn_10ws_empty_v1`, 100,000 timesteps, or `N_FUTURE_BLOCKS`.

- [ ] **Step 3: Update the configuration cell**

Set these exact values:

```python
EXTRACTOR       = "candidate-cnn"
STATE_CONTEXT   = "full"
FEATURES_DIM    = 256
GRID_SIZE       = 64
TIMESTEPS       = 1_000_000
LR              = 3e-4
N_STEPS         = 960
BATCH_SIZE      = 64
N_EPOCHS        = 10
GAMMA           = 1.0
GAE_LAMBDA      = 0.98
SEED            = 0
HOLDOUT_EVAL_FREQ = 50_000
CHECKPOINT_FREQ = 50_000
OUTPUT_DIR = "/content/drive/MyDrive/CNN-RL-outputs/candidate_cnn_state_v3"
```

Keep the selected rollout and lambda values in one clearly marked configuration cell so they can be replaced after screening. Remove `N_FUTURE_BLOCKS` because schema 3 fixes it at 16.

- [ ] **Step 4: Prepare fixed scenarios and construct the final command**

Add an idempotent scenario-preparation code cell before training:

```python
SCENARIO_PATH = "./data/fixed_eval_scenarios.json"
if not os.path.exists(SCENARIO_PATH):
    !python run_ablation.py --prepare-eval-scenarios --scenario-path {SCENARIO_PATH}
```

The training command must pass state context, all PPO hyperparameters, workspace codes, auto-resume, checkpoint frequency, fixed scenarios, holdout frequency, final report, and optional ONNX export. Do not pass removed future-count or deprecated `--n-eval` options.

- [ ] **Step 5: Update notebook prose and artifact display**

Explain that periodic selection uses seeds 1000-1004 and final output uses all 20. Display `best_model.sb3`, `holdout_selection.csv`, `evaluation_scenarios.csv`, `run_config.json`, final model, and checkpoints. Point placement visualization at `best_model.sb3` after a final run.

- [ ] **Step 6: Clear notebook outputs and validate JSON**

Run: `python -m json.tool Colab_train.ipynb > $null`

Expected: exit code 0.

Run: `python -m pytest test_requirements.py test_parallel_training_config.py -q`

Expected: PASS.

- [ ] **Step 7: Commit the Colab workflow**

```bash
git add AllocRL/Colab_train.ipynb AllocRL/test_requirements.py AllocRL/test_parallel_training_config.py
git commit -m "docs: update colab for schema3 training"
```

---

### Task 7: Run End-to-End Operational Verification

**Files:**
- Modify: `ABLATION.md`
- Test: all `test_*.py`

**Interfaces:**
- Consumes: all Stage A-C features.
- Produces: verified commands and a documented readiness record without launching the full multi-million-step experiment matrix.

- [ ] **Step 1: Compile all executable modules**

Run: `python -m compileall -q alloc_env baseline_policies.py evaluation_runner.py evaluation_scenarios.py evaluate_baselines.py holdout_model_selection.py ablation_report.py run_ablation.py train.py`

Expected: exit code 0.

- [ ] **Step 2: Run the complete test suite**

Run: `python -m pytest -q`

Expected: all tests PASS. Record explicit optional-dependency skips in the command output; no unexpected test is skipped.

- [ ] **Step 3: Verify dry-run command counts and budgets**

Run: `python run_ablation.py --stage smoke --dry-run`

Expected: five commands, all at 20,000 timesteps and seed 0.

Run: `python run_ablation.py --stage screening --dry-run`

Expected: sixty commands, all at 300,000 timesteps, covering three seeds and four hyperparameter pairs.

Run: `python run_ablation.py --stage final --selected-gae-lambda 0.98 --selected-n-steps 960 --dry-run`

Expected: twenty-five commands, all at 1,000,000 timesteps and seeds 0-4, each with `--final-holdout-report`.

- [ ] **Step 4: Run one-scenario baseline smoke**

Run: `python evaluate_baselines.py --scenarios ./data/fixed_eval_scenarios.json --output ./output_ablation/baselines/smoke.csv --limit 1`

Expected: exactly two result rows sharing seed 1000, one random-valid and one greedy-immediate-area.

- [ ] **Step 5: Run one short row-E training smoke**

Run: `python train.py --data-dir ./data --output-dir ./output_stage_c_smoke --timesteps 1024 --extractor candidate-cnn --state-context full --n-steps 512 --batch-size 64 --seed 0 --eval-scenarios ./data/fixed_eval_scenarios.json --holdout-eval-freq 0 --checkpoint-freq 0 --no-export-onnx`

Expected: training exits successfully, writes final SB3 model and complete `run_config.json`, and original CSV evaluation occurs once. No `best_model.sb3` is expected because periodic holdout evaluation is disabled for this short smoke.

- [ ] **Step 6: Verify save/load/evaluate for all extractors**

Run: `python smoke_test.py --all-extractors --timesteps 1024`

Expected: all three extractors save, reload, and complete one evaluation episode; candidate CNN diagnostics contain positive gradient norm and weight change.

- [ ] **Step 7: Record verification and experiment boundary in documentation**

Append the exact completed commands and observed test count to `ABLATION.md`. State that full smoke/screening/final experiments are computational runs, not unit-test prerequisites, and that no reward change is authorized before the final acceptance report.

- [ ] **Step 8: Commit the final operational documentation**

```bash
git add AllocRL/ABLATION.md
git commit -m "docs: record schema3 training verification"
```
