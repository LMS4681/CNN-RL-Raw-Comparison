# Overnight Raw Observation vs Candidate CNN Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and publish a reproducible comparison repository whose single Colab notebook trains `raw-direct/full` and `candidate-cnn/full` sequentially for about three active-training hours each, saves durable Drive artifacts, and generates an honest Korean preliminary report.

**Architecture:** Branch from the approved `CNN-RL` snapshot in an isolated worktree and add an `AllocRL/comparison` package. The raw arm uses a parameter-free deterministic flatten extractor while both arms share an explicit PPO actor/critic MLP contract. A verified-generation wall-clock callback, idempotent subprocess orchestrator, checkpoint evaluator, and report builder make one Colab notebook resumable and self-reporting without changing allocation, reward, split, or holdout semantics.

**Tech Stack:** Python 3.12, PyTorch, Gymnasium 1.3.0, Stable-Baselines3 2.9.0, sb3-contrib 2.9.0, pytest, Matplotlib, Jupyter/Google Colab, Git/GitHub, Google Drive.

**Design source:** `docs/superpowers/specs/2026-07-21-raw-observation-cnn-comparison-design.md`

## Global Constraints

- The baseline implementation is `https://github.com/LMS4681/CNN-RL.git` commit `cd4e14fc1725a4ff159e59d6874d3602f3b65a06` and observation schema 3.
- The immutable input hashes are fixed scenarios `6125f53939a1b8eef8662b2628c0da2f1d0f26b5b541a99252858326b38cd814` and split manifest `d3df1d0076248b4bcbddb4c910a3cb81481da65c7415c6b3cacf9e055cc3f9df`.
- Publish a separate public repository named `LMS4681/CNN-RL-Raw-Comparison`; do not push comparison implementation commits to the original repository's `main`.
- Preserve the 913-block episode, deterministic ship-disjoint split, fixed ten-workspace order, action mask, no-rotation semantics, reward conservation, normalization scales, and fixed scenario bundle.
- Compare `raw-direct/full` against the baseline candidate-CNN feature extractor with its SB3 2.9 policy defaults made explicit, using seed `0`, one environment, and the same Colab VM/GPU.
- Raw-direct excludes `grids`, applies future/pending masks, and concatenates normalized non-grid arrays in this exact order: `block`, `ws_meta`, `future_blocks`, `future_mask`, `future_demand`, `pending_blocks`, `pending_mask`, `pending_summary`.
- Preserve the shared schema-3 environment, so raw still pays grid-construction cost even though its extractor never reads grids; disclose this throughput limitation.
- Raw-direct has no `Conv2d` or learned feature-extractor `Linear`; its exact output dimension is `2772`.
- Both arms explicitly use `net_arch={"pi": [64, 64], "vf": [64, 64]}`, ReLU, and a shared feature extractor; only the extractor/input width differs.
- Each arm targets `10_800` recorded training-subprocess wall seconds, including PPO, selection and checkpoint I/O. Persist at `10_000` timesteps or `300` wall seconds, whichever comes first; use timestep ceiling `2_000_000_000` only as a fail-safe.
- Use selection scenarios `1000..1004`; use scenarios `1005..1019` as the primary unselected final test and all 20 only as a secondary result.
- Use PPO settings `learning_rate=3e-4`, `n_steps=960`, `batch_size=64`, `n_epochs=10`, `gamma=1.0`, `gae_lambda=0.98`.
- Run raw-direct first, then candidate-CNN, after both 1,024-step smoke checks pass.
- Store generated models, checkpoints, TensorBoard events, and full logs only in Drive; Git stores code, tests, notebook, small manifests, and reports.
- Resume only the exact filename/timestep/SHA256/config SHA referenced by `run_state.json`; never use broad `--auto-resume` for the overnight arms.
- All non-Git shell and pytest commands below run from `CNN-RL-Raw-Comparison/AllocRL`; Git commands run from the repository root.
- A single-seed result is labeled preliminary. Never claim statistical significance or general CNN superiority.
- Use TDD for every code task, `apply_patch` for manual source edits, a focused test run per task, one full regression run before each task commit, and an independent review gate per task.

---

## File Structure

- Create `AllocRL/comparison/__init__.py`: stable public imports for comparison components.
- Create `AllocRL/comparison/raw_direct_extractor.py`: parameter-free masked raw observation concatenation.
- Create `AllocRL/comparison/wall_clock_callback.py`: cumulative wall-clock budget and verified versioned checkpoint/state persistence.
- Create `AllocRL/comparison/artifact_manifest.py`: environment, parameter, timing, checksum, and completion metadata.
- Create `AllocRL/comparison/checkpoint_evaluator.py`: selected/common checkpoint evaluation and validation/test split output.
- Create `AllocRL/comparison/report_builder.py`: paired summaries, plots, Korean complete/partial report generation.
- Create `AllocRL/comparison/experiment_runner.py`: smoke, raw, CNN, evaluation, and reporting orchestration.
- Create `AllocRL/configs/overnight_seed0.json`: immutable overnight experiment contract.
- Modify `.gitignore`: whitelist the root README, provenance, notebook, report, and plan/spec paths while retaining generated-artifact ignores.
- Create `notebooks/overnight_compare.ipynb`: Colab Pro GPU preflight, Drive mount, clone/tag checkout, and one-command run.
- Create `AllocRL/test_raw_direct_extractor.py`: extractor order, masks, grid exclusion, parameterlessness, policy contract.
- Create `AllocRL/test_wall_clock_callback.py`: fake-clock stop, exact resume, verified state generations, checkpoint recovery.
- Create `AllocRL/test_artifact_manifest.py`: deterministic environment/parameter/checksum metadata.
- Create `AllocRL/test_checkpoint_evaluator.py`: common-step selection and `1000..1004`/`1005..1019` separation.
- Create `AllocRL/test_comparison_report.py`: exact summaries, honest missing-data behavior, Korean report text and plots.
- Create `AllocRL/test_comparison_experiment.py`: command equality, order, smoke gate, resume, complete/partial markers.
- Create `AllocRL/test_comparison_notebook.py`: valid clean notebook, pinned tag, GPU/Drive preflight, one runner invocation.
- Create `AllocRL/requirements-comparison.in` and `AllocRL/requirements-comparison.txt`: exact direct inputs and hashed non-GPU comparison dependency lock.
- Create `UPSTREAM_BASELINE.md`: baseline repository/SHA and experiment boundary.
- Create `reports/README.md`: generated report/artifact publication rules.
- Modify `AllocRL/train.py`: add raw extractor, explicit policy network, wall-clock CLI/callback, runtime metadata and fallback evaluation.
- Modify `AllocRL/test_parallel_training_config.py`: extractor/CLI/policy compatibility coverage.
- Modify `AllocRL/test_train_resume_cli.py`: comparison policy contract compatibility and wall-clock resume coverage.
- Create `README.md`: point to the overnight notebook, Drive output contract, and preliminary-result limitations.
- Modify `AllocRL/smoke_test.py` and `AllocRL/test_smoke_workflows.py`: add the `raw-direct` save/load/evaluate smoke path.

---

### Task 0: Isolate the Comparison Branch and Expose Root Artifacts

**Files:**
- Modify: `.gitignore`

**Interfaces:**
- Consumes: the approved documentation commit whose `AllocRL/` tree still matches baseline `cd4e14f`.
- Produces: an isolated comparison worktree/branch where root notebook, report, provenance, and documentation files can be tracked without exposing generated training artifacts.

- [ ] **Step 1: Enter the worktree created by `superpowers:using-git-worktrees` and verify isolation**

Run from the repository root:

```powershell
git branch --show-current
git diff --exit-code cd4e14fc1725a4ff159e59d6874d3602f3b65a06 -- AllocRL
```

Expected: the branch is not `main`; the second command exits 0 before implementation starts. If `AllocRL` differs, stop and inspect rather than discarding changes.

- [ ] **Step 2: Prove root publication paths are currently ignored**

Run: `git check-ignore -q README.md`

Expected: exit 0.

- [ ] **Step 3: Whitelist the exact publication paths**

Apply this block immediately after `!/AllocRL/**` in `.gitignore`:

```gitignore
!/README.md
!/UPSTREAM_BASELINE.md
!/notebooks/
!/notebooks/**
!/reports/
!/reports/**
!/docs/
!/docs/**
```

- [ ] **Step 4: Verify source paths are visible and generated artifacts remain ignored**

Run:

```powershell
git check-ignore -q README.md; if ($LASTEXITCODE -ne 1) { exit 1 }
git check-ignore -q AllocRL/output/probe.sb3; if ($LASTEXITCODE -ne 0) { exit 1 }
```

Expected: exit 0.

- [ ] **Step 5: Commit**

```powershell
git add .gitignore
git commit -m "chore: expose comparison publication files"
```

---

### Task 1: Add the Parameter-Free Raw-Direct Policy Contract

**Files:**
- Create: `AllocRL/comparison/__init__.py`
- Create: `AllocRL/comparison/raw_direct_extractor.py`
- Create: `AllocRL/test_raw_direct_extractor.py`
- Modify: `AllocRL/train.py:127-152,395-445,1326-1435`
- Modify: `AllocRL/test_parallel_training_config.py`
- Modify: `AllocRL/test_train_resume_cli.py`
- Modify: `AllocRL/smoke_test.py`
- Modify: `AllocRL/test_smoke_workflows.py`

**Interfaces:**
- Consumes: schema-3 `gym.spaces.Dict` and observation tensors.
- Produces: `RAW_DIRECT_KEYS`, `RAW_DIRECT_FEATURE_DIM = 2772`, `RawDirectExtractor`, `explicit_policy_net_arch()`, and the `--extractor raw-direct` training option.

- [ ] **Step 1: Write failing raw-vector contract tests**

```python
from test_feature_extractors import clone_observation, observation, observation_space


def expected_raw_concat(values):
    future_mask = values["future_mask"]
    pending_mask = values["pending_mask"]
    return torch.cat(
        (
            values["block"],
            values["ws_meta"].flatten(1),
            (values["future_blocks"] * future_mask.unsqueeze(-1)).flatten(1),
            future_mask.flatten(1),
            values["future_demand"].flatten(1),
            (values["pending_blocks"] * pending_mask.unsqueeze(-1)).flatten(1),
            pending_mask.flatten(1),
            values["pending_summary"].flatten(1),
        ),
        dim=1,
    )


def test_raw_direct_feature_dimension_and_order():
    values = observation(batch_size=1)
    extractor = RawDirectExtractor(observation_space())
    output = extractor(values)
    assert extractor.features_dim == 2772
    assert output.shape == (1, 2772)
    torch.testing.assert_close(output, expected_raw_concat(values))


def test_raw_direct_masks_invalid_slots():
    changed = observation(batch_size=1)
    changed["future_mask"][0, 3] = 0
    changed["future_blocks"][0, 3] = 1
    changed["pending_mask"][0, 2, 7] = 0
    changed["pending_blocks"][0, 2, 7] = 1
    expected = clone_observation(changed)
    expected["future_blocks"][0, 3] = 0
    expected["pending_blocks"][0, 2, 7] = 0
    extractor = RawDirectExtractor(observation_space())
    torch.testing.assert_close(extractor(changed), extractor(expected))


def test_raw_direct_ignores_grids_and_has_no_learned_layers():
    values = observation(batch_size=1)
    extractor = RawDirectExtractor(observation_space())
    changed = clone_observation(values)
    changed["grids"].fill_(1)
    torch.testing.assert_close(extractor(values), extractor(changed))
    assert not any(isinstance(m, (nn.Conv2d, nn.Linear)) for m in extractor.modules())
    assert sum(p.numel() for p in extractor.parameters()) == 0
```

- [ ] **Step 2: Write failing policy-contract tests**

```python
@pytest.mark.parametrize("extractor", ["raw-direct", "candidate-cnn"])
def test_comparison_arms_share_explicit_policy_mlp(extractor):
    kwargs = build_policy_kwargs(extractor=extractor, features_dim=256)
    assert kwargs["net_arch"] == {"pi": [64, 64], "vf": [64, 64]}
    assert kwargs["activation_fn"] is nn.ReLU
    assert kwargs["share_features_extractor"] is True


def test_raw_direct_is_a_cli_choice():
    captured = {}

    def fake_train(args):
        captured["extractor"] = args.extractor

    with patch.object(sys, "argv", ["train.py", "--extractor", "raw-direct"]), \
         patch.object(train_module, "train", fake_train):
        train_module.main()
    assert captured["extractor"] == "raw-direct"


def test_raw_direct_is_in_real_smoke_matrix():
    assert smoke_test.EXTRACTORS == (
        "structured", "fixed-grid", "candidate-cnn", "raw-direct"
    )


def test_smoke_cli_accepts_explicit_cuda(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        smoke_test, "_run_selected_extractors",
        lambda extractors, output_dir, timesteps, device: captured.update(
            extractors=extractors, device=device
        ),
    )
    assert smoke_test.main([
        "--extractor", "raw-direct", "--timesteps", "1024",
        "--device", "cuda", "--output-dir", str(tmp_path),
    ]) == 0
    assert captured == {"extractors": ("raw-direct",), "device": "cuda"}
```

- [ ] **Step 3: Run tests and verify RED**

Run: `python -m pytest test_raw_direct_extractor.py test_parallel_training_config.py test_train_resume_cli.py test_smoke_workflows.py -q`

Expected: FAIL because `comparison.raw_direct_extractor`, the CLI choice, and explicit net architecture do not exist.

- [ ] **Step 4: Implement the exact parameter-free extractor**

```python
RAW_DIRECT_KEYS = (
    "block",
    "ws_meta",
    "future_blocks",
    "future_mask",
    "future_demand",
    "pending_blocks",
    "pending_mask",
    "pending_summary",
)
RAW_DIRECT_FEATURE_DIM = 2772


class RawDirectExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        validate_observation_space(observation_space)
        _ = features_dim  # accepted for the shared SB3 extractor constructor contract
        super().__init__(observation_space, RAW_DIRECT_FEATURE_DIM)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        future_mask = observations["future_mask"].to(
            dtype=observations["future_blocks"].dtype
        )
        pending_mask = observations["pending_mask"].to(
            dtype=observations["pending_blocks"].dtype
        )
        parts = (
            observations["block"],
            observations["ws_meta"].flatten(1),
            (observations["future_blocks"] * future_mask.unsqueeze(-1)).flatten(1),
            future_mask.flatten(1),
            observations["future_demand"].flatten(1),
            (observations["pending_blocks"] * pending_mask.unsqueeze(-1)).flatten(1),
            pending_mask.flatten(1),
            observations["pending_summary"].flatten(1),
        )
        output = torch.cat(parts, dim=1)
        if output.shape[1] != RAW_DIRECT_FEATURE_DIM:
            raise RuntimeError(
                f"raw-direct feature width must be {RAW_DIRECT_FEATURE_DIM}, "
                f"got {output.shape[1]}"
            )
        return output
```

Import `validate_observation_space` from `alloc_env.cnn_extractor`. Export `RAW_DIRECT_KEYS`, `RAW_DIRECT_FEATURE_DIM`, and `RawDirectExtractor` from `comparison/__init__.py`; there are no additional symbols in that file.

- [ ] **Step 5: Register the extractor and pin both policy MLPs**

```python
def explicit_policy_net_arch() -> dict[str, list[int]]:
    return {"pi": [64, 64], "vf": [64, 64]}


return {
    "features_extractor_class": extractors[extractor],
    "features_extractor_kwargs": {"features_dim": features_dim},
    "share_features_extractor": True,
    "net_arch": explicit_policy_net_arch(),
    "activation_fn": torch.nn.ReLU,
}
```

Add `raw-direct` to the extractor map and parser choices. Record `policy_net_arch`, `policy_activation`, and `extractor_output_dim` in `run_config.json` and add all three exact names to `CONFIG_COMPATIBILITY_KEYS` and its expected-key test. Resolve output dimension as `2772` for raw-direct and `args.features_dim` for existing extractors. Add `raw-direct` as the fourth `smoke_test.EXTRACTORS` entry. Add smoke CLI `--device` with choices `cpu,cuda` and default `cpu`, pass it through `_run_selected_extractors(extractors, output_dir, timesteps, device)` and `train_tiny_model(extractor=extractor, timesteps=timesteps, device=device)`, and replace the hard-coded `device="cpu"` in the smoke PPO constructor with that argument.

- [ ] **Step 6: Run focused and full tests**

Run: `python -m pytest test_raw_direct_extractor.py test_parallel_training_config.py test_train_resume_cli.py test_smoke_workflows.py -q`

Expected: PASS.

Run: `python -m pytest -q`

Expected: all existing tests plus new tests PASS; dependency deprecation warnings may remain but no test fails.

- [ ] **Step 7: Commit**

```bash
git add AllocRL/comparison/__init__.py AllocRL/comparison/raw_direct_extractor.py AllocRL/test_raw_direct_extractor.py AllocRL/train.py AllocRL/test_parallel_training_config.py AllocRL/test_train_resume_cli.py AllocRL/smoke_test.py AllocRL/test_smoke_workflows.py
git commit -m "feat: add raw-direct comparison policy"
```

---

### Task 2: Add Durable Three-Hour Wall-Clock Training

**Files:**
- Create: `AllocRL/comparison/wall_clock_callback.py`
- Create: `AllocRL/test_wall_clock_callback.py`
- Modify: `AllocRL/train.py:62-70,395-445,896-1180,1326-1435`
- Modify: `AllocRL/test_train_resume_cli.py`

**Interfaces:**
- Consumes: SB3 model `num_timesteps`, output directory, monotonic/UTC clocks, `target_seconds`, `checkpoint_freq`, `heartbeat_seconds`, and canonical experiment-config SHA256.
- Produces: `WallClockState`, `read_wall_clock_state(path)`, `resolve_state_checkpoint(output_dir, state)`, `WallClockBudgetCallback`, `run_state.json`, `progress_timing.csv`, and CLI flags `--max-training-seconds`, `--wall-clock-state`, `--wall-clock-heartbeat-seconds`, and `--comparison-config-sha256`.

- [ ] **Step 1: Write fake-clock RED tests**

```python
def test_wall_clock_stops_at_cumulative_budget(tmp_path, fake_clock):
    callback, model = prepared_callback(
        tmp_path, fake_clock, target_seconds=10_800, checkpoint_freq=10_000
    )
    callback._on_training_start()
    fake_clock.advance(10_799)
    assert callback._on_step() is True
    fake_clock.advance(1)
    assert callback._on_step() is False
    state = read_wall_clock_state(tmp_path / "run_state.json")
    assert state.status == "complete"
    assert state.completed_training_seconds == pytest.approx(10_800)
    assert state.last_checkpoint_timestep == model.num_timesteps
    assert state.last_checkpoint_file
    assert len(state.last_checkpoint_sha256) == 64
    assert state.config_sha256 == TEST_CONFIG_SHA256


def test_resume_uses_only_remaining_budget(tmp_path, fake_clock):
    first, model = prepared_callback(tmp_path, fake_clock, target_seconds=10_800)
    first._on_training_start()
    model.num_timesteps = 120_000
    fake_clock.advance(7_200)
    first.persist_checkpoint(status="running")

    callback, _ = prepared_callback(
        tmp_path, fake_clock, target_seconds=10_800, model=model
    )
    callback._on_training_start()
    fake_clock.advance(3_599)
    assert callback._on_step() is True
    fake_clock.advance(1)
    assert callback._on_step() is False


def test_state_never_advances_past_readable_checkpoint(tmp_path, fake_clock):
    callback, model = prepared_callback(tmp_path, fake_clock)
    model.save_raises = OSError("drive unavailable")
    with pytest.raises(OSError, match="drive unavailable"):
        callback.persist_checkpoint(status="running")
    assert not (tmp_path / "run_state.json").exists()


def test_heartbeat_persists_before_timestep_interval(tmp_path, fake_clock):
    callback, model = prepared_callback(
        tmp_path,
        fake_clock,
        target_seconds=10_800,
        checkpoint_freq=10_000,
        heartbeat_seconds=300,
    )
    callback._on_training_start()
    model.num_timesteps = 17
    fake_clock.advance(300)
    assert callback._on_step() is True
    state = read_wall_clock_state(tmp_path / "run_state.json")
    assert state.last_checkpoint_timestep == 17
    assert state.last_regular_checkpoint_timestep == 0
    assert state.completed_training_seconds == pytest.approx(300)


def test_resume_rejects_model_not_named_by_state(tmp_path, fake_clock):
    first, model = prepared_callback(tmp_path, fake_clock)
    first._on_training_start()
    model.num_timesteps = 100
    first.persist_checkpoint(status="running")
    model.num_timesteps = 200
    resumed, _ = prepared_callback(tmp_path, fake_clock, model=model)
    with pytest.raises(ValueError, match="state checkpoint timestep 100.*model 200"):
        resumed._on_training_start()


def test_archive_verified_before_state_crash_keeps_prior_generation(
    tmp_path, fake_clock, monkeypatch
):
    callback, model = prepared_callback(tmp_path, fake_clock)
    callback._on_training_start()
    model.num_timesteps = 100
    callback.persist_checkpoint(status="running")
    prior = read_wall_clock_state(tmp_path / "run_state.json")

    model.num_timesteps = 200
    monkeypatch.setattr(
        wall_clock_module,
        "atomic_write_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("state write failed")),
    )
    with pytest.raises(OSError, match="state write failed"):
        callback.persist_checkpoint(status="running")

    current = read_wall_clock_state(tmp_path / "run_state.json")
    assert current == prior
    assert resolve_state_checkpoint(tmp_path, current).name == prior.last_checkpoint_file
    assert any("_g2.sb3" in path.name for path in (tmp_path / "checkpoints").iterdir())


def test_selection_elapsed_time_is_charged_to_same_budget(tmp_path, fake_clock):
    callback, _ = prepared_callback(tmp_path, fake_clock, target_seconds=10_800)
    callback._on_training_start()
    fake_clock.advance(10_800)  # fake selection callback ran before wall-clock callback
    assert callback._on_step() is False
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest test_wall_clock_callback.py test_train_resume_cli.py -q`

Expected: FAIL because the callback, state schema, and CLI do not exist.

- [ ] **Step 3: Implement immutable state and verified JSON replacement**

```python
@dataclass(frozen=True)
class WallClockState:
    schema_version: int
    target_training_seconds: float
    completed_training_seconds: float
    last_checkpoint_timestep: int
    last_regular_checkpoint_timestep: int
    last_checkpoint_file: str
    last_checkpoint_sha256: str
    config_sha256: str
    generation: int
    restart_count: int
    max_unrecorded_seconds: float
    status: Literal["running", "complete"]
    started_at_utc: str
    updated_at_utc: str
    completed_at_utc: str | None


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
        json.loads(path.read_text(encoding="utf-8"))
    finally:
        temporary.unlink(missing_ok=True)
```

Validate exact keys, `schema_version == 1`, finite non-negative seconds, non-negative timestep/generation/restart count, 64-character lowercase SHA256 values, a basename-only checkpoint filename, status in `{"running", "complete"}`, and unchanged target/config SHA on resume. This replacement protects against torn JSON observed by the same mount; it is not documented as a Google Drive crash-durability guarantee.

- [ ] **Step 4: Implement checkpoint-before-state persistence**

Give `WallClockBudgetCallback` this constructor so every dependency used by tests is explicit:

```python
def __init__(
    self,
    output_dir: str | Path,
    *,
    target_seconds: float,
    checkpoint_freq: int,
    heartbeat_seconds: float,
    config_sha256: str,
    monotonic: Callable[[], float] = time.monotonic,
    utc_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    archive_timestep_reader: Callable[[Path], int | None] = model_num_timesteps,
) -> None:
    super().__init__(verbose=0)
    if target_seconds <= 0 or checkpoint_freq <= 0 or heartbeat_seconds <= 0:
        raise ValueError("wall-clock limits and persistence intervals must be positive")
    if not re.fullmatch(r"[0-9a-f]{64}", config_sha256):
        raise ValueError("comparison config SHA256 must be lowercase hexadecimal")
    self._output_dir = Path(output_dir)
    self._state_path = self._output_dir / "run_state.json"
    self._target_seconds = float(target_seconds)
    self._checkpoint_freq = int(checkpoint_freq)
    self._heartbeat_seconds = float(heartbeat_seconds)
    self._config_sha256 = config_sha256
    self._monotonic = monotonic
    self._utc_now = utc_now
    self._archive_timestep_reader = archive_timestep_reader
    self._completed_before_segment = 0.0
    self._last_checkpoint_timestep = 0
    self._last_regular_checkpoint_timestep = 0
    self._last_persisted_monotonic = 0.0
    self._segment_started = 0.0
```

`WallClockBudgetCallback._on_step()` computes:

```python
elapsed = self._completed_before_segment + (
    self._monotonic() - self._segment_started
)
current_timestep = int(self.model.num_timesteps)
current_regular_boundary = (
    current_timestep // self._checkpoint_freq
) * self._checkpoint_freq
should_checkpoint = (
    current_regular_boundary > self._last_regular_checkpoint_timestep
)
should_heartbeat = (
    self._monotonic() - self._last_persisted_monotonic
    >= self._heartbeat_seconds
)
should_stop = elapsed >= self._target_seconds
```

On an absolute regular boundary, heartbeat, or stop, save locally to a unique temporary directory, flush the archive, copy it as `checkpoints/model_<timestep>_g<generation>.sb3.partial`, rename it to the never-overwritten final generation name, reopen it with `model_num_timesteps`, and verify its SHA256. Heartbeats do not move `last_regular_checkpoint_timestep`; crossing `10_000`, `20_000`, and later absolute boundaries does. Recompute elapsed after checkpoint I/O, append `generation,timestep,recorded_training_seconds,updated_at_utc,status,checkpoint_file` to `progress_timing.csv`, then write state. Only after a verified stop generation and reread state agree may `_on_step()` return `False`. Never create both this callback and `Sb3CheckpointCallback` in a wall-clock run.

- [ ] **Step 5: Integrate training CLI without changing non-comparison defaults**

```python
parser.add_argument("--max-training-seconds", type=float, default=0.0)
parser.add_argument("--wall-clock-state", default=None)
parser.add_argument("--wall-clock-heartbeat-seconds", type=float, default=300.0)
parser.add_argument("--comparison-config-sha256", default=None)
```

When the limit is positive, require a 64-character config SHA, use `output_dir / "run_state.json"` unless explicitly provided, skip the ordinary `Sb3CheckpointCallback`, and append `WallClockBudgetCallback` last so allocation diagnostics and holdout-selection time are charged. A zero limit preserves current callback behavior. After `learn()` returns, require `run_state.status == "complete"`; if the timestep ceiling returned first, raise and leave the arm incomplete. Save the conventional final model only after that gate. Generalized selected/fallback evaluation is implemented in Task 4.

- [ ] **Step 6: Run focused and full tests**

Run: `python -m pytest test_wall_clock_callback.py test_train_resume_cli.py test_holdout_model_selection.py -q`

Expected: PASS.

Run: `python -m pytest -q`

Expected: all tests PASS; dependency deprecation warnings may remain.

- [ ] **Step 7: Commit**

```bash
git add AllocRL/comparison/wall_clock_callback.py AllocRL/test_wall_clock_callback.py AllocRL/train.py AllocRL/test_train_resume_cli.py
git commit -m "feat: persist wall-clock training budgets"
```

---

### Task 3: Record Reproducible Runtime and Artifact Metadata

**Files:**
- Create: `AllocRL/comparison/artifact_manifest.py`
- Create: `AllocRL/test_artifact_manifest.py`
- Modify: `AllocRL/train.py:1039-1115`

**Interfaces:**
- Consumes: model, root/arm output directories, canonical experiment config, start/end timestamps, fixed input paths, Git SHA, dependency-lock SHA, and optional CUDA runtime.
- Produces: `collect_environment(command: Sequence[str], provenance: Mapping[str, str]) -> dict[str, Any]`, `count_trainable_parameters(model) -> dict[str, int]`, `sha256_file(path: str | Path) -> str`, `canonical_json_sha256(payload: Mapping[str, Any]) -> str`, `append_environment_segment(path: str | Path, payload: Mapping[str, Any]) -> None`, `write_runtime_metrics(path: str | Path, metrics: Mapping[str, Any]) -> None`, `write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None`, root `environment.json`, root `manifest.json`, and per-arm `environment_segments.jsonl`/`runtime_metrics.json`.

- [ ] **Step 1: Write deterministic metadata tests**

```python
def test_parameter_counts_are_split_by_model_component(tiny_policy):
    counts = count_trainable_parameters(tiny_policy)
    assert counts["total"] == sum(
        p.numel() for p in tiny_policy.parameters() if p.requires_grad
    )
    assert set(counts) == {"total", "feature_extractor", "policy", "value"}
    assert counts["total"] == (
        counts["feature_extractor"] + counts["policy"] + counts["value"]
    )


def test_environment_manifest_contains_required_provenance(monkeypatch):
    monkeypatch.setattr(platform, "platform", lambda: "test-platform")
    manifest = collect_environment(command=["python", "train.py"])
    assert REQUIRED_ENVIRONMENT_KEYS <= manifest.keys()
    assert manifest["command"] == ["python", "train.py"]
    assert manifest["vm_boot_id"]
    assert "gpu_uuid" in manifest


def test_sha256_file_is_content_based(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == hashlib.sha256(b"abc").hexdigest()


def test_manifest_has_one_canonical_checkpoint_inventory(tmp_path):
    manifest = minimal_manifest(
        checkpoints={
            "raw_direct": {"selected": artifact("best_model.sb3", "a" * 64, 50_000)},
            "candidate_cnn": {"selected": artifact("best_model.sb3", "b" * 64, 60_000)},
        }
    )
    write_manifest(tmp_path / "manifest.json", manifest)
    loaded = json.loads((tmp_path / "manifest.json").read_text("utf-8"))
    assert loaded["checkpoints"]["raw_direct"]["selected"]["timestep"] == 50_000


def test_environment_capture_redacts_requirement_credentials():
    assert sanitize_requirement_line(
        "pkg @ https://user:secret@example.test/pkg.whl?token=abc"
    ) == "pkg @ https://example.test/pkg.whl"
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest test_artifact_manifest.py -q`

Expected: FAIL because the metadata module does not exist.

- [ ] **Step 3: Implement environment and runtime capture**

`collect_environment()` records UTC timestamp, command argv, Python/platform, comparison Git SHA/dirty flag, baseline SHA, config/scenario/split/lock SHA256, Linux VM boot ID (or a Windows test fallback), Torch/CUDA/cuDNN, resolved device, GPU name and UUID, GPU total memory, CPU count, process ID, and sanitized `pip freeze`. `write_runtime_metrics()` records target/recorded/end-to-end training seconds, overrun, restart count, max unrecorded seconds, start/end timestep, steps/second, disjoint parameter counts, peak CUDA memory, evaluation seconds, selected checkpoint timestep/selection count/selection tuple, and checkpoint identity.

Use `torch.cuda.get_device_name`, `torch.cuda.get_device_properties`, `torch.cuda.max_memory_allocated`, `/proc/sys/kernel/random/boot_id`, and timeout-bounded `nvidia-smi --query-gpu=uuid --format=csv,noheader` plus `python -m pip freeze`. Strip URL userinfo/query/fragment before recording requirement lines. Missing CUDA yields explicit `null` values, not fabricated GPU data.

Partition parameters by object identity exactly once: feature-extractor parameter IDs first; value MLP and value-head IDs second; every remaining trainable policy parameter ID in `policy`. Assert the three disjoint sets union to all trainable IDs before returning counts.

- [ ] **Step 4: Integrate model-created and training-ended writes**

The orchestrator writes canonical root `environment.json` and `manifest.json`; each training subprocess appends one canonical JSON line to its arm's `environment_segments.jsonl` before model creation and writes `runtime_metrics.json` at exit. At training end, update metrics from wall-clock state, `progress_timing.csv`, and `model.num_timesteps`. Task 4 fills manifest checkpoint entries `selected`, `final`, and `common` with relative path, label, SHA256, and stored timestep. Never include credentials, Drive tokens, environment variables, raw remote URLs, or GitHub authentication data.

- [ ] **Step 5: Run focused and full tests**

Run: `python -m pytest test_artifact_manifest.py test_parallel_training_config.py -q`

Expected: PASS.

Run: `python -m pytest -q`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add AllocRL/comparison/artifact_manifest.py AllocRL/test_artifact_manifest.py AllocRL/train.py
git commit -m "feat: record comparison runtime metadata"
```

---

### Task 4: Evaluate Selected and Common-Step Checkpoints Honestly

**Files:**
- Create: `AllocRL/comparison/checkpoint_evaluator.py`
- Create: `AllocRL/test_checkpoint_evaluator.py`

**Interfaces:**
- Consumes: raw/CNN output directories, fixed scenario bundle, run configs, and model loader.
- Produces: `readable_checkpoint_inventory(output_dir: Path, regular_interval: int) -> dict[int, Path]`, `select_common_timestep(raw_dir: Path, cnn_dir: Path, regular_interval: int) -> int`, `resolve_selected_or_fallback(output_dir: Path) -> CheckpointRef`, `evaluate_checkpoint(model_path: Path, run_config: Mapping[str, Any], scenarios: Sequence[dict], checkpoint_label: str, arm: str, model_loader=MaskablePPO.load) -> list[dict]`, per-arm all-20 CSV, primary-15 CSV, and `common_step_evaluation.csv`.

- [ ] **Step 1: Write failing common-step tests**

```python
def test_common_step_uses_largest_shared_checkpoint(tmp_path):
    raw = checkpoint_dir(tmp_path / "raw", [10_000, 20_000, 30_000])
    cnn = checkpoint_dir(tmp_path / "cnn", [10_000, 20_000])
    assert select_common_timestep(raw, cnn) == 20_000


def test_final_test_excludes_selection_scenarios():
    records = [{"seed": seed} for seed in range(1000, 1020)]
    selection, primary = split_holdout_records(records)
    assert [row["seed"] for row in selection] == list(range(1000, 1005))
    assert [row["seed"] for row in primary] == list(range(1005, 1020))


def test_missing_best_uses_exact_state_checkpoint(tmp_path):
    state_path, checkpoint_path = write_verified_state_checkpoint(
        tmp_path, timestep=12_345
    )
    selected = resolve_selected_or_fallback(tmp_path)
    assert selected.path == checkpoint_path
    assert selected.label == "fallback_final"
    assert selected.timestep == 12_345
```

- [ ] **Step 2: Write failing exact evaluation-row tests**

```python
def test_evaluate_checkpoint_labels_source_and_timestep(fake_model, scenarios):
    rows = evaluate_checkpoint(
        model_path=Path("model_20000_steps.sb3"),
        run_config=complete_run_config(),
        scenarios=scenarios,
        model_loader=lambda *_args, **_kwargs: fake_model,
        checkpoint_label="common_step",
    )
    assert len(rows) == 20
    assert {row["checkpoint"] for row in rows} == {"common_step"}
    assert {row["checkpoint_timestep"] for row in rows} == {20_000}
```

- [ ] **Step 3: Run tests and verify RED**

Run: `python -m pytest test_checkpoint_evaluator.py -q`

Expected: FAIL because the evaluator does not exist.

- [ ] **Step 4: Implement checkpoint selection and evaluation**

```python
@dataclass(frozen=True)
class CheckpointRef:
    path: Path
    label: Literal["best_model", "fallback_final", "final", "common_step"]
    timestep: int
    sha256: str
```

Discover only readable `*.sb3` archives using `train.model_num_timesteps` and SHA256; never rank by filename alone. For the common-step protocol, retain only timesteps divisible by configured `checkpoint_freq`, map duplicate heartbeat generations to the newest verified generation, and choose the maximum exact intersection. Raise a partial-result error if no exact intersection exists. `resolve_selected_or_fallback()` uses readable `best_model.sb3` only when `holdout_selection.csv` proves it was selected; otherwise it returns the exact archive named and hashed by complete `run_state.json` with label `fallback_final`.

Load adjacent `run_config.json`, reconstruct `ObservationScales` with `ObservationScales.from_dict`, restore `active_workspace_codes` and `state_context`, load `MaskablePPO` without a training environment, and call `evaluation_runner.evaluate_scenarios(lambda _: ModelActionPolicy(model, name=arm), list(scenarios), workspace_codes=run_config["active_workspace_codes"], observation_scales=scales, state_context_mode=run_config["state_context"])`. Do not reuse `evaluate_selected_holdout_report()`, whose fixed `best_model` label remains unchanged for baseline compatibility.

Write exact columns from the existing evaluation runner plus:

```text
arm
checkpoint
checkpoint_timestep
checkpoint_sha256
evaluation_partition
```

`evaluation_partition` is `selection` for `1000..1004` and `primary_test` for `1005..1019`. Require exact, unique seed set `1000..1019`; write all 20 rows to `<arm>/evaluation_scenarios.csv`, the exact 15-row subset to `<arm>/evaluation_primary_test.csv`, and paired arm rows at the common checkpoint to `comparison/common_step_evaluation.csv`, all as UTF-8 CSV with deterministic field order. Update root `manifest.json` for every arm's `selected`, `final`, and `common` entries using `{path, label, sha256, timestep}`.

- [ ] **Step 5: Run focused and full tests**

Run: `python -m pytest test_checkpoint_evaluator.py test_evaluation_scenarios.py -q`

Expected: PASS.

Run: `python -m pytest -q`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add AllocRL/comparison/checkpoint_evaluator.py AllocRL/test_checkpoint_evaluator.py
git commit -m "feat: evaluate paired comparison checkpoints"
```

---

### Task 5: Generate the Korean Complete or Partial Report

**Files:**
- Create: `AllocRL/comparison/report_builder.py`
- Create: `AllocRL/test_comparison_report.py`
- Create: `reports/README.md`

**Interfaces:**
- Consumes: per-arm evaluation CSVs, common-step CSV, runtime metrics, environment manifest, training/loss CSVs.
- Produces: `build_comparison_summary(root: str | Path) -> dict[str, Any]`, `build_paired_differences(root: str | Path) -> list[dict]`, `write_complete_report(root: str | Path) -> Path`, `write_partial_report(root: str | Path, failure: str) -> Path`, `summary.json`, two PNG plots, paired-difference CSV, and Korean Markdown.

- [ ] **Step 1: Write failing metric and honesty tests**

```python
def test_summary_uses_only_primary_test_for_primary_means(tmp_path):
    write_eval_fixture(tmp_path, selection_score=0.99, primary_score=0.40)
    summary = build_comparison_summary(tmp_path)
    assert summary["raw_direct"]["primary_test"]["mean_terminal_score"] == 0.40


def test_paired_differences_match_scenario_seeds(tmp_path):
    write_paired_fixture(
        tmp_path,
        raw={1005: (0.2, 0.10, 4.0, 3.0)},
        cnn={1005: (0.5, 0.04, 2.5, 1.0)},
    )
    rows = build_paired_differences(tmp_path)
    assert rows == [{
        "seed": 1005,
        "terminal_score_delta_cnn_minus_raw": pytest.approx(0.3),
        "dropout_rate_delta_cnn_minus_raw": pytest.approx(-0.06),
        "mean_delay_days_delta_cnn_minus_raw": pytest.approx(-1.5),
        "delayed_count_delta_cnn_minus_raw": pytest.approx(-2.0),
    }]


def test_missing_arm_creates_partial_not_complete_report(tmp_path):
    write_raw_only_fixture(tmp_path)
    path = write_partial_report(tmp_path, failure="candidate runtime stopped")
    assert path.name == "PARTIAL_REPORT.md"
    assert not (tmp_path / "COMPLETE.json").exists()
    assert "후보 CNN 결과가 없어 우열을 결론내리지 않음" in path.read_text("utf-8")


def test_korean_report_is_utf8_and_has_no_replacement_character(tmp_path):
    write_complete_fixture(tmp_path)
    text = write_complete_report(tmp_path).read_text(encoding="utf-8")
    assert "예비 결과" in text
    assert "seed 0" in text
    assert "통계적 유의성" in text
    assert "\ufffd" not in text
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest test_comparison_report.py -q`

Expected: FAIL because the report builder does not exist.

- [ ] **Step 3: Implement strict CSV loading and summaries**

Validate headers, unique rows, and exact disjoint seed sets before aggregation. Compute arm means for terminal score, dropout, delay and delayed count; copy recorded/end-to-end seconds, restart/max-unrecorded seconds, total timesteps, steps/second, disjoint parameter counts, peak GPU memory, selected checkpoint timestep/selection tuple/count, and common timestep from canonical metadata. Produce CNN-minus-raw paired differences for all four headline scenario metrics. Never treat 15 scenarios as 15 independent training runs.

- [ ] **Step 4: Implement plots and Korean Markdown**

Create:

1. `learning_curves.png` with episode terminal score versus timestep and checkpoint progress versus recorded subprocess wall time from `progress_timing.csv`;
2. `holdout_comparison.png` with primary-test means and scenario-paired differences;
3. `preliminary_comparison_ko.md` with purpose, architecture, controls, hardware, each 3-hour-budget selected-or-fallback result and checkpoint timestep/selection score/count or fallback reason, common-step result, efficiency, the fact that raw still incurs shared environment grid construction, limitations, and next steps.

Every conclusion sentence is templated from actual JSON values. Missing input produces `자료 없음` and a partial-report explanation, never `0` or an estimated value. Serialize summary JSON with sorted keys and paired CSV with a fixed header/order so a clean rerun is byte-identical; PNG bytes are not part of that reproducibility assertion.

- [ ] **Step 5: Run focused and full tests**

Run: `python -m pytest test_comparison_report.py -q`

Expected: PASS and temporary PNGs are non-empty.

Run: `python -m pytest -q`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add AllocRL/comparison/report_builder.py AllocRL/test_comparison_report.py reports/README.md
git commit -m "feat: build preliminary comparison report"
```

---

### Task 6: Orchestrate Smoke, Raw, CNN, Evaluation, and Reporting

**Files:**
- Create: `AllocRL/comparison/experiment_runner.py`
- Create: `AllocRL/configs/overnight_seed0.json`
- Create: `AllocRL/test_comparison_experiment.py`

**Interfaces:**
- Consumes: immutable production JSON config, Python executable, repository root and Drive output root.
- Produces: frozen `ExperimentConfig`, `load_experiment_config(path: str | Path) -> ExperimentConfig`, `ExperimentConfig.for_test(**operational_overrides: int | float) -> ExperimentConfig`, `build_smoke_command(arm: str, config: ExperimentConfig) -> list[str]`, `build_train_command(arm: str, config: ExperimentConfig, resume_path: Path | None = None) -> list[str]`, `run_overnight_experiment(config_path: str | Path, output_root: str | Path) -> None`, verified idempotent stage journal, `PARTIAL_REPORT.md`, and `COMPLETE.json`.

- [ ] **Step 1: Write failing command equality and order tests**

```python
def test_arm_commands_differ_only_in_extractor_and_output(config):
    raw = build_train_command("raw_direct", config)
    cnn = build_train_command("candidate_cnn", config)
    assert normalized_common_args(raw) == normalized_common_args(cnn)
    assert value_after(raw, "--extractor") == "raw-direct"
    assert value_after(cnn, "--extractor") == "candidate-cnn"
    assert value_after(raw, "--timesteps") == "2000000000"
    assert "--auto-resume" not in raw


def test_runner_smokes_both_before_long_training(fake_subprocess, config):
    run_overnight_experiment(config.path, config.output_root)
    assert fake_subprocess.stage_names[:5] == [
        "smoke_raw_direct",
        "smoke_candidate_cnn",
        "train_raw_direct",
        "evaluate_raw_direct",
        "train_candidate_cnn",
    ]


def test_smoke_failure_prevents_long_training(fake_subprocess, config):
    fake_subprocess.fail("smoke_candidate_cnn")
    with pytest.raises(ExperimentStageError):
        run_overnight_experiment(config.path, config.output_root)
    assert "train_raw_direct" not in fake_subprocess.stage_names
    assert (config.output_root / "comparison" / "PARTIAL_REPORT.md").is_file()
```

- [ ] **Step 2: Write failing completion/resume tests**

```python
def test_completed_arm_is_not_rerun(fake_subprocess, config):
    mark_arm_complete(config.output_root / "raw_direct")
    run_overnight_experiment(config.path, config.output_root)
    assert "train_raw_direct" not in fake_subprocess.stage_names


def test_resume_command_uses_only_checkpoint_named_by_state(tmp_path, config):
    state, named = write_verified_state_checkpoint(tmp_path, timestep=10_000)
    write_orphan_checkpoint(tmp_path, timestep=20_000)
    command = build_train_command("raw_direct", config, resume_path=named)
    assert Path(value_after(command, "--resume-from")) == named
    assert "--auto-resume" not in command


def test_stale_in_progress_stage_is_retried(fake_subprocess, config):
    write_stage(config.output_root, "evaluate_raw_direct", status="in_progress")
    run_overnight_experiment(config.path, config.output_root)
    assert fake_subprocess.stage_names.count("evaluate_raw_direct") == 1


def test_different_vm_or_gpu_prevents_complete_marker(fake_subprocess, config):
    fake_subprocess.environment("raw_direct", boot_id="vm-a", gpu_uuid="gpu-a")
    fake_subprocess.environment("candidate_cnn", boot_id="vm-b", gpu_uuid="gpu-a")
    with pytest.raises(ExperimentIntegrityError, match="same Colab VM"):
        run_overnight_experiment(config.path, config.output_root)
    assert not (config.output_root / "COMPLETE.json").exists()


@pytest.mark.parametrize("failed_stage", [
    "train_raw_direct",
    "train_candidate_cnn",
    "evaluate_common_step",
    "build_report",
])
def test_each_interrupted_stage_is_idempotently_retried(
    failed_stage, fake_subprocess, config
):
    fake_subprocess.kill_once(failed_stage)
    with pytest.raises(ExperimentStageError):
        run_overnight_experiment(config.path, config.output_root)
    completed_before = completed_stage_hashes(config.output_root)
    run_overnight_experiment(config.path, config.output_root)
    assert_completed_hashes_unchanged(config.output_root, completed_before)
    assert fake_subprocess.stage_names.count(failed_stage) == 2


def test_complete_marker_requires_every_stage(fake_subprocess, config):
    run_overnight_experiment(config.path, config.output_root)
    marker = json.loads((config.output_root / "COMPLETE.json").read_text("utf-8"))
    assert marker["status"] == "complete"
    assert marker["stages"] == REQUIRED_COMPLETE_STAGES
```

- [ ] **Step 3: Run tests and verify RED**

Run: `python -m pytest test_comparison_experiment.py -q`

Expected: FAIL because the runner and config do not exist.

- [ ] **Step 4: Create the immutable seed-0 config**

```json
{
  "schema_version": 1,
  "baseline_commit": "cd4e14fc1725a4ff159e59d6874d3602f3b65a06",
  "fixed_scenarios_sha256": "6125f53939a1b8eef8662b2628c0da2f1d0f26b5b541a99252858326b38cd814",
  "split_manifest_path": "data/data_split_manifest.json",
  "split_manifest_sha256": "d3df1d0076248b4bcbddb4c910a3cb81481da65c7415c6b3cacf9e055cc3f9df",
  "seed": 0,
  "state_context": "full",
  "target_training_seconds_per_arm": 10800,
  "timesteps_ceiling": 2000000000,
  "learning_rate": 0.0003,
  "n_steps": 960,
  "batch_size": 64,
  "n_epochs": 10,
  "gamma": 1.0,
  "gae_lambda": 0.98,
  "n_envs": 1,
  "vec_env": "auto",
  "device": "auto",
  "checkpoint_freq": 10000,
  "checkpoint_heartbeat_seconds": 300,
  "holdout_eval_freq": 50000,
  "holdout_selection_count": 5,
  "smoke_timesteps": 1024,
  "scenario_path": "data/fixed_eval_scenarios.json",
  "dependency_lock_path": "requirements-comparison.txt"
}
```

Reject missing/extra keys, either wrong immutable input hash, wrong baseline SHA, non-positive budgets, non-seed-0 configuration, and altered fixed PPO/ceiling/heartbeat values. `ExperimentConfig.for_test()` is a separate constructor that permits only operational overrides `target_training_seconds_per_arm`, `timesteps_ceiling`, `checkpoint_freq`, `checkpoint_heartbeat_seconds`, `holdout_eval_freq`, and `smoke_timesteps`; it preserves seed, data hashes, observation, model, and PPO values and cannot be loaded from the production JSON path.

- [ ] **Step 5: Implement fail-fast sequential orchestration**

Use `subprocess.run(command, check=True, cwd=allocrl_dir)` with argument lists, never shell strings. Smoke commands are exactly `python smoke_test.py --extractor <raw-direct|candidate-cnn> --timesteps 1024 --device cuda --output-dir <root>/smoke/<arm>`. Long commands include `--timesteps 2000000000`, `--max-training-seconds 10800`, `--wall-clock-heartbeat-seconds 300`, the canonical config SHA, `--checkpoint-freq 10000`, `--holdout-eval-freq 50000`, fixed scenarios, and `--no-export-onnx`; they omit both `--auto-resume` and `--final-holdout-report`. On resume, resolve and re-hash only the exact state archive and add `--resume-from <exact path>`.

Use exact ordered stages `preflight`, both smokes, raw train/evaluate, CNN train/evaluate, common evaluation, report, integrity verification. `stage_journal.json` records for each stage `{status,input_sha256,output_sha256,started_at_utc,completed_at_utc,error}`. At startup, change prior `in_progress` to `interrupted`; skip a completed stage only when its current input/output hashes still match. Hold a root lease containing a random token, boot ID, PID and heartbeat; a context-managed daemon thread refreshes it every 60 seconds even while `subprocess.run()` blocks. Refuse a live second writer and allow explicit stale takeover only after 900 seconds.

After any training subprocess, require complete state and a state checkpoint whose basename, SHA256 and stored timestep agree; normal process exit without that gate is failure. Before `COMPLETE.json`, require canonical baseline/comparison/config/scenario/split/lock hashes, exact unique scenario seed sets, all selected/final/common checkpoint hashes, and equality across arm environments for boot ID, GPU UUID, Torch/CUDA/cuDNN and lock hash. Any caught exception writes `comparison/PARTIAL_REPORT.md`; a VM kill is recognized and documented on the next invocation.

- [ ] **Step 6: Run focused and full tests**

Run: `python -m pytest test_comparison_experiment.py test_checkpoint_evaluator.py test_comparison_report.py -q`

Expected: PASS.

Run: `python -m pytest -q`

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add AllocRL/comparison/experiment_runner.py AllocRL/configs/overnight_seed0.json AllocRL/test_comparison_experiment.py
git commit -m "feat: orchestrate overnight comparison"
```

---

### Task 7: Add the Colab Notebook, Dependency Lock, and Operator Docs

**Files:**
- Create: `notebooks/overnight_compare.ipynb`
- Create: `AllocRL/test_comparison_notebook.py`
- Create: `AllocRL/requirements-comparison.in`
- Create: `AllocRL/requirements-comparison.txt`
- Create: `UPSTREAM_BASELINE.md`
- Create: `README.md`

**Interfaces:**
- Consumes: public repository tag `overnight-v1`, Colab Pro GPU runtime, Google Drive authorization.
- Produces: a clean notebook that runs the full experiment once and displays complete/partial report paths.

- [ ] **Step 1: Write failing notebook contract tests**

```python
def test_notebook_has_exact_clone_and_runner_contract():
    source = notebook_code("../notebooks/overnight_compare.ipynb")
    assert "LMS4681/CNN-RL-Raw-Comparison.git" in source
    assert "overnight-v1" in source
    assert "requirements-comparison.txt" in source
    assert "%cd /content/CNN-RL-Raw-Comparison/AllocRL" in source
    assert "comparison.experiment_runner" in source
    assert "/content/drive/MyDrive/CNN-RL-comparison/overnight-20260721" in source


def test_notebook_requires_gpu_and_has_no_saved_outputs():
    notebook = load_notebook("../notebooks/overnight_compare.ipynb")
    source = notebook_code_from_data(notebook)
    assert "torch.cuda.is_available()" in source
    assert "raise RuntimeError" in source
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"])
    assert all(cell.get("execution_count") is None for cell in notebook["cells"])


def test_comparison_lock_is_hashed_and_does_not_install_torch():
    text = Path("requirements-comparison.txt").read_text(encoding="utf-8")
    assert "stable-baselines3==2.9.0" in text
    assert "sb3-contrib==2.9.0" in text
    assert "gymnasium==1.3.0" in text
    assert "--hash=sha256:" in text
    assert not re.search(r"^(torch==|nvidia-|triton==)", text, re.MULTILINE)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest test_comparison_notebook.py -q`

Expected: FAIL because notebook, lock and docs do not exist.

- [ ] **Step 3: Create the pinned dependency contract**

Create `requirements-comparison.in` with the exact direct contract:

```text
gymnasium==1.3.0
stable-baselines3==2.9.0
sb3-contrib==2.9.0
matplotlib
numpy
pandas
tensorboard
tqdm
rich
```

Generate the hashed universal lock from the repository root:

```powershell
uv pip compile AllocRL/requirements-comparison.in --universal --generate-hashes --no-emit-package torch --no-emit-package triton --no-emit-package nvidia-cublas-cu12 --no-emit-package nvidia-cuda-cupti-cu12 --no-emit-package nvidia-cuda-nvrtc-cu12 --no-emit-package nvidia-cuda-runtime-cu12 --no-emit-package nvidia-cudnn-cu12 --no-emit-package nvidia-cufft-cu12 --no-emit-package nvidia-curand-cu12 --no-emit-package nvidia-cusolver-cu12 --no-emit-package nvidia-cusparse-cu12 --no-emit-package nvidia-cusparselt-cu12 --no-emit-package nvidia-nccl-cu12 --no-emit-package nvidia-nvjitlink-cu12 --no-emit-package nvidia-nvtx-cu12 --output-file AllocRL/requirements-comparison.txt
```

Assert every emitted requirement is exact and hashed, install it in the implementation verification environment with `uv pip install --require-hashes -r requirements-comparison.txt`, run `python -m pip check`, and run the full tests. Do not list or reinstall Torch/NVIDIA/triton. Record the resulting lock SHA256 in `UPSTREAM_BASELINE.md`, root manifest, and both arm environments.

- [ ] **Step 4: Build the clean notebook**

Notebook cells, in order:

1. Explain preliminary single-seed scope and six-hour expectation.
2. Mount Drive.
3. Require CUDA, print `nvidia-smi`, Torch/CUDA/cuDNN and RAM.
4. clone `--branch overnight-v1 --depth 1` to `/content/CNN-RL-Raw-Comparison`, record `git rev-parse HEAD`, and fail if the checkout is dirty.
5. capture the preinstall Torch version/path, install `AllocRL/requirements-comparison.txt` with `--require-hashes`, run `pip check`, then assert Torch version/path and CUDA availability are unchanged.
6. `%cd /content/CNN-RL-Raw-Comparison/AllocRL` and verify the config's exact baseline/scenario/split hashes plus the dependency-lock hash recorded in `UPSTREAM_BASELINE.md`.
7. run:

```python
!python -m comparison.experiment_runner \
  --config ./configs/overnight_seed0.json \
  --output-root /content/drive/MyDrive/CNN-RL-comparison/overnight-20260721
```

8. display `COMPLETE.json` and `preliminary_comparison_ko.md`, or display `PARTIAL_REPORT.md` and the exact resume instruction.

- [ ] **Step 5: Document operation and limitations**

`README.md` states: use one GPU notebook, run all once, keep the tab/runtime active, expect roughly six hours plus setup/evaluation, Drive is authoritative, rerun all to resume from the last verified generation, at most 300 seconds plus the current callback interval can be unrecorded after a VM kill, and Colab cannot guarantee uninterrupted completion. `UPSTREAM_BASELINE.md` records the approved baseline SHA, both immutable input hashes, lock SHA, and no-comparison-commits-on-original-main rule.

- [ ] **Step 6: Validate notebook and run focused/full tests**

Run: `python -m json.tool ../notebooks/overnight_compare.ipynb > $null`

Expected: exit 0.

Run: `python -m pytest test_comparison_notebook.py test_requirements.py -q`

Expected: PASS.

Run: `python -m pytest -q`

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add notebooks/overnight_compare.ipynb AllocRL/test_comparison_notebook.py AllocRL/requirements-comparison.in AllocRL/requirements-comparison.txt UPSTREAM_BASELINE.md README.md
git commit -m "docs: add overnight Colab comparison"
```

---

### Task 8: Run Local End-to-End Verification

**Files:**
- Test: all comparison and existing test modules.
- Generated only in a temporary directory: short smoke artifacts.

**Interfaces:**
- Consumes: completed Tasks 1-7.
- Produces: verified code/tag candidate without starting the real six-hour run.

- [ ] **Step 1: Compile executable modules**

Run:

```powershell
python -m compileall -q comparison train.py evaluation_runner.py evaluation_scenarios.py
```

Expected: exit 0.

- [ ] **Step 2: Run focused comparison suite**

Run:

```powershell
python -m pytest test_raw_direct_extractor.py test_wall_clock_callback.py test_artifact_manifest.py test_checkpoint_evaluator.py test_comparison_report.py test_comparison_experiment.py test_comparison_notebook.py -q
```

Expected: all comparison tests PASS.

- [ ] **Step 3: Run the complete regression suite**

Run: `python -m pytest -q`

Expected: all tests PASS; only explicitly documented dependency deprecation warnings may remain.

- [ ] **Step 4: Run the fault-injection resume matrix**

Run: `python -m pytest test_wall_clock_callback.py test_comparison_experiment.py -q`

The named tests inject interruption after archive copy/before state, after raw completion, during CNN training, during common evaluation, and during report generation. Expected: each rerun executes only the interrupted stage, ignores orphan archives, preserves completed-stage output hashes, and creates no `COMPLETE.json` until integrity verification succeeds.

- [ ] **Step 5: Run a short real two-arm end-to-end**

Use `ExperimentConfig.for_test(target_training_seconds_per_arm=30, timesteps_ceiling=1_000_000, checkpoint_freq=128, checkpoint_heartbeat_seconds=5, holdout_eval_freq=0, smoke_timesteps=1_024)` with a temporary output root and the production seed/data/PPO contract. Run the actual sequential runner. Verify both SB3 archives load, both states are complete, runtime metadata exists, fallback evaluation contains exactly 20 unique scenario rows per arm, both arms share at least checkpoint 128, common-step rows exist, and the Korean report contains only real metrics.

- [ ] **Step 6: Verify clean diff and artifact exclusions**

Run: `git diff --check` and `git status --short --ignored`.

Expected: source tree clean after commit; `.sb3`, ONNX, TensorBoard, Drive outputs and temporary smoke outputs are ignored and absent from staged files.

- [ ] **Step 7: Commit any verification documentation only if changed**

If the verification run changes no tracked file, do not create an empty commit. If `README.md` receives exact observed commands/counts, commit only that file:

```bash
git add README.md
git commit -m "docs: record comparison verification"
```

---

### Task 9: Publish the Separate GitHub Repository

**Files:**
- No source changes expected.
- Git tag: `overnight-v1`.

**Interfaces:**
- Consumes: reviewed verification commit.
- Produces: public `https://github.com/LMS4681/CNN-RL-Raw-Comparison` with `main` and immutable `overnight-v1` tag.

- [ ] **Step 1: Verify publishing identity and range**

Run: `git status --short --branch`, `git log --oneline cd4e14f..HEAD`, and `git remote -v`.

Expected: clean comparison branch; original `origin` still points to `LMS4681/CNN-RL`; no comparison implementation is on original `main`.

Run this tracked-content secret gate before any public push:

```powershell
$hits = git grep -n -I -E '(ghp_[A-Za-z0-9]{20,}|github_pat_|AIza[0-9A-Za-z_-]{20,}|-----BEGIN (RSA|OPENSSH|EC) PRIVATE KEY-----)' HEAD
if ($LASTEXITCODE -eq 0) { $hits; exit 1 }
if ($LASTEXITCODE -ne 1) { exit $LASTEXITCODE }
```

Expected: exit 0 with no matches. Confirm that the same owner already publishes the inherited `AllocRL/data` snapshot in the baseline repository; if its public authorization has changed, stop rather than pushing data.

- [ ] **Step 2: Create the empty public repository**

Using the authenticated GitHub UI, create owner `LMS4681`, repository `CNN-RL-Raw-Comparison`, visibility Public, with no generated README, `.gitignore`, or license. If the repository already exists, verify ownership and that overwriting unrelated history is not required; otherwise stop.

- [ ] **Step 3: Add a distinct remote and immutable tag**

```powershell
git remote add comparison https://github.com/LMS4681/CNN-RL-Raw-Comparison.git
git tag -a overnight-v1 -m "Overnight raw-direct versus candidate-CNN comparison"
```

If either name already exists, inspect it rather than replacing it.

- [ ] **Step 4: Push without force**

```powershell
git push comparison HEAD:main
git push comparison overnight-v1
```

Expected: both pushes exit 0; no force push.

- [ ] **Step 5: Verify remote content**

Run: `git ls-remote comparison refs/heads/main refs/tags/overnight-v1 refs/tags/overnight-v1^{}`.

Expected: remote main and dereferenced annotated tag resolve to the reviewed implementation commit. Create a GitHub release for `overnight-v1` and, where the account supports it, a tag ruleset preventing updates/deletion of `overnight-*`. Open the GitHub notebook path and confirm the Colab badge/link resolves. Runtime manifests still record the actual checked-out comparison commit even if repository policy later changes.

---

### Task 10: Launch and Monitor the Overnight Colab Run

**Files:**
- External Drive artifacts only.

**Interfaces:**
- Consumes: public `overnight-v1`, Colab Pro runtime, user Drive authorization.
- Produces: complete or resumable partial Drive run.

- [ ] **Step 1: Open the tagged notebook in Colab**

Open:

```text
https://colab.research.google.com/github/LMS4681/CNN-RL-Raw-Comparison/blob/overnight-v1/notebooks/overnight_compare.ipynb
```

- [ ] **Step 2: Select a GPU runtime and run all**

Use the user-authorized Colab Pro account. Select GPU, run all, manually approve Drive mount, and verify the notebook records one resolved GPU. If CUDA is unavailable, stop before smoke rather than running CPU overnight.

- [ ] **Step 3: Observe the preflight gate**

Wait through dependency install and both 1,024-step smokes. Verify both smoke archives load and the notebook begins `train_raw_direct`. Do not leave the run unattended before this gate passes.

- [ ] **Step 4: Confirm durable Drive progress**

Verify `manifest.json`, `environment.json`, `stage_journal.json`, lease heartbeat, raw `run_state.json`, `progress_timing.csv`, and the first verified raw checkpoint appear under:

```text
/content/drive/MyDrive/CNN-RL-comparison/overnight-20260721/
```

After this point the notebook may run unattended. Do not start a second runtime for this design.

- [ ] **Step 5: Handle interruption without inventing completion**

If the runtime stops, reopen the same tagged notebook and run all. The runner skips hash-verified completed stages and continues from the exact state checkpoint. If Colab preserved the VM, boot/GPU continuity can still pass. If it assigns a new VM/GPU, finish recoverable stages but generate a partial/degraded report: common-step quality may be shown, while the 3-hour wall-time comparison is explicitly non-comparable and `COMPLETE.json` is not created. If reconnection is not performed, preserve the last journal/state artifacts; do not mark the study complete.

---

### Task 11: Validate and Publish the Next-Day Preliminary Report

**Files:**
- Copy into comparison repository only after validation: `reports/preliminary_comparison_ko.md`, `reports/summary.json`, `reports/learning_curves.png`, `reports/holdout_comparison.png`, `reports/scenario_paired_differences.csv`, `reports/artifact_manifest.json`.
- Never copy: model/checkpoint archives, TensorBoard event files, full Drive training directories.

**Interfaces:**
- Consumes: Drive `COMPLETE.json` or `PARTIAL_REPORT.md` and checksummed artifacts.
- Produces: reviewed Korean report commit in the separate GitHub repository.

- [ ] **Step 1: Verify completion marker and checksums**

Require `COMPLETE.json`; both arm states `complete`; each recorded time at least 10,800 seconds with reported overrun; exact baseline/comparison/config/scenario/split/lock hashes; one VM boot ID/GPU UUID and matching Torch/CUDA/cuDNN across all segments; exact state filename/SHA/stored-timestep agreement; exact unique disjoint selection seeds `1000..1004` and primary seeds `1005..1019`; 20 final rows and 15 primary-test rows per arm; and readable selected/final/common models with manifest hashes. If any condition fails, publish only the partial/degraded report and exact missing conditions.

- [ ] **Step 2: Recompute summary from raw CSVs**

Run the report builder again in a clean runtime against the Drive root. Compare canonical regenerated `summary.json` and paired CSV byte-for-byte with the overnight outputs; regenerate PNGs and validate their dimensions/non-empty content without requiring byte equality. Investigate any canonical-data mismatch before reporting results.

- [ ] **Step 3: Review claims against the single-seed limitation**

Confirm the Korean report says `예비`, identifies seed 0, calls each primary result either `3시간 budget 내 validation-selected checkpoint` or an explicit `fallback_final`, lists selected timestep/selection count/selection tuple or fallback reason for each arm, distinguishes the primary 15 scenarios from the all-20 secondary set, reports the common-step comparison, includes hardware/throughput/parameters/restarts/max-unrecorded time, contains no Unicode replacement character, and makes no significance claim.

- [ ] **Step 4: Commit small report artifacts**

```bash
cp /content/drive/MyDrive/CNN-RL-comparison/overnight-20260721/manifest.json reports/artifact_manifest.json
git add reports/preliminary_comparison_ko.md reports/summary.json reports/learning_curves.png reports/holdout_comparison.png reports/scenario_paired_differences.csv reports/artifact_manifest.json
git commit -m "docs: publish overnight comparison results"
git push comparison HEAD:main
```

- [ ] **Step 5: Final verification**

Run `git status --short`, `git ls-remote comparison refs/heads/main`, and open the rendered GitHub report. Report exact test count, Drive completion status, both model budgets, primary metrics, common-step metrics, and the report URL to the user.
