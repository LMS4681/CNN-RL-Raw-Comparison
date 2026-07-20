"""Wall-clock training budget and durable checkpoint regression tests."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from comparison import wall_clock_callback as wall_clock_module
from comparison.wall_clock_callback import (
    WallClockBudgetCallback,
    read_progress_timing,
    read_wall_clock_state,
    reconcile_progress_timing,
    resolve_state_checkpoint,
)


TEST_CONFIG_SHA256 = "a" * 64


class FakeClock:
    def __init__(self) -> None:
        self.seconds = 0.0
        self.started_at = datetime(2026, 7, 21, tzinfo=timezone.utc)

    def monotonic(self) -> float:
        return self.seconds

    def utc_now(self) -> datetime:
        return self.started_at + timedelta(seconds=self.seconds)

    def advance(self, seconds: float) -> None:
        self.seconds += seconds


class FakeModel:
    """Complete local archive double used by the callback tests."""

    def __init__(self) -> None:
        self.num_timesteps = 0
        self.save_raises: Exception | None = None

    def save(self, path: str | Path) -> None:
        if self.save_raises is not None:
            raise self.save_raises
        Path(path).write_text(str(self.num_timesteps), encoding="utf-8")


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


def read_fake_archive(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def prepared_callback(
    output_dir: Path,
    fake_clock: FakeClock,
    *,
    target_seconds: float = 10_800,
    checkpoint_freq: int = 10_000,
    heartbeat_seconds: float = 300,
    model: FakeModel | None = None,
) -> tuple[WallClockBudgetCallback, FakeModel]:
    selected_model = model or FakeModel()
    callback = WallClockBudgetCallback(
        output_dir,
        target_seconds=target_seconds,
        checkpoint_freq=checkpoint_freq,
        heartbeat_seconds=heartbeat_seconds,
        config_sha256=TEST_CONFIG_SHA256,
        monotonic=fake_clock.monotonic,
        utc_now=fake_clock.utc_now,
        archive_timestep_reader=read_fake_archive,
    )
    callback.model = selected_model
    return callback, selected_model


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


def test_read_wall_clock_state_rejects_duplicate_json_fields(tmp_path, fake_clock):
    callback, model = prepared_callback(tmp_path, fake_clock)
    callback._on_training_start()
    model.num_timesteps = 1; callback.persist_checkpoint(status="running")
    path = tmp_path / "run_state.json"; raw = path.read_text(encoding="utf-8").rstrip()
    path.write_text(raw[:-1] + ',"generation":999}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_wall_clock_state(path)


def test_read_wall_clock_state_requires_real_utc_timestamps(tmp_path, fake_clock):
    callback, model = prepared_callback(tmp_path, fake_clock)
    callback._on_training_start(); model.num_timesteps = 1
    callback.persist_checkpoint(status="running")
    path = tmp_path / "run_state.json"
    payload = __import__("json").loads(path.read_text(encoding="utf-8"))
    payload["started_at_utc"] = "2026-07-21T09:00:00+09:00"
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="started_at_utc"):
        read_wall_clock_state(path)


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
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("state write failed")
        ),
    )
    with pytest.raises(OSError, match="state write failed"):
        callback.persist_checkpoint(status="running")

    current = read_wall_clock_state(tmp_path / "run_state.json")
    assert current == prior
    assert resolve_state_checkpoint(tmp_path, current).name == (
        prior.last_checkpoint_file
    )
    assert any(
        "_g2.sb3" in path.name
        for path in (tmp_path / "checkpoints").iterdir()
    )
    assert [row.generation for row in read_progress_timing(
        tmp_path / "progress_timing.csv"
    )] == [1]


def test_state_checkpoint_must_be_a_direct_regular_file(tmp_path, fake_clock):
    callback, model = prepared_callback(tmp_path, fake_clock)
    callback._on_training_start()
    model.num_timesteps = 100
    callback.persist_checkpoint(status="running")
    state = read_wall_clock_state(tmp_path / "run_state.json")

    checkpoint = tmp_path / "checkpoints" / state.last_checkpoint_file
    resolved = resolve_state_checkpoint(tmp_path, state)

    assert resolved == checkpoint.resolve()
    assert resolved.parent == (tmp_path / "checkpoints").resolve()
    assert resolved.is_file() and not resolved.is_symlink()


def test_direct_regular_file_containment_rejects_an_outside_regular_file(
    tmp_path
):
    root = tmp_path / "checkpoints"
    root.mkdir()
    inside = root / "inside.sb3"
    inside.write_bytes(b"inside")
    outside = tmp_path / "outside.sb3"
    outside.write_bytes(b"outside")

    assert wall_clock_module.resolve_direct_regular_file(
        root, inside, label="checkpoint"
    ) == inside.resolve()
    with pytest.raises(ValueError, match="direct regular"):
        wall_clock_module.resolve_direct_regular_file(
            root, outside, label="checkpoint"
        )


def test_state_checkpoint_rejects_symlink_file_escape(tmp_path, fake_clock):
    callback, model = prepared_callback(tmp_path, fake_clock)
    callback._on_training_start()
    model.num_timesteps = 100
    callback.persist_checkpoint(status="running")
    state = read_wall_clock_state(tmp_path / "run_state.json")
    checkpoint = tmp_path / "checkpoints" / state.last_checkpoint_file
    outside = tmp_path / "outside.sb3"
    outside.write_bytes(checkpoint.read_bytes())
    checkpoint.unlink()
    try:
        checkpoint.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows host")

    with pytest.raises(ValueError, match="direct regular"):
        resolve_state_checkpoint(tmp_path, state)


def test_state_checkpoint_rejects_symlinked_checkpoint_directory(
    tmp_path, fake_clock
):
    callback, model = prepared_callback(tmp_path, fake_clock)
    callback._on_training_start()
    model.num_timesteps = 100
    callback.persist_checkpoint(status="running")
    state = read_wall_clock_state(tmp_path / "run_state.json")
    checkpoints = tmp_path / "checkpoints"
    checkpoint = checkpoints / state.last_checkpoint_file
    outside = tmp_path / "outside-checkpoints"
    outside.mkdir()
    (outside / checkpoint.name).write_bytes(checkpoint.read_bytes())
    checkpoint.unlink()
    checkpoints.rmdir()
    try:
        checkpoints.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable on this Windows host")

    with pytest.raises(ValueError, match="direct regular"):
        resolve_state_checkpoint(tmp_path, state)


def test_progress_failure_is_repaired_from_authoritative_state_on_resume(
    tmp_path, fake_clock, monkeypatch
):
    callback, model = prepared_callback(tmp_path, fake_clock)
    callback._on_training_start()
    model.num_timesteps = 100
    callback.persist_checkpoint(status="running")

    original = wall_clock_module.atomic_write_progress_timing
    monkeypatch.setattr(
        wall_clock_module,
        "atomic_write_progress_timing",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("progress write failed")
        ),
    )
    model.num_timesteps = 200
    fake_clock.advance(5)
    with pytest.raises(OSError, match="progress write failed"):
        callback.persist_checkpoint(status="running")

    state = read_wall_clock_state(tmp_path / "run_state.json")
    assert state.generation == 2
    assert [row.generation for row in read_progress_timing(
        tmp_path / "progress_timing.csv"
    )] == [1]

    monkeypatch.setattr(
        wall_clock_module, "atomic_write_progress_timing", original
    )
    resumed, _ = prepared_callback(tmp_path, fake_clock, model=model)
    resumed._on_training_start()
    assert [row.generation for row in read_progress_timing(
        tmp_path / "progress_timing.csv"
    )] == [1, 2]


def test_same_callback_retry_adopts_committed_state_before_next_generation(
    tmp_path, fake_clock, monkeypatch
):
    callback, model = prepared_callback(tmp_path, fake_clock)
    callback._on_training_start()
    model.num_timesteps = 100
    callback.persist_checkpoint(status="running")
    original = wall_clock_module.atomic_write_progress_timing
    monkeypatch.setattr(
        wall_clock_module,
        "atomic_write_progress_timing",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("progress")),
    )
    model.num_timesteps = 200
    fake_clock.advance(5)
    with pytest.raises(OSError, match="progress"):
        callback.persist_checkpoint(status="running")

    monkeypatch.setattr(
        wall_clock_module, "atomic_write_progress_timing", original
    )
    callback.persist_checkpoint(status="running")

    state = read_wall_clock_state(tmp_path / "run_state.json")
    assert state.generation == 3
    assert [row.generation for row in read_progress_timing(
        tmp_path / "progress_timing.csv"
    )] == [1, 2, 3]


def test_reconcile_removes_only_valid_future_tail(tmp_path, fake_clock):
    callback, model = prepared_callback(tmp_path, fake_clock)
    callback._on_training_start()
    model.num_timesteps = 100
    state = callback.persist_checkpoint(status="running")
    path = tmp_path / "progress_timing.csv"
    with path.open("a", encoding="utf-8", newline="") as stream:
        stream.write(
            "2,200,1.0,2026-07-21T00:00:01+00:00,running,"
            "model_200_g2.sb3\n"
        )

    rows = reconcile_progress_timing(path, state)
    assert [row.generation for row in rows] == [1]
    assert read_progress_timing(path) == rows


def test_reconcile_without_state_removes_wholly_valid_orphan_sequence(tmp_path):
    path = tmp_path / "progress_timing.csv"
    path.write_text(
        "generation,timestep,recorded_training_seconds,updated_at_utc,status,checkpoint_file\n"
        "2,200,1.0,2026-07-21T00:00:01+00:00,running,model_200_g2.sb3\n",
        encoding="utf-8",
    )
    assert reconcile_progress_timing(path, None) == []
    assert read_progress_timing(path) == []


def test_reconcile_rejects_same_generation_conflict(tmp_path, fake_clock):
    callback, model = prepared_callback(tmp_path, fake_clock)
    callback._on_training_start()
    model.num_timesteps = 100
    state = callback.persist_checkpoint(status="running")
    path = tmp_path / "progress_timing.csv"
    text = path.read_text(encoding="utf-8").replace(
        "model_100_g1.sb3", "different.sb3"
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="same-generation"):
        reconcile_progress_timing(path, state)


@pytest.mark.parametrize(
    "row",
    [
        "1,100,nan,2026-07-21T00:00:00+00:00,running,model.sb3",
        "1,100,1.0,not-utc,running,model.sb3",
        "1,100,1.0,2026-07-21T00:00:00+00:00,running,../model.sb3",
        "1,100,1.0,2026-07-21T00:00:00+00:00,unknown,model.sb3",
    ],
)
def test_progress_reader_fails_closed_on_malformed_rows(tmp_path, row):
    path = tmp_path / "progress_timing.csv"
    path.write_text(
        "generation,timestep,recorded_training_seconds,updated_at_utc,status,checkpoint_file\n"
        + row
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="progress timing"):
        read_progress_timing(path)


def test_selection_elapsed_time_is_charged_to_same_budget(
    tmp_path, fake_clock
):
    callback, _ = prepared_callback(tmp_path, fake_clock, target_seconds=10_800)
    callback._on_training_start()
    fake_clock.advance(10_800)
    assert callback._on_step() is False


def test_checkpoint_io_time_is_included_in_persisted_elapsed(
    tmp_path, fake_clock
):
    callback, model = prepared_callback(tmp_path, fake_clock)
    callback._on_training_start()
    original_save = model.save

    def timed_save(path):
        fake_clock.advance(7)
        original_save(path)

    model.save = timed_save
    callback.persist_checkpoint(status="running")

    state = read_wall_clock_state(tmp_path / "run_state.json")
    assert state.completed_training_seconds == pytest.approx(7)


def test_progress_rows_record_verified_generation(tmp_path, fake_clock):
    callback, model = prepared_callback(tmp_path, fake_clock)
    callback._on_training_start()
    model.num_timesteps = 10_001
    callback.persist_checkpoint(status="running")

    with (tmp_path / "progress_timing.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [
        {
            "generation": "1",
            "timestep": "10001",
            "recorded_training_seconds": "0.0",
            "updated_at_utc": "2026-07-21T00:00:00+00:00",
            "status": "running",
            "checkpoint_file": "model_10001_g1.sb3",
        }
    ]
    assert read_wall_clock_state(
        tmp_path / "run_state.json"
    ).last_regular_checkpoint_timestep == 10_000


def test_copy_sha_mismatch_never_advances_state(
    tmp_path, fake_clock, monkeypatch
):
    callback, _ = prepared_callback(tmp_path, fake_clock)
    original_copy = wall_clock_module.shutil.copyfile

    def corrupting_copy(source, destination):
        result = original_copy(source, destination)
        with Path(destination).open("ab") as stream:
            stream.write(b" ")
        return result

    monkeypatch.setattr(wall_clock_module.shutil, "copyfile", corrupting_copy)
    with pytest.raises(ValueError, match="SHA256"):
        callback.persist_checkpoint(status="running")

    assert not (tmp_path / "run_state.json").exists()


def test_config_sha_must_be_a_lowercase_hex_string(tmp_path, fake_clock):
    with pytest.raises(ValueError, match="comparison config SHA256"):
        WallClockBudgetCallback(
            tmp_path,
            target_seconds=10,
            checkpoint_freq=10,
            heartbeat_seconds=1,
            config_sha256=None,
            monotonic=fake_clock.monotonic,
            utc_now=fake_clock.utc_now,
            archive_timestep_reader=read_fake_archive,
        )
