"""
블록 배치 강화학습 - CNN+MaskablePPO 학습 + ONNX export.

사용법:
    py train.py --data-dir ./data --timesteps 100000

의존성:
    pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import pickle
import re
import sys
import time
import warnings
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Windows cp949 콘솔에서 Unicode 출력 에러 방지
if sys.platform == "win32" and os.environ.get("PYTHONIOENCODING") is None:
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback

from comparison.wall_clock_callback import (
    WallClockBudgetCallback,
    read_wall_clock_state,
    resolve_state_checkpoint,
)
from comparison.artifact_manifest import (
    append_environment_segment,
    collect_environment,
    count_trainable_parameters,
    read_json_object,
    sha256_file,
    write_runtime_metrics,
)

from alloc_env.observation_state import (
    FUTURE_DAY_WINDOWS,
    GRID_SIZE,
    N_WORKSPACES,
    ORDERED_FUTURE_COUNT,
    PENDING_QUEUE_SLOTS,
    ObservationScales,
)


DEFAULT_ACTIVE_WORKSPACE_CODES = (
    "PE049,PE050,PE055,PE054,PE056,PE048,PE044,PE059,PE060,PE061"
)
DEFAULT_SUPPLEMENTAL_WORKSPACES = {
    "PE054": ("500-B", 51.0, 31.0),
}
DEFAULT_EXCLUDED_START_MONTHS = (7, 11)
DEFAULT_MONTHLY_JITTER = 20
DEFAULT_EMPIRICAL_PROFILE_PROBABILITY = 0.2
TRAINING_DATA_SCHEMA_VERSION = 2
OBSERVATION_SCHEMA_VERSION = 3
REWARD_SCHEMA_VERSION = 2
MODEL_FILENAME = "block_placement_ppo.sb3"
LEGACY_MODEL_FILENAME = "block_placement_ppo.zip"


def _cuda_device_index(device: Any) -> int | None:
    resolved = str(device)
    if resolved == "cuda":
        return torch.cuda.current_device()
    if resolved.startswith("cuda:"):
        return int(resolved.split(":", 1)[1])
    return None


def comparison_runtime_metrics(
    model,
    wall_clock_state,
    *,
    start_timestep: int,
    end_to_end_seconds: float,
    evaluation_seconds: float,
    selected_checkpoint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the per-arm metrics payload from the durable wall-clock state."""
    recorded_seconds = float(wall_clock_state.completed_training_seconds)
    end_timestep = int(model.num_timesteps)
    trained_steps = end_timestep - int(start_timestep)
    cuda_index = _cuda_device_index(getattr(model, "device", "cpu"))
    return {
        "target_training_seconds": float(wall_clock_state.target_training_seconds),
        "recorded_training_seconds": recorded_seconds,
        "end_to_end_training_seconds": float(end_to_end_seconds),
        "overrun_seconds": max(
            0.0,
            recorded_seconds - float(wall_clock_state.target_training_seconds),
        ),
        "restart_count": int(wall_clock_state.restart_count),
        "max_unrecorded_seconds": float(wall_clock_state.max_unrecorded_seconds),
        "start_timestep": int(start_timestep),
        "end_timestep": end_timestep,
        "steps_per_second": (
            trained_steps / recorded_seconds if recorded_seconds > 0 else None
        ),
        "parameter_counts": count_trainable_parameters(model.policy),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(cuda_index))
            if cuda_index is not None
            else None
        ),
        "evaluation_seconds": float(evaluation_seconds),
        **(
            selected_checkpoint
            if selected_checkpoint is not None
            else {
                "selected_checkpoint_timestep": int(
                    wall_clock_state.last_checkpoint_timestep
                ),
                "selection_count": 0,
                "selection_tuple": None,
                "checkpoint_identity": {
                    "filename": wall_clock_state.last_checkpoint_file,
                    "sha256": wall_clock_state.last_checkpoint_sha256,
                },
            }
        ),
    }


def comparison_runtime_provenance(args) -> dict[str, str]:
    """Return the immutable comparison inputs required for arm metadata."""
    fields = {
        "baseline_sha256": getattr(args, "comparison_baseline_sha256", None),
        "config_sha256": getattr(args, "comparison_config_sha256", None),
        "scenario_sha256": getattr(args, "comparison_scenario_sha256", None),
        "split_sha256": getattr(args, "comparison_split_sha256", None),
        "lock_sha256": getattr(args, "comparison_lock_sha256", None),
    }
    missing = sorted(key for key, value in fields.items() if not value)
    if missing:
        raise ValueError(
            "comparison runtime metadata requires immutable provenance: "
            + ", ".join(missing)
        )
    baseline = fields["baseline_sha256"]
    if not isinstance(baseline, str) or re.fullmatch(r"[0-9a-f]{40}", baseline) is None:
        raise ValueError("comparison provenance baseline must be a git commit SHA")
    for key in ("config_sha256", "scenario_sha256", "split_sha256", "lock_sha256"):
        value = fields[key]
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"comparison provenance {key} must be SHA-256")
    requested_device = str(getattr(args, "device", "auto"))
    fields["resolved_device"] = (
        "cuda" if requested_device == "auto" and torch.cuda.is_available() else
        "cpu" if requested_device == "auto" else requested_device
    )
    return fields


def runtime_selected_checkpoint(
    output_dir: str | Path, wall_clock_state, *, selection_count: int = 1
) -> dict[str, Any]:
    """Use a readable selected holdout model, else the verified state archive."""
    output = Path(output_dir)
    selection_path = output / "holdout_selection.csv"
    best_path = output / "best_model.sb3"
    if selection_path.is_file() and best_path.is_file():
        try:
            with selection_path.open(encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source)
                expected_fields = (
                    "timestep",
                    "mean_terminal_score",
                    "mean_dropout_rate",
                    "mean_delay_days",
                    "is_best",
                )
                if tuple(reader.fieldnames or ()) != expected_fields:
                    raise ValueError("holdout selection CSV header is incompatible")
                best_rows = [row for row in reader if row["is_best"] == "1"]
            if best_rows:
                row = best_rows[-1]
                timestep = int(row["timestep"])
                ranking = [
                    float(row["mean_terminal_score"]),
                    -float(row["mean_dropout_rate"]),
                    -float(row["mean_delay_days"]),
                ]
                if model_num_timesteps(best_path) == timestep:
                    return {
                        "selected_checkpoint_timestep": timestep,
                        "selection_count": int(selection_count),
                        "selection_tuple": ranking,
                        "checkpoint_identity": {
                            "filename": best_path.name,
                            "sha256": sha256_file(best_path),
                        },
                    }
        except (csv.Error, KeyError, TypeError, ValueError):
            pass
    checkpoint = output / "checkpoints" / wall_clock_state.last_checkpoint_file
    if not checkpoint.is_file() or sha256_file(checkpoint) != wall_clock_state.last_checkpoint_sha256:
        raise ValueError("complete wall-clock state checkpoint is not verifiable")
    return {
        "selected_checkpoint_timestep": int(wall_clock_state.last_checkpoint_timestep),
        "selection_count": 0,
        "selection_tuple": None,
        "checkpoint_identity": {
            "filename": checkpoint.name,
            "sha256": wall_clock_state.last_checkpoint_sha256,
        },
    }


class Sb3CheckpointCallback(CheckpointCallback):
    """Store SB3 ZIP containers with an extension not filtered by DLP tools."""

    def _checkpoint_path(
        self, checkpoint_type: str = "", extension: str = ""
    ) -> str:
        if checkpoint_type == "" and extension == "zip":
            extension = "sb3"
        return super()._checkpoint_path(checkpoint_type, extension)


def parse_workspace_codes(value: str | None) -> list[str] | None:
    if value is None:
        return None
    codes = [code.strip().upper() for code in value.split(",") if code.strip()]
    return codes or None


def load_allocation_scenario(
    data_dir: str | Path,
    strategy,
    active_workspace_codes: list[str] | None = None,
):
    """Load the fixed ten-yard scenario used by training and model tools."""
    from alloc_env.data_loader import (
        apply_allowable_block_patterns,
        clone_empty_workspaces,
        load_target_blocks,
        load_workspaces,
        select_workspaces_in_order,
    )

    data_dir = Path(data_dir)
    workspaces = load_workspaces(
        str(data_dir / "선행건조 작업장 기준정보.csv"),
        str(data_dir / "선행건조 지번 기준정보.csv"),
        strategy,
        supplemental_workspaces=DEFAULT_SUPPLEMENTAL_WORKSPACES,
    )
    apply_allowable_block_patterns(workspaces)
    blocks = load_target_blocks(
        str(data_dir / "블록데이터.csv"),
        excluded_start_months=DEFAULT_EXCLUDED_START_MONTHS,
    )
    selected = select_workspaces_in_order(
        workspaces, active_workspace_codes
    )
    return blocks, clone_empty_workspaces(selected)


def set_global_seed(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mask_fn(env):
    return env.action_masks()


def build_policy_kwargs(
    extractor: str = "candidate-cnn",
    features_dim: int = 256,
) -> dict:
    from alloc_env.cnn_extractor import (
        CandidateCnnExtractor,
        FixedGridExtractor,
        StructuredExtractor,
    )
    from comparison.raw_direct_extractor import RawDirectExtractor

    extractors = {
        "structured": StructuredExtractor,
        "fixed-grid": FixedGridExtractor,
        "candidate-cnn": CandidateCnnExtractor,
        "raw-direct": RawDirectExtractor,
    }
    if extractor not in extractors:
        raise ValueError(
            f"Unknown extractor '{extractor}'. "
            f"Choose one of: {', '.join(extractors)}"
        )

    return {
        "features_extractor_class": extractors[extractor],
        "features_extractor_kwargs": {"features_dim": features_dim},
        "share_features_extractor": True,
        "net_arch": explicit_policy_net_arch(),
        "activation_fn": torch.nn.ReLU,
    }


def explicit_policy_net_arch() -> dict[str, list[int]]:
    return {"pi": [64, 64], "vf": [64, 64]}


def observation_float_count(observation_space: gym.spaces.Dict) -> int:
    return sum(
        int(np.prod(space.shape))
        for space in observation_space.spaces.values()
    )


def estimate_rollout_buffer_mb(
    observation_space: gym.spaces.Dict,
    n_steps: int,
    n_envs: int,
) -> float:
    return (
        observation_float_count(observation_space)
        * 4
        * n_steps
        * n_envs
        / 1024
        / 1024
    )


def resolve_vec_env_type(vec_env: str, n_envs: int) -> str:
    if n_envs < 1:
        raise ValueError("--n-envs must be at least 1")
    if n_envs == 1:
        return "single"
    if vec_env == "auto":
        return "dummy" if sys.platform == "win32" else "subproc"
    if vec_env in {"dummy", "subproc"}:
        return vec_env
    raise ValueError(f"Unknown vec env type: {vec_env}")


def _validate_production_env_contract(
    workspaces: Sequence,
    grid_size: int,
    observation_scales: ObservationScales | None,
) -> None:
    if len(workspaces) != N_WORKSPACES:
        raise ValueError(
            f"production environments require exactly {N_WORKSPACES} "
            f"workspaces, got {len(workspaces)}"
        )
    if grid_size != GRID_SIZE:
        raise ValueError(
            f"production grid_size must be {GRID_SIZE}, got {grid_size}"
        )
    if not isinstance(observation_scales, ObservationScales):
        raise TypeError(
            "observation_scales must be the full-source ObservationScales"
        )


def make_env(
    blocks,
    workspaces,
    strategy,
    use_synthetic=False,
    generator_dist=None,
    generator_source_blocks=None,
    generator_monthly_jitter=DEFAULT_MONTHLY_JITTER,
    generator_empirical_profile_probability=(
        DEFAULT_EMPIRICAL_PROFILE_PROBABILITY
    ),
    generator_target_month_counts=None,
    synthetic_n_blocks=None,
    vary_layout=True,
    grid_size=GRID_SIZE,
    state_context_mode="full",
    observation_scales=None,
    env_seed=0,
):
    """환경 팩토리 (SubprocVecEnv용)."""
    from alloc_env.alloc_env import BlockPlacementEnv
    from alloc_env.block_generator import SyntheticBlockGenerator

    _validate_production_env_contract(
        workspaces, grid_size, observation_scales
    )

    def _init():
        local_generator = (
            SyntheticBlockGenerator(
                dist=generator_dist,
                seed=env_seed,
                source_blocks=generator_source_blocks,
                monthly_jitter=generator_monthly_jitter,
                empirical_profile_probability=(
                    generator_empirical_profile_probability
                ),
                target_month_counts=generator_target_month_counts,
            )
            if generator_dist is not None
            else None
        )
        env = BlockPlacementEnv(
            blocks, workspaces, strategy,
            use_synthetic=use_synthetic,
            generator=local_generator,
            synthetic_n_blocks=synthetic_n_blocks,
            vary_layout=vary_layout,
            grid_size=grid_size,
            state_context_mode=state_context_mode,
            observation_scales=observation_scales,
        )
        env.action_space.seed(env_seed)
        env.observation_space.seed(env_seed)
        return env
    return _init


def create_training_env(
    blocks,
    workspaces,
    strategy,
    generator,
    observation_scales: ObservationScales,
    grid_size: int = GRID_SIZE,
    n_envs: int = 1,
    vec_env: str = "auto",
    state_context_mode: str = "full",
    seed: int = 0,
    episode_n_blocks: int = 913,
):
    """Create the training env, optionally vectorized for parallel rollout."""
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    _validate_production_env_contract(
        workspaces, grid_size, observation_scales
    )
    resolved_vec_env = resolve_vec_env_type(vec_env, n_envs)
    env_kwargs = {
        "blocks": blocks,
        "workspaces": workspaces,
        "strategy": strategy,
        "use_synthetic": True,
        "generator_dist": generator.dist if generator is not None else None,
        "generator_source_blocks": (
            generator.source_blocks if generator is not None else None
        ),
        "generator_monthly_jitter": (
            generator.monthly_jitter
            if generator is not None else DEFAULT_MONTHLY_JITTER
        ),
        "generator_empirical_profile_probability": (
            generator.empirical_profile_probability
            if generator is not None
            else DEFAULT_EMPIRICAL_PROFILE_PROBABILITY
        ),
        "generator_target_month_counts": (
            generator.target_month_counts
            if generator is not None and generator.source_blocks
            else None
        ),
        "synthetic_n_blocks": episode_n_blocks,
        "vary_layout": False,
        "grid_size": grid_size,
        "state_context_mode": state_context_mode,
        "observation_scales": observation_scales,
    }

    if resolved_vec_env == "single":
        return ActionMasker(
            make_env(**env_kwargs, env_seed=seed)(), mask_fn
        )

    env_fns = [
        make_env(**env_kwargs, env_seed=seed + rank)
        for rank in range(n_envs)
    ]
    if resolved_vec_env == "dummy":
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns)


def create_evaluation_env(
    blocks,
    workspaces,
    strategy,
    observation_scales: ObservationScales,
    grid_size: int = GRID_SIZE,
    state_context_mode: str = "full",
    seed: int = 0,
):
    """CSV 원본 블록으로 평가하는 마스크 적용 환경을 생성합니다."""
    from sb3_contrib.common.wrappers import ActionMasker

    from alloc_env.alloc_env import BlockPlacementEnv

    _validate_production_env_contract(
        workspaces, grid_size, observation_scales
    )
    env = BlockPlacementEnv(
        blocks,
        workspaces,
        strategy,
        use_synthetic=False,
        grid_size=grid_size,
        state_context_mode=state_context_mode,
        observation_scales=observation_scales,
    )
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return ActionMasker(env, mask_fn)


# ── 체크포인트 / 자동 이어학습 유틸 ─────────────────────────────────

# 관측 공간·네트워크 구조에 영향을 주는 키. 이어학습하려면 이 값들이 모두 같아야 한다.
CONFIG_COMPATIBILITY_KEYS = tuple(sorted({
    "training_data_schema_version",
    "observation_schema_version",
    "reward_schema_version",
    "extractor",
    "state_context",
    "grid_size",
    "ordered_future_count",
    "pending_queue_slots",
    "future_day_windows",
    "observation_scales",
    "features_dim",
    "extractor_output_dim",
    "policy_net_arch",
    "policy_activation",
    "active_workspace_codes",
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
}))


def current_run_config(
    args,
    active_workspace_codes: Sequence[str],
    source_manifest: Mapping[str, object],
    observation_scales: ObservationScales,
) -> dict:
    if len(active_workspace_codes) != N_WORKSPACES:
        raise ValueError(
            f"run configuration requires exactly {N_WORKSPACES} active "
            f"workspace codes, got {len(active_workspace_codes)}"
        )
    required_manifest_keys = ("split_seed", "source_sha256")
    missing_manifest_keys = [
        key for key in required_manifest_keys if key not in source_manifest
    ]
    if missing_manifest_keys:
        missing = ", ".join(missing_manifest_keys)
        raise ValueError(
            f"source manifest is missing required compatibility fields: {missing}"
        )
    return {
        "training_data_schema_version": TRAINING_DATA_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "extractor": args.extractor,
        "features_dim": args.features_dim,
        "extractor_output_dim": (
            2772 if args.extractor == "raw-direct" else args.features_dim
        ),
        "policy_net_arch": explicit_policy_net_arch(),
        "policy_activation": "ReLU",
        "active_workspace_codes": list(active_workspace_codes),
        "state_context": args.state_context,
        "grid_size": GRID_SIZE,
        "ordered_future_count": ORDERED_FUTURE_COUNT,
        "pending_queue_slots": PENDING_QUEUE_SLOTS,
        "future_day_windows": [list(item) for item in FUTURE_DAY_WINDOWS],
        "observation_scales": observation_scales.to_dict(),
        "data_split_seed": int(source_manifest["split_seed"]),
        "source_sha256": str(source_manifest["source_sha256"]),
        "episode_block_count": int(source_manifest["source_row_count"]),
        "target_month_counts": dict(source_manifest["source_month_counts"]),
        "excluded_start_months": list(DEFAULT_EXCLUDED_START_MONTHS),
        "monthly_jitter": int(args.monthly_jitter),
        "empirical_profile_probability": float(
            args.empirical_profile_probability
        ),
        "learning_rate": float(args.lr),
        "n_steps": int(args.n_steps),
        "batch_size": int(args.batch_size),
        "n_epochs": int(args.n_epochs),
        "gamma": float(args.gamma),
        "gae_lambda": float(args.gae_lambda),
        "seed": int(args.seed),
        "eval_scenarios": args.eval_scenarios,
    }


def write_run_config(output_dir, config) -> None:
    import json
    with open(Path(output_dir) / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_run_config(path: str | Path) -> dict:
    """Load and validate a training run configuration JSON object."""
    import json

    config_path = Path(path)
    if config_path.is_dir():
        config_path = config_path / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run configuration not found: {config_path}")

    try: return read_json_object(config_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as error:
        raise ValueError(f"Run configuration must be a JSON object: {config_path}") from error


def load_model_run_config(model_path: str | Path) -> dict:
    """Load run_config.json stored beside a model or its checkpoint parent."""
    model_dir = Path(model_path).resolve().parent
    candidates = (
        model_dir / "run_config.json",
        model_dir.parent / "run_config.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return load_run_config(candidate)
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"No run_config.json found for model {model_path}. Searched: {searched}"
    )


def resolve_model_archive_path(path: str | Path) -> Path:
    """Resolve an SB3 archive and reject files transformed by security tools."""
    import zipfile

    requested = Path(path).expanduser().resolve()
    candidates = [requested]
    if requested.suffix == "":
        candidates.extend(
            [Path(f"{requested}.sb3"), Path(f"{requested}.zip")]
        )

    for candidate in candidates:
        if not candidate.is_file():
            continue
        if not zipfile.is_zipfile(candidate):
            raise ValueError(
                f"Model file is not a readable SB3 archive: {candidate}. "
                "A company security filter may have transformed a .zip file; "
                "save new models with the .sb3 extension."
            )
        return candidate

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Saved model not found. Searched: {searched}")


def model_num_timesteps(
    path: Path,
    loader=MaskablePPO.load,
) -> int | None:
    """Read a training archive's stored timestep without attaching an env."""
    try:
        model = loader(str(path), device="cpu")
        value = getattr(model, "num_timesteps", None)
        return int(value) if value is not None else None
    except (
        EOFError,
        OSError,
        RuntimeError,
        ValueError,
        pickle.UnpicklingError,
        zipfile.BadZipFile,
    ):
        return None
    except AssertionError as error:
        if str(error) in {
            "No data found in the saved file",
            "No params found in the saved file",
        }:
            return None
        raise


def find_resumable_model(
    output_dir,
    loader=MaskablePPO.load,
) -> Path | None:
    """Select the readable training state with the greatest stored timestep."""
    root = Path(output_dir)
    final_paths = [root / MODEL_FILENAME, root / LEGACY_MODEL_FILENAME]
    checkpoints_dir = root / "checkpoints"
    checkpoint_paths = (
        list(checkpoints_dir.glob("*.sb3"))
        if checkpoints_dir.is_dir()
        else []
    )
    candidates = [
        path
        for path in [*final_paths, *checkpoint_paths]
        if path.is_file() and path.name != "best_model.sb3"
    ]

    ranked = []
    for path in candidates:
        timesteps = model_num_timesteps(path, loader=loader)
        if timesteps is None:
            continue
        ranked.append((
            timesteps,
            int(path in final_paths),
            path.stat().st_mtime_ns,
            path.name,
            path,
        ))
    if not ranked:
        return None
    return max(ranked, key=lambda item: item[:4])[-1]


def config_mismatches(
    saved: Mapping,
    current: Mapping,
) -> dict[str, tuple[Any, Any]]:
    return {
        key: (saved.get(key), current.get(key))
        for key in CONFIG_COMPATIBILITY_KEYS
        if saved.get(key) != current.get(key)
    }


def configs_compatible(
    saved: Mapping,
    current: Mapping,
) -> bool:
    return not config_mismatches(saved, current)


def require_current_training_data_schema(
    config: dict,
    source: str,
) -> None:
    saved_version = config.get("training_data_schema_version")
    if saved_version == TRAINING_DATA_SCHEMA_VERSION:
        return
    raise ValueError(
        f"[{source}] Saved model training_data_schema_version is "
        f"incompatible: saved={saved_version}, "
        f"current={TRAINING_DATA_SCHEMA_VERSION}. Use the matching legacy "
        "code or train a new model."
    )


def require_current_observation_schema(
    config: Mapping[str, object],
    source: str,
) -> None:
    saved_version = config.get("observation_schema_version")
    if saved_version == OBSERVATION_SCHEMA_VERSION:
        return
    raise ValueError(
        f"[{source}] Saved model observation_schema_version is "
        f"incompatible: saved={saved_version}, "
        f"current={OBSERVATION_SCHEMA_VERSION}. Schema-2 models cannot be "
        "used with the schema-3 environment; train or select a schema-3 model."
    )


def observation_contract_from_run_config(
    config: dict,
    source: str,
) -> tuple[list[str], str, ObservationScales]:
    require_current_training_data_schema(config, source=source)
    require_current_observation_schema(config, source=source)

    fixed_values = {
        "grid_size": GRID_SIZE,
        "ordered_future_count": ORDERED_FUTURE_COUNT,
        "pending_queue_slots": PENDING_QUEUE_SLOTS,
        "future_day_windows": [list(item) for item in FUTURE_DAY_WINDOWS],
    }
    for key, expected in fixed_values.items():
        actual = config.get(key)
        if actual != expected:
            raise ValueError(
                f"[{source}] Saved model {key} is incompatible: "
                f"saved={actual}, current={expected}"
            )

    workspace_codes = config.get("active_workspace_codes")
    if (
        isinstance(workspace_codes, (str, bytes))
        or not isinstance(workspace_codes, Sequence)
        or len(workspace_codes) != N_WORKSPACES
        or any(not isinstance(code, str) for code in workspace_codes)
    ):
        raise ValueError(
            f"[{source}] active_workspace_codes must contain exactly "
            f"{N_WORKSPACES} string codes"
        )

    state_context = config.get("state_context")
    if state_context not in {"full", "current"}:
        raise ValueError(
            f"[{source}] state_context must be 'full' or 'current', "
            f"got {state_context!r}"
        )
    try:
        scales = ObservationScales.from_dict(config["observation_scales"])
    except KeyError as error:
        raise ValueError(
            f"[{source}] Saved model is missing observation_scales"
        ) from error
    return list(workspace_codes), str(state_context), scales


def require_compatible_run_config(
    saved: dict,
    current: dict,
    source: str,
) -> None:
    mismatches = config_mismatches(saved, current)
    if not mismatches:
        return
    details = "\n".join(
        f"  {key}: saved={saved_value!r}, current={current_value!r}"
        for key, (saved_value, current_value) in mismatches.items()
    )
    raise ValueError(
        f"[{source}] Saved model configuration is incompatible:\n"
        f"{details}\n"
        "Use a matching configuration or a new output directory."
    )


def resolve_resume_path(args, output_dir, current_config):
    """이어학습 경로를 결정한다.

    - --resume-from 이 명시되면 그 경로(없으면 에러).
    - 아니고 --auto-resume 이면 output-dir에서 호환 가능한 최신 모델을 자동 탐지.
      기존 설정과 구조가 다르면 관측/네트워크 불일치를 막기 위해 ValueError.
    반환: Path(이어학습) 또는 None(새로 학습).
    """
    import json

    max_training_seconds = float(
        getattr(args, "max_training_seconds", 0.0)
    )
    if max_training_seconds < 0:
        raise ValueError("--max-training-seconds must be non-negative")
    if max_training_seconds > 0:
        return _resolve_wall_clock_resume_path(
            args,
            output_dir,
            current_config,
            target_seconds=max_training_seconds,
        )

    if args.resume_from:
        candidate = resolve_model_archive_path(args.resume_from)
        saved_config = load_model_run_config(candidate)
        require_compatible_run_config(
            saved_config, current_config, source="resume-from"
        )
        return candidate

    if not getattr(args, "auto_resume", False):
        return None

    candidate = find_resumable_model(output_dir)
    if candidate is None:
        print("[auto-resume] 기존 체크포인트 없음 → 새로 학습합니다.")
        return None

    cfg_path = Path(output_dir) / "run_config.json"
    if not cfg_path.exists():
        print("[auto-resume] run_config.json 없음 → 호환성 확인 불가, 새로 학습합니다.")
        return None

    saved_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    require_compatible_run_config(
        saved_cfg, current_config, source="auto-resume"
    )
    print(f"[auto-resume] 호환 체크포인트 발견 → 이어학습: {candidate}")
    return candidate


def _wall_clock_state_path(args, output_dir: str | Path) -> Path:
    configured = getattr(args, "wall_clock_state", None)
    if configured is not None:
        return Path(configured).expanduser().resolve()
    return (Path(output_dir) / "run_state.json").resolve()


def _resolve_wall_clock_resume_path(
    args,
    output_dir: str | Path,
    current_config: dict,
    *,
    target_seconds: float,
) -> Path | None:
    """Resolve only the exact verified generation named by wall-clock state."""
    config_sha256 = getattr(args, "comparison_config_sha256", None)
    if not isinstance(config_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", config_sha256
    ) is None:
        raise ValueError(
            "wall-clock training requires --comparison-config-sha256 as "
            "64-character lowercase hexadecimal"
        )
    if getattr(args, "auto_resume", False):
        raise ValueError(
            "--auto-resume is not allowed in wall-clock mode; pass the exact "
            "state-named archive with --resume-from"
        )

    output_dir = Path(output_dir).resolve()
    state_path = _wall_clock_state_path(args, output_dir)
    if not state_path.exists():
        if getattr(args, "resume_from", None):
            raise ValueError(
                "wall-clock resume requires an existing verified run_state.json"
            )
        return None

    state = read_wall_clock_state(state_path)
    if state.target_training_seconds != float(target_seconds):
        raise ValueError(
            "wall-clock target changed on resume: "
            f"state {state.target_training_seconds}, requested {target_seconds}"
        )
    if state.config_sha256 != config_sha256:
        raise ValueError("wall-clock config SHA256 changed on resume")
    expected = resolve_state_checkpoint(output_dir, state)
    requested = getattr(args, "resume_from", None)
    if not requested:
        raise ValueError(
            "existing wall-clock state requires --resume-from with the exact "
            "state-named checkpoint"
        )
    candidate = resolve_model_archive_path(requested)
    if candidate.resolve() != expected.resolve():
        raise ValueError(
            "--resume-from must be the exact checkpoint named by run_state.json: "
            f"expected {expected}, got {candidate}"
        )
    stored_timestep = model_num_timesteps(candidate)
    if stored_timestep != state.last_checkpoint_timestep:
        raise ValueError(
            "state checkpoint timestep "
            f"{state.last_checkpoint_timestep} does not match archive "
            f"{stored_timestep}"
        )
    saved_config = load_model_run_config(candidate)
    require_compatible_run_config(
        saved_config, current_config, source="resume-from"
    )
    return candidate


def load_requested_evaluation_scenarios(
    path: str | Path | None,
) -> list[dict] | None:
    if path is None:
        return None
    scenario_path = Path(path).expanduser().resolve()
    if not scenario_path.is_file():
        raise FileNotFoundError(
            f"Fixed evaluation scenarios not found: {scenario_path}. "
            "Run `py -B run_ablation.py --prepare-eval-scenarios` first."
        )
    from evaluation_scenarios import read_scenarios

    return read_scenarios(scenario_path)


def write_evaluation_metrics(
    path: str | Path,
    rows: list[dict],
) -> None:
    from evaluation_runner import write_evaluation_metrics as write_metrics

    write_metrics(path, rows)


def evaluate_fixed_scenarios(
    model,
    scenario_records: list[dict],
    observation_scales: ObservationScales,
    state_context_mode: str,
    workspace_codes: list[str] | None = None,
) -> list[dict]:
    from evaluation_runner import ModelActionPolicy, evaluate_scenarios

    return evaluate_scenarios(
        lambda seed: ModelActionPolicy(model),
        scenario_records,
        workspace_codes=workspace_codes,
        observation_scales=observation_scales,
        state_context_mode=state_context_mode,
    )


def create_holdout_eval_callback(
    scenarios,
    evaluate_fn,
    output_dir: str | Path,
    eval_freq: int,
    selection_count: int,
):
    """Build fixed-holdout selection unless it is explicitly disabled."""
    if eval_freq < 0:
        raise ValueError("holdout eval frequency must be non-negative")
    if scenarios is None or eval_freq == 0:
        return None

    from holdout_model_selection import FixedHoldoutEvalCallback

    return FixedHoldoutEvalCallback(
        scenarios,
        evaluate_fn,
        output_dir,
        eval_freq=eval_freq,
        selection_count=selection_count,
    )


def _detach_callback_references(callbacks) -> None:
    """Remove model references retained by SB3 callback state."""
    seen: set[int] = set()

    def detach(callback) -> None:
        if callback is None or id(callback) in seen:
            return
        seen.add(id(callback))
        for child in getattr(callback, "callbacks", ()):
            detach(child)
        detach(getattr(callback, "callback", None))
        if hasattr(callback, "locals"):
            callback.locals.clear()
        if hasattr(callback, "globals"):
            callback.globals.clear()
        if hasattr(callback, "parent"):
            callback.parent = None
        if hasattr(callback, "model"):
            callback.model = None

    for callback in callbacks:
        detach(callback)
    callbacks.clear()


def _detach_model_environment(model) -> None:
    if model is None:
        return
    if hasattr(model, "env"):
        model.env = None
    if hasattr(model, "_vec_normalize_env"):
        model._vec_normalize_env = None


def _release_training_resources(model, callbacks, training_env) -> None:
    _detach_callback_references(callbacks)
    _detach_model_environment(model)
    if training_env is not None:
        training_env.close()


class _TrainingResourceLifecycle:
    def __init__(self) -> None:
        self.training_env = None
        self.model = None
        self.callbacks = []
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _release_training_resources(
            self.model, self.callbacks, self.training_env
        )
        self.model = None
        self.training_env = None


def evaluate_selected_holdout_report(
    model_class,
    *,
    output_dir: str | Path,
    training_env,
    scenario_records: list[dict],
    device: str,
    evaluate_fn,
) -> list[dict]:
    """Load the selected checkpoint and evaluate all fixed scenarios."""
    from evaluation_runner import ModelActionPolicy
    from holdout_model_selection import (
        BEST_MODEL_FILENAME,
        validate_fixed_holdout_scenarios,
    )

    selected_model = None
    attached_env = None
    try:
        validate_fixed_holdout_scenarios(scenario_records)
        best_path = resolve_model_archive_path(
            Path(output_dir) / BEST_MODEL_FILENAME
        )
        selected_model = model_class.load(
            str(best_path), env=training_env, device=device
        )
        if hasattr(selected_model, "get_env"):
            attached_env = selected_model.get_env()
        rows = evaluate_fn(
            lambda _seed: ModelActionPolicy(selected_model, name="model"),
            scenario_records,
        )
        return [{**row, "checkpoint": "best_model"} for row in rows]
    finally:
        _detach_model_environment(selected_model)
        if attached_env is not None:
            attached_env.close()
        elif training_env is not None:
            training_env.close()


def train(args):
    """Run training with deterministic cleanup on every exit path."""
    resources = _TrainingResourceLifecycle()
    try:
        return _train(args, resources)
    finally:
        resources.close()


def _train(args, resources: _TrainingResourceLifecycle):
    """MaskablePPO 학습 실행."""
    from sb3_contrib import MaskablePPO

    from alloc_env.strategy import BaseGridStrategy
    from alloc_env.callbacks import AllocationCallback, TrainingMetricsCallback
    from alloc_env.block_generator import SyntheticBlockGenerator

    fixed_scenarios = load_requested_evaluation_scenarios(
        getattr(args, "eval_scenarios", None)
    )
    if args.final_holdout_report and fixed_scenarios is None:
        raise ValueError(
            "--final-holdout-report requires --eval-scenarios"
        )
    set_global_seed(args.seed)

    data_dir = Path(args.data_dir)
    print("=" * 60)
    print("  블록 배치 강화학습 - MaskablePPO")
    print("=" * 60)

    # ── 1. 데이터 로드 ────────────────────────────────────────────
    strategy = BaseGridStrategy(step=5.0)
    active_workspace_codes = parse_workspace_codes(args.active_workspace_codes)
    full_blocks, workspaces = load_allocation_scenario(
        data_dir, strategy, active_workspace_codes
    )
    from alloc_env.alloc_env import DROPOUT_THRESHOLD
    from alloc_env.observation_state import build_observation_scales

    observation_scales = build_observation_scales(
        full_blocks,
        workspaces,
        DROPOUT_THRESHOLD,
    )
    active_workspace_code_list = [
        workspace.code for workspace in workspaces
    ]

    def run_fixed_holdout(policy_factory, selected_scenarios):
        from evaluation_runner import evaluate_scenarios

        return evaluate_scenarios(
            policy_factory,
            list(selected_scenarios),
            workspace_codes=active_workspace_code_list,
            observation_scales=observation_scales,
            state_context_mode=args.state_context,
        )

    from alloc_env.data_split import split_blocks_by_ship

    source_split = split_blocks_by_ship(
        full_blocks,
        data_dir / "블록데이터.csv",
        split_seed=20260716,
        holdout_fraction=0.20,
    )
    target_month_counts = Counter(
        (block.in_date.year, block.in_date.month) for block in full_blocks
    )

    print(f"블록 {len(full_blocks)}개, 작업장 {len(workspaces)}개")
    if active_workspace_codes:
        print(
            f"Active workspaces: {len(workspaces)} "
            f"({', '.join(ws.code for ws in workspaces)})"
        )
    else:
        print(f"Active workspaces: all {len(workspaces)}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    max_training_seconds = float(
        getattr(args, "max_training_seconds", 0.0)
    )
    wall_clock_enabled = max_training_seconds > 0
    if max_training_seconds < 0:
        raise ValueError("--max-training-seconds must be non-negative")
    if wall_clock_enabled and args.checkpoint_freq <= 0:
        raise ValueError(
            "wall-clock training requires --checkpoint-freq to be positive"
        )
    run_config = current_run_config(
        args,
        active_workspace_code_list,
        source_split.manifest,
        observation_scales,
    )
    resume_path = resolve_resume_path(args, output_dir, run_config)
    is_resume = resume_path is not None
    write_run_config(output_dir, run_config)

    # ── 2. Synthetic 블록 생성기 ─────────────────────────────────
    generator = SyntheticBlockGenerator.from_blocks(
        source_split.training_blocks,
        seed=args.seed,
        monthly_jitter=args.monthly_jitter,
        empirical_profile_probability=(
            args.empirical_profile_probability
        ),
        target_month_counts=target_month_counts,
    )
    print(
        "[Synthetic] fixed total="
        f"{len(full_blocks)}, monthly jitter=+/-{args.monthly_jitter}, "
        "empirical profile probability="
        f"{args.empirical_profile_probability:.2f}"
    )

    # ── 3. 환경 생성 (학습: synthetic, 평가: CSV 원본) ────────────
    env = create_training_env(
        source_split.training_blocks,
        workspaces,
        strategy,
        generator,
        observation_scales=observation_scales,
        episode_n_blocks=len(full_blocks),
        grid_size=args.grid_size,
        n_envs=args.n_envs,
        vec_env=args.vec_env,
        state_context_mode=args.state_context,
        seed=args.seed,
    )
    resources.training_env = env
    resolved_vec_env = resolve_vec_env_type(args.vec_env, args.n_envs)

    # 메모리 사용량 예측
    G = args.grid_size
    buffer_mb = estimate_rollout_buffer_mb(
        env.observation_space, args.n_steps, args.n_envs
    )
    print(f"Obs space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(
        f"Training envs: {args.n_envs} ({resolved_vec_env}), "
        f"device={args.device}"
    )
    print(
        f"Rollout buffer 예상 메모리: {buffer_mb:.0f} MB "
        f"(grid={G}×{G}, n_steps={args.n_steps}, n_envs={args.n_envs})"
    )

    # ── 4. 모델 생성 (CNN+MLP 하이브리드) ─────────────────────────
    policy_kwargs = build_policy_kwargs(
        extractor=args.extractor,
        features_dim=args.features_dim,
    )
    if wall_clock_enabled:
        append_environment_segment(
            output_dir / "environment_segments.jsonl",
            collect_environment(
                sys.argv, provenance=comparison_runtime_provenance(args)
            ),
        )

    print(f"Feature extractor: {args.extractor}")
    if is_resume:
        print(f"기존 모델에서 이어 학습: {resume_path}")
        model = MaskablePPO.load(
            str(resume_path),
            env=env,
            device=args.device,
            tensorboard_log=str(output_dir / "tb_logs"),
        )
    else:
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            policy_kwargs=policy_kwargs,
            seed=args.seed,
            device=args.device,
            tensorboard_log=str(output_dir / "tb_logs"),
        )
    resources.model = model

    # ── 5. 콜백 설정 ──────────────────────────────────────────────
    callback = [
        AllocationCallback(
            log_dir=args.output_dir, verbose=1, append=is_resume
        ),
        TrainingMetricsCallback(
            log_dir=args.output_dir, verbose=1, append=is_resume
        ),
    ]
    resources.callbacks = callback
    holdout_callback = create_holdout_eval_callback(
        fixed_scenarios,
        run_fixed_holdout,
        output_dir,
        eval_freq=args.holdout_eval_freq,
        selection_count=args.holdout_selection_count,
    )
    if holdout_callback is not None:
        callback.append(holdout_callback)
    if args.checkpoint_freq > 0 and not wall_clock_enabled:
        # SB3 CheckpointCallback은 콜백 호출 횟수 기준이라 n_envs로 나눠 step 단위를 맞춘다.
        save_freq = max(args.checkpoint_freq // max(args.n_envs, 1), 1)
        callback.append(Sb3CheckpointCallback(
            save_freq=save_freq,
            save_path=str(output_dir / "checkpoints"),
            name_prefix="block_placement_ppo",
            verbose=1,
        ))
        print(
            f"중간 체크포인트: 약 {args.checkpoint_freq} step마다 "
            f"→ {output_dir / 'checkpoints'}"
        )
    if wall_clock_enabled:
        callback.append(WallClockBudgetCallback(
            output_dir,
            target_seconds=max_training_seconds,
            checkpoint_freq=args.checkpoint_freq,
            heartbeat_seconds=float(
                getattr(args, "wall_clock_heartbeat_seconds", 300.0)
            ),
            config_sha256=getattr(
                args, "comparison_config_sha256", None
            ),
            state_path=_wall_clock_state_path(args, output_dir),
        ))
        print(
            "Wall-clock budget: "
            f"{max_training_seconds:.0f} seconds, verified checkpoints → "
            f"{output_dir / 'checkpoints'}"
        )

    # ── 6. 학습 ──────────────────────────────────────────────────
    # 이어학습이면 reset_num_timesteps=False → 기존 step 뒤에 args.timesteps 만큼 추가 학습.
    print(f"\n학습 시작: {args.timesteps} timesteps "
          f"({'이어학습(추가)' if is_resume else '신규'})")
    print(f"TensorBoard: tensorboard --logdir {Path(args.output_dir) / 'tb_logs'}")
    training_start_timestep = None
    training_started = None
    if wall_clock_enabled:
        training_start_timestep = int(model.num_timesteps)
        training_started = time.monotonic()
    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=True,
        callback=callback,
        reset_num_timesteps=not is_resume,
    )

    if wall_clock_enabled:
        state_path = _wall_clock_state_path(args, output_dir)
        try:
            wall_clock_state = read_wall_clock_state(state_path)
        except FileNotFoundError as error:
            raise RuntimeError(
                "training returned before a complete wall-clock state was "
                "persisted"
            ) from error
        if wall_clock_state.status != "complete":
            raise RuntimeError(
                "training timestep ceiling returned before the wall-clock "
                "budget completed; the arm remains incomplete"
            )

    # ── 7. 모델 저장 ─────────────────────────────────────────────
    sb3_path = str(output_dir / MODEL_FILENAME)
    model.save(sb3_path)
    print(f"\nSB3 모델 저장: {sb3_path}")

    # ── 8. ONNX export ───────────────────────────────────────────
    if args.export_onnx:
        onnx_path = str(output_dir / "block_placement_ppo.onnx")
        if try_export_to_onnx(model, env, onnx_path):
            print(f"ONNX 모델 저장: {onnx_path}")

    # ── 9. 학습 결과 평가 ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  학습 완료 - 최종 평가")
    print("=" * 60)
    evaluation_started = time.monotonic()
    eval_env = create_evaluation_env(
        full_blocks,
        workspaces,
        strategy,
        observation_scales=observation_scales,
        grid_size=args.grid_size,
        state_context_mode=args.state_context,
        seed=args.seed,
    )
    try:
        csv_row = evaluate_original_csv_row(
            model, eval_env, n_eval=args.n_eval
        )
    finally:
        eval_env.close()
    evaluation_seconds = time.monotonic() - evaluation_started
    write_evaluation_metrics(
        output_dir / "evaluation_csv.csv",
        [csv_row],
    )
    if wall_clock_enabled:
        write_runtime_metrics(
            output_dir / "runtime_metrics.json",
            comparison_runtime_metrics(
                model,
                wall_clock_state,
                start_timestep=training_start_timestep,
                end_to_end_seconds=time.monotonic() - training_started,
                evaluation_seconds=evaluation_seconds,
                selected_checkpoint=runtime_selected_checkpoint(
                    output_dir,
                    wall_clock_state,
                    selection_count=args.holdout_selection_count,
                ),
            ),
        )

    resources.close()
    model = None
    holdout_callback = None
    callback = None
    env = None
    gc.collect()

    if args.final_holdout_report:
        print("\n  Selected model fixed holdout report")
        selected_model_env = create_evaluation_env(
            full_blocks,
            workspaces,
            strategy,
            observation_scales=observation_scales,
            grid_size=args.grid_size,
            state_context_mode=args.state_context,
            seed=args.seed,
        )
        scenario_rows = evaluate_selected_holdout_report(
            MaskablePPO,
            output_dir=output_dir,
            training_env=selected_model_env,
            scenario_records=fixed_scenarios,
            device=args.device,
            evaluate_fn=run_fixed_holdout,
        )
        write_evaluation_metrics(
            output_dir / "evaluation_scenarios.csv", scenario_rows
        )


def evaluate(model, env, n_eval: int = 5, return_metrics: bool = False):
    """Compatibility wrapper for deterministic model evaluation."""
    from evaluation_runner import ModelActionPolicy, evaluate_policy

    metrics = evaluate_policy(
        ModelActionPolicy(model), env, episodes=n_eval
    )
    print(
        f"\n  Mean evaluation: reward={metrics['mean_reward']:.2f}, "
        f"terminal score={metrics['mean_terminal_score']:.2f}, "
        f"dropout={metrics['mean_dropout_rate']:.1%}, "
        f"retained choices={metrics['mean_retained_choice_ratio']:.3f} "
        f"(n={n_eval})"
    )
    return metrics if return_metrics else metrics["mean_reward"]


def evaluate_original_csv(model, env, n_eval: int = 5) -> dict[str, float]:
    """Evaluate the original CSV exactly once while accepting legacy n_eval."""
    from evaluation_runner import ModelActionPolicy, evaluate_policy

    warnings.warn(
        "--n-eval is deprecated and ignored; original CSV evaluation always "
        "runs exactly one episode.",
        FutureWarning,
        stacklevel=2,
    )
    return evaluate_policy(ModelActionPolicy(model), env, episodes=1)


def evaluate_original_csv_row(model, env, n_eval: int = 5) -> dict:
    """Evaluate once and build the single original-CSV metric row."""
    return {
        "source": "original_csv",
        "policy": "model",
        **evaluate_original_csv(model, env, n_eval=n_eval),
    }


def export_to_onnx(model, env, onnx_path: str):
    """SB3 모델을 ONNX 형식으로 export (Dict obs 대응, 동적 키).

    관측 키 집합과 텐서 shape는 observation_space에서 직접 읽는다.
    """
    import inspect

    import torch
    import onnx

    policy = model.policy
    obs_space = env.observation_space

    if not hasattr(obs_space, "spaces"):
        raise ValueError(
            "ONNX export는 Dict 관측 공간을 기대합니다 (flat obs 미지원)."
        )

    obs_keys = sorted(obs_space.spaces)
    dummy_obs = {
        key: torch.zeros(
            1,
            *obs_space.spaces[key].shape,
            device=policy.device,
        )
        for key in obs_keys
    }

    # Actor 네트워크만 export (추론에 필요한 부분)
    class PolicyWrapper(torch.nn.Module):
        def __init__(self, policy, obs_keys):
            super().__init__()
            self.policy = policy
            self._obs_keys = list(obs_keys)

        def forward(self, *obs_tensors):
            obs_dict = dict(zip(self._obs_keys, obs_tensors))
            features = self.policy.extract_features(
                obs_dict, self.policy.pi_features_extractor
            )
            latent_pi = self.policy.mlp_extractor.forward_actor(features)
            logits = self.policy.action_net(latent_pi)
            input_anchor = sum(
                tensor.flatten(start_dim=1)[:, :1] * 0.0
                for tensor in obs_tensors
            )
            return logits + input_anchor

    wrapper = PolicyWrapper(policy, obs_keys)
    wrapper.eval()

    # Dict obs를 키 순서대로 개별 인자로 전달
    dummy_inputs = tuple(dummy_obs[key] for key in obs_keys)
    dynamic_axes = {key: {0: "batch"} for key in obs_keys}
    dynamic_axes["action_logits"] = {0: "batch"}

    export_kwargs = {
        "input_names": obs_keys,
        "output_names": ["action_logits"],
        "dynamic_axes": dynamic_axes,
        "opset_version": 17,
    }
    try:
        export_parameters = inspect.signature(torch.onnx.export).parameters
    except (TypeError, ValueError):
        export_parameters = {}
    if "dynamo" in export_parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(
        wrapper,
        dummy_inputs,
        onnx_path,
        **export_kwargs,
    )

    # 검증
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    with torch.no_grad():
        output = wrapper(*dummy_inputs)
    if not bool(torch.isfinite(output).all().item()):
        raise ValueError("ONNX actor output must be finite")
    print(f"  ONNX inputs: {[inp.name for inp in onnx_model.graph.input]}")


def try_export_to_onnx(model, env, onnx_path: str | Path) -> bool:
    """Export an optional ONNX artifact without invalidating saved SB3 output."""
    try:
        export_to_onnx(model, env, str(onnx_path))
    except Exception as exc:
        path = Path(onnx_path)
        try:
            path.unlink(missing_ok=True)
        except OSError as cleanup_error:
            print(f"[경고] 불완전한 ONNX 파일 제거 실패: {cleanup_error}")
        print(
            f"\n[경고] ONNX 모델 변환 실패: {type(exc).__name__}: {exc}\n"
            "SB3 모델은 이미 저장되어 있으며 최종 평가는 계속합니다."
        )
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="블록 배치 RL 학습")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="CSV 데이터 디렉토리 경로")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="모델 출력 디렉토리")
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="총 학습 타임스텝")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="학습률 (learning rate)")
    parser.add_argument("--n-steps", type=int, default=960,
                        help="PPO n_steps (913-block episode 근사, batch 64 배수)")
    parser.add_argument(
        "--grid-size",
        type=int,
        default=GRID_SIZE,
        choices=[GRID_SIZE],
        help="fixed observation grid size",
    )
    parser.add_argument("--batch-size", type=int, default=64,
                        help="미니배치 크기")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="PPO epochs per update")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="감가율 (discount factor)")
    parser.add_argument("--gae-lambda", type=float, default=0.98,
                        help="GAE bias-variance parameter")
    parser.add_argument("--n-eval", type=int, default=5,
                        help=(
                            "deprecated and ignored; original CSV evaluation "
                            "always runs exactly one episode"
                        ))
    parser.add_argument(
        "--eval-scenarios",
        type=str,
        default=None,
        help="fixed evaluation scenario JSON prepared by run_ablation.py",
    )
    parser.add_argument(
        "--holdout-eval-freq",
        type=int,
        default=50_000,
        help="fixed-holdout selection frequency; 0 disables selection",
    )
    parser.add_argument(
        "--holdout-selection-count",
        type=int,
        choices=[5],
        default=5,
        help="number of fixed scenarios used for periodic model selection",
    )
    parser.add_argument(
        "--final-holdout-report",
        action="store_true",
        help="evaluate selected best_model.sb3 on all fixed scenarios",
    )
    parser.add_argument(
        "--extractor",
        type=str,
        default="candidate-cnn",
        choices=["structured", "fixed-grid", "candidate-cnn", "raw-direct"],
        help="feature extractor ablation mode",
    )
    parser.add_argument(
        "--state-context",
        default="full",
        choices=["full", "current"],
        help="state context ablation mode",
    )
    parser.add_argument("--features-dim", type=int, default=256,
                        help="policy feature vector dimension")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="PyTorch device for policy training")
    parser.add_argument("--seed", type=int, default=0,
                        help="global and environment random seed")
    parser.add_argument(
        "--monthly-jitter",
        type=int,
        default=DEFAULT_MONTHLY_JITTER,
        help="balanced profile monthly count jitter (fixed total)",
    )
    parser.add_argument(
        "--empirical-profile-probability",
        type=float,
        default=DEFAULT_EMPIRICAL_PROFILE_PROBABILITY,
        help="probability of using the empirical monthly count profile",
    )
    parser.add_argument("--n-envs", type=int, default=1,
                        help="number of parallel training environments")
    parser.add_argument("--vec-env", type=str, default="auto",
                        choices=["auto", "dummy", "subproc"],
                        help="vector env backend when --n-envs > 1")
    parser.add_argument("--active-workspace-codes", type=str,
                        default=DEFAULT_ACTIVE_WORKSPACE_CODES,
                        help=(
                            "comma-separated active workspace codes. "
                            "Only selected workspaces enter observation and "
                            "action spaces. Use empty string to enable all "
                            "workspaces."
                        ))
    parser.add_argument("--resume-from", type=str, default=None,
                        help="이어 학습할 기존 SB3 모델 zip 경로(명시적)")
    parser.add_argument("--auto-resume", action="store_true", default=False,
                        help=("output-dir에 호환 가능한 기존 모델/체크포인트가 있으면 "
                              "자동으로 이어학습. 설정(추출기/관측/구조)이 다르면 중단."))
    parser.add_argument("--checkpoint-freq", type=int, default=0,
                        help=("중간 체크포인트 저장 주기(env step 단위). 0=비활성. "
                              "예: 10000. 세션 끊김 대비 + auto-resume 복구 지점."))
    parser.add_argument("--max-training-seconds", type=float, default=0.0)
    parser.add_argument("--wall-clock-state", default=None)
    parser.add_argument(
        "--wall-clock-heartbeat-seconds", type=float, default=300.0
    )
    parser.add_argument("--comparison-config-sha256", default=None)
    parser.add_argument("--comparison-baseline-sha256", default=None)
    parser.add_argument("--comparison-scenario-sha256", default=None)
    parser.add_argument("--comparison-split-sha256", default=None)
    parser.add_argument("--comparison-lock-sha256", default=None)
    parser.add_argument("--export-onnx", action="store_true", default=True,
                        help="ONNX export 수행")
    parser.add_argument("--no-export-onnx", action="store_false", dest="export_onnx")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
