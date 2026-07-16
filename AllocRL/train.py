"""
블록 배치 강화학습 - CNN+MaskablePPO 학습 + ONNX export.

사용법:
    py train.py --data-dir ./data --timesteps 100000

의존성:
    pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Windows cp949 콘솔에서 Unicode 출력 에러 방지
if sys.platform == "win32" and os.environ.get("PYTHONIOENCODING") is None:
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback


DEFAULT_ACTIVE_WORKSPACE_CODES = "PE052,PE055,PE051,PE050,PE049"
OBSERVATION_SCHEMA_VERSION = 2
REWARD_SCHEMA_VERSION = 2
MODEL_FILENAME = "block_placement_ppo.sb3"
LEGACY_MODEL_FILENAME = "block_placement_ppo.zip"


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

    extractors = {
        "structured": StructuredExtractor,
        "fixed-grid": FixedGridExtractor,
        "candidate-cnn": CandidateCnnExtractor,
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
    }


def estimate_rollout_buffer_mb(
    n_workspaces: int,
    grid_size: int,
    n_steps: int,
    n_envs: int = 1,
    n_future_blocks: int = 0,
) -> float:
    from alloc_env.alloc_env import FUTURE_BLOCK_FEATURE_DIM

    future_floats = (
        n_future_blocks * (FUTURE_BLOCK_FEATURE_DIM + 1)  # future_blocks + mask
        if n_future_blocks > 0 else 0
    )
    obs_bytes = (
        10
        + n_workspaces * 4 * grid_size * grid_size
        + n_workspaces * 3
        + future_floats
    ) * 4
    return obs_bytes * n_steps * n_envs / 1024 / 1024


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


def make_env(
    blocks,
    workspaces,
    strategy,
    use_synthetic=False,
    generator_dist=None,
    synthetic_n_blocks=None,
    vary_layout=True,
    grid_size=64,
    n_future_blocks=4,
    env_seed=0,
):
    """환경 팩토리 (SubprocVecEnv용)."""
    from alloc_env.alloc_env import BlockPlacementEnv
    from alloc_env.block_generator import SyntheticBlockGenerator

    def _init():
        local_generator = (
            SyntheticBlockGenerator(dist=generator_dist, seed=env_seed)
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
            n_future_blocks=n_future_blocks,
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
    grid_size: int = 64,
    n_envs: int = 1,
    vec_env: str = "auto",
    n_future_blocks: int = 4,
    seed: int = 0,
):
    """Create the training env, optionally vectorized for parallel rollout."""
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    resolved_vec_env = resolve_vec_env_type(vec_env, n_envs)
    env_kwargs = {
        "blocks": blocks,
        "workspaces": workspaces,
        "strategy": strategy,
        "use_synthetic": True,
        "generator_dist": generator.dist if generator is not None else None,
        "synthetic_n_blocks": len(blocks),
        "vary_layout": True,
        "grid_size": grid_size,
        "n_future_blocks": n_future_blocks,
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
    grid_size: int = 64,
    n_future_blocks: int = 4,
    seed: int = 0,
):
    """CSV 원본 블록으로 평가하는 마스크 적용 환경을 생성합니다."""
    from sb3_contrib.common.wrappers import ActionMasker

    from alloc_env.alloc_env import BlockPlacementEnv

    env = BlockPlacementEnv(
        blocks,
        workspaces,
        strategy,
        use_synthetic=False,
        grid_size=grid_size,
        n_future_blocks=n_future_blocks,
    )
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return ActionMasker(env, mask_fn)


# ── 체크포인트 / 자동 이어학습 유틸 ─────────────────────────────────

# 관측 공간·네트워크 구조에 영향을 주는 키. 이어학습하려면 이 값들이 모두 같아야 한다.
ARCH_CONFIG_KEYS = (
    "observation_schema_version",
    "reward_schema_version",
    "extractor",
    "n_future_blocks",
    "grid_size",
    "features_dim",
    "active_workspace_codes",
)


def current_run_config(args, active_workspace_codes) -> dict:
    return {
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "extractor": args.extractor,
        "n_future_blocks": args.n_future_blocks,
        "grid_size": args.grid_size,
        "features_dim": args.features_dim,
        "active_workspace_codes": list(active_workspace_codes or []),
        "seed": args.seed,
        "eval_scenarios": getattr(args, "eval_scenarios", None),
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

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Run configuration must be a JSON object: {config_path}")
    return config


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


def find_resumable_model(output_dir):
    """이어학습 가능한 최신 산출물 경로를 찾는다.

    우선순위: 완주 모델(.sb3, legacy .zip) > checkpoints/ 중 최대 step.
    완주 모델은 항상 가장 많은 step을 가지므로 우선한다. 완주 모델이 없으면
    (예: 세션 끊김/크래시) checkpoints/ 중 step이 가장 큰 파일을 쓴다.
    """
    import re
    import zipfile

    output_dir = Path(output_dir)
    for filename in (MODEL_FILENAME, LEGACY_MODEL_FILENAME):
        final = output_dir / filename
        if final.is_file():
            return resolve_model_archive_path(final)
    ckpt_dir = output_dir / "checkpoints"
    if ckpt_dir.is_dir():
        best, best_steps = None, -1
        for pattern in ("*.sb3", "*.zip"):
            for p in ckpt_dir.glob(pattern):
                if not zipfile.is_zipfile(p):
                    continue
                m = re.search(r"(\d+)_steps", p.stem)
                steps = int(m.group(1)) if m else 0
                if steps > best_steps:
                    best_steps, best = steps, p
        return best
    return None


def configs_compatible(saved: dict, current: dict):
    for key in ARCH_CONFIG_KEYS:
        if saved.get(key) != current.get(key):
            return False, key
    return True, None


def require_compatible_run_config(
    saved: dict,
    current: dict,
    source: str,
) -> None:
    compatible, bad_key = configs_compatible(saved, current)
    if compatible:
        return
    raise ValueError(
        f"[{source}] Saved model configuration is incompatible "
        f"(key='{bad_key}': saved={saved.get(bad_key)} != "
        f"current={current.get(bad_key)}). Use a matching configuration "
        "or a new output directory."
    )


def resolve_resume_path(args, output_dir, current_config):
    """이어학습 경로를 결정한다.

    - --resume-from 이 명시되면 그 경로(없으면 에러).
    - 아니고 --auto-resume 이면 output-dir에서 호환 가능한 최신 모델을 자동 탐지.
      기존 설정과 구조가 다르면 관측/네트워크 불일치를 막기 위해 ValueError.
    반환: Path(이어학습) 또는 None(새로 학습).
    """
    import json

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
    import csv

    if not rows:
        raise ValueError("At least one evaluation metric row is required")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_fixed_scenarios(
    model,
    scenario_records: list[dict],
    grid_size: int,
    n_future_blocks: int,
    workspace_codes: list[str] | None = None,
) -> list[dict]:
    from alloc_env.data_loader import select_workspaces_in_order
    from alloc_env.strategy import BaseGridStrategy
    from evaluation_scenarios import materialize_scenario

    rows = []
    for scenario in scenario_records:
        strategy = BaseGridStrategy(step=5.0)
        blocks, workspaces = materialize_scenario(scenario, strategy)
        workspaces = select_workspaces_in_order(
            workspaces, workspace_codes
        )
        env = create_evaluation_env(
            blocks=blocks,
            workspaces=workspaces,
            strategy=strategy,
            grid_size=grid_size,
            n_future_blocks=n_future_blocks,
            seed=int(scenario["seed"]),
        )
        try:
            metrics = evaluate(
                model, env, n_eval=1, return_metrics=True
            )
        finally:
            env.close()
        rows.append({"seed": int(scenario["seed"]), **metrics})
    return rows


def train(args):
    """MaskablePPO 학습 실행."""
    from sb3_contrib import MaskablePPO

    from alloc_env.data_loader import (
        apply_allowable_block_patterns,
        load_blocks,
        load_workspaces,
        select_workspaces,
    )
    from alloc_env.strategy import BaseGridStrategy
    from alloc_env.callbacks import AllocationCallback, TrainingMetricsCallback
    from alloc_env.block_generator import SyntheticBlockGenerator

    fixed_scenarios = load_requested_evaluation_scenarios(
        getattr(args, "eval_scenarios", None)
    )
    set_global_seed(args.seed)

    data_dir = Path(args.data_dir)
    ws_csv   = str(data_dir / "선행건조 작업장 기준정보.csv")
    lot_csv  = str(data_dir / "선행건조 지번 기준정보.csv")
    blk_csv  = str(data_dir / "블록데이터.csv")

    print("=" * 60)
    print("  블록 배치 강화학습 - MaskablePPO")
    print("=" * 60)

    # ── 1. 데이터 로드 ────────────────────────────────────────────
    strategy = BaseGridStrategy(step=5.0)
    workspaces = load_workspaces(ws_csv, lot_csv, strategy)
    apply_allowable_block_patterns(workspaces)
    blocks = load_blocks(blk_csv, workspaces)
    active_workspace_codes = parse_workspace_codes(args.active_workspace_codes)
    total_workspace_count = len(workspaces)
    workspaces = select_workspaces(workspaces, active_workspace_codes)

    print(f"블록 {len(blocks)}개, 작업장 {len(workspaces)}개")
    if active_workspace_codes:
        print(
            f"Active workspaces: {len(workspaces)}/{total_workspace_count} "
            f"({', '.join(ws.code for ws in workspaces)})"
        )
    else:
        print(f"Active workspaces: all {len(workspaces)}")

    # ── 2. Synthetic 블록 생성기 ─────────────────────────────────
    generator = SyntheticBlockGenerator.from_csv(blk_csv, seed=args.seed)
    print("[Synthetic] CSV 분포 기반 블록 생성기 초기화 완료")

    # ── 3. 환경 생성 (학습: synthetic, 평가: CSV 원본) ────────────
    env = create_training_env(
        blocks,
        workspaces,
        strategy,
        generator,
        grid_size=args.grid_size,
        n_envs=args.n_envs,
        vec_env=args.vec_env,
        n_future_blocks=args.n_future_blocks,
        seed=args.seed,
    )
    resolved_vec_env = resolve_vec_env_type(args.vec_env, args.n_envs)

    # 메모리 사용량 예측
    N = len(workspaces)
    G = args.grid_size
    buffer_mb = estimate_rollout_buffer_mb(
        N, G, args.n_steps, args.n_envs, args.n_future_blocks
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

    # ── 3. 출력 디렉토리 사전 생성 ────────────────────────────────
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 4. 모델 생성 (CNN+MLP 하이브리드) ─────────────────────────
    policy_kwargs = build_policy_kwargs(
        extractor=args.extractor,
        features_dim=args.features_dim,
    )
    # 이어학습 경로 결정: --resume-from(명시) 우선, 없으면 --auto-resume 자동 탐지
    run_config = current_run_config(
        args, [workspace.code for workspace in workspaces]
    )
    resume_path = resolve_resume_path(args, output_dir, run_config)
    is_resume = resume_path is not None
    # 현재 설정 기록 (다음 auto-resume 호환성 검사 + 크래시 후 복구용)
    write_run_config(output_dir, run_config)

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

    # ── 5. 콜백 설정 ──────────────────────────────────────────────
    callback = [
        AllocationCallback(
            log_dir=args.output_dir, verbose=1, append=is_resume
        ),
        TrainingMetricsCallback(
            log_dir=args.output_dir, verbose=1, append=is_resume
        ),
    ]
    if args.checkpoint_freq > 0:
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

    # ── 6. 학습 ──────────────────────────────────────────────────
    # 이어학습이면 reset_num_timesteps=False → 기존 step 뒤에 args.timesteps 만큼 추가 학습.
    print(f"\n학습 시작: {args.timesteps} timesteps "
          f"({'이어학습(추가)' if is_resume else '신규'})")
    print(f"TensorBoard: tensorboard --logdir {Path(args.output_dir) / 'tb_logs'}")
    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=True,
        callback=callback,
        reset_num_timesteps=not is_resume,
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
    eval_env = create_evaluation_env(
        blocks,
        workspaces,
        strategy,
        grid_size=args.grid_size,
        n_future_blocks=args.n_future_blocks,
        seed=args.seed,
    )
    try:
        csv_metrics = evaluate(
            model, eval_env, n_eval=args.n_eval, return_metrics=True
        )
    finally:
        eval_env.close()
    write_evaluation_metrics(
        output_dir / "evaluation_csv.csv",
        [{"source": "original_csv", **csv_metrics}],
    )

    if fixed_scenarios is not None:
        print("\n  Fixed evaluation scenarios")
        scenario_rows = evaluate_fixed_scenarios(
            model,
            fixed_scenarios,
            grid_size=args.grid_size,
            n_future_blocks=args.n_future_blocks,
            workspace_codes=[
                workspace.code for workspace in workspaces
            ],
        )
        write_evaluation_metrics(
            output_dir / "evaluation_scenarios.csv", scenario_rows
        )


def evaluate(model, env, n_eval: int = 5, return_metrics: bool = False):
    """Evaluate deterministic episodes and optionally return quality metrics."""
    from alloc_env.alloc_env import DELAY_THRESHOLD
    from alloc_env.simulator import SimulationResult
    from evaluation_scenarios import compute_retained_choice_ratio

    if n_eval < 1:
        raise ValueError("n_eval must be at least 1")

    rewards = []
    terminal_scores = []
    dropout_rates = []
    mean_delay_days = []
    delayed_counts = []
    retained_choice_ratios = []
    for ep in range(n_eval):
        obs, info = env.reset()
        total_reward = 0.0
        done = False
        episode_choice_ratios = []
        diagnostic_env = getattr(env, "unwrapped", env)
        while not done:
            action_masks = env.action_masks() if hasattr(env, 'action_masks') else None
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            if hasattr(diagnostic_env, "future_workspace_choice_indices"):
                future_indices = (
                    diagnostic_env.future_workspace_choice_indices()
                )
                choices_before = (
                    diagnostic_env.future_workspace_choice_count(
                        future_indices
                    )
                )
            else:
                future_indices = []
                choices_before = 0
            if hasattr(
                diagnostic_env,
                "future_workspace_choice_count_after_action",
            ):
                choices_after = (
                    diagnostic_env.future_workspace_choice_count_after_action(
                        int(action), future_indices
                    )
                )
            else:
                choices_after = None
            obs, reward, terminated, truncated, info = env.step(action)
            if (
                choices_after is None
                and hasattr(
                    diagnostic_env, "future_workspace_choice_count"
                )
            ):
                choices_after = (
                    diagnostic_env.future_workspace_choice_count(
                        future_indices
                    )
                )
            elif choices_after is None:
                choices_after = 0
            episode_choice_ratios.append(
                compute_retained_choice_ratio(
                    choices_before, choices_after
                )
            )
            total_reward += float(reward)
            done = terminated or truncated

        rewards.append(total_reward)
        terminal_score = info.get(
            "terminal_score", info.get("terminal_reward", total_reward)
        )
        terminal_scores.append(terminal_score)
        result = info.get("raw_result")
        delay_days = list(result.delay_days) if result is not None else []
        dropout_count = sum(
            delay == SimulationResult.DROPOUT for delay in delay_days
        )
        placed_delays = [
            delay
            for delay in delay_days
            if delay != SimulationResult.DROPOUT
        ]
        dropout_rates.append(
            dropout_count / len(delay_days) if delay_days else 0.0
        )
        mean_delay_days.append(
            float(np.mean(placed_delays)) if placed_delays else 0.0
        )
        delayed_counts.append(
            sum(delay > DELAY_THRESHOLD for delay in placed_delays)
        )
        retained_choice_ratios.append(
            float(np.mean(episode_choice_ratios))
            if episode_choice_ratios
            else 1.0
        )
        print(
            f"  Episode {ep+1}: "
            f"total reward = {total_reward:.2f}, "
            f"terminal score = {terminal_score:.2f}"
        )

    metrics = {
        "mean_reward": float(np.mean(rewards)),
        "mean_terminal_score": float(np.mean(terminal_scores)),
        "mean_dropout_rate": float(np.mean(dropout_rates)),
        "mean_delay_days": float(np.mean(mean_delay_days)),
        "mean_delayed_count": float(np.mean(delayed_counts)),
        "mean_retained_choice_ratio": float(
            np.mean(retained_choice_ratios)
        ),
    }
    print(
        f"\n  Mean evaluation: reward={metrics['mean_reward']:.2f}, "
        f"terminal score={metrics['mean_terminal_score']:.2f}, "
        f"dropout={metrics['mean_dropout_rate']:.1%}, "
        f"retained choices={metrics['mean_retained_choice_ratio']:.3f} "
        f"(n={n_eval})"
    )
    return metrics if return_metrics else metrics["mean_reward"]


def export_to_onnx(model, env, onnx_path: str):
    """SB3 모델을 ONNX 형식으로 export (Dict obs 대응, 동적 키).

    관측 키 집합을 하드코딩하지 않고 observation_space에서 읽어오므로,
    n_future_blocks > 0으로 future_blocks/future_mask가 추가되어도 그대로 export된다.
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

    # 키 순서 고정 (gymnasium Dict는 키를 정렬 순으로 유지 → 결정적).
    obs_keys = list(obs_space.spaces.keys())
    dummy_obs = {
        key: torch.zeros(1, *space.shape, device=policy.device)
        for key, space in obs_space.spaces.items()
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
            return self.policy.action_net(latent_pi)

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
    parser.add_argument("--n-steps", type=int, default=554,
                        help="PPO n_steps (에피소드 길이×2 권장)")
    parser.add_argument("--grid-size", type=int, default=64,
                        help="점유 그리드 해상도 (64 or 128, 메모리에 영향)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="미니배치 크기")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="PPO epochs per update")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="감가율 (discount factor)")
    parser.add_argument("--gae-lambda", type=float, default=0.98,
                        help="GAE bias-variance parameter")
    parser.add_argument("--n-eval", type=int, default=5,
                        help="평가 에피소드 수")
    parser.add_argument(
        "--eval-scenarios",
        type=str,
        default=None,
        help="fixed evaluation scenario JSON prepared by run_ablation.py",
    )
    parser.add_argument(
        "--extractor",
        type=str,
        default="candidate-cnn",
        choices=["structured", "fixed-grid", "candidate-cnn"],
        help="feature extractor ablation mode",
    )
    parser.add_argument("--n-future-blocks", type=int, default=4,
                        help="ordered future blocks included in observations")
    parser.add_argument("--features-dim", type=int, default=256,
                        help="policy feature vector dimension")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="PyTorch device for policy training")
    parser.add_argument("--seed", type=int, default=0,
                        help="global and environment random seed")
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
    parser.add_argument("--export-onnx", action="store_true", default=True,
                        help="ONNX export 수행")
    parser.add_argument("--no-export-onnx", action="store_false", dest="export_onnx")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
