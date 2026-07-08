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


DEFAULT_ACTIVE_WORKSPACE_CODES = "PE052,PE055,PE051,PE050,PE049"


def parse_workspace_codes(value: str | None) -> list[str] | None:
    if value is None:
        return None
    codes = [code.strip().upper() for code in value.split(",") if code.strip()]
    return codes or None


def mask_fn(env):
    return env.action_masks()


def build_policy_kwargs(
    extractor: str = "cnn",
    features_dim: int = 256,
    cnn_out_dim: int = 64,
    embed_dim: int = 64,
    num_heads: int = 4,
) -> dict:
    from alloc_env.cnn_extractor import (
        BlockSetAttentionCnnExtractor,
        OccupancyCnnExtractor,
        PointerAttentionCnnExtractor,
    )

    extractors = {
        "cnn": OccupancyCnnExtractor,
        "pointer-attn": PointerAttentionCnnExtractor,
        "block-attn": BlockSetAttentionCnnExtractor,
    }
    if extractor not in extractors:
        raise ValueError(
            f"Unknown extractor '{extractor}'. "
            f"Choose one of: {', '.join(extractors)}"
        )

    extractor_kwargs = {
        "features_dim": features_dim,
        "cnn_out_dim": cnn_out_dim,
    }
    if extractor in ("pointer-attn", "block-attn"):
        extractor_kwargs.update({
            "embed_dim": embed_dim,
            "num_heads": num_heads,
        })

    return {
        "features_extractor_class": extractors[extractor],
        "features_extractor_kwargs": extractor_kwargs,
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
        + n_workspaces * 3 * grid_size * grid_size
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
    generator=None,
    synthetic_n_blocks=None,
    vary_layout=True,
    grid_size=64,
    active_workspace_codes=None,
    n_future_blocks=0,
):
    """환경 팩토리 (SubprocVecEnv용)."""
    from alloc_env.alloc_env import BlockPlacementEnv

    def _init():
        return BlockPlacementEnv(
            blocks, workspaces, strategy,
            use_synthetic=use_synthetic,
            generator=generator,
            synthetic_n_blocks=synthetic_n_blocks,
            active_workspace_codes=active_workspace_codes,
            vary_layout=vary_layout,
            grid_size=grid_size,
            n_future_blocks=n_future_blocks,
        )
    return _init


def create_training_env(
    blocks,
    workspaces,
    strategy,
    generator,
    grid_size: int = 64,
    n_envs: int = 1,
    vec_env: str = "auto",
    active_workspace_codes=None,
    n_future_blocks: int = 0,
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
        "generator": generator,
        "synthetic_n_blocks": len(blocks),
        "active_workspace_codes": active_workspace_codes,
        "vary_layout": True,
        "grid_size": grid_size,
        "n_future_blocks": n_future_blocks,
    }

    if resolved_vec_env == "single":
        return ActionMasker(make_env(**env_kwargs)(), mask_fn)

    env_fns = [make_env(**env_kwargs) for _ in range(n_envs)]
    if resolved_vec_env == "dummy":
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns)


def create_evaluation_env(
    blocks,
    workspaces,
    strategy,
    grid_size: int = 64,
    active_workspace_codes=None,
    n_future_blocks: int = 0,
):
    """CSV 원본 블록으로 평가하는 마스크 적용 환경을 생성합니다."""
    from sb3_contrib.common.wrappers import ActionMasker

    from alloc_env.alloc_env import BlockPlacementEnv

    env = BlockPlacementEnv(
        blocks,
        workspaces,
        strategy,
        use_synthetic=False,
        active_workspace_codes=active_workspace_codes,
        grid_size=grid_size,
        n_future_blocks=n_future_blocks,
    )
    return ActionMasker(env, mask_fn)


# ── 체크포인트 / 자동 이어학습 유틸 ─────────────────────────────────

# 관측 공간·네트워크 구조에 영향을 주는 키. 이어학습하려면 이 값들이 모두 같아야 한다.
ARCH_CONFIG_KEYS = (
    "extractor",
    "n_future_blocks",
    "grid_size",
    "features_dim",
    "cnn_out_dim",
    "extractor_embed_dim",
    "extractor_heads",
    "active_workspace_codes",
)


def current_run_config(args, active_workspace_codes) -> dict:
    return {
        "extractor": args.extractor,
        "n_future_blocks": args.n_future_blocks,
        "grid_size": args.grid_size,
        "features_dim": args.features_dim,
        "cnn_out_dim": args.cnn_out_dim,
        "extractor_embed_dim": args.extractor_embed_dim,
        "extractor_heads": args.extractor_heads,
        "active_workspace_codes": list(active_workspace_codes or []),
    }


def write_run_config(output_dir, config) -> None:
    import json
    with open(Path(output_dir) / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def find_resumable_model(output_dir):
    """이어학습 가능한 최신 산출물 경로를 찾는다.

    우선순위: 완주 모델(block_placement_ppo.zip) > checkpoints/ 중 최대 step.
    완주 모델은 항상 가장 많은 step을 가지므로 우선한다. 완주 모델이 없으면
    (예: 세션 끊김/크래시) checkpoints/ 중 step이 가장 큰 파일을 쓴다.
    """
    import re
    output_dir = Path(output_dir)
    final = output_dir / "block_placement_ppo.zip"
    if final.exists():
        return final
    ckpt_dir = output_dir / "checkpoints"
    if ckpt_dir.is_dir():
        best, best_steps = None, -1
        for p in ckpt_dir.glob("*.zip"):
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


def resolve_resume_path(args, output_dir, current_config):
    """이어학습 경로를 결정한다.

    - --resume-from 이 명시되면 그 경로(없으면 에러).
    - 아니고 --auto-resume 이면 output-dir에서 호환 가능한 최신 모델을 자동 탐지.
      기존 설정과 구조가 다르면 관측/네트워크 불일치를 막기 위해 ValueError.
    반환: Path(이어학습) 또는 None(새로 학습).
    """
    import json

    if args.resume_from:
        resume_path = Path(args.resume_from).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume model not found: {resume_path}")
        return resume_path

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
    ok, bad_key = configs_compatible(saved_cfg, current_config)
    if not ok:
        raise ValueError(
            f"[auto-resume] 기존 모델과 설정이 다릅니다 (key='{bad_key}': "
            f"saved={saved_cfg.get(bad_key)} != now={current_config.get(bad_key)}). "
            f"관측/네트워크 구조가 달라 이어학습할 수 없습니다. "
            f"OUTPUT_DIR을 새 폴더로 바꾸거나 자동 이어학습을 끄세요."
        )
    print(f"[auto-resume] 호환 체크포인트 발견 → 이어학습: {candidate}")
    return candidate


def train(args):
    """MaskablePPO 학습 실행."""
    from sb3_contrib import MaskablePPO

    from alloc_env.data_loader import (
        load_workspaces, load_blocks, apply_allowable_block_patterns,
    )
    from alloc_env.strategy import BaseGridStrategy
    from alloc_env.callbacks import AllocationCallback, TrainingMetricsCallback
    from alloc_env.block_generator import SyntheticBlockGenerator

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

    print(f"블록 {len(blocks)}개, 작업장 {len(workspaces)}개")
    if active_workspace_codes:
        print(
            f"Active workspaces: {len(active_workspace_codes)}/{len(workspaces)} "
            f"({', '.join(active_workspace_codes)})"
        )
    else:
        print(f"Active workspaces: all {len(workspaces)}")

    # ── 2. Synthetic 블록 생성기 ─────────────────────────────────
    generator = SyntheticBlockGenerator.from_csv(blk_csv)
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
        active_workspace_codes=active_workspace_codes,
        n_future_blocks=args.n_future_blocks,
    )
    resolved_vec_env = resolve_vec_env_type(args.vec_env, args.n_envs)

    if args.extractor == "block-attn" and args.n_future_blocks == 0:
        print(
            "[경고] --extractor block-attn 인데 --n-future-blocks 0 입니다. "
            "미래 블록 없이는 블록-집합 attention이 단일 토큰으로 퇴화하여 "
            "MLP와 유사하게 동작합니다. lookahead 이점을 보려면 "
            "--n-future-blocks 3~5 를 권장합니다."
        )

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
        cnn_out_dim=args.cnn_out_dim,
        embed_dim=args.extractor_embed_dim,
        num_heads=args.extractor_heads,
    )
    # 이어학습 경로 결정: --resume-from(명시) 우선, 없으면 --auto-resume 자동 탐지
    run_config = current_run_config(args, active_workspace_codes)
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
            policy_kwargs=policy_kwargs,
            device=args.device,
            tensorboard_log=str(output_dir / "tb_logs"),
        )

    # ── 5. 콜백 설정 ──────────────────────────────────────────────
    callback = [
        AllocationCallback(log_dir=args.output_dir, verbose=1),
        TrainingMetricsCallback(log_dir=args.output_dir, verbose=1),
    ]
    if args.checkpoint_freq > 0:
        from stable_baselines3.common.callbacks import CheckpointCallback

        # SB3 CheckpointCallback은 콜백 호출 횟수 기준이라 n_envs로 나눠 step 단위를 맞춘다.
        save_freq = max(args.checkpoint_freq // max(args.n_envs, 1), 1)
        callback.append(CheckpointCallback(
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
    sb3_path = str(output_dir / "block_placement_ppo")
    model.save(sb3_path)
    print(f"\nSB3 모델 저장: {sb3_path}")

    # ── 8. ONNX export ───────────────────────────────────────────
    if args.export_onnx:
        onnx_path = str(output_dir / "block_placement_ppo.onnx")
        export_to_onnx(model, env, onnx_path)
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
        active_workspace_codes=active_workspace_codes,
        n_future_blocks=args.n_future_blocks,
    )
    evaluate(model, eval_env, n_eval=args.n_eval)


def evaluate(model, env, n_eval: int = 5):
    """학습된 모델로 n_eval 에피소드 평가."""
    from sb3_contrib import MaskablePPO

    rewards = []
    terminal_rewards = []
    for ep in range(n_eval):
        obs, info = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action_masks = env.action_masks() if hasattr(env, 'action_masks') else None
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        terminal_reward = info.get("terminal_reward", total_reward)
        terminal_rewards.append(terminal_reward)
        print(
            f"  Episode {ep+1}: "
            f"total = {total_reward:.2f}, terminal = {terminal_reward:.2f}"
        )

    mean_r = np.mean(rewards)
    mean_terminal = np.mean(terminal_rewards)
    print(
        f"\n  평균 reward: total={mean_r:.2f}, "
        f"terminal={mean_terminal:.2f} (n={n_eval})"
    )
    return mean_r


def export_to_onnx(model, env, onnx_path: str):
    """SB3 모델을 ONNX 형식으로 export (Dict obs 대응, 동적 키).

    관측 키 집합을 하드코딩하지 않고 observation_space에서 읽어오므로,
    n_future_blocks > 0으로 future_blocks/future_mask가 추가되어도 그대로 export된다.
    """
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

    torch.onnx.export(
        wrapper,
        dummy_inputs,
        onnx_path,
        input_names=obs_keys,
        output_names=["action_logits"],
        dynamic_axes=dynamic_axes,
        opset_version=18,
    )

    # 검증
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"  ONNX inputs: {[inp.name for inp in onnx_model.graph.input]}")


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
    parser.add_argument("--n-eval", type=int, default=5,
                        help="평가 에피소드 수")
    parser.add_argument("--extractor", type=str, default="cnn",
                        choices=["cnn", "pointer-attn", "block-attn"],
                        help="feature extractor: cnn, pointer-attn, or "
                             "block-attn (블록-집합 attention, 미래 lookahead)")
    parser.add_argument("--n-future-blocks", type=int, default=0,
                        help="관측에 포함할 미래 블록 개수 (0=미포함, 기존 계약 유지). "
                             "block-attn extractor와 함께 3~5 권장.")
    parser.add_argument("--features-dim", type=int, default=256,
                        help="policy feature vector dimension")
    parser.add_argument("--cnn-out-dim", type=int, default=64,
                        help="workspace CNN output dimension")
    parser.add_argument("--extractor-embed-dim", type=int, default=64,
                        help="pointer-attn token embedding dimension")
    parser.add_argument("--extractor-heads", type=int, default=4,
                        help="pointer-attn attention head count")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="PyTorch device for policy training")
    parser.add_argument("--n-envs", type=int, default=1,
                        help="number of parallel training environments")
    parser.add_argument("--vec-env", type=str, default="auto",
                        choices=["auto", "dummy", "subproc"],
                        help="vector env backend when --n-envs > 1")
    parser.add_argument("--active-workspace-codes", type=str,
                        default=DEFAULT_ACTIVE_WORKSPACE_CODES,
                        help=(
                            "comma-separated active workspace codes. "
                            "Observation/action shape keeps all workspaces; "
                            "inactive workspaces are masked. Use empty string "
                            "to enable all workspaces."
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
