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


def mask_fn(env):
    return env.action_masks()


def build_policy_kwargs(
    extractor: str = "cnn",
    features_dim: int = 256,
    cnn_out_dim: int = 64,
    embed_dim: int = 64,
    num_heads: int = 4,
):
    from alloc_env.cnn_extractor import (
        OccupancyCnnExtractor,
        PointerAttentionCnnExtractor,
    )

    extractors = {
        "cnn": OccupancyCnnExtractor,
        "pointer-attn": PointerAttentionCnnExtractor,
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
    if extractor == "pointer-attn":
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
) -> float:
    obs_bytes = (10 + n_workspaces * 3 * grid_size * grid_size + n_workspaces * 3) * 4
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
):
    """환경 팩토리 (SubprocVecEnv용)."""
    from alloc_env.alloc_env import BlockPlacementEnv

    def _init():
        return BlockPlacementEnv(
            blocks, workspaces, strategy,
            use_synthetic=use_synthetic,
            generator=generator,
            synthetic_n_blocks=synthetic_n_blocks,
            vary_layout=vary_layout,
            grid_size=grid_size,
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
        "vary_layout": True,
        "grid_size": grid_size,
    }

    if resolved_vec_env == "single":
        return ActionMasker(make_env(**env_kwargs)(), mask_fn)

    env_fns = [make_env(**env_kwargs) for _ in range(n_envs)]
    if resolved_vec_env == "dummy":
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns)


def create_evaluation_env(blocks, workspaces, strategy, grid_size: int = 64):
    """CSV 원본 블록으로 평가하는 마스크 적용 환경을 생성합니다."""
    from sb3_contrib.common.wrappers import ActionMasker

    from alloc_env.alloc_env import BlockPlacementEnv

    env = BlockPlacementEnv(
        blocks,
        workspaces,
        strategy,
        use_synthetic=False,
        grid_size=grid_size,
    )
    return ActionMasker(env, mask_fn)


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

    print(f"블록 {len(blocks)}개, 작업장 {len(workspaces)}개")

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
    )
    resolved_vec_env = resolve_vec_env_type(args.vec_env, args.n_envs)

    # 메모리 사용량 예측
    N = len(workspaces)
    G = args.grid_size
    buffer_mb = estimate_rollout_buffer_mb(N, G, args.n_steps, args.n_envs)
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
    print(f"Feature extractor: {args.extractor}")
    if args.resume_from:
        resume_path = Path(args.resume_from).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume model not found: {resume_path}")
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

    # ── 6. 학습 ──────────────────────────────────────────────────
    print(f"\n학습 시작: {args.timesteps} timesteps")
    print(f"TensorBoard: tensorboard --logdir {Path(args.output_dir) / 'tb_logs'}")
    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=True,
        callback=callback,
        reset_num_timesteps=not bool(args.resume_from),
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
        blocks, workspaces, strategy, grid_size=args.grid_size
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
    """SB3 모델을 ONNX 형식으로 export (Dict obs 대응)."""
    import torch
    import onnx

    policy = model.policy
    obs_space = env.observation_space

    # Dict obs의 각 키별 더미 입력 생성
    dummy_obs = {}
    if hasattr(obs_space, 'spaces'):
        # Dict observation space
        for key, space in obs_space.spaces.items():
            dummy_obs[key] = torch.zeros(1, *space.shape, device=policy.device)
    else:
        # Flat observation space (fallback)
        dummy_obs = torch.zeros(1, obs_space.shape[0], device=policy.device)

    # Actor 네트워크만 export (추론에 필요한 부분)
    class PolicyWrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, block, grids, ws_meta):
            obs_dict = {"block": block, "grids": grids, "ws_meta": ws_meta}
            features = self.policy.extract_features(
                obs_dict, self.policy.pi_features_extractor
            )
            latent_pi = self.policy.mlp_extractor.forward_actor(features)
            return self.policy.action_net(latent_pi)

    wrapper = PolicyWrapper(policy)
    wrapper.eval()

    # Dict obs를 개별 인자로 전달
    dummy_inputs = (
        dummy_obs["block"],
        dummy_obs["grids"],
        dummy_obs["ws_meta"],
    )

    torch.onnx.export(
        wrapper,
        dummy_inputs,
        onnx_path,
        input_names=["block", "grids", "ws_meta"],
        output_names=["action_logits"],
        dynamic_axes={
            "block": {0: "batch"},
            "grids": {0: "batch"},
            "ws_meta": {0: "batch"},
            "action_logits": {0: "batch"},
        },
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
                        choices=["cnn", "pointer-attn"],
                        help="feature extractor: cnn or pointer-attn")
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
    parser.add_argument("--resume-from", type=str, default=None,
                        help="이어 학습할 기존 SB3 모델 zip 경로")
    parser.add_argument("--export-onnx", action="store_true", default=True,
                        help="ONNX export 수행")
    parser.add_argument("--no-export-onnx", action="store_false", dest="export_onnx")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
