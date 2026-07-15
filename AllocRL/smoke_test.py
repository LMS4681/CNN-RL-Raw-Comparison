# -*- coding: utf-8 -*-
"""
CNN + Dict Obs 통합 스모크 테스트.

1. OccupancyGridRenderer 렌더링 검증
2. Three approved feature extractors forward pass 검증
3. BlockPlacementEnv reset/step 동작 검증
4. SyntheticBlockGenerator.generate_workspaces() 검증
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
BASE = Path(__file__).resolve().parent
os.chdir(BASE)

import numpy as np
from datetime import date

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}")


print("=" * 60)
print("  CNN + Dict Obs 통합 스모크 테스트")
print("=" * 60)

# ── 1. 데이터 로드 ────────────────────────────────────────────────
print("\n[1] 데이터 로드")
from alloc_env.data_loader import load_workspaces, load_blocks
from alloc_env.strategy import BaseGridStrategy

strategy = BaseGridStrategy(step=5.0)
data_dir = BASE / "data"
ws_csv = str(data_dir / "선행건조 작업장 기준정보.csv")
lot_csv = str(data_dir / "선행건조 지번 기준정보.csv")
blk_csv = str(data_dir / "블록데이터.csv")

workspaces = load_workspaces(ws_csv, lot_csv, strategy)
blocks = load_blocks(blk_csv, workspaces)
check("작업장 로드", len(workspaces) > 0, f"{len(workspaces)}개")
check("블록 로드", len(blocks) > 0, f"{len(blocks)}개")
print(f"  작업장 {len(workspaces)}개, 블록 {len(blocks)}개")

# ── 2. 점유 그리드 렌더러 ─────────────────────────────────────────
print("\n[2] OccupancyGridRenderer 테스트")
from alloc_env.occupancy_grid import OccupancyGridRenderer, GridCache

renderer = OccupancyGridRenderer(grid_size=128)
env_date = date(2026, 4, 1)

# 단일 작업장 렌더링
ws0 = workspaces[0]
grid = renderer.render(ws0, env_date)
check("그리드 shape", grid.shape == (3, 128, 128), f"got {grid.shape}")
check("그리드 dtype", grid.dtype == np.float32)
check("Ch0 범위 [0,1]", 0.0 <= grid[0].min() and grid[0].max() <= 1.0)
check("Ch1 범위 [0,1]", 0.0 <= grid[1].min() and grid[1].max() <= 1.0)
check("Ch2 (경계 마스크) 존재", grid[2].sum() > 0, f"sum={grid[2].sum():.0f}")

# 전체 작업장 렌더링
grids = renderer.render_all(workspaces, env_date)
check("전체 그리드 shape",
      grids.shape == (len(workspaces), 3, 128, 128),
      f"got {grids.shape}")

# 스케일 값
scale = renderer.compute_scale_value(ws0)
check("scale > 0", scale > 0, f"scale={scale:.4f} m/px")
print(f"  작업장 {ws0.code}: {ws0.length:.0f}×{ws0.breadth:.0f}m, scale={scale:.4f} m/px")

# 캐시 테스트
cache = GridCache(renderer, len(workspaces))
cache.invalidate_all()
cached = cache.get_grids(workspaces, env_date)
check("캐시 그리드 shape", cached.shape == grids.shape)
# 두 번째 호출 (캐시 hit)
cached2 = cache.get_grids(workspaces, env_date)
check("캐시 hit 일관성", np.allclose(cached, cached2))

# ── 3. CNN Feature Extractor ──────────────────────────────────────
print("\n[3] Feature extractor 테스트")
import torch
import gymnasium as gym
from gymnasium import spaces
from alloc_env.alloc_env import FUTURE_BLOCK_FEATURE_DIM
from alloc_env.cnn_extractor import (
    CandidateCnnExtractor,
    FixedGridExtractor,
    StructuredExtractor,
)

N = len(workspaces)
G = 128
K = 4
extractor_grid_size = 32
obs_space = spaces.Dict({
    "block": spaces.Box(0, 1, shape=(10,), dtype=np.float32),
    "grids": spaces.Box(
        0, 1, shape=(N, 4, extractor_grid_size, extractor_grid_size),
        dtype=np.float32,
    ),
    "ws_meta": spaces.Box(0, 1, shape=(N, 3), dtype=np.float32),
    "future_blocks": spaces.Box(
        0, 1, shape=(K, FUTURE_BLOCK_FEATURE_DIM), dtype=np.float32,
    ),
    "future_mask": spaces.Box(0, 1, shape=(K,), dtype=np.float32),
})

dummy_obs = {
    "block": torch.randn(1, 10),
    "grids": torch.randn(
        1, N, 4, extractor_grid_size, extractor_grid_size
    ),
    "ws_meta": torch.randn(1, N, 3),
    "future_blocks": torch.randn(1, K, FUTURE_BLOCK_FEATURE_DIM),
    "future_mask": torch.ones(1, K),
}
for extractor_class in (
    StructuredExtractor,
    FixedGridExtractor,
    CandidateCnnExtractor,
):
    extractor = extractor_class(obs_space, features_dim=256)
    feat = extractor(dummy_obs)
    name = extractor_class.__name__
    check(f"{name} output shape", feat.shape == (1, 256), f"got {feat.shape}")
    check(f"{name} output finite", torch.isfinite(feat).all().item())

# ── 4. 작업장 레이아웃 합성 생성 ──────────────────────────────────
print("\n[4] generate_workspaces() 테스트")
from alloc_env.block_generator import SyntheticBlockGenerator

gen = SyntheticBlockGenerator.from_csv(blk_csv, seed=42)
synth_ws = gen.generate_workspaces(workspaces, scale_range=(0.7, 1.3))
check("합성 작업장 수 보존", len(synth_ws) == len(workspaces))

# 크기가 변형되었는지 확인
changed = any(
    abs(s.length - o.length) > 0.1 or abs(s.breadth - o.breadth) > 0.1
    for s, o in zip(synth_ws, workspaces)
)
check("크기 변형 발생", changed, "원본과 동일")

# 범위 확인
for s, o in zip(synth_ws, workspaces):
    ratio_l = s.length / max(o.length, 0.1)
    ratio_b = s.breadth / max(o.breadth, 0.1)
    in_range = (0.5 <= ratio_l <= 1.5) and (0.5 <= ratio_b <= 1.5)
    if not in_range:
        check(f"스케일 범위 {s.code}", False,
              f"L:{ratio_l:.2f} B:{ratio_b:.2f}")
        break
else:
    check("스케일 범위 전체 유효 (0.5~1.5)", True)

# ── 5. 환경 reset/step ────────────────────────────────────────────
print("\n[5] BlockPlacementEnv 통합 테스트")
from alloc_env.alloc_env import BlockPlacementEnv

env = BlockPlacementEnv(
    blocks[:20],  # 빠른 테스트를 위해 20개만
    workspaces,
    strategy,
    use_synthetic=False,
    grid_size=G,
)

obs, info = env.reset()
check("obs Dict 키", set(obs.keys()) == {"block", "grids", "ws_meta"})
check("obs['block'] shape", obs["block"].shape == (10,), f"got {obs['block'].shape}")
check("obs['grids'] shape",
      obs["grids"].shape == (N, 4, G, G),
      f"got {obs['grids'].shape}")
check("obs['ws_meta'] shape",
      obs["ws_meta"].shape == (N, 3),
      f"got {obs['ws_meta'].shape}")
check("obs 값 범위 [0,1]",
      obs["block"].min() >= -0.01 and obs["block"].max() <= 1.01)

# step 실행
mask = env.action_masks()
valid_actions = np.where(mask)[0]
check("마스크에 유효 액션 존재", len(valid_actions) > 0, f"{len(valid_actions)}개")

if len(valid_actions) > 0:
    action = int(valid_actions[0])
    obs2, reward, terminated, truncated, info2 = env.step(action)
    check("step 후 obs 키", set(obs2.keys()) == {"block", "grids", "ws_meta"})
    check("step 중 reward finite", np.isfinite(reward), f"reward={reward}")
    check("step 후 current_step 증가", info2["current_step"] == 1)

# 전체 에피소드 실행
obs, info = env.reset()
total_steps = 0
done = False
while not done:
    mask = env.action_masks()
    valid = np.where(mask)[0]
    if len(valid) == 0:
        print("  ⚠️ 유효 액션 없음 — 에피소드 강제 종료")
        break
    action = int(valid[0])
    obs, reward, terminated, truncated, info = env.step(action)
    total_steps += 1
    done = terminated or truncated

check("전체 에피소드 완료", done, f"steps={total_steps}")
if done:
    check("Terminal score 존재", "terminal_score" in info,
          f"keys={list(info.keys())}")
    check("Resolved reward 존재", "resolved_reward" in info,
          f"keys={list(info.keys())}")
    check("Terminal residual 존재", "terminal_residual" in info,
          f"keys={list(info.keys())}")
    check("Episode reward 존재", "episode_reward" in info,
          f"keys={list(info.keys())}")
    if "terminal_score" in info:
        print(f"  Terminal score: {info['terminal_score']:.4f}")

# ── 6. Synthetic 모드 ─────────────────────────────────────────────
print("\n[6] Synthetic 모드 (블록+레이아웃 변형)")
env_syn = BlockPlacementEnv(
    blocks[:20],
    workspaces,
    strategy,
    use_synthetic=True,
    generator=gen,
    synthetic_n_blocks=15,
    vary_layout=True,
    grid_size=G,
)

obs1, _ = env_syn.reset()
obs2, _ = env_syn.reset()
# 두 에피소드의 그리드가 다른지 확인 (레이아웃 변형)
grids_diff = not np.allclose(obs1["grids"], obs2["grids"])
check("에피소드 간 그리드 변형", grids_diff,
      "두 에피소드의 그리드가 동일함")

# ── 결과 요약 ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
total = PASS + FAIL
print(f"  결과: {PASS}/{total} 통과, {FAIL}개 실패")
print("=" * 60)

if FAIL > 0:
    sys.exit(1)
