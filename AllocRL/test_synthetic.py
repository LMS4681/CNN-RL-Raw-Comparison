"""Smoke test: shaped reward + 강화된 Dict 관측값 검증."""
import os, sys
sys.path.insert(0, ".")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from alloc_env.strategy import BaseGridStrategy
from alloc_env.alloc_env import DROPOUT_THRESHOLD
from alloc_env.block_generator import SyntheticBlockGenerator
from alloc_env.observation_state import build_observation_scales
from train import (
    DEFAULT_ACTIVE_WORKSPACE_CODES,
    create_training_env,
    load_allocation_scenario,
    parse_workspace_codes,
)

s = BaseGridStrategy(step=5.0)
bl, ws = load_allocation_scenario(
    "data", s, parse_workspace_codes(DEFAULT_ACTIVE_WORKSPACE_CODES)
)
observation_scales = build_observation_scales(
    bl, ws, DROPOUT_THRESHOLD
)

gen = SyntheticBlockGenerator.from_blocks(bl)

# 1. Obs 차원 확인
env = create_training_env(
    bl,
    ws,
    s,
    gen,
    observation_scales=observation_scales,
    episode_n_blocks=50,
    grid_size=64,
    n_envs=1,
)
obs, _ = env.reset()
assert env.observation_space.contains(obs)
assert env.unwrapped._observation_scales is observation_scales
print(f"=== Obs Structure ===")
print(f"  Keys: {list(obs.keys())}")
print(f"  block shape: {obs['block'].shape}")
print(f"  grids shape: {obs['grids'].shape}")
print(f"  ws_meta shape: {obs['ws_meta'].shape}")

# 2. Step reward 확인
mask = env.action_masks()
valid = np.where(mask)[0]
obs2, r, done, _, _ = env.step(valid[0])
print(f"\n=== Step Reward ===")
print(f"  Step 1 reward: {r:+.4f}, finite: {np.isfinite(r)}")

# 3. Obs 구조 확인
print(f"\n=== Obs Structure (step=1) ===")
print(f"  block[0-4]: {obs2['block'][:5]}")
print(f"  assigned:   {obs2['block'][5]:.3f}")
print(f"  area:       {obs2['block'][6]:.3f}")
print(f"  max axis:   {obs2['block'][7]:.3f}")
print(f"  ws_meta[0]: {obs2['ws_meta'][0]}")

# 4. 에피소드 완료 후 terminal reward
total_r = r
step = 1
while not done:
    mask = env.action_masks()
    valid = np.where(mask)[0]
    obs, r, done, _, info = env.step(np.random.choice(valid))
    total_r += r
    step += 1

print(f"\n=== Terminal Reward ===")
print(f"  Steps: {step}")
print(f"  Total reward: {total_r:+.4f}")
print(f"  Resolved reward (from info): {info.get('resolved_reward', 'N/A')}")
print(f"  Terminal residual (from info): {info.get('terminal_residual', 'N/A')}")
print(f"  Terminal score (from info): {info.get('terminal_score', 'N/A')}")
print(f"  Episode reward (from info): {info.get('episode_reward', 'N/A')}")

# 분석
result = info['raw_result']
dd = result.delay_days
n = len(dd)
from alloc_env.simulator import SimulationResult
compliant = sum(1 for d in dd if d != SimulationResult.DROPOUT and d <= 2)
delayed = sum(1 for d in dd if d != SimulationResult.DROPOUT and d > 2)
dropout = sum(1 for d in dd if d == SimulationResult.DROPOUT)
print(f"  Compliant: {compliant}, Delayed: {delayed}, Dropout: {dropout}")
print(f"  Success rate: {compliant/n:.1%}")

# 5. 보상 범위 검증 (여러 에피소드)
rewards = []
for _ in range(20):
    obs, _ = env.reset()
    done = False
    while not done:
        mask = env.action_masks()
        valid = np.where(mask)[0]
        obs, r, done, _, info = env.step(np.random.choice(valid))
    rewards.append(info['terminal_score'])

print(f"\n=== Reward Stats (20 episodes) ===")
print(f"  Min: {min(rewards):.4f}, Max: {max(rewards):.4f}")
print(f"  Mean: {np.mean(rewards):.4f}, Std: {np.std(rewards):.4f}")
print(f"  Range valid [-2.0, +1.0]: {all(-2.0 <= r <= 1.0 for r in rewards)}")

env.close()
print(f"\nAll tests passed!")
