"""Smoke test: reward 범위 및 정합성 검증."""
import os, sys
sys.path.insert(0, ".")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from alloc_env.data_loader import load_workspaces, load_blocks
from alloc_env.strategy import BaseGridStrategy
from alloc_env.alloc_env import BlockPlacementEnv

s = BaseGridStrategy(step=5.0)
ws = load_workspaces(
    "data/선행건조 작업장 기준정보.csv",
    "data/선행건조 지번 기준정보.csv", s)
bl = load_blocks("data/블록데이터.csv", ws)
env = BlockPlacementEnv(bl, ws, s)

obs, info = env.reset()
total_r = 0.0
step_rewards = []
step = 0

while True:
    mask = env.action_masks()
    valid = np.where(mask)[0]
    action = np.random.choice(valid)
    obs, r, done, trunc, info = env.step(action)
    step_rewards.append(r)
    total_r += r
    step += 1

    if step <= 5:
        print(f"  Step {step:>3}: action={action:>2}, reward={r:>+.4f}, total={total_r:>+.4f}")

    if done:
        break

print(f"\n=== Episode Summary ===")
print(f"  Steps: {step}")
print(f"  Step rewards:  min={min(step_rewards[:-1]):>+.4f}  max={max(step_rewards[:-1]):>+.4f}  mean={np.mean(step_rewards[:-1]):>+.4f}")
print(f"  Last step reward (includes terminal): {step_rewards[-1]:>+.4f}")
print(f"  Resolved reward: {info.get('resolved_reward', 'N/A')}")
print(f"  Terminal residual: {info.get('terminal_residual', 'N/A')}")
print(f"  Terminal score: {info.get('terminal_score', 'N/A')}")
print(f"  Episode reward: {info.get('episode_reward', 'N/A')}")
print(f"  Total reward: {total_r:>+.4f}")
print(f"  Success rate: {sum(1 for d in info['raw_result'].delay_days if d != 2147483647 and d <= 2) / len(info['raw_result'].delay_days):.1%}")
print(f"  Dropout count: {sum(1 for d in info['raw_result'].delay_days if d == 2147483647)}")
