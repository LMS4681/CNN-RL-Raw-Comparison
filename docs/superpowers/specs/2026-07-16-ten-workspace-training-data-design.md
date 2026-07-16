# Ten-Workspace Training Data Design

## Objective

Train the existing candidate-CNN MaskablePPO flow against the ten real target
workspaces while starting every episode from an empty yard and treating all
eligible CSV rows as allocation targets.

## Target Workspaces

Action indices follow this fixed order:

1. `PE049` (C-1)
2. `PE050` (C-2)
3. `PE055` (500-A)
4. `PE054` (500-B)
5. `PE056` (700-Q)
6. `PE048` (700-A)
7. `PE044` (900-A)
8. `PE059` (800-T)
9. `PE060` (H-dong)
10. `PE061` (900-C)

The workspace master CSV omits `PE054`. Its three lot records span 51 m by
31 m, so the loader supplies `PE054` as 500-B with those dimensions and keeps
the source lot geometry.

## Target Blocks

- Use both historically placed and unplaced rows as new allocation targets.
- Use `건조착수` and `건조완료` for every target, including historically
  placed rows.
- Exclude targets whose start month is July or November.
- The current source data must produce exactly 913 targets.
- Ignore historical workspace codes, coordinates, and placement flags.
- Do not create pre-placed obstacles for training or CSV evaluation.

## Episode Generation

- Every training episode contains exactly 913 blocks.
- Use the seven observed months from December 2025 through June 2026.
- In 80% of episodes, distribute 913 evenly, add bounded monthly jitter of
  plus or minus 20, and rebalance to an exact total of 913.
- In 20% of episodes, retain the empirical month counts
  `64, 122, 106, 142, 153, 151, 175`.
- Select exact start dates from working days inside the assigned month.
- Bootstrap complete source rows so length, breadth, height, weight, and
  duration remain correlated.
- Keep the ten workspace dimensions and lot geometry fixed across episodes.

The fixed total stabilizes episode length and PPO rollout composition. Monthly
variation prevents the policy from memorizing one congestion calendar, while
the empirical-profile episodes preserve exposure to the real seasonal load.

## Model Compatibility

The ten-workspace action and observation shapes are incompatible with existing
five-workspace checkpoints. Colab therefore uses a new output directory and a
960-step rollout, which is close to one 913-block episode and divisible by the
64-sample minibatch size.
