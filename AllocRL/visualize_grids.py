# -*- coding: utf-8 -*-
"""
CNN 입력용 점유 그리드 시각화 - 4채널 그리드를 이미지로 저장.

각 작업장의 실제 schema-3 관측 그리드를 컬러 이미지로 변환하여 저장합니다.
- Ch0 (collision exclusion): 빨간색
- Ch1 (remaining working days): 초록색
- Ch2 (post-candidate lot state): 파란색
- Ch3 (candidate exclusion): 자홍색

실행:
  py visualize_grids.py --data-dir ./data --output-dir ./output/grid_images

옵션:
  --grid-size 64     그리드 해상도 (기본 64)
  --synthetic        합성 기배치를 추가하여 렌더링
  --date 2026-04-15  환경 날짜 지정
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
from datetime import date, datetime
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
os.chdir(BASE)

CHANNEL_LABELS = (
    "collision exclusion",
    "remaining working days",
    "post-candidate lot state",
    "candidate exclusion",
)


def visualize_grids(args):
    import matplotlib
    matplotlib.use('Agg')  # GUI 없는 백엔드
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.font_manager as fm

    # 한글 폰트 설정 (Windows: Malgun Gothic)
    font_path = "C:/Windows/Fonts/malgun.ttf"
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams['axes.unicode_minus'] = False

    from alloc_env.alloc_env import BlockPlacementEnv, DROPOUT_THRESHOLD
    from alloc_env.observation_state import build_observation_scales
    from alloc_env.strategy import BaseGridStrategy
    from alloc_env.occupancy_grid import OccupancyGridRenderer
    from alloc_env.block_generator import SyntheticBlockGenerator
    from train import (
        DEFAULT_ACTIVE_WORKSPACE_CODES,
        load_allocation_scenario,
        parse_workspace_codes,
    )

    data_dir = Path(args.data_dir)
    blk_csv = str(data_dir / "블록데이터.csv")

    # ── 1. 데이터 로드 ────────────────────────────────────────────
    strategy = BaseGridStrategy(step=5.0)
    blocks, workspaces = load_allocation_scenario(
        data_dir,
        strategy,
        parse_workspace_codes(DEFAULT_ACTIVE_WORKSPACE_CODES),
    )
    observation_scales = build_observation_scales(
        blocks, workspaces, DROPOUT_THRESHOLD
    )
    print(f"작업장 {len(workspaces)}개, 블록 {len(blocks)}개 로드")

    # ── 2. 환경 날짜 ──────────────────────────────────────────────
    if args.date:
        env_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        env_date = min(b.in_date for b in blocks) if blocks else date(2026, 4, 1)
    print(f"환경 날짜: {env_date}")

    # ── 3. 합성 모드 (선택) ───────────────────────────────────────
    if args.synthetic:
        gen = SyntheticBlockGenerator.from_csv(blk_csv, seed=42)

        # 기배치 블록 합성 (각 작업장에 3~5개)
        n_pre = len(workspaces) * 4
        preplaced = gen.generate_preplaced(n_pre, workspaces, env_date)
        ws_map = {ws.code: ws for ws in workspaces}
        for ws_code, pp in preplaced:
            if ws_code in ws_map:
                ws_map[ws_code].add_pre_placement(pp)
        print(f"[Synthetic] 작업장 변형 + 기배치 {n_pre}개 생성")

    # ── 4. 렌더링 ─────────────────────────────────────────────────
    G = args.grid_size
    renderer = OccupancyGridRenderer(grid_size=G)
    visualization_blocks = blocks
    if args.date:
        later_blocks = [block for block in blocks if block.in_date >= env_date]
        if later_blocks:
            visualization_blocks = later_blocks

    env = BlockPlacementEnv(
        visualization_blocks,
        workspaces,
        strategy,
        use_synthetic=False,
        grid_size=G,
        state_context_mode="full",
        observation_scales=observation_scales,
    )
    obs, _ = env.reset(seed=args.seed)
    grids = obs["grids"]  # (N, 4, G, G), including candidate placement
    env_date = env.unwrapped._env_date
    env.close()
    print(f"그리드 shape: {grids.shape}")

    # ── 5. 출력 디렉토리 ──────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 6-A. 개별 작업장 이미지 저장 ──────────────────────────────
    individual_dir = out_dir / "individual"
    individual_dir.mkdir(exist_ok=True)

    for i, ws in enumerate(workspaces):
        grid = grids[i]  # (4, G, G)
        coordinate_map = renderer.coordinate_map(ws)

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        fig.suptitle(
            f"{ws.code} ({ws.name})  |  "
            f"{ws.length:.0f}×{ws.breadth:.0f}m  |  "
            f"x={coordinate_map.x_px_per_m:.3f} px/m, "
            f"y={coordinate_map.y_px_per_m:.3f} px/m  |  date={env_date}",
            fontsize=13, fontweight='bold'
        )

        # Ch0
        ax = axes[0]
        ax.imshow(grid[0], cmap='Reds', vmin=0, vmax=1, origin='lower')
        ax.set_title(CHANNEL_LABELS[0], fontsize=11)
        ax.set_xlabel(f"excluded={grid[0].sum():.0f}px")

        # Ch1
        ax = axes[1]
        im1 = ax.imshow(grid[1], cmap='YlGn', vmin=0, vmax=1, origin='lower')
        ax.set_title(CHANNEL_LABELS[1], fontsize=11)
        plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

        # Ch2
        ax = axes[2]
        ax.imshow(grid[2], cmap='Blues', vmin=0, vmax=1, origin='lower')
        ax.set_title(CHANNEL_LABELS[2], fontsize=11)
        ax.set_xlabel(f"mean state={grid[2].mean():.3f}")

        # Ch3
        ax = axes[3]
        ax.imshow(grid[3], cmap='magma', vmin=0, vmax=1, origin='lower')
        ax.set_title(CHANNEL_LABELS[3], fontsize=11)
        ax.set_xlabel(f"excluded={grid[3].sum():.0f}px")

        # RGB 합성 + 후보 배치(자홍색)
        ax = axes[4]
        rgb = np.stack([grid[0], grid[1], grid[2]], axis=-1)
        rgb[..., 0] = np.maximum(rgb[..., 0], grid[3])
        rgb[..., 2] = np.maximum(rgb[..., 2], grid[3])
        ax.imshow(rgb, origin='lower')
        ax.set_title("RGB + 후보 배치(자홍색)", fontsize=11)

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        img_path = individual_dir / f"{ws.code}_{G}x{G}.png"
        fig.savefig(str(img_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  저장: {img_path.name}")

    # ── 6-B. 전체 작업장 그리드 오버뷰 ────────────────────────────
    n_ws = len(workspaces)
    cols = min(6, n_ws)
    rows = (n_ws + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    fig.suptitle(
        f"전체 작업장 점유 그리드 오버뷰  |  {G}×{G}  |  date={env_date}",
        fontsize=14, fontweight='bold'
    )

    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i in range(len(axes)):
        ax = axes[i]
        if i < n_ws:
            ws = workspaces[i]
            grid = grids[i]
            rgb = np.stack([grid[0], grid[1], grid[2]], axis=-1)
            rgb[..., 0] = np.maximum(rgb[..., 0], grid[3])
            rgb[..., 2] = np.maximum(rgb[..., 2], grid[3])
            ax.imshow(rgb, origin='lower')
            exclusion = grid[0].mean() * 100
            ax.set_title(f"{ws.code}\n{ws.length:.0f}×{ws.breadth:.0f}m\nexcl={exclusion:.0f}%",
                         fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= n_ws:
            ax.set_visible(False)

    # 범례
    legend_patches = [
        mpatches.Patch(color='red', label=CHANNEL_LABELS[0]),
        mpatches.Patch(color='green', label=CHANNEL_LABELS[1]),
        mpatches.Patch(color='blue', label=CHANNEL_LABELS[2]),
        mpatches.Patch(color='magenta', label=CHANNEL_LABELS[3]),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4, fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    overview_path = out_dir / f"grid_overview_{G}x{G}.png"
    fig.savefig(str(overview_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n오버뷰 저장: {overview_path}")

    # ── 7. 통계 요약 ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  그리드 통계 요약 ({G}×{G})")
    print(f"{'='*60}")
    print(
        f"{'작업장':>8} {'크기(m)':>14} {'x px/m':>8} {'y px/m':>8} "
        f"{'lot':>6} {'excl%':>6} {'remaining':>9}"
    )
    print(f"{'-'*60}")
    for i, ws in enumerate(workspaces):
        grid = grids[i]
        lot_state = grid[2].mean()
        exclusion = grid[0].mean() * 100
        avg_ttl = grid[1][grid[1] > 0].mean() * 60 if grid[1].any() else 0
        coordinate_map = renderer.coordinate_map(ws)
        print(f"{ws.code:>8} {ws.length:>6.0f}×{ws.breadth:<6.0f} "
              f"x={coordinate_map.x_px_per_m:.3f} px/m "
              f"y={coordinate_map.y_px_per_m:.3f} px/m "
              f"{lot_state:>5.2f} {exclusion:>5.1f}% "
              f"{avg_ttl:>6.1f}일")

    print(f"\n총 이미지 {n_ws + 1}개 저장 → {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="CNN 입력용 점유 그리드 시각화")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./output/grid_images")
    parser.add_argument("--grid-size", type=int, default=64, choices=[64],
                        help="고정 그리드 해상도")
    parser.add_argument("--synthetic", action="store_true",
                        help="합성 기배치를 추가하여 렌더링")
    parser.add_argument("--date", type=str, default=None,
                        help="환경 날짜 (YYYY-MM-DD)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    visualize_grids(args)


if __name__ == "__main__":
    main()
