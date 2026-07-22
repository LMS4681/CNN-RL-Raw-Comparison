"""Export deterministic evaluation placement results as CSV and date-frame PNGs.

Colab usage:

    !python visualize_eval_placement.py \
      --data-dir ./data \
      --model-path /content/drive/MyDrive/CNN_RL_output/block_placement_ppo.sb3 \
      --output-dir /content/drive/MyDrive/CNN_RL_output/eval_visualization
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from alloc_env.block import Block, PrePlacedBlock, SAFETY_DISTANCE
from alloc_env.simulator import SimulationResult
from alloc_env.workspace import Workspace
from train import (
    load_model_run_config,
    observation_contract_from_run_config,
    resolve_model_archive_path,
)

EPSILON = 1e-5


def _setup_korean_font() -> None:
    """한글 라벨이 tofu(□)로 깨지지 않도록 사용 가능한 한글 폰트를 선택.

    Colab에서는 `apt-get install -y fonts-nanum` 후 이 함수가 NanumGothic을 찾아 씀.
    폰트가 없으면 조용히 무시(기존 동작 유지).
    """
    import matplotlib
    from matplotlib import font_manager

    # 알려진 경로의 폰트를 직접 등록 (matplotlib 폰트 캐시가 오래된 경우 대비)
    known_paths = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    ]
    for p in known_paths:
        try:
            if Path(p).exists():
                font_manager.fontManager.addfont(p)
        except Exception:
            pass

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["NanumGothic", "Malgun Gothic", "AppleGothic", "NanumBarunGothic"]:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return


def _rect_bounds(cx: float, cy: float, length: float, breadth: float) -> tuple[float, float, float, float]:
    return (
        cx - length / 2.0,
        cx + length / 2.0,
        cy - breadth / 2.0,
        cy + breadth / 2.0,
    )


def _block_bounds(block: Block) -> tuple[float, float, float, float]:
    return _rect_bounds(block.ref_x, block.ref_y, block.length, block.breadth)


def _preplaced_bounds(pp: PrePlacedBlock) -> tuple[float, float, float, float]:
    return _rect_bounds(pp.pos_x, pp.pos_y, pp.length, pp.breadth)


def _rects_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    safety_distance: float = 0.0,
) -> bool:
    a_left, a_right, a_bottom, a_top = a
    b_left, b_right, b_bottom, b_top = b
    sep_x = (
        a_right + safety_distance <= b_left + EPSILON
        or b_right + safety_distance <= a_left + EPSILON
    )
    sep_y = (
        a_top + safety_distance <= b_bottom + EPSILON
        or b_top + safety_distance <= a_bottom + EPSILON
    )
    return not (sep_x or sep_y)


def _periods_overlap(a_start: date, a_end: date, b_start: date, b_end: date) -> bool:
    return a_start <= b_end and b_start <= a_end


def _block_periods_overlap(a_start: date, a_end: date, b_start: date, b_end: date) -> bool:
    return a_start <= b_end and b_start <= a_end


def _block_preplaced_periods_overlap(
    block_start: date,
    block_end: date,
    preplaced_start: date,
    preplaced_end: date,
) -> bool:
    return block_start <= preplaced_end and preplaced_start <= block_end


def _is_active(block: Block, env_date: date) -> bool:
    return block.in_date <= env_date <= block.out_date


def _is_preplaced_active(pp: PrePlacedBlock, env_date: date) -> bool:
    return pp.start_date <= env_date <= pp.end_date


def _workspace_contains(ws: Workspace, bounds: tuple[float, float, float, float]) -> bool:
    left, right, bottom, top = bounds
    return (
        left >= ws.origin_x - EPSILON
        and bottom >= ws.origin_y - EPSILON
        and right <= ws.origin_x + ws.length + EPSILON
        and top <= ws.origin_y + ws.breadth + EPSILON
    )


def _workspace_display_name(ws: Workspace) -> str:
    return ws.name or ws.code


def block_plot_label(block: Block) -> str:
    return f"{block.name}\n{block.in_date.isoformat()}~{block.out_date.isoformat()}"


def _iter_dates(start: date, end: date, stride: int = 1) -> Iterable[date]:
    stride = max(1, stride)
    current = start
    step = timedelta(days=stride)
    while current <= end:
        yield current
        current += step


def find_placement_violations(result: SimulationResult) -> list[dict[str, str]]:
    """Find dropout, missing workspace, out-of-bounds, and time-overlap collisions."""
    workspace_by_code = {ws.code: ws for ws in result.workspaces}
    violations: list[dict[str, str]] = []

    placed_blocks: list[tuple[int, Block]] = []
    for idx, block in enumerate(result.blocks):
        delay = result.delay_days[idx] if idx < len(result.delay_days) else SimulationResult.DROPOUT
        if delay == SimulationResult.DROPOUT:
            violations.append({
                "type": "dropout",
                "block_index": str(idx),
                "block_name": block.name,
                "workspace_code": block.workspace_code or "",
                "detail": "Block was not placed before dropout threshold",
            })
            continue

        if not block.workspace_code or block.workspace_code not in workspace_by_code:
            violations.append({
                "type": "missing_workspace",
                "block_index": str(idx),
                "block_name": block.name,
                "workspace_code": block.workspace_code or "",
                "detail": "Placed block has no known workspace_code",
            })
            continue

        workspace = workspace_by_code[block.workspace_code]
        if not _workspace_contains(workspace, _block_bounds(block)):
            violations.append({
                "type": "out_of_bounds",
                "block_index": str(idx),
                "block_name": block.name,
                "workspace_code": block.workspace_code,
                "detail": (
                    f"bounds={tuple(round(v, 3) for v in _block_bounds(block))}, "
                    f"workspace=({workspace.origin_x}, {workspace.origin_y}, "
                    f"{workspace.length}, {workspace.breadth})"
                ),
            })

        for pp in workspace.pre_placements:
            if not _block_preplaced_periods_overlap(
                block.in_date,
                block.out_date,
                pp.start_date,
                pp.end_date,
            ):
                continue
            block_bounds = _block_bounds(block)
            preplaced_bounds = _preplaced_bounds(pp)
            if _rects_overlap(block_bounds, preplaced_bounds):
                violations.append({
                    "type": "preplaced_overlap",
                    "block_index": str(idx),
                    "block_name": block.name,
                    "workspace_code": block.workspace_code,
                    "detail": f"overlaps preplaced block {pp.label}",
                })
            elif _rects_overlap(block_bounds, preplaced_bounds, SAFETY_DISTANCE):
                violations.append({
                    "type": "preplaced_safety_distance",
                    "block_index": str(idx),
                    "block_name": block.name,
                    "workspace_code": block.workspace_code,
                    "detail": f"closer than {SAFETY_DISTANCE:.1f}m to preplaced block {pp.label}",
                })

        placed_blocks.append((idx, block))

    for left_pos, (i, left) in enumerate(placed_blocks):
        for j, right in placed_blocks[left_pos + 1:]:
            if left.workspace_code != right.workspace_code:
                continue
            if not _block_periods_overlap(left.in_date, left.out_date, right.in_date, right.out_date):
                continue
            left_bounds = _block_bounds(left)
            right_bounds = _block_bounds(right)
            if _rects_overlap(left_bounds, right_bounds):
                violations.append({
                    "type": "overlap",
                    "block_index": str(i),
                    "block_name": left.name,
                    "workspace_code": left.workspace_code or "",
                    "detail": f"overlaps block {j}:{right.name}",
                })
            elif _rects_overlap(left_bounds, right_bounds, SAFETY_DISTANCE):
                violations.append({
                    "type": "safety_distance",
                    "block_index": str(i),
                    "block_name": left.name,
                    "workspace_code": left.workspace_code or "",
                    "detail": f"closer than {SAFETY_DISTANCE:.1f}m to block {j}:{right.name}",
                })

    return violations


def _write_assignments(result: SimulationResult, assignments: list[int] | None, output_dir: Path) -> None:
    path = output_dir / "assignments.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "block_index", "block_name", "assignment", "workspace_code",
            "x", "y", "length", "breadth", "in_date", "out_date",
            "delay_days", "status",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, block in enumerate(result.blocks):
            delay = result.delay_days[idx] if idx < len(result.delay_days) else SimulationResult.DROPOUT
            writer.writerow({
                "block_index": idx,
                "block_name": block.name,
                "assignment": assignments[idx] if assignments and idx < len(assignments) else "",
                "workspace_code": block.workspace_code or "",
                "x": f"{block.ref_x:.3f}",
                "y": f"{block.ref_y:.3f}",
                "length": f"{block.length:.3f}",
                "breadth": f"{block.breadth:.3f}",
                "in_date": block.in_date.isoformat(),
                "out_date": block.out_date.isoformat(),
                "delay_days": delay,
                "status": "dropout" if delay == SimulationResult.DROPOUT else "placed",
            })


def _write_violations(violations: list[dict[str, str]], output_dir: Path) -> None:
    path = output_dir / "placement_violations.csv"
    fieldnames = ["type", "block_index", "block_name", "workspace_code", "detail"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in violations:
            writer.writerow(item)


def _plot_workspace_frame(
    result: SimulationResult,
    env_date: date,
    save_path: Path,
    active_workspace_codes: list[str] | None = None,
) -> None:
    active_codes = (
        {code.upper() for code in active_workspace_codes}
        if active_workspace_codes else None
    )
    workspaces = [
        ws for ws in result.workspaces
        if active_codes is None or ws.code.upper() in active_codes
    ]
    if not workspaces:
        return

    cols = min(4, len(workspaces))
    rows = math.ceil(len(workspaces) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 3.6))
    if not isinstance(axes, np.ndarray):  # type: ignore[name-defined]
        axes_list = [axes]
    else:
        axes_list = list(axes.ravel())

    for ax in axes_list[len(workspaces):]:
        ax.axis("off")

    blocks_by_workspace: dict[str, list[Block]] = {}
    for block in result.blocks:
        if block.workspace_code and _is_active(block, env_date):
            blocks_by_workspace.setdefault(block.workspace_code, []).append(block)

    for ax, ws in zip(axes_list, workspaces):
        ax.set_title(_workspace_display_name(ws), fontsize=9)
        ax.set_xlim(ws.origin_x, ws.origin_x + ws.length)
        ax.set_ylim(ws.origin_y, ws.origin_y + ws.breadth)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)

        ax.add_patch(plt.Rectangle(
            (ws.origin_x, ws.origin_y),
            ws.length,
            ws.breadth,
            fill=False,
            edgecolor="black",
            linewidth=1.2,
        ))
        for lot in ws.lots:
            ax.add_patch(plt.Rectangle(
                (lot.origin_x, lot.origin_y),
                lot.length,
                lot.breadth,
                fill=False,
                edgecolor="lightgray",
                linewidth=0.5,
            ))

        for pp in ws.pre_placements:
            if not _is_preplaced_active(pp, env_date):
                continue
            left, _, bottom, _ = _preplaced_bounds(pp)
            ax.add_patch(plt.Rectangle(
                (left, bottom),
                pp.length,
                pp.breadth,
                facecolor="#9ca3af",
                edgecolor="#4b5563",
                alpha=0.45,
            ))

        for block in blocks_by_workspace.get(ws.code, []):
            left, _, bottom, _ = _block_bounds(block)
            in_bounds = _workspace_contains(ws, _block_bounds(block))
            color = "#2563eb" if in_bounds else "#dc2626"
            ax.add_patch(plt.Rectangle(
                (left, bottom),
                block.length,
                block.breadth,
                facecolor=color,
                edgecolor="white",
                alpha=0.75,
            ))
            ax.text(
                block.ref_x,
                block.ref_y,
                block_plot_label(block),
                ha="center",
                va="center",
                fontsize=4,
                color="white",
            )

    fig.suptitle(f"Placement on {env_date.isoformat()}", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def export_evaluation_visualization(
    result: SimulationResult,
    assignments: list[int] | None,
    output_dir: str | Path,
    frame_stride_days: int = 1,
    active_workspace_codes: list[str] | None = None,
) -> list[dict[str, str]]:
    output_dir = Path(output_dir)
    frames_dir = output_dir / "placement_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    _write_assignments(result, assignments, output_dir)
    violations = find_placement_violations(result)
    _write_violations(violations, output_dir)

    placed_blocks = [
        block for i, block in enumerate(result.blocks)
        if i < len(result.delay_days)
        and result.delay_days[i] != SimulationResult.DROPOUT
        and block.workspace_code
    ]
    if not placed_blocks:
        return violations

    start = min(block.in_date for block in placed_blocks)
    end = max(block.out_date for block in placed_blocks)
    for env_date in _iter_dates(start, end, stride=frame_stride_days):
        _plot_workspace_frame(
            result,
            env_date,
            frames_dir / f"placement_{env_date.isoformat()}.png",
            active_workspace_codes=active_workspace_codes,
        )
    return violations


def evaluate_model_and_export(
    data_dir: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    frame_stride_days: int = 1,
) -> list[dict[str, str]]:
    from alloc_env.observation_state import GRID_SIZE
    from alloc_env.strategy import BaseGridStrategy
    from evaluation_runner import model_class_from_run_config
    from train import create_evaluation_env, load_allocation_scenario

    _setup_korean_font()  # 한글 라벨 깨짐 방지 (폰트 없으면 무시)

    model_path = resolve_model_archive_path(model_path)
    run_config = load_model_run_config(model_path)
    model_class = model_class_from_run_config(run_config)
    active_codes, state_context, observation_scales = (
        observation_contract_from_run_config(
            run_config, source="placement visualization"
        )
    )

    data_dir = Path(data_dir)
    strategy = BaseGridStrategy(step=5.0)
    blocks, workspaces = load_allocation_scenario(
        data_dir, strategy, active_codes
    )

    env = create_evaluation_env(
        blocks,
        workspaces,
        strategy,
        observation_scales=observation_scales,
        grid_size=GRID_SIZE,
        state_context_mode=state_context,
        seed=int(run_config.get("seed", 0)),
    )
    try:
        model = model_class.load(str(model_path), env=env, device="auto")
        obs, _ = env.reset()
        done = False
        info = {}
        while not done:
            action_masks = (
                env.action_masks() if hasattr(env, "action_masks") else None
            )
            action, _ = model.predict(
                obs, action_masks=action_masks, deterministic=True
            )
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
    finally:
        env.close()

    result = info.get("raw_result")
    if result is None:
        raise RuntimeError("Evaluation did not return raw_result.")
    assignments = info.get("assignments")
    return export_evaluation_visualization(
        result,
        assignments,
        output_dir,
        frame_stride_days=frame_stride_days,
        active_workspace_codes=[workspace.code for workspace in workspaces],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize deterministic evaluation placement")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model-path", default="./output/block_placement_ppo.sb3")
    parser.add_argument("--output-dir", default="./output/eval_visualization")
    parser.add_argument("--frame-stride-days", type=int, default=1)
    args = parser.parse_args()

    violations = evaluate_model_and_export(
        args.data_dir,
        args.model_path,
        args.output_dir,
        frame_stride_days=args.frame_stride_days,
    )
    print(f"Visualization files saved to: {Path(args.output_dir).resolve()}")
    print(f"Violation count: {len(violations)}")


if __name__ == "__main__":
    main()
