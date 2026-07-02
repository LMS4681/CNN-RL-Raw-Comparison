"""Diagnose whether synthetic training blocks match the CSV problem shape.

Colab usage:

    !python diagnose_synthetic_data.py \
      --data-dir ./data \
      --output-dir /content/drive/MyDrive/CNN_RL_output/diagnostics
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from alloc_env.block import Block
from alloc_env.block_generator import SyntheticBlockGenerator
from alloc_env.constraints import BlockPatternConstraint, DimensionConstraint, ValidWorkspacePicker
from alloc_env.data_loader import apply_allowable_block_patterns, load_blocks, load_workspaces
from alloc_env.strategy import BaseGridStrategy
from alloc_env.workspace import Workspace


NUMERIC_FIELDS = {
    "length": lambda block: block.length,
    "breadth": lambda block: block.breadth,
    "height": lambda block: block.height,
    "weight": lambda block: block.weight,
    "duration_days": lambda block: block.original_duration,
}


def _safe_stats(values: Iterable[float]) -> dict[str, float]:
    arr = np.array(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def _name_prefix(name: str) -> str:
    if not name:
        return ""
    chars = []
    for ch in name:
        if ch.isdigit():
            break
        chars.append(ch)
    return "".join(chars).rstrip("-_") or name[:3]


def _valid_workspace_counts(blocks: list[Block], workspaces: list[Workspace]) -> list[int]:
    picker = ValidWorkspacePicker(
        blocks,
        workspaces,
        [DimensionConstraint(), BlockPatternConstraint()],
    )
    return [
        len(picker.get_valid_workspaces(i))
        for i in range(len(blocks))
    ]


def summarize_blocks(blocks: list[Block], workspaces: list[Workspace]) -> dict[str, float | int | str]:
    counts = _valid_workspace_counts(blocks, workspaces) if blocks else []
    count_arr = np.array(counts, dtype=np.float64)

    summary: dict[str, float | int | str] = {
        "count": len(blocks),
        "zero_valid_workspace_count": int(sum(1 for count in counts if count == 0)),
        "zero_valid_workspace_ratio": (
            float(sum(1 for count in counts if count == 0) / len(counts))
            if counts else 0.0
        ),
        "valid_workspace_mean": float(count_arr.mean()) if count_arr.size else 0.0,
        "valid_workspace_min": int(count_arr.min()) if count_arr.size else 0,
        "valid_workspace_max": int(count_arr.max()) if count_arr.size else 0,
    }

    for field, getter in NUMERIC_FIELDS.items():
        stats = _safe_stats(getter(block) for block in blocks)
        for stat_name, value in stats.items():
            summary[f"{field}_{stat_name}"] = value

    prefix_counts = Counter(_name_prefix(block.name) for block in blocks)
    top_prefixes = ", ".join(
        f"{prefix}:{count}" for prefix, count in prefix_counts.most_common(5)
    )
    summary["top_name_prefixes"] = top_prefixes
    return summary


def build_diagnostic_report(
    csv_blocks: list[Block],
    synthetic_blocks: list[Block],
    workspaces: list[Workspace],
) -> dict[str, dict[str, float | int | str]]:
    return {
        "csv": summarize_blocks(csv_blocks, workspaces),
        "synthetic": summarize_blocks(synthetic_blocks, workspaces),
    }


def write_diagnostic_report(
    report: Mapping[str, Mapping[str, float | int | str]],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_fields = sorted({
        key
        for dataset_summary in report.values()
        for key in dataset_summary.keys()
    })
    summary_path = output_dir / "synthetic_diagnostic_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset"] + all_fields)
        writer.writeheader()
        for dataset, summary in report.items():
            writer.writerow({"dataset": dataset, **summary})

    json_path = output_dir / "synthetic_diagnostic_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def diagnose_from_data_dir(
    data_dir: str | Path,
    output_dir: str | Path,
    n_samples: int | None = None,
    seed: int = 0,
) -> dict[str, dict[str, float | int | str]]:
    data_dir = Path(data_dir)
    ws_csv = str(data_dir / "선행건조 작업장 기준정보.csv")
    lot_csv = str(data_dir / "선행건조 지번 기준정보.csv")
    blk_csv = str(data_dir / "블록데이터.csv")

    strategy = BaseGridStrategy(step=5.0)
    workspaces = load_workspaces(ws_csv, lot_csv, strategy)
    apply_allowable_block_patterns(workspaces)
    csv_blocks = load_blocks(blk_csv, workspaces)

    sample_count = n_samples or len(csv_blocks)
    base_date = min((block.in_date for block in csv_blocks), default=date(2026, 4, 1))
    generator = SyntheticBlockGenerator.from_csv(blk_csv, seed=seed)
    synthetic_blocks = generator.generate(sample_count, base_date)

    report = build_diagnostic_report(csv_blocks, synthetic_blocks, workspaces)
    workspace_pattern_count = sum(
        1 for ws in workspaces if ws.allowable_block_patterns
    )
    report["workspace_patterns"] = {
        "workspace_count": len(workspaces),
        "workspaces_with_patterns": workspace_pattern_count,
        "workspaces_with_patterns_ratio": (
            workspace_pattern_count / len(workspaces) if workspaces else 0.0
        ),
    }
    write_diagnostic_report(report, output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose CSV vs synthetic block data")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./output/diagnostics")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    report = diagnose_from_data_dir(
        args.data_dir,
        args.output_dir,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    print(f"Diagnostic files saved to: {Path(args.output_dir).resolve()}")
    for dataset in ["csv", "synthetic"]:
        summary = report[dataset]
        print(
            f"{dataset}: count={summary['count']}, "
            f"zero_valid={summary['zero_valid_workspace_count']}, "
            f"valid_mean={summary['valid_workspace_mean']:.2f}, "
            f"length_mean={summary['length_mean']:.2f}, "
            f"breadth_mean={summary['breadth_mean']:.2f}"
        )


if __name__ == "__main__":
    main()
