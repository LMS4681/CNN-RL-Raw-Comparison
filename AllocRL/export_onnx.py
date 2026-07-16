"""Export a saved MaskablePPO policy using its recorded run configuration."""

from __future__ import annotations

import argparse
from pathlib import Path


def export_saved_model(
    data_dir: str | Path = "./data",
    model_path: str | Path = "./output/block_placement_ppo.sb3",
    onnx_path: str | Path | None = None,
    device: str = "auto",
) -> Path:
    """Recreate the saved observation space and export the actor to ONNX."""
    from sb3_contrib import MaskablePPO

    from alloc_env.strategy import BaseGridStrategy
    from train import (
        create_evaluation_env,
        export_to_onnx,
        load_allocation_scenario,
        load_model_run_config,
        require_current_training_data_schema,
        resolve_model_archive_path,
    )

    model_path = resolve_model_archive_path(model_path)
    run_config = load_model_run_config(model_path)
    require_current_training_data_schema(run_config, source="ONNX export")
    data_dir = Path(data_dir).expanduser().resolve()

    strategy = BaseGridStrategy(step=5.0)
    blocks, workspaces = load_allocation_scenario(
        data_dir,
        strategy,
        run_config.get("active_workspace_codes"),
    )

    env = create_evaluation_env(
        blocks,
        workspaces,
        strategy,
        grid_size=int(run_config["grid_size"]),
        n_future_blocks=int(run_config["n_future_blocks"]),
        seed=int(run_config.get("seed", 0)),
    )
    try:
        model = MaskablePPO.load(str(model_path), env=env, device=device)
        destination = (
            Path(onnx_path).expanduser().resolve()
            if onnx_path is not None
            else model_path.with_suffix(".onnx")
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        export_to_onnx(model, env, str(destination))
    finally:
        env.close()

    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a saved AllocRL MaskablePPO policy to ONNX"
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument(
        "--model-path", default="./output/block_placement_ppo.sb3"
    )
    parser.add_argument("--onnx-path", default=None)
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"]
    )
    args = parser.parse_args()

    destination = export_saved_model(
        data_dir=args.data_dir,
        model_path=args.model_path,
        onnx_path=args.onnx_path,
        device=args.device,
    )
    print(f"ONNX export OK: {destination}")


if __name__ == "__main__":
    main()
