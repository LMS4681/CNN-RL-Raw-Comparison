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
    from alloc_env.observation_state import GRID_SIZE
    from alloc_env.strategy import BaseGridStrategy
    from evaluation_runner import model_class_from_run_config
    from train import (
        create_evaluation_env,
        export_to_onnx,
        load_allocation_scenario,
        load_model_run_config,
        observation_contract_from_run_config,
        resolve_model_archive_path,
    )

    model_path = resolve_model_archive_path(model_path)
    run_config = load_model_run_config(model_path)
    model_class = model_class_from_run_config(run_config)
    workspace_codes, state_context, observation_scales = (
        observation_contract_from_run_config(
            run_config, source="ONNX export"
        )
    )
    data_dir = Path(data_dir).expanduser().resolve()

    strategy = BaseGridStrategy(step=5.0)
    blocks, workspaces = load_allocation_scenario(
        data_dir,
        strategy,
        workspace_codes,
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
        model = model_class.load(str(model_path), env=env, device=device)
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
