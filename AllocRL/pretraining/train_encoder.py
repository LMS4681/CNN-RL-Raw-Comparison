from __future__ import annotations

import argparse
import bisect
import copy
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from pretraining.dataset import (
    load_pretraining_shard,
    read_dataset_manifest,
)
from pretraining.model import CandidatePretrainingModel


CHECKPOINT_SCHEMA_VERSION = 1
COMPLETION_SCHEMA_VERSION = 1
RESUME_SCHEMA_VERSION = 1


class PretrainingGateError(RuntimeError):
    pass


@dataclass(frozen=True)
class EncoderTrainingConfig:
    schema_version: int
    seed: int
    optimizer: str
    learning_rate: float
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    minimum_relative_baseline_improvement: float
    minimum_shuffled_grid_degradation: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> EncoderTrainingConfig:
        config = cls(**{
            name: values[name]
            for name in cls.__dataclass_fields__
        })
        if config.schema_version != 1:
            raise ValueError("pretraining config schema_version must be 1")
        if config.optimizer != "AdamW":
            raise ValueError("pretraining optimizer must be AdamW")
        if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        for name in ("batch_size", "max_epochs", "early_stopping_patience"):
            value = getattr(config, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "minimum_relative_baseline_improvement",
            "minimum_shuffled_grid_degradation",
        ):
            value = getattr(config, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        return config


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, values: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="\n") as destination:
        json.dump(values, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)


def _atomic_torch_save(values: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as destination:
        torch.save(values, destination)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)


def _load_resume_state(
    path: Path,
    *,
    config_sha256: str,
    dataset_manifest_sha256: str,
    smoke_state_count: int | None,
) -> Mapping[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("invalid pretraining_last.pt") from error
    expected_keys = {
        "resume_schema_version",
        "observation_schema_version",
        "config_sha256",
        "dataset_manifest_sha256",
        "smoke_state_count",
        "next_epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "best_loss",
        "best_epoch",
        "best_state",
        "patience",
        "history",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise ValueError("pretraining resume payload keys differ")
    if payload["resume_schema_version"] != RESUME_SCHEMA_VERSION:
        raise ValueError("pretraining resume schema differs")
    if payload["observation_schema_version"] != 4:
        raise ValueError("pretraining resume observation schema differs")
    if payload["config_sha256"] != config_sha256:
        raise ValueError("pretraining resume config SHA256 differs")
    if payload["dataset_manifest_sha256"] != dataset_manifest_sha256:
        raise ValueError("pretraining resume dataset SHA256 differs")
    if payload["smoke_state_count"] != smoke_state_count:
        raise ValueError("pretraining resume smoke subset differs")
    next_epoch = payload["next_epoch"]
    history = payload["history"]
    if (
        isinstance(next_epoch, bool)
        or not isinstance(next_epoch, int)
        or next_epoch < 1
        or not isinstance(history, list)
        or len(history) != next_epoch
    ):
        raise ValueError("pretraining resume epoch history is invalid")
    return payload


def _save_resume_state(
    path: Path,
    *,
    model: CandidatePretrainingModel,
    optimizer: torch.optim.Optimizer,
    config_sha256: str,
    dataset_manifest_sha256: str,
    smoke_state_count: int | None,
    next_epoch: int,
    best_loss: float,
    best_epoch: int,
    best_state: Mapping[str, torch.Tensor],
    patience: int,
    history: list[dict[str, float | int]],
) -> None:
    _atomic_torch_save(
        {
            "resume_schema_version": RESUME_SCHEMA_VERSION,
            "observation_schema_version": 4,
            "config_sha256": config_sha256,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "smoke_state_count": smoke_state_count,
            "next_epoch": next_epoch,
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": float(best_loss),
            "best_epoch": int(best_epoch),
            "best_state": {
                key: value.detach().cpu()
                for key, value in best_state.items()
            },
            "patience": int(patience),
            "history": list(history),
        },
        path,
    )


class PretrainingShardDataset(Dataset):
    def __init__(
        self,
        root: Path,
        manifest: Mapping[str, object],
        split_name: str,
    ) -> None:
        self.root = Path(root)
        self.manifest = manifest
        self.entries = list(manifest["splits"][split_name]["shards"])
        self.ends: list[int] = []
        total = 0
        for entry in self.entries:
            total += int(entry["state_count"])
            self.ends.append(total)
        self._cached_index: int | None = None
        self._cached_shard: dict | None = None

    def __len__(self) -> int:
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index: int) -> dict[str, dict[str, torch.Tensor]]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.ends, index)
        start = 0 if shard_index == 0 else self.ends[shard_index - 1]
        if self._cached_index != shard_index:
            self._cached_shard = load_pretraining_shard(
                self.root, self.manifest, self.entries[shard_index]
            )
            self._cached_index = shard_index
        assert self._cached_shard is not None
        local_index = index - start
        return {
            section: {
                key: torch.from_numpy(values[local_index])
                for key, values in arrays.items()
            }
            for section, arrays in self._cached_shard.items()
        }


def validate_dataset_contract(
    configured: Mapping[str, object],
    manifest: Mapping[str, object],
    *,
    smoke_mode: bool,
) -> None:
    dataset_config = manifest.get("config")
    if not isinstance(dataset_config, Mapping):
        raise ValueError("dataset manifest is missing its config contract")
    keys = (
        "train_state_count",
        "validation_state_count",
        "train_episode_seeds",
        "validation_episode_seeds",
        "states_per_shard",
        "replay_every_n_states",
        "replay_resolved_blocks",
        "replay_max_decisions",
    )
    for key in keys:
        if smoke_mode and key in {
            "train_state_count",
            "validation_state_count",
        }:
            continue
        if configured.get(key) != dataset_config.get(key):
            raise ValueError(
                f"pretraining dataset config mismatch for {key}: "
                f"configured={configured.get(key)!r}, "
                f"dataset={dataset_config.get(key)!r}"
            )


def _observation_space(sample: Mapping[str, torch.Tensor]) -> gym.spaces.Dict:
    return gym.spaces.Dict({
        key: gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=tuple(value.shape),
            dtype=np.float32,
        )
        for key, value in sample.items()
    })


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _to_device(
    values: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in values.items()}


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.to(dtype=values.dtype)
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)


def _evaluate(
    model: CandidatePretrainingModel,
    loader: DataLoader,
    device: torch.device,
    *,
    optionality_baseline: float,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    sample_count = 0
    future_fit_sum = 0.0
    future_fit_count = 0.0
    no_grid_sum = 0.0
    optionality_sum = 0.0
    optionality_count = 0.0
    baseline_optionality_sum = 0.0
    shuffled_total = 0.0
    counterfactual_delta = 0.0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            observations = _to_device(batch["observations"], device)
            targets = _to_device(batch["targets"], device)
            predictions = model(observations)
            components = model.loss_components(predictions, targets)
            total = model.loss(predictions, targets)
            batch_size = observations["block"].shape[0]
            sample_count += batch_size
            totals["total_loss"] = totals.get("total_loss", 0.0) + (
                float(total) * batch_size
            )
            for name, value in components.items():
                totals[name] = totals.get(name, 0.0) + float(value) * batch_size

            action_mask = targets["action_mask"].bool()
            future_mask = action_mask.unsqueeze(-1).expand_as(
                targets["future_fit"]
            )
            future_losses = F.binary_cross_entropy(
                predictions.future_fit,
                targets["future_fit"],
                reduction="none",
            )
            future_fit_sum += float((future_losses * future_mask).sum())
            future_fit_count += float(future_mask.sum())

            no_grid = dict(observations)
            no_grid["grids"] = torch.zeros_like(observations["grids"])
            no_grid_predictions = model(no_grid)
            no_grid_losses = F.binary_cross_entropy(
                no_grid_predictions.future_fit,
                targets["future_fit"],
                reduction="none",
            )
            no_grid_sum += float((no_grid_losses * future_mask).sum())

            optionality_errors = (
                predictions.future_optionality_after
                - targets["future_optionality_after"]
            ).abs()
            optionality_sum += float(
                (optionality_errors * action_mask).sum()
            )
            optionality_count += float(action_mask.sum())
            baseline_optionality_sum += float(
                (
                    (
                        targets["future_optionality_after"]
                        - optionality_baseline
                    ).abs()
                    * action_mask
                ).sum()
            )

            shuffled = dict(observations)
            shuffled["grids"] = torch.roll(
                observations["grids"], shifts=1, dims=1
            )
            shuffled_total += float(
                model.loss(model(shuffled), targets)
            ) * batch_size

            counterfactual = dict(observations)
            counterfactual_grids = observations["grids"].clone()
            counterfactual_grids[:, :, 0] = torch.roll(
                counterfactual_grids[:, :, 0].flatten(-2),
                shifts=1,
                dims=-1,
            ).reshape_as(counterfactual_grids[:, :, 0])
            counterfactual["grids"] = counterfactual_grids
            changed = model(counterfactual)
            counterfactual_delta += float(
                (
                    changed.largest_free_rectangle_ratio
                    - predictions.largest_free_rectangle_ratio
                ).abs().mean()
                + (
                    changed.free_component_count_normalized
                    - predictions.free_component_count_normalized
                ).abs().mean()
            ) * batch_size

    if sample_count == 0:
        raise ValueError("validation dataset is empty")
    result = {
        name: value / sample_count for name, value in totals.items()
    }
    result.update({
        "future_fit_bce": future_fit_sum / max(future_fit_count, 1.0),
        "no_grid_future_fit_bce": no_grid_sum / max(future_fit_count, 1.0),
        "optionality_mae": optionality_sum / max(optionality_count, 1.0),
        "mean_baseline_optionality_mae": (
            baseline_optionality_sum / max(optionality_count, 1.0)
        ),
        "shuffled_grid_total_loss": shuffled_total / sample_count,
        "counterfactual_geometry_delta": counterfactual_delta / sample_count,
    })
    return result


def _target_mean(loader: DataLoader) -> float:
    total = 0.0
    count = 0.0
    for batch in loader:
        targets = batch["targets"]
        mask = targets["action_mask"].bool()
        total += float((targets["future_optionality_after"] * mask).sum())
        count += float(mask.sum())
    return total / max(count, 1.0)


def _gate_results(
    metrics: Mapping[str, float],
    config: EncoderTrainingConfig,
) -> dict[str, bool]:
    def all_finite(value: object) -> bool:
        if isinstance(value, Mapping):
            return all(all_finite(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(all_finite(item) for item in value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return math.isfinite(value)
        return True

    future_baseline = metrics["no_grid_future_fit_bce"]
    optionality_baseline = metrics["mean_baseline_optionality_mae"]
    normal_loss = metrics["validation_total_loss"]
    future_improvement = (
        future_baseline - metrics["future_fit_bce"]
    ) / max(future_baseline, 1e-12)
    optionality_improvement = (
        optionality_baseline - metrics["optionality_mae"]
    ) / max(optionality_baseline, 1e-12)
    shuffled_degradation = (
        metrics["shuffled_grid_total_loss"] - normal_loss
    ) / max(normal_loss, 1e-12)
    return {
        "finite": all_finite(metrics),
        "future_fit": future_improvement
        >= config.minimum_relative_baseline_improvement,
        "optionality": optionality_improvement
        >= config.minimum_relative_baseline_improvement,
        "grid_dependence": shuffled_degradation
        >= config.minimum_shuffled_grid_degradation,
        "counterfactual_geometry": metrics["counterfactual_geometry_delta"]
        > 1e-6,
    }


def write_extractor_checkpoint(
    model: CandidatePretrainingModel,
    path: Path,
    *,
    observation_schema_version: int,
    config_sha256: str,
    dataset_manifest_sha256: str,
    best_epoch: int,
) -> Path:
    state = {
        key: value.detach().cpu()
        for key, value in model.extractor.state_dict().items()
    }
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "observation_schema_version": observation_schema_version,
        "config_sha256": config_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "best_epoch": int(best_epoch),
        "extractor_state_dict": state,
    }
    _atomic_torch_save(payload, Path(path))
    return Path(path)


def publish_pretraining_completion(
    checkpoint_path: Path,
    metrics_path: Path,
    marker_path: Path,
    *,
    config_sha256: str,
    dataset_manifest_sha256: str,
    gates: Mapping[str, bool],
    smoke_mode: bool,
) -> Path:
    marker_path = Path(marker_path)
    marker_path.unlink(missing_ok=True)
    failed = sorted(name for name, passed in gates.items() if not passed)
    if failed and not smoke_mode:
        raise PretrainingGateError(
            "pretraining gates failed: " + ", ".join(failed)
        )
    receipt = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "observation_schema_version": 4,
        "checkpoint_filename": Path(checkpoint_path).name,
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "metrics_filename": Path(metrics_path).name,
        "metrics_sha256": _sha256_file(metrics_path),
        "config_sha256": config_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "gates": {name: bool(value) for name, value in gates.items()},
        "smoke_mode": bool(smoke_mode),
        "production_eligible": not smoke_mode and not failed,
    }
    _atomic_json(marker_path, receipt)
    return marker_path


def train_candidate_encoder(
    config_path: Path,
    dataset_root: Path,
    output_dir: Path,
    *,
    smoke_state_count: int | None = None,
    max_epochs: int | None = None,
    device: str | None = None,
) -> Path:
    config_path = Path(config_path)
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    config = EncoderTrainingConfig.from_mapping(raw_config)
    epochs = config.max_epochs if max_epochs is None else int(max_epochs)
    if epochs < 1:
        raise ValueError("max_epochs override must be positive")
    smoke_mode = smoke_state_count is not None
    if smoke_state_count is not None and smoke_state_count < 1:
        raise ValueError("smoke_state_count must be positive")

    manifest_path = dataset_root / "dataset_manifest.json"
    manifest = read_dataset_manifest(manifest_path)
    if manifest.get("observation_schema_version") != 4:
        raise ValueError("pretraining dataset must use observation schema 4")
    validate_dataset_contract(raw_config, manifest, smoke_mode=smoke_mode)
    config_sha256 = _sha256_file(config_path)
    dataset_manifest_sha256 = _sha256_file(manifest_path)

    train_dataset: Dataset = PretrainingShardDataset(
        dataset_root, manifest, "train"
    )
    validation_dataset: Dataset = PretrainingShardDataset(
        dataset_root, manifest, "validation"
    )
    if smoke_state_count is not None:
        train_dataset = Subset(
            train_dataset,
            range(min(smoke_state_count, len(train_dataset))),
        )
        validation_dataset = Subset(
            validation_dataset,
            range(min(smoke_state_count, len(validation_dataset))),
        )
    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError("pretraining datasets must be non-empty")

    _seed_everything(config.seed)
    deterministic_train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    first_sample = train_dataset[0]
    model = CandidatePretrainingModel(
        _observation_space(first_sample["observations"]),
        features_dim=256,
    )
    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate
    )
    optionality_baseline = _target_mean(deterministic_train_loader)

    best_loss = math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    history = []
    start_epoch = 0
    resume_path = output_dir / "pretraining_last.pt"
    if resume_path.is_file():
        resume = _load_resume_state(
            resume_path,
            config_sha256=config_sha256,
            dataset_manifest_sha256=dataset_manifest_sha256,
            smoke_state_count=smoke_state_count,
        )
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        best_loss = float(resume["best_loss"])
        best_epoch = int(resume["best_epoch"])
        best_state = {
            key: value.clone()
            for key, value in resume["best_state"].items()
        }
        patience = int(resume["patience"])
        history = list(resume["history"])
        start_epoch = int(resume["next_epoch"])

    epoch_range = (
        range(0)
        if patience >= config.early_stopping_patience
        else range(start_epoch, epochs)
    )
    for epoch in epoch_range:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(config.seed + epoch),
        )
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch in train_loader:
            observations = _to_device(
                batch["observations"], resolved_device
            )
            targets = _to_device(batch["targets"], resolved_device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(model(observations), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite pretraining loss")
            loss.backward()
            optimizer.step()
            batch_size = observations["block"].shape[0]
            train_loss += float(loss.detach()) * batch_size
            train_count += batch_size

        validation = _evaluate(
            model,
            validation_loader,
            resolved_device,
            optionality_baseline=optionality_baseline,
        )
        validation_loss = validation["total_loss"]
        history.append({
            "epoch": epoch,
            "train_total_loss": train_loss / max(train_count, 1),
            "validation_total_loss": validation_loss,
        })
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        assert best_state is not None
        _save_resume_state(
            resume_path,
            model=model,
            optimizer=optimizer,
            config_sha256=config_sha256,
            dataset_manifest_sha256=dataset_manifest_sha256,
            smoke_state_count=smoke_state_count,
            next_epoch=epoch + 1,
            best_loss=best_loss,
            best_epoch=best_epoch,
            best_state=best_state,
            patience=patience,
            history=history,
        )
        if patience >= config.early_stopping_patience:
            break
    if best_state is None:
        raise RuntimeError("pretraining produced no finite checkpoint")
    model.load_state_dict(best_state, strict=True)

    final_metrics = _evaluate(
        model,
        validation_loader,
        resolved_device,
        optionality_baseline=optionality_baseline,
    )
    metrics = {
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "validation_total_loss": final_metrics.pop("total_loss"),
        **final_metrics,
        "history": history,
        "smoke_mode": smoke_mode,
    }
    gates = _gate_results(metrics, config)
    metrics["gates"] = gates

    checkpoint_path = write_extractor_checkpoint(
        model,
        output_dir / "candidate_encoder_pretrained.pt",
        observation_schema_version=4,
        config_sha256=config_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        best_epoch=best_epoch,
    )
    metrics_path = output_dir / "pretraining_metrics.json"
    _atomic_json(metrics_path, metrics)
    return publish_pretraining_completion(
        checkpoint_path,
        metrics_path,
        output_dir / "PRETRAINING_COMPLETE.json",
        config_sha256=config_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        gates=gates,
        smoke_mode=smoke_mode,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pretrain the candidate CNN feature extractor"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-state-count", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args(argv)
    marker = train_candidate_encoder(
        args.config,
        args.dataset_root,
        args.output_dir,
        smoke_state_count=args.smoke_state_count,
        max_epochs=args.max_epochs,
        device=args.device,
    )
    print(marker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
