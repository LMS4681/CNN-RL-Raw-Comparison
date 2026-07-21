from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .block import Block


DEFAULT_SPLIT_SEED = 20260716
DEFAULT_HOLDOUT_FRACTION = 0.20


@dataclass(frozen=True)
class BlockSourceSplit:
    training_blocks: Sequence[Block]
    holdout_blocks: Sequence[Block]
    manifest: dict


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _holdout_fraction(ship_no: str, split_seed: int) -> float:
    payload = f"{split_seed}:{ship_no}".encode("utf-8")
    prefix = hashlib.sha256(payload).digest()[:8]
    return int.from_bytes(prefix, "big", signed=False) / float(2**64)


def split_blocks_by_ship(
    blocks: Sequence[Block],
    source_path: str | Path,
    split_seed: int = DEFAULT_SPLIT_SEED,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
) -> BlockSourceSplit:
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")
    if any(not block.ship_no.strip() for block in blocks):
        raise ValueError("Every target must have a non-empty ship_no")

    holdout_ships = {
        ship_no
        for ship_no in {block.ship_no for block in blocks}
        if _holdout_fraction(ship_no, split_seed) < holdout_fraction
    }
    training = tuple(
        block.clone() for block in blocks if block.ship_no not in holdout_ships
    )
    holdout = tuple(
        block.clone() for block in blocks if block.ship_no in holdout_ships
    )
    if not training or not holdout:
        raise ValueError(
            "Ship split must produce non-empty training and holdout sources"
        )

    def month_counts(items: Sequence[Block]) -> dict[str, int]:
        counts = Counter(
            f"{block.in_date.year:04d}-{block.in_date.month:02d}"
            for block in items
        )
        return dict(sorted(counts.items()))

    training_ships = sorted({block.ship_no for block in training})
    holdout_ship_list = sorted(holdout_ships)
    manifest = {
        "split_seed": int(split_seed),
        "holdout_fraction": float(holdout_fraction),
        "source_sha256": sha256_file(source_path),
        "source_row_count": len(blocks),
        "source_month_counts": month_counts(blocks),
        "training_row_count": len(training),
        "holdout_row_count": len(holdout),
        "training_ship_nos": training_ships,
        "holdout_ship_nos": holdout_ship_list,
        "training_month_counts": month_counts(training),
        "holdout_month_counts": month_counts(holdout),
    }
    return BlockSourceSplit(training, holdout, manifest)


def write_split_manifest(path: str | Path, manifest: Mapping) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
