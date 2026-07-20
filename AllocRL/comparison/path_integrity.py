"""Filesystem containment checks shared by durable comparison artifacts."""

from __future__ import annotations

import os
import stat
from pathlib import Path


def _is_link_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def resolve_direct_regular_file(
    root: str | Path,
    candidate: str | Path,
    *,
    label: str,
) -> Path:
    """Resolve one immediate child while rejecting links and path escapes."""
    directory = Path(root)
    path = Path(candidate)
    try:
        if _is_link_or_junction(directory) or _is_link_or_junction(path):
            raise ValueError(f"{label} must be a direct regular file")
        resolved_directory = directory.resolve(strict=True)
        resolved = path.resolve(strict=True)
        if not resolved_directory.is_dir() or resolved.parent != resolved_directory:
            raise ValueError(f"{label} must be a direct regular file")
        if not stat.S_ISREG(os.lstat(path).st_mode):
            raise ValueError(f"{label} must be a direct regular file")
    except FileNotFoundError:
        raise
    except (OSError, RuntimeError) as error:
        raise ValueError(f"{label} must be a direct regular file") from error
    return resolved
