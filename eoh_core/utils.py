"""
Common utility helpers shared across the EOH codebase.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


def log(message: str) -> None:
    """Lightweight stdout logger with flush for long running jobs."""
    print(message, flush=True)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create *path* if it does not exist and return the resolved Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp() -> str:
    """Return a compact timestamp string for file naming."""
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def abort(message: str, exit_code: int = 1) -> None:
    """Print *message* to stderr and terminate the process."""
    print(message, file=sys.stderr, flush=True)
    raise SystemExit(exit_code)


def normalize_list(items: Iterable[str]) -> list[str]:
    """Return a deduplicated list while preserving the original order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        ordered.append(item)
        seen.add(item)
    return ordered
