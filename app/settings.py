"""Project-relative configuration helpers for the app."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configured_path(environment_variable: str, default_relative_path: str) -> Path:
    """Resolve a configured path, interpreting relative values from the project root."""
    configured = os.environ.get(environment_variable)
    path = Path(configured) if configured else PROJECT_ROOT / default_relative_path
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path
