from __future__ import annotations

import sys
from pathlib import Path


def add_project_src_to_path() -> Path:
    """Make the local package importable when running scripts from the repo."""
    root = Path(__file__).resolve().parents[1]
    src = root / "script" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root
