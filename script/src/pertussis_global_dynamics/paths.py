from __future__ import annotations

from pathlib import Path


def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "data").is_dir() and (parent / "outputs").is_dir() and (parent / "manuscript").is_dir():
            return parent
    return Path(__file__).resolve().parents[3]


ROOT = _find_project_root()
DATA = ROOT / "data"
RAW = DATA / "raw"
CLEAN = DATA / "clean"
DATA_INTERMEDIATE = DATA / "intermediate"
METADATA = DATA / "metadata"
OUTPUTS = ROOT / "outputs"
SCRIPT = ROOT / "script"
R = SCRIPT / "R"
MODELS = SCRIPT / "models"
FITTED = MODELS / "fitted_objects"
MANUSCRIPT = ROOT / "manuscript"
FIGURE_DATA = MANUSCRIPT / "figure_data"
REPORTS = MANUSCRIPT / "reports"
