"""Taxonomy loading and validation."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAXONOMY_PATH = PROJECT_ROOT / "configs" / "taxonomy.json"


def load_taxonomy(path: Path | str = DEFAULT_TAXONOMY_PATH) -> dict:
    """Load the workbook-derived node taxonomy."""
    with Path(path).open("r", encoding="utf-8") as handle:
        taxonomy = json.load(handle)
    labels = [node["label"] for node in taxonomy["nodes"]]
    if len(labels) != len(set(labels)):
        raise ValueError("Taxonomy contains duplicate node labels")
    return taxonomy


def taxonomy_labels(path: Path | str = DEFAULT_TAXONOMY_PATH) -> set[str]:
    return {node["label"] for node in load_taxonomy(path)["nodes"]}
