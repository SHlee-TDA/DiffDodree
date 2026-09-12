from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DIR = PROJECT_ROOT / "tutorials"
EXPECTED_NOTEBOOKS = (
    "01_data_inspection.ipynb",
    "02_structure_graph.ipynb",
    "03_model_training.ipynb",
    "04_model_sampling_tutorial.ipynb",
    "05_hwanip_counterfactual_experiment.ipynb",
    "06_evaluation_and_export.ipynb",
)

# Canonical Cell 1, with the DiffDodree Drive directory set for the GitHub release.
CANONICAL_SETUP_SHA256 = "862dd093fc639f90e4290f5195fb7be4253cce6fda5bc8fef832ccb5d86c8600"


def _source(cell: Mapping[str, Any]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def _load_notebook(name: str) -> Mapping[str, Any]:
    return json.loads((TUTORIAL_DIR / name).read_text(encoding="utf-8"))


def test_exact_tutorial_inventory() -> None:
    actual = tuple(sorted(path.name for path in TUTORIAL_DIR.glob("*.ipynb")))
    assert actual == EXPECTED_NOTEBOOKS
    assert not (TUTORIAL_DIR / "00_environment_setup.ipynb").exists()


def test_every_notebook_starts_with_exact_canonical_setup() -> None:
    assert CANONICAL_SETUP_SHA256 != "TO_BE_FILLED_BY_BUILDER"
    for name in EXPECTED_NOTEBOOKS:
        notebook = _load_notebook(name)
        cells = notebook.get("cells", [])
        assert cells, f"{name} has no cells"
        first = cells[0]
        assert first["cell_type"] == "code", f"{name} Cell 1 must be code"
        assert "environment-setup" in first.get("metadata", {}).get("tags", [])
        source = _source(first).strip()
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        assert digest == CANONICAL_SETUP_SHA256, f"{name} Cell 1 differs from the canonical block"


def test_device_resolution_and_no_hardcoded_device_calls() -> None:
    forbidden = (
        re.compile(r"\.cuda\s*\("),
        re.compile(r"torch\.device\s*\(\s*['\"]cuda(?:[^'\"]*)['\"]\s*\)"),
    )
    for name in EXPECTED_NOTEBOOKS:
        notebook = _load_notebook(name)
        code = "\n".join(
            _source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"
        )
        assert "resolve_device(" in _source(notebook["cells"][0])
        for pattern in forbidden:
            assert pattern.search(code) is None, f"{name} contains {pattern.pattern}"


def test_notebooks_have_no_committed_binary_or_large_outputs() -> None:
    for name in EXPECTED_NOTEBOOKS:
        notebook = _load_notebook(name)
        for cell in notebook["cells"]:
            assert cell.get("outputs", []) == [], f"{name} contains committed Cell output"
        assert (TUTORIAL_DIR / name).stat().st_size < 250_000
