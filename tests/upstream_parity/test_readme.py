"""Executable quick-start acceptance test for the source-style THRML API."""

import runpy
from pathlib import Path


def test_upstream_readme_quick_example() -> None:
    """Catch the documented THRML-style Ising example drifting from the public API."""

    root = Path(__file__).resolve().parents[2]
    namespace = runpy.run_path(root / "examples" / "thrml_ising.py", run_name="__main__")

    assert namespace["samples"].shape == (1_000, 5)
