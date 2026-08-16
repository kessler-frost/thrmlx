#!/usr/bin/env python3
"""Remove only generated, project-local thrmlx development state."""

from pathlib import Path
from shutil import rmtree


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if repo_root.name != "thrmlx" or not (repo_root / "pyproject.toml").is_file():
        raise SystemExit(f"Refusing to clean an unexpected directory: {repo_root}")

    generated_directories = (
        repo_root / ".venv",
        repo_root / ".pytest_cache",
        repo_root / ".ruff_cache",
        repo_root / ".ty",
        repo_root / "build",
        repo_root / "dist",
        repo_root / "src" / "thrmlx.egg-info",
    )
    for directory in generated_directories:
        rmtree(directory, ignore_errors=True)
    for cache in repo_root.rglob("__pycache__"):
        rmtree(cache, ignore_errors=True)

    print("Removed thrmlx's project-local environment, build artifacts, and tool caches.")


if __name__ == "__main__":
    main()
