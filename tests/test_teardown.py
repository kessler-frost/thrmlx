import runpy
from pathlib import Path
from shutil import copy2

import pytest


def copy_teardown(root: Path) -> Path:
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    source = Path(__file__).parents[1] / "scripts" / "teardown.py"
    destination = scripts / "teardown.py"
    copy2(source, destination)
    (root / "pyproject.toml").touch()
    return destination


def test_teardown_removes_only_named_project_generated_state(tmp_path: Path) -> None:
    root = tmp_path / "thrmlx"
    script = copy_teardown(root)
    generated = [root / ".venv", root / ".pytest_cache", root / "dist"]
    for directory in generated:
        directory.mkdir()
        (directory / "generated").touch()
    bytecode_cache = root / "src" / "thrmlx" / "__pycache__"
    bytecode_cache.mkdir(parents=True)
    sentinel = root / "keep" / "sentinel"
    sentinel.parent.mkdir()
    sentinel.touch()

    runpy.run_path(str(script), run_name="__main__")

    assert all(not directory.exists() for directory in generated)
    assert not bytecode_cache.exists()
    assert sentinel.is_file()


def test_teardown_refuses_an_unexpected_repository_name(tmp_path: Path) -> None:
    root = tmp_path / "another-project"
    script = copy_teardown(root)

    with pytest.raises(SystemExit, match="Refusing"):
        runpy.run_path(str(script), run_name="__main__")
