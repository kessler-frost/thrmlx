# Contributing

Start with the behavioral contract in
[`docs/superpowers/specs/2026-08-16-thrmlx-v0-design.md`](docs/superpowers/specs/2026-08-16-thrmlx-v0-design.md)
and the repository rules in [`AGENTS.md`](AGENTS.md).

Set up the normal development environment with `uv sync --frozen --group dev`; undo local setup with
`python3 scripts/teardown.py`.
Implement behavior test-first and use hand-derived expected values or exact enumeration rather than
mirroring production formulas in tests.

Checkpoint commands:

```bash
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run ty check src
uv build
```

The paired local performance benchmark is opt-in because it installs pinned upstream THRML and JAX:

```bash
uv sync --frozen --group benchmark
uv run --frozen --group benchmark python -m benchmarks.run --smoke
python3 scripts/teardown.py
```

Never compare a Metal result to a JAX CPU result without naming both devices and retaining cold and
warm timing modes separately. CI validates benchmark contracts but does not publish shared-runner
timings.

The test suite currently runs faster serially. `uv run pytest -n auto` is an available comparison
gate; keep it only when its full pass/fail/error counts match the serial run, and prefer it as the
default only after measured wall time improves.

To smoke-test the built wheel without pip or a persistent environment:

```bash
uv run --isolated --no-project --with ./dist/thrmlx-0.1.0-py3-none-any.whl python -c "import thrmlx; print(thrmlx.__version__)"
```

Pull requests should explain the behavioral change, its correctness evidence, and any measured
performance effect. Do not introduce a compatibility layer, sparse representation, or custom
Metal kernel without a concrete caller or benchmark demonstrating the need.
