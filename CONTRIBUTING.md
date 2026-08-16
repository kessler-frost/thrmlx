# Contributing

Start with the behavioral contract in
[`docs/superpowers/specs/2026-08-16-thrmlx-v0-design.md`](docs/superpowers/specs/2026-08-16-thrmlx-v0-design.md)
and the repository rules in [`AGENTS.md`](AGENTS.md).

Set up the project with `uv sync --all-groups`; undo local setup with `./scripts/teardown.sh`.
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

Pull requests should explain the behavioral change, its correctness evidence, and any measured
performance effect. Do not introduce a compatibility layer, sparse representation, or custom
Metal kernel without a concrete caller or benchmark demonstrating the need.
