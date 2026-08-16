# Direct THRML API Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use inline test-driven development for every task. Steps use checkbox syntax for tracking.

**Goal:** Make `import thrmlx` a complete MLX-backed replacement for the pinned THRML public API after a root-module rename.

**Architecture:** Keep the existing implementation modules and normalize their consumer seams to upstream THRML names. The root package and `models` package become the only export adapters; no second package root or JAX compatibility layer is created.

**Tech Stack:** Python 3.10+, MLX 0.32, uv, pytest, Ruff, ty; upstream THRML 0.1.4 commit `9c4e6fbb800f5e5c627122e668ff1b158ef3782b`.

**Spec:** `docs/superpowers/specs/2026-08-16-thrmlx-direct-thrml-api-design.md`

## Global Constraints

- Keep `thrmlx` as the sole distribution and import root; do not package `thrml` or add JAX/Equinox runtime dependencies.
- Preserve every pinned upstream root/model export under `thrmlx` and `thrmlx.models` after the root rename.
- Retain the documented MLX `SamplingSchedule` vocabulary as aliases, but prioritize THRML public
  keyword names and reject ambiguous duplicate schedule aliases.
- Use real MLX program execution in tests, literal export expectations, and the full serial/parallel release gates before push.

### Task 1: Establish consumer-facing failing tests

**Files:**
- Create: `tests/test_thrmlx_thrml_api.py`

- [ ] Write literal root/model export, deep-module import, upstream-keyword, and real one-spin Gibbs tests.
- [ ] Run `uv run pytest tests/test_thrmlx_thrml_api.py -q` and confirm missing `models`, missing
  `AbstractFactorizedEBM`, and unsupported upstream keyword failures.

### Task 2: Normalize the public surface

**Files:**
- Modify: `src/thrmlx/__init__.py`, `src/thrmlx/models/__init__.py`, `src/thrmlx/schedule.py`,
  `src/thrmlx/block_management.py`, `src/thrmlx/block_sampling.py`

- [ ] Add missing package exports and normalize public keyword names without duplicating algorithms.
- [ ] Run `uv run pytest tests/test_thrmlx_thrml_api.py -q` and confirm the complete direct-import contract is green.
- [ ] Commit `feat: complete direct THRML API compatibility`.

### Task 3: Publish and verify the replacement contract

**Files:**
- Modify: `README.md`, `UPSTREAM.md`, `PROJECT_LOG.md`

- [ ] Document `uv add` Git installation, renamed imports, MLX array/key substitution, and explicit JAX exclusions.
- [ ] Run serial and xdist pytest, Ruff format/lint, ty, parity report, build, examples, and GitHub CI.
- [ ] Commit docs, push main, and verify the remote CI run.
