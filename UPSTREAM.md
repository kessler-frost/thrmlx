# Upstream THRML provenance

thrmlx is a source-derived, MLX backend port of
[THRML](https://github.com/extropic-ai/thrml), Extropic's Apache-2.0 licensed
probabilistic-computing simulator.

## Pinned reference

| Field | Value |
| --- | --- |
| Repository | https://github.com/extropic-ai/thrml |
| Version | 0.1.4 |
| Commit | 9c4e6fbb800f5e5c627122e668ff1b158ef3782b |
| Collected upstream acceptance objectives | 60 |
| License | Apache License 2.0 |

The reference commit, rather than a moving branch, defines the behavior that
the compatibility ledger measures.

## What is retained and what changes

The port retains THRML's problem-domain concepts: nodes, blocks, state
packing, sampling schedules, discrete energy-based models, observers, Ising
wrappers, sampled moments, and contrastive gradient estimation. It changes
the execution backend from JAX/Equinox to MLX arrays and Metal-oriented
execution for local Apple Silicon.

thrmlx deliberately does not promise JAX array compatibility, Equinox
PyTrees, JAX transformations, or bit-identical random streams. Every upstream
objective is represented in [tests/upstream_parity/manifest.json](tests/upstream_parity/manifest.json).
An objective is only described as compatible after its MLX-translated test is
green.

The pinned suite's MNIST test depends on approximately 30 MB of upstream
NumPy fixtures and JAX/Optax training code. Its MLX translation is a
deterministic compact fixture with the same 28-by-28 binary-image shape,
visible labels, contrastive update, and classification acceptance check. The
exact upstream dataset and optimizer implementation are intentionally not
vendored; this divergence is documented in the test and project log.

## Attribution and source policy

THRML is Apache-2.0 licensed. Files copied verbatim from THRML retain their
upstream copyright and license header. Files structurally adapted for MLX
name the reference in their module documentation. New code is identified as
new in its commit message or nearby documentation. This repository does not
claim Extropic affiliation, sponsorship, or endorsement.

The existing GitHub repository is not in THRML's GitHub fork network; the
accurate description is a source-derived fork or port.

## Refresh procedure

With an upstream checkout at the pinned commit, collect the reference suite
using the dependency environment from this repository:

~~~bash
cd /path/to/thrml
uv run --project /path/to/thrmlx --frozen --group benchmark \
  --with 'networkx>=3,<4' --with 'optax>=0.2,<0.3' pytest --collect-only -q tests
~~~

Compare that output with the literal upstream IDs in the ledger before
changing the pin. Update the pin, the manifest's objectives, and translated
tests as one reviewable change. A new pin does not inherit the old
compatibility result.
