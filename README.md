# thrmlx

[![CI](https://github.com/kessler-frost/thrmlx/actions/workflows/ci.yml/badge.svg)](https://github.com/kessler-frost/thrmlx/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

Run [THRML](https://github.com/extropic-ai/thrml) models locally on Apple Silicon with MLX.

`thrmlx` is an independent, source-derived fork of THRML. It preserves the THRML 0.1.4 model and
sampling APIs while replacing the JAX/Equinox runtime with MLX and Metal.

## What it supports

- Ising and factorized energy-based models
- Spin, categorical, and mixed-variable factors
- Block Gibbs sampling, clamping, observers, moments, and contrastive gradients
- The complete pinned THRML API under the `thrmlx` package root

All 60 translated objectives from the THRML 0.1.4 test suite pass. See
[UPSTREAM.md](UPSTREAM.md) for the compatibility boundary and source provenance.

## Install

Add the public Git repository to any uv project:

```bash
uv add "thrmlx @ git+https://github.com/kessler-frost/thrmlx.git"
```

Then replace the package root in THRML-facing imports:

```diff
-from thrml import Block, SamplingSchedule, SpinNode, sample_states
-from thrml.models import IsingEBM, IsingSamplingProgram
+from thrmlx import Block, SamplingSchedule, SpinNode, sample_states
+from thrmlx.models import IsingEBM, IsingSamplingProgram
```

Use `mlx.core` arrays and `mx.random.key` instead of JAX arrays and keys. JAX/Equinox features
such as `jit`, `vmap`, `grad`, and PyTree integration are backend-specific and are not emulated.

## Example

```python
import mlx.core as mx

from thrmlx import Block, SamplingSchedule, SpinNode, sample_states
from thrmlx.models import IsingEBM, IsingSamplingProgram, hinton_init

nodes = [SpinNode() for _ in range(5)]
edges = list(zip(nodes, nodes[1:]))
model = IsingEBM(nodes, edges, mx.zeros(5), mx.full(4, 0.5), mx.array(1.0))

free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
init_key, sample_key = mx.random.split(mx.random.key(0), 2)

samples = sample_states(
    sample_key,
    program,
    SamplingSchedule(n_warmup=100, n_samples=1_000, steps_per_sample=2),
    hinton_init(init_key, model, free_blocks, ()),
    [],
    [Block(nodes)],
)[0]

assert samples.shape == (1_000, 5)
```

Blocks are graph-color groups: connected nodes must not be updated in the same block. More examples
cover [generic factors](examples/factor_sampling.py),
[discrete EBMs](examples/discrete_ebm.py), and
[contrastive learning](examples/train_mnist_fixture.py).

## Performance

Measured on an M4 Pro Mac mini with five warm repetitions per workload. `thrmlx` used MLX 0.32 on
Metal; pinned THRML 0.1.4 used JAX CPU because its JAX-Metal stack was incompatible on this Mac.
These results compare the available local backends, not the frameworks on the same accelerator.

| Workload | thrmlx / Metal | THRML / JAX CPU | Relative |
| --- | ---: | ---: | ---: |
| Dense bipartite Ising / RBM | 8.3 ms | 832.7 ms | 99.9× |
| Line and grid Ising sampling | 74.0–105.9 ms | 265.7–292.8 ms | 2.8–3.6× |
| Spin, categorical, and mixed factors | 97.5–155.0 ms | 305.0–385.2 ms | 2.2–3.9× |
| Moments and contrastive updates | 129.2–243.4 ms | 366.1–704.3 ms | 2.8–2.9× |

The [full benchmark result](benchmarks/results/2026-08-16-m4-pro-source-matrix.json) includes all
nine workloads, raw repetitions, device labels, and their matching upstream objectives.

## Development

```bash
uv sync --frozen --group dev
uv run pytest -q
uv run ruff format --check .
uv run ruff check .
uv run ty check src
uv build --no-sources
```

Run `python3 scripts/teardown.py` to remove the project environment. See
[PROJECT_LOG.md](PROJECT_LOG.md) for milestones and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for licensing and attribution.

`thrmlx` is not affiliated with, sponsored by, or endorsed by Extropic.
