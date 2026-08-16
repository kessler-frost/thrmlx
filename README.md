# thrmlx

[![CI](https://github.com/kessler-frost/thrmlx/actions/workflows/ci.yml/badge.svg)](https://github.com/kessler-frost/thrmlx/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

An MLX backend port of [THRML](https://github.com/extropic-ai/thrml) for fast local Apple-Silicon
execution. It keeps THRML's model-building, block-Gibbs sampling, observation, and
contrastive-learning surface, replacing JAX/Equinox with MLX.

> [!NOTE]
> Independent, unofficial source-derived port; not affiliated with or endorsed by Extropic.

**THRML v0.1.4 API · 60/60 translated objectives green · Apache-2.0**

## Install and migrate

```bash
uv add "thrmlx @ git+https://github.com/kessler-frost/thrmlx.git"
```

Replace the package root in THRML code:

```diff
-from thrml import Block, SamplingSchedule, SpinNode, sample_states
-from thrml.models import IsingEBM, IsingSamplingProgram
+from thrmlx import Block, SamplingSchedule, SpinNode, sample_states
+from thrmlx.models import IsingEBM, IsingSamplingProgram
```

`thrmlx` exports the pinned THRML root, `models`, and public submodule API, including original
argument names such as `SamplingSchedule(n_warmup=..., n_samples=..., steps_per_sample=...)`.
Use `mlx.core` arrays and `mx.random.key` in place of JAX arrays and keys. JAX/Equinox transforms
(`jit`, `vmap`, `grad`, PyTrees) are intentionally not emulated.

## THRML-style sampling

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

See [examples](examples) for native dense Ising, generic factors, discrete EBMs, and the compact
MNIST-shaped contrastive-learning fixture.

## Apple-Silicon benchmarks

Fresh local measurement on an M4 Pro / 48 GB Mac mini: MLX 0.32 Metal versus source-pinned THRML
0.1.4 through JAX CPU. These are five warm-repetition medians; they are **not** a same-accelerator
comparison.

| Workload group | thrmlx / Metal | THRML / JAX CPU | Speedup |
| --- | ---: | ---: | ---: |
| Dense bipartite Ising / RBM | 8.3 ms | 832.7 ms | 99.9× |
| Line and grid Ising sampling | 74.0–105.9 ms | 265.7–292.8 ms | 2.8–3.6× |
| Spin, categorical, and mixed factors | 97.5–155.0 ms | 305.0–385.2 ms | 2.2–3.9× |
| Moments and contrastive updates | 129.2–243.4 ms | 366.1–704.3 ms | 2.8–2.9× |

The [full result JSON](benchmarks/results/2026-08-16-m4-pro-source-matrix.json) has every
workload, raw repetitions, device labels, and mapped upstream objectives. Reproduce it with:

```bash
uv sync --frozen --group benchmark
uv run --frozen --group benchmark python -m benchmarks.source_matrix \
  --output benchmarks/results/local-source-matrix.json
```

## Develop

```bash
uv sync --frozen --group dev
uv run pytest -q
uv run ruff format --check .
uv run ruff check .
uv run ty check src
```

More detail: [upstream compatibility and boundaries](UPSTREAM.md), [project history](PROJECT_LOG.md),
and [licensing/provenance](THIRD_PARTY_NOTICES.md). Run `python3 scripts/teardown.py` to remove the
project-local environment.
