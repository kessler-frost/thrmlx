# thrmlx

[![CI](https://github.com/kessler-frost/thrmlx/actions/workflows/ci.yml/badge.svg)](https://github.com/kessler-frost/thrmlx/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

thrmlx is a source-derived [THRML](https://github.com/extropic-ai/thrml) port with an MLX backend
for fast local Apple-Silicon execution. The target is THRML's model-building, block-Gibbs sampling,
observation, and contrastive-learning use cases, implemented with MLX instead of THRML's JAX/Equinox
backend.

> [!IMPORTANT]
> thrmlx is an independent, unofficial, source-derived port. It is not affiliated with, sponsored
> by, or endorsed by Extropic.

The source compatibility ledger tracks all 60 tests collected from THRML v0.1.4, and **60 / 60
MLX translations are green**. The port covers THRML-style nodes, blocks, generic samplers,
factorized spin/categorical/mixed EBMs, observers, Ising wrappers, sampled moments, contrastive
gradients, the quick-start example, and a compact MNIST-shaped end-to-end fixture. This is semantic
source compatibility, not JAX/Equinox transformation or random-bitstream compatibility; see
[UPSTREAM.md](UPSTREAM.md).

## Apple-Silicon THRML source-use-case matrix

Measured 2026-08-16 on an Apple M4 Pro / 48 GB Mac mini (macOS 26.5.2, Python 3.12.12), using
MLX 0.32.0 on Metal and source-pinned THRML 0.1.4 at
`9c4e6fbb800f5e5c627122e668ff1b158ef3782b` through JAX 0.11.0 CPU. Each row is a matched,
materialized source-style workload with five warm repetitions; raw samples, devices, and linked
upstream objectives are committed in [the result JSON](benchmarks/results/2026-08-16-m4-pro-source-matrix.json).

| THRML use case | thrmlx / MLX Metal warm median | THRML / JAX CPU warm median | Speedup |
| --- | ---: | ---: | ---: |
| Batched bipartite Ising / RBM | 8.27 ms | 492.31 ms | 59.5× |
| Sparse line Ising sampling | 52.65 ms | 239.59 ms | 4.6× |
| Checkerboard grid Ising sampling | 51.76 ms | 254.63 ms | 4.9× |
| Low-level spin-factor Gibbs program | 53.63 ms | 249.40 ms | 4.7× |
| Categorical factor Gibbs program | 86.93 ms | 332.35 ms | 3.8× |
| Mixed spin/categorical factor program | 50.93 ms | 301.49 ms | 5.9× |
| Ising moment observer | 65.95 ms | 227.28 ms | 3.4× |
| Semi-visible contrastive gradient | 130.58 ms | 547.79 ms | 4.2× |
| MNIST-shaped contrastive update | 94.39 ms | 420.46 ms | 4.5× |

This is a local Apple-Silicon outcome, not a same-accelerator framework comparison: MLX uses Metal
while this upstream THRML installation runs through JAX CPU. A JIT-enabled THRML/JAX-Metal smoke
program remains incompatible on this Mac, so no JAX-Metal throughput claim is published. The matrix
is deliberately source-use-case-specific, not a general GPU or cloud-price comparison.

## Quick start

Requires Python 3.10+, uv, Apple Silicon, and macOS 14 or newer.

```bash
git clone https://github.com/kessler-frost/thrmlx.git
cd thrmlx
uv sync --frozen --group dev
uv run --frozen --group dev python examples/two_spin.py
uv run --frozen --group dev python examples/generic_block_sampling.py
uv run --frozen --group dev python examples/factor_sampling.py
uv run --frozen --group dev python examples/discrete_ebm.py
uv run --frozen --group dev python examples/thrml_ising.py
```

Undo the project-local setup with:

```bash
python3 scripts/teardown.py
```

The same example as library code:

```python
import mlx.core as mx

from thrmlx import Ising, SamplingSchedule, sample

model = Ising(
    fields=mx.array([0.0, 0.0]),
    couplings=mx.array([[0.0, 0.8], [0.8, 0.0]]),
)
trace = sample(
    mx.random.key(7),
    model,
    SamplingSchedule(warmup=200, samples=4),
    chains=4096,
)
```

`trace` is a boolean MLX array with shape `(4096, 4, 2)`. Both the chain and sample axes stay
present when their size is one.

### THRML-style Ising program

The source-style node/block program API is available when its graph representation is a better fit:

```python
import mlx.core as mx

from thrmlx import Block, SamplingSchedule, SpinNode, sample_states
from thrmlx.models import IsingEBM, IsingSamplingProgram, hinton_init

nodes = [SpinNode() for _ in range(5)]
edges = [(nodes[i], nodes[i + 1]) for i in range(4)]
model = IsingEBM(
    nodes,
    edges,
    mx.zeros((5,), dtype=mx.float32),
    mx.full((4,), 0.5, dtype=mx.float32),
    mx.array(1.0, dtype=mx.float32),
)
free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
initial_key, sampling_key = mx.random.split(mx.random.key(0), 2)
samples = sample_states(
    sampling_key,
    program,
    SamplingSchedule(warmup=100, samples=1_000, sweeps_per_sample=2),
    hinton_init(initial_key, model, free_blocks, ()),
    [],
    [Block(nodes)],
)[0]
assert samples.shape == (1_000, 5)
```

## Semantics

Boolean states are canonical: `False = -1` and `True = +1`. For signed state `s`, fields `b`,
symmetric zero-diagonal couplings `J`, and inverse temperature `beta`, `model.energy(state)` returns
the reduced energy

```text
-beta * (b @ s + 0.5 * s.T @ J @ s)
```

The Gibbs conditional is `P(s_i = +1 | rest) = sigmoid(2 * beta * local_field_i)`.

When update blocks are omitted, `Ising` derives a deterministic first-fit coloring from nonzero
couplings. A supplied coloring must partition every spin exactly once and cannot put coupled spins
in the same block. Blocks update in declaration order; spins within one block update simultaneously.

Schedule timing follows THRML: run `warmup` sweeps, record the first sample immediately, then run
`sweeps_per_sample` sweeps before every later sample. Setting `sweeps_per_sample=0` intentionally
records the same post-warmup state repeatedly.

### Clamping

```python
from thrmlx import Clamp

clamp = Clamp(
    mask=mx.array([True, False]),
    values=mx.array([True, False]),
)
trace = sample(
    mx.random.key(8),
    model,
    SamplingSchedule(warmup=100, samples=10),
    chains=1024,
    clamp=clamp,
)
```

A one-dimensional mask is shared by all chains. A `(chains, spins)` mask can fix different sites
per chain. Values broadcast to the mask, override the initial state, and remain enforced after
every block update.

### Random keys

`sample` requires one explicit MLX PRNG key and never reads or seeds MLX's implicit global random
state. Reusing a key with the same inputs reproduces the same trace for a pinned MLX version and
device. Split keys for independent calls:

```python
first_key, second_key = mx.random.split(mx.random.key(9), 2)
```

Bitwise compatibility across MLX versions, devices, or JAX is not promised.

## Development and verification

```bash
uv run pytest -q
uv run ruff format --check .
uv run ruff check .
uv run ty check src
uv build
```

Linux CPU MLX is a secondary semantic-checking target. Set it up with
`uv sync --frozen --group dev --extra cpu`; the same `python3 scripts/teardown.py` removes that
environment.

Reproduce the paired local benchmark with its optional, pinned THRML/JAX dependency group:

```bash
uv sync --frozen --group benchmark
uv run --frozen --group benchmark python -m benchmarks.source_matrix \
  --output benchmarks/results/local-source-matrix.json
python3 scripts/teardown.py
```

It emits JSON containing hardware/software provenance, cold time to first result, five materialized
warm timings, source-objective links, and separate MLX/THRML devices. It intentionally does not
turn a failed adapter into a partial comparison.

## Scope and roadmap

The source-compatibility implementation is complete. The benchmark matrix covers the corresponding
THRML workloads: line/grid Ising, bipartite RBM, categorical factors, mixed grids, clamped positive
phase, moment observers, contrastive gradients, and the MNIST fixture. Every result identifies the
green upstream objectives it exercises, its accelerator, and whether the JAX comparison is CPU-only.

## License and provenance

Apache-2.0. See [LICENSE](LICENSE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and
[UPSTREAM.md](UPSTREAM.md).
