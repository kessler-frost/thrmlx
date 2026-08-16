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

The source compatibility ledger currently tracks all 60 tests collected from THRML v0.1.4. The
block-state foundation has **14 / 60 translated objectives green**, so this is not yet full THRML
parity. The current Ising adapter and newly compatible node/block state layer are followed by
generic sampling, factors, observers, and training in ledger order. See [UPSTREAM.md](UPSTREAM.md).

## Preliminary Ising-only Apple-Silicon baseline

On 2026-08-16, the primary two-block dense RBM workload ran on a Mac mini with Apple M4 Pro and
48 GB memory (macOS 26.5.2, Python 3.12.12). The request uses 128 visible + 128 latent spins,
16,384 bipartite edges, 1,024 chains, 20 warmup sweeps, 32 recorded states per chain, and seven
warm repetitions. The full machine, package, workload, and raw-timing provenance is committed in
[the result JSON](benchmarks/results/2026-08-16-m4-pro.json).

| Adapter | Device | Cold time to first result | Warm median time | Warm recorded states/s |
| --- | --- | ---: | ---: | ---: |
| `thrmlx` 0.1.0 / MLX 0.32.0 | Metal GPU | 80.1 ms | 8.94 ms | 3,664,966 |
| THRML 0.1.4 / JAX 0.11.0 | JAX CPU | 1.30 s | 532.8 ms | 61,504 |

For this specific, sufficiently batched local workload, thrmlx records states about 59.6× faster.
This is an **Ising-only preliminary result**, not a claim about every THRML use case. It is not a
same-accelerator framework comparison: MLX uses Metal while the [official JAX macOS installation
path](https://docs.jax.dev/en/latest/installation.html) is CPU. A JIT-enabled THRML/JAX-Metal smoke
program is currently incompatible on this Mac, so no JAX-Metal throughput row is published. Each
upstream objective will receive a paired benchmark only after its translated compatibility test is
green.

## Quick start

Requires Python 3.10+, uv, Apple Silicon, and macOS 14 or newer.

```bash
git clone https://github.com/kessler-frost/thrmlx.git
cd thrmlx
uv sync --frozen --group dev
uv run --frozen --group dev python examples/two_spin.py
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
uv run --frozen --group benchmark python -m benchmarks.run \
  --output benchmarks/results/local.json
python3 scripts/teardown.py
```

It emits JSON containing hardware/software provenance, the complete workload/schedule, cold time
to first result, seven materialized warm timings, and separate MLX/THRML devices. It intentionally
does not turn a failed adapter into a partial comparison.

## Scope and roadmap

The current compatibility work proceeds through node/block-state management, generic block
sampling, discrete factors and observers, then Ising learning and the MNIST fixture. The benchmark
matrix will cover the corresponding THRML workloads: line/grid Ising, bipartite RBM, categorical
factors, mixed grids, clamped positive phase, moment observers, contrastive gradients, and the
MNIST fixture. Results will identify the exact green upstream objectives they exercise.

## License and provenance

Apache-2.0. See [LICENSE](LICENSE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and
[UPSTREAM.md](UPSTREAM.md).
