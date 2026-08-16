# thrmlx

[![CI](https://github.com/kessler-frost/thrmlx/actions/workflows/ci.yml/badge.svg)](https://github.com/kessler-frost/thrmlx/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

`thrmlx` is a small, MLX-native Python library for fast local Ising-model sampling. It provides
dense pairwise models, deterministic graph coloring, compiled batched block-Gibbs chains, explicit
random keys, and optional clamping without exposing JAX graph/program machinery.

> [!IMPORTANT]
> `thrmlx` is an independent, unofficial project. It is not affiliated with, sponsored by, or
> endorsed by Extropic. THRML is referenced only as a behavioral and conceptual influence.

Version 0.1 targets correctness and a clean local-Mac interface. Exact enumeration and fixed-key
statistical tests are release gates; benchmark results are measurements, not promises.

## Quick start

Requires Python 3.10+, uv, Apple Silicon, and macOS 14 or newer.

```bash
git clone https://github.com/kessler-frost/thrmlx.git
cd thrmlx
uv sync --all-groups
uv run python examples/two_spin.py
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
`uv sync --all-groups --extra cpu`; the same `python3 scripts/teardown.py` removes that environment.

Run the bounded local benchmark with:

```bash
uv run python benchmarks/dense_sampling.py
```

It emits JSON containing the model, block, schedule, MLX/device, elapsed-time, and throughput
metadata needed to compare runs honestly.

## Scope and roadmap

Version 0.1 deliberately uses dense coupling matrices. It does not include generic factors,
categorical variables, sparse storage, annealing, differentiation through discrete samples, a
THRML compatibility facade, or training APIs. The next candidate milestone is sampled moments and
contrastive KL-gradient estimation after the sampler contract and local benchmark baseline settle.

## License and provenance

Apache-2.0. See [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
