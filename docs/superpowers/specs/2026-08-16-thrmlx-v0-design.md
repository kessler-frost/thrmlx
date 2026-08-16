# thrmlx v0.1 Design

## Outcome

`thrmlx` is an unofficial, MLX-native Python library for fast Ising-model sampling on local
Apple Silicon. Version 0.1 provides a small interface for constructing a pairwise Ising model,
validating or deriving colored update blocks, computing reduced energy, and running reproducible
batched block-Gibbs chains with optional clamping.

The north star is measured local performance without weakening distributional correctness.
Correctness is established first against exact enumeration on small models; benchmarks become a
gate only after the behavioral contract is stable.

## Relationship to THRML

The behavioral reference is Extropic's THRML v0.1.4 at commit
`9c4e6fbb800f5e5c627122e668ff1b158ef3782b`. `thrmlx` preserves the high-value semantics:

- boolean states with `False = -1` and `True = +1`;
- reduced Ising energy including inverse temperature;
- simultaneous conditionals within a valid color block;
- ordered block updates within a sweep;
- warmup followed by an immediate first recorded sample;
- immutable clamped spins;
- explicit functional random keys.

It does not copy THRML's JAX/Equinox implementation machinery. Public JAX concepts such as
PyTrees, caller-managed `vmap`, factor programs, and traced static objects do not cross the MLX
interface. The initial implementation is original code written against this behavioral contract.

## Considered approaches

### Mechanical THRML port

Retain `SpinNode`, `Block`, `IsingEBM`, `IsingSamplingProgram`, and `sample_states` almost
verbatim. This minimizes migration edits but preserves shallow concepts whose complexity comes
from JAX's graph compiler and transformation rules.

### Pure array library

Expose only functions over fields, dense coupling matrices, and state arrays. This is small but
makes validation, coloring, energy conventions, and compilation policy every caller's concern.

### Selected: MLX-native deep module with a future compatibility adapter

Expose three primary names—`Ising`, `SamplingSchedule`, and `sample`—plus the small `Clamp` value
carrier. `Ising` owns graph validation, deterministic coloring, reduced-energy semantics, and the
private sampling plan. `sample` owns state initialization, clamping, random-key derivation,
ordered sweeps, recording, batching, and compilation. A compatibility adapter may later expose
selected THRML names if real migration demand appears; v0.1 does not speculate on that interface.

## Public interface

```python
from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True, slots=True)
class SamplingSchedule:
    warmup: int = 0
    samples: int = 1
    sweeps_per_sample: int = 1


@dataclass(frozen=True, slots=True)
class Clamp:
    mask: mx.array
    values: mx.array


class Ising:
    def __init__(
        self,
        fields: mx.array,
        couplings: mx.array,
        blocks: Sequence[Sequence[int]] | None = None,
        *,
        beta: float = 1.0,
    ) -> None: ...

    @property
    def n_spins(self) -> int: ...

    @property
    def blocks(self) -> tuple[tuple[int, ...], ...]: ...

    def energy(self, state: mx.array) -> mx.array: ...


def sample(
    key: mx.array,
    model: Ising,
    schedule: SamplingSchedule,
    *,
    chains: int = 1,
    initial: mx.array | None = None,
    clamp: Clamp | None = None,
) -> mx.array: ...
```

Example:

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

`trace` is boolean with shape `(4096, 4, 2)`.

## Model semantics and validation

For signed spins `s` in `{-1, +1}^N`, fields `b`, symmetric dense couplings `J`, and inverse
temperature `beta`, the public reduced energy is

```text
energy(s) = -beta * (b @ s + 0.5 * s.T @ J @ s)
```

The conditional used to update spin `i` is

```text
P(s_i = +1 | rest) = sigmoid(2 * beta * (b_i + sum_j J_ij s_j))
```

`energy` accepts boolean state with shape `(..., N)` and returns floating values with shape
`(...)`. It includes `beta`, matching THRML's reduced-energy convention.

Construction validates once and may materialize scalar validation results:

- `fields` is a finite floating MLX array with shape `(N,)` and `N >= 1`;
- `couplings` is a finite floating MLX array with shape `(N, N)`;
- `couplings` is symmetric and has an exactly zero diagonal;
- `beta` is finite and strictly positive;
- provided blocks are a nonempty ordered partition of `range(N)`;
- every block is nonempty and contains no repeated or out-of-range index;
- no nonzero coupling connects two spins inside one block.

When `blocks` is omitted, `Ising` derives a deterministic first-fit greedy coloring in ascending
spin order from nonzero off-diagonal couplings. A coloring need not be minimal to be correct.
Block order is part of the chain definition because it changes finite trajectories.

Configuration type failures raise `TypeError`; invalid values and shapes raise `ValueError` with
the violated invariant in the message.

## Sampling semantics

The canonical state dtype is `mx.bool_`. `False` represents `-1`, and `True` represents `+1`.
`sample` always retains chain and sample axes and returns shape `(chains, schedule.samples, N)`.

`SamplingSchedule` requires `warmup >= 0`, `samples >= 1`, and `sweeps_per_sample >= 0`. Sampling
timing matches THRML:

1. Run `warmup` complete sweeps.
2. Record the first sample immediately.
3. Before each later sample, run `sweeps_per_sample` complete sweeps.

A sweep updates blocks in their declared order. Every spin in one block observes the same
pre-block state and is sampled simultaneously.

When `initial` is absent, `sample` draws an independent boolean initial state for every chain.
When supplied, it must be boolean with shape `(chains, N)`. A `Clamp` mask is boolean with shape
`(N,)` or `(chains, N)`; values are boolean and broadcastable to the mask. Clamped values override
initial state and are enforced after every block update. A mask entry of `False` leaves that spin
free and its corresponding value is ignored.

The caller supplies one explicit MLX key. `sample` never seeds or consumes MLX's implicit global
random state. Reusing a key with identical inputs reproduces the same trace for a pinned MLX
version and device; callers split keys to request independent traces. Bitwise compatibility across
MLX versions or with JAX is not promised.

## Implementation modules

```text
src/thrmlx/
  __init__.py       public interface exports
  model.py          Ising construction, validation, coloring, and energy
  schedule.py       SamplingSchedule and Clamp validation
  sampling.py       state preparation, compiled block updates, sweeps, recording
```

`Ising` is the deep module at the model seam. Its private implementation stores dense fields,
couplings, immutable blocks, and prebuilt MLX index arrays. Callers do not maintain adjacency,
node-index maps, masks, or kernels.

`sample` is the only execution seam. Its implementation may compile fixed-shape block updates or
a sweep and may change caching strategy without changing callers. Dense matrices are deliberate
for v0.1: they are clear, Metal-friendly, and sufficient for small and medium Ising models. Sparse
storage is introduced only after profiles establish a real workload and MLX primitive strategy.

## Correctness strategy

Every behavior is implemented red-green-refactor. Stable tests include:

- exact reduced energies for all states of one- and two-spin models;
- exact single-spin conditional probabilities;
- auto-coloring and validation of manual colorings;
- schedule timing, including an unchanged first draw when warmup is zero;
- permanent scalar and per-chain clamping;
- output dtype and unsqueezed shape for one and many chains/samples;
- deterministic repeated keys and independence from MLX global RNG seeding;
- empirical two-spin probabilities and moments against exact enumeration;
- valid nonminimal colorings producing the same stationary distribution;
- compiled and eager block-update parity where both paths exist.

Statistical tests use fixed keys, many parallel chains, and tolerances derived from sampling error.
They validate a target distribution rather than merely checking plausible-looking spins.

## Performance strategy

Correctness precedes optimization. The first benchmark records warm and compiled throughput for
representative dense models and reports model size, block sizes, chains, sweeps, MLX version,
device, and wall time. No performance claim is made from the two-spin example. A custom Metal
kernel or sparse adapter requires a benchmark demonstrating a material improvement while exact
distribution and parity tests remain green.

## Packaging and project policy

- Distribution and import name: `thrmlx`.
- Python: `>=3.10`.
- Runtime dependency: `mlx>=0.32,<0.33`.
- Build backend: `uv_build>=0.12.5,<0.13`.
- Development tools: uv, pytest, pytest-xdist, Ruff, and ty.
- License: Apache-2.0.
- Provenance: `THIRD_PARTY_NOTICES.md` records the THRML behavioral reference and pinned commit.
- Branding: README states that the project is unofficial and is not affiliated with or endorsed
  by Extropic. Extropic logos and branding are excluded.
- Setup documentation uses `uv sync --frozen`; `scripts/teardown.py` removes only project-local
  generated environments and caches.
- CI runs unit, lint, type, build, and package-smoke checks on Linux CPU MLX and Apple-Silicon
  macOS. CI performance numbers are informational, never release gates.

## Adopted Emma Frost guidance

The project adopts the generally useful practices from `emma-frost`:

- state a measurable north star and the next concrete deliverable;
- keep the default path thin and add infrastructure only after a measured blocker;
- iterate through change, test/evaluate, compare, and record;
- default to one agent and use bounded Terra/Luna parallel work only when it shortens the path;
- add focused tests at stable seams and for reproduced bugs;
- run broad tests, Ruff, ty, and build checks at merge and release checkpoints;
- use uv, pathlib, top-level imports, and structured results;
- keep project-owned execution single-process and thread-free.

ARC gameplay, MTPLX/Qwen lifecycle, Kaggle, recordings, and experiment-specific canaries are not
transferable and are excluded.

## Deferred work

The v0.1 scope excludes generic factors, categorical variables, custom conditional samplers,
sparse storage, contrastive training, automatic differentiation through sampling, annealing,
observers, a THRML compatibility facade, PyPI publication, and performance promises.

The next candidate is sampled-moment and contrastive KL-gradient estimation. It begins only after
v0.1 sampling correctness and benchmark evidence are recorded.
