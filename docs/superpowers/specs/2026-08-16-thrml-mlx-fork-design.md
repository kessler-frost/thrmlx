# THRML-to-MLX Fork Design

## Outcome

thrmlx becomes an Apache-2.0, source-derived MLX port of upstream THRML v0.1.4 at
9c4e6fbb800f5e5c627122e668ff1b158ef3782b. Its purpose is to make THRML's model-building,
block-Gibbs sampling, observation, and contrastive-training use cases execute efficiently on
Apple Silicon through MLX rather than JAX.

The package and import name remain thrmlx. This avoids colliding with an installed upstream thrml
package while retaining a clear port identity. The port adopts THRML's domain vocabulary and
behavior: SpinNode, CategoricalNode, Block, SamplingSchedule, factorized discrete EBMs, sampling
programs, observers, Ising wrappers, sampled moments, and contrastive gradients. It deliberately
does not preserve JAX transforms, Equinox modules, JAX arrays, PyTree tracing, or bit-identical
random streams.

## Fork provenance

The repository will state plainly that it is a derivative port of upstream THRML. THIRD_PARTY_NOTICES,
README, and a new UPSTREAM.md will identify the upstream repository, Apache-2.0 license, pinned
commit, retained notices, divergence policy, and refresh procedure. The GitHub repository cannot
retroactively become a GitHub-network fork without replacing it, so it is described as a
source-derived fork/port and receives an upstream Git remote.

No upstream source is copied without preserving Apache-2.0 attribution. Code requiring a different
MLX representation is rewritten, and the provenance record distinguishes copied, adapted, and new
modules.

## Compatibility target

Upstream collects 60 tests, including its README example and MNIST training test. They exercise
nodes, blocks, packed global/block state, Gibbs scheduling, spin/categorical EBM factors,
interactions, observers, Ising moments, contrastive KL gradients, and end-to-end training.

The original JAX/Equinox test files cannot run unchanged against MLX: they import and assert JAX
arrays, PyTrees, jit, and gradient mechanics that do not exist in MLX. The acceptance suite is a
vendored, MLX-translated copy under tests/upstream_parity. Every translated test retains its
upstream file/test ID in a marker or name. tests/upstream_parity/manifest.json records the source
SHA, test ID, port-test path, status, and intent. A parity claim means all manifest entries are
green; skipped or unsupported entries are visible objective failures, not silently excluded tests.

## Architecture

An MLX-native StaticPlan owns node-to-slot packing, per-type block layout, interaction lowering,
active masks, factor data, and compiled update kernels. Public program objects retain THRML's
structural composition but carry ordinary Python objects plus MLX arrays.

nodes and blocks feed StaticPlan; StaticPlan lowers factors into a SamplingProgram; the sampling
program invokes MLX compiled sweeps and observers/readout.

A block update simultaneously samples all members from its pre-block state. A sweep updates
declared blocks in order. All random draws use explicit MLX keys. Boolean spins retain False=-1
and True=+1; categoricals use uint8. The current dense path is an optimization inside the generic
factor plan.

Contrastive gradients are analytic sampled-moment differences. MLX automatic differentiation may
verify closed-form energies, but it does not differentiate through the discrete sampler.

## Delivery sequence

1. Fork foundation: provenance/branding, upstream manifest, nodes, blocks, static packing, and
   translated block-management tests.
2. Generic sampler: program, schedule, conditionals, clamping, readout, and block-sampling tests.
3. Discrete EBM: interactions, spin/categorical factors, heterogeneous sampling, observers, and
   factor/discrete/observer tests.
4. Ising and learning: wrappers, moments, contrastive gradients, MNIST fixture, and translated
   Ising/training/README tests.
5. Performance matrix: every green upstream objective gets paired local benchmarks.

The benchmark matrix covers line/grid Ising, bipartite RBM, categorical factor, mixed heterogeneous
grid, clamped positive phase, observer/moment collection, contrastive-gradient step, and MNIST
fixture step. Every result records cold time, materialized warm median, work units, model/block
structure, chain/batch count, upstream-test coverage, device, versions, and raw repetitions.

## JAX comparator policy

JAX CPU is the supported macOS THRML reference. JAX Metal becomes a same-GPU comparator only if
upstream THRML executes a JIT-enabled end-to-end workload on Metal; record package/JAX versions and
ENABLE_PJRT_COMPATIBILITY. MLX Metal is the port's native backend.

On the development M4 Pro, the Apple Metal plugin initializes with JAX 0.4.30/0.4.35 and
jax-metal 0.1.0, but THRML fails because those JAX releases lack jax.tree.flatten_with_path. JAX
0.5.2 plus jax-metal 0.1.1 fails before the program runs with UNIMPLEMENTED:
default_memory_space is not supported. No JAX Metal throughput row is published until a full THRML
smoke program succeeds with JIT enabled; disabling JIT is not a valid performance workaround.

## Scope boundaries

This rebase does not publish to PyPI, change Apache-2.0, claim affiliation with Extropic, or
silently depend on JAX in production. It does not promise bitwise random-stream identity across
backends. The prior array-only API can remain only as a thin convenience adapter mapped to the
THRML-compatible object model.
