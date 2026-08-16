# Project log

## 2026-08-16 — Direct `thrmlx` THRML API compatibility

- Made the pinned THRML v0.1.4 public surface directly available under the `thrmlx` root: root
  exports, `thrmlx.models`, and all documented public module paths now retain THRML names while
  using MLX arrays and explicit MLX keys.
- Aligned upstream public keyword names for block specs, schedule construction, state extraction,
  block sampling, scheduled sampling, and observer-driven sampling. `SamplingSchedule` also
  retains the original MLX-native spellings as non-ambiguous aliases.
- Added direct-import contract tests that validate every pinned public export and execute an
  end-to-end source-style factor Gibbs program. The boundary intentionally excludes JAX/Equinox
  transformations and random-bitstream identity; callers replace the package root and migrate
  direct JAX calls to `mlx.core`.
- Reran the complete nine-case source-workload matrix after the compatibility changes. The updated
  five-repetition local result is 2.2×–99.9× MLX Metal versus the pinned THRML/JAX CPU adapter;
  raw repetitions, hardware/software provenance, and device labels remain in the committed result
  JSON rather than being reduced to a benchmark claim without context.

## 2026-08-16 — Complete source-use-case matrix and batched MLX execution

- Added a reproducible nine-row source-use-case matrix covering the completed computational THRML
  surface: dense RBMs, line/grid Ising models, spin/categorical/mixed factor programs, moment
  observers, semi-visible contrastive gradients, and the MNIST-shaped fixture update. Every row
  records the mapped green source objectives, full hardware/software provenance, materialized cold
  timing, and five warm timings in
  `benchmarks/results/2026-08-16-m4-pro-source-matrix.json`.
- On the Apple M4 Pro / 48 GB development Mac, MLX Metal was 3.4×–59.5× faster than the pinned
  upstream THRML/JAX CPU adapter across those matched local workloads. This remains explicitly a
  Metal-versus-CPU result, not a same-accelerator or cloud-pricing claim.
- Removed Python-side per-chain traversal from contrastive gradients. Generic block sampling now
  preserves leading batch axes through factor interactions and observers, so positive/negative
  phases and the MNIST-shaped update execute as native MLX batches. Regression tests cover batched
  spin conditionals and batched moment accumulation.

## 2026-08-16 — Observer, Ising, and fixture parity

- Ported observers and the source-style Ising API: `StateObserver`, mixed-value moment
  accumulation, observation-aware sampling, sparse-edge `IsingEBM`, `IsingSamplingProgram`,
  Hinton initialization, sampled first/second moments, and two-phase contrastive gradients.
- Added the executable THRML-style Ising README example. It exercises the same public API covered
  by the translated upstream quick-start objective.
- Added a deterministic, compact 28-by-28 binary-image/two-label fixture that makes one
  contrastive update and classifies the fixture at 100%. It preserves the upstream test's
  end-to-end EBM-training intent while avoiding the upstream repository's 30 MB MNIST test-data
  payload in this source-derived MLX repository.
- The compatibility ledger is complete: all 60 pinned THRML v0.1.4 acceptance objectives have a
  green MLX translation. This does not claim JAX transform or random-bitstream compatibility.

## 2026-08-16 — Discrete EBM parity

- Ported the factorized discrete EBM layer: spin, categorical, and mixed factors lower into static
  directed interactions backed by MLX arrays; exact Boolean Bernoulli and categorical-softmax
  conditionals consume those interactions without JAX or Equinox.
- Translated and passed all 24 upstream discrete-EBM objectives: factor validation, interaction
  lowering, binary/categorical/mixed Boltzmann marginals, clamped triplets, ragged mixed updates,
  energies, equivalent representations, a mixed checkerboard grid, and a 1,024-node grid sweep.
  Compatibility reached 52 green / 8 planned objectives at this milestone.
- Added `examples/discrete_ebm.py`, a runnable THRML-style coupled spin/categorical sampling
  program. Benchmark rows for this newly green surface follow after the remaining API work and
  stable paired measurement harness are in place.

## 2026-08-16 — v0.1 direction

- Chose an MLX-native deep interface over a mechanical port of THRML's JAX graph/program API.
- Fixed the initial public seam at `Ising`, `SamplingSchedule`, `Clamp`, and `sample`.
- Selected dense symmetric coupling matrices and deterministic greedy coloring for v0.1.
- Made exact enumeration and statistical agreement the correctness gate before optimization.
- Deferred training, generic factors, sparse storage, a THRML compatibility facade, and PyPI
  publication.

## 2026-08-16 — sampler correctness baseline

- Verified 88 tests serially in 0.25 seconds and with xdist in 2.31 seconds on the development Mac;
  both runs reported 88 passed, zero failed, and zero errors. Retained serial pytest as the normal
  gate because parallel startup overhead dominates this small suite.
- Ran the fixed-key distribution file three times; every run reported five passed.
- Proved the distribution tests detect a Gibbs sign error by temporarily changing the conditional
  from `sigmoid(2 * beta * local_field)` to its negative. All five distribution tests failed after
  verifying the mutation was present, then passed after restoring and verifying the correct sign.
- Added a structured local benchmark. Performance remains informational until representative
  workloads and repeated warm-run results establish a baseline.

## 2026-08-16 — Apple-Silicon performance baseline

- Added a framework-neutral, two-block dense RBM workload and paired adapters: `thrmlx` on MLX
  Metal and upstream THRML v0.1.4 at `9c4e6fbb800f5e5c627122e668ff1b158ef3782b` on JAX CPU.
  THRML/JAX remain a pinned `benchmark` dependency group, never a `thrmlx` runtime dependency;
  PyPI publication remains out of scope.
- Measured the Mac mini M4 Pro / 48 GB primary request (256 spins, 16,384 edges, 1,024 chains,
  20 warmup sweeps, 32 samples, seven warm repetitions). MLX recorded 3,664,966 warm states/s
  (8.94 ms median); THRML/JAX CPU recorded 61,504 (532.8 ms median). Cold timings were 80.1 ms
  and 1.30 s respectively. The committed JSON is the source of truth for unrounded values.
- Kept the table explicitly device-labeled and declined to call it a same-accelerator comparison.
  Future JAX Metal, sparse-topology, or custom-Metal-kernel work must add a separately reproducible
  result rather than overwrite this baseline.

## 2026-08-16 — THRML MLX fork foundation

- Reframed thrmlx as a source-derived THRML v0.1.4 port with an MLX backend and committed a
  60-objective upstream compatibility ledger. The original JAX/Equinox tests cannot execute
  unchanged against MLX, so the ledger maps each objective to a translated MLX test and makes
  incomplete work visible.
- Ported the fourteen upstream block-management objectives: identity-stable nodes, same-type
  blocks, static global/block state locations, nested tuple/dictionary templates, allocation,
  round trips, and state validation. The current status is 14 green / 46 planned; it is not a
  full THRML parity claim.
- Retained the existing MLX-native dense Ising adapter as a convenience layer. The next port
  target is generic THRML block sampling; benchmark rows follow only when their corresponding
  objective is green.
- Probed JIT-enabled THRML on JAX Metal without disabling JIT. JAX Metal initialized on the M4
  Pro, but the documented 0.4.30/0.4.35 compatibility pair lacked jax.tree.flatten_with_path used
  by THRML; the newer 0.5.2/0.1.1 pair failed with default_memory_space unsupported. No JAX-Metal
  performance row is published.

## 2026-08-16 — Generic THRML block sampling

- Ported THRML's generic program seam to MLX: conditional samplers receive statically lowered
  interaction slices, active masks, and explicit MLX tail state; BlockGibbsSpec preserves free
  superblock ordering and clamped blocks.
- Translated and passed all six upstream block-sampling objectives, including scheduled recording,
  sampler-count/state guardrails, and nested tuple/dictionary state. Compatibility now stands at
  20 green / 40 planned objectives.
- Added generic_block_sampling.py as a non-Ising THRML-style example. It is intentionally a
  correctness example rather than a fabricated framework-performance result; factor and observer
  ports are the next benchmark-eligible surface.

## 2026-08-16 — THRML factor and interaction contracts

- Ported abstract and weighted factors, factor-backed sampling-program lowering, and strict
  directed-interaction validation to MLX. A FactorSamplingProgram now composes factor-generated
  interactions with explicit interactions before static generic-program lowering.
- Translated and passed the five upstream factor objectives and three interaction objectives.
  Compatibility now stands at 28 green / 32 planned objectives.
- Added factor_sampling.py, a THRML-style weighted pair-factor program whose recorded MLX trace
  is exercised in Apple Silicon CI. Discrete spin/categorical EBM factors are next.
