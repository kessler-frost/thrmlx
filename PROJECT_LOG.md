# Project log

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
