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
