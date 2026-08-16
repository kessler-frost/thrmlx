# Apple-Silicon Performance Baseline Design

## Outcome

This milestone adds a reproducible, evidence-backed Apple-Silicon performance baseline for
`thrmlx`. It compares the public sampler to upstream THRML running on JAX and places the measured
result near the top of the README. The benchmark is a locally run development tool; CI verifies its
contract but does not publish timings from shared hardware.

The goal is to measure a representative local-Mac workload accurately, not to make a universal
performance claim. Every reported result identifies the backend and execution device, hardware,
software versions, workload, schedule, and timing mode.

## Comparison contract

The benchmark runs one identical bipartite dense Ising / RBM workload through two independent
adapters:

| Adapter | Runtime | Device on the baseline Mac | Purpose |
| --- | --- | --- | --- |
| `thrmlx` | MLX 0.32 | Apple Metal GPU | The library's intended local execution path. |
| `thrml` | upstream THRML v0.1.4 on JAX | JAX CPU backend | The behavioral reference implementation on the officially supported local JAX path. |

The upstream THRML package is installed from its pinned public Git commit, not copied into this
repository. It is a benchmark-only dependency group and cannot become a runtime dependency of
`thrmlx`.

JAX's official installation guide documents standard macOS JAX as CPU and calls Mac GPU support
experimental. Consequently the comparison must always label the JAX number as CPU. It is useful to
someone choosing a fast local-Mac sampler, but it is not a same-accelerator framework benchmark.
The report may add a JAX Metal result only if an independently verified, pinned Apple plugin makes
the same workload execute on Metal; it must remain a separate row.

## Workload

The primary workload is a deterministic, fully connected bipartite RBM with 128 visible and 128
latent spins (256 total). It has two update blocks, one for each partition. Fields and weights are
generated in host NumPy from a documented seed and are converted to each runtime's native array
type. The coupling matrix in `thrmlx` is the symmetric dense expansion of the exact edge-weight
vector given to THRML.

A bipartite RBM is selected because it gives both libraries a valid two-block Gibbs sweep—the
normal blocked-sampling shape rather than an artifact of automatic coloring. It does not claim to
measure sparse storage efficiency. Benchmark work is one request with 1,024 chains, 20 warmup
sweeps, 32 recorded samples, and one sweep per later sample. A recorded sample is one complete
256-spin chain state; throughput is therefore recorded states per second.

The benchmark uses float32 parameters and boolean spins. It does not compare random bitstreams or
require traces to be bitwise equal across runtimes; each runtime owns a different random-key
algorithm. A small, separately tested two-block model verifies that each adapter emits boolean
traces of the expected shape and has empirical one- and two-spin moments consistent with exact
enumeration.

## Timing protocol

Each adapter exposes the same runner contract:

```python
def run(key_seed: int) -> array: ...  # bool, (chains, samples, spins)
```

The benchmark reports two non-interchangeable timing modes:

1. **Cold time to first result:** construct the runner and execute one complete request, blocking
   until its output is materialized. This includes the framework's initial compilation and captures
   interactive first-use cost.
2. **Warm steady state:** construct once, perform an unmeasured warm-up request, then time seven
   independently keyed complete requests. Every result is materialized before the timer stops.
   Report the median elapsed time and median recorded-states-per-second, alongside the seven raw
   elapsed values.

The benchmark never mixes cold and warm values, times lazy scheduling without materializing the
result, or reports a best-of-many run as its headline. It records only sampling work: reporting,
JSON serialization, and runner construction after the cold measurement occur outside timed windows.

## Results and provenance

`benchmarks/run.py` writes a structured JSON report to standard output. A checked-in baseline at
`benchmarks/results/2026-08-16-m4-pro.json` records the exact local run used in the README. Its
schema includes:

- schema version and ISO-8601 timestamp;
- macOS/platform, processor, Python version, and backend/device metadata;
- MLX, JAX, JAXLIB, Equinox, and THRML versions plus the pinned THRML commit;
- full workload and schedule values;
- each adapter's cold elapsed time, warm raw elapsed values, median elapsed time, and throughput;
- an explicit `comparison_note` naming the unequal accelerator backends.

The README table is derived from that JSON and links to both the committed data and the reproduction
command. It carries no “faster than” headline if a runner fails or reports a backend other than the
documented one.

## Components

```text
benchmarks/
  __init__.py                 importable benchmark package
  contract.py                 immutable workload, result schema, and timing helpers
  thrmlx_runner.py            MLX-native workload adapter
  thrml_runner.py             upstream THRML/JAX workload adapter
  run.py                      CLI that executes both adapters and emits JSON
  results/2026-08-16-m4-pro.json  checked-in reproducible baseline
tests/
  test_benchmark.py           workload/result-schema/unit timing-contract tests
README.md                     visible result table and reproduction command
pyproject.toml, uv.lock       benchmark-only dependency group
PROJECT_LOG.md                durable decision and measured outcome
```

`contract.py` is the only shared representation of the model and metrics. Adapters only translate
that data into their native public API and block until results are ready. The runner intentionally
does not expose THRML nodes, programs, JAX arrays, or JAX installation details through `thrmlx`.

## Validation and release gates

- Tests first prove the workload's coupling expansion and adapter-independent metadata.
- Tests prove median computation, shape/dtype checks, required provenance fields, and that timing
  materialization is part of the runner callback.
- When benchmark dependencies are installed on macOS, a focused smoke test runs both adapters on a
  deliberately small workload and checks output shape/dtype plus the expected JAX CPU device.
- Existing exact-enumeration distribution tests remain required. The benchmark is not a correctness
  substitute.
- At the milestone checkpoint, run the full test suite serially and with `-n auto`, compare exact
  pass/fail/error counts, then run Ruff, ty, `uv build`, and wheel smoke checks.
- GitHub Actions remains a semantic/package gate only; no CI timing is placed in the README.

## Scope boundaries

This milestone does not change the public `thrmlx` sampler, add a THRML compatibility API, add
custom Metal kernels, add sparse coupling storage, publish to PyPI, or benchmark Extropic cloud
hardware. A JAX reference sampler, JAX Metal plugin comparison, and alternate graph topologies are
future follow-ups after this baseline reveals a specific opportunity.
