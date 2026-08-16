# Apple-Silicon Performance Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`
> (recommended for independent tasks) or `superpowers:executing-plans` to implement this plan
> task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a reproducible, correctly labeled Mac performance baseline comparing `thrmlx` on
MLX Metal with upstream THRML on JAX CPU.

**Architecture:** `benchmarks.contract` owns one deterministic bipartite workload and a structured
result schema. `thrmlx_runner` and `thrml_runner` consume that immutable workload and return
materialized boolean traces through their respective public APIs. `run.py` applies one timing policy
to both runners, emits JSON, and is the only CLI. Documentation consumes a committed result JSON,
not a manually transcribed claim.

**Tech Stack:** Python 3.12 on macOS/Apple Silicon, MLX 0.32, JAX CPU, upstream THRML v0.1.4 pinned
to `9c4e6fbb800f5e5c627122e668ff1b158ef3782b`, NumPy, uv, pytest, Ruff, ty.

**Spec:** `docs/superpowers/specs/2026-08-16-performance-baseline-design.md`

## Global Constraints

- Do not add JAX, THRML, Equinox, or NumPy to runtime dependencies. NumPy belongs in the development
  group because the framework-neutral workload contract is tested without benchmark runtimes; JAX,
  THRML, and Equinox remain benchmark-only.
- Pin THRML by its public Git commit; `uv.lock` records every resolved wheel/version.
- Keep JAX results labeled `CPU` unless the actual report proves an Apple Metal device; never compare
  the two rows as identical-accelerator benchmarks.
- Use the exact 128-visible / 128-latent, two-block, float32 workload; 1,024 chains; 20 warmup
  sweeps; 32 samples; one sweep per later sample; seven warm timing repetitions.
- A complete trace must be materialized before a timing stops. Output shape is `(1024, 32, 256)` and
  dtype is boolean for both adapters.
- Keep imports at file tops, use `pathlib.Path` for paths, use `uv` for all Python commands, and do
  not add threads, multiprocessing, implicit RNG seeding, or a second project environment.
- Follow red-green-refactor for benchmark behavior. Use the scoped existing
  `python3 scripts/teardown.py` to remove environments/caches after optional benchmark setup.

---

### Task 1: Add the benchmark dependency boundary and deterministic workload contract

**Files:**

- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `benchmarks/__init__.py`
- Create: `benchmarks/contract.py`
- Modify: `tests/test_benchmark.py`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**

- Produces `BenchmarkWorkload`, `BenchmarkConfig`, `workload()`, and `expanded_couplings()` from
  `benchmarks.contract`.
- `BenchmarkWorkload` has host `numpy.ndarray` fields: `fields` (`float32`, `(256,)`),
  `edge_weights` (`float32`, `(128, 128)`), `blocks` (`((0..127), (128..255))`), and scalar `beta`.
- `BenchmarkConfig` defines constants `chains=1024`, `warmup=20`, `samples=32`,
  `sweeps_per_sample=1`, and `warm_repetitions=7`.
- `expanded_couplings(workload)` returns a symmetric `(256, 256)` `float32` matrix with zero
  diagonal and the exact bipartite weights in both off-diagonal quadrants.

- [x] **Step 1: Write the failing contract tests**

```python
from benchmarks.contract import BenchmarkConfig, expanded_couplings, workload


def test_primary_workload_is_a_two_block_bipartite_ising_model() -> None:
    model = workload()
    couplings = expanded_couplings(model)

    assert model.fields.shape == (256,)
    assert model.fields.dtype.name == "float32"
    assert model.edge_weights.shape == (128, 128)
    assert model.blocks == (tuple(range(128)), tuple(range(128, 256)))
    assert couplings.shape == (256, 256)
    assert (couplings[:128, :128] == 0).all()
    assert (couplings[128:, 128:] == 0).all()
    assert (couplings == couplings.T).all()


def test_primary_config_describes_the_published_work_unit() -> None:
    assert BenchmarkConfig() == BenchmarkConfig(
        chains=1024, warmup=20, samples=32, sweeps_per_sample=1, warm_repetitions=7
    )
```

- [x] **Step 2: Run the contract tests and verify red**

Run: `uv run pytest tests/test_benchmark.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks.contract'`.

- [x] **Step 3: Add benchmark-only dependencies and implementation**

Add NumPy to the `dev` group and a `benchmark` dependency group with `jax>=0.4` and
`thrml @ git+https://github.com/extropic-ai/thrml@9c4e6fbb800f5e5c627122e668ff1b158ef3782b`.
Run `uv lock`, then create the contract module using a locally seeded NumPy `Generator`, bounded
float32 fields/weights, and direct block tuples. Build the coupling expansion by filling only the
two off-diagonal quadrants; do not construct Python edge objects here.

```python
@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    chains: int = 1024
    warmup: int = 20
    samples: int = 32
    sweeps_per_sample: int = 1
    warm_repetitions: int = 7


def expanded_couplings(model: BenchmarkWorkload) -> np.ndarray:
    couplings = np.zeros((model.n_spins, model.n_spins), dtype=np.float32)
    couplings[:model.n_visible, model.n_visible:] = model.edge_weights
    couplings[model.n_visible:, :model.n_visible] = model.edge_weights.T
    return couplings
```

Change normal developer/CI setup from `--all-groups` to `--group dev`; reserve
`uv sync --group benchmark` for benchmarking. Keep Linux CPU MLX's `--extra cpu` in its existing
commands. Verify the locked THRML source revision with `uv tree --group benchmark` and direct
content search of `uv.lock`.

- [x] **Step 4: Run the focused contract tests and checks**

Run: `uv run --group benchmark pytest tests/test_benchmark.py -q`

Expected: PASS. Then run:

```bash
uv run --group benchmark ruff format --check benchmarks tests/test_benchmark.py
uv run --group benchmark ruff check benchmarks tests/test_benchmark.py
```

- [x] **Step 5: Commit the boundary and contract**

```bash
git add pyproject.toml uv.lock benchmarks/__init__.py benchmarks/contract.py \
  tests/test_benchmark.py .github/workflows/ci.yml
git commit -m "chore: add benchmark dependency group"
```

### Task 2: Add independently materialized MLX and THRML runners

**Files:**

- Create: `benchmarks/thrmlx_runner.py`
- Create: `benchmarks/thrml_runner.py`
- Modify: `tests/test_benchmark.py`

**Interfaces:**

- Consumes `BenchmarkWorkload`, `BenchmarkConfig`, and `expanded_couplings`.
- Produces `make_runner(model, config) -> Callable[[int], object]` from each runner module. Each
  callable constructs its native key from a seed, runs one full request, materializes the trace,
  validates `bool` and `(chains, samples, n_spins)`, and returns that already materialized native
  trace so tests can inspect the real backend result without moving host transfer out of the timer.
- THRML runner uses only public imports: `Block`, `SamplingSchedule`, `SpinNode`, `sample_states`,
  `IsingEBM`, `IsingSamplingProgram`, and `hinton_init`.

- [x] **Step 1: Write failing runner smoke tests**

```python
import jax.numpy as jnp
import mlx.core as mx

from benchmarks.contract import BenchmarkConfig, BenchmarkWorkload, workload
from benchmarks.thrmlx_runner import make_runner as make_mlx_runner
from benchmarks.thrml_runner import make_runner as make_thrml_runner


def _smoke_config() -> BenchmarkConfig:
    return BenchmarkConfig(chains=8, warmup=2, samples=3, sweeps_per_sample=1, warm_repetitions=2)


def _small_workload() -> BenchmarkWorkload:
    return workload(n_visible=3, n_latent=2)


def test_thrmlx_runner_materializes_one_complete_boolean_trace() -> None:
    trace = make_mlx_runner(_small_workload(), _smoke_config())(0)
    assert trace.shape == (8, 3, 5)
    assert trace.dtype == mx.bool_


def test_thrml_runner_materializes_one_complete_boolean_trace() -> None:
    trace = make_thrml_runner(_small_workload(), _smoke_config())(0)
    assert trace.shape == (8, 3, 5)
    assert trace.dtype == jnp.bool_
```

- [x] **Step 2: Run the runner tests and verify red**

Run: `uv run --group benchmark pytest tests/test_benchmark.py -q`

Expected: FAIL with module import errors for `benchmarks.thrmlx_runner` and
`benchmarks.thrml_runner`.

- [x] **Step 3: Implement the MLX runner**

Build one `thrmlx.Ising` from the expanded coupling matrix and declared blocks, and one
`thrmlx.SamplingSchedule` from the config. The returned callable must invoke the public `sample`
function with `chains=config.chains`, force `mx.eval(trace)`, and validate the expected shape/dtype
inside the callable.

```python
def make_runner(model: BenchmarkWorkload, config: BenchmarkConfig) -> Callable[[int], None]:
    ising = Ising(mx.array(model.fields), mx.array(expanded_couplings(model)), blocks=model.blocks)
    schedule = SamplingSchedule(
        warmup=config.warmup, samples=config.samples, sweeps_per_sample=config.sweeps_per_sample
    )

    def run(seed: int) -> mx.array:
        trace = sample(mx.random.key(seed), ising, schedule, chains=config.chains)
        mx.eval(trace)
        assert trace.dtype == mx.bool_
        assert trace.shape == (config.chains, config.samples, model.n_spins)
        return trace

    return run
```

- [x] **Step 4: Implement the upstream THRML runner**

Create a stable list of `SpinNode` objects; create every visible-to-latent edge in row-major order;
flatten `edge_weights` in exactly that order; make two `Block`s; and capture a constructed
`IsingEBM`/ `IsingSamplingProgram` in the runner factory. The callable splits one typed JAX key for
init/sampling, calls `hinton_init` with batch shape `(config.chains,)`, executes a `jax.jit`-compiled
`jax.vmap` of `sample_states`, then calls `jax.block_until_ready` on the full-state readout. Assert
the materialized readout is `bool` and the same `(chains, samples, spins)` shape.

Use the imported THRML schedule names (`n_warmup`, `n_samples`, `steps_per_sample`) rather than
adapting the library's user-facing naming into a second API. Do not add a JAX implementation to the
`thrmlx` package.

- [x] **Step 5: Run focused tests and record the actual JAX backend**

Run:

```bash
uv run --group benchmark pytest tests/test_benchmark.py -q
uv run --group benchmark python -c 'import jax; print(jax.default_backend()); print(jax.devices())'
```

Expected: both runner tests PASS; baseline macOS JAX output identifies `cpu`.

- [x] **Step 6: Commit the adapters**

```bash
git add benchmarks/thrmlx_runner.py benchmarks/thrml_runner.py tests/test_benchmark.py
git commit -m "feat: add MLX and THRML benchmark runners"
```

### Task 3: Implement structured timing and a command-line JSON report

**Files:**

- Modify: `benchmarks/contract.py`
- Create: `benchmarks/run.py`
- Modify: `tests/test_benchmark.py`

**Interfaces:**

- Produces `Timing`, `AdapterResult`, `BenchmarkResult`, and `measure(make_runner, seeds, clock)`.
- `measure` creates a fresh runner for cold timing, then a fresh runner for a single unmeasured warm
  request followed by seven fully materialized measured invocations.
- `python benchmarks/run.py` emits one JSON object to stdout. It accepts `--output PATH` to also
  write byte-identical formatted JSON via `Path.write_text`.

- [ ] **Step 1: Write failing timing and JSON-schema tests**

```python
def test_measure_keeps_cold_and_warm_timings_separate() -> None:
    calls: list[int] = []
    ticks = iter([10.0, 15.0, 20.0, 21.0, 30.0, 32.0, 40.0, 43.0])

    def factory() -> Callable[[int], None]:
        return calls.append

    result = measure(factory, seeds=(1, 2, 3), clock=lambda: next(ticks))

    assert calls == [1, 2, 3]
    assert result.cold_elapsed_seconds == 5.0
    assert result.warm_elapsed_seconds == (2.0, 3.0)
    assert result.warm_median_elapsed_seconds == 2.5


def test_json_report_contains_provenance_and_two_named_adapters(capsys: pytest.CaptureFixture[str]) -> None:
    run_main(["--smoke"])
    report = json.loads(capsys.readouterr().out)

    assert report["schema_version"] == 1
    assert set(report["adapters"]) == {"thrmlx", "thrml"}
    assert report["adapters"]["thrml"]["device"] == "cpu"
    assert report["comparison_note"]
```

- [ ] **Step 2: Run the tests and verify red**

Run: `uv run --group benchmark pytest tests/test_benchmark.py -q`

Expected: FAIL with missing `measure`/ `run_main` imports.

- [ ] **Step 3: Implement timing and provenance serialization**

Use `statistics.median`, `time.perf_counter`, `platform.platform`, `platform.processor`,
`sys.version`, `importlib.metadata.version`, and JAX/MLX device APIs. Use seed `0` for cold, seed
`1` for unmeasured warmup, and seeds `2..8` for the seven timed repetitions. Convert all result
values to built-in JSON scalar/list/dict values before `json.dumps(..., indent=2, sort_keys=True)`.
The normal CLI runs the primary config; `--smoke` replaces only workload/config cardinalities for
the test suite. `--output` is optional and uses `Path`.

- [ ] **Step 4: Run tests and one explicit smoke report**

Run:

```bash
uv run --group benchmark pytest tests/test_benchmark.py -q
uv run --group benchmark python benchmarks/run.py --smoke
```

Expected: PASS; output has named cold and warm values for both adapters and states JAX device
`cpu` on the baseline Mac.

- [ ] **Step 5: Commit the timing CLI**

```bash
git add benchmarks/contract.py benchmarks/run.py tests/test_benchmark.py
git commit -m "feat: add reproducible benchmark report"
```

### Task 4: Produce, document, verify, and publish the actual baseline

**Files:**

- Create: `benchmarks/results/2026-08-16-m4-pro.json`
- Modify: `README.md`
- Modify: `PROJECT_LOG.md`
- Modify: `CONTRIBUTING.md`

**Interfaces:**

- Consumes the normal `benchmarks/run.py` JSON report.
- Produces a README result table derived exactly from `benchmarks/results/2026-08-16-m4-pro.json`.

- [ ] **Step 1: Run the normal benchmark and capture a result artifact**

Run:

```bash
mkdir -p benchmarks/results
uv run --group benchmark python benchmarks/run.py \
  --output benchmarks/results/2026-08-16-m4-pro.json
```

Verify the file content, report schema, M4 Pro hardware metadata, `metal` MLX device, `cpu` JAX
device, pinned THRML commit, full primary dimensions, seven warm repetitions, and positive throughput
values. If either adapter fails, do not write a partial comparison table; repair the runner through a
new failing regression test first.

- [ ] **Step 2: Add the README result table and reproducible setup**

Place a `## Measured Apple-Silicon baseline` section immediately after the project introduction.
State the date, Mac mini M4 Pro / 48 GB, versions, workload, schedule, and accelerator caveat.
Use a table with `Adapter`, `Device`, `Cold time to first result`, `Warm median time`, and `Warm
recorded states/s`. Link to the exact JSON data. Document:

```bash
uv sync --frozen --group benchmark
uv run --group benchmark python benchmarks/run.py \
  --output benchmarks/results/local.json
python3 scripts/teardown.py
```

Do not call the result a promise, a universal speedup, or a same-device comparison. Remove the old
single-engine `dense_sampling.py` command or rewrite it to invoke the new report; do not retain two
conflicting benchmark protocols.

- [ ] **Step 3: Record the decision and test full quality gates**

Add the benchmark contract, actual result, and no-PyPI policy to `PROJECT_LOG.md`. Run serial and
parallel full suites, ensuring their pass/fail/error counts are identical, then run:

```bash
uv run --group benchmark pytest -q
uv run --group benchmark pytest -q -n auto
uv run --group benchmark ruff format --check .
uv run --group benchmark ruff check .
uv run --group benchmark ty check src
uv build
```

Create a clean temporary uv-managed environment and install the built wheel; run the two-spin
example. Inspect `git diff --check`, `git status --short`, and a content search proving the README
numbers occur in the checked-in JSON.

- [ ] **Step 4: Commit and push**

```bash
git add README.md PROJECT_LOG.md CONTRIBUTING.md benchmarks tests pyproject.toml uv.lock \
  .github/workflows/ci.yml
git commit -m "docs: publish Apple-Silicon performance baseline"
git push origin main
```

Verify `git status --short --branch` is clean and `gh api repos/kessler-frost/thrmlx/commits/main
--jq .sha` matches `git rev-parse HEAD`.
