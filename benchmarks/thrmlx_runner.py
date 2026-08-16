"""MLX adapter for the shared local sampling benchmark."""

from collections.abc import Callable

import mlx.core as mx

from benchmarks.contract import BenchmarkConfig, BenchmarkWorkload, expanded_couplings
from thrmlx import Ising, SamplingSchedule, sample


def make_runner(model: BenchmarkWorkload, config: BenchmarkConfig) -> Callable[[int], mx.array]:
    """Create a materializing MLX sampling request for one workload/configuration pair."""

    ising = Ising(
        mx.array(model.fields),
        mx.array(expanded_couplings(model)),
        blocks=model.blocks,
        beta=model.beta,
    )
    schedule = SamplingSchedule(
        warmup=config.warmup,
        samples=config.samples,
        sweeps_per_sample=config.sweeps_per_sample,
    )
    expected_shape = (config.chains, config.samples, model.n_spins)

    def run(seed: int) -> mx.array:
        trace = sample(mx.random.key(seed), ising, schedule, chains=config.chains)
        mx.eval(trace)
        assert trace.dtype == mx.bool_
        assert trace.shape == expected_shape
        return trace

    return run
