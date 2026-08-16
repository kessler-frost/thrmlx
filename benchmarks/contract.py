"""Deterministic workload and timing configuration shared by benchmark adapters."""

from collections.abc import Callable
from dataclasses import dataclass
from statistics import median

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Sampling request and timing-repetition sizes for the primary comparison."""

    chains: int = 1_024
    warmup: int = 20
    samples: int = 32
    sweeps_per_sample: int = 1
    warm_repetitions: int = 7


@dataclass(frozen=True, slots=True)
class BenchmarkWorkload:
    """Dense bipartite Ising parameters stored in a framework-neutral representation."""

    fields: npt.NDArray[np.float32]
    edge_weights: npt.NDArray[np.float32]
    n_visible: int
    n_latent: int
    beta: float = 1.0

    @property
    def n_spins(self) -> int:
        """Return the total spin count."""

        return self.n_visible + self.n_latent

    @property
    def blocks(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return the visible and latent update blocks in sweep order."""

        return (tuple(range(self.n_visible)), tuple(range(self.n_visible, self.n_spins)))


@dataclass(frozen=True, slots=True)
class Timing:
    """Cold and repeated warm durations for one materializing runner."""

    cold_elapsed_seconds: float
    warm_elapsed_seconds: tuple[float, ...]

    @property
    def warm_median_elapsed_seconds(self) -> float:
        """Return the median fully materialized warm duration."""

        return float(median(self.warm_elapsed_seconds))


def workload(n_visible: int = 128, n_latent: int = 128) -> BenchmarkWorkload:
    """Build the fixed-seed, moderate-coupling RBM workload."""

    generator = np.random.default_rng(20_260_816)
    fields = generator.normal(0.0, 0.05, n_visible + n_latent).astype(np.float32)
    edge_weights = generator.normal(0.0, 0.05, (n_visible, n_latent)).astype(np.float32)
    return BenchmarkWorkload(fields, edge_weights, n_visible, n_latent)


def expanded_couplings(model: BenchmarkWorkload) -> npt.NDArray[np.float32]:
    """Expand bipartite weights into a symmetric dense Ising coupling matrix."""

    couplings = np.zeros((model.n_spins, model.n_spins), dtype=np.float32)
    couplings[: model.n_visible, model.n_visible :] = model.edge_weights
    couplings[model.n_visible :, : model.n_visible] = model.edge_weights.T
    return couplings


def measure(
    make_runner: Callable[[], Callable[[int], object]],
    *,
    cold_seed: int,
    warmup_seed: int,
    warm_seeds: tuple[int, ...],
    clock: Callable[[], float],
) -> Timing:
    """Measure one cold request and a sequence of independently seeded warm requests."""

    cold_started = clock()
    make_runner()(cold_seed)
    cold_elapsed_seconds = clock() - cold_started

    warm_runner = make_runner()
    warm_runner(warmup_seed)
    warm_elapsed_seconds: list[float] = []
    for seed in warm_seeds:
        warm_started = clock()
        warm_runner(seed)
        warm_elapsed_seconds.append(clock() - warm_started)
    return Timing(cold_elapsed_seconds, tuple(warm_elapsed_seconds))
