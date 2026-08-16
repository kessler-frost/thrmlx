import math

import mlx.core as mx
import pytest

from tests.helpers import (
    assert_probabilities_within_sampling_error,
    empirical_probabilities,
    exact_distribution,
)
from thrmlx import Clamp, Ising, SamplingSchedule, sample


@pytest.mark.parametrize(("coupling", "seed"), [(0.8, 101), (-0.8, 102)])
def test_two_spin_histogram_and_correlation_match_exact_distribution(
    coupling: float, seed: int
) -> None:
    chains = 50_000
    model = Ising(
        mx.zeros((2,)),
        mx.array([[0.0, coupling], [coupling, 0.0]]),
    )
    trace = sample(
        mx.random.key(seed),
        model,
        SamplingSchedule(warmup=16),
        chains=chains,
    )
    final = trace[:, -1, :]
    _, expected = exact_distribution(model)

    observed = empirical_probabilities(final)
    signed = 2 * final.astype(mx.float32) - 1
    observed_correlation = mx.mean(signed[:, 0] * signed[:, 1]).item()

    assert_probabilities_within_sampling_error(observed, expected, chains)
    assert observed_correlation == pytest.approx(math.tanh(coupling), abs=0.015)


def test_dense_three_spin_histogram_matches_exact_enumeration() -> None:
    chains = 80_000
    model = Ising(
        mx.array([0.1, -0.25, 0.2]),
        mx.array(
            [
                [0.0, 0.3, -0.2],
                [0.3, 0.0, 0.4],
                [-0.2, 0.4, 0.0],
            ]
        ),
        beta=1.2,
    )
    trace = sample(
        mx.random.key(103),
        model,
        SamplingSchedule(warmup=24),
        chains=chains,
    )
    _, expected = exact_distribution(model)

    observed = empirical_probabilities(trace[:, -1, :])

    assert_probabilities_within_sampling_error(observed, expected, chains)


def test_minimal_and_nonminimal_colorings_reach_the_same_distribution() -> None:
    chains = 60_000
    fields = mx.array([0.1, -0.1, 0.2])
    couplings = mx.array(
        [
            [0.0, 0.5, 0.0],
            [0.5, 0.0, -0.35],
            [0.0, -0.35, 0.0],
        ]
    )
    minimal = Ising(fields, couplings)
    nonminimal = Ising(fields, couplings, blocks=((0,), (1,), (2,)))
    keys = mx.random.split(mx.random.key(104), 2)
    schedule = SamplingSchedule(warmup=20)

    minimal_trace = sample(keys[0], minimal, schedule, chains=chains)
    nonminimal_trace = sample(keys[1], nonminimal, schedule, chains=chains)
    _, expected = exact_distribution(minimal)

    minimal_observed = empirical_probabilities(minimal_trace[:, -1, :])
    nonminimal_observed = empirical_probabilities(nonminimal_trace[:, -1, :])

    assert_probabilities_within_sampling_error(minimal_observed, expected, chains)
    assert_probabilities_within_sampling_error(nonminimal_observed, expected, chains)


def test_clamped_free_spin_matches_its_exact_conditional() -> None:
    chains = 50_000
    coupling = 0.7
    model = Ising(
        mx.zeros((2,)),
        mx.array([[0.0, coupling], [coupling, 0.0]]),
    )
    clamp = Clamp(
        mask=mx.array([True, False]),
        values=mx.array([True, False]),
    )

    trace = sample(
        mx.random.key(105),
        model,
        SamplingSchedule(warmup=8),
        chains=chains,
        clamp=clamp,
    )

    observed_positive = mx.mean(trace[:, -1, 1].astype(mx.float32)).item()
    expected_positive = 1.0 / (1.0 + math.exp(-2.0 * coupling))

    assert observed_positive == pytest.approx(expected_positive, abs=0.012)
