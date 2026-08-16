import math
from itertools import product

import mlx.core as mx

from thrmlx import Ising


def exact_distribution(model: Ising) -> tuple[mx.array, mx.array]:
    states = mx.array(list(product((False, True), repeat=model.n_spins)))
    probabilities = mx.softmax(-model.energy(states))
    return states, probabilities


def empirical_probabilities(states: mx.array) -> mx.array:
    n_spins = states.shape[-1]
    weights = mx.array([1 << (n_spins - spin - 1) for spin in range(n_spins)])
    labels = mx.sum(states.astype(mx.int32) * weights, axis=-1)
    state_ids = mx.arange(1 << n_spins)
    counts = mx.sum(labels[:, None] == state_ids[None, :], axis=0)
    return counts.astype(mx.float32) / states.shape[0]


def assert_probabilities_within_sampling_error(
    observed: mx.array,
    expected: mx.array,
    draws: int,
    *,
    numerical_allowance: float = 0.003,
) -> None:
    for observed_probability, expected_probability in zip(
        observed.tolist(), expected.tolist(), strict=True
    ):
        standard_error = math.sqrt(expected_probability * (1.0 - expected_probability) / draws)
        tolerance = 6.0 * standard_error + numerical_allowance
        assert abs(observed_probability - expected_probability) <= tolerance
