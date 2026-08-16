"""MLX translations of upstream THRML Ising-model objectives."""

from itertools import pairwise, product
from math import exp

import mlx.core as mx
import pytest

from thrmlx import Block, SamplingSchedule, sample_with_observation
from thrmlx.models.ising import (
    IsingEBM,
    IsingSamplingProgram,
    IsingTrainingSpec,
    estimate_kl_grad,
    estimate_moments,
    hinton_init,
)
from thrmlx.observers import StateObserver
from thrmlx.pgm import SpinNode


def _signed(value: bool) -> int:
    return 1 if value else -1


def _exact_line_distribution(
    model: IsingEBM, nodes: list[SpinNode]
) -> dict[tuple[bool, ...], float]:
    states = list(product((False, True), repeat=len(nodes) - 1))
    weights = []
    for free_values in states:
        state = mx.array([True, *free_values])
        weights.append(exp(-model.energy([state], [Block(nodes)]).item()))
    normalizer = sum(weights)
    return {state: weight / normalizer for state, weight in zip(states, weights, strict=True)}


def test_upstream_testline_sample() -> None:
    """Catch Ising wrapper samples that diverge from the clamped line Boltzmann law."""

    nodes = [SpinNode() for _ in range(4)]
    model = IsingEBM(
        nodes,
        list(pairwise(nodes)),
        mx.array([0.1, -0.2, 0.3, -0.1], dtype=mx.float32),
        mx.array([0.4, -0.3, 0.2], dtype=mx.float32),
        mx.array(1.0, dtype=mx.float32),
    )
    free_blocks = [Block(nodes[1::2]), Block(nodes[2::2])]
    program = IsingSamplingProgram(model, free_blocks, [Block([nodes[0]])])
    trace = StateObserver([Block(nodes)])
    _, samples = sample_with_observation(
        mx.random.key(43524),
        program,
        SamplingSchedule(warmup=30, samples=4_000, sweeps_per_sample=1),
        hinton_init(mx.random.key(7), model, free_blocks, ()),
        [mx.array([True])],
        trace.init(),
        trace,
    )
    full_trace = samples[0]

    for state, probability in _exact_line_distribution(model, nodes).items():
        empirical = mx.mean(
            mx.all(full_trace[:, 1:] == mx.array(state), axis=1).astype(mx.float32)
        ).item()
        assert empirical == pytest.approx(probability, abs=0.035)


def test_upstream_testmomentaccumulator_first_moments() -> None:
    """Catch first Ising moments that disagree with raw state observations."""

    nodes = [SpinNode() for _ in range(3)]
    edges = [(nodes[0], nodes[1]), (nodes[1], nodes[2])]
    model = IsingEBM(
        nodes,
        edges,
        mx.array([0.2, -0.1, 0.4], dtype=mx.float32),
        mx.array([0.3, -0.2], dtype=mx.float32),
        mx.array(1.0, dtype=mx.float32),
    )
    blocks = [Block([nodes[0], nodes[2]]), Block([nodes[1]])]
    program = IsingSamplingProgram(model, blocks, [])
    schedule = SamplingSchedule(warmup=10, samples=500, sweeps_per_sample=1)
    initial = hinton_init(mx.random.key(42), model, blocks, ())
    first_moments, _ = estimate_moments(
        mx.random.key(42), nodes, edges, program, schedule, initial, []
    )
    observer = StateObserver([Block(nodes)])
    _, samples = sample_with_observation(
        mx.random.key(42), program, schedule, initial, [], observer.init(), observer
    )
    signed_samples = 2 * samples[0].astype(mx.float32) - 1

    assert mx.allclose(first_moments, mx.mean(signed_samples, axis=0), atol=1e-6).item()


def test_upstream_testmomentaccumulator_second_moments() -> None:
    """Catch pair moments that use a different state ordering than the observer trace."""

    nodes = [SpinNode() for _ in range(3)]
    edges = [(nodes[0], nodes[1]), (nodes[1], nodes[2])]
    model = IsingEBM(
        nodes,
        edges,
        mx.array([0.2, -0.1, 0.4], dtype=mx.float32),
        mx.array([0.3, -0.2], dtype=mx.float32),
        mx.array(1.0, dtype=mx.float32),
    )
    blocks = [Block([nodes[0], nodes[2]]), Block([nodes[1]])]
    program = IsingSamplingProgram(model, blocks, [])
    schedule = SamplingSchedule(warmup=10, samples=500, sweeps_per_sample=1)
    initial = hinton_init(mx.random.key(42), model, blocks, ())
    _, second_moments = estimate_moments(
        mx.random.key(42), nodes, edges, program, schedule, initial, []
    )
    observer = StateObserver([Block(nodes)])
    _, samples = sample_with_observation(
        mx.random.key(42), program, schedule, initial, [], observer.init(), observer
    )
    signed_samples = 2 * samples[0].astype(mx.float32) - 1
    expected = mx.array(
        [
            mx.mean(signed_samples[:, 0] * signed_samples[:, 1]),
            mx.mean(signed_samples[:, 1] * signed_samples[:, 2]),
        ]
    )

    assert mx.allclose(second_moments, expected, atol=1e-6).item()


def test_upstream_testestimateklgrad_estimate_kl_grad() -> None:
    """Catch two-phase Monte Carlo gradients with wrong positive/negative signs."""

    nodes = [SpinNode(), SpinNode()]
    model = IsingEBM(
        nodes,
        [(nodes[0], nodes[1])],
        mx.array([0.2, -0.3], dtype=mx.float32),
        mx.array([0.4], dtype=mx.float32),
        mx.array(1.0, dtype=mx.float32),
    )
    positive_blocks = [Block([nodes[1]])]
    negative_blocks = [Block([nodes[0]]), Block([nodes[1]])]
    schedule = SamplingSchedule(warmup=10, samples=80, sweeps_per_sample=1)
    training = IsingTrainingSpec(
        model,
        [Block([nodes[0]])],
        [],
        positive_blocks,
        negative_blocks,
        schedule,
        schedule,
    )
    data = [mx.array([[True]])]
    positive_initial = hinton_init(mx.random.key(1), model, positive_blocks, (96, 1))
    negative_initial = hinton_init(mx.random.key(2), model, negative_blocks, (96,))
    gradient_weights, gradient_biases, _, _ = estimate_kl_grad(
        mx.random.key(44),
        training,
        nodes,
        [(nodes[0], nodes[1])],
        data,
        [],
        positive_initial,
        negative_initial,
    )

    probabilities = []
    for left, right in product((False, True), repeat=2):
        probabilities.append(exp(-model.energy([mx.array([left, right])], [Block(nodes)]).item()))
    normalizer = sum(probabilities)
    negative_biases = [
        sum(
            probability * _signed(state[index])
            for probability, state in zip(
                probabilities, product((False, True), repeat=2), strict=True
            )
        )
        / normalizer
        for index in range(2)
    ]
    positive_right = (exp(0.1) - exp(-0.1)) / (exp(0.1) + exp(-0.1))
    expected_biases = mx.array(
        [-(1 - negative_biases[0]), -(positive_right - negative_biases[1])], dtype=mx.float32
    )
    expected_weight = mx.array(
        [
            -(
                positive_right
                - (
                    sum(
                        probability * _signed(left) * _signed(right)
                        for probability, (left, right) in zip(
                            probabilities, product((False, True), repeat=2), strict=True
                        )
                    )
                    / normalizer
                )
            )
        ],
        dtype=mx.float32,
    )

    assert mx.allclose(gradient_biases, expected_biases, atol=0.08).item()
    assert mx.allclose(gradient_weights, expected_weight, atol=0.08).item()


def test_upstream_testestimateklgradfullyvisible_fully_visible_ising() -> None:
    """Catch fully visible training that samples instead of preserving positive data moments."""

    nodes = [SpinNode() for _ in range(4)]
    edges = [(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[3])]
    model = IsingEBM(
        nodes,
        edges,
        mx.array([0.1, -0.2, 0.3, -0.4], dtype=mx.float32),
        mx.array([0.2, -0.1, 0.3], dtype=mx.float32),
        mx.array(1.0, dtype=mx.float32),
    )
    negative_blocks = [Block([nodes[0], nodes[2]]), Block([nodes[1], nodes[3]])]
    training = IsingTrainingSpec(
        model,
        [Block(nodes)],
        [],
        [],
        negative_blocks,
        SamplingSchedule(warmup=0, samples=1, sweeps_per_sample=0),
        SamplingSchedule(warmup=10, samples=20, sweeps_per_sample=1),
    )
    data = [mx.array([[True, False, True, False], [False, True, False, True]], dtype=mx.bool_)]
    negative_initial = hinton_init(mx.random.key(3), model, negative_blocks, (32,))
    gradient_weights, gradient_biases, positive_moments, _ = estimate_kl_grad(
        mx.random.key(4),
        training,
        nodes,
        edges,
        data,
        [],
        [],
        negative_initial,
    )

    assert gradient_weights.shape == (len(edges),)
    assert gradient_biases.shape == (len(nodes),)
    assert mx.all(mx.isfinite(gradient_weights)).item()
    assert mx.all(mx.isfinite(gradient_biases)).item()
    assert positive_moments[0][0].tolist() == (2 * data[0].astype(mx.int8) - 1).tolist()
