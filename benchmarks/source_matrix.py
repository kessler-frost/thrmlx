"""Reproducible MLX-versus-THRML measurements for each completed source use case."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp
import mlx.core as mx
from thrml import Block as JBlock
from thrml import BlockGibbsSpec as JBlockGibbsSpec
from thrml import CategoricalNode as JCategoricalNode
from thrml import SamplingSchedule as JSamplingSchedule
from thrml import SpinNode as JSpinNode
from thrml import sample_states as jsample_states
from thrml.factor import FactorSamplingProgram as JFactorSamplingProgram
from thrml.models.discrete_ebm import CategoricalEBMFactor as JCategoricalEBMFactor
from thrml.models.discrete_ebm import CategoricalGibbsConditional as JCategoricalGibbsConditional
from thrml.models.discrete_ebm import DiscreteEBMFactor as JDiscreteEBMFactor
from thrml.models.discrete_ebm import SpinEBMFactor as JSpinEBMFactor
from thrml.models.discrete_ebm import SpinGibbsConditional as JSpinGibbsConditional
from thrml.models.ising import IsingEBM as JIsingEBM
from thrml.models.ising import IsingSamplingProgram as JIsingSamplingProgram
from thrml.models.ising import IsingTrainingSpec as JIsingTrainingSpec
from thrml.models.ising import estimate_kl_grad as jestimate_kl_grad
from thrml.models.ising import estimate_moments as jestimate_moments
from thrml.models.ising import hinton_init as jhinton_init

from benchmarks.contract import BenchmarkConfig, Timing, measure, workload
from benchmarks.thrml_runner import make_runner as make_dense_thrml_runner
from benchmarks.thrmlx_runner import make_runner as make_dense_thrmlx_runner
from thrmlx import Block as MBlock
from thrmlx import BlockGibbsSpec as MBlockGibbsSpec
from thrmlx import CategoricalNode as MCategoricalNode
from thrmlx import FactorSamplingProgram as MFactorSamplingProgram
from thrmlx import SamplingSchedule as MSamplingSchedule
from thrmlx import SpinNode as MSpinNode
from thrmlx import sample_states as msample_states
from thrmlx.models.discrete_ebm import CategoricalEBMFactor as MCategoricalEBMFactor
from thrmlx.models.discrete_ebm import CategoricalGibbsConditional as MCategoricalGibbsConditional
from thrmlx.models.discrete_ebm import DiscreteEBMFactor as MDiscreteEBMFactor
from thrmlx.models.discrete_ebm import SpinEBMFactor as MSpinEBMFactor
from thrmlx.models.discrete_ebm import SpinGibbsConditional as MSpinGibbsConditional
from thrmlx.models.ising import IsingEBM as MIsingEBM
from thrmlx.models.ising import IsingSamplingProgram as MIsingSamplingProgram
from thrmlx.models.ising import IsingTrainingSpec as MIsingTrainingSpec
from thrmlx.models.ising import estimate_kl_grad as mestimate_kl_grad
from thrmlx.models.ising import estimate_moments as mestimate_moments
from thrmlx.models.ising import hinton_init as mhinton_init

THRML_COMMIT = "9c4e6fbb800f5e5c627122e668ff1b158ef3782b"
Runner = Callable[[int], object]
RunnerFactory = Callable[[bool], Runner]


@dataclass(frozen=True, slots=True)
class Case:
    """One reproducible source-use-case benchmark with matched backend runners."""

    identifier: str
    title: str
    objectives: tuple[str, ...]
    work_units: int
    unit: str
    make_thrmlx: RunnerFactory
    make_thrml: RunnerFactory


def _sampling_shape(smoke: bool) -> tuple[int, int, int]:
    return (4, 24, 2) if smoke else (20, 256, 5)


def _line_layout(size: int) -> tuple[list[int], list[int], list[tuple[int, int]]]:
    return (
        list(range(0, size, 2)),
        list(range(1, size, 2)),
        [(index, index + 1) for index in range(size - 1)],
    )


def _grid_layout(side_length: int) -> tuple[list[int], list[int], list[tuple[int, int]]]:
    color_zero: list[int] = []
    color_one: list[int] = []
    edges: list[tuple[int, int]] = []
    for row in range(side_length):
        for column in range(side_length):
            index = row * side_length + column
            (color_zero if (row + column) % 2 == 0 else color_one).append(index)
            if row + 1 < side_length:
                edges.append((index, (row + 1) * side_length + column))
            if column + 1 < side_length:
                edges.append((index, row * side_length + column + 1))
    return color_zero, color_one, edges


def _make_mlx_ising(
    layout: tuple[list[int], list[int], list[tuple[int, int]]],
    smoke: bool,
) -> Runner:
    warmup, samples, _ = _sampling_shape(smoke)
    first_color, second_color, edges = layout
    nodes = [MSpinNode() for _ in range(len(first_color) + len(second_color))]
    blocks = [
        MBlock([nodes[index] for index in first_color]),
        MBlock([nodes[index] for index in second_color]),
    ]
    model = MIsingEBM(
        nodes,
        [(nodes[left], nodes[right]) for left, right in edges],
        mx.linspace(-0.2, 0.2, len(nodes), dtype=mx.float32),
        mx.full((len(edges),), 0.12, dtype=mx.float32),
        mx.array(1.0, dtype=mx.float32),
    )
    program = MIsingSamplingProgram(model, blocks, [])
    schedule = MSamplingSchedule(warmup=warmup, samples=samples, sweeps_per_sample=1)
    full_block = MBlock(nodes)

    def run(seed: int) -> mx.array:
        initial = mhinton_init(mx.random.key(seed), model, blocks, ())
        trace = msample_states(
            mx.random.key(seed + 1), program, schedule, initial, [], [full_block]
        )[0]
        if not isinstance(trace, mx.array):
            raise TypeError("Ising benchmark must produce an MLX array trace")
        mx.eval(trace)
        return trace

    return run


def _make_thrml_ising(
    layout: tuple[list[int], list[int], list[tuple[int, int]]],
    smoke: bool,
) -> Runner:
    warmup, samples, _ = _sampling_shape(smoke)
    first_color, second_color, edges = layout
    nodes = [JSpinNode() for _ in range(len(first_color) + len(second_color))]
    blocks = [
        JBlock([nodes[index] for index in first_color]),
        JBlock([nodes[index] for index in second_color]),
    ]
    model = JIsingEBM(
        nodes,
        [(nodes[left], nodes[right]) for left, right in edges],
        jnp.linspace(-0.2, 0.2, len(nodes), dtype=jnp.float32),
        jnp.full((len(edges),), 0.12, dtype=jnp.float32),
        jnp.array(1.0, dtype=jnp.float32),
    )
    program = JIsingSamplingProgram(model, blocks, [])
    schedule = JSamplingSchedule(warmup, samples, 1)
    full_block = JBlock(nodes)

    def run(seed: int) -> jax.Array:
        initial = jhinton_init(jax.random.key(seed), model, blocks, ())
        return jax.block_until_ready(
            jsample_states(jax.random.key(seed + 1), program, schedule, initial, [], [full_block])[
                0
            ]
        )

    return run


def _make_mlx_spin_factor(smoke: bool) -> Runner:
    warmup, samples, _ = _sampling_shape(smoke)
    first_color, second_color, edges = _line_layout(64)
    nodes = [MSpinNode() for _ in range(64)]
    blocks = [
        MBlock([nodes[index] for index in first_color]),
        MBlock([nodes[index] for index in second_color]),
    ]
    factors = [
        MSpinEBMFactor([MBlock(nodes)], mx.linspace(-0.1, 0.1, 64, dtype=mx.float32)),
        MSpinEBMFactor(
            [
                MBlock([nodes[left] for left, _ in edges]),
                MBlock([nodes[right] for _, right in edges]),
            ],
            mx.full((len(edges),), 0.12, dtype=mx.float32),
        ),
    ]
    program = MFactorSamplingProgram(
        MBlockGibbsSpec(blocks, []), [MSpinGibbsConditional(), MSpinGibbsConditional()], factors
    )
    schedule = MSamplingSchedule(warmup=warmup, samples=samples, sweeps_per_sample=1)

    def run(seed: int) -> mx.array:
        keys = mx.random.split(mx.random.key(seed), 2)
        initial = [
            mx.random.bernoulli(shape=(len(block),), key=key)
            for block, key in zip(blocks, keys, strict=True)
        ]
        trace = msample_states(
            mx.random.key(seed + 1), program, schedule, initial, [], [MBlock(nodes)]
        )[0]
        if not isinstance(trace, mx.array):
            raise TypeError("spin-factor benchmark must produce an MLX array trace")
        mx.eval(trace)
        return trace

    return run


def _make_thrml_spin_factor(smoke: bool) -> Runner:
    warmup, samples, _ = _sampling_shape(smoke)
    first_color, second_color, edges = _line_layout(64)
    nodes = [JSpinNode() for _ in range(64)]
    blocks = [
        JBlock([nodes[index] for index in first_color]),
        JBlock([nodes[index] for index in second_color]),
    ]
    factors = [
        JSpinEBMFactor([JBlock(nodes)], jnp.linspace(-0.1, 0.1, 64, dtype=jnp.float32)),
        JSpinEBMFactor(
            [
                JBlock([nodes[left] for left, _ in edges]),
                JBlock([nodes[right] for _, right in edges]),
            ],
            jnp.full((len(edges),), 0.12, dtype=jnp.float32),
        ),
    ]
    program = JFactorSamplingProgram(
        JBlockGibbsSpec(blocks, []), [JSpinGibbsConditional(), JSpinGibbsConditional()], factors, []
    )
    schedule = JSamplingSchedule(warmup, samples, 1)

    def run(seed: int) -> jax.Array:
        keys = jax.random.split(jax.random.key(seed), 2)
        initial = [
            jax.random.bernoulli(key, shape=(len(block),))
            for block, key in zip(blocks, keys, strict=True)
        ]
        return jax.block_until_ready(
            jsample_states(
                jax.random.key(seed + 1), program, schedule, initial, [], [JBlock(nodes)]
            )[0]
        )

    return run


def _make_mlx_categorical_factor(smoke: bool) -> Runner:
    warmup, samples, _ = _sampling_shape(smoke)
    first_color, second_color, edges = _line_layout(32)
    nodes = [MCategoricalNode() for _ in range(32)]
    blocks = [
        MBlock([nodes[index] for index in first_color]),
        MBlock([nodes[index] for index in second_color]),
    ]
    factors = [
        MCategoricalEBMFactor(
            [
                MBlock([nodes[left] for left, _ in edges]),
                MBlock([nodes[right] for _, right in edges]),
            ],
            mx.full((len(edges), 3, 3), 0.05, dtype=mx.float32),
        )
    ]
    program = MFactorSamplingProgram(
        MBlockGibbsSpec(blocks, []),
        [MCategoricalGibbsConditional(3), MCategoricalGibbsConditional(3)],
        factors,
    )
    schedule = MSamplingSchedule(warmup=warmup, samples=samples, sweeps_per_sample=1)

    def run(seed: int) -> mx.array:
        initial = [mx.zeros((len(block),), dtype=mx.uint8) for block in blocks]
        trace = msample_states(
            mx.random.key(seed), program, schedule, initial, [], [MBlock(nodes)]
        )[0]
        if not isinstance(trace, mx.array):
            raise TypeError("categorical-factor benchmark must produce an MLX array trace")
        mx.eval(trace)
        return trace

    return run


def _make_thrml_categorical_factor(smoke: bool) -> Runner:
    warmup, samples, _ = _sampling_shape(smoke)
    first_color, second_color, edges = _line_layout(32)
    nodes = [JCategoricalNode() for _ in range(32)]
    blocks = [
        JBlock([nodes[index] for index in first_color]),
        JBlock([nodes[index] for index in second_color]),
    ]
    factors = [
        JCategoricalEBMFactor(
            [
                JBlock([nodes[left] for left, _ in edges]),
                JBlock([nodes[right] for _, right in edges]),
            ],
            jnp.full((len(edges), 3, 3), 0.05, dtype=jnp.float32),
        )
    ]
    program = JFactorSamplingProgram(
        JBlockGibbsSpec(blocks, []),
        [JCategoricalGibbsConditional(3), JCategoricalGibbsConditional(3)],
        factors,
        [],
    )
    schedule = JSamplingSchedule(warmup, samples, 1)

    def run(seed: int) -> jax.Array:
        initial = [jnp.zeros((len(block),), dtype=jnp.uint8) for block in blocks]
        return jax.block_until_ready(
            jsample_states(jax.random.key(seed), program, schedule, initial, [], [JBlock(nodes)])[0]
        )

    return run


def _make_mlx_mixed_factor(smoke: bool) -> Runner:
    warmup, samples, _ = _sampling_shape(smoke)
    spin_nodes = [MSpinNode() for _ in range(32)]
    categorical_nodes = [MCategoricalNode() for _ in range(32)]
    spin_block = MBlock(spin_nodes)
    categorical_block = MBlock(categorical_nodes)
    factor = MDiscreteEBMFactor(
        [MBlock([spin_nodes[index] for index in range(32) for _ in range(2)])],
        [
            MBlock(
                [categorical_nodes[max(index - 1, 0)] for index in range(32) for _ in range(1)]
                + categorical_nodes
            )
        ],
        mx.full((64, 3), 0.08, dtype=mx.float32),
    )
    program = MFactorSamplingProgram(
        MBlockGibbsSpec([spin_block, categorical_block], []),
        [MSpinGibbsConditional(), MCategoricalGibbsConditional(3)],
        [factor],
    )
    schedule = MSamplingSchedule(warmup=warmup, samples=samples, sweeps_per_sample=1)

    def run(seed: int) -> mx.array:
        trace = msample_states(
            mx.random.key(seed),
            program,
            schedule,
            [mx.zeros((32,), dtype=mx.bool_), mx.zeros((32,), dtype=mx.uint8)],
            [],
            [spin_block],
        )[0]
        if not isinstance(trace, mx.array):
            raise TypeError("mixed-factor benchmark must produce an MLX array trace")
        mx.eval(trace)
        return trace

    return run


def _make_thrml_mixed_factor(smoke: bool) -> Runner:
    warmup, samples, _ = _sampling_shape(smoke)
    spin_nodes = [JSpinNode() for _ in range(32)]
    categorical_nodes = [JCategoricalNode() for _ in range(32)]
    spin_block = JBlock(spin_nodes)
    categorical_block = JBlock(categorical_nodes)
    factor = JDiscreteEBMFactor(
        [JBlock([spin_nodes[index] for index in range(32) for _ in range(2)])],
        [
            JBlock(
                [categorical_nodes[max(index - 1, 0)] for index in range(32) for _ in range(1)]
                + categorical_nodes
            )
        ],
        jnp.full((64, 3), 0.08, dtype=jnp.float32),
    )
    program = JFactorSamplingProgram(
        JBlockGibbsSpec([spin_block, categorical_block], []),
        [JSpinGibbsConditional(), JCategoricalGibbsConditional(3)],
        [factor],
        [],
    )
    schedule = JSamplingSchedule(warmup, samples, 1)

    def run(seed: int) -> jax.Array:
        return jax.block_until_ready(
            jsample_states(
                jax.random.key(seed),
                program,
                schedule,
                [jnp.zeros((32,), dtype=jnp.bool_), jnp.zeros((32,), dtype=jnp.uint8)],
                [],
                [spin_block],
            )[0]
        )

    return run


def _make_mlx_moment_observer(smoke: bool) -> Runner:
    warmup, samples, _ = _sampling_shape(smoke)
    layout = _line_layout(32)
    first_color, second_color, edges = layout
    nodes = [MSpinNode() for _ in range(32)]
    blocks = [
        MBlock([nodes[index] for index in first_color]),
        MBlock([nodes[index] for index in second_color]),
    ]
    model = MIsingEBM(
        nodes,
        [(nodes[left], nodes[right]) for left, right in edges],
        mx.zeros((32,), dtype=mx.float32),
        mx.full((len(edges),), 0.1, dtype=mx.float32),
        mx.array(1.0, dtype=mx.float32),
    )
    program = MIsingSamplingProgram(model, blocks, [])
    schedule = MSamplingSchedule(warmup=warmup, samples=samples, sweeps_per_sample=1)

    def run(seed: int) -> mx.array:
        initial = mhinton_init(mx.random.key(seed), model, blocks, ())
        first, second = mestimate_moments(
            mx.random.key(seed + 1),
            nodes,
            [(nodes[left], nodes[right]) for left, right in edges],
            program,
            schedule,
            initial,
            [],
        )
        mx.eval(first, second)
        return first

    return run


def _make_thrml_moment_observer(smoke: bool) -> Runner:
    warmup, samples, _ = _sampling_shape(smoke)
    layout = _line_layout(32)
    first_color, second_color, edges = layout
    nodes = [JSpinNode() for _ in range(32)]
    blocks = [
        JBlock([nodes[index] for index in first_color]),
        JBlock([nodes[index] for index in second_color]),
    ]
    model = JIsingEBM(
        nodes,
        [(nodes[left], nodes[right]) for left, right in edges],
        jnp.zeros((32,), dtype=jnp.float32),
        jnp.full((len(edges),), 0.1, dtype=jnp.float32),
        jnp.array(1.0, dtype=jnp.float32),
    )
    program = JIsingSamplingProgram(model, blocks, [])
    schedule = JSamplingSchedule(warmup, samples, 1)

    def run(seed: int) -> jax.Array:
        first, second = jestimate_moments(
            jax.random.key(seed),
            nodes,
            [(nodes[left], nodes[right]) for left, right in edges],
            program,
            schedule,
            jhinton_init(jax.random.key(seed + 1), model, blocks, ()),
            [],
        )
        jax.block_until_ready(second)
        return jax.block_until_ready(first)

    return run


def _make_mlx_gradient(smoke: bool) -> Runner:
    warmup, samples, _ = _sampling_shape(smoke)
    chains = 8 if smoke else 32
    nodes = [MSpinNode(), MSpinNode()]
    edge = (nodes[0], nodes[1])
    model = MIsingEBM(
        nodes,
        [edge],
        mx.array([0.2, -0.3], dtype=mx.float32),
        mx.array([0.4], dtype=mx.float32),
        mx.array(1.0, dtype=mx.float32),
    )
    positive_blocks = [MBlock([nodes[1]])]
    negative_blocks = [MBlock([nodes[0]]), MBlock([nodes[1]])]
    training = MIsingTrainingSpec(
        model,
        [MBlock([nodes[0]])],
        [],
        positive_blocks,
        negative_blocks,
        MSamplingSchedule(warmup=warmup, samples=samples, sweeps_per_sample=1),
        MSamplingSchedule(warmup=warmup, samples=samples, sweeps_per_sample=1),
    )

    def run(seed: int) -> mx.array:
        positive = mhinton_init(mx.random.key(seed), model, positive_blocks, (chains, 1))
        negative = mhinton_init(mx.random.key(seed + 1), model, negative_blocks, (chains,))
        weights, biases, _, _ = mestimate_kl_grad(
            mx.random.key(seed + 2),
            training,
            nodes,
            [edge],
            [mx.array([[True]])],
            [],
            positive,
            negative,
        )
        mx.eval(weights, biases)
        return weights

    return run


def _make_thrml_gradient(smoke: bool) -> Runner:
    warmup, samples, _ = _sampling_shape(smoke)
    chains = 8 if smoke else 32
    nodes = [JSpinNode(), JSpinNode()]
    edge = (nodes[0], nodes[1])
    model = JIsingEBM(
        nodes,
        [edge],
        jnp.array([0.2, -0.3], dtype=jnp.float32),
        jnp.array([0.4], dtype=jnp.float32),
        jnp.array(1.0, dtype=jnp.float32),
    )
    positive_blocks = [JBlock([nodes[1]])]
    negative_blocks = [JBlock([nodes[0]]), JBlock([nodes[1]])]
    training = JIsingTrainingSpec(
        model,
        [JBlock([nodes[0]])],
        [],
        positive_blocks,
        negative_blocks,
        JSamplingSchedule(warmup, samples, 1),
        JSamplingSchedule(warmup, samples, 1),
    )

    def run(seed: int) -> jax.Array:
        weights, _, _, _ = jestimate_kl_grad(
            jax.random.key(seed + 2),
            training,
            nodes,
            [edge],
            [jnp.array([[True]])],
            [],
            jhinton_init(jax.random.key(seed), model, positive_blocks, (chains, 1)),
            jhinton_init(jax.random.key(seed + 1), model, negative_blocks, (chains,)),
        )
        return jax.block_until_ready(weights)

    return run


def _make_mlx_mnist_update(smoke: bool) -> Runner:
    image_count = 28 * 28
    chains = 8 if smoke else 32
    warmup, samples, _ = _sampling_shape(smoke)
    image_nodes = [MSpinNode() for _ in range(image_count)]
    label_nodes = [MSpinNode(), MSpinNode()]
    nodes = [*image_nodes, *label_nodes]
    edges = [(image_nodes[0], label_nodes[0]), (image_nodes[0], label_nodes[1])]
    model = MIsingEBM(
        nodes,
        edges,
        mx.zeros((len(nodes),), dtype=mx.float32),
        mx.array([-0.8, 0.8], dtype=mx.float32),
        mx.array(1.0, dtype=mx.float32),
    )
    image_block = MBlock(image_nodes)
    label_block = MBlock(label_nodes)
    training = MIsingTrainingSpec(
        model,
        [MBlock(nodes)],
        [],
        [],
        [image_block, label_block],
        MSamplingSchedule(warmup=0, samples=1, sweeps_per_sample=0),
        MSamplingSchedule(warmup=warmup, samples=samples, sweeps_per_sample=1),
    )
    images = mx.zeros((8, image_count), dtype=mx.bool_)
    images[4:, 0] = True
    labels = mx.zeros((8, 2), dtype=mx.bool_)
    labels[:4, 0] = True
    labels[4:, 1] = True
    data = mx.concatenate([images, labels], axis=-1)

    def run(seed: int) -> mx.array:
        initial = mhinton_init(mx.random.key(seed), model, [image_block, label_block], (chains,))
        weights, biases, _, _ = mestimate_kl_grad(
            mx.random.key(seed + 1), training, nodes, edges, [data], [], [], initial
        )
        mx.eval(weights, biases)
        return weights

    return run


def _make_thrml_mnist_update(smoke: bool) -> Runner:
    image_count = 28 * 28
    chains = 8 if smoke else 32
    warmup, samples, _ = _sampling_shape(smoke)
    image_nodes = [JSpinNode() for _ in range(image_count)]
    label_nodes = [JSpinNode(), JSpinNode()]
    nodes = [*image_nodes, *label_nodes]
    edges = [(image_nodes[0], label_nodes[0]), (image_nodes[0], label_nodes[1])]
    model = JIsingEBM(
        nodes,
        edges,
        jnp.zeros((len(nodes),), dtype=jnp.float32),
        jnp.array([-0.8, 0.8], dtype=jnp.float32),
        jnp.array(1.0, dtype=jnp.float32),
    )
    image_block = JBlock(image_nodes)
    label_block = JBlock(label_nodes)
    training = JIsingTrainingSpec(
        model,
        [JBlock(nodes)],
        [],
        [],
        [image_block, label_block],
        JSamplingSchedule(0, 1, 0),
        JSamplingSchedule(warmup, samples, 1),
    )
    images = jnp.zeros((8, image_count), dtype=jnp.bool_).at[4:, 0].set(True)
    labels = jnp.zeros((8, 2), dtype=jnp.bool_).at[:4, 0].set(True).at[4:, 1].set(True)
    data = jnp.concatenate([images, labels], axis=-1)

    def run(seed: int) -> jax.Array:
        weights, _, _, _ = jestimate_kl_grad(
            jax.random.key(seed + 1),
            training,
            nodes,
            edges,
            [data],
            [],
            [],
            jhinton_init(jax.random.key(seed), model, [image_block, label_block], (chains,)),
        )
        return jax.block_until_ready(weights)

    return run


def _make_dense_thrmlx(smoke: bool) -> Runner:
    config = BenchmarkConfig(
        chains=8 if smoke else 1_024,
        warmup=2 if smoke else 20,
        samples=3 if smoke else 32,
        sweeps_per_sample=1,
        warm_repetitions=2 if smoke else 5,
    )
    return make_dense_thrmlx_runner(workload(3, 2) if smoke else workload(), config)


def _make_dense_thrml(smoke: bool) -> Runner:
    config = BenchmarkConfig(
        chains=8 if smoke else 1_024,
        warmup=2 if smoke else 20,
        samples=3 if smoke else 32,
        sweeps_per_sample=1,
        warm_repetitions=2 if smoke else 5,
    )
    return make_dense_thrml_runner(workload(3, 2) if smoke else workload(), config)


def cases() -> tuple[Case, ...]:
    """Return the fixed source-use-case matrix in documentation display order."""

    return (
        Case(
            "dense_rbm",
            "Batched bipartite Ising / RBM",
            ("TestSampling::test_binary", "TestEstimateKLGrad::test_estimate_kl_grad"),
            32_768,
            "recorded states",
            _make_dense_thrmlx,
            _make_dense_thrml,
        ),
        Case(
            "ising_line",
            "Sparse line Ising sampling",
            ("TestLine::test_sample",),
            256,
            "recorded states",
            lambda smoke: _make_mlx_ising(_line_layout(64), smoke),
            lambda smoke: _make_thrml_ising(_line_layout(64), smoke),
        ),
        Case(
            "ising_grid",
            "Checkerboard grid Ising sampling",
            ("TestBigGrid::test_big", "TestHeteroGrid::test_grid"),
            256,
            "recorded states",
            lambda smoke: _make_mlx_ising(_grid_layout(8), smoke),
            lambda smoke: _make_thrml_ising(_grid_layout(8), smoke),
        ),
        Case(
            "spin_factor",
            "Low-level spin-factor Gibbs program",
            ("TestSampling::test_binary", "TestBlockSample::test_binary_bias"),
            256,
            "recorded states",
            _make_mlx_spin_factor,
            _make_thrml_spin_factor,
        ),
        Case(
            "categorical_factor",
            "Categorical factor Gibbs program",
            ("TestSampling::test_categorical", "TestBlockSample::test_categorical_triplet"),
            256,
            "recorded states",
            _make_mlx_categorical_factor,
            _make_thrml_categorical_factor,
        ),
        Case(
            "mixed_factor",
            "Mixed spin/categorical factor program",
            ("TestSampling::test_mixed", "TestBlockSample::test_ragged_mixed"),
            256,
            "recorded states",
            _make_mlx_mixed_factor,
            _make_thrml_mixed_factor,
        ),
        Case(
            "moment_observer",
            "Ising moment observer",
            (
                "TestMomentAccumulator::test_first_moments",
                "TestMomentAccumulator::test_second_moments",
            ),
            256,
            "observed states",
            _make_mlx_moment_observer,
            _make_thrml_moment_observer,
        ),
        Case(
            "contrastive_gradient",
            "Semi-visible contrastive gradient",
            ("TestEstimateKLGrad::test_estimate_kl_grad",),
            1,
            "gradient estimates",
            _make_mlx_gradient,
            _make_thrml_gradient,
        ),
        Case(
            "mnist_fixture_update",
            "MNIST-shaped contrastive update",
            (
                "TestTrainMnist::test_train_mnist",
                "TestEstimateKLGradFullyVisible::test_fully_visible_ising",
            ),
            1,
            "fixture updates",
            _make_mlx_mnist_update,
            _make_thrml_mnist_update,
        ),
    )


def case_ids() -> tuple[str, ...]:
    """Return the stable identifiers used by reports and tests."""

    return tuple(case.identifier for case in cases())


def _timing_report(timing: Timing, work_units: int) -> dict[str, float | list[float]]:
    return {
        "cold_elapsed_seconds": timing.cold_elapsed_seconds,
        "warm_elapsed_seconds": list(timing.warm_elapsed_seconds),
        "warm_median_elapsed_seconds": timing.warm_median_elapsed_seconds,
        "warm_units_per_second": work_units / timing.warm_median_elapsed_seconds,
    }


def _measure_case(
    factory: RunnerFactory, smoke: bool, work_units: int
) -> dict[str, float | list[float]]:
    repetitions = 2 if smoke else 5
    timing = measure(
        lambda: factory(smoke),
        cold_seed=0,
        warmup_seed=1,
        warm_seeds=tuple(range(2, 2 + repetitions)),
        clock=time.perf_counter,
    )
    return _timing_report(timing, work_units)


def report(smoke: bool) -> dict[str, object]:
    """Run every matched use case and return a JSON-serializable provenance report."""

    entries = []
    for case in cases():
        entries.append(
            {
                "id": case.identifier,
                "title": case.title,
                "objectives": list(case.objectives),
                "unit": case.unit,
                "work_units": case.work_units if not smoke else min(case.work_units, 24),
                "adapters": {
                    "thrmlx": {
                        "device": mx.default_device().type.name,
                        "timing": _measure_case(
                            case.make_thrmlx,
                            smoke,
                            case.work_units if not smoke else min(case.work_units, 24),
                        ),
                    },
                    "thrml": {
                        "device": jax.default_backend(),
                        "timing": _measure_case(
                            case.make_thrml,
                            smoke,
                            case.work_units if not smoke else min(case.work_units, 24),
                        ),
                    },
                },
            }
        )
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "comparison_note": (
            "MLX runs on Apple Metal GPU while upstream THRML runs through JAX CPU on this Mac. "
            "These are local-use measurements, not same-accelerator framework comparisons."
        ),
        "host": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version,
        },
        "software": {
            "jax": version("jax"),
            "jaxlib": version("jaxlib"),
            "mlx": version("mlx"),
            "thrml": version("thrml"),
            "thrml_commit": THRML_COMMIT,
        },
        "cases": entries,
    }


def main(arguments: Sequence[str] | None = None) -> None:
    """Run the matrix and print or save its JSON evidence."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    parsed = parser.parse_args(arguments)
    result = json.dumps(report(parsed.smoke), indent=2, sort_keys=True)
    if parsed.output is not None:
        parsed.output.parent.mkdir(parents=True, exist_ok=True)
        parsed.output.write_text(f"{result}\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
