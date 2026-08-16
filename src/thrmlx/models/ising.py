"""THRML-style Ising EBMs, sampling programs, and contrastive estimators for MLX."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

# MLX 0.32 ships its native extension without typing metadata.
import mlx.core as mx  # ty: ignore[unresolved-import]

from thrmlx.block_management import Block, State
from thrmlx.block_sampling import BlockGibbsSpec, BlockSamplingProgram, sample_with_observation
from thrmlx.factor import FactorSamplingProgram
from thrmlx.models.discrete_ebm import SpinEBMFactor, SpinGibbsConditional
from thrmlx.models.ebm import AbstractFactorizedEBM, EBMFactor
from thrmlx.observers import MomentAccumulatorObserver
from thrmlx.pgm import AbstractNode, ArraySpec
from thrmlx.schedule import SamplingSchedule

Edge: TypeAlias = tuple[AbstractNode, AbstractNode]


class IsingEBM(AbstractFactorizedEBM):
    """A sparse-edge Ising EBM with ``-beta * (bias + pair)`` energy convention."""

    def __init__(
        self,
        nodes: Sequence[AbstractNode],
        edges: Sequence[Edge],
        biases: mx.array,
        weights: mx.array,
        beta: mx.array,
    ) -> None:
        if not nodes:
            raise ValueError("an Ising EBM requires at least one node")
        if len({type(node) for node in nodes}) != 1:
            raise ValueError("all Ising EBM nodes must have the same type")
        if not isinstance(biases, mx.array) or biases.shape != (len(nodes),):
            raise ValueError("biases must have shape (number of nodes,)")
        if not isinstance(weights, mx.array) or weights.shape != (len(edges),):
            raise ValueError("weights must have shape (number of edges,)")
        if (
            not isinstance(beta, mx.array)
            or beta.shape != ()
            or not mx.issubdtype(beta.dtype, mx.floating)
        ):
            raise TypeError("beta must be a scalar floating MLX array")
        self.nodes = tuple(nodes)
        self.edges = tuple(edges)
        self.biases = biases
        self.weights = weights
        self.beta = beta
        super().__init__({type(self.nodes[0]): ArraySpec((), mx.bool_)})

    @property
    def factors(self) -> tuple[EBMFactor, ...]:
        factors: list[EBMFactor] = [SpinEBMFactor([Block(self.nodes)], self.beta * self.biases)]
        if self.edges:
            factors.append(
                SpinEBMFactor(
                    [
                        Block([edge[0] for edge in self.edges]),
                        Block([edge[1] for edge in self.edges]),
                    ],
                    self.beta * self.weights,
                )
            )
        return tuple(factors)


class IsingSamplingProgram(FactorSamplingProgram):
    """A factor-backed Gibbs program specialized to a sparse-edge Ising EBM."""

    def __init__(
        self,
        ebm: IsingEBM,
        free_blocks: Sequence[Block[AbstractNode] | Sequence[Block[AbstractNode]]],
        clamped_blocks: Sequence[Block[AbstractNode]],
    ) -> None:
        spec = BlockGibbsSpec(free_blocks, clamped_blocks, ebm.node_shape_dtypes)
        sampler = SpinGibbsConditional()
        super().__init__(spec, [sampler for _ in spec.free_blocks], ebm.factors)


class IsingTrainingSpec:
    """Static positive/negative Gibbs programs and schedules for contrastive training."""

    def __init__(
        self,
        ebm: IsingEBM,
        data_blocks: Sequence[Block[AbstractNode]],
        conditioning_blocks: Sequence[Block[AbstractNode]],
        positive_sampling_blocks: Sequence[Block[AbstractNode] | Sequence[Block[AbstractNode]]],
        negative_sampling_blocks: Sequence[Block[AbstractNode] | Sequence[Block[AbstractNode]]],
        schedule_positive: SamplingSchedule,
        schedule_negative: SamplingSchedule,
    ) -> None:
        self.ebm = ebm
        self.program_positive = IsingSamplingProgram(
            ebm,
            positive_sampling_blocks,
            [*data_blocks, *conditioning_blocks],
        )
        self.program_negative = IsingSamplingProgram(
            ebm, negative_sampling_blocks, conditioning_blocks
        )
        self.schedule_positive = schedule_positive
        self.schedule_negative = schedule_negative


def hinton_init(
    key: mx.array,
    model: IsingEBM,
    blocks: Sequence[Block[AbstractNode]],
    batch_shape: tuple[int, ...],
) -> list[mx.array]:
    """Initialize free spins independently from their bias-only marginals."""

    if any(type(size) is not int or size < 0 for size in batch_shape):
        raise ValueError("batch_shape must contain non-negative integer dimensions")
    node_indices = {node: index for index, node in enumerate(model.nodes)}
    if not blocks:
        return []
    keys = mx.random.split(key, len(blocks))
    state: list[mx.array] = []
    for block, block_key in zip(blocks, keys, strict=True):
        indices = mx.array([node_indices[node] for node in block], dtype=mx.int32)
        probabilities = mx.sigmoid(model.beta * model.biases[indices])
        state.append(
            mx.random.bernoulli(probabilities, shape=(*batch_shape, len(block)), key=block_key)
        )
    return state


def _signed_transform(
    state: Sequence[State], blocks: Sequence[Block[AbstractNode]]
) -> list[mx.array]:
    if any(not isinstance(value, mx.array) for value in state):
        raise TypeError("Ising moment observation requires array states")
    return [2 * value.astype(mx.int8) - 1 for value in state if isinstance(value, mx.array)]


def estimate_moments(
    key: mx.array,
    first_moment_nodes: Sequence[AbstractNode],
    second_moment_edges: Sequence[Edge],
    program: BlockSamplingProgram,
    schedule: SamplingSchedule,
    init_state: Sequence[mx.array],
    clamped_data: Sequence[mx.array],
) -> tuple[mx.array, mx.array]:
    """Estimate spin and pair moments over one unbatched Gibbs trajectory."""

    moment_spec: list[Sequence[Sequence[AbstractNode]]] = []
    if first_moment_nodes:
        moment_spec.append([(node,) for node in first_moment_nodes])
    if second_moment_edges:
        moment_spec.append(list(second_moment_edges))
    if not moment_spec:
        empty = mx.zeros((0,), dtype=model_dtype(program))
        return empty, empty
    observer = MomentAccumulatorObserver(moment_spec, _signed_transform)
    carry, _ = sample_with_observation(
        key,
        program,
        schedule,
        init_state,
        clamped_data,
        observer.init(),
        observer,
    )
    if not isinstance(carry, list):
        raise TypeError("moment observer must return a list carry")
    first = (
        mx.zeros((0,), dtype=mx.float32) if not first_moment_nodes else carry[0] / schedule.samples
    )
    second_index = 1 if first_moment_nodes else 0
    second = (
        mx.zeros((0,), dtype=mx.float32)
        if not second_moment_edges
        else carry[second_index] / schedule.samples
    )
    return first, second


def model_dtype(program: BlockSamplingProgram) -> mx.Dtype:
    """Use the first state template dtype when an empty moment request needs a dtype."""

    first_spec = next(iter(program.gibbs_spec.node_shape_dtypes.values()))
    if not isinstance(first_spec, ArraySpec):
        raise TypeError("Ising programs require scalar ArraySpec state templates")
    return first_spec.dtype


def estimate_kl_grad(
    key: mx.array,
    training_spec: IsingTrainingSpec,
    bias_nodes: Sequence[AbstractNode],
    weight_edges: Sequence[Edge],
    data: Sequence[mx.array],
    conditioning_values: Sequence[mx.array],
    init_state_positive: Sequence[mx.array],
    init_state_negative: Sequence[mx.array],
) -> tuple[mx.array, mx.array, tuple[mx.array, mx.array], tuple[mx.array, mx.array]]:
    """Estimate contrastive KL gradients from positive and negative Gibbs phases."""

    if not data:
        raise ValueError("data must contain at least one clamped state array")
    positive_key, negative_key = mx.random.split(key, 2)
    n_data = data[0].shape[0]
    if any(value.shape[0] != n_data for value in data):
        raise ValueError("all data arrays must share a leading batch axis")
    if init_state_positive:
        n_chains_positive, positive_batch = init_state_positive[0].shape[:2]
        if positive_batch != n_data:
            raise ValueError("positive initial-state batch axis must match data")
        positive_clamped = [
            mx.broadcast_to(value, (n_chains_positive, *value.shape)) for value in data
        ] + [
            mx.broadcast_to(value, (n_chains_positive, n_data, *value.shape))
            for value in conditioning_values
        ]
        moments_bias_positive, moments_weight_positive = estimate_moments(
            positive_key,
            bias_nodes,
            weight_edges,
            training_spec.program_positive,
            training_spec.schedule_positive,
            init_state_positive,
            positive_clamped,
        )
    else:
        if sum(value.shape[-1] for value in data) != len(bias_nodes):
            raise ValueError("fully visible data must include every requested bias node")
        spins = mx.concatenate(data, axis=-1).astype(training_spec.ebm.beta.dtype)
        spins = 2 * spins - 1
        node_indices = {node: index for index, node in enumerate(bias_nodes)}
        left = mx.array([node_indices[edge[0]] for edge in weight_edges], dtype=mx.int32)
        right = mx.array([node_indices[edge[1]] for edge in weight_edges], dtype=mx.int32)
        moments_bias_positive = spins[None]
        moments_weight_positive = (spins[:, left] * spins[:, right])[None]

    if not init_state_negative:
        raise ValueError("negative-phase initial state must not be empty")
    n_chains_negative = init_state_negative[0].shape[0]
    negative_clamped = [
        mx.broadcast_to(value, (n_chains_negative, *value.shape)) for value in conditioning_values
    ]
    moments_bias_negative, moments_weight_negative = estimate_moments(
        negative_key,
        bias_nodes,
        weight_edges,
        training_spec.program_negative,
        training_spec.schedule_negative,
        init_state_negative,
        negative_clamped,
    )
    beta = training_spec.ebm.beta
    gradient_biases = -beta * (
        mx.mean(moments_bias_positive, axis=(0, 1)) - mx.mean(moments_bias_negative, axis=0)
    )
    gradient_weights = -beta * (
        mx.mean(moments_weight_positive, axis=(0, 1)) - mx.mean(moments_weight_negative, axis=0)
    )
    return (
        gradient_weights,
        gradient_biases,
        (moments_bias_positive, moments_weight_positive),
        (moments_bias_negative, moments_weight_negative),
    )
