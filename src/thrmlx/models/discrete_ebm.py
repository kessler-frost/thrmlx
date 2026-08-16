"""Discrete EBM factors and exact Gibbs conditionals for the MLX backend."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import prod

# MLX 0.32 ships its native extension without typing metadata.
import mlx.core as mx  # ty: ignore[unresolved-import]

from thrmlx.block_management import Block, BlockSpec, State, StateSpec, from_global_state
from thrmlx.conditional_samplers import BernoulliConditional, SoftmaxConditional
from thrmlx.factor import WeightedFactor
from thrmlx.interaction import InteractionGroup
from thrmlx.models.ebm import EBMFactor
from thrmlx.pgm import AbstractNode, ArraySpec


@dataclass(frozen=True, slots=True)
class DiscreteEBMInteraction:
    """The static weight table and count of spin tails for one directed update."""

    n_spin: int
    weights: mx.array


def _spin_product(spin_values: Sequence[mx.array]) -> mx.array:
    if not spin_values:
        return mx.array(1)
    stacked = mx.stack(spin_values, axis=-1).astype(mx.int8)
    return mx.prod(2 * stacked - 1, axis=-1)


def _batch_gather(values: mx.array, *indices: mx.array) -> mx.array:
    n_indices = len(indices)
    if n_indices == 0:
        return values
    batch_shape = values.shape[:-n_indices]
    flattened_values = values.reshape((-1, *values.shape[-n_indices:]))
    flattened_indices = tuple(index.reshape(-1) for index in indices)
    batch_indices = mx.arange(flattened_indices[0].shape[0])
    return flattened_values[(batch_indices, *flattened_indices)].reshape(batch_shape)


def _batch_gather_with_k(values: mx.array, *indices: mx.array) -> mx.array:
    n_indices = len(indices)
    if n_indices == 0:
        return values
    batch_shape = values.shape[:-n_indices]
    n_categories = batch_shape[-1]
    expanded_indices = tuple(
        mx.broadcast_to(index[..., None], (*index.shape, n_categories)).reshape(-1)
        for index in indices
    )
    flattened_values = values.reshape((prod(batch_shape), *values.shape[-n_indices:]))
    return _batch_gather(flattened_values, *expanded_indices).reshape(batch_shape)


def _split_states(states: Sequence[State], n_spin: int) -> tuple[list[mx.array], list[mx.array]]:
    spin_states = list(states[:n_spin])
    categorical_states = list(states[n_spin:])
    if any(
        not isinstance(state, mx.array) or state.ndim < 2 or state.dtype != mx.bool_
        for state in spin_states
    ):
        raise RuntimeError("Spin states must be scalar bool.")
    if any(
        not isinstance(state, mx.array)
        or state.ndim < 2
        or not mx.issubdtype(state.dtype, mx.unsignedinteger)
        for state in categorical_states
    ):
        raise RuntimeError("Categorical states must be scalar unsigned integer.")
    return spin_states, categorical_states


class DiscreteEBMFactor(EBMFactor, WeightedFactor):
    """Parallel factors ``s_1 ... s_M W[c_1, ..., c_N]`` over discrete nodes."""

    def __init__(
        self,
        spin_node_groups: Sequence[Block[AbstractNode]],
        categorical_node_groups: Sequence[Block[AbstractNode]],
        weights: mx.array,
    ) -> None:
        self.spin_node_groups = tuple(spin_node_groups)
        self.categorical_node_groups = tuple(categorical_node_groups)
        WeightedFactor.__init__(self, weights, self.spin_node_groups + self.categorical_node_groups)
        if weights.ndim != 1 + len(self.categorical_node_groups):
            raise RuntimeError(
                "The shape of the weight tensor must be [b, x_1, ..., x_k], where "
                "k is the length of categorical_node_groups."
            )
        spin_types = {group.node_type for group in self.spin_node_groups}
        categorical_types = {group.node_type for group in self.categorical_node_groups}
        if spin_types & categorical_types:
            raise RuntimeError("A node cannot be both categorical and spin.")
        self.is_spin = {node_type: True for node_type in spin_types}
        self.is_spin.update({node_type: False for node_type in categorical_types})

    def to_interaction_groups(self) -> list[InteractionGroup]:
        interaction_groups: list[InteractionGroup] = []
        n_spin = len(self.spin_node_groups)
        n_categorical = len(self.categorical_node_groups)
        n_total = n_spin + n_categorical

        if n_spin:
            spin_indices = list(range(n_spin))
            head_nodes: list[AbstractNode] = []
            tail_nodes = [[] for _ in range(n_total - 1)]
            for head_index in spin_indices:
                tail_spin_indices = spin_indices[:head_index] + spin_indices[head_index + 1 :]
                head_nodes.extend(self.spin_node_groups[head_index].nodes)
                for tail_index, group_index in enumerate(tail_spin_indices):
                    tail_nodes[tail_index].extend(self.spin_node_groups[group_index].nodes)
                for categorical_index, categorical_group in enumerate(self.categorical_node_groups):
                    tail_nodes[n_spin - 1 + categorical_index].extend(categorical_group.nodes)
            tiler = [1] * self.weights.ndim
            tiler[0] = n_spin
            interaction_groups.append(
                InteractionGroup(
                    DiscreteEBMInteraction(n_spin - 1, mx.tile(self.weights, tiler)),
                    Block(head_nodes),
                    [Block(nodes) for nodes in tail_nodes],
                )
            )

        for head_index in range(n_categorical):
            tail_indices = list(range(n_categorical))
            tail_indices.pop(head_index)
            axes = (0, head_index + 1, *(index + 1 for index in tail_indices))
            interaction_groups.append(
                InteractionGroup(
                    DiscreteEBMInteraction(n_spin, mx.transpose(self.weights, axes)),
                    self.categorical_node_groups[head_index],
                    [
                        *self.spin_node_groups,
                        *(self.categorical_node_groups[index] for index in tail_indices),
                    ],
                )
            )
        return interaction_groups

    def energy(self, global_state: Sequence[State], block_spec: BlockSpec) -> mx.array:
        spin_values = from_global_state(global_state, block_spec, self.spin_node_groups)
        categorical_values = from_global_state(
            global_state, block_spec, self.categorical_node_groups
        )
        if any(not isinstance(value, mx.array) for value in [*spin_values, *categorical_values]):
            raise TypeError("discrete EBM factors require array node states")
        spins = [value for value in spin_values if isinstance(value, mx.array)]
        categories = [value for value in categorical_values if isinstance(value, mx.array)]
        selected_weights = _batch_gather(self.weights, *categories)
        spin_product = _spin_product(spins).astype(self.weights.dtype)
        return -mx.sum(selected_weights * spin_product)


def _merge_groups(groups: Sequence[InteractionGroup], n_tail_groups: int) -> list[InteractionGroup]:
    if not groups:
        return []
    head_nodes: list[AbstractNode] = []
    tail_nodes = [[] for _ in range(n_tail_groups)]
    weights: list[mx.array] = []
    for group in groups:
        interaction = group.interaction
        if not isinstance(interaction, DiscreteEBMInteraction):
            raise TypeError("square discrete EBM factors only merge discrete interactions")
        head_nodes.extend(group.head_nodes.nodes)
        for index, block in enumerate(group.tail_nodes):
            tail_nodes[index].extend(block.nodes)
        weights.append(interaction.weights)
    first_interaction = groups[0].interaction
    if not isinstance(first_interaction, DiscreteEBMInteraction):
        raise TypeError("square discrete EBM factors only merge discrete interactions")
    return [
        InteractionGroup(
            DiscreteEBMInteraction(first_interaction.n_spin, mx.concatenate(weights, axis=0)),
            Block(head_nodes),
            [Block(nodes) for nodes in tail_nodes],
        )
    ]


class SquareDiscreteEBMFactor(DiscreteEBMFactor):
    """A discrete EBM factor with equal category size along every category axis."""

    def __init__(
        self,
        spin_node_groups: Sequence[Block[AbstractNode]],
        categorical_node_groups: Sequence[Block[AbstractNode]],
        weights: mx.array,
    ) -> None:
        super().__init__(spin_node_groups, categorical_node_groups, weights)
        if weights.ndim > 2 and len(set(weights.shape[1:])) != 1:
            raise RuntimeError("Interaction tensor is not square.")

    def to_interaction_groups(self) -> list[InteractionGroup]:
        groups = super().to_interaction_groups()
        spin_groups = [group for group in groups if self.is_spin[group.head_nodes.node_type]]
        categorical_groups = [
            group for group in groups if not self.is_spin[group.head_nodes.node_type]
        ]
        n_tails = len(self.node_groups) - 1
        return _merge_groups(spin_groups, n_tails) + _merge_groups(categorical_groups, n_tails)


class SpinEBMFactor(SquareDiscreteEBMFactor):
    """A square discrete factor containing only Boolean spins."""

    def __init__(self, node_groups: Sequence[Block[AbstractNode]], weights: mx.array) -> None:
        super().__init__(node_groups, [], weights)


class CategoricalEBMFactor(DiscreteEBMFactor):
    """A discrete factor containing only categorical variables."""

    def __init__(self, node_groups: Sequence[Block[AbstractNode]], weights: mx.array) -> None:
        super().__init__([], node_groups, weights)


class SquareCategoricalEBMFactor(SquareDiscreteEBMFactor):
    """A square discrete factor containing only categorical variables."""

    def __init__(self, node_groups: Sequence[Block[AbstractNode]], weights: mx.array) -> None:
        super().__init__([], node_groups, weights)


class SpinGibbsConditional(BernoulliConditional):
    """Exact conditional sampler for spin heads in a discrete EBM."""

    def compute_parameters(
        self,
        key: mx.array,
        interactions: Sequence[object],
        active_flags: Sequence[mx.array],
        states: Sequence[Sequence[State]],
        sampler_state: object,
        output_spec: StateSpec,
    ) -> tuple[mx.array, object]:
        gamma = mx.zeros(output_spec.shape if isinstance(output_spec, ArraySpec) else ())
        for interaction, active, state in zip(interactions, active_flags, states, strict=True):
            if not isinstance(interaction, DiscreteEBMInteraction):
                raise RuntimeError("Unsupported interaction found")
            spin_states, categorical_states = _split_states(state, interaction.n_spin)
            weights = _batch_gather(interaction.weights, *categorical_states)
            contribution = mx.sum(
                weights
                * active.astype(weights.dtype)
                * _spin_product(spin_states).astype(weights.dtype),
                axis=-1,
            )
            gamma = gamma.astype(weights.dtype) + contribution
        return gamma, sampler_state


class CategoricalGibbsConditional(SoftmaxConditional):
    """Exact conditional sampler for categorical heads in a discrete EBM."""

    def __init__(self, n_categories: int) -> None:
        if type(n_categories) is not int or n_categories < 1:
            raise ValueError("n_categories must be a positive integer")
        self.n_categories = n_categories

    def compute_parameters(
        self,
        key: mx.array,
        interactions: Sequence[object],
        active_flags: Sequence[mx.array],
        states: Sequence[Sequence[State]],
        sampler_state: object,
        output_spec: StateSpec,
    ) -> tuple[mx.array, object]:
        if not isinstance(output_spec, ArraySpec):
            raise TypeError("categorical sampling requires an ArraySpec output")
        theta = mx.zeros((*output_spec.shape, self.n_categories))
        for interaction, active, state in zip(interactions, active_flags, states, strict=True):
            if not isinstance(interaction, DiscreteEBMInteraction):
                raise RuntimeError("Unsupported interaction found")
            spin_states, categorical_states = _split_states(state, interaction.n_spin)
            weights = _batch_gather_with_k(interaction.weights, *categorical_states)
            spin_product = _spin_product(spin_states)[..., None].astype(weights.dtype)
            contribution = mx.sum(
                spin_product * weights * active[..., None].astype(weights.dtype),
                axis=-2,
            )
            theta = theta.astype(weights.dtype) + contribution
        return theta, sampler_state
