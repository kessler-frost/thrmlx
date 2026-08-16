"""Observers that inspect THRML-style MLX block-Gibbs chains."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Protocol

# MLX 0.32 ships its native extension without typing metadata.
import mlx.core as mx  # ty: ignore[unresolved-import]

from thrmlx.block_management import (
    Block,
    BlockSpec,
    State,
    block_state_to_global,
    from_global_state,
)
from thrmlx.pgm import AbstractNode


class SamplingProgram(Protocol):
    """Structural program view required by observers."""

    @property
    def gibbs_spec(self) -> BlockSpec:
        """Return the immutable block-state specification being observed."""


class AbstractObserver(ABC):
    """Inspect a Gibbs-chain state and optionally retain a running carry."""

    @abstractmethod
    def __call__(
        self,
        program: SamplingProgram,
        state_free: Sequence[State],
        state_clamped: Sequence[State],
        carry: object,
        iteration: int,
    ) -> tuple[object, object]:
        """Observe one post-sweep state and return its updated carry and record."""

    def init(self) -> object:
        """Return the initial observer carry."""

        return None


class StateObserver(AbstractObserver):
    """Record raw state for an ordered set of requested node blocks."""

    def __init__(self, blocks_to_sample: Sequence[Block[AbstractNode]]) -> None:
        self.blocks_to_sample = tuple(blocks_to_sample)

    def __call__(
        self,
        program: SamplingProgram,
        state_free: Sequence[State],
        state_clamped: Sequence[State],
        carry: object,
        iteration: int,
    ) -> tuple[None, list[State]]:
        gibbs_spec = program.gibbs_spec
        global_state = block_state_to_global(list(state_free) + list(state_clamped), gibbs_spec)
        return None, from_global_state(global_state, gibbs_spec, self.blocks_to_sample)


def _identity(state: Sequence[State], blocks: Sequence[Block[AbstractNode]]) -> Sequence[State]:
    return state


class MomentAccumulatorObserver(AbstractObserver):
    """Accumulate requested products of transformed node values over observations."""

    def __init__(
        self,
        moment_spec: Sequence[Sequence[Sequence[AbstractNode]]],
        f_transform: Callable[
            [Sequence[State], Sequence[Block[AbstractNode]]], Sequence[State]
        ] = _identity,
    ) -> None:
        self.f_transform = f_transform
        flat_nodes: list[AbstractNode] = []
        node_indices: dict[AbstractNode, int] = {}
        nodes_by_type: dict[type[AbstractNode], list[AbstractNode]] = {}
        indices_by_type: dict[type[AbstractNode], list[int]] = {}
        moment_indices: list[mx.array] = []

        for moment_group in moment_spec:
            if not moment_group:
                raise ValueError("each moment group must contain at least one moment")
            width = len(moment_group[0])
            if any(len(moment) != width for moment in moment_group):
                raise ValueError("all moments in a group must have equal width")
            rows: list[list[int]] = []
            for moment in moment_group:
                row: list[int] = []
                for node in moment:
                    index = node_indices.get(node)
                    if index is None:
                        index = len(flat_nodes)
                        node_indices[node] = index
                        flat_nodes.append(node)
                    row.append(index)
                    node_type = type(node)
                    nodes_by_type.setdefault(node_type, []).append(node)
                    indices_by_type.setdefault(node_type, []).append(index)
                rows.append(row)
            moment_indices.append(mx.array(rows, dtype=mx.int32))

        self.flat_nodes = tuple(flat_nodes)
        self.moment_indices = tuple(moment_indices)
        self.blocks_to_sample = tuple(Block(nodes) for nodes in nodes_by_type.values())
        self.type_indices = tuple(
            mx.array(indices, dtype=mx.int32) for indices in indices_by_type.values()
        )

    def init(self) -> list[mx.array]:
        """Allocate one running sum per requested moment type."""

        return [mx.zeros((indices.shape[0],), dtype=mx.float32) for indices in self.moment_indices]

    def __call__(
        self,
        program: SamplingProgram,
        state_free: Sequence[State],
        state_clamped: Sequence[State],
        carry: object,
        iteration: int,
    ) -> tuple[list[mx.array], None]:
        if not isinstance(carry, list) or len(carry) != len(self.moment_indices):
            raise ValueError("moment observer carry does not match the requested moments")
        gibbs_spec = program.gibbs_spec
        global_state = block_state_to_global(list(state_free) + list(state_clamped), gibbs_spec)
        sampled_state = from_global_state(global_state, gibbs_spec, self.blocks_to_sample)
        transformed_state = list(self.f_transform(sampled_state, self.blocks_to_sample))
        if any(not isinstance(state, mx.array) for state in transformed_state):
            raise TypeError("moment transforms must return MLX arrays")
        arrays = [state for state in transformed_state if isinstance(state, mx.array)]
        dtype = mx.result_type(*(array.dtype for array in arrays))
        flat_state = mx.zeros((len(self.flat_nodes),), dtype=dtype)
        for state, indices in zip(arrays, self.type_indices, strict=True):
            flat_state[indices] = state

        updated_carry = []
        for previous, indices in zip(carry, self.moment_indices, strict=True):
            products = mx.prod(flat_state[indices], axis=-1)
            dtype = mx.result_type(previous.dtype, products.dtype, mx.float32)
            updated_carry.append(previous.astype(dtype) + products.astype(dtype))
        return updated_carry, None
