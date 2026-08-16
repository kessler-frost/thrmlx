"""Validated dense pairwise Ising models."""

import math
from collections.abc import Sequence
from itertools import combinations
from sys import float_info

# MLX 0.32 ships its native extension without typing metadata.
import mlx.core as mx  # ty: ignore[unresolved-import]


def _validate_parameter_array(name: str, value: mx.array) -> None:
    if not isinstance(value, mx.array):
        raise TypeError(f"{name} must be an MLX array")
    if not mx.issubdtype(value.dtype, mx.floating):
        raise TypeError(f"{name} must have a floating dtype")
    if not mx.all(mx.isfinite(value)).item():
        raise ValueError(f"{name} must contain only finite values")


def _validate_shapes(fields: mx.array, couplings: mx.array) -> int:
    if fields.ndim != 1:
        raise ValueError("fields must have shape (N,)")
    if fields.shape[0] < 1:
        raise ValueError("an Ising model must contain at least one spin")
    n_spins = fields.shape[0]
    if couplings.shape != (n_spins, n_spins):
        raise ValueError("couplings must have shape (N, N) matching fields")
    return n_spins


def _validate_coupling_matrix(couplings: mx.array) -> None:
    if not mx.array_equal(couplings, couplings.T).item():
        raise ValueError("couplings must be symmetric")
    if not mx.all(mx.diagonal(couplings) == 0).item():
        raise ValueError("couplings must have an exactly zero diagonal")


def _auto_color(adjacency: list[list[bool]]) -> tuple[tuple[int, ...], ...]:
    assignments: list[int] = []
    colors: list[list[int]] = []
    for spin, neighbors in enumerate(adjacency):
        neighbor_colors = {assignments[neighbor] for neighbor in range(spin) if neighbors[neighbor]}
        color = next(candidate for candidate in range(spin + 1) if candidate not in neighbor_colors)
        if color == len(colors):
            colors.append([])
        colors[color].append(spin)
        assignments.append(color)
    return tuple(tuple(block) for block in colors)


def _validate_blocks(
    blocks: Sequence[Sequence[int]], n_spins: int, adjacency: list[list[bool]]
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
        raise TypeError("blocks must be a sequence of index sequences")
    normalized = tuple(tuple(block) for block in blocks)
    if not normalized:
        raise ValueError("blocks must be nonempty")
    if any(not block for block in normalized):
        raise ValueError("every block must be nonempty")

    flat = tuple(index for block in normalized for index in block)
    if any(type(index) is not int for index in flat):
        raise TypeError("block indices must be integers")
    if any(index < 0 or index >= n_spins for index in flat):
        raise ValueError("block indices must be in range for the model")
    if len(flat) != n_spins or set(flat) != set(range(n_spins)):
        raise ValueError("blocks must be a partition of every spin exactly once")
    if any(
        adjacency[left][right] for block in normalized for left, right in combinations(block, 2)
    ):
        raise ValueError("a nonzero coupling cannot connect spins in the same update block")
    return normalized


class Ising:
    """A dense pairwise Ising model with a valid ordered update coloring."""

    def __init__(
        self,
        fields: mx.array,
        couplings: mx.array,
        blocks: Sequence[Sequence[int]] | None = None,
        *,
        beta: float = 1.0,
    ) -> None:
        _validate_parameter_array("fields", fields)
        _validate_parameter_array("couplings", couplings)
        n_spins = _validate_shapes(fields, couplings)
        _validate_coupling_matrix(couplings)
        if isinstance(beta, bool) or not isinstance(beta, (int, float)):
            raise TypeError("beta must be a real number")
        if beta <= 0 or beta > float_info.max or not math.isfinite(beta):
            raise ValueError("beta must be finite and strictly positive")

        dtype = mx.result_type(fields.dtype, couplings.dtype)
        adjacency = (couplings != 0).tolist()
        normalized_blocks = (
            _auto_color(adjacency)
            if blocks is None
            else _validate_blocks(blocks, n_spins, adjacency)
        )

        self._fields = fields.astype(dtype)
        self._couplings = couplings.astype(dtype)
        self._blocks = normalized_blocks
        self._block_indices = tuple(mx.array(block) for block in normalized_blocks)
        self._beta = float(beta)

    @property
    def n_spins(self) -> int:
        """Number of binary spins in the model."""

        return self._fields.shape[0]

    @property
    def blocks(self) -> tuple[tuple[int, ...], ...]:
        """Ordered independent-set blocks used for a Gibbs sweep."""

        return self._blocks

    def energy(self, state: mx.array) -> mx.array:
        """Return reduced energy for boolean states shaped ``(..., N)``."""

        if not isinstance(state, mx.array):
            raise TypeError("state must be an MLX array")
        if state.dtype != mx.bool_:
            raise TypeError("state must have boolean dtype")
        if state.ndim == 0 or state.shape[-1] != self.n_spins:
            raise ValueError("state must have shape (..., N) matching the model")

        signed = 2 * state.astype(self._fields.dtype) - 1
        field_energy = signed @ self._fields
        coupling_energy = 0.5 * mx.sum((signed @ self._couplings) * signed, axis=-1)
        return -self._beta * (field_energy + coupling_energy)
