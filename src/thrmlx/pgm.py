"""THRML domain nodes and MLX state-template descriptions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import count

# MLX 0.32 ships its native extension without typing metadata.
import mlx.core as mx  # ty: ignore[unresolved-import]

_node_ids = count()


@dataclass(frozen=True, slots=True)
class ArraySpec:
    """Shape and dtype of one node-state array, excluding batch and node axes."""

    shape: tuple[int, ...]
    dtype: mx.Dtype

    def __init__(self, shape: Sequence[int] = (), dtype: mx.Dtype = mx.float32) -> None:
        normalized_shape = tuple(shape)
        if any(type(size) is not int or size < 0 for size in normalized_shape):
            raise ValueError("ArraySpec shape entries must be non-negative integers")
        object.__setattr__(self, "shape", normalized_shape)
        object.__setattr__(self, "dtype", dtype)


class AbstractNode:
    """Identity-stable base class for every variable in a sampling program."""

    __slots__ = ("_identifier",)
    _identifier: int

    def __new__(cls) -> AbstractNode:
        if cls is AbstractNode:
            raise TypeError("only subclasses of AbstractNode may be instantiated")
        node = super().__new__(cls)
        node._identifier = next(_node_ids)
        return node

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AbstractNode) and self._identifier == other._identifier

    def __hash__(self) -> int:
        return self._identifier

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, AbstractNode):
            raise TypeError("nodes can only be ordered against nodes")
        return self._identifier < other._identifier


class SpinNode(AbstractNode):
    """A binary variable represented as False for -1 and True for +1."""


class CategoricalNode(AbstractNode):
    """A categorical variable represented by an unsigned integer label."""


DEFAULT_NODE_SHAPE_DTYPES = {
    SpinNode: ArraySpec((), mx.bool_),
    CategoricalNode: ArraySpec((), mx.uint8),
}
