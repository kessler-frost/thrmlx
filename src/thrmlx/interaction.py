"""Directed conditional-update dependencies adapted from THRML for MLX."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TypeAlias

# MLX 0.32 ships its native extension without typing metadata.
import mlx.core as mx  # ty: ignore[unresolved-import]

from thrmlx.block_management import Block
from thrmlx.pgm import AbstractNode

Interaction: TypeAlias = mx.array | tuple["Interaction", ...] | dict[str, "Interaction"]


def _walk_interaction(interaction: Interaction) -> Iterator[mx.array]:
    if isinstance(interaction, mx.array):
        yield interaction
        return
    if isinstance(interaction, tuple):
        for value in interaction:
            yield from _walk_interaction(value)
        return
    if isinstance(interaction, dict):
        if any(not isinstance(key, str) for key in interaction):
            raise TypeError("interaction dictionary keys must be strings")
        for value in interaction.values():
            yield from _walk_interaction(value)
        return
    raise TypeError("interactions contain only MLX arrays, tuples, and dictionaries")


class InteractionGroup:
    """Static directed interactions used when resampling a group of head nodes."""

    def __init__(
        self,
        interaction: Interaction,
        head_nodes: Block[AbstractNode],
        tail_nodes: Sequence[Block[AbstractNode]],
    ) -> None:
        if any(len(tail_block) != len(head_nodes) for tail_block in tail_nodes):
            raise ValueError("all tail blocks must have the same length as head nodes")
        if any(
            array.ndim == 0 or array.shape[0] != len(head_nodes)
            for array in _walk_interaction(interaction)
        ):
            raise ValueError("interaction arrays must have leading dimension equal to head nodes")

        self.interaction = interaction
        self.head_nodes = head_nodes
        self.tail_nodes = tuple(tail_nodes)
