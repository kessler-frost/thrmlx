"""Conditional-sampler protocol for MLX THRML programs."""

from __future__ import annotations

from typing import TYPE_CHECKING

# MLX 0.32 ships its native extension without typing metadata.
import mlx.core as mx  # ty: ignore[unresolved-import]

from thrmlx.block_management import State, StateSpec
from thrmlx.interaction import Interaction

if TYPE_CHECKING:
    from collections.abc import Sequence


class AbstractConditionalSampler:
    """Base class for one free-block update rule."""

    def init(self) -> object:
        """Return the initial carry passed to the first update of this sampler."""

        return None

    def sample(
        self,
        key: mx.array,
        interactions: Sequence[Interaction],
        active_flags: Sequence[mx.array],
        states: Sequence[Sequence[State]],
        sampler_state: object,
        output_spec: StateSpec,
    ) -> tuple[State, object]:
        """Produce a replacement state for one block."""

        raise NotImplementedError
