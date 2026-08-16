"""Conditional samplers for MLX THRML block-Gibbs programs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

# MLX 0.32 ships its native extension without typing metadata.
import mlx.core as mx  # ty: ignore[unresolved-import]

from thrmlx.block_management import State, StateSpec
from thrmlx.interaction import Interaction
from thrmlx.pgm import ArraySpec

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


class AbstractParametricConditionalSampler(AbstractConditionalSampler, ABC):
    """Sample a conditional distribution after deriving its parameters."""

    @abstractmethod
    def compute_parameters(
        self,
        key: mx.array,
        interactions: Sequence[Interaction],
        active_flags: Sequence[mx.array],
        states: Sequence[Sequence[State]],
        sampler_state: object,
        output_spec: StateSpec,
    ) -> tuple[mx.array, object]:
        """Return distribution parameters and the next sampler state."""

    @abstractmethod
    def sample_given_parameters(
        self,
        key: mx.array,
        parameters: mx.array,
        sampler_state: object,
        output_spec: StateSpec,
    ) -> tuple[State, object]:
        """Draw a state from precomputed distribution parameters."""

    def sample(
        self,
        key: mx.array,
        interactions: Sequence[Interaction],
        active_flags: Sequence[mx.array],
        states: Sequence[Sequence[State]],
        sampler_state: object,
        output_spec: StateSpec,
    ) -> tuple[State, object]:
        sample_key, parameter_key = mx.random.split(key, 2)
        parameters, next_state = self.compute_parameters(
            parameter_key,
            interactions,
            active_flags,
            states,
            sampler_state,
            output_spec,
        )
        return self.sample_given_parameters(sample_key, parameters, next_state, output_spec)


class BernoulliConditional(AbstractParametricConditionalSampler, ABC):
    """Sample Boolean spins with log odds twice the supplied field."""

    def sample_given_parameters(
        self,
        key: mx.array,
        parameters: mx.array,
        sampler_state: object,
        output_spec: StateSpec,
    ) -> tuple[State, object]:
        if (
            not isinstance(output_spec, ArraySpec)
            or output_spec.shape
            or output_spec.dtype != mx.bool_
        ):
            raise RuntimeError("Spin states must be bool.")
        return mx.random.bernoulli(mx.sigmoid(2 * parameters), key=key), sampler_state


class SoftmaxConditional(AbstractParametricConditionalSampler, ABC):
    """Sample unsigned categorical states from unnormalized log probabilities."""

    def sample_given_parameters(
        self,
        key: mx.array,
        parameters: mx.array,
        sampler_state: object,
        output_spec: StateSpec,
    ) -> tuple[State, object]:
        if (
            not isinstance(output_spec, ArraySpec)
            or output_spec.shape
            or not mx.issubdtype(output_spec.dtype, mx.unsignedinteger)
        ):
            raise RuntimeError("Categorical states must be unsigned integer.")
        n_categories = parameters.shape[-1]
        max_categories = mx.iinfo(output_spec.dtype).max + 1
        if n_categories > max_categories:
            raise RuntimeError(
                f"n_categories={n_categories} exceeds what dtype {output_spec.dtype} can represent;"
                "pass a wider integer dtype via node_shape_dtypes."
            )
        return mx.random.categorical(parameters, axis=-1, key=key).astype(
            output_spec.dtype
        ), sampler_state
