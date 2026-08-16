"""Factor abstractions adapted from THRML for MLX sampling programs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

# MLX 0.32 ships its native extension without typing metadata.
import mlx.core as mx  # ty: ignore[unresolved-import]

from thrmlx.block_management import Block
from thrmlx.block_sampling import BlockGibbsSpec, BlockSamplingProgram
from thrmlx.conditional_samplers import AbstractConditionalSampler
from thrmlx.interaction import InteractionGroup
from thrmlx.pgm import AbstractNode


class AbstractFactor(ABC):
    """A parallel batch of undirected factors over equal-length node groups."""

    def __init__(self, node_groups: Sequence[Block[AbstractNode]]) -> None:
        self.node_groups = tuple(node_groups)
        if not self.node_groups:
            raise ValueError("a factor cannot have empty node groups")
        if len({len(group) for group in self.node_groups}) != 1:
            raise ValueError("factor node groups must contain the same number of nodes")

    @abstractmethod
    def to_interaction_groups(self) -> list[InteractionGroup]:
        """Lower this undirected factor batch into directed sampling interactions."""


class WeightedFactor(AbstractFactor):
    """A factor with an MLX weight tensor batched over its parallel node groups."""

    def __init__(
        self,
        weights: mx.array,
        node_groups: Sequence[Block[AbstractNode]],
    ) -> None:
        super().__init__(node_groups)
        if not isinstance(weights, mx.array):
            raise TypeError("weights must be an MLX array")
        if weights.ndim == 0 or weights.shape[0] != len(self.node_groups[0]):
            raise ValueError("weights must have leading axis equal to factor node count")
        self.weights = weights


class FactorSamplingProgram(BlockSamplingProgram):
    """Build a generic Gibbs program from factors and explicit interactions."""

    def __init__(
        self,
        gibbs_spec: BlockGibbsSpec,
        samplers: Sequence[AbstractConditionalSampler],
        factors: Sequence[AbstractFactor],
        other_interaction_groups: Sequence[InteractionGroup] = (),
    ) -> None:
        interaction_groups = list(other_interaction_groups)
        for factor in factors:
            interaction_groups.extend(factor.to_interaction_groups())
        super().__init__(gibbs_spec, samplers, interaction_groups)
