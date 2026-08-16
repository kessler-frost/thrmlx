"""Factorized energy-based models translated from THRML to MLX."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

# MLX 0.32 ships its native extension without typing metadata.
import mlx.core as mx  # ty: ignore[unresolved-import]

from thrmlx.block_management import Block, BlockSpec, State, StateSpec, block_state_to_global
from thrmlx.factor import AbstractFactor
from thrmlx.pgm import DEFAULT_NODE_SHAPE_DTYPES, AbstractNode


class AbstractEBM(ABC):
    """A model that assigns an energy to a state over a block layout."""

    @abstractmethod
    def energy(self, state: Sequence[State], blocks: Sequence[Block[AbstractNode]]) -> mx.array:
        """Evaluate the scalar energy of ``state`` arranged by ``blocks``."""


class EBMFactor(AbstractFactor, ABC):
    """An undirected factor that contributes an energy term."""

    @abstractmethod
    def energy(self, global_state: Sequence[State], block_spec: BlockSpec) -> mx.array:
        """Evaluate this factor using packed global state."""


class AbstractFactorizedEBM(AbstractEBM, ABC):
    """An EBM whose total energy is the sum of factor energies."""

    def __init__(
        self,
        node_shape_dtypes: Mapping[type[AbstractNode], StateSpec] | None = None,
    ) -> None:
        self.node_shape_dtypes = dict(
            DEFAULT_NODE_SHAPE_DTYPES if node_shape_dtypes is None else node_shape_dtypes
        )

    def energy(self, state: Sequence[State], blocks: Sequence[Block[AbstractNode]]) -> mx.array:
        block_spec = BlockSpec(blocks, self.node_shape_dtypes)
        global_state = block_state_to_global(state, block_spec)
        total = mx.array(0.0)
        for factor in self.factors:
            total = total + factor.energy(global_state, block_spec)
        return total

    @property
    @abstractmethod
    def factors(self) -> tuple[EBMFactor, ...]:
        """Return the EBM's energy factors."""


class FactorizedEBM(AbstractFactorizedEBM):
    """A factorized EBM backed by an explicit, ordered factor collection."""

    def __init__(
        self,
        factors: Sequence[EBMFactor],
        node_shape_dtypes: Mapping[type[AbstractNode], StateSpec] | None = None,
    ) -> None:
        super().__init__(node_shape_dtypes)
        self._factors = tuple(factors)

    @property
    def factors(self) -> tuple[EBMFactor, ...]:
        return self._factors
