"""MLX translations of upstream THRML factor objectives."""

import mlx.core as mx
import pytest

from thrmlx import (
    AbstractConditionalSampler,
    AbstractFactor,
    AbstractNode,
    ArraySpec,
    Block,
    BlockGibbsSpec,
    FactorSamplingProgram,
    InteractionGroup,
    WeightedFactor,
)


class FactorNode(AbstractNode):
    """A factor-node type distinct from the generic sampling fixtures."""


class PointlessFactor(AbstractFactor):
    """A factor used to test structural validation without adding interactions."""

    def to_interaction_groups(self):
        return []


class SimpleWeightedFactor(WeightedFactor):
    """A weighted factor used to test leading batch-axis validation."""

    def to_interaction_groups(self):
        return []


class DirectedFactor(AbstractFactor):
    """A factor that produces one concrete interaction for program-lowering coverage."""

    def to_interaction_groups(self):
        return [
            InteractionGroup(
                mx.ones((1,)),
                self.node_groups[0],
                [self.node_groups[1]],
            )
        ]


class NoopSampler(AbstractConditionalSampler):
    """A sampler only needed to construct a factor-backed program."""

    def sample(self, key, interactions, active_flags, states, sampler_state, output_spec):
        return mx.zeros((1,)), sampler_state


def _parallel_groups(size: int = 3) -> list[Block[FactorNode]]:
    return [Block([FactorNode() for _ in range(size)]) for _ in range(3)]


def test_upstream_testfactorcreate_good() -> None:
    """Catch a valid parallel factor layout being rejected."""

    factor = PointlessFactor(_parallel_groups())

    assert len(factor.node_groups) == 3


def test_upstream_testfactorcreate_empty() -> None:
    """Catch a factor that silently accepts an empty node-group list."""

    with pytest.raises(ValueError, match="empty"):
        PointlessFactor([])


def test_upstream_testfactorcreate_ragged() -> None:
    """Catch a factor that accepts node groups with unequal parallel lengths."""

    with pytest.raises(ValueError, match="same number"):
        PointlessFactor([*_parallel_groups(), Block([FactorNode() for _ in range(4)])])


def test_upstream_testweighted_good() -> None:
    """Catch a correctly batched weight tensor being rejected."""

    factor = SimpleWeightedFactor(mx.zeros((3, 1, 3)), _parallel_groups())

    assert factor.weights.shape == (3, 1, 3)


def test_upstream_testweighted_bad() -> None:
    """Catch a weighted factor that ignores the leading parallel-node axis."""

    with pytest.raises(ValueError, match="weights"):
        SimpleWeightedFactor(mx.zeros((4, 1, 3)), _parallel_groups())


def test_factor_sampling_program_lowers_factor_interactions() -> None:
    """Catch a factor-backed program that omits its factors during static lowering."""

    free_block = Block([FactorNode()])
    clamp_block = Block([FactorNode()])

    program = FactorSamplingProgram(
        BlockGibbsSpec(
            [free_block],
            [clamp_block],
            {FactorNode: ArraySpec((), mx.float32)},
        ),
        [NoopSampler()],
        [DirectedFactor([free_block, clamp_block])],
    )

    assert len(program.per_block_interactions[0]) == 1
