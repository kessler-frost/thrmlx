"""Build a THRML-style factor program and sample it through MLX."""

import mlx.core as mx

from thrmlx import (
    AbstractConditionalSampler,
    AbstractNode,
    ArraySpec,
    Block,
    BlockGibbsSpec,
    FactorSamplingProgram,
    InteractionGroup,
    SamplingSchedule,
    WeightedFactor,
    sample_states,
)


class ScalarNode(AbstractNode):
    """A scalar continuous variable used by this weighted pair factor."""


class WeightedPairFactor(WeightedFactor):
    """Direct each pair weight from the clamped tail into the free head."""

    def to_interaction_groups(self):
        return [InteractionGroup(self.weights, self.node_groups[0], [self.node_groups[1]])]


class WeightedTailSampler(AbstractConditionalSampler):
    """Replace each free state with its factor-weighted tail state."""

    def sample(self, key, interactions, active_flags, states, sampler_state, output_spec):
        return mx.sum(
            interactions[0] * active_flags[0].astype(mx.float32) * states[0][0],
            axis=-1,
        ), sampler_state


def main() -> None:
    """Create a one-pair weighted factor and record its conditional trace."""

    free_block = Block([ScalarNode()])
    clamp_block = Block([ScalarNode()])
    specification = BlockGibbsSpec(
        [free_block],
        [clamp_block],
        {ScalarNode: ArraySpec((), mx.float32)},
    )
    program = FactorSamplingProgram(
        specification,
        [WeightedTailSampler()],
        [WeightedPairFactor(mx.array([2.0]), [free_block, clamp_block])],
    )
    trace = sample_states(
        mx.random.key(13),
        program,
        SamplingSchedule(warmup=1, samples=3, sweeps_per_sample=1),
        [mx.array([0.0], dtype=mx.float32)],
        [mx.array([1.5], dtype=mx.float32)],
        [free_block],
    )

    print(trace[0].tolist())


if __name__ == "__main__":
    main()
