"""Run a THRML-style custom conditional sampler with MLX state and keys."""

import mlx.core as mx

from thrmlx import (
    AbstractConditionalSampler,
    AbstractNode,
    ArraySpec,
    Block,
    BlockGibbsSpec,
    BlockSamplingProgram,
    InteractionGroup,
    SamplingSchedule,
    sample_states,
)


class ScalarNode(AbstractNode):
    """A scalar continuous variable for this custom sampling program."""


class WeightedTailSampler(AbstractConditionalSampler):
    """Replace each head state with its weighted clamped-tail value."""

    def sample(self, key, interactions, active_flags, states, sampler_state, output_spec):
        return mx.sum(
            interactions[0] * active_flags[0].astype(mx.float32) * states[0][0],
            axis=-1,
        ), sampler_state


def main() -> None:
    """Record three samples from a two-head generic conditional program."""

    free_nodes = [ScalarNode(), ScalarNode()]
    clamp_nodes = [ScalarNode(), ScalarNode()]
    free_block = Block(free_nodes)
    clamp_block = Block(clamp_nodes)
    specification = BlockGibbsSpec(
        [free_block],
        [clamp_block],
        {ScalarNode: ArraySpec((), mx.float32)},
    )
    program = BlockSamplingProgram(
        specification,
        [WeightedTailSampler()],
        [
            InteractionGroup(
                mx.array([0.25, -2.0], dtype=mx.float32),
                free_block,
                [clamp_block],
            )
        ],
    )
    trace = sample_states(
        mx.random.key(12),
        program,
        SamplingSchedule(warmup=1, samples=3, sweeps_per_sample=1),
        [mx.array([0.0, 0.0], dtype=mx.float32)],
        [mx.array([3.0, 0.5], dtype=mx.float32)],
        [free_block],
    )

    print(trace[0].tolist())


if __name__ == "__main__":
    main()
