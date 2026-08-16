"""MLX translations of upstream THRML block-sampling objectives."""

import mlx.core as mx
import pytest

from thrmlx import (
    AbstractConditionalSampler,
    AbstractNode,
    ArraySpec,
    Block,
    BlockGibbsSpec,
    BlockSamplingProgram,
    InteractionGroup,
    SamplingSchedule,
    SpinNode,
    sample_blocks,
    sample_single_block,
    sample_states,
)


class ScalarNode(AbstractNode):
    """A floating-point node used in the generic sampler port."""


class NestedNode(AbstractNode):
    """A node with tuple/dictionary state used to test generic state updates."""


class EmptySampler(AbstractConditionalSampler):
    """A sampler that satisfies the unused first free block."""

    def sample(self, key, interactions, active_flags, states, sampler_state, output_spec):
        return mx.zeros((1,), dtype=mx.float32), sampler_state


class WeightedTailSampler(AbstractConditionalSampler):
    """A test sampler whose numeric result exposes interaction and tail gathers."""

    def sample(self, key, interactions, active_flags, states, sampler_state, output_spec):
        weighted_tail = interactions[0] * active_flags[0].astype(mx.float32) * states[0][0]
        return mx.sum(weighted_tail, axis=-1), sampler_state


class IncrementNestedSampler(AbstractConditionalSampler):
    """A sampler that proves nested tail state reaches the user update rule."""

    def sample(self, key, interactions, active_flags, states, sampler_state, output_spec):
        source = states[0][0]
        return {
            "categorical": source["categorical"][:, 0, :] + 1,
            "floating": source["floating"][:, 0, :] + 1,
        }, sampler_state


def _signed_neighbor_program() -> tuple[
    BlockSamplingProgram,
    list[mx.array],
    list[mx.array],
]:
    free_nodes = [ScalarNode(), ScalarNode(), ScalarNode()]
    clamp_nodes = [ScalarNode(), ScalarNode()]
    free_blocks = [Block([free_nodes[0]]), Block(free_nodes[1:])]
    clamp_blocks = [Block(clamp_nodes)]
    spec = BlockGibbsSpec(
        free_blocks,
        clamp_blocks,
        {ScalarNode: ArraySpec((), mx.float32)},
    )
    group = InteractionGroup(
        mx.array([0.25, -2.0], dtype=mx.float32),
        Block(free_nodes[1:]),
        [Block(clamp_nodes)],
    )
    return (
        BlockSamplingProgram(spec, [EmptySampler(), WeightedTailSampler()], [group]),
        [mx.array([2.0], dtype=mx.float32), mx.array([1.0, 4.0], dtype=mx.float32)],
        [mx.array([3.0, 0.5], dtype=mx.float32)],
    )


def test_interaction_group_rejects_tail_length_that_differs_from_its_heads() -> None:
    """Catch an interaction layout that cannot align head and tail entries."""

    with pytest.raises(ValueError, match="same length"):
        InteractionGroup(
            mx.ones((2,)),
            Block([SpinNode(), SpinNode()]),
            [Block([SpinNode()])],
        )


def test_upstream_testplusminus_sample_block() -> None:
    """Catch interaction or tail-state slices that do not align to each free node."""

    program, state_free, state_clamp = _signed_neighbor_program()

    updated, _ = sample_single_block(
        mx.random.key(7),
        state_free,
        state_clamp,
        program,
        1,
        None,
    )

    assert updated.tolist() == pytest.approx([0.75, -1.0])


def test_upstream_testplusminus_sample_blocks() -> None:
    """Catch sweeps that expose an earlier superblock update too late or too early."""

    program, state_free, state_clamp = _signed_neighbor_program()

    updated, sampler_states = sample_blocks(
        mx.random.key(8),
        state_free,
        state_clamp,
        program,
        [None, None],
    )

    assert sampler_states == [None, None]
    assert updated[0].tolist() == pytest.approx([0.0])
    assert updated[1].tolist() == pytest.approx([0.75, -1.0])


def test_upstream_testplusminus_sample_states() -> None:
    """Catch scheduled recording that omits post-warmup or later generic samples."""

    program, state_free, state_clamp = _signed_neighbor_program()

    trace = sample_states(
        mx.random.key(9),
        program,
        SamplingSchedule(warmup=1, samples=3, sweeps_per_sample=1),
        state_free,
        state_clamp,
        [program.gibbs_spec.free_blocks[1]],
    )

    assert len(trace) == 1
    assert trace[0].shape == (3, 2)
    assert trace[0].tolist() == [[0.75, -1.0]] * 3


def test_upstream_testplusminus_state_gaurdrailing() -> None:
    """Catch a sweep that accepts free state with a template-incompatible dtype."""

    program, state_free, state_clamp = _signed_neighbor_program()
    wrong_state = [mx.array([True]), state_free[1]]

    with pytest.raises(TypeError, match="dtype"):
        sample_blocks(mx.random.key(10), wrong_state, state_clamp, program, [None, None])


def test_upstream_testsamplervalidation_mismatched_sampler_list_raises() -> None:
    """Catch a program that fails to require one conditional sampler per free block."""

    program, _, _ = _signed_neighbor_program()

    with pytest.raises(ValueError, match="Expected 2 samplers"):
        BlockSamplingProgram(program.gibbs_spec, [EmptySampler()], [])


def test_upstream_testpytreestate_pytree_state() -> None:
    """Catch generic sampling that loses values in tuple/dictionary block state."""

    nodes = [NestedNode(), NestedNode()]
    block = Block(nodes)
    specs = {
        NestedNode: {
            "categorical": ArraySpec((2,), mx.int8),
            "floating": ArraySpec((3,), mx.float32),
        }
    }
    program = BlockSamplingProgram(
        BlockGibbsSpec([block], [], specs),
        [IncrementNestedSampler()],
        [InteractionGroup(mx.ones((2,)), block, [block])],
    )
    initial = [
        {
            "categorical": mx.array([[1, 2], [3, 4]], dtype=mx.int8),
            "floating": mx.array(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                dtype=mx.float32,
            ),
        }
    ]

    updated, _ = sample_single_block(mx.random.key(11), initial, [], program, 0, None)

    assert updated["categorical"].tolist() == [[2, 3], [4, 5]]
    assert updated["floating"].tolist() == [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]
