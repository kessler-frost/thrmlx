"""Generic THRML block-Gibbs programs adapted to explicit MLX execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import cast

# MLX 0.32 ships its native extension without typing metadata.
import mlx.core as mx  # ty: ignore[unresolved-import]

from thrmlx.block_management import (
    Block,
    BlockSpec,
    State,
    StateSpec,
    _take_state,
    block_state_to_global,
    verify_block_state,
)
from thrmlx.conditional_samplers import AbstractConditionalSampler
from thrmlx.interaction import Interaction, InteractionGroup
from thrmlx.observers import AbstractObserver, StateObserver
from thrmlx.pgm import DEFAULT_NODE_SHAPE_DTYPES, AbstractNode
from thrmlx.schedule import SamplingSchedule


@dataclass(frozen=True, slots=True)
class _CompiledInteraction:
    """Static interaction data and tail locations for one target free block."""

    interaction: Interaction
    active: mx.array
    tail_buckets: tuple[int, ...]
    tail_positions: tuple[mx.array, ...]


class BlockGibbsSpec(BlockSpec):
    """BlockSpec augmented with ordered free and clamped Gibbs partitions."""

    def __init__(
        self,
        free_superblocks: Sequence[Block[AbstractNode] | Sequence[Block[AbstractNode]]],
        clamped_blocks: Sequence[Block[AbstractNode]],
        node_shape_dtypes: Mapping[type[AbstractNode], StateSpec] | None = None,
    ) -> None:
        free_blocks: list[Block[AbstractNode]] = []
        superblocks: list[tuple[Block[AbstractNode], ...]] = []
        sampling_order: list[tuple[int, ...]] = []

        for superblock in free_superblocks:
            blocks = (superblock,) if isinstance(superblock, Block) else tuple(superblock)
            if not blocks:
                raise ValueError("a Gibbs superblock must contain at least one block")
            start_index = len(free_blocks)
            free_blocks.extend(blocks)
            superblocks.append(blocks)
            sampling_order.append(tuple(range(start_index, len(free_blocks))))

        templates = (
            cast(Mapping[type[AbstractNode], StateSpec], DEFAULT_NODE_SHAPE_DTYPES)
            if node_shape_dtypes is None
            else node_shape_dtypes
        )
        super().__init__(free_blocks + list(clamped_blocks), templates)
        self.free_blocks = tuple(free_blocks)
        self.clamped_blocks = tuple(clamped_blocks)
        self.superblocks = tuple(superblocks)
        self.sampling_order = tuple(sampling_order)


def _slice_interaction(interaction: Interaction, indices: mx.array) -> Interaction:
    if isinstance(interaction, mx.array):
        return mx.take(interaction, indices, axis=0)
    if isinstance(interaction, (bool, float, int)):
        return interaction
    if isinstance(interaction, tuple):
        return tuple(_slice_interaction(value, indices) for value in interaction)
    if isinstance(interaction, dict):
        return {key: _slice_interaction(value, indices) for key, value in interaction.items()}
    if is_dataclass(interaction) and not isinstance(interaction, type):
        values = {
            field.name: _slice_interaction(getattr(interaction, field.name), indices)
            for field in fields(interaction)
        }
        return type(interaction)(**values)
    raise TypeError(
        "interaction must be an MLX array, static scalar, tuple, dictionary, or dataclass"
    )


def _compile_block_interactions(
    gibbs_spec: BlockGibbsSpec,
    block: Block[AbstractNode],
    interaction_groups: Sequence[InteractionGroup],
) -> tuple[_CompiledInteraction, ...]:
    compiled: list[_CompiledInteraction] = []
    for group in interaction_groups:
        occurrences = [
            [index for index, head in enumerate(group.head_nodes) if head == node] for node in block
        ]
        interaction_count = max((len(indices) for indices in occurrences), default=0)
        if interaction_count == 0:
            continue

        interaction_indices = [
            indices + [0] * (interaction_count - len(indices)) for indices in occurrences
        ]
        active = [
            [True] * len(indices) + [False] * (interaction_count - len(indices))
            for indices in occurrences
        ]
        tail_buckets: list[int] = []
        tail_positions: list[mx.array] = []
        for tail_block in group.tail_nodes:
            first_location = gibbs_spec.node_global_location_map.get(tail_block[0])
            if first_location is None:
                raise ValueError("interaction tail node is absent from the Gibbs specification")
            tail_buckets.append(first_location[0])
            positions = [
                [gibbs_spec.node_global_location_map[tail_block[index]][1] for index in indices]
                + [0] * (interaction_count - len(indices))
                for indices in occurrences
            ]
            tail_positions.append(mx.array(positions))

        indices_array = mx.array(interaction_indices)
        compiled.append(
            _CompiledInteraction(
                interaction=_slice_interaction(group.interaction, indices_array),
                active=mx.array(active),
                tail_buckets=tuple(tail_buckets),
                tail_positions=tuple(tail_positions),
            )
        )
    return tuple(compiled)


class BlockSamplingProgram:
    """Lower static directed interactions for reusable generic Gibbs sweeps."""

    def __init__(
        self,
        gibbs_spec: BlockGibbsSpec,
        samplers: Sequence[AbstractConditionalSampler],
        interaction_groups: Sequence[InteractionGroup],
    ) -> None:
        if len(samplers) != len(gibbs_spec.free_blocks):
            raise ValueError(
                f"Expected {len(gibbs_spec.free_blocks)} samplers, received {len(samplers)}"
            )
        self.gibbs_spec = gibbs_spec
        self.samplers = tuple(samplers)
        self._compiled_interactions = tuple(
            _compile_block_interactions(gibbs_spec, block, interaction_groups)
            for block in gibbs_spec.free_blocks
        )
        self.per_block_interactions = tuple(
            tuple(compiled.interaction for compiled in block_compiled)
            for block_compiled in self._compiled_interactions
        )
        self.per_block_interaction_active = tuple(
            tuple(compiled.active for compiled in block_compiled)
            for block_compiled in self._compiled_interactions
        )


def _sample_single_block(
    key: mx.array,
    state_free: Sequence[State],
    state_clamp: Sequence[State],
    program: BlockSamplingProgram,
    block: int,
    sampler_state: object,
    global_state: Sequence[State],
) -> tuple[State, object]:
    compiled_interactions = program._compiled_interactions[block]
    interaction_states = [
        [
            _take_state(
                program.gibbs_spec.global_sd_order[bucket],
                global_state[bucket],
                positions,
            )
            for bucket, positions in zip(
                compiled.tail_buckets, compiled.tail_positions, strict=True
            )
        ]
        for compiled in compiled_interactions
    ]
    free_block = program.gibbs_spec.free_blocks[block]
    sampler = program.samplers[block]
    return sampler.sample(
        key,
        [compiled.interaction for compiled in compiled_interactions],
        [compiled.active for compiled in compiled_interactions],
        interaction_states,
        sampler_state,
        program.gibbs_spec.node_shape_dtypes[free_block.node_type],
    )


def sample_single_block(
    key: mx.array,
    state_free: Sequence[State],
    state_clamp: Sequence[State],
    program: BlockSamplingProgram,
    block: int,
    sampler_state: object,
) -> tuple[State, object]:
    """Sample one free block from the current free and clamped state."""

    global_state = block_state_to_global(list(state_free) + list(state_clamp), program.gibbs_spec)
    return _sample_single_block(
        key,
        state_free,
        state_clamp,
        program,
        block,
        sampler_state,
        global_state,
    )


def sample_blocks(
    key: mx.array,
    state_free: Sequence[State],
    state_clamp: Sequence[State],
    program: BlockSamplingProgram,
    sampler_states: Sequence[object],
) -> tuple[list[State], list[object]]:
    """Perform one full ordered Gibbs sweep using only the supplied MLX key."""

    if len(sampler_states) != len(program.gibbs_spec.free_blocks):
        raise ValueError("number of sampler states must equal number of free blocks")
    free_state = list(state_free)
    updated_sampler_states = list(sampler_states)
    verify_block_state(
        program.gibbs_spec.free_blocks,
        free_state,
        program.gibbs_spec.node_shape_dtypes,
        block_axis=-1,
    )
    verify_block_state(
        program.gibbs_spec.clamped_blocks,
        state_clamp,
        program.gibbs_spec.node_shape_dtypes,
        block_axis=-1,
    )

    keys = mx.random.split(key, len(program.gibbs_spec.free_blocks))
    for block_indices in program.gibbs_spec.sampling_order:
        global_state = block_state_to_global(free_state + list(state_clamp), program.gibbs_spec)
        updates = {
            block: _sample_single_block(
                keys[block],
                free_state,
                state_clamp,
                program,
                block,
                updated_sampler_states[block],
                global_state,
            )
            for block in block_indices
        }
        for block, (state, sampler_state) in updates.items():
            free_state[block] = state
            updated_sampler_states[block] = sampler_state
    return free_state, updated_sampler_states


def _stack_recorded_states(states: Sequence[State]) -> State:
    first = states[0]
    if isinstance(first, mx.array):
        return mx.stack(states, axis=0)
    if isinstance(first, tuple):
        tuple_states = cast(Sequence[tuple[State, ...]], states)
        return tuple(
            _stack_recorded_states([state[index] for state in tuple_states])
            for index in range(len(first))
        )
    dictionary_states = cast(Sequence[dict[str, State]], states)
    return {
        key: _stack_recorded_states([state[key] for state in dictionary_states]) for key in first
    }


def _stack_observations(observations: Sequence[object]) -> object:
    first = observations[0]
    if first is None:
        return None
    if isinstance(first, mx.array):
        return mx.stack(cast(Sequence[mx.array], observations), axis=0)
    if isinstance(first, tuple):
        tuple_observations = cast(Sequence[tuple[object, ...]], observations)
        return tuple(
            _stack_observations([observation[index] for observation in tuple_observations])
            for index in range(len(first))
        )
    if isinstance(first, list):
        list_observations = cast(Sequence[list[object]], observations)
        return [
            _stack_observations([observation[index] for observation in list_observations])
            for index in range(len(first))
        ]
    if isinstance(first, dict):
        dictionary_observations = cast(Sequence[dict[str, object]], observations)
        return {
            key: _stack_observations([observation[key] for observation in dictionary_observations])
            for key in first
        }
    raise TypeError("observer outputs must be MLX arrays, containers, or None")


def sample_with_observation(
    key: mx.array,
    program: BlockSamplingProgram,
    schedule: SamplingSchedule,
    state_free: Sequence[State],
    state_clamp: Sequence[State],
    observation_carry_init: object,
    observer: AbstractObserver,
) -> tuple[object, object]:
    """Run a Gibbs chain and record an observer after every scheduled sample."""

    free_state = list(state_free)
    sampler_states = [sampler.init() for sampler in program.samplers]
    n_sweeps = schedule.warmup + (schedule.samples - 1) * schedule.sweeps_per_sample
    keys = mx.random.split(key, max(1, n_sweeps))
    key_index = 0
    for _ in range(schedule.warmup):
        free_state, sampler_states = sample_blocks(
            keys[key_index],
            free_state,
            state_clamp,
            program,
            sampler_states,
        )
        key_index += 1

    carry, first_observation = observer(
        program,
        free_state,
        state_clamp,
        observation_carry_init,
        0,
    )
    observations = [first_observation]
    for iteration in range(1, schedule.samples):
        for _ in range(schedule.sweeps_per_sample):
            free_state, sampler_states = sample_blocks(
                keys[key_index],
                free_state,
                state_clamp,
                program,
                sampler_states,
            )
            key_index += 1
        carry, observation = observer(
            program,
            free_state,
            state_clamp,
            carry,
            iteration,
        )
        observations.append(observation)
    return carry, _stack_observations(observations)


def sample_states(
    key: mx.array,
    program: BlockSamplingProgram,
    schedule: SamplingSchedule,
    state_free: Sequence[State],
    state_clamp: Sequence[State],
    nodes_to_sample: Sequence[Block[AbstractNode]],
) -> list[State]:
    """Warm a generic Gibbs chain and record requested states on a sample axis."""
    _, observed = sample_with_observation(
        key,
        program,
        schedule,
        state_free,
        state_clamp,
        None,
        StateObserver(nodes_to_sample),
    )
    if not isinstance(observed, list):
        raise TypeError("state observer must return a list of states")
    return cast(list[State], observed)
