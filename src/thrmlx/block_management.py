"""THRML-compatible static block state management backed by MLX arrays."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Generic, TypeAlias, TypeVar, cast

# MLX 0.32 ships its native extension without typing metadata.
import mlx.core as mx  # ty: ignore[unresolved-import]

from thrmlx.pgm import AbstractNode, ArraySpec

Node = TypeVar("Node", bound=AbstractNode)
StateSpec: TypeAlias = ArraySpec | tuple["StateSpec", ...] | dict[str, "StateSpec"]
State: TypeAlias = mx.array | tuple["State", ...] | dict[str, "State"]
StateKey: TypeAlias = tuple[object, ...]


class Block(Generic[Node]):
    """A same-type sequence of variables that can share a block update."""

    def __init__(self, nodes: Sequence[Node]) -> None:
        self.nodes = tuple(nodes)
        if self.nodes and len({type(node) for node in self.nodes}) != 1:
            raise ValueError("all nodes in a block must have the same type")

    @property
    def node_type(self) -> type[Node]:
        """Return the sole concrete node type represented by this block."""

        if not self.nodes:
            raise ValueError("an empty block has no node type")
        return type(self.nodes[0])

    def __contains__(self, node: object) -> bool:
        return node in self.nodes

    def __getitem__(self, index: int) -> Node:
        return self.nodes[index]

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    def __add__(self, other: object) -> Block[Node]:
        if not isinstance(other, Block):
            return NotImplemented
        if self.nodes and other.nodes and self.node_type is not other.node_type:
            raise ValueError("cannot add blocks of different node types")
        return Block(self.nodes + other.nodes)

    def __repr__(self) -> str:
        return f"Block(nodes={self.nodes!r})"


class BlockSpec:
    """Map node blocks into deterministic global state buckets."""

    def __init__(
        self,
        blocks: Sequence[Block[AbstractNode]],
        node_shape_dtypes: Mapping[type[AbstractNode], StateSpec],
    ) -> None:
        self.blocks = tuple(blocks)
        self.node_shape_dtypes = dict(node_shape_dtypes)
        self.global_sd_order: list[StateSpec] = []
        self.all_block_sds: list[StateSpec] = []
        self.block_to_global_slice_spec: list[list[int]] = []
        self.node_global_location_map: dict[AbstractNode, tuple[int, int]] = {}
        sd_index_map: dict[StateKey, int] = {}
        node_counts: list[int] = []
        known_nodes: set[AbstractNode] = set()

        for block_index, block in enumerate(self.blocks):
            if not block:
                raise ValueError("encountered an empty block in BlockSpec")
            if block.node_type not in self.node_shape_dtypes:
                raise ValueError(f"missing state template for {block.node_type.__name__}")

            state_spec = self.node_shape_dtypes[block.node_type]
            state_key = _state_spec_key(state_spec)
            bucket_index = sd_index_map.get(state_key)
            if bucket_index is None:
                bucket_index = len(self.global_sd_order)
                sd_index_map[state_key] = bucket_index
                self.global_sd_order.append(state_spec)
                self.block_to_global_slice_spec.append([])
                node_counts.append(0)

            self.all_block_sds.append(state_spec)
            self.block_to_global_slice_spec[bucket_index].append(block_index)
            for node in block:
                if node in known_nodes:
                    raise ValueError("a node cannot show up twice in BlockSpec")
                known_nodes.add(node)
                self.node_global_location_map[node] = (bucket_index, node_counts[bucket_index])
                node_counts[bucket_index] += 1


def _state_spec_key(state_spec: StateSpec) -> StateKey:
    if isinstance(state_spec, ArraySpec):
        return ("array", state_spec.shape, str(state_spec.dtype))
    if isinstance(state_spec, tuple):
        return ("tuple", *(_state_spec_key(value) for value in state_spec))
    if isinstance(state_spec, dict):
        if any(not isinstance(key, str) for key in state_spec):
            raise TypeError("state-template dictionary keys must be strings")
        return ("dict", *((key, _state_spec_key(state_spec[key])) for key in sorted(state_spec)))
    raise TypeError("state templates contain only ArraySpec, tuples, and dictionaries")


def _allocate_state(state_spec: StateSpec, batch_shape: tuple[int, ...], node_count: int) -> State:
    if isinstance(state_spec, ArraySpec):
        return mx.zeros((*batch_shape, node_count, *state_spec.shape), dtype=state_spec.dtype)
    if isinstance(state_spec, tuple):
        return tuple(_allocate_state(value, batch_shape, node_count) for value in state_spec)
    if isinstance(state_spec, dict):
        return {
            key: _allocate_state(value, batch_shape, node_count)
            for key, value in state_spec.items()
        }
    raise TypeError("state templates contain only ArraySpec, tuples, and dictionaries")


def _walk_spec_and_state(
    state_spec: StateSpec,
    state: State,
) -> Iterator[tuple[ArraySpec, mx.array]]:
    if isinstance(state_spec, ArraySpec):
        if not isinstance(state, mx.array):
            raise TypeError("state leaf must be an MLX array")
        yield state_spec, state
        return

    if isinstance(state_spec, tuple):
        if not isinstance(state, tuple) or len(state) != len(state_spec):
            raise TypeError("state structure does not match tuple template")
        for value_spec, value in zip(state_spec, state, strict=True):
            yield from _walk_spec_and_state(value_spec, value)
        return

    if isinstance(state_spec, dict):
        if not isinstance(state, dict) or state.keys() != state_spec.keys():
            raise TypeError("state structure does not match dictionary template")
        for key, value_spec in state_spec.items():
            yield from _walk_spec_and_state(value_spec, state[key])
        return

    raise TypeError("state templates contain only ArraySpec, tuples, and dictionaries")


def _check_state_compat(state_spec: StateSpec, state: State) -> tuple[int, ...]:
    """Return shared leading axes after verifying every state leaf against its template."""

    batch_shape: tuple[int, ...] | None = None
    for array_spec, value in _walk_spec_and_state(state_spec, state):
        if value.dtype != array_spec.dtype:
            raise TypeError("state dtype does not match template dtype")
        if value.ndim < len(array_spec.shape):
            raise ValueError("state shape does not contain the template shape")
        state_shape = value.shape[-len(array_spec.shape) :] if array_spec.shape else tuple()
        if state_shape != array_spec.shape:
            raise ValueError("state shape does not match template shape")
        current_batch_shape = value.shape[: value.ndim - len(array_spec.shape)]
        if batch_shape is None:
            batch_shape = current_batch_shape
        elif batch_shape != current_batch_shape:
            raise ValueError("state leaves have inconsistent batch shapes")

    if batch_shape is None:
        raise ValueError("state template must contain at least one array")
    return batch_shape


def _concat_states(state_spec: StateSpec, states: Sequence[State]) -> State:
    if isinstance(state_spec, ArraySpec):
        first_state = states[0]
        if not isinstance(first_state, mx.array):
            raise TypeError("state leaf must be an MLX array")
        node_axis = first_state.ndim - len(state_spec.shape) - 1
        return mx.concatenate(states, axis=node_axis)
    if isinstance(state_spec, tuple):
        tuple_states = cast(Sequence[tuple[State, ...]], states)
        return tuple(
            _concat_states(value_spec, [state[index] for state in tuple_states])
            for index, value_spec in enumerate(state_spec)
        )
    if isinstance(state_spec, dict):
        dictionary_states = cast(Sequence[dict[str, State]], states)
        return {
            key: _concat_states(value_spec, [state[key] for state in dictionary_states])
            for key, value_spec in state_spec.items()
        }
    raise TypeError("state templates contain only ArraySpec, tuples, and dictionaries")


def _take_state(state_spec: StateSpec, state: State, positions: mx.array) -> State:
    if isinstance(state_spec, ArraySpec):
        if not isinstance(state, mx.array):
            raise TypeError("state leaf must be an MLX array")
        node_axis = state.ndim - len(state_spec.shape) - 1
        return mx.take(state, positions, axis=node_axis)
    if isinstance(state_spec, tuple):
        if not isinstance(state, tuple):
            raise TypeError("state structure does not match tuple template")
        return tuple(
            _take_state(value_spec, state[index], positions)
            for index, value_spec in enumerate(state_spec)
        )
    if isinstance(state_spec, dict):
        if not isinstance(state, dict):
            raise TypeError("state structure does not match dictionary template")
        return {
            key: _take_state(value_spec, state[key], positions)
            for key, value_spec in state_spec.items()
        }
    raise TypeError("state templates contain only ArraySpec, tuples, and dictionaries")


def make_empty_block_state(
    blocks: Sequence[Block[AbstractNode]],
    node_shape_dtypes: Mapping[type[AbstractNode], StateSpec],
    batch_shape: tuple[int, ...] = (),
) -> list[State]:
    """Allocate zero-initialized state with the node axis after all batch axes."""

    return [
        _allocate_state(node_shape_dtypes[block.node_type], batch_shape, len(block))
        for block in blocks
    ]


def block_state_to_global(block_state: Sequence[State], spec: BlockSpec) -> list[State]:
    """Pack compatible block-local state into one global state entry per template."""

    if len(block_state) != len(spec.blocks):
        raise ValueError("number of block states does not match the BlockSpec")
    return [
        _concat_states(
            spec.global_sd_order[bucket], [block_state[index] for index in block_indexes]
        )
        for bucket, block_indexes in enumerate(spec.block_to_global_slice_spec)
    ]


def get_node_locations(nodes: Block[AbstractNode], spec: BlockSpec) -> tuple[int, mx.array]:
    """Return the global bucket and contiguous member positions for a block."""

    if not nodes:
        raise ValueError("an empty block has no node locations")
    locations = [spec.node_global_location_map.get(node) for node in nodes]
    if any(location is None for location in locations):
        raise ValueError("block contains a node absent from the BlockSpec")
    bucket_indices = {location[0] for location in locations if location is not None}
    if len(bucket_indices) != 1:
        raise ValueError("a block cannot span global state buckets")
    return bucket_indices.pop(), mx.array(
        [location[1] for location in locations if location is not None]
    )


def from_global_state(
    global_state: Sequence[State],
    spec_from: BlockSpec,
    blocks_to_extract: Sequence[Block[AbstractNode]],
) -> list[State]:
    """Extract requested block-local states from global packed state."""

    if len(global_state) != len(spec_from.global_sd_order):
        raise ValueError("number of global states does not match the BlockSpec")
    extracted: list[State] = []
    for block in blocks_to_extract:
        bucket, positions = get_node_locations(block, spec_from)
        extracted.append(
            _take_state(spec_from.global_sd_order[bucket], global_state[bucket], positions)
        )
    return extracted


def verify_block_state(
    blocks: Sequence[Block[AbstractNode]],
    states: Sequence[State],
    node_shape_dtypes: Mapping[type[AbstractNode], StateSpec],
    block_axis: int | None = None,
) -> None:
    """Raise when block-local state disagrees with node templates or block lengths."""

    if len(blocks) != len(states):
        raise ValueError("number of states is not equal to number of blocks")
    for block, state in zip(blocks, states, strict=True):
        state_spec = node_shape_dtypes.get(block.node_type)
        if state_spec is None:
            raise ValueError(f"missing state template for {block.node_type.__name__}")
        batch_shape = _check_state_compat(state_spec, state)
        if block_axis is None:
            continue
        if not -len(batch_shape) <= block_axis < len(batch_shape):
            raise ValueError("block axis is outside the state batch shape")
        if batch_shape[block_axis] != len(block):
            raise ValueError("state block length does not match block nodes")
