"""MLX translations of upstream THRML block-management objectives."""

import mlx.core as mx
import pytest

from thrmlx import (
    DEFAULT_NODE_SHAPE_DTYPES,
    AbstractNode,
    ArraySpec,
    Block,
    BlockSpec,
    CategoricalNode,
    SpinNode,
    block_state_to_global,
    from_global_state,
    get_node_locations,
    make_empty_block_state,
    verify_block_state,
)
from thrmlx.block_management import _check_state_compat


class CustomNode(AbstractNode):
    """A node type used to prove blocks retain their concrete type."""


class FloatNode(AbstractNode):
    """A vector-valued node used to exercise a second state bucket."""


class NestedNode(AbstractNode):
    """A node with nested state leaves used to exercise structural validation."""


def _assert_state_equal(actual: object, expected: object) -> None:
    if isinstance(actual, mx.array) and isinstance(expected, mx.array):
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert mx.array_equal(actual, expected).item()
        return
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected, strict=True):
            _assert_state_equal(actual_value, expected_value)
        return
    if isinstance(actual, dict) and isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_state_equal(actual[key], expected[key])
        return
    raise AssertionError(f"state structures differ: {actual!r} and {expected!r}")


def _mixed_blocks_and_specs() -> tuple[list[Block[AbstractNode]], dict[type[AbstractNode], object]]:
    return (
        [
            Block([SpinNode(), SpinNode()]),
            Block([SpinNode()]),
            Block([FloatNode(), FloatNode(), FloatNode()]),
            Block([NestedNode(), NestedNode()]),
        ],
        {
            SpinNode: ArraySpec((), mx.bool_),
            FloatNode: ArraySpec((2,), mx.float32),
            NestedNode: (
                {"flags": ArraySpec((), mx.bool_), "values": ArraySpec((2,), mx.float32)},
                ArraySpec((1,), mx.int16),
            ),
        },
    )


def test_upstream_testduplicate_good() -> None:
    """Catch a BlockSpec that rejects a layout with distinct node identities."""

    block_spec = BlockSpec(
        [Block([CustomNode(), CustomNode()]), Block([CustomNode()])],
        {CustomNode: DEFAULT_NODE_SHAPE_DTYPES[SpinNode]},
    )

    assert len(block_spec.blocks) == 2


def test_upstream_testduplicate_duplicate() -> None:
    """Catch removal of duplicate-node detection from BlockSpec."""

    node = SpinNode()

    with pytest.raises(ValueError, match="twice"):
        BlockSpec([Block([node]), Block([node])], DEFAULT_NODE_SHAPE_DTYPES)


def test_block_rejects_mixed_concrete_node_types() -> None:
    """Catch a Block that permits a sampler-unsafe mixture of node types."""

    with pytest.raises(ValueError, match="same type"):
        Block([SpinNode(), CategoricalNode()])


def test_empty_block_has_no_node_type() -> None:
    """Catch an empty Block that presents a fictitious concrete node type."""

    with pytest.raises(ValueError, match="empty"):
        _ = Block([]).node_type


def test_upstream_testblocks_empty_state() -> None:
    """Catch allocation that puts node state before the block-member axis."""

    blocks, specs = _mixed_blocks_and_specs()

    block_state = make_empty_block_state(blocks, specs, batch_shape=(2,))

    assert block_state[0].shape == (2, 2)
    assert block_state[1].shape == (2, 1)
    assert block_state[2].shape == (2, 3, 2)
    assert block_state[3][0]["flags"].shape == (2, 2)
    assert block_state[3][0]["values"].shape == (2, 2, 2)
    assert block_state[3][1].shape == (2, 2, 1)


def test_upstream_testblocks_shape_transforms() -> None:
    """Catch incorrect packing or extraction slices between block and global state."""

    blocks, specs = _mixed_blocks_and_specs()
    block_spec = BlockSpec(blocks, specs)
    block_state = make_empty_block_state(blocks, specs, batch_shape=(2,))

    unpacked = from_global_state(block_state_to_global(block_state, block_spec), block_spec, blocks)

    _assert_state_equal(unpacked, block_state)


def test_upstream_testblocks_node_lookup() -> None:
    """Catch locations that ignore the preceding block in the same state bucket."""

    blocks, specs = _mixed_blocks_and_specs()
    block_spec = BlockSpec(blocks, specs)

    bucket, positions = get_node_locations(blocks[1], block_spec)

    assert bucket == 0
    assert positions.tolist() == [2]


def test_upstream_testblockcompat_good() -> None:
    """Catch a compatible nested state that loses its common batch shape."""

    spec = (
        {"flags": ArraySpec((), mx.int8)},
        ArraySpec((2,), mx.float32),
    )
    state = (
        {"flags": mx.zeros((4, 2, 10), dtype=mx.int8)},
        mx.zeros((4, 2, 10, 2), dtype=mx.float32),
    )

    assert _check_state_compat(spec, state) == (4, 2, 10)


def test_upstream_testblockcompat_bad_dtype() -> None:
    """Catch validation that accepts a leaf with a different dtype."""

    with pytest.raises(TypeError, match="dtype"):
        _check_state_compat(ArraySpec((), mx.bool_), mx.zeros((1,), dtype=mx.int32))


def test_upstream_testblockcompat_bad_shape() -> None:
    """Catch validation that accepts a leaf with the wrong state suffix."""

    with pytest.raises(ValueError, match="shape"):
        _check_state_compat(ArraySpec((2,), mx.float32), mx.zeros((2, 1), dtype=mx.float32))


def test_upstream_testblockcompat_missing_array() -> None:
    """Catch validation that accepts a non-array where an MLX array is required."""

    with pytest.raises(TypeError, match="array"):
        _check_state_compat(ArraySpec((), mx.float32), 1.0)


def test_upstream_testblockcompat_bad_structure() -> None:
    """Catch validation that accepts a tuple template and array state as equivalent."""

    with pytest.raises(TypeError, match="structure"):
        _check_state_compat((ArraySpec((), mx.float32),), mx.zeros((1,), dtype=mx.float32))


def test_upstream_testblockcompat_good_state() -> None:
    """Catch verify_block_state rejecting a state it allocated itself."""

    blocks, specs = _mixed_blocks_and_specs()
    state = make_empty_block_state(blocks, specs, batch_shape=(3,))

    verify_block_state(blocks, state, specs, block_axis=-1)


def test_upstream_testblockcompat_wrong_state_len() -> None:
    """Catch a missing block state being silently accepted."""

    blocks, specs = _mixed_blocks_and_specs()

    with pytest.raises(ValueError, match="number of states"):
        verify_block_state(blocks, make_empty_block_state(blocks, specs)[:-1], specs)


def test_upstream_testblockcompat_bad_block() -> None:
    """Catch verify_block_state ignoring the expected type for a block."""

    blocks = [Block([SpinNode()])]

    with pytest.raises(TypeError, match="dtype"):
        verify_block_state(blocks, [mx.zeros((1,), dtype=mx.int32)], DEFAULT_NODE_SHAPE_DTYPES)


def test_upstream_testblockcompat_length_mismatch() -> None:
    """Catch a block state with four values being accepted for a three-node block."""

    blocks = [Block([FloatNode(), FloatNode(), FloatNode()])]
    specs = {FloatNode: ArraySpec((2,), mx.float32)}

    with pytest.raises(ValueError, match="block length"):
        verify_block_state(blocks, [mx.zeros((4, 2), dtype=mx.float32)], specs, block_axis=-1)
