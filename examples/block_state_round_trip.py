"""Pack and recover THRML-style block state through the MLX backend."""

import mlx.core as mx

from thrmlx import (
    AbstractNode,
    ArraySpec,
    Block,
    BlockSpec,
    SpinNode,
    block_state_to_global,
    from_global_state,
    make_empty_block_state,
)


class VectorNode(AbstractNode):
    """A node whose state is a two-element MLX vector."""


def main() -> None:
    """Allocate block-local state, pack it, then recover the original blocks."""

    blocks = [
        Block([SpinNode(), SpinNode()]),
        Block([SpinNode()]),
        Block([VectorNode(), VectorNode()]),
    ]
    templates = {
        SpinNode: ArraySpec((), mx.bool_),
        VectorNode: ArraySpec((2,), mx.float32),
    }
    spec = BlockSpec(blocks, templates)
    block_state = make_empty_block_state(blocks, templates, batch_shape=(2,))
    global_state = block_state_to_global(block_state, spec)
    recovered = from_global_state(global_state, spec, blocks)

    assert recovered[1].shape == (2, 1)
    print(global_state[0].shape)


if __name__ == "__main__":
    main()
