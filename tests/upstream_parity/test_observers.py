"""MLX translation of the upstream THRML observer objective."""

import mlx.core as mx

from thrmlx import ArraySpec, Block, BlockGibbsSpec, CategoricalNode, SpinNode
from thrmlx.observers import MomentAccumulatorObserver, StateObserver


class _Program:
    """Minimal program carrier used to observe a static state layout."""

    def __init__(self, gibbs_spec: BlockGibbsSpec) -> None:
        self.gibbs_spec = gibbs_spec


def test_upstream_testmomentobserver_preserves_mixed_node_values() -> None:
    """Catch a moment observer that corrupts Boolean/unsigned mixed state values."""

    spin = SpinNode()
    category = CategoricalNode()
    blocks = [Block([spin]), Block([category])]
    program = _Program(
        BlockGibbsSpec(
            blocks,
            [],
            {SpinNode: ArraySpec((), mx.bool_), CategoricalNode: ArraySpec((), mx.uint8)},
        )
    )
    observer = MomentAccumulatorObserver([[(spin, category)]])

    carry, recorded = observer(
        program,
        [mx.array([True]), mx.array([2], dtype=mx.uint8)],
        [],
        observer.init(),
        0,
    )

    assert carry[0].tolist() == [2.0]
    assert recorded is None


def test_state_observer_records_requested_nodes_in_program_layout() -> None:
    """Catch observer state gathering that depends on free-block order alone."""

    spin = SpinNode()
    category = CategoricalNode()
    blocks = [Block([spin]), Block([category])]
    program = _Program(
        BlockGibbsSpec(
            blocks,
            [],
            {SpinNode: ArraySpec((), mx.bool_), CategoricalNode: ArraySpec((), mx.uint8)},
        )
    )
    observer = StateObserver([blocks[1], blocks[0]])

    carry, recorded = observer(
        program,
        [mx.array([True]), mx.array([2], dtype=mx.uint8)],
        [],
        observer.init(),
        0,
    )

    assert carry is None
    assert [state.tolist() for state in recorded] == [[2], [True]]
