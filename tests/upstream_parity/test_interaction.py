"""MLX translations of upstream THRML interaction-objective validation."""

import mlx.core as mx
import pytest

from thrmlx import AbstractNode, Block, InteractionGroup


class InteractionNode(AbstractNode):
    """A node type used only for interaction-layout validation."""


def _interaction_inputs() -> tuple[Block[InteractionNode], list[Block[InteractionNode]]]:
    return (
        Block([InteractionNode() for _ in range(3)]),
        [Block([InteractionNode() for _ in range(3)]) for _ in range(2)],
    )


def test_upstream_testinteractioninputs_good() -> None:
    """Catch a well-aligned interaction group being rejected."""

    head, tails = _interaction_inputs()

    group = InteractionGroup(mx.zeros((3,)), head, tails)

    assert group.head_nodes == head


def test_upstream_testinteractioninputs_bad_tail() -> None:
    """Catch an interaction group that permits a ragged tail block."""

    head, _ = _interaction_inputs()

    with pytest.raises(ValueError, match="same length"):
        InteractionGroup(
            mx.zeros((3,)),
            head,
            [Block([InteractionNode() for _ in range(4)])],
        )


def test_upstream_testinteractioninputs_bad_interaction() -> None:
    """Catch an interaction tree that accepts scalar or mismatched array leaves."""

    head, tails = _interaction_inputs()

    with pytest.raises(ValueError, match="leading dimension"):
        InteractionGroup((mx.zeros((3,)), mx.array(1.0)), head, tails)
    with pytest.raises(ValueError, match="leading dimension"):
        InteractionGroup((mx.zeros((3,)), mx.zeros((4,))), head, tails)
