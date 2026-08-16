"""MLX translations of upstream THRML discrete-EBM objectives."""

from math import exp

import mlx.core as mx
import pytest

from thrmlx import (
    ArraySpec,
    Block,
    BlockGibbsSpec,
    CategoricalNode,
    FactorSamplingProgram,
    SamplingSchedule,
    SpinNode,
    block_state_to_global,
    sample_blocks,
    sample_single_block,
    sample_states,
)
from thrmlx.models.discrete_ebm import (
    CategoricalEBMFactor,
    CategoricalGibbsConditional,
    DiscreteEBMFactor,
    SpinEBMFactor,
    SpinGibbsConditional,
    SquareDiscreteEBMFactor,
)
from thrmlx.models.ebm import FactorizedEBM


def _spin_block(size: int) -> Block[SpinNode]:
    return Block([SpinNode() for _ in range(size)])


def _categorical_block(size: int) -> Block[CategoricalNode]:
    return Block([CategoricalNode() for _ in range(size)])


def test_upstream_testfactor_good() -> None:
    """Catch a valid mixed discrete factor being rejected at construction."""

    factor = DiscreteEBMFactor(
        [_spin_block(4)],
        [_categorical_block(4)],
        mx.zeros((4, 3), dtype=mx.float32),
    )

    assert factor.weights.shape == (4, 3)


def test_upstream_testfactor_wrong_n_cat() -> None:
    """Catch a factor that accepts a weight rank inconsistent with categories."""

    with pytest.raises(RuntimeError, match="weight tensor"):
        DiscreteEBMFactor(
            [_spin_block(4)],
            [_categorical_block(4)],
            mx.zeros((4, 3, 3), dtype=mx.float32),
        )


def test_upstream_testfactor_duplicated_type() -> None:
    """Catch one node type being both spin and categorical in a factor."""

    block = _spin_block(4)

    with pytest.raises(RuntimeError, match="categorical and spin"):
        DiscreteEBMFactor([block], [block], mx.zeros((4, 3), dtype=mx.float32))


def test_upstream_testsquare_good() -> None:
    """Catch a square categorical interaction tensor being rejected."""

    factor = SquareDiscreteEBMFactor(
        [],
        [_categorical_block(4) for _ in range(3)],
        mx.zeros((4, 3, 3, 3), dtype=mx.float32),
    )

    assert factor.weights.shape == (4, 3, 3, 3)


def test_upstream_testsquare_bad() -> None:
    """Catch a nonsquare categorical table silently enabling group merging."""

    with pytest.raises(RuntimeError, match="square"):
        SquareDiscreteEBMFactor(
            [],
            [_categorical_block(4) for _ in range(3)],
            mx.zeros((4, 3, 2, 1), dtype=mx.float32),
        )


def test_upstream_testinteractions_to_interactions() -> None:
    """Catch mixed-factor lowering that omits a choice of conditional head."""

    factor = DiscreteEBMFactor(
        [_spin_block(4), _spin_block(4)],
        [_categorical_block(4), _categorical_block(4)],
        mx.zeros((4, 3, 3), dtype=mx.float32),
    )

    groups = factor.to_interaction_groups()

    assert len(groups) == 3
    assert groups[0].interaction.n_spin == 1


def test_upstream_testinteractions_to_interactions_binary() -> None:
    """Catch spin-factor lowering that loses any variable's directed update."""

    blocks = [_spin_block(4) for _ in range(3)]
    weights = mx.array([0.25, -0.5, 1.0, 2.0], dtype=mx.float32)
    group = SpinEBMFactor(blocks, weights).to_interaction_groups()[0]

    assert len(group.head_nodes) == 12
    assert len(group.tail_nodes) == 2
    assert all(len(tail) == 12 for tail in group.tail_nodes)
    assert group.interaction.weights.shape == (12,)


def test_upstream_testenergy_bin() -> None:
    """Catch the sign or spin encoding in binary factor energies."""

    blocks = [_spin_block(2) for _ in range(3)]
    weights = mx.array([2.0, -3.0], dtype=mx.float32)
    factor = SpinEBMFactor([blocks[2], blocks[0], blocks[1]], weights)
    ebm = FactorizedEBM([factor])
    state = [
        mx.array([True, False]),
        mx.array([True, True]),
        mx.array([False, False]),
    ]

    assert ebm.energy(state, blocks).item() == pytest.approx(5.0)


def test_upstream_testenergy_cat() -> None:
    """Catch categorical-table lookup that indexes a factor in the wrong order."""

    blocks = [_categorical_block(2) for _ in range(2)]
    weights = mx.array(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
            [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
        ],
        dtype=mx.float32,
    )
    ebm = FactorizedEBM([CategoricalEBMFactor([blocks[1], blocks[0]], weights)])
    state = [mx.array([2, 0], dtype=mx.uint8), mx.array([1, 2], dtype=mx.uint8)]

    assert ebm.energy(state, blocks).item() == pytest.approx(-21.0)


def test_upstream_testenergy_mixed() -> None:
    """Catch mixed spin/categorical factor energies with the wrong spin sign."""

    spin_block = _spin_block(2)
    cat_block = _categorical_block(2)
    weights = mx.array([[1.0, 4.0, 8.0], [2.0, 5.0, 9.0]], dtype=mx.float32)
    ebm = FactorizedEBM([DiscreteEBMFactor([spin_block], [cat_block], weights)])
    state = [mx.array([1, 2], dtype=mx.uint8), mx.array([True, False])]

    assert ebm.energy(state, [cat_block, spin_block]).item() == pytest.approx(5.0)


def test_factor_energy_reads_global_state_through_a_block_spec() -> None:
    """Catch factor energy that happens to rely on caller-local block order."""

    spins = _spin_block(2)
    categories = _categorical_block(2)
    factor = DiscreteEBMFactor(
        [spins],
        [categories],
        mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32),
    )
    spec = BlockGibbsSpec(
        [categories, spins],
        [],
        {
            SpinNode: ArraySpec((), mx.bool_),
            CategoricalNode: ArraySpec((), mx.uint8),
        },
    )
    global_state = block_state_to_global(
        [mx.array([1, 0], dtype=mx.uint8), mx.array([True, False])],
        spec,
    )

    assert factor.energy(global_state, spec).item() == pytest.approx(1.0)


def test_upstream_testsamplertype_good() -> None:
    """Catch a compatible spin conditional failing in a mixed-factor program."""

    free = _spin_block(1)
    clamp_spin = _spin_block(1)
    clamp_category = _categorical_block(1)
    factor = DiscreteEBMFactor(
        [free, clamp_spin],
        [clamp_category],
        mx.array([[100.0, -100.0, -100.0]], dtype=mx.float32),
    )
    program = FactorSamplingProgram(
        BlockGibbsSpec([free], [clamp_spin, clamp_category]),
        [SpinGibbsConditional()],
        [factor],
    )

    sampled, _ = sample_single_block(
        mx.random.key(3434),
        [mx.array([False])],
        [mx.array([True]), mx.array([0], dtype=mx.uint8)],
        program,
        0,
        None,
    )

    assert sampled.tolist() == [True]


def test_upstream_testsamplertype_bad_bin() -> None:
    """Catch spin conditionals that accept a non-Boolean spin state."""

    free = _spin_block(1)
    clamp_spin = _spin_block(1)
    clamp_category = _categorical_block(1)
    factor = DiscreteEBMFactor(
        [free, clamp_spin],
        [clamp_category],
        mx.zeros((1, 3), dtype=mx.float32),
    )
    program = FactorSamplingProgram(
        BlockGibbsSpec(
            [free],
            [clamp_spin, clamp_category],
            {SpinNode: ArraySpec((), mx.float32), CategoricalNode: ArraySpec((), mx.uint8)},
        ),
        [SpinGibbsConditional()],
        [factor],
    )

    with pytest.raises(RuntimeError, match="bool"):
        sample_single_block(
            mx.random.key(3434),
            [mx.array([0.0], dtype=mx.float32)],
            [mx.array([1.0], dtype=mx.float32), mx.array([0], dtype=mx.uint8)],
            program,
            0,
            None,
        )


def test_upstream_testsamplertype_bad_cat() -> None:
    """Catch spin conditionals that accept a signed categorical tail state."""

    free = _spin_block(1)
    clamp_spin = _spin_block(1)
    clamp_category = _categorical_block(1)
    factor = DiscreteEBMFactor(
        [free, clamp_spin],
        [clamp_category],
        mx.zeros((1, 3), dtype=mx.float32),
    )
    program = FactorSamplingProgram(
        BlockGibbsSpec(
            [free],
            [clamp_spin, clamp_category],
            {SpinNode: ArraySpec((), mx.bool_), CategoricalNode: ArraySpec((), mx.int8)},
        ),
        [SpinGibbsConditional()],
        [factor],
    )

    with pytest.raises(RuntimeError, match="unsigned"):
        sample_single_block(
            mx.random.key(3434),
            [mx.array([False])],
            [mx.array([True]), mx.array([0], dtype=mx.int8)],
            program,
            0,
            None,
        )


def test_upstream_testblocksample_binary_bias() -> None:
    """Catch a spin bias conditional with an incorrect Boltzmann sign or factor of two."""

    block = _spin_block(3)
    program = FactorSamplingProgram(
        BlockGibbsSpec([block], []),
        [SpinGibbsConditional()],
        [SpinEBMFactor([block], mx.array([100.0, -100.0, 100.0], dtype=mx.float32))],
    )

    sampled, _ = sample_single_block(
        mx.random.key(342), [mx.array([False, False, False])], [], program, 0, None
    )

    assert sampled.tolist() == [True, False, True]


def test_batched_spin_conditionals_preserve_leading_chain_axes() -> None:
    """Catch MLX conditional updates that only work after Python-side chain unbatching."""

    block = _spin_block(2)
    program = FactorSamplingProgram(
        BlockGibbsSpec([block], []),
        [SpinGibbsConditional()],
        [SpinEBMFactor([block], mx.array([100.0, -100.0], dtype=mx.float32))],
    )
    trace = sample_states(
        mx.random.key(342),
        program,
        SamplingSchedule(warmup=1, samples=3, sweeps_per_sample=1),
        [mx.zeros((4, 2), dtype=mx.bool_)],
        [],
        [block],
    )[0]

    assert trace.shape == (3, 4, 2)
    assert trace.tolist() == [[[True, False]] * 4] * 3


def test_upstream_testblocksample_categorical_bias() -> None:
    """Catch a categorical bias conditional that samples the wrong category axis."""

    block = _categorical_block(3)
    program = FactorSamplingProgram(
        BlockGibbsSpec([block], []),
        [CategoricalGibbsConditional(3)],
        [
            CategoricalEBMFactor(
                [block],
                mx.array(
                    [[0.0, 100.0, 0.0], [100.0, 0.0, 0.0], [0.0, 0.0, 100.0]],
                    dtype=mx.float32,
                ),
            )
        ],
    )

    sampled, _ = sample_single_block(
        mx.random.key(342), [mx.array([0, 0, 0], dtype=mx.uint8)], [], program, 0, None
    )

    assert sampled.dtype == mx.uint8
    assert sampled.tolist() == [1, 0, 2]


def test_upstream_testblocksample_categorical_uint8_rejects_too_many_categories() -> None:
    """Catch category draws that silently overflow an unsigned output dtype."""

    block = _categorical_block(1)
    program = FactorSamplingProgram(
        BlockGibbsSpec([block], []),
        [CategoricalGibbsConditional(300)],
        [CategoricalEBMFactor([block], mx.zeros((1, 300), dtype=mx.float32))],
    )

    with pytest.raises(RuntimeError, match=r"n_categories=300.*uint8"):
        sample_single_block(
            mx.random.key(342), [mx.array([0], dtype=mx.uint8)], [], program, 0, None
        )


def test_upstream_testblocksample_categorical_triplet() -> None:
    """Catch three-body categorical conditionals that misalign clamped tail slices."""

    blocks = [_categorical_block(2) for _ in range(3)]
    weights = mx.full((2, 3, 3, 3), -100.0, dtype=mx.float32)
    weights[0, 1, 2, 0] = 100.0
    weights[1, 0, 1, 2] = 100.0
    factor = CategoricalEBMFactor(blocks, weights)
    state = [
        mx.array([1, 0], dtype=mx.uint8),
        mx.array([2, 1], dtype=mx.uint8),
        mx.array([0, 2], dtype=mx.uint8),
    ]

    expected = ([1, 0], [2, 1], [0, 2])
    for index in range(3):
        program = FactorSamplingProgram(
            BlockGibbsSpec(
                [blocks[index]],
                [block for block_index, block in enumerate(blocks) if block_index != index],
            ),
            [CategoricalGibbsConditional(3)],
            [factor],
        )
        clamped = [value for block_index, value in enumerate(state) if block_index != index]
        sampled, _ = sample_single_block(
            mx.random.key(342), [state[index]], clamped, program, 0, None
        )

        assert sampled.tolist() == expected[index]


def test_upstream_testsampling_binary() -> None:
    """Catch binary EBM traces whose empirical Gibbs marginal is incorrect."""

    block = _spin_block(1)
    field = 0.35
    program = FactorSamplingProgram(
        BlockGibbsSpec([block], []),
        [SpinGibbsConditional()],
        [SpinEBMFactor([block], mx.array([field], dtype=mx.float32))],
    )
    trace = sample_states(
        mx.random.key(4243),
        program,
        SamplingSchedule(warmup=20, samples=4_000, sweeps_per_sample=1),
        [mx.array([False])],
        [],
        [block],
    )[0]
    expected_true_probability = 1 / (1 + exp(-2 * field))

    assert mx.mean(trace.astype(mx.float32)).item() == pytest.approx(
        expected_true_probability, abs=0.04
    )


def test_upstream_testsampling_categorical() -> None:
    """Catch categorical EBM traces whose empirical softmax marginal is incorrect."""

    block = _categorical_block(1)
    logits = (0.4, -0.2, 0.8)
    program = FactorSamplingProgram(
        BlockGibbsSpec([block], []),
        [CategoricalGibbsConditional(3)],
        [CategoricalEBMFactor([block], mx.array([logits], dtype=mx.float32))],
    )
    trace = sample_states(
        mx.random.key(443),
        program,
        SamplingSchedule(warmup=20, samples=4_000, sweeps_per_sample=1),
        [mx.array([0], dtype=mx.uint8)],
        [],
        [block],
    )[0][:, 0]
    normalizer = sum(exp(logit) for logit in logits)

    for category, logit in enumerate(logits):
        empirical = mx.mean((trace == category).astype(mx.float32)).item()
        assert empirical == pytest.approx(exp(logit) / normalizer, abs=0.04)


def test_upstream_testsampling_mixed() -> None:
    """Catch mixed EBM traces that disagree with their enumerated Boltzmann law."""

    spin_block = _spin_block(1)
    categorical_block = _categorical_block(1)
    categorical_bias = (0.2, -0.2, 0.5)
    spin_categorical = (-1.0, 0.0, 1.0)
    factors = [
        CategoricalEBMFactor([categorical_block], mx.array([categorical_bias], dtype=mx.float32)),
        DiscreteEBMFactor(
            [spin_block],
            [categorical_block],
            mx.array([spin_categorical], dtype=mx.float32),
        ),
    ]
    program = FactorSamplingProgram(
        BlockGibbsSpec([spin_block, categorical_block], []),
        [SpinGibbsConditional(), CategoricalGibbsConditional(3)],
        factors,
    )
    traces = sample_states(
        mx.random.key(443),
        program,
        SamplingSchedule(warmup=30, samples=6_000, sweeps_per_sample=1),
        [mx.array([False]), mx.array([0], dtype=mx.uint8)],
        [],
        [spin_block, categorical_block],
    )
    spin_trace = traces[0][:, 0]
    categorical_trace = traces[1][:, 0]
    unnormalized = [
        exp((1 if spin else -1) * spin_categorical[category] + categorical_bias[category])
        for spin in (False, True)
        for category in range(3)
    ]
    normalizer = sum(unnormalized)

    for spin_index, spin in enumerate((False, True)):
        for category in range(3):
            index = 3 * spin_index + category
            empirical = mx.mean(
                ((spin_trace == spin) & (categorical_trace == category)).astype(mx.float32)
            ).item()
            assert empirical == pytest.approx(unnormalized[index] / normalizer, abs=0.035)


def test_upstream_testblocksample_ragged_mixed() -> None:
    """Catch ragged mixed interactions that lose an update while padding other heads."""

    free_spins = _spin_block(2)
    free_categories = _categorical_block(2)
    clamp_spins = _spin_block(2)
    clamp_categories = _categorical_block(2)
    spin_tail_factor = SpinEBMFactor(
        [Block([free_spins[0]]), Block([clamp_spins[0]])],
        mx.array([100.0], dtype=mx.float32),
    )
    categorical_tail_factor = DiscreteEBMFactor(
        [Block([free_spins[1]])],
        [Block([clamp_categories[0]]), Block([clamp_categories[1]])],
        mx.full((1, 3, 3), -100.0, dtype=mx.float32),
    )
    categorical_tail_factor.weights[0, 1, 2] = 100.0
    free_category_factor = CategoricalEBMFactor(
        [Block([clamp_categories[0]]), Block([clamp_categories[1]]), Block([free_categories[0]])],
        mx.full((1, 3, 3, 3), -100.0, dtype=mx.float32),
    )
    free_category_factor.weights[0, 1, 2, 0] = 100.0
    spin_pair_factor = DiscreteEBMFactor(
        [Block([clamp_spins[0]]), Block([clamp_spins[1]])],
        [Block([free_categories[1]])],
        mx.array([[-100.0, -100.0, 100.0]], dtype=mx.float32),
    )
    program = FactorSamplingProgram(
        BlockGibbsSpec([free_spins, free_categories], [clamp_spins, clamp_categories]),
        [SpinGibbsConditional(), CategoricalGibbsConditional(3)],
        [spin_tail_factor, categorical_tail_factor, free_category_factor, spin_pair_factor],
    )

    sampled_spins, _ = sample_single_block(
        mx.random.key(34233434),
        [mx.array([False, False]), mx.array([0, 0], dtype=mx.uint8)],
        [mx.array([True, True]), mx.array([1, 2], dtype=mx.uint8)],
        program,
        0,
        None,
    )
    sampled_categories, _ = sample_single_block(
        mx.random.key(34233434),
        [mx.array([False, False]), mx.array([0, 0], dtype=mx.uint8)],
        [mx.array([True, True]), mx.array([1, 2], dtype=mx.uint8)],
        program,
        1,
        None,
    )

    assert sampled_spins.tolist() == [True, True]
    assert sampled_categories.tolist() == [0, 2]


def test_upstream_testequivalence_equivalence() -> None:
    """Catch equivalent binary and categorical edge models with different sample laws."""

    class GridNode(SpinNode):
        """A shared node identity type whose state encoding varies by model."""

    left = Block([GridNode()])
    right = Block([GridNode()])
    table = ((0.3, -0.4), (0.7, 0.2))
    weight_table = mx.array([table], dtype=mx.float32)
    first_field = (-table[0][0] - table[0][1] + table[1][0] + table[1][1]) / 4
    second_field = (-table[0][0] + table[0][1] - table[1][0] + table[1][1]) / 4
    pair_field = (table[0][0] - table[0][1] - table[1][0] + table[1][1]) / 4
    categorical_ebm = FactorizedEBM(
        [CategoricalEBMFactor([left, right], weight_table)],
        {GridNode: ArraySpec((), mx.uint8)},
    )
    spin_ebm = FactorizedEBM(
        [
            SpinEBMFactor(
                [Block([left[0], right[0]])],
                mx.array([first_field, second_field], dtype=mx.float32),
            ),
            SpinEBMFactor([left, right], mx.array([pair_field], dtype=mx.float32)),
        ],
        {GridNode: ArraySpec((), mx.bool_)},
    )
    offsets = []
    for left_value in (False, True):
        for right_value in (False, True):
            spin_energy = spin_ebm.energy(
                [mx.array([left_value]), mx.array([right_value])], [left, right]
            ).item()
            categorical_energy = categorical_ebm.energy(
                [
                    mx.array([int(left_value)], dtype=mx.uint8),
                    mx.array([int(right_value)], dtype=mx.uint8),
                ],
                [left, right],
            ).item()
            offsets.append(categorical_energy - spin_energy)

    assert offsets == pytest.approx([offsets[0]] * 4)

    spin_program = FactorSamplingProgram(
        BlockGibbsSpec([left, right], [], {GridNode: ArraySpec((), mx.bool_)}),
        [SpinGibbsConditional(), SpinGibbsConditional()],
        spin_ebm.factors,
    )
    categorical_program = FactorSamplingProgram(
        BlockGibbsSpec([left, right], [], {GridNode: ArraySpec((), mx.uint8)}),
        [CategoricalGibbsConditional(2), CategoricalGibbsConditional(2)],
        categorical_ebm.factors,
    )
    schedule = SamplingSchedule(warmup=20, samples=4_000, sweeps_per_sample=1)
    spin_trace = sample_states(
        mx.random.key(2232),
        spin_program,
        schedule,
        [mx.array([False]), mx.array([False])],
        [],
        [left, right],
    )
    categorical_trace = sample_states(
        mx.random.key(2232),
        categorical_program,
        schedule,
        [mx.array([0], dtype=mx.uint8), mx.array([0], dtype=mx.uint8)],
        [],
        [left, right],
    )

    for left_value in (False, True):
        for right_value in (False, True):
            spin_frequency = mx.mean(
                ((spin_trace[0][:, 0] == left_value) & (spin_trace[1][:, 0] == right_value)).astype(
                    mx.float32
                )
            ).item()
            categorical_frequency = mx.mean(
                (
                    (categorical_trace[0][:, 0] == int(left_value))
                    & (categorical_trace[1][:, 0] == int(right_value))
                ).astype(mx.float32)
            ).item()
            assert spin_frequency == pytest.approx(categorical_frequency, abs=0.04)


def test_upstream_testheterogrid_grid() -> None:
    """Catch directed lowering errors on a checkerboard grid mixing spin and categorical nodes."""

    top_left = SpinNode()
    top_right = CategoricalNode()
    bottom_left = CategoricalNode()
    bottom_right = SpinNode()
    spin_block = Block([top_left, bottom_right])
    categorical_block = Block([top_right, bottom_left])
    factor = DiscreteEBMFactor(
        [Block([top_left, top_left, bottom_right, bottom_right])],
        [Block([top_right, bottom_left, top_right, bottom_left])],
        mx.array(
            [[0.2, -0.1, 0.4], [-0.3, 0.5, 0.1], [0.4, 0.0, -0.2], [0.1, 0.3, -0.4]],
            dtype=mx.float32,
        ),
    )
    program = FactorSamplingProgram(
        BlockGibbsSpec([[spin_block], [categorical_block]], []),
        [SpinGibbsConditional(), CategoricalGibbsConditional(3)],
        [factor],
    )
    traces = sample_states(
        mx.random.key(42424),
        program,
        SamplingSchedule(warmup=10, samples=500, sweeps_per_sample=1),
        [mx.array([False, True]), mx.array([0, 1], dtype=mx.uint8)],
        [],
        [spin_block, categorical_block],
    )

    assert traces[0].shape == (500, 2)
    assert traces[0].dtype == mx.bool_
    assert traces[1].shape == (500, 2)
    assert traces[1].dtype == mx.uint8
    assert mx.all(traces[1] < 3).item()


def test_upstream_testbiggrid_big() -> None:
    """Catch large-grid program lowering that cannot complete a full checkerboard sweep."""

    side_length = 32
    nodes = [[SpinNode() for _ in range(side_length)] for _ in range(side_length)]
    color_zero: list[SpinNode] = []
    color_one: list[SpinNode] = []
    edge_left: list[SpinNode] = []
    edge_right: list[SpinNode] = []
    for row in range(side_length):
        for column in range(side_length):
            node = nodes[row][column]
            (color_zero if (row + column) % 2 == 0 else color_one).append(node)
            if row + 1 < side_length:
                edge_left.append(node)
                edge_right.append(nodes[row + 1][column])
            if column + 1 < side_length:
                edge_left.append(node)
                edge_right.append(nodes[row][column + 1])
    blocks = [Block(color_zero), Block(color_one)]
    factor = SpinEBMFactor(
        [Block(edge_left), Block(edge_right)],
        mx.zeros((len(edge_left),), dtype=mx.float32),
    )
    program = FactorSamplingProgram(
        BlockGibbsSpec(blocks, []),
        [SpinGibbsConditional(), SpinGibbsConditional()],
        [factor],
    )

    updated, _ = sample_blocks(
        mx.random.key(424),
        [mx.zeros((len(color_zero),), dtype=mx.bool_), mx.zeros((len(color_one),), dtype=mx.bool_)],
        [],
        program,
        [None, None],
    )

    assert [state.shape for state in updated] == [(len(color_zero),), (len(color_one),)]
