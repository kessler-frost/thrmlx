"""Direct ``thrmlx`` import contracts derived from pinned THRML 0.1.4 exports."""

from importlib import import_module
from inspect import signature

import mlx.core as mx
import pytest

import thrmlx
from thrmlx import (
    Block,
    BlockGibbsSpec,
    CategoricalNode,
    FactorSamplingProgram,
    SamplingSchedule,
    SpinNode,
    block_state_to_global,
    from_global_state,
    get_node_locations,
    sample_blocks,
    sample_single_block,
    sample_states,
    sample_with_observation,
)
from thrmlx.factor import WeightedFactor
from thrmlx.models import DiscreteEBMFactor, SpinEBMFactor, SpinGibbsConditional
from thrmlx.observers import StateObserver

UPSTREAM_ROOT_EXPORTS = {
    "AbstractConditionalSampler",
    "AbstractFactor",
    "AbstractNode",
    "AbstractObserver",
    "AbstractParametricConditionalSampler",
    "BernoulliConditional",
    "Block",
    "BlockGibbsSpec",
    "BlockSamplingProgram",
    "BlockSpec",
    "CategoricalNode",
    "FactorSamplingProgram",
    "InteractionGroup",
    "MomentAccumulatorObserver",
    "SamplingSchedule",
    "SoftmaxConditional",
    "SpinNode",
    "StateObserver",
    "WeightedFactor",
    "__version__",
    "block_state_to_global",
    "from_global_state",
    "get_node_locations",
    "make_empty_block_state",
    "models",
    "sample_blocks",
    "sample_single_block",
    "sample_states",
    "sample_with_observation",
    "verify_block_state",
}

UPSTREAM_MODEL_EXPORTS = {
    "AbstractEBM",
    "AbstractFactorizedEBM",
    "CategoricalEBMFactor",
    "CategoricalGibbsConditional",
    "DiscreteEBMFactor",
    "DiscreteEBMInteraction",
    "EBMFactor",
    "FactorizedEBM",
    "IsingEBM",
    "IsingSamplingProgram",
    "IsingTrainingSpec",
    "SpinEBMFactor",
    "SpinGibbsConditional",
    "SquareCategoricalEBMFactor",
    "SquareDiscreteEBMFactor",
    "estimate_kl_grad",
    "estimate_moments",
    "hinton_init",
}

UPSTREAM_PARAMETER_NAMES = {
    "BlockGibbsSpec": ("free_super_blocks", "clamped_blocks", "node_shape_dtypes"),
    "sample_single_block": (
        "key",
        "state_free",
        "clamp_state",
        "program",
        "block",
        "sampler_state",
        "global_state",
    ),
    "sample_blocks": ("key", "state_free", "clamp_state", "program", "sampler_state"),
    "sample_with_observation": (
        "key",
        "program",
        "schedule",
        "init_chain_state",
        "state_clamp",
        "observation_carry_init",
        "f_observe",
    ),
    "sample_states": (
        "key",
        "program",
        "schedule",
        "init_state_free",
        "state_clamp",
        "nodes_to_sample",
    ),
    "from_global_state": ("global_state", "spec_from", "blocks_to_extract"),
    "get_node_locations": ("nodes", "spec"),
}


def _one_spin_program() -> tuple[Block[SpinNode], FactorSamplingProgram, list[mx.array]]:
    """Build a deterministic one-spin program using only direct THRML-style objects."""

    block = Block([SpinNode()])
    program = FactorSamplingProgram(
        BlockGibbsSpec(free_super_blocks=[block], clamped_blocks=[]),
        [SpinGibbsConditional()],
        [SpinEBMFactor([block], mx.array([100.0], dtype=mx.float32))],
        [],
    )
    return block, program, [mx.array([False], dtype=mx.bool_)]


def test_thrmlx_root_contains_every_pinned_thrml_export() -> None:
    """Catch a direct-import migration that omits a public THRML root symbol."""

    assert set(thrmlx.__all__) >= UPSTREAM_ROOT_EXPORTS
    assert thrmlx.models is import_module("thrmlx.models")


def test_thrmlx_models_contains_every_pinned_thrml_model_export() -> None:
    """Catch a source-compatible root whose model package silently loses a public EBM class."""

    assert set(thrmlx.models.__all__) >= UPSTREAM_MODEL_EXPORTS
    assert (
        thrmlx.models.AbstractFactorizedEBM
        is import_module("thrmlx.models.ebm").AbstractFactorizedEBM
    )


def test_direct_entry_points_keep_pinned_thrml_parameter_names() -> None:
    """Catch a Python-compatible call whose THRML keyword migration silently breaks."""

    for name, expected in UPSTREAM_PARAMETER_NAMES.items():
        entry_point = getattr(thrmlx, name)
        assert tuple(signature(entry_point).parameters) == expected


def test_discrete_ebm_factors_retain_the_upstream_weighted_factor_contract() -> None:
    """Catch discrete EBM factors that lose their public weighted-factor base class."""

    spin_block = Block([SpinNode()])
    categorical_block = Block([CategoricalNode()])
    factor = DiscreteEBMFactor(
        [spin_block],
        [categorical_block],
        mx.array([[0.25, -0.25]], dtype=mx.float32),
    )

    assert isinstance(factor, WeightedFactor)
    assert factor.node_groups == (spin_block, categorical_block)


def test_thrmlx_preserves_every_pinned_thrml_module_path() -> None:
    """Catch a root rename that leaves a documented THRML submodule import broken."""

    module_paths = (
        "thrmlx.block_management",
        "thrmlx.block_sampling",
        "thrmlx.conditional_samplers",
        "thrmlx.factor",
        "thrmlx.interaction",
        "thrmlx.observers",
        "thrmlx.pgm",
        "thrmlx.models.discrete_ebm",
        "thrmlx.models.ebm",
        "thrmlx.models.ising",
    )

    assert [import_module(path).__name__ for path in module_paths] == list(module_paths)


def test_upstream_schedule_keywords_drive_a_real_mlx_gibbs_program() -> None:
    """Catch THRML schedule keyword aliases that construct but do not control MLX sampling."""

    block, program, initial = _one_spin_program()
    schedule = SamplingSchedule(n_warmup=1, n_samples=2, steps_per_sample=1)
    trace = sample_states(
        mx.random.key(12),
        program,
        schedule,
        init_state_free=initial,
        state_clamp=[],
        nodes_to_sample=[block],
    )

    assert (schedule.warmup, schedule.samples, schedule.sweeps_per_sample) == (1, 2, 1)
    assert (schedule.n_warmup, schedule.n_samples, schedule.steps_per_sample) == (1, 2, 1)
    assert trace[0].tolist() == [[True], [True]]


def test_upstream_state_and_sampler_keywords_run_the_direct_api() -> None:
    """Catch a public sampler that only supports the MLX port's original argument spellings."""

    block, program, initial = _one_spin_program()
    sampler_state = [sampler.init() for sampler in program.samplers]
    global_state = block_state_to_global(initial, program.gibbs_spec)
    sampled_block, _ = sample_single_block(
        mx.random.key(13),
        state_free=initial,
        clamp_state=[],
        program=program,
        block=0,
        sampler_state=sampler_state[0],
        global_state=global_state,
    )
    sampled_state, _ = sample_blocks(
        mx.random.key(14),
        state_free=initial,
        clamp_state=[],
        program=program,
        sampler_state=sampler_state,
    )
    carry, observed = sample_with_observation(
        mx.random.key(15),
        program=program,
        schedule=SamplingSchedule(n_warmup=1, n_samples=1, steps_per_sample=1),
        init_chain_state=initial,
        state_clamp=[],
        observation_carry_init=None,
        f_observe=StateObserver([block]),
    )

    assert sampled_block.tolist() == [True]
    assert sampled_state[0].tolist() == [True]
    assert carry is None
    assert observed[0].tolist() == [[True]]
    assert from_global_state(
        global_state=global_state,
        spec_from=program.gibbs_spec,
        blocks_to_extract=[block],
    )[0].tolist() == [False]
    assert get_node_locations(nodes=block, spec=program.gibbs_spec)[1].tolist() == [0]


def test_schedule_rejects_ambiguous_upstream_and_mlx_keyword_aliases() -> None:
    """Catch a schedule constructor that silently chooses one conflicting public vocabulary."""

    with pytest.raises(TypeError, match="both"):
        SamplingSchedule(n_warmup=1, warmup=2)
