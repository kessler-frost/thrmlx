"""Source-derived THRML interfaces with an MLX backend."""

from thrmlx.block_management import (
    Block,
    BlockSpec,
    block_state_to_global,
    from_global_state,
    get_node_locations,
    make_empty_block_state,
    verify_block_state,
)
from thrmlx.block_sampling import (
    BlockGibbsSpec,
    BlockSamplingProgram,
    sample_blocks,
    sample_single_block,
    sample_states,
    sample_with_observation,
)
from thrmlx.conditional_samplers import (
    AbstractConditionalSampler,
    AbstractParametricConditionalSampler,
    BernoulliConditional,
    SoftmaxConditional,
)
from thrmlx.factor import AbstractFactor, FactorSamplingProgram, WeightedFactor
from thrmlx.interaction import InteractionGroup
from thrmlx.model import Ising
from thrmlx.models.ising import (
    IsingEBM,
    IsingSamplingProgram,
    IsingTrainingSpec,
    estimate_kl_grad,
    estimate_moments,
    hinton_init,
)
from thrmlx.observers import AbstractObserver, MomentAccumulatorObserver, StateObserver
from thrmlx.pgm import (
    DEFAULT_NODE_SHAPE_DTYPES,
    AbstractNode,
    ArraySpec,
    CategoricalNode,
    SpinNode,
)
from thrmlx.sampling import sample
from thrmlx.schedule import Clamp, SamplingSchedule

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_NODE_SHAPE_DTYPES",
    "AbstractConditionalSampler",
    "AbstractFactor",
    "AbstractNode",
    "AbstractObserver",
    "AbstractParametricConditionalSampler",
    "ArraySpec",
    "BernoulliConditional",
    "Block",
    "BlockGibbsSpec",
    "BlockSamplingProgram",
    "BlockSpec",
    "CategoricalNode",
    "Clamp",
    "FactorSamplingProgram",
    "InteractionGroup",
    "Ising",
    "IsingEBM",
    "IsingSamplingProgram",
    "IsingTrainingSpec",
    "MomentAccumulatorObserver",
    "SamplingSchedule",
    "SoftmaxConditional",
    "SpinNode",
    "StateObserver",
    "WeightedFactor",
    "__version__",
    "block_state_to_global",
    "estimate_kl_grad",
    "estimate_moments",
    "from_global_state",
    "get_node_locations",
    "hinton_init",
    "make_empty_block_state",
    "sample",
    "sample_blocks",
    "sample_single_block",
    "sample_states",
    "sample_with_observation",
    "verify_block_state",
]
