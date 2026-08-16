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
)
from thrmlx.conditional_samplers import AbstractConditionalSampler
from thrmlx.factor import AbstractFactor, FactorSamplingProgram, WeightedFactor
from thrmlx.interaction import InteractionGroup
from thrmlx.model import Ising
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
    "ArraySpec",
    "Block",
    "BlockGibbsSpec",
    "BlockSamplingProgram",
    "BlockSpec",
    "CategoricalNode",
    "Clamp",
    "FactorSamplingProgram",
    "InteractionGroup",
    "Ising",
    "SamplingSchedule",
    "SpinNode",
    "WeightedFactor",
    "__version__",
    "block_state_to_global",
    "from_global_state",
    "get_node_locations",
    "make_empty_block_state",
    "sample",
    "sample_blocks",
    "sample_single_block",
    "sample_states",
    "verify_block_state",
]
