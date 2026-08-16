"""Sample a THRML-style mixed spin/categorical EBM on Apple Silicon."""

import mlx.core as mx

from thrmlx import (
    Block,
    BlockGibbsSpec,
    CategoricalNode,
    FactorSamplingProgram,
    SamplingSchedule,
    SpinNode,
    sample_states,
)
from thrmlx.models.discrete_ebm import (
    CategoricalEBMFactor,
    CategoricalGibbsConditional,
    DiscreteEBMFactor,
    SpinGibbsConditional,
)


def main() -> None:
    """Print a compact trace from a coupled Boolean/categorical model."""

    spin = Block([SpinNode()])
    category = Block([CategoricalNode()])
    factors = [
        CategoricalEBMFactor([category], mx.array([[0.2, -0.2, 0.5]], dtype=mx.float32)),
        DiscreteEBMFactor(
            [spin],
            [category],
            mx.array([[-1.0, 0.0, 1.0]], dtype=mx.float32),
        ),
    ]
    program = FactorSamplingProgram(
        BlockGibbsSpec([spin, category], []),
        [SpinGibbsConditional(), CategoricalGibbsConditional(3)],
        factors,
    )
    spin_trace, category_trace = sample_states(
        mx.random.key(17),
        program,
        SamplingSchedule(warmup=20, samples=4, sweeps_per_sample=2),
        [mx.array([False]), mx.array([0], dtype=mx.uint8)],
        [],
        [spin, category],
    )
    print({"spin": spin_trace.tolist(), "category": category_trace.tolist()})


if __name__ == "__main__":
    main()
