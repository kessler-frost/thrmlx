"""THRML energy-based model interfaces backed by MLX."""

from thrmlx.models.discrete_ebm import (
    CategoricalEBMFactor,
    CategoricalGibbsConditional,
    DiscreteEBMFactor,
    DiscreteEBMInteraction,
    SpinEBMFactor,
    SpinGibbsConditional,
    SquareCategoricalEBMFactor,
    SquareDiscreteEBMFactor,
)
from thrmlx.models.ebm import AbstractEBM, AbstractFactorizedEBM, EBMFactor, FactorizedEBM
from thrmlx.models.ising import (
    IsingEBM,
    IsingSamplingProgram,
    IsingTrainingSpec,
    estimate_kl_grad,
    estimate_moments,
    hinton_init,
)

__all__ = [
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
]
