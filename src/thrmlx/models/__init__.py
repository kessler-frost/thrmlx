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
from thrmlx.models.ebm import AbstractEBM, EBMFactor, FactorizedEBM

__all__ = [
    "AbstractEBM",
    "CategoricalEBMFactor",
    "CategoricalGibbsConditional",
    "DiscreteEBMFactor",
    "DiscreteEBMInteraction",
    "EBMFactor",
    "FactorizedEBM",
    "SpinEBMFactor",
    "SpinGibbsConditional",
    "SquareCategoricalEBMFactor",
    "SquareDiscreteEBMFactor",
]
