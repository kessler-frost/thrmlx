"""Sampling configuration and clamped-spin values."""

from dataclasses import dataclass

# MLX 0.32 ships its native extension without typing metadata.
import mlx.core as mx  # ty: ignore[unresolved-import]


def _validate_count(name: str, value: int, minimum: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _validate_boolean_array(name: str, value: mx.array) -> None:
    if not isinstance(value, mx.array):
        raise TypeError(f"{name} must be an MLX array")
    if value.dtype != mx.bool_:
        raise TypeError(f"{name} must have boolean dtype")


@dataclass(frozen=True, slots=True)
class SamplingSchedule:
    """Number and spacing of recorded block-Gibbs states."""

    warmup: int = 0
    samples: int = 1
    sweeps_per_sample: int = 1

    def __post_init__(self) -> None:
        _validate_count("warmup", self.warmup, 0)
        _validate_count("samples", self.samples, 1)
        _validate_count("sweeps_per_sample", self.sweeps_per_sample, 0)


@dataclass(frozen=True, slots=True)
class Clamp:
    """Boolean mask and values for spins held fixed during sampling."""

    mask: mx.array
    values: mx.array

    def __post_init__(self) -> None:
        _validate_boolean_array("mask", self.mask)
        _validate_boolean_array("values", self.values)
        if self.mask.ndim not in (1, 2):
            raise ValueError("mask rank must be one or two")
        padded_values_shape = (1,) * (self.mask.ndim - self.values.ndim) + self.values.shape
        incompatible_dimension = any(
            value_size not in (1, mask_size)
            for value_size, mask_size in zip(padded_values_shape, self.mask.shape, strict=False)
        )
        if self.values.ndim > self.mask.ndim or incompatible_dimension:
            raise ValueError("values must broadcast exactly to the mask shape")
