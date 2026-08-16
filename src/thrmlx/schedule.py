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


def _resolve_schedule_count(
    upstream_name: str,
    upstream_value: int | None,
    mlx_name: str,
    mlx_value: int | None,
    default: int,
) -> int:
    if upstream_value is not None and mlx_value is not None:
        raise TypeError(f"{upstream_name} and {mlx_name} cannot both be supplied")
    value = default if upstream_value is None else upstream_value
    return value if mlx_value is None else mlx_value


@dataclass(frozen=True, slots=True, init=False)
class SamplingSchedule:
    """Number and spacing of recorded block-Gibbs states."""

    warmup: int
    samples: int
    sweeps_per_sample: int

    def __init__(
        self,
        n_warmup: int | None = None,
        n_samples: int | None = None,
        steps_per_sample: int | None = None,
        *,
        warmup: int | None = None,
        samples: int | None = None,
        sweeps_per_sample: int | None = None,
    ) -> None:
        resolved_warmup = _resolve_schedule_count("n_warmup", n_warmup, "warmup", warmup, 0)
        resolved_samples = _resolve_schedule_count("n_samples", n_samples, "samples", samples, 1)
        resolved_sweeps = _resolve_schedule_count(
            "steps_per_sample", steps_per_sample, "sweeps_per_sample", sweeps_per_sample, 1
        )
        _validate_count("warmup", resolved_warmup, 0)
        _validate_count("samples", resolved_samples, 1)
        _validate_count("sweeps_per_sample", resolved_sweeps, 0)
        object.__setattr__(self, "warmup", resolved_warmup)
        object.__setattr__(self, "samples", resolved_samples)
        object.__setattr__(self, "sweeps_per_sample", resolved_sweeps)

    @property
    def n_warmup(self) -> int:
        """Return the THRML name for ``warmup``."""

        return self.warmup

    @property
    def n_samples(self) -> int:
        """Return the THRML name for ``samples``."""

        return self.samples

    @property
    def steps_per_sample(self) -> int:
        """Return the THRML name for ``sweeps_per_sample``."""

        return self.sweeps_per_sample


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
