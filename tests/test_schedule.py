from dataclasses import FrozenInstanceError

import mlx.core as mx
import pytest

from thrmlx import Clamp, SamplingSchedule


def test_schedule_defaults_and_boundary_values() -> None:
    default = SamplingSchedule()
    boundary = SamplingSchedule(warmup=0, samples=1, sweeps_per_sample=0)

    assert (default.warmup, default.samples, default.sweeps_per_sample) == (0, 1, 1)
    assert (boundary.warmup, boundary.samples, boundary.sweeps_per_sample) == (0, 1, 0)


@pytest.mark.parametrize(
    ("arguments", "field"),
    [
        ({"warmup": True}, "warmup"),
        ({"warmup": 1.5}, "warmup"),
        ({"samples": False}, "samples"),
        ({"samples": 1.5}, "samples"),
        ({"sweeps_per_sample": True}, "sweeps_per_sample"),
        ({"sweeps_per_sample": 1.5}, "sweeps_per_sample"),
    ],
)
def test_schedule_rejects_non_integer_counts(arguments: dict[str, object], field: str) -> None:
    with pytest.raises(TypeError, match=field):
        SamplingSchedule(**arguments)


@pytest.mark.parametrize(
    ("arguments", "field"),
    [
        ({"warmup": -1}, "warmup"),
        ({"samples": 0}, "samples"),
        ({"samples": -1}, "samples"),
        ({"sweeps_per_sample": -1}, "sweeps_per_sample"),
    ],
)
def test_schedule_rejects_counts_outside_their_domain(
    arguments: dict[str, int], field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        SamplingSchedule(**arguments)


def test_schedule_is_immutable() -> None:
    schedule = SamplingSchedule()

    with pytest.raises(FrozenInstanceError):
        schedule.samples = 2  # type: ignore[misc]


@pytest.mark.parametrize("field", ["mask", "values"])
def test_clamp_requires_mlx_arrays(field: str) -> None:
    arguments = {
        "mask": mx.array([True, False]),
        "values": mx.array([False, True]),
    }
    arguments[field] = [True, False]

    with pytest.raises(TypeError, match=field):
        Clamp(**arguments)


@pytest.mark.parametrize("field", ["mask", "values"])
def test_clamp_requires_boolean_arrays(field: str) -> None:
    arguments = {
        "mask": mx.array([True, False]),
        "values": mx.array([False, True]),
    }
    arguments[field] = mx.array([0, 1])

    with pytest.raises(TypeError, match=field):
        Clamp(**arguments)


@pytest.mark.parametrize("shape", [(), (1, 1, 2)])
def test_clamp_rejects_mask_rank_other_than_one_or_two(shape: tuple[int, ...]) -> None:
    mask = mx.ones(shape, dtype=mx.bool_)
    values = mx.ones(shape, dtype=mx.bool_)

    with pytest.raises(ValueError, match=r"mask.*rank"):
        Clamp(mask=mask, values=values)


def test_clamp_rejects_values_that_do_not_broadcast_to_mask() -> None:
    with pytest.raises(ValueError, match=r"values.*broadcast"):
        Clamp(
            mask=mx.ones((3, 2), dtype=mx.bool_),
            values=mx.ones((2, 1), dtype=mx.bool_),
        )


def test_clamp_accepts_shared_values_for_per_chain_mask() -> None:
    clamp = Clamp(
        mask=mx.array([[True, False], [False, True], [True, True]]),
        values=mx.array([True, False]),
    )

    assert clamp.mask.shape == (3, 2)
    assert clamp.values.shape == (2,)


def test_clamp_is_immutable() -> None:
    clamp = Clamp(mask=mx.array([True]), values=mx.array([False]))

    with pytest.raises(FrozenInstanceError):
        clamp.mask = mx.array([False])  # type: ignore[misc]
