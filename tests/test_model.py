import math

import mlx.core as mx
import pytest

from thrmlx import Ising


def test_one_spin_reduced_energy_includes_beta() -> None:
    model = Ising(
        fields=mx.array([0.75]),
        couplings=mx.array([[0.0]]),
        beta=2.0,
    )

    energies = model.energy(mx.array([[False], [True]]))

    assert energies.shape == (2,)
    assert energies.tolist() == pytest.approx([1.5, -1.5])


def test_two_spin_reduced_energy_has_the_expected_sign_and_pair_count() -> None:
    model = Ising(
        fields=mx.array([0.2, -0.4]),
        couplings=mx.array([[0.0, 0.7], [0.7, 0.0]]),
        beta=1.5,
    )
    states = mx.array(
        [
            [False, False],
            [False, True],
            [True, False],
            [True, True],
        ]
    )

    energies = model.energy(states)

    assert energies.tolist() == pytest.approx([-1.35, 1.95, 0.15, -0.75])


def test_energy_retains_all_leading_batch_dimensions() -> None:
    model = Ising(mx.array([0.5]), mx.array([[0.0]]))
    states = mx.array([[[False], [True]], [[True], [False]]])

    energies = model.energy(states)

    assert energies.shape == (2, 2)
    assert energies.flatten().tolist() == pytest.approx([0.5, -0.5, -0.5, 0.5])


@pytest.mark.parametrize("field", ["fields", "couplings"])
def test_model_requires_mlx_parameter_arrays(field: str) -> None:
    arguments = {
        "fields": mx.array([0.0]),
        "couplings": mx.array([[0.0]]),
    }
    arguments[field] = [0.0]

    with pytest.raises(TypeError, match=field):
        Ising(**arguments)


@pytest.mark.parametrize("field", ["fields", "couplings"])
def test_model_requires_floating_parameter_arrays(field: str) -> None:
    arguments = {
        "fields": mx.array([0.0]),
        "couplings": mx.array([[0.0]]),
    }
    arguments[field] = mx.array([0]) if field == "fields" else mx.array([[0]])

    with pytest.raises(TypeError, match=field):
        Ising(**arguments)


@pytest.mark.parametrize(
    ("fields", "couplings", "field"),
    [
        (mx.array([math.nan]), mx.array([[0.0]]), "fields"),
        (mx.array([0.0]), mx.array([[math.inf]]), "couplings"),
    ],
)
def test_model_requires_finite_parameters(
    fields: mx.array, couplings: mx.array, field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        Ising(fields, couplings)


@pytest.mark.parametrize(
    ("fields", "couplings", "message"),
    [
        (mx.array(0.0), mx.array([[0.0]]), "fields.*shape"),
        (mx.array([]), mx.zeros((0, 0)), "at least one"),
        (mx.zeros((1, 1)), mx.array([[0.0]]), "fields.*shape"),
        (mx.array([0.0, 0.0]), mx.zeros((2, 1)), "couplings.*shape"),
    ],
)
def test_model_rejects_invalid_parameter_shapes(
    fields: mx.array, couplings: mx.array, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Ising(fields, couplings)


@pytest.mark.parametrize("beta", [0.0, -1.0, math.nan, math.inf])
def test_model_requires_finite_positive_beta(beta: float) -> None:
    with pytest.raises(ValueError, match="beta"):
        Ising(mx.array([0.0]), mx.array([[0.0]]), beta=beta)


@pytest.mark.parametrize("beta", [True, "1.0"])
def test_model_rejects_non_numeric_beta(beta: object) -> None:
    with pytest.raises(TypeError, match="beta"):
        Ising(mx.array([0.0]), mx.array([[0.0]]), beta=beta)


def test_model_requires_symmetric_couplings() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        Ising(
            mx.array([0.0, 0.0]),
            mx.array([[0.0, 1.0], [0.0, 0.0]]),
        )


def test_model_requires_zero_coupling_diagonal() -> None:
    with pytest.raises(ValueError, match="diagonal"):
        Ising(mx.array([0.0]), mx.array([[0.1]]))


@pytest.mark.parametrize(
    ("blocks", "message"),
    [
        ((), "nonempty"),
        (((0,), ()), "block.*nonempty"),
        (((0,), (0, 1)), "partition"),
        (((0,),), "partition"),
        (((0,), (2,)), "range"),
        (((0.0,), (1,)), "indices.*integers"),
    ],
)
def test_model_rejects_invalid_manual_partitions(
    blocks: tuple[tuple[object, ...], ...], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        Ising(mx.zeros((2,)), mx.zeros((2, 2)), blocks=blocks)


def test_model_rejects_coupled_spins_inside_one_update_block() -> None:
    with pytest.raises(ValueError, match=r"coupling.*block"):
        Ising(
            mx.zeros((2,)),
            mx.array([[0.0, 1.0], [1.0, 0.0]]),
            blocks=((0, 1),),
        )


def test_model_accepts_valid_nonminimal_coloring() -> None:
    model = Ising(mx.zeros((3,)), mx.zeros((3, 3)), blocks=((0,), (1,), (2,)))

    assert model.blocks == ((0,), (1,), (2,))


@pytest.mark.parametrize(
    ("couplings", "expected"),
    [
        (mx.zeros((3, 3)), ((0, 1, 2),)),
        (
            mx.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
            ((0, 2), (1,)),
        ),
        (
            mx.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),
            ((0,), (1,), (2,)),
        ),
        (
            mx.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ),
            ((0, 2), (1, 3)),
        ),
    ],
)
def test_model_auto_colors_in_ascending_first_fit_order(
    couplings: mx.array, expected: tuple[tuple[int, ...], ...]
) -> None:
    model = Ising(mx.zeros((couplings.shape[0],)), couplings)

    assert model.blocks == expected


@pytest.mark.parametrize(
    "state",
    [
        [[False]],
        mx.array([0]),
        mx.array([True, False]),
    ],
)
def test_energy_requires_boolean_mlx_states_with_matching_last_axis(state: object) -> None:
    model = Ising(mx.array([0.0]), mx.array([[0.0]]))

    with pytest.raises((TypeError, ValueError), match="state"):
        model.energy(state)
