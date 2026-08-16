"""Contract checks for the per-use-case THRML benchmark matrix."""

import pytest


def test_matrix_covers_every_source_use_case_family() -> None:
    """Catch a benchmark table that drops a completed source-compatibility family."""

    pytest.importorskip("thrml")
    from benchmarks.source_matrix import case_ids

    assert set(case_ids()) == {
        "dense_rbm",
        "ising_line",
        "ising_grid",
        "spin_factor",
        "categorical_factor",
        "mixed_factor",
        "moment_observer",
        "contrastive_gradient",
        "mnist_fixture_update",
    }
