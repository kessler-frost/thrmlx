import mlx.core as mx
import pytest

from thrmlx import Clamp, Ising, SamplingSchedule, sample


def test_sample_retains_chain_sample_and_spin_axes() -> None:
    model = Ising(mx.array([0.0]), mx.array([[0.0]]))

    trace = sample(
        mx.random.key(0),
        model,
        SamplingSchedule(samples=1),
        chains=1,
    )

    assert trace.shape == (1, 1, 1)
    assert trace.dtype == mx.bool_


def test_sample_returns_requested_batched_shape() -> None:
    model = Ising(mx.zeros((3,)), mx.zeros((3, 3)))

    trace = sample(
        mx.random.key(1),
        model,
        SamplingSchedule(samples=4),
        chains=5,
    )

    assert trace.shape == (5, 4, 3)


@pytest.mark.parametrize(
    "key",
    [
        [0, 0],
        mx.array([0, 0]),
        mx.array([[0, 0]], dtype=mx.uint32),
    ],
)
def test_sample_requires_an_mlx_prng_key(key: object) -> None:
    model = Ising(mx.array([0.0]), mx.array([[0.0]]))

    with pytest.raises((TypeError, ValueError), match="key"):
        sample(key, model, SamplingSchedule())


@pytest.mark.parametrize("chains", [True, 0, -1, 1.5])
def test_sample_requires_a_positive_integer_chain_count(chains: object) -> None:
    model = Ising(mx.array([0.0]), mx.array([[0.0]]))

    with pytest.raises((TypeError, ValueError), match="chains"):
        sample(mx.random.key(0), model, SamplingSchedule(), chains=chains)


@pytest.mark.parametrize(
    "initial",
    [
        [[False]],
        mx.array([[0]]),
        mx.array([False]),
        mx.array([[False, True]]),
        mx.array([[False], [True]]),
    ],
)
def test_sample_validates_initial_state(initial: object) -> None:
    model = Ising(mx.array([0.0]), mx.array([[0.0]]))

    with pytest.raises((TypeError, ValueError), match="initial"):
        sample(
            mx.random.key(0),
            model,
            SamplingSchedule(),
            chains=1,
            initial=initial,
        )


def test_zero_warmup_records_initial_before_any_sweep() -> None:
    model = Ising(mx.array([100.0]), mx.array([[0.0]]))
    initial = mx.array([[False], [False]])

    trace = sample(
        mx.random.key(0),
        model,
        SamplingSchedule(warmup=0, samples=3, sweeps_per_sample=0),
        chains=2,
        initial=initial,
    )

    assert trace.tolist() == [
        [[False], [False], [False]],
        [[False], [False], [False]],
    ]


def test_warmup_updates_state_before_first_record() -> None:
    model = Ising(mx.array([100.0]), mx.array([[0.0]]))

    trace = sample(
        mx.random.key(0),
        model,
        SamplingSchedule(warmup=1, samples=1),
        chains=2,
        initial=mx.array([[False], [False]]),
    )

    assert trace.tolist() == [[[True]], [[True]]]


def test_sweep_updates_blocks_in_declared_order() -> None:
    couplings = mx.array([[0.0, 100.0], [100.0, 0.0]])
    initial = mx.array([[False, True]])
    schedule = SamplingSchedule(warmup=1, samples=1)
    forward = Ising(mx.zeros((2,)), couplings, blocks=((0,), (1,)))
    reverse = Ising(mx.zeros((2,)), couplings, blocks=((1,), (0,)))

    forward_trace = sample(mx.random.key(0), forward, schedule, initial=initial)
    reverse_trace = sample(mx.random.key(0), reverse, schedule, initial=initial)

    assert forward_trace.tolist() == [[[True, True]]]
    assert reverse_trace.tolist() == [[[False, False]]]


def test_one_block_updates_all_independent_spins() -> None:
    model = Ising(
        mx.array([100.0, -100.0]),
        mx.zeros((2, 2)),
        blocks=((0, 1),),
    )

    trace = sample(
        mx.random.key(3),
        model,
        SamplingSchedule(warmup=1),
        chains=2,
        initial=mx.array([[False, True], [True, False]]),
    )

    assert trace.tolist() == [[[True, False]], [[True, False]]]


def test_sample_does_not_mutate_supplied_initial_state() -> None:
    model = Ising(mx.array([100.0]), mx.array([[0.0]]))
    initial = mx.array([[False]])

    sample(
        mx.random.key(0),
        model,
        SamplingSchedule(warmup=1),
        initial=initial,
    )

    assert initial.tolist() == [[False]]


def test_shared_clamp_overrides_initial_and_survives_every_draw() -> None:
    model = Ising(mx.array([-100.0, 100.0]), mx.zeros((2, 2)))
    clamp = Clamp(
        mask=mx.array([True, False]),
        values=mx.array([True, False]),
    )

    trace = sample(
        mx.random.key(5),
        model,
        SamplingSchedule(warmup=2, samples=3, sweeps_per_sample=2),
        chains=4,
        initial=mx.zeros((4, 2), dtype=mx.bool_),
        clamp=clamp,
    )

    assert mx.all(trace[:, :, 0]).item()
    assert mx.all(trace[:, :, 1]).item()


def test_per_chain_clamp_broadcasts_values_and_fixes_selected_sites() -> None:
    model = Ising(mx.zeros((2,)), mx.zeros((2, 2)))
    clamp = Clamp(
        mask=mx.array([[True, False], [False, True], [True, True]]),
        values=mx.array([True, False]),
    )

    trace = sample(
        mx.random.key(6),
        model,
        SamplingSchedule(warmup=1, samples=3),
        chains=3,
        clamp=clamp,
    )

    assert mx.all(trace[0, :, 0]).item()
    assert mx.all(~trace[1, :, 1]).item()
    assert mx.all(trace[2, :, 0]).item()
    assert mx.all(~trace[2, :, 1]).item()


@pytest.mark.parametrize(
    "clamp",
    [
        Clamp(mask=mx.array([True, False]), values=mx.array([True, False])),
        Clamp(
            mask=mx.array([[True], [False]]),
            values=mx.array([[True], [False]]),
        ),
    ],
)
def test_sample_validates_clamp_against_model_and_chains(clamp: Clamp) -> None:
    model = Ising(mx.array([0.0]), mx.array([[0.0]]))

    with pytest.raises(ValueError, match="clamp"):
        sample(
            mx.random.key(0),
            model,
            SamplingSchedule(),
            chains=1,
            clamp=clamp,
        )


def test_identical_explicit_key_reproduces_trace() -> None:
    model = Ising(mx.zeros((4,)), mx.zeros((4, 4)))
    schedule = SamplingSchedule(warmup=2, samples=5, sweeps_per_sample=2)
    key = mx.random.key(91)

    first = sample(key, model, schedule, chains=16)
    second = sample(key, model, schedule, chains=16)

    assert mx.array_equal(first, second).item()


def test_global_mlx_seed_does_not_change_explicit_key_trace() -> None:
    model = Ising(mx.zeros((4,)), mx.zeros((4, 4)))
    schedule = SamplingSchedule(warmup=2, samples=5, sweeps_per_sample=2)
    key = mx.random.key(92)

    mx.random.seed(1)
    first = sample(key, model, schedule, chains=16)
    mx.random.seed(999)
    second = sample(key, model, schedule, chains=16)

    assert mx.array_equal(first, second).item()


def test_split_keys_produce_distinct_traces() -> None:
    model = Ising(mx.zeros((8,)), mx.zeros((8, 8)))
    schedule = SamplingSchedule(warmup=2, samples=8, sweeps_per_sample=1)
    keys = mx.random.split(mx.random.key(93), 2)

    first = sample(keys[0], model, schedule, chains=64)
    second = sample(keys[1], model, schedule, chains=64)

    assert not mx.array_equal(first, second).item()
