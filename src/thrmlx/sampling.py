"""Batched block-Gibbs sampling with explicit MLX random keys."""

# MLX 0.32 ships its native extension without typing metadata.
import mlx.core as mx  # ty: ignore[unresolved-import]

from thrmlx.model import Ising
from thrmlx.schedule import Clamp, SamplingSchedule


def _validate_key(key: mx.array) -> None:
    if not isinstance(key, mx.array):
        raise TypeError("key must be an MLX PRNG key")
    if key.dtype != mx.uint32 or key.shape != (2,):
        raise ValueError("key must have uint32 dtype and shape (2,)")


def _validate_chains(chains: int) -> None:
    if type(chains) is not int:
        raise TypeError("chains must be an integer")
    if chains < 1:
        raise ValueError("chains must be at least one")


def _prepare_initial(
    initial: mx.array | None,
    *,
    chains: int,
    n_spins: int,
    key: mx.array,
) -> mx.array:
    if initial is None:
        return mx.random.bernoulli(shape=(chains, n_spins), key=key)
    if not isinstance(initial, mx.array):
        raise TypeError("initial must be an MLX array")
    if initial.dtype != mx.bool_:
        raise TypeError("initial must have boolean dtype")
    if initial.shape != (chains, n_spins):
        raise ValueError("initial must have shape (chains, N) matching the request")
    return mx.array(initial)


def _prepare_clamp(
    clamp: Clamp | None,
    *,
    chains: int,
    n_spins: int,
) -> tuple[mx.array, mx.array]:
    if clamp is None:
        empty = mx.zeros((chains, n_spins), dtype=mx.bool_)
        return empty, empty
    if not isinstance(clamp, Clamp):
        raise TypeError("clamp must be a Clamp instance")
    expected_mask_shape = (n_spins,) if clamp.mask.ndim == 1 else (chains, n_spins)
    if clamp.mask.shape != expected_mask_shape:
        raise ValueError("clamp mask must match the model spins and requested chains")
    return (
        mx.broadcast_to(clamp.mask, (chains, n_spins)),
        mx.broadcast_to(clamp.values, (chains, n_spins)),
    )


@mx.compile
def _update_block(
    state: mx.array,
    fields: mx.array,
    couplings: mx.array,
    block: mx.array,
    key: mx.array,
    beta: mx.array,
    clamp_mask: mx.array,
    clamp_values: mx.array,
) -> mx.array:
    signed = 2 * state.astype(fields.dtype) - 1
    local_fields = fields[block] + signed @ couplings[:, block]
    probabilities = mx.sigmoid(2 * beta * local_fields)
    draws = mx.random.bernoulli(probabilities, key=key)
    updated = mx.array(state)
    updated[:, block] = draws
    return mx.where(clamp_mask, clamp_values, updated)


def _run_sweeps(
    state: mx.array,
    model: Ising,
    keys: mx.array,
    key_offset: int,
    sweeps: int,
    beta: mx.array,
    clamp_mask: mx.array,
    clamp_values: mx.array,
) -> tuple[mx.array, int]:
    for _ in range(sweeps):
        for block in model._block_indices:
            state = _update_block(
                state,
                model._fields,
                model._couplings,
                block,
                keys[key_offset],
                beta,
                clamp_mask,
                clamp_values,
            )
            key_offset += 1
    return state, key_offset


def sample(
    key: mx.array,
    model: Ising,
    schedule: SamplingSchedule,
    *,
    chains: int = 1,
    initial: mx.array | None = None,
    clamp: Clamp | None = None,
) -> mx.array:
    """Draw batched block-Gibbs states shaped ``(chains, samples, spins)``."""

    _validate_key(key)
    _validate_chains(chains)
    if not isinstance(model, Ising):
        raise TypeError("model must be an Ising instance")
    if not isinstance(schedule, SamplingSchedule):
        raise TypeError("schedule must be a SamplingSchedule")

    total_sweeps = schedule.warmup + (schedule.samples - 1) * schedule.sweeps_per_sample
    keys = mx.random.split(key, 1 + total_sweeps * len(model.blocks))
    state = _prepare_initial(initial, chains=chains, n_spins=model.n_spins, key=keys[0])
    clamp_mask, clamp_values = _prepare_clamp(
        clamp,
        chains=chains,
        n_spins=model.n_spins,
    )
    state = mx.where(clamp_mask, clamp_values, state)
    beta = mx.array(model._beta, dtype=model._fields.dtype)
    state, key_offset = _run_sweeps(
        state,
        model,
        keys,
        1,
        schedule.warmup,
        beta,
        clamp_mask,
        clamp_values,
    )
    records = [mx.array(state)]

    for _ in range(1, schedule.samples):
        state, key_offset = _run_sweeps(
            state,
            model,
            keys,
            key_offset,
            schedule.sweeps_per_sample,
            beta,
            clamp_mask,
            clamp_values,
        )
        records.append(mx.array(state))

    return mx.stack(records, axis=1)
