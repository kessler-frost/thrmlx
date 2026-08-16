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


@mx.compile
def _update_block(
    state: mx.array,
    fields: mx.array,
    couplings: mx.array,
    block: mx.array,
    key: mx.array,
    beta: mx.array,
) -> mx.array:
    signed = 2 * state.astype(fields.dtype) - 1
    local_fields = fields + signed @ couplings
    probabilities = mx.sigmoid(2 * beta * local_fields[:, block])
    draws = mx.random.bernoulli(probabilities, key=key)
    updated = mx.array(state)
    updated[:, block] = draws
    return updated


def _run_sweeps(
    state: mx.array,
    model: Ising,
    keys: mx.array,
    key_offset: int,
    sweeps: int,
    beta: mx.array,
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
    if clamp is not None:
        raise NotImplementedError("clamping is implemented in the next v0.1 increment")

    total_sweeps = schedule.warmup + (schedule.samples - 1) * schedule.sweeps_per_sample
    keys = mx.random.split(key, 1 + total_sweeps * len(model.blocks))
    state = _prepare_initial(initial, chains=chains, n_spins=model.n_spins, key=keys[0])
    beta = mx.array(model._beta, dtype=model._fields.dtype)
    state, key_offset = _run_sweeps(state, model, keys, 1, schedule.warmup, beta)
    records = [mx.array(state)]

    for _ in range(1, schedule.samples):
        state, key_offset = _run_sweeps(
            state,
            model,
            keys,
            key_offset,
            schedule.sweeps_per_sample,
            beta,
        )
        records.append(mx.array(state))

    return mx.stack(records, axis=1)
