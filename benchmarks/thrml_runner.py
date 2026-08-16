"""Upstream THRML/JAX adapter for the shared local sampling benchmark."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

from benchmarks.contract import BenchmarkConfig, BenchmarkWorkload


def make_runner(model: BenchmarkWorkload, config: BenchmarkConfig) -> Callable[[int], jax.Array]:
    """Create a materializing upstream THRML request for one workload/configuration pair."""

    nodes = [SpinNode() for _ in range(model.n_spins)]
    visible_nodes = nodes[: model.n_visible]
    latent_nodes = nodes[model.n_visible :]
    edges = [(visible, latent) for visible in visible_nodes for latent in latent_nodes]
    blocks = [Block(visible_nodes), Block(latent_nodes)]
    full_state = Block(nodes)
    ising = IsingEBM(
        nodes,
        edges,
        jnp.array(model.fields),
        jnp.array(model.edge_weights.reshape(-1)),
        jnp.array(model.beta, dtype=jnp.float32),
    )
    program = IsingSamplingProgram(ising, blocks, [])
    schedule = SamplingSchedule(config.warmup, config.samples, config.sweeps_per_sample)
    expected_shape = (config.chains, config.samples, model.n_spins)

    def sample_one(initial: list[jax.Array], key: jax.Array) -> jax.Array:
        return sample_states(key, program, schedule, initial, [], [full_state])[0]

    sample_batch = jax.jit(jax.vmap(sample_one, in_axes=(0, 0)))

    def run(seed: int) -> jax.Array:
        _, initialization_key, sampling_key = jax.random.split(jax.random.key(seed), 3)
        initial = hinton_init(initialization_key, ising, blocks, (config.chains,))
        trace = jax.block_until_ready(
            sample_batch(initial, jax.random.split(sampling_key, config.chains))
        )
        assert trace.dtype == jnp.bool_
        assert trace.shape == expected_shape
        return trace

    return run
