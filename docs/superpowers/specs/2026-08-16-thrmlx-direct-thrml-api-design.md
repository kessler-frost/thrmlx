# Direct THRML API Compatibility Design

## Outcome

`thrmlx` remains both the distribution and import name. A user installs the public Git repository
with `uv add "thrmlx @ git+https://github.com/kessler-frost/thrmlx.git"` and changes only their
root module import from `thrml` to `thrmlx`. The complete pinned THRML 0.1.4 public API is then
available from `thrmlx` and `thrmlx.models`, implemented by MLX.

No `thrml` package is shipped. That avoids a collision with upstream THRML and keeps the MLX backend
explicit in every environment.

## Contract

- `thrmlx` re-exports every upstream root public object and the `models` module, while retaining
  additive MLX-native `Ising`, `Clamp`, and `sample` helpers.
- `thrmlx.models` re-exports every upstream model object, including `AbstractFactorizedEBM`.
- Existing module paths are preserved under the renamed root: for example,
  `thrmlx.block_sampling`, `thrmlx.models.discrete_ebm`, and `thrmlx.models.ising`.
- Public function keyword names follow THRML 0.1.4 where they differed from the first MLX port:
  `free_super_blocks`, `init_state_free`, `init_chain_state`, `f_observe`, `clamp_state`,
  `sampler_state`, `spec_from`, and `blocks_to_extract`.
- `SamplingSchedule` accepts upstream names (`n_warmup`, `n_samples`, `steps_per_sample`) and
  established MLX names (`warmup`, `samples`, `sweeps_per_sample`), exposes both attribute triplets,
  and rejects duplicate aliases in one call.

## Backend boundary

This is source API compatibility under the `thrmlx` root. Arrays and random keys must be MLX values;
JAX arrays, Equinox PyTrees, `jax.jit`, `jax.vmap`, `jax.grad`, and JAX random-bitstream identity are
not emulated. The unchanged domain program is run with `mlx.core` instead of JAX array construction.

## Proof

Tests use a literal inventory copied from the pinned upstream public exports, import every direct
`thrmlx` module path, check normal keyword invocation, and run a one-spin MLX Gibbs program through
the upstream schedule vocabulary. This tests the consumer contract independently of the 60-object
semantic translation ledger, which remains the behavioral oracle.
