# thrmlx agent guidance

## Direction

- North star: maximize measured local Apple-Silicon sampling throughput without weakening
  distributional correctness against exact enumeration.
- Next deliverable: a correct, documented v0.1 dense Ising sampler. Keep the default path thin;
  add sparse storage, custom kernels, compatibility layers, and training only after a measured need.
- Iterate through change, evaluate, compare, and record. Put durable decisions in `PROJECT_LOG.md`.

## Behavioral contract

- Treat `docs/superpowers/specs/2026-08-16-thrmlx-v0-design.md` as the v0.1 source of truth.
- Preserve boolean spins (`False = -1`, `True = +1`), reduced energy including `beta`, ordered
  color-block sweeps, THRML schedule timing, immutable clamps, unsqueezed output axes, and explicit
  MLX random keys. Never use MLX's implicit global random state in library code.
- THRML v0.1.4 commit `9c4e6fbb800f5e5c627122e668ff1b158ef3782b` is a behavioral oracle, not
  an implementation template. Keep the code original and the project clearly unofficial.

## Engineering loop

- Implement behavior red-green-refactor. Derive expected values independently, use exact
  enumeration for small models, and add focused tests at stable seams or for reproduced bugs.
- Run focused tests while iterating. Run the complete serial suite, Ruff, ty, build, and package
  smoke checks at merge/release checkpoints. Adopt parallel pytest only after serial and parallel
  pass/fail/error counts are identical.
- Use uv for environments, dependencies, commands, and builds. Use Ruff and ty. Keep imports at
  file tops and use `pathlib.Path` for file or directory paths.
- Keep conditionals and exception handling minimal. Prefer one validated path over parallel modes.
- Do not add Python threads, multiprocessing, `asyncio.to_thread`, or thread-backed async file I/O.
  Use core asyncio only if genuinely asynchronous work enters the project.
- Pair every setup/install command or script with a scoped teardown/uninstall counterpart.

## Coordination

- Default to one agent. Use bounded Terra or Luna subagents only when independent work materially
  shortens the critical path, and split producer/consumer responsibilities when contract
  falsification matters.
- Self-review internal specs and plans, then continue. Ask the user only when a missing choice
  materially changes the result, needs new authority, or creates external impact.
