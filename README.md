# thrmlx

`thrmlx` is an MLX-native Python library for fast local Ising-model sampling on Apple Silicon.
The v0.1 goal is deliberately small: correct dense pairwise models, deterministic graph coloring,
batched block-Gibbs chains, explicit random keys, and optional clamping.

> [!IMPORTANT]
> `thrmlx` is an independent, unofficial project. It is not affiliated with, sponsored by, or
> endorsed by Extropic. THRML is referenced only as a behavioral and conceptual influence.

The implementation is under active development. The public v0.1 contract is documented in
[`docs/superpowers/specs/2026-08-16-thrmlx-v0-design.md`](docs/superpowers/specs/2026-08-16-thrmlx-v0-design.md).

## Development setup

Requires Python 3.10+, uv, and a platform supported by MLX. The primary target is an Apple-Silicon
Mac running macOS 14 or newer.

```bash
uv sync --all-groups
```

Remove the generated environment and project-local caches with the paired teardown command:

```bash
./scripts/teardown.sh
```

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
