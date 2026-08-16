"""Run the reproducible MLX-versus-THRML local sampling benchmark."""

import argparse
import json
import platform
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import jax
import mlx.core as mx

from benchmarks.contract import BenchmarkConfig, BenchmarkWorkload, Timing, measure, workload
from benchmarks.thrml_runner import make_runner as make_thrml_runner
from benchmarks.thrmlx_runner import make_runner as make_thrmlx_runner

THRML_COMMIT = "9c4e6fbb800f5e5c627122e668ff1b158ef3782b"


def _timing_report(timing: Timing, config: BenchmarkConfig) -> dict[str, float | list[float]]:
    recorded_states = config.chains * config.samples
    return {
        "cold_elapsed_seconds": timing.cold_elapsed_seconds,
        "cold_recorded_states_per_second": recorded_states / timing.cold_elapsed_seconds,
        "warm_elapsed_seconds": list(timing.warm_elapsed_seconds),
        "warm_median_elapsed_seconds": timing.warm_median_elapsed_seconds,
        "warm_recorded_states_per_second": recorded_states / timing.warm_median_elapsed_seconds,
    }


def _adapter_report(
    make_runner: Callable[[BenchmarkWorkload, BenchmarkConfig], Callable[[int], object]],
    model: BenchmarkWorkload,
    config: BenchmarkConfig,
    *,
    device: str,
) -> dict[str, object]:
    timing = measure(
        lambda: make_runner(model, config),
        cold_seed=0,
        warmup_seed=1,
        warm_seeds=tuple(range(2, 2 + config.warm_repetitions)),
        clock=time.perf_counter,
    )
    return {"device": device, "timing": _timing_report(timing, config)}


def _report(model: BenchmarkWorkload, config: BenchmarkConfig) -> dict[str, object]:
    return {
        "adapters": {
            "thrmlx": _adapter_report(
                make_thrmlx_runner,
                model,
                config,
                device=mx.default_device().type.name,
            ),
            "thrml": _adapter_report(
                make_thrml_runner,
                model,
                config,
                device=jax.default_backend(),
            ),
        },
        "comparison_note": (
            "This is an Apple-Silicon local-use comparison: MLX runs on Metal GPU while upstream "
            "THRML runs through JAX's CPU backend. It is not a same-accelerator framework "
            "benchmark."
        ),
        "host": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version,
        },
        "schema_version": 1,
        "software": {
            "jax": version("jax"),
            "jaxlib": version("jaxlib"),
            "mlx": version("mlx"),
            "thrml": version("thrml"),
            "thrml_commit": THRML_COMMIT,
        },
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "workload": {
            "beta": model.beta,
            "chains": config.chains,
            "edge_count": model.n_visible * model.n_latent,
            "n_latent": model.n_latent,
            "n_spins": model.n_spins,
            "n_visible": model.n_visible,
            "recorded_states": config.chains * config.samples,
            "samples": config.samples,
            "sweeps_per_sample": config.sweeps_per_sample,
            "warm_repetitions": config.warm_repetitions,
            "warmup": config.warmup,
        },
    }


def _smoke_config() -> BenchmarkConfig:
    return BenchmarkConfig(chains=8, warmup=2, samples=3, sweeps_per_sample=1, warm_repetitions=2)


def main(arguments: Sequence[str] | None = None) -> None:
    """Emit a full primary report or a deliberately small validation report as JSON."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    parsed = parser.parse_args(arguments)
    model = workload(3, 2) if parsed.smoke else workload()
    config = _smoke_config() if parsed.smoke else BenchmarkConfig()
    report = json.dumps(_report(model, config), indent=2, sort_keys=True)
    if parsed.output is not None:
        parsed.output.parent.mkdir(parents=True, exist_ok=True)
        parsed.output.write_text(f"{report}\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
