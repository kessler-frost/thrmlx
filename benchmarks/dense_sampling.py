import json
import platform
import time
from importlib.metadata import version

import mlx.core as mx

from thrmlx import Ising, SamplingSchedule, sample

N_SPINS = 64
CHAINS = 512
WARMUP = 10
SAMPLES = 10
SWEEPS_PER_SAMPLE = 1
COUPLING = 0.2


def ring_couplings(n_spins: int, coupling: float) -> mx.array:
    return mx.array(
        [
            [
                coupling if column in {(row - 1) % n_spins, (row + 1) % n_spins} else 0.0
                for column in range(n_spins)
            ]
            for row in range(n_spins)
        ]
    )


def main() -> None:
    blocks = (tuple(range(0, N_SPINS, 2)), tuple(range(1, N_SPINS, 2)))
    model = Ising(mx.zeros((N_SPINS,)), ring_couplings(N_SPINS, COUPLING), blocks=blocks)
    schedule = SamplingSchedule(
        warmup=WARMUP,
        samples=SAMPLES,
        sweeps_per_sample=SWEEPS_PER_SAMPLE,
    )
    warm_trace = sample(mx.random.key(0), model, SamplingSchedule(warmup=1), chains=32)
    mx.eval(warm_trace)

    started = time.perf_counter()
    trace = sample(mx.random.key(1), model, schedule, chains=CHAINS)
    mx.eval(trace)
    elapsed_seconds = time.perf_counter() - started

    result = {
        "block_sizes": [len(block) for block in model.blocks],
        "chains": CHAINS,
        "device": str(mx.default_device()),
        "elapsed_seconds": elapsed_seconds,
        "mlx_version": version("mlx"),
        "n_spins": N_SPINS,
        "platform": platform.platform(),
        "recorded_samples": CHAINS * SAMPLES,
        "recorded_samples_per_second": CHAINS * SAMPLES / elapsed_seconds,
        "samples": SAMPLES,
        "sweeps_per_sample": SWEEPS_PER_SAMPLE,
        "warmup": WARMUP,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
