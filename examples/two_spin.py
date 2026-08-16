import mlx.core as mx

from thrmlx import Clamp, Ising, SamplingSchedule, sample


def main() -> None:
    model = Ising(
        fields=mx.array([0.0, 0.0]),
        couplings=mx.array([[0.0, 0.8], [0.8, 0.0]]),
    )
    clamp = Clamp(
        mask=mx.array([False, False]),
        values=mx.array([False, False]),
    )
    trace = sample(
        mx.random.key(7),
        model,
        SamplingSchedule(warmup=200, samples=4),
        chains=4_096,
        clamp=clamp,
    )
    signed = 2 * trace[:, -1].astype(mx.float32) - 1
    correlation = mx.mean(signed[:, 0] * signed[:, 1]).item()
    print(f"trace shape: {trace.shape}")
    print(f"final spin correlation: {correlation:.3f}")


if __name__ == "__main__":
    main()
