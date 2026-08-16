import json
from importlib import import_module
from importlib.metadata import version

import mlx.core as mx
import pytest

from benchmarks import dense_sampling
from benchmarks.contract import BenchmarkConfig, expanded_couplings, measure, workload


def test_dense_benchmark_reports_installed_mlx_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    dense_sampling.main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["mlx_version"] == version("mlx")
    assert result["cold_recorded_samples_per_second"] > 0
    assert result["warm_recorded_samples_per_second"] > 0


def test_primary_workload_expands_every_bipartite_edge_without_intrablock_edges() -> None:
    model = workload()
    couplings = expanded_couplings(model)

    assert model.fields.shape == (256,)
    assert model.fields.dtype.name == "float32"
    assert model.edge_weights.shape == (128, 128)
    assert model.blocks == (tuple(range(128)), tuple(range(128, 256)))
    assert couplings.shape == (256, 256)
    assert couplings.dtype.name == "float32"
    assert (couplings[:128, :128] == 0).all()
    assert (couplings[128:, 128:] == 0).all()
    assert (couplings[:128, 128:] == model.edge_weights).all()
    assert (couplings[128:, :128] == model.edge_weights.T).all()


def test_primary_config_defines_a_fixed_32k_recorded_state_work_unit() -> None:
    config = BenchmarkConfig()

    assert config.chains * config.samples == 32_768
    assert config.warmup + (config.samples - 1) * config.sweeps_per_sample == 51
    assert config.warm_repetitions == 7


def test_thrmlx_runner_materializes_one_complete_boolean_trace() -> None:
    pytest.importorskip("thrml")
    runner_module = import_module("benchmarks.thrmlx_runner")
    model = workload(n_visible=3, n_latent=2)
    config = BenchmarkConfig(chains=8, warmup=2, samples=3, sweeps_per_sample=1, warm_repetitions=2)

    trace = runner_module.make_runner(model, config)(0)

    assert trace.shape == (8, 3, 5)
    assert trace.dtype == mx.bool_


def test_thrml_runner_materializes_one_complete_boolean_trace() -> None:
    pytest.importorskip("thrml")
    runner_module = import_module("benchmarks.thrml_runner")
    model = workload(n_visible=3, n_latent=2)
    config = BenchmarkConfig(chains=8, warmup=2, samples=3, sweeps_per_sample=1, warm_repetitions=2)

    trace = runner_module.make_runner(model, config)(0)

    assert trace.shape == (8, 3, 5)
    assert str(trace.dtype) == "bool"


def test_measure_keeps_cold_and_warm_timings_separate() -> None:
    calls: list[object] = []
    events: list[str] = []
    ticks = iter([10.0, 15.0, 20.0, 21.0, 30.0, 33.0])

    def make_runner() -> object:
        calls.append("factory")
        events.append("factory")

        def run(seed: int) -> None:
            calls.append(seed)
            events.append(f"run:{seed}")

        return run

    def clock() -> float:
        events.append("clock")
        return next(ticks)

    timing = measure(
        make_runner,
        cold_seed=0,
        warmup_seed=1,
        warm_seeds=(2, 3),
        clock=clock,
    )

    assert calls == ["factory", 0, "factory", 1, 2, 3]
    assert events == [
        "clock",
        "factory",
        "run:0",
        "clock",
        "factory",
        "run:1",
        "clock",
        "run:2",
        "clock",
        "clock",
        "run:3",
        "clock",
    ]
    assert timing.cold_elapsed_seconds == 5.0
    assert timing.warm_elapsed_seconds == (1.0, 3.0)
    assert timing.warm_median_elapsed_seconds == 2.0


def test_smoke_report_names_both_adapters_and_their_execution_devices(
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("thrml")
    run_module = import_module("benchmarks.run")

    run_module.main(["--smoke"])

    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == 1
    assert set(report["adapters"]) == {"thrmlx", "thrml"}
    assert report["adapters"]["thrmlx"]["device"] == "gpu"
    assert report["adapters"]["thrml"]["device"] == "cpu"
    assert report["adapters"]["thrmlx"]["timing"]["warm_elapsed_seconds"]
    assert report["adapters"]["thrml"]["timing"]["warm_elapsed_seconds"]
