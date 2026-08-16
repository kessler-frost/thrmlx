import json
from importlib.metadata import version

import pytest

from benchmarks import dense_sampling


def test_dense_benchmark_reports_installed_mlx_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    dense_sampling.main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["mlx_version"] == version("mlx")
    assert result["cold_recorded_samples_per_second"] > 0
    assert result["warm_recorded_samples_per_second"] > 0
