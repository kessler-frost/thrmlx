"""Compact MLX fixture translating THRML's end-to-end MNIST training objective."""

import mlx.core as mx

from thrmlx.training import train_mnist_fixture


def test_upstream_testtrainmnist_train_mnist() -> None:
    """Catch an MNIST-shaped visible-data training fixture that cannot classify its labels."""

    result = train_mnist_fixture(mx.random.key(20260816))

    assert result.accuracy >= 0.9
    assert result.bias_gradient_shape == (786,)
    assert result.weight_gradient_shape == (2,)
