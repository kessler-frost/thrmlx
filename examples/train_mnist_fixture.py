"""Run the compact MNIST-shaped contrastive-training compatibility fixture."""

import mlx.core as mx

from thrmlx.training import train_mnist_fixture

if __name__ == "__main__":
    print(train_mnist_fixture(mx.random.key(20260816)))
