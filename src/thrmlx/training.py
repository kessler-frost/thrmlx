"""Small end-to-end training fixtures for the MLX THRML compatibility surface."""

from __future__ import annotations

from dataclasses import dataclass

# MLX 0.32 ships its native extension without typing metadata.
import mlx.core as mx  # ty: ignore[unresolved-import]

from thrmlx.block_management import Block
from thrmlx.block_sampling import sample_states
from thrmlx.models.ising import (
    IsingEBM,
    IsingSamplingProgram,
    IsingTrainingSpec,
    estimate_kl_grad,
    hinton_init,
)
from thrmlx.pgm import SpinNode
from thrmlx.schedule import SamplingSchedule


@dataclass(frozen=True, slots=True)
class MNISTFixtureResult:
    """Summary of a compact 28-by-28 binary-image contrastive-training fixture."""

    accuracy: float
    bias_gradient_shape: tuple[int, ...]
    weight_gradient_shape: tuple[int, ...]


def _fixture_data() -> tuple[mx.array, mx.array]:
    images = mx.zeros((8, 28 * 28), dtype=mx.bool_)
    images[4:, 0] = True
    labels = mx.zeros((8, 2), dtype=mx.bool_)
    labels[:4, 0] = True
    labels[4:, 1] = True
    return images, labels


def train_mnist_fixture(key: mx.array) -> MNISTFixtureResult:
    """Run one contrastive update on an MNIST-shaped binary image/label fixture."""

    images, labels = _fixture_data()
    image_nodes = [SpinNode() for _ in range(28 * 28)]
    label_nodes = [SpinNode(), SpinNode()]
    nodes = [*image_nodes, *label_nodes]
    edges = [(image_nodes[0], label_nodes[0]), (image_nodes[0], label_nodes[1])]
    model = IsingEBM(
        nodes,
        edges,
        mx.zeros((len(nodes),), dtype=mx.float32),
        mx.array([-0.8, 0.8], dtype=mx.float32),
        mx.array(1.0, dtype=mx.float32),
    )
    image_block = Block(image_nodes)
    label_block = Block(label_nodes)
    negative_blocks = [image_block, label_block]
    training = IsingTrainingSpec(
        model,
        [Block(nodes)],
        [],
        [],
        negative_blocks,
        SamplingSchedule(warmup=0, samples=1, sweeps_per_sample=0),
        SamplingSchedule(warmup=4, samples=16, sweeps_per_sample=1),
    )
    initialization_key, gradient_key, classification_key = mx.random.split(key, 3)
    negative_initial = hinton_init(initialization_key, model, negative_blocks, (32,))
    gradient_weights, gradient_biases, _, _ = estimate_kl_grad(
        gradient_key,
        training,
        nodes,
        edges,
        [mx.concatenate([images, labels], axis=-1)],
        [],
        [],
        negative_initial,
    )
    updated_model = IsingEBM(
        nodes,
        edges,
        model.biases - 0.02 * gradient_biases,
        model.weights - 0.02 * gradient_weights,
        model.beta,
    )
    classifier = IsingSamplingProgram(updated_model, [label_block], [image_block])
    classification_keys = mx.random.split(classification_key, images.shape[0] * 2).reshape(
        (images.shape[0], 2, 2)
    )
    predictions: list[int] = []
    for image, keys in zip(images, classification_keys, strict=True):
        initial = hinton_init(keys[0], updated_model, [label_block], ())
        trace = sample_states(
            keys[1],
            classifier,
            SamplingSchedule(warmup=20, samples=100, sweeps_per_sample=1),
            initial,
            [image],
            [label_block],
        )[0]
        if not isinstance(trace, mx.array):
            raise TypeError("MNIST fixture label sampling must return an MLX array")
        scores = mx.mean(2 * trace.astype(mx.float32) - 1, axis=0)
        predictions.append(int(mx.argmax(scores).item()))
    targets = [int(mx.argmax(label).item()) for label in labels]
    accuracy = sum(
        prediction == target for prediction, target in zip(predictions, targets, strict=True)
    ) / len(targets)
    return MNISTFixtureResult(
        accuracy,
        gradient_biases.shape,
        gradient_weights.shape,
    )
