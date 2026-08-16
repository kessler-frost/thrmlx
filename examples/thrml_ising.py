"""The source-style THRML Ising quick start, running on MLX."""

import mlx.core as mx

from thrmlx import Block, SamplingSchedule, SpinNode, sample_states
from thrmlx.models import IsingEBM, IsingSamplingProgram, hinton_init

nodes = [SpinNode() for _ in range(5)]
edges = [(nodes[index], nodes[index + 1]) for index in range(4)]
model = IsingEBM(
    nodes,
    edges,
    mx.zeros((5,), dtype=mx.float32),
    mx.full((4,), 0.5, dtype=mx.float32),
    mx.array(1.0, dtype=mx.float32),
)
free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
initial_key, sampling_key = mx.random.split(mx.random.key(0), 2)
initial_state = hinton_init(initial_key, model, free_blocks, ())
samples = sample_states(
    sampling_key,
    program,
    SamplingSchedule(warmup=100, samples=1_000, sweeps_per_sample=2),
    initial_state,
    [],
    [Block(nodes)],
)[0]

if __name__ == "__main__":
    print(samples.shape)
