# THRML MLX Generic Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Port THRML's generic block-Gibbs program surface to MLX and translate the six upstream block-sampling objectives.

**Architecture:** BlockGibbsSpec extends the verified static BlockSpec with ordered free superblocks and clamped blocks. InteractionGroup lowers directed head/tail dependencies once into BlockSamplingProgram; sample_single_block only gathers static slices and invokes one user conditional sampler. Sampling uses explicit MLX keys, Python-owned state containers, and the existing SamplingSchedule semantics.

**Tech Stack:** Python 3.10+, MLX 0.32, uv, pytest, Ruff, ty; pinned upstream THRML v0.1.4 commit 9c4e6fbb800f5e5c627122e668ff1b158ef3782b.

**Spec:** docs/superpowers/specs/2026-08-16-thrml-mlx-fork-design.md

## Global Constraints

- Keep thrmlx as the package name; production code depends on MLX only.
- Keep state templates to ArraySpec leaves, tuples, and dictionaries with string keys.
- User-defined conditional samplers receive ordinary Python lists plus MLX arrays; they never receive JAX, PyTrees, or hidden global RNG state.
- A sweep reads one pre-superblock global state; all blocks in that superblock sample from it, then their updates become visible together.
- Interaction trees accept MLX arrays, tuples, and dictionaries with string keys. Every array has a leading axis equal to the InteractionGroup head length.
- A green objective requires a translated test with the exact manifest port identifier; set only the six block-sampling entries green.
- No new benchmark number until a generic sampler example runs through the same translated test path.

---

## File Structure

| File | Responsibility |
| --- | --- |
| src/thrmlx/conditional_samplers.py | Abstract conditional sampler contract. |
| src/thrmlx/interaction.py | Directed InteractionGroup validation. |
| src/thrmlx/block_sampling.py | Gibbs specification, program lowering, explicit-key sweep, and state recording. |
| src/thrmlx/__init__.py | Generic sampling exports. |
| tests/upstream_parity/test_block_sampling.py | Six translated upstream block-sampling objectives. |
| examples/generic_block_sampling.py | A runnable signed-neighbor THRML-style sampling program. |
| tests/upstream_parity/manifest.json | The six verified status transitions. |

## Task 1: Define generic sampler and interaction boundaries

**Files:**
- Create: src/thrmlx/conditional_samplers.py, src/thrmlx/interaction.py
- Test: tests/upstream_parity/test_block_sampling.py

**Interfaces:**

~~~python
class AbstractConditionalSampler:
    def init(self) -> object: ...
    def sample(
        self,
        key: mx.array,
        interactions: list[Interaction],
        active_flags: list[mx.array],
        states: list[list[State]],
        sampler_state: object,
        output_spec: StateSpec,
    ) -> tuple[State, object]: ...


class InteractionGroup:
    def __init__(
        self,
        interaction: Interaction,
        head_nodes: Block[AbstractNode],
        tail_nodes: Sequence[Block[AbstractNode]],
    ) -> None: ...
~~~

- [ ] **Step 1: Write the failing interaction test**

~~~python
def test_interaction_group_rejects_tail_length_that_differs_from_its_heads() -> None:
    with pytest.raises(ValueError, match="same length"):
        InteractionGroup(mx.ones((2,)), Block([SpinNode(), SpinNode()]), [Block([SpinNode()])])
~~~

The production mutation is removing the head/tail alignment validation.

- [ ] **Step 2: Run the test to verify it fails**

Run: uv run pytest tests/upstream_parity/test_block_sampling.py -k interaction_group -v

Expected: FAIL because InteractionGroup does not exist.

- [ ] **Step 3: Implement the small boundaries**

AbstractConditionalSampler raises NotImplementedError from sample and returns None from init. InteractionGroup validates tail block lengths and recursively validates every interaction array leading dimension. It stores no derived execution plan.

- [ ] **Step 4: Verify green**

Run: uv run pytest tests/upstream_parity/test_block_sampling.py -k interaction_group -v

Expected: PASS.

## Task 2: Lower a generic Gibbs program and execute one block

**Files:**
- Create: src/thrmlx/block_sampling.py
- Modify: src/thrmlx/__init__.py
- Modify: tests/upstream_parity/test_block_sampling.py

**Interfaces:**

~~~python
class BlockGibbsSpec(BlockSpec):
    def __init__(
        self,
        free_superblocks: Sequence[Block[AbstractNode] | Sequence[Block[AbstractNode]]],
        clamped_blocks: Sequence[Block[AbstractNode]],
        node_shape_dtypes: Mapping[type[AbstractNode], StateSpec] = DEFAULT_NODE_SHAPE_DTYPES,
    ) -> None: ...


class BlockSamplingProgram:
    def __init__(
        self,
        gibbs_spec: BlockGibbsSpec,
        samplers: Sequence[AbstractConditionalSampler],
        interaction_groups: Sequence[InteractionGroup],
    ) -> None: ...


def sample_single_block(
    key: mx.array,
    state_free: Sequence[State],
    state_clamp: Sequence[State],
    program: BlockSamplingProgram,
    block: int,
    sampler_state: object,
) -> tuple[State, object]: ...
~~~

- [ ] **Step 1: Write the failing signed-neighbor test**

~~~python
def test_upstream_testplusminus_sample_block() -> None:
    program, free_state, clamp_state = make_signed_neighbor_program()
    updated, _ = sample_single_block(mx.random.key(7), free_state, clamp_state, program, 1, None)
    assert updated.tolist() == pytest.approx([1.75, 4.5])
~~~

Use literal inputs and the hand-calculated values. This catches a missing active mask, a reversed tail lookup, or a superblock update read from a mutated global state.

- [ ] **Step 2: Run the test to verify it fails**

Run: uv run pytest tests/upstream_parity/test_block_sampling.py::test_upstream_testplusminus_sample_block -v

Expected: FAIL because the generic program does not exist.

- [ ] **Step 3: Implement static lowering and single-block execution**

BlockSamplingProgram checks sampler count, precomputes head occurrence indices, active masks, interaction slices, tail state bucket indices, and tail positions. sample_single_block packs free plus clamped state only when needed, gathers every tail state with mx.take at the node axis, and calls the selected sampler with the block type template.

- [ ] **Step 4: Verify green**

Run: uv run pytest tests/upstream_parity/test_block_sampling.py::test_upstream_testplusminus_sample_block -v

Expected: PASS.

## Task 3: Execute full sweeps and record generic states

**Files:**
- Modify: src/thrmlx/block_sampling.py
- Modify: tests/upstream_parity/test_block_sampling.py
- Create: examples/generic_block_sampling.py

**Interfaces:**

~~~python
def sample_blocks(
    key: mx.array,
    state_free: Sequence[State],
    state_clamp: Sequence[State],
    program: BlockSamplingProgram,
    sampler_states: Sequence[object],
) -> tuple[list[State], list[object]]: ...


def sample_states(
    key: mx.array,
    program: BlockSamplingProgram,
    schedule: SamplingSchedule,
    state_free: Sequence[State],
    state_clamp: Sequence[State],
    nodes_to_sample: Sequence[Block[AbstractNode]],
) -> list[State]: ...
~~~

- [ ] **Step 1: Write failing sweep, schedule, validation, and nested-state tests**

~~~python
def test_upstream_testplusminus_sample_blocks() -> None:
    updated, _ = sample_blocks(mx.random.key(8), free_state, clamp_state, program, [None, None])
    assert updated[0].tolist() == pytest.approx([2.0])


def test_upstream_testsamplervalidation_mismatched_sampler_list_raises() -> None:
    with pytest.raises(ValueError, match="Expected 2 samplers"):
        BlockSamplingProgram(spec, [sampler], [])
~~~

Also translate test_sample_states, test_state_gaurdrailing, and test_pytree_state using only tuple/dict state templates. Each test asserts a real numerical state outcome, not that a sampler mock was called.

- [ ] **Step 2: Run the five tests to verify they fail**

Run: uv run pytest tests/upstream_parity/test_block_sampling.py -k 'sample_blocks or sample_states or gaurdrailing or mismatched or pytree' -v

Expected: FAIL because sweep and recording functions are absent.

- [ ] **Step 3: Implement sweep and recording behavior**

Validate free and clamped state before a sweep. Split one supplied MLX key once per free block. Update superblock members from their shared pre-superblock packed state, then materialize all updates. Apply warmup sweeps, record post-warmup state, and apply steps_per_sample sweeps before each later recording. Stack recorded state at leading sample axis.

- [ ] **Step 4: Verify all six translated objectives and example**

Run:

~~~bash
uv run pytest tests/upstream_parity/test_block_sampling.py -v
uv run python examples/generic_block_sampling.py
~~~

Expected: all six mapped tests pass and the example prints a three-sample floating-point trace.

- [ ] **Step 5: Mark only verified objectives green and commit**

Set the six tests/test_block_sampling.py entries to green and leave 40 planned.

~~~bash
git add src/thrmlx/conditional_samplers.py src/thrmlx/interaction.py src/thrmlx/block_sampling.py src/thrmlx/__init__.py tests/upstream_parity/test_block_sampling.py tests/upstream_parity/manifest.json examples/generic_block_sampling.py
git commit -m "feat: port generic THRML block sampling"
~~~

## Task 4: Verify and publish the generic sampling milestone

**Files:**
- Modify: README.md, PROJECT_LOG.md, .github/workflows/ci.yml

- [ ] **Step 1: Update current coverage and example coverage**

Change README to 20 / 60 translated objectives green. Add the generic block-sampling example to Apple Silicon CI and record the transition in PROJECT_LOG.md. Do not add a comparison number in this task.

- [ ] **Step 2: Run quality gates**

Run serial then parallel:

~~~bash
uv run pytest -q
uv run pytest -n auto -q
uv run ruff format --check .
uv run ruff check .
uv run ty check src
uv build
~~~

Expected: serial and parallel have identical counts; all other gates pass.

- [ ] **Step 3: Commit and push**

~~~bash
git add README.md PROJECT_LOG.md .github/workflows/ci.yml
git commit -m "docs: publish generic sampling parity status"
git push origin main
~~~

## Self-Review

- **Spec coverage:** The plan ports the next dependency-complete upstream module: generic program construction and all six block-sampling objectives. It retains explicit key semantics and dynamic MLX array execution. Factors, observers, Ising learning, MNIST, and benchmark measurements remain separately scoped.
- **Placeholder scan:** Every public type, translated objective, validation condition, expected numeric contract, and quality command is identified in the task that produces it.
- **Type consistency:** InteractionGroup feeds BlockSamplingProgram; BlockGibbsSpec and StateSpec come from the existing block foundation; sample_states uses the existing SamplingSchedule.

## Execution Record

Completed on 2026-08-16 in commits following the plan: all six translated
block-sampling objectives passed, documentation reported 20 / 60 green, and
serial/parallel pytest both reported 122 passed.
