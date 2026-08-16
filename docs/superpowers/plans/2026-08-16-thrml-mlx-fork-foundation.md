# THRML MLX Fork Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Establish thrmlx as an auditable, source-derived THRML port and implement the MLX-native node/block state foundation required by every remaining upstream use case.

**Architecture:** Keep thrmlx as the import name while adopting THRML's public domain objects. pgm.py provides identity-stable nodes and typed state templates; block_management.py owns state packing and validation behind a small MLX-array/Python-container seam. A committed parity manifest maps every upstream test objective to a translated test and keeps unsupported work visible without making a false parity claim.

**Tech Stack:** Python 3.10+, MLX 0.32, uv, pytest, Ruff, ty; upstream THRML v0.1.4 source commit 9c4e6fbb800f5e5c627122e668ff1b158ef3782b as the behavioral reference.

**Spec:** docs/superpowers/specs/2026-08-16-thrml-mlx-fork-design.md

## Global Constraints

- Keep package/import name thrmlx; do not publish to PyPI or add JAX to production dependencies.
- Preserve Apache-2.0 attribution and call the project a source-derived MLX port, not an affiliated Extropic project or a GitHub-network fork.
- Treat the upstream 60 collected test IDs as the compatibility ledger; call the suite complete only when every manifest entry is translated and green.
- Use MLX arrays at runtime; do not emulate JAX arrays, PyTrees, JIT transforms, or upstream random streams.
- State templates support one recursively nested structure built from ArraySpec, tuples, and dictionaries with string keys; reject other container types.
- Nodes compare and hash by a process-local monotonic identity. A Block contains one concrete node type and BlockSpec rejects empty blocks.
- Execute on this main checkout; the user explicitly authorized direct commit and push.

---

## File Structure

| File | Responsibility |
| --- | --- |
| UPSTREAM.md | Pinned upstream identity, attribution, divergence policy, and refresh instructions. |
| tests/upstream_parity/manifest.json | 60 upstream objective IDs with source, port test, intent, and status. |
| tools/parity_report.py | Read-only JSON status report used in CI. |
| src/thrmlx/pgm.py | Nodes, ArraySpec, and default state templates. |
| src/thrmlx/block_management.py | Blocks, static locations, allocation, conversion, and validation. |
| tests/upstream_parity/test_block_management.py | MLX translations of the 14 upstream block-management objectives. |
| examples/block_state_round_trip.py | THRML-style block-state use example. |

## Task 1: Create provenance and the parity ledger

**Files:**
- Create: UPSTREAM.md, tests/upstream_parity/__init__.py, tests/upstream_parity/manifest.json, tests/test_parity_manifest.py, tools/parity_report.py
- Modify: README.md, THIRD_PARTY_NOTICES.md

**Interfaces:**
- Produces python tools/parity_report.py, which prints total, green, planned, and complete JSON keys.

- [ ] **Step 1: Write the failing coverage test**

~~~python
def test_manifest_tracks_every_upstream_test_objective() -> None:
    payload = json.loads(MANIFEST_PATH.read_text())
    assert payload["upstream"]["commit"] == "9c4e6fbb800f5e5c627122e668ff1b158ef3782b"
    assert len(payload["tests"]) == 60
    assert {entry["id"] for entry in payload["tests"]} == EXPECTED_UPSTREAM_IDS
~~~

Use the literal 60 IDs emitted by the pinned upstream pytest --collect-only -q tests run. This fails if an upstream objective is silently omitted.

- [ ] **Step 2: Run the test and confirm the red state**

Run: uv run pytest tests/test_parity_manifest.py::test_manifest_tracks_every_upstream_test_objective -v

Expected: FAIL because the manifest does not exist.

- [ ] **Step 3: Implement the ledger and provenance**

Create this exact top-level manifest shape and one entry for every collected upstream ID:

~~~json
{
  "upstream": {
    "repository": "https://github.com/extropic-ai/thrml",
    "version": "0.1.4",
    "commit": "9c4e6fbb800f5e5c627122e668ff1b158ef3782b",
    "collected_tests": 60
  },
  "tests": [
    {
      "id": "tests/test_block_management.py::TestBlocks::test_shape_transforms",
      "port": "tests/upstream_parity/test_block_management.py::test_upstream_testblocks_shape_transforms",
      "intent": "block/global state round trip",
      "status": "planned"
    }
  ]
}
~~~

Set every status to planned initially. UPSTREAM.md identifies upstream, pin, Apache-2.0 terms, source-derived status, divergence rules, and the exact upstream collection command. README and notices call the current result an Ising-only preliminary benchmark, not all-use-case THRML parity.

- [ ] **Step 4: Implement the report and verify green**

Implement the report with Path(__file__).resolve().parents[1] and json.loads; it does not fetch or write. Initially it prints:

~~~json
{"complete": false, "green": 0, "planned": 60, "total": 60}
~~~

Run:

~~~bash
uv run pytest tests/test_parity_manifest.py -v
uv run python tools/parity_report.py
~~~

Expected: test passes and report contains the exact four fields above.

- [ ] **Step 5: Commit the ledger**

~~~bash
git add UPSTREAM.md THIRD_PARTY_NOTICES.md README.md tests/upstream_parity tests/test_parity_manifest.py tools/parity_report.py
git commit -m "docs: establish THRML compatibility ledger"
~~~

## Task 2: Add node and block compatibility objects

**Files:**
- Create: src/thrmlx/pgm.py, src/thrmlx/block_management.py, tests/upstream_parity/test_block_management.py
- Modify: src/thrmlx/__init__.py, tests/upstream_parity/manifest.json

**Interfaces:**

~~~python
@dataclass(frozen=True, slots=True)
class ArraySpec:
    shape: tuple[int, ...]
    dtype: mx.Dtype


class AbstractNode: ...


class SpinNode(AbstractNode): ...


class CategoricalNode(AbstractNode): ...


class Block:
    def __init__(self, nodes: Sequence[AbstractNode]) -> None: ...
    @property
    def node_type(self) -> type[AbstractNode]: ...
~~~

- [ ] **Step 1: Write failing tests**

~~~python
def test_upstream_testduplicate_duplicate_rejects_a_node_in_two_blocks() -> None:
    node = SpinNode()
    with pytest.raises(ValueError, match="twice"):
        BlockSpec([Block([node]), Block([node])], DEFAULT_NODE_SHAPE_DTYPES)


def test_block_rejects_mixed_concrete_node_types() -> None:
    with pytest.raises(ValueError, match="same type"):
        Block([SpinNode(), CategoricalNode()])
~~~

The tests fail after a duplicate-node check is removed or a block begins accepting mixed node types.

- [ ] **Step 2: Run the test and confirm the red state**

Run: uv run pytest tests/upstream_parity/test_block_management.py -k 'duplicate or mixed' -v

Expected: FAIL with missing THRML-compatible public objects.

- [ ] **Step 3: Implement the minimal public seam**

Implement identity-stable nodes with a module-private itertools.count(). Implement frozen, slots ArraySpec with tuple-normalized shapes. SpinNode and CategoricalNode are empty node subclasses. Block stores a tuple, supports length/iteration/indexing/membership, rejects mixed concrete types, and rejects node_type on an empty block. Export the new names while retaining Ising, Clamp, SamplingSchedule, and sample.

- [ ] **Step 4: Verify green and commit**

Run: uv run pytest tests/upstream_parity/test_block_management.py -k 'duplicate or mixed' -v

Expected: PASS.

~~~bash
git add src/thrmlx/pgm.py src/thrmlx/block_management.py src/thrmlx/__init__.py tests/upstream_parity/test_block_management.py tests/upstream_parity/manifest.json
git commit -m "feat: add THRML-compatible nodes and blocks"
~~~

## Task 3: Port static block state packing

**Files:**
- Modify: src/thrmlx/block_management.py, tests/upstream_parity/test_block_management.py, tests/upstream_parity/manifest.json
- Create: examples/block_state_round_trip.py

**Interfaces:**

~~~python
class BlockSpec:
    def __init__(
        self, blocks: Sequence[Block], node_shape_dtypes: Mapping[type[AbstractNode], StateSpec]
    ) -> None: ...


def make_empty_block_state(
    blocks: Sequence[Block],
    node_shape_dtypes: Mapping[type[AbstractNode], StateSpec],
    batch_shape: tuple[int, ...] = (),
) -> list[State]: ...
def block_state_to_global(block_state: Sequence[State], spec: BlockSpec) -> list[State]: ...
def from_global_state(
    global_state: Sequence[State], spec: BlockSpec, blocks: Sequence[Block]
) -> list[State]: ...
def get_node_locations(block: Block, spec: BlockSpec) -> tuple[int, mx.array]: ...
def verify_block_state(
    blocks: Sequence[Block],
    states: Sequence[State],
    node_shape_dtypes: Mapping[type[AbstractNode], StateSpec],
    block_axis: int | None = None,
) -> None: ...
~~~

StateSpec is recursively ArraySpec, tuple, or a dictionary with string keys; State substitutes mx.array leaves.

- [ ] **Step 1: Write the failing round-trip and dtype tests**

~~~python
def test_upstream_testblocks_shape_transforms_round_trips_nested_templates() -> None:
    blocks, templates = mixed_blocks_and_specs()
    spec = BlockSpec(blocks, templates)
    block_state = make_empty_block_state(blocks, templates, batch_shape=(2,))
    assert_state_equal(
        from_global_state(block_state_to_global(block_state, spec), spec, blocks), block_state
    )


def test_upstream_testblockcompat_bad_dtype_rejects_nested_array() -> None:
    with pytest.raises(TypeError, match="dtype"):
        verify_block_state(
            [Block([SpinNode()])], [mx.zeros((1,), dtype=mx.int32)], DEFAULT_NODE_SHAPE_DTYPES
        )
~~~

Use literal (2, 3, 2) shapes and dtypes in fixtures. These catch wrong slice placement and accidental dtype acceptance.

- [ ] **Step 2: Run the tests and confirm the red state**

Run: uv run pytest tests/upstream_parity/test_block_management.py -k 'shape_transforms or bad_dtype' -v

Expected: FAIL because packing and validation are absent.

- [ ] **Step 3: Implement one recursive state walker**

Implement private _walk_spec_and_state(spec, state) to validate container kind, tuple length, dictionary keys, MLX leaf type, shape suffix, and dtype. Reuse it in allocation and validation. BlockSpec uses first-seen template-bucket ordering, maps a node to (bucket_index, node_index), and rejects empty blocks, absent templates, and duplicate identities. Global conversion concatenates bucket data across node axis; extraction uses mx.take(..., axis=0).

- [ ] **Step 4: Verify all 14 translated block-management objectives and example**

Run:

~~~bash
uv run pytest tests/upstream_parity/test_block_management.py -v
uv run python examples/block_state_round_trip.py
~~~

Expected: all 14 mapped tests pass; the example prints (2, 3) for its packed spin shape.

- [ ] **Step 5: Mark exactly the verified upstream scope green and commit**

Set the 14 tests/test_block_management.py manifest entries to green; leave the other 46 planned.

~~~bash
git add src/thrmlx/block_management.py tests/upstream_parity/test_block_management.py tests/upstream_parity/manifest.json examples/block_state_round_trip.py
git commit -m "feat: port THRML block state management to MLX"
~~~

## Task 4: Publish partial compatibility status and verify

**Files:**
- Modify: README.md, PROJECT_LOG.md, .github/workflows/ci.yml, tests/test_parity_manifest.py

**Interfaces:**
- Consumes the Task 1 report and Task 3 manifest updates.
- Produces a CI report and public status that say 14 / 60 translated upstream objectives green; not full THRML parity.

- [ ] **Step 1: Write the failing partial-status test**

~~~python
def test_parity_report_distinguishes_foundation_coverage_from_full_parity() -> None:
    result = subprocess.run(
        [sys.executable, "tools/parity_report.py"], text=True, capture_output=True, check=True
    )
    assert json.loads(result.stdout) == {"complete": False, "green": 14, "planned": 46, "total": 60}
~~~

This fails if a partial port is reported as complete.

- [ ] **Step 2: Run the test and confirm the red state**

Run: uv run pytest tests/test_parity_manifest.py::test_parity_report_distinguishes_foundation_coverage_from_full_parity -v

Expected: FAIL because the ledger is still all planned before Task 3 finishes.

- [ ] **Step 3: Wire status to docs and CI**

After Task 3 passes, update README and PROJECT_LOG.md with source commit, coverage, retained MLX Ising adapter, and the JAX Metal incompatibility result. Add uv run python tools/parity_report.py after pytest in CI. Do not add a JAX-Metal throughput row: a JIT-enabled THRML smoke program is currently incompatible.

- [ ] **Step 4: Verify serial, parallel, and release gates**

Run serial:

~~~bash
uv run pytest -q
uv run ruff format --check .
uv run ruff check .
uv run ty check src
uv build
~~~

Then run: uv run pytest -n auto -q.

Expected: identical serial/parallel pass/fail/error counts; format, lint, types, and build pass.

- [ ] **Step 5: Commit, push, and record the revision**

~~~bash
git add README.md PROJECT_LOG.md .github/workflows/ci.yml tests/test_parity_manifest.py
git commit -m "docs: publish THRML MLX foundation status"
git push origin main
git rev-parse HEAD
~~~

## Self-Review

- **Spec coverage:** Task 1 creates provenance and every-objective accounting. Tasks 2–3 port nodes, blocks, and static packing, including all 14 upstream block-management objectives. Task 4 reports the partial state and preserves the JAX Metal policy. Generic sampling, factors, observers, Ising training, MNIST, and their benchmark rows belong to the later, independently testable milestones named in the design.
- **Placeholder scan:** The upstream commit, test counts, data schema, state container types, public interfaces, expected report, and exact test commands are fixed above.
- **Type consistency:** Task 3 defines every BlockSpec state function consumed by later tests; Task 4 consumes only the report and manifest statuses from prior tasks.
