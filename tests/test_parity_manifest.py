"""Compatibility-ledger behavior."""

import asyncio
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "tests" / "upstream_parity" / "manifest.json"
EXPECTED_UPSTREAM_IDS = {
    "tests/test_block_management.py::TestBlocks::test_empty_state",
    "tests/test_block_management.py::TestBlocks::test_node_lookup",
    "tests/test_block_management.py::TestBlocks::test_shape_transforms",
    "tests/test_block_management.py::TestBlockCompat::test_bad_block",
    "tests/test_block_management.py::TestBlockCompat::test_bad_dtype",
    "tests/test_block_management.py::TestBlockCompat::test_bad_shape",
    "tests/test_block_management.py::TestBlockCompat::test_bad_structure",
    "tests/test_block_management.py::TestBlockCompat::test_good",
    "tests/test_block_management.py::TestBlockCompat::test_good_state",
    "tests/test_block_management.py::TestBlockCompat::test_length_mismatch",
    "tests/test_block_management.py::TestBlockCompat::test_missing_array",
    "tests/test_block_management.py::TestBlockCompat::test_wrong_state_len",
    "tests/test_block_management.py::TestDuplicate::test_duplicate",
    "tests/test_block_management.py::TestDuplicate::test_good",
    "tests/test_block_sampling.py::TestPlusMinus::test_sample_block",
    "tests/test_block_sampling.py::TestPlusMinus::test_sample_blocks",
    "tests/test_block_sampling.py::TestPlusMinus::test_sample_states",
    "tests/test_block_sampling.py::TestPlusMinus::test_state_gaurdrailing",
    "tests/test_block_sampling.py::TestSamplerValidation::test_mismatched_sampler_list_raises",
    "tests/test_block_sampling.py::TestPyTreeState::test_pytree_state",
    "tests/test_discrete_ebm.py::TestFactor::test_duplicated_type",
    "tests/test_discrete_ebm.py::TestFactor::test_good",
    "tests/test_discrete_ebm.py::TestFactor::test_wrong_n_cat",
    "tests/test_discrete_ebm.py::TestSamplerType::test_bad_bin",
    "tests/test_discrete_ebm.py::TestSamplerType::test_bad_cat",
    "tests/test_discrete_ebm.py::TestSamplerType::test_good",
    "tests/test_discrete_ebm.py::TestSquare::test_bad",
    "tests/test_discrete_ebm.py::TestSquare::test_good",
    "tests/test_discrete_ebm.py::TestSampling::test_binary",
    "tests/test_discrete_ebm.py::TestSampling::test_categorical",
    "tests/test_discrete_ebm.py::TestSampling::test_mixed",
    "tests/test_discrete_ebm.py::TestInteractions::test_to_interactions",
    "tests/test_discrete_ebm.py::TestInteractions::test_to_interactions_binary",
    "tests/test_discrete_ebm.py::TestBlockSample::test_binary_bias",
    "tests/test_discrete_ebm.py::TestBlockSample::test_categorical_bias",
    "tests/test_discrete_ebm.py::TestBlockSample::test_categorical_triplet",
    "tests/test_discrete_ebm.py::TestBlockSample::test_categorical_uint8_rejects_too_many_categories",
    "tests/test_discrete_ebm.py::TestBlockSample::test_ragged_mixed",
    "tests/test_discrete_ebm.py::TestEnergy::test_bin",
    "tests/test_discrete_ebm.py::TestEnergy::test_cat",
    "tests/test_discrete_ebm.py::TestEnergy::test_mixed",
    "tests/test_discrete_ebm.py::TestEquivalence::test_equivalence",
    "tests/test_discrete_ebm.py::TestHeteroGrid::test_grid",
    "tests/test_discrete_ebm.py::TestBigGrid::test_big",
    "tests/test_factor.py::TestFactorCreate::test_empty",
    "tests/test_factor.py::TestFactorCreate::test_good",
    "tests/test_factor.py::TestFactorCreate::test_ragged",
    "tests/test_factor.py::TestWeighted::test_bad",
    "tests/test_factor.py::TestWeighted::test_good",
    "tests/test_interaction.py::TestInteractionInputs::test_bad_interaction",
    "tests/test_interaction.py::TestInteractionInputs::test_bad_tail",
    "tests/test_interaction.py::TestInteractionInputs::test_good",
    "tests/test_ising.py::TestLine::test_sample",
    "tests/test_ising.py::TestMomentAccumulator::test_first_moments",
    "tests/test_ising.py::TestMomentAccumulator::test_second_moments",
    "tests/test_ising.py::TestEstimateKLGrad::test_estimate_kl_grad",
    "tests/test_ising.py::TestEstimateKLGradFullyVisible::test_fully_visible_ising",
    "tests/test_observers.py::TestMomentObserver::test_preserves_mixed_node_values",
    "tests/test_readme.py::test_readme_quick_example",
    "tests/test_train_mnist.py::TestTrainMnist::test_train_mnist",
}


def test_manifest_tracks_every_upstream_test_objective() -> None:
    """Fail if a future port silently omits an upstream test objective."""

    payload = json.loads(MANIFEST_PATH.read_text())

    assert payload["upstream"]["commit"] == "9c4e6fbb800f5e5c627122e668ff1b158ef3782b"
    assert len(payload["tests"]) == 60
    assert {entry["id"] for entry in payload["tests"]} == EXPECTED_UPSTREAM_IDS


def test_parity_report_summarizes_the_committed_ledger() -> None:
    """Fail if the report stops exposing incomplete parity to its caller."""

    async def run_report() -> tuple[int, str, str]:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "tools/parity_report.py",
            cwd=REPOSITORY_ROOT,
            stderr=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return process.returncode, stdout.decode(), stderr.decode()

    returncode, stdout, stderr = asyncio.run(run_report())

    assert returncode == 0, stderr
    assert json.loads(stdout) == {
        "complete": False,
        "green": 52,
        "planned": 8,
        "total": 60,
    }
