"""Engineering-evidence-only tests for Experiment Stack v1."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from fidmem.assets.resolver import AssetState, load_asset_lock
from fidmem.assets.stack import ExperimentStack, PhysicalAsset, load_experiment_stack


ROOT = Path(__file__).resolve().parents[2]


def test_approved_stack_deduplicates_shared_snapshots() -> None:
    stack = load_experiment_stack(
        ROOT / "configs/experiment_stacks/gist_residual_v1.yaml"
    )
    assert stack.status == "CANDIDATE_ASSETS_UNVERIFIED"
    assert (
        stack.logical_roles["gist_text_encoder"]
        == stack.logical_roles["embedding_model"]
    )
    assert stack.logical_roles["residual_model"] == stack.logical_roles["visual_model"]
    assert len(stack.physical_assets) == 5


def test_videomme_source_replaces_longtvqa_without_changing_model_snapshots() -> None:
    """Changing source identity, roles, or verified model bytes must be detected."""
    stack = load_experiment_stack(
        ROOT / "configs/experiment_stacks/gist_residual_v1.yaml"
    )
    lock = load_asset_lock(
        ROOT / "configs/experiment_stacks/gist_residual_v1.assets.lock.json"
    )

    assert stack.logical_roles["source_dataset"] == "videomme_v2_metadata"
    assert stack.physical_assets["videomme_v2_metadata"].model_dump(mode="json") == {
        "repo_id": "MME-Benchmarks/Video-MME-v2",
        "repo_type": "dataset",
        "immutable_revision": "6e4bebb03202e1ddbf3d37703e560e51c5aa2d64",
        "backend": "huggingface_hub",
        "dtype": None,
        "include_files": ["README.md", "subtitle.zip", "test.parquet"],
    }
    assert "longtvqa_metadata" not in stack.physical_assets
    assert stack.logical_roles["gist_text_encoder"] == "bge_m3"
    assert stack.logical_roles["embedding_model"] == "bge_m3"
    assert stack.logical_roles["residual_model"] == "qwen3_vl_8b_instruct"
    assert stack.logical_roles["visual_model"] == "qwen3_vl_8b_instruct"
    assert stack.logical_roles["answerer"] == "qwen3_8b"

    expected_model_hashes = {
        "bge_m3": "78f38848464972447ce0a70b3c29cff2480f9586234590ace6ba3e43334f0591",
        "siglip2_so400m_patch14_384": "8e9391da171c97d0ecbf619cc0f26178e158ed569a7527a8df8a24552616f021",
        "qwen3_vl_8b_instruct": "c37cd4285c6cb9089b0ab8a8c27cce3bfedfd4b680061a5403cdb42f8b965b96",
        "qwen3_8b": "4a133fa0d16ea3fdd42e987d7aa5135f3f8562ce45c55bf478b2f777f53e6fad",
    }
    assert set(lock.physical_assets) == set(expected_model_hashes) | {
        "videomme_v2_metadata"
    }
    for asset_id, snapshot_hash in expected_model_hashes.items():
        entry = lock.physical_assets[asset_id]
        assert entry.state is AssetState.VERIFIED
        assert entry.local_snapshot_sha256 == snapshot_hash


def test_final_targets_exclude_source_dataset() -> None:
    """Video-MME-v2 must stay source data rather than a final target benchmark."""
    stack = load_experiment_stack(
        ROOT / "configs/experiment_stacks/gist_residual_v1.yaml"
    )

    assert set(stack.target_benchmarks) == {"longvideobench", "lvbench", "mlvu"}
    assert {
        benchmark.repo_id for benchmark in stack.target_benchmarks.values()
    } == {"LongVideoBench", "LVBench", "MLVU"}


@pytest.mark.parametrize(
    "revision", ["main", "master", "latest", "feature/x", "abc123"]
)
def test_stack_rejects_mutable_or_short_revision(revision: str) -> None:
    payload = load_experiment_stack(
        ROOT / "configs/experiment_stacks/gist_residual_v1.yaml"
    ).model_dump(mode="json")
    payload["physical_assets"]["bge_m3"]["immutable_revision"] = revision
    with pytest.raises(ValidationError, match="full lowercase commit SHA"):
        ExperimentStack.model_validate(payload)


def test_dataset_include_files_are_normalized_and_safe() -> None:
    asset = PhysicalAsset(
        repo_id="owner/dataset",
        repo_type="dataset",
        backend="huggingface_hub",
        include_files=("test.parquet", "README.md", "test.parquet"),
    )

    assert asset.include_files == ("README.md", "test.parquet")



@pytest.mark.parametrize(
    "unsafe_name",
    (
        "",
        "   ",
        ".",
        "./README.md",
        "data//x",
        "data/./x",
        "/README.md",
        "C:\\README.md",
        "C:README.md",
        "\\README.md",
        "data\\x",
        "../README.md",
        "data/../test.parquet",
    ),
)
def test_dataset_include_files_reject_noncanonical_paths(unsafe_name: str) -> None:
    with pytest.raises(ValidationError, match="include_files"):
        PhysicalAsset(
            repo_id="owner/dataset",
            repo_type="dataset",
            backend="huggingface_hub",
            include_files=(unsafe_name,),
        )


def test_dataset_include_files_accept_canonical_nested_posix_path() -> None:
    asset = PhysicalAsset(
        repo_id="owner/dataset",
        repo_type="dataset",
        backend="huggingface_hub",
        include_files=("videos/001.zip",),
    )

    assert asset.include_files == ("videos/001.zip",)
