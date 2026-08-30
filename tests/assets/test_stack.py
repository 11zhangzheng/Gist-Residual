"""Engineering-evidence-only tests for Experiment Stack v1."""

from pathlib import Path

import pytest
from pydantic import ValidationError

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
