"""Engineering-evidence-only E01-E03 Experiment Stack wiring checks."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from fidmem.experiments.execution_pack import (
    CheckFailure,
    load_experiment_config,
    load_registry,
    resolve_environment_references,
)


ROOT = Path(__file__).resolve().parents[2]


def test_e01_uses_existing_runner_with_longtvqa_setup_command() -> None:
    config = load_experiment_config(ROOT / "configs/experiments/e01_dataset.yaml")
    assert config["execution"]["command"] == [
        "python",
        "-m",
        "fidmem.assets.setup",
        "e01",
    ]
    assert config["inputs"]["stack_config"].endswith("gist_residual_v1.yaml")


def test_e02_uses_existing_authority_schema_setup_command() -> None:
    config = load_experiment_config(ROOT / "configs/experiments/e02_authority.yaml")
    assert config["execution"]["command"] == [
        "python",
        "-m",
        "fidmem.assets.setup",
        "e02",
    ]
    assert config["inputs"]["authority_draft"] == "env:FIDMEM_AUTHORITY_DRAFT"
    e02 = load_registry(ROOT / "configs/experiments/registry.yaml").experiment("E02")
    assert e02.dependencies == ("E01",)
    assert e02.required_gates == ("environment_ready", "dataset_frozen")


def test_e03_points_to_stack_contract_without_changing_canary_size() -> None:
    config = load_experiment_config(ROOT / "configs/experiments/e03_canary.yaml")
    assert config["execution"]["command"] == [
        "python",
        "-m",
        "fidmem.providers.stack_v1_cli",
        "run",
    ]
    assert config["inputs"]["question_count_min"] == 10
    assert config["inputs"]["question_count_max"] == 20
    assert (
        config["inputs"]["provider_backend_factory"]
        == "env:FIDMEM_PROVIDER_BACKEND_FACTORY"
    )


def test_videomme_pilot_policy_is_video_disjoint_and_question_independent() -> None:
    """Changing deterministic pilot selection or group sizes must fail this contract."""
    policy = OmegaConf.to_container(
        OmegaConf.load(
            ROOT / "configs/experiment_stacks/videomme_v2_pilot_split_policy.yaml"
        ),
        resolve=True,
    )

    assert policy == {
        "schema_version": 1,
        "split_policy_id": "videomme-v2-pilot-split-v1",
        "dataset_scope": "PARTIAL_DATASET_PILOT",
        "split_unit": "video_id",
        "pool": {
            "video_count": 45,
            "seed": "videomme-v2-partial-pilot-pool-v1",
            "algorithm": "videomme-v2-archive-aware-hash-v1",
            "question_independent": True,
        },
        "split": {
            "seed": "videomme-v2-partial-pilot-split-v1",
            "algorithm": "videomme-v2-partial-pilot-split-v1",
            "video_groups": {
                "oracle": 25,
                "canary": 4,
                "holdout": 4,
                "development": 12,
            },
        },
    }


def test_environment_references_are_resolved_before_run_hashing(monkeypatch) -> None:
    monkeypatch.setenv("FIDMEM_GIT_COMMIT", "a" * 40)
    assert resolve_environment_references(
        {"source": {"git_commit": "env:FIDMEM_GIT_COMMIT"}}
    ) == {"source": {"git_commit": "a" * 40}}
    monkeypatch.delenv("FIDMEM_GIT_COMMIT")
    with pytest.raises(CheckFailure, match="FIDMEM_GIT_COMMIT"):
        resolve_environment_references("env:FIDMEM_GIT_COMMIT")
