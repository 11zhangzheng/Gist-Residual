"""Engineering-evidence-only Authority Draft asset gate tests."""

from pathlib import Path

import pytest

from fidmem.assets.authority_draft import build_authority_draft


ROOT = Path(__file__).resolve().parents[2]


def test_authority_draft_rejects_unverified_checked_in_assets(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not VERIFIED"):
        build_authority_draft(
            project_root=ROOT,
            asset_lock_path=ROOT
            / "configs/experiment_stacks/gist_residual_v1.assets.lock.json",
            manifests_root=tmp_path,
            split_policy_path=ROOT
            / "configs/experiment_stacks/longtvqa_split_policy.yaml",
            prompt_config_path=ROOT
            / "configs/experiment_stacks/gist_residual_v1.prompts.yaml",
            observation_config_path=ROOT
            / "configs/experiment_stacks/gist_residual_v1.observation_configs.yaml",
            evidence_root=tmp_path / "evidence",
        )
