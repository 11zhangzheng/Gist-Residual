"""Static engineering validation for setup wrappers on hosts without Bash."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_all_setup_wrappers_are_thin_bash_entrypoints() -> None:
    setup = ROOT / "scripts/setup"
    expected = {
        "01_resolve_stack_assets.sh": "fidmem.assets.cli resolve",
        "02_download_models.sh": "fidmem.assets.cli download",
        "03_verify_models.sh": "fidmem.assets.cli verify",
        "04_download_videomme_v2_metadata.sh": "fidmem.assets.cli download",
        "05_verify_videomme_v2_metadata.sh": "fidmem.assets.setup metadata",
        "06_prepare_videomme_v2_videos.sh": "fidmem.assets.setup videos",
        "07_build_videomme_v2_manifests.sh": "fidmem.assets.setup manifests",
        "08_build_authority_draft.sh": "fidmem.assets.setup authority-draft",
    }
    assert {path.name for path in setup.glob("*.sh")} == set(expected)
    for name, command in expected.items():
        text = (setup / name).read_text(encoding="utf-8")
        assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
        assert 'export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"' in text
        assert command in text
        assert '"$@"' in text
