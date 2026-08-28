import json
from pathlib import Path

import pytest

from fidmem.experiments.results import load_result_schemas, validate_result


def test_templates_are_explicitly_not_formal_results() -> None:
    schemas = load_result_schemas("configs/experiments/result_schemas.yaml")
    template = json.loads(
        Path("configs/experiments/result_templates/evaluation.template.json").read_text(
            encoding="utf-8"
        )
    )
    with pytest.raises(ValueError, match="template"):
        validate_result(template, schemas)


def test_null_placeholder_and_nan_are_rejected() -> None:
    schemas = load_result_schemas("configs/experiments/result_schemas.yaml")
    base = {
        "result_family": "evaluation",
        "experiment_id": "E13",
        "run_id": "run",
        "authority_sha256": "a" * 64,
        "rows": [
            {
                "dataset": "d",
                "split": "evaluation",
                "policy": "bc",
                "seed": 1,
                "cost_preference": 0.1,
                "accuracy": 0.5,
                "total_cost": 1.0,
                "gpu_seconds": 1.0,
                "wall_seconds": 1.0,
                "frames": 1,
                "visual_tokens": 1,
                "text_tokens": 1,
                "peak_memory_bytes": 1,
            }
        ],
        "provenance": {"complete": True},
    }
    validate_result(base, schemas)
    for invalid in (None, float("nan"), "RESEARCH_OWNER_DECISION_REQUIRED"):
        candidate = json.loads(json.dumps(base))
        candidate["rows"][0]["accuracy"] = invalid
        with pytest.raises(ValueError):
            validate_result(candidate, schemas)
