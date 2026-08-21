from pathlib import Path

import pytest
from pydantic import ValidationError

from fidmem.config import AppConfig, load_config


def test_load_config_exposes_the_approved_resource_budget() -> None:
    config = load_config(Path("configs/base.yaml"))

    assert config.retrieval.top_k == 5
    assert config.oracle.max_depth == 5
    assert config.oracle.beam_size == 8
    assert config.visual.low_frames == 12
    assert config.visual.high_frames == 32
    assert config.budget.a800_gpu_hours == 800
    assert config.budget.v100_gpu_hours == 200


def test_app_config_rejects_high_frame_count_below_low_frame_count() -> None:
    with pytest.raises(ValidationError, match="high_frames"):
        AppConfig.model_validate(
            {
                "retrieval": {"top_k": 5},
                "oracle": {"max_depth": 5, "beam_size": 8},
                "visual": {"low_frames": 32, "high_frames": 12},
                "budget": {"a800_gpu_hours": 800, "v100_gpu_hours": 200},
            }
        )


@pytest.mark.parametrize(
    ("budget_field", "excessive_hours"),
    (("a800_gpu_hours", 801), ("v100_gpu_hours", 201)),
)
def test_app_config_rejects_gpu_budget_above_hardware_cap(
    budget_field: str, excessive_hours: int
) -> None:
    budget = {"a800_gpu_hours": 800, "v100_gpu_hours": 200}
    budget[budget_field] = excessive_hours

    with pytest.raises(ValidationError, match=budget_field):
        AppConfig.model_validate(
            {
                "retrieval": {"top_k": 5},
                "oracle": {"max_depth": 5, "beam_size": 8},
                "visual": {"low_frames": 12, "high_frames": 32},
                "budget": budget,
            }
        )
