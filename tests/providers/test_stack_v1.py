"""Engineering-evidence-only provider contract and resume tests."""

from pathlib import Path

import pytest

from fidmem.actions.environment import ActionObservation
from fidmem.assets.resolver import initial_lock, write_asset_lock
from fidmem.assets.stack import load_experiment_stack
from fidmem.providers.stack_v1 import (
    ExecutionRequest,
    MeasuredOperation,
    ProviderExecutionResult,
    canonical_import_record,
    check_stack_assets,
    execute_batch,
)
from fidmem.providers.stack_v1_cli import canary_gate_fields
from fidmem.experiments.gates import evaluate_gate
from fidmem.types import ActionInstance, ActionType, RouterState


ROOT = Path(__file__).resolve().parents[2]


def _request() -> ExecutionRequest:
    return ExecutionRequest(
        evidence_class="engineering",
        question_id="fixture-question",
        video_id="fixture-video",
        model_role="residual_model",
        model_id="fixture-model",
        model_revision="a" * 40,
        config_sha256="b" * 64,
        state=RouterState(
            question="fixture?",
            options=("A", "B"),
            evidence=(),
            action_history=(),
            remaining_budget=10,
            candidate_event_ids=("event-1",),
            candidate_fidelity_levels={"event-1": "GIST"},
            context_frontiers={"event-1": (0, 0)},
            cost_preference=0.5,
        ),
        action=ActionInstance(ActionType.EXPAND_RESIDUAL, "event-1", None),
        input_payload={"fixture": "engineering evidence only"},
    )


class FixtureBackend:
    def __init__(self) -> None:
        self.checked = 0
        self.executed = 0

    def check(self, request: ExecutionRequest) -> None:
        assert request.input_payload["fixture"] == "engineering evidence only"
        self.checked += 1

    def execute(self, request: ExecutionRequest) -> ProviderExecutionResult:
        self.executed += 1
        return ProviderExecutionResult(
            request_key=request.request_key,
            provider="engineering-fixture",
            device_name="cpu-fixture",
            decode_config={"fixture": True},
            raw_response={"fixture": "not production"},
            observation=ActionObservation(
                action_type=ActionType.EXPAND_RESIDUAL,
                target_event_id="event-1",
            ),
            measured_operations=(
                MeasuredOperation(
                    measurement_source="runtime-measured",
                    scope="residual",
                    amortizable=True,
                    cache_status="miss",
                    operation="fixture-residual",
                    gpu_seconds=0,
                    wall_seconds=0.001,
                    input_frames=1,
                    visual_tokens=0,
                    text_tokens=1,
                    peak_memory_bytes=0,
                    device_name="cpu-fixture",
                ),
            ),
        )


def test_provider_result_is_existing_importer_schema() -> None:
    request = _request()
    record = canonical_import_record(request, FixtureBackend().execute(request))
    assert record.evidence_class == "engineering"
    assert record.cost_records[0].wall_seconds == 0.001
    assert record.raw_response is None


def test_check_never_invokes_model_and_resume_skips_complete_request(
    tmp_path: Path,
) -> None:
    backend = FixtureBackend()
    request = _request()
    checked = execute_batch(
        (request,), backend=backend, output_dir=tmp_path, resume=False, check_only=True
    )
    assert checked["status"] == "CHECK_PASSED"
    assert backend.executed == 0
    execute_batch(
        (request,), backend=backend, output_dir=tmp_path, resume=False, check_only=False
    )
    assert backend.executed == 1
    resumed = execute_batch(
        (request,), backend=backend, output_dir=tmp_path, resume=True, check_only=False
    )
    assert resumed["resume_hits"] == 1
    assert backend.executed == 1


def test_unverified_stack_fails_closed_before_e03(tmp_path: Path) -> None:
    lock_path = tmp_path / "gist_residual_v1.assets.lock.json"
    write_asset_lock(
        lock_path,
        initial_lock(
            load_experiment_stack(
                ROOT / "configs/experiment_stacks/gist_residual_v1.yaml"
            )
        ),
    )
    with pytest.raises(ValueError, match="asset .* is not VERIFIED"):
        check_stack_assets(
            stack_path=ROOT / "configs/experiment_stacks/gist_residual_v1.yaml",
            lock_path=lock_path,
            authority_path=None,
        )


def test_canary_validation_adapter_satisfies_existing_gate_contract() -> None:
    validation = {
        "production_provenance_valid": True,
        "production_namespace_isolated": True,
        "cost_reconciliation_passed": True,
        "cross_question_cache_isolation_valid": True,
        "provider_model_device_identity_consistent": True,
        "missing_observation_count": 0,
        "unexpected_observation_question_count": 0,
        "duplicate_collision_count": 0,
    }
    fields = canary_gate_fields(validation, resume_passed=True)
    status, checks = evaluate_gate("production_canary", fields, {})
    assert status == "PASS"
    assert all(checks.values())
