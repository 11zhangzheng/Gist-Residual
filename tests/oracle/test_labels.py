from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from fidmem.agent.answerer import FrozenAnswerer
from fidmem.oracle.labels import (
    AnswerStabilityAudit,
    BeamAuditCase,
    BeamSearchAudit,
    CostNormalization,
    OraclePilotAudit,
    PilotQuestionTiming,
    PilotTimingAudit,
    StabilitySample,
    answer_stability_audit,
    compare_beam_to_exhaustive,
    fit_train_cost_normalization,
    preference_labels,
    summarize_pilot_timings,
    sufficiency_label,
)
from fidmem.oracle.search import OraclePath, canonical_oracle
from fidmem.types import (
    ActionInstance,
    ActionType,
    EvidenceItem,
    FidelityLevel,
    RouterState,
)


def _path(*, score: float, cost: float, correct: bool, answer: str = "A") -> OraclePath:
    return OraclePath(
        transitions=(),
        answer=answer,
        answer_score=score,
        correct=correct,
        total_cost=cost,
        utility=score,
    )


def _state(
    *, evidence: tuple[EvidenceItem, ...] = (), history: tuple[ActionInstance, ...] = ()
) -> RouterState:
    return RouterState(
        question="Q",
        options=("A", "B"),
        evidence=evidence,
        action_history=history,
        remaining_budget=10,
        candidate_event_ids=(),
        candidate_fidelity_levels={},
        context_frontiers={},
        cost_preference=0.3,
    )


def test_canonical_prefers_lowest_cost_correct_path_and_falls_back_by_score_then_cost() -> (
    None
):
    assert (
        canonical_oracle(
            (
                _path(score=0.95, cost=6, correct=True),
                _path(score=0.8, cost=3, correct=True),
            )
        ).total_cost
        == 3
    )

    fallback = canonical_oracle(
        (
            _path(score=0.6, cost=1, correct=False, answer="B"),
            _path(score=0.8, cost=5, correct=False, answer="B"),
            _path(score=0.8, cost=2, correct=False, answer="B"),
        )
    )
    assert (fallback.answer_score, fallback.total_cost) == (0.8, 2)


def test_preference_labels_use_only_manifested_train_normalization_and_keep_all_tied_optima(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "run-manifest.json"
    normalization = fit_train_cost_normalization(
        (1.0, 3.0, 2.0), split="train", run_manifest=manifest_path
    )
    paths = (
        _path(score=0.4, cost=1, correct=False, answer="B"),
        _path(score=0.9, cost=3, correct=True),
        _path(score=0.9, cost=3, correct=True),
    )

    labels = preference_labels(paths, normalization)

    assert tuple(label.cost_preference for label in labels) == (0.0, 0.1, 0.3, 1.0)
    assert all(path.answer_score == 0.9 for path in labels[0].optimal_paths)
    assert labels[-1].optimal_paths[0].total_cost == 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["oracle_cost_normalization"] == {
        "constant": 3.0,
        "method": "max_train_total_cost",
        "sample_count": 3,
        "source_split": "train",
    }

    with pytest.raises(ValueError, match="train"):
        fit_train_cost_normalization((99.0,), split="dev", run_manifest={})


def test_sufficiency_runs_the_same_frozen_stop_answer_for_any_state_origin() -> None:
    prompts: list[str] = []
    answerer = FrozenAnswerer(
        lambda prompt: prompts.append(prompt)
        or ("A" if '"content":"enough"' in prompt else "B")
    )
    evidence = (
        EvidenceItem(
            event_id="e1",
            fidelity_level=FidelityLevel.RESIDUAL,
            content="enough",
            score=1,
        ),
    )
    failed_origin = _state(
        evidence=evidence,
        history=(ActionInstance(ActionType.VERIFY_VISUAL, "e1", "low"),),
    )
    empty = _state()

    assert sufficiency_label(failed_origin, answerer, gold_answer="A") == 1
    assert sufficiency_label(empty, answerer, gold_answer="A") == 0
    assert len(prompts) == 2
    assert all(
        "Question:" in prompt and "Evidence:" in prompt and "Answer:" in prompt
        for prompt in prompts
    )


def test_pilot_beam_and_stability_audits_have_fixed_reviewable_shapes() -> None:
    timing = summarize_pilot_timings(
        tuple(
            PilotQuestionTiming(question_id=f"q{i:03d}", a800_gpu_seconds=float(i + 1))
            for i in range(100)
        )
    )
    beam = compare_beam_to_exhaustive(
        (
            BeamAuditCase(
                question_id="q0",
                beam_action_signature=("A",),
                exhaustive_action_signature=("A",),
                beam_cost=3,
                exhaustive_cost=3,
            ),
            BeamAuditCase(
                question_id="q1",
                beam_action_signature=("B",),
                exhaustive_action_signature=("C",),
                beam_cost=5,
                exhaustive_cost=3,
            ),
        )
    )
    stability = answer_stability_audit(
        tuple(
            StabilitySample(
                state_id=f"s{i:03d}",
                answers=("A", "B", "A") if i == 0 else ("A", "A", "A"),
            )
            for i in range(100)
        )
    )
    report = OraclePilotAudit(timing=timing, beam=beam, stability=stability)

    assert (
        report.timing.question_count,
        report.timing.mean_a800_gpu_seconds,
        report.timing.p90_a800_gpu_seconds,
    ) == (100, 50.5, 90.0)
    assert (
        report.beam.case_count,
        report.beam.path_hit_rate,
        report.beam.mean_cost_gap,
    ) == (2, 0.5, 1.0)
    assert (
        report.stability.state_count,
        report.stability.repeats_per_state,
        report.stability.answer_flip_rate,
    ) == (100, 3, 0.01)


def test_audit_models_reject_directly_forged_summaries() -> None:
    timings = tuple(
        PilotQuestionTiming(question_id=f"q{i:03d}", a800_gpu_seconds=1)
        for i in range(100)
    )
    with pytest.raises(ValidationError, match="mean_a800_gpu_seconds"):
        PilotTimingAudit(
            question_count=100,
            mean_a800_gpu_seconds=999,
            p90_a800_gpu_seconds=999,
            per_question=timings,
        )

    beam_cases = (
        BeamAuditCase(
            question_id="q0",
            beam_action_signature=("A",),
            exhaustive_action_signature=("A",),
            beam_cost=3,
            exhaustive_cost=3,
        ),
    )
    with pytest.raises(ValidationError, match="case_count"):
        BeamSearchAudit(
            case_count=99,
            path_hit_rate=0,
            mean_cost_gap=999,
            cases=beam_cases,
        )

    samples = tuple(
        StabilitySample(state_id=f"s{i:03d}", answers=("A", "B", "A"))
        for i in range(100)
    )
    with pytest.raises(ValidationError, match="flipped_state_count"):
        AnswerStabilityAudit(
            state_count=100,
            repeats_per_state=3,
            flipped_state_count=0,
            answer_flip_rate=0,
            samples=samples,
        )


def test_oracle_pilot_json_rejects_nested_raw_summary_mismatches() -> None:
    payload = {
        "timing": {
            "question_count": 100,
            "mean_a800_gpu_seconds": 999,
            "p90_a800_gpu_seconds": 999,
            "per_question": [
                {"question_id": f"q{i:03d}", "a800_gpu_seconds": 1} for i in range(100)
            ],
        },
        "beam": {
            "case_count": 99,
            "path_hit_rate": 0,
            "mean_cost_gap": 999,
            "cases": [
                {
                    "question_id": "q0",
                    "beam_action_signature": ["A"],
                    "exhaustive_action_signature": ["A"],
                    "beam_cost": 3,
                    "exhaustive_cost": 3,
                }
            ],
        },
        "stability": {
            "state_count": 100,
            "repeats_per_state": 3,
            "flipped_state_count": 0,
            "answer_flip_rate": 0,
            "samples": [
                {"state_id": f"s{i:03d}", "answers": ["A", "B", "A"]}
                for i in range(100)
            ],
        },
    }

    with pytest.raises(ValidationError):
        OraclePilotAudit.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    "payload",
    (
        {"constant": 1, "sample_count": 1},
        {"constant": 1, "sample_count": 1, "source_split": "dev"},
        {"constant": float("nan"), "sample_count": 1, "source_split": "train"},
        {"constant": 1, "sample_count": 0, "source_split": "train"},
    ),
)
def test_cost_normalization_json_rejects_untrusted_provenance_or_scale(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        CostNormalization.model_validate_json(json.dumps(payload))


def test_oracle_pilot_config_preregisters_counts_search_bounds_and_device() -> None:
    config = OmegaConf.to_container(
        OmegaConf.load("configs/experiment/oracle_pilot.yaml"), resolve=True
    )
    assert config == {
        "oracle_pilot": {
            "question_count": 100,
            "device": "A800",
            "beam_size": 8,
            "max_depth": 5,
            "exhaustive_subset_size": 20,
            "stability_state_count": 100,
            "stability_repeats": 3,
            "flip_rate_threshold": 0.02,
            "cost_preferences": [0.0, 0.1, 0.3, 1.0],
        }
    }
