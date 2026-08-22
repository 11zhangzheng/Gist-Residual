from __future__ import annotations

import pytest
import torch
from fidmem.router.dataset import OracleBCRecord, RouterCollator
from tests.router._fixtures import authoritative_record
from fidmem.router.model import (
    EncoderIdentity,
    MemoryRouter,
    RouterModelConfig,
    TestTextEncoder,
)
from fidmem.types import (
    ActionInstance,
    ActionType,
    EvidenceItem,
    FidelityLevel,
    RouterState,
)


def _record(
    index: int,
    *,
    with_candidates: bool = True,
    question: str | None = None,
) -> OracleBCRecord:
    evidence = (
        (
            EvidenceItem(
                event_id=f"e{index}",
                fidelity_level=FidelityLevel.RESIDUAL,
                content="blue bottle behind carton",
                score=0.9,
                acquisition_step=1,
            ),
        )
        if with_candidates
        else ()
    )
    candidates = (f"e{index}",) if with_candidates else ()
    state = RouterState(
        question=question or f"What color is bottle {index}?",
        options=("blue", "red"),
        evidence=evidence,
        action_history=(ActionInstance(ActionType.SEARCH_GIST, None, None),),
        remaining_budget=9.0,
        candidate_event_ids=candidates,
        candidate_fidelity_levels=(
            {f"e{index}": FidelityLevel.RESIDUAL} if with_candidates else {}
        ),
        context_frontiers=(
            {f"e{index}": (index % 3, (index + 1) % 3)} if with_candidates else {}
        ),
        cost_preference=0.3,
    )
    actions = (
        (
            ActionInstance(ActionType.EXPAND_RESIDUAL, f"e{index}", None),
            ActionInstance(ActionType.VERIFY_VISUAL, f"e{index}", "low"),
            ActionInstance(ActionType.STOP, None, None),
        )
        if with_candidates
        else (
            ActionInstance(ActionType.SEARCH_GIST, None, None),
            ActionInstance(ActionType.STOP, None, None),
        )
    )
    video_id = f"v{index // 2}"
    return authoritative_record(
        state=state,
        actions=actions,
        legal_action_mask=(False, True, True) if with_candidates else (True, True),
        target_action_index=1 if with_candidates else 0,
        video_id=video_id,
        question_id=f"q{index}",
        sufficiency_target=1 if index % 2 else 0,
        cost_to_go=float(index % 4),
        observation_snapshot_id="cached-graph-sha256",
    )


def _model() -> MemoryRouter:
    identity = EncoderIdentity.test_identity("model-test-v1")
    config = RouterModelConfig(
        encoder=identity,
        encoder_output_dim=16,
        hidden_dim=24,
        action_type_embedding_dim=8,
        fidelity_embedding_dim=4,
        max_question_tokens=128,
        max_item_tokens=96,
    )
    return MemoryRouter(
        config,
        text_encoder=TestTextEncoder(identity, vocab_size=257, output_dim=16),
    )


def test_forward_scores_each_action_instance_and_hard_masks_before_softmax() -> None:
    batch = RouterCollator(max_question_bytes=128, max_item_bytes=96)(
        [_record(0), _record(1, with_candidates=False)]
    )
    model = _model().to(dtype=torch.float64)

    action_logits, sufficiency_logit, cost_to_go = model(batch)

    assert action_logits.shape == (2, 3)
    assert sufficiency_logit.shape == (2,)
    assert cost_to_go.shape == (2,)
    assert (
        action_logits.dtype
        == sufficiency_logit.dtype
        == cost_to_go.dtype
        == torch.float64
    )
    assert torch.isfinite(action_logits[batch.legal_action_mask]).all()
    assert torch.isfinite(sufficiency_logit).all()
    assert torch.isfinite(cost_to_go).all()
    minimum = torch.finfo(action_logits.dtype).min
    assert torch.equal(
        action_logits[~batch.legal_action_mask],
        torch.full_like(action_logits[~batch.legal_action_mask], minimum),
    )
    probabilities = action_logits.softmax(dim=-1)
    assert torch.equal(
        probabilities[~batch.legal_action_mask],
        torch.zeros_like(probabilities[~batch.legal_action_mask]),
    )


def test_forward_fails_closed_when_a_batch_row_has_no_legal_action() -> None:
    batch = RouterCollator(max_question_bytes=128, max_item_bytes=96)([_record(0)])
    batch.legal_action_mask[0] = False

    with pytest.raises(ValueError, match="at least one legal action"):
        _model()(batch)


def test_collator_rejects_overlong_text_instead_of_silently_truncating() -> None:
    record = _record(0, question="x" * 129)

    with pytest.raises(ValueError, match="question.*128"):
        RouterCollator(max_question_bytes=128, max_item_bytes=96)([record])
