from __future__ import annotations

from pathlib import Path

import pytest
import torch

from fidmem.eval.baselines import (
    BCPolicyAdapter,
    BaselinePolicyError,
    DAggerPolicyAdapter,
    FullResidualPolicy,
    GistOnlyPolicy,
    GistResidualPolicy,
    GistVisualPolicy,
    QuestionOnlyPolicy,
    RulePolicy,
    TextAdaptivePolicy,
    UniformFramesPolicy,
)
from fidmem.router.dagger import ArtifactReference, BCPolicy, PolicyIdentity
from fidmem.router.dagger_workflow import _build_manifest
from fidmem.router.dataset import TestByteTokenizer
from fidmem.router.model import EncoderIdentity, MemoryRouter, RouterModelConfig
from fidmem.types import (
    ActionInstance,
    ActionType,
    EvidenceItem,
    FidelityLevel,
    RouterState,
)


def _state(
    *,
    candidates: tuple[str, ...] = (),
    evidence: tuple[EvidenceItem, ...] = (),
    history: tuple[ActionInstance, ...] = (),
) -> RouterState:
    return RouterState(
        question="Which option is supported?",
        options=("A", "B"),
        evidence=evidence,
        action_history=history,
        remaining_budget=100.0,
        candidate_event_ids=candidates,
        candidate_fidelity_levels={
            event_id: FidelityLevel.GIST for event_id in candidates
        },
        context_frontiers={event_id: (0, 0) for event_id in candidates},
        cost_preference=0.1,
    )


def _action(
    action_type: ActionType,
    event_id: str | None = None,
    visual_budget: str | None = None,
) -> ActionInstance:
    return ActionInstance(action_type, event_id, visual_budget)  # type: ignore[arg-type]


def test_fixed_and_rule_policies_return_exact_stable_mask_members() -> None:
    search = _action(ActionType.SEARCH_GIST)
    stop = _action(ActionType.STOP)
    assert GistOnlyPolicy()(_state(), (search, stop)) is search

    residual_two = _action(ActionType.EXPAND_RESIDUAL, "e2")
    residual_one = _action(ActionType.EXPAND_RESIDUAL, "e1")
    visual_low = _action(ActionType.VERIFY_VISUAL, "e1", "low")
    visual_high = _action(ActionType.VERIFY_VISUAL, "e1", "high")
    context = _action(ActionType.EXPAND_CONTEXT, "e1")
    legal = (
        residual_two,
        visual_high,
        stop,
        residual_one,
        context,
        visual_low,
    )
    state = _state(candidates=("e1", "e2"))

    assert GistOnlyPolicy()(state, legal) is stop
    assert GistResidualPolicy()(state, legal) is residual_one
    assert GistVisualPolicy()(state, legal) is visual_high
    assert UniformFramesPolicy(("e1", "e2"))(state, legal) is visual_low
    assert FullResidualPolicy(("e1", "e2"))(state, legal) is residual_one
    assert RulePolicy(sufficiency_threshold=0.9)(state, legal) is residual_one


def test_fixed_policy_fails_closed_without_preference_or_stop() -> None:
    residual = _action(ActionType.EXPAND_RESIDUAL, "e1")
    with pytest.raises(BaselinePolicyError, match="STOP is not legal"):
        GistVisualPolicy()(_state(candidates=("e1",)), (residual,))


def test_restricted_controllers_never_receive_action_or_candidate_sets() -> None:
    stop = _action(ActionType.STOP)
    visual = EvidenceItem(
        event_id="e1",
        fidelity_level=FidelityLevel.VISUAL,
        content="SECRET MULTIMODAL",
        score=1.0,
        attachments=("secret.png",),
    )
    gist = EvidenceItem(
        event_id="e1",
        fidelity_level=FidelityLevel.GIST,
        content="public gist",
        score=0.2,
    )
    searched = _action(ActionType.SEARCH_GIST)
    state = _state(candidates=("e1",), evidence=(visual, gist), history=(searched,))

    def question_controller(view):
        assert vars(view) == {
            "question": state.question,
            "options": state.options,
            "remaining_budget": state.remaining_budget,
            "cost_preference": state.cost_preference,
        }
        return ActionType.STOP

    def text_controller(view):
        assert view.gist_text == ("public gist",)
        assert view.action_history == (ActionType.SEARCH_GIST,)
        assert "SECRET" not in repr(view)
        assert "secret.png" not in repr(view)
        assert not hasattr(view, "candidate_event_ids")
        return ActionType.STOP

    assert QuestionOnlyPolicy(question_controller)(state, (stop,)) is stop
    assert TextAdaptivePolicy(text_controller)(state, (stop,)) is stop
    unavailable = QuestionOnlyPolicy(lambda _view: ActionType.VERIFY_VISUAL)
    assert unavailable(state, (stop,)) is stop


def _bc_policy() -> tuple[MemoryRouter, BCPolicy]:
    identity = EncoderIdentity.test_identity("eval-bc")
    model = MemoryRouter(
        RouterModelConfig(
            encoder=identity,
            encoder_output_dim=8,
            hidden_dim=12,
            action_type_embedding_dim=4,
            fidelity_embedding_dim=2,
            max_question_tokens=64,
            max_item_tokens=32,
        )
    )
    return model, BCPolicy(model, tokenizer=TestByteTokenizer("eval-bc"))


def _checkpoint(model: MemoryRouter, path: Path) -> Path:
    torch.save({"model": model.state_dict()}, path)
    return path


def _mismatched_manifest(*, checkpoint: Path, actual: PolicyIdentity):
    wrong = actual.model_copy(update={"behavior_sha256": "f" * 64})
    checkpoint_ref = ArtifactReference(
        path="checkpoint.pt", sha256=actual.checkpoint_sha256
    )
    return _build_manifest(
        run_identity="1" * 64,
        round_number=1,
        generation="round-0001",
        previous_generation_id=None,
        previous_generation_manifest_sha256=None,
        manifest_path="manifest.json",
        status="stopped",
        stop_reason="max_rounds",
        source_policy=ArtifactReference(
            path="source.pt", sha256=actual.checkpoint_sha256
        ),
        source_policy_identity=actual,
        checkpoint=checkpoint_ref,
        checkpoint_policy_identity=wrong,
        seen_keys=ArtifactReference(path="seen.json", sha256="2" * 64),
        dev_artifact=ArtifactReference(path="dev.json", sha256="3" * 64),
        deviation_artifact=ArtifactReference(path="deviations.json", sha256="4" * 64),
        base_dataset_identity="5" * 64,
        aggregated_dataset_identity="6" * 64,
        train_subset_question_ids=("q1",),
        train_subset_sha256="7" * 64,
        context_identities=("8" * 64,),
        seen_key_count=0,
        deviation_count=0,
        new_deviation_count=0,
        thresholds={"utility_gain": 0.005, "regret_improvement_ratio": 0.02},
        metrics={"dev_utility": 1.0, "cost_regret": 0.0},
        budget_bin_width=1.0,
    )


def test_bc_and_dagger_adapters_cross_check_actual_policy_checkpoint(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="actual Task 10 BCPolicy"):
        BCPolicyAdapter(lambda *_args: None, checkpoint=tmp_path / "fake.pt")

    model, policy = _bc_policy()
    checkpoint = _checkpoint(model, tmp_path / "checkpoint.pt")
    adapter = BCPolicyAdapter(policy, checkpoint=checkpoint)
    stop = _action(ActionType.STOP)
    assert adapter(_state(), (stop,)) is stop

    other_model, other_policy = _bc_policy()
    with pytest.raises(ValueError, match="model state"):
        BCPolicyAdapter(other_policy, checkpoint=checkpoint)

    manifest = _mismatched_manifest(
        checkpoint=checkpoint, actual=adapter.policy_identity
    )
    with pytest.raises(ValueError, match="differs from manifest"):
        DAggerPolicyAdapter(
            policy,
            checkpoint=checkpoint,
            manifest=manifest,
        )
