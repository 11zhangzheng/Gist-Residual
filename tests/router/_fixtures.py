from __future__ import annotations

from fidmem.actions.environment import ActionObservation, EnvironmentTransition
from fidmem.agent.answerer import FrozenAnswerer
from fidmem.data.longroute import (
    BUILDER_VERSION,
    MANIFEST_VERSION,
    DatasetManifest,
    LongRouteExample,
    SourceManifestProvenance,
    SourceVideoProvenance,
    VirtualSegment,
)
from fidmem.oracle.labels import COST_PREFERENCES, CostNormalization, PreferenceLabel
from fidmem.oracle.search import OraclePath
from fidmem.router.dataset import (
    FrozenComponentIdentity,
    OracleBCRecord,
    materialize_oracle_record,
    seal_sufficiency_label,
)
from fidmem.types import ActionInstance, RouterState


def authoritative_record(
    *,
    state: RouterState,
    actions: tuple[ActionInstance, ...],
    legal_action_mask: tuple[bool, ...],
    target_action_index: int,
    video_id: str,
    question_id: str,
    sufficiency_target: int,
    cost_to_go: float,
    split: str = "train",
    observation_snapshot_id: str = "cached-observations-v1",
    normalization: CostNormalization | None = None,
) -> OracleBCRecord:
    """Build a genuine canonical Task8/Task9 roundtrip fixture."""

    normalization = normalization or CostNormalization(
        constant=10, sample_count=100, source_split="train"
    )
    event_ids = sorted(
        {
            event_id
            for event_id in (
                *(state.candidate_event_ids),
                *(item.event_id for item in state.evidence),
                *(action.event_id for action in (*state.action_history, *actions)),
            )
            if event_id is not None
        }
    )
    if not event_ids:
        event_ids = ["fixture-event"]
    local_ids = [event_id.removeprefix(f"{video_id}:") for event_id in event_ids]
    duration = 600 / len(local_ids)
    segments = tuple(
        VirtualSegment(
            source_video_id=video_id,
            event_id=event_id,
            source_start_sec=index * duration,
            source_end_sec=(index + 1) * duration,
            global_start_sec=index * duration,
            global_end_sec=(index + 1) * duration,
        )
        for index, event_id in enumerate(local_ids)
    )
    target_action = actions[target_action_index]
    target_local = (
        target_action.event_id.removeprefix(f"{video_id}:")
        if target_action.event_id is not None
        else local_ids[0]
    )
    target_event_id = f"{video_id}:{target_local}"
    example = LongRouteExample(
        question_id=question_id,
        split=split,
        question=state.question,
        options=state.options,
        answer=state.options[0],
        target_source_video_id=video_id,
        target_event_id=target_event_id,
        target_position=local_ids.index(target_local),
        supporting_event_ids=(target_event_id,),
        template="single_event",
        segments=segments,
        duration_sec=600,
    )
    source_hash = "1" * 64
    asset_hash = "2" * 64
    source = SourceManifestProvenance(
        identity="fixture-source",
        dataset_name="unit",
        dataset_version="v1",
        source_uri="file:///fixture-source.json",
        license="test",
        canonical_sha256=source_hash,
        videos=(
            SourceVideoProvenance(
                video_id=video_id,
                source_uri=f"file:///{video_id}.mp4",
                content_sha256=asset_hash,
            ),
        ),
    )
    manifest = DatasetManifest(
        manifest_version=MANIFEST_VERSION,
        schema_version=MANIFEST_VERSION,
        builder_version=BUILDER_VERSION,
        seed=7,
        source_manifest_hashes={source.identity: source_hash},
        source_manifests=(source,),
        builder_config={"fixture": True},
        group_assignment={video_id: split},
        split_statistics={"train": int(split == "train"), "dev": int(split == "dev")},
        multi_event_ratio=0,
        leakage_audit_uri="generation/leakage.json",
        leakage_parquet_uri="generation/leakage.parquet",
        examples=(example,),
        asset_sha256s={video_id: asset_hash},
        generation_uri="generation",
    )
    step_cost = cost_to_go * normalization.constant
    next_state = state.model_copy(
        update={"action_history": state.action_history + (target_action,)}
    )
    transition = EnvironmentTransition(
        state=state,
        action=target_action,
        observation=ActionObservation(
            action_type=target_action.action_type,
            target_event_id=target_action.event_id,
        ),
        next_state=next_state,
        step_cost=step_cost,
    )
    labels = tuple(
        PreferenceLabel(
            cost_preference=preference,
            utility=1 - preference * cost_to_go,
            optimal_paths=(
                OraclePath(
                    transitions=(transition,),
                    answer=example.answer,
                    answer_score=1,
                    correct=True,
                    total_cost=step_cost,
                    utility=1 - preference * cost_to_go,
                ),
            ),
        )
        for preference in COST_PREFERENCES
    )
    predicted = example.answer if sufficiency_target else "__incorrect__"
    artifact = seal_sufficiency_label(
        state=state,
        question_id=question_id,
        gold_answer=example.answer,
        answerer=FrozenAnswerer(lambda _: predicted),
        answerer_identity=FrozenComponentIdentity(
            implementation="tests.frozen-answerer",
            model_id="fixture-answerer",
            revision="1",
            artifact_sha256="3" * 64,
        ),
    )
    return materialize_oracle_record(
        observation_snapshot_id=observation_snapshot_id,
        state=state,
        action_instances=actions,
        legal_action_mask=legal_action_mask,
        preference_labels=labels,
        normalization=normalization,
        manifest=manifest,
        example=example,
        sufficiency_artifact=artifact,
    )
