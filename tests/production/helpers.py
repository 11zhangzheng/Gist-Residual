from __future__ import annotations

import hashlib
from pathlib import Path

from fidmem.production.authority import (
    CanonicalConfigIdentity,
    CostContract,
    DatasetIdentity,
    GPUIdentity,
    ModelIdentities,
    ModelIdentity,
    ObservationConfigurations,
    ProductionAuthorityDraft,
    PromptIdentities,
    PromptIdentity,
    RepositoryIdentity,
    RuntimeIdentity,
    canonical_json,
    canonical_sha256,
    production_cost_schema_sha256,
)
from fidmem.production.manifests import (
    DatasetManifest,
    QuestionManifest,
    QuestionManifestRecord,
    VideoManifest,
    VideoManifestRecord,
)


def fake_repository(_project_root: Path | None = None) -> RepositoryIdentity:
    return RepositoryIdentity(
        git_commit="1" * 40,
        dirty_worktree=True,
        source_tree_sha256="2" * 64,
        repository_root_name="repo",
    )


def fake_gpu_runtime() -> RuntimeIdentity:
    return RuntimeIdentity(
        machine_identity="gpu-host",
        gpu_count=1,
        gpus=(GPUIdentity(name="NVIDIA A800", uuid="GPU-test-uuid"),),
        driver_version="555.42",
        cuda_version="12.4",
        pytorch_version="2.5.1+cu124",
        python_version="3.12.4",
        inference_backend="vllm",
        inference_backend_version="0.6.0",
    )


def fake_cpu_runtime() -> RuntimeIdentity:
    return RuntimeIdentity(
        machine_identity="cpu-host",
        gpu_count=0,
        gpus=(),
        driver_version="unavailable",
        cuda_version="unavailable",
        pytorch_version="2.5.1+cpu",
        python_version="3.12.4",
        inference_backend="unconfigured",
        inference_backend_version="unavailable",
    )


def _write(path: Path, value: object) -> str:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complete_draft(tmp_path: Path) -> ProductionAuthorityDraft:
    videos = VideoManifest(
        dataset_name="ApprovedDataset",
        dataset_version="0123456789abcdef",
        records=(
            VideoManifestRecord(
                video_id="v1",
                content_sha256="3" * 64,
                uri="videos/v1.mp4",
                duration_seconds=60.0,
                group="canary",
            ),
        ),
    )
    questions = QuestionManifest(
        dataset_name="ApprovedDataset",
        dataset_version="0123456789abcdef",
        records=(
            QuestionManifestRecord(
                question_id="q1",
                video_id="v1",
                record_sha256="4" * 64,
                question_types=("visual",),
                group="canary",
            ),
        ),
    )
    split_policy = {"version": 1, "unit": "video_id", "groups": ["canary"]}
    split_path = tmp_path / "split-policy.json"
    video_path = tmp_path / "videos.json"
    question_path = tmp_path / "questions.json"
    dataset_path = tmp_path / "dataset.json"
    split_sha = _write(split_path, split_policy)
    video_sha = _write(video_path, videos.model_dump(mode="json"))
    question_sha = _write(question_path, questions.model_dump(mode="json"))
    dataset_manifest = DatasetManifest(
        dataset_name="ApprovedDataset",
        dataset_version="0123456789abcdef",
        dataset_scope="FULL_DATASET",
        source_metadata_sha256="5" * 64,
        source_archive_index_sha256="6" * 64,
        subset_selection_manifest_sha256=None,
        selected_video_count=len(videos.records),
        selected_question_count=len(questions.records),
        available_video_count=len(videos.records),
        available_question_count=len(questions.records),
        split_policy_id="video-id-v1",
        split_policy_sha256=split_sha,
        video_manifest_sha256=videos.manifest_sha256,
        question_manifest_sha256=questions.manifest_sha256,
    )
    dataset_sha = _write(dataset_path, dataset_manifest.model_dump(mode="json"))

    def model(role: str) -> ModelIdentity:
        evidence = {
            "identity_kind": "provider_revision",
            "provider": "approved-provider",
            "canonical_id": f"org/{role}-model",
            "immutable_revision": "0123456789abcdef0123456789abcdef01234567",
        }
        evidence_path = tmp_path / f"{role}-model-identity.json"
        evidence_sha256 = _write(evidence_path, evidence)
        return ModelIdentity(
            provider=evidence["provider"],
            canonical_id=evidence["canonical_id"],
            immutable_revision=evidence["immutable_revision"],
            identity_kind="provider_revision",
            identity_evidence_path=evidence_path.name,
            identity_evidence_sha256=evidence_sha256,
            artifact_sha256=None,
            dtype="bfloat16",
            runtime_settings={"temperature": 0.0, "max_tokens": 128},
        )

    def prompt(name: str, content: str) -> PromptIdentity:
        return PromptIdentity(
            name=name,
            version="1",
            content=content,
            sha256=hashlib.sha256(content.encode()).hexdigest(),
        )

    def config(name: str) -> CanonicalConfigIdentity:
        content = {"name": name, "version": 1}
        return CanonicalConfigIdentity(
            version="1", content=content, sha256=canonical_sha256(content)
        )

    return ProductionAuthorityDraft(
        repository=fake_repository(),
        dataset=DatasetIdentity(
            dataset_name="ApprovedDataset",
            dataset_version="0123456789abcdef",
            split="canary",
            split_policy_id="video-id-v1",
            split_policy_path=split_path.name,
            split_policy_sha256=split_sha,
            dataset_manifest_path=dataset_path.name,
            dataset_manifest_sha256=dataset_sha,
            question_manifest_path=question_path.name,
            question_manifest_sha256=question_sha,
            video_manifest_path=video_path.name,
            video_manifest_sha256=video_sha,
        ),
        models=ModelIdentities(
            gist_text_encoder=model("gist-text"),
            gist_visual_encoder=model("gist-visual"),
            residual_model=model("residual"),
            visual_model=model("visual"),
            answerer=model("answerer"),
            embedding_model=model("embedding"),
        ),
        prompts=PromptIdentities(
            gist_summary=prompt("gist-summary", "Summarize the transcript."),
            residual_generation=prompt("residual", "Extract novel event details."),
            visual_event=prompt("visual-event", "Describe reusable event evidence."),
            visual_question=prompt("visual-question", "Verify the requested detail."),
            answerer_template=prompt("answerer", "Question Options Evidence Answer"),
        ),
        observation_configurations=ObservationConfigurations(
            segmentation=config("segmentation"),
            frame_sampling=config("frame-sampling"),
            retrieval=config("retrieval"),
            observation_budget=config("observation-budget"),
        ),
        cost=CostContract(
            cost_record_schema_version="1",
            cost_accounting_version="fidmem-cost-v1",
            units={
                "gpu_seconds": "seconds",
                "wall_seconds": "seconds",
                "input_frames": "count",
                "visual_tokens": "count",
                "text_tokens": "count",
                "peak_memory_bytes": "bytes",
            },
            aggregation_semantics={
                "gpu_seconds": "sum",
                "wall_seconds": "sum",
                "input_frames": "sum",
                "visual_tokens": "sum",
                "text_tokens": "sum",
                "peak_memory_bytes": "max",
                "cache_hits": "count",
                "cache_misses": "count",
                "amortizable_event_work": "charge_once_per_authority_cache_key",
            },
            schema_sha256=production_cost_schema_sha256(),
        ),
    )
