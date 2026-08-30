# Video-MME-v2 Source Dataset Migration Design

## Status

Approved in chat on 2026-08-30. This design changes the source and Router-development dataset from LongTVQA to Video-MME-v2 while preserving the existing Production Authority schema, Experiment Execution Pack, E00-E17 DAG, gate framework, Oracle protocol, Answerer stability threshold, and model stack.

## Objective

Use official Video-MME-v2 metadata, timestamped subtitles, and raw MP4 files as the source assets for event segmentation and Gist/Residual/Visual observation generation. Video-MME-v2 is not an independent final target benchmark. The final target benchmark set is LongVideoBench, LVBench, and MLVU.

The first deliverable may be a deterministic `PARTIAL_DATASET_PILOT` when the server cannot hold the full dataset. Pilot evidence must remain explicitly namespaced and must never be reported as a full Video-MME-v2 result.

## Immutable identities and licensing

The only approved upstream is the official Hugging Face dataset:

- repository: `MME-Benchmarks/Video-MME-v2`
- repository type: `dataset`
- immutable revision observed and approved for this migration: `6e4bebb03202e1ddbf3d37703e560e51c5aa2d64`
- metadata: `test.parquet`
- subtitles: `subtitle.zip`
- videos: `videos/001.zip` through `videos/040.zip`

The implementation must pin the revision above for every metadata, archive-index, and archive download request. It must record the upstream LFS SHA-256 for every downloaded object and must not use public URL mirrors, video-platform URLs, or third-party copies as substitutes.

Video-MME-v2 is restricted to academic research. The local data must not be redistributed, republished, copied to public artifacts, or committed to Git. Production manifests may contain local absolute paths and hashes, but no media content.

## Preserved experiment architecture

No new experiment DAG or Authority type will be introduced. The migration reuses:

- the existing asset-lock state machine and immutable revision checks;
- the existing thin setup wrappers and `--check`/resume convention;
- `VideoManifest`, `QuestionManifest`, `DatasetManifest`, and `SelectionManifest`;
- E00 environment recording;
- E01 dataset freeze and video-disjointness gate;
- E02 Authority Draft validation and runtime seal;
- E03 provider runner, canonical observation importer, raw response persistence, CostRecord reconciliation, namespace isolation, and resume validation;
- E04 Oracle protocol and all downstream E05-E17 dependencies.

The model stack remains unchanged:

- Gist text and embedding: `BAAI/bge-m3` shared physical snapshot;
- Gist visual: `google/siglip2-so400m-patch14-384`;
- Residual and Visual: `Qwen/Qwen3-VL-8B-Instruct` shared physical snapshot;
- Answerer: `Qwen/Qwen3-8B`;
- dtype: `bfloat16`;
- backend: Hugging Face Transformers.

## Dataset adapter boundary

A Video-MME-v2 adapter will replace the LongTVQA-specific setup implementation without changing the generic execution pack. Its responsibilities are:

1. parse and validate the complete official annotation table;
2. validate the one-video/four-question group structure and stable question identities;
3. build an immutable mapping from each `video_id` to its official ZIP archive;
4. select either full scope or a deterministic pilot scope;
5. download and resume only the official archives required by the selected scope;
6. verify archive LFS hashes before extraction;
7. extract only selected MP4 members with path-traversal protection and atomic replacement;
8. verify MP4 identity, duration, decodability, question coverage, and subtitle coverage;
9. build the existing manifest types and deterministic selection manifests;
10. generate a pending human-audit manifest without generating or inferring a human PASS result.

The generic asset lock will represent `videomme_v2_metadata` as the `source_dataset` logical role. Metadata verification covers the pinned annotation, subtitle, README, and repository identity. Raw MP4 identity is carried by the E01 video manifest and Source Gate rather than being duplicated into the model/metadata snapshot hash.

## Commands and resumability

The setup entrypoints will remain thin Bash wrappers around Python modules. Video-MME-v2 metadata and video preparation must expose the following behavior:

- `--check`: validate the pinned remote/local identities, required storage roots, free-space safety margin, annotation schema, deterministic selection plan, and existing partial files without downloading or extracting media;
- `--resume`: continue Hugging Face downloads using the pinned revision, reuse archives whose LFS SHA-256 matches, reuse extracted MP4s whose recorded content SHA-256 matches, and never charge or regenerate completed observation work;
- `--verify-only`: perform no network downloads, re-hash local metadata/archives/videos, probe every selected MP4, and run the deterministic random-decode sample.

Downloads and extraction write to temporary sibling paths and become visible through atomic rename. A failed archive or MP4 remains explicitly failed and is never accepted by the manifest builder.

## Official archive index

The implementation will read each official ZIP central directory at the pinned revision using HTTP range requests. It will not download video payloads while building the index. For every ZIP member it records:

- upstream repository and revision;
- archive path and archive LFS SHA-256;
- member path, uncompressed size, and ZIP CRC;
- normalized `video_id` derived from the `.mp4` filename.

The index fails closed on duplicate video IDs, unsafe paths, non-MP4 members presented as videos, metadata video IDs missing from all archives, or archive members absent from metadata. The canonical archive-index SHA-256 becomes an input to the subset selection manifest.

## Deterministic pilot scope

The initial pilot target is 45 selected videos. Video-MME-v2 contains four related questions per video, so 45 videos provide 180 questions while preserving complete question groups.

Selection uses two frozen seeds:

- pool seed: `videomme-v2-partial-pilot-pool-v1`
- split seed: `videomme-v2-partial-pilot-split-v1`

The pool algorithm is versioned as `videomme-v2-archive-aware-hash-v1`:

1. validate all 800 unique metadata video IDs and their archive-index entries;
2. rank all video IDs by canonical SHA-256 of `{algorithm, seed, video_id}`;
3. take the archive containing the highest-ranked video not covered by already selected archives;
4. repeat until the union of videos in selected archives contains at least 45 metadata video IDs;
5. rank the union with the same canonical rank and select the first 45 video IDs;
6. download the selected archives, but extract only the 45 selected MP4 files.

This algorithm is independent of question text, answer labels, difficulty, model predictions, or manual preference. The selection manifest records the complete metadata SHA-256, archive-index SHA-256, seeds, algorithm version, selected archive identities, selected video IDs, and its own canonical SHA-256.

## Pilot split

The split algorithm ranks the 45 selected video IDs by canonical SHA-256 of `{split_seed, video_id}` and assigns whole videos, including all four questions, as follows:

- Oracle: first 25 videos, exactly 100 questions;
- Canary: next 4 videos, exactly 16 questions;
- source-holdout: next 4 videos, exactly 16 questions;
- development: remaining 12 videos, exactly 48 questions.

No video or question group may cross assignments. Canary and Oracle are therefore video-disjoint, and development/source-holdout remain disjoint from both. Gold answers are available only through the existing explicit Oracle/evaluation scopes; development and Canary manifests do not expose gold-answer hashes.

The pilot is sufficient only for E00-E04 method and production-chain validation. It is not sufficient for paper-scale Router training, final accuracy claims, or a full Video-MME-v2 result.

## Full dataset continuation

When storage permits, the same adapter downloads all 40 pinned archives and builds a separate `FULL_DATASET` manifest. Full data uses distinct artifact, cache, Authority, and experiment run namespaces. Pilot observations may be reused only when their source MP4 SHA-256, config SHA-256, model identity, and Authority binding all match; otherwise they remain isolated pilot evidence.

The full split policy will be frozen as a separate checked-in policy and will never be inferred from pilot outcomes. Pilot results cannot select or tune the full split.

## Manifest and Authority representation

The generic dataset manifest will add provenance fields that are valid for any source dataset:

- `dataset_scope`: `PARTIAL_DATASET_PILOT` or `FULL_DATASET`;
- `source_metadata_sha256`;
- `source_archive_index_sha256`;
- `subset_selection_manifest_sha256` for pilot scope;
- selected and available video/question counts.

The official dataset name remains `MME-Benchmarks/Video-MME-v2`, and the dataset version remains the upstream immutable revision. The pilot marker is not encoded by altering the official dataset name or revision.

Production Authority remains unchanged. It binds the dataset, video, question, split-policy, and subset-selection artifacts through their repository-relative paths and SHA-256 identities. A pilot Authority is valid only for pilot-namespaced runs.

## Source Gate and human audit

For every selected MP4 the Source Gate requires:

- exactly one file for the metadata `video_id`;
- whole-file SHA-256;
- positive duration, dimensions, and frame rate from the existing video probe;
- successful deterministic midpoint decode for at least 20 selected videos;
- no duplicate content hash across different video IDs;
- complete annotation mapping;
- complete subtitle mapping when the official metadata declares subtitles for the video;
- no unexpected selected or extracted MP4.

The existing human-audit gate remains fail closed. The adapter creates a deterministic 100-question audit manifest for the 180-question pilot, binding question ID, video ID, and audit-manifest SHA-256. It never creates a completed result, reviewer identity, completion timestamp, or PASS outcome. E01 formal completion waits for a real human result bound to that manifest.

## Target benchmark role migration

Current configuration and runbook references will be updated so that:

- source/Router development: Video-MME-v2;
- final independent targets: LongVideoBench, LVBench, and MLVU;
- Video-MME-v2 is absent from the final target benchmark list and E13 main benchmark matrix.

Historical design and plan documents remain unchanged as historical records. Current stack configuration, experiment registry wording, E01/E03 documentation, setup wrappers, and integration tests reflect the new role assignment.

## Research decisions and hard stops

Dataset role, model stack, pilot selection protocol, pilot split, DAG, gates, Authority rules, Oracle protocol, Answerer stability threshold, and final benchmark set are frozen by this design.

The following existing production fields remain `RESEARCH_OWNER_DECISION_REQUIRED` and block E02 sealing until separately frozen:

- segmentation policy and version;
- complete per-role frame-sampling policy;
- Gist summary prompt;
- Answerer prompt template;
- tokenizer, pooling, normalization, batching, processor, decoding, seed, and precision settings required by the six model roles;
- E04 observation graph and Oracle headroom threshold where still unresolved in the existing config.

The Transformers provider factory may be implemented only after these settings are frozen. It must use the verified local snapshots and existing provider contract; it cannot use fixtures or silently choose generation parameters.

## Failure behavior

Every phase fails closed. In particular:

- a changed upstream revision requires a new lock and a new clean commit;
- remote or local hash mismatch stops extraction and manifest generation;
- insufficient free disk stops before archive download;
- a missing/corrupt/duplicate video stops E01;
- missing human audit stops E01 formal completion;
- unresolved production configuration stops Authority at E02;
- missing real backend factory stops E03;
- failed Canary stops E04 and every later experiment;
- pilot evidence is never promoted to full-dataset evidence by renaming or copying artifacts.

## Testing and verification

Implementation follows test-driven development. Engineering tests may use small generated ZIP/Parquet/MP4 fixtures, but those fixtures are never accepted as production evidence.

Tests cover:

- official annotation schema and four-question grouping;
- archive central-directory indexing and unsafe-member rejection;
- deterministic pilot selection and stable selection hash;
- exact 25/4/4/12 video split and 100/16/16/48 question counts;
- video-disjointness and gold-answer scoping;
- check/resume/verify-only behavior;
- archive and MP4 hash mismatch failures;
- atomic extraction recovery;
- `PARTIAL_DATASET_PILOT` propagation into dataset and Authority-bound artifacts;
- unchanged model sharing identities;
- final target benchmark set excluding Video-MME-v2 and including MLVU;
- existing E00-E17 registry and gate dependencies remaining unchanged.

Real-data verification records commands, exit codes, immutable revision, archive and MP4 hashes, counts, disk use, artifact paths, and gate verdicts. E03/E04 additionally record Authority hash, model revisions, GPU identity, observation counts, measured costs, resume evidence, and results.
