# Provider-Neutral Observation Import Design

**Date:** 2026-08-25
**Status:** Approved in conversation
**Scope:** M0 observation ingestion boundary only

## Goal

Separate expensive observation generation from local experiment orchestration. A GPU or external provider worker produces authoritative JSONL records; the local repository validates and imports them without invoking a model, then emits reviewable artifacts for R002/R003.

## Input Contract

Each JSONL line is one immutable observation atom with:

- `schema_version` equal to `1`;
- `question_id` and `video_id`;
- `provider_identity` containing non-empty provider, model revision, decode configuration, and device name;
- `state` validated as `RouterState`;
- `action` validated as `ActionInstance`;
- `observation` validated as `ActionObservation`;
- at least one nested authoritative `CostRecord` for every non-STOP action;
- a deterministic record identifier derived locally from canonical content.

The importer never accepts model predictions as ground truth and does not create Oracle labels. It only records observations and measured resource usage.

## Import Behavior

The importer reads JSONL line by line, rejects malformed or duplicate identifiers, validates that action and observation identities match, and rejects non-STOP records without measured cost metadata. Valid records are written canonically to `observations.jsonl` using an atomic replace.

With `--resume`, records already present with identical canonical content are counted as cache hits and are not duplicated or recharged. The same identifier with different content is a hard error.

## Outputs

One import produces:

- `observations.jsonl`: canonical validated records;
- `cost.csv`: one row per nested `CostRecord`, including question/video/action/provider identity;
- `summary.json`: counts, cache hits/misses, GPU/wall/frame/token totals, and P90 GPU seconds;
- `manifest.json`: schema version, input path/hash, provider identities, config hash, run id, and output paths.

These artifacts are engineering evidence. They become paper evidence only when the source worker uses the frozen preregistered provider and dataset ground truth is later used by Oracle/evaluation.

## Error Handling

The command fails closed with exit code `2` for malformed JSON, schema mismatch, identity mismatch, missing authoritative cost, duplicate conflict, or an empty import. It writes no partial final artifacts. Existing complete artifacts remain intact if a new import fails.

## CLI Boundary

`build-observations` gains `--input-jsonl`. Without this option, the existing mock counter path remains available only for R001 engineering smoke and is explicitly marked `engineering_smoke`. With the option, the authoritative importer runs and records `execution_status=completed` plus artifact paths.

No GPU, network, model SDK, API key, or SSH credential is used by the importer.

## Testing

Tests use real Pydantic domain models and temporary files. They cover successful import, canonical artifacts, measured cost aggregation, resume idempotence, conflicting duplicates, malformed records, missing cost metadata, and preservation of prior artifacts after failure.

## Self-Review

- No placeholders or undecided provider SDKs.
- Input/output identities are explicit.
- Ground truth remains outside the observation importer.
- Failure is atomic and fail-closed.
- Scope is limited to the M0 provider boundary.
