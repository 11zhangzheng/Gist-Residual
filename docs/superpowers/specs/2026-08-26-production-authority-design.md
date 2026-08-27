# Production Authority Engineering Closure Design

**Date:** 2026-08-26
**Status:** Approved; conformance clarifications incorporated 2026-08-27
**Scope:** Production Authority, provenance binding, dataset split infrastructure, and production/development namespace isolation only

## 1. Goal and non-goals

Build the engineering gate that must succeed before any Production Canary observation can be generated. After this work, an experiment owner can supply real dataset, model, prompt, configuration, and GPU-host identities; the repository can validate and seal them into one deterministic `PRODUCTION_AUTHORITY.json`, then require its SHA-256 throughout the observation import and reporting chain.

This scope does not select a dataset or model, download benchmark media, call a provider or VLM, generate production observations, run Canary or Oracle, train Router/DAgger, or execute a benchmark. Completing this design produces engineering evidence only.

The frozen Oracle protocol remains unchanged:

- `question_count = 100`
- `beam_size = 8`
- `max_depth = 5`
- `exhaustive_subset_size = 20`
- `stability_state_count = 100`
- `stability_repeats = 3`
- `flip_rate_threshold = 0.02`

## 2. Repository-level research contract

Create a concise root `AGENTS.md` containing stable rules only:

- preserve the Gist → Residual → Raw Visual, learned cost-aware Router, accuracy–cost Pareto, question-independent Residual cache, and frozen observation-model/Answerer hypotheses;
- distinguish engineering, production, and paper evidence;
- reject mock, synthetic, fixture, dry-run, placeholder, and estimated-cost promotion;
- require a sealed Production Authority before production observation generation;
- enforce video-level split isolation, question-scoped Visual cache, event-scoped question-independent Residual reuse, GT path restrictions, and deployable-feature restrictions;
- preserve user Git changes and prohibit destructive or publishing operations by default;
- require the plan, tracker, review, runbook, and frozen configs as task context;
- require tests for every production invariant and a truth-separated completion report.

`AGENTS.md` is not timestamped and contains no changing test counts or experiment results.

## 3. Component boundaries

Add a focused `fidmem.production` package:

```text
src/fidmem/production/
  __init__.py
  authority.py       # schema, canonical hash, draft/validate/seal lifecycle
  authority_io.py    # draft parsing only
  manifests.py       # dataset/question/video manifests and split gates
  provenance.py      # namespace paths and Authority-bound cache/artifact gates
  generation.py      # immutable generation publication and atomic CURRENT pointer
  cli_support.py     # production-only CLI validation helpers
  observation_import.py  # compatibility adapter only; no importer implementation
  canary.py              # compatibility adapter only; no metric implementation
```

Keep `src/fidmem/experiments/observation_import.py` as the sole observation record schema, validation, cost/summary, resume, cache-key, and materialization source of truth. Explicit production mode consumes sealed Authority, provenance, cache, and immutable-generation helpers but does not decide dataset/model identity itself. Read-only Canary committed-run validation lives in `src/fidmem/experiments/canary_validation.py`; production adapters contain no experiment metric logic.

`src/fidmem/cli.py` exposes validation/sealing and explicit production import entry points. It never turns an ordinary engineering import into production implicitly.

## 4. Versioned Authority schema

### 4.1 Draft and sealed types

`ProductionAuthorityDraft` has `schema_version = 1`, `lifecycle = "draft"`, and `production_ready = false`. Authority sections may be absent so the experiment owner can prepare the document incrementally. Draft serialization is never a production credential.

`SealedProductionAuthority` has `schema_version = 1`, `lifecycle = "sealed"`, `production_ready = true`, every required section populated, and an `authority_sha256` matching its canonical content.

Unknown fields are rejected in both types. Blank strings are rejected wherever a value is present.

### 4.2 Repository identity

`RepositoryIdentity` contains:

- `git_commit`: full 40-hex commit;
- `dirty_worktree`: boolean;
- `source_tree_sha256`: SHA-256 of a deterministic allowlisted inventory of execution-affecting `src/**`, `configs/**`, and root dependency/build descriptors. It excludes `PRODUCTION_AUTHORITY.json`, artifacts, production outputs/caches, `.aris`, `refine-logs`, reports, logs, temporary files, and pytest/Python caches;
- `repository_root_name`: non-secret repository identity.

Validation recomputes the identity from the target repository. A clean or dirty tree is allowed, but the declared identity must equal the actual tree being sealed.

### 4.3 Dataset identity

`DatasetIdentity` contains:

- `dataset_name`;
- `dataset_version` or immutable revision;
- `split`;
- `split_policy_id` and `split_policy_sha256`;
- relative `dataset_manifest_path`, `question_manifest_path`, and `video_manifest_path`;
- SHA-256 for each of those three manifests.

Validation resolves paths under the repository or explicitly approved data root, requires files to exist, and recomputes each hash. A plan-level dataset family name without a manifest is insufficient.

### 4.4 Model identities

`ModelIdentity` is used for these six required roles:

- `gist_text_encoder`;
- `gist_visual_encoder`;
- `residual_model`;
- `visual_model`;
- `answerer`;
- `embedding_model`.

Each entry contains `provider`, `canonical_id`, `immutable_revision`, `identity_kind`, `identity_evidence_path`, `identity_evidence_sha256`, `dtype`, and canonical `runtime_settings`.

For `identity_kind = local_artifact`, `local_snapshot_path`, `local_snapshot_sha256`, and `artifact_sha256` are all required and must identify the same verified checkpoint bytes. For `identity_kind = provider_revision`, local snapshot and `artifact_sha256` fields are forbidden; the evidence file must be exact JSON binding `identity_kind`, `provider`, `canonical_id`, and `immutable_revision` to a provider-backed immutable deployment/revision. A hosted provider that cannot supply independently verifiable immutable revision evidence fails closed. An arbitrary 64-hex value is never accepted as hosted-model proof.

Mutable or placeholder identities are rejected, including `latest`, `main`, `master`, smoke/test identities, template markers, and empty revisions.

### 4.5 Prompts and observation configuration

Every `PromptIdentity` contains `name`, `version`, raw `content`, and `sha256`. Prompt SHA-256 has an explicitly distinct domain: the exact UTF-8 bytes of `content`, with no JSON encoding or newline normalization. Validation recomputes that raw-content digest.

The sealed Authority requires prompt entries for every configured production generation/template role, including Gist summary, Residual generation, Visual event observation, Visual question verification, and Answerer template.

`CanonicalConfigIdentity` contains canonical JSON `content`, `version`, and `sha256`. The Authority requires segmentation, frame sampling, retrieval, and observation-budget configurations. Validation recomputes every hash. Mutating content without updating its digest fails; changing both content and digest produces a different Authority rather than silently resuming the old run.

### 4.6 Runtime seal

`RuntimeIdentity` contains:

- hostname or non-secret machine identity;
- GPU count and GPU name/UUID pairs;
- driver version;
- CUDA runtime/version;
- PyTorch version;
- Python version;
- inference backend name and version.

The public seal path always captures runtime identity on the executing host. It does not accept a caller-provided production runtime blob. Unit tests may inject a deterministic probe at the internal function boundary; the CLI cannot. Production validation requires at least one visible GPU and complete driver/CUDA/backend identity. A CUDA-disabled development host therefore cannot seal a production Authority.

### 4.7 Cost contract

`CostContract` contains:

- `cost_record_schema_version`;
- `cost_accounting_version`;
- units for GPU time, wall time, frames, visual/text tokens, and peak memory;
- aggregation semantics for sums, maxima, cache hits, cache misses, and amortizable event-level work;
- `schema_sha256` binding the canonical CostRecord schema.

Validation rejects estimated values labeled as measured and verifies that the declared schema matches the production `CostRecord` fields used by the importer.

## 5. Deterministic hashing

All production semantic identities use the shared `canonical_json_bytes()` and `canonical_sha256()` primitive: UTF-8, no BOM, sorted keys, compact separators, `ensure_ascii=false`, `allow_nan=false`, and no trailing newline. `authority_sha256` is excluded from its own hash input; all other sealed fields, including `production_ready=true`, participate. Deterministic selection hashes a structured canonical object containing separately named seed/video/question fields; it never hashes ambiguous string concatenation.

`authority_sha256` is the semantic Authority identity. `authority_file_sha256` hashes the exact sealed serialized bytes. `authority_path` is non-semantic location metadata. Moving the same sealed file does not change either semantic or file identity. Loading recomputes semantic identity and fails on tampering; production manifests bind both hashes plus the path.

## 6. Lifecycle

### Draft

- A YAML template may be copied and edited.
- Missing fields and explicit template markers are allowed.
- It remains `lifecycle: draft` and `production_ready: false`.
- It cannot authorize production imports.

### Validate

Validation returns a machine-readable report listing all failures without writing `PRODUCTION_AUTHORITY.json`. It checks required fields, placeholders, hashes, manifest schema/existence, split overlap, model immutability, prompt/config content, local snapshots, actual Git identity, actual runtime, and CostRecord consistency.

### Seal

Seal reruns validation using fresh repository and runtime probes, constructs the sealed model, computes `authority_sha256`, verifies a parse round trip, and atomically writes `PRODUCTION_AUTHORITY.json`. Any failure leaves an existing sealed Authority untouched and emits no incomplete replacement.

## 7. Dataset-neutral split infrastructure

### 7.1 Manifests

`VideoManifest` contains dataset identity plus unique video records with `video_id`, immutable content SHA-256, URI/path identity, duration metadata, and assigned split group.

`QuestionManifest` contains unique question records with `question_id`, `video_id`, record SHA-256, question type tags, and assigned split group. Gold answers may be present only in explicitly authorized Oracle/evaluation manifests; selection logic never reads them.

`DatasetManifest` binds the question/video manifest hashes, dataset revision, and split policy.

### 7.2 Split policy

Allowed experiment groups are `development`, `canary`, `oracle`, and `holdout`. A `video_id` may appear in exactly one group. Questions inherit the group of their video. Development and holdout overlap is always rejected; every other cross-group video overlap is rejected as well.

Deterministic selection ranks eligible records by `canonical_sha256({"seed": ..., "video_id": ..., "question_id": ...})`, then takes the requested count while preserving video-group constraints. The selection manifest records seed, source hashes, chosen IDs, and its own canonical SHA-256.

## 8. Production provenance and namespace isolation

### 8.1 Explicit namespace

Observation records gain:

- `evidence_class`: `engineering` or `production`, defaulting to `engineering` for backward-compatible R001 fixtures;
- `authority_sha256`: absent for engineering, required for production.

Engineering records carrying Authority and production records lacking Authority are rejected.

Production artifacts live under:

```text
artifacts/production/<authority_sha256>/runs/<run_id>/
artifacts/production/<authority_sha256>/cache/
```

Engineering artifacts always live under `artifacts/development/runs/<run_id>/`. Production import never reads development cache files, even if record content otherwise matches. Merely supplying `--production-authority` to dry-run, ingest, build-gist, engineering-smoke, or any other generic engineering command cannot enter the production namespace.

### 8.2 Artifact binding

The same Authority hash is required in every provider-worker production observation row, canonical observation row, cost row, summary, manifest, production cache envelope/cache manifest, CLI state/command history, and report/cost aggregation.

The production manifest also binds the sealed Authority file path and file SHA-256. Report aggregation rejects missing, mixed, or mismatched hashes before reading totals.

### 8.3 Import and resume

Production import requires the explicit real-input path `build-observations --input-jsonl ... --production-authority <path>` and every input record must declare `evidence_class=production` with the same sealed Authority. Absence retains engineering mode. A blocked production attempt never falls back to engineering.

Before writing artifacts, the importer validates every record, sealed Authority, namespace path, existing run manifest, and existing canonical observations. One run may contain exactly one Authority. Resume under the same Authority treats identical complete records as cache hits without regeneration or recharge. Resume under a different Authority fails before modifying artifacts.

Record IDs derive from complete canonical record content, including `evidence_class` and `authority_sha256`. Collision, duplicate-ID tampering, or identical IDs with different content fail closed.

Production artifact sets are built and validated as immutable generation directories. Only an atomic `CURRENT.json` pointer switch makes a complete generation visible. Observation, cost, manifest, state, cache, or report failure leaves the prior committed generation byte-identical and reportable; reports resolve exactly one CURRENT generation.

### 8.4 Cache and Canary ownership

The real production import and no-op resume paths read/write cache entries only through `AuthorityBoundCache`. Cache envelopes bind evidence class, Authority, payload, and payload hash. Event-level amortizable Residual keys omit `question_id`; question-level Visual keys include `question_id`; all keys include visual budget/config/model/revision as applicable. Missing, mixed, or cross-Authority cache envelopes fail closed.

Canary validation is read-only experiment logic in `fidmem.experiments.canary_validation`. It may consume production identity/provenance APIs and the CURRENT committed generation, but it never executes a provider/model and is not a production identity source of truth.
## 9. CLI and templates

Add provider-neutral commands or equivalent subcommands:

```text
authority-validate --draft <path>
authority-seal --draft <path> --output PRODUCTION_AUTHORITY.json
build-observations --input-jsonl <path> --production-authority <path> --resume
```

`authority-validate` may run on a development machine and report the missing runtime seal. `authority-seal` must run on the intended production host.

Add `configs/production/authority.example.yaml` and `configs/production/canary.example.yaml`. Both state `TEMPLATE / NOT PRODUCTION`, remain drafts, and deliberately fail production validation until the experiment owner replaces every template value. The Canary template references an Authority path/hash but cannot initiate model execution.

## 10. Failure behavior

All production gates fail closed with machine-readable reasons and nonzero exit status. Validation never substitutes defaults for missing production identity. Seal and import preserve existing valid artifacts after malformed hash, placeholder identity, missing runtime, split overlap, Authority mismatch, namespace mismatch, schema error, collision, or tampering.

No production failure path falls back to engineering mode.

## 11. Test strategy

Tests use hand-derived canonical fixtures and real Pydantic/file behavior. Runtime and Git probes are injected only at the system boundary; no external model or GPU call occurs.

Required tests cover:

- incomplete Authority fail closed;
- placeholder model identity rejected;
- malformed SHA rejected;
- mutated prompt rejected;
- mutated configuration rejected;
- mismatched model revision/artifact identity rejected;
- missing runtime identity rejected;
- Authority canonical round trip and tampering detection;
- mismatched `authority_sha256` rejected;
- mixed-Authority observations rejected;
- production/development namespace isolation;
- video-level split overlap rejected;
- deterministic selection stability;
- same-Authority resume succeeds without duplicate cost;
- different-Authority resume fails without modifying artifacts;
- cost, summary, manifest, cache, state, and report binding/reconciliation;
- frozen Oracle configuration remains unchanged.

Verification order is focused tests with `-W error`, full tests with `-W error`, formatter check, `py_compile`/`compileall`, and `git diff --check`.

## 12. Evidence boundary and completion gate

The implementation, templates, validation reports, and tests are engineering evidence. A sealed Authority is production-control evidence, not an experimental result. Production observations become production evidence only after a real approved host/provider/model/dataset execution passes the Authority-bound import and reconciliation gates. Paper evidence still requires the later preregistered Oracle/evaluation path against dataset ground truth.

This engineering task is complete only when a real experiment owner can fill the template with authoritative dataset/model/prompt/config identities, run `authority-seal` on the target GPU host, and obtain a valid `PRODUCTION_AUTHORITY.json` without changing Python code. No Authority is generated during this task.
