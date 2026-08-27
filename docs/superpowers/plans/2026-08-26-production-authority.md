# Production Authority Engineering Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task in the current working directory. Steps use checkbox (`- [ ]`) syntax for tracking. The user forbids worktrees, branches, commits, and pushes for this task.

**Goal:** Build a fail-closed, dataset-neutral Production Authority lifecycle and bind its deterministic SHA-256 to every production observation, cache, cost, manifest, state, and report artifact without running a production experiment.

**Architecture:** Add a typed `fidmem.production` package for Authority, split manifests, and provenance. The existing provider-neutral importer remains the observation boundary, but explicit production mode requires a sealed Authority and an Authority-specific namespace; development behavior stays isolated and backward compatible.

**Tech Stack:** Python 3.11+, Pydantic v2, OmegaConf/PyYAML-compatible YAML loading, standard-library JSON/hashlib/platform/subprocess/tempfile, pytest, argparse.

**Spec:** `docs/superpowers/specs/2026-08-26-production-authority-design.md`

## Global Constraints

- Work only in `D:\Desktop\Gist`; do not create a worktree or create/switch branches.
- Preserve every existing tracked and untracked user modification. Do not reset, checkout, clean, stash, rebase, delete unknown files, commit, push, merge, publish, or generate `PRODUCTION_AUTHORITY.json` during this task.
- Do not download a benchmark, call a paid API/VLM/provider, run Canary/Oracle/Router/DAgger, or generate a production observation.
- Keep `configs/experiment/oracle_pilot.yaml` values exactly `100/8/5/20/100/3/0.02` for question count, beam size, depth, exhaustive subset, stability states, repeats, and flip threshold.
- Every production behavior change follows RED → GREEN → REFACTOR with warnings treated as errors.
- Engineering, production-control, production, and paper evidence remain distinct. Templates and synthetic tests never become production evidence.
- Use native `apply_patch`; if the known Windows sandbox helper fails, use the already authorized `git apply --check` then `git apply` fallback on exact task files.

---

### Task 1: Repository Contract and Authority Core Types

**Files:**
- Create: `AGENTS.md`
- Create: `src/fidmem/production/__init__.py`
- Create: `src/fidmem/production/authority.py`
- Create: `tests/production/__init__.py`
- Create: `tests/production/test_authority_models.py`

**Interfaces:**
- Produces: `ProductionAuthorityDraft`, `SealedProductionAuthority`, `RepositoryIdentity`, `DatasetIdentity`, `ModelIdentity`, `PromptIdentity`, `CanonicalConfigIdentity`, `RuntimeIdentity`, `CostContract`, `canonical_json()`, `canonical_sha256()`, and `load_sealed_authority()`.
- Consumes: Pydantic models and standard-library JSON/SHA-256 only.

- [ ] **Step 1: Protect the working tree and write the repository contract**

Run `git status --short`, `git diff --check`, and `git diff --name-status` first. Create concise `AGENTS.md` with the approved research invariants, evidence classes, Authority gate, leakage rules, Git safety, required context, test discipline, and completion-report fields. Do not include current test counts or experiment numbers.

- [ ] **Step 2: Write failing core-model tests**

```python
import pytest
from pydantic import ValidationError

from fidmem.production.authority import (
    ProductionAuthorityDraft,
    PromptIdentity,
    canonical_sha256,
)


def test_draft_can_be_incomplete_but_never_production_ready() -> None:
    draft = ProductionAuthorityDraft()
    assert draft.lifecycle == "draft"
    assert draft.production_ready is False


def test_prompt_content_is_bound_to_its_hash() -> None:
    prompt = PromptIdentity(
        name="residual",
        version="1",
        content="Extract novel event details.",
        sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="prompt sha256"):
        prompt.verify_content_hash()


def test_malformed_sha_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PromptIdentity(name="p", version="1", content="x", sha256="bad")
```

The expected digest in a positive test must be a hand-checked literal computed outside the code under test.

- [ ] **Step 3: Run RED**

Run:

```powershell
D:\Anaconda\python.exe -W error -m pytest -q tests/production/test_authority_models.py
```

Expected: collection fails because `fidmem.production.authority` does not exist.

- [ ] **Step 4: Implement minimal frozen schemas and canonical hashing**

Use `ConfigDict(frozen=True, extra="forbid")`. SHA fields use lowercase `[0-9a-f]{64}`. Canonical serialization is:

```python
def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
```

`SealedProductionAuthority.authority_sha256` is recomputed from `model_dump(mode="json", exclude={"authority_sha256"})`. `load_sealed_authority(path)` validates lifecycle/readiness and detects tampering.

- [ ] **Step 5: Run GREEN and inspect the focused diff**

Run the Task 1 test command again. Then run Black on the two Python files and `git diff --check`. Do not commit.

---

### Task 2: Dataset, Question, Video, and Split Manifests

**Files:**
- Create: `src/fidmem/production/manifests.py`
- Create: `tests/production/test_manifests.py`

**Interfaces:**
- Produces: `ExperimentGroup`, `VideoManifestRecord`, `QuestionManifestRecord`, `VideoManifest`, `QuestionManifest`, `DatasetManifest`, `SelectionManifest`, `validate_split_isolation()`, and `select_questions_deterministically()`.
- Consumes: canonical hashing from Task 1.

- [ ] **Step 1: Write failing split and deterministic-selection tests**

```python
def test_video_cannot_cross_development_and_holdout() -> None:
    videos = VideoManifest(records=(
        video("v1", "development"),
        video("v1", "holdout"),
    ))
    with pytest.raises(ValueError, match="video_id.*multiple experiment groups"):
        validate_split_isolation(videos, question_manifest())


def test_questions_inherit_their_video_group() -> None:
    with pytest.raises(ValueError, match="question split differs"):
        validate_split_isolation(
            video_manifest(video("v1", "canary")),
            question_manifest(question("q1", "v1", "oracle")),
        )


def test_selection_is_stable_and_does_not_read_gold_answer() -> None:
    first = select_questions_deterministically(manifest, group="canary", count=2, seed="r002-v1")
    changed_gold = manifest.model_copy(update={"questions": questions_with_changed_gold})
    second = select_questions_deterministically(changed_gold, group="canary", count=2, seed="r002-v1")
    assert first.question_ids == second.question_ids
```

Use literal SHA values and real Pydantic manifests; do not mock selection.

- [ ] **Step 2: Run RED**

Run:

```powershell
D:\Anaconda\python.exe -W error -m pytest -q tests/production/test_manifests.py
```

Expected: collection fails because `fidmem.production.manifests` does not exist.

- [ ] **Step 3: Implement manifest schemas and video-level gates**

Use experiment groups `development`, `canary`, `oracle`, and `holdout`. Reject duplicate question IDs, duplicate `(video_id, group)` rows, any video assigned to multiple groups, a question group different from its video, and gold-answer material in unauthorized scopes.

Selection rank is `canonical_sha256({"seed": seed, "video_id": video_id, "question_id": question_id})`; sort by `(digest, video_id, question_id)`. The output binds source manifest hashes, seed, selected IDs, and its own canonical SHA-256.

- [ ] **Step 4: Run GREEN and mutation checks**

Run the Task 2 tests. Confirm that changing the group comparison or including gold-answer text in the rank would fail at least one test. Run Black and `git diff --check`. Do not commit.

---

### Task 3: Authority Validation, Runtime/Repository Probes, and Atomic Seal

**Files:**
- Modify: `src/fidmem/production/authority.py`
- Create: `tests/production/helpers.py`
- Create: `tests/production/test_authority_lifecycle.py`

**Interfaces:**
- Produces: `AuthorityValidationIssue`, `AuthorityValidationReport`, `probe_repository()`, `probe_runtime()`, `validate_authority_draft()`, and `seal_authority()`.
- Consumes: Task 1 schemas and Task 2 manifest loaders/gates.

- [ ] **Step 1: Add complete test-only draft builders**

`tests/production/helpers.py` builds real temporary dataset/question/video manifests, six non-placeholder model identities, five prompt identities, four configuration identities, a repository identity, runtime probe result, and cost contract. Expected hashes are produced when arranging fixture files; assertions still compare against literal or independently calculated values.

- [ ] **Step 2: Write failing validation tests**

Cover one behavior per test:

```python
@pytest.mark.parametrize("bad_id", [
    "text-1b-2b", "shared-frozen-vlm", "frozen-answerer/v1",
    "REPLACE_WITH_64_HEX", "latest", "main", "offline-smoke-v1",
])
def test_placeholder_or_mutable_model_identity_is_rejected(tmp_path, bad_id) -> None:
    report = validate_authority_draft(draft_with_model_id(tmp_path, bad_id), project_root=tmp_path, runtime_probe=fake_gpu_probe)
    assert "model_identity_not_immutable" in report.error_codes


def test_mutated_prompt_is_rejected(tmp_path) -> None:
    report = validate_authority_draft(draft_with_mutated_prompt(tmp_path), project_root=tmp_path, runtime_probe=fake_gpu_probe)
    assert "prompt_hash_mismatch" in report.error_codes


def test_mutated_config_is_rejected(tmp_path) -> None:
    report = validate_authority_draft(draft_with_mutated_config(tmp_path), project_root=tmp_path, runtime_probe=fake_gpu_probe)
    assert "config_hash_mismatch" in report.error_codes


def test_runtime_without_gpu_is_rejected(tmp_path) -> None:
    report = validate_authority_draft(complete_draft(tmp_path), project_root=tmp_path, runtime_probe=fake_cpu_probe)
    assert "production_gpu_missing" in report.error_codes
```

Also cover incomplete drafts, manifest file/hash mismatch, local snapshot mismatch, repository mismatch, CostRecord schema mismatch, and mismatched revision/artifact identity.

- [ ] **Step 3: Run validation RED**

Run:

```powershell
D:\Anaconda\python.exe -W error -m pytest -q tests/production/test_authority_lifecycle.py -k "validation or rejected or mismatch"
```

Expected: imports or missing functions fail.

- [ ] **Step 4: Implement deterministic validation**

`validate_authority_draft()` returns all issues in stable `(code, path, message)` order. It never mutates the draft or writes an Authority. `probe_repository()` binds the full commit, dirty flag, tracked diff/content, and every non-ignored untracked path/content hash. `probe_runtime()` captures the local hostname, GPU names/UUIDs, driver, CUDA, PyTorch, Python, and inference backend without printing secrets.

- [ ] **Step 5: Run validation GREEN**

Run the Task 3 validation selection and then all Task 1–3 tests.

- [ ] **Step 6: Write failing seal/tampering tests**

```python
def test_seal_is_canonical_and_detects_tampering(tmp_path) -> None:
    sealed = seal_authority(complete_draft(tmp_path), output_path=tmp_path / "PRODUCTION_AUTHORITY.json", project_root=tmp_path, runtime_probe=fake_gpu_probe)
    assert load_sealed_authority(tmp_path / "PRODUCTION_AUTHORITY.json") == sealed
    mutate_prompt_in_file(tmp_path / "PRODUCTION_AUTHORITY.json")
    with pytest.raises(ValueError, match="authority_sha256"):
        load_sealed_authority(tmp_path / "PRODUCTION_AUTHORITY.json")


def test_failed_seal_preserves_existing_authority(tmp_path) -> None:
    before = existing_authority_bytes(tmp_path)
    with pytest.raises(AuthorityValidationError):
        seal_authority(incomplete_draft(), output_path=tmp_path / "PRODUCTION_AUTHORITY.json", project_root=tmp_path, runtime_probe=fake_gpu_probe)
    assert (tmp_path / "PRODUCTION_AUTHORITY.json").read_bytes() == before
```

- [ ] **Step 7: Implement atomic seal and run GREEN**

Write a same-directory temporary file, flush/fsync, parse it back with `load_sealed_authority()`, then `os.replace`. The public CLI path always uses `probe_runtime`; only the Python function accepts an injected probe for tests. Run all Task 3 tests, Black, compile, and `git diff --check`. Do not generate an Authority in the repository and do not commit.

---

### Task 4: Authority-Bound Provenance and Cache Namespaces

**Files:**
- Create: `src/fidmem/production/provenance.py`
- Modify: `src/fidmem/storage/cache.py`
- Create: `tests/production/test_provenance.py`
- Modify: `tests/storage/test_cache.py`

**Interfaces:**
- Produces: `EvidenceClass`, `ProductionContext`, `AuthorityBoundCacheEnvelope`, `engineering_run_root()`, `production_run_root()`, `require_single_authority()`, `AuthorityBoundCache.get_bound()`, and `AuthorityBoundCache.put_bound()`.
- Consumes: sealed Authority loading and existing `ContentAddressedCache`.

- [ ] **Step 1: Write failing namespace and cache tests**

```python
def test_production_and_development_namespaces_are_disjoint(tmp_path) -> None:
    prod = production_run_root(tmp_path, "a" * 64, "R002-canary")
    dev = engineering_run_root(tmp_path, "R001")
    assert prod != dev
    assert "production" in prod.parts and "development" in dev.parts


def test_bound_cache_rejects_different_authority(tmp_path) -> None:
    cache = AuthorityBoundCache(ContentAddressedCache(tmp_path / "cache"))
    cache.put_bound("key", {"value": 1}, authority_sha256="a" * 64)
    with pytest.raises(ValueError, match="authority"):
        cache.get_bound("key", expected_authority_sha256="b" * 64)
```

Also assert engineering cache cannot carry Authority and production cache cannot omit it.

- [ ] **Step 2: Run RED**

Run Task 4 tests; expect missing provenance symbols.

- [ ] **Step 3: Implement explicit roots and cache envelopes**

Production roots are `artifacts/production/<authority>/runs/<run>` and `artifacts/production/<authority>/cache`; engineering roots are under `artifacts/development`. The cache envelope canonical payload contains `schema_version`, `evidence_class`, `authority_sha256`, `payload`, and `payload_sha256`. Reads verify both hashes before returning the payload.

- [ ] **Step 4: Run GREEN and existing cache regression tests**

Run:

```powershell
D:\Anaconda\python.exe -W error -m pytest -q tests/production/test_provenance.py tests/storage/test_cache.py
```

Run Black and `git diff --check`. Do not commit.

---

### Task 5: Bind Production Observation Import, Artifacts, and Resume

**Files:**
- Modify: `src/fidmem/experiments/observation_import.py`
- Modify: `tests/experiments/test_observation_import.py`
- Modify: `tests/integration/test_observation_import_cli.py`

**Interfaces:**
- Consumes: `SealedProductionAuthority`, `ProductionContext`, and Authority-bound cache/provenance helpers.
- Produces: production-aware `ObservationImportRecord` and `import_observations(..., authority_path: str | Path | None = None)` as the sole import/materialization entry point.

- [ ] **Step 1: Write failing record-binding tests**

Add `evidence_class: Literal["engineering", "production"] = "engineering"` and optional `authority_sha256` to the desired record API. Tests assert:

- production without Authority fails;
- engineering with Authority fails;
- malformed or mismatched Authority fails;
- mixed Authority rows fail;
- canonical IDs change when Authority changes;
- the existing engineering fixture remains engineering and contains no production hash.

Run only the new `-k authority` tests and verify RED because fields/validation are absent.

- [ ] **Step 2: Implement minimal record gate and run GREEN**

Validate evidence class and Authority at model/import boundaries. Do not infer production from provider names or output paths. Run record tests with `-W error`.

- [ ] **Step 3: Write failing artifact-binding tests**

For a test-sealed Authority, assert the exact hash appears in canonical observations, every cost row, summary, run manifest, cache envelope/manifest, CLI state/history, report, and generation marker. Assert manifest semantic Authority hash, Authority file hash/path, and raw-cost aggregation reconcile.

- [ ] **Step 4: Implement staged production artifact set**

Build and validate an immutable generation under a same-parent staging directory; include its internal marker before atomically switching `CURRENT.json`. Any stage failure preserves the prior generation byte-identical. Engineering imports keep their current behavior and development classification.

- [ ] **Step 5: Write resume RED tests**

```python
def test_resume_with_same_authority_is_idempotent(tmp_path) -> None:
    first = production_import(tmp_path, authority="a" * 64, resume=False)
    before = artifact_bytes(first)
    second = production_import(tmp_path, authority="a" * 64, resume=True)
    assert second.cache_hits == first.record_count
    assert second.cache_misses == 0
    assert artifact_bytes(second) == before


def test_resume_with_different_authority_preserves_run(tmp_path) -> None:
    first = production_import(tmp_path, authority="a" * 64, resume=False)
    before = artifact_bytes(first)
    with pytest.raises(ValueError, match="different Authority"):
        production_import(tmp_path, authority="b" * 64, resume=True)
    assert artifact_bytes(first) == before
```

- [ ] **Step 6: Implement resume gate and run full importer GREEN**

Resolve and validate `CURRENT.json`, the generation marker, and every bound artifact before reading canonical records. Reject an incomplete generation, mixed hash, different hash, collision, or tampered record before writes. Run:

```powershell
D:\Anaconda\python.exe -W error -m pytest -q tests/experiments/test_observation_import.py tests/integration/test_observation_import_cli.py
```

Run Black, compile, and `git diff --check`. Do not commit.

---

### Task 6: CLI Lifecycle, Production Report Gate, Templates, and Runbook

**Files:**
- Modify: `src/fidmem/cli.py`
- Create: `configs/production/authority.example.yaml`
- Create: `configs/production/canary.example.yaml`
- Modify: `docs/RUNBOOK.md`
- Create: `tests/integration/test_production_authority_cli.py`
- Modify: `tests/integration/test_end_to_end.py`

**Interfaces:**
- Produces CLI commands `authority-validate`, `authority-seal`, and production-aware `build-observations --production-authority`; extends `report` with Authority verification.
- Consumes lifecycle/import APIs from Tasks 3–5.

- [ ] **Step 1: Write CLI RED tests**

Test that:

- `authority-validate` returns exit 2 plus stable issue codes for the invalid example template;
- `authority-seal` on the CUDA-disabled development host returns exit 2 and writes no Authority;
- production import requires a sealed Authority and writes only to the Authority namespace;
- production report rejects missing/mixed Authority before aggregating costs;
- engineering report remains explicitly engineering;
- the frozen Oracle YAML values remain exactly `100/8/5/20/100/3/0.02`.

Run the new integration file and verify parser/missing-command failures.

- [ ] **Step 2: Implement CLI dispatch and fail-closed status**

Add parser arguments without changing existing command semantics. Authority validation emits machine-readable JSON. Seal uses the actual host probe with no runtime override argument. `build-observations --production-authority` derives its run root from the sealed hash. State and command history record `evidence_class` and `authority_sha256`.

- [ ] **Step 3: Bind report aggregation**

In production mode, load the sealed Authority and verify state, manifest, summary, cost rows, cache manifest, and commit marker share one hash before totals are returned. On mismatch, persist `blocked`, return exit 2, and do not emit a completed report.

- [ ] **Step 4: Add deliberately invalid templates**

`authority.example.yaml` contains `lifecycle: draft`, `production_ready: false`, explicit `TEMPLATE / NOT PRODUCTION` markers, all schema sections, and no real identity. `canary.example.yaml` contains only canary selection/import configuration and an unset Authority reference; it cannot execute a model.

- [ ] **Step 5: Update the runbook**

Document draft → validate → target-host seal → production import/report commands, expected fail-closed development-host behavior, namespace paths, and the prohibition against counting templates or test fixtures as production evidence.

- [ ] **Step 6: Run CLI/integration GREEN**

Run:

```powershell
D:\Anaconda\python.exe -W error -m pytest -q tests/integration/test_production_authority_cli.py tests/integration/test_observation_import_cli.py tests/integration/test_end_to_end.py tests/integration/test_resume.py tests/integration/test_unwired_stages.py
```

Then run all production/experiment tests. Run Black, compile, and `git diff --check`. Do not commit.

---

### Task 7: Final Integrity Review, Records, and Verification

**Files:**
- Modify: `refine-logs/EXPERIMENT_TRACKER.md`
- Create: `refine-logs/EXPERIMENT_TRACKER_<timestamp>.md`
- Modify: `refine-logs/EXPERIMENT_CODE_REVIEW.md`
- Create: `refine-logs/EXPERIMENT_CODE_REVIEW_<timestamp>.md`
- Modify: `MANIFEST.md`

**Interfaces:**
- Consumes: all implementation and raw verification output.
- Produces: truthful engineering-readiness records only; no Production Authority or experimental result.

- [ ] **Step 1: Run focused verification**

```powershell
D:\Anaconda\python.exe -W error -m pytest -q tests/production tests/experiments tests/integration tests/storage tests/costs
```

Record the exact pass/skip/time output.

- [ ] **Step 2: Run full verification**

```powershell
D:\Anaconda\python.exe -W error -m pytest -q
```

Record the exact pass/skip/time output. If quota or execution capacity becomes unavailable, stop and resume later rather than weakening the suite.

- [ ] **Step 3: Run formatter, compile, and diff gates**

```powershell
D:\Anaconda\python.exe -m black --check src tests
D:\Anaconda\python.exe -m compileall -q src tests

git diff --check
```

Also scan new/untracked task files for trailing whitespace because `git diff --check` does not cover every untracked file.

- [ ] **Step 4: Review implementation against the approved spec**

Run the `experiment-bridge` reviewer checklist using a fresh reviewer if available and save a trace. If unavailable, perform the same checklist locally and mark it `[local-only]`. Confirm every required artifact binding, split gate, resume rule, runtime seal, and production/development isolation behavior has a passing test.

- [ ] **Step 5: Update tracker and review records using output protocols**

Write timestamped tracker/review files first, copy identical bytes to fixed names, and append one MANIFEST row per output. Keep R002 `BLOCKED` until the experiment owner supplies real dataset/model/prompt/GPU identities and seals Authority on the production host. State explicitly that no Authority, Canary, production observation, production cost, or paper result was generated.

- [ ] **Step 6: Final working-tree audit**

Run `git status --short`, `git diff --stat`, `git diff --check`, and list untracked files. Distinguish pre-existing modifications from this plan's files. Do not stage, commit, push, or clean.

## Plan Self-Review

- **Spec coverage:** Tasks 1–3 cover repository rules and Authority draft/validate/seal; Task 2 covers dataset-neutral manifests and leakage; Task 4 covers cache and namespace provenance; Task 5 covers observation/artifact/resume binding; Task 6 covers CLI/templates/report/runbook and frozen Oracle parameters; Task 7 covers tests, review, records, and evidence boundaries.
- **No unspecified implementation steps:** Every schema, public interface, artifact, CLI command, failure class, and verification command used by later tasks is introduced by an earlier task.
- **Type consistency:** `authority_sha256`, `authority_file_sha256`, `evidence_class`, `ProductionAuthorityDraft`, `SealedProductionAuthority`, `AuthorityValidationReport`, `ProductionContext`, and `authority_path` use the same names throughout.
- **Evidence boundary:** The plan never downloads data, calls a provider, selects a formal dataset/model, creates repository `PRODUCTION_AUTHORITY.json`, or upgrades engineering tests into production/paper evidence.
