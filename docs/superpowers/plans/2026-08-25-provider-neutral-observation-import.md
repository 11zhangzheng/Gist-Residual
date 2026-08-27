# Provider-Neutral Observation Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import authoritative precomputed observation JSONL into canonical, resumable, paper-auditable M0 artifacts without invoking a GPU model or external API.

**Architecture:** Add a focused importer module that validates existing domain models, computes local canonical identifiers, and atomically emits JSONL/CSV/JSON artifacts. Keep provider execution outside the repository boundary; extend `build-observations` only to dispatch to the importer when `--input-jsonl` is supplied.

**Tech Stack:** Python 3.11+, Pydantic v2 domain models, standard-library JSON/CSV/hashlib/tempfile, pytest, argparse.

---

### Task 1: Validate Authoritative Observation Records

**Files:**
- Create: `src/fidmem/experiments/__init__.py`
- Create: `src/fidmem/experiments/observation_import.py`
- Create: `tests/experiments/test_observation_import.py`

- [ ] **Step 1: Write the failing record-validation tests**

Create real `RouterState`, `ActionInstance`, `ActionObservation`, `OperationMetadata`, and `CostRecord` payloads. Assert `ObservationImportRecord.model_validate(payload)` succeeds for a measured residual action and rejects a non-STOP action whose metadata has no `cost_record`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `D:\Anaconda\python.exe -m pytest -q tests/experiments/test_observation_import.py -k record`

Expected: collection fails because `fidmem.experiments.observation_import` does not exist.

- [ ] **Step 3: Implement the minimal record model**

Define `ProviderIdentity` and `ObservationImportRecord` as frozen Pydantic models. Validate schema version `1`, non-empty identities, action/observation action type and target identity, and authoritative nested cost metadata for non-STOP actions. Add a `record_id` property computed from canonical JSON excluding any caller-supplied identifier.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `D:\Anaconda\python.exe -m pytest -q tests/experiments/test_observation_import.py -k record`

Expected: record tests pass.

### Task 2: Import Atomically and Resume Idempotently

**Files:**
- Modify: `src/fidmem/experiments/observation_import.py`
- Modify: `tests/experiments/test_observation_import.py`

- [ ] **Step 1: Write failing import/resume tests**

Write two valid JSONL records, call `import_observations(input_path, output_dir, resume=False)`, and assert canonical `observations.jsonl` contains two records. Run again with `resume=True`; assert two hits, zero misses, unchanged output bytes, and no duplicated cost rows. Add a conflict test where an existing record id maps to changed content and assert a hard failure.

- [ ] **Step 2: Run tests and verify RED**

Run: `D:\Anaconda\python.exe -m pytest -q tests/experiments/test_observation_import.py -k "import or resume or conflict"`

Expected: failure because `import_observations` is missing.

- [ ] **Step 3: Implement canonical atomic import**

Read non-empty JSONL lines, parse each record, reject duplicate ids within one input, merge identical records only when `resume=True`, reject content conflicts, and write to a temporary file in the output directory before `os.replace`. Do not modify existing outputs until all input records validate.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `D:\Anaconda\python.exe -m pytest -q tests/experiments/test_observation_import.py -k "import or resume or conflict"`

Expected: import/resume tests pass.

### Task 3: Emit Cost, Summary, and Manifest Artifacts

**Files:**
- Modify: `src/fidmem/experiments/observation_import.py`
- Modify: `tests/experiments/test_observation_import.py`

- [ ] **Step 1: Write failing artifact tests**

Assert the importer writes `cost.csv`, `summary.json`, and `manifest.json`. Verify exact GPU/wall/frame/token totals from nested cost records, deterministic P90 nearest-rank behavior, provider identity enumeration, input SHA-256, config SHA-256, cache hit/miss counts, and artifact paths.

- [ ] **Step 2: Run tests and verify RED**

Run: `D:\Anaconda\python.exe -m pytest -q tests/experiments/test_observation_import.py -k artifacts`

Expected: missing artifact assertions fail.

- [ ] **Step 3: Implement deterministic artifact writers**

Flatten nested cost records into CSV rows ordered by record id and metadata index. Aggregate numeric fields without estimates. Compute P90 using sorted values and `ceil(0.9*n)-1`. Atomically write summary and manifest after observations and cost data are ready.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `D:\Anaconda\python.exe -m pytest -q tests/experiments/test_observation_import.py -k artifacts`

Expected: artifact tests pass.

### Task 4: Wire CLI Dispatch and Fail Closed

**Files:**
- Modify: `src/fidmem/cli.py`
- Create: `tests/integration/test_observation_import_cli.py`
- Modify: `tests/integration/test_resume.py`

- [ ] **Step 1: Write failing CLI tests**

Invoke `main(["build-observations", ..., "--input-jsonl", path])` and assert exit code `0`, completed execution state, and four artifacts. Invoke malformed or missing-cost input and assert exit code `2`, blocked state, and no partial replacement. Keep the existing no-input mock path assertion but require `mode="engineering_smoke"` in its output.

- [ ] **Step 2: Run tests and verify RED**

Run: `D:\Anaconda\python.exe -m pytest -q tests/integration/test_observation_import_cli.py tests/integration/test_resume.py`

Expected: argparse rejects `--input-jsonl` or required output fields are absent.

- [ ] **Step 3: Implement minimal CLI wiring**

Add `--input-jsonl` to `build-observations`. Dispatch to `import_observations` when supplied; record completed status and artifact paths. Catch importer validation errors, persist a blocked reason, print machine-readable JSON, and return exit code `2`. Mark the legacy counter path as `engineering_smoke`.

- [ ] **Step 4: Run integration tests and verify GREEN**

Run: `D:\Anaconda\python.exe -m pytest -q tests/integration/test_observation_import_cli.py tests/integration/test_resume.py tests/integration/test_unwired_stages.py`

Expected: all targeted integration tests pass.

### Task 5: Verify and Record M0 Readiness

**Files:**
- Modify: `refine-logs/EXPERIMENT_TRACKER.md`
- Create: timestamped `refine-logs/EXPERIMENT_TRACKER_<timestamp>.md`
- Modify: `MANIFEST.md`

- [ ] **Step 1: Run focused module verification**

Run: `D:\Anaconda\python.exe -m pytest -q tests/experiments tests/integration tests/actions tests/oracle tests/costs`

Expected: all tests pass; CUDA-specific tests may skip.

- [ ] **Step 2: Run diff checks**

Run: `git diff --check`

Expected: no whitespace errors.

- [ ] **Step 3: Update experiment records**

Record the exact test result and mark the provider-neutral importer `DONE`. Keep R002/R003 `BLOCKED` until a frozen provider generates authoritative records on the approved GPU server or approved external API.

- [ ] **Step 4: Request code review before deployment**

Use `superpowers:requesting-code-review` plus the `experiment-bridge` reviewer checklist. Save blocking/non-blocking findings to `refine-logs/EXPERIMENT_CODE_REVIEW.md`. Do not connect to a server or API until blocking findings are cleared and the user explicitly supplies/approves credentials.

## Plan Self-Review

- Spec coverage: validation, atomicity, resume, artifacts, CLI, tests, and deployment gate each map to one task.
- Placeholder scan: no TBD/TODO or unspecified implementation step remains.
- Type consistency: all records reuse `RouterState`, `ActionInstance`, `ActionObservation`, and nested `CostRecord` from existing modules.
