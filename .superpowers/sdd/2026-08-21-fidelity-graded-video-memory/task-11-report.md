# Task 11 implementation report

Status: ROUND1_DONE

## Scope and design

- Task 11 now exposes one stable `fidmem.router.dagger` API backed by three
  single-responsibility modules: pure cached rollout/utility logic, atomic
  resumable workflow logic, and the production Task 10 adapter/config loader.
- Task 10 is consumed only through public `OracleBCDataset`, model/tokenizer,
  `train_bc`, `load_checkpoint`, split, and runtime contracts. No Task 10 file
  was modified and no Task 10 private dataset helper is imported.
- Every real DAgger `MemoryEnvironment` must contain an injected
  `ForbiddenObservationGenerator`. Non-STOP rollout reads only
  `CachedUtilityGraph.get` / `CachedAnswerEvaluator.get` and advances only via
  `MemoryEnvironment.replay`.
- `DAggerQuestionContext` binds question/video identity, the immutable base BC
  dataset, cached snapshot identities, and exact Task 8/9/10 provenance.
- Each round atomically persists content-hashed seen keys, immutable deviation
  artifacts, a new policy checkpoint, and a self-hashed round manifest. Resume
  revalidates the source-policy chain, checkpoint/data/subset/context/seen
  identities, and never relabels a persisted state key.

## Root-cause investigation

The rejected implementation constrained observation lookup but not the full
Oracle dependency graph: it accepted an arbitrary callable evaluator, ignored
non-empty `SearchResult.pending`, and selected correctness-first
`canonical_oracle` instead of the Task 9 preference utility. The round API also
stopped at an in-memory result and therefore could not preserve a fixed subset,
seen keys, aggregated deviations, trained checkpoints, or resumable identity.

The production Task 10 API intentionally rejects exact-resume when the dataset
identity changes. DAgger therefore retrains each round's newly aggregated
base-BC-plus-deviation dataset into a new checkpoint, while the prior checkpoint
remains the recorded source policy used for rollout. This is standard aggregate
DAgger training and avoids modifying Task 10's frozen resume contract.

## RED / GREEN ledger

### Baseline

`D:\Anaconda\python.exe -m pytest tests/router/test_dagger.py -q`

Result before Round 1 tests: `7 passed in 13.00s`. The prior tests did not cover
any authoritative review failure.

### RED 1: cache/evaluation/utility/device/state-key contracts

`D:\Anaconda\python.exe -m pytest tests/router/test_dagger_round1.py -q`

Collection failed on missing `CacheArtifactIdentity`, proving that no explicit
immutable cache authority existed.

### RED 2: strict context and resumable multi-round workflow

`D:\Anaconda\python.exe -m pytest tests/router/test_dagger_workflow.py -q`

Collection failed on missing `DAggerConfig`, proving that no runner, trainer
boundary, manifest, seen-key persistence, or resume API existed.

### RED 3: actual forbidden executor injection

`D:\Anaconda\python.exe -m pytest tests/router/test_dagger_security.py -q`

Result: `1 failed`; an environment with a callable executor was accepted.
After validation was added, the same test passed.

### RED 4: worker/order-independent resume identity

`D:\Anaconda\python.exe -m pytest tests/router/test_dagger_order_resume.py -q`

Result: `1 failed` with `round manifest run identity mismatch` when the same
contexts arrived in reverse order. Canonical context identity sorting fixed it.

### GREEN

`D:\Anaconda\python.exe -m pytest tests/router/test_dagger.py tests/router/test_dagger_round1.py tests/router/test_dagger_workflow.py tests/router/test_dagger_task10_adapter.py tests/router/test_dagger_security.py -q`

Final focused result: `18 passed in 3.43s`.

The production-adapter tests additionally prove that the config is strictly
loaded, all configured runner/trainer fields are consumed, aggregate size is
base BC plus deviations, and the injected production bridge calls the Task 10
`train_bc` contract and returns a content-hashed checkpoint.

## Final verification

- Offline workflow smoke:
  `D:\Anaconda\python.exe -m pytest tests/router/test_dagger_workflow.py::test_multiround_training_persists_seen_keys_manifests_and_resumes tests/router/test_dagger_round1.py::test_non_stop_rollout_reads_cache_and_replays_with_forbidden_executor -q`
  -> `2 passed in 2.96s`.
- Final Router suite:
  `D:\Anaconda\python.exe -m pytest tests/router -q`
  -> `72 passed in 14.59s`.
- Full default environment, no OMP/MKL workaround:
  `D:\Anaconda\python.exe -m pytest -q`
  -> `320 passed, 1 skipped in 57.18s` (under the 180-second bound).
- `D:\Anaconda\python.exe -m compileall -q src tests` -> exit 0.
- Scoped `ruff check` -> exit 0.
- Scoped `ruff format --check` -> `10 files already formatted`.
- `git diff --check` -> exit 0, no output.

## Residual limits

- No provider, VLM, Answerer, or judge I/O is permitted or used during DAgger
  correction; absent observation/evaluation cache state fails closed.
- The Task 10 adapter inherits Task 10's exact git/runtime/tokenizer checkpoint
  checks. CUDA checkpoint replay remains unexercised on this CPU host; no CUDA
  exact-resume claim is made.
