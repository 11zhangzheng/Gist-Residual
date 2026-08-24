# Task 12 implementation report

- Original Task 12 baseline: `bab7ce6` (`fix: preserve indeterminate dagger commits`).
- Initial Task 12 commit: `92f44ad` (`feat: evaluate router accuracy-cost tradeoffs`).
- Round 1 scope: make the evaluation claims content-bound and independently recomputable; retain CPU-only synthetic execution and do not add a CLI.

## Round 1 RED / GREEN ledger

- Task 7 Answerer authority RED: 2 failed, 4 passed because `FrozenAnswerer` did not expose actual model/revision/decode/adapter identity or authoritative per-answer cost usage. GREEN: 6 passed in `tests/agent/test_runner.py`; complete `tests/agent tests/actions` later passed 43/43.
- Task 7 horizon RED: `AgentRunner` rejected the approved `max_transitions` argument. GREEN: default five-transition behavior and custom seven-transition forced STOP both passed.
- Task 12 review RED: the seven fresh authority/boundary/integration probes initially failed 7/7. After content factories, strict manifests, full cost accounting, restricted controller views, and legal fixed traces, the same probes passed 7/7.
- Cross-run authority RED: Pareto accepted runs with different Answerer/shared identities. GREEN: Pareto and Cost@Accuracy now require the same sealed benchmark, shared runtime identity, and cost preference.
- Amortization RED: five base-memory tokens over two actual group queries failed because integer divisibility was required. GREEN: aggregate resources retain fractional amortized equivalents; the integration result is exactly 2.5 + 2.5 tokens.
- Seed/config/family RED: `evaluate_run` did not apply the run seed, config accepted a missing baseline, and caller-declared policy family could disagree with the actual object. GREEN: all three are derived or strictly validated.
- Combined test collection RED: the eval and router review modules shared a basename. GREEN: the eval review module now has a unique name and both directories collect together.

## Content-bound evaluation authority

- `FrozenAnswerer.identity` is recomputed from the prompt template, model artifact, immutable revision, decode configuration, and actual adapter code/closure state. Formal evaluation fails closed when that identity or authoritative `AnswererAdapterResult` usage is unavailable; legacy string adapters remain answer-compatible for non-formal tests.
- `CacheBinding` recomputes its identity from the actual namespace and canonical cache file contents. `environment_sha256` binds canonical events, action/cost semantics, actual executor code/closure state, and that cache identity.
- `EvaluationQuestion.create` derives its record hash from question/options/gold, initial state, video events, split, canonical source manifest, budgets, Oracle authority, environment, and cache. Caller-supplied hashes or error signals are forbidden.
- `BenchmarkManifest.create` derives provenance, source, group assignment, leakage-audit, normalization, base-memory-cost, and question identities. Formal `train` questions are unrepresentable, and a run must cover every benchmark question exactly once and in order.
- Raw question results and run manifests are strict, self-hashed Pydantic records with nested instance revalidation. Metrics round-trip through serialized validation before recomputing summaries; `model_copy`, `model_construct`, numeric strings, NaN/Inf, forged `RunPoint`, question omission, summary spoofing, and relation tampering fail closed.

## Fair policies and information boundaries

- Every policy step receives the exact tuple returned by the same `MemoryEnvironment.valid_actions`; the runner accepts only the identical `ActionInstance` object from that tuple.
- Question-only controllers receive only question, options, remaining budget, and cost preference. Text-Adaptive controllers additionally receive only Gist text and generic action-history types. Neither controller receives legal actions, candidate identifiers, frontier/fidelity attributes, multimodal evidence, or their existence set; an internal canonical selector maps predicted `ActionType` to one exact legal instance or deterministic STOP.
- Prompt/VLM rationale remains private: it is absent from acquired evidence, final Answerer prompts, and raw records. Prompt cost is separately retained and included in the total.
- BC requires an actual Task 10 `BCPolicy` and matching checkpoint state/content. BC+DAgger additionally requires a sealed final stopped Task 11 manifest whose checkpoint artifact and policy identity match the actual adapter.

## Fixed traces and approved horizon ruling

- Uniform and full-Residual use the benchmark-authoritative temporal event order. They SEARCH, repeatedly take only legal CONTEXT actions until all events are exposed, then respectively VERIFY low visual or EXPAND residual for every event in temporal order, and STOP.
- The approved minimal Task 7 extension adds `AgentRunner(max_transitions=5)` while preserving five as the default through run, restore, and forced STOP.
- Rule, Prompt/VLM, Text-Adaptive, Question-only, BC, and BC+DAgger always use the five-transition Router horizon. Only Uniform and full-Residual derive a full-coverage horizon from authoritative event count; the manifest records both horizon and category.
- Ruling risk: longer horizons make fixed exhaustive baselines and adaptive Router policies structurally different. The paper tables must state that difference explicitly; otherwise accuracy/cost interpretation could incorrectly attribute horizon opportunity to routing quality.
- Real six-event synthetic traces prove all actions remain legal and exceed five transitions. Frame-, evidence-token-, and total-cost overruns still preserve the complete measured trajectory, mark the question invalid, and count it incorrect in primary/fixed-budget accuracy.

## Cost, utility, metrics, and taxonomy

- `CostBreakdown` retains base-memory amortization, environment acquisition, policy/router measurement, Prompt controller usage, and final Answerer usage. Total GPU/wall/frame/token/peak-memory and scalar cost are recomputed from those components.
- Utility uses the exact Task 9 formula `answer_score - lambda * total_cost / normalizer`, with `lambda` restricted to `{0.0, 0.1, 0.3, 1.0}` and normalization bound to the training artifact. The same Oracle path has zero regret.
- Primary accuracy and fixed-budget accuracy use every benchmark question as denominator, so invalid questions cannot be dropped. Valid-only accuracy and invalid rate are reported separately.
- Pareto is the stable non-dominated set maximizing accuracy and minimizing cost. Cost@Accuracy is the lowest finite total cost meeting the threshold, or explicit `None` when unreachable. Public cross-run APIs accept only fully validated `EvaluationRun` objects.
- Error signals are derived from actual trajectories plus sealed Oracle support/fidelity authority. Only valid incorrect questions receive one primary cause, with exact priority recall -> Answerer -> premature stop -> insufficient fidelity -> over-retrieval; correct and invalid questions have no primary cause, while secondary efficiency flags may remain multi-valued.

## Configuration and reproducibility

- `main_eval.yaml` lists the exact eleven policies, four total/frame/token budgets, three seeds, frozen preferences, shared-identity placeholders, reporting fields, and A800-training/V100-evaluation assignment.
- The strict loader validates the exact matrix; `evaluation_matrix` consumes all policy/seed/budget/preference/hardware/shared-identity fields, and `evaluate_run` applies Python, NumPy, and Torch seeds. GPU assignment is recorded but CPU synthetic tests never invoke real GPU/VLM/network work.

## Round 1 verification

- `D:\Anaconda\python.exe -m pytest tests/eval -q` -> 24 passed in 4.08s.
- `D:\Anaconda\python.exe -m pytest tests/router tests/eval -q` -> 128 passed, 1 skipped in 22.08s.
- `D:\Anaconda\python.exe -m pytest tests/agent tests/actions -q` -> 43 passed in 9.91s.
- `D:\Anaconda\python.exe -m pytest -q` -> 383 passed, 2 skipped in 63.24s; bounded 180-second rule not reached.
- `D:\Anaconda\python.exe -m compileall -q src tests` -> exit 0.
- `git diff --check` -> clean; Git emitted only LF-to-CRLF conversion warnings.
- `ruff check src/fidmem/agent/answerer.py src/fidmem/eval tests/eval` -> clean after mechanical fixes; the pre-existing compact Task 7 runner style was intentionally not reformatted.

## Residual scope

No real benchmark, VLM, GPU, or network execution was authorized for Task 12. Formal runs must replace configuration placeholders with sealed artifacts and supply authoritative cached observations/cost records. The approved fixed-baseline versus Router horizon distinction remains an interpretation caveat, not a hidden manifest difference.
