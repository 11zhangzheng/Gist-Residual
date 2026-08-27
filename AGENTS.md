# Repository Research Rules

## Research invariants

- Preserve the Gist → Residual → Raw Visual fidelity hierarchy, the learned cost-aware Memory Router, accuracy–cost Pareto evaluation, question-independent Residual cache, and frozen observation models/Answerer.
- Do not change the paper hypothesis to accommodate a local result.

## Evidence classes

- Keep engineering evidence, production evidence, and paper evidence explicitly separate.
- Mock, synthetic, fixture, placeholder, and dry-run artifacts are engineering evidence only.
- Estimated cost is never measured cost. Production evidence requires raw measured CostRecord data and complete provenance.

## Production gate

- A validated, sealed Production Authority is required before production observation generation.
- Missing fields, placeholder identities, invalid hashes, identity mismatches, or an unsealed Authority fail closed.

## Leakage and cache isolation

- Do not tune on final/test splits; split videos by `video_id` by default.
- Question-level Visual cache never crosses questions. Only event-level, question-independent Residual may be reused across questions.
- Ground truth is available only on explicit Oracle/evaluation paths. Deployable Router features never read ground truth or future-only information.

## Git safety

- Start by reading `git status`, `git diff`, and `HEAD`; preserve all user modifications and unknown files.
- Do not reset, clean, stash user changes, checkout over files, rebase, commit, push, merge, or publish unless the user explicitly requests it.

## Required context

Before formal research work, read the files that exist among `AGENTS.md`, `refine-logs/EXPERIMENT_PLAN.md`, `refine-logs/EXPERIMENT_TRACKER.md`, `refine-logs/EXPERIMENT_CODE_REVIEW.md`, `docs/RUNBOOK.md`, and relevant frozen configs.

## Test and reporting discipline

- Every new production invariant requires a test. Never delete tests, weaken assertions, or relax fail-closed gates to obtain a pass.
- Completion reports distinguish changes, omissions, engineering/production/paper evidence, tests, current experiment gate, Git status, and remaining blockers.
