# Paper Experiment Execution Pack

This directory is the manual, fail-closed runbook for the Fidelity-Graded Video
Memory paper. The pack prepares execution; it does not contain production
results. Checked-in templates, `--check` output, mocks, fixtures, dry-runs, and
estimated resources remain engineering evidence.

## Frozen protocol

The paper hypothesis is unchanged: a learned cost-aware Router selects among
Gist, question-independent Residual, Context, question-level Visual, and STOP,
while observation models and the Answerer remain frozen. All policies share the
same atomic observations, Answerer template, scoring, cache graph, and maximum
budgets. Formal costs include offline construction, cache misses, online
actions, Answerer, and Router overhead.

The dependency DAG is:

```text
E00 Environment
  -> E01 Dataset/video-disjoint manifests
  -> E02 Production Authority seal
  -> E03 Production Canary
       -> E04 Oracle viability + Beam8 audit
       -> E05 Answerer stability
       -> E06 leakage + label audits
E04 + E05 + E06
  -> E07 Gist recall
  -> E08 fixed baselines
  -> E09 Rule/Prompt/Text-Adaptive baselines
  -> E10 BC Router training
  -> E11 BC evaluation
       -> E12 optional DAgger
       -> E13 main benchmarks
            -> E14 ablations
            -> E15 efficiency/cache/failure analysis
            -> E16 cross-dataset -> E17 optional cross-model
```

Every arrow is enforced by hashed gate artifacts. A missing, failed, tampered,
or protocol-mismatched gate returns `FAIL_CLOSED` before creating a run.

## Prerequisites

1. Use Python 3.11+ with the project dependencies installed. The bash wrappers
   add `src/` to `PYTHONPATH`; an editable install is optional.
2. Install `nvidia-smi` on GPU hosts and export the approved inference backend
   identity as `FIDMEM_INFERENCE_BACKEND`.
3. Freeze dataset files, manifests, local checkpoints or provider-backed
   immutable model revisions, prompts, configs, and the target runtime.
4. Fill only the config for the experiment being prepared. Checked-in values
   containing `RESEARCH_OWNER_DECISION_REQUIRED` are intentionally invalid.
5. Seal Production Authority on the actual target GPU host. E03 and every later
   formal stage reference that sealed file instead of repeating model/dataset
   identity.

The registry is `configs/experiments/registry.yaml`. Validate it without model
execution:

```bash
PYTHONPATH=src python -m fidmem.experiments.pack_cli --validate-registry
PYTHONPATH=src python -m fidmem.experiments.pack_cli --list
```

## First-time Experiment Stack v1 asset setup

E00-E17 remain the only experiment DAG. Asset preparation is a pre-Authority
input to E01/E02, not a second runner or gate framework. Copy `.env.example`,
set storage roots for the current host, and provide only a researcher-approved
LongTVQA raw-video root:

```bash
export FIDMEM_DATA_ROOT=/data/fidmem/datasets
export FIDMEM_MODEL_ROOT=/data/fidmem/models
export FIDMEM_CACHE_ROOT=/data/fidmem/cache
export FIDMEM_ARTIFACT_ROOT="$PWD/artifacts"
export FIDMEM_GIT_COMMIT="$(git rev-parse HEAD)"
export FIDMEM_LONGTVQA_VIDEO_ROOT=/approved/LongTVQA/videos

bash scripts/setup/01_resolve_stack_assets.sh --check
bash scripts/setup/01_resolve_stack_assets.sh
bash scripts/setup/02_download_models.sh --check
bash scripts/setup/02_download_models.sh --resume
bash scripts/setup/03_verify_models.sh --verify-only
bash scripts/setup/04_download_longtvqa_metadata.sh --check
bash scripts/setup/04_download_longtvqa_metadata.sh --resume
bash scripts/setup/05_verify_longtvqa_metadata.sh --check
bash scripts/setup/06_verify_longtvqa_videos.sh --check
```

`--check` performs metadata, revision, dependency, storage, permission, and
cache checks and never calls `snapshot_download`. `--dry-run` prints the known
plan without downloading. `--resume` uses the Hugging Face snapshot/cache
mechanism. `--verify-only` hashes existing local files. Logical roles map to
five physical assets: BGE-M3 is shared by Gist text and embedding, and one
Qwen3-VL snapshot is shared by Residual and Visual.

The checked-in lock is only a candidate: it contains four metadata-resolved
commit SHAs, leaves Qwen3-VL unresolved, and contains no `VERIFIED` asset.
Running a resolver is not verification. E02 re-hashes every local snapshot and
accepts only a fully `VERIFIED` lock.

After approved raw videos are present, generate the automatic Source Gate and
the deterministic 100-item human audit manifest:

```bash
bash scripts/setup/06_verify_longtvqa_videos.sh \
  --output "$FIDMEM_ARTIFACT_ROOT/longtvqa-source-gate"
# A real researcher completes the generated audit; do not edit its status.
export FIDMEM_LONGTVQA_HUMAN_AUDIT_RESULT=/approved/audit-result.json
# First freeze configs/experiment_stacks/longtvqa_split_policy.yaml.
bash scripts/setup/07_build_longtvqa_manifests.sh \
  --output "$FIDMEM_ARTIFACT_ROOT/dataset-freeze"
export FIDMEM_E01_RESULTS_ROOT="$FIDMEM_ARTIFACT_ROOT/dataset-freeze"
bash scripts/setup/08_build_authority_draft.sh --check
bash scripts/setup/08_build_authority_draft.sh
```

The Source Gate requires all video/QA/subtitle mappings, readable videos and
durations, content SHA-256, no duplicate identities, at least 20 deterministic
episode decodes, constructible QA/actions, and a bound 100-item human audit.
The metadata repository is never described as the raw-video dataset.

Keep `FIDMEM_ARTIFACT_ROOT` inside the checkout because the approved Authority
validator binds manifest/evidence paths beneath the repository root. Dataset,
model, and cache roots may remain on external mounted storage.

Current prompt/config audits intentionally block sealing: the complete Gist
prompt, Answerer template, segmentation policy, frame-sampling policy, and all
per-model runtime/decode settings remain
`RESEARCH_OWNER_DECISION_REQUIRED`. Retrieval top-5 with 0.6/0.4 weights and
Visual budgets 12/32 are frozen from the existing implementation. The Draft
keeps runtime absent and `production_ready=false`; E02 probes and seals runtime
only on the selected GPU host.

## Manual execution contract

Every numbered script accepts:

```text
--check
--resume
--config PATH
--gpus 0[,1...]
--run-id ID
--output-root PATH
--gate-root PATH
--help
```

Always run `--check` first:

```bash
bash scripts/experiments/03_canary.sh --check --gpus 0 --run-id canary-v1
bash scripts/experiments/03_canary.sh --gpus 0 --run-id canary-v1
bash scripts/experiments/03_canary.sh --resume --gpus 0 --run-id canary-v1
```

`--check` validates registry consistency, config composition, unresolved
required fields, Authority, upstream gates, explicit GPU selection and free
VRAM, disk, environment variables, paths, source identity, output identity,
and Router cache separation. It never invokes `execution.command`.

The configured executor is invoked as an argument list without a shell. It
receives `CUDA_VISIBLE_DEVICES`, `FIDMEM_RUN_DIR`, `FIDMEM_EXPERIMENT_ID`, and
`FIDMEM_CONFIG_SNAPSHOT`. It must write only the declared outputs below
`$FIDMEM_RUN_DIR`. It must never select a replacement model, dataset, GPU, or
checkpoint when an input is missing.

## Run directories and lifecycle

Formal runs live under:

```text
artifacts/experiments/<experiment-id>/<run-id>/
  metadata.json
  config.snapshot.json
  upstream-gates.snapshot.json
  stdout.log
  stderr.log
  STATUS.json
  PREPARED
  RUNNING
  COMPLETED | FAILED
  results/
```

`metadata.json` binds experiment, protocol, config hash, Authority hash,
selected GPUs, exact executor command, invocation, and upstream gate hashes.
Resume requires the same identity. A completed run is a no-op on resume. A
failed run never receives `COMPLETED`.

## Recording gates

Gate thresholds are frozen in `configs/experiments/gates/`. Build a gate only
from a real result artifact. For example:

```bash
PYTHONPATH=src python -m fidmem.experiments.gate_cli \
  --gate production_canary \
  --experiment E03 \
  --run-id canary-v1 \
  --protocol-version fidelity-memory-paper-v1 \
  --config-sha256 <64-hex-from-preflight> \
  --authority-sha256 <sealed-authority-hash> \
  --result artifacts/experiments/E03/canary-v1/results/canary_validation.json \
  --thresholds configs/experiments/gates/production_canary.yaml \
  --output artifacts/experiment-gates/production_canary.json
```

An existing different gate file is never overwritten. Archive the old gate and
create a new protocol/run identity explicitly if the approved protocol changes.
Oracle viability and Gist recall currently contain unresolved thresholds and
therefore cannot pass until the research owner freezes them before inspecting
formal results.

## Observation/training separation

E03 is the only registered production observation-generation stage. E04-E09
consume the frozen observation graph. E10 and E12 are `router_training` stages;
their configs declare `may_generate_observations: false`, and preflight requires
an existing frozen observation cache. Router training must fail rather than
regenerate Residual or Visual observations.

Ground truth is available only to E04/E06 and formal evaluation scoring paths.
Router execution features must never read labels or future-only information.

## Results

Machine-readable contracts are in
`configs/experiments/result_schemas.yaml`. Non-result skeletons are under
`configs/experiments/result_templates/`; `template_only: true` makes the formal
validator reject them. Future real results can be compiled with:

```bash
PYTHONPATH=src python scripts/experiments/analysis/export_tables.py \
  artifacts/experiments/E13/<run>/results/main_summary.json \
  --output artifacts/paper-tables/table_data.json
```

The compiler prepares data for accuracy/cost tables, Pareto curves, Oracle
headroom, Router metrics, cache savings, ablations, and generalization. It does
not fabricate missing rows, confidence intervals, or plots.

## Manual order and stop rules

Run E00 through E11 in order, recording each produced gate. E12 is optional and
must not block the BC-only E13 run; include BC+DAgger in a later frozen policy
manifest only if its gate passes. E13 precedes E14-E17. Stop immediately when a
required gate fails. Do not lower thresholds after seeing results; create a
new approved protocol version if a prospective change is scientifically
necessary.

Current unresolved owner choices are the Qwen3-VL immutable revision, approved
raw LongTVQA videos, video-level split assignments/seeds, completed human
timestamp audit, the prompt/config items listed above, the concrete frozen
Transformers backend factory/request materializer, target GPU/runtime, Oracle
headroom/missing-rate thresholds, Gist recall threshold, transfer protocol,
and second backbone.
