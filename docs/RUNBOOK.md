# fidmem reproducibility runbook

This runbook is the executable M0–M4 path for the first offline pilot. Every
command accepts `--config`, `--run-id`, `--dry-run`, and `--resume`; artifacts
are written below `artifacts/runs/<run-id>/`.

## M0 — environment and identity

```powershell
python -m fidmem.cli evaluate --config configs/experiment/pilot.yaml --run-id pilot --dry-run
python -m fidmem.cli report --config configs/experiment/pilot.yaml --run-id pilot
```

The dry run must remain below 800 A800 GPU-hours and 200 V100 GPU-hours. Record
the Git commit, config hash, video hash, cache namespace, and model revisions in
the resulting report before using real models.

## M1 — ingest and cheap memory

```powershell
python -m fidmem.cli ingest --config configs/experiment/pilot.yaml --run-id pilot --video data/tiny_video.mp4
python -m fidmem.cli build-gist --config configs/experiment/pilot.yaml --run-id pilot --resume
```

Ingest is CPU/ffmpeg work. The base Gist path does not call a VLM; only the
cheap text and visual encoders are allowed.

## M2 — observations and Oracle

```powershell
python -m fidmem.cli build-observations --config configs/experiment/pilot.yaml --run-id pilot --resume
python -m fidmem.cli build-oracle --config configs/experiment/pilot.yaml --run-id pilot --resume
```

A800 is reserved for VLM/Answerer observations and V100 for embeddings and
Router work. Observation outputs are content-addressed. Two workers must never
write the same unclaimed cache key; use separate run IDs or the RunStore claim
protocol.

## M3 — Router training and correction

```powershell
python -m fidmem.cli train-router --config configs/experiment/train_bc.yaml --run-id bc-smoke --resume
python -m fidmem.cli run-dagger --config configs/experiment/dagger.yaml --run-id dagger-smoke --resume
```

The Router is the only trainable component. Checkpoints must preserve config,
dataset, Git, device, and RNG identity. Stop DAgger after round 2 unless the
registered utility/cost-regret thresholds authorize round 3.

## M4 — evaluation and reporting

```powershell
python -m fidmem.cli evaluate --config configs/experiment/main_eval.yaml --run-id main --dry-run
python -m fidmem.cli report --config configs/experiment/main_eval.yaml --run-id main
```

All policies use the same Answerer template, cache graph, frame/token budgets,
and scoring rule. Do not train on formal benchmark questions. Stop the pilot
if leakage audit, answerer identity, cache identity, or budget checks fail.

To resume after interruption, rerun the same command with `--resume`. Completed
items remain complete and are not charged again; incomplete items are retried
only after their lease/state is reclaimed.

## Production Authority gate and R002 Canary bootstrap

Production evidence is a separate, fail-closed namespace. The checked-in
configs/production/authority.example.yaml and
configs/production/canary.example.yaml files are deliberately invalid
templates; neither is evidence and neither authorizes a provider or model.

First fill an Authority draft only with approved, immutable dataset manifests,
model revisions, prompt/config hashes, the canonical cost contract, and the
current repository identity. A local/HF model must bind verified checkpoint
bytes. A hosted model must bind provider-backed immutable revision evidence;
providers without verifiable immutable revision evidence fail closed. Validate
without executing a model:

~~~powershell
python -m fidmem.cli authority-validate --draft path/to/authority.draft.yaml --project-root .
~~~

The example template must exit 2 with stable issue codes. Validation probes the
current repository and runtime. Seal only on the actual target GPU host after
all issues are resolved:

~~~powershell
python -m fidmem.cli authority-seal --draft path/to/authority.draft.yaml --project-root . --output path/to/PRODUCTION_AUTHORITY.json
~~~

Sealing writes canonical JSON atomically and only after fresh repository,
CUDA/PyTorch/backend, GPU UUID, manifest, model, prompt, config, and cost-schema
checks pass. Semantic identities use UTF-8 canonical JSON with sorted keys,
compact separators, Unicode preserved, NaN rejected, and no BOM/trailing
newline. Prompt SHA-256 hashes exact raw UTF-8 prompt bytes. `authority_sha256`
is semantic identity, `authority_file_sha256` is exact file identity, and the
path is non-semantic metadata. A failed seal preserves any existing output.

Provider output for a Canary must already contain the sealed Authority hash,
immutable provider/model/revision/device/config identity, raw response plus raw
response hash, and measured CostRecord values. Import and report it with:

~~~powershell
python -m fidmem.cli build-observations --config configs/base.yaml --artifact-root artifacts --run-id R002-production-canary --production-authority path/to/PRODUCTION_AUTHORITY.json --input-jsonl path/to/provider-observations.jsonl --resume
python -m fidmem.cli report --config configs/base.yaml --artifact-root artifacts --run-id R002-production-canary --production-authority path/to/PRODUCTION_AUTHORITY.json
~~~

Production artifacts live below
artifacts/production/<authority_sha256>/runs/<run-id>/. Engineering commands
always write below artifacts/development/runs/<run-id>/. Supplying an Authority
to dry-run, ingest, build-gist, or smoke paths is rejected rather than promoted.
The importer validates an immutable generation and atomically switches
CURRENT.json only after every artifact is complete. A failure leaves the prior
generation byte-identical. Report aggregation resolves one CURRENT generation
and checks the single Authority before reading totals.

A no-op resume does not rewrite or recharge completed observations. Production
import/resume uses AuthorityBoundCache and never development cache. Event-level
amortizable observations exclude question_id from their cache key; question-
level visual verification includes it and therefore cannot cross questions.
Dry-run estimates, mocks, fixtures, synthetic observations, invalid templates,
and engineering smoke output must never be counted as production or paper
evidence. Canary accuracy is not a pass criterion. Do not expand to the
100-question Oracle pilot unless provenance, cost reconciliation, cache
isolation, and resume validation all pass.
