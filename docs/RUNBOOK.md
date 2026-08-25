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
