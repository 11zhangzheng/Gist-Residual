# Manual GPU Runbook

## Environment and paths

Activate the approved environment and verify the repository checkout. The
wrapper adds `src/` to `PYTHONPATH`.

```bash
cd /path/to/Gist
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvidia-smi
export FIDMEM_INFERENCE_BACKEND=<approved-backend-identity>
export FIDMEM_DATA_ROOT=/data/fidmem/datasets
export FIDMEM_MODEL_ROOT=/data/fidmem/models
export FIDMEM_CACHE_ROOT=/data/fidmem/cache
export FIDMEM_ARTIFACT_ROOT="$PWD/artifacts"
export FIDMEM_GIT_COMMIT="$(git rev-parse HEAD)"
```

Mount dataset and model storage read-only where practical. Put paths and
immutable identities in the selected experiment YAML or sealed Authority. Do
not edit Python source, download a replacement checkpoint, or substitute a
dataset when preflight fails.

Before E01/E02 on a new host, follow the first-time setup in `README.md`.
Resolve metadata, download by immutable commit with `--resume`, then verify.
Residual/Visual must share one Qwen3-VL directory; Gist-text/embedding must
share one BGE-M3 directory. Do not copy either snapshot or obtain LongTVQA
videos from an unapproved third-party source.

The Linux order immediately before the first Canary is:

```bash
bash scripts/setup/01_resolve_stack_assets.sh --check
bash scripts/setup/01_resolve_stack_assets.sh
bash scripts/setup/02_download_models.sh --resume
bash scripts/setup/03_verify_models.sh --verify-only
bash scripts/setup/04_download_longtvqa_metadata.sh --resume
bash scripts/setup/05_verify_longtvqa_metadata.sh --check
bash scripts/setup/06_verify_longtvqa_videos.sh --check
bash scripts/setup/07_build_longtvqa_manifests.sh --check
bash scripts/experiments/00_environment.sh --check --run-id environment-v1
bash scripts/experiments/00_environment.sh --run-id environment-v1
# Record environment_ready from E00 results before checking E01.
bash scripts/experiments/01_dataset.sh --check --run-id longtvqa-freeze-v1
bash scripts/experiments/01_dataset.sh --run-id longtvqa-freeze-v1
# Record dataset_frozen from results/source_gate.json before checking E02.
bash scripts/setup/08_build_authority_draft.sh --check
bash scripts/setup/08_build_authority_draft.sh
bash scripts/experiments/02_authority.sh --check --gpus 0 --run-id authority-v1
bash scripts/experiments/02_authority.sh --gpus 0 --run-id authority-v1
# Record authority_sealed from results/authority_gate_result.json before E03.
bash scripts/experiments/03_canary.sh --check --gpus 0 --run-id canary-v1
bash scripts/experiments/03_canary.sh --gpus 0 --run-id canary-v1
```

Each command must pass before continuing. The split policy, human audit,
prompt/runtime settings, provider request manifest/backend factory, source
commit, and Authority paths must be frozen before their corresponding
experiment `--check` can pass.
Use `python -m fidmem.experiments.gate_cli` exactly as documented in the
README's “Recording gates” section, with each preflight config hash and the
corresponding frozen threshold file. Never hand-write a PASS gate artifact.

## GPU selection and safety

Always pass physical indices explicitly:

```bash
bash scripts/experiments/03_canary.sh --check --gpus 0 --run-id canary-v1
```

Preflight queries visible GPU UUID/name/free VRAM and prints the selected
device. It fails if an index is absent or free VRAM is below the config. Inspect
`nvidia-smi` yourself for meaningful existing workloads before launch. The pack
never kills another process and never selects a different GPU automatically.

## Launch and monitoring

Use `tmux` or `screen` on remote hosts:

```bash
tmux new -s fidmem-canary
bash scripts/experiments/03_canary.sh --gpus 0 --run-id canary-v1
tail -f artifacts/experiments/E03/canary-v1/stdout.log
```

The executor receives `CUDA_VISIBLE_DEVICES` and writes stdout/stderr into the
immutable run directory. Monitor GPU processes separately with `nvidia-smi`.

## Interruption and resume

Interrupt with Ctrl-C or the scheduler's normal termination mechanism; do not
delete the run directory. Resume with the identical config, Authority, gates,
GPU selection, executor command, and run ID:

```bash
bash scripts/experiments/03_canary.sh --resume --gpus 0 --run-id canary-v1
```

A changed config hash, Authority, command, GPU selection, or upstream gate hash
fails before execution. Completed observations must be reused and not charged
again. A `FAILED` marker is not evidence of completion.

## Failure recovery

1. Read `STATUS.json`, `stderr.log`, and the executor's item-level state.
2. Correct only the external cause or approved config input.
3. If identity changed, choose a new run ID; do not mutate a completed run.
4. If an upstream gate is failed, stop. Do not lower its threshold.
5. Never run Router training when the frozen observation cache is missing.

## Artifact archiving

After completion, copy the entire run directory, gate artifacts, sealed
Authority, dataset/selection manifests, and Authority-bound cache manifest.
Verify SHA-256 after transfer. Keep raw provider responses and CostRecords; a
summary without raw provenance is not production or paper evidence.
