# Experiment Matrix

All scales and resource classes below come from the approved protocol. No GPU
hours are estimated before a measured Canary.

| ID | Purpose | Dataset / approximate scale | GPU | Dependencies | Required gate | Expected output | Paper role | Status |
|---|---|---|---|---|---|---|---|---|
| E00 | Environment/source snapshot | none | no | none | none | environment identity | reproducibility | PREPARED, not run |
| E01 | Dataset/manifests | LongTVQA frozen metadata + approved raw videos | no | E00 | environment | manifests + selections + Source Gate | data protocol | WIRED; BLOCKED: verified assets/raw videos/human audit/split freeze |
| E02 | Authority seal | Stack v1 verified lock + frozen identities | target GPU metadata | E01 | environment, dataset | sealed Authority | provenance | WIRED; BLOCKED: asset/prompt/config/runtime freeze |
| E03 | Production Canary | 10–20 real LongTVQA questions | yes | E02 | Authority | canonical provider JSONL/observations/cost/validation | gate only | CONTRACT WIRED; BLOCKED: frozen Transformers backend/request manifest |
| E04 | Oracle + Beam audit | 100 questions; 20 exhaustive | yes | E03 | Canary | trajectories/headroom/Beam report | Oracle figure | BLOCKED |
| E05 | Answerer stability | 100 states × 3 | yes | E03 | Canary | raw answers/flip rates | label reliability | BLOCKED |
| E06 | Leakage/label audit | all groups; ≥100 LongRoute labels | optional/light | E03 | Canary | audit reports | integrity | BLOCKED |
| E07 | Gist recall | frozen dev | yes | E04–E06 | all M0 gates | retrieval JSONL | retrieval diagnostic | BLOCKED |
| E08 | Fixed baselines | frozen dev | yes | E07 | Gist recall | fixed predictions/costs | fixed baselines | BLOCKED |
| E09 | Adaptive baselines | frozen dev | yes | E08 | Gist recall | Rule/Prompt/Text-Adaptive | controller isolation | BLOCKED |
| E10 | BC training | 2k–4k only after Oracle decision; 3 seeds | yes | E09 | M0/M1 gates | checkpoints/history | learned Router | BLOCKED |
| E11 | BC evaluation | frozen dev, 3 seeds | yes | E10 | BC gate | raw BC evaluation | M2 gate | BLOCKED |
| E12 | Optional DAgger | 2 rounds, conditional 3rd | yes | E11 | BC evaluation | generations/checkpoints | optional correction | BLOCKED |
| E13 | Main benchmarks | Video-MME long, LVBench, LongVideoBench; 3 seeds | yes | E11 | BC/leakage/label | predictions/cost/Pareto | main C1 | BLOCKED |
| E14 | Ablations | frozen benchmark matrix | yes | E13 | main benchmark | ablation matrix | ablation table | BLOCKED |
| E15 | Efficiency/cache | Q={1,2,4,8,16} | no | E13 | main benchmark | cost/cache/Pareto data | main C2 | BLOCKED |
| E16 | Cross-dataset | owner-frozen transfer | yes | E13 | main benchmark | transfer results | appendix | BLOCKED/optional |
| E17 | Cross-model | second frozen backbone | yes | E16 | main benchmark | model-transfer results | appendix | BLOCKED/optional |
