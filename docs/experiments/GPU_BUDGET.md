# GPU Budget Ledger

No GPU-hour number is invented in this pack. Resource estimates remain unknown
until E03 records measured throughput. The approved global ceiling remains 800
A800 GPU-hours and 200 V100 GPU-hours; the core target is 600/150.

| ID | Resource class | Scale driver | Initial GPU estimate | Estimation source after gate |
|---|---|---|---|---|
| E00 | CPU metadata | one host | not applicable | n/a |
| E01 | CPU/I/O | dataset bytes/manifests | not applicable | measured wall time |
| E02 | target GPU metadata | one seal | not applicable | n/a |
| E03 | observation inference | 10–20 questions | UNKNOWN | measured Canary CostRecords |
| E04 | Oracle/Answerer | 100 questions + 20 audit | UNKNOWN | Canary per-action cost and observed graph |
| E05 | Answerer | 300 executions | UNKNOWN | Canary Answerer throughput |
| E06 | CPU/light embedding | manifests and label sample | UNKNOWN if embedding GPU used | measured audit |
| E07 | retrieval/Answerer | frozen dev | UNKNOWN | Canary + Oracle throughput |
| E08 | evaluation | five fixed policies | UNKNOWN | shared-cache baseline pilot |
| E09 | evaluation/controller | three adaptive policies | UNKNOWN | E08 plus controller measurements |
| E10 | Router training | 2k–4k trajectories, 3 seeds | UNKNOWN | Oracle graph size and BC pilot |
| E11 | Router evaluation | 3 seeds | UNKNOWN | E10/E08 throughput |
| E12 | Router training | two rounds, conditional third | UNKNOWN | BC and round-1 measurements |
| E13 | benchmark evaluation | 3 datasets × policies × 3 seeds | UNKNOWN | M1/M2 measured throughput |
| E14 | ablations | registered matrix | UNKNOWN | E13 per-policy throughput |
| E15 | CPU analysis | raw result rows | not applicable | measured CPU time |
| E16 | transfer evaluation | owner-frozen protocol | UNKNOWN | E13 throughput |
| E17 | second backbone | owner-frozen protocol | UNKNOWN | separate-backbone Canary required |

Each formal run appends measured GPU seconds, wall seconds, frames, visual/text
tokens, peak memory, cache status, and device identity. Later budget tooling may
project larger runs only from these measured rows; projected values must remain
separate from actual usage.
