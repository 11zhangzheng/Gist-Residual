# Experiment Tracker

**Budget ceiling:** 800 A800 GPUh / 200 V100 GPUh  
**Planned core:** 600 A800 GPUh / 150 V100 GPUh  
**Status values:** TODO / RUNNING / DONE / FAILED / CUT

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | 基础链路与成本守恒 | mock models, tiny video | fixture | state validity, exact cost sum, resume | MUST | TODO | 0 GPU；未通过不得接真实模型 |
| R002 | M0 | 测单动作真实成本 | fixed G/R/V paths, 50 QA | LongRoute pilot | mean/P90 GPU-s per action, frames, tokens | MUST | TODO | A800垂直切片 |
| R003 | M0 | 外推 Oracle 总预算 | Beam Oracle, 100 QA | LongRoute pilot | GPUh/question, missing observations, correct-path rate | MUST | TODO | 决定2k或4k训练规模 |
| R004 | M0 | Answerer标签稳定性 | 3 repeats, 100 states | pilot states | answer flip rate | MUST | TODO | >2%才启用边界多数投票 |
| R005 | M0 | 数据泄漏审计 | id+SHA256+embedding | train vs all eval | confirmed/near duplicate count | MUST | TODO | confirmed必须为0 |
| R006 | M0 | Beam teacher审计 | Beam8 vs exhaustive/UCS | tractable pilot subset | optimal-path hit, cost gap | MUST | TODO | gap明显才改搜索 |
| R010 | M1 | 固定输入基线 | uniform frames | core dev subsets | accuracy, total cost | MUST | TODO | 同Answerer和预算 |
| R011 | M1 | 最低层基线 | Gist-only | core dev subsets | accuracy, base/online cost | MUST | TODO | 主Gist无VLM |
| R012 | M1 | 固定中保真路径 | Gist→Residual | core dev subsets | accuracy, residual hit/cost | MUST | TODO | 共享原子缓存 |
| R013 | M1 | 固定视觉路径 | Gist→Visual-low/high | core dev subsets | accuracy, frame/GPU cost | MUST | TODO | 两预算点 |
| R014 | M1 | 全量记忆基线 | full Residual upfront | core dev subsets | accuracy, preprocessing cost | MUST | TODO | 全离线成本计入 |
| R015 | M1 | 文本迁移基线 | Text-Adaptive Router | core dev subsets | accuracy, cost, stop rate | MUST | TODO | 只用ASR/Gist文本状态 |
| R016 | M1 | Prompt Agent基线 | same-action Prompt Controller | core dev subsets | accuracy, cost, policy latency | MUST | TODO | reasoning不传给Answerer |
| R017 | M1 | Gist召回校准 | fusion weights, K=3/5 | LongRoute dev | evidence recall@K, offline cost | MUST | TODO | top-5不足则No-Go |
| R020 | M2 | 生成训练teacher | canonical+4 λ labels | LongRoute train 2k | correct-path %, cost distribution | MUST | TODO | R003通过后启动 |
| R021 | M2 | BC训练重复1 | BC seed 11 | LongRoute train/dev | action acc, utility, regret | MUST | TODO | V100 |
| R022 | M2 | BC训练重复2 | BC seed 22 | LongRoute train/dev | action acc, utility, regret | MUST | TODO | V100 |
| R023 | M2 | BC训练重复3 | BC seed 33 | LongRoute train/dev | action acc, utility, regret | MUST | TODO | V100 |
| R024 | M2 | DAgger round 1 | best BC + cached labels | LongRoute train/dev | utility gain, cost regret | MUST | TODO | 不调用新VLM |
| R025 | M2 | DAgger round 2 | round1 policy | LongRoute train/dev | utility gain, cost regret | MUST | TODO | 默认最后一轮 |
| R026 | M2 | DAgger round 3门控 | round2 policy | LongRoute train/dev | gain vs stopping threshold | CONDITIONAL | TODO | 仅改善≥门槛运行 |
| R027 | M2 | Question-only隔离 | no evidence/history | LongRoute dev | utility, accuracy, cost | MUST | TODO | 检验顺序状态必要性 |
| R030 | M3 | 主数据集1 | all core systems | Video-MME long | accuracy-cost frontier | MUST | TODO | 主表/主图 |
| R031 | M3 | 主数据集2 | all core systems | LVBench | accuracy-cost frontier | MUST | TODO | 主表/主图 |
| R032 | M3 | 主数据集3 | all core systems | LongVideoBench | accuracy-cost frontier | MUST | TODO | 主表/主图 |
| R033 | M3 | Router策略统计 | BC+DAgger | all 3 core sets | actions, premature stop, overhead | MUST | TODO | Router overhead<5% |
| R034 | M3 | Oracle上限 | canonical Oracle | core evaluable subset | accuracy, cost, policy regret | MUST | TODO | 同Answerer |
| R035 | M3 | λ条件控制 | λ=0/0.1/0.3/1.0 | all 3 core sets | frontier coverage, monotonicity | MUST | TODO | 归一化常数冻结 |
| R040 | M3 | Memory层消融 | no Residual | Video-MME long, LVBench | accuracy, cost | MUST | TODO | 其余配置固定 |
| R041 | M3 | Context消融 | no Context | Video-MME long, LVBench | temporal category acc, cost | MUST | TODO | 无增益则删模块 |
| R042 | M3 | Visual消融 | no Visual | Video-MME long, LVBench | detail acc, cost | MUST | TODO | 检验最高保真必要性 |
| R043 | M3 | Router辅助头消融 | no sufficiency / no cost-to-go | LongRoute dev + one core set | accuracy, regret, stop errors | MUST | TODO | 两变体共享观察 |
| R044 | M3 | Gist构建消融 | ASR+embedding / main / Gist+ | one core set | recall, accuracy, preprocessing cost | MUST | TODO | Gist+用7B VLM |
| R045 | M3 | 搜索规模敏感性 | K=3/5, depth=3/5 | LongRoute dev | recall, utility, cost | NICE | TODO | 预算不足先CUT |
| R050 | M4 | 缓存摊销主实验 | cache/no-cache/full-upfront | multi-query videos | cost at Q=1/2/4/8/16, accuracy | MUST | TODO | 3个问题顺序seed |
| R051 | M4 | 错误归因 | full router vs Oracle | all core errors | five-category distribution | MUST | TODO | Router与非Router分开 |
| R052 | M4 | Residual质量审计 | 50 events | sampled events | proposition overlap, new detail rate | MUST | TODO | overlap>30%先改Prompt |
| R053 | M4 | 统计聚合 | 3 Router seeds | all main runs | mean, std, paired bootstrap CI | MUST | TODO | VLM输出固定 |
| R054 | M4 | 空缓存复现 | frozen core config | representative core subset | result/cost delta | MUST | TODO | 独立run directory |
| R055 | M4 | 预算与完整性审计 | all run manifests | all | actual GPUh, missing records, invalid runs | MUST | TODO | 超预算结果不进入主表 |
| R056 | M4 | 主表/主图导出 | verified results only | all | table schema, figure data hashes | MUST | TODO | 不填造结果 |
| R060 | Optional | Offline RL | cached discrete Q-learning | LongRoute train/dev | gain over DAgger | NICE | TODO | 核心结论成立且有余量 |
| R061 | Optional | 第二backbone鲁棒性 | alternate 7B–8B VLM | one core set | Pareto transfer | NICE | TODO | 不占核心预算 |
| R062 | Optional | 新benchmark | full method + strongest baselines | Video-MME-v2 | accuracy-cost | NICE | TODO | 数据稳定后再决定 |

## Decision Log

| Date | Gate | Evidence | Decision | Owner |
|---|---|---|---|---|
| 2026-08-21 | Scope freeze | Approved design spec | Offline LVQA only; freeze Answerer; BC+DAgger core | Project lead |

## Budget Ledger

| Milestone | Planned A800 GPUh | Actual A800 GPUh | Planned V100 GPUh | Actual V100 GPUh | Remaining After Planned Milestone |
|---|---:|---:|---:|---:|---:|
| M0 | 100 | 0 | 30 | 0 | 700 / 170 |
| M1 | 120 | 0 | 20 | 0 | 580 / 150 |
| M2 | 180 | 0 | 70 | 0 | 400 / 80 |
| M3 | 150 | 0 | 20 | 0 | 250 / 60 |
| M4 | 50 | 0 | 10 | 0 | 200 / 50 |
| Reserve | 200 | 0 | 50 | 0 | 0 / 0 |
