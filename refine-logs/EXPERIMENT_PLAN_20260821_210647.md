# 实验计划

**问题：** 长视频问答系统往往承担高额的全量离线理解成本，Prompt/VLM Agent 的顺序控制又昂贵且不稳定。  
**方法主旨：** 用最低成本 Oracle trajectory 监督一个独立轻量 Router，在 Gist–Residual–Raw Visual 间执行预算条件化的顺序路由，以更低全口径成本获取足够证据。  
**日期：** 2026-08-21  
**设计规范：** `docs/superpowers/specs/2026-08-21-fidelity-graded-video-memory-design.md`

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1：学习式成本感知 Router 获得更优准确率–全口径成本 Pareto 前沿 | 隔离并证明真正的方法贡献是控制策略，而非更强 Answerer、更多帧或 Prompt 工程 | 同一冻结 Answerer、同一原子观察和统一 Prompt 下，至少两个小时级 benchmark 上 Pareto dominance；等准确率成本降低≥40%，或等成本准确率提高≥2点；优于同 action space Prompt Controller | B1, B2, B5 |
| C2：按需 Fidelity 升级与事件级持久缓存降低首次及多查询成本 | 证明 Memory 不只是一次性检索技巧，而是能沉淀复用的增量记忆 | `Q=1` 与 `Q={2,4,8,16}` 全口径成本；`Q>=8` 边际成本显著下降且 accuracy 不降；问题级 Visual cache 不计入复用收益 | B1, B3, B4 |
| Anti-claim：收益来自输入更多证据、更长 Prompt 或不同 Answerer | 保证实验 attribution 可成立 | 统一 Answerer checkpoint/template；固定 frame/token budget 对比；报告每题真实 evidence 长度；所有策略共享原子观察 cache | B1, B2 |
| Anti-claim：Beam Oracle 标签不可靠或数据泄漏 | Router 训练证据必须可信 | 小子集 exhaustive/uniform-cost 审计 Beam gap；100状态标签翻转审计；video id/hash/embedding 泄漏报告为0 | B2, B5 |

## Paper Storyline

- Main paper must prove:
  - 同骨干、同预算下 Learned Router 的整条 Pareto 前沿优于固定和 Prompt 控制。
  - 优势来自 sequential evidence-dependent routing，而非 question-only 分类或辅助 Prompt。
  - Residual 和事件级缓存分别带来信息增益与多查询成本收益。
- Appendix can support:
  - 第二 VLM backbone、offline RL、Video-MME-v2。
  - Beam 与 exhaustive 的更大规模比较、更多 λ、更多 K/深度敏感性。
  - 开放题与 LLM Judge 稳定性。
- Experiments intentionally cut:
  - 流式视频、联合训练 VLM/Answerer、完整知识图谱、自由动作生成。
  - 未通过 B1 门槛前的 RL 和第二 backbone。

## Experiment Blocks

### Block 1: 主 Accuracy–Total Cost Pareto

- **Claim tested:** C1、C2。
- **Why this block exists:** 直接回答“是否在真实长视频上用更少全口径计算获得相同或更高准确率”。
- **Dataset / split / task:** Video-MME long、LVBench、LongVideoBench；全部为 multiple-choice；只使用官方评测划分。
- **Compared systems:** uniform frames、Gist-only、Gist→Residual、Gist→Visual、full Residual、Rule Router、Prompt Controller、Text-Adaptive Router、BC、BC+DAgger、canonical Oracle upper bound。
- **Metrics:** 首要为 MC accuracy、Pareto frontier、Cost@Accuracy、fixed-budget accuracy；次要为 GPU-seconds、wall time、frames、visual/text tokens、peak memory、平均动作数。
- **Setup details:** 同一冻结7B–8B VLM/Answerer；统一 Answerer template；λ=`0,0.1,0.3,1.0`；K=5、max depth=5；Router 三 seeds；VLM 原子输出单份缓存共享。
- **Success criterion:** 至少两个小时级 benchmark 出现稳定 Pareto dominance；等准确率成本降低≥40%或等成本准确率提高≥2点；Router 开销<在线总成本5%。
- **Failure interpretation:** 若 top-K recall 低，问题属于 Gist；若 Oracle 无优势，Memory action space 不成立；若 Oracle 有优势但 Router 无优势，问题属于策略学习。
- **Table / figure target:** Main Table 1（质量与成本）；Figure 3（每数据集 Pareto）；Figure 4（动作轨迹示例）。
- **Priority:** MUST-RUN。

### Block 2: Controller Novelty Isolation

- **Claim tested:** Learned sequential control 本身有效。
- **Why this block exists:** 排除“只要问题分类或文本 Adaptive-RAG 就够了”。
- **Dataset / split / task:** LongRoute-Train dev、Video-MME long、LVBench。
- **Compared systems:** Rule、Question-only、Text-Adaptive、Prompt Controller、BC、BC+DAgger；同 action mask 与 cache graph。
- **Metrics:** accuracy、normalized utility、Oracle regret、premature-stop、unnecessary-expansion、policy latency。
- **Setup details:** Router 三 seeds，报告均值与标准差；Prompt Controller temperature=0；Evidence 格式和预算完全一致。
- **Success criterion:** BC 超过 Rule；BC+DAgger 在至少两个核心指标上优于 BC；Learned Router 优于 Prompt Controller 且策略开销显著更低。
- **Failure interpretation:** Question-only 接近完整 Router 表明 evidence-dependent 状态无贡献；Prompt Controller 更强表明小 Router 表征或标签不足；DAgger 无提升则保留 BC 并删除 DAgger 主张。
- **Table / figure target:** Main Table 2（Controller 对比）；Appendix training curves。
- **Priority:** MUST-RUN。

### Block 3: Memory Fidelity 与简洁性消融

- **Claim tested:** 三层 Fidelity 和辅助状态确有必要，且无需更复杂结构。
- **Why this block exists:** 隔离 Gist、Residual、Context、Visual、sufficiency、cost-to-go 的贡献。
- **Dataset / split / task:** Video-MME long、LVBench；LongRoute-Train dev 用于 top-K/evidence localization 分析。
- **Compared systems:** full method；无 Residual、无 Context、无 Visual；Gist only；Gist+；embedding+ASR only；无 sufficiency head；无 cost-to-go；K={3,5}；depth={3,5}。
- **Metrics:** accuracy、total cost、top-K recall、各动作频率、Residual 新命题率、cache bytes/event。
- **Setup details:** 除被消融部分外复用同一 Router checkpoint 或按预注册方式重训；不得修改 Answerer。
- **Success criterion:** Residual 或 Visual 至少一者在细粒度类别提供明确增益；full method 位于消融 Pareto 前沿；Gist+ 的收益必须与其额外离线成本同时呈现。
- **Failure interpretation:** Residual 无增益且重合率高则改 Prompt/schema；Context 无增益则从主方法删除；辅助头无增益则保留更简单单头版本。
- **Table / figure target:** Main Table 3；Appendix sensitivity plots。
- **Priority:** MUST-RUN；K/depth 完整网格为 NICE-TO-HAVE。

### Block 4: 跨查询缓存摊销

- **Claim tested:** C2。
- **Why this block exists:** 证明按需生成的高保真 Memory 会随查询沉淀，而非只在单题中节省帧。
- **Dataset / split / task:** M3-Bench 严格隔离评测划分中每视频多问题样本；若问题数不足，则在 Video-MME/LongVideoBench 按视频聚合官方问题。
- **Compared systems:** full method with event cache、no persistent cache、full Residual upfront、Gist-only。
- **Metrics:** `Q={1,2,4,8,16}` 平均总成本、边际成本、event cache hit rate、accuracy；问题级 Visual cache 每题重新计费。
- **Setup details:** 每视频随机5个问题顺序 seed；至少3个顺序 seed；缓存从空状态开始；只复用事件级关键帧/特征/通用描述与 Residual。
- **Success criterion:** `Q>=8` 时 full method 的边际成本显著低于 no-cache/full-upfront，accuracy 不下降；首次查询成本仍计入。
- **Failure interpretation:** cache hit 高但成本无降说明缓存读取/Answerer 占主导；hit 低说明 Residual 过度问题特定或问题事件不重叠。
- **Table / figure target:** Figure 5（cost vs Q）；Appendix cache breakdown。
- **Priority:** MUST-RUN。

### Block 5: Oracle、泄漏与失败诊断

- **Claim tested:** 标签与结论可靠；定位系统剩余瓶颈。
- **Why this block exists:** 避免把召回或 Answerer 错误归因给 Router，也验证 Beam teacher 和数据隔离。
- **Dataset / split / task:** 100题 Oracle pilot；LongRoute-Train train/dev；三个主 benchmark 的全部错误题。
- **Compared systems:** Beam vs exhaustive/uniform-cost subset；single-run vs repeated Answerer audit；full Router vs Oracle。
- **Metrics:** Beam optimal-path hit、cost gap、answer flip rate、泄漏命中数；五类错误比例：召回、过早停止、保真度不足、过度检索、Answerer。
- **Setup details:** 100代表状态各3次稳定性审计；flip rate>2%才对边界状态多数投票；embedding near-duplicate threshold 0.985。
- **Success criterion:** 无确认泄漏；Beam cost gap 可忽略且不改变主要标签；标签翻转可控；Router 专属错误与非 Router 错误分开。
- **Failure interpretation:** Beam gap 大则增加 beam 或改搜索；flip 高则先解决确定性；泄漏非零则重建 split 并作废受影响结果。
- **Table / figure target:** Main error figure；Appendix integrity table。
- **Priority:** MUST-RUN。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Planned Cost | Risk |
|---|---|---|---|---:|---|
| M0 Sanity（周1–2） | 验证缓存、成本、统一 Answerer、100题 Oracle 垂直切片 | R001–R006 | 端到端可恢复；成本可复算；外推不超预算 | 100 A800h / 30 V100h | VLM吞吐或标签抖动 |
| M1 Baselines（周3–6） | 建立固定路径、Prompt/Text-Adaptive 基线和 Gist 召回 | R010–R016 | Oracle 明显优于固定路径；top-5 recall 可接受 | 120 / 20 | Gist召回不足 |
| M2 Main Method（周7–9） | 2k–4k轨迹、BC、DAgger、多偏好 Router | R020–R026 | BC>Rule；BC+DAgger 达到或接近 Oracle | 180 / 70 | imitation drift |
| M3 Decision（周10–12） | 三个主 benchmark、Pareto、关键消融 | R030–R044 | 至少两个 benchmark 过成功门槛 | 150 / 20 | 主张不成立 |
| M4 Polish（周13–16） | 缓存摊销、失败分析、统计与复现 | R050–R056 | C2成立；核心结果从空缓存可复现 | 50 / 10 | 多查询重叠不足 |
| Reserve | 仅补主结论相关失败运行 | 由失败单申请 | 不得启动无关扩展 | 200 / 50 | 预算透支 |

## First Three Runs to Launch

1. **R001**：tiny/mock 端到端与成本守恒，不使用真实 VLM。
2. **R002**：50题固定路径垂直切片，测真实 Gist/Residual/Visual 单动作成本。
3. **R003**：100题 Oracle pilot，输出 mean/P90 A800 GPU-hour 外推和缺失原子观察分布。

## Compute and Data Budget

- **Planned core:** 600 A800 GPU-hours、150 V100 GPU-hours。
- **Hard reserve:** 200 A800 GPU-hours、50 V100 GPU-hours。
- **Hard ceiling:** 800 A800 GPU-hours、200 V100 GPU-hours。
- **Data preparation:** LongRoute-Train 先2,000题，R003外推通过后最多4,000题；20%–30%多事件难例；100条人工审计。
- **Seeds:** Router/策略训练3 seeds；昂贵 VLM observation 只生成一次；问题顺序实验3 seeds。
- **Human evaluation:** 100条 LongRoute 多事件题；50条 Gist/Residual 命题重合；错误分类抽检至少100题。
- **Biggest bottleneck:** Oracle 原子 Visual observation；预计超过核心预算45%时按 high-Visual比例→beam 8到6/4→Residual后仍错才Visual→最后缩训练题数的顺序降本。

## Risks and Mitigations

- **VideoARM/Light-Omni 等弱化新颖性：** 主张只落在可学习多保真成本路由和 Oracle supervision；投稿前独立 novelty audit，不使用未经验证的“首个”。
- **Gist top-K recall 低：** 先调 ASR/text/visual score fusion 和 K=3/5，不立即增加 VLM Gist；若 top-5仍低，停止扩大 Router。
- **Residual 重复 Gist：** 50事件命题重合审计；>30%先改 Prompt/schema。
- **Answerer 限制上限：** 报告 Oracle with same Answerer，并单列“充分证据但 Answerer 错误”。
- **Prompt Controller 不公平：** 同合法动作、同缓存、同 Answerer template；Prompt reasoning 不传给 Answerer。
- **多偏好 λ 单位敏感：** NormalizedCost 常数只用 train split 冻结；主要结论仍以原始成本 Pareto/Cost@Accuracy 为准。
- **开放题 Judge 噪声：** 主实验优先 MC；开放题只进入附录且 Oracle/最终评测使用同一 Judge。

## Final Checklist

- [ ] Main paper tables are covered
- [ ] Novelty is isolated
- [ ] Simplicity is defended
- [ ] Frontier contribution is justified or explicitly not claimed
- [ ] Nice-to-have runs are separated from must-run runs
- [ ] Offline preprocessing and cache generation are included in cost
- [ ] Formal benchmark videos never enter Router training
- [ ] Core plan stays under 600/150 planned and 800/200 hard ceiling
