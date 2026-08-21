# Fidelity-Graded Video Memory Agent 设计规范

**日期：** 2026-08-21  
**状态：** 已完成讨论，待用户书面审阅  
**范围：** 离线长/超长视频问答；首版不支持流式视频

## 1. 项目目标

构建一个面向长视频问答的轻量记忆 Agent。系统只在离线阶段构建低成本、全覆盖的 Gist Memory；在收到问题后，由独立训练的轻量 Memory Controller 依据当前证据、操作历史和剩余预算，按需执行 Residual 展开、上下文扩展或原始视觉验证，并在证据充分时停止。

首版冻结视觉理解模型和最终 Answerer，只训练 Router，以隔离控制策略本身的贡献。

## 2. 研究边界与主张

### 2.1 明确不做

- 不支持实时或流式视频输入。
- 不联合训练 VLM、Answerer 与 Router。
- 不把“动态 coarse-to-fine 分层记忆”单独宣称为首创。
- 不在首版默认使用在线强化学习。
- 不把模型 API token 数作为唯一成本指标。

本文默认的“成本”均指全口径计算成本，包括离线基础记忆的按查询均摊成本、在线动作成本、首次缓存生成成本和最终回答成本。成本同时以 GPU-seconds、输入帧/视觉与文本 token、端到端延迟和峰值显存报告，不选择性使用单一有利指标。

### 2.2 论文主张

**C1：学习式成本感知路由。** 在相同冻结 Answerer、相同最大视觉预算和相同原子观察条件下，轻量 Learned Memory Controller 相比固定策略、规则 Router 和 Prompt/VLM Controller，获得更优的准确率–端到端成本 Pareto 前沿。

**C2：按需保真度升级与缓存复用。** Gist–Residual–Raw Visual 的逐级解压和问题无关 Residual 缓存，能够同时降低首次查询成本与多查询摊销成本。

创新定位聚焦于“多保真度视频记忆上的可学习成本感知顺序控制”和“最低成本 Oracle trajectory 监督”。在完成独立 novelty audit 前，不使用“首个按需增量长视频记忆系统”或“首个可学习视频记忆 Router”等不可充分验证的优先权表述。

### 2.3 必须排除的替代解释

- 收益来自更强或更大的 Answerer。
- 收益来自读取了更多帧或获得了更高视觉 token 预算。
- 收益来自全量预计算但未计入报告的离线成本。
- 收益来自测试集轨迹泄漏或直接在正式 benchmark 上训练 Router。
- 收益来自不同方法使用了不同缓存或不同原子视觉观察。
- 收益来自 Answerer 接收了不同格式或包含额外策略引导的 Prompt，而不是 Router 获取的证据本身。

为排除最后一项，所有基线和本方法使用完全相同的 Answerer 模板。模板只接收统一序列化的 `question + options + acquired evidence`；Visual 动作获取的帧属于 Evidence，不得附加方法专属的推理或策略提示。证据长度本身是策略决策的结果，不能强行固定，但必须计入 token/帧成本，并在固定总 Evidence token/frame budget 的对比中再次验证。

## 3. 总体架构

```text
Video
  -> Cheap event segmentation / ASR / sparse frames / embeddings
  -> Gist Memory Index
  -> Learned Memory Controller
       - SEARCH_GIST
       - EXPAND_RESIDUAL(event)
       - EXPAND_CONTEXT(event)
       - VERIFY_VISUAL(event, budget)
       - STOP
  -> Evidence Store
  -> Frozen Answerer
```

### 3.1 离线基础阶段

每个视频只执行以下全覆盖操作：

1. 视频解码和元数据提取；
2. 基于 shot boundary、ASR 停顿和最大时间窗的廉价事件切分；
3. ASR 或已有字幕对齐；
4. 每个事件稀疏采样低分辨率关键帧；
5. 提取冻结视觉和文本 embedding；
6. 仅根据 ASR/字幕，用 1B–2B 轻量文本模型生成极短 Gist 描述并建立多模态检索索引；
7. 保存原视频时间指针。

分段过程不得调用大 VLM。目标事件长度为约 20–40 秒；过长事件强制切分，极短相邻镜头合并。

### 3.2 在线查询阶段

Controller 从问题和当前状态构造合法动作集合，逐步获取证据。所有昂贵操作经过统一缓存和成本计量。最终 Answerer 只接收 Controller 已获取的 Evidence，以及 Visual 动作实际选取的帧。

## 4. Fidelity-Graded Memory

### 4.1 Gist Memory

Gist 是唯一全量构建的语义层，包含：

- `video_id`、`event_id`；
- `start_sec`、`end_sec`；
- ASR/字幕片段；
- 稀疏关键帧路径；
- 视觉与文本 embedding；
- 极短事件描述；
- 原视频指针和缓存版本信息。

主版本 Gist 不调用 VLM。它用 ASR/字幕和 1B–2B 轻量文本模型生成文本摘要，同时保留冻结视觉编码器提取的廉价 embedding，以支持无语音或语音不足的视频。Gist 用于全视频粗召回，不要求覆盖颜色、计数、精细空间关系等细节。

必须保留两个消融：`embedding + ASR only` 用于验证轻量文本摘要是否值得成本；`Gist+` 使用共享的 7B–8B VLM 读取稀疏帧生成视觉增强 Gist，用于检验额外视觉语义带来的召回收益是否抵得过全量构建成本。

### 4.2 Residual Memory

Residual 默认不存在，只在候选事件被选中后生成。它必须是问题无关的结构化事件细节，使其可以跨查询复用：

```json
{
  "entities": [],
  "actions": [],
  "attributes": [],
  "spatial_relations": [],
  "counts": [],
  "state_changes": [],
  "exceptions": [],
  "unstructured_details": []
}
```

Residual 使用比 Gist 更高的帧密度和分辨率，但不直接回答当前问题。生成 Prompt 必须同时输入现有 Gist，并明确要求只补充 Gist 未覆盖的细节、禁止复述已有命题。`unstructured_details` 用于保留无法稳定归入固定字段的新增信息。生成结果持久化到事件级缓存。

### 4.3 Raw Visual / Visual Verification

Raw Visual 离线阶段只保存时间指针。`VERIFY_VISUAL` 是问题相关的高成本操作，包含两个离散预算：

- `low`：8–12 帧；
- `high`：24–32 帧。

首版不允许 Router 输出任意帧数。Visual 使用两级缓存：

- 事件级通用缓存：采样关键帧、帧级视觉特征和通用帧描述，可跨查询复用；
- 问题级专属缓存：针对精确问题的验证答案和必要的验证记录，不计入跨查询缓存收益。

多查询摊销实验只将事件级通用缓存视为可复用收益，问题级专属缓存按每个新问题重新计费。

### 4.4 Context

`EXPAND_CONTEXT(event)` 每次将已知上下文边界向前、向后各扩展一个相邻事件，默认只返回新事件的 Gist，以及此前已经存在的 Residual。再次执行只允许扩展尚未访问的下一圈事件；若没有未访问邻居，该动作被 Mask。它不隐式触发新的高成本 Residual 或 Visual 操作，避免隐藏成本。

## 5. Controller 状态与动作

### 5.1 状态

```text
RouterState = question
            + current evidence
            + action history
            + remaining budget
            + current candidate events
            + candidate fidelity / context frontier
            + cost preference
```

Evidence Gap 在首版采用隐式表示，由 Router 从状态中学习，不额外调用 LLM 生成缺失信息文本。显式 Evidence Gap 仅作为后续消融。

### 5.2 动作实例

Router 不直接生成自由文本 `event_id`。Gist 检索先得到 top-K 候选，系统再构造合法动作实例：

- `SEARCH_GIST`；
- `EXPAND_RESIDUAL(e_i)`；
- `EXPAND_CONTEXT(e_i)`；
- `VERIFY_VISUAL(e_i, low)`；
- `VERIFY_VISUAL(e_i, high)`；
- `STOP`。

同一事件不得重复执行相同无收益动作。首版 `K=5`、最大轨迹长度为 5；若吞吐 pilot 超出预算，可将 `K` 降为 3。

合法动作由环境进行硬 Mask，而不是依赖 Router 学会避免错误：已有 Residual 的事件不能再次选择 `EXPAND_RESIDUAL`；同一视觉预算档位已经验证后不能重复选择；`EXPAND_CONTEXT` 只能在仍有未访问相邻事件时选择；预算不足的动作不可选。训练和推理使用同一 Mask 逻辑。

### 5.3 模型

Router 使用约 100M–300M 参数的文本编码器与轻量候选动作评分层。每个 EvidenceItem 独立编码后池化，避免将无限历史直接拼接到单个序列。

Router 输出：

- candidate-aware action-instance score；
- evidence sufficiency probability；
- normalized cost-to-go estimate。

共享的 7B–8B 开源 VLM 负责 Gist/Residual/Visual 与最终回答。正式对比中 checkpoint、解码设置、图像分辨率和最大预算保持一致。

## 6. 核心数据接口

### 6.1 EventRecord

```text
EventRecord(
  video_id, event_id, start_sec, end_sec,
  asr_text, keyframe_paths, visual_embedding,
  text_embedding, gist_text, raw_video_uri,
  memory_version
)
```

### 6.2 EvidenceItem

```text
EvidenceItem(
  source_layer, event_id, content,
  confidence, acquisition_cost, cache_key
)
```

### 6.3 RouterState

```text
RouterState(
  question, options, evidence,
  action_history, remaining_budget,
  candidate_event_ids, candidate_fidelity_levels,
  context_frontiers, cost_preference
)
```

### 6.4 ActionInstance、Transition 与 Trajectory

```text
ActionInstance(action_type, event_id, visual_budget)
Transition(state_before, action, observation, state_after, step_cost)
Trajectory(transitions, predicted_answer, correct, total_cost)
```

所有记录必须带配置版本、模型版本和随机种子。

## 7. 缓存与可恢复执行

缓存键必须包含：

```text
video_hash + event_range + model_version
+ prompt_hash + sampling_config
```

缓存采用内容寻址和原子写入。每个任务记录 `pending/running/complete/failed` 状态；中断后只重跑未完成或校验失败的条目。不同 Router、固定路径和 Prompt Agent 必须共享同一原子观察缓存。

## 8. LongRoute-Train 与数据隔离

### 8.1 合成长视频课程训练

使用 NExT-QA、ActivityNet-QA 等训练划分中的带答案视频，将目标视频或事件与 9–19 个无关片段拼接，构造 10–60 分钟训练视频。保留原选择题答案，并随机化：

- 证据位置；
- 干扰片段数；
- 相似干扰事件；
- 视频总长度；
- 上下文范围。

LongRoute-Train 至少包含两类难例：

- 相似干扰：将视觉或语义相近、但关键属性或状态不同的事件放在目标事件附近；
- 多事件证据：使用可程序验证的时序、计数或属性比较模板，将两个或多个源事件组合为必须跨区域取证的问题。

多事件问题只能使用可从源标注确定答案的模板，不使用未经校验的自由生成答案。首版目标占训练问题的 20%–30%，并人工抽检至少 100 条问题、答案和证据位置。

原始视频必须按 video id 隔离到 train/dev，正式 benchmark 视频不得出现在 LongRoute-Train。

### 8.2 真实长视频校准

在许可证和官方划分允许的前提下，使用与正式评测严格隔离的 M3-Bench 训练部分进行小规模校准。该步骤不替代合成训练，也不得访问正式评测答案。

## 9. Oracle Trajectory

### 9.1 目标

对每个问题选择能够回答正确的最低成本路径：

```text
tau* = argmin C(tau), subject to Answer(tau) = gold answer
```

该约束式解作为 canonical Oracle，用于评估策略到“最低成本正确路径”的 regret。为训练可控制的完整 Pareto 前沿，同一缓存轨迹图还按 4 个预注册成本偏好计算：

```text
J_lambda(tau) = AnswerScore(tau) - lambda * NormalizedCost(tau)
lambda in {0.0, 0.1, 0.3, 1.0}
```

RouterState 接收 `cost_preference`，训练一个条件 Router，而不是维护四套独立模型。`NormalizedCost` 的归一化常数只由训练划分确定并在正式评测前冻结。评测同时报告离散偏好点及预算扫掠得到的 Pareto 曲线。

若所有路径均答错，canonical Oracle 先按答案置信度排序，再选择最低成本路径。Multiple-choice 任务使用确定性精确判分；开放题若进入附录，使用与最终实验完全相同且版本冻结的 Judge、Prompt 和阈值。Oracle 与最终实验不得使用不同判分标准。

Answerer 使用 temperature 0 和固定 seed。第2周对至少 100 个代表状态各重复运行 3 次，测量标签翻转率；只有翻转率超过 2% 时，才对置信度接近边界或发生翻转的状态使用三次多数投票，避免将全部 Oracle 成本放大三倍。

### 9.2 可承受搜索

1. `SEARCH_GIST` 产生 top-K；
2. 逐级生成候选 Residual、Context 和必要的 Visual 原子观察；
3. 原子观察只生成一次并落盘；
4. 使用 beam size 8、最大深度 5 组合轨迹；
5. 禁止重复动作和无效循环；
6. 只对 Residual 后仍无法正确回答的候选优先生成 Visual。

Oracle 训练规模先限制为 2,000–4,000 个问题。

首版以 Beam search 为主。A* 不作为默认替换：在答案正确性只有执行 Answerer 后才可观测的状态图上，缺少既有信息量又可证明 admissible 的启发函数，不能仅凭“后续都执行最便宜动作”保证全局最优。系统将在一个可穷举的小规模子集上运行 uniform-cost/exhaustive search，报告 Beam 的最优路径命中率和 cost gap；只有该审计暴露明显缺口时才研究 A* 或增大 Beam。

## 10. Router 训练

### 10.1 Behavior Cloning

```text
L = L_action
  + 0.3 * L_sufficiency
  + 0.1 * L_cost_to_go
```

动作分类始终是主损失；`0.3` 和 `0.1` 是初始预注册配置，后续只在训练 dev 上调整。训练集包含 Oracle 路径上的状态–动作对，以及近优但错误路径访问的状态。

Sufficiency 标签由“在当前状态立即 STOP 并使用统一 Answerer 是否答对”确定，而不是由该状态来自正确或错误轨迹决定。所有立即回答正确的中间状态均为正样本；立即回答错误的状态为负样本。这样避免把一条最终失败轨迹中偶然已经充分的中间状态错误标成负例。

### 10.2 DAgger 式纠偏

让 BC Router 在训练环境中运行，对其偏离 Oracle 后访问的状态重新标注，以降低序列误差累积。标注只在已有原子观察缓存图上计算单步最优合法动作，不重新运行完整 VLM Oracle。默认迭代 2 轮，只有第2轮 dev utility 仍有明确提升时才运行第3轮。

### 10.3 Offline RL

Offline RL 严格限定为附录扩展。主论文的核心结论必须由 BC+DAgger 独立支撑。只有 BC+DAgger 已达到核心门槛，且主要剩余误差明确来自长期成本分配时才运行；不得为了 RL 重新在线生成大规模 VLM 观察，也不得占用核心算力预算。

## 11. 成本模型

```text
C_total(Q) = C_base_memory / Q
           + C_online_actions
           + C_answer
```

`Q` 是同一视频的累计问题数。必须同时报告 `Q=1` 和 `Q in {2,4,8,16}`。

每个操作记录：

- GPU-seconds；
- wall-clock latency；
- 输入帧数和分辨率；
- 视觉与文本 token；
- 峰值显存；
- 缓存命中状态；
- Router 自身耗时。

离线基础记忆和首次缓存生成不得从总成本中排除。

## 12. 实验协议

### 12.1 主数据集

- Video-MME long；
- LVBench；
- LongVideoBench。

EgoSchema 公开 500 题和严格隔离的 M3-Bench 用作迁移或附录实验。Video-MME-v2 只有在核心实验完成且资源充足时加入。

### 12.2 基线家族

1. 固定输入/固定路径：uniform frames、Gist-only、Gist→Residual、Gist→Visual、全量 Residual；
2. 自适应控制：同 action space 的 Prompt Controller、可公平复现的 VideoARM/VCA 风格策略，以及 `Text-Adaptive Router`；后者将文本 Adaptive-RAG 直接迁移到视频，只基于 ASR/Gist 文本决定继续检索、调用视觉或停止，不使用 candidate fidelity、视频上下文 frontier 和多模态 Evidence 表征；
3. Router 变体：Rule、Question-only、BC、BC+DAgger、可选 offline RL。

所有可控对比共享 Answerer、原子观察、缓存、最大视觉预算和解码配置。

### 12.3 五个实验块

- B1：准确率–总成本 Pareto；
- B2：Controller 隔离对比；
- B3：Memory 层级与辅助头消融；
- B4：多查询缓存摊销；
- B5：失败轨迹与 Evidence Gap 类型分析。

B5 固定使用五类互斥的首要失败原因：

1. 召回错误：正确事件未进入 Gist top-K；
2. 过早停止：证据不足时执行 STOP；
3. 保真度不足：存在有用升级动作但未选择；
4. 过度检索：额外高成本动作不改变答案或充分性；
5. Answerer 错误：已提供 Oracle 充分证据但冻结 Answerer 仍答错。

召回错误和 Answerer 错误单独报告，不计为 Controller action-selection error。

### 12.4 指标

- multiple-choice accuracy；
- Accuracy–Cost curve、Pareto AUC、Cost@Accuracy；
- 分问题类型和视频长度准确率；
- Oracle utility regret；
- premature-stop rate；
- unnecessary-expansion rate；
- Gist top-K evidence recall；
- 平均动作数与动作分布；
- GPU 时间、端到端延迟、帧、token、显存和缓存命中率。

### 12.5 成功门槛

- 至少两个小时级 benchmark 上形成稳定 Pareto 优势；
- 相比同 Answerer 的最强全量记忆方案，准确率下降不超过 1 点时总成本降低至少 40%，或相同总成本下准确率提高至少 2 点；
- Learned Router 优于同 action space Prompt Controller；
- Router 计算成本低于总在线成本的 5%；
- BC+DAgger 相对 Oracle 的正确率差距不超过 3 点；
- `Q >= 8` 时缓存显著降低边际成本且不降低准确率。

可以补充报告经训练集常数归一化后的 `Accuracy / NormalizedCost`，但不把“提升 60%”设为主要成功门槛：该比值对成本单位和归一化方式敏感，不能替代尺度更稳健的 Pareto dominance、Cost@Accuracy 与固定预算 Accuracy。

若主 Pareto 不成立，不增加 RL 或更复杂 Memory，优先检查 Gist top-K recall、Residual 信息覆盖和成本核算。

## 13. 工程结构

```text
configs/
src/fidmem/
  data/
  memory/
  retrieval/
  actions/
  oracle/
  router/
  agent/
  costs/
  eval/
scripts/
tests/
artifacts/
docs/
```

技术栈：Python、PyTorch、Transformers、FAISS、Hydra、Parquet/DuckDB、pytest。`artifacts/` 保存视频派生物、模型输出、轨迹和运行结果，并排除在 Git 版本控制之外。

## 14. 测试与验收

必须覆盖：

- 非法动作和 STOP 后动作被拒绝；
- `candidate_fidelity_levels` 与 `context_frontiers` 在动作后正确更新，并产生一致的硬 Mask；
- 相同 Residual 不重复收费；
- 缓存键确定且版本敏感；
- toy graph 的 Oracle 返回最低成本正确路径；
- 轨迹总成本等于操作成本之和；
- 中断任务能够从 manifest 恢复；
- 数据集之间不存在 video id 或内容哈希泄漏；
- 一个短视频 toy case 能从空缓存完成端到端问答；
- 固定 seed 和固定缓存时结果可复现。

## 15. 四个月里程碑

| 周期 | 交付物 | 决策门槛 |
|---|---|---|
| 第1–2周 | 数据类型、切分、缓存、成本计量、toy pipeline；用最小垂直切片跑通100题 Oracle 原子观察 | 单视频可运行，成本可复算；获得单题 A800 成本估计 |
| 第3–4周 | 三层 Memory、检索、动作、冻结 Answerer | 50题固定路径稳定 |
| 第5–6周 | Oracle 缓存、beam search、LongRoute-Train v0 | 500题可恢复生成，Oracle 优于固定路径 |
| 第7–8周 | 2k–4k轨迹、BC、Rule/Prompt baseline | BC 优于 Rule |
| 第9周 | DAgger、方法冻结 | DAgger 有效或明确保留 BC |
| 第10–12周 | 三个主 benchmark、Pareto、关键消融 | 至少两个数据集达到门槛 |
| 第13–14周 | 缓存摊销、鲁棒性、可选扩展 | C2 得到支持 |
| 第15周 | 失败分析、统计、论文图表 | 主表主图完备 |
| 第16周 | 补跑、复现包和缓冲 | 核心结果可从空缓存复现 |

## 16. 算力预算

### 16.1 冻结预算

- 核心项目 A800 总预算不超过 800 GPU-hours；
- 核心项目 V100 总预算不超过 200 GPU-hours；
- 第2周吞吐 pilot 后允许约 ±30% 校准，但任何扩大必须由主张收益支撑。

### 16.2 分级预算

| 阶段 | A800 | V100 |
|---|---:|---:|
| 可行性验证 | 120–220 GPUh | 30–60 GPUh |
| 核心论文实验 | 500–800 GPUh | 100–200 GPUh |
| 可选补充 | 只有剩余预算时启动 | 只有剩余预算时启动 |

### 16.3 停止规则

- 可行性阶段超过 220 A800 GPU-hours 仍不能证明 Oracle 优于固定路径，则停止扩展训练集；
- 核心实验达到 800 A800 GPU-hours 后，只补跑影响主结论的实验；
- offline RL、第二 backbone 和 Video-MME-v2 不得占用核心预算；
- Oracle 原子观察预计超过 A800 核心预算的 45% 时，按下一条规定的顺序收缩搜索与生成成本。
- 第2周根据100题垂直切片的单题均值和 P90 成本外推 2,000–4,000 题预算；若超限，依次减少 high-Visual 生成比例、将 Beam 从8降到6或4、只为 Residual 后仍答错的样本生成 Visual，最后才缩减训练题数。

## 17. 主要风险与处置

### Gist 召回不足

先提高低成本 embedding/ASR 融合和 top-K，不直接增加 Visual 调用。若 top-5 evidence recall 仍低，方法的“逐级解压”前提不成立。

### Residual 无法恢复判别细节

分析缺失字段分布，调整统一 schema 与帧采样；不得改成问题相关 Residual 来掩盖问题，否则缓存主张失效。

### Gist 与 Residual 信息重叠过高

Residual 生成必须输入 Gist 并只请求新增命题。第3–4周人工抽检50个事件，同时计算规范化命题重合率；若人工判定或自动命题匹配的重合率超过30%，先迭代 Prompt/schema，再扩大原子观察生成。重合率阈值只作为工程门槛，不作为论文效果指标。

### Oracle 标签不唯一

训练 action distribution 或等价最优动作集合，不强迫 Router 模仿单条任意轨迹；主要评价 utility regret，而非轨迹完全匹配率。

### Prompt Controller 对比不公平

使用完全相同的合法动作、缓存和最大预算，只替换控制策略。

### 成本测量受系统噪声影响

GPU 操作预热后重复测量，分别报告设备时间和端到端时间；缓存命中与未命中分开统计。

### Answerer 输出导致 Oracle 标签抖动

固定 checkpoint、Prompt、temperature、seed 和判分器版本。先按第9.1节进行100状态三次重复审计；标签翻转率不超过2%时保持单次确定性生成，超过2%时只对边界或已观察到翻转的状态多数投票，并在附录报告翻转率。

### 数据泄漏

除 video id 比对外，对视频帧 embedding 或内容哈希进行近重复检测，并保存审计报告。
