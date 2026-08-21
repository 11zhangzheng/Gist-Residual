# Fidelity-Graded Video Memory Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个冻结 VLM/Answerer、仅训练轻量 Router 的长视频问答系统，支持 Gist–Residual–Raw Visual 按需升级、成本可审计缓存、Oracle trajectory、BC+DAgger 训练与 Pareto 评测。

**Architecture:** 视频先经过无 VLM 的轻量 Gist 预处理，查询时由候选感知 Router 在合法动作集合上顺序决策。所有昂贵观察内容寻址缓存，Answerer 与基线共享统一输入模板；Oracle 在缓存状态图上搜索，训练与评测共享同一状态转移和成本模型。

**Tech Stack:** Python 3.11、PyTorch 2.x、Transformers 4.x、FAISS CPU、Hydra、Pydantic 2、DuckDB/Parquet、Typer、pytest。

**Spec:** `docs/superpowers/specs/2026-08-21-fidelity-graded-video-memory-design.md`

## Global Constraints

- 首版只支持离线长视频问答，不实现流式输入。
- VLM 与最终 Answerer 全程冻结；主训练对象仅为 100M–300M Router。
- 主 Gist 不调用 VLM：ASR/字幕经 1B–2B 文本模型生成短摘要，并融合廉价视觉 embedding。
- Residual 必须问题无关；问题级 Visual 缓存不得计入跨查询复用收益。
- 所有策略共享 Answerer 模板、原子观察缓存、最大帧预算和判分标准。
- 核心预算不超过 800 A800 GPU-hours 和 200 V100 GPU-hours。
- 所有任务遵循测试先行；每个任务结束时必须能独立运行其测试和 smoke command。

## File Map

```text
pyproject.toml                         # 依赖、CLI 和 pytest 配置
.gitignore                            # 模型、视频、缓存和运行产物
configs/base.yaml                     # 全局默认配置
configs/model/*.yaml                  # Gist/Residual/Visual/Answerer/Router 配置
configs/experiment/*.yaml             # pilot、train、main、ablation 配置
src/fidmem/types.py                   # 稳定领域类型
src/fidmem/config.py                  # 配置加载和校验
src/fidmem/storage/cache.py           # 内容寻址缓存
src/fidmem/storage/run_store.py       # DuckDB/Parquet 运行记录与恢复
src/fidmem/costs/tracker.py           # 全口径成本计量
src/fidmem/data/video.py              # 视频探测、解码、采样
src/fidmem/data/segmentation.py       # 廉价事件切分
src/fidmem/data/leakage.py            # ID/hash/embedding 近重复审计
src/fidmem/memory/gist.py             # Gist 构建
src/fidmem/memory/residual.py         # Residual 生成与去重
src/fidmem/memory/visual.py           # 两级 Visual 缓存
src/fidmem/retrieval/index.py         # 多模态 top-K 检索
src/fidmem/actions/environment.py     # 动作、硬 Mask、状态转移
src/fidmem/agent/answerer.py          # 统一冻结 Answerer 接口
src/fidmem/agent/runner.py            # 顺序执行循环
src/fidmem/data/longroute.py          # 合成长视频训练集
src/fidmem/oracle/search.py           # Beam/exhaustive Oracle
src/fidmem/oracle/labels.py           # 多偏好和充分性标签
src/fidmem/router/model.py            # Router 三头模型
src/fidmem/router/train_bc.py         # Behavior Cloning
src/fidmem/router/dagger.py           # 缓存图上的 DAgger
src/fidmem/eval/baselines.py          # 固定、Prompt、Text-Adaptive 基线
src/fidmem/eval/metrics.py            # 准确率、成本、regret、错误分类
src/fidmem/eval/runner.py             # benchmark 运行与聚合
src/fidmem/cli.py                     # 统一命令入口
tests/                                # 与上述模块镜像的单元/集成测试
```

---

### Task 1: 项目骨架、稳定类型与配置校验

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `configs/base.yaml`
- Create: `src/fidmem/__init__.py`
- Create: `src/fidmem/types.py`
- Create: `src/fidmem/config.py`
- Test: `tests/test_types.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `EventRecord`, `EvidenceItem`, `RouterState`, `ActionInstance`, `Transition`, `Trajectory`, `AppConfig`。
- Consumes: 仅 Python 标准库、Pydantic 与 Hydra/OmegaConf。

- [ ] **Step 0: 初始化当前空目录的 Git 仓库**

```bash
git init
git branch -M main
```

Expected: `git status` 显示位于 `main`，且现有 `docs/`、`refine-logs/` 与 `MANIFEST.md` 为未跟踪文件。

- [ ] **Step 1: 写类型校验失败测试**

```python
from fidmem.types import ActionInstance, ActionType, FidelityLevel, RouterState

def test_router_state_rejects_unknown_candidate_fidelity() -> None:
    state = RouterState(
        question="What color is the bottle?",
        options=("red", "blue"), evidence=(), action_history=(),
        remaining_budget=1.0, candidate_event_ids=("e1",),
        candidate_fidelity_levels={"e1": FidelityLevel.GIST},
        context_frontiers={"e1": (0, 0)}, cost_preference=0.3,
    )
    assert state.candidate_fidelity_levels["e1"] is FidelityLevel.GIST
    assert ActionInstance(ActionType.EXPAND_RESIDUAL, "e1", None).event_id == "e1"
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run: `pytest tests/test_types.py -q`  
Expected: collection error mentioning `fidmem.types`。

- [ ] **Step 3: 实现冻结领域类型**

在 `types.py` 使用 `str, Enum` 和 frozen Pydantic models；`ActionType` 固定为 `SEARCH_GIST/EXPAND_RESIDUAL/EXPAND_CONTEXT/VERIFY_VISUAL/STOP`，`FidelityLevel` 固定为 `GIST/RESIDUAL/VISUAL`，Visual budget 只允许 `low/high/None`。`RouterState` 校验 candidate id 与 fidelity/context 字典键集合完全一致，预算非负。

- [ ] **Step 4: 建立配置并验证预算常数**

`configs/base.yaml` 写入 `retrieval.top_k: 5`、`oracle.max_depth: 5`、`oracle.beam_size: 8`、`visual.low_frames: 12`、`visual.high_frames: 32`、`budget.a800_gpu_hours: 800`、`budget.v100_gpu_hours: 200`；`AppConfig` 对这些范围做显式校验。

- [ ] **Step 5: 运行类型与配置测试**

Run: `pytest tests/test_types.py tests/test_config.py -q`  
Expected: PASS。

- [ ] **Step 6: 提交**

```bash
git add pyproject.toml .gitignore configs src/fidmem tests/test_types.py tests/test_config.py
git commit -m "chore: scaffold fidelity memory project"
```

### Task 2: 内容寻址缓存与可恢复运行存储

**Files:**
- Create: `src/fidmem/storage/cache.py`
- Create: `src/fidmem/storage/run_store.py`
- Test: `tests/storage/test_cache.py`
- Test: `tests/storage/test_run_store.py`

**Interfaces:**
- Consumes: `EventRecord` 与 JSON-serializable payload。
- Produces: `CacheKey.build(...) -> str`、`ContentAddressedCache.get/put`、`RunStore.claim/complete/fail/pending`。

- [ ] **Step 1: 写缓存键和原子恢复测试**

```python
def test_prompt_change_invalidates_cache(tmp_path):
    cache = ContentAddressedCache(tmp_path)
    a = cache.key("vhash", (0.0, 30.0), "model-v1", "prompt-a", {"frames": 12})
    b = cache.key("vhash", (0.0, 30.0), "model-v1", "prompt-b", {"frames": 12})
    assert a != b
    cache.put(a, {"value": 3})
    assert cache.get(a) == {"value": 3}
```

- [ ] **Step 2: 确认测试失败后实现 canonical JSON + SHA-256 key**

序列化使用 `json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False)`；写入先落同目录临时文件，经 `flush` 和 `os.fsync` 后 `os.replace`，禁止直接覆盖目标文件。

- [ ] **Step 3: 实现 DuckDB 状态机**

表字段固定为 `run_id, item_key, status, attempt, worker_id, started_at, finished_at, error_type, error_message, output_uri`。`claim` 只能将 `pending/failed` 变为 `running`；`complete` 必须提供已存在的 output URI。

- [ ] **Step 4: 测试崩溃恢复**

模拟 `running` 条目超过 lease 后回收为 `pending`，已 `complete` 条目不得重跑。

- [ ] **Step 5: 运行测试并提交**

Run: `pytest tests/storage -q`  
Expected: PASS。

```bash
git add src/fidmem/storage tests/storage
git commit -m "feat: add resumable content-addressed storage"
```

### Task 3: 全口径成本计量

**Files:**
- Create: `src/fidmem/costs/tracker.py`
- Test: `tests/costs/test_tracker.py`

**Interfaces:**
- Produces: `CostRecord`, `CostTracker.measure(operation, cache_status, frames, tokens)`、`amortized_total(base, online, answer, query_count)`。
- Consumes: CUDA events when available；CPU 环境退化到 `perf_counter_ns`。

- [ ] **Step 1: 写成本守恒和摊销测试**

```python
def test_amortized_total_counts_first_cache_generation():
    total = amortized_total(base_gpu_s=80.0, online_gpu_s=12.0,
                            answer_gpu_s=3.0, query_count=4)
    assert total == 35.0
```

- [ ] **Step 2: 实现设备时间与端到端时间双计时**

`CostRecord` 固定字段：`operation, gpu_seconds, wall_seconds, input_frames, visual_tokens, text_tokens, peak_memory_bytes, cache_status, device_name`。CUDA 路径在计时前后同步，正式测量前由调用者执行至少一次 warmup。

- [ ] **Step 3: 实现成本聚合**

按 `video_id/question_id/action_type/cache_status` 聚合；基础记忆以 `base_cost / query_count` 摊销；问题级 Visual cache 每个新问题重新计费。

- [ ] **Step 4: 运行 CPU 单测和 CUDA 条件测试**

Run: `pytest tests/costs -q`  
Expected: CPU 必过；无 CUDA 时 GPU 专项标记 skip。

- [ ] **Step 5: 提交**

```bash
git add src/fidmem/costs tests/costs
git commit -m "feat: account for end-to-end memory costs"
```

### Task 4: 视频接入、廉价事件切分与泄漏审计

**Files:**
- Create: `src/fidmem/data/video.py`
- Create: `src/fidmem/data/segmentation.py`
- Create: `src/fidmem/data/leakage.py`
- Test: `tests/data/test_segmentation.py`
- Test: `tests/data/test_leakage.py`
- Create: `tests/fixtures/tiny_video.mp4`

**Interfaces:**
- Produces: `probe_video`, `sample_frames`, `segment_video`, `LeakageAuditor.audit(train, eval)`。
- Consumes: ffprobe/ffmpeg、ASR timestamp spans、shot timestamps。

- [ ] **Step 1: 写事件边界性质测试**

```python
def test_segments_cover_timeline_without_large_gaps():
    events = segment_timestamps(120.0, shots=(0.0, 12.0, 55.0, 120.0),
                                speech_breaks=(28.0, 82.0), min_sec=8, max_sec=40)
    assert events[0].start_sec == 0.0
    assert events[-1].end_sec == 120.0
    assert all(8 <= e.duration_sec <= 40 for e in events[:-1])
```

- [ ] **Step 2: 实现确定性切分**

先合并小于8秒的相邻 shot，再优先选择靠近 ASR 停顿的边界，最后按40秒硬切；输出事件按时间排序、无重叠、无超过0.5秒空洞。

- [ ] **Step 2a: 生成确定性视频 fixture**

```bash
ffmpeg -y -f lavfi -i "testsrc=size=320x240:rate=10:duration=4" -pix_fmt yuv420p tests/fixtures/tiny_video.mp4
```

Expected: ffprobe 报告时长约4秒、分辨率320×240、10 FPS。

- [ ] **Step 3: 实现三层泄漏检查**

依次比较规范化 video id、文件 SHA-256、每视频8帧 embedding centroid cosine。`cosine >= 0.985` 标为 `near_duplicate`，输出 Parquet 审计表，不自动删除数据。

- [ ] **Step 4: 运行 fixture 集成测试**

Run: `pytest tests/data -q`  
Expected: tiny video 可探测、采帧、切段；复制文件被 hash 检出。

- [ ] **Step 5: 提交**

```bash
git add src/fidmem/data tests/data tests/fixtures
git commit -m "feat: ingest and segment long videos cheaply"
```

### Task 5: Gist Memory 与多模态检索

**Files:**
- Create: `src/fidmem/memory/gist.py`
- Create: `src/fidmem/retrieval/index.py`
- Create: `configs/model/gist.yaml`
- Test: `tests/memory/test_gist.py`
- Test: `tests/retrieval/test_index.py`

**Interfaces:**
- Produces: `GistBuilder.build(event) -> EventRecord`、`GistIndex.search(question, k) -> tuple[ScoredEvent, ...]`。
- Consumes: ASR、轻量文本摘要器、冻结视觉编码器与 `ContentAddressedCache`。

- [ ] **Step 1: 写“主 Gist 不调用 VLM”测试**

使用会在调用时抛异常的 `ForbiddenVLM` 注入 `GistBuilder`；断言主配置仍生成 gist，且 silent event 仍具有视觉 embedding。

- [ ] **Step 2: 实现 Gist builder**

文本摘要器输出最多40 token；无 ASR 时 `gist_text="[no speech]"`。视觉编码器只处理每事件4张低分辨率帧。`Gist+` 独立配置允许共享 VLM，但缓存 namespace 与主 Gist 分离。

- [ ] **Step 3: 实现融合检索**

归一化 text/visual score，默认 `0.6 * text + 0.4 * visual`；稳定 tie-break 使用 `(score desc, start_sec asc, event_id asc)`；返回 top-K 和分项分数。

- [ ] **Step 4: 测试 top-K、确定性与空 ASR**

Run: `pytest tests/memory/test_gist.py tests/retrieval/test_index.py -q`  
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/fidmem/memory/gist.py src/fidmem/retrieval configs/model/gist.yaml tests/memory tests/retrieval
git commit -m "feat: build lightweight searchable gist memory"
```

### Task 6: Residual、两级 Visual Cache 与 Context frontier

**Files:**
- Create: `src/fidmem/memory/residual.py`
- Create: `src/fidmem/memory/visual.py`
- Create: `configs/model/residual.yaml`
- Create: `configs/model/visual.yaml`
- Test: `tests/memory/test_residual.py`
- Test: `tests/memory/test_visual.py`

**Interfaces:**
- Produces: `ResidualGenerator.expand(event, gist)`、`VisualVerifier.observe_event/verify_question`、`expand_context(events, frontier)`。
- Consumes: 冻结 VLM adapter、事件帧采样器、两级缓存。

- [ ] **Step 1: 写 schema、去重复和缓存隔离测试**

断言 Residual JSON 恰含 `entities/actions/attributes/spatial_relations/counts/state_changes/exceptions/unstructured_details`；重复 Gist 命题被过滤；两个不同问题共享 event cache key、但 question cache key 不同。

- [ ] **Step 2: 实现 Residual 解析和命题去重**

VLM Prompt 包含 Gist 并明确禁止复述。解析失败只允许一次 JSON repair；之后记录 `schema_error`。使用规范化字符串和 embedding cosine `>=0.92` 去掉与 Gist 重复的命题，同时保留原始响应用于审计。

- [ ] **Step 3: 实现 Visual 两级缓存**

事件级 key 不包含 question；问题级 key 包含规范化 question/options hash。`low=12`、`high=32` 帧，采样配置进入 cache key。成本聚合器只摊销事件级条目。

- [ ] **Step 4: 实现 Context frontier**

每次扩展左右各一事件并更新 `(left_radius, right_radius)`；只返回新进入边界的 Gist 和已存在 Residual，禁止隐式生成新 Residual/Visual。

- [ ] **Step 5: 运行测试并提交**

Run: `pytest tests/memory -q`  
Expected: PASS。

```bash
git add src/fidmem/memory configs/model/residual.yaml configs/model/visual.yaml tests/memory
git commit -m "feat: add progressive residual and visual memory"
```

### Task 7: 动作环境、硬 Mask 与统一 Answerer

**Files:**
- Create: `src/fidmem/actions/environment.py`
- Create: `src/fidmem/agent/answerer.py`
- Create: `src/fidmem/agent/runner.py`
- Create: `configs/model/answerer.yaml`
- Test: `tests/actions/test_environment.py`
- Test: `tests/agent/test_runner.py`

**Interfaces:**
- Produces: `MemoryEnvironment.valid_actions/step`、`FrozenAnswerer.answer`、`AgentRunner.run`。
- Consumes: GistIndex、Memory providers、Router policy。

- [ ] **Step 1: 写硬 Mask 状态机测试**

覆盖：Residual 后同动作消失；low Visual 后 low 消失但 high 仍合法；Context 无新邻居后消失；预算不足动作消失；STOP 后 `step` 抛 `TerminalStateError`。

- [ ] **Step 2: 实现纯状态转移核心**

把 `valid_actions(state)` 写成无 I/O 纯函数；I/O 观察由 action executor 返回，再由 reducer 更新 `evidence/history/budget/fidelity/frontier`。每次成本从环境统一扣除，Router 无权修改。

- [ ] **Step 3: 实现统一 Answerer 模板**

固定模板字段顺序为 `Question/Options/Evidence/Answer`。Evidence 统一按时间和 acquisition step 排序；禁止加入策略名称、动作理由或方法专属指令。Visual frames 仅作为对应 EvidenceItem 的附件。

- [ ] **Step 4: 实现最多5步 Runner**

每步先生成合法动作，policy 只能从其中选择；达到最大深度时强制 STOP 并记录 `forced_stop=true`；完整 Transition 写入 RunStore。

- [ ] **Step 5: 运行集成测试并提交**

Run: `pytest tests/actions tests/agent -q`  
Expected: mock memory/answerer 的完整轨迹 PASS。

```bash
git add src/fidmem/actions src/fidmem/agent configs/model/answerer.yaml tests/actions tests/agent
git commit -m "feat: execute masked memory actions consistently"
```

### Task 8: LongRoute-Train 合成与隔离

**Files:**
- Create: `src/fidmem/data/longroute.py`
- Create: `configs/experiment/longroute.yaml`
- Test: `tests/data/test_longroute.py`

**Interfaces:**
- Produces: `LongRouteBuilder.build(seed) -> DatasetManifest`。
- Consumes: NExT-QA/ActivityNet-QA train split manifest 与 LeakageAuditor。

- [ ] **Step 1: 写确定性拼接与分组隔离测试**

同 seed 产生相同目标位置和 distractors；同一 source video 不得跨 train/dev；eval hash 命中时 builder 失败并输出审计记录。

- [ ] **Step 2: 实现单事件课程**

每题拼接1个目标与9–19个 distractors，目标位置均匀采样；时长不足10分钟时继续添加 distractor，超过60分钟时重新采样；保存每段 source id 与全局时间偏移。

- [ ] **Step 3: 实现难例**

相似干扰按 embedding 最近邻抽样。多事件题只使用可程序判定的 `before/after`、属性比较和计数模板，目标比例20%–30%；生成 manifest 中保存 supporting event ids。

- [ ] **Step 4: 生成100题人工审计包**

导出 `question/options/answer/source_events/global_offsets/contact_sheet`，并提供 `valid/invalid/reason` CSV 表头，禁止将未审计自由生成答案混入训练。

- [ ] **Step 5: 运行测试并提交**

Run: `pytest tests/data/test_longroute.py -q`  
Expected: PASS。

```bash
git add src/fidmem/data/longroute.py configs/experiment/longroute.yaml tests/data/test_longroute.py
git commit -m "feat: synthesize leak-free long-route training data"
```

### Task 9: Oracle 状态图、多偏好标签与稳定性审计

**Files:**
- Create: `src/fidmem/oracle/search.py`
- Create: `src/fidmem/oracle/labels.py`
- Create: `configs/experiment/oracle_pilot.yaml`
- Test: `tests/oracle/test_search.py`
- Test: `tests/oracle/test_labels.py`

**Interfaces:**
- Produces: `beam_search`, `exhaustive_search`, `canonical_oracle`, `preference_labels`, `sufficiency_label`。
- Consumes: MemoryEnvironment 的缓存状态图与统一 Answerer。

- [ ] **Step 1: 写 toy graph 最优性测试**

构造 `GIST(cost=1, wrong) -> RESIDUAL(cost=2, correct)`、`GIST -> VISUAL(cost=5, correct)`，断言 canonical Oracle 选择总成本3的路径；λ=1时允许选择更便宜路径，λ=0时选择最高 AnswerScore。

- [ ] **Step 2: 实现 Beam search**

优先队列键固定为 `(-utility, total_cost, depth, action_signature)`；`beam_size=8,max_depth=5`。所有 observation 从 cache graph 读取；缺失原子观察返回待生成清单，不在搜索器内部调用 VLM。

- [ ] **Step 3: 实现 canonical 与四偏好标签**

canonical：正确路径中最小成本；无正确路径时按 AnswerScore 降序、成本升序。偏好固定 `{0.0,0.1,0.3,1.0}`，成本归一化常数只从 train split 估计并写入 run manifest。

- [ ] **Step 4: 实现充分性标签**

对任意状态执行统一 STOP Answer；答对为1、答错为0，与该状态来自正确或失败轨迹无关。

- [ ] **Step 5: 实现100题 pilot 和 Beam 审计**

100题记录单题 mean/P90 A800 时间；其中可承受子集运行 exhaustive/uniform-cost，与 Beam 比较路径命中率和 cost gap。另抽100个状态三次推理，计算 answer flip rate。

- [ ] **Step 6: 运行测试并提交**

Run: `pytest tests/oracle -q`  
Expected: toy graph Beam 与 exhaustive 一致，多偏好标签确定。

```bash
git add src/fidmem/oracle configs/experiment/oracle_pilot.yaml tests/oracle
git commit -m "feat: generate cost-aware oracle trajectories"
```

### Task 10: Router 三头模型与 Behavior Cloning

**Files:**
- Create: `src/fidmem/router/model.py`
- Create: `src/fidmem/router/dataset.py`
- Create: `src/fidmem/router/train_bc.py`
- Create: `configs/model/router.yaml`
- Create: `configs/experiment/train_bc.yaml`
- Test: `tests/router/test_model.py`
- Test: `tests/router/test_training.py`

**Interfaces:**
- Produces: `MemoryRouter.forward(batch) -> action_logits,sufficiency_logit,cost_to_go`、`train_bc`。
- Consumes: Oracle state/action Parquet 与合法动作 mask。

- [ ] **Step 1: 写 shape、Mask 与 overfit 测试**

非法动作 logit 必须在 softmax 前填为 dtype 最小值；32样本 toy dataset 在200步内 action accuracy 达到95%以上，证明训练链路可学习。

- [ ] **Step 2: 实现候选感知编码**

问题、每个 EvidenceItem 与每个 ActionInstance 分别编码；加入 budget、cost preference、fidelity 和 frontier embedding；action scorer 对合法 action instance 输出单个 logit。

- [ ] **Step 3: 实现三项损失**

`loss = CE(action) + 0.3*BCE(sufficiency) + 0.1*SmoothL1(cost_to_go)`；配置保存三个权重，主实验前只允许在 train dev 上修改一次并记录原因。

- [ ] **Step 4: 实现按视频分组的数据加载与训练恢复**

同一视频的状态只进入一个 split；checkpoint 保存模型、优化器、scheduler、step、随机状态、git commit 和配置 hash。默认3 seeds，昂贵 observation 不随 seed 重建。

- [ ] **Step 5: 运行测试和单卡 smoke train**

Run: `pytest tests/router -q`  
Run: `python -m fidmem.cli train-router --config configs/experiment/train_bc.yaml --max-steps 20`  
Expected: 测试 PASS；产生可恢复 checkpoint。

- [ ] **Step 6: 提交**

```bash
git add src/fidmem/router configs/model/router.yaml configs/experiment/train_bc.yaml tests/router
git commit -m "feat: train lightweight memory router by imitation"
```

### Task 11: DAgger 单步纠偏与策略冻结

**Files:**
- Create: `src/fidmem/router/dagger.py`
- Create: `configs/experiment/dagger.yaml`
- Test: `tests/router/test_dagger.py`

**Interfaces:**
- Produces: `collect_deviations`, `label_best_next_action`, `run_dagger_round`。
- Consumes: BC policy、缓存状态图、Oracle utility evaluator。

- [ ] **Step 1: 写“不调用 VLM”测试**

注入 `ForbiddenObservationGenerator`，让 BC policy 进入偏离状态；断言 DAgger 只从缓存子图选择单步最优动作并产生新样本。

- [ ] **Step 2: 实现 rollout 与去重**

状态 key 由 question id、已获取 cache keys、budget bin、cost preference 和 frontier 组成；同 key 只标注一次。每轮收集 train questions 的固定子集，避免分布随机器数变化。

- [ ] **Step 3: 实现2轮默认、3轮条件门槛**

第2轮后若 dev utility 相对第1轮提升小于0.5个百分点且 cost regret 改善小于2%，停止；否则允许第3轮。停止决策写入 manifest。

- [ ] **Step 4: 运行测试并提交**

Run: `pytest tests/router/test_dagger.py -q`  
Expected: PASS，mock VLM 调用计数为0。

```bash
git add src/fidmem/router/dagger.py configs/experiment/dagger.yaml tests/router/test_dagger.py
git commit -m "feat: correct router drift with cached dagger labels"
```

### Task 12: 公平基线、指标与 benchmark Runner

**Files:**
- Create: `src/fidmem/eval/baselines.py`
- Create: `src/fidmem/eval/metrics.py`
- Create: `src/fidmem/eval/error_taxonomy.py`
- Create: `src/fidmem/eval/runner.py`
- Create: `configs/experiment/main_eval.yaml`
- Test: `tests/eval/test_baselines.py`
- Test: `tests/eval/test_metrics.py`

**Interfaces:**
- Produces: 固定策略、Rule、Prompt、Text-Adaptive policies；`evaluate_run`；Pareto/Cost@Accuracy/error taxonomy。
- Consumes: 相同 MemoryEnvironment、Answerer、cache graph 和 benchmark manifest。

- [ ] **Step 1: 写公平性测试**

所有 policy 必须收到同一 `valid_actions`，Answerer template hash 完全相同，固定预算下 frames/tokens 超限即标记 invalid；Prompt policy 的文本不得进入最终 Answerer evidence。

- [ ] **Step 2: 实现三类基线**

固定策略：uniform、Gist-only、Gist→Residual、Gist→Visual、full Residual；自适应：Rule、Prompt/VLM controller、Text-Adaptive Router；学习式：Question-only、BC、BC+DAgger。每个策略只返回 ActionInstance。

- [ ] **Step 3: 实现核心指标**

输出 MC accuracy、GPU/wall/frame/token cost、peak memory、Pareto frontier、Cost@Accuracy、fixed-budget accuracy、Oracle utility regret、premature stop、unnecessary expansion、top-K recall、action distribution。

- [ ] **Step 4: 实现五类互斥错误归因**

优先级固定：召回错误→Answerer错误→过早停止→保真度不足→过度检索。每题只分配一个 primary cause，同时可保存 secondary flags。

- [ ] **Step 5: 运行合成 benchmark 集成测试**

Run: `pytest tests/eval -q`  
Expected: 已知数据上 Pareto 非支配点、Cost@Accuracy 和错误分类与手算一致。

- [ ] **Step 6: 提交**

```bash
git add src/fidmem/eval configs/experiment/main_eval.yaml tests/eval
git commit -m "feat: evaluate router accuracy-cost tradeoffs"
```

### Task 13: CLI、端到端复现与验收门槛

**Files:**
- Create: `src/fidmem/cli.py`
- Create: `configs/experiment/pilot.yaml`
- Create: `docs/RUNBOOK.md`
- Create: `tests/integration/test_end_to_end.py`
- Create: `tests/integration/test_resume.py`

**Interfaces:**
- Produces: `fidmem ingest/build-gist/build-observations/build-oracle/train-router/run-dagger/evaluate/report` 命令。
- Consumes: 前述全部模块。

- [ ] **Step 1: 写端到端失败测试**

tiny video + mock models 必须完成 ingest→gist→oracle→BC smoke→evaluate；中途终止 observation worker 后重启，complete 条目不重复计费。

- [ ] **Step 2: 实现 Typer CLI 和 dry-run**

所有命令支持 `--config`、`--run-id`、`--dry-run`、`--resume`。dry-run 输出样本数、预计原子观察数、cache hit 数和根据 pilot 外推的 GPU-hour，超过预算时返回非零退出码。

- [ ] **Step 3: 实现报告命令**

`report` 读取 DuckDB/Parquet，生成机器可读 JSON 和 Markdown：配置 hash、数据 hash、模型版本、成本汇总、结果表、失败分类、未完成条目和预算余额。

- [ ] **Step 4: 编写精确 Runbook**

按 M0–M4 给出命令顺序、输入目录规范、GPU 分配、恢复步骤和停止门槛；明确 A800 跑 VLM/Answerer，V100 跑 embedding/Router，禁止两个 worker 写同一未加锁 cache key。

- [ ] **Step 5: 执行完整验证**

Run: `pytest -q`  
Run: `python -m fidmem.cli --help`  
Run: `python -m fidmem.cli evaluate --config configs/experiment/pilot.yaml --dry-run`  
Expected: 全部测试 PASS；CLI 列出8个子命令；dry-run 在核心预算内。

- [ ] **Step 6: 提交**

```bash
git add src/fidmem/cli.py configs/experiment/pilot.yaml docs/RUNBOOK.md tests/integration
git commit -m "feat: provide reproducible end-to-end workflow"
```

## Implementation Checkpoints

1. Task 1–4 后：只验证基础设施、切分、缓存和成本，不接真实 VLM。
2. Task 5–7 后：完成50题固定路径与统一 Answerer 垂直切片。
3. Task 8–9 后：完成100题 Oracle pilot，重新核准总算力；未通过门槛不得进入大规模轨迹生成。
4. Task 10–11 后：比较 Rule、BC、BC+DAgger；BC 未超过 Rule 时暂停正式 benchmark。
5. Task 12–13 后：执行 `refine-logs/EXPERIMENT_TRACKER.md` 中 MUST runs。

## Plan Self-Review Checklist

- [ ] 每个设计规范模块都有对应任务。
- [ ] 所有跨模块类型均由 Task 1 定义。
- [ ] 所有昂贵模型调用都经过 Task 2 cache 与 Task 3 cost tracker。
- [ ] Offline RL、第二 backbone 和 Video-MME-v2 未进入核心实现路径。
- [ ] 没有任务要求在正式 benchmark 上训练 Router。
- [ ] 所有最终 Answerer 调用共享 Task 7 模板。
