# Fidelity-Graded Video Memory Agent — 项目进度报告

> 生成日期：2026-08-24
> 工作区：`D:/Desktop/Gist`（主实现位于 worktree `D:/Desktop/Gist/.worktrees/fidelity-memory-agent`）
> 分支：`feat/fidelity-memory-agent`（基线 `b29dec4`）

---

## 1. 项目概述

**Fidelity-Graded Video Memory Agent（`fidmem`）** 是一个面向长/超长视频问答的论文驱动研究系统，核心目标：**冻结视觉理解模型与最终 Answerer，只训练一个轻量 Memory Router（100M–300M 参数）**，以验证「可学习成本感知路由」在准确率–成本 Pareto 前沿上的优势。

工作流程：

```text
Video → 廉价事件切分/ASR/稀疏帧/embedding → Gist Memory Index
      → Learned Memory Controller (SEARCH_GIST / EXPAND_RESIDUAL /
        EXPAND_CONTEXT / VERIFY_VISUAL / STOP)
      → Evidence Store → Frozen Answerer
```

三条核心主张：

- **C1**：轻量 Learned Memory Controller 相对固定策略、规则 Router 和 Prompt/VLM Controller 获得更优的准确率–端到端成本 Pareto 前沿；
- **C2**：Gist–Residual–Raw Visual 的逐级解压 + 问题无关 Residual 缓存，同时降低首次查询成本与多查询摊销成本。

**工程方法**：Superpowers 的 Subagent-Driven Development（SDD），把设计规范拆成 **13 个任务**，测试先行（TDD），每个任务经多轮独立 review 修复后才关闭。

**技术栈**：Python 3.11、PyTorch（`torch>=2.1,<3`）、Transformers、FAISS CPU、Hydra、Pydantic 2、DuckDB/Parquet、Typer、pytest。

---

## 2. 进度总览

| # | 任务 | 状态 |
|---|---|---|
| 1 | 项目骨架、稳定类型与配置校验 | ✅ 完成 |
| 2 | 内容寻址缓存与可恢复运行存储 | ✅ 完成 |
| 3 | 全口径成本计量 | ✅ 完成 |
| 4 | 视频接入、廉价事件切分与泄漏审计 | ✅ 完成 |
| 5 | Gist Memory 与多模态检索 | ✅ 完成 |
| 6 | Residual、两级 Visual Cache 与 Context frontier | ✅ 完成 |
| 7 | 动作环境、硬 Mask 与统一 Answerer | ✅ 完成 |
| 8 | LongRoute-Train 合成与隔离 | ✅ 完成 |
| 9 | Oracle 状态图、多偏好标签与稳定性审计 | ✅ 完成 |
| 10 | Router 三头模型与 Behavior Cloning | ✅ 完成（round 4 review clean） |
| 11 | DAgger 单步纠偏与策略冻结 | ✅ 完成（round 4 review clean） |
| 12 | 公平基线、指标与 benchmark Runner | 🔄 进行中 |
| 13 | CLI、端到端复现与验收门槛 | ⬜ 未开始 |

**整体进度：约 85%（11/13 任务完成）。**

---

## 3. 量化数据

| 指标 | 数值 |
|---|---|
| feature 分支最新提交 | `bab7ce6` |
| 全量测试结果（Task 11 round 4） | 356 passed, 2 skipped |
| Router 测试结果 | 104 passed, 1 skipped |
| DAgger 测试结果 | 50 passed, 1 skipped |
| Task 10 round 3 新增测试 | 6 passed |
| Router 生产模型可训练参数量 | 121,049,935（落在 100M–300M 冻结区间内） |
| BC 训练目标函数 | `CE(action) + 0.3·BCE(sufficiency) + 0.1·SmoothL1(cost_to_go)` |
| 预注册种子 | 13 / 37 / 73 |

---

## 4. Task 10–11 round 4 验证结论

Task 10 与 Task 11 均已完成 round 4 独立 review，review clean。实际提交链为：`477b4f2 → e8fd813 → 00bf3b9 → d58d7ff → d600e89`（Task 10）；`5ae79ac → b0e4ee2 → 0228261 → d4fb0cf → bab7ce6`（Task 11）。 Task 10 的最终实现覆盖 **数据/checkpoint 溯源信任链的进一步加固**：

- 每个 segment 必须恰有一个真实 source owner，且 asset hash、event identity 与时间范围均需匹配；
- `SufficiencyLabelArtifact` 改为自算 self-hash，禁止调用方注入 label 或自定义 judge；
- loader 强制要求 content-addressed 的 `.authority.json` sidecar，缺失/篡改即失败关闭；
- 伪造的 row provenance 无法自洽（Task8 lineage 全链路重算比对）；
- `_git_commit` 在无仓库或无构建元数据时失败关闭。

Task 11 的最终实现覆盖无新 VLM 的缓存 DAgger、多轮原子发布、身份绑定、恢复与 indeterminate commit 处理。

**验证结果（2026-08-24 实测）：**

| 检查项 | 结果 |
|---|---|
| Task 11 DAgger 全套 | ✅ 50 passed, 1 skipped |
| `pytest tests/router` | ✅ 104 passed, 1 skipped |
| `pytest`（全量） | ✅ 356 passed, 2 skipped |
| `compileall src tests` | ✅ exit 0 |
| `git diff --check` | ✅ exit 0 |
| Task 11 round 4 full verification | ✅ 无 OMP/MKL workaround，75.41s，低于 180s bound |

**结论：Task 10–11 已完成且独立 review clean；不要将 Task 12 误报为已完成。**

---

## 5. 当前下一步

Task 12 正在进行：实现公平基线、评测指标与 benchmark Runner，并核对 BC/BC+DAgger 的统一评测口径。Task 13 尚未开始，后续负责 CLI、端到端复现与验收门槛。

## 6. 剩余工作（Task 12–13）

1. **Task 12 — 公平基线与评测**：固定策略 / Rule / Prompt / Text-Adaptive / BC / BC+DAgger 基线，Pareto、Cost@Accuracy、五类互斥错误归因。
2. **Task 13 — CLI 与端到端复现**：Typer CLI（8 个子命令）、dry-run 预算外推、`docs/RUNBOOK.md`、验收门槛。

---

## 7. 残余风险

- **CUDA 复现性未验证**：Router 的 121M 生产模型从未做过真实 GPU 训练，CUDA 位级 replay、确定性与显存峰值均未实测。
- **生产预训练编码器未下载**：测试全程 CPU-only + `local_files_only`，正式训练需预先本地化的不可变 HF snapshot。
- **CI symlink residual**：无 symlink 权限的 Windows 主机跳过真实 symlink-alias 回归，需在 Linux CI 执行。
- **测试可诊断性 residual**：此前声称的 OpenMP/MKL Barrier 死锁未复现，尚未确认；`OMP_NUM_THREADS=1` 与 `MKL_NUM_THREADS=1` 仅保留为 CI workaround。仍需补充有超时的 Barrier/future 诊断，避免测试挂起时不可诊断。
- **若干 minor 问题被延期**：Task 1–9 各留了少量 minor（如原子写失败路径测试、线程竞争未证明跨进程互斥、POSIX `fcntl.flock` 分支需 Linux CI 验证等），详见 SDD ledger `progress.md`。

---

## 8. 环境要点

- **Python**：`D:/Anaconda/python.exe`（Anaconda 环境，已装 torch/pydantic/hydra 等）。
- **无系统 ffmpeg**：用 `imageio-ffmpeg==0.6.0` 提供打包的 ffmpeg，用于确定性 MP4 fixture。
- **smoke 命令**需 `PYTHONPATH='src'` 前缀（worktree 未 editable-install）。
- **测试隔离**：宿主无 CUDA，GPU 专项测试以 skip 处理。





