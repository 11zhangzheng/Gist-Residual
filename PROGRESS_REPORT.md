# Fidelity-Graded Video Memory Agent — 项目进度报告

> 更新日期：2026-09-01 UTC
>
> 工作区：`/home/zhangzheng/projects/Gist-Residual`
>
> 当前 HEAD：`86517e2c051730ebc32ad8b8de57b4e91ddca6f8`（`Refactor metadata handling and enhance video preparation logic`）
>
> 当前状态：Video-MME-v2 代码迁移及 45-video pilot 下载已完成；数据资产布局和 manifest 绝对路径仍需修复并重新冻结。正式实验停在 E00 前置条件，尚无 Production Authority、production result 或 paper result。

---

## 1. 当前研究协议

### 数据集角色

| 角色 | 数据集 |
|---|---|
| Source / Router development | **Video-MME-v2** |
| Final independent target benchmarks | **LongVideoBench、LVBench、MLVU** |

Video-MME-v2 不再是 final independent target benchmark。现有 Production Authority、Experiment Execution Pack、E00–E17 DAG、冻结 Gate、Oracle protocol、Answerer stability threshold 和 provenance 规则保持不变。

### 核心研究假设

系统仍以冻结的视觉/语言模型和轻量 Memory Router 验证成本感知的分级视频记忆：

```text
Video → event segmentation / Gist Memory Index
      → SEARCH_GIST / EXPAND_RESIDUAL / EXPAND_CONTEXT /
        VERIFY_VISUAL / STOP
      → Evidence Store → Frozen Answerer
```

---

## 2. 冻结模型栈与资产状态

| Logical role | Hugging Face snapshot | Immutable revision | 状态 |
|---|---|---|---|
| Gist text / embedding | `BAAI/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` | 已下载并重新哈希验证；两个 logical roles 共用一个 snapshot |
| Gist visual | `google/siglip2-so400m-patch14-384` | `e8e487298228002f3d8a82e0cd5c8ea9c567f57f` | 已下载并重新哈希验证 |
| Residual / Visual VLM | `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | 已下载并重新哈希验证；两个 logical roles 共用一个 snapshot |
| Answerer | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | 已下载并重新哈希验证 |

- dtype：`bfloat16`
- backend：Hugging Face Transformers
- 模型目录占用：约 **41 GiB**
- asset lock SHA-256：`c459cc2f97c7dee52d96b8520fa9d7731b74508ca1ddc5f7350f6fc1f1db817d`
- `03_verify_models.sh --check` 已对四个物理 snapshot 完成磁盘身份校验，没有重复下载共享角色资产。

---

## 3. Video-MME-v2 数据准备

### 3.1 官方 metadata / annotations

- 官方来源：`MME-Benchmarks/Video-MME-v2`
- 固定 revision：`6e4bebb03202e1ddbf3d37703e560e51c5aa2d64`
- metadata 覆盖：**800 videos、3,200 questions**，每个 video 4 questions
- video ID：精确覆盖 `001`–`800`
- metadata semantic SHA-256：`04cf8ecdf9e25cd76bfce496330bc06e9bd8e2d529ddf3d9e845f6d172960a3a`
- 官方 README SHA-256：`57ddaf7bb20cb6715518a72a4ffd14afefe211c54a51605ec4ab648277b0b6ca`
- subtitle archive SHA-256：`adbd3cfd98bd03756398d1c8b63c7bcddf0e5c2494b6a0736ed890456021c287`
- parquet SHA-256：`8dc7f8c8830aa49dd08a82592f8276899472a145155dde3bea5dd6914a65a9b4`

### 3.2 当前下载范围

当前磁盘无法在既定安全余量内承载完整 Video-MME-v2，因此使用确定性、按 `video_id` 选择的 **`PARTIAL_DATASET_PILOT`**：

- selection seed：`videomme-v2-partial-pilot-pool-v1`
- selection algorithm：`videomme-v2-archive-aware-hash-v1`
- pilot pool：**45 videos、180 questions**
- 官方 video archives：`017`、`027`、`035`
- selection semantic SHA-256：`cecf538035e0031bc5b048b5ab94602e6791ed7f2689dd0d32533e65e8d0b15f`
- archive index semantic SHA-256：`61f94457b33c4e71b66f3f523928b992233fe4c289ce48a64ca6b162187e38ae`

验证结果：

- 当前 live raw root 中 45/45 raw MP4、45/45 subtitles 和 3/3 archives 均存在
- 历史 raw verification 记录 45/45 文件身份通过、20/20 midpoint decode smoke check 通过
- 三个归档的待提取成员数均为 0
- 下载器支持 `--check`、`--resume`、`--verify-only`
- 已修复 resume 模式误读旧报告的问题；恢复判断现在使用可恢复 state

当前 live raw root 为：

```text
/mnt/disk1/zhangzheng/fidmem/datasets/videomme_v2_metadata/Raw/Video-MME-v2
```

该目录位于被冻结的 metadata snapshot 内，会改变 snapshot 的目录级 identity；因此当前 metadata asset lock 的 re-verification **FAIL**。此外，历史 manifest 仍绑定已不存在的旧 raw root `/mnt/disk1/zhangzheng/Tvqa_data/Raw/Video-MME-v2`。必须先把 raw media 放回独立于 metadata snapshot 的稳定路径，并重新执行 `--verify-only` 和 manifest 生成，才能进入 E01。

**证据边界：该子集只能用于方法验证和 E03/E04 pilot，不得表述为 full Video-MME-v2 benchmark result。**

---

## 4. Manifest 与 video-disjoint split

当前工程产物目录：

```text
artifacts/dataset-preparation/videomme-v2-pilot-v1/
```

### 历史 Manifest 身份

| 产物 | SHA-256 |
|---|---|
| `media_preparation.json` | `e7e67835b23587a1467b8c2809315f65ec6e1b7bc61ea7259300d12006ed786c` |
| `archive_index.json` | `d3fe233c1c5b75fe6b0efea996ec7bf2f04f640291e25059ca57cc78d1cbea1e` |
| `metadata_verification.json` | `670015369508d23aa9f85d98c0598c5a832e28c759e043200073194b959f1347` |
| `raw_video_verification.json` | `4083fab7ed151d800e84bdadc5fbd6653432747f7eae32cf56b15194e817ce5b` |
| `subset_selection_manifest.json` | `d9507f3c598a33d87268180dc79887d75f3de5853b63d7ecc6bce003077723ad` |
| `split_policy.json` | `8d9d75e2148c5bff9e1adbef2499ab8bf68b5bc239df1dd021a5b68714f0a68b` |
| `dataset_manifest.json` | `f069f23deeb7bb28f39b4b3601b7918fdd3a43b1424cb4345fc684ce6f8c4ab6` |
| `video_manifest.json` | `13f0da02a33a8973aba7d585529eb16e3d61188866fff6445a70d43e82b98716` |
| `question_manifest.json` | `45f2d731f8bfb0e87d6a573c44fd8a4791572f23e58094702c3cdcc3d078a2ee` |
| `canary_selection_manifest.json` | `f862efc0c43417645e379b6da7358edf847ba6eae782fa7dcb316e1d722f6505` |
| `oracle_selection_manifest.json` | `fcb1a11cc353a7434fc8bcd846144b39004902c269f27f81bdb191f805abc034` |
| `human_audit_manifest.json` | `2c79b7ff59c088848429a866ae6c44447e313b6c4438381d97b3f1597c85ce3d` |
| `source_gate.json` | `ff050c1ea9ac26f5a7a8ee878c8f02b917285fb9462987969ed3e5a41fbade1e` |

Source policy SHA-256：`dbd4ef4a51d548d705c4763d4415c1e4e4680f4639f3c456f20417205b5480fa`

这些文件的内容哈希均可复核，但它们当前不是可执行的最终冻结 manifest：`video_manifest.json` 中 45/45 `uri` 均指向已不存在的旧 raw root，`raw_video_verification.json` 和 `media_preparation.json` 也绑定相同旧路径。重新生成后文件哈希将变化，报告中的本组哈希应保留为历史工程证据，而不是 E01/E02 Authority 输入。

### Split 分布

| Split | Videos | Questions | 用途 |
|---|---:|---:|---|
| `development` | 12 | 48 | Source / Router development |
| `canary` | 4 | 16 | E03 Production Canary |
| `oracle` | 25 | 100 | E04 Oracle Pilot |
| `holdout` | 4 | 16 | Source holdout |
| **合计** | **45** | **180** | `PARTIAL_DATASET_PILOT` |

四个 split 已通过 video-disjoint 审计；Canary 与 Oracle 没有共享 `video_id`。

---

## 5. E00–E17 实验状态

| Stage | 当前状态 | 说明 |
|---|---|---|
| E00 Environment & Asset Freeze | **仅 `--check` 通过，正式阶段未运行** | 配置 SHA-256 `7f79b0c67e4df482b40b4e03158027ede686947fe1f22029db0be56f325ccd23`；当前 `selected_gpus=[]`、NVIDIA driver 不可用；metadata snapshot identity 仍需恢复 |
| E01 Production Readiness Audit | **未就绪，正式人工审计未签署** | live raw root 与冻结 metadata snapshot 重叠，历史 manifest 路径失效；Execution Pack 继续 fail closed |
| E02 Authority Seal | **未运行** | 尚无 Production Authority hash；不能启动正式 Canary |
| E03 Production Canary | **未运行** | 16 questions 已准备，但 E00/E01/E02 gate 尚未满足 |
| E04 Oracle Pilot | **未运行** | 100 questions 已准备；必须等待 E03 Canary 通过 |
| E05–E17 | **未开始** | 严格等待 DAG 上游阶段和既有 Gate |

目前没有正式 stage record、GPU runtime、observation count、measured cost、result 或 gate verdict。不得把 E00 `--check` 描述为 E00 正式完成。

---

## 6. 验证与测试证据

本轮工程验证记录：

```text
pytest -q tests/assets tests/production tests/integration
203 passed in 17.70s

python -m fidmem.production.pack_cli --validate-registry
valid: true

artifact-chain consistency audit
ARTIFACT_CHAIN_OK videos=45 questions=180 canary=16 oracle=100 raw_decode=20

git diff --check
PASS
```

补充说明：

- 四个模型 snapshot 已执行真实磁盘 rehash，而非仅验证 marker。
- 当前 raw root 下 45/45 MP4、45/45 subtitle 和 3/3 archive 均存在；但历史 manifest URI 的存在性检查为 0/45。
- 对当前 live root 运行 `06_prepare_videomme_v2_videos.sh --verify-only` 时，在读取数据前即按 fail-closed 规则报错：`Video-MME-v2 metadata snapshot differs from asset lock`。原因是 raw media 被放入 metadata snapshot 内，而不是 metadata 三个冻结文件本身发生已知内容变化。
- 完整 `pytest -q` 曾启动，但现有 CPU-only Router 训练用例包含 200-step 长运行，手动停止前未出现失败；因此本报告**不声称 full suite 已通过**。
- 当前节点没有可用 NVIDIA runtime，未伪造 GPU 验证。

---

## 7. 磁盘与运行环境

| 路径/资产 | 当前状态 |
|---|---|
| 模型 snapshots | 约 41 GiB |
| Video-MME-v2 metadata | 约 8.7 MiB |
| Pilot raw media | 约 14 GiB |
| `/mnt/disk1` | 约 101 GiB 可用，99% 已用 |
| `/home` | 约 22 GiB 可用，98% 已用 |
| GPU | `nvidia-smi` 无法连接 driver |

在保留安全余量且不删除既有数据的前提下，当前不适合继续下载 full Video-MME-v2。待扩容后，应复用相同官方来源、冻结 revision、下载器和 manifest 规则补齐全量数据。

---

## 8. Git 与 provenance 状态

- 当前 HEAD：`86517e2c051730ebc32ad8b8de57b4e91ddca6f8`
- `origin/main`：`63e0fcab8a817b0c6015fbf8c0fe74be45903c9f`；本地 `main` 领先 1 个 commit
- E00 `--check` 记录的 execution source-tree hash：`08568c76f2c378d7935a870502db0f1acba943543aa249591531dbdcd521a5de`
- 当前工作树仅 `PROGRESS_REPORT.md` 有未提交变更。
- 本地 commit `86517e2` 已包含 metadata/preparation 修复和报告初次刷新；当前报告的后续一致性更正尚未提交。没有 tag 或外部发布记录。

`PROGRESS_REPORT.md` 属于非执行报告，按现有 source identity 规则不进入 execution source-tree hash；正式 E00 前仍需重新生成并核对全部 provenance 证据。

---

## 9. 当前阻断项与下一步

### 正式实验阻断项

1. raw media 当前位于冻结 metadata snapshot 内，导致 metadata asset lock re-verification 失败。
2. 历史 manifest 中 45/45 video URI 指向不存在的旧 raw root，必须从 live media 重新生成。
3. 当前机器 NVIDIA driver / GPU runtime 不可用。
4. E01 仍需人工完成 Production Readiness Audit 并签署现有 Gate。
5. E00 正式运行尚未生成可供 Execution Pack 接受的 environment gate。
6. E02 Authority Seal 尚未建立，因此 E03 不能正式启动。
7. 当前报告仍在 dirty worktree；本地新 commit 尚未出现在 `origin/main`。

### 下一正式阶段

**下一正式阶段仍是 E00 Environment & Asset Freeze，但必须先修复数据资产布局。** 推荐顺序：

1. 将 45-video raw media 放到 metadata snapshot 之外的稳定目录，不删除现有数据；恢复 metadata lock identity。
2. 对新 raw root 执行 `--verify-only`，重新生成 dataset/video/question、Canary/Oracle 和 Source Gate preparation artifacts，并复核新的哈希及 video-disjoint 性。
3. 在满足冻结环境要求的目标 GPU 节点，重新执行 E00 正式环境与资产冻结。
4. 完成人工 E01 Production Readiness Audit。
5. 对所有冻结字段生成 E02 Authority Seal。
6. E02 通过后运行 E03 Canary（16 questions）；Canary 通过后运行 E04 Oracle Pilot（100 questions）。

当前未发现新的 `RESEARCH_OWNER_DECISION_REQUIRED`；现有阻断均属于环境、provenance 或既定人工 Gate，而不是需要改变论文定义的研究参数。

---

## 10. 证据分级声明

- 当前数据、代码、manifest、split、模型校验和测试结果属于 **engineering evidence**。
- 尚无 E00–E04 正式阶段结果，因此没有 **production evidence**。
- 尚未执行 final target benchmark，因此没有 **paper evidence**。
- `PARTIAL_DATASET_PILOT` 的任何后续结果都必须显式保留该标签，不得写成 full Video-MME-v2 结果。
