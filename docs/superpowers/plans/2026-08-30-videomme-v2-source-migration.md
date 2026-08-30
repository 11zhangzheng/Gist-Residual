# Video-MME-v2 Source Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace LongTVQA with a pinned official Video-MME-v2 source adapter, prepare a deterministic 45-video pilot with auditable raw MP4s and video-disjoint manifests, preserve the verified model stack, and advance the real server through every currently satisfiable E00-E02 gate without fabricating missing human or research-owner evidence.

**Architecture:** Extend the existing asset-lock and Production Authority path instead of creating a second execution framework. A dataset-specific adapter validates the pinned Parquet/subtitle metadata, indexes official ZIP central directories through seekable range reads, selects archives and videos deterministically, downloads with hash-bound resume, extracts atomically, and emits the existing manifest family plus explicit pilot provenance. E01 remains fail closed on the real human audit; E02 remains fail closed on any unresolved production config or backend identity.

**Tech Stack:** Python 3.12 in the existing `fidmem-a800` conda environment, Pydantic v2, DuckDB Parquet reader, `huggingface_hub>=0.24,<1`, standard-library `zipfile`/`hashlib`/`tempfile`, ffprobe/ffmpeg through existing video utilities, OmegaConf, pytest, Bash, Git.

**Spec:** `docs/superpowers/specs/2026-08-30-videomme-v2-source-migration-design.md`

## Global Constraints

- Work in `/home/zhangzheng/projects/Gist-Residual` at the current clean checkout; inspect `git status --short --branch`, `git diff`, and `git rev-parse HEAD` before each implementation batch.
- Never reset, clean, stash, rebase, force-push, push, overwrite unknown changes, delete existing datasets/models/artifacts, or acquire videos from a third-party source.
- The only source upstream is dataset `MME-Benchmarks/Video-MME-v2` at immutable revision `6e4bebb03202e1ddbf3d37703e560e51c5aa2d64`.
- The pilot scope is exactly `PARTIAL_DATASET_PILOT`, 45 videos and 180 complete four-question groups, selected with `videomme-v2-partial-pilot-pool-v1` and `videomme-v2-archive-aware-hash-v1`.
- The pilot split is exactly Oracle 25 videos/100 questions, Canary 4/16, source-holdout 4/16, and development 12/48, ranked with `videomme-v2-partial-pilot-split-v1`.
- Final target benchmarks are exactly LongVideoBench, LVBench, and MLVU. Video-MME-v2 is source/Router-development data, not a final independent target benchmark.
- Preserve model identities and sharing: BGE-M3 for Gist text/embedding, SigLIP2 for Gist visual, one Qwen3-VL snapshot for Residual/Visual, Qwen3-8B for Answerer, `bfloat16`, Hugging Face Transformers.
- Checked-in fixture ZIP/Parquet/MP4 data is engineering evidence only. Real production evidence requires official bytes, immutable hashes, a sealed Authority, raw responses, and measured CostRecords.
- Do not create a human-audit result, reviewer identity, completion timestamp, PASS outcome, provider credential, provider factory, runtime setting, prompt, segmentation policy, frame-sampling policy, or Oracle threshold that was not actually supplied and frozen.
- Do not launch Router training, DAgger, a full benchmark, large observation generation, E03, or E04 unless all upstream gates pass. A pilot result is never a full Video-MME-v2 result.
- Commit implementation and the final verified checked-in asset lock locally in reviewable commits; never push.

---

### Task 1: Generic Dataset Scope and Provenance Contract

**Files:**
- Modify: `src/fidmem/production/manifests.py`
- Modify: `src/fidmem/production/authority.py`
- Modify: `tests/production/test_manifests.py`
- Modify: `tests/production/helpers.py`
- Modify: `tests/production/test_authority_lifecycle.py`

**Interfaces:**
- Produces: `DatasetScope = Literal["PARTIAL_DATASET_PILOT", "FULL_DATASET"]` and the schema-v2 `DatasetManifest` fields `dataset_scope`, `source_metadata_sha256`, `source_archive_index_sha256`, `subset_selection_manifest_sha256`, `selected_video_count`, `selected_question_count`, `available_video_count`, and `available_question_count`.
- Consumes: existing `canonical_sha256()`, `VideoManifest`, `QuestionManifest`, `DatasetIdentity`, and Authority file/hash validation.

- [ ] **Step 1: Write failing manifest provenance tests**

Add tests that construct a pilot manifest with literal lowercase 64-hex hashes and assert that pilot scope requires `subset_selection_manifest_sha256`, selected counts cannot exceed available counts, and full scope rejects a subset hash:

```python
def test_partial_dataset_manifest_requires_selection_identity() -> None:
    with pytest.raises(ValidationError, match="subset selection"):
        DatasetManifest(
            dataset_name="MME-Benchmarks/Video-MME-v2",
            dataset_version="6e4bebb03202e1ddbf3d37703e560e51c5aa2d64",
            dataset_scope="PARTIAL_DATASET_PILOT",
            source_metadata_sha256="1" * 64,
            source_archive_index_sha256="2" * 64,
            subset_selection_manifest_sha256=None,
            selected_video_count=45,
            selected_question_count=180,
            available_video_count=800,
            available_question_count=3200,
            split_policy_id="videomme-v2-pilot-split-v1",
            split_policy_sha256="3" * 64,
            video_manifest_sha256="4" * 64,
            question_manifest_sha256="5" * 64,
        )
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
/mnt/disk1/zhangzheng/zhangzheng/conda_envs/fidmem-a800/bin/python -W error -m pytest -q tests/production/test_manifests.py
```

Expected: the new test fails because the scope/provenance fields and validator do not exist.

- [ ] **Step 3: Implement schema-v2 validation**

Make every new field required. Use a model validator with these exact invariants:

```python
if self.selected_video_count > self.available_video_count:
    raise ValueError("selected video count exceeds available video count")
if self.selected_question_count > self.available_question_count:
    raise ValueError("selected question count exceeds available question count")
if self.dataset_scope == "PARTIAL_DATASET_PILOT" and self.subset_selection_manifest_sha256 is None:
    raise ValueError("partial dataset requires a subset selection identity")
if self.dataset_scope == "FULL_DATASET" and self.subset_selection_manifest_sha256 is not None:
    raise ValueError("full dataset forbids a subset selection identity")
```

Update the production test helper's `DatasetManifest` fixture with `FULL_DATASET`, equal selected/available counts, fixed source hashes, and no subset hash.

- [ ] **Step 4: Bind the new provenance in Authority validation**

After `DatasetManifest.model_validate_json()`, require `dataset_name` and `dataset_version` to equal `draft.dataset`, require selected counts to equal the video/question manifest record counts, and preserve the existing split/video/question hash checks. Add a tampering test that changes `dataset_scope` or a source hash in the manifest file and expects `dataset_manifest_mismatch` or `manifest_hash_mismatch`.

- [ ] **Step 5: Run GREEN and commit**

Run the two focused production test files, then `git diff --check`. Commit only these files:

```bash
git add src/fidmem/production/manifests.py src/fidmem/production/authority.py tests/production/test_manifests.py tests/production/helpers.py tests/production/test_authority_lifecycle.py
git commit -m "Bind dataset scope provenance to authority"
```

---

### Task 2: Selective Dataset Assets and Lock Reconciliation

**Files:**
- Modify: `src/fidmem/assets/stack.py`
- Modify: `src/fidmem/assets/resolver.py`
- Modify: `src/fidmem/assets/cli.py`
- Modify: `tests/assets/test_stack.py`
- Modify: `tests/assets/test_resolver.py`
- Modify: `tests/assets/test_cli.py`

**Interfaces:**
- Produces: `PhysicalAsset.include_files: tuple[str, ...]`, `resolve_entry(entry: AssetLockEntry, *, info_loader: Callable[[str, str], tuple[str, tuple[str, ...]]], required_files: tuple[str, ...] = ()) -> AssetLockEntry`, and `reconcile_lock(stack: ExperimentStack, previous: AssetLock) -> AssetLock`.
- Consumes: existing immutable revision validation, `AssetLock.create()`, `verify_entry()`, and asset CLI lifecycle.

- [ ] **Step 1: Write failing selective-resolution and reconciliation tests**

Use a dataset fixture whose remote listing contains `README.md`, `test.parquet`, `subtitle.zip`, and `videos/001.zip`. Assert that resolving with the first three required files excludes the video archive. Build a previous lock with VERIFIED model entries, reconcile it against a stack where only `source_dataset` changes, and assert all identical model entries retain paths/hashes/states while the new dataset entry is RESOLVED.

```python
resolved = resolve_entry(
    dataset_entry,
    info_loader=lambda _repo, _type: (REVISION, REMOTE_FILES),
    required_files=("README.md", "subtitle.zip", "test.parquet"),
)
assert resolved.expected_files == ("README.md", "subtitle.zip", "test.parquet")
```

Also assert a missing required file fails with `required remote files are missing`.

- [ ] **Step 2: Run RED**

Run:

```bash
/mnt/disk1/zhangzheng/zhangzheng/conda_envs/fidmem-a800/bin/python -W error -m pytest -q tests/assets/test_stack.py tests/assets/test_resolver.py tests/assets/test_cli.py
```

Expected: `PhysicalAsset` rejects `include_files` and the new function/signature is absent.

- [ ] **Step 3: Implement selective resolution**

Normalize `include_files` to a sorted, duplicate-free tuple and reject absolute paths, `..`, and empty names. In `resolve_entry`, validate every required file exists in the remote listing, then store only required files when the tuple is non-empty; continue storing the full remote listing for model assets.

- [ ] **Step 4: Implement identity-preserving reconciliation**

`reconcile_lock()` must match old entries by the exact tuple `(repo_id, repo_type, immutable_revision, backend, dtype)`. Preserve a matched entry byte-for-byte except for its dictionary key. Initialize unmatched assets with the same lifecycle rules as `initial_lock()`. Recompute logical mappings and `lock_sha256`. Never preserve an entry across a changed repository or revision.

Expose CLI action `reconcile`; `--check` reports `preserved_asset_ids` and `reset_asset_ids` without writing. The non-check action atomically writes the reconciled checked-in lock.

- [ ] **Step 5: Run GREEN and inspect the diff**

Run the Task 2 tests and `git diff --check`. Do not commit yet because Task 3 updates the stack and lock in the same consistent commit.

---

### Task 3: Frozen Stack, Target Roles, and Candidate Asset Lock

**Files:**
- Modify: `configs/experiment_stacks/gist_residual_v1.yaml`
- Modify: `configs/experiment_stacks/gist_residual_v1.assets.lock.json`
- Create: `configs/experiment_stacks/videomme_v2_pilot_split_policy.yaml`
- Delete: `configs/experiment_stacks/longtvqa_split_policy.yaml`
- Modify: `tests/assets/test_stack.py`
- Modify: `tests/integration/test_experiment_stack_wiring.py`

**Interfaces:**
- Produces: physical asset `videomme_v2_metadata`, logical `source_dataset: videomme_v2_metadata`, final targets `longvideobench`, `lvbench`, `mlvu`, and frozen pilot split policy `videomme-v2-pilot-split-v1`.
- Consumes: Task 2 selective asset and reconciliation interfaces.

- [ ] **Step 1: Add failing role and target tests**

Assert the stack contains exactly the approved source repository/revision/include files, does not list Video-MME-v2 under `target_benchmarks`, and has exactly the three final targets. Assert model logical roles still share the same physical IDs and their checked-in lock entries remain VERIFIED with unchanged local snapshot hashes.

- [ ] **Step 2: Update the stack and frozen policy**

Use this exact dataset asset:

```yaml
videomme_v2_metadata:
  repo_id: MME-Benchmarks/Video-MME-v2
  repo_type: dataset
  immutable_revision: 6e4bebb03202e1ddbf3d37703e560e51c5aa2d64
  backend: huggingface_hub
  dtype: null
  include_files: [README.md, subtitle.zip, test.parquet]
```

Create a frozen policy containing `dataset_scope: PARTIAL_DATASET_PILOT`, pool count 45, the exact pool/split seeds and algorithm versions from Global Constraints, and group video counts `oracle: 25`, `canary: 4`, `holdout: 4`, `development: 12`. It must not contain hand-picked video IDs or question-dependent criteria.

- [ ] **Step 3: Reconcile the candidate lock without downloading**

Run CLI help first, then:

```bash
PYTHONPATH=src /mnt/disk1/zhangzheng/zhangzheng/conda_envs/fidmem-a800/bin/python -m fidmem.assets.cli reconcile --check
PYTHONPATH=src /mnt/disk1/zhangzheng/zhangzheng/conda_envs/fidmem-a800/bin/python -m fidmem.assets.cli reconcile
```

Expected: four model entries are preserved VERIFIED, `videomme_v2_metadata` is RESOLVED, `longtvqa_metadata` is absent, and the lock hash validates.

- [ ] **Step 4: Run stack/lock tests and commit Tasks 2-3**

Run the Task 2 suite plus `tests/integration/test_experiment_stack_wiring.py`, then `git diff --check`. Commit:

```bash
git add src/fidmem/assets/stack.py src/fidmem/assets/resolver.py src/fidmem/assets/cli.py tests/assets/test_stack.py tests/assets/test_resolver.py tests/assets/test_cli.py tests/integration/test_experiment_stack_wiring.py configs/experiment_stacks/gist_residual_v1.yaml configs/experiment_stacks/gist_residual_v1.assets.lock.json configs/experiment_stacks/videomme_v2_pilot_split_policy.yaml configs/experiment_stacks/longtvqa_split_policy.yaml
git commit -m "Freeze Video-MME-v2 source stack"
```

---

### Task 4: Official Metadata and Subtitle Adapter

**Files:**
- Create: `src/fidmem/assets/videomme_v2.py`
- Create: `tests/assets/test_videomme_v2.py`
- Delete: `src/fidmem/assets/longtvqa.py`
- Delete: `tests/assets/test_longtvqa.py`

**Interfaces:**
- Produces: `VideoMMEQuestion`, `MetadataVerificationReport`, `ParsedVideoMME`, `HumanAuditItem`, `HumanAuditManifest`, `verify_metadata(root, immutable_revision)`, `build_human_audit_manifest(metadata, selected_video_ids, seed, count=100)`, and `validate_human_audit_result(manifest, result_path)`.
- Consumes: DuckDB, `zipfile.ZipFile`, `canonical_sha256()`, and Task 1 manifest types.

- [ ] **Step 1: Create failing official-shape fixture tests**

Build a temporary Parquet through DuckDB with the exact columns `video_id`, `url`, `group_type`, `group_structure`, `question_id`, `question`, `options`, `answer`, `level`, `second_head`, and `third_head`. Create four rows per video and a subtitle ZIP with one JSONL member named from each video ID. Tests must reject duplicate question IDs, any video with other than four questions, blank options/answers, subtitle/video mismatches, unexpected files, and a revision other than the frozen 40-hex value.

- [ ] **Step 2: Run RED**

Run:

```bash
/mnt/disk1/zhangzheng/zhangzheng/conda_envs/fidmem-a800/bin/python -W error -m pytest -q tests/assets/test_videomme_v2.py -k metadata
```

Expected: import fails because `fidmem.assets.videomme_v2` does not exist.

- [ ] **Step 3: Implement deterministic metadata parsing**

Read Parquet without pandas:

```python
connection = duckdb.connect()
cursor = connection.execute("SELECT * FROM read_parquet(?)", [str(parquet_path)])
columns = tuple(item[0] for item in cursor.description)
rows = tuple(dict(zip(columns, values, strict=True)) for values in cursor.fetchall())
```

Require exactly 3,200 rows, 800 unique three-digit video IDs, 3,200 unique stable question IDs, and four rows per video for the real official snapshot; expose count overrides only as private test parameters. Parse question types from nonblank `group_type`, `level`, `second_head`, and `third_head`, using `videomme-v2-unlabeled` only when all four are blank. Hash the ordered metadata files as `{path, size, sha256}` records.

- [ ] **Step 4: Implement pending human-audit behavior**

Rank only questions whose `video_id` is selected using `canonical_sha256({"seed": seed, "video_id": item.video_id, "question_id": item.question_id})`. Emit exactly 100 items with status `PENDING_HUMAN_AUDIT`. `validate_human_audit_result()` requires a bound manifest hash, real nonblank reviewer/completion identities, the same 100 question IDs, and every outcome PASS. It never writes a result.

- [ ] **Step 5: Run GREEN and commit**

Run all adapter tests and existing production manifest tests. Commit:

```bash
git add src/fidmem/assets/videomme_v2.py tests/assets/test_videomme_v2.py src/fidmem/assets/longtvqa.py tests/assets/test_longtvqa.py
git commit -m "Add Video-MME-v2 metadata adapter"
```

---

### Task 5: Official Archive Index and Deterministic Pilot Selection

**Files:**
- Modify: `src/fidmem/assets/videomme_v2.py`
- Modify: `tests/assets/test_videomme_v2.py`

**Interfaces:**
- Produces: `OfficialFileIdentity`, `ArchiveMemberIdentity`, `ArchiveIndex`, `PilotSelectionManifest`, `build_archive_index(file_identities, opener)`, `select_pilot(metadata, archive_index, count=45)`, and `full_scope_media(archive_index) -> tuple[tuple[str, ...], tuple[str, ...]]`.
- Consumes: `HfApi.dataset_info(revision=IMMUTABLE_REVISION, files_metadata=True)`, `HfFileSystem(token=False).open()`, and Task 4 parsed metadata.

- [ ] **Step 1: Write failing archive safety and selection tests**

Create three engineering ZIPs with 20 numbered MP4 members apiece and an injected seekable opener. Assert rejection of `../escape.mp4`, absolute paths, duplicate video stems, non-MP4 video members, unknown metadata IDs, missing metadata IDs, absent upstream SHA-256, and mismatched file size. Assert two repeated selections have the same selected archive/video IDs and selection hash and never inspect question text/answer.

Also assert `full_scope_media()` returns all 40 archive paths and all 800 video IDs in canonical order and does not create a pilot subset-selection hash.

- [ ] **Step 2: Run RED**

Run:

```bash
/mnt/disk1/zhangzheng/zhangzheng/conda_envs/fidmem-a800/bin/python -W error -m pytest -q tests/assets/test_videomme_v2.py -k "archive or pilot"
```

Expected: the archive/index types and functions are absent.

- [ ] **Step 3: Implement pinned remote identity loading**

Accept only siblings `videos/001.zip` through `videos/040.zip`. For each sibling require `size > 0` and `lfs.sha256` matching lowercase 64-hex. Open remote ZIPs with:

```python
path = f"datasets/{DATASET_ID}@{IMMUTABLE_REVISION}/{archive_path}"
with HfFileSystem(token=False).open(path, "rb", block_size=1024 * 1024) as stream:
    index = inspect_zip_central_directory(stream, official_identity)
```

`build_archive_index()` records archive path/hash/size and member path/CRC/compressed/uncompressed sizes without reading MP4 payload bytes. Canonically sort archives and members before hashing.

- [ ] **Step 4: Implement archive-aware pilot selection**

Rank every video by `canonical_sha256({"algorithm": POOL_ALGORITHM, "seed": POOL_SEED, "video_id": video_id})`. Add the archive of the highest-ranked uncovered video until the archive union contains at least 45 videos, rank that union with the same function, and choose the first 45. Bind metadata hash, archive-index hash, seeds, algorithm, selected archives, selected videos, available/selected counts, and `selection_sha256`.

- [ ] **Step 5: Run GREEN and commit**

Run the complete adapter test file and `git diff --check`. Commit:

```bash
git add src/fidmem/assets/videomme_v2.py tests/assets/test_videomme_v2.py
git commit -m "Index official Video-MME-v2 archives"
```

---

### Task 6: Resumable Verified Download and Atomic Extraction

**Files:**
- Modify: `src/fidmem/assets/videomme_v2.py`
- Modify: `tests/assets/test_videomme_v2.py`

**Interfaces:**
- Produces: `DatasetPreparationResult`, `DownloadPlan`, `check_download_capacity(plan: DownloadPlan, root: Path) -> None`, `download_pinned_file(identity: OfficialFileIdentity, destination: Path, resume: bool, http_getter: Callable[..., None]) -> Path`, `extract_selected_media(video_ids: tuple[str, ...], archive_index: ArchiveIndex, archive_root: Path, video_root: Path, subtitle_zip: Path) -> tuple[Path, ...]`, and `prepare_videos(metadata: ParsedVideoMME, raw_root: Path, cache_root: Path, *, scope: Literal["pilot", "full"], check: bool, resume: bool, verify_only: bool) -> DatasetPreparationResult`.
- Consumes: `huggingface_hub.hf_hub_url()`, `huggingface_hub.file_download.http_get()`, Task 5 identities, and existing filesystem SHA-256/video utilities.

- [ ] **Step 1: Write failing resume, hash, capacity, and extraction tests**

Use an injected `http_getter(url, stream, resume_size, expected_size)` that appends fixture bytes. Assert a partial sibling resumes from its exact size, a completed hash-matching archive is reused, a mismatch never replaces the last verified archive, insufficient space fails before invoking the getter, and extraction writes only the selected 45 MP4s. Include ZIP symlink/path traversal and CRC failure tests.

Assert pilot scope plans only the deterministically selected archives/videos, while full scope plans all 40 archives and 800 videos and applies the same capacity/hash/extraction checks.

- [ ] **Step 2: Run RED**

Run:

```bash
/mnt/disk1/zhangzheng/zhangzheng/conda_envs/fidmem-a800/bin/python -W error -m pytest -q tests/assets/test_videomme_v2.py -k "download or resume or extract or capacity"
```

Expected: download/extraction functions are absent.

- [ ] **Step 3: Implement fail-closed capacity and resume**

Required bytes equal remaining selected archive bytes plus selected uncompressed MP4 bytes plus 20 GiB safety margin. Download each ZIP into a sibling whose suffix is `.partial`, pass its size as `resume_size`, fsync, recompute whole-file SHA-256, and only then `os.replace()` the final archive. `--verify-only` rejects missing partial/final assets without a network call. Log paths, byte counts, and states, never credentials.

- [ ] **Step 4: Implement atomic selected extraction**

Normalize every member with `PurePosixPath`; reject absolute paths, `..`, symlinks, directories masquerading as media, and unexpected selected names. Extract a selected member to a temporary sibling, fsync, hash, and atomically replace the MP4 named from its video ID. Extract the selected subtitle JSONLs with the same rules. On resume, reuse an existing output only when its recorded SHA-256 and size still match.

- [ ] **Step 5: Run GREEN and commit**

Run all Video-MME tests and `git diff --check`. Commit:

```bash
git add src/fidmem/assets/videomme_v2.py tests/assets/test_videomme_v2.py
git commit -m "Add resumable Video-MME-v2 media preparation"
```

---

### Task 7: Source Gate, Resolved Split, and E01 Artifacts

**Files:**
- Modify: `src/fidmem/assets/videomme_v2.py`
- Modify: `src/fidmem/assets/setup.py`
- Modify: `tests/assets/test_videomme_v2.py`
- Modify: `tests/assets/test_authority_draft.py`

**Interfaces:**
- Produces: `RawVideoVerificationReport`, `ResolvedSplitPolicy`, `build_pilot_split()`, `verify_raw_videos()`, `build_manifests()`, `write_dataset_preparation()`, and `prepare_e01()`.
- Consumes: Task 1 schema-v2 manifests, Task 4 audit contract, Task 5 selection, Task 6 extracted media, existing `probe_video()`/`sample_frames()`, and `build_authority_draft()`.

- [ ] **Step 1: Write failing Source Gate and exact split tests**

Create 45 selected fixture videos with injected probes/decoders and 180 questions. Assert exact group counts 25/4/4/12 and 100/16/16/48, strict video disjointness, Canary/Oracle selection sizes 16/100, gold hashes only on Oracle/holdout records, duplicate content rejection, at least 20 deterministic midpoint decodes, and missing annotation/subtitle/video failures.

- [ ] **Step 2: Run RED**

Run:

```bash
/mnt/disk1/zhangzheng/zhangzheng/conda_envs/fidmem-a800/bin/python -W error -m pytest -q tests/assets/test_videomme_v2.py -k "source_gate or split or manifest or human"
```

Expected: split/source functions are absent or still LongTVQA-specific.

- [ ] **Step 3: Implement the resolved split and manifest construction**

Rank the 45 selected IDs by `canonical_sha256({"seed": SPLIT_SEED, "video_id": video_id})`, slice 25/4/4/12 in Oracle/Canary/holdout/development order, and write a canonical resolved split artifact binding its checked-in source policy SHA-256 and full `video_groups`. Build schema-v2 `DatasetManifest` with official metadata/index/subset hashes and 45/180 selected versus 800/3200 available counts.

- [ ] **Step 4: Separate data preparation from formal E01 completion**

`write_dataset_preparation()` writes metadata verification, archive index, subset selection, raw-video verification, pending human audit, resolved split policy, video/question/dataset manifests, Canary/Oracle selection manifests, and `source_gate.json` with `status: PENDING_HUMAN_AUDIT` when no result exists. `prepare_e01()` reads those exact artifacts, validates every hash again, requires the bound real human result, and only then returns `source_gate: PASS` for the formal experiment. Neither function generates a PASS result.

- [ ] **Step 5: Bind Authority to the resolved split artifact**

Change Authority draft construction to use `Path(os.environ["FIDMEM_E01_RESULTS_ROOT"]) / "split_policy.json"`, not the checked-in algorithm policy, and validate that the dataset manifest binds the same split-policy file SHA-256. Add an Authority test that mutates the resolved mapping and expects fail closed.

- [ ] **Step 6: Run GREEN and commit**

Run Video-MME, Authority draft, Authority lifecycle, and manifest tests. Commit:

```bash
git add src/fidmem/assets/videomme_v2.py src/fidmem/assets/setup.py tests/assets/test_videomme_v2.py tests/assets/test_authority_draft.py
git commit -m "Build Video-MME-v2 pilot manifests"
```

---

### Task 8: CLI, Wrappers, Environment, Experiment Config, and Runbooks

**Files:**
- Modify: `.env.example`
- Rename: `scripts/setup/04_download_longtvqa_metadata.sh` to `scripts/setup/04_download_videomme_v2_metadata.sh`
- Rename: `scripts/setup/05_verify_longtvqa_metadata.sh` to `scripts/setup/05_verify_videomme_v2_metadata.sh`
- Rename: `scripts/setup/06_verify_longtvqa_videos.sh` to `scripts/setup/06_prepare_videomme_v2_videos.sh`
- Rename: `scripts/setup/07_build_longtvqa_manifests.sh` to `scripts/setup/07_build_videomme_v2_manifests.sh`
- Modify: `tests/assets/test_setup_wrappers.py`
- Modify: `configs/experiments/e01_dataset.yaml`
- Modify: `configs/experiments/e13_main.yaml`
- Modify: `configs/experiments/registry.yaml`
- Modify: `docs/experiments/README.md`
- Modify: `docs/experiments/GPU_RUNBOOK.md`
- Modify: `docs/experiments/EXPERIMENT_MATRIX.md`
- Modify: `docs/experiments/GPU_BUDGET.md`
- Modify: `docs/RUNBOOK.md`
- Modify: `tests/integration/test_experiment_stack_wiring.py`

**Interfaces:**
- Produces: setup CLI flags `--check`, `--resume`, `--verify-only`, `--scope {pilot,full}`, `--output`, and environment names `FIDMEM_VIDEOMME_V2_RAW_ROOT`, `FIDMEM_VIDEOMME_V2_PREPARATION_ROOT`, `FIDMEM_VIDEOMME_V2_HUMAN_AUDIT_RESULT`.
- Consumes: Tasks 3-7 adapter/setup functions and the unchanged experiment pack CLI.

- [ ] **Step 1: Write failing wrapper/config integration tests**

Assert the exact eight setup wrapper names and entrypoints, E01 required environment/path keys including `FIDMEM_VIDEOMME_V2_PREPARATION_ROOT`, source asset `videomme_v2_metadata`, resolved split output, E13 benchmark list `[longvideobench, lvbench, mlvu]`, and unchanged E00-E17 IDs/dependencies/gates. Assert no current config, active runbook, setup wrapper, or `src/fidmem/assets` file contains `LongTVQA`.

- [ ] **Step 2: Run RED**

Run:

```bash
/mnt/disk1/zhangzheng/zhangzheng/conda_envs/fidmem-a800/bin/python -W error -m pytest -q tests/assets/test_setup_wrappers.py tests/integration/test_experiment_stack_wiring.py
```

Expected: old wrapper names/environment/config references fail.

- [ ] **Step 3: Expose actual setup CLI semantics**

The `videos` command accepts all three modes and defaults to pilot scope; `--check` may read official remote metadata/ZIP central directories but never downloads payload MP4s or extracts. `--resume` downloads/extracts required official pilot assets. `--verify-only` performs no network download and re-hashes/probes local assets. The `manifests` command writes preparation artifacts even when audit is pending. The formal `e01` command reads `FIDMEM_VIDEOMME_V2_PREPARATION_ROOT`, revalidates those artifacts, and requires the audit result.

For `--scope full`, the video command plans/downloads/verifies all 40 archives and 800 MP4s under a distinct `FULL_DATASET` namespace. Full manifest/E01 construction fails closed until a separate checked-in frozen full split policy exists; it never derives that policy from pilot outcomes.

- [ ] **Step 4: Migrate active configuration and documentation**

Use `/mnt/disk1/zhangzheng/Tvqa_data/Raw/Video-MME-v2` as the documented server raw root, while `.env.example` retains the portable example `/data/fidmem/datasets/Video-MME-v2`. Replace source and target roles only; leave historical specs/plans unchanged. Document pilot namespace, expected 45/180 counts, archive hash/resume behavior, pending audit stop, academic-research-only local handling/no redistribution, and the exact setup/E00-E02 commands.

- [ ] **Step 5: Run CLI help and integration GREEN**

Run `--help` for asset CLI, setup CLI subcommands, and E00-E03 wrappers. Run the focused tests and registry validation:

```bash
PYTHONPATH=src /mnt/disk1/zhangzheng/zhangzheng/conda_envs/fidmem-a800/bin/python -m fidmem.experiments.pack_cli --validate-registry
```

- [ ] **Step 6: Commit**

Run `git diff --check`, inspect `git diff --stat`, and commit the wrapper/config/doc migration:

```bash
git add .env.example scripts/setup tests/assets/test_setup_wrappers.py configs/experiments/e01_dataset.yaml configs/experiments/e13_main.yaml configs/experiments/registry.yaml docs/experiments docs/RUNBOOK.md tests/integration/test_experiment_stack_wiring.py
git commit -m "Wire Video-MME-v2 into experiment pack"
```

---

### Task 9: Full Engineering Regression and Frozen Source Commit

**Files:**
- Verify: all changed source/config/test files
- Modify only if a real regression identifies a scoped defect

**Interfaces:**
- Produces: a clean implementation commit identity suitable for `FIDMEM_GIT_COMMIT` and subsequent production artifact provenance.
- Consumes: Tasks 1-8.

- [ ] **Step 1: Run focused suites with warnings as errors**

Run:

```bash
/mnt/disk1/zhangzheng/zhangzheng/conda_envs/fidmem-a800/bin/python -W error -m pytest -q tests/assets tests/production tests/integration
```

- [ ] **Step 2: Run the complete test suite**

Run:

```bash
/mnt/disk1/zhangzheng/zhangzheng/conda_envs/fidmem-a800/bin/python -W error -m pytest -q
```

Expected: PASS with no warning promoted to an error.

- [ ] **Step 3: Verify source-role invariants**

Run `rg` over active source/config/scripts/runbooks for LongTVQA, over target configs for Video-MME-v2, and over model role mappings. Expected: no active LongTVQA source reference, no Video-MME-v2 final-target reference, and unchanged shared model physical identities.

- [ ] **Step 4: Freeze a clean implementation identity**

Run `git status --short --branch`, `git diff --check`, and `git rev-parse HEAD`. If a scoped regression fix was required, commit only that fix and its test. Do not begin real asset work until the tree is clean.

---

### Task 10: Real Metadata Download, Verification, and Checked-In Verified Lock

**Files:**
- Modify: `configs/experiment_stacks/gist_residual_v1.assets.lock.json`
- Create outside Git: `/mnt/disk1/zhangzheng/fidmem/datasets/videomme_v2_metadata/`
- Reuse outside Git: `/mnt/disk1/zhangzheng/fidmem/models/{bge_m3,siglip2_so400m_patch14_384,qwen3_vl_8b_instruct,qwen3_8b}/`

**Interfaces:**
- Produces: a checked-in fully VERIFIED asset lock with a new lock SHA-256 and a clean Git commit.
- Consumes: approved official network source and existing verified model snapshots.

- [ ] **Step 1: Export real server paths without logging credentials**

Set `FIDMEM_DATA_ROOT=/mnt/disk1/zhangzheng/fidmem/datasets`, `FIDMEM_MODEL_ROOT=/mnt/disk1/zhangzheng/fidmem/models`, `FIDMEM_CACHE_ROOT=/mnt/disk1/zhangzheng/fidmem/cache`, `FIDMEM_ARTIFACT_ROOT=$PWD/artifacts`, `FIDMEM_VIDEOMME_V2_RAW_ROOT=/mnt/disk1/zhangzheng/Tvqa_data/Raw/Video-MME-v2`, `FIDMEM_VIDEOMME_V2_PREPARATION_ROOT=$PWD/artifacts/dataset-preparation/videomme-v2-pilot-v1`, `CUDA_VISIBLE_DEVICES=0`, and `FIDMEM_GIT_COMMIT` to the current clean 40-hex HEAD. Keep `FIDMEM_PROVIDER_BACKEND_FACTORY` unset.

- [ ] **Step 2: Run actual help and no-download checks**

Run the real commands in this order and record command, exit code, key output, and target path:

```bash
bash scripts/setup/01_resolve_stack_assets.sh --check
bash scripts/setup/04_download_videomme_v2_metadata.sh --check
bash scripts/setup/03_verify_models.sh --verify-only
```

All must pass before download.

- [ ] **Step 3: Download and verify only the pinned metadata files**

Run:

```bash
bash scripts/setup/04_download_videomme_v2_metadata.sh --resume
bash scripts/setup/05_verify_videomme_v2_metadata.sh --check
bash scripts/setup/03_verify_models.sh --verify-only
```

Expected dataset snapshot files are exactly `README.md`, `subtitle.zip`, and `test.parquet`; no `videos/*.zip` is downloaded by the metadata asset command.

- [ ] **Step 4: Verify and commit the new asset lock**

Run the all-asset local verification command:

```bash
bash scripts/setup/01_resolve_stack_assets.sh --verify-only
```

Confirm all five physical entries are VERIFIED and model snapshot hashes are unchanged. Run the full test suite and commit only the verified lock:

```bash
git add configs/experiment_stacks/gist_residual_v1.assets.lock.json
git commit -m "Freeze verified Video-MME-v2 asset identities"
```

Then confirm `git status --short --branch` is clean. Never push.

---

### Task 11: Real Pilot Archive Download, Extraction, and Data Preparation

**Files:**
- Create outside Git: `/mnt/disk1/zhangzheng/Tvqa_data/Raw/Video-MME-v2/archives/`
- Create outside Git: `/mnt/disk1/zhangzheng/Tvqa_data/Raw/Video-MME-v2/videos/`
- Create outside Git: `/mnt/disk1/zhangzheng/Tvqa_data/Raw/Video-MME-v2/subtitles/`
- Create: `artifacts/dataset-preparation/videomme-v2-pilot-v1/`

**Interfaces:**
- Produces: official archive index, deterministic selection, verified 45-video pilot, pending human-audit manifest, resolved split, dataset/video/question manifests, and Canary/Oracle selections.
- Consumes: Task 10 verified metadata lock and Task 6 resumable downloader.

- [ ] **Step 1: Check disk and build the no-payload remote plan**

Run:

```bash
df -h /mnt/disk1/zhangzheng
bash scripts/setup/06_prepare_videomme_v2_videos.sh --check --scope pilot --output artifacts/dataset-preparation/videomme-v2-pilot-v1
```

Record selected archive count/paths/LFS hashes, 45 selected video IDs, expected archive bytes, expected extracted bytes, available bytes, and selection SHA-256. Stop before download if the 20 GiB safety margin cannot be preserved.

- [ ] **Step 2: Download and extract the official pilot**

Run the resumable command in tmux session `fidmem-videomme-v2-download`:

```bash
bash scripts/setup/06_prepare_videomme_v2_videos.sh --resume --scope pilot --output artifacts/dataset-preparation/videomme-v2-pilot-v1
```

Monitor with `tmux capture-pane -pt fidmem-videomme-v2-download -S -80`, disk usage, and file counts. Do not delete any pre-existing data if capacity changes; safely interrupt and retain partials for resume.

- [ ] **Step 3: Perform offline full verification**

Run:

```bash
bash scripts/setup/06_prepare_videomme_v2_videos.sh --verify-only --scope pilot --output artifacts/dataset-preparation/videomme-v2-pilot-v1
bash scripts/setup/07_build_videomme_v2_manifests.sh --check --output artifacts/dataset-preparation/videomme-v2-pilot-v1
bash scripts/setup/07_build_videomme_v2_manifests.sh --output artifacts/dataset-preparation/videomme-v2-pilot-v1
```

Expected: archive/MP4 hashes pass; 45 videos and 180 questions exist; split counts are exactly 25/4/4/12 and 100/16/16/48; at least 20 midpoint decodes pass; Source Gate remains `PENDING_HUMAN_AUDIT` because no human result exists.

- [ ] **Step 4: Re-run verify-only to prove resume idempotence**

Capture archive/video mtimes, run `--resume` once more, and confirm no verified archive/video is downloaded or rewritten. Recompute all manifest hashes and confirm they are identical. This is engineering resume evidence, not provider billing evidence.

---

### Task 12: E00, E01/E02 Fail-Closed Audit, and Conditional Handoff

**Files:**
- Create: `artifacts/experiments/E00/environment-a800-v1/`
- Conditionally create: `artifacts/experiments/E01/videomme-v2-pilot-freeze-v1/`
- Conditionally create: `artifacts/authority/ProductionAuthorityDraft.json`
- Conditionally create: `artifacts/experiments/E02/authority-a800-v1/`

**Interfaces:**
- Produces: real environment evidence and every currently valid gate artifact; otherwise produces an explicit blocker report with no fabricated PASS.
- Consumes: clean Task 10 Git identity, Task 11 manifests, real human result if later supplied, frozen production configs if later supplied, and actual GPU 0 runtime.

- [ ] **Step 1: Re-audit the target host and run E00 check**

Run the actual CLI help first, then `nvidia-smi`, `nvidia-smi -L`, Python/PyTorch/CUDA/cuDNN checks, ffmpeg, disk, Git status/diff/HEAD, and:

```bash
bash scripts/experiments/00_environment.sh --check --run-id environment-a800-v1
```

Confirm GPU 0 identity, full Git commit, clean state, real absolute paths, and planned artifact directory before formal E00.

- [ ] **Step 2: Execute E00 and record `environment_ready`**

Run:

```bash
bash scripts/experiments/00_environment.sh --run-id environment-a800-v1
```

Build the gate only through `fidmem.experiments.gate_cli` from E00's real `metadata.json`, `results/environment.json`, and frozen `environment_ready.yaml`:

```bash
E00_CONFIG_SHA256="$(/mnt/disk1/zhangzheng/zhangzheng/conda_envs/fidmem-a800/bin/python -c 'import json; print(json.load(open("artifacts/experiments/E00/environment-a800-v1/metadata.json", encoding="utf-8"))["config_sha256"])')"
PYTHONPATH=src /mnt/disk1/zhangzheng/zhangzheng/conda_envs/fidmem-a800/bin/python -m fidmem.experiments.gate_cli --gate environment_ready --experiment E00 --run-id environment-a800-v1 --protocol-version fidelity-memory-paper-v1 --config-sha256 "$E00_CONFIG_SHA256" --result artifacts/experiments/E00/environment-a800-v1/results/environment.json --thresholds configs/experiments/gates/environment_ready.yaml --output artifacts/experiment-gates/environment_ready.json
```

Do not hand-write PASS.

- [ ] **Step 3: Run formal E01 check and honor the human-audit stop**

Set `FIDMEM_E01_RESULTS_ROOT` only after a formal E01 completes. Export `FIDMEM_VIDEOMME_V2_PREPARATION_ROOT=$PWD/artifacts/dataset-preparation/videomme-v2-pilot-v1` and the reserved, currently absent result path `FIDMEM_VIDEOMME_V2_HUMAN_AUDIT_RESULT=$PWD/artifacts/longtvqa-human-audit-result.json`; pointing at an absent path is not evidence and allows the adapter to report the exact missing audit. Run:

```bash
bash scripts/experiments/01_dataset.sh --check --run-id videomme-v2-pilot-freeze-v1
```

Given the currently known missing result, expected exit is fail closed with `human timestamp audit result is missing`. Report the pending audit-manifest path/SHA-256 and do not create `dataset_frozen`.

- [ ] **Step 4: Conditionally complete E01 after real human evidence exists**

Only after the real result validates, rerun E01 check, execute E01, and build `dataset_frozen` from the real `results/source_gate.json`. Confirm all produced manifest hashes equal Task 11 and no media or gold answers leaked into development/Canary artifacts. Use E01's real config hash:

```bash
E01_CONFIG_SHA256="$(/mnt/disk1/zhangzheng/zhangzheng/conda_envs/fidmem-a800/bin/python -c 'import json; print(json.load(open("artifacts/experiments/E01/videomme-v2-pilot-freeze-v1/metadata.json", encoding="utf-8"))["config_sha256"])')"
PYTHONPATH=src /mnt/disk1/zhangzheng/zhangzheng/conda_envs/fidmem-a800/bin/python -m fidmem.experiments.gate_cli --gate dataset_frozen --experiment E01 --run-id videomme-v2-pilot-freeze-v1 --protocol-version fidelity-memory-paper-v1 --config-sha256 "$E01_CONFIG_SHA256" --result artifacts/experiments/E01/videomme-v2-pilot-freeze-v1/results/source_gate.json --thresholds configs/experiments/gates/dataset_frozen.yaml --output artifacts/experiment-gates/dataset_frozen.json
```

- [ ] **Step 5: Audit Authority Draft and E02 without weakening unresolved gates**

Only if `dataset_frozen` exists, set `FIDMEM_E01_RESULTS_ROOT` to the formal E01 results and run:

```bash
bash scripts/setup/08_build_authority_draft.sh --check
```

Record every unresolved prompt, segmentation, frame-sampling, per-model runtime/decode, and runtime field. Do not write or seal a production-ready Authority while any remain. Keep `FIDMEM_PROVIDER_BACKEND_FACTORY` unset.

- [ ] **Step 6: Conditionally seal E02 on GPU 0**

Only after research-owner fields are frozen in a new clean commit and the draft check reports no unresolved fields, build the draft, run E02 `--check --gpus 0 --run-id authority-a800-v1`, execute E02, and record `authority_sealed` from the real result. Re-verify dataset/model/prompt/config/runtime identities and report the Authority SHA-256. Read `config_sha256` from E02's real `metadata.json` and `authority_sha256` from its real gate result:

```bash
E02_CONFIG_SHA256="$(/mnt/disk1/zhangzheng/zhangzheng/conda_envs/fidmem-a800/bin/python -c 'import json; print(json.load(open("artifacts/experiments/E02/authority-a800-v1/metadata.json", encoding="utf-8"))["config_sha256"])')"
E02_AUTHORITY_SHA256="$(/mnt/disk1/zhangzheng/zhangzheng/conda_envs/fidmem-a800/bin/python -c 'import json; print(json.load(open("artifacts/experiments/E02/authority-a800-v1/results/authority_gate_result.json", encoding="utf-8"))["authority_sha256"])')"
PYTHONPATH=src /mnt/disk1/zhangzheng/zhangzheng/conda_envs/fidmem-a800/bin/python -m fidmem.experiments.gate_cli --gate authority_sealed --experiment E02 --run-id authority-a800-v1 --protocol-version fidelity-memory-paper-v1 --config-sha256 "$E02_CONFIG_SHA256" --authority-sha256 "$E02_AUTHORITY_SHA256" --result artifacts/experiments/E02/authority-a800-v1/results/authority_gate_result.json --thresholds configs/experiments/gates/authority_sealed.yaml --output artifacts/experiment-gates/authority_sealed.json
```

- [ ] **Step 7: Stop before E03 unless every production prerequisite is real**

E03 is launchable only if E02 is sealed, the 16-question Canary selection is bound to that Authority, a production Transformers backend factory and request manifest exist, provider/model revisions match, and GPU/disk checks pass. If any item is absent, report `STOP AT AUTHORITY` and list exact missing paths/fields. Do not run E03 or E04 from this plan; once those prerequisites are genuinely frozen, write a focused E03/E04 execution plan with expected call counts, VRAM range, output namespace, resume command, Authority/selection hashes, and measured CostRecord reconciliation.

---

## Completion Evidence

The implementation handoff must report:

- the clean Git commit and checked-in asset-lock SHA-256;
- whether the source migration tests and full suite pass;
- official metadata revision and hashes;
- pilot/full scope marker, selected archive identities, videos/questions, and disk use;
- archive-index, subset-selection, split-policy, dataset, video, question, Canary, Oracle, and human-audit manifest paths/SHA-256 values;
- unchanged model paths/revisions/snapshot hashes and sharing identities;
- E00/E01/E02 command lines, exit codes, artifact paths, gate verdicts, and blockers;
- whether E03 is launchable, without describing pilot preparation as a paper benchmark result.
