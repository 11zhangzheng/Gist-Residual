"""Strict Oracle records and deterministic video-grouped BC datasets.

Rows contain snapshots of already-cached observations. Loading and splitting
these rows is intentionally data-only: it has no provider or VLM callback and
therefore cannot rebuild expensive observations for another random seed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

import duckdb
import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    model_validator,
)
from torch import Tensor
from torch.utils.data import Dataset

from fidmem.types import ActionInstance, ActionType, FidelityLevel, RouterState


_ACTION_INDEX = {action: index for index, action in enumerate(ActionType)}
_FIDELITY_INDEX = {level: index for index, level in enumerate(FidelityLevel)}
_NONE_FIDELITY = len(_FIDELITY_INDEX)
_VISUAL_BUDGET_INDEX = {None: 0, "low": 1, "high": 2}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


class OracleBCRecord(BaseModel):
    """One supervised state with a complete candidate-instance action set."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, revalidate_instances="always"
    )

    record_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    observation_snapshot_id: str = Field(min_length=1)
    state: RouterState
    action_instances: tuple[ActionInstance, ...] = Field(min_length=1)
    legal_action_mask: tuple[StrictBool, ...] = Field(min_length=1)
    target_action_index: StrictInt = Field(ge=0)
    sufficiency_target: Literal[0, 1]
    cost_to_go: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_nested_state_and_action_mask(self) -> "OracleBCRecord":
        RouterState.model_validate(self.state.model_dump(mode="python"))
        for action in self.action_instances:
            ActionInstance.model_validate(action.model_dump(mode="python"))
        if len(set(self.action_instances)) != len(self.action_instances):
            raise ValueError("action_instances must be unique")
        if len(self.legal_action_mask) != len(self.action_instances):
            raise ValueError("legal_action_mask must match action_instances")
        if not any(self.legal_action_mask):
            raise ValueError("record must contain at least one legal action")
        if self.target_action_index >= len(self.action_instances):
            raise ValueError("target_action_index is out of range")
        if not self.legal_action_mask[self.target_action_index]:
            raise ValueError("target action must be legal")
        if not math.isfinite(self.cost_to_go):
            raise ValueError("cost_to_go must be finite")
        return self


class SplitManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    seed: StrictInt
    dataset_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    train_video_ids: tuple[str, ...]
    dev_video_ids: tuple[str, ...]
    test_video_ids: tuple[str, ...]
    train_record_ids: tuple[str, ...]
    dev_record_ids: tuple[str, ...]
    test_record_ids: tuple[str, ...]


class OracleBCDataset(Dataset[OracleBCRecord]):
    """Canonical, immutable view of cached Oracle supervision."""

    def __init__(self, records: Sequence[OracleBCRecord]) -> None:
        validated = tuple(
            OracleBCRecord.model_validate(record.model_dump(mode="python"))
            for record in records
        )
        if not validated:
            raise ValueError("Oracle BC dataset must not be empty")
        ids = tuple(record.record_id for record in validated)
        if len(set(ids)) != len(ids):
            raise ValueError("Oracle BC record ids must be unique")
        self._records = tuple(sorted(validated, key=lambda record: record.record_id))
        payload = "\n".join(
            _canonical_json(record.model_dump(mode="json")) for record in self._records
        )
        self.identity = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def records(self) -> tuple[OracleBCRecord, ...]:
        return self._records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> OracleBCRecord:
        return self._records[index]

    def subset(self, record_ids: Sequence[str]) -> "OracleBCDataset":
        wanted = set(record_ids)
        selected = tuple(
            record for record in self._records if record.record_id in wanted
        )
        if len(selected) != len(wanted):
            raise ValueError("split manifest contains unknown record ids")
        return OracleBCDataset(selected)


def build_grouped_split(
    records: Sequence[OracleBCRecord],
    *,
    seed: int,
    train_fraction: float = 0.8,
    dev_fraction: float = 0.1,
) -> SplitManifest:
    """Assign whole videos to deterministic splits, independent of row order."""

    for name, value in (
        ("train_fraction", train_fraction),
        ("dev_fraction", dev_fraction),
    ):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if train_fraction + dev_fraction > 1:
        raise ValueError("train_fraction + dev_fraction must not exceed one")
    dataset = OracleBCDataset(records)
    videos = sorted(
        {record.video_id for record in dataset.records},
        key=lambda video_id: (
            hashlib.sha256(f"{seed}\0{video_id}".encode("utf-8")).digest(),
            video_id,
        ),
    )
    count = len(videos)
    train_count = min(count, int(math.floor(count * train_fraction)))
    if train_fraction > 0 and train_count == 0:
        train_count = 1
    remaining = count - train_count
    dev_count = min(remaining, int(math.floor(count * dev_fraction)))
    if dev_fraction > 0 and remaining > 0 and dev_count == 0:
        dev_count = 1
    train_videos = tuple(sorted(videos[:train_count]))
    dev_videos = tuple(sorted(videos[train_count : train_count + dev_count]))
    test_videos = tuple(sorted(videos[train_count + dev_count :]))

    def record_ids(video_ids: tuple[str, ...]) -> tuple[str, ...]:
        selected = set(video_ids)
        return tuple(
            record.record_id
            for record in dataset.records
            if record.video_id in selected
        )

    return SplitManifest(
        seed=seed,
        dataset_hash=dataset.identity,
        train_video_ids=train_videos,
        dev_video_ids=dev_videos,
        test_video_ids=test_videos,
        train_record_ids=record_ids(train_videos),
        dev_record_ids=record_ids(dev_videos),
        test_record_ids=record_ids(test_videos),
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("refusing to replace a symlink")
    handle = tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_oracle_records(path: str | Path, records: Sequence[OracleBCRecord]) -> None:
    """Write strict JSONL or a one-column Parquet without pickle payloads."""

    target = Path(path)
    dataset = OracleBCDataset(records)
    rows = tuple(
        _canonical_json(record.model_dump(mode="json")) for record in dataset.records
    )
    if target.suffix.casefold() == ".jsonl":
        _atomic_write(target, ("\n".join(rows) + "\n").encode("utf-8"))
        return
    if target.suffix.casefold() != ".parquet":
        raise ValueError("Oracle records path must end in jsonl or parquet")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError("refusing to replace a symlink")
    descriptor, raw_path = tempfile.mkstemp(dir=target.parent, suffix=".parquet")
    os.close(descriptor)
    temporary = Path(raw_path)
    try:
        connection = duckdb.connect()
        try:
            connection.execute(
                "CREATE TABLE oracle_records(record_json VARCHAR NOT NULL)"
            )
            connection.executemany(
                "INSERT INTO oracle_records VALUES (?)", ((row,) for row in rows)
            )
            quoted = str(temporary.resolve()).replace("'", "''")
            connection.execute(f"COPY oracle_records TO '{quoted}' (FORMAT PARQUET)")
        finally:
            connection.close()
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def load_oracle_records(path: str | Path) -> tuple[OracleBCRecord, ...]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.casefold()
    if suffix == ".jsonl":
        raw_rows = tuple(
            line
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    elif suffix == ".parquet":
        connection = duckdb.connect()
        try:
            cursor = connection.execute(
                "SELECT * FROM read_parquet(?)", [str(source.resolve())]
            )
            columns = tuple(item[0] for item in cursor.description)
            if columns != ("record_json",):
                raise ValueError("Oracle parquet must contain exactly record_json")
            raw_rows = tuple(row[0] for row in cursor.fetchall())
        finally:
            connection.close()
    else:
        raise ValueError("Oracle records path must end in jsonl or parquet")
    if not raw_rows or any(not isinstance(row, str) for row in raw_rows):
        raise ValueError("Oracle record source must contain non-empty JSON strings")
    return OracleBCDataset(
        tuple(OracleBCRecord.model_validate_json(row) for row in raw_rows)
    ).records


@dataclass
class RouterBatch:
    question_token_ids: Tensor
    question_token_mask: Tensor
    evidence_token_ids: Tensor
    evidence_token_mask: Tensor
    evidence_item_mask: Tensor
    evidence_fidelity: Tensor
    evidence_numeric: Tensor
    history_token_ids: Tensor
    history_token_mask: Tensor
    history_item_mask: Tensor
    history_action_type: Tensor
    action_token_ids: Tensor
    action_token_mask: Tensor
    legal_action_mask: Tensor
    action_type: Tensor
    action_fidelity: Tensor
    action_visual_budget: Tensor
    action_frontier: Tensor
    action_evidence_affinity: Tensor
    state_numeric: Tensor
    target_action_index: Tensor
    sufficiency_target: Tensor
    cost_to_go_target: Tensor

    def to(self, device: torch.device | str) -> "RouterBatch":
        return RouterBatch(
            **{
                field.name: getattr(self, field.name).to(device)
                for field in fields(self)
            }
        )


def _action_text(action: ActionInstance) -> str:
    return "|".join(
        (action.action_type.value, action.event_id or "", action.visual_budget or "")
    )


def _encode_bytes(text: str, *, maximum: int, label: str) -> list[int]:
    encoded = text.encode("utf-8")
    if len(encoded) > maximum:
        raise ValueError(f"{label} exceeds configured maximum of {maximum} UTF-8 bytes")
    return [value + 1 for value in encoded]


def _padded_tokens(
    items: Sequence[Sequence[str]], *, maximum: int, label: str
) -> tuple[Tensor, Tensor, Tensor]:
    batch_size = len(items)
    max_items = max(1, max((len(row) for row in items), default=0))
    encoded: list[list[list[int]]] = []
    max_tokens = 1
    for row in items:
        encoded_row = [
            _encode_bytes(text, maximum=maximum, label=label) for text in row
        ]
        encoded.append(encoded_row)
        for item in encoded_row:
            max_tokens = max(max_tokens, len(item))
    token_ids = torch.zeros((batch_size, max_items, max_tokens), dtype=torch.long)
    token_mask = torch.zeros_like(token_ids, dtype=torch.bool)
    item_mask = torch.zeros((batch_size, max_items), dtype=torch.bool)
    for row_index, row in enumerate(encoded):
        for item_index, item in enumerate(row):
            if item:
                token_ids[row_index, item_index, : len(item)] = torch.tensor(item)
                token_mask[row_index, item_index, : len(item)] = True
            item_mask[row_index, item_index] = True
    return token_ids, token_mask, item_mask


class RouterCollator:
    def __init__(self, *, max_question_bytes: int, max_item_bytes: int) -> None:
        if max_question_bytes < 1 or max_item_bytes < 1:
            raise ValueError("text byte limits must be positive")
        self.max_question_bytes = max_question_bytes
        self.max_item_bytes = max_item_bytes

    def __call__(self, records: Sequence[OracleBCRecord]) -> RouterBatch:
        validated = tuple(
            OracleBCRecord.model_validate(record.model_dump(mode="python"))
            for record in records
        )
        if not validated:
            raise ValueError("cannot collate an empty batch")
        question_rows = [
            (record.state.question + "\n" + "\n".join(record.state.options),)
            for record in validated
        ]
        question_ids, question_mask, _ = _padded_tokens(
            question_rows, maximum=self.max_question_bytes, label="question"
        )
        evidence_rows = [
            [item.content for item in record.state.evidence] for record in validated
        ]
        evidence_ids, evidence_token_mask, evidence_item_mask = _padded_tokens(
            evidence_rows, maximum=self.max_item_bytes, label="evidence item"
        )
        history_rows = [
            [_action_text(action) for action in record.state.action_history]
            for record in validated
        ]
        history_ids, history_token_mask, history_item_mask = _padded_tokens(
            history_rows, maximum=self.max_item_bytes, label="history action"
        )
        action_rows = [
            [_action_text(action) for action in record.action_instances]
            for record in validated
        ]
        action_ids, action_token_mask, action_item_mask = _padded_tokens(
            action_rows, maximum=self.max_item_bytes, label="action instance"
        )
        batch_size, max_actions = action_item_mask.shape
        max_evidence = evidence_item_mask.shape[1]

        evidence_fidelity = torch.full(
            (batch_size, max_evidence), _NONE_FIDELITY, dtype=torch.long
        )
        evidence_numeric = torch.zeros(
            (batch_size, max_evidence, 2), dtype=torch.float32
        )
        history_action_type = torch.zeros_like(history_item_mask, dtype=torch.long)
        legal_mask = torch.zeros((batch_size, max_actions), dtype=torch.bool)
        action_type = torch.zeros((batch_size, max_actions), dtype=torch.long)
        action_fidelity = torch.full(
            (batch_size, max_actions), _NONE_FIDELITY, dtype=torch.long
        )
        action_visual_budget = torch.zeros((batch_size, max_actions), dtype=torch.long)
        action_frontier = torch.zeros((batch_size, max_actions, 2), dtype=torch.float32)
        affinity = torch.zeros(
            (batch_size, max_actions, max_evidence), dtype=torch.bool
        )
        state_numeric = torch.zeros((batch_size, 2), dtype=torch.float32)

        for row_index, record in enumerate(validated):
            state = record.state
            state_numeric[row_index] = torch.tensor(
                (state.remaining_budget, state.cost_preference)
            )
            for evidence_index, item in enumerate(state.evidence):
                evidence_fidelity[row_index, evidence_index] = _FIDELITY_INDEX[
                    item.fidelity_level
                ]
                evidence_numeric[row_index, evidence_index] = torch.tensor(
                    (item.score, float(item.acquisition_step))
                )
            for history_index, action in enumerate(state.action_history):
                history_action_type[row_index, history_index] = _ACTION_INDEX[
                    action.action_type
                ]
            legal_mask[row_index, : len(record.legal_action_mask)] = torch.tensor(
                record.legal_action_mask
            )
            for action_index, action in enumerate(record.action_instances):
                action_type[row_index, action_index] = _ACTION_INDEX[action.action_type]
                action_visual_budget[row_index, action_index] = _VISUAL_BUDGET_INDEX[
                    action.visual_budget
                ]
                if (
                    action.event_id is not None
                    and action.event_id in state.candidate_fidelity_levels
                ):
                    action_fidelity[row_index, action_index] = _FIDELITY_INDEX[
                        state.candidate_fidelity_levels[action.event_id]
                    ]
                    action_frontier[row_index, action_index] = torch.tensor(
                        state.context_frontiers[action.event_id], dtype=torch.float32
                    )
                    for evidence_index, item in enumerate(state.evidence):
                        affinity[row_index, action_index, evidence_index] = (
                            item.event_id == action.event_id
                        )
        return RouterBatch(
            question_token_ids=question_ids[:, 0],
            question_token_mask=question_mask[:, 0],
            evidence_token_ids=evidence_ids,
            evidence_token_mask=evidence_token_mask,
            evidence_item_mask=evidence_item_mask,
            evidence_fidelity=evidence_fidelity,
            evidence_numeric=evidence_numeric,
            history_token_ids=history_ids,
            history_token_mask=history_token_mask,
            history_item_mask=history_item_mask,
            history_action_type=history_action_type,
            action_token_ids=action_ids,
            action_token_mask=action_token_mask,
            legal_action_mask=legal_mask,
            action_type=action_type,
            action_fidelity=action_fidelity,
            action_visual_budget=action_visual_budget,
            action_frontier=action_frontier,
            action_evidence_affinity=affinity,
            state_numeric=state_numeric,
            target_action_index=torch.tensor(
                [record.target_action_index for record in validated], dtype=torch.long
            ),
            sufficiency_target=torch.tensor(
                [record.sufficiency_target for record in validated], dtype=torch.float32
            ),
            cost_to_go_target=torch.tensor(
                [record.cost_to_go for record in validated], dtype=torch.float32
            ),
        )
