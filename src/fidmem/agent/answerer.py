"""Frozen, strategy-neutral final answering over acquired evidence."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, is_dataclass
from time import perf_counter_ns
from types import CodeType, ModuleType

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fidmem.costs.tracker import CostRecord
from fidmem.types import EvidenceItem, FidelityLevel


_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class AnswererResponseError(RuntimeError):
    """Raised when the frozen adapter does not produce a parseable answer."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _identity_value(value: object) -> object:
    """Return a deterministic, finite payload for actual callable state."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("callable identity state must be finite")
        return value
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, type):
        return {"type": f"{value.__module__}:{value.__qualname__}"}
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return _identity_value(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _identity_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return tuple(_identity_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        items = tuple(_identity_value(item) for item in value)
        return tuple(sorted(items, key=_canonical_json))
    if isinstance(value, (CodeType, ModuleType)):
        return {"type": type(value).__name__, "name": getattr(value, "__name__", "")}
    if callable(value):
        return {"callable_sha256": callable_identity_sha256(value)}
    state = getattr(value, "__dict__", None)
    if isinstance(state, dict):
        return {
            "type": f"{type(value).__module__}:{type(value).__qualname__}",
            "state": _identity_value(state),
        }
    raise TypeError(
        "unsupported callable identity state: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def callable_identity_sha256(adapter: Callable[..., object]) -> str:
    """Hash executable code plus referenced closure/global/instance state."""

    target = (
        adapter
        if inspect.isfunction(adapter) or inspect.ismethod(adapter)
        else adapter.__call__
    )
    code = getattr(target, "__code__", None)
    if not isinstance(code, CodeType):
        raise TypeError("answer adapter must expose Python executable code")
    closure = inspect.getclosurevars(target)
    globals_payload = {
        name: _identity_value(value)
        for name, value in sorted(closure.globals.items())
        if not isinstance(value, ModuleType)
    }
    payload = {
        "module": getattr(target, "__module__", type(adapter).__module__),
        "qualname": getattr(target, "__qualname__", type(adapter).__qualname__),
        "code_sha256": hashlib.sha256(code.co_code).hexdigest(),
        "constants": _identity_value(code.co_consts),
        "names": code.co_names,
        "defaults": _identity_value(getattr(target, "__defaults__", None)),
        "kwdefaults": _identity_value(getattr(target, "__kwdefaults__", None)),
        "nonlocals": _identity_value(closure.nonlocals),
        "globals": globals_payload,
        "instance_state": _identity_value(getattr(adapter, "__dict__", {})),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


class AnswererAdapterResult(BaseModel):
    """Adapter response with authoritative measured usage for formal evaluation."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        arbitrary_types_allowed=True,
        revalidate_instances="always",
    )

    response: str
    cost_record: CostRecord
    total_cost: float = Field(ge=0)

    @model_validator(mode="after")
    def cost_record_is_valid(self) -> "AnswererAdapterResult":
        self.cost_record.validate_values()
        return self


class FrozenAnswererIdentity(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        revalidate_instances="always",
        protected_namespaces=(),
    )

    model_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_revision: str = Field(min_length=1)
    decode_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    adapter_sha256: str = Field(pattern=_SHA256_PATTERN)
    template_sha256: str = Field(pattern=_SHA256_PATTERN)
    identity_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def self_hash_matches(self) -> "FrozenAnswererIdentity":
        payload = self.model_dump(mode="json", exclude={"identity_sha256"})
        expected = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        if self.identity_sha256 != expected:
            raise ValueError("Answerer identity self hash mismatch")
        return self


class AnswerResult(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        arbitrary_types_allowed=True,
        revalidate_instances="always",
    )

    answer: str
    prompt: str
    cost_record: CostRecord
    total_cost: float = Field(ge=0)
    usage_authoritative: bool

    @model_validator(mode="after")
    def cost_record_is_valid(self) -> "AnswerResult":
        self.cost_record.validate_values()
        return self


AnswerAdapter = Callable[[str], str | AnswererAdapterResult]


class FrozenAnswerer:
    """Use one fixed serialization for every routing strategy."""

    def __init__(
        self,
        adapter: AnswerAdapter,
        *,
        model_artifact_sha256: str | None = None,
        model_revision: str | None = None,
        decode_config: dict[str, object] | None = None,
    ) -> None:
        self._adapter = adapter
        identity_values = (model_artifact_sha256, model_revision, decode_config)
        if any(value is not None for value in identity_values) and not all(
            value is not None for value in identity_values
        ):
            raise ValueError(
                "model artifact, revision, and decode config must be supplied together"
            )
        self._identity: FrozenAnswererIdentity | None = None
        if model_artifact_sha256 is not None:
            if not isinstance(model_revision, str) or not model_revision:
                raise ValueError("model revision must not be blank")
            if not isinstance(decode_config, dict):
                raise TypeError("decode config must be a dictionary")
            template_sha256 = hashlib.sha256(
                self.render_prompt(
                    "sentinel-question",
                    ("A", "B"),
                    (
                        EvidenceItem(
                            event_id="sentinel",
                            fidelity_level=FidelityLevel.GIST,
                            content="sentinel-content",
                            score=0.0,
                        ),
                    ),
                ).encode("utf-8")
            ).hexdigest()
            payload = {
                "model_artifact_sha256": model_artifact_sha256,
                "model_revision": model_revision,
                "decode_config_sha256": hashlib.sha256(
                    _canonical_json(decode_config).encode("utf-8")
                ).hexdigest(),
                "adapter_sha256": callable_identity_sha256(adapter),
                "template_sha256": template_sha256,
            }
            self._identity = FrozenAnswererIdentity(
                **payload,
                identity_sha256=hashlib.sha256(
                    _canonical_json(payload).encode("utf-8")
                ).hexdigest(),
            )

    @property
    def identity(self) -> FrozenAnswererIdentity:
        if self._identity is None:
            raise ValueError("FrozenAnswerer lacks a formal evaluation identity")
        return FrozenAnswererIdentity.model_validate(
            self._identity.model_dump(mode="python")
        )

    @staticmethod
    def _ordered(evidence: Sequence[EvidenceItem]) -> tuple[EvidenceItem, ...]:
        return tuple(
            sorted(
                evidence,
                key=lambda item: (
                    item.start_sec,
                    item.acquisition_step,
                    item.event_id,
                    item.fidelity_level.value,
                    item.content,
                    item.attachments,
                    item.score,
                ),
            )
        )

    @classmethod
    def render_prompt(
        cls,
        question: str,
        options: Sequence[str],
        evidence: Sequence[EvidenceItem],
    ) -> str:
        if not question.strip():
            raise ValueError("question must not be blank")
        normalized_options = tuple(str(option) for option in options)
        payload = [
            {
                "event_id": item.event_id,
                "fidelity": item.fidelity_level.value,
                "content": item.content,
                "attachments": list(item.attachments),
            }
            for item in cls._ordered(evidence)
        ]
        return (
            f"Question:\n{json.dumps(question, ensure_ascii=False, separators=(',', ':'))}\n"
            f"Options:\n{json.dumps(normalized_options, ensure_ascii=False, separators=(',', ':'))}\n"
            f"Evidence:\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
            "Answer:\n"
        )

    @staticmethod
    def _parse(raw_response: str) -> str:
        if not isinstance(raw_response, str) or not raw_response.strip():
            raise AnswererResponseError("frozen answerer returned an empty response")
        answer = raw_response.strip()
        if answer.casefold().startswith("answer:"):
            answer = answer.split(":", 1)[1].strip()
        if not answer:
            raise AnswererResponseError(
                "frozen answerer returned an unparseable answer"
            )
        return answer

    def answer(
        self,
        question: str,
        options: Sequence[str],
        evidence: Sequence[EvidenceItem],
    ) -> AnswerResult:
        prompt = self.render_prompt(question, options, evidence)
        wall_start_ns = perf_counter_ns()
        raw = self._adapter(prompt)
        measured_wall_seconds = (perf_counter_ns() - wall_start_ns) / 1_000_000_000
        if isinstance(raw, AnswererAdapterResult):
            response = AnswererAdapterResult(
                response=raw.response,
                cost_record=raw.cost_record,
                total_cost=raw.total_cost,
            )
            return AnswerResult(
                answer=self._parse(response.response),
                prompt=prompt,
                cost_record=response.cost_record,
                total_cost=response.total_cost,
                usage_authoritative=True,
            )
        if not isinstance(raw, str):
            raise AnswererResponseError(
                "frozen answerer adapter must return text or AnswererAdapterResult"
            )
        estimated_text_tokens = len(prompt.split()) + len(raw.split())
        return AnswerResult(
            answer=self._parse(raw),
            prompt=prompt,
            cost_record=CostRecord(
                operation="frozen_answerer",
                gpu_seconds=0.0,
                wall_seconds=measured_wall_seconds,
                input_frames=0,
                visual_tokens=0,
                text_tokens=estimated_text_tokens,
                peak_memory_bytes=0,
                cache_status="miss",
                device_name="cpu",
            ),
            total_cost=0.0,
            usage_authoritative=False,
        )


__all__ = [
    "AnswerAdapter",
    "AnswererAdapterResult",
    "AnswererResponseError",
    "AnswerResult",
    "FrozenAnswerer",
    "FrozenAnswererIdentity",
    "callable_identity_sha256",
]
