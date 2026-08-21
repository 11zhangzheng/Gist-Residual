"""Frozen, strategy-neutral final answering over acquired evidence."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict

from fidmem.types import EvidenceItem


class AnswererResponseError(RuntimeError):
    """Raised when the frozen adapter does not produce a parseable answer."""


class AnswerResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str
    prompt: str


AnswerAdapter = Callable[[str], str]


class FrozenAnswerer:
    """Use one fixed serialization for every routing strategy."""

    def __init__(self, adapter: AnswerAdapter) -> None:
        self._adapter = adapter

    @staticmethod
    def _ordered(evidence: Sequence[EvidenceItem]) -> tuple[EvidenceItem, ...]:
        return tuple(sorted(
            evidence,
            key=lambda item: (item.start_sec, item.acquisition_step, item.event_id, item.fidelity_level.value, item.content, item.attachments, item.score),
        ))

    @classmethod
    def render_prompt(cls, question: str, options: Sequence[str], evidence: Sequence[EvidenceItem]) -> str:
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
            raise AnswererResponseError("frozen answerer returned an unparseable answer")
        return answer

    def answer(self, question: str, options: Sequence[str], evidence: Sequence[EvidenceItem]) -> AnswerResult:
        prompt = self.render_prompt(question, options, evidence)
        return AnswerResult(answer=self._parse(self._adapter(prompt)), prompt=prompt)
