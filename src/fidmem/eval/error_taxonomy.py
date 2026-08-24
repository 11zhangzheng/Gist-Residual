"""Auditable, mutually exclusive primary error attribution."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, model_validator


class ErrorCause(str, Enum):
    RECALL = "recall_error"
    ANSWERER = "answerer_error"
    PREMATURE_STOP = "premature_stop"
    INSUFFICIENT_FIDELITY = "insufficient_fidelity"
    OVER_RETRIEVAL = "over_retrieval"


ERROR_PRIORITY = (
    ErrorCause.RECALL,
    ErrorCause.ANSWERER,
    ErrorCause.PREMATURE_STOP,
    ErrorCause.INSUFFICIENT_FIDELITY,
    ErrorCause.OVER_RETRIEVAL,
)


class ErrorSignals(BaseModel):
    """Signals produced by benchmark/oracle audits, never caller labels."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    gist_top_k_contains_answer: bool
    oracle_evidence_sufficient: bool = False
    answerer_correct_with_oracle_evidence: bool | None = None
    stopped_with_insufficient_evidence: bool = False
    useful_fidelity_upgrade_available: bool = False
    unnecessary_expansion: bool = False

    @model_validator(mode="after")
    def oracle_answer_signal_has_authority(self) -> "ErrorSignals":
        if (
            self.answerer_correct_with_oracle_evidence is not None
            and not self.oracle_evidence_sufficient
        ):
            raise ValueError(
                "Answerer correctness requires Oracle-sufficient evidence"
            )
        return self


class ErrorClassification(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    primary: ErrorCause | None
    secondary: tuple[ErrorCause, ...]


def _active_causes(signals: ErrorSignals) -> tuple[ErrorCause, ...]:
    active: list[ErrorCause] = []
    if not signals.gist_top_k_contains_answer:
        active.append(ErrorCause.RECALL)
    if (
        signals.oracle_evidence_sufficient
        and signals.answerer_correct_with_oracle_evidence is False
    ):
        active.append(ErrorCause.ANSWERER)
    if signals.stopped_with_insufficient_evidence:
        active.append(ErrorCause.PREMATURE_STOP)
    if signals.useful_fidelity_upgrade_available:
        active.append(ErrorCause.INSUFFICIENT_FIDELITY)
    if signals.unnecessary_expansion:
        active.append(ErrorCause.OVER_RETRIEVAL)
    return tuple(active)


def classify_error(
    signals: ErrorSignals, *, invalid: bool = False
) -> ErrorClassification:
    """Choose the first active cause in the frozen priority order.

    Invalid samples are reported separately because their outcome is not an
    ordinary controller/answering error. Secondary flags retain every active
    auditable cause for trajectory analysis.
    """

    if invalid:
        return ErrorClassification(primary=None, secondary=())
    active = _active_causes(signals)
    return ErrorClassification(
        primary=next((cause for cause in ERROR_PRIORITY if cause in active), None),
        secondary=tuple(cause for cause in ERROR_PRIORITY if cause in active),
    )


__all__ = [
    "ERROR_PRIORITY",
    "ErrorCause",
    "ErrorClassification",
    "ErrorSignals",
    "classify_error",
]
