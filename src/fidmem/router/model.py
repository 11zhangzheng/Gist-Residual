"""Candidate-instance-aware three-head memory router."""

from __future__ import annotations

from typing import Literal, NamedTuple

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn

from .dataset import RouterBatch, TokenizerIdentity


class EncoderIdentity(BaseModel):
    """Pinned tokenizer/backbone identity; production never follows a branch."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    kind: Literal["pretrained", "test"]
    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    tokenizer: TokenizerIdentity
    trust_remote_code: Literal[False] = False

    @classmethod
    def test_identity(cls, name: str) -> "EncoderIdentity":
        return cls(
            kind="test",
            model_id=name,
            revision="offline-test-v1",
            tokenizer=TokenizerIdentity.byte_identity(name),
            trust_remote_code=False,
        )

    @model_validator(mode="after")
    def production_revision_must_be_immutable(self) -> "EncoderIdentity":
        if self.kind == "pretrained":
            for value in (self.revision, self.tokenizer.revision):
                if len(value) != 40 or any(
                    char not in "0123456789abcdef" for char in value
                ):
                    raise ValueError(
                        "pretrained revision must be a 40-character commit hash"
                    )
        return self


class RouterModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    encoder: EncoderIdentity = Field(
        default_factory=lambda: EncoderIdentity.test_identity("byte-test-v1")
    )
    encoder_output_dim: int = Field(default=192, ge=4)
    hidden_dim: int = Field(default=256, ge=8)
    action_type_embedding_dim: int = Field(default=32, ge=2)
    fidelity_embedding_dim: int = Field(default=16, ge=2)
    max_question_tokens: int = Field(default=512, ge=1)
    max_item_tokens: int = Field(default=256, ge=1)
    production: bool = False
    enforce_parameter_range: bool = False
    min_total_parameters: int = Field(default=100_000_000, ge=1)
    max_total_parameters: int = Field(default=150_000_000, ge=1, le=300_000_000)

    @model_validator(mode="after")
    def validate_dimensions(self) -> "RouterModelConfig":
        if self.production and self.encoder.kind != "pretrained":
            raise ValueError("production Router requires a pretrained encoder identity")
        if self.max_total_parameters < self.min_total_parameters:
            raise ValueError("maximum parameter count must cover minimum")
        return self


class TestTextEncoder(nn.Module):
    """Small offline encoder for tests; production config rejects its identity."""

    __test__ = False

    def __init__(
        self,
        identity: EncoderIdentity,
        vocab_size: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        if identity.kind != "test":
            raise ValueError("TestTextEncoder requires a test identity")
        self.identity = identity
        self.embedding = nn.Embedding(vocab_size, output_dim, padding_idx=0)

    def forward(self, token_ids: Tensor, token_mask: Tensor) -> Tensor:
        del token_mask
        return self.embedding(token_ids)


class PretrainedTextEncoder(nn.Module):
    def __init__(self, identity: EncoderIdentity, backbone: nn.Module) -> None:
        super().__init__()
        if identity.kind != "pretrained":
            raise ValueError("pretrained wrapper requires a pretrained identity")
        self.identity = identity
        self.backbone = backbone

    def forward(self, token_ids: Tensor, token_mask: Tensor) -> Tensor:
        result = self.backbone(input_ids=token_ids, attention_mask=token_mask)
        hidden = getattr(result, "last_hidden_state", None)
        if not isinstance(hidden, Tensor):
            raise ValueError("pretrained encoder did not return last_hidden_state")
        return hidden


class ProductionEncoderFactory:
    """Lazy Hugging Face loader; importing this module never downloads models."""

    @staticmethod
    def load(identity: EncoderIdentity) -> tuple[PretrainedTextEncoder, object]:
        if identity.kind != "pretrained" or identity.trust_remote_code:
            raise ValueError("production encoder identity is not safe")
        from pathlib import Path

        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer

        tokenizer_snapshot = Path(
            snapshot_download(
                identity.tokenizer.model_id,
                revision=identity.tokenizer.revision,
                local_files_only=True,
            )
        ).resolve()
        if tokenizer_snapshot.name != identity.tokenizer.revision:
            raise ValueError("local tokenizer snapshot does not match pinned revision")
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_snapshot),
            local_files_only=True,
            trust_remote_code=False,
        )
        backbone = AutoModel.from_pretrained(
            identity.model_id,
            local_files_only=True,
            revision=identity.revision,
            trust_remote_code=False,
        )
        return PretrainedTextEncoder(identity, backbone), tokenizer


class RouterOutput(NamedTuple):
    action_logits: Tensor
    sufficiency_logit: Tensor
    cost_to_go: Tensor


def _masked_mean(values: Tensor, mask: Tensor, dimension: int) -> Tensor:
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=dimension) / weights.sum(dim=dimension).clamp_min(
        1
    )


class MemoryRouter(nn.Module):
    """Scores each available ActionInstance rather than five action classes."""

    def __init__(
        self,
        config: RouterModelConfig,
        *,
        text_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_dim
        if text_encoder is None:
            if config.encoder.kind != "test":
                raise ValueError(
                    "pretrained encoder must be built by ProductionEncoderFactory"
                )
            text_encoder = TestTextEncoder(
                config.encoder, vocab_size=257, output_dim=config.encoder_output_dim
            )
        identity = getattr(text_encoder, "identity", None)
        if identity != config.encoder:
            raise ValueError(
                "text encoder identity does not match Router config identity"
            )
        self.text_encoder = text_encoder
        self.token_projection = nn.Linear(config.encoder_output_dim, h)
        self.action_type_embedding = nn.Embedding(5, config.action_type_embedding_dim)
        self.fidelity_embedding = nn.Embedding(4, config.fidelity_embedding_dim)
        self.visual_budget_embedding = nn.Embedding(3, 4)
        self.question_projection = nn.Sequential(
            nn.LayerNorm(h), nn.Linear(h, h), nn.GELU()
        )
        self.evidence_projection = nn.Sequential(
            nn.Linear(h + config.fidelity_embedding_dim + 2, h),
            nn.GELU(),
            nn.LayerNorm(h),
        )
        self.history_projection = nn.Sequential(
            nn.Linear(h + config.action_type_embedding_dim, h),
            nn.GELU(),
            nn.LayerNorm(h),
        )
        self.action_projection = nn.Sequential(
            nn.Linear(
                h
                + config.action_type_embedding_dim
                + config.fidelity_embedding_dim
                + 4
                + 2,
                h,
            ),
            nn.GELU(),
            nn.LayerNorm(h),
        )
        self.state_projection = nn.Sequential(
            nn.Linear(4 * h + 2, h), nn.GELU(), nn.LayerNorm(h)
        )
        self.action_scorer = nn.Sequential(
            nn.Linear(3 * h, h), nn.GELU(), nn.Linear(h, 1)
        )
        self.sufficiency_head = nn.Sequential(
            nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1)
        )
        self.cost_to_go_head = nn.Sequential(
            nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1)
        )
        if config.enforce_parameter_range:
            count = self.total_parameter_count
            if not config.min_total_parameters <= count <= config.max_total_parameters:
                raise ValueError(
                    f"total parameter count {count} is outside configured range "
                    f"[{config.min_total_parameters}, {config.max_total_parameters}]"
                )

    @property
    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def _encode_tokens(self, token_ids: Tensor, token_mask: Tensor) -> Tensor:
        original_shape = token_ids.shape[:-1]
        length = token_ids.shape[-1]
        flat_ids = token_ids.reshape(-1, length)
        flat_mask = token_mask.reshape(-1, length)
        encoded = self.text_encoder(flat_ids, flat_mask)
        embedded = self.token_projection(encoded)
        pooled = _masked_mean(embedded, flat_mask, 1)
        return pooled.reshape(*original_shape, self.config.hidden_dim)

    def forward(self, batch: RouterBatch) -> RouterOutput:
        legal = batch.legal_action_mask
        if legal.ndim != 2 or not legal.any(dim=1).all():
            raise ValueError("each batch row must contain at least one legal action")
        dtype = self.token_projection.weight.dtype
        question = self.question_projection(
            self._encode_tokens(batch.question_token_ids, batch.question_token_mask)
        )
        evidence_text = self._encode_tokens(
            batch.evidence_token_ids, batch.evidence_token_mask
        )
        evidence = self.evidence_projection(
            torch.cat(
                (
                    evidence_text,
                    self.fidelity_embedding(batch.evidence_fidelity),
                    batch.evidence_numeric.to(dtype=dtype),
                ),
                dim=-1,
            )
        )
        evidence_summary = _masked_mean(evidence, batch.evidence_item_mask, 1)
        history_text = self._encode_tokens(
            batch.history_token_ids, batch.history_token_mask
        )
        history = self.history_projection(
            torch.cat(
                (history_text, self.action_type_embedding(batch.history_action_type)),
                dim=-1,
            )
        )
        history_summary = _masked_mean(history, batch.history_item_mask, 1)
        action_text = self._encode_tokens(
            batch.action_token_ids, batch.action_token_mask
        )
        actions = self.action_projection(
            torch.cat(
                (
                    action_text,
                    self.action_type_embedding(batch.action_type),
                    self.fidelity_embedding(batch.action_fidelity),
                    self.visual_budget_embedding(batch.action_visual_budget),
                    batch.action_frontier.to(dtype=dtype),
                ),
                dim=-1,
            )
        )
        affinity = batch.action_evidence_affinity.to(dtype=dtype)
        candidate_evidence = torch.bmm(affinity, evidence)
        candidate_evidence = candidate_evidence / affinity.sum(
            dim=-1, keepdim=True
        ).clamp_min(1)
        candidate_representation = actions + candidate_evidence
        candidate_summary = _masked_mean(
            candidate_representation, batch.legal_action_mask, 1
        )
        state = self.state_projection(
            torch.cat(
                (
                    question,
                    evidence_summary,
                    history_summary,
                    candidate_summary,
                    batch.state_numeric.to(dtype=dtype),
                ),
                dim=-1,
            )
        )
        expanded_state = state.unsqueeze(1).expand(-1, actions.shape[1], -1)
        raw_logits = self.action_scorer(
            torch.cat((expanded_state, actions, candidate_evidence), dim=-1)
        ).squeeze(-1)
        action_logits = raw_logits.masked_fill(
            ~legal, torch.finfo(raw_logits.dtype).min
        )
        sufficiency_logit = self.sufficiency_head(state).squeeze(-1)
        cost_to_go = torch.nn.functional.softplus(
            self.cost_to_go_head(state).squeeze(-1)
        )
        if not (
            torch.isfinite(raw_logits[legal]).all()
            and torch.isfinite(sufficiency_logit).all()
            and torch.isfinite(cost_to_go).all()
        ):
            raise FloatingPointError("router produced non-finite output")
        return RouterOutput(action_logits, sufficiency_logit, cost_to_go)
