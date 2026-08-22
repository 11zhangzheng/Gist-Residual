"""Candidate-instance-aware three-head memory router."""

from __future__ import annotations

from typing import NamedTuple

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn

from .dataset import RouterBatch


class RouterModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    byte_vocab_size: int = Field(default=257, ge=257)
    hidden_dim: int = Field(default=256, ge=8)
    token_embedding_dim: int = Field(default=192, ge=4)
    action_type_embedding_dim: int = Field(default=32, ge=2)
    fidelity_embedding_dim: int = Field(default=16, ge=2)
    num_encoder_layers: int = Field(default=0, ge=0)
    num_attention_heads: int = Field(default=4, ge=1)
    feedforward_multiplier: int = Field(default=4, ge=1)
    dropout: float = Field(default=0.0, ge=0, lt=1)
    max_question_bytes: int = Field(default=512, ge=1)
    max_item_bytes: int = Field(default=256, ge=1)
    enforce_parameter_range: bool = False
    min_trainable_parameters: int = Field(default=100_000_000, ge=1)
    max_trainable_parameters: int = Field(default=300_000_000, ge=1)

    @model_validator(mode="after")
    def validate_dimensions(self) -> "RouterModelConfig":
        if self.hidden_dim % self.num_attention_heads:
            raise ValueError("hidden_dim must be divisible by num_attention_heads")
        if self.max_trainable_parameters < self.min_trainable_parameters:
            raise ValueError("maximum parameter count must cover minimum")
        return self


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

    def __init__(self, config: RouterModelConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_dim
        self.token_embedding = nn.Embedding(
            config.byte_vocab_size, config.token_embedding_dim, padding_idx=0
        )
        self.token_projection = nn.Linear(config.token_embedding_dim, h)
        if config.num_encoder_layers:
            layer = nn.TransformerEncoderLayer(
                d_model=h,
                nhead=config.num_attention_heads,
                dim_feedforward=h * config.feedforward_multiplier,
                dropout=config.dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.sequence_encoder: nn.Module = nn.TransformerEncoder(
                layer, config.num_encoder_layers, enable_nested_tensor=False
            )
        else:
            self.sequence_encoder = nn.Identity()
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
            nn.Linear(3 * h + 2, h), nn.GELU(), nn.LayerNorm(h)
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
            count = self.trainable_parameter_count
            if (
                not config.min_trainable_parameters
                <= count
                <= config.max_trainable_parameters
            ):
                raise ValueError(
                    f"trainable parameter count {count} is outside configured range "
                    f"[{config.min_trainable_parameters}, {config.max_trainable_parameters}]"
                )

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
        embedded = self.token_projection(self.token_embedding(flat_ids))
        if not isinstance(self.sequence_encoder, nn.Identity):
            safe_mask = flat_mask.clone()
            empty = ~safe_mask.any(dim=1)
            safe_mask[empty, 0] = True
            embedded = self.sequence_encoder(embedded, src_key_padding_mask=~safe_mask)
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
        state = self.state_projection(
            torch.cat(
                (
                    question,
                    evidence_summary,
                    history_summary,
                    batch.state_numeric.to(dtype=dtype),
                ),
                dim=-1,
            )
        )

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
