"""Graph-conditioned bidirectional Transformer denoiser."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, NamedTuple

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class DenoiserConfig:
    num_pitch_tokens: int
    num_duration_tokens: int
    num_family_tokens: int
    num_families: int
    num_chapters: int
    num_meters: int
    num_position_buckets: int
    max_events_per_measure: int
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    ffn_dim: int = 256
    dropout: float = 0.1
    embedding_init_std: float = 0.02
    hwanip_message_scale: float = 0.0

    def __post_init__(self) -> None:
        integer_fields = (
            self.num_pitch_tokens,
            self.num_duration_tokens,
            self.num_family_tokens,
            self.num_families,
            self.num_chapters,
            self.num_meters,
            self.num_position_buckets,
            self.max_events_per_measure,
            self.d_model,
            self.n_layers,
            self.n_heads,
            self.ffn_dim,
        )
        if any(value <= 0 for value in integer_fields):
            raise ValueError("All DenoiserConfig dimensions must be positive")
        if self.num_family_tokens != self.num_families + 1:
            raise ValueError("num_family_tokens must equal num_families + one MASK token")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= self.dropout < 1.0 or self.embedding_init_std <= 0:
            raise ValueError("Invalid dropout or embedding_init_std")
        if self.hwanip_message_scale < 0:
            raise ValueError("hwanip_message_scale must be non-negative")

    @property
    def family_mask_id(self) -> int:
        return self.num_families

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DenoiserConfig":
        required = {
            "num_pitch_tokens",
            "num_duration_tokens",
            "num_family_tokens",
            "num_families",
            "num_chapters",
            "num_meters",
            "num_position_buckets",
            "max_events_per_measure",
        }
        missing = required - set(value)
        if missing:
            raise KeyError(f"DenoiserConfig is missing required fields: {sorted(missing)}")
        known = set(cls.__dataclass_fields__)
        return cls(**{name: item for name, item in value.items() if name in known})


class DenoiserOutput(NamedTuple):
    pitch_logits: Tensor
    duration_logits: Tensor
    family_logits: Tensor


class GugakDenoiser(nn.Module):
    """Denoise pitch, duration, and family tokens for one complete score.

    Attribute names intentionally match the original ``YGDenoiser`` so the
    frozen v0/v1 state dictionaries can be loaded with ``strict=True`` when the
    dimensions and position-bucket design agree.
    """

    def __init__(self, config: DenoiserConfig, transition_matrix: Tensor) -> None:
        super().__init__()
        self.config = config
        self.d = config.d_model
        expected = (config.num_families, config.num_families)
        matrix = torch.as_tensor(transition_matrix, dtype=torch.float32)
        if tuple(matrix.shape) != expected:
            raise ValueError(f"transition_matrix must have shape {expected}, got {tuple(matrix.shape)}")
        if not torch.isfinite(matrix).all() or bool((matrix < 0).any()):
            raise ValueError("transition_matrix must be finite and non-negative")

        self.emb_p = nn.Embedding(config.num_pitch_tokens, self.d)
        self.emb_d = nn.Embedding(config.num_duration_tokens, self.d)
        self.emb_z = nn.Embedding(config.num_family_tokens, self.d)
        self.emb_ch = nn.Embedding(config.num_chapters, self.d)
        self.emb_mt = nn.Embedding(config.num_meters, self.d)
        self.emb_pos = nn.Embedding(config.num_position_buckets, self.d)
        self.emb_slot = nn.Embedding(config.max_events_per_measure, self.d)
        self.emb_anchor = nn.Embedding(2, self.d)

        self.register_buffer("PG_out", matrix.clone())
        self.register_buffer("PG_in", matrix.T.contiguous())
        self.W_out = nn.Linear(config.num_families, self.d, bias=False)
        self.W_in = nn.Linear(config.num_families, self.d, bias=False)
        self.role_mask = nn.Parameter(torch.zeros(self.d))
        self.s_proj = nn.Sequential(nn.Linear(2, self.d), nn.SiLU(), nn.Linear(self.d, self.d))

        layer = nn.TransformerEncoderLayer(
            d_model=self.d,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        # norm_first=True is intentional and incompatible with the nested-tensor
        # fast path in current PyTorch releases; disable it explicitly to avoid a
        # misleading warning in every tutorial Notebook.
        self.enc = nn.TransformerEncoder(
            layer, num_layers=config.n_layers, enable_nested_tensor=False
        )
        self.in_norm = nn.LayerNorm(self.d)
        self.norm = nn.LayerNorm(self.d)
        self.head_p = nn.Linear(self.d, config.num_pitch_tokens)
        self.head_d = nn.Linear(self.d, config.num_duration_tokens)
        self.head_z = nn.Linear(self.d, config.num_families)
        self._init_scales(config.embedding_init_std)

    def _init_scales(self, std: float) -> None:
        embeddings = (
            self.emb_p,
            self.emb_d,
            self.emb_z,
            self.emb_ch,
            self.emb_mt,
            self.emb_pos,
            self.emb_slot,
            self.emb_anchor,
        )
        for embedding in embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=std)
        out_norm = float(self.PG_out.norm(dim=1).mean().clamp(min=1e-6))
        in_norm = float(self.PG_in.norm(dim=1).mean().clamp(min=1e-6))
        nn.init.normal_(self.W_out.weight, std=std / out_norm)
        nn.init.normal_(self.W_in.weight, std=std / in_norm)
        nn.init.normal_(self.role_mask, std=std)
        nn.init.normal_(self.s_proj[0].weight, std=1.0)
        nn.init.zeros_(self.s_proj[0].bias)
        nn.init.normal_(self.s_proj[-1].weight, std=std / math.sqrt(self.d))
        nn.init.zeros_(self.s_proj[-1].bias)

    def graph_role(self, family: Tensor) -> Tensor:
        is_mask = family == self.config.family_mask_id
        clipped = family.clamp(max=self.config.num_families - 1)
        role = self.W_out(self.PG_out[clipped]) + self.W_in(self.PG_in[clipped])
        return torch.where(is_mask.unsqueeze(-1), self.role_mask.expand_as(role), role)

    @staticmethod
    def _expand_measure_condition(value: Tensor, batch: int, measures: int, name: str) -> Tensor:
        if value.shape == (measures,):
            return value.unsqueeze(0).expand(batch, -1)
        if value.shape == (batch, measures):
            return value
        raise ValueError(
            f"{name} must have shape ({measures},) or ({batch}, {measures}), got {tuple(value.shape)}"
        )

    def forward(
        self,
        pitch: Tensor,
        duration: Tensor,
        family: Tensor,
        corruption: Tensor,
        anchor: Tensor,
        chapter: Tensor,
        meter: Tensor,
        position: Tensor,
        hwanip_pairs: Tensor | None = None,
    ) -> DenoiserOutput:
        if pitch.ndim != 3 or duration.shape != pitch.shape:
            raise ValueError("pitch and duration must have identical (B, T, S) shapes")
        batch, measures, slots = pitch.shape
        if slots != self.config.max_events_per_measure:
            raise ValueError(f"Expected {self.config.max_events_per_measure} slots, got {slots}")
        if family.shape != (batch, measures):
            raise ValueError(f"family must have shape {(batch, measures)}, got {tuple(family.shape)}")
        if corruption.shape != (batch, 2):
            raise ValueError(f"corruption must have shape {(batch, 2)}")
        anchor = self._expand_measure_condition(anchor, batch, measures, "anchor")
        chapter = self._expand_measure_condition(chapter, batch, measures, "chapter")
        meter = self._expand_measure_condition(meter, batch, measures, "meter")
        position = self._expand_measure_condition(position, batch, measures, "position")

        hidden = self.emb_p(pitch) + self.emb_d(duration)
        measure_context = (
            self.emb_z(family)
            + self.graph_role(family)
            + self.emb_anchor(anchor.long())
            + self.emb_ch(chapter)
            + self.emb_mt(meter)
            + self.emb_pos(position)
        )
        hidden = hidden + measure_context.unsqueeze(2)
        slot_ids = torch.arange(slots, device=pitch.device)
        hidden = hidden + self.emb_slot(slot_ids)[None, None, :, :]
        hidden = hidden + self.s_proj(corruption)[:, None, None, :]
        if hwanip_pairs is not None and self.config.hwanip_message_scale > 0:
            if hwanip_pairs.ndim != 2 or hwanip_pairs.shape[1] != 2:
                raise ValueError("hwanip_pairs must have shape (R, 2)")
            pairs = hwanip_pairs.to(device=pitch.device, dtype=torch.long)
            if pairs.numel() and bool(((pairs < 0) | (pairs >= measures)).any()):
                raise ValueError("hwanip_pairs contains an out-of-range measure position")
            # Directed source→target message. No new parameters are introduced, so
            # legacy Checkpoint state dictionaries remain strictly loadable.
            pooled = hidden.mean(dim=2)
            source, target = pairs[:, 0], pairs[:, 1]
            message = torch.zeros_like(pooled).index_add(1, target, pooled[:, source])
            counts = torch.zeros(measures, dtype=hidden.dtype, device=hidden.device).index_add(
                0, target, torch.ones_like(target, dtype=hidden.dtype)
            )
            message = message / counts.clamp(min=1)[None, :, None]
            hidden = hidden + self.config.hwanip_message_scale * message.unsqueeze(2)
        hidden = self.in_norm(hidden)
        hidden = self.enc(hidden.reshape(batch, measures * slots, self.d))
        hidden = self.norm(hidden.reshape(batch, measures, slots, self.d))
        return DenoiserOutput(
            self.head_p(hidden),
            self.head_d(hidden),
            self.head_z(hidden.mean(dim=2)),
        )


# Compatibility name used by the original notebooks and checkpoints.
YGDenoiser = GugakDenoiser
