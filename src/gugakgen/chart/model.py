"""Restoration model q_theta over chart coordinates, and its losses (Methods draft, 2.6).

One training example is one phrase. A contiguous span of its events is hidden
and the model predicts the hidden deviations ``eta`` and durations from the
visible events, the medoid notes they sit on, and the context
``kappa_t = (f_{t-1}, f_t, f_{t+1}, B_t)`` with the neighbouring boundary notes.

The ``eta`` head is shared by all types (one distribution of deviations for the
piece); the type embedding adjusts it. Context and type can be replaced by a
null token during training (``context_dropout``, ``type_dropout``) so that the
model cannot rely on them as an address of the original measure.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gugakgen.chart.coords import Scale, reference_steps, surrogate_tables, to_chart, to_quanta
from gugakgen.chart.edit_distance import Event
from gugakgen.chart.typology import Typology


@dataclass(frozen=True)
class Layout:
    """Token layout shared by data preparation, the model and the sampler."""

    scale: Scale
    durations: tuple[int, ...]  # duration vocabulary in quanta, ascending
    num_types: int
    max_events: int
    meters: tuple[int, ...]  # phrase totals B in quanta

    @property
    def num_steps(self) -> int:
        return len(self.scale)

    @property
    def eta_offset(self) -> int:
        return self.num_steps - 1

    @property
    def num_eta(self) -> int:  # eta in [-(n-1), n-1], then REST
        return 2 * self.num_steps - 1

    @property
    def eta_rest(self) -> int:
        return self.num_eta

    @property
    def eta_mask(self) -> int:
        return self.num_eta + 1

    @property
    def eta_pad(self) -> int:
        return self.num_eta + 2

    @property
    def dur_mask(self) -> int:
        return len(self.durations)

    @property
    def dur_pad(self) -> int:
        return len(self.durations) + 1

    @property
    def pitch_rest(self) -> int:  # pitch token: scale step, or this for a rest
        return self.num_steps

    # context tokens
    @property
    def type_null(self) -> int:
        return self.num_types + 2  # after START, END

    @property
    def type_start(self) -> int:
        return self.num_types

    @property
    def type_end(self) -> int:
        return self.num_types + 1

    @property
    def boundary_none(self) -> int:  # no neighbouring phrase
        return self.num_steps + 1

    @property
    def boundary_null(self) -> int:
        return self.num_steps + 2

    def pitch_token(self, pitch: int | None) -> int:
        return self.pitch_rest if pitch is None else self.scale.step(pitch)

    @classmethod
    def from_typology(cls, typology: Typology) -> Layout:
        phrases = typology.phrases
        return cls(
            scale=Scale.from_phrases(phrases),
            durations=tuple(sorted({to_quanta(d) for phrase in phrases for _, d in phrase})),
            num_types=typology.k,
            max_events=max(len(phrase) for phrase in phrases),
            meters=tuple(sorted({sum(to_quanta(d) for _, d in phrase) for phrase in phrases})),
        )


def encode_phrase(
    phrase: Sequence[Event],
    type_index: int,
    previous: tuple[int, Sequence[Event]] | None,
    following: tuple[int, Sequence[Event]] | None,
    typology: Typology,
    layout: Layout,
) -> dict[str, np.ndarray]:
    """Arrays for one phrase. ``previous``/``following`` are (type, events) of the neighbours."""

    medoid = typology.medoids[type_index]
    chart = to_chart(phrase, medoid, layout.scale)
    refs = reference_steps(medoid, layout.scale)
    size = layout.max_events
    if len(chart) > size:
        raise ValueError(f"A phrase has {len(chart)} events; the layout allows {size}")
    eta = np.full(size, layout.eta_pad, dtype=np.int64)
    dur = np.full(size, layout.dur_pad, dtype=np.int64)
    slot = np.zeros(size, dtype=np.int64)
    ref = np.zeros(size, dtype=np.int64)
    ref_dur = np.zeros(size, dtype=np.int64)
    inserted = np.zeros(size, dtype=np.int64)
    for j, event in enumerate(chart):
        eta[j] = layout.eta_rest if event.eta is None else event.eta + layout.eta_offset
        dur[j] = layout.durations.index(event.quanta)
        slot[j] = event.slot
        ref[j] = refs[event.slot]
        ref_dur[j] = min(to_quanta(medoid[event.slot][1]), 15)
        inserted[j] = int(event.inserted)

    def last_pitch(events: Sequence[Event]) -> int:
        return layout.pitch_token(events[-1][0])

    def first_pitch(events: Sequence[Event]) -> int:
        return layout.pitch_token(events[0][0])

    return {
        "eta": eta,
        "dur": dur,
        "slot": slot,
        "ref": ref,
        "ref_dur": ref_dur,
        "inserted": inserted,
        "length": np.int64(len(chart)),
        "type": np.int64(type_index),
        "prev_type": np.int64(layout.type_start if previous is None else previous[0]),
        "next_type": np.int64(layout.type_end if following is None else following[0]),
        "prev_pitch": np.int64(layout.boundary_none if previous is None else last_pitch(previous[1])),
        "next_pitch": np.int64(
            layout.boundary_none if following is None else first_pitch(following[1])
        ),
        "meter": np.int64(layout.meters.index(sum(event.quanta for event in chart))),
    }


def encode_piece(
    phrases: Sequence[Sequence[Event]], typology: Typology, layout: Layout
) -> dict[str, np.ndarray]:
    """Stack ``encode_phrase`` over a piece whose phrase ``t`` has type ``typology.labels[t]``."""

    labels = typology.labels
    rows = [
        encode_phrase(
            phrase,
            labels[t],
            None if t == 0 else (labels[t - 1], phrases[t - 1]),
            None if t == len(phrases) - 1 else (labels[t + 1], phrases[t + 1]),
            typology,
            layout,
        )
        for t, phrase in enumerate(phrases)
    ]
    return {key: np.stack([row[key] for row in rows]) for key in rows[0]}


def margin_tables(typology: Typology, layout: Layout) -> tuple[np.ndarray, np.ndarray]:
    """``surrogate_tables`` for every phrase of the original: shapes (T, K, S, P) and (T, K)."""

    tables = [
        surrogate_tables(phrase, typology.medoids, layout.scale, layout.max_events)
        for phrase in typology.phrases
    ]
    return np.stack([cost for cost, _ in tables]), np.stack([const for _, const in tables])


class ChartDenoiser(nn.Module):
    def __init__(
        self,
        layout: Layout,
        d_model: int = 96,
        n_layers: int = 3,
        n_heads: int = 4,
        ffn_dim: int = 192,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layout = layout
        size = layout.max_events
        self.eta = nn.Embedding(layout.eta_pad + 1, d_model)
        self.dur = nn.Embedding(layout.dur_pad + 1, d_model)
        self.slot = nn.Embedding(size, d_model)
        self.ref = nn.Embedding(layout.num_steps, d_model)
        self.ref_dur = nn.Embedding(16, d_model)
        self.inserted = nn.Embedding(2, d_model)
        self.order = nn.Embedding(size + 1, d_model)
        self.own_type = nn.Embedding(layout.type_null + 1, d_model)
        self.prev_type = nn.Embedding(layout.type_null + 1, d_model)
        self.next_type = nn.Embedding(layout.type_null + 1, d_model)
        self.prev_pitch = nn.Embedding(layout.boundary_null + 1, d_model)
        self.next_pitch = nn.Embedding(layout.boundary_null + 1, d_model)
        self.meter = nn.Embedding(len(layout.meters), d_model)
        block = nn.TransformerEncoderLayer(
            d_model, n_heads, ffn_dim, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(block, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.eta_head = nn.Linear(d_model, layout.num_eta)
        self.rest_head = nn.Linear(d_model, 1)
        self.dur_head = nn.Linear(d_model, len(layout.durations))

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Return log-probabilities over pitch tokens ``(B, S, P)`` and durations ``(B, S, D)``.

        The eta logits are read out at ``pitch step - reference step``, so the
        distribution is normalised over the pitches that exist in the scale.
        """

        layout = self.layout
        size = batch["eta"].shape[1]
        order = torch.arange(1, size + 1, device=batch["eta"].device)
        events = (
            self.eta(batch["eta"])
            + self.dur(batch["dur"])
            + self.slot(batch["slot"])
            + self.ref(batch["ref"])
            + self.ref_dur(batch["ref_dur"])
            + self.inserted(batch["inserted"])
            + self.order(order)[None]
        )
        context = (
            self.own_type(batch["type"])
            + self.prev_type(batch["prev_type"])
            + self.next_type(batch["next_type"])
            + self.prev_pitch(batch["prev_pitch"])
            + self.next_pitch(batch["next_pitch"])
            + self.meter(batch["meter"])
            + self.order.weight[0]
        )
        hidden = torch.cat([context[:, None], events], dim=1)
        padding = torch.arange(size, device=hidden.device)[None] >= batch["length"][:, None]
        padding = torch.cat([torch.zeros_like(padding[:, :1]), padding], dim=1)
        hidden = self.norm(self.encoder(hidden, src_key_padding_mask=padding))[:, 1:]
        steps = torch.arange(layout.num_steps, device=hidden.device)
        index = steps[None, None] - batch["ref"][..., None] + layout.eta_offset
        pitch_logits = torch.cat(
            [self.eta_head(hidden).gather(-1, index), self.rest_head(hidden)], dim=-1
        )
        return F.log_softmax(pitch_logits, dim=-1), F.log_softmax(self.dur_head(hidden), dim=-1)


def pitch_targets(batch: dict[str, Tensor], layout: Layout) -> Tensor:
    """Pitch token of every event from its eta token and reference step (pads map to 0)."""

    eta = batch["eta"]
    step = (eta - layout.eta_offset + batch["ref"]).clamp(0, layout.num_steps - 1)
    return torch.where(eta == layout.eta_rest, torch.full_like(step, layout.pitch_rest), step)


def hide(
    batch: dict[str, Tensor],
    hidden: Tensor,
    layout: Layout,
    *,
    null_context: Tensor | None = None,
    null_type: Tensor | None = None,
) -> dict[str, Tensor]:
    """Model input with ``hidden`` events masked; optionally null context/type per row."""

    out = dict(batch)
    out["eta"] = torch.where(hidden, torch.full_like(batch["eta"], layout.eta_mask), batch["eta"])
    out["dur"] = torch.where(hidden, torch.full_like(batch["dur"], layout.dur_mask), batch["dur"])
    if null_context is not None:
        for key, null in (
            ("prev_type", layout.type_null),
            ("next_type", layout.type_null),
            ("prev_pitch", layout.boundary_null),
            ("next_pitch", layout.boundary_null),
        ):
            out[key] = torch.where(null_context, torch.full_like(batch[key], null), batch[key])
    if null_type is not None:
        out["type"] = torch.where(
            null_type, torch.full_like(batch["type"], layout.type_null), batch["type"]
        )
    return out


def random_spans(
    length: Tensor, size: int, generator: torch.Generator, low: float, high: float
) -> Tensor:
    """One contiguous hidden span per row of ``(rows, size)``, covering a ``U(low, high)`` share of its events."""

    rows, device = length.shape[0], length.device
    ratio = low + (high - low) * torch.rand(rows, generator=generator, device=device)
    span = (ratio * length).round().clamp(min=1).long().minimum(length)
    start = (torch.rand(rows, generator=generator, device=device) * (length - span + 1)).long()
    position = torch.arange(size, device=device)[None]
    return (position >= start[:, None]) & (position < (start + span)[:, None])


@dataclass(frozen=True)
class LossWeights:
    margin: float = 0.0  # lambda_M
    beats: float = 0.0  # lambda_B
    margin_size: float = 0.05
    softmin_temperature: float = 0.05


def losses(
    log_pitch: Tensor,
    log_dur: Tensor,
    batch: dict[str, Tensor],
    hidden: Tensor,
    layout: Layout,
    weights: LossWeights,
    margin_cost: Tensor | None = None,
    margin_const: Tensor | None = None,
) -> dict[str, Tensor]:
    """L_rec, and the soft structural terms L_M and L_B of the draft's eq. 10.

    L_M: expected surrogate distance (alignment fixed) to the own medoid must be
    smaller, by ``margin_size``, than the soft minimum over the other medoids.
    L_B: the expected total duration of the hidden span must equal its true total.
    L_G and L_R are not defined here: with the type sequence fixed and the return
    positions copied (setting (ga) of the draft) they have nothing to act on.
    """

    target_pitch = pitch_targets(batch, layout)
    count = hidden.sum().clamp(min=1)
    nll_pitch = -(log_pitch.gather(-1, target_pitch[..., None]).squeeze(-1) * hidden).sum() / count
    target_dur = batch["dur"].clamp(max=len(layout.durations) - 1)
    nll_dur = -(log_dur.gather(-1, target_dur[..., None]).squeeze(-1) * hidden).sum() / count
    out = {"pitch": nll_pitch, "duration": nll_dur, "rec": nll_pitch + nll_dur}
    total = out["rec"]

    valid = torch.arange(hidden.shape[1], device=hidden.device)[None] < batch["length"][:, None]
    if weights.margin > 0:
        if margin_cost is None or margin_const is None:
            raise ValueError("L_M needs the surrogate tables of the batch")
        truth = F.one_hot(target_pitch, layout.num_steps + 1).to(log_pitch.dtype)
        belief = torch.where(hidden[..., None], log_pitch.exp(), truth) * valid[..., None]
        expected = torch.einsum("bsp,bksp->bk", belief, margin_cost) + margin_const
        own = expected.gather(1, batch["type"][:, None]).squeeze(1)
        others = expected.masked_fill(
            F.one_hot(batch["type"], layout.num_types).bool(), float("inf")
        )
        tau = weights.softmin_temperature
        nearest_other = -tau * torch.logsumexp(-others / tau, dim=1)
        out["margin"] = F.relu(weights.margin_size + own - nearest_other).mean()
        total = total + weights.margin * out["margin"]
    if weights.beats > 0:
        quanta = torch.tensor(layout.durations, dtype=log_dur.dtype, device=log_dur.device)
        expected = ((log_dur.exp() * quanta).sum(-1) * hidden).sum(1)
        true = (quanta[target_dur] * hidden).sum(1)
        phrase_total = (quanta[target_dur] * valid).sum(1)
        out["beats"] = (((expected - true) / phrase_total) ** 2).mean()
        total = total + weights.beats * out["beats"]
    out["total"] = total
    return out
