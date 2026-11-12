"""Masked (absorbing-state) discrete diffusion over grid phrases, and the graph loss (generator v1).

Forward process: every cell of a phrase is replaced by MASK independently with
probability ``r`` (linear schedule ``alpha_r = 1 - r``; D3PM absorbing state,
Austin et al. 2021, App. A.2.2 eq. 7). The denoiser predicts the clean token of
every masked cell and never predicts MASK (MDLM's SUBS parameterisation,
Sahoo et al. 2024, eq. 9). It is not conditioned on ``r`` (MDLM 3.2.2).

Losses (decided 2026-11-12):

* ``diffusion_loss``: the continuous-time ELBO, which for the linear schedule
  weights the masked-token cross-entropy by ``1/r`` (MDLM eq. 12 with
  ``alpha_r = 1 - r``). Computed on the original phrases.
* ``graph_loss`` (option (i)): computed at off-data points. New type sequences
  are given as conditions with every cell masked; the one-step predictions are
  read back into the original's types by the expected contour distance to the
  fixed medoids; the expected transition counts are compared with each
  sequence's own transition counts by total variation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gugakgen.chart.edit_distance import Event
from gugakgen.chart.grid import Grid
from gugakgen.chart.typology import Typology


@dataclass(frozen=True)
class Layout:
    grid: Grid
    num_types: int
    meters: tuple[int, ...]  # phrase lengths in cells, e.g. (16, 24)

    @property
    def max_cells(self) -> int:
        return max(self.meters)

    # boundary-note tokens: sounding states 0..P, then these
    @property
    def boundary_mask(self) -> int:
        return self.grid.num_states

    @property
    def boundary_none(self) -> int:
        return self.grid.num_states + 1

    @property
    def boundary_null(self) -> int:
        return self.grid.num_states + 2

    # type tokens: 0..K-1, then these
    @property
    def type_start(self) -> int:
        return self.num_types

    @property
    def type_end(self) -> int:
        return self.num_types + 1

    @property
    def type_null(self) -> int:
        return self.num_types + 2


class Tables:
    """Per-(type, meter) medoid references and contour costs, as tensors."""

    def __init__(self, typology: Typology, layout: Layout, device: torch.device) -> None:
        grid, size = layout.grid, layout.max_cells
        K, M = layout.num_types, len(layout.meters)
        ref = np.zeros((K, M, size), dtype=np.int64)
        onset = np.zeros((K, M, size), dtype=np.int64)
        cost = np.zeros((M, K, size, grid.num_states), dtype=np.float32)
        for m, cells in enumerate(layout.meters):
            cost[m, :, :cells] = grid.contour_costs(typology.medoids, cells)
            for c, medoid in enumerate(typology.medoids):
                ref[c, m, :cells], onset[c, m, :cells] = grid.reference(medoid, cells)
        self.ref = torch.as_tensor(ref, device=device)
        self.onset = torch.as_tensor(onset, device=device)
        self.cost = torch.as_tensor(cost, device=device)


# ----------------------------------------------------------------------------------------------
# Rows: one phrase with its conditions


def boundary_state(grid: Grid, tokens: Sequence[int], last: bool) -> int | None:
    """Sounding state at the phrase's last cell (``last``) or first cell; None if masked."""

    states = grid.sounding(tokens)
    return states[-1] if last else states[0]


def make_rows(
    phrases_tokens: Sequence[Sequence[int]],
    types: Sequence[int],
    layout: Layout,
    *,
    boundaries: bool = True,
) -> dict[str, np.ndarray]:
    """Arrays for a whole piece (one row per phrase). Neighbour boundary notes are read
    from the given tokens (masked cells give the boundary MASK token)."""

    grid, size, count = layout.grid, layout.max_cells, len(types)
    tokens = np.full((count, size), grid.mask, dtype=np.int64)
    length = np.zeros(count, dtype=np.int64)
    for t, phrase in enumerate(phrases_tokens):
        tokens[t, : len(phrase)] = phrase
        length[t] = len(phrase)

    def boundary(t: int, last: bool) -> int:
        if t < 0 or t >= count:
            return layout.boundary_none
        if not boundaries:
            return layout.boundary_mask
        state = boundary_state(grid, phrases_tokens[t], last)
        return layout.boundary_mask if state is None else state

    return {
        "tokens": tokens,
        "length": length,
        "meter": np.array([layout.meters.index(int(n)) for n in length], dtype=np.int64),
        "type": np.asarray(types, dtype=np.int64),
        "prev_type": np.array([types[t - 1] if t else layout.type_start for t in range(count)]),
        "next_type": np.array([types[t + 1] if t < count - 1 else layout.type_end for t in range(count)]),
        "prev_b": np.array([boundary(t - 1, True) for t in range(count)], dtype=np.int64),
        "next_b": np.array([boundary(t + 1, False) for t in range(count)], dtype=np.int64),
    }


# ----------------------------------------------------------------------------------------------
# Model


class GridDenoiser(nn.Module):
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
        grid, size = layout.grid, layout.max_cells
        self.layout = layout
        self.token = nn.Embedding(grid.mask + 1, d_model)
        self.position = nn.Embedding(size + 1, d_model)  # 0 is the context slot
        self.reference = nn.Embedding(grid.num_pitches, d_model)
        self.reference_onset = nn.Embedding(2, d_model)
        types = layout.type_null + 1
        self.own_type = nn.Embedding(types, d_model)
        self.prev_type = nn.Embedding(types, d_model)
        self.next_type = nn.Embedding(types, d_model)
        self.meter = nn.Embedding(len(layout.meters), d_model)
        self.prev_b = nn.Embedding(layout.boundary_null + 1, d_model)
        self.next_b = nn.Embedding(layout.boundary_null + 1, d_model)
        block = nn.TransformerEncoderLayer(d_model, n_heads, ffn_dim, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, grid.num_outputs)

    def forward(self, batch: dict[str, Tensor], tables: Tables) -> Tensor:
        """Log-probabilities ``(B, J, P + 2)`` of the clean token of every cell."""

        layout, grid = self.layout, self.layout.grid
        size = batch["tokens"].shape[1]
        position = torch.arange(1, size + 1, device=batch["tokens"].device)
        # the medoid of the conditioned type, stretched to this phrase's length
        own = batch["type"].clamp(max=layout.num_types - 1)
        ref = tables.ref[own, batch["meter"], :size]
        ref_on = tables.onset[own, batch["meter"], :size]
        known_type = (batch["type"] < layout.num_types)[:, None, None]
        cells = (
            self.token(batch["tokens"])
            + self.position(position)[None]
            + known_type * (self.reference(ref) + self.reference_onset(ref_on))
        )
        context = (
            self.own_type(batch["type"])
            + self.prev_type(batch["prev_type"])
            + self.next_type(batch["next_type"])
            + self.meter(batch["meter"])
            + self.prev_b(batch["prev_b"])
            + self.next_b(batch["next_b"])
            + self.position.weight[0]
        )
        hidden = torch.cat([context[:, None], cells], dim=1)
        padding = torch.arange(size, device=hidden.device)[None] >= batch["length"][:, None]
        padding = torch.cat([torch.zeros_like(padding[:, :1]), padding], dim=1)
        logits = self.head(self.norm(self.encoder(hidden, src_key_padding_mask=padding))[:, 1:])
        first = torch.zeros_like(logits[0, :, 0], dtype=torch.bool)
        first[0] = True
        logits = logits.masked_fill(first[None, :, None] & (torch.arange(grid.num_outputs, device=logits.device) == grid.hold), -1e4)
        return F.log_softmax(logits.float(), dim=-1)


def null_context(batch: dict[str, Tensor], rows: Tensor, layout: Layout) -> dict[str, Tensor]:
    """Replace neighbour types and boundary notes by the null token on ``rows``."""

    out = dict(batch)
    for key, null in (
        ("prev_type", layout.type_null),
        ("next_type", layout.type_null),
        ("prev_b", layout.boundary_null),
        ("next_b", layout.boundary_null),
    ):
        out[key] = torch.where(rows, torch.full_like(batch[key], null), batch[key])
    return out


# ----------------------------------------------------------------------------------------------
# Losses


def corrupt(batch: dict[str, Tensor], r: Tensor, generator: torch.Generator, mask_token: int) -> tuple[dict[str, Tensor], Tensor]:
    """Mask each valid cell independently with probability ``r`` (one ``r`` per row)."""

    size = batch["tokens"].shape[1]
    valid = torch.arange(size, device=r.device)[None] < batch["length"][:, None]
    masked = (torch.rand(batch["tokens"].shape, generator=generator, device=r.device) < r[:, None]) & valid
    out = dict(batch)
    out["tokens"] = torch.where(masked, torch.full_like(batch["tokens"], mask_token), batch["tokens"])
    return out, masked


def diffusion_loss(log_probs: Tensor, clean: Tensor, masked: Tensor, r: Tensor, length: Tensor) -> Tensor:
    """ELBO term per row: (1/r) * sum over masked cells of -log p(clean) / cells; mean over rows."""

    target = clean.clamp(max=log_probs.shape[-1] - 1)
    nll = -log_probs.gather(-1, target[..., None]).squeeze(-1) * masked
    return (nll.sum(1) / (r * length)).mean()


def sounding_marginals(probs: Tensor, grid: Grid) -> Tensor:
    """q[b, j, s]: probability that state ``s`` sounds at cell ``j`` (cells independent)."""

    states = probs[..., : grid.num_states]  # onsets of pitches 0..P-1, then onset of rest = state P
    hold = probs[..., grid.hold : grid.hold + 1]
    q = [states[:, 0]]
    for j in range(1, probs.shape[1]):
        q.append(states[:, j] + hold[:, j] * q[-1])
    return torch.stack(q, dim=1)


def soft_types(probs: Tensor, meter: Tensor, length: Tensor, tables: Tables, grid: Grid, gamma: float) -> Tensor:
    """a[b, c] = softmax_c(-d_c / gamma), d_c the expected contour distance to medoid c."""

    q = sounding_marginals(probs, grid)
    cost = tables.cost[meter][:, :, : q.shape[1]]  # (B, K, J, S)
    distance = torch.einsum("bjs,bkjs->bk", q, cost) / length[:, None]
    return F.softmax(-distance / gamma, dim=-1)


def transition_counts(assign: Tensor) -> Tensor:
    """Expected transition counts per walk: assign (W, T, K) -> (W, K, K)."""

    return torch.einsum("wtk,wtl->wkl", assign[:, :-1], assign[:, 1:])


def graph_loss(assign: Tensor, labels: Tensor, num_types: int) -> Tensor:
    """Option (i): mean over walks of TV(expected counts, the walk's own label counts)."""

    expected = transition_counts(assign)
    target = transition_counts(F.one_hot(labels, num_types).to(assign.dtype))
    p = expected / expected.sum(dim=(1, 2), keepdim=True)
    q = target / target.sum(dim=(1, 2), keepdim=True)
    return 0.5 * (p - q).abs().sum(dim=(1, 2)).mean()


# ----------------------------------------------------------------------------------------------
# Reverse process (generation)


def combine_returns(source: Tensor, target: Tensor) -> Tensor:
    """Joint prediction of one return pair: mean of the two log-probabilities (not normalised)."""

    return 0.5 * (source + target)


@torch.no_grad()
def reverse_sample(
    model: GridDenoiser,
    tables: Tables,
    types: Sequence[int],
    lengths: Sequence[int],
    *,
    steps: int,
    generator: torch.Generator,
    returns: Sequence[tuple[int, int]] = (),
    given: dict[int, Sequence[int]] | None = None,
    temperature: float = 1.0,
) -> list[list[int]]:
    """Ancestral sampling from all-masked phrases (MDLM 4.1).

    Going from time r to s < r, a masked cell is revealed with probability
    (r - s) / r and its token is drawn from the prediction. ``given`` phrases are
    fixed. For each return pair (source, target) the two positions are one
    variable: their predictions are combined and the drawn tokens are written to
    both (joint decoding). The combination is the normalised geometric mean
    (log-probabilities averaged). Adding them would count what both predictions
    share (type, medoid contour, the cell itself) twice and sharpen the draw
    towards the argmax (code review of 45618a9: entropy per cell 0.25 → 0.03).
    """

    layout, grid = model.layout, model.layout.grid
    device = tables.cost.device
    count, size = len(types), layout.max_cells
    given = given or {}
    target_of = {target: source for source, target in returns}
    for target, source in target_of.items():
        if lengths[target] != lengths[source]:
            raise ValueError(f"return pair ({source}, {target}) has phrases of different lengths")
    state = torch.full((count, size), grid.mask, dtype=torch.long, device=device)
    valid = torch.arange(size, device=device)[None] < torch.as_tensor(lengths, device=device)[:, None]
    for t, phrase in given.items():
        state[t, : len(phrase)] = torch.as_tensor(list(phrase), device=device)
    for target, source in target_of.items():
        state[target] = state[source]
    for k in range(steps):
        r, s = 1.0 - k / steps, 1.0 - (k + 1) / steps
        tokens = [state[t, : lengths[t]].tolist() for t in range(count)]
        rows = make_rows(tokens, types, layout)
        batch = {key: torch.as_tensor(value, device=device) for key, value in rows.items()}
        log_probs = model(batch, tables) / temperature
        for target, source in target_of.items():
            log_probs[source] = combine_returns(log_probs[source], log_probs[target])
        log_probs = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
        masked = (state == grid.mask) & valid
        reveal = masked & (torch.rand(state.shape, generator=generator, device=device) < (r - s) / r)
        drawn = torch.multinomial(log_probs.exp().reshape(-1, grid.num_outputs), 1, generator=generator)
        drawn = drawn.reshape(count, size)
        for target in target_of:
            reveal[target] = False
        state = torch.where(reveal, drawn, state)
        for target, source in target_of.items():
            state[target] = state[source]
    return [state[t, : lengths[t]].tolist() for t in range(count)]


def decode_piece(grid: Grid, tokens: Sequence[Sequence[int]]) -> list[tuple[Event, ...]]:
    return [tuple(grid.decode(phrase)) for phrase in tokens]
