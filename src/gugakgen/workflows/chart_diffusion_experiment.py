"""Generator v1 experiment: type walks on G + masked discrete diffusion + graph loss (i).

Arms differ only in the weight of the graph loss (0 = diffusion only). For each
arm and training seed:

1. held-out runs (folds of distinct phrases) record the held-out ELBO along
   training; the training length with the lowest mean held-out ELBO is chosen;
2. a full run on all 72 phrases to that length, with monitoring: at
   ``monitor.points`` evenly spaced steps a few pieces are generated end to end
   (no gradient) and their surface phrase graph is compared with the conditions;
3. 60 pieces are generated on new type walks;
4. for the data-copying test, each fold model is retrained to the selected
   length and generates pieces (``evaluation.copy_test``);
5. ``evaluation.json`` scores every model on the axes fixed on 2026-11-12
   (see ``gugakgen.chart.evaluation``) and judges the criteria that need no
   other generator.

Every unit writes one JSON under ``artifacts/chart_diffusion_runs/<RUN_TAG>/``;
existing units are read back, so a run resumes under the same RUN_TAG, and no
other RUN_TAG is touched.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gugakgen.chart.coords import cached_distance
from gugakgen.chart.diffusion import (
    GridDenoiser,
    Layout,
    Tables,
    corrupt,
    decode_piece,
    diffusion_loss,
    graph_loss,
    make_rows,
    null_context,
    reverse_sample,
    soft_types,
)
from gugakgen.chart.evaluation import (
    EVAL_PROTOCOL,
    Reference,
    as_phrase,
    bootstrap_ci,
    copy_test,
    judge,
    model_scores,
)
from gugakgen.chart.grid import Grid
from gugakgen.chart.metrics import _intervals, _jsd, _ngrams, _notes, summarise
from gugakgen.chart.typology import Typology, build_typology, load_phrases
from gugakgen.chart.walk import TypeWalk
from gugakgen.config import load_yaml
from gugakgen.device import RuntimeDevice
from gugakgen.reproducibility import set_global_seed
from gugakgen.workflows.chart_experiment import (
    _commit,
    _unit,
    address_audit,
    heldout_folds,
    typology_report,
)

# 2026-11-12 (2nd): the graph loss starts at step 0, the held-out grid is finer and ends at
# 1000 steps, and monitoring is spread over the chosen length. The first full run
# (RUN_TAG 20261112_1717) chose step 100 for both arms, before the graph loss started at
# step 200, so both arms were the same model.
# 2026-11-12 (3rd, code review of 45618a9): return pairs are decoded from the mean of the two
# log-probabilities (not the sum), the copy test also reports Z on measures that are not identical
# to a training measure, coverage has no bootstrap interval, the held-out baseline is a row mean
# like the ELBO, a full unit resumes from its checkpoint, and meta also pins the piece config and
# the source files.
PROTOCOL = "chart_diffusion_v1_eval3_2026-11-12"
SOURCE_FILES = (
    "src/gugakgen/chart/diffusion.py", "src/gugakgen/chart/edit_distance.py", "src/gugakgen/chart/evaluation.py",
    "src/gugakgen/chart/grid.py", "src/gugakgen/chart/metrics.py", "src/gugakgen/chart/typology.py",
    "src/gugakgen/chart/walk.py", "src/gugakgen/workflows/chart_diffusion_experiment.py",
    "src/gugakgen/workflows/chart_experiment.py",
)
Batch = dict[str, torch.Tensor]


# ----------------------------------------------------------------------------------------------
# Setup


def load_setup(config_path: Path, project_root: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    piece = load_yaml(project_root / config["piece_config"])
    measures_path = project_root / piece["data"]["canonical_measures"]
    phrases, chapters = load_phrases(measures_path)
    typology = build_typology(phrases, chapters, k=int(config["typology"]["k"]))
    grid = Grid.from_phrases(phrases, int(config["grid"]["cells_per_beat"]))
    tokens = [grid.encode(phrase) for phrase in phrases]
    layout = Layout(grid, typology.k, tuple(sorted({len(t) for t in tokens})))
    returns: list[tuple[int, int]] = []
    for relation in piece["structure"]["hwanip_relations"]:
        (s0, s1), (t0, t1) = relation["source_measures"], relation["target_measures"]
        if s1 - s0 != t1 - t0:
            raise ValueError("Only one-to-one return relations are supported")
        returns.extend((s0 - 1 + i, t0 - 1 + i) for i in range(s1 - s0 + 1))
    walk = TypeWalk(
        typology.transition_counts(),
        len(phrases),
        typology.labels[0],
        typology.labels[-1],
        returns,
    )
    return {
        "config": config,
        "typology": typology,
        "grid": grid,
        "layout": layout,
        "tokens": tokens,
        "lengths": [len(t) for t in tokens],
        "rows": make_rows(tokens, typology.labels, layout),
        "returns": tuple(returns),
        "walk": walk,
        "edges": typology.transition_counts() > 0,
        "measures_sha256": hashlib.sha256(measures_path.read_bytes()).hexdigest(),
    }


def walk_rows(walks: np.ndarray, lengths: Sequence[int], layout: Layout) -> dict[str, np.ndarray]:
    """Rows for whole pieces whose cells are all masked: shape (W, T, ...)."""

    count, positions = walks.shape
    size = layout.max_cells
    length = np.broadcast_to(np.asarray(lengths), (count, positions))
    prev_type = np.concatenate([np.full((count, 1), layout.type_start), walks[:, :-1]], axis=1)
    next_type = np.concatenate([walks[:, 1:], np.full((count, 1), layout.type_end)], axis=1)
    prev_b = np.full((count, positions), layout.boundary_mask)
    next_b = np.full((count, positions), layout.boundary_mask)
    prev_b[:, 0] = next_b[:, -1] = layout.boundary_none
    return {
        "tokens": np.full((count, positions, size), layout.grid.mask),
        "length": length.copy(),
        "meter": np.vectorize(layout.meters.index)(length),
        "type": walks,
        "prev_type": prev_type,
        "next_type": next_type,
        "prev_b": prev_b,
        "next_b": next_b,
    }


def to_device(arrays: Mapping[str, np.ndarray], device: torch.device) -> Batch:
    return {key: torch.as_tensor(np.asarray(value), dtype=torch.long, device=device) for key, value in arrays.items()}


# ----------------------------------------------------------------------------------------------
# Scores


def heldout_elbo(
    model: GridDenoiser, tables: Tables, rows: Batch, index: Sequence[int], config: Mapping[str, Any], *, null: bool
) -> float:
    """Deterministic ELBO estimate (nats per cell) on the listed phrases: fixed r grid, fixed masks."""

    if not index:
        return float("nan")
    layout = model.layout
    pick = torch.as_tensor(list(index), device=rows["tokens"].device)
    base = {key: value[pick] for key, value in rows.items()}
    if null:
        base = null_context(base, torch.ones(len(pick), dtype=torch.bool, device=pick.device), layout)
    generator = torch.Generator(device=pick.device).manual_seed(0)
    values = []
    with torch.no_grad():
        for r_value in config["heldout"]["r_grid"]:
            for _ in range(int(config["heldout"]["draws"])):
                r = torch.full((len(pick),), float(r_value), device=pick.device)
                noisy, masked = corrupt(base, r, generator, layout.grid.mask)
                values.append(float(diffusion_loss(model(noisy, tables), base["tokens"], masked, r, base["length"])))
    return float(np.mean(values))


def full_mask_recall(model: GridDenoiser, tables: Tables, rows: Batch, *, null: bool) -> float:
    """Share of the original phrases reproduced exactly by the argmax from an all-masked input."""

    layout = model.layout
    batch = dict(rows)
    valid = torch.arange(layout.max_cells, device=rows["tokens"].device)[None] < rows["length"][:, None]
    batch["tokens"] = torch.where(valid, torch.full_like(rows["tokens"], layout.grid.mask), rows["tokens"])
    if null:
        batch = null_context(batch, torch.ones(len(valid), dtype=torch.bool, device=valid.device), layout)
    with torch.no_grad():
        guess = model(batch, tables).argmax(-1)
    same = ((guess == rows["tokens"]) | ~valid).all(dim=1)
    return float(same.float().mean())


def _counts(labels: Sequence[int], k: int) -> np.ndarray:
    counts = np.zeros((k, k))
    for a, b in zip(labels[:-1], labels[1:], strict=True):
        counts[a, b] += 1
    return counts


def _tv(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(a / a.sum() - b / b.sum()).sum())


def surface_types(phrases: Sequence[Sequence], typology: Typology) -> list[int]:
    return [
        int(np.argmin([cached_distance(phrase, medoid) for medoid in typology.medoids]))
        for phrase in phrases
    ]


def piece_metrics(
    phrases: Sequence[tuple], walk: Sequence[int], setup: Mapping[str, Any], first_pass: float
) -> dict[str, float]:
    typology: Typology = setup["typology"]
    returns, edges = setup["returns"], setup["edges"]
    free = [t for t in range(len(walk)) if t not in {target for _, target in returns}]
    surface = surface_types(phrases, typology)
    original = typology.phrases
    notes, original_notes = _notes(phrases), _notes(original)
    seen2, seen4 = set(_ngrams(original_notes, 2)), set(_ngrams(original_notes, 4))
    return {
        # structure: does the locally generated surface realise the conditioned type walk?
        "first_pass_membership": first_pass,
        "membership": float(np.mean([typology.is_member(phrases[t], walk[t]) for t in free])),
        "surface_edge_validity": float(np.mean([bool(edges[a, b]) for a, b in zip(surface[:-1], surface[1:], strict=True)])),
        "surface_graph_tv": _tv(_counts(surface, typology.k), _counts(walk, typology.k)),
        "returns_exact": float(all(phrases[s] == phrases[t] for s, t in returns)),
        # novelty
        "walk_positions_differing": float(np.mean(np.asarray(walk) != np.asarray(typology.labels))),
        "original_phrase_rate": float(np.mean([phrases[t] in set(original) for t in free])),
        "unseen_note_transition_rate": float(np.mean([g not in seen2 for g in _ngrams(notes, 2)])),
        "note_4gram_novelty": float(np.mean([g not in seen4 for g in _ngrams(notes, 4)])),
        # style (reference)
        "interval_jsd": _jsd(_intervals(notes), _intervals(original_notes)),
        "duration_jsd": _jsd(
            defaultdict(int, _count_durations(notes)), defaultdict(int, _count_durations(original_notes))
        ),
    }


def _count_durations(notes: Sequence) -> dict[float, int]:
    out: dict[float, int] = defaultdict(int)
    for _, duration in notes:
        out[duration] += 1
    return out


def generate_piece(
    model: GridDenoiser, tables: Tables, setup: Mapping[str, Any], seed: int, *, attempts: int | None = None
) -> tuple[list[tuple], list[int], float]:
    """One piece on a new type walk. Returns phrases, the walk and the first-pass membership rate."""

    config, typology, grid = setup["config"], setup["typology"], setup["grid"]
    sampling = config["sampling"]
    walk = [int(x) for x in setup["walk"].sample(np.random.default_rng(seed))]
    generator = torch.Generator(device=tables.cost.device).manual_seed(seed)
    returns, lengths = setup["returns"], setup["lengths"]
    kwargs = {"steps": int(sampling["steps"]), "generator": generator, "returns": returns,
              "temperature": float(sampling["temperature"])}
    tokens = reverse_sample(model, tables, walk, lengths, **kwargs)
    target_of = {target: source for source, target in returns}
    free = [t for t in range(len(walk)) if t not in target_of]

    def rejected(state: list[list[int]]) -> list[int]:
        return [t for t in free if not typology.is_member(tuple(grid.decode(state[t])), walk[t])]

    bad = rejected(tokens)
    first_pass = 1.0 - len(bad) / len(free)
    for _ in range(int(sampling["rejection_attempts"]) if attempts is None else attempts):
        if not bad:
            break
        redo = set(bad) | {target for target, source in target_of.items() if source in bad}
        given = {t: tokens[t] for t in range(len(walk)) if t not in redo}
        tokens = reverse_sample(model, tables, walk, lengths, given=given, **kwargs)
        bad = rejected(tokens)
    return decode_piece(grid, tokens), walk, first_pass


def generation_table(model: GridDenoiser, tables: Tables, setup: Mapping[str, Any], seeds: Sequence[int]) -> dict[str, Any]:
    rows, pieces = [], []
    for seed in seeds:
        phrases, walk, first_pass = generate_piece(model, tables, setup, seed)
        rows.append(piece_metrics(phrases, walk, setup, first_pass))
        pieces.append({"seed": int(seed), "walk": [int(x) for x in walk],
                       "phrases": [[[e[0], float(e[1])] for e in p] for p in phrases]})
    return {"metrics": summarise(rows), "per_piece": rows, "pieces": pieces}


# ----------------------------------------------------------------------------------------------
# Training


def train_model(
    setup: Mapping[str, Any],
    arm: Mapping[str, Any],
    train_index: Sequence[int],
    seed: int,
    steps: int,
    runtime: RuntimeDevice,
    callbacks: Mapping[int, Callable[[GridDenoiser, Tables], None]] | None = None,
) -> tuple[GridDenoiser, Tables, list[dict[str, float]]]:
    config, layout = setup["config"], setup["layout"]
    training, graph = config["training"], config["graph"]
    set_global_seed(seed)
    device = runtime.device
    model = GridDenoiser(layout, **config["model"]).to(device)
    tables = Tables(setup["typology"], layout, device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    rows = to_device(setup["rows"], device)
    index = torch.as_tensor(list(train_index), device=device)
    pool_rng = np.random.default_rng(seed)
    pool = np.array([setup["walk"].sample(pool_rng) for _ in range(int(graph["walk_pool"]))])
    pool_rows = to_device(walk_rows(pool, setup["lengths"], layout), device)
    pool_labels = torch.as_tensor(pool, device=device)
    weight = float(arm["lambda_graph"])
    generator = torch.Generator(device=device).manual_seed(seed)
    history: list[dict[str, float]] = []
    model.train()
    for step in range(1, steps + 1):
        pick = index[torch.randint(len(index), (training["batch_size"],), generator=generator, device=device)]
        clean = {key: value[pick] for key, value in rows.items()}
        r_min = float(training["r_min"])
        r = r_min + (1 - r_min) * torch.rand(len(pick), generator=generator, device=device)
        noisy, masked = corrupt(clean, r, generator, layout.grid.mask)
        drop = torch.rand(len(pick), generator=generator, device=device) < float(training["context_dropout"])
        noisy = null_context(noisy, drop, layout)
        terms = {"diffusion": diffusion_loss(model(noisy, tables), clean["tokens"], masked, r, clean["length"])}
        total = terms["diffusion"]
        ramp = min(1.0, max(0.0, (step - graph["start_step"]) / max(graph["ramp_steps"], 1)))
        if weight > 0 and ramp > 0:
            walks = torch.randint(len(pool), (int(graph["walks_per_step"]),), generator=generator, device=device)
            batch = {key: value[walks].flatten(0, 1) for key, value in pool_rows.items()}
            probs = model(batch, tables).exp()
            assign = soft_types(probs, batch["meter"], batch["length"], tables, layout.grid, float(graph["gamma"]))
            terms["graph"] = graph_loss(assign.view(len(walks), -1, layout.num_types), pool_labels[walks], layout.num_types)
            total = total + weight * ramp * terms["graph"]
        rate = training["learning_rate"] * min(1.0, step / max(training["warmup_steps"], 1))
        for group in optimiser.param_groups:
            group["lr"] = rate
        optimiser.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"])
        optimiser.step()
        if step % max(steps // 20, 1) == 0 or step == steps:
            history.append({"step": step, **{k: float(v.detach()) for k, v in terms.items()}})
        if callbacks and step in callbacks:
            model.eval()
            callbacks[step](model, tables)
            model.train()
    return model.eval(), tables, history


def source_digest() -> str:
    """SHA-256 over the source files of this experiment (catches uncommitted edits and runs without git)."""

    package = Path(__file__).resolve().parents[1]  # src/gugakgen
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        digest.update(name.encode())
        digest.update((package / name.removeprefix("src/gugakgen/")).read_bytes())
    return digest.hexdigest()


# Digest of the source files as they were when this module was imported. If the files on disk
# change afterwards (e.g. a git pull in a running Colab session), the code in memory is stale:
# run 20261112_1955 recorded commit f46935c in meta.json but ran the 45618a9 code.
LOADED_SOURCE = source_digest()


def check_loaded_source() -> None:
    if source_digest() != LOADED_SOURCE:
        raise RuntimeError(
            "소스 파일이 바뀌었는데 메모리에는 이전 코드가 남아 있습니다(git pull 뒤 재시작하지 않음). "
            "Runtime → Restart session 후 처음부터 다시 실행하세요."
        )


def monitor_steps(steps: int, points: int) -> list[int]:
    """``points`` evenly spaced steps ending at ``steps`` (fewer if the run is shorter)."""

    return sorted({max(1, round(steps * (i + 1) / points)) for i in range(points)})


def positional_baseline(setup: Mapping[str, Any], held: Sequence[int]) -> float:
    """Context-free reference for the held-out ELBO (nats per cell).

    Each held-out cell is predicted by the training phrases of the same length at the same
    cell (add-1/2 smoothing). A model that ignores type, neighbours and the unmasked cells
    reaches exactly this ELBO, so the curve should dip below it before it overfits.
    """

    tokens, outputs = setup["tokens"], setup["grid"].num_outputs
    train = [t for t in range(len(tokens)) if t not in set(held)]
    rows = []  # mean over the cells of a phrase, then over phrases, as in ``heldout_elbo``
    for t in held:
        same = [tokens[u] for u in train if len(tokens[u]) == len(tokens[t])]
        losses = []
        for j, token in enumerate(tokens[t]):
            counts = np.full(outputs, 0.5)
            for other in same:
                counts[other[j]] += 1
            losses.append(-np.log(counts[token] / counts.sum()))
        rows.append(np.mean(losses))
    return float(np.mean(rows))


def graph_active_steps(config: Mapping[str, Any], arm: str, steps: int) -> int:
    """Steps on which the graph loss had a positive weight (0 for the diffusion-only arm)."""

    if float(config["arms"][arm]["lambda_graph"]) <= 0:
        return 0
    return max(0, steps - int(config["graph"]["start_step"]))


def eval_steps(config: Mapping[str, Any], steps: int) -> list[int]:
    return sorted({int(s) for s in config["heldout"]["eval_steps"] if int(s) < steps} | {steps})


def heldout_unit(setup: Mapping[str, Any], arm_name: str, seed: int, fold: int, steps: int, runtime: RuntimeDevice) -> dict[str, Any]:
    config, typology = setup["config"], setup["typology"]
    held = heldout_folds(typology, config["heldout"]["folds"], config["heldout"]["fold_seed"])[fold]
    train = [t for t in range(len(typology.labels)) if t not in set(held)]
    curve: list[dict[str, Any]] = []

    def record(step: int) -> Callable[[GridDenoiser, Tables], None]:
        def call(model: GridDenoiser, tables: Tables) -> None:
            rows = to_device(setup["rows"], tables.cost.device)
            curve.append({
                "step": step,
                "heldout_elbo": heldout_elbo(model, tables, rows, held, config, null=False),
                "heldout_elbo_null_context": heldout_elbo(model, tables, rows, held, config, null=True),
                "train_elbo": heldout_elbo(model, tables, rows, train, config, null=False),
            })
        return call

    _, _, history = train_model(
        setup, config["arms"][arm_name], train, seed, steps, runtime,
        {s: record(s) for s in eval_steps(config, steps)},
    )
    return {"arm": arm_name, "seed": seed, "fold": fold, "steps": steps,
            "heldout_measures": [t + 1 for t in held], "final_losses": history[-1], "curve": curve}


def select_steps(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_step: dict[int, list[float]] = defaultdict(list)
    for unit in units:
        for point in unit["curve"]:
            by_step[point["step"]].append(point["heldout_elbo"])
    means = {step: float(np.mean(v)) for step, v in sorted(by_step.items())}
    return {"selected": min(means, key=means.__getitem__), "heldout_elbo_by_step": means}


def full_unit(
    setup: Mapping[str, Any], arm_name: str, seed: int, steps: int, seeds: Sequence[int],
    runtime: RuntimeDevice, checkpoint: Path,
) -> dict[str, Any]:
    config = setup["config"]
    monitor = config["monitor"]
    log: list[dict[str, Any]] = []
    if checkpoint.exists():
        # The run stopped after training but before this unit's JSON was written (e.g. during the
        # 60-piece generation): load the saved model instead of training again or overwriting it.
        saved = torch.load(checkpoint, map_location=runtime.device, weights_only=False)
        expected = {"protocol": PROTOCOL, "arm": arm_name, "seed": seed, "steps": steps, "config": config}
        if any(saved.get(key) != value for key, value in expected.items()) or "history" not in saved:
            raise FileExistsError(f"다른 설정으로 만든 Checkpoint가 있습니다. 덮어쓰지 않고 멈춥니다: {checkpoint}")
        model = GridDenoiser(setup["layout"], **config["model"]).to(runtime.device)
        model.load_state_dict(saved["state_dict"])
        model.eval()
        tables = Tables(setup["typology"], setup["layout"], runtime.device)
        return _full_result(model, tables, setup, arm_name, seed, steps, seeds, runtime, checkpoint,
                            saved["history"], saved["monitor"])

    def watch(step: int) -> Callable[[GridDenoiser, Tables], None]:
        def call(model: GridDenoiser, tables: Tables) -> None:
            rows = []
            for k in range(int(monitor["pieces"])):
                phrases, walk, first_pass = generate_piece(model, tables, setup, 10_000 + k, attempts=0)
                rows.append(piece_metrics(phrases, walk, setup, first_pass))
            log.append({"step": step, **{key: float(np.mean([r[key] for r in rows])) for key in (
                "first_pass_membership", "surface_edge_validity", "surface_graph_tv", "original_phrase_rate")}})
        return call

    points = {s: watch(s) for s in monitor_steps(steps, int(monitor["points"]))}
    everything = list(range(len(setup["typology"].labels)))
    model, tables, history = train_model(setup, config["arms"][arm_name], everything, seed, steps, runtime, points)
    torch.save({"protocol": PROTOCOL, "arm": arm_name, "seed": seed, "steps": steps, "config": config,
                "history": history, "monitor": log,
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}}, checkpoint)
    return _full_result(model, tables, setup, arm_name, seed, steps, seeds, runtime, checkpoint, history, log)


def _full_result(
    model: GridDenoiser, tables: Tables, setup: Mapping[str, Any], arm_name: str, seed: int, steps: int,
    seeds: Sequence[int], runtime: RuntimeDevice, checkpoint: Path,
    history: list[dict[str, float]], log: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = to_device(setup["rows"], runtime.device)
    return {
        "arm": arm_name, "seed": seed, "steps": steps, "checkpoint": checkpoint.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "history": history, "monitor": log,
        "recall": {"with_context": full_mask_recall(model, tables, rows, null=False),
                   "null_context": full_mask_recall(model, tables, rows, null=True)},
        "generation": generation_table(model, tables, setup, seeds),
    }


def copytest_unit(
    setup: Mapping[str, Any], arm_name: str, seed: int, fold: int, steps: int, count: int, runtime: RuntimeDevice
) -> dict[str, Any]:
    """Retrain the fold model to the selected length and generate pieces for the data-copying test.

    The seed, data and settings are those of the held-out run, but on a GPU the retrained model
    need not be bit-identical to that run at the same step (non-deterministic kernels).
    """

    config, typology = setup["config"], setup["typology"]
    held = heldout_folds(typology, config["heldout"]["folds"], config["heldout"]["fold_seed"])[fold]
    train = [t for t in range(len(typology.labels)) if t not in set(held)]
    model, tables, _ = train_model(setup, config["arms"][arm_name], train, seed, steps, runtime)
    first = int(config["evaluation"]["copy_test"]["first_seed"])
    pieces = []
    for sample_seed in range(first, first + count):
        phrases, walk, first_pass = generate_piece(model, tables, setup, sample_seed)
        pieces.append({"seed": sample_seed, "walk": walk, "first_pass": first_pass,
                       "phrases": [[[e[0], float(e[1])] for e in p] for p in phrases]})
    return {"arm": arm_name, "seed": seed, "fold": fold, "steps": steps,
            "heldout_measures": [t + 1 for t in held], "pieces": pieces}


# ----------------------------------------------------------------------------------------------
# Whole run


def run_experiment(
    config_path: Path, project_root: Path, run_tag: str, runtime: RuntimeDevice,
    *, run_mode: str = "full", log: Callable[[str], None] = print,
) -> dict[str, Any]:
    if run_mode not in {"smoke", "full"}:
        raise ValueError("run_mode must be 'smoke' or 'full'")
    check_loaded_source()
    setup = load_setup(config_path, project_root)
    config = setup["config"]
    smoke = run_mode == "smoke"
    steps = int(config["training"]["smoke_steps" if smoke else "steps"])
    seeds = [int(s) for s in config["training"]["seeds"]][: 1 if smoke else None]
    folds = range(1 if smoke else int(config["heldout"]["folds"]))
    first, pieces = int(config["sampling"]["first_seed"]), 2 if smoke else int(config["sampling"]["pieces"])
    sample_seeds = list(range(first, first + pieces))
    copy_pieces = 2 if smoke else int(config["evaluation"]["copy_test"]["pieces"])
    run_dir = project_root / config["artifacts"]["run_dir"] / (run_tag + ("_smoke" if smoke else ""))
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {"protocol": PROTOCOL, "run_tag": run_tag, "run_mode": run_mode, "commit": _commit(project_root),
            "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
            "piece_config_sha256": hashlib.sha256((project_root / config["piece_config"]).read_bytes()).hexdigest(),
            "source_sha256": source_digest(),
            "measures_sha256": setup["measures_sha256"], "runtime": runtime.as_dict(),
            "gpu": torch.cuda.get_device_name(0) if runtime.backend == "cuda" else None,
            "autocast": False,  # v1 trains in float32; runtime.amp_enabled is not used here
            "torch": torch.__version__, "python": platform.python_version()}
    previous = run_dir / "meta.json"
    if previous.exists():
        old = json.loads(previous.read_text(encoding="utf-8"))
        for key in ("protocol", "commit", "config_sha256", "piece_config_sha256", "source_sha256",
                    "measures_sha256", "run_mode"):
            if old.get(key) != meta[key]:
                raise RuntimeError(f"RUN_TAG {run_tag!r}는 다른 {key}로 시작한 실행입니다. 새 RUN_TAG를 쓰세요.")
        if old.get("gpu") != meta["gpu"]:
            log(f"주의: 처음 실행한 GPU({old.get('gpu')})와 지금 GPU({meta['gpu']})가 다릅니다. 이어서 실행하지만 기록해 두세요.")
    else:
        previous.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    report: dict[str, Any] = {"meta": meta, "typology": typology_report(setup["typology"]),
                              "address_audit": address_audit(setup["typology"]),
                              "returns": [[s + 1, t + 1] for s, t in setup["returns"]],
                              "selected_steps": {}, "heldout": [], "full": [], "copytest": [],
                              "heldout_baseline": {}}
    for fold in folds:
        held = heldout_folds(setup["typology"], config["heldout"]["folds"], config["heldout"]["fold_seed"])[fold]
        report["heldout_baseline"][str(fold)] = positional_baseline(setup, held)
    started = time.time()
    for arm in config["arms"]:
        units = [
            _unit(run_dir / f"heldout_{arm}_s{seed}_f{fold}.json",
                  lambda arm=arm, seed=seed, fold=fold: heldout_unit(setup, arm, seed, fold, steps, runtime))
            for seed in seeds for fold in folds
        ]
        report["heldout"].extend(units)
        choice = select_steps(units)
        choice["graph_active_steps"] = graph_active_steps(config, arm, choice["selected"])
        report["selected_steps"][arm] = choice
        log(f"{arm}: 보류 평가 완료, 학습 길이 {choice['selected']} step ({(time.time() - started) / 60:.1f}분 경과)")
        for seed in seeds:
            name = f"full_{arm}_s{seed}"
            report["full"].append(_unit(
                run_dir / f"{name}.json",
                lambda arm=arm, seed=seed, name=name, length=choice["selected"]: full_unit(
                    setup, arm, seed, length, sample_seeds, runtime, run_dir / f"{name}.pt"),
            ))
            log(f"{arm} · seed {seed}: 전체 학습·생성 완료 ({(time.time() - started) / 60:.1f}분 경과)")
        for seed in seeds:
            for fold in folds:
                name = f"copytest_{arm}_s{seed}_f{fold}"
                report["copytest"].append(_unit(
                    run_dir / f"{name}.json",
                    lambda arm=arm, seed=seed, fold=fold, length=choice["selected"]: copytest_unit(
                        setup, arm, seed, fold, length, copy_pieces, runtime),
                ))
        log(f"{arm}: 복제 검정용 보류 모형 생성 완료 ({(time.time() - started) / 60:.1f}분 경과)")
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    report["evaluation"] = _unit(run_dir / "evaluation.json", lambda: evaluate_report(report, setup))
    log(f"평가 완료 ({(time.time() - started) / 60:.1f}분 경과)")
    return report


def build_reference(setup: Mapping[str, Any]) -> Reference:
    typology: Typology = setup["typology"]
    return Reference(
        typology.phrases, typology.chapters, typology.medoids, setup["edges"], setup["returns"],
        quantum=1.0 / setup["grid"].cells_per_beat, knn_k=int(setup["config"]["evaluation"]["knn_k"]),
    )


def evaluate_report(report: Mapping[str, Any], setup: Mapping[str, Any]) -> dict[str, Any]:
    """Scores of every full model and the data-copying tests, plus the fixed criteria."""

    settings = setup["config"]["evaluation"]
    ref = build_reference(setup)
    typology: Typology = setup["typology"]
    models = []
    for unit in report["full"]:
        pieces = [[as_phrase(p) for p in piece["phrases"]] for piece in unit["generation"]["pieces"]]
        scores = model_scores(pieces, ref, bootstrap=int(settings["bootstrap"]),
                              seed=int(settings["bootstrap_seed"]), pair_sample=int(settings["pair_sample"]))
        first_pass = [row["first_pass_membership"] for row in unit["generation"]["per_piece"]]
        scores["per_piece"]["first_pass_membership"] = {
            "mean": float(np.mean(first_pass)),
            "ci95": bootstrap_ci(first_pass, int(settings["bootstrap"]), int(settings["bootstrap_seed"]))}
        models.append({"arm": unit["arm"], "seed": unit["seed"], **scores})
    tests = []
    for unit in report.get("copytest", []):
        held = {m - 1 for m in unit["heldout_measures"]}
        train = [typology.phrases[t] for t in range(len(typology.phrases)) if t not in held]
        test = [typology.phrases[t] for t in sorted(held)]
        generated = [as_phrase(piece["phrases"][t]) for piece in unit["pieces"] for t in ref.free]
        tests.append({"arm": unit["arm"], "seed": unit["seed"], "fold": unit["fold"],
                      **copy_test(generated, train, test)})
    combined: dict[str, dict[int, float]] = defaultdict(dict)
    combined_novel: dict[str, dict[int, float]] = defaultdict(dict)
    for (arm, seed), zs in _group(tests, "z").items():
        combined[arm][seed] = float(np.sum(zs) / np.sqrt(len(zs)))  # Stouffer over folds
    for (arm, seed), zs in _group(tests, "z_novel").items():
        combined_novel[arm][seed] = float(np.sum(zs) / np.sqrt(len(zs)))
    first_pass = defaultdict(dict)
    for model in models:
        first_pass[model["arm"]][model["seed"]] = model["per_piece"]["first_pass_membership"]["mean"]
    arms = list(setup["config"]["arms"])
    main = settings["main_arm"]
    active = graph_active_steps(setup["config"], main, int(report["selected_steps"][main]["selected"]))
    criteria = judge(first_pass, combined_novel, main_arm=main, ablation_arm=settings["ablation_arm"],
                     graph_active_steps=active)
    return {"protocol": EVAL_PROTOCOL, "arms": arms,
            "reference": {"distinct_phrases": len(ref.distinct), "barcode_bars": int(len(ref.barcode)),
                          "knn_k": ref.knn_k, "nn1_mean": float(ref.nn1.mean()),
                          "repeat_rate": ref.repeat_rate, "repeat_rate_free": ref.repeat_rate_free},
            "models": models, "copy_tests": tests,
            "copy_z_combined": {arm: dict(v) for arm, v in combined.items()},
            "copy_z_novel_combined": {arm: dict(v) for arm, v in combined_novel.items()}, "criteria": criteria}


def _group(tests: Sequence[Mapping[str, Any]], key: str) -> dict[tuple[str, int], list[float]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for test in tests:
        grouped[(test["arm"], test["seed"])].append(test[key])
    return grouped


# ----------------------------------------------------------------------------------------------
# Tables for the notebook


def heldout_baseline(report: Mapping[str, Any]) -> float:
    """Mean context-free held-out ELBO over the folds (NaN for runs made before it was recorded)."""

    values = list(report.get("heldout_baseline", {}).values())
    return float(np.mean(values)) if values else float("nan")


def curve_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for unit in report["heldout"]:
        for point in unit["curve"]:
            for key in ("heldout_elbo", "heldout_elbo_null_context", "train_elbo"):
                grouped[(unit["arm"], point["step"])][key].append(point[key])
    return [{"arm": arm, "step": step, **{k: round(float(np.mean(v)), 4) for k, v in values.items()}}
            for (arm, step), values in sorted(grouped.items())]


def monitor_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for unit in report["full"]:
        for point in unit["monitor"]:
            for key, value in point.items():
                if key != "step":
                    grouped[(unit["arm"], point["step"])][key].append(value)
    return [{"arm": arm, "step": step, **{k: round(float(np.mean(v)), 3) for k, v in values.items()}}
            for (arm, step), values in sorted(grouped.items())]


GENERATION_COLUMNS = (
    "first_pass_membership", "membership", "surface_edge_validity", "surface_graph_tv",
    "walk_positions_differing", "original_phrase_rate", "unseen_note_transition_rate",
    "note_4gram_novelty", "interval_jsd", "duration_jsd", "returns_exact",
)


def generation_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One row per arm: mean over training seeds of the per-model means, with the seed range."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in report["full"]:
        grouped[unit["arm"]].append(unit["generation"]["metrics"])
    rows = []
    for arm, tables in grouped.items():
        row: dict[str, Any] = {"arm": arm, "models": len(tables)}
        for key in GENERATION_COLUMNS:
            values = [t[key]["mean"] for t in tables]
            row[key] = round(float(np.mean(values)), 3)
            row[f"{key}_range"] = f"{min(values):.3f}–{max(values):.3f}"
        rows.append(row)
    return rows


def recall_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for unit in report["full"]:
        for key, value in unit["recall"].items():
            grouped[unit["arm"]][key].append(value)
    return [{"arm": arm, **{k: round(float(np.mean(v)), 3) for k, v in values.items()}} for arm, values in grouped.items()]


def export_musicxml(unit: Mapping[str, Any], setup: Mapping[str, Any], out_dir: Path, count: int = 3) -> list[Path]:
    """Write the first ``count`` generated pieces of a full unit as MusicXML (never overwrites)."""

    from gugakgen.data.musicxml import write_musicxml
    from gugakgen.data.records import EventRecord, MeasureRecord, Meter, ScoreRecord

    typology: Typology = setup["typology"]
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for piece in unit["generation"]["pieces"][:count]:
        path = out_dir / f"{unit['arm']}_s{unit['seed']}_piece{piece['seed']}.musicxml"
        if path.exists():
            raise FileExistsError(f"이미 있습니다: {path}")
        measures = [
            MeasureRecord(
                number=t + 1,
                chapter=typology.chapters[t],
                meter=Meter(round(sum(d for _, d in phrase)), 4),
                events=tuple(EventRecord(p, d) for p, d in phrase),
                family="abcdefghi"[piece["walk"][t]],
            )
            for t, phrase in enumerate(piece["phrases"])
        ]
        score = ScoreRecord("generated", "Piri", tuple(measures), {"walk": piece["walk"]})
        written.append(write_musicxml(score, path, part_name="Piri"))
    return written


# ----------------------------------------------------------------------------------------------
# Evaluation tables for the notebook

AXES = {
    "geometry": ("phrase_graph_tv", "pooled_phrase_graph_tv", "surface_edge_validity", "first_pass_membership",
                 "ph_bottleneck", "ph_wasserstein", "ph_cycles"),
    "statistics": ("interval_jsd", "duration_jsd", "pitch_jsd", "note_3gram_jsd", "note_4gram_jsd",
                   "oa_intra_original_vs_inter", "oa_intra_original_vs_intra_generated"),
    "music": ("pitch_outside_scale", "beat_fill_error", "offgrid_onset", "chapter_cadence_ok", "final_pitch_ok"),
    "novelty": ("exact_copy_rate", "authenticity", "density", "coverage", "nn_distance_mean",
                "distinct_measure_rate", "intra_generated_distance", "longest_shared_run",
                "longest_shared_run_share"),
}


def _value(model: Mapping[str, Any], key: str) -> float:
    if key in model["per_piece"]:
        return float(model["per_piece"][key]["mean"])
    return float(model["set"][key]["value"])


def evaluation_rows(evaluation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One row per metric: mean over training seeds for each arm, with the seed range."""

    by_arm: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for model in evaluation["models"]:
        by_arm[model["arm"]].append(model)
    rows = []
    for axis, keys in AXES.items():
        for key in keys:
            row: dict[str, Any] = {"axis": axis, "metric": key}
            for arm, models in by_arm.items():
                values = [_value(m, key) for m in models]
                row[arm] = round(float(np.mean(values)), 3)
                row[f"{arm} (seeds)"] = f"{min(values):.3f}–{max(values):.3f}"
            rows.append(row)
    return rows


def criteria_rows(evaluation: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"criterion": key, "status": value["status"], "rule": value["rule"],
             "detail": {k: v for k, v in value.items() if k not in {"status", "rule"}} or ""}
            for key, value in evaluation["criteria"].items()]


def copytest_rows(evaluation: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in test.items()}
            for test in evaluation["copy_tests"]]
    novel = evaluation.get("copy_z_novel_combined", {})
    for arm, by_seed in evaluation["copy_z_combined"].items():
        for seed, z in by_seed.items():
            row = {"arm": arm, "seed": seed, "fold": "combined", "z": round(float(z), 3)}
            if arm in novel:
                row["z_novel"] = round(float(novel[arm][seed]), 3)
            rows.append(row)
    return rows
