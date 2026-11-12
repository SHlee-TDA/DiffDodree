"""Chart-model experiment: held-out prediction, address audit and generation (setting (ga)).

Every unit of work (one training run, one generation table) writes its own JSON
under ``artifacts/chart_runs/<RUN_TAG>/``. An existing unit is read back instead
of recomputed, so an interrupted Colab session resumes with the same RUN_TAG,
and nothing from another RUN_TAG is ever overwritten.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gugakgen.chart.coords import Scale
from gugakgen.chart.metrics import (
    baseline_edit,
    baseline_recombine,
    evaluate_piece,
    original_own_medoid_distance,
    summarise,
)
from gugakgen.chart.model import (
    ChartDenoiser,
    Layout,
    LossWeights,
    encode_piece,
    hide,
    losses,
    margin_tables,
    pitch_targets,
    random_spans,
)
from gugakgen.chart.sampling import (
    ChainConfig,
    EmpiricalPredictor,
    ModelPredictor,
    Predictor,
    run_chain,
)
from gugakgen.chart.typology import Typology, build_typology, load_phrases
from gugakgen.config import load_yaml
from gugakgen.device import RuntimeDevice
from gugakgen.reproducibility import set_global_seed

PROTOCOL = "chart_v0_2026-11-11"


# ----------------------------------------------------------------------------------------------
# Setup


def load_setup(config_path: Path, project_root: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    piece = load_yaml(project_root / config["piece_config"])
    measures_path = project_root / piece["data"]["canonical_measures"]
    phrases, chapters = load_phrases(measures_path)
    typology = build_typology(phrases, chapters, k=int(config["typology"]["k"]))
    layout = Layout.from_typology(typology)
    returns: list[tuple[int, int]] = []
    for relation in piece["structure"]["hwanip_relations"]:
        (s0, s1), (t0, t1) = relation["source_measures"], relation["target_measures"]
        if s1 - s0 != t1 - t0:
            raise ValueError("Only one-to-one return relations are supported")
        returns.extend((s0 - 1 + i, t0 - 1 + i) for i in range(s1 - s0 + 1))
    for source, target in returns:
        if typology.labels[source] != typology.labels[target]:
            raise ValueError(f"Return pair {(source + 1, target + 1)} has two different types (eq. 5)")
    return {
        "config": config,
        "typology": typology,
        "layout": layout,
        "returns": tuple(returns),
        "encoded": encode_piece(typology.phrases, typology, layout),
        "margin": margin_tables(typology, layout),
        "measures_sha256": hashlib.sha256(measures_path.read_bytes()).hexdigest(),
    }


def typology_report(typology: Typology) -> dict[str, Any]:
    sizes = Counter(typology.names)
    return {
        "k": typology.k,
        "sequence": typology.sequence_by_chapter(),
        "medoid_measures": [index + 1 for index in typology.medoid_phrases],
        "type_sizes": {name: sizes[name] for name in sorted(sizes)},
        "silhouette": typology.silhouette,
        "transition_edges": int((typology.transition_counts() > 0).sum()),
        "contact": typology.contact_summary(),
        "original_own_medoid_distance": original_own_medoid_distance(typology),
    }


def address_audit(typology: Typology) -> dict[str, int]:
    """How often the context of the draft's eq. 8 identifies one measure of the original."""

    labels, phrases = typology.labels, typology.phrases
    count = len(labels)
    totals = [sum(d for _, d in phrase) for phrase in phrases]
    kappa = [
        (labels[t - 1] if t else "start", labels[t], labels[t + 1] if t < count - 1 else "end", totals[t])
        for t in range(count)
    ]
    with_notes = [
        (*kappa[t], phrases[t - 1][-1] if t else None, phrases[t + 1][0] if t < count - 1 else None)
        for t in range(count)
    ]
    contents = defaultdict(set)
    for key, phrase in zip(kappa, phrases, strict=True):
        contents[key].add(phrase)
    kappa_counts, note_counts = Counter(kappa), Counter(with_notes)
    return {
        "measures": count,
        "kappa_unique_position": sum(kappa_counts[key] == 1 for key in kappa),
        "kappa_single_content": sum(len(contents[key]) == 1 for key in kappa),
        "kappa_and_boundary_notes_unique_position": sum(note_counts[key] == 1 for key in with_notes),
    }


def heldout_folds(typology: Typology, folds: int, seed: int) -> list[list[int]]:
    """Folds of phrase indices grouped by identical content; medoid phrases are never held out."""

    groups: dict[tuple, list[int]] = defaultdict(list)
    for t, phrase in enumerate(typology.phrases):
        groups[phrase].append(t)
    medoids = {typology.phrases[index] for index in typology.medoid_phrases}
    candidates = [rows for phrase, rows in groups.items() if phrase not in medoids]
    order = np.random.default_rng(seed).permutation(len(candidates))
    return [sorted(t for i in order[fold::folds] for t in candidates[i]) for fold in range(folds)]


# ----------------------------------------------------------------------------------------------
# Training and prediction scores


def train_model(
    setup: Mapping[str, Any],
    arm: Mapping[str, float],
    rows: Sequence[int],
    seed: int,
    steps: int,
    runtime: RuntimeDevice,
    checkpoints: Mapping[int, Callable[[ChartDenoiser], None]] | None = None,
) -> tuple[ChartDenoiser, list[dict[str, float]]]:
    """Train on the listed phrases. ``checkpoints[step]`` is called with the model in eval mode."""

    config, layout = setup["config"], setup["layout"]
    training = config["training"]
    set_global_seed(seed)
    device = runtime.device
    model = ChartDenoiser(layout, **config["model"]).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    data = {key: torch.as_tensor(value[list(rows)], device=device) for key, value in setup["encoded"].items()}
    cost = torch.as_tensor(setup["margin"][0][list(rows)], device=device)
    const = torch.as_tensor(setup["margin"][1][list(rows)], device=device)
    weights = LossWeights(
        margin=float(arm["lambda_margin"]),
        beats=float(arm["lambda_beats"]),
        margin_size=float(config["loss"]["margin_size"]),
        softmin_temperature=float(config["loss"]["softmin_temperature"]),
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    history: list[dict[str, float]] = []
    model.train()
    for step in range(1, steps + 1):
        pick = torch.randint(len(rows), (training["batch_size"],), generator=generator, device=device)
        batch = {key: value[pick] for key, value in data.items()}
        hidden = random_spans(
            batch["length"], layout.max_events, generator, training["span_min"], training["span_max"]
        )
        uniform = torch.rand(2, len(pick), generator=generator, device=device)
        inputs = hide(
            batch,
            hidden,
            layout,
            null_context=uniform[0] < arm["context_dropout"],
            null_type=uniform[1] < arm["type_dropout"],
        )
        log_pitch, log_dur = model(inputs)
        terms = losses(log_pitch, log_dur, batch, hidden, layout, weights, cost[pick], const[pick])
        rate = training["learning_rate"] * min(1.0, step / max(training["warmup_steps"], 1))
        for group in optimiser.param_groups:
            group["lr"] = rate
        optimiser.zero_grad(set_to_none=True)
        terms["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"])
        optimiser.step()
        if step % max(steps // 10, 1) == 0 or step == steps:
            history.append({"step": step, **{k: float(v.detach()) for k, v in terms.items()}})
        if checkpoints and step in checkpoints:
            checkpoints[step](model.eval())
            model.train()
    return model.eval(), history


def prediction_scores(
    predictor: Predictor, encoded: Mapping[str, np.ndarray], rows: Sequence[int], layout: Layout
) -> dict[str, float]:
    """Scores over every contiguous span of every listed phrase (no sampling noise).

    ``*_nll``: mean negative log-probability (nats) of a hidden event's pitch and
    duration. ``full_mask_recall``: share of phrases reproduced exactly by the
    argmax when all their events are hidden.
    """

    if not rows:
        return {}
    cases: list[tuple[int, int, int]] = [
        (row, start, start + span)
        for row in rows
        for span in range(1, int(encoded["length"][row]) + 1)
        for start in range(int(encoded["length"][row]) - span + 1)
    ]
    index = np.array([row for row, _, _ in cases])
    batch = {key: value[index] for key, value in encoded.items()}
    hidden = np.zeros((len(cases), layout.max_events), dtype=bool)
    for i, (_, start, stop) in enumerate(cases):
        hidden[i, start:stop] = True
    log_pitch, log_dur = predictor(batch, hidden)
    tensors = {key: torch.as_tensor(value) for key, value in batch.items()}
    pitch = pitch_targets(tensors, layout).numpy()
    duration = batch["dur"].clip(max=len(layout.durations) - 1)
    pitch_ll = np.take_along_axis(log_pitch, pitch[..., None], -1).squeeze(-1)
    dur_ll = np.take_along_axis(log_dur, duration[..., None], -1).squeeze(-1)
    pitch_hit = log_pitch.argmax(-1) == pitch
    dur_hit = log_dur.argmax(-1) == duration
    full = np.array([stop - start == int(encoded["length"][row]) for row, start, stop in cases])
    recalled = ((pitch_hit & dur_hit) | ~hidden).all(axis=1)[full]
    # weight phrases equally: average within a phrase first
    per_row: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for i, (row, _, _) in enumerate(cases):
        mask = hidden[i]
        per_row[row].append(
            (float(-pitch_ll[i][mask].mean()), float(-dur_ll[i][mask].mean()), float(pitch_hit[i][mask].mean()))
        )
    means = np.array([np.mean(values, axis=0) for values in per_row.values()])
    return {
        "pitch_nll": float(means[:, 0].mean()),
        "duration_nll": float(means[:, 1].mean()),
        "pitch_top1": float(means[:, 2].mean()),
        "full_mask_recall": float(recalled.mean()),
        "phrases": len(per_row),
    }


# ----------------------------------------------------------------------------------------------
# Units of work


def _unit(path: Path, compute: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    result = compute()
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    temporary.rename(path)
    return result


def eval_steps(config: Mapping[str, Any], steps: int) -> list[int]:
    return sorted({int(s) for s in config["heldout"]["eval_steps"] if int(s) < steps} | {steps})


def select_steps(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Training length with the lowest held-out NLL (pitch + duration, context given),
    averaged over the folds and training seeds of one arm."""

    by_step: dict[int, list[float]] = defaultdict(list)
    for unit in units:
        for point in unit["curve"]:
            scores = point["with_context"]
            by_step[point["step"]].append(scores["pitch_nll"] + scores["duration_nll"])
    means = {step: float(np.mean(values)) for step, values in sorted(by_step.items())}
    return {"selected": min(means, key=means.__getitem__), "heldout_nll_by_step": means}


def heldout_unit(
    setup: Mapping[str, Any], arm_name: str, seed: int, fold: int, steps: int, runtime: RuntimeDevice
) -> dict[str, Any]:
    config, typology, layout, encoded = (setup[k] for k in ("config", "typology", "layout", "encoded"))
    folds = heldout_folds(typology, config["heldout"]["folds"], config["heldout"]["fold_seed"])
    held = folds[fold]
    train = [t for t in range(len(typology.labels)) if t not in set(held)]
    curve: list[dict[str, Any]] = []

    def record(step: int) -> Callable[[ChartDenoiser], None]:
        def call(model: ChartDenoiser) -> None:
            scores = {
                "with_context": prediction_scores(ModelPredictor(model, runtime.device), encoded, held, layout),
                "null_context": prediction_scores(
                    ModelPredictor(model, runtime.device, null_context=True), encoded, held, layout
                ),
                "train": prediction_scores(ModelPredictor(model, runtime.device), encoded, train, layout),
            }
            curve.append({"step": step, **scores})

        return call

    points = {int(step): record(int(step)) for step in eval_steps(config, steps)}
    model, history = train_model(setup, config["arms"][arm_name], train, seed, steps, runtime, points)
    predictors: dict[str, Predictor] = {
        "model": ModelPredictor(model, runtime.device),
        "model_null_context": ModelPredictor(model, runtime.device, null_context=True),
        "empirical_by_type": EmpiricalPredictor(encoded, layout, rows=train),
        "empirical_global": EmpiricalPredictor(encoded, layout, rows=train, by_type=False),
    }
    return {
        "arm": arm_name,
        "seed": seed,
        "fold": fold,
        "steps": steps,
        "heldout_measures": [t + 1 for t in held],
        "final_losses": history[-1],
        "curve": curve,
        "heldout": {name: prediction_scores(p, encoded, held, layout) for name, p in predictors.items()},
        "train": {
            name: prediction_scores(predictors[name], encoded, train, layout)
            for name in ("model", "model_null_context")
        },
    }


def _generate_rows(
    setup: Mapping[str, Any], make: Callable[[np.random.Generator], tuple[list, dict[str, float]]], pieces: int
) -> dict[str, Any]:
    first = int(setup["config"]["sampling"]["first_seed"])
    rows, chain = [], defaultdict(list)
    for seed in range(first, first + pieces):
        phrases, stats = make(np.random.default_rng(seed))
        rows.append(evaluate_piece(phrases, setup["typology"], setup["returns"]))
        for key, value in stats.items():
            chain[key].append(value)
    return {
        "pieces": pieces,
        "metrics": summarise(rows),
        "chain": {key: float(np.mean(values)) for key, values in chain.items()},
        "per_piece": rows,
    }


def chain_settings(setup: Mapping[str, Any]) -> list[tuple[str, ChainConfig]]:
    sampling = setup["config"]["sampling"]
    base = {
        "steps": int(sampling["steps"]),
        "temperature": float(sampling["temperature"]),
        "initial": sampling["initial"],
        "returns": setup["returns"],
    }
    settings = [(f"rho{rho}", ChainConfig(rho=float(rho), **base)) for rho in sampling["rhos"]]
    settings.append(
        (
            f"rho{sampling['free_rho']}_no_rejection",
            ChainConfig(rho=float(sampling["free_rho"]), enforce_membership=False, **base),
        )
    )
    return settings


def chain_tables(setup: Mapping[str, Any], predictor: Predictor, pieces: int) -> dict[str, Any]:
    def table(chain: ChainConfig) -> dict[str, Any]:
        def make(rng: np.random.Generator) -> tuple[list, dict[str, float]]:
            phrases, stats = run_chain(setup["typology"], setup["layout"], predictor, chain, rng)
            return phrases, stats.as_dict()

        return _generate_rows(setup, make, pieces)

    return {name: table(chain) for name, chain in chain_settings(setup)}


def full_unit(
    setup: Mapping[str, Any],
    arm_name: str,
    seed: int,
    steps: int,
    pieces: int,
    runtime: RuntimeDevice,
    checkpoint: Path,
) -> dict[str, Any]:
    if checkpoint.exists():
        raise FileExistsError(f"Checkpoint가 이미 있습니다: {checkpoint}")
    config, typology, layout, encoded = (setup[k] for k in ("config", "typology", "layout", "encoded"))
    everything = list(range(len(typology.labels)))
    model, history = train_model(setup, config["arms"][arm_name], everything, seed, steps, runtime)
    torch.save(
        {
            "protocol": PROTOCOL,
            "arm": arm_name,
            "seed": seed,
            "steps": steps,
            "config": config,
            "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        },
        checkpoint,
    )
    return {
        "arm": arm_name,
        "seed": seed,
        "steps": steps,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "history": history,
        # the October probe in chart form: are original phrases reproduced from a fully hidden input?
        "train_recall": {
            "with_context": prediction_scores(ModelPredictor(model, runtime.device), encoded, everything, layout),
            "null_context": prediction_scores(
                ModelPredictor(model, runtime.device, null_context=True), encoded, everything, layout
            ),
            "null_context_and_type": prediction_scores(
                ModelPredictor(model, runtime.device, null_context=True, null_type=True),
                encoded,
                everything,
                layout,
            ),
        },
        "generation": chain_tables(setup, ModelPredictor(model, runtime.device), pieces),
    }


def baseline_unit(setup: Mapping[str, Any], pieces: int) -> dict[str, Any]:
    typology, layout, returns = setup["typology"], setup["layout"], setup["returns"]
    scale = Scale.from_phrases(typology.phrases)
    out: dict[str, Any] = {
        "T0_recombine": _generate_rows(
            setup, lambda rng: (baseline_recombine(typology, returns, rng), {}), pieces
        )
    }
    for rate in setup["config"]["sampling"]["edit_baseline_rates"]:
        out[f"T2_edit_{rate}"] = _generate_rows(
            setup,
            lambda rng, rate=rate: (baseline_edit(typology, returns, rng, scale, float(rate)), {}),
            pieces,
        )
    empirical = EmpiricalPredictor(setup["encoded"], layout)
    for name, table in chain_tables(setup, empirical, pieces).items():
        out[f"chart_empirical_{name}"] = table
    return out


# ----------------------------------------------------------------------------------------------
# Whole run


def _commit(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_experiment(
    config_path: Path,
    project_root: Path,
    run_tag: str,
    runtime: RuntimeDevice,
    *,
    run_mode: str = "full",
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    if run_mode not in {"smoke", "full"}:
        raise ValueError("run_mode must be 'smoke' or 'full'")
    setup = load_setup(config_path, project_root)
    config = setup["config"]
    smoke = run_mode == "smoke"
    steps = int(config["training"]["smoke_steps" if smoke else "steps"])
    seeds = [int(s) for s in config["training"]["seeds"]][: 1 if smoke else None]
    folds = range(2 if smoke else int(config["heldout"]["folds"]))
    pieces = 2 if smoke else int(config["sampling"]["pieces"])
    run_dir = project_root / config["artifacts"]["run_dir"] / (run_tag + ("_smoke" if smoke else ""))
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "protocol": PROTOCOL,
        "run_tag": run_tag,
        "run_mode": run_mode,
        "commit": _commit(project_root),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "measures_sha256": setup["measures_sha256"],
        "runtime": runtime.as_dict(),
        "torch": torch.__version__,
        "python": platform.python_version(),
    }
    previous = run_dir / "meta.json"
    if previous.exists():
        old = json.loads(previous.read_text(encoding="utf-8"))
        for key in ("commit", "config_sha256", "measures_sha256", "run_mode"):
            if old[key] != meta[key]:
                raise RuntimeError(
                    f"RUN_TAG {run_tag!r}는 다른 {key}로 시작한 실행입니다. 새 RUN_TAG를 쓰세요."
                )
    else:
        previous.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    report: dict[str, Any] = {
        "selected_steps": {},
        "meta": meta,
        "typology": typology_report(setup["typology"]),
        "address_audit": address_audit(setup["typology"]),
        "returns": [[s + 1, t + 1] for s, t in setup["returns"]],
        "heldout": [],
        "full": [],
    }
    started = time.time()
    for arm in config["arms"]:
        units = [
            _unit(
                run_dir / f"heldout_{arm}_s{seed}_f{fold}.json",
                lambda arm=arm, seed=seed, fold=fold: heldout_unit(
                    setup, arm, seed, fold, steps, runtime
                ),
            )
            for seed in seeds
            for fold in folds
        ]
        report["heldout"].extend(units)
        choice = select_steps(units)
        report["selected_steps"][arm] = choice
        log(
            f"{arm}: 보류 평가 완료, 보류 손실이 가장 낮은 학습 길이 {choice['selected']} step "
            f"({(time.time() - started) / 60:.1f}분 경과)"
        )
        for seed in seeds:
            name = f"full_{arm}_s{seed}"
            report["full"].append(
                _unit(
                    run_dir / f"{name}.json",
                    lambda arm=arm, seed=seed, name=name, length=choice["selected"]: full_unit(
                        setup, arm, seed, length, pieces, runtime, run_dir / f"{name}.pt"
                    ),
                )
            )
            log(f"{arm} · seed {seed}: 전체 학습·생성 완료 ({(time.time() - started) / 60:.1f}분 경과)")
    report["baselines"] = _unit(run_dir / "baselines.json", lambda: baseline_unit(setup, pieces))
    log(f"기준선: 완료 ({(time.time() - started) / 60:.1f}분 경과)")
    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return report


# ----------------------------------------------------------------------------------------------
# Tables for the notebook


def curve_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Held-out and training NLL (pitch + duration) by training length, averaged over folds and seeds."""

    grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for unit in report["heldout"]:
        for point in unit["curve"]:
            for key in ("with_context", "null_context", "train"):
                grouped[(unit["arm"], point["step"])][key].append(
                    point[key]["pitch_nll"] + point[key]["duration_nll"]
                )
            grouped[(unit["arm"], point["step"])]["train_recall"].append(
                point["train"]["full_mask_recall"]
            )
    return [
        {"arm": arm, "step": step, **{key: round(float(np.mean(v)), 3) for key, v in values.items()}}
        for (arm, step), values in sorted(grouped.items())
    ]


def heldout_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One row per (arm, predictor) at the selected training length: mean over folds within
    a training seed, then mean and range over seeds. Learning-free predictors appear once."""

    grouped: dict[tuple[str, str], dict[int, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for unit in report["heldout"]:
        selected = report["selected_steps"][unit["arm"]]["selected"]
        point = next(p for p in unit["curve"] if p["step"] == selected)
        entries = {
            "model": point["with_context"],
            "model_null_context": point["null_context"],
            "empirical_by_type": unit["heldout"]["empirical_by_type"],
            "empirical_global": unit["heldout"]["empirical_global"],
        }
        for predictor, scores in entries.items():
            arm = unit["arm"] if predictor.startswith("model") else "(학습 없음)"
            grouped[(arm, predictor)][unit["seed"]].append(scores)
    rows = []
    for (arm, predictor), by_seed in grouped.items():
        row: dict[str, Any] = {"arm": arm, "predictor": predictor, "seeds": len(by_seed)}
        if arm in report["selected_steps"]:
            row["steps"] = report["selected_steps"][arm]["selected"]
        for key in ("pitch_nll", "duration_nll", "pitch_top1", "full_mask_recall"):
            per_seed = [float(np.mean([s[key] for s in scores])) for scores in by_seed.values()]
            row[key] = round(float(np.mean(per_seed)), 3)
            row[f"{key}_range"] = f"{min(per_seed):.3f}–{max(per_seed):.3f}"
        rows.append(row)
    return rows


def recall_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for unit in report["full"]:
        for condition, scores in unit["train_recall"].items():
            grouped[(unit["arm"], condition)].append(scores["full_mask_recall"])
    return [
        {"arm": arm, "condition": condition, "full_mask_recall": round(float(np.mean(v)), 3),
         "range": f"{min(v):.3f}–{max(v):.3f}"}
        for (arm, condition), v in grouped.items()
    ]


GENERATION_COLUMNS = (
    "original_phrase_rate",
    "unseen_note_transition_rate",
    "note_4gram_novelty",
    "own_medoid_distance",
    "interval_jsd",
    "duration_jsd",
    "unseen_boundary_rate",
    "type_membership",
)


def generation_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One row per (generator, setting): mean over training seeds of the per-model means."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for unit in report["full"]:
        for setting, table in unit["generation"].items():
            grouped[(unit["arm"], setting)].append(table)
    for name, table in report["baselines"].items():
        generator, _, setting = name.partition("_rho")
        grouped[(generator, f"rho{setting}" if setting else "—")].append(table)
    rows = []
    for (generator, setting), tables in grouped.items():
        row: dict[str, Any] = {"generator": generator, "setting": setting, "models": len(tables)}
        for key in GENERATION_COLUMNS:
            row[key] = round(float(np.mean([t["metrics"][key]["mean"] for t in tables])), 4)
        for key in ("membership_rejection_rate", "first_step_membership_rate", "unchanged_share"):
            values = [t["chain"][key] for t in tables if key in t["chain"]]
            row[key] = round(float(np.mean(values)), 3) if values else None
        rows.append(row)
    return rows
