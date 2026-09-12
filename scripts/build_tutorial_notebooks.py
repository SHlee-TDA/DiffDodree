#!/usr/bin/env python3
"""Build the six small, output-free Phase 1 tutorial notebooks."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DIR = PROJECT_ROOT / "tutorials"

CANONICAL_SETUP = r'''# Cell 1 — 공통 실행 환경 설정
# 이 Cell은 모든 tutorial Notebook의 첫 번째 Cell이며 여러 번 실행해도 안전합니다.

from pathlib import Path
import importlib.util
import os
import subprocess
import sys

# Google Drive의 project folder가 다르면 이 값만 수정합니다.
COLAB_PROJECT_ROOT = Path(
    "/content/drive/MyDrive/DiffDodree"
)

# 필요하면 환경변수로 project root를 직접 지정할 수 있습니다.
PROJECT_ROOT_OVERRIDE = os.environ.get("GUGAKGEN_PROJECT_ROOT")


def find_local_project_root(start: Path) -> Path:
    """현재 위치에서 pyproject.toml을 기준으로 project root를 찾습니다."""
    candidates = [start.resolve(), *start.resolve().parents]

    for candidate in candidates:
        if (candidate / "pyproject.toml").exists():
            return candidate

    raise FileNotFoundError(
        "project root를 찾지 못했습니다. "
        "GUGAKGEN_PROJECT_ROOT를 지정하거나 project folder에서 Notebook을 실행하세요."
    )


try:
    # 순수 local Python에는 상위 ``google`` namespace가 없을 수 있습니다.
    IN_COLAB = importlib.util.find_spec("google.colab") is not None
except ModuleNotFoundError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive

    # 이미 mount되어 있으면 기존 mount를 그대로 사용합니다.
    drive.mount("/content/drive", force_remount=False)

if PROJECT_ROOT_OVERRIDE:
    project_root = Path(PROJECT_ROOT_OVERRIDE).expanduser().resolve()
elif IN_COLAB:
    project_root = COLAB_PROJECT_ROOT.resolve()
else:
    project_root = find_local_project_root(Path.cwd())

if not project_root.exists():
    raise FileNotFoundError(
        f"project folder가 존재하지 않습니다: {project_root}\n"
        "COLAB_PROJECT_ROOT 또는 GUGAKGEN_PROJECT_ROOT를 확인하세요."
    )

bootstrap_script = project_root / "scripts" / "bootstrap_notebook.py"

if not bootstrap_script.exists():
    raise FileNotFoundError(
        f"Notebook bootstrap script를 찾지 못했습니다: {bootstrap_script}"
    )

environment_profile = "colab" if IN_COLAB else "local"

# 필요한 dependency만 확인하고 누락된 package만 설치합니다.
# Colab profile에서는 Colab이 제공한 CUDA용 PyTorch를 교체하지 않습니다.
subprocess.run(
    [
        sys.executable,
        str(bootstrap_script),
        "--project-root",
        str(project_root),
        "--profile",
        environment_profile,
        "--install-missing",
    ],
    check=True,
)

# 현재 kernel에서 src package를 즉시 import할 수 있게 합니다.
src_root = project_root / "src"

if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

os.environ["GUGAKGEN_PROJECT_ROOT"] = str(project_root)
os.chdir(project_root)

from gugakgen.device import resolve_device
from gugakgen.reproducibility import set_global_seed

runtime = resolve_device(preference="auto")
device = runtime.device

SEED = 2026
set_global_seed(SEED)

print("환경 설정 완료")
print(f"- 실행 환경: {'Google Colab' if IN_COLAB else 'Local Jupyter'}")
print(f"- Project root: {project_root}")
print(f"- Device: {device}")
print(f"- AMP enabled: {runtime.amp_enabled}")
print(f"- Seed: {SEED}")'''


def markdown(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip()}


def code(source: str, *, tags: Iterable[str] = ()) -> dict[str, Any]:
    metadata = {"tags": list(tags)} if tags else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": source.strip(),
    }


def notebook(cells: list[dict[str, Any]]) -> dict[str, Any]:
    # nbformat 4.5부터 Cell id가 schema의 정식 field입니다. Notebook별로
    # deterministic id를 부여해 diff가 안정적이고 validation warning이 없게 합니다.
    identified_cells = [
        {**cell, "id": f"cell-{index:02d}"}
        for index, cell in enumerate(cells, start=1)
    ]
    return {
        "cells": identified_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP_CELL = code(CANONICAL_SETUP, tags=("environment-setup",))


NOTEBOOKS: dict[str, list[dict[str, Any]]] = {
    "01_data_inspection.ipynb": [
        SETUP_CELL,
        markdown(r'''
# 01. 양청도드리 Data inspection

이 Notebook은 전문가가 전사하고 마디 단위로 구분한 양청도드리 피리 악보를 불러와
Phase 1의 symbolic representation을 확인합니다. MusicXML의 pitch와 duration을 그대로
읽으며, 율명 대응은 전문가가 검증한 mapping만 사용합니다.

이 단계에서는 시김새, 농현, 음색, 미세 timing을 추론하지 않습니다. 확인 대상은 악보의
마디, pitch token, duration token, family label입니다.
'''),
        code(r'''
# Piece configuration과 MusicXML source를 확인합니다.
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from IPython.display import display

from gugakgen.config import load_yaml
from gugakgen.data.musicxml import combine_scores, parse_musicxml, score_to_tokens
from gugakgen.data.vocabulary import ScoreVocabulary
from gugakgen.structure.transition_graph import TransitionGraph

piece_config = load_yaml(project_root / "configs/yangcheongdodeuri.yaml")
piece_info = piece_config["piece"]
data_config = piece_config["data"]
chapter_ranges = tuple(tuple(item["measures"]) for item in piece_config["chapters"])

source_paths = []
for pattern in data_config["source_globs"]:
    source_paths.extend(sorted(project_root.glob(pattern)))

if len(source_paths) != 2:
    raise RuntimeError(f"피리 MusicXML 2개를 예상했지만 {len(source_paths)}개를 찾았습니다: {source_paths}")

display(pd.DataFrame({"MusicXML source": [str(path.relative_to(project_root)) for path in source_paths]}))
'''),
        code(r'''
# 두 MusicXML segment를 72개 마디의 단일 악보로 연결합니다.
segments = []
measure_offset = 0
default_meters = ((4, 4), (6, 4))

for source_path, default_meter in zip(source_paths, default_meters):
    segment = parse_musicxml(
        source_path,
        piece_id=piece_info["id"],
        part_name=piece_info["part"],
        measure_offset=measure_offset,
        default_meter=default_meter,
        chapter_ranges=chapter_ranges,
        reject_chords=data_config["reject_chords"],
    )
    segments.append(segment)
    measure_offset += len(segment.measures)

score = combine_scores(segments)
score.validate_meters()

vocabulary = ScoreVocabulary.from_json(project_root / data_config["vocabulary"])
transition_graph = TransitionGraph.from_json(project_root / data_config["transition_graph"])
tokens = score_to_tokens(
    score,
    vocabulary,
    original_path=transition_graph.original_path,
    num_position_buckets=4,
)

summary = pd.DataFrame(
    [
        {"항목": "악곡", "값": piece_info["title_ko"]},
        {"항목": "파트", "값": piece_info["part"]},
        {"항목": "마디 수", "값": len(score.measures)},
        {"항목": "event 수", "값": sum(len(measure.events) for measure in score.measures)},
        {"항목": "pitch token 수", "값": vocabulary.num_pitch_tokens},
        {"항목": "duration token 수", "값": vocabulary.num_duration_tokens},
        {"항목": "family 수", "값": vocabulary.num_families},
        {"항목": "최대 event/마디", "값": vocabulary.max_events_per_measure},
    ]
)
display(summary)
'''),
        code(r'''
# 전문가가 정한 마디별 symbolic record와 token을 함께 확인합니다.
measure_rows = []
for index, measure in enumerate(score.measures):
    measure_rows.append(
        {
            "마디": measure.number,
            "장": measure.chapter,
            "장단": f"{measure.meter.numerator}/{measure.meter.denominator}",
            "event 수": len(measure.events),
            "pitch": [event.pitch for event in measure.events],
            "duration_q": [float(event.duration_q) for event in measure.events],
            "family": transition_graph.original_path[index],
        }
    )

measure_table = pd.DataFrame(measure_rows)
display(measure_table.head(12))
print("Token array shape:", tokens.pitch.shape)
print("첫 마디 pitch token:", tokens.pitch[0].tolist())
print("첫 마디 duration token:", tokens.duration[0].tolist())
'''),
        code(r'''
# 전문가 검증 율명 mapping으로 pitch distribution을 표시합니다.
yulmyung = {int(midi): name for midi, name in piece_config["pitch"]["midi_to_yulmyung"].items()}
pitch_counts = Counter(
    event.pitch
    for measure in score.measures
    for event in measure.events
    if event.pitch is not None
)
pitch_table = pd.DataFrame(
    [
        {"MIDI pitch": pitch, "율명": yulmyung.get(pitch, str(pitch)), "event 수": count}
        for pitch, count in sorted(pitch_counts.items())
    ]
)
display(pitch_table)

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.bar(pitch_table["율명"], pitch_table["event 수"], color="#4c6ef5")
ax.set_title("양청도드리 피리 pitch distribution")
ax.set_xlabel("율명")
ax.set_ylabel("event 수")
ax.grid(axis="y", alpha=0.2)
plt.show()
'''),
        code(r'''
# 원하는 마디를 선택하여 record와 token을 대화식으로 확인합니다.
import ipywidgets as widgets


def inspect_measure(measure_number: int = 1) -> None:
    index = measure_number - 1
    display(measure_table.iloc[[index]])
    print("pitch token   :", tokens.pitch[index].tolist())
    print("duration token:", tokens.duration[index].tolist())
    print("family token  :", int(tokens.family[index]))


widgets.interact(
    inspect_measure,
    measure_number=widgets.IntSlider(value=1, min=1, max=len(score.measures), step=1, description="마디"),
)
'''),
        markdown(r'''
## 확인 기준

- 모든 마디의 duration 합이 해당 장단과 정확히 일치해야 합니다.
- pitch 집합은 전문가가 검증한 율명 mapping의 범위 안에 있어야 합니다.
- PAD와 MASK는 음악 event가 아니라 Transformer 입력을 위한 특별 token입니다.
- 이 Notebook의 결과는 Data validation이며 생성 성능에 대한 결론이 아닙니다.
'''),
    ],
    "02_structure_graph.ipynb": [
        SETUP_CELL,
        markdown(r'''
# 02. Transition Graph와 환입 relation

이 Notebook은 전문가가 구분한 마디를 A–I family sequence로 읽고, 인접 마디 사이의
local transition Graph와 떨어진 장 사이의 환입 relation을 함께 확인합니다.

local Graph는 관찰된 family transition을 나타내며, 환입 relation은 제1장 material과
제4장 환입 구간 사이의 non-local correspondence를 별도로 표현합니다.
'''),
        code(r'''
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.patches import FancyArrowPatch

from gugakgen.config import load_yaml
from gugakgen.workflows import load_piece_context

train_config = load_yaml(project_root / "configs/train_maskgit.yaml")
context = load_piece_context(
    train_config["piece_config"],
    project_root=project_root,
    model_section=train_config["model"],
)

print("마디 수:", context.tokens.num_measures)
print("family:", context.graph.node_order)
print("환입 pair:", [(source + 1, target + 1) for source, target in context.hwanip_relations.pairs])
'''),
        code(r'''
# family와 local transition을 table로 확인합니다.
family_rows = []
for name in context.graph.node_order:
    node = context.graph.metadata["nodes"][name]
    family_rows.append(
        {
            "family": name,
            "마디 수": node["count"],
            "medoid 마디": node["medoid_measure"],
            "소속 마디": node["member_measures"],
        }
    )
display(pd.DataFrame(family_rows))

edge_table = pd.DataFrame(
    [
        {
            "source": edge.source,
            "target": edge.target,
            "count": edge.count,
            "probability": edge.probability,
        }
        for edge in context.graph.edges
    ]
).sort_values(["source", "probability"], ascending=[True, False])
display(edge_table)
'''),
        code(r'''
# local transition Graph를 시각화합니다.
graph_nx = nx.DiGraph()
for name in context.graph.node_order:
    graph_nx.add_node(name)
for edge in context.graph.edges:
    graph_nx.add_edge(edge.source, edge.target, weight=edge.count, probability=edge.probability)

positions = nx.circular_layout(graph_nx)
weights = [0.8 + 0.35 * graph_nx[u][v]["weight"] for u, v in graph_nx.edges]

fig, ax = plt.subplots(figsize=(8, 7))
nx.draw_networkx_nodes(graph_nx, positions, node_size=1500, node_color="#dbe4ff", ax=ax)
nx.draw_networkx_labels(graph_nx, positions, font_size=12, font_weight="bold", ax=ax)
nx.draw_networkx_edges(
    graph_nx,
    positions,
    width=weights,
    alpha=0.55,
    arrows=True,
    arrowsize=18,
    connectionstyle="arc3,rad=0.08",
    ax=ax,
)
ax.set_title("양청도드리 family transition Graph")
ax.axis("off")
plt.show()
'''),
        code(r'''
# 장 경계와 non-local 환입 relation을 하나의 timeline에서 확인합니다.
family_ids = context.tokens.family
family_names = [context.vocabulary.decode_family(int(value)) for value in family_ids]
family_y = {name: index for index, name in enumerate(context.graph.node_order)}
x = np.arange(1, context.tokens.num_measures + 1)
y = np.asarray([family_y[name] for name in family_names])

fig, ax = plt.subplots(figsize=(15, 5))
ax.step(x, y, where="mid", color="#495057", linewidth=1.2)
ax.scatter(x, y, c=context.tokens.chapter, cmap="tab10", s=30, zorder=3)

for chapter in context.config["chapters"]:
    start, end = chapter["measures"]
    ax.axvline(start - 0.5, color="#adb5bd", linewidth=0.8)
    ax.text((start + end) / 2, len(family_y) + 0.25, f"제{chapter['chapter']}장", ha="center")

for source, target in context.hwanip_relations.pairs:
    arc = FancyArrowPatch(
        (source + 1, family_y[family_names[source]] + 0.15),
        (target + 1, family_y[family_names[target]] + 0.15),
        connectionstyle="arc3,rad=-0.35",
        arrowstyle="-|>",
        mutation_scale=12,
        color="#e03131",
        alpha=0.8,
    )
    ax.add_patch(arc)

ax.set_yticks(range(len(family_y)), context.graph.node_order)
ax.set_xlim(0.5, context.tokens.num_measures + 0.5)
ax.set_ylim(-0.7, len(family_y) + 0.8)
ax.set_xlabel("마디")
ax.set_ylabel("family")
ax.set_title("장 경계, family path, 환입 relation")
ax.grid(axis="x", alpha=0.1)
plt.show()
'''),
        code(r'''
# 원곡 path와 환입 relation의 deterministic validation 결과를 확인합니다.
transition_validity = context.graph.transition_validity(context.tokens.family)
family_return = context.hwanip_relations.family_satisfaction(context.tokens.family)
surface_signatures = [measure.signature() for measure in context.score.measures]
surface_return = context.hwanip_relations.surface_satisfaction(surface_signatures)

display(
    pd.DataFrame(
        [
            {"metric": "transition validity", "value": transition_validity},
            {"metric": "환입 family satisfaction", "value": family_return},
            {"metric": "환입 surface satisfaction", "value": surface_return},
        ]
    )
)
'''),
        markdown(r'''
## 해석 주의

transition Graph와 환입 relation은 서로 다른 구조 층위입니다. transition Graph의
validity가 높다고 해서 새로운 제1장 material이 제4장에 관계적으로 돌아온다는 뜻은
아닙니다. 그 주장은 `05_hwanip_counterfactual_experiment.ipynb`의 Counterfactual
experiment로 별도 검증합니다.
'''),
    ],
    "03_model_training.ipynb": [
        SETUP_CELL,
        markdown(r'''
# 03. Transformer training

이 Notebook은 양청도드리 한 악보 안에서 반복적으로 corruption task를 구성하여
Transformer denoiser를 학습합니다. `smoke`는 MacBook CPU/MPS에서 pipeline을 확인하는
짧은 실행이고, `full`은 Colab 또는 Linux CUDA 환경에서 연구용 Checkpoint를 만드는 실행입니다.

학습에는 pitch, duration, family reconstruction과 함께 Graph, meter, 환입 Loss function을
사용합니다. `full` 실행은 CUDA가 없으면 즉시 중단됩니다.
'''),
        code(r'''
# 연구자가 주로 수정하는 training option입니다.
from copy import deepcopy

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from IPython.display import display

from gugakgen.config import load_yaml

RUN_MODE = "smoke"  # "smoke" 또는 "full"
SMOKE_STEPS = 3
FULL_STEPS = 6000
BATCH_SIZE = 32

# 각 Masking mode의 합은 1.0이어야 합니다.
MASKING_PROBABILITIES = {
    "event": 0.25,
    "measure": 0.15,
    "span": 0.15,
    "structure": 0.15,
    "joint": 0.15,
    "hwanip_target": 0.15,
}

if RUN_MODE not in {"smoke", "full"}:
    raise ValueError("RUN_MODE는 'smoke' 또는 'full'이어야 합니다.")
if abs(sum(MASKING_PROBABILITIES.values()) - 1.0) > 1e-9:
    raise ValueError("MASKING_PROBABILITIES의 합은 1.0이어야 합니다.")
if RUN_MODE == "full" and runtime.backend != "cuda":
    raise RuntimeError("full training에는 CUDA가 필요합니다. Colab GPU runtime을 선택하세요.")

print("RUN_MODE:", RUN_MODE)
print("Device:", runtime.as_dict())
'''),
        code(r'''
# 원본 YAML을 보존하고 tutorial 전용 runtime configuration을 만듭니다.
base_config_path = project_root / "configs/train_maskgit.yaml"
training_config = deepcopy(load_yaml(base_config_path))
training_config["training"]["smoke_steps"] = SMOKE_STEPS
training_config["training"]["steps"] = FULL_STEPS
training_config["training"]["batch_size"] = BATCH_SIZE
training_config["training"]["masking_modes"] = list(MASKING_PROBABILITIES)
training_config["training"]["masking_probabilities"] = list(MASKING_PROBABILITIES.values())

runtime_config_dir = project_root / "artifacts/runtime_configs"
runtime_config_dir.mkdir(parents=True, exist_ok=True)
runtime_config_path = runtime_config_dir / "tutorial_train_maskgit.yaml"
runtime_config_path.write_text(
    yaml.safe_dump(training_config, allow_unicode=True, sort_keys=False),
    encoding="utf-8",
)

display(
    pd.DataFrame(
        [{"Masking mode": key, "probability": value} for key, value in MASKING_PROBABILITIES.items()]
    )
)
print("Runtime configuration:", runtime_config_path.relative_to(project_root))
'''),
        code(r'''
# 같은 API로 smoke 또는 full training을 실행합니다.
from gugakgen.workflows.train import train_from_config

training_result = train_from_config(
    runtime_config_path,
    project_root=project_root,
    run_mode=RUN_MODE,
    runtime=runtime,
)

print("Training 완료")
print("Checkpoint:", training_result.checkpoint_path)
print("SHA-256:", training_result.checkpoint_sha256)
print("History:", training_result.history_path)
'''),
        code(r'''
# Loss function history를 table과 curve로 확인합니다.
history = pd.DataFrame(training_result.history)
display(history)

loss_columns = [
    column
    for column in history.columns
    if column.startswith("eval_") and column not in {"eval_total"}
]
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
history.plot(x="step", y=["train_total", "eval_total"], marker="o", ax=axes[0])
axes[0].set_title("Total Loss function")
axes[0].set_ylabel("Loss")
axes[0].grid(alpha=0.2)

history.plot(x="step", y=loss_columns, marker="o", ax=axes[1])
axes[1].set_title("Evaluation Loss function components")
axes[1].set_ylabel("Loss")
axes[1].grid(alpha=0.2)
plt.tight_layout()
plt.show()
'''),
        markdown(r'''
## 실행 결과의 용도

`*_smoke.pt`는 code path와 Device 호환성을 확인하기 위한 Checkpoint이며 논문 결과로
사용하지 않습니다. `full` 결과만 연구 실험에 사용하고, Checkpoint SHA-256과 runtime
metadata를 함께 보존합니다. MPS 결과와 CUDA 결과가 완전히 같은 floating-point 값을
가질 필요는 없지만, 동일 Seed의 실험 조건과 artifact provenance는 유지해야 합니다.
'''),
    ],
    "04_model_sampling_tutorial.ipynb": [
        SETUP_CELL,
        markdown(r'''
# 04. Model Sampling tutorial

이 Notebook은 공동연구자가 가장 자주 사용하는 interface입니다. pretrained Checkpoint를
불러온 뒤 Temperature, Graph guidance, Seed, transition path projection, 환입 mode를 조정하여
악보를 생성합니다. 모든 마디는 Exact-meter DP를 통과하므로 장단 합은 항상 정확합니다.

화면에서는 raw family path와 projected family path를 구분하여 표시합니다. MusicXML과 MIDI
export는 버튼 또는 마지막 fallback Cell에서 실행할 수 있습니다.
'''),
        code(r'''
from dataclasses import replace

import pandas as pd
from IPython.display import FileLink, clear_output, display

from gugakgen.config import load_yaml
from gugakgen.data.musicxml import tokens_to_score, write_musicxml
from gugakgen.workflows.generate import load_generation_bundle

checkpoint_candidates = [
    project_root / "artifacts/checkpoints/yangcheongdodeuri_maskgit.pt",
    project_root / "artifacts/m2_denoiser_v0.pt",
    project_root / "artifacts/checkpoints/yangcheongdodeuri_maskgit_smoke.pt",
]
checkpoint_path = next((path for path in checkpoint_candidates if path.exists()), None)
if checkpoint_path is None:
    raise FileNotFoundError("사용 가능한 Checkpoint가 없습니다. 먼저 03 Notebook을 실행하세요.")

bundle = load_generation_bundle(
    checkpoint_path,
    project_root=project_root,
    runtime=runtime,
)
piece_config = load_yaml(project_root / "configs/yangcheongdodeuri.yaml")
yulmyung = {int(midi): name for midi, name in piece_config["pitch"]["midi_to_yulmyung"].items()}

print("Checkpoint:", checkpoint_path.relative_to(project_root))
print("Checkpoint SHA-256:", bundle.checkpoint.sha256)
print("Device:", bundle.runtime.as_dict())
if "smoke" in checkpoint_path.name:
    print("주의: smoke Checkpoint는 interface 확인용이며 연구 결과용이 아닙니다.")
'''),
        code(r'''
# ipywidgets를 사용할 수 없는 환경에서는 DEFAULT_PARAMETERS를 그대로 사용합니다.
DEFAULT_PARAMETERS = {
    "seed": 2026,
    "num_steps": 12,
    "temperature_surface": 1.4,
    "temperature_family": 1.0,
    "graph_guidance": 0.5,
    "transition_projection": True,
    "exact_meter": True,
    "hwanip_mode": "learned",
}

try:
    import ipywidgets as widgets

    WIDGETS_AVAILABLE = True
    controls = {
        "seed": widgets.IntText(value=DEFAULT_PARAMETERS["seed"], description="Seed"),
        "num_steps": widgets.IntSlider(value=12, min=1, max=24, step=1, description="Sampling step"),
        "temperature_surface": widgets.FloatSlider(
            value=1.4, min=0.0, max=2.5, step=0.1, description="Surface Temperature"
        ),
        "temperature_family": widgets.FloatSlider(
            value=1.0, min=0.0, max=2.0, step=0.1, description="Family Temperature"
        ),
        "graph_guidance": widgets.FloatSlider(
            value=0.5, min=0.0, max=2.0, step=0.1, description="Graph guidance"
        ),
        "transition_projection": widgets.Checkbox(value=True, description="Path projection"),
        "exact_meter": widgets.Checkbox(value=True, disabled=True, description="Exact-meter DP"),
        "hwanip_mode": widgets.Dropdown(
            options=[("Learned relation", "learned"), ("Guided relation", "guided"), ("환입 끄기", "off")],
            value="learned",
            description="환입 mode",
        ),
    }
    display(widgets.VBox(list(controls.values())))
except ImportError:
    WIDGETS_AVAILABLE = False
    controls = {}
    print("ipywidgets를 사용할 수 없어 Python parameter fallback을 사용합니다.")


def current_parameters() -> dict:
    if not WIDGETS_AVAILABLE:
        return dict(DEFAULT_PARAMETERS)
    return {name: widget.value for name, widget in controls.items()}
'''),
        code(r'''
# 현재 control 값으로 한 개 악보를 생성합니다.
def generate_score(parameters: dict):
    if not parameters["exact_meter"]:
        raise ValueError("Phase 1에서는 Exact-meter DP를 해제할 수 없습니다.")

    sampling_config = replace(
        bundle.sampling_config,
        num_steps=int(parameters["num_steps"]),
        temperature_pitch_duration=float(parameters["temperature_surface"]),
        temperature_family=float(parameters["temperature_family"]),
        graph_guidance_scale=float(parameters["graph_guidance"]),
        hwanip_guidance_scale=1.0 if parameters["hwanip_mode"] == "guided" else 0.0,
        project_transition_path=bool(parameters["transition_projection"]),
    )

    original_relations = bundle.sampler.hwanip_relations
    if parameters["hwanip_mode"] == "off":
        bundle.sampler.hwanip_relations = None
    try:
        sampled = bundle.sampler.sample(
            seed=int(parameters["seed"]),
            config=sampling_config,
            score_anchors=bundle.score_anchors,
            family_anchors=bundle.family_anchors,
        )
    finally:
        bundle.sampler.hwanip_relations = original_relations

    score = tokens_to_score(
        sampled.tokens,
        bundle.context.vocabulary,
        piece_id=bundle.context.score.piece_id,
        part=bundle.context.score.part,
        meter_id_to_value=bundle.context.meter_id_to_value,
    )
    return sampled, score


def yul_string(measure) -> str:
    return " ".join(
        "쉼" if event.pitch is None else yulmyung.get(event.pitch, str(event.pitch))
        for event in measure.events
    )


def display_generation(sampled, score) -> None:
    raw_names = [bundle.context.vocabulary.decode_family(int(value)) for value in sampled.raw_family]
    final_names = [bundle.context.vocabulary.decode_family(int(value)) for value in sampled.tokens.family]
    rows = []
    for index, measure in enumerate(score.measures):
        rows.append(
            {
                "마디": index + 1,
                "장": measure.chapter,
                "raw family": raw_names[index],
                "projected family": final_names[index],
                "변경": raw_names[index] != final_names[index],
                "율명": yul_string(measure),
                "duration_q": [float(event.duration_q) for event in measure.events],
            }
        )
    table = pd.DataFrame(rows)
    print("Structure edit 수:", sampled.structure_edits)
    print("Final transition validity:", bundle.context.graph.transition_validity(sampled.tokens.family))
    display(table.head(16))
    display(table.loc[table["마디"].isin([1, 2, 3, 4, 25, 26, 27, 28])])


last_generation = {}
parameters = current_parameters()
sampled, generated_score = generate_score(parameters)
last_generation.update(parameters=parameters, sampled=sampled, score=generated_score)
display_generation(sampled, generated_score)
'''),
        code(r'''
# Button으로 parameter를 다시 읽고 생성 결과를 갱신합니다.
if WIDGETS_AVAILABLE:
    generate_button = widgets.Button(description="악보 생성", button_style="primary", icon="play")
    generation_output = widgets.Output()

    def on_generate_clicked(_button) -> None:
        with generation_output:
            clear_output(wait=True)
            parameters = current_parameters()
            sampled, generated_score = generate_score(parameters)
            last_generation.clear()
            last_generation.update(parameters=parameters, sampled=sampled, score=generated_score)
            display_generation(sampled, generated_score)

    generate_button.on_click(on_generate_clicked)
    display(generate_button, generation_output)
'''),
        code(r'''
# MusicXML/MIDI export 함수와 Button을 준비합니다.
def export_last_generation() -> tuple[Path, Path]:
    if not last_generation:
        raise RuntimeError("먼저 악보를 생성하세요.")
    output_dir = project_root / "artifacts/tutorial_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(last_generation["parameters"]["seed"])
    xml_path = output_dir / f"yangcheongdodeuri_seed_{seed}.musicxml"
    midi_path = output_dir / f"yangcheongdodeuri_seed_{seed}.mid"
    export_config = piece_config["export"]
    write_musicxml(
        last_generation["score"],
        xml_path,
        divisions_per_quarter=export_config["divisions_per_quarter"],
        key_fifths=export_config["key_fifths"],
        part_id=export_config["part_id"],
        part_name=export_config["part_name"],
    )

    from music21 import converter

    converter.parse(str(xml_path)).write("midi", fp=str(midi_path))
    print("MusicXML:", xml_path.relative_to(project_root))
    print("MIDI:", midi_path.relative_to(project_root))
    display(FileLink(str(xml_path)), FileLink(str(midi_path)))
    return xml_path, midi_path


if WIDGETS_AVAILABLE:
    export_button = widgets.Button(description="MusicXML/MIDI 내보내기", icon="download")
    export_output = widgets.Output()

    def on_export_clicked(_button) -> None:
        with export_output:
            clear_output(wait=True)
            export_last_generation()

    export_button.on_click(on_export_clicked)
    display(export_button, export_output)
'''),
        code(r'''
# Restart and Run All에서 자동 export가 필요하면 True로 바꿉니다.
EXPORT_ON_RUN_ALL = False
if EXPORT_ON_RUN_ALL:
    export_last_generation()
'''),
        markdown(r'''
## 결과 해석

`raw family`는 Transformer와 guidance가 만든 직접 출력이고, `projected family`는 관찰된
transition edge만 통과하도록 Viterbi projection을 적용한 결과입니다. 따라서 논문에서는
두 결과를 분리해 보고해야 합니다. MIDI는 악보 확인을 위한 편의 출력이며 피리 음색,
시김새, 농현 또는 실제 연주 표현을 재현하지 않습니다.
'''),
    ],
    "05_hwanip_counterfactual_experiment.ipynb": [
        SETUP_CELL,
        markdown(r'''
# 05. 환입 Counterfactual experiment

이 Notebook은 새로운 제1장 material이 제4장 환입 구간에 관계적으로 반영되는지 확인합니다.
먼저 제1장의 네 마디만 새로 생성한 뒤, 그 material을 고정한 상태에서 제4장 25–28마디를
다시 생성합니다.

비교 조건은 Proposed Graph-conditioned Model, No-return ablation, Rule Baseline,
Negative control입니다. 동일 Seed와 Sampling configuration을 사용하여 조건 간 차이를
paired comparison으로 읽습니다.
'''),
        code(r'''
from dataclasses import replace

import numpy as np
import pandas as pd
from IPython.display import display

from gugakgen.data.records import TokenizedScore
from gugakgen.structure.hwanip_relations import HwanipRelation, HwanipRelations
from gugakgen.workflows.generate import load_generation_bundle

checkpoint_candidates = [
    project_root / "artifacts/checkpoints/yangcheongdodeuri_maskgit.pt",
    project_root / "artifacts/checkpoints/yangcheongdodeuri_maskgit_smoke.pt",
    project_root / "artifacts/m2_denoiser_v0.pt",
]
checkpoint_path = next((path for path in checkpoint_candidates if path.exists()), None)
if checkpoint_path is None:
    raise FileNotFoundError("사용 가능한 Checkpoint가 없습니다.")

bundle = load_generation_bundle(checkpoint_path, project_root=project_root, runtime=runtime)
true_relations = bundle.context.hwanip_relations
N_TRIALS = 2  # tutorial 확인용. 논문 실험에서는 사전등록한 sample 수로 늘립니다.
SAMPLING_STEPS = 4
SEED_START = 3100

experiment_config = replace(
    bundle.sampling_config,
    num_steps=SAMPLING_STEPS,
    copy_rejection_attempts=2,
    hwanip_guidance_scale=1.0,
)

print("Checkpoint:", checkpoint_path.relative_to(project_root))
print("환입 pair:", [(a + 1, b + 1) for a, b in true_relations.pairs])
if "smoke" in checkpoint_path.name:
    print("주의: 현재 결과는 experiment code 확인용이며 연구 결과가 아닙니다.")
'''),
        code(r'''
# TokenizedScore에서 마디 signature와 Rule Baseline을 구성하는 helper입니다.
def token_signatures(tokens: TokenizedScore) -> list[tuple]:
    return [
        (
            tuple(int(value) for value in tokens.pitch[index]),
            tuple(int(value) for value in tokens.duration[index]),
        )
        for index in range(tokens.num_measures)
    ]


reference_signatures = set(token_signatures(bundle.context.tokens))


def rule_return(tokens: TokenizedScore) -> TokenizedScore:
    pitch = tokens.pitch.copy()
    duration = tokens.duration.copy()
    family = tokens.family.copy()
    event_counts = tokens.event_counts.copy()
    for source, target in true_relations.pairs:
        pitch[target] = pitch[source]
        duration[target] = duration[source]
        family[target] = family[source]
        event_counts[target] = event_counts[source]
    return replace(
        tokens,
        pitch=pitch,
        duration=duration,
        family=family,
        event_counts=event_counts,
    )


negative_relations = HwanipRelations(
    relations=(
        HwanipRelation(
            relation_id="negative_chapter_2_to_chapter_4",
            source_measures=(5, 8),
            target_measures=(25, 28),
            relation_type="return_of",
        ),
    ),
    num_measures=bundle.context.tokens.num_measures,
)
'''),
        code(r'''
# 제1장 material 생성과 제4장 conditional generation을 분리합니다.
def generate_new_chapter_one(seed: int) -> TokenizedScore:
    # 제2장 이후 악보를 고정하고 제1장 surface만 생성합니다.
    stage_one = bundle.sampler.sample(
        seed=seed,
        config=experiment_config,
        score_anchors=tuple(range(5, 73)),
        family_anchors=bundle.family_anchors,
    )
    return stage_one.tokens


def generate_return(
    initial: TokenizedScore,
    *,
    seed: int,
    relations: HwanipRelations | None,
) -> TokenizedScore:
    # 생성한 제1장과 환입 target 밖의 모든 마디를 고정합니다.
    fixed_measures = tuple([*range(1, 25), *range(29, 73)])
    original_relations = bundle.sampler.hwanip_relations
    bundle.sampler.hwanip_relations = relations
    try:
        result = bundle.sampler.sample(
            seed=seed,
            config=experiment_config,
            score_anchors=fixed_measures,
            family_anchors=bundle.family_anchors,
            initial=initial,
        )
    finally:
        bundle.sampler.hwanip_relations = original_relations
    return result.tokens
'''),
        code(r'''
# 네 조건을 동일 Seed로 실행하고 true 환입 relation에 대해 평가합니다.
rows = []
details = {}

for trial in range(N_TRIALS):
    seed = SEED_START + trial
    new_chapter_one = generate_new_chapter_one(seed)
    conditions = {
        "Proposed Graph-conditioned Model": generate_return(
            new_chapter_one, seed=seed + 100, relations=true_relations
        ),
        "No-return ablation": generate_return(
            new_chapter_one, seed=seed + 100, relations=None
        ),
        "Rule Baseline": rule_return(new_chapter_one),
        "Negative control": generate_return(
            new_chapter_one, seed=seed + 100, relations=negative_relations
        ),
    }

    for condition, tokens in conditions.items():
        signatures = token_signatures(tokens)
        rates = true_relations.counterfactual_return_rates(signatures, reference_signatures)
        row = {
            "trial": trial,
            "seed": seed,
            "condition": condition,
            **rates,
            "family satisfaction": true_relations.family_satisfaction(tokens.family),
            "surface satisfaction": true_relations.surface_satisfaction(signatures),
        }
        rows.append(row)
        details[(trial, condition)] = tokens

result_table = pd.DataFrame(rows)
display(result_table)
'''),
        code(r'''
# 조건별 평균을 계산합니다. None인 r_new/r_copy는 해당 subset이 없다는 뜻입니다.
summary_table = (
    result_table.groupby("condition", as_index=False)
    .agg(
        n_new=("n_new", "sum"),
        n_copy=("n_copy", "sum"),
        r_new=("r_new", "mean"),
        r_copy=("r_copy", "mean"),
        family_satisfaction=("family satisfaction", "mean"),
        surface_satisfaction=("surface satisfaction", "mean"),
    )
)
display(summary_table)

minimum_new_slots = 4
for _, row in summary_table.iterrows():
    if row["condition"] != "Rule Baseline" and row["n_new"] < minimum_new_slots:
        print(f"{row['condition']}: new material이 적어 r_new 해석을 보류합니다.")
'''),
        markdown(r'''
## 해석 원칙

Rule Baseline은 환입을 규칙으로 복사하므로 구조적 upper bound/control이며 learned result가
아닙니다. Proposed Model의 주장은 `r_new`가 충분한 new-material sample에서 No-return
ablation과 Negative control보다 높을 때만 지지됩니다. smoke Checkpoint 또는 두 trial의
결과는 실행 예시일 뿐 논문 evidence가 아닙니다.
'''),
    ],
    "06_evaluation_and_export.ipynb": [
        SETUP_CELL,
        markdown(r'''
# 06. Evaluation과 research artifact export

이 Notebook은 생성 결과를 raw, projected, accepted 단계가 섞이지 않도록 평가합니다.
transition validity, Graph path edit, exact-copy novelty, 환입 satisfaction, Exact-meter validity를
계산하고 paper-ready CSV와 provenance manifest를 저장합니다.

평가 범위는 symbolic 악보입니다. 시김새, 농현, 음색, 연주 timing은 metric과 expert protocol에서
제외하며 향후 연구 범위로 명시합니다.
'''),
        code(r'''
import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
from IPython.display import FileLink, display

from gugakgen.data.musicxml import tokens_to_score, write_musicxml
from gugakgen.workflows.generate import load_generation_bundle

checkpoint_candidates = [
    project_root / "artifacts/checkpoints/yangcheongdodeuri_maskgit.pt",
    project_root / "artifacts/m2_denoiser_v0.pt",
    project_root / "artifacts/checkpoints/yangcheongdodeuri_maskgit_smoke.pt",
]
checkpoint_path = next((path for path in checkpoint_candidates if path.exists()), None)
if checkpoint_path is None:
    raise FileNotFoundError("사용 가능한 Checkpoint가 없습니다.")

bundle = load_generation_bundle(checkpoint_path, project_root=project_root, runtime=runtime)
N_SAMPLES = 4  # tutorial용. 논문 보고에서는 사전등록한 sample 수를 사용합니다.
SEED_START = 4200
EXPORT_COUNT = 2
evaluation_config = replace(
    bundle.sampling_config,
    num_steps=4,
    copy_rejection_attempts=2,
)

print("Checkpoint:", checkpoint_path.relative_to(project_root))
print("SHA-256:", bundle.checkpoint.sha256)
if "smoke" in checkpoint_path.name:
    print("주의: smoke Checkpoint metric은 논문 결과로 사용하지 않습니다.")
'''),
        code(r'''
# novelty와 structure metric helper를 정의합니다.
def token_signatures(tokens) -> list[tuple]:
    return [
        (
            tuple(int(value) for value in tokens.pitch[index]),
            tuple(int(value) for value in tokens.duration[index]),
        )
        for index in range(tokens.num_measures)
    ]


original_signatures = token_signatures(bundle.context.tokens)
original_signature_set = set(original_signatures)


def evaluate_sample(seed: int):
    sampled = bundle.sampler.sample(
        seed=seed,
        config=evaluation_config,
        score_anchors=bundle.score_anchors,
        family_anchors=bundle.family_anchors,
    )
    signatures = token_signatures(sampled.tokens)
    exact_copy_any = float(np.mean([value in original_signature_set for value in signatures]))
    exact_copy_position = float(
        np.mean([value == original_signatures[index] for index, value in enumerate(signatures)])
    )
    meter_valid = all(
        bundle.sampler.meter_decoder.is_valid(
            sampled.tokens.pitch[index],
            sampled.tokens.duration[index],
            float(sampled.tokens.target_beats[index]),
        )
        for index in range(sampled.tokens.num_measures)
    )
    metrics = {
        "seed": seed,
        "raw_transition_validity": bundle.context.graph.transition_validity(sampled.raw_family),
        "projected_transition_validity": bundle.context.graph.transition_validity(sampled.tokens.family),
        "structure_edits": sampled.structure_edits,
        "structure_edit_rate": sampled.structure_edits / sampled.tokens.num_measures,
        "exact_copy_any": exact_copy_any,
        "exact_copy_position": exact_copy_position,
        "novel_measure_rate": 1.0 - exact_copy_any,
        "hwanip_family_satisfaction": bundle.context.hwanip_relations.family_satisfaction(
            sampled.tokens.family
        ),
        "hwanip_surface_satisfaction": bundle.context.hwanip_relations.surface_satisfaction(
            signatures
        ),
        "exact_meter_valid": bool(meter_valid),
    }
    return sampled, metrics
'''),
        code(r'''
# 여러 Seed의 결과를 생성하고 raw/projected metric을 분리해 수집합니다.
samples = []
metric_rows = []
for offset in range(N_SAMPLES):
    seed = SEED_START + offset
    sampled, metrics = evaluate_sample(seed)
    samples.append(sampled)
    metric_rows.append(metrics)

metrics_table = pd.DataFrame(metric_rows)
display(metrics_table)

metric_columns = [column for column in metrics_table.columns if column != "seed"]
paper_table = pd.DataFrame(
    {
        "metric": metric_columns,
        "mean": [metrics_table[column].astype(float).mean() for column in metric_columns],
        "std": [metrics_table[column].astype(float).std(ddof=1) for column in metric_columns],
        "n": N_SAMPLES,
    }
)
display(paper_table)
'''),
        code(r'''
# MusicXML, CSV, JSON manifest를 하나의 evaluation folder에 저장합니다.
output_dir = project_root / "artifacts/evaluation/tutorial_phase1"
output_dir.mkdir(parents=True, exist_ok=True)

metrics_csv = output_dir / "sample_metrics.csv"
paper_csv = output_dir / "paper_summary.csv"
metrics_table.to_csv(metrics_csv, index=False)
paper_table.to_csv(paper_csv, index=False)

exported_files = []
export_config = bundle.context.config["export"]
for sampled, metrics in zip(samples[:EXPORT_COUNT], metric_rows[:EXPORT_COUNT]):
    score = tokens_to_score(
        sampled.tokens,
        bundle.context.vocabulary,
        piece_id=bundle.context.score.piece_id,
        part=bundle.context.score.part,
        meter_id_to_value=bundle.context.meter_id_to_value,
    )
    destination = output_dir / f"yangcheongdodeuri_seed_{metrics['seed']}.musicxml"
    write_musicxml(
        score,
        destination,
        divisions_per_quarter=export_config["divisions_per_quarter"],
        key_fifths=export_config["key_fifths"],
        part_id=export_config["part_id"],
        part_name=export_config["part_name"],
    )
    exported_files.append(destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest_path = output_dir / "summary_manifest.json"
manifest = {
    "schema_version": 1,
    "scope": "symbolic pitch-duration score; 시김새/농현/audio excluded",
    "checkpoint": str(checkpoint_path.relative_to(project_root)),
    "checkpoint_sha256": bundle.checkpoint.sha256,
    "device": bundle.runtime.as_dict(),
    "sample_count": N_SAMPLES,
    "seeds": metrics_table["seed"].tolist(),
    "sampling_config": {
        key: getattr(evaluation_config, key) for key in evaluation_config.__dataclass_fields__
    },
    "metrics": metric_rows,
    "files": [],
}

files_for_manifest = [metrics_csv, paper_csv, *exported_files]
manifest["files"] = [
    {"path": str(path.relative_to(project_root)), "sha256": sha256(path)}
    for path in files_for_manifest
]
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print("Evaluation artifact:", output_dir.relative_to(project_root))
display(FileLink(str(metrics_csv)), FileLink(str(paper_csv)), FileLink(str(manifest_path)))
'''),
        markdown(r'''
## 논문 보고 전 Checklist

- smoke Checkpoint 결과를 제외합니다.
- 모든 condition에 동일한 Seed set과 Sampling 수를 사용합니다.
- raw, guided, projected, accepted metric을 한 값으로 합치지 않습니다.
- mean과 std뿐 아니라 sample-level CSV를 보존합니다.
- 공동연구자의 expert evaluation에서는 pitch, duration, 마디 구성, transition structure,
  환입 관계만 질문하고 시김새·농현·음색은 평가 대상에서 제외합니다.
'''),
    ],
}


def main() -> None:
    TUTORIAL_DIR.mkdir(parents=True, exist_ok=True)
    expected = set(NOTEBOOKS)
    for existing in TUTORIAL_DIR.glob("*.ipynb"):
        if existing.name not in expected:
            raise RuntimeError(f"예상하지 않은 tutorial Notebook이 있습니다: {existing}")
    for name, cells in NOTEBOOKS.items():
        path = TUTORIAL_DIR / name
        path.write_text(
            json.dumps(notebook(cells), ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
