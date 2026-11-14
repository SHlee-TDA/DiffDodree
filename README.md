# DiffDodree

**〈양청도드리〉 한 곡에서 배우는 구조 보존 생성: 프레이즈 그래프 위의 마스크형 이산 확산**

[![웹 데모](https://img.shields.io/badge/웹_데모-GitHub_Pages-b5382b)](https://shlee-tda.github.io/DiffDodree/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SHlee-TDA/DiffDodree/blob/main/notebooks/tutorial.ipynb)
[![tests](https://github.com/SHlee-TDA/DiffDodree/actions/workflows/ci.yml/badge.svg)](https://github.com/SHlee-TDA/DiffDodree/actions/workflows/ci.yml)
[![code: MIT](https://img.shields.io/badge/code-MIT-blue)](LICENSE)

정악 〈양청도드리〉 피리 선율 72마디만으로 학습해, 원곡의 마디 구조와 환입(還入)을 지키면서 새 선율을 만드는 연구의 코드와 결과입니다.

1. **프레이즈 그래프**: 마디 사이의 길이 가중 편집 거리로 72마디를 아홉 유형(A–I)으로 묶고, 원곡의 유형 전이로 그래프 G를 만듭니다.
2. **프레이즈 다양체**: 유형마다 대표 마디를 중심으로 한 변주의 덩어리를 그래프의 뼈대에 붙인 것으로 봅니다. 원곡은 그 위를 지나는 하나의 곡선입니다.
3. **생성**: G 위의 1차 마르코프 보행으로 새 유형열을 뽑고(첫·끝 유형과 환입 조건은 정확히 만족), 각 마디의 음은 박당 4칸 격자 위의 **마스크형(흡수 상태) 이산 확산**으로 채웁니다. 환입의 두 구간은 한 변수로 함께 생성합니다.
4. **구조 손실**: 원곡에 없는 유형열을 조건으로 준 예측에서 유형 전이를 다시 읽어 조건과 맞춥니다(가중치 λ). λ는 '구조를 따르는 정도'와 '원곡에서 벗어나는 정도' 사이의 손잡이입니다.

> 논문은 투고 준비 중입니다. 서지 정보는 게재 후 추가합니다. 비교 생성기(R0–R2, T0/T2, TLJ-A) 실험은 아직 진행 중이며 이 저장소에 포함되지 않았습니다.

## 바로 해 보기

| 방법 | 설명 |
|---|---|
| [웹 데모](https://shlee-tda.github.io/DiffDodree/) | 프레이즈 그래프·다양체를 보고, 브라우저에서 곡을 생성(ONNX)하고, 곡이 다양체 위를 흘러가는 모습을 재생하며 악보를 보고 MusicXML로 내려받습니다. |
| [Colab 튜토리얼](https://colab.research.google.com/github/SHlee-TDA/DiffDodree/blob/main/notebooks/tutorial.ipynb) | 데이터 → 유형·그래프 → 다양체 → 보행 → 확산 → 학습(CPU 1분) → 생성 → 논문 결과 재현 확인. GPU가 필요 없습니다. |
| 로컬 | 아래 설치 후 `python -m pytest`, `python figures/make_all.py` |

```bash
git clone https://github.com/SHlee-TDA/DiffDodree.git
```

```bash
cd DiffDodree && python -m pip install -e ".[figures,dev]"
```

Python 3.10 이상, PyTorch 2.1 이상. 그림의 한글은 Apple SD Gothic Neo(macOS) 또는 NanumGothic(`apt-get install fonts-nanum`)으로 그립니다.

## 저장소 구조

```
src/gugakgen/chart/      생성기 v1: 편집 거리, 유형화, 격자, 보행, 확산 모형·손실, 평가
src/gugakgen/workflows/  실험 실행 (chart_diffusion_experiment.py)
configs/                 곡 설정과 실험 설정 (논문 실행이 해시로 고정한 파일이라 수정하지 않습니다)
data/                    〈양청도드리〉 피리 MusicXML (이용 조건: data/README.md)
artifacts/               처리된 72마디 등 고정 입력
results/runs/            논문에 쓴 네 실행의 요약 (평가, 학습 곡선, 일부 생성곡)
results/checkpoints/     λ 실험의 모형 8개 (학습 Seed 260807)
figures/                 논문 그림 재현 스크립트와 결과 (make_all.py)
notebooks/tutorial.ipynb Colab 튜토리얼
experiments/             전체 실험을 다시 돌리는 Colab 노트북
site/                    GitHub Pages 웹 데모 (정적 HTML + ONNX)
scripts/                 결과 요약, 웹 내보내기, 튜토리얼 생성
tests/                   pytest, site/tests/parity.mjs (JS 포트 검증)
```

## 재현하기

세 수준으로 재현할 수 있습니다. 자세한 대응표는 [docs/reproducibility.md](docs/reproducibility.md)에 있습니다.

1. **그림**: `python figures/make_all.py`는 `results/runs`의 요약에서 논문의 그림 1–15, A1과 부록 K 악보를 다시 그립니다. 저자 환경(macOS, Apple SD Gothic Neo)에서 26개 이미지 모두 픽셀 단위로 같습니다. 다른 글꼴에서는 글자 모양만 다릅니다.
2. **모형과 생성곡**: CPU에서 `configs/chart_diffusion_v1_lambda_sweep.yaml`로 다시 학습하면 저자 환경에서 `results/checkpoints`와 변수가 비트 단위로 같고, 그 체크포인트는 논문 실행의 생성곡을 그대로 다시 만듭니다(튜토리얼 8절). 스레드 수·PyTorch 판·GPU가 다르면 1e-3 수준의 차이가 날 수 있습니다.
3. **전체 실험**: [`experiments/chart_diffusion_colab.ipynb`](experiments/chart_diffusion_colab.ipynb)가 보류 평가, 학습 길이 선택, 60곡 생성, 복제 검정, 평가까지 실행합니다. 결과는 `artifacts/chart_diffusion_runs/<RUN_TAG>/`에 쓰이며 같은 이름이 있으면 덮어쓰지 않습니다. 각 실행의 `meta.json`은 commit, 설정·곡 설정·소스 파일의 SHA-256을 기록합니다.

| 논문 | 실행 | 코드 | 비고 |
|---|---|---|---|
| 그림 1, 8–11, 13, A1 | `20261112_2112` | `1efc936` | Colab T4, λ ∈ {0, 1}, 학습 Seed 3 |
| 그림 14–15, 부록 K | `20261113_1331_cpu` | `a06050c` | λ ∈ {0, 0.25, 0.5, 1}, 학습 길이를 조건마다 선택 |
| 그림 12 | `20261113_1653_cpu_lambda_sweep` + `20261113_2206_cpu_lambda_sweep_seeds12` | `a06050c`+diff, `d8b87e8` | λ 8개 × 학습 Seed 15, 모두 125 step. 현재 코드와 소스 해시가 같습니다. |

체크포인트 전체(132개, 약 120 MB)와 원본 `report.json`(약 280 MB)은 용량 때문에 저장소에 넣지 않았습니다. `scripts/slim_runs.py`가 원본 실행 폴더에서 `results/runs`를 만듭니다.

```bash
python -m pytest -q
```

```bash
node site/tests/parity.mjs
```

## 웹 데모의 생성에 대해

웹 데모는 `results/checkpoints`의 모형을 ONNX로 바꿔(`scripts/export_web.py`, PyTorch와의 확률 차이 ≤ 4e-7) 브라우저에서 같은 역과정·재생성 규칙을 실행합니다. 난수는 브라우저의 것이라 논문의 곡을 그대로 다시 만들지는 않지만, 같은 모형에서 30곡씩 뽑아 본 결과 재생성 전 유형 소속과 원곡 마디 복제율이 논문 실행의 60곡과 맞았습니다(λ = 0: 0.801 / 7.8% 대 0.796 / 8.3%, λ = 0.1: 0.846 / 9.1% 대 0.849 / 10.2%, λ = 1: 0.933 / 15.4% 대 0.929 / 15.1%). 소리는 브라우저 합성음이며 피리 녹음이 아닙니다.

## 이전 세대 코드

MaskGIT 기반 v0, chart v0 튜토리얼, condition dropout 실험, tutorials 01–06은 태그 [`v0-maskgit-legacy`](https://github.com/SHlee-TDA/DiffDodree/tree/v0-maskgit-legacy)에 남아 있습니다. `src/gugakgen/chart/model.py`·`sampling.py`와 `workflows/chart_experiment.py`(chart v0)는 v1이 일부 함수를 가져다 쓰므로 그대로 두었습니다.

## 라이선스와 인용

- 코드: [MIT License](LICENSE).
- 〈양청도드리〉 전사 악보와 이로부터 만든 파일: MIT 대상이 아니며, 제공자 확인 전까지 연구 목적 열람·재현만 허용합니다. [data/README.md](data/README.md)
- 인용: [CITATION.cff](CITATION.cff) (논문 게재 후 갱신).

---

### English summary

DiffDodree learns from a single piece — the 72-measure piri melody of the court-music piece *Yangcheongdodeuri* — and generates new melodies that keep its structure. Measures are clustered into nine types by a duration-weighted edit distance, giving a **phrase graph**; with the variation inside each type it forms a **phrase manifold** on which the original is one curve. A new type sequence is drawn as a Markov walk on the graph (with exact start/end and return-section constraints), and each measure is filled by **masked (absorbing-state) discrete diffusion** on a 4-cells-per-beat grid; the returning sections are decoded jointly. An off-data **graph loss** (weight λ) trades adherence to the structure against departure from the training measures. The [web demo](https://shlee-tda.github.io/DiffDodree/) runs the trained models in the browser (ONNX), and the [Colab tutorial](https://colab.research.google.com/github/SHlee-TDA/DiffDodree/blob/main/notebooks/tutorial.ipynb) walks through the method on CPU. Code is MIT-licensed; the transcribed score is not (see `data/README.md`).
