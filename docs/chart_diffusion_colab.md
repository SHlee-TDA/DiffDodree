# 생성기 v1 실험을 Colab에서 실행하기

2026-11-11–12에 정한 설계를 구현했습니다. Notebook은
[`experiments/chart_diffusion_colab.ipynb`](../experiments/chart_diffusion_colab.ipynb)입니다.
학습은 Colab GPU에서 합니다. Mac에서는 테스트와 smoke 실행만 합니다.

## 설계와 코드

| 결정 | 구현 (`src/gugakgen/`) |
|---|---|
| 유형: 편집거리 k-medoids, k = 9 | `chart/edit_distance.py`, `chart/typology.py` (개정안의 유형열 재현) |
| 유형열 Y′: G 위 1차 Markov 보행, 시작·끝 유형 고정, 장 경계 조건 없음 | `chart/walk.py`. 환입 자리(1–4 = 25–28마디)는 같은 유형이어야 하므로, 이 조건까지 포함한 조건부 분포에서 정확히 뽑음 |
| 4박·6박 마디가 섞인 유형 허용 | 대표형을 위상으로 늘이거나 줄여 기준으로 씀 (`chart/grid.py`의 `reference`) |
| 정간 격자 | `chart/grid.py`. 박당 칸 수는 설정값(지금 4). 3으로 바꾸면 지금 전사의 48마디를 적을 수 없음 |
| 마스크형 이산 확산 | `chart/diffusion.py`. 흡수 상태 순방향(D3PM), 마스크를 예측하지 않는 복원(MDLM SUBS), 시간 입력 없음 |
| 손실 = ELBO + λ · 그래프 손실 (i) | `chart/diffusion.py`의 `diffusion_loss`, `graph_loss`. 그래프 손실은 새 유형열을 조건으로 모두 가린 입력에서 계산 |
| 환입 | 생성 때 1–4마디와 25–28마디를 한 변수로 묶어 공동 디코딩 (`reverse_sample`) |
| 감시 | 전체 학습 중 고른 학습 길이를 5등분한 시점마다 곡 4개를 끝까지 생성해 표면 그래프 확인 (기울기 없음) |

임시로 정한 것: 확산 시간의 하한 `r_min = 0.02`, 이웃 문맥 가림 확률 0.5(10월 주소 암기 대책을 그대로 둠),
그래프 손실의 시작 시점(0 step)과 증가 기간(50 step), 온도 γ = 0.5, λ = 1, 새 유형열 8개/step.
생성 후 유형 소속(편집거리)을 어긴 마디는 최대 4번 다시 생성하고, 거부 전 소속 비율을 따로 기록합니다.

**코드 검토(commit 45618a9)에서 고친 것.** 환입 두 자리의 예측을 더하던 것을 평균으로 바꿨습니다(더하면 두 예측이 공유하는 정보가 두 번 계산되어 1–4마디가 거의 argmax로 뽑히고, 둘째 실행에서 그 자리의 원곡 복사율이 74%였습니다).
복제 검정에 `z_novel`을 더하고 동점 보정을 넣었으며, coverage의 bootstrap 구간을 없앴고, 보류 기준선을 ELBO와 같은 행 평균으로 바꿨습니다.
전체 학습 뒤 생성 중에 끊기면 Checkpoint를 읽어 생성만 다시 하고, RUN_TAG는 곡 설정(`yangcheongdodeuri.yaml`)과 소스 파일 hash까지 같아야 이어 씁니다.
남은 한계: 대표형은 보류하지 않으므로 학습 길이는 대표형에서 먼 마디의 ELBO로 정해집니다. GPU에서 복제 검정용 재학습 모형은 보류 실행의 같은 step 모형과 비트 단위로 같지는 않습니다. 이웃 경계의 MASK 토큰은 확산 손실에서 학습되지 않습니다(생성 결과에 주는 영향은 검토에서 무시할 만한 수준으로 측정됨).

**첫 실행(`20261112_1717`)에서 고친 것.** 보류 ELBO 최저점이 100 step이어서 두 조건 모두 100 step으로 학습했는데,
그래프 손실은 200 step부터 켜지도록 되어 있어 λ = 1 모형이 λ = 0 모형과 똑같았습니다(기준 5의 'fail'은 무효).
그래서 그래프 손실을 0 step부터 켜고(50 step 동안 λ를 올림), 보류 평가를 25 step 간격부터 촘촘히 1000 step까지 보며,
감시 시점을 학습 길이에 맞춰 나눴습니다. 고른 길이에서 그래프 손실이 한 번도 켜지지 않았으면 기준 5는 `invalid`로 표시됩니다.
CPU 예비 확인(Seed 1개, 8곡)에서 100 step의 거부 전 유형 소속은 λ = 0이 0.71, λ = 1이 0.92였고 보류 ELBO 최저값은 거의 같았습니다.

## 평가 (2026-11-12 고정, `chart/evaluation.py`)

학습이 끝나면 같은 실행 안에서 평가까지 계산해 `evaluation.json`에 씁니다. 저장된 생성곡만 쓰므로 Mac의 CPU에서도 다시 계산할 수 있습니다.

| 축 | 지표 |
|---|---|
| 기하 | 마디 수준: 표면 유형 그래프와 원곡 G의 TV(곡별, 60곡 합산), 표면 전이의 G 포함 비율, 거부 전 유형 소속. 음표 수준: 정재훈 교수팀의 음표 그래프(1/가중치 최단 경로 거리)로 만든 1차원 persistence barcode와 원곡 barcode의 bottleneck·1-Wasserstein 거리 |
| 통계 | 음정·음가·음높이·음 3/4-gram 분포의 JSD. 마디 사이 편집거리 분포의 겹침 넓이(원곡 안 대 원곡–생성, 원곡 안 대 생성 안) |
| 음악 | 음계 밖 음, 박 채움 오류, 격자 밖 시작점(이 셋은 격자 모형에서 구성상 0), 장 끝 종지음, 곡 끝 음 |
| 새로움 | 원곡 마디 그대로 비율(원곡 자체의 반복률과 비교), authenticity(최근접 원곡 마디의 원곡 안 최근접 거리보다 가까우면 복제), density·coverage(k = 5, 서로 다른 원곡 마디 51개 기준; Naeem et al.의 원 정의는 중복 포함), 최근접 거리, 서로 다른 마디 비율, 원곡과 겹치는 최장 음 구간, data-copying 검정 C_T(보류 모형 9개 × 20곡, 묶음별 Z를 Stouffer로 합침; 동점 보정 포함) |

**복제 검정을 두 가지로 계산하는 이유.** 원곡은 72마디 중 33마디(46%)가 곡 안 다른 자리에 똑같이 다시 나옵니다.
보류 묶음은 대표형과, 학습 마디와 내용이 같은 마디를 빼고 만들기 때문에 보류 마디는 학습 마디와의 거리가 0일 수 없습니다.
그래서 원곡처럼 반복하기만 해도 생성 마디 전부로 계산한 `z`는 음수로 치우칩니다(원곡 자체를 넣으면 `z` ≪ −2, `z_novel` = 0).
판정 기준 4의 |Z| < 2는 학습 마디와 똑같은 마디를 뺀 `z_novel`로 보고, 뺀 비율(`exact_train_share`)은 원곡의 반복률과 함께 보고합니다.

판정 기준 다섯 가지는 고정했습니다. 이번 실행은 우리 방법만 평가하므로 기준 5(λ = 1이 학습 Seed 3개 모두에서 거부 전 유형 소속이 더 높음)와 기준 4의 |Z| < 2만 판정하고, 기준선(R0–R2, T0/T2)과 TLJ-A가 필요한 나머지는 '비교 대기'로 둡니다.

평가에는 `scipy`와 `ripser`가 필요하며, Notebook의 첫 설정 셀이 없으면 설치합니다.

## 1. 실험 branch를 GitHub에 올리기 (Mac)

```bash
git push -u origin claude/method-section-model-improve-d5ad4d
```

## 2. Notebook 열기

```
https://colab.research.google.com/github/SHlee-TDA/DiffDodree/blob/claude/method-section-model-improve-d5ad4d/experiments/chart_diffusion_colab.ipynb
```

`Runtime → Change runtime type`에서 GPU(T4)를 고르고 `Runtime → Run all`을 누릅니다.
코드 경로만 확인하려면 설정 셀에서 `RUN_MODE = "smoke"`로 바꿉니다.

## 덮어쓰기 방지와 재개

- **새 commit을 받은 뒤에는 Runtime → Restart session 후 실행합니다.** 같은 session에서 다시 Run all 하면 메모리에 남은 이전 코드로 돕니다(`20261112_1955`는 meta에 f46935c가 적혔지만 45618a9 코드로 돌았음). 이제 첫 셀과 `run_experiment`가 이 경우 멈춥니다.

- 결과는 `artifacts/chart_diffusion_runs/<RUN_TAG>/`에만 쓰입니다(smoke는 `<RUN_TAG>_smoke`).
- 학습 1회가 끝날 때마다 JSON 하나로 저장합니다. 연결이 끊기면 처음 출력된 `RUN_TAG`를 넣고 다시 실행합니다.
- 같은 `RUN_TAG`를 다른 commit·설정으로 이어 쓰거나, 이미 있는 Checkpoint·MusicXML을 덮어쓰려 하면 멈춥니다.
- Colab은 `MyDrive/DiffDodree_chart` clone을 씁니다(v0 실험과 같은 폴더, 결과 폴더는 다름).

## 결과 위치 (`MyDrive/DiffDodree_chart/artifacts/chart_diffusion_runs/<RUN_TAG>/`)

| 파일 | 내용 |
|---|---|
| `report.json` | 고른 학습 길이, 보류 곡선, 감시 기록, 암기 정도, 생성 시 기록한 지표 |
| `heldout_<조건>_s<Seed>_f<묶음>.json` | 보류 평가 한 번 |
| `full_<조건>_s<Seed>.json`, `.pt` | 전체 학습 모형의 기록·생성곡(유형열과 음표)·Checkpoint |
| `copytest_<조건>_s<Seed>_f<묶음>.json` | 복제 검정용 보류 모형의 생성곡 20곡 |
| `evaluation.json` | 네 축의 지표(곡 단위 95% bootstrap 구간 포함), 복제 검정, 판정 기준 |
| `musicxml/` | 들어 볼 생성곡 (조건마다 3곡) |
