# 차트 복원 모형 실험을 Colab에서 실행하기

Methods 개정안(2026-11-11)의 2.3–2.7절을 설정 (가)(유형열 Y·환입 R·총음가 B 고정)로 구현한 첫 실험입니다.
Notebook은 [`experiments/chart_model_colab.ipynb`](../experiments/chart_model_colab.ipynb)입니다.
학습은 Colab GPU에서 합니다. Mac에서는 테스트와 smoke 실행만 합니다.

## 무엇을 구현했나

| 개정안 | 코드 (`src/gugakgen/chart/`) | 비고 |
|---|---|---|
| 2.3 편집거리, 가중 k-medoids | `edit_distance.py`, `typology.py` | 개정안의 유형열·실루엣(0.425/0.424/0.415)·접촉 쌍(13쌍, 31/56)을 재현 (`tests/test_chart.py`) |
| 2.4 Y, G, R, B | `typology.py`, `workflows/chart_experiment.py` | R은 `configs/yangcheongdodeuri.yaml`의 제1장→제4장 |
| 2.5 국소 좌표 (u, η) | `coords.py` | η는 음계 칸 수. 72마디 모두 좌표에서 그대로 복원됨 |
| 2.6 복원 모형 q_θ, 손실 | `model.py` | L_rec, L_M(여유), L_B(음가 합). L_G·L_R은 (가)에서 작용할 대상이 없어 넣지 않음 |
| 2.6 훼손–복원 연쇄 | `sampling.py` | 가린 구간 음가 합 유지, 유형 소속 거부, 환입 복사 |
| 2.7 평가, 기준선 | `metrics.py` | T0(재조합), T2(규칙 변이), 빈도표 차트 |

개정안이 정하지 않아 임시로 정한 것: 쪼개기·합치기의 비용식(`edit_distance.py` 머리말), 대응 음이 없는 사건과
쉼표의 좌표(`coords.py` 머리말), 접촉 쌍의 동률 처리(`typology.py`).

기존 MaskGIT 코드·설정·체크포인트와 선율형 A–I(`phrase_graph_v0.json`)는 바꾸지 않았습니다.

## 1. 실험 branch를 GitHub에 올리기 (Mac)

```bash
git push -u origin claude/method-section-model-improve-d5ad4d
```

## 2. Notebook 열기

```
https://colab.research.google.com/github/SHlee-TDA/DiffDodree/blob/claude/method-section-model-improve-d5ad4d/experiments/chart_model_colab.ipynb
```

`Runtime → Change runtime type`에서 GPU(T4)를 고르고 `Runtime → Run all`을 누릅니다.
코드 경로만 확인하려면 설정 셀에서 `RUN_MODE = "smoke"`로 바꿉니다.

## 3. 실험 구성 (`configs/chart_v0.yaml`)

1. **보류 평가.** 서로 다른 마디 51개 중 대표 마디 9개를 뺀 42개를 5묶음으로 나눕니다(같은 내용의 마디는 같은 묶음).
   한 묶음을 빼고 학습해, 뺀 마디에서 가린 사건의 (음높이, 음가) 예측 손실을 학습 길이별로 잽니다. 학습 조건 3개 × Seed 3개 × 5묶음.
2. **학습 길이 선택.** 조건마다 보류 손실이 가장 낮은 step을 고릅니다.
3. **전체 학습과 생성.** 그 step까지 72마디 전체로 학습하고, Seed 260807–260866으로 60곡씩 생성합니다
   (가림 비율 ρ = 0.3, 0.6, 1.0, 그리고 유형 소속 거부를 끈 대조 1개).
4. **기준선.** T0, T2(0.15, 0.3), 빈도표 차트를 같은 Seed·같은 지표로 평가합니다.

## 덮어쓰기 방지와 재개

- 모든 결과는 `artifacts/chart_runs/<RUN_TAG>/`에만 쓰입니다(smoke는 `<RUN_TAG>_smoke`).
- 학습 1회·생성 표 1개가 끝날 때마다 JSON 한 개로 저장합니다. 연결이 끊기면 설정 셀의 `RUN_TAG`에 처음 출력된 값을 넣고
  다시 실행합니다. 끝난 단위는 건너뜁니다.
- 같은 `RUN_TAG`를 다른 commit이나 다른 설정으로 이어서 쓰면 멈춥니다. 이미 있는 Checkpoint도 덮어쓰지 않습니다.
- Colab은 `MyDrive/DiffDodree_chart`라는 별도 clone을 씁니다. 기존 `DiffDodree`, `DiffDodree_conddrop`은 건드리지 않습니다.

## 결과 위치 (`MyDrive/DiffDodree_chart/artifacts/chart_runs/<RUN_TAG>/`)

| 파일 | 내용 |
|---|---|
| `report.json` | 유형화 요약, 문맥 주소성 점검, 보류 평가, 고른 학습 길이, 생성 평가, 기준선 |
| `heldout_<조건>_s<Seed>_f<묶음>.json` | 보류 평가 한 번의 학습 길이별 손실 |
| `full_<조건>_s<Seed>.json`, `.pt` | 전체 학습 모형의 평가와 Checkpoint |
| `baselines.json` | 학습 없는 기준선 |

표를 읽는 법은 Notebook 마지막 셀에 있습니다.
