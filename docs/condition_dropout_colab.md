# 조건 드롭아웃 실험을 Colab에서 실행하기

기존 모델은 (장, 박자, 장 안 위치) 조합만 보고 원곡 1장·4장을 그대로 외워 냅니다.
이 실험은 학습 중 장·위치 조건을 무작위로 가린 모델(`conddrop`)과 기존 모델(`baseline`)을
같은 설정·같은 Seed로 학습해 비교합니다. Notebook은
[`experiments/condition_dropout_colab.ipynb`](../experiments/condition_dropout_colab.ipynb)입니다.

MacBook(메모리 8GB)에서는 두 모델 학습이 스왑으로 사실상 멈추므로 Colab GPU에서 실행합니다.

## 1. 실험 branch를 GitHub에 올리기 (Mac)

Colab은 GitHub에서 code를 받습니다. 실험 code가 들어 있는 branch를 먼저 push합니다.

```bash
git push -u origin claude/generated-song-verse-1-4-identity-182b42
```

다른 branch 이름으로 올렸다면 Notebook Cell 0의 `BRANCH`를 같은 이름으로 바꿉니다.

## 2. Notebook 열기

브라우저에서 아래 주소를 엽니다. branch 이름의 `/`는 그대로 둡니다.

```
https://colab.research.google.com/github/SHlee-TDA/DiffDodree/blob/claude/generated-song-verse-1-4-identity-182b42/experiments/condition_dropout_colab.ipynb
```

`Runtime → Change runtime type`에서 GPU(T4로 충분)를 고릅니다.

## 3. 실행

위에서 아래로 실행합니다(`Runtime → Run all`).

| Cell | 하는 일 |
|---|---|
| Cell 0 | Drive를 연결하고 `MyDrive/DiffDodree_conddrop`에 실험 branch를 clone합니다. 기존 `MyDrive/DiffDodree`는 건드리지 않습니다. |
| Cell 1 | 다른 tutorial과 같은 공통 환경 설정입니다. |
| 설정 | `RUN_MODE`, `STEPS`, `EVAL_SEEDS`, `RUN_TAG`를 정합니다. 처음에는 바꾸지 않아도 됩니다. |
| 학습 | baseline → conddrop 순서로 **한 번에 하나씩** 학습합니다. |
| 평가 | 두 모델을 같은 60개 Seed로 생성해 표 세 개를 보여 줍니다. |

코드 경로만 먼저 확인하고 싶으면 `RUN_MODE = "smoke"`로 3 step만 돌릴 수 있습니다.

## 덮어쓰기 방지

- 결과 파일 이름에는 실행 시각으로 만든 `RUN_TAG`(예: `20261025_1030`)가 붙습니다.
- 같은 이름의 Checkpoint·history·평가 JSON이 이미 있으면 쓰지 않고 멈춥니다.
- `DiffDodree_conddrop` clone에 commit하지 않은 수정이 있으면 Cell 0이 멈춥니다.

## 연결이 끊겼을 때

학습 중 Checkpoint는 runtime의 임시 폴더에 쓰이고, 학습이 끝난 뒤에 Drive로 복사됩니다.
그래서 학습 도중 끊기면 그 모델은 처음부터 다시 학습해야 합니다.

다시 실행할 때는 설정 셀의 `RUN_TAG`에 처음 출력된 값을 넣습니다.
이미 끝난 모델은 건너뛰고, 남은 모델만 학습한 뒤 평가합니다.

## 결과 위치

| 파일 | 위치 (`MyDrive/DiffDodree_conddrop/` 아래) |
|---|---|
| Checkpoint | `artifacts/checkpoints/yangcheongdodeuri_maskgit{,_conddrop}_<RUN_TAG>.pt` |
| 학습 history (암기 정도 포함) | `artifacts/checkpoints/*_<RUN_TAG>_history.json` |
| 평가 결과 | `artifacts/evaluation/conddrop_<RUN_TAG>.json` |

Drive가 Mac과 동기화되어 있으면 Mac의 `내 드라이브/DiffDodree_conddrop`에서도 같은 파일을 볼 수 있습니다.
표를 읽는 법은 Notebook 마지막 셀에 있습니다.
