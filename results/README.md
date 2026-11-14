# 논문 실행 결과

## `runs/<RUN_TAG>/`

`scripts/slim_runs.py`가 원본 실행 폴더(`artifacts/chart_diffusion_runs/<RUN_TAG>/`)에서 만든 요약입니다.

| 파일 | 내용 |
|---|---|
| `meta.json` | protocol, commit, 설정·곡 설정·소스 파일·마디 자료의 SHA-256, 실행 환경 |
| `config_used.yaml` | 실행에 쓴 설정 (있을 때) |
| `evaluation.json` | 네 축의 평가 지표, 복제 검정, 판정 기준 (원본 그대로) |
| `report_slim.json` | 원본 `report.json`에서 생성곡을 뺀 것. 보류 곡선, 고른 학습 길이, 학습 기록, 감시, 곡별 지표는 모두 남음. 생성곡은 `slim.pieces_kept_for_seeds`의 학습 Seed만 남김 |

| 실행 | 내용 | 생성곡을 남긴 학습 Seed |
|---|---|---|
| `20261112_2112` | Colab T4, λ ∈ {0, 1}, 학습 Seed 3 | 전부 |
| `20261113_1331_cpu` | CPU, λ ∈ {0, 0.25, 0.5, 1}, 조건마다 학습 길이 선택 | 260807 |
| `20261113_1653_cpu_lambda_sweep` | CPU, λ 8개, 모두 125 step, 학습 Seed 3 | 260807 |
| `20261113_2206_cpu_lambda_sweep_seeds12` | 같은 설정, 학습 Seed 12개 추가 | 없음 |

## `checkpoints/20261113_1653_cpu_lambda_sweep/`

λ ∈ {0, 0.05, 0.1, 0.15, 0.25, 0.5, 1, 2}, 학습 Seed 260807의 전체 학습 모형(각 약 1 MB).
`torch.load(..., weights_only=False)`로 읽으면 `state_dict`, 설정, 학습 기록이 들어 있습니다. 웹 데모의 ONNX 모형도
이 파일에서 만듭니다(`scripts/export_web.py`).

이 폴더의 결과는 전사 악보에서 만든 것이므로 MIT License 대상이 아닙니다([data/README.md](../data/README.md)).
