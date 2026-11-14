# 재현 안내

## 1. 논문 그림 ↔ 파일 ↔ 스크립트 ↔ 실행

`python figures/make_all.py`가 아래 순서로 모든 그림을 `figures/output/`에 다시 만듭니다.
중간 산출물 `figures/figure_data.json`, `figures/tda_order_check.json`도 저장소에 들어 있습니다.

| 논문 | 파일 | 스크립트 (함수) | 입력 |
|---|---|---|---|
| 그림 1 | `f01_overview.png` | `make_figures.py` (`fig1`) | 원곡, 실행 `20261112_2112` |
| 그림 2 | `f02_tda_order.png` | `check_tda_order.py` → `make_figures.py` (`fig_tda_order`) | 원곡 |
| 그림 3 | `f03_structure.png` | `make_figures.py` (`fig2`) | 원곡 |
| 그림 4 | `f04_manifold.png` | `make_figures.py` (`fig_manifold`, `manifold/*.svg`) | 개념도 |
| 그림 5 | `f05_type_variation.png` | `make_figures.py` (`fig_type_variation`) | 원곡 |
| 그림 6 | `f06_new_walks.png` | `make_figures.py` (`fig5`) | 원곡, 보행 표본 |
| 그림 7 | `f07_local_diffusion.png` | `make_figures.py` (`fig6`) | 개념도 |
| 그림 8 | `f08_structure_loss.png` | `make_figures.py` (`fig_graph_compare`) | `20261112_2112` |
| 그림 9 | `f09_return_joint.png` | `make_figures.py` (`fig8`) | `20261112_2112` |
| 그림 10 | `f10_copy_test.png` | `make_figures.py` (`fig_copy_test`) | `20261112_2112` |
| 그림 11 | `f11_results.png` | `make_figures.py` (`fig10`) | `20261112_2112` |
| 그림 12 | `f15_lambda_sweep.png` (+ `_stats.json`) | `make_sweep_figure.py` | `20261113_1653_…` + `20261113_2206_…` |
| 그림 13 | `f12_example.png` | `make_figures.py` (`fig11`) | `20261112_2112` |
| 그림 14 | `f13_excerpt_lambda.png` | `render_scores.py` | `20261113_1331_cpu` |
| 그림 15 | `f14_excerpt_return.png` | `render_scores.py` | `20261113_1331_cpu` |
| 그림 A1 | `fA1_heldout.png` | `make_figures.py` (`figA1`) | `20261112_2112` |
| 부록 K | `scores/score_*.png` | `render_scores.py` | `20261113_1331_cpu` |

파일 이름은 원고 초기 번호를 따르므로 그림 12–15의 번호와 다릅니다.

**확인한 것 (2026-11-14, macOS, Apple SD Gothic Neo, matplotlib·verovio·pymupdf 최신판)**: 26개 이미지가 원고의
그림과 픽셀 단위로 같고, `figure_data.json`, `f15_lambda_sweep_stats.json`, `tda_order_check.json`이 원고 때의
파일과 같습니다.

## 2. 실행과 코드 판

| 실행 | protocol | commit | 소스 해시(앞 16자) | 설정 |
|---|---|---|---|---|
| `20261112_2112` | eval3 | `1efc936` | `a7a2ef38fa24395e` | `configs/chart_diffusion_v1.yaml` (당시 판) |
| `20261113_1331_cpu` | eval4 | `a06050c` | `61702043c12dd43d` | `configs/chart_diffusion_v1.yaml` (당시 판) |
| `20261113_1653_cpu_lambda_sweep` | eval5 | `a06050c` + `uncommitted.diff` (= `d8b87e8`) | `45495be9f52c6ca4` | `configs/chart_diffusion_v1_lambda_sweep.yaml` |
| `20261113_2206_cpu_lambda_sweep_seeds12` | eval5 | `d8b87e8` | `45495be9f52c6ca4` | `configs/chart_diffusion_v1_lambda_sweep_more_seeds.yaml` |

현재 `main`의 소스 해시는 `45495be9f52c6ca4`로 λ 실험 두 실행과 같습니다. 앞의 두 실행을 같은 코드로 다시
돌리려면 해당 commit을 checkout합니다. 설정 파일과 `configs/yangcheongdodeuri.yaml`은 실행의 `meta.json`이 SHA-256으로
고정하므로 수정하지 않습니다(바꾸려면 새 파일을 만듭니다).

## 3. 체크포인트와 생성곡

- `results/checkpoints/20261113_1653_cpu_lambda_sweep/`: λ 8개, 학습 Seed 260807, 125 step.
- CPU에서 다시 학습하면 저자 환경(Apple M1, PyTorch 2.15, 기본 스레드 수)에서 비트 단위로 같습니다.
  `torch.set_num_threads`를 바꾸거나 다른 판·GPU에서는 1e-3 수준의 차이가 날 수 있습니다.
- 체크포인트로 `generate_piece(seed)`를 부르면 `results/runs/…/report_slim.json`의 같은 Seed 곡이 그대로 나옵니다
  (`scripts/export_web.py`와 튜토리얼 8절에서 확인).

## 4. 웹 데모

`python scripts/export_web.py`가 `site/models/*.onnx`와 `site/data/*.json`을 다시 만듭니다. ONNX 출력은 PyTorch와
확률 차이 4e-7 이하이며, `node site/tests/parity.mjs`가 JavaScript 편집 거리·유형 소속·보행·다양체 배치·MusicXML이
Python과 같은지 확인합니다.

## 5. 저장소에 없는 것

- 전체 체크포인트 132개(약 120 MB)와 원본 `report.json`(약 280 MB). 요청하면 별도 보관소(예: Zenodo, GitHub Release)로
  공개할 수 있습니다. `scripts/slim_runs.py`가 원본에서 `results/runs`를 만듭니다.
- 비교 생성기(R0–R2, T0/T2, TLJ-A)와 그 결과: 진행 중.
