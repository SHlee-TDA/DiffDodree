# 논문 그림 재현

```bash
python figures/make_all.py
```

`results/runs/`의 실행 요약에서 논문의 모든 그림을 `figures/output/`에 다시 만듭니다(약 2분).
필요한 패키지: `pip install -e ".[figures]"`. 그림 속 한글은 Apple SD Gothic Neo 또는 NanumGothic을 씁니다.

| 단계 | 스크립트 | 입력 → 출력 |
|---|---|---|
| 1 | `extract_data.py` | 원곡 + `results/runs/20261112_2112` → `figure_data.json` |
| 2 | `check_tda_order.py` | `figure_data.json` → `tda_order_check.json` (음 기록 순서에 따른 바코드) |
| 3 | `make_figures.py` | 위 두 파일 → 그림 1–11, 13, A1 (`f01`–`f12`, `fA1`) |
| 4 | `make_sweep_figure.py` | `results/runs/20261113_1653_…`, `…_2206_…` → 그림 12 (`f15_lambda_sweep.png`, `_stats.json`) |
| 5 | `render_scores.py` | `results/runs/20261113_1331_cpu` → 그림 14–15 (`f13`, `f14`), 부록 K (`scores/`) |

논문 번호와 파일 이름의 전체 대응은 [docs/reproducibility.md](../docs/reproducibility.md)에 있습니다.
2026-11-14에 저자 환경에서 다시 만든 26개 이미지는 원고의 그림과 픽셀 단위로 같았습니다.
`output/`의 그림은 전사 악보에서 만든 것이므로 MIT License 대상이 아닙니다([data/README.md](../data/README.md)).
