# 2026-10 탐색 분석 스크립트

`docs/results/2026-10_memorization/README.md`의 수치를 만든 스크립트입니다. 탐색 단계에서 쓴 것이라
재사용 API(`src/`)가 아니며, 같은 결과를 다시 확인하는 용도로 둡니다. 저장소 root에서 실행합니다.
체크포인트는 저장소에 없으므로 경로를 인자로 줍니다. 메모리 8GB Mac에서는 CPU, 한 번에 하나씩 실행합니다.

| 스크립트 | 결과 | 인자 |
|---|---|---|
| `anchor_exp.py` | 2절 anchor 표시 수정 전후 비교 → `data/anchor_flag_*.json` | `<project_root> <checkpoint> <n_seeds> <out.json>` |
| `leak_probe.py` | 2절 주소 암기 진단 (표준출력) | `<project_root> <checkpoint> [...]` |
| `stage2_null.py` | 4절 2단계 실험 → `data/stage2_null_cpu.json` | `<checkpoint> <runtime_config.yaml> <out.json>` |
| `reeval_dtw.py` | 5.2절 A·B → `data/reeval_dtw.json` | `<checkpoint> <runtime_config.yaml> <out.json>` |
| `listening.py` | 들어 보기용 mp3·악보 (편집 거리 지표는 철회) | `<checkpoint> <runtime_config.yaml> <out_dir>` |

5.2절 A의 수치는 재사용 API `gugakgen.workflows.conddrop_experiment.evaluate_checkpoint`로도 그대로 나옵니다.
`anchor_exp.py`·`leak_probe.py`는 legacy `m2_denoiser_v0.pt`(저장소 `artifacts/`)와
`m2_denoiser_rel4_maskgit_v1.pt`(원래 작업 폴더 `artifacts/`, 저장소에 없음)를 사용했습니다.
`listening.py`는 `lame`(mp3 인코더)이 필요합니다.
