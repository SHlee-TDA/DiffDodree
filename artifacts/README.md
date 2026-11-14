# 고정 입력

| 파일 | 용도 |
|---|---|
| `measures_v0.json` | 72마디의 (음높이, 길이) 기록. 생성기 v1의 유일한 학습 자료 (`configs/yangcheongdodeuri.yaml`의 `canonical_measures`) |
| `vocabulary_v0.json`, `phrase_graph_v0.json` | 이전 세대(v0) 파이프라인의 어휘·유형 그래프. v1은 쓰지 않지만 곡 설정이 경로를 가리키므로 함께 둡니다 |
| `release_manifest.json` | 원본 MusicXML과 위 파일의 크기·SHA-256 |

각 실험 실행의 `meta.json`은 `measures_v0.json`의 SHA-256(`0d954194…`)을 기록합니다. 악보를 고치면 이 값이 달라지므로
기존 결과와 섞이지 않습니다. 실행 결과는 `artifacts/chart_diffusion_runs/<RUN_TAG>/`에 쓰이며 git으로 추적하지 않습니다.
논문에 쓴 실행의 요약은 `results/runs/`에 있습니다.

이 폴더의 파일은 전사 악보에서 만든 것이므로 MIT License 대상이 아닙니다([data/README.md](../data/README.md)).
