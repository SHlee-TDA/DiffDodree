# 배포 전 검증 기록

검증일: 2026-09-12. Git index에 등록된 Phase 1 파일만 별도 `/private/tmp` directory로
복원하고, 원래 작업 폴더의 virtualenv·미추적 파일이 없는 상태에서 실행했습니다.

| 항목 | 결과 |
|---|---|
| Source 구성 | 피리 MusicXML 2개, frozen JSON 3개, legacy Checkpoint 1개 |
| Input integrity | `artifacts/release_manifest.json`의 6개 SHA-256과 byte 수 일치 |
| Python / PyTorch | 3.13.6 / 2.14.0 |
| pytest | 15 passed, 1 skipped (CUDA 미제공) |
| Notebook 01–06 | 각각 새로운 Jupyter kernel에서 전체 Cell 순차 실행 통과 |
| Training | MPS smoke, 3 steps |
| Export | 04의 `export_last_generation()`을 추가 호출하여 MusicXML/MIDI 생성 통과 |
| Source Notebook output | 실행 결과를 source에 저장하지 않음 |

이 검증은 macOS의 isolated dependency 환경에서 수행했습니다. 실제 Colab Linux/CUDA,
Google 로그인·Drive 권한, browser widget interaction 및 한글·율명 렌더링의 시각적
확인은 아직 수행하지 않았습니다. 생성한 smoke 결과는 논문 evidence가 아닙니다.

현재 [Colab 검증 절차](colab_validation.md)를 따라 확인한 뒤 GPU, commit SHA,
Checkpoint SHA, runtime version, Notebook별 결과를 별도 기록으로 추가합니다.
