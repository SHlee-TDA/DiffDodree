# 실행에 필요한 frozen artifact

| File | 용도 |
|---|---|
| `measures_v0.json` | 72개 마디의 canonical pitch-duration record |
| `vocabulary_v0.json` | pitch, duration, family token mapping |
| `phrase_graph_v0.json` | frozen family label과 observed transition Graph |
| `m2_denoiser_v0.pt` | legacy compatibility·Sampling integration test용 Checkpoint |
| `release_manifest.json` | 배포 input의 byte 수와 SHA-256 |

포함된 legacy Checkpoint는 새 환입 Model의 full training 결과를 대신하지 않습니다.
03 Notebook이 생성하는 `checkpoints/*_smoke.pt`도 실행 확인용입니다.
새로운 full Checkpoint가 생기면 각 Notebook이 출력한 선택 경로와 SHA-256을 확인합니다.
04/06은 full → legacy → smoke 순, 05는 full → smoke → legacy 순으로 선택합니다.
정식 비교 실험에서는 자동 fallback에 의존하지 말고 Checkpoint를 명시적으로 고정해야 합니다.

`checkpoints/`, `runtime_configs/`, `tutorial_outputs/`, `evaluation/`은 실행 중 생성되며
Git 추적에서 제외됩니다. 연구 결과는 별도 영구 저장소에 configuration·Seed·commit과
함께 보존합니다. 현재 중간 Checkpoint의 영구 저장과 정확한 resume 보완은 후속 작업입니다.
