# DiffDodree 연구 진행 계획

2026-09-12 기준. 10월까지 단독 연구를 진행하고 11월에 국악 전문가 공동연구를 시작합니다.
아래 일정은 실행 순서의 제안이며 실험 결과와 GPU 이용 여건에 따라 조정합니다.

| 시기 | 작업 | 확인할 결과 |
|---|---|---|
| 9월 전반 | GitHub 배포와 새 checkout 검증, Colab/CUDA smoke | 누락 파일 없이 동일 tutorial 실행 |
| 9월 후반 | 중간 Checkpoint 영구 저장·resume 보완, pilot training | 중단 후 복구, 수치 안정성, runtime·메모리 기록 |
| 10월 전반 | 학습 조건별 Ablation, 여러 training Seed와 generation Seed | learned structure와 Sampling constraint의 기여 분리 |
| 10월 후반 | Counterfactual·novelty 분석, 평가용 악보와 논문 초안 | 재현 가능한 table·manifest, blinded 평가 자료 |
| 11월 | 전문가와 환입 정의·출력 검토, 평가·수정 | 음악적 타당성 및 claim 범위 확정 |

## 본실험 전에 확정할 것

1. Single-score internal learning의 성공을 관찰 가능한 구조 보존과 변형으로 정의합니다.
   철학적 이해나 독립 작품에 대한 일반화를 metric만으로 입증했다고 주장하지 않습니다.
2. 현재 05의 No-return은 동일 Checkpoint에서 relation을 끄는 inference Ablation입니다.
   training 단계의 기여를 분석하려면 conditioning/Loss function을 제거한 Model을 따로 학습합니다.
3. raw transition, projection 후 transition, 수정량을 분리하고 Rule Baseline과 비교합니다.
4. Exact-meter와 환입 satisfaction만으로 음악적 품질을 판단하지 않습니다. exact-copy 비율,
   새 제1장 material에 대한 환입 반응, 전문가 판단을 함께 사용합니다.
5. 초깃값인 6000 steps와 tutorial의 소수 sample은 확정된 논문 실험 설계가 아닙니다.
   반복 수, Seed, aggregation, confidence interval, 중단 기준을 pilot 이후 고정합니다.
6. 새 full Checkpoint를 각 조건에 명시적으로 지정합니다. smoke/legacy fallback 결과를
   Proposed Model의 최종 결과로 합치지 않습니다.

## 운영상 보완할 것

- Colab 중간 Checkpoint를 Drive에 주기적으로 복사하고 RNG·AMP state까지 포함한 resume 점검
- Model·data·Graph·configuration·commit·runtime version을 연결한 run manifest
- 논문용 metric 계산을 `src/`의 재사용 API로 옮기고 tutorial은 inspection 중심으로 유지
- 공동연구 전에 전문가 확인이 필요한 relation alignment를 별도 목록으로 관리

현재 `expert_validation: required`인 환입 alignment는 전문가 협의 전의 연구 가정으로
기록합니다. 기존 마디 구분과 율명 대응의 검증 상태와 혼동하지 않습니다.
