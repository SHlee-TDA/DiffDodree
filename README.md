# DiffDodree

양청도드리의 마디 전이와 환입 관계를 보존하는 symbolic score generation 연구입니다.
Single-score internal learning을 사용하며, Python package 이름은 `gugakgen`입니다.
현재 구현은 Transformer 기반 MaskGIT iterative Masking/denoising입니다.
프로젝트 이름 자체가 특정 Diffusion model의 구현이나 성능 입증을 뜻하지는 않습니다.

전문가가 구분한 72개 피리 마디와 검증한 율명–pitch 대응을 유지합니다.
시김새, 농현, audio synthesis는 후속 연구 범위입니다.

## 시작하기

처음에는 [Colab 실행·검증 안내](docs/colab_validation.md)를 따라 Drive에 이 저장소를
clone한 뒤 아래 tutorial을 엽니다. **GitHub에서 Notebook을 여는 것만으로 source와
data가 runtime에 clone되지는 않습니다.** 모든 Notebook의 Cell 1은
`/content/drive/MyDrive/DiffDodree`를 기본 경로로 사용합니다.

| Notebook | 내용 | Colab |
|---|---|---|
| [01](tutorials/01_data_inspection.ipynb) | 악보·마디·율명·duration inspection | [열기](https://colab.research.google.com/github/SHlee-TDA/DiffDodree/blob/main/tutorials/01_data_inspection.ipynb) |
| [02](tutorials/02_structure_graph.ipynb) | transition Graph와 환입 시각화 | [열기](https://colab.research.google.com/github/SHlee-TDA/DiffDodree/blob/main/tutorials/02_structure_graph.ipynb) |
| [03](tutorials/03_model_training.ipynb) | smoke/full training과 Loss history | [열기](https://colab.research.google.com/github/SHlee-TDA/DiffDodree/blob/main/tutorials/03_model_training.ipynb) |
| [04](tutorials/04_model_sampling_tutorial.ipynb) | Sampling controls와 악보 export | [열기](https://colab.research.google.com/github/SHlee-TDA/DiffDodree/blob/main/tutorials/04_model_sampling_tutorial.ipynb) |
| [05](tutorials/05_hwanip_counterfactual_experiment.ipynb) | 제1장–제4장 Counterfactual experiment | [열기](https://colab.research.google.com/github/SHlee-TDA/DiffDodree/blob/main/tutorials/05_hwanip_counterfactual_experiment.ipynb) |
| [06](tutorials/06_evaluation_and_export.ipynb) | metric·CSV·manifest export | [열기](https://colab.research.google.com/github/SHlee-TDA/DiffDodree/blob/main/tutorials/06_evaluation_and_export.ipynb) |

## Local Jupyter

Python 3.10 이상을 사용합니다. virtualenv는 Google Drive 밖 local disk에 두면
많은 dependency file을 import할 때의 동기화 지연을 줄일 수 있습니다.
사용할 Python 환경을 활성화하고 repository root에서 실행합니다.

```bash
python scripts/bootstrap_notebook.py --project-root . --profile local --install-missing
python -m pip install 'pytest>=8,<10'
python -m pytest -q tests
```

Jupyter에서 tutorial을 열고 Cell 1부터 실행합니다. `src/`는 Cell 1에서 import 경로에
추가되므로 editable install이 필수는 아닙니다. Colab에서는 `--profile colab`을 사용하며
기존 CUDA PyTorch를 유지하기 위해 `requirements/dev.txt`를 통째로 설치하지 않습니다.

## 배포 구성과 현재 상태

- `src/gugakgen/`: data, structure, Model, Sampling, training/generation API
- `configs/`: 양청도드리 구조와 training configuration
- `scripts/`: dependency bootstrap과 6개 tutorial generator
- `tests/`: Device, 공통 Cell 1, MusicXML/Graph/DP/Sampling integration
- `data/`: 현재 pipeline에 필요한 피리 MusicXML 2개; [설명](data/README.md)
- `artifacts/`: frozen input 3개와 legacy Checkpoint; [설명](artifacts/README.md)
- `docs/`: Colab 검증 절차와 [연구 roadmap](docs/research_roadmap.md)

새 배포 사본에서 수행한 [검증 기록](docs/validation_status.md)을 함께 제공합니다.

로컬 Phase 1 검증에서 pytest 15 passed / 1 CUDA skip 및 6개 Notebook의 순차 실행을
확인했습니다. 실제 Colab/CUDA 검증과 새 Model의 full training은 별도로 수행해야 합니다.
`*_smoke.pt`는 실행 확인용이며 논문 결과에 사용하지 않습니다. 생성 output, virtualenv,
과거 root Notebook은 이 배포의 추적 대상에서 제외됩니다. 원래 작업 폴더의 파일은 유지됩니다.

현재 open-source 또는 데이터 재배포 license는 별도로 지정하지 않았습니다.
코드·전사 악보·Checkpoint의 이용 조건은 저장소 소유자에게 확인합니다.
