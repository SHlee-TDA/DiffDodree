# Colab에서 DiffDodree 검증하기

첫 검증은 브라우저에서 수행합니다. GPU 연산과 함께 Cell 1, Drive mount, widget,
한글·율명 표시, 악보 export를 직접 확인하기 위한 절차입니다.

## 1. 최초 한 번 Drive에 clone

[빈 Colab Notebook](https://colab.research.google.com/#create=true)을 열고 작업할 Google
계정으로 로그인합니다. 아래 임시 준비 Cell을 실행합니다. 저장소에 별도의 00 Notebook을
추가할 필요는 없습니다. 원래 `MyDrive/국악생성인공지능` 폴더와 다른 새 clone을 만듭니다.

```python
from pathlib import Path
import subprocess
from google.colab import drive

drive.mount("/content/drive", force_remount=False)
clone_path = Path("/content/drive/MyDrive/DiffDodree")
repo_url = "https://github.com/SHlee-TDA/DiffDodree.git"
if not clone_path.exists():
    subprocess.run(["git", "clone", repo_url, str(clone_path)], check=True)
elif not (clone_path / ".git").is_dir():
    raise RuntimeError("DiffDodree 경로에 다른 폴더가 있습니다. 기존 내용을 확인하세요.")
else:
    origin = subprocess.check_output(
        ["git", "-C", str(clone_path), "remote", "get-url", "origin"], text=True
    ).strip()
    if origin != repo_url:
        raise RuntimeError(f"다른 저장소입니다: {origin}")
    print("기존 clone을 사용합니다. 아래 update 절차로 최신 commit을 확인하세요.")
subprocess.run(["git", "-C", str(clone_path), "log", "-1", "--oneline"], check=True)
```

private repository라면 GitHub 인증이 별도로 필요합니다. Colab의 Notebook 열기 권한과
runtime의 `git clone` 인증은 별개입니다. Notebook source에 credential을 저장하지 않습니다.

## 2. Tutorial 열기

[01 Notebook 열기](https://colab.research.google.com/github/SHlee-TDA/DiffDodree/blob/main/tutorials/01_data_inspection.ipynb).
나머지 5개 링크는 [README](../README.md)에 있습니다.

1. `Runtime → Change runtime type`에서 GPU를 선택합니다. 처음에는 제공되는 T4로 충분한지 확인합니다.
2. Cell 1을 실행하고 Google Drive 연결 승인을 완료합니다.
3. 출력이 `Project root: /content/drive/MyDrive/DiffDodree`, `Device: cuda`인지 확인합니다.
4. Cell 1을 다시 실행해 중복 mount나 dependency 재설치 문제가 없는지 확인합니다.
5. 각 Notebook에서 kernel을 재시작하고 위에서 아래로 실행합니다.

01/02는 CPU에서도 확인할 수 있습니다. CUDA 검증 시에는 아래 Cell로 backend를 명시적으로
확인합니다. GPU가 할당되지 않았다면 현재 조건을 통과한 것으로 기록하지 않습니다.

```python
import torch
assert runtime.backend == "cuda", runtime.as_dict()
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Runtime:", runtime.as_dict())
```

## 3. Pytest와 수동 확인

Cell 1 실행 후 임시 code cell에서 다음을 실행합니다. `requirements/dev.txt`에는 local
PyTorch dependency가 포함되므로 Colab 검증에는 필요한 pytest만 추가합니다.

```python
import os
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "pytest>=8,<10"], check=True)
test_env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "tests"],
    cwd=project_root, env=test_env, check=True,
)
```

현재 suite에서는 CUDA runtime에 대해 15 passed / 1 MPS skip을 예상합니다.
이 suite의 1-step Sampling integration은 CPU를 명시하므로, CUDA training·Sampling은
아래 Notebook 실행으로 추가 확인합니다.

| Notebook | 통과 조건 |
|---|---|
| 01 | 72개 마디, 피리 source 2개, 율명·duration table과 pitch plot 표시 |
| 02 | family Graph, 장 경계, 제1장–제4장 환입 표시 |
| 03 | `RUN_MODE = "smoke"`, CUDA에서 3-step 완료, finite Loss, Checkpoint 생성 |
| 04 | parameter 변경 후 생성, raw/projected 비교, export Button으로 MusicXML와 MIDI 생성 |
| 05 | 4개 조건의 소규모 실행 완료; 결과는 tutorial 검증용으로 표시 |
| 06 | metric table, MusicXML, CSV, summary manifest 저장 |

04의 export는 기본 Run All에서 실행하지 않습니다. **export Button을 직접 누르거나
`export_last_generation()`을 실행**해야 MIDI 경로까지 검증됩니다.
한글·율명이 네모로 보이거나 widget이 표시되지 않는 경우도 기록합니다.
이번 검증에서는 `full`로 전환하지 않습니다. 중간 Checkpoint 영구 저장과 resume를
확인한 다음 장시간 training으로 진행합니다.

## 4. Update와 기록

코드를 수정한 다음에는 Drive clone을 update하고 kernel을 재시작합니다.
실행 중간에는 checkout을 변경하지 않습니다. 작업 폴더와 GitHub Notebook의 commit을
맞추고, 재현용 실행에서는 `main` 대신 기록한 commit URL을 사용하면 좋습니다.

```python
subprocess.run(["git", "-C", str(project_root), "status", "--short"], check=True)
subprocess.run(["git", "-C", str(project_root), "pull", "--ff-only"], check=True)
subprocess.run(["git", "-C", str(project_root), "rev-parse", "HEAD"], check=True)
```

수정한 파일 때문에 pull이 실패하면 변경 내용을 먼저 확인합니다. validation 기록에는
commit SHA, Python/PyTorch/CUDA version, GPU, 선택한 Checkpoint SHA, pytest 결과와
Notebook별 성공 여부를 남깁니다. generated file은 Git에 올리지 않지만 Drive에서 보존합니다.

## API·CLI 대안

2026-09-12 확인 기준, Google은 [Colab REST API](https://developers.google.com/colab/api/reference/rest/v1beta/runtimes)와
[google-colab-cli](https://github.com/googlecolab/google-colab-cli)를 제공합니다.
API는 runtime 생성/조회와 Jupyter 접속 정보를 제공하고, CLI는 script·Notebook 실행을
지원합니다. API 이용에는 OAuth 인증이 필요하며 계정별 실제 접근·GPU 할당은 확인해야 합니다.
CLI/REST 인증 연결은 이 배포 작업에서 구성하지 않았습니다.

브라우저의 tutorial 사용성 검증을 마친 뒤, 반복 training·Ablation 자동화용으로 CLI를
검토합니다. 최초 로그인·권한 동의와 widget 표시 검증은 별도 절차로 남습니다.
