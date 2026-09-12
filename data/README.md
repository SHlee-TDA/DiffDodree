# 양청도드리 피리 source

현재 배포에는 `configs/yangcheongdodeuri.yaml`이 참조하는 피리 MusicXML 2개만 포함합니다.

- `02 Cheon-Yangcheongdodeuri_musicxml_1-6/`: 제1–6장, 55개 마디
- `02 Cheon-Yangcheongdodeuri_musicxml_7/`: 제7장, 17개 마디

마디 구분과 MusicXML의 pitch 대응은 연구자의 국악 전문가 협업 맥락에서 제공되었습니다.
Python pipeline은 이 표현을 유지하고 시김새나 tuning을 새로 추정하지 않습니다.
제1–6장은 4/4, 제7장은 6/4의 전사상 meter를 사용합니다. 이것이 서양식 강세 구조를
부여한다는 뜻은 아닙니다.

현재 배포의 SHA-256은 `artifacts/release_manifest.json`에 기록합니다.
악보 수정 시 frozen measures, vocabulary, Graph의 일치 여부를 다시 확인해야 합니다.
다른 악기의 원본 파일과 이전 실험은 원래 작업 폴더에 유지하지만 현재 배포에는 포함하지 않습니다.

전사 악보의 이용·재배포 조건은 저장소 소유자에게 확인합니다.
