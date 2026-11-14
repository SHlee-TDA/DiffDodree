# 웹 데모 (GitHub Pages)

빌드 단계가 없는 정적 사이트입니다. `.github/workflows/pages.yml`이 `main`에 push될 때 이 폴더를 그대로 배포합니다.

| 파일 | 내용 |
|---|---|
| `index.html`, `css/style.css` | 페이지 |
| `js/core.js` | 편집 거리, 유형 소속, 유형 보행, 격자, 역과정·재생성, 다양체 배치, MusicXML (Python과 같은 규칙) |
| `js/worker.js` | onnxruntime-web으로 모형을 돌리는 Web Worker |
| `js/manifold.js` | 프레이즈 다양체 <i>M</i>(캔버스): 대표 마디 MDS로 놓은 엽, 닮은 유형 사이의 목, 엽마다의 국소 좌표(τ, η) 위를 음표 단위로 지나는 곡선 γ(t) |
| `js/viz.js`, `js/audio.js`, `js/app.js` | 프레이즈 그래프 <i>G</i>, λ 그림, 합성음, 화면 |
| `models/*.onnx`, `data/*.json` | `python scripts/export_web.py`로 만든 파일 |
| `tests/parity.mjs` | JS 포트가 Python 값(`data/parity.json`)과 같은지 확인: `node site/tests/parity.mjs` |

로컬에서 보기: `python -m http.server -d site 8000` 후 <http://localhost:8000>.
외부 라이브러리: onnxruntime-web 1.20.1, Verovio 4.3.1 (jsDelivr), Noto·STIX Two 글꼴 (Google Fonts).
