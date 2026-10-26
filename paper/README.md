# 논문 원고

한국음악연구(한국국악학회) 투고를 목표로 한 초고입니다. 투고 규정 원문은 아직 확인하지 못했습니다.
서식 규정(분량, 글꼴, 각주, 참고문헌 양식, 파일 형식)은 확인 후 변환 단계에서 반영합니다.

- `manuscript.md`: 내용 중심 초고(Pandoc Markdown)
- `drafting_brief.md`: 초안 집필 착수 메모. 확인된 결과, 중심 주장 후보, 절별 근거, 쓰기 전에 정할 것
- 결과 근거: `../docs/results/2026-10_memorization/README.md`

## 변환

```bash
pandoc manuscript.md -o manuscript.docx
```

`.docx`를 한컴오피스(웹 포함)에서 열어 `.hwp`로 저장하는 것이 현재 계획입니다.
학회가 `.docx`/PDF를 받는지 먼저 확인합니다. 규정 확인 후에는 `--reference-doc`으로
서식 템플릿을 지정할 수 있습니다.
