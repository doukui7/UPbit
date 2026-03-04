# 프로젝트 작업 규칙

- 사용자가 명시적으로 요청하지 않으면 UI 레이블/제목/설명 문구를 영어로 바꾸지 않는다.
- 기존에 표시되던 한국 종목명(코드 + 한글명) 표기를 임의로 삭제하거나 축소하지 않는다.
- 기존 동작을 리팩터링할 때는 화면 표시 요소(텍스트/라벨/종목명 병기)가 유지되는지 먼저 확인한다.
- 요청한 것 이외의 내용은 임의로 수정하지 않는다.
- 요청 범위를 벗어나는 수정이 필요하면 작업 전에 반드시 사용자 확인을 받는다.
- 한글로 작성된 화면 문구/레이블/설명/가이드를 영어로 변경하지 않는다.
- 소스 파일은 EUC-KR(CP949) 인코딩으로 저장한다.

# Project Rules

- Do not change UI labels, titles, subtitles, or help text to English unless the user explicitly asks.
- Do not remove Korean instrument name displays (code + Korean name) unless explicitly requested.
- Do not change anything outside the user's explicit request without permission.
- If out-of-scope changes are required, get user confirmation before editing.
- Set source file encoding to EUC-KR (CP949).
