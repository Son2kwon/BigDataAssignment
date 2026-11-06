# Phase 5. M1 피처 모델 + SHAP — 상세 기록

브랜치 `phase/5-m1`. 계획은 [PLAN.md](../PLAN.md) Phase 5 절, 규칙은 [CLAUDE.md](../CLAUDE.md). 단계마다 6개 항목(한 일 / 결정과 근거 / 검증 / 산출물 / 계획과 다른 점 / 미결)을 쓴다.

---

## 5.0a summary.json 반올림 (2026-11-06)

### 한 일
- `afp/io.py`에 `round_floats`·`write_summary`를 추가했다. 중첩 dict·list 안의 float(numpy float 포함)를 소수 10자리(`io.SUMMARY_DIGITS`)로 반올림하고, `ensure_ascii=False, indent=2, default=str`로 쓴다. NaN·inf·int·bool·문자열은 그대로 둔다
- `summary.json`을 쓰는 세 곳(`01_parse.py`, `02_split.py`, `03_baselines.py`)이 모두 이 함수를 쓴다. Phase 5 스크립트도 같은 함수를 쓴다
- 테스트 `code/tests/test_io.py` 3개: 중첩·타입 처리, 2.9e-12 차이가 반올림 후 사라짐, 파일 형식(끝 줄바꿈 옵션)

### 결정과 근거
- 자릿수 10: PR #11의 머신 간 τ 차이가 최대 2.9e-12였다. 10자리면 그 차이가 사라지고, 결과 해석에 쓰는 자릿수(4~6자리)보다 충분히 많다
- 공용 함수 위치는 `afp/io.py` (모든 스크립트가 이미 import하는 경로·로드 모듈)
- 끝 줄바꿈: phase1·2 파일은 원래 끝 줄바꿈이 없고 phase4는 있다. 반올림 외의 차이를 만들지 않으려고 `trailing_newline` 옵션으로 기존 형식을 유지했다
- 기각: CSV까지 같은 함수로 묶기. CSV는 이미 `float_precision`으로 자릿수를 정해서 쓰고 있고, `b1_grid.csv`의 차이(소수 다섯째 자리)는 반올림으로 없앨 수 없는 크기다

### 검증
- 적용 후 첫 실행 (`01_parse.py` 검증 모드 → `02_split.py` → `03_baselines.py`, 합계 3분 18초) [확인]:
  - `results/phase1/summary.json`, `results/phase2/summary.json`: 바이트 단위로 같다 (10자리를 넘는 float가 없었다)
  - `results/phase4/summary.json`: float 407개가 바뀌었고 **모두 `round(이전 값, 10)`과 정확히 같다**. 구조·키 순서·그 밖의 값(65개 float 포함)은 같다. 최대 차이 4.99e-11
  - 나머지 `results/`·`preds/`·`splits.parquet` 95개 파일은 바이트 단위로 같다. 예외는 `b1_grid.csv`의 알려진 플랫폼 차이(고르지 않은 C ≥ 1 행 6개, 소수 다섯째 자리)뿐이라 커밋하지 않고 되돌렸다
- 두 번째 실행 (같은 세 스크립트): 첫 실행 결과와 `results/`·`preds/`·`splits.parquet` 96개 파일이 **모두 바이트 단위로 같다** [확인]. `b1_grid.csv`도 두 실행 사이에서는 같다 (이 머신 안에서는 결정적이고, 커밋된 파일과의 차이는 플랫폼 차이)
- `uv run --no-sync pytest`: 111개 통과 (기존 108 + 새 3)

### 산출물
- `code/afp/io.py` (`SUMMARY_DIGITS`, `round_floats`, `write_summary`), `code/tests/test_io.py`
- `results/phase4/summary.json` (반올림만 바뀜)

### 계획과 다른 점
- 없음

### 미결
- 반올림 경계에 걸린 값은 드물게 머신 사이에서 마지막 자리가 갈릴 수 있다 (PLAN 5절에 기록). 그런 diff도 플랫폼 차이로 보고 커밋하지 않는다
