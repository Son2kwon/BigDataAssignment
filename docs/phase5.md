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

---

## 5.0 운영점 루프 이동 (P5-D2, 2026-11-06)

### 한 일
- `03_baselines.py`에 있던 운영점 선택·적용 루프를 `afp/evaluate.py`의 모델 무관 함수로 옮겼다. 입력은 모델마다 `(preds, fold_preds)` 한 쌍이다: `preds` = train OOF + test 예측 (traj_id, N, p_fail), `fold_preds` = test fold 모델별 예측 (traj_id, N, fold, p_fail, OOF 모델만. 없으면 None)
  - `evaluate.operating_points(models, sets, costs, k_star)`: 단독·결합 정책 선택(train OOF, 3.5%), test 적용(P4-D6 fold별 적용 평균), 민감도(`MAX_LOSS_GRID` × 목적 함수 2), fold별 train 범위
  - `evaluate.cap_rows`: K\*·제안서 K=35 참고 행(`cap_proposal`)과 K 민감도
  - `evaluate.bootstrap_tasks`, `cap_bootstrap_tasks`, `boot_task`, `cap_stat`: 부트스트랩 작업 (판별 + 운영점 + K\* 대비 짝지은 차이)
  - `evaluate.attach_ci`, `operating_table`, `sensitivity_table`, `discrimination_ci`, `judged_and_saves_more`, `loss_ci_width`, `frame`, `parallel`, `test_preds`
  - 상수 `OP_COLUMNS`, `SENS_COLUMNS`, `OP_SCHEMA`, `K_PROPOSAL`(=35)도 `evaluate`로 옮겼다
- `03_baselines.py`는 이 함수들을 부르기만 한다 (B1 학습·오라클·이전성 분석·완료 기준 검사는 그대로)
- 테스트 1개 추가 (`test_operating_points_loop_matches_direct_selection`): 합성 데이터에서 루프가 고른 (τ, K)가 `select_from_curves`를 직접 부른 값과 같고, test 행이 `test_point`와 같고, fold 점수가 있는 모델에만 fold 범위가 붙고, 부트스트랩 → 운영점 표가 `OP_COLUMNS` 형식으로 구간·판정을 채우는지

### 결정과 근거
- 모델 입력 형식을 "예측 파일 두 개"로 정했다. M1~M3 스크립트가 B0·B1과 같은 함수를 그대로 부르고, 학습 방식(OOF인지 규칙인지)은 `fold_preds`가 있는지로만 갈린다
- 부트스트랩 재표본은 평가 집합의 이슈와 `io.SEED`로만 정해진다. 그래서 다른 스크립트(04_m1.py)에서 돌려도 K\* 행의 구간이 Phase 4와 같다 (5.3에서 확인)
- 기각: B0·B1 비교 행까지 M1 스크립트에서 다시 계산. Phase 4 결과 파일을 읽어 붙이는 쪽이 계산을 반복하지 않고, K\* 행이 같은지로 두 스크립트가 같은 재표본을 썼는지도 확인할 수 있다

### 검증
- `uv run --no-sync python code/03_baselines.py --force` (2분 59초, 12코어 M4 Pro): 완료 기준 C1~C13 통과, **`results/`·`preds/`·`splits.parquet` 96개 파일이 리팩터 전(5.0a 두 번째 실행)과 바이트 단위로 같다** [확인]. `b1_grid.csv`의 플랫폼 차이는 리팩터 전과 같은 값이라 되돌렸다
- `uv run --no-sync pytest`: 112개 통과

### 산출물
- `code/afp/evaluate.py` (운영점 루프 절), `code/03_baselines.py` (얇아짐), `code/tests/test_operating_point.py`

### 계획과 다른 점
- 없음 (PLAN 5.0 그대로)

### 미결
- 없음

---

## 5.1 피처 (2026-11-06)

### 한 일
- `code/afp/features.py`: **스텝 단위 값**(`step_values`)과 **궤적 단위 M1 피처**(`m1_features`)를 나눴다. 궤적 피처는 스텝 단위 값을 앞 N스텝(`step_idx <= N`)에서 집계한 것뿐이라, M2(Phase 6)가 같은 스텝 값을 시퀀스로 쓰면 두 모델의 입력 정보가 같은 정의에서 나온다
- 피처 **73개** (+ B1 포함 모델은 `b1_score` 1개 = 74개). 산출물 `data/processed/features/m1_n.parquet` (traj_id, N, 피처 73개, 334,789행 = 70B train+test 궤적의 N별 진행 중 궤적 합. N=3 74,181 / 5 71,175 / 8 62,780 / 10 55,518 / 15 40,098 / 20 31,037). 계산 1.2초
- train 분포 표 `results/phase5/feature_dist.csv` (N 6개 × 피처 73개: 평균, 표준편차, 5·50·95 분위수, 0이 아닌 비율, 성공·실패 평균)

### 피처 사전

| 이름 | 정의 (앞 N스텝 집계) | 타입 | 근거 | 사용한 스텝 단위 값 |
| --- | --- | --- | --- | --- |
| `cmd_<명령>` (28개) | 명령 종류별 횟수. 명령 = Phase 1 사전 26개 + `other` + `none` | 횟수 | 제안서 §3 "명령 종류별 빈도" | `command_name` |
| `n_error` | 에러 관측 스텝 수 (= B0 `b0_err` 점수) | 횟수 | 제안서 "에러 누적", B0 규칙 재사용 | `is_error` |
| `n_edit_failed` | 편집 거부 스텝 수 (= B0 `b0_editfail` 점수) | 횟수 | B0 규칙 재사용 | `edit_failed` |
| `n_edit_ok` | 거부되지 않은 `edit` 횟수 | 횟수 | PLAN 5.1 "`edit` 횟수". 전체 `edit` 횟수는 `cmd_edit`, 성공한 편집을 따로 둔다 | `command_name`, `edit_failed` |
| `n_repeat_consec` | 바로 앞 스텝과 같은 명령인 스텝 수 (연속 반복) | 횟수 | 제안서 "같은 명령 반복" (연속) | `repeat_prev` |
| `n_repeat_any` | 앞 스텝 어딘가와 같은 명령인 스텝 수 (전체 반복) | 횟수 | 같은 명령 반복 (전체) | `repeat_any` |
| `max_run_len` | 같은 명령 연속 구간의 최대 길이 | 횟수 | 연속 반복의 세기 (같은 명령을 3번 이상 연달아 내는 "막힘") | `run_len` |
| `n_unique_files` | 열린 파일 경로의 고유 개수 | 횟수 | 제안서 "열어 본 고유 파일 수" | `open_file` |
| `n_steps_no_file` | 열린 파일이 없는 스텝 수 | 횟수 | 파일을 열지 않고 탐색만 하는 구간 | `open_file` |
| `obs_len_mean`, `_max`, `_min`, `_last` | 관측 문자 수의 평균·최대·최소·마지막 스텝 값 | 실수 | 제안서 "관측 길이 통계". 마지막 값은 "마지막 스텝" 후보 | `obs_len` |
| `issue_mentions_file` | 이슈 본문에 파일 경로 언급이 있다 (0/1) | 0/1 | 아래 `issue_file_opened`의 0이 "언급 없음"인지 "안 열었음"인지 가른다 | (이슈 본문) |
| `issue_file_opened` | 이슈 본문에 언급된 파일을 한 번이라도 열었다 (0/1) | 0/1 | PLAN 5.1 추가 후보. 이슈 본문은 운영 중에도 보인다 (정답 파일과 다름, 5.5) | `opens_issue_file` |
| `n_issue_file_steps` | 언급된 파일이 열려 있던 스텝 수 | 횟수 | 같은 근거, 머문 정도 | `opens_issue_file` |
| `last_is_error`, `last_edit_failed` | 마지막(N번째) 스텝이 에러·편집 거부 | 0/1 | PLAN 5.1 추가 후보 "마지막 스텝의 에러 여부" | `is_error`, `edit_failed` |
| `last_cmd_<명령>` (28개) | 마지막 스텝의 명령 종류 원-핫 | 0/1 | PLAN 5.1 추가 후보 "마지막 스텝의 명령" | `command_name` |
| `b1_score` (B1 포함 모델만) | B1(`b1_n`) 점수: train OOF, test fold k (P4-D7 ①) | 실수 | 이슈 난이도 기준선 위에 궤적 피처가 더하는 정보를 재려고 | (B1 예측 파일) |

모든 피처는 Float64로 저장한다. 결측은 없다 (진행 중 궤적은 N스텝을 모두 가진다. 관측이 없으면 `obs_len`=0).

### 결정과 근거
- **비율 피처(횟수 / N)는 넣지 않는다.** N별로 모델을 따로 학습하고(D2), 시점 N에 진행 중인 궤적은 모두 정확히 N스텝을 봤으므로 같은 모델 안에서 비율은 횟수에 상수를 곱한 값이다. 로지스틱 회귀(표준화 후)와 트리에서 정보가 같다. N을 피처로 넣은 단일 모델(보조 실험)을 만들 때만 의미가 있다
- 같은 이유로 **같은 N 안에서 다른 피처의 선형 변환인 값을 뺐다**: 고유 명령 수(= N − `n_repeat_any`), 관측 길이 합(= N × `obs_len_mean`). 처음 만든 75개에서 2개를 빼 73개
- **에러·편집 실패는 B0 규칙의 정의를 그대로 쓴다**: `steps.parquet`의 `is_error`·`edit_failed`(Phase 1 `parse.is_error`, `parse.EDIT_FAILED`)를 다시 계산하지 않고 읽는다. 그래서 `n_error`는 `b0_err` 점수와 같다 (테스트 `test_step_definitions_reuse_b0_rules`)
- **같은 명령** = 명령 원문(코드 블록 내용)의 앞뒤 공백을 지운 문자열이 같은 것. 명령 이름만 같은 것(예: 다른 파일 `open`)은 반복으로 세지 않는다. 이름 수준 반복은 `cmd_*` 횟수가 이미 담는다
- **이슈 언급 파일** (`features.FILE_MENTION`) [가정: 휴리스틱]: 이슈 본문에서 확장자가 코드·설정 파일(py, cfg, toml, json, yml, txt, rst, md, c, h, js 등)인 토큰. 경로가 있는 언급(`pkg/a.py`)은 열린 경로의 끝이 `/pkg/a.py`인지, 이름만 있는 언급(`utils.py`)은 열린 파일 이름이 같은지로 맞춘다. URL 안의 경로는 앞에 `/`가 있어 빠진다. N=10 진행 중 train 궤적의 22.4%가 언급이 있는 이슈의 궤적이다. 이름만 같은 다른 파일(`__init__.py` 등)도 맞는 것으로 센다
- 마지막 스텝 명령은 원-핫 28개로 넣었다. 묶지 않은 이유: 명령 사전이 이미 빈도 0.1% 이상으로 정해져 있고, XGBoost는 쓸모없는 원-핫을 거의 쓰지 않는다. LR은 L2 규제가 있다
- 기각: 관측 원문의 키워드 피처(예: "Traceback" 횟수 외의 단어). 에러 판별 규칙(B0)과 겹치고 M3(원문 Transformer)의 몫이다

### 검증
- 손으로 센 예제 2개(스텝 단위 값, 궤적 피처), 이슈 언급 매칭 규칙, B0 규칙 재사용, `check_features`가 1:1이 아니거나 무한값이 있으면 멈추는지 (`code/tests/test_features.py`)
- 누수 테스트는 5.2

### 피처별 train 분포 요약 (N=10 진행 중 train 궤적 46,167개, `results/phase5/feature_dist.csv`)

성공·실패 평균 차이가 큰 피처 (실패 쪽이 크면 ↑):

| 피처 | 평균 | 0이 아닌 비율 | 성공 평균 | 실패 평균 |
| --- | --- | --- | --- | --- |
| `n_error` | 1.99 | 78.9% | 1.72 | 2.03 ↑ |
| `n_edit_failed` | 0.74 | 32.6% | 0.50 | 0.77 ↑ |
| `n_edit_ok` | 1.76 | 83.2% | 2.13 | 1.70 |
| `n_repeat_consec` | 0.32 | 16.5% | 0.16 | 0.35 ↑ |
| `n_repeat_any` | 1.01 | 58.0% | 0.88 | 1.03 ↑ |
| `max_run_len` | 1.30 | 100% | 1.16 | 1.32 ↑ |
| `n_steps_no_file` | 1.20 | 46.1% | 0.88 | 1.25 ↑ |
| `cmd_python` | 1.07 | 62.5% | 1.33 | 1.03 |
| `cmd_ls` | 0.73 | 38.0% | 0.58 | 0.76 ↑ |
| `cmd_rm` | 0.06 | 5.3% | 0.12 | 0.05 |
| `issue_file_opened` | 0.082 | 8.2% | 0.114 | 0.076 |
| `last_is_error` | 0.25 | 24.5% | 0.20 | 0.25 ↑ |
| `last_edit_failed` | 0.12 | 12.2% | 0.07 | 0.13 ↑ |
| `last_cmd_rm` | 0.043 | 4.3% | 0.109 | 0.032 |
| `obs_len_last` | 2,021 | 100% | 1,791 | 2,058 ↑ |

- 성공 궤적은 편집이 받아들여지고(`n_edit_ok`), 재현 스크립트를 돌리고(`cmd_python`), 그 스크립트를 지우는(`cmd_rm`, `last_cmd_rm`) 경향이 있다. SWE-agent의 "재현 → 수정 → 확인 → 정리" 흐름과 부합한다 [가정: 해석]
- 실패 궤적은 에러·편집 거부·같은 명령 연속·파일을 열지 않는 스텝이 많다. 차이는 평균의 10~50% 수준이고 분포가 크게 겹친다 (B0 규칙의 전체 AUROC가 0.5 근처였던 것과 부합)
- `cmd_submit`, `cmd_none`, `last_cmd_submit`, `last_cmd_none`은 거의 항상 0이다 (진행 중 궤적은 아직 제출하지 않았다). 그대로 둔다 (상수 피처는 두 학습기에 영향이 없다)
- `obs_len_*`는 꼬리가 길다 (N=10 `obs_len_max` q95 6,298, 최대는 훨씬 큼). LR은 `log1p` 후 표준화한다 (5.3)

### M2(Phase 6)가 쓸 스텝 단위 API

```python
from afp import features
values = features.load_step_values(max_n, trajs)   # traj_id, step_idx + 아래 값 (앞 max_n스텝, (traj_id, step_idx) 정렬)
# 또는 이미 읽은 스텝으로: features.step_values(steps, mentions, traj_issue)
```

| 컬럼 | 타입 | 정의 | 시점 |
| --- | --- | --- | --- |
| `command_name` | str (28종, `features.COMMAND_NAMES`) | Phase 1 명령 사전 | 스텝 t |
| `is_error`, `edit_failed` | bool | B0 규칙 (Phase 1 `parse`) | 스텝 t 관측 |
| `obs_len` | int | 관측 문자 수 (없으면 0) | 스텝 t 관측 |
| `open_file` | str/null | 관측 끝 상태 줄의 열린 파일 | 스텝 t 관측 |
| `repeat_prev`, `repeat_any` | bool | 바로 앞 / 앞 어딘가와 같은 명령 원문 | 스텝 1..t |
| `run_len` | int | 이 스텝에서 끝나는 같은 명령 연속 길이 | 스텝 1..t |
| `opens_issue_file` | bool | 열린 파일이 이슈 본문 언급 파일 | 스텝 t + 이슈 본문 |

- 모든 값은 스텝 1..t만 보고 정해진다. 시퀀스를 길이 N으로 자르면 그 시점의 M1 피처와 같은 정보다 (`m1_features`는 이 값의 집계일 뿐)
- M2는 `command_name`을 임베딩, 나머지를 수치 입력으로 쓰면 된다 (`obs_len`은 `log1p` 권장, M1 LR과 같게). B1 점수를 넣는다면 P4-D7 규칙(fold k ← B1 fold k)을 따른다

### 산출물
- `code/afp/features.py`, `code/tests/test_features.py`, `code/tests/feature_helpers.py`
- `data/processed/features/m1_n.parquet` (git 밖), `results/phase5/feature_dist.csv`

### 계획과 다른 점
- 피처 목록은 PLAN 5.1 그대로이고, 추가한 것은 `n_edit_ok`, `max_run_len`, `n_steps_no_file`, `issue_mentions_file`, `n_issue_file_steps`, `obs_len_last`·`_min`이다 (모두 같은 스텝 단위 값의 집계)

### 미결
- `[결정 필요]` 이슈 언급 파일 매칭 규칙 (휴리스틱). 선택지: (a) 지금 규칙 유지 (권장: 운영 중 계산 가능하고 단순), (b) 이름만 있는 언급은 흔한 이름(`__init__.py`, `setup.py` 등)을 빼고 맞춤. (b)는 SHAP 결과를 보고 정하면 테스트 정보가 들어가므로, 바꾼다면 train 분포만 보고 정한다
