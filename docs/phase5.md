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

> 메모 (5.2): 이 절을 쓴 뒤 `issue_file_mentions`가 이슈 13개의 "고칠 파일 목록" 꼬리를 지우도록 고쳤다. `issue_mentions_file`·`issue_file_opened`·`n_issue_file_steps` 세 피처가 궤적 11개에서 바뀌었고, 아래 표의 반올림 값은 그대로다 (5.2 검증).

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

---

## 5.2 누수 테스트 (2026-11-06)

### 한 일
`code/tests/test_features_leakage.py` (7개, 합성 3개 시드 포함):
- **N 뒤 스텝** (`test_features_ignore_steps_after_n`, 시드 3개 × N 4개): N 뒤 스텝의 명령·관측 길이·에러·편집 실패·열린 파일을 모두 바꾸거나, 지우거나, 엉뚱한 스텝(999번)을 덧붙이거나, 행 순서를 섞어도 시점 N 피처가 바이트 단위로 같다. 기존 `test_error_count_ignores_steps_after_n`(B0)의 틀을 모든 피처로 넓혔다
- **궤적 길이**: `n_steps`는 "N에 진행 중인가"만 정하고 값에는 들어가지 않는다 (N 뒤 길이를 100 늘려도 같다)
- **금지 컬럼**: `features.STEP_INPUT_COLUMNS`에 `io.LEAKY_COLUMNS`가 없고 `io.check_safe_columns`를 통과한다. 입력에 `exit_status`·`generated_patch`·`eval_logs`·`target`을 넣어도 결과가 같다 (읽지 않는다). 실제 로드(`features.load_step_values`)는 `io.scan_steps`·`io.load_trajectories`로 컬럼을 명시해 누수 컬럼을 요청하면 멈춘다. 정적 점검(`test_leakage.py`의 `scan_code`)도 `features.py`·`04_m1.py`를 검사한다 (허용 목록 밖 파일에 누수 컬럼 이름이 있으면 실패)
- **정답 파일 목록 꼬리**: 합성 본문에 `baselines.HINT` 꼬리를 붙여도 언급이 같다. 실데이터 13개 이슈 모두 꼬리를 지운 본문과 언급이 같다
- **실데이터** (데이터가 있을 때만): 모든 N에서 피처 행이 진행 중 궤적과 1:1, 결측·무한값 0 (`features.check_features`). 스텝 21개 이상인 궤적 300개 표본은 앞 N스텝만 읽어 계산해도(N=3, 10) 전체 스텝으로 계산한 값과 같다

### 결정과 근거
- 스텝 단위 값이 "스텝 1..t만 본다"는 조건을 집계 뒤 결과로 검사한다. 스텝 단위(`run_len` 등 누적 값)와 집계(`last_*`, 정렬) 둘 다 깨지면 이 테스트가 잡는다
- 이슈 본문은 운영 중에도 보이므로 입력으로 허용한다. 단 **이슈 13개 본문 끝의 "고칠 파일 목록"(정답 패치 정보, Phase 4 B1 입력 정리와 같은 꼬리)은 언급 추출 전에 지운다** (`features.issue_file_mentions`가 `baselines.strip_hint`를 먼저 적용). 이 테스트를 쓰면서 처음 만든 5.1 코드가 이 꼬리를 지우지 않아 13개 이슈(모두 train)에서 정답 파일이 "언급 파일"로 들어갈 수 있었음을 찾아 고쳤다. 5.3 학습은 고친 뒤 피처로만 했다

### 검증
- `uv run --no-sync pytest code/tests/test_features_leakage.py`: 8개 통과. `test_leakage.py`(정적 점검 포함) 통과
- 꼬리 제거 전후 피처 차이 (실데이터, 다시 만든 `m1_n.parquet` 비교) [확인]: 세 피처만 바뀐다. `issue_mentions_file` 52행(궤적 9개), `issue_file_opened`·`n_issue_file_steps` 46행(궤적 11개). 나머지 70개 피처는 모든 행이 같다. `feature_dist.csv`는 이 세 피처의 행만 바뀌었다 (N=10 `issue_file_opened` 평균 0.08153 → 0.08134)

### 산출물
- `code/tests/test_features_leakage.py`

### 계획과 다른 점
- 5.1 코드의 누수 하나(정답 파일 목록 꼬리)를 5.2에서 고쳤다 (위)

### 미결
- 없음

---

## 5.5 사후 분석: 정답 파일 열람 (피처 아님, 2026-11-06)

### 한 일
- `afp/leakage.py`에 5.5 절을 추가했다 (`patch_files`, `load_patch_files`, `solution_proxy`, `opened_files`, `hint_files`). 원본 패치 컬럼은 이 모듈만 읽는다 (`io.read_raw(..., allow_leaky=True)`)
- 실행 스크립트 `code/04b_solution_files.py` (4초). 출력 `results/phase5/solution_file_open.csv`, `solution_file_open_by_n.csv`, `solution_file_hint_check.csv`, `solution_files_summary.json`. M1 학습(04_m1.py)과 독립이고, 출력은 결과 표뿐이다 (피처·예측 입력에 들어가지 않는다)
- 테스트 `code/tests/test_solution_files.py` 4개 (패치 파일 추출 규칙, 고칠 파일 목록 파싱, 근사의 leave-one-out, 열람 매칭과 첫 스텝)

### 결정과 근거
- **정답 패치가 데이터에 없다** [확인]: 원본 컬럼은 `instance_id, model_name, target, trajectory, exit_status, generated_patch, eval_logs`이고, `eval_logs`에는 예측 패치·테스트 패치 적용 기록과 pytest 출력만 있다. 이 머신의 `data/`에는 swe-agent 궤적만 있다 (SWE-bench 이슈 데이터 없음)
- 그래서 **근사**를 썼다 [가정]: 궤적 t의 정답 파일 근사 = 같은 이슈의 **다른** 성공 궤적들이 낸 `generated_patch`가 고친 기존 비테스트 파일의 합집합. 새로 만든 파일·지운 파일·테스트 경로(`tests/`, `test_*.py`, `*_test.py`, `conftest.py`)는 뺀다. 자기 패치를 빼서(leave-one-out) 성공 궤적이 "자기가 고친 파일을 열었다"로 자동으로 맞는 것을 막는다
- 열람 = 관측 끝 상태 줄의 열린 파일(`open_file`)이 `/근사 경로`로 끝나는 스텝이 있다 (`open`·`edit`·`goto`·`scroll`·`create` 뒤에 열린 파일이 바뀐다). N별 표는 첫 열람 스텝이 N 이하인지
- 기각: 정답 패치를 HF Hub에서 바로 받기 (CLAUDE.md: 로컬 parquet만, revision 고정). 받으려면 `data/download.sh`에 데이터셋과 revision·체크섬을 추가해야 한다 → 미결

### 검증 / 결과 [확인, `results/phase5/solution_file_open*.csv`]

| 범위 | 성공 궤적: 열람 비율 (궤적 수) | 실패 궤적: 열람 비율 (궤적 수) |
| --- | --- | --- |
| 데이터 카드 (제안서 §5 인용) | 83% | 40% |
| 근사, 전체 모델 | 99.7% (13,313) | 84.3% (22,018) |
| 근사, 70B 평가 집합 (train + test) | 99.7% (12,393) | 84.3% (20,478) |

시점 N별 (70B, 시점 N 진행 중 궤적, 앞 N스텝 안에서 처음 열었는지):

| N | 3 | 5 | 8 | 10 | 15 | 20 |
| --- | --- | --- | --- | --- | --- | --- |
| 성공 | 41.9% | 69.7% | 87.5% | 91.1% | 93.7% | 96.0% |
| 실패 | 31.3% | 54.2% | 70.9% | 75.4% | 79.1% | 79.3% |
| 차이 (%p) | 10.6 | 15.5 | 16.6 | 15.7 | 14.6 | 16.7 |

- **데이터 카드 값(83% / 40%)은 재현되지 않았다.** 방향(성공이 더 자주 연다)은 같고, 앞 N스텝에서도 N=5부터 15%p 안팎의 차이가 계속된다
- 재현되지 않은 이유 [가정]:
  1. 근사가 정답 파일이 아니다. 성공한 패치들은 서로 같은 파일을 고치는 경향이 강하고, 성공 궤적은 그 파일을 고치려고 반드시 열었으므로 성공 쪽 비율이 거의 100%가 된다
  2. **근사를 만들 수 있는 궤적은 "다른 성공 궤적이 있는 이슈"뿐이다.** 성공이 없는 이슈의 궤적 44,705개(이슈 2,829개, 대부분 실패)가 빠진다. 이 이슈들의 실패 궤적은 정답 파일을 찾지 못했을 가능성이 높아서, 실패 쪽 비율이 크게 올라간다. 데이터 카드는 정답 패치로 모든 궤적을 셌을 것이다
  3. "열람" 정의가 다를 수 있다 (데이터 카드의 정의는 확인하지 못했다)
- 근사 대조: 본문에 "고칠 파일 목록"(정답 파일)이 있는 이슈 13개 중 성공 궤적이 있는 이슈는 1개(`pylint-dev__astroid-1196`)뿐이고, 그 이슈의 근사 파일 1개가 목록 파일과 같다 (1/1). 대조 표본이 너무 작아 근사의 정확도는 알 수 없다
- 이 값은 **피처로 쓰지 않는다** (정답 패치 정보, CLAUDE.md 누수 규칙). M1의 `issue_file_opened`는 이슈 본문 언급 파일 기준이라 다른 값이다 (N=10 train: 성공 11.4% / 실패 7.6%, 5.1)

### 산출물
- `code/afp/leakage.py` (5.5 절), `code/04b_solution_files.py`, `code/tests/test_solution_files.py`
- `results/phase5/solution_file_open.csv`, `solution_file_open_by_n.csv`, `solution_file_hint_check.csv`, `solution_files_summary.json`

### 계획과 다른 점
- PLAN 5.5는 "성공 83% / 실패 40% 재현"이었다. 정답 패치가 없어 근사로 바꿨고 재현되지 않았다

### 미결
- `[결정 필요]` 정답 패치 출처. 선택지: (a) 근사 결과를 한계와 함께 그대로 보고 (권장: 이 분석은 피처가 아니라 발표의 동기 설명용이고, 방향과 N별 차이는 보인다), (b) SWE-agent 궤적 데이터셋의 이슈 원본(정답 패치 포함) 데이터셋을 `data/download.sh`에 revision·SHA-256 고정으로 추가하고 다시 계산. (b)는 다운로드 경로 변경이라 팀 결정이 필요하다

---

## 5.3 학습과 평가 (2026-11-06)

### 한 일
- `code/04_m1.py`: 모델 4종 × N 6개 = **24개 설정**을 N마다 따로 학습한다 (D2). `m1_lr`(로지스틱 회귀), `m1_xgb`(XGBoost), 각각 B1 포함(`_b1`)·제외
- 학습: `oof.oof_predict`로 train 5-fold OOF 점수와 test fold 모델별 점수. `oof_predict`에 두 옵션을 더했다
  - `test_for_fold(k, test)`: fold k 모델에 넣을 test 행을 따로 준다. B1 포함 모델의 fold k 모델은 B1 fold k test 점수로 채점한다 (P4-D7 ①). 테스트 `test_test_for_fold_gives_fold_specific_input`
  - `keep_models`: fold 모델을 남긴다 (② 민감도와 SHAP에 쓴다)
- B1 입력 (P4-D7): train = `preds.load_feature("b1_n", "train")` (OOF), test fold k 모델 = `load_feature("b1_n", "test", fold=k)` (①). ② 민감도 = `load_feature_other_folds("b1_n", k)`의 j≠k 4개로 M1 fold k를 각각 채점. 평균 test 점수(③)는 쓰지 않는다 (fold별 입력이 실제로 다른지 완료 기준 C5로 확인)
- 예측 저장: `preds.write`(train OOF + test 5개 fold 평균, 판별 지표용) + `preds.write_test_folds`(fold별, 운영점용 P4-D6). `preds/m1_{lr,lr_b1,xgb,xgb_b1}.parquet` + `.test_folds.parquet`
- 평가: 5.0의 공용 함수 그대로 (`evaluate.discrimination_table`, `operating_points`, `bootstrap_tasks`, `operating_table`). K\*=40과 K=35 참고 행도 같은 함수로 다시 계산하고, Phase 4 결과와 같은지 확인한다 (완료 기준 C8)
- 비교 표는 Phase 4 결과 파일(`results/phase4/discrimination*.csv`, `operating_points.csv`)을 읽어 M1 행과 합친다

### 결정과 근거
- **학습기 설정**
  - LR: 피처를 `log1p` 한 뒤 표준화(fold 안에서 학습), L2, lbfgs (`tol=1e-6`, `max_iter=5000`). 피처가 모두 0 이상의 횟수·길이·확률이고 관측 길이 꼬리가 길어서 log 변환 후 표준화했다. BLAS 스레드 1개 (B1과 같은 이유)
  - XGBoost: `hist`, 학습률 0.05, subsample 0.8, colsample_bytree 0.8, min_child_weight 1, λ=1, `n_jobs=1`, `random_state=io.SEED`. 작업 단위로 병렬화하고 모델 안에서는 스레드 1개라 결과가 결정적이다 (다시 학습한 점수가 그리드 점수와 비트 단위로 같은지 완료 기준 C3)
- **그리드** (각 8개, 순서 = 단순한 것부터. `results/phase5/m1_grid.csv`):
  - LR: C ∈ {0.01, 0.1, 1, 10} × class_weight ∈ {없음, balanced}
  - XGBoost: max_depth ∈ {3, 6} × n_estimators ∈ {200, 600} × 불균형 처리 ∈ {없음, balanced(`scale_pos_weight` = 성공 수 / 실패 수)}
  - 선택: (모델, N)마다 **train OOF AUROC** 최대. 최고값에서 0.002 안이면 그리드 앞쪽(단순한) 설정 (B1과 같은 규칙)
  - 클래스 불균형 처리는 그리드에 넣었다. 실패가 다수(N=10 진행 중 train의 86%)라 balanced는 실패 쪽 무게를 줄인다. 점수 척도가 바뀌어도 τ는 같은 OOF 척도에서 고르므로 운영점 규칙과 충돌하지 않는다
- **한계 (그리드 선택의 낙관)**: 같은 5-fold OOF로 하이퍼파라미터를 고르고 그 OOF 점수로 운영점·좋은 N도 고른다. 그리드가 작아서(8개) 낙관은 작을 것으로 보지만 [가정] 크기를 따로 재지 않았다 (중첩 교차검증 안 함). test 지표에는 영향이 없다 (test는 고른 뒤 한 번 채점)
- **한계 (그리드 경계)**: XGBoost는 24개 중 23개가 그리드에서 가장 작은 모델(깊이 3, 트리 200개, 불균형 처리 없음)을 골랐고(N=20 B1 제외만 balanced), 모델이 커질수록 OOF AUROC가 내려간다 (N=10 `m1_xgb_b1` 0.717 → 깊이 6·트리 600개 0.693). 최적점이 그리드 경계에 있으므로 더 작은 모델(깊이 2, 트리 100개 등)이 더 나을 수 있다. LR은 C에 거의 둔감하다 (N=10 OOF AUROC 차이 < 0.001)
  - **그리드는 넓히지 않았다.** 이 경계 결과를 확인한 개발 실행(`--skip-bootstrap --skip-shap`)이 test 판별 지표도 출력했고 그것을 읽었다 (N=10 test AUROC 약 0.70). 그 뒤에 그리드를 바꾸면 "테스트 결과를 본 뒤 규칙을 바꾸지 않는다"(CLAUDE.md)를 지켰다고 말할 수 없다. 그래서 그리드는 처음 코드에 쓴 그대로 두고 경계 문제는 미결로 넘긴다
- **N별 개별 모델** (D2). 피처 정의가 N과 무관하게 같고 N마다 하이퍼파라미터를 고른다
- **libomp**: XGBoost macOS 휠은 OpenMP 런타임(`libomp.dylib`)이 시스템에 있어야 로드된다. 이 머신에 없어서 `brew install libomp`로 설치했다 (저장소 변경 없음). Linux에서는 보통 `libgomp`가 이미 있다

### 검증
- `uv run --no-sync python code/04_m1.py` (12코어 M4 Pro **4분 28초**: 피처 1.4초, 그리드 OOF 192개 70초, 고른 설정 다시 학습 15개 5초, 운영점 선택·적용 40초, 부트스트랩 25개 × 1,000회 127초, ② 민감도 6초, SHAP 17초) → 완료 기준 C1~C10 통과 [확인] (`results/phase5/summary.json`의 `criteria`)
  - C3 고른 설정을 다시 학습한 OOF·fold 점수가 그리드 점수와 비트 단위로 같다
  - C5 B1 포함 모델의 test 입력이 fold마다 다르다 (③ 평균 점수가 아님)
  - C8 이 스크립트가 다시 계산한 K\*=40·K=35 행(부트스트랩 구간 포함)이 `results/phase4/operating_points.csv`와 같다 (같은 평가 집합·시드 → 같은 재표본)
- 재실행 바이트 비교: 5.6 절
- 개발 실행 순서 (투명성): 부트스트랩·SHAP을 뺀 개발 실행 1회 → 그리드 경계 확인 (위 한계) → 그리드를 바꾸지 않고 전체 실행. 개발 실행과 전체 실행의 예측·판별 값은 같다 (같은 코드, `preds.write`가 다시 쓰지 않음)

### 결과: 고른 하이퍼파라미터 (`results/phase5/m1_grid.csv`)

| 모델 | N=3 | 5 | 8 | 10 | 15 | 20 |
| --- | --- | --- | --- | --- | --- | --- |
| `m1_lr` | C=0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| `m1_lr_b1` | C=0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| `m1_xgb` | d3·200 | d3·200 | d3·200 | d3·200 | d3·200 | d3·200·balanced |
| `m1_xgb_b1` | d3·200 | d3·200 | d3·200 | d3·200 | d3·200 | d3·200 |

LR은 모두 class_weight 없음 (balanced와 OOF AUROC 차이 ≤0.001이라 동률 규칙으로 앞쪽). LR C=0.01도 동률 규칙 (C에 따른 차이 < 0.001)

### 결과: 판별 지표 (5.6에 정리)

### 산출물
- `code/04_m1.py`, `code/afp/oof.py` (`test_for_fold`, `keep_models`), `code/tests/test_oof.py`
- `data/processed/features/m1_n.parquet`, `data/processed/preds/m1_*.parquet`·`.test_folds.parquet` (git 밖)
- `results/phase5/`: `m1_grid.csv`, `discrimination.csv`, `discrimination_ci.csv`, `operating_points.csv`, `operating_points_sensitivity.csv`, `cost_curves/`(48개), `b1_input_sensitivity.csv`, `discrimination_compare.csv`, `operating_points_compare.csv`, `loss_ci_width_compare.csv`, `loss_ci_width_summary.csv`, `good_n.csv`, `summary.json`

### 계획과 다른 점
- 없음 (PLAN 5.3 그대로). 하이퍼파라미터 그리드는 작업 지시의 "작은 그리드"로 각 8개

### 미결
- `[결정 필요]` XGBoost 그리드 경계 (위 한계). 선택지: (a) 지금 그리드 결과를 주 결과로 두고 경계 문제를 한계로 보고 (권장: test를 본 뒤 그리드를 바꾸지 않는다는 규칙을 지킨다), (b) 더 작은 모델(깊이 2, 트리 50~100개)까지 넓힌 그리드를 **보조 실험**으로 돌려 train OOF AUROC만 비교하고 test 결과는 주 결과를 바꾸지 않는다. (b)를 하더라도 주 표는 (a)
- `[결정 필요]` 좋은 N 규칙의 동률 폭 `GOOD_N_TIE = 0.005` (아래 5.6). OOF AUROC가 N에 따라 단조 증가해서 이번에는 동률 폭이 결과를 바꾸지 않았다

---

## 5.4 SHAP (2026-11-06)

### 한 일
- `m1_xgb`, `m1_xgb_b1`의 N=5, 10, 20 (6개 설정)에서 fold 모델 5개 각각에 `shap.TreeExplainer`를 적용했다. 대상은 test 궤적이고, fold k 모델에는 그 모델이 채점에 쓴 입력(B1 포함이면 B1 fold k 점수)을 넣었다. 값의 단위는 실패의 log-odds
- 표: `results/phase5/shap_importance.csv` (피처별 평균 |SHAP| = fold 5개 평균, fold 사이 최소·최대, fold별 순위 범위, 피처 값이 0이 아닐 때의 평균 SHAP), `shap_stability.csv` (상위 10개의 fold 사이 안정성)
- 그림: `results/phase5/figures/` PNG 24개. 설정마다 요약 그림 1개(상위 15개, beeswarm)와 상위 3개 피처의 의존 그림. 그림의 SHAP 값은 표본마다 fold 5개 값의 평균이다 (= 5개 모델 출력 평균의 분해). 점의 색은 피처 값이고, `b1_score`는 fold 평균 값으로 칠했다

### 결과: 상위 피처 (평균 |SHAP|, test. 괄호는 fold 5개 사이 순위 범위)

| 순위 | `m1_xgb` N=5 | N=10 | N=20 | `m1_xgb_b1` N=5 | N=10 | N=20 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `n_edit_ok` 0.181 (1~3) | `n_edit_ok` 0.313 (1) | `n_edit_ok` 0.369 (1) | `b1_score` 0.451 (1) | `b1_score` 0.411 (1) | `n_edit_ok` 0.332 (1~2) |
| 2 | `cmd_find_file` 0.174 (1~3) | `cmd_find_file` 0.149 (2~4) | `n_unique_files` 0.180 (2~7) | `n_edit_ok` 0.135 (2) | `n_edit_ok` 0.292 (2) | `b1_score` 0.313 (1~2) |
| 3 | `obs_len_min` 0.156 (2~4) | `obs_len_min` 0.143 (2~3) | `cmd_python` 0.175 (2~7) | `cmd_find_file` 0.102 (3~6) | `obs_len_mean` 0.108 (3~5) | `n_unique_files` 0.164 (3~6) |
| 4 | `cmd_open` 0.131 (2~5) | `obs_len_mean` 0.124 (2~5) | `n_steps_no_file` 0.167 (2~6) | `obs_len_min` 0.101 (3~5) | `n_edit_failed` 0.101 (3~7) | `cmd_python` 0.143 (3~10) |
| 5 | `obs_len_mean` 0.090 (5~6) | `n_edit_failed` 0.102 (4~8) | `obs_len_min` 0.152 (4~6) | `cmd_open` 0.094 (3~8) | `obs_len_min` 0.098 (3~5) | `obs_len_mean` 0.132 (4~7) |
| 6 | `n_edit_failed` 0.079 (5~7) | `cmd_python` 0.093 (4~9) | `obs_len_mean` 0.149 (3~7) | `obs_len_max` 0.089 (4~7) | `cmd_find_file` 0.086 (3~7) | `n_steps_no_file` 0.132 (4~8) |

**fold 사이 안정성** (`shap_stability.csv`): 6개 설정 모두 fold 모델 5개의 피처 순위 Spearman 상관이 최소 0.91, 평균 0.94. 전체 상위 10개 중 각 fold의 상위 10개에 들어가는 수는 8~10개, 5개 fold 모두의 상위 10개에 공통인 피처는 7~9개다. 1위 피처는 6개 설정 중 4개에서 모든 fold가 같다 (N=5 `m1_xgb`는 1~3위 셋이 fold마다 자리를 바꾸고, N=20 `m1_xgb_b1`은 `n_edit_ok`와 `b1_score`가 1·2위를 바꾼다)

### 해석: "초반 실패 신호" (인과가 아니다)
SHAP은 **이 모델이 점수를 낼 때 어떤 피처에 기댔는지**를 보여 줄 뿐이다. 피처를 바꾸면 성공 확률이 바뀐다는 뜻이 아니고, 피처끼리 서로 얽혀 있다 (예: 편집 성공 수는 "파일을 찾았는가"와 같이 움직인다). 아래는 모델이 쓰는 패턴의 기술이다 [가정: 해석].

- **편집이 받아들여졌는가(`n_edit_ok`)가 모든 N에서 가장 강한 궤적 신호다.** 값이 작으면 실패 쪽(N=10 그림에서 약 +0.1~+0.8 log-odds), 값이 크면 성공 쪽이다. 반대로 편집 거부(`n_edit_failed`)·마지막 스텝 에러(`last_is_error`)가 많으면 실패 쪽이다. "초반에 고치려는 시도가 문법 검사를 통과하지 못한다"가 실패 신호다
- **탐색 단계에서 머무는가**: N=5·10에서 `find_file`·`open`을 아직 하지 않은 궤적(값 0)은 실패 쪽이다 (모델이 탐색의 시작 자체를 성공 신호로 쓴다). N=20에서는 `n_steps_no_file`(파일 없이 보낸 스텝)이 많고 `n_unique_files`가 많은(여러 파일을 헤맨) 궤적이 실패 쪽이다
- **재현 스크립트 흐름**: `cmd_python`(재현·확인 실행)·`last_cmd_rm`(재현 스크립트 정리)이 크면 성공 쪽이다 (5.1 분포와 부합)
- **관측 길이**: `obs_len_mean`이 길면 실패 쪽이다 (긴 에러 출력·긴 파일 목록) [가정]
- **B1 포함 모델**: 이슈 난이도 점수 `b1_score`가 N=5·10에서 1위이고, N=20에서는 `n_edit_ok`와 같은 크기로 내려온다. 궤적이 길어질수록 궤적 자체의 신호가 이슈 난이도만큼 중요해진다. 궤적 피처의 순위는 B1을 넣어도 거의 같다
- 반복(`n_repeat_consec`, `max_run_len`)은 분포상 실패 쪽 차이가 있었지만(5.1) 상위권이 아니다. N=20에서 `n_repeat_any`가 7위 정도로 들어온다. 이슈 파일 언급(`issue_file_opened`)은 상위 10위 밖이다

### 산출물
- `results/phase5/shap_importance.csv`, `shap_stability.csv`, `figures/*.png` (24개)

### 계획과 다른 점
- 없음

### 미결
- 의존 그림은 피처마다 상위 3개만 저장했다. 다른 피처가 필요하면 `shap_analysis`의 `DEPENDENCE_TOP`을 바꿔 다시 그린다 (SHAP은 test 해석용이라 다시 그려도 선택 규칙에 영향이 없다)

