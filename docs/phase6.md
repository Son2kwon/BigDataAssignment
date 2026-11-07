# Phase 6. M2 LSTM — 상세 기록

- 담당: 차승준 (PLAN 10절). 코드·문서 작성은 Claude Code, 학습 실행은 차승준이 한다 (2026-11-07 지시: "코드·아키텍처를 다 만든 뒤 실행할 명령을 알려 주고, 실행 결과를 받아 분석·문서화")
- 브랜치 `phase/6-m2`. 실행 스크립트 `code/05_m2.py`, 공용 모듈 `code/afp/sequences.py`(6.1)·`code/afp/lstm.py`(6.2·6.3), `code/afp/evaluate.py`에 짝지은 비교·N 요약 추가
- 모든 수치 표기는 Phase 5와 같다: test는 70B 이슈 602개, 구간은 이슈 단위 부트스트랩 1,000회 95%, 기준선 비교는 train에서 고른 **K\*=40 대비** (제안서 K=35는 참고 행)

---

## 6.0 설계와 사전 등록 (2026-11-07, 학습 실행 전에 커밋)

### M1(Phase 5) 결과에서 가져온 설계 근거

| M1에서 확인한 것 (출처) | M2 설계 |
| --- | --- |
| 신호의 중심은 **횟수 피처**다. SHAP 1위는 모든 N에서 `n_edit_ok`이고, 상위에 `cmd_find_file`·`n_unique_files`가 있다 (`docs/phase5.md` 5.4) | 스텝마다 **그 시점의 M1 누적 집계 41개**를 입력에 넣는다 (`sequences.CUM_FEATURES`). 작은 데이터에서 LSTM이 정확한 횟수를 스스로 배우기 어렵다 [가정]. 이렇게 하면 M2는 매 시점 M1과 같은 정보를 갖고, M1과의 차이는 "순서 정보 + 모든 N을 함께 학습"에서 나온다. 효과는 절제 실험으로 잰다 |
| 가장 작은 XGBoost(깊이 3, 트리 200)가 24개 중 23개에서 골라졌고, 모델이 커질수록 OOF AUROC가 내려갔다 (5.3). 신호가 약하고 과적합이 쉽다. 실효 표본은 궤적 6.2만 개가 아니라 학습 이슈 2,989개다 | 작은 hidden(16~64), dropout, weight decay, **이슈 단위 검증 조각으로 조기 종료**, fold 모델마다 **seed 3개 앙상블** |
| 그리드 경계에서 골랐는데, 개발 실행이 test를 출력해서 그리드를 넓힐 수 없었다 (5.3, P5-D4) | 개발 실행(기본)은 **train OOF만** 계산하고, test는 `--final`에서 한 번 채점한다. 고른 hidden이 그리드 끝이면 **train OOF만 보고 자동으로 한 단계 넓힌다** (규칙을 아래에 미리 적음) |
| 비용 절약은 N=15·20에서 크고 N≤10에서는 작다 (5.6) | **모델 하나로 모든 N** (P6-D2): 시점 t마다 은닉 상태로 예측하고, 위치 log1p(t)를 입력에 넣는다. 설정 선택 기준은 N 6개 OOF AUROC의 평균이라 특정 N에 치우치지 않는다 |
| B1을 넣으면 전체 AUROC가 +0.01~0.03 오르고 이슈 내 AUROC는 그대로다 (5.6) | B1 포함/제외 **둘 다** (P6-D1). B1은 머리(head)에만 넣고 순환부에는 넣지 않는다 (h_t는 궤적 정보만 담는다) |
| N=10에서는 스텝 기준으로 이기지만 토큰 기준으로는 K\*와 같았다 (5.6) | 모든 운영점 표와 `n_summary.csv`에 절약 토큰과 K\* 대비 토큰 차이 [95%]를 넣는다 |
| "M2가 M1보다 나은가"를 따로 재야 한다 | 미리 정한 짝으로 **같은 부트스트랩 표본에서 M2 − M1 차이**를 잰다 (`evaluate.paired_stat`) |

### 사전 등록 (이 절은 결과를 본 뒤 고치지 않는다)

**모델** (`code/05_m2.py` `MODEL_SPECS`)
- `m2_lstm`: B1 없음 / `m2_lstm_b1`: B1(`b1_n`) 점수를 머리에 넣는다. 위치 t의 B1 = 그리드 {0, 3, 5, 8, 10, 15, 20}에서 t 이하 가장 큰 N의 `b1_n` 점수다. 평가 시점에서는 `b1_n(N)` 그대로다. train은 OOF, test fold k 모델에는 B1 fold k 점수를 넣는다 (P4-D7 ①). ②는 민감도로만 보고한다

**입력** (6.1): 스텝 t = 명령 임베딩(8차원) + 스텝 값 11개 + 누적 집계 41개(log1p). 정적 입력은 `issue_mentions_file`이다. 표준화는 fold 모델의 학습 조각에서만 한다. 궤적의 마지막 스텝은 입력에 넣지 않는다 (예측이 필요한 위치는 t < n_steps뿐)

**망** (`afp/lstm.py`): 1층 단방향 LSTM(`pack_padded_sequence`) → dropout → [h_t, 정적, (B1 logit 표준화)] → 선형 → 위치마다 logit

**손실**: 진행 중인 모든 위치(t = 1..min(n_steps−1, 20))의 BCE 평균. 클래스 가중은 없다 (M1에서 balanced가 OOF AUROC를 올리지 않았다)

**검증·조기 종료** (CLAUDE.md 누수 방지): fold k 모델은 fold k를 뺀 train으로만 학습한다. 그 안에서 이슈 단위로 1/8을 검증 조각으로 뗀다 (`StratifiedGroupKFold`, `io.SEED`). 검증 점수는 N_GRID 6개 시점 AUROC의 평균이다. patience 4, 최대 40 epoch이고, 최고 epoch의 가중치를 쓴다. fold k와 test는 학습·검증·표준화 어디에도 쓰지 않는다

**고정값**: AdamW lr 1e-3, weight decay 1e-4, 배치 512, 그래디언트 노름 1.0, 임베딩 8, seed 3개(`lstm.seeds_for(fold)` = `io.SEED`와 fold 번호로만 정함), 확률 평균

**그리드와 선택**: hidden {16, 32, 64} × dropout {0.3, 0.1} = 6개를 모델마다 따로 고른다. 선택 기준은 **train OOF AUROC의 N 6개 평균** 최대이고, 최고값에서 0.002 안이면 앞쪽(작은 hidden, 같은 hidden이면 큰 dropout)을 고른다 (M1·B1과 같은 동률 폭)

**그리드 확장 (train OOF만)**: 고른 hidden이 지금 그리드의 최솟값 또는 최댓값이면, 사다리 {8, 16, 32, 64, 128, 256}의 바깥 값 하나를 dropout 두 개와 함께 더하고 다시 고른다. 최대 2번까지다. 그 뒤에도 끝이면 개발 실행이 `[주의]`를 출력하고 멈춰서 보고한다

**절제 실험 (train OOF만, test를 채점하지 않는다)**: `m2_lstm`에서 고른 설정으로 (a) 누적 집계를 뺀 `m2x_nocum` (명령 + 스텝 값만, 원래 제안서의 M2), (b) 순환을 뺀 `m2x_norec` (스텝마다 같은 층, 시점 t의 입력만 봄 = "N을 입력으로 넣은 단일 M1형 모델"에 가깝다). base − 절제의 OOF AUROC 차이로 "누적 집계의 몫"과 "순서(순환)의 몫"을 본다

**평가 (`--final`)**: 03·04와 같은 `afp/evaluate.py` 함수를 쓴다. 판별 지표는 fold 평균 점수(P4-D2), 운영점은 test를 fold 모델별로 채점해 평균한다(P4-D6). 운영점은 train OOF 3.5%로 고르고, 판정은 test 손실 95% 상한 ≤5%다. 모든 N(3~20)을 보고한다

**M2 vs M1 짝지은 비교** (`m2_vs_m1.csv`): `m2_lstm` ↔ `m1_xgb`, `m2_lstm_b1` ↔ `m1_xgb_b1` (같은 B1 조건에서 가장 강한 M1). N마다 AUROC·이슈 내 AUROC 차이와, 두 모델이 각자 train에서 고른 운영점(단독·결합)의 손실·절약 스텝·절약 토큰 차이를 같은 재표본(시드 `io.SEED`)에서 잰다. M1 운영점은 같은 함수로 이 머신의 M1 예측 파일에서 다시 고른다

**좋은 N** (train OOF만, `good_n.csv`): P5-D5가 재검토 중이므로 두 규칙을 같이 낸다. ① `auroc_first` = M1 규칙 (OOF AUROC 최대에서 0.005 안의 N 중 train 결합 절약 스텝 최대) ② `saved_first` = train 결합 절약 스텝 최대

**해석 규칙**: 여러 N·정책 중 하나만 통과한 결과는 강조하지 않고 N 전체의 경향으로 쓴다. M2가 M1보다 나쁘거나 비슷해도 그대로 보고한다

**실행 순서**: 개발 실행(train OOF만) → 출력의 `[주의]`(그리드 경계, epoch 상한)를 확인 → 문제가 없으면 `--final` 한 번. `--final` 결과를 본 뒤에는 설정을 바꾸지 않는다

### 설계 확인용 진단 (2026-11-07, train만, 사전 등록 전)

고정값(lr)이 학습되는 범위에 있는지만 확인했다. 전체 train에서 fold 0을 빼고 학습해 fold 0(train 이슈)에 적용했다. seed는 1개, 조기 종료 없이 30 epoch, GPU다. test는 쓰지 않았다. 스크립트는 저장소 밖 임시 파일이다.

| hidden, lr | 최고 검증 epoch | 검증 점수(N 평균 AUROC) 최고 | fold 0 AUROC N=3 / 10 / 20 |
| --- | --- | --- | --- |
| 16, 1e-3 | 23 | 0.720 | 0.627 / 0.675 / 0.708 |
| 32, 1e-3 | 12 | 0.724 | 0.625 / 0.674 / 0.707 |
| 32, 3e-3 | 9 | 0.729 | 0.631 / 0.673 / 0.711 |
| 64, 1e-3 | 12 | 0.730 | 0.634 / 0.679 / 0.715 |
| 128, 1e-3 | 12 | 0.722 | 0.635 / 0.677 / 0.703 |

- 검증 점수는 epoch 9~12쯤 최고점을 찍고 천천히 내려간다. 그래서 patience 4로 조기 종료하는 설정이 맞다. hidden·lr에 따른 차이는 0.01 안이다. 그래서 lr은 1e-3으로 고정하고 hidden만 그리드에 둔다
- 같은 설정을 두 번 학습하면 검증 곡선이 같다 [확인]
- 참고 (같은 수치가 아님): M1 train OOF AUROC(train 전체, fold 5개)는 LR N=10 0.685, XGB N=10 0.699다 (`results/phase5/discrimination.csv`). 위 값은 fold 0 하나, seed 1개라 바로 비교할 수 없다

---

## 6.1 시퀀스 입력 (2026-11-07)

### 한 일
- `code/afp/sequences.py`: 궤적 × 위치 t(1..20) 입력 배열 `SequenceData`를 만든다 (`build`, `load`). B1 위치 채우기 `b1_matrix`, 예측 파일 형식 변환 `preds_frame`, 작업 프로세스용 배열 캐시 `save_arrays`·`load_arrays`도 있다
- 입력 정의

| 묶음 | 내용 | 출처 |
| --- | --- | --- |
| 명령 | `command_name` 28종 인덱스 (패딩 28) → 임베딩 | `features.step_values` |
| 스텝 값 11개 (`STEP_FEATURES`) | `is_error`, `edit_failed`, `edit_ok`(편집이고 거부 안 됨), `obs_len_log`, `has_file`, `new_file`(그 파일이 처음 열린 스텝), `repeat_prev`, `repeat_any`, `run_len_log`, `opens_issue_file`, `position_log`(log1p t) | `features.step_values` + 같은 값의 함수 |
| 누적 41개 (`CUM_FEATURES`) | 시점 t의 M1 피처(`features.m1_features(N=t)`)에서 스텝 t 값과 겹치는 `last_*` 30개·`obs_len_last`와 정적 `issue_mentions_file`을 뺀 것, log1p | `features.m1_features` 그대로 |
| 정적 1개 | `issue_mentions_file` | 같은 함수 |

- 길이 L = min(n_steps − 1, 20). 시점 t 진행 중 ⇔ n_steps > t ⇔ t ≤ L이라, 예측이 필요한 위치가 정확히 1..L이다. 입력 스텝이 없는 궤적(n_steps = 1)은 뺀다 (어떤 시점에도 평가 대상이 아니다)

### 결정과 근거
- **누적 집계를 입력에 넣는다** (P6-D5): 6.0 표 첫 행. 정의는 M1 함수를 그대로 써서 M1과 M2가 같은 정의에서 나온다 (P5-D3 "M2는 같은 스텝 단위 값을 쓴다"). 새 정보(예: `ai_len`, 관측 원문)는 넣지 않는다. M1과 비교하기 위해서다
- **마지막 스텝을 입력에서 뺀다**: 시점 N 예측은 위치 N 출력이라 마지막 스텝이 들어가도 단방향 LSTM에서는 새지 않는다. 하지만 학습에 쓰이지 않는 스텝이므로 아예 넣지 않는 쪽이 검증하기 쉽다
- **B1 위치 채우기**: `b1_n`은 N 그리드 시점에만 있다. 위치 t에는 t 이하 가장 큰 그리드 시점의 점수를 쓴다. 시점 N에 진행 중인 궤적은 그 이하 모든 그리드 시점에도 진행 중이라 점수가 항상 있다. 평가 위치에서는 `b1_n(N)` 그대로다
- 기각: 시점마다 다른 B1 대신 `b1_static`(N=0 모델 하나). P4-D3이 `b1_n`을 주 결과로 정했고, M1+B1과 같은 입력이어야 비교가 깔끔하다

### 검증 (`code/tests/test_sequences.py`, 테스트 함수 9개·케이스 11개 통과 [확인])
- `test_sequences_ignore_steps_after_n`: N 뒤 스텝을 바꿔도 위치 1..N의 명령·스텝 값·누적·정적 입력이 같다 (N=3, 5, 10, 15, seed 3개). 앞 N스텝만 넣고 만든 입력과도 같다
- `test_last_step_is_not_input`: 마지막 스텝을 바꿔도 입력이 같고, 패딩 위치 값이 0·패딩 인덱스다
- `test_cumulative_inputs_equal_m1_features`: 위치 t 누적 입력 = log1p(M1 피처 N=t)
- `test_b1_matrix_forward_fill`, `test_new_file_and_edit_ok`, `test_preds_frame_matches_running_trajectories`(시점마다 `metrics.align` 통과), `test_array_cache_roundtrip`, `test_sequences_read_no_leaky_columns`
- `test_real_sequences_ignore_later_steps`: 실데이터 train 표본 300궤적. 앞 10스텝만 넣어 만든 입력이 전체 입력의 위치 1..10과 같다
- 실데이터 크기: train 62,311궤적 중 입력이 있는 궤적 62,225개 (n_steps = 1인 86개 제외), 시퀀스 만들기 1.7초 [확인, 6.0 진단 실행]

### 산출물
- `code/afp/sequences.py`, `code/tests/test_sequences.py`
- 캐시 `data/processed/m2/seq_train/` (git 밖, 개발 실행이 만든다)

### 계획과 다른 점
- PLAN 6.1은 "명령 임베딩 + 스텝 수치 피처"였다. 여기에 **누적 집계 41개**를 더했다 (proposal-changes.md 19번). 원래 형태는 절제 실험 `m2x_nocum`으로 같이 잰다

### 미결
- 없음
