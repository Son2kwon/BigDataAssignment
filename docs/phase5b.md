# Phase 5b. M1 개선 보조 실험 ① 마지막 k스텝 창 피처 — 상세 기록

- 담당: 차승준. 코드·문서 작성은 Claude Code, 학습 실행은 차승준이 한다 (Phase 6과 같은 방식)
- 브랜치 `phase/5b`. 실행 스크립트 `code/04e_m1_window.py`, 피처 `code/afp/features.py` (`window_features`), 테스트 `code/tests/test_features_window.py`
- 설계·결정은 PLAN "Phase 5b~5e" 절과 11절 P5b-D1~D11 (사전 등록 2026-11-12). 이 문서는 그 설계를 링크하고 구현·결과만 적는다
- 표기: 기준선 비교는 train에서 고른 **K\*=40 대비**다. 이 Phase의 수치는 모두 **train OOF**다 (test는 읽지 않는다)

---

## 5b.0 공통 규칙 (5b~5e, P5b-D1~D3)

5c~5e 문서는 이 절을 링크한다.

1. **보조 실험이다.** 공식 M1(Phase 5) 결과·파일은 바꾸지 않는다. Phase 하나가 방법 하나이고, 각 Phase에서 고른 묶음(S1~S3)이 다음 Phase의 출발점이다
2. **개발은 train OOF로만 한다.** 5b~5e 스크립트는 test 궤적의 피처도 만들지 않는다. 5e를 마친 뒤 `--final`을 **한 번** 실행한다. `--final`의 결론은 "공식 → S1 → S2 → S3 → S4" 흐름으로만 쓰고, 나머지 묶음은 부록 표에 둔다. 보고는 P6-D11 형식(공식과 나란히, 보조 표시)이고, 공식 M1과는 짝지은 부트스트랩으로 비교한다
3. **모델·그리드·N** (P5b-D2): `m1_xgb`, `m1_xgb_b1`. 묶음마다 `XGB_GRID` 8개에서 공식과 같은 규칙(train OOF AUROC 최대, GRID_TIE 0.002 안이면 그리드 앞쪽)으로 다시 고른다. N은 N_GRID (3, 5, 8, 10, 15, 20) 전부. 학습기·고정값은 `04_m1.py`를 import해서 그대로 쓴다
4. **다음 묶음 선택** (P5b-D3, 5e 제외): train OOF 이슈 내 AUROC(쌍 가중)의 N∈{8, 10, 15, 20} 평균이 가장 큰 묶음. 0.002 안이면 피처가 적은 묶음 (같으면 앞 번호). 고르는 모델은 `m1_xgb`다 (P5b-D12, 아래)
5. **보고**: 모든 묶음을 학습하고 보고한다. 판별 지표(전체 AUROC, 이슈 내 AUROC 쌍 가중·이슈 평균)와 운영점 표(절약 스텝, 절약 토큰, K\* 대비 스텝·토큰 차이 [95%])를 낸다. 결과가 나빠도 그대로 쓴다
6. **OOF로 고른 후보 수**를 Phase마다 적는다 (α·γ·λ·t·K·묶음). 5b~5e 전체에서 선택이 여러 단계라 S1~S4의 train OOF 지표는 낙관적이다 (12절 리스크)
7. **train OOF 구간** (P5b-D13): train 이슈 단위 부트스트랩 1,000회 (시드 `io.SEED`)다. 운영점(τ·K)은 원래 데이터에서 고른 값을 고정하고 재표본마다 다시 고르지 않는다. 그래서 선택 불확실성은 구간에 들어 있지 않다. 모든 묶음·모델·N이 같은 재표본을 쓴다 (짝지은 차이)
8. 하지 않는 것: P5(관측 내용) 피처 (P5b-D11, 팀 결정으로 범위에서 뺌)

## 5b.1 설계·구현 (2026-11-13, 학습 실행 전)

### 한 일

- `afp/features.py`에 창·최근성 피처를 더했다 (`window_features`, `window_columns`, `proportional_k`, `WINDOW_KS`, `RECENCY_FEATURES`). 기존 M1 피처(`m1_features`)와 스텝 단위 값(`step_values`, M2가 같이 씀)은 바꾸지 않았다. `check_features`에 검사할 열(`columns`, N마다 다르면 함수)을 받는 인자를 더했다. 기본값은 예전과 같다
- 창 피처 (P5b-D5, k마다 7개, 창 = 스텝 N−k+1..N, k ≥ N이면 열을 만들지 않음): `n_error_w{k}`, `n_edit_failed_w{k}`, `n_edit_ok_w{k}`, `n_repeat_consec_w{k}`, `n_repeat_any_w{k}`, `n_new_files_w{k}`, `obs_len_mean_w{k}`
  - `n_new_files_w{k}`: 궤적에서 `open_file` 값이 처음 나온 스텝이 창 안인 파일 수 (= 창 시작 전에 한 번도 열리지 않은 파일 중 창 안에서 처음 연 파일)
  - `n_repeat_any_w{k}`: 스텝 값 `repeat_any`를 그대로 더한다. 창 이전 명령과 같아도 센다
- 최근성 피처 (P5b-D6, 3개, N=3에도 있음): `steps_since_edit_ok`, `steps_since_new_file` (사건이 없으면 N), `cur_error_run` (스텝 N에서 끝나는 연속 에러 길이)
- `code/04e_m1_window.py`: 묶음 ①~⑤와 확인용 비례 k 묶음을 `m1_xgb`·`m1_xgb_b1` × N 6개 × 그리드 8개로 학습한다 (train 5-fold OOF). 같은 (모델, N, 열)은 한 번만 학습한다 (N=3의 ②~④·비례 k = ①, N=5의 ③ = ① 등). 이어서 묶음별 판별 지표, train 운영점(공용 `evaluate.candidate_curves`·`metrics.select_from_curves`, 손실 ≤3.5%, 단독·결합), 부트스트랩 구간, S1 선택, 비례 k 확인, 가설 확인을 한다
  - 묶음 피처 수 (N ≥ 8): ① 73 / ② 80 / ③ 80 / ④ 87 / ⑤ 90 / 비례 k 80. N=5: ② ④ 80, ③ = ① 73, ⑤ 83. N=3: ⑤만 76, 나머지 73
  - `--smoke`: train 이슈 1/10, 그리드 앞 2개, 부트스트랩 20회. `data/processed/phase5b_smoke/`에만 쓴다
- 테스트 `code/tests/test_features_window.py` (PLAN 5b 테스트 ①~⑤ + 손 계산): 손으로 센 8스텝 궤적, ① 스텝 N 이후를 무작위로 바꾸기·지우기·덧붙이기·행 섞기·궤적 길이 늘리기에도 값이 같음, ② N ≤ k이면 그 k의 열이 없음, ③ 창 경계 (스텝 N−k → 0, N−k+1 → 1, N → 1, k=3·5), ④ 사건 없으면 `steps_since_*` = N, ⑤ 새 피처를 붙여도 `check_features` 통과 (없어야 할 열을 요구하면 결측으로 걸림), 비례 k 값

### 결정과 근거 (기각한 안 포함)

| ID | 결정 | 근거 / 기각한 안 |
| --- | --- | --- |
| P5b-D12 | S1을 고르는 모델은 `m1_xgb` (B1 없음). `m1_xgb_b1`로 골랐을 때의 묶음도 `selection.csv`·`summary.json`에 같이 내고, 다르면 `[결정 필요]`로 보고한다 | PLAN 규칙이 모델을 정하지 않았다. 창 피처는 궤적 피처이고 선택 지표가 이슈 내 AUROC라, B1(이슈 안에서 같은 값)이 없는 모델이 피처의 몫을 바로 보여 준다. P6 종합 권장 주 모델도 M1 XGB다. 기각: 두 모델 평균 (규칙이 복잡해지고 B1 모델 쪽 차이가 작아 정보가 적음) `[결정 필요]` 되돌릴 수 있음 |
| P5b-D13 | train OOF 운영점·판별 구간은 train 이슈 단위 부트스트랩 (1,000회, τ·K 고정). 묶음 − ① 짝지은 차이도 같은 재표본에서 잰다 | PLAN이 OOF 운영점 표에 "K\* 대비 토큰 차이 [95%]"를 요구한다. 기존 부트스트랩 함수는 test 행만 받아서, 같은 `metrics.bootstrap`·`evaluate.apply_point`로 train 버전을 스크립트에 만들었다. 한계: 선택 불확실성이 빠져 구간이 좁다 (5b.0 7) |
| P5b-D14 | 비례 k 확인(P5b-D8)의 비교 대상은 창이 하나인 고정 k 묶음 ②·③ 중 선택 지표가 큰 쪽. 차이가 0.002를 넘으면 `[결정 필요]`로 보고한다 | 비례 k 묶음도 창이 하나라 ②·③과 피처 수가 같다. ④·⑤(창 두 개, 최근성)와 비교하면 창 정의가 아니라 피처 수 차이를 잰다. S1 대비 차이도 같이 낸다 `[결정 필요]` 되돌릴 수 있음 |
| (구현) | test 궤적은 피처도 만들지 않는다. `--final`은 5e를 마친 뒤 별도 스크립트로 한다 | CLAUDE.md "개발 중 test 지표 계산 금지". 5b~5e를 한 번에 평가하는 `--final`이라 5b 스크립트에 넣지 않았다 |
| (구현) | ①을 새로 학습하고, 이 머신의 공식 M1 OOF(`preds/m1_xgb*.parquet`, 2026-11-07 Linux 실행)와 비트 단위로 같은지 확인한다 (완료 기준 C3) | XGBoost 결과가 플랫폼마다 달라서 (P6-D10) 묶음 비교는 같은 머신의 ①과 해야 한다. 커밋된 `results/phase5/`는 Mac 값이라 참고 표(`oof_reference.csv`)에만 둔다 |

### 검증

- `uv run --no-sync pytest`: 186개 통과 [확인] (새 테스트 10개 포함)
- `--smoke` 전체 경로 통과 (16초, 완료 기준 C1~C6 통과) [확인]
- 실데이터: 공식 피처 73개가 `04_m1.py` 캐시(`features/m1_n.parquet`)의 train 행과 같다, 새 피처는 모든 N에서 진행 중 궤적과 1:1이고 결측·무한값이 없다 [확인]
- 실행 시간 측정 (train 전체, 1프로세스): OOF 1회(fold 5개) 깊이 3·트리 200 2.6초, 깊이 6·트리 600 8.2초 / 운영점 1.7초 / 부트스트랩 10회 0.9초 (N=3) [확인]. 전체 = 학습 설계 56개(중복 제거) × 설정 8개 = OOF 448회 + 부트스트랩 72개 × 1,000회라 24코어로 10~20분 [가정]

### 산출물 (실행 후)

- `results/phase5b/`: `grid.csv`(묶음·모델·N·설정별 OOF AUROC), `discrimination_oof.csv`(판별 + ① 대비 차이 [95%]), `operating_points_oof.csv`(운영점, K\* 대비 차이 `d_*` [95%], ① 대비 차이 `do_*` [95%], fold 범위), `selection.csv`(S1 선택 표), `prop_k_check.csv`, `feature_dist.csv`(새 피처 분포), `oof_reference.csv`(다른 Phase의 train OOF 판별, 참고), `summary.json`(완료 기준 C1~C6)
- `data/processed/phase5b/<model>__<bundle>.parquet`: 묶음별 train OOF 점수 (git 밖, 5c 출발점)

### 계획과 다른 점

- 없음. 정하지 않았던 세부를 P5b-D12~D14로 정했다

### 미결

- `[결정 필요]` P5b-D12 (S1을 고르는 모델), P5b-D14 (비례 k 비교 대상). 둘 다 권장안으로 진행한다

## 5b.2 결과

(학습 실행 후 작성)
