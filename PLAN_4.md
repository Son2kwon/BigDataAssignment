# Phase 4 상세 계획 — 기준선 B0·B1 + 운영점 선택(3.5)

[PLAN.md](PLAN.md) Phase 4(4.1~4.4)와 Phase 3에서 미룬 3.5(운영점 선택)를 처음부터 끝까지 실행할 수 있게 푼 문서다. PLAN.md가 "무엇이 끝나면 다음으로 넘어가는지"를 정한다면, 이 문서는 Phase 4 안에서 "어떤 파일에 무엇을 어떤 순서로 만들고 무엇으로 검증하는지"를 정한다. Phase 4가 끝나면 이 문서의 결과 요약을 PLAN.md "Phase 4 결과" 절로 옮긴다.

> **2026-10-31 완료.** 결과·완료 기준·이 계획과 다르게 한 것은 PLAN.md "Phase 4 결과" 절에 있다. 아래는 작업 전 계획 기록이다 (9절 결정 권장안은 모두 권장대로 구현, P4-D5는 그리드가 `issue` 가중을 골랐다).

표기는 PLAN.md와 같다.
- **[확인]** 저장소 파일이나 데이터를 직접 보고 확인한 사실 (2026-10-29, main `f69f270`, 로컬 `data/processed/`)
- **[탐색]** 이 계획을 쓰려고 임시 스크립트로 돌려 본 값. 결과가 아니라 규모·방향 확인용이고, 최종 값은 `03_baselines.py`가 다시 낸다
- **[가정]** 확인하지 않은 추정
- **[결정 필요]** 팀이 고를 선택지. 9절에 모아 두었다

---

## 1. 목표와 범위

Phase 4는 **모델 비교의 바닥과 천장**을 정하고, 이후 모든 모델(M1·M2·M3)이 그대로 따라 쓸 **예측 파일·OOF·운영점 선택 규칙**을 만든다.

| 산출 | 왜 필요한가 |
| --- | --- |
| B0 고정 상한 K, 에러 횟수 규칙 | 운영 기준선. 예측 모델은 같은 성공 손실에서 이것보다 많이 아껴야 한다 |
| B1 이슈 본문 TF-IDF + LR | 이슈 난이도 기준선. M1이 이것 위에 더하는 정보가 "궤적 신호"다 |
| B1 학습 이슈 OOF 점수 | M1(Phase 5)의 입력 피처. in-fold 점수를 넣으면 이슈 난이도가 누수된다 |
| 오라클(같은 이슈 다른 궤적 성공률) | 참고 천장. 이슈 난이도만으로 얼마까지 맞힐 수 있는지 |
| 3.5 운영점 선택 함수 | 학습 OOF 곡선에서 손실 ≤5% 최대 절약 τ(와 K)를 고르고 테스트에 고정 적용 |
| 공용 OOF·예측 파일·평가 드라이버 | Phase 5~7이 같은 코드로 학습·평가되게 한다 |

범위 밖: M1 피처(Phase 5), 그림 정리(Phase 9), 외부 데이터(Phase 8). 다중 시점 정책(D8)은 Phase 4에서 하지 않는다.

## 2. 출발점

### 이미 있는 것 [확인]

- `data/processed/trajectories.parquet`: `traj_id, instance_id, model_name, target, n_steps, issue_text, issue_msg_len, system_len, …` (80,036행)
- `data/processed/steps.parquet`: `traj_id, step_idx, is_error, edit_failed, ai_len, obs_len, …` (2,115,623행). **에러 관측 판별 규칙은 Phase 1에서 이미 정했다** (`parse.ERROR`, 스텝의 30.0%). 4.1의 "에러 규칙 정하기"는 끝난 일이고 여기서는 쓰기만 한다
- `data/processed/splits.parquet`: `traj_id, split, fold` (train fold 0~4, 테스트 -1)
- `afp/metrics.py`: `align, discrimination, within_issue_auroc, context_costs, cost_curve(cap=), at_threshold, point_metrics, bootstrap`. 예측 형식은 `traj_id, N, p_fail`이고 시점 N의 진행 중 궤적과 정확히 1:1이어야 한다
- `afp/io.py`: 컬럼을 명시해야 읽히는 로더, 누수 컬럼 거부. `afp/leakage.scan_code`가 `code/` 아래 새 파일에서 누수 컬럼 이름을 쓰면 pytest를 실패시킨다

### 계획 작성 중 확인한 데이터 사실

**분할별 규모** [확인]

| split | 궤적 | 성공률 | 이슈 |
| --- | --- | --- | --- |
| train | 62,460 | 16.27% | 3,013 |
| test_same | 12,332 | 18.67% | 578 |
| test_cross | 5,244 | 17.58% | 578 |

**고정 상한 K** (성공 손실 / 절약 스텝, 분할별로 따로 계산) [확인]

| K | train | test_same |
| --- | --- | --- |
| 33 | 5.09% / 25.97% | 5.39% / 23.51% |
| **34** | **4.71% / 25.06%** | 4.95% / 22.64% |
| 35 | 4.32% / 24.19% | 4.47% / 21.81% |
| 40 | 3.17% / 20.49% | 3.56% / 18.31% |

- **학습 이슈에서 고르면 K\*=34**다 (손실 ≤5%인 가장 작은 K). 제안서의 K=35는 전체 데이터(테스트 포함)에서 고른 값이다. 3.5 원칙("τ·K는 학습에서만 고른다")을 B0에도 적용하면 비교 기준선은 K=34가 된다 → 9절 P4-D1
- train fold별 K=34 손실: 4.89 / 5.27 / 4.66 / 4.16 / 4.58% (fold 1은 5% 초과). **"손실 ≤5%"는 fold 사이에서도 ±0.5%p 흔들린다.** 운영점 결과는 반드시 구간과 같이 보고한다
- test_cross는 K ≤ 50 어디서도 손실 ≤5%를 만족하지 못한다 (K=35에서 12.4%, Phase 3). test_cross 운영점은 "train에서 고른 τ·K를 그대로 적용한 결과"로만 보고한다

**이슈 본문** [확인]
- 3,591개 이슈 중 **13개(모두 train)는 본문이 2가지**다. 한쪽은 끝에 `Here is the list of files that is recommended to be edited in order to fix the issue.\nFILES TO MODIFY:\n- <경로>`가 붙어 있다. 이 변형은 궤적 26개(13개 이슈 × 2)에만 있고 26개 모두 실패
- 이 꼬리는 **고칠 파일 목록 = 정답 패치에서 나온 정보**다. 제안서 §3 "정답 패치를 알아야 계산되는 값은 피처로 쓰지 않는다"에 해당하므로 B1 입력에서 지운다. 지우지 않으면 이슈 안에서 점수가 달라져 "B1 이슈 내 AUROC = 0.5" 검증도 깨진다
- 본문 길이: 중앙값 724자, 평균 1,192자, 최대 37,707자, 빈 본문 0개

**에러 횟수 점수** (`p_fail` = 앞 N스텝 `is_error` 개수, 학습 없음) [탐색]

| N | train AUROC | test_same AUROC / 이슈 내 | test_cross AUROC / 이슈 내 |
| --- | --- | --- | --- |
| 3 | 0.523 | 0.516 / 0.528 | 0.530 / 0.559 |
| 5 | 0.547 | 0.532 / 0.532 | 0.558 / 0.577 |
| 10 | 0.533 | 0.534 / 0.545 | 0.533 / 0.609 |
| 20 | 0.508 | 0.519 / 0.586 | 0.479 / 0.578 |

약한 신호다. 점수가 정수라 동점이 많아서 비용 곡선이 몇 개의 점으로만 이루어진다 (N=10 학습 궤적 에러 수 사분위 1/2/3, 90% 4).

**B1 규모 확인** (단어 1~2gram TF-IDF + LR, 학습 이슈 전체로 학습 후 테스트 채점) [탐색]
- test_same N=0 AUROC 0.64 / PR-AUC 0.28 (기준값 0.19), test_cross N=0 0.69 / 0.30. C를 줄일수록(0.3 > 1 > 3 > 10) 조금씩 좋아졌다. 4가지 C 학습·채점 합계 19초
- **이 탐색은 테스트 셋을 봤으므로 하이퍼파라미터 선택에 쓰지 않는다.** 4.2의 그리드는 C를 양쪽으로 넓게 두고 학습 OOF로만 고른다

**오라클** [탐색]
- 전체 데이터, 궤적이 2개 이상인 이슈의 궤적 80,026개에서 "같은 이슈 다른 궤적 성공률"(leave-one-out) AUROC **0.9448** → 제안서 0.945 재현
- 테스트 셋에서 LOO 오라클: test_same N=0 0.941, N=10 0.930 / test_cross 0.914, 0.905
- **LOO 오라클의 이슈 내 AUROC는 정의상 0이다** (실측 0.0). 이슈 안에서 성공 궤적은 자기를 뺀 (k−1)/(m−1), 실패 궤적은 k/(m−1)을 받아 항상 실패 쪽이 더 "성공할 것"처럼 보인다. 그래서 이슈 내 AUROC 표에는 넣지 않는다
- 자기 자신이 섞이지 않는 변형: test_cross 궤적을 같은 이슈의 70B(test_same) 성공률로 채점하면 N=0 0.917, N=10 0.907, 이슈 내 0.5

## 3. 코드 배치와 인터페이스

```
code/
├── afp/
│   ├── baselines.py   B0 규칙 점수, B1 본문 정리·학습·채점, 오라클 (새 파일)
│   ├── oof.py         fold별 학습 → OOF 예측 + 테스트 fold 평균 예측 (새 파일, Phase 5~7 재사용)
│   ├── preds.py       예측 파일 쓰기·읽기·검증, 메타데이터 (새 파일)
│   ├── evaluate.py    모델 하나의 예측 → 분할×N 지표 표, 운영점 표 (새 파일, Phase 5~7 재사용)
│   └── metrics.py     3.5 select_operating_point, select_combined, fixed_cap_curve 추가
├── 03_baselines.py    Phase 4 실행 스크립트
└── tests/
    ├── test_baselines.py
    ├── test_oof.py
    ├── test_operating_point.py  (metrics 3.5 부분)
    └── test_phase4_data.py      (data/processed가 있을 때만)
```

### 공용 상수

- `N_GRID = (3, 5, 8, 10, 15, 20)` (제안서). B1은 여기에 N=0도 낸다. 상수는 `afp/evaluate.py` 한 곳
- `MAX_LOSS = 0.05`, `N_BOOT = 1000`, 시드는 `io.SEED`

### 예측 파일 (`afp/preds.py`)

- 경로: `data/processed/preds/<model_id>.parquet` (git 밖)
- 컬럼: **`traj_id, N, p_fail`만** (하네스 형식 그대로). 분할·fold는 `splits.parquet`와 join해서 얻는다
- 한 파일에 train(OOF) + test_same + test_cross 예측이 같이 있다. N마다 그 시점 진행 중 궤적 전부에 정확히 1개 (`metrics.align`이 검사하는 조건을 저장 전에 `preds.validate`가 분할마다 미리 검사)
- parquet 메타데이터 `afp.preds`: `model_id`, 설정(JSON), `oof_scheme`(예: `"5fold-oof/test-fold-mean"`), N 목록, git 커밋, sklearn 버전, 생성 시각
- `model_id` 규칙: 기준선 `b0_*`, `b1_*`, 참고용 `ref_*`. **`ref_*` 예측은 다른 모델의 입력으로 쓰지 않는다** (`preds.load_feature`가 `ref_` 접두어를 거부)

### OOF 규칙 (`afp/oof.py`, 4.3)

```python
oof_predict(data, fit, predict, folds) -> pl.DataFrame  # traj_id, N, p_fail
```
- train fold k 궤적의 점수 = fold k를 뺀 4개 fold로 학습한 모델의 점수
- 테스트 궤적 점수 = **5개 fold 모델 점수의 평균** [결정 필요 P4-D2. PLAN.md 4.3은 "학습 이슈 전체로 학습한 B1"이라고 적었다]
  - 이유 1: 3.5에서 τ를 OOF 점수 위에서 고르고 테스트에 그대로 적용한다. 전체 학습 모델은 데이터가 25% 많아 점수 분포가 OOF와 다를 수 있다 (LR은 정규화 강도가 데이터 양에 상대적이라 더 자신 있게 낸다). 같은 4/5 크기 모델 5개의 평균이 OOF와 척도가 가깝다
  - 이유 2: M1은 학습 때 B1 OOF 점수를 피처로 보고, 테스트 때 B1 테스트 점수를 본다. 두 점수의 척도가 같아야 한다
  - 전체 학습 버전도 같은 함수 옵션(`test="refit"`)으로 낼 수 있게 두고, 4.6에서 두 방식의 테스트 AUROC와 OOF↔테스트 점수 분위수를 비교 표로 남긴다
- 학습에 쓴 이슈 집합을 fold마다 기록하고, "자기 이슈를 본 모델의 점수를 받은 train 궤적 0개"를 실행 때마다 검사한다
- fit 함수는 fold 안에서 **벡터라이저(어휘·IDF)까지 새로 학습**한다. 전체 train으로 어휘를 만든 뒤 fold를 나누면 검증 fold 단어의 IDF가 섞인다

## 4. 작업 순서

```
4.0 공용 인터페이스 ─┬─ 4.1 B0 ──────────┐
                     ├─ 4.2 B1 → 4.3 OOF ┼─ 4.5 운영점(3.5) ─ 4.6 실행·표 ─ 4.7 검증 ─ 4.8 문서
                     └─ 4.4 오라클 ───────┘
```

4.1, 4.2~4.3, 4.4는 4.0의 예측 파일 형식만 합의하면 병렬로 진행할 수 있다.

### 4.0 공용 인터페이스 (0.5일)

| 작업 | 내용 |
| --- | --- |
| 4.0.1 | `afp/preds.py`: `write(model_id, preds, config)`, `read(model_id)`, `validate(preds, trajs_by_split, n_grid)`. 저장 전에 분할·N별 커버리지 검사, NaN 거부, 메타데이터 기록. 같은 내용이면 다시 쓰지 않고(바이트 동일 유지), 다르면 `--force` 없이는 멈춤 (02_split.py와 같은 방식) |
| 4.0.2 | `afp/evaluate.py`: `load_eval_sets()` → `{"train": …, "test_same": …, "test_cross": …}` (각각 `metrics.TRAJ_COLUMNS` 형태) + 평가 집합별 `costs`(`metrics.context_costs`, 1회 계산 후 `data/processed/context_costs.parquet`에 캐시) |
| 4.0.3 | `evaluate.discrimination_table(model_id, preds, sets, n_grid)` → 행 = (model, split, N): `n_running, base_rate, auroc, pr_auc, within_pair, within_issue_mean, within_n_issues, within_n_pairs`. train 행은 OOF 점수 기준(`split="train_oof"`로 표기) |
| 4.0.4 | `evaluate.curves(...)`: 분할·N별 `cost_curve`(costs 포함). 저장용으로는 τ 행을 최대 200개로 줄인다 (손실 0.1%p 간격 보간 없이, 기존 행에서 고름) |

### 4.1 B0: 고정 상한·에러 규칙 (0.5일)

| ID | 점수·정책 | 비고 |
| --- | --- | --- |
| `b0_cap` | 정책 "스텝 > K면 종료", K = 1..max | 시점 N 예측이 아니라 정책이라 `metrics.fixed_cap_curve(trajs, costs)`로 K별 (손실, 절약 스텝, 절약 토큰) 표를 낸다 |
| `b0_err` | `p_fail` = 앞 N스텝 `is_error` 개수 | 제안서의 "에러 횟수 규칙". τ를 훑는 것 = 문턱 m을 훑는 것 |
| `b0_editfail` | `p_fail` = 앞 N스텝 `edit_failed` 개수 | 보조. 편집 거부만 센다 |

- `fixed_cap_curve`는 닫힌 식(벡터 계산)으로 만들고, 테스트에서 "N=K에서 진행 중 궤적 전부 p_fail=1"을 `cost_curve`에 넣은 결과와 K 몇 개에서 일치하는지 확인한다. 실데이터 값은 Phase 3 표(K=35 전체 4.9%/24.6%/42.2%, test_same 4.47%/21.81%)와 같아야 한다
- 에러 비율(개수/N)은 고정 N에서 개수와 순위가 같으므로 따로 만들지 않는다
- B0 규칙은 학습이 없으므로 "OOF 점수" = 학습 궤적에 같은 규칙을 적용한 점수다. 운영점(m, K)은 4.5 규칙대로 train에서 고른다
- 에러 점수는 `io.scan_steps(["traj_id", "step_idx", "is_error", "edit_failed"]).filter(step_idx <= 20)`만 읽는다 (원문 컬럼을 읽지 않아 메모리 수십 MB)
- 피처는 앞 N스텝만 본다는 테스트 (5.2의 누수 방지 테스트를 여기서 먼저 만든다): N 이후 스텝의 `is_error`를 바꿔도 시점 N 점수가 같아야 한다

### 4.2 B1: 이슈 본문 TF-IDF + 로지스틱 회귀 (1일)

**입력 정리** (`baselines.issue_table`)
1. `io.load_trajectories(["traj_id", "instance_id", "issue_text"])`만 읽는다
2. `strip_hint(text)`: `"Here is the list of files that is recommended to be edited"`부터 끝까지 지운다 (2절, 13개 이슈)
3. 정리 후 `instance_id`마다 본문이 정확히 1개인지 assert (3,591개). 하나라도 2개면 멈춘다
4. 테스트: `strip_hint`가 꼬리 없는 본문은 그대로 두고, 꼬리 있는 본문을 꼬리 없는 쪽과 같게 만드는지 (실데이터 13쌍)

**학습 단위: 이슈 행 + 가중치**

같은 이슈의 궤적은 입력(본문)이 같으므로 궤적 단위 LR은 "이슈 행 2개(성공/실패) + 가중치(개수)" LR과 수학적으로 같다. 궤적 62,460행 대신 이슈 3,013×2행으로 학습한다. PLAN.md 4.2의 "궤적 단위 라벨 vs 이슈별 성공률 타깃" 비교는 가중치만 바꾸면 된다.

| 가중 방식 | 이슈 i의 (성공 행, 실패 행) 가중치 | 의미 |
| --- | --- | --- |
| `traj` | (k_i, m_i − k_i) | 궤적 단위 라벨. 궤적이 많은 이슈의 비중이 크다. 하네스 지표(궤적 단위)와 맞는다 |
| `issue` | (k_i/m_i, 1 − k_i/m_i) | 이슈별 성공률을 타깃으로, 이슈마다 무게 1 |

- 테스트: 작은 합성 데이터에서 `traj` 가중 이슈 행 LR의 계수·확률이 궤적을 복제한 LR과 수치 허용 오차(예: 1e-5) 안에서 같은지 (`solver="lbfgs"`, 같은 `C`)
- 시점 N 버전(`b1_n`): 가중치를 "N에 아직 진행 중인 성공·실패 궤적 수"로 바꿔 N마다 다시 학습한다. 진행 중 궤적이 없는 이슈는 그 N의 학습에서 빠진다
- 고정 버전(`b1_static`): N=0 모델 하나의 점수를 모든 N에 그대로 쓴다

**하이퍼파라미터 그리드** (N=0, `traj` 가중, 학습 5-fold OOF AUROC로 선택)

| 항목 | 후보 |
| --- | --- |
| 분석 단위 | `word` (1,2)-gram, `char_wb` (3,5)-gram |
| `min_df` | 2 |
| `max_features` | 200,000 |
| `sublinear_tf` | True |
| `C` | 0.03, 0.1, 0.3, 1, 3 |
| 가중 방식 | `traj`, `issue` |

- 2 × 5 × 2 = 20개 설정 × 5 fold = 100회 학습. fold 학습 이슈 약 2,400개라 단어 기준 회당 1~2초 [가정: 문자 n-gram은 3~5배]. 전체 5분 안팎
- 선택 기준: train OOF AUROC (궤적 단위, N=0). 동점(차이 0.002 미만)이면 더 단순한 쪽(`word`, 큰 정규화 = 작은 C)
- 고른 설정을 `b1_n`의 모든 N에 그대로 쓴다 (N마다 다시 고르지 않음: 설정 선택에 쓰는 OOF가 M1 입력이 되므로 선택 횟수를 줄여 낙관 편향을 줄인다)
- 그리드 전체의 OOF AUROC를 `results/phase4/b1_grid.csv`로 남긴다
- 선택 편향: 설정 선택과 최종 OOF 점수가 같은 fold를 쓴다(중첩 CV 아님). 그리드가 작아 편향은 작다고 보고 [가정] 그대로 쓴다. 학습 OOF 지표는 "선택에 쓴 값"으로 표시하고 결과 주장은 테스트 지표로만 한다

**(선택) 4.2b 저장소 사전 확률**: 테스트 이슈의 23.5%는 학습에 없는 저장소에서 온다 (Phase 2.4). 저장소별 성공률 OOF 타깃 인코딩을 TF-IDF 점수와 섞은 `b1_repo`를 하나 더 만들어 볼 수 있다. B1 본 결과가 끝난 뒤 시간이 남으면 한다

### 4.3 B1 OOF 점수 (4.2와 같이, 0.5일)

- `oof.oof_predict`로 `b1_static`, `b1_n` 각각 train OOF + 테스트 fold 평균 점수를 만든다
- **모든 궤적(80,036개)에 대해 N ∈ {0} ∪ N_GRID의 진행 중 궤적마다 점수 1개**. M1(Phase 5)은 `preds.load_feature("b1_n")`을 `(traj_id, N)`으로 join해서 쓴다
- 검사 (실행 때마다, 실패하면 멈춤):
  - 자기 이슈를 학습에 쓴 모델의 점수를 받은 train 궤적 0개
  - 같은 이슈의 궤적은 (N마다) 점수가 정확히 같다 → 이슈 내 AUROC가 정확히 0.5 (|x − 0.5| < 1e-12), train_oof·test_same·test_cross 모든 N
  - in-fold 점수와 비교: 같은 설정을 학습 이슈 전체로 학습해 train에 채점한 in-fold AUROC가 OOF AUROC보다 **훨씬 높아야** 한다 (이 차이가 4.3이 막으려는 누수의 크기다. 값을 `summary.json`에 남김)
- 단위 테스트 (`test_oof.py`, 합성 데이터): 라벨이 이슈로만 정해지고 피처가 이슈 ID 원-핫인 데이터에서 in-fold AUROC ≈ 1, OOF AUROC ≈ 0.5 (처음 보는 이슈는 구별 못 함). fold 경계를 하나라도 어기면 이 테스트가 깨진다

### 4.4 오라클 (참고용 천장, 0.5일)

| ID | 점수 (`p_fail` = 1 − 성공률) | 이슈 내 AUROC |
| --- | --- | --- |
| `ref_oracle_loo` | 같은 평가 집합에서 같은 이슈의 **다른 궤적 전부**(N 이전에 끝난 것 포함) 성공률 | 정의상 0 → 보고하지 않음 (2절) |
| `ref_oracle_xmodel` | test_cross: 같은 이슈 test_same(70B) 성공률 / test_same: 같은 이슈 test_cross(8B·405B) 성공률 | 이슈 안에서 상수 → 0.5 |

- 재현 검사: 전체 데이터 N=0 LOO AUROC = 0.9448 (궤적 2개 이상 이슈, 80,026개). 제안서 0.945와 소수 셋째 자리 일치를 assert
- 궤적이 1개인 이슈(LOO 정의 불가)는 10개이고 모두 train이라 테스트 오라클에는 영향이 없다. 모든 테스트 이슈에 test_same·test_cross 궤적이 둘 다 있어 `ref_oracle_xmodel`도 빠지는 궤적이 없다 [확인]. 그래도 코드는 점수를 못 주는 궤적이 생기면 학습 성공률(16.27%)을 넣고 개수를 `summary.json`에 적는다 (하네스는 진행 중 궤적 전부에 예측을 요구)
- 스텝 0개 궤적 1개(`00-03183`, test_cross, 실패)는 N=0에서도 진행 중이 아니라 어떤 예측 대상에도 들어가지 않는다 [확인]. B1 N=0 행 수가 분할 궤적 수보다 1 적은 이유다
- 오라클은 정답 라벨을 쓰므로 **배포 불가능한 참고값**이다. 판별 표에는 점선/별도 행으로, 운영점 비교 표에는 넣지 않는다
- 오라클은 `target`을 쓰지만 누수 컬럼(`exit_status` 등)은 쓰지 않는다. 코드는 `baselines.py`의 `oracle_*` 함수로 분리하고 `ref_` 예측은 피처로 못 쓰게 막는다 (3절)

### 4.5 운영점 선택 (PLAN 3.5, 0.5~1일)

`afp/metrics.py`에 추가한다.

```python
select_operating_point(curve, max_loss=0.05, objective="saved_steps") -> dict
#   curve = cost_curve(train, oof_preds, N[, cap=K]) 출력.
#   lost_success <= max_loss인 행 중 objective 최대. 동점이면 큰 τ(덜 끊는 쪽).
#   만족하는 행이 없으면 τ=inf 행 (아무것도 안 끊음; cap이 있으면 cap 자체가 손실을 넘을 수 있으므로 feasible=False 표시).
select_cap(cap_curve, max_loss=0.05) -> dict           # B0: 손실 ≤ max_loss인 가장 작은 K
select_combined(trajs, preds, n, costs, k_grid, max_loss) -> dict
#   K마다 select_operating_point(cost_curve(..., cap=K)) → objective 최대인 (K, τ)
```

- **선택은 train OOF 곡선 하나(5 fold를 합친 62,460개)에서 한다.** 고른 (τ, K)의 fold별 손실·절약을 같이 기록해 흔들림을 본다 (2절: K=34 fold 손실 4.16~5.27%)
- 목적 함수: 절약 **스텝** 최대 (제안서 비교 단위). 절약 토큰은 같은 운영점에서 함께 보고 [결정 필요 P4-D4]
- 결합 정책 K 그리드: `range(N + 1, 61)`. (N, K) 하나당 `cost_curve` 1회(학습 6만 행, 수십 ms [가정])라 N 6개 × K 최대 57개 ≈ 340회로 충분히 빠르다
- 테스트 적용: `metrics.point_metrics(test, preds, N, tau=τ*, costs=, cap=K*)`. 테스트에서 τ를 다시 고르지 않는다
- 부트스트랩 (테스트 분할마다, 1,000회): `bootstrap(test, preds, stat)`의 stat에 **같은 표본에서 계산한 고정 상한 기준선과의 차이**를 같이 넣는다 → "모델 절약 − K\* 절약"의 짝지은 95% 구간. 손실은 "≤5%를 만족했는지"를 구간 상한과 함께 보고한다
- 민감도: max_loss ∈ {0.02, 0.03, 0.05, 0.08, 0.10}에서도 같은 선택을 해서 `operating_points_sensitivity.csv`에 둔다 (곡선 모양을 숫자로 남김)
- 단위 테스트 (`test_operating_point.py`, 손으로 셀 수 있는 곡선): 제약을 지키는 행 중 최대 절약을 고르는지, 동점 처리, 만족 행이 없을 때, `select_cap`이 실데이터 train에서 K=34를 고르는지(`test_phase4_data.py`), `select_combined`의 τ=inf 행이 고정 상한과 같은지

### 4.6 실행 스크립트 `code/03_baselines.py` (0.5일)

```
uv run --no-sync python code/03_baselines.py [--force] [--n-boot 1000] [--skip-bootstrap] [--models b0_err,b1_n,...]
```

1. 입력 확인: `trajectories.parquet`, `steps.parquet`, `splits.parquet`가 없으면 앞 단계 실행 안내 후 종료. `splits`를 `split.validate_splits`로 다시 검증
2. 평가 집합·costs 준비 (4.0.2)
3. 예측 생성 → `data/processed/preds/`: `b0_err`, `b0_editfail`, `b1_static`, `b1_n`, `ref_oracle_loo`, `ref_oracle_xmodel` (+ 선택 `b1_repo`)
4. 평가 → `results/phase4/`
5. 완료 기준 검사(4.7) → 통과하면 `완료 기준 통과` 출력, 아니면 assert로 멈춤

**`results/phase4/` (git에 올림, 작은 CSV·JSON)**

| 파일 | 내용 |
| --- | --- |
| `discrimination.csv` | (model, split, N)별 AUROC, PR-AUC, 기준값, 이슈 내 AUROC 2종, 이슈·쌍 수 |
| `discrimination_ci.csv` | 테스트 분할 지표의 부트스트랩 95% 구간 |
| `fixed_cap.csv` | 분할별 K = 1..80의 손실·절약 스텝·절약 토큰 |
| `operating_points.csv` | (model, N, 정책 ∈ {단독, 결합})별 train에서 고른 τ·K, train·test_same·test_cross의 정지율·손실·절약 스텝·절약 토큰 + 구간 + K\* 대비 차이 구간 |
| `operating_points_sensitivity.csv` | max_loss별 같은 표 (주 지표만) |
| `cost_curves/<model>_<split>_N<N>.csv` | 줄인 비용 곡선 (그림용) |
| `b1_grid.csv` | B1 하이퍼파라미터 그리드 OOF AUROC |
| `b1_scale.csv` | 테스트 fold 평균 vs 전체 재학습: AUROC, 점수 분위수, OOF 분위수 (P4-D2 근거) |
| `summary.json` | 완료 기준 수치, 고른 설정, 누수 검사 결과, 실행 시간, 커밋 |

- 예상 실행 시간 [가정]: 예측 생성 5~10분(B1 그리드가 대부분), 부트스트랩 1,000회 × (모델 5 × N 6 × 테스트 분할 2) ≈ 60조합 × 약 20초(Phase 3 측정) ≈ 20분. 개발 중에는 `--skip-bootstrap` 또는 `--n-boot 100`
- 결정성: 다시 실행하면 `results/phase4/`와 `preds/*.parquet`이 바이트 단위로 같아야 한다. 저장된 예측이 재계산과 다르면 `--force` 없이는 멈춘다. LR은 `lbfgs` + 고정 시드, 정렬 순서 고정 (Phase 3에서 겪은 polars join 순서 문제와 같은 대응)
- 누수: 스크립트와 새 모듈은 누수 컬럼 이름을 쓰지 않는다. `leakage.scan_code` 테스트가 그대로 지킨다. `generated_patch`가 필요한 분석은 Phase 4에 없다

### 4.7 완료 기준

PLAN.md Phase 4의 완료 기준("N별 B0·B1 AUROC/PR-AUC/이슈 내 AUROC 표, B1 이슈 내 AUROC = 0.5")을 아래처럼 검사 가능한 항목으로 바꾼다. 모두 `03_baselines.py`가 assert하고 `summary.json`에 남긴다.

| # | 기준 |
| --- | --- |
| C1 | 예측 파일 6개(+선택 1)가 모든 분할·N에서 진행 중 궤적과 1:1 (`preds.validate`) |
| C2 | B1(`b1_static`, `b1_n`) 이슈 내 AUROC = 0.5 (|x − 0.5| < 1e-12), train_oof·test_same·test_cross × 모든 N |
| C3 | OOF: 자기 이슈를 본 모델 점수를 받은 train 궤적 0개. 3,591개 이슈 모두 정리 후 본문 1개 |
| C4 | `fixed_cap_curve`가 Phase 3 값 재현: 전체 K=30/35/40/50 → 7.0/4.9/3.7/2.2% 손실, 29.4/24.6/20.8/15.8% 절약 스텝, K=35 절약 토큰 42.2%. test_same K=35 4.47%/21.81% |
| C5 | train에서 고른 고정 상한 K\* = 34 (2절 값이 바뀌면 원인을 기록) |
| C6 | 오라클 전체 데이터 LOO AUROC 0.945 (소수 셋째 자리) |
| C7 | `discrimination.csv`에 B0·B1·오라클 × 3분할 × N 표가 있고, 테스트 분할에는 구간이 있음 |
| C8 | `operating_points.csv`에 B0·B1의 단독·결합 정책 운영점과 K\* 대비 차이 구간이 있음 |
| C9 | `uv run --no-sync pytest` 전체 통과 (기존 56개 + 새 테스트), 재실행 시 `git diff results/` 없음 |

결과가 기준선보다 나쁘거나 B0 에러 규칙이 0.5 근처여도 그대로 보고한다 (제안서 §5). "이긴다"는 완료 기준이 아니다.

### 4.8 문서 (0.5일)

- PLAN.md: "Phase 4 결과 (날짜 완료)" 절 추가 (Phase 1~3 결과 절과 같은 형식: 실행 명령, 완료 기준 일치 여부, 확정한 규칙, 핵심 표). 2절 "현재 상태"·4절 "격차"·7절 일정·8절 명령 갱신. 3.5 완료 표시
- README.md: 실행 순서에 `03_baselines.py` 추가 (소요 시간 포함)
- docs/proposal-changes.md: 제안서와 달라진 것이 생기면 추가 — 후보는 (1) 비교 기준 K=35 → 학습에서 고른 K\*=34 (P4-D1을 그렇게 정할 경우) (2) 이슈 본문 13개의 파일 목록 꼬리 제거 (3) 테스트 점수 fold 평균 (P4-D2)
- 11절 미결 사항 표에 P4-D1~D5 결정 결과 반영

## 5. 테스트 목록

| 파일 | 테스트 | 확인하는 것 |
| --- | --- | --- |
| `test_baselines.py` | `test_strip_hint_removes_file_list_only` | 꼬리 제거, 꼬리 없는 본문 불변 |
| | `test_issue_rows_weighted_lr_equals_trajectory_lr` | 이슈 행 + 가중치 LR = 궤적 복제 LR |
| | `test_b1_score_constant_within_issue` | 이슈 내 점수 동일 → 하네스 이슈 내 AUROC 0.5 |
| | `test_error_count_ignores_steps_after_n` | N 이후 스텝을 바꿔도 시점 N 점수 불변 |
| | `test_oracle_loo_within_issue_is_zero` | LOO 오라클 인공물을 문서화 (0이어야 함) |
| | `test_ref_preds_rejected_as_features` | `ref_*` 피처 사용 거부 |
| `test_oof.py` | `test_oof_issue_memorization_is_chance` | 이슈 ID 원-핫: in-fold ≈ 1, OOF ≈ 0.5 |
| | `test_oof_never_scores_seen_issue` | fold별 학습 이슈와 채점 이슈 교집합 0 |
| | `test_test_scores_are_fold_mean` | 테스트 점수 = fold 모델 5개 평균 |
| | `test_vectorizer_fit_inside_fold` | 검증 fold에만 있는 단어가 어휘에 없음 |
| `test_operating_point.py` | `test_select_max_saving_under_loss` | 제약 + 최대 절약 + 동점 규칙 |
| | `test_select_infeasible_returns_no_stop` | 만족 행 없음 처리 |
| | `test_fixed_cap_curve_matches_cost_curve` | 닫힌 식 = 하네스 |
| | `test_select_combined_inf_row_is_cap` | 결합 정책 τ=inf = 고정 상한 |
| `test_phase4_data.py` (데이터 있을 때만) | `test_real_fixed_cap_values`, `test_real_cap_selected_on_train_is_34`, `test_real_issue_text_unique_after_strip`, `test_real_preds_cover_running` | C3~C5 실데이터 |
| `test_leakage.py` (기존) | `scan_code` | 새 파일이 누수 컬럼 이름을 안 씀 |

## 6. Phase 5로 넘기는 것

| 항목 | 형태 |
| --- | --- |
| B1 OOF·테스트 점수 | `data/processed/preds/b1_n.parquet` (+ `b1_static`), `preds.load_feature`로 `(traj_id, N)` join |
| OOF 학습기 | `oof.oof_predict(data, fit, predict, folds)` — M1 LR/XGBoost, M2가 그대로 쓴다 |
| 평가 | `evaluate.discrimination_table`, `evaluate.curves`, `metrics.select_operating_point/select_combined` |
| 비교 기준값 | `operating_points.csv`의 K\* 행과 B0·B1 행. M1 표는 같은 열을 낸다 |
| 앞 N스텝 테스트 패턴 | `test_error_count_ignores_steps_after_n` → M1 피처 누수 테스트(5.2)의 틀 |

## 7. 일정과 분담

PLAN.md 7절 기준 9주차(지금) 작업. 합계 약 3~4일 [가정]. PLAN.md는 Phase 4를 1일로 잡았지만 3.5와 공용 OOF·평가 코드가 들어오면서 커졌다. 대신 Phase 5~7은 이 코드를 그대로 쓴다.

| 순서 | 작업 | 예상 | 담당 (PLAN.md 10절 제안) |
| --- | --- | --- | --- |
| 1 | 4.0 공용 인터페이스 (예측 파일 형식 먼저 합의) | 0.5일 | 공동 |
| 2a | 4.1 B0 + `fixed_cap_curve` | 0.5일 | B |
| 2b | 4.2 B1 + 4.3 OOF | 1.5일 | A |
| 2c | 4.4 오라클 | 0.5일 | B |
| 3 | 4.5 운영점 선택 | 0.5~1일 | B |
| 4 | 4.6 실행 스크립트·표, 4.7 검증 | 0.5일 | 공동 |
| 5 | 4.8 문서 | 0.5일 | 공동 |

10주차 시작에 Phase 5(M1)가 시작할 수 있어야 한다. B1이 늦어지면 M1은 B1 없는 버전부터 시작하고, B1 포함 버전은 `b1_n.parquet`이 나온 뒤 붙인다.

브랜치: `phase/4`. 작업 단위(4.0, 4.1, …)마다 커밋하고 Phase 1~3처럼 PR 하나로 main에 합친다.

## 8. 검증 명령

```bash
uv run --no-sync python code/01_parse.py          # 결과 있으면 검증만
uv run --no-sync python code/02_split.py --skip-leakage
uv run --no-sync python code/03_baselines.py --n-boot 100    # 개발용
uv run --no-sync python code/03_baselines.py                 # 최종 (부트스트랩 1,000회)
uv run --no-sync pytest
git diff --stat results/                          # 재실행 후 변화 없어야 함
```

- `--no-sync`: `dl` 그룹(torch)을 지우지 않으려고 (README)
- git worktree에서 작업하면 `afp/io.py`의 `ROOT`가 worktree 루트를 가리켜 `data/`에 `download.sh`만 있다. 메인 체크아웃의 데이터를 안쪽에 링크한다: `ln -s <메인 체크아웃>/data/processed data/processed`, `ln -s <메인 체크아웃>/data/swe-agent-trajectories data/swe-agent-trajectories` (`.gitignore`의 `data/*`가 링크도 git에서 뺀다)

## 9. 결정 필요

| ID | 결정할 것 | 선택지 | 권장 |
| --- | --- | --- | --- |
| P4-D1 | 모델 비교의 고정 상한 기준선 | (a) 학습에서 고른 K\*=34 (b) 제안서 K=35 | **(a)를 주 기준으로, (b)를 같은 표에 참고 행으로.** 3.5 원칙과 일관되고, K=35는 테스트를 포함한 전체 데이터에서 고른 값이다. 두 값의 test_same 차이는 손실 0.48%p, 절약 0.83%p라 결론이 바뀔 가능성은 낮다 [확인: 2절 표]. (a)로 정하면 proposal-changes.md에 기록 |
| P4-D2 | OOF 모델의 테스트 점수 | (a) 5개 fold 모델 평균 (b) 학습 이슈 전체로 재학습 (PLAN.md 4.3 문구) | **(a)**. τ 이전과 M1 입력 척도가 OOF와 같다. (b)도 계산해 `b1_scale.csv`로 차이를 보이고, 차이가 작으면 어느 쪽이든 무방하다고 기록 |
| P4-D3 | B1 대표 행 | (a) `b1_n` (N마다 진행 중 궤적으로 재학습) (b) `b1_static` (N=0 모델 하나) | **(a)를 주 결과, (b)를 보조**. 시점 N의 비교 대상(진행 중 궤적)에서 이슈 난이도만으로 낼 수 있는 최선에 더 가깝다. M1은 같은 N의 `b1_n`을 피처로 쓴다 |
| P4-D4 | 운영점 목적 함수 | 절약 스텝 / 절약 토큰 | **스텝** (제안서 비교 단위가 24.6% 스텝). 토큰은 같은 운영점에서 함께 보고. 토큰 기준으로 고른 운영점은 민감도 표에만 |
| P4-D5 | B1 가중 방식 | `traj` / `issue` | 4.2 그리드에서 train OOF AUROC로 고른다. 미리 정하지 않는다 |
| (D3 유지) | 이슈 내 AUROC 주 지표 | 쌍 가중 / 이슈 평균 | PLAN.md D3 그대로. Phase 4 표는 둘 다 낸다 |

## 10. 리스크

| 리스크 | 영향 | 대응 |
| --- | --- | --- |
| B1이 약함 (탐색 AUROC 0.64~0.69, 오라클 0.94) | M1 "B1 포함/제외" 차이가 작게 나와 "이슈 난이도 통제"가 약해 보임 | 오라클 행을 같이 보여 "B1이 잡는 난이도는 일부"임을 명시. 이슈 내 AUROC가 난이도 통제의 주 지표라는 점을 유지. 시간 있으면 4.2b |
| τ가 OOF → 테스트로 옮겨지지 않음 (점수 척도 차이) | 테스트 손실이 5%를 크게 넘거나 너무 보수적 | P4-D2 fold 평균, `b1_scale.csv`로 분위수 비교, 테스트 정지율을 train 정지율과 같이 보고 |
| "손실 ≤5%" 판정이 표본 잡음에 묻힘 (fold 간 ±0.5%p, 테스트 구간 ±1.4%p) | 기준선 대비 우열이 불확실 | 짝지은 부트스트랩 차이 구간, 민감도 표, fold별 손실 기록 |
| 에러 규칙 점수가 정수라 비용 곡선이 성김 | 5% 근처 운영점이 없어 손실 2%나 8%로 튐 | 그대로 보고 (규칙의 성질). 동점 깨기용 2차 점수는 넣지 않는다 (넣으면 규칙이 아니라 모델이 됨) |
| 이슈 본문의 정답 파일 힌트 (13개 이슈) | B1 누수, 이슈 내 0.5 검증 깨짐 | `strip_hint` + 이슈당 본문 1개 assert (C3) |
| 오라클 LOO 인공물 | 이슈 내 AUROC 0을 "신호 없음"으로 오해 | 이슈 내 표에서 LOO 제외, `ref_oracle_xmodel`로 대체, 테스트로 문서화 |
| 테스트를 본 탐색값이 설정 선택에 스며듦 | 테스트 성능 과대평가 | 2절 탐색값은 선택에 쓰지 않음. 그리드는 C를 양쪽으로 넓게, 선택은 train OOF만 |
| 부트스트랩 시간 (약 20분) | 반복 개발이 느림 | `--skip-bootstrap`, `--n-boot 100`. 최종 1회만 1,000회 |
| 팀원 동시 작업 (main에 다른 Phase 커밋) | 공용 모듈 충돌 | 4.0 인터페이스를 먼저 커밋·공유. `metrics.py`는 함수 추가만 하고 기존 시그니처는 바꾸지 않음 |
