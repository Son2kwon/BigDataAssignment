# Phase 8. 일반화 테스트 — 상세 기록

- 담당: 차승준. 코드와 문서는 Copilot이 작성하고, 실행은 차승준이 한다 (Phase 7과 같은 진행 방식)
- 브랜치는 `phase/8`(`phase/7`에서 분기)이다. 실행 스크립트는 `code/07_m1_cross.py`, 공용 모듈은 `code/afp/cross.py`다
- 수치 표기는 Phase 5~7과 같다. 구간은 이슈 단위 부트스트랩 1,000회의 95%다. 기준선 비교는 70B train에서 고른 **K\*=40 대비**이고, 제안서 K=35는 참고 행으로 둔다

---

## 8.1 보조 실험: 70B M1을 8B·405B 궤적에 적용 (2026-11-10, 실행 전에 커밋)

### 배경

- 제안서의 결과 5번은 "70B 같은 모델 테스트 vs 8B·405B 다른 모델 테스트"였다. 2026-11-05 회의에서 테스트 분할을 70B 무작위 이슈로 바꾸면서 이 항목을 범위에서 뺐다 (`docs/proposal-changes.md` 14번, PLAN Phase 8.1 취소선)
- 2026-11-10 차승준 요청: "학습한 M1을 우리가 준비한 다른 데이터셋에 돌릴 수 있는가"
- 데이터별로 확인한 결과는 아래와 같다 [확인]

| 데이터 | M1을 그대로 적용? | 이유 |
| --- | --- | --- |
| `swe-agent-trajectories`의 8B·405B 궤적 (5,244개) | **가능** | 프레임워크·행동 어휘·메시지 형식이 같다. 이미 `trajectories.parquet`·`steps.parquet`에 파싱되어 있다 (80,036궤적 전체). 맥락 길이 캐시 `context_costs.parquet`도 모든 궤적을 덮는다 |
| `swe-rebench-openhands-trajectories`, `terminalbench-trajectories` | 불가 | OpenHands·Terminus 프레임워크라 행동 어휘(`cmd_*`, 편집 성공/실패)가 없다. 프레임워크와 무관한 피처만으로 축소 모델을 따로 만들어야 한다 (PLAN 8.3, D1: 최소 범위 밖) |
| `swebench-experiments` | 아직 불가 | git 저장소만 있고 궤적은 받지 않았다. `analysis.download_logs`로 받은 뒤 Phase 1 파서용 어댑터가 필요하다 (PLAN 8.2) |

- 그래서 이 보조 실험의 범위는 8B·405B뿐이다
- 주 평가(70B train/test), 분할, 기존 결과 파일은 바꾸지 않는다. 결과는 `results/phase8/`에만 쓴다

### 사전 등록 (이 절은 결과를 본 뒤 고치지 않는다)

**평가 집합** (`afp/cross.py`, 8B·405B 궤적은 `splits.parquet`에 없다)

8B·405B가 시도한 이슈는 모두 70B도 시도했다. 그래서 그 이슈가 70B 분할의 어느 쪽인지로 나눈다.

| 집합 | 정의 | 궤적 / 이슈 [확인] | 성공률 | 스텝 중앙값 | 역할 |
| --- | --- | --- | --- | --- | --- |
| `x8b_unseen` | 70B **test** 이슈의 8B 궤적 | 707 / 87 | 15.7% | 27 | **주 결과** (M1 fold 모델이 이 이슈의 70B 궤적을 학습에 쓰지 않았다) |
| `x405b_unseen` | 70B test 이슈의 405B 궤적 | 255 / 22 | 15.7% | 31 | **주 결과** |
| `x8b_all` | 8B 궤적 전부 | 4,053 / 528 | 15.1% | 26 | 참고 (70B train 이슈가 섞여 있어 같은 이슈의 70B 궤적이 학습에 쓰였다) |
| `x405b_all` | 405B 궤적 전부 | 1,191 / 100 | 25.9% | 27 | 참고 |
| `test70b` | 70B test (Phase 5와 같은 집합) | 12,481 / 602 | 16.7% | — | 참고: 같은 코드·같은 예측으로 나란히 계산한 기준 |

- 출처: 2026-11-10 `cross.describe(cross.load_cross_sets())` (`trajectories.parquet`, `splits.parquet`). 실행하면 `results/phase8/sets.csv`가 같은 값과 N별 진행 중 궤적 수를 낸다
- 시점 N에 진행 중인 궤적 (괄호는 성공) [확인]:

| N | 405B test 이슈 | 405B train 이슈 | 8B test 이슈 | 8B train 이슈 |
| --- | --- | --- | --- | --- |
| 5 | 248 (40) | 904 (254) | 680 (110) | 3,175 (456) |
| 10 | 224 (30) | 753 (163) | 583 (76) | 2,630 (293) |
| 20 | 170 (4) | 506 (47) | 425 (28) | 1,921 (119) |

- `x405b_unseen`은 이슈 22개라서 구간이 매우 넓다. N=20에서는 진행 중 성공이 4개뿐이라 손실 지표가 사실상 의미가 없다. **결과는 개별 N이 아니라 N 전체의 경향으로 쓴다**

**모델**
- `m1_xgb`, `m1_lr`만 쓴다. B1 포함판(`*_b1`)은 뺀다. 8B·405B 궤적용 B1 fold 점수가 없다. B1을 새로 만들면 B1이 같은 이슈 본문을 이미 학습했다는 문제도 생긴다
- fold 모델은 저장하지 않았다. 그래서 `04_m1.py`의 `design`·`run_oof`·`predict_model`을 그대로 불러(`importlib`) 저장된 예측 파일의 설정(`preds.metadata(m)["config"]["selected"]`)으로 다시 만든다
- **C3**: 다시 만든 fold 모델의 train OOF와 test fold 점수가 저장된 예측(`preds/m1_*.parquet`, `.test_folds.parquet`)과 1e-9 안에서 같아야 진행한다. 다르면 멈춘다
- 8B·405B 궤적은 fold 모델 5개로 각각 채점한다. 판별 지표는 5개의 평균 점수로 낸다 (P4-D2). 운영점은 fold 모델별로 적용한 뒤 평균한다 (P4-D6)

**고르는 것이 없다**
- τ·K: 저장된 train OOF 예측에서 `04_m1.py`와 같은 함수로 다시 고른다 (`evaluate.candidate_curves` → `metrics.select_from_curves`, 선택 기준 3.5%, 목적 절약 스텝, 단독·결합). 8B·405B에서 다시 고르지 않는다
- K\*: 70B train에서 고른 값(40)이다 (C4). 제안서 K=35 행을 같이 둔다
- 하이퍼파라미터·N·피처는 Phase 5 그대로다
- 그래서 개발/최종(`--final`) 분리가 없다. 이 데이터로 고르는 값이 하나도 없기 때문이다. `--smoke`는 지표를 출력하지 않고, 70B test 참고 행도 계산하지 않는다

**보고**
- 집합 × 모델 × N 6개: AUROC·PR-AUC·이슈 내 AUROC(쌍 가중 주, 이슈 평균 보조) [95%]
- 운영점(단독·결합): 정지율, 손실 [95%], 판정(구간 상한 ≤5%), 절약 스텝·**절약 토큰**, 같은 집합의 K\* 대비 차이 [95%]
- 같은 집합의 K\*=40 행과 K=35 행
- 질문:
  1. 다른 모델 궤적에서 판별이 70B test 대비 얼마나 떨어지는가
  2. 70B train에서 고른 τ·K의 손실이 다른 모델에서도 판정을 통과하는가. 8B·405B는 궤적이 길어서 K\*=40만으로도 손실이 70B보다 클 것으로 본다 [가정]
  3. 결론이 스텝과 토큰에서 같은 방향인가

**완료 기준** (`summary.json`의 `criteria`)

| 기준 | 내용 |
| --- | --- |
| C1 | 집합이 8B·405B 궤적뿐이고, `splits.parquet`·70B train과 겹치지 않으며, `_unseen` 이슈 ⊂ 70B test 이슈 |
| C2 | 피처가 진행 중 궤적과 1:1이고 결측·무한값이 없다 (`features.check_features`) |
| C3 | 다시 만든 fold 모델이 저장된 M1 예측을 1e-9 안에서 재현한다 |
| C4 | K\* = 40 |
| C5 | 모든 집합·모델·N·정책에 운영점, fold 범위, 구간이 있다 |

- 스크립트 밖에서 확인하는 것: pytest 통과, 재실행 시 `results/phase8/` 바이트 동일

### 한 일
- `code/afp/cross.py`
  - `cross_sets`: 집합을 만들고 검증한다. 8B·405B가 splits에 있거나, 70B가 시도하지 않은 이슈가 있거나, 한 이슈가 70B train/test 양쪽에 있으면 멈춘다
  - `describe`: 집합별 궤적·이슈 수, 성공률, N별 진행 중 궤적 수
- `code/afp/evaluate.py`: `attach_ci`·`operating_table`에 `test_splits` 인자를 더했다 (기본 `("test",)`이라 기존 호출은 그대로다). 구간을 붙일 평가 집합 이름을 넘길 수 있다
- `code/07_m1_cross.py`: 위 사전 등록대로 피처 → fold 모델 다시 만들기·채점(C3) → 운영점(train OOF) → 판별·운영점 적용 → 부트스트랩 → 표
- `code/tests/test_cross.py` (7개): 집합 분할, 잘못된 입력 4종 거부, 진행 중 집계, `test_splits` 구간 부착

### 결정과 근거
- **P8-D1** 범위: 8B·405B만, 보조 실험으로 한다. 발표에 쓸지는 `[결정 필요]`
  - 선택지: (a) 부록 (b) 본문에 "다른 모델 적용" 한 줄 (c) 쓰지 않음
  - 권장: (a) 부록. 2026-11-05 결정을 뒤집지 않는다. 테스트 이슈 수(87·22)가 작아 구간이 넓다
- **P8-D2** 주 집합 = `_unseen`: 70B test 이슈만 쓰면 "같은 이슈의 70B 궤적을 본 모델"이라는 혼입이 없다. `_all`은 표본이 커서 참고로 둔다
  - 기각: `_all`만 쓰기. 이슈 난이도 기억이 섞인다 (B1 없는 M1도 같은 이슈의 70B 궤적에서 이슈 특성을 배운다)
- **P8-D3** 다시 학습하고 재현을 확인: fold 모델 파일이 없어서 다시 만드는 수밖에 없다. 같은 코드·시드·설정에 `n_jobs=1`이라 결정적이고, C3가 같은 모델임을 보장한다
  - 기각: 70B 전체로 모델 하나를 새로 학습하기. Phase 5 모델과 다른 모델이 되고, P4-D6 운영점과도 맞지 않는다
- **P8-D4** 70B 참고 행(`test70b`)은 이 머신의 저장된 예측(Linux)으로 다시 계산한다. 커밋된 Phase 5 결과는 Mac 실행이라 XGBoost 점수와 운영점이 조금 다르다 (P6-D10)
  - smoke에서 `m1_xgb`가 고른 τ·K가 커밋된 `results/phase5/operating_points.csv`의 train 행과 4개 중 0개 같았다. `m1_lr`은 4개 모두 같았다 [확인]. 선택 함수는 같고, XGBoost 예측의 플랫폼 차이 탓이다. 그래서 다른 모델 집합과 70B를 비교할 때는 같은 예측으로 계산한 `test70b` 행과 비교한다. Phase 5 표와 직접 비교하지 않는다
  - `summary.json`의 `chosen_points_same_as_committed_phase5_csv`에 같은 개수를 기록한다

### 검증
- `uv run --no-sync pytest`: 176개 통과 (`test_cross.py` 7개 추가) [확인]
- `--smoke` (집합마다 이슈 25개, N 5·10, 부트스트랩 20회, 약 11초): 완료 기준 C1~C5 모두 통과했다. C3에서 다시 만든 모델의 OOF·test fold 점수가 저장된 예측과 같다 (허용 1e-9). 두 번 실행하면 smoke 산출물(표 5개, parquet 1개)이 바이트 단위로 같다 [확인]
- smoke는 지표를 출력하지 않는다. 70B test 행도 계산하지 않는다

### 산출물
- 코드: `code/afp/cross.py`, `code/07_m1_cross.py`, `code/tests/test_cross.py`, `code/afp/evaluate.py`(`test_splits`)
- 실행 후 `results/phase8/`: `sets.csv`, `discrimination.csv`, `operating_points.csv`(train 행 + 집합별 행 + `b0_cap`), `n_summary.csv`(집합별 K\*·K=35 + 모델 운영점, 절약 스텝·토큰과 K\* 대비 [95%]), `summary.json`
- 실행 후 `data/processed/phase8/m1_cross_preds.parquet` (git 밖): 8B·405B 궤적의 fold 모델별 점수

### 계획과 다른 점
- PLAN 8.1은 2026-11-05에 범위에서 뺐다. 이번에는 사용자 요청으로 보조 실험으로만 되살렸다. 주 결과·분할은 바뀌지 않는다 (`docs/proposal-changes.md` 21번)
- 제안서의 `test_cross`는 "8B·405B 전부"였다. 여기서는 이슈 혼입을 피하려고 70B test 이슈 부분(`_unseen`)을 주로 둔다

### 미결
- P8-D1 `[결정 필요]` 발표에 쓸지와 형식
- 실행 결과 (차승준 실행 후 8.2 결과 절에 적는다)
- 8.2 SWE-bench experiments, 8.3 openhands·terminalbench 축소 모델은 하지 않았다 (위 표의 이유)
