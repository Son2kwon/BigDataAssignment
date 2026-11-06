# Phase 5 리뷰 체크리스트 (PR #13, `phase/5-m1`)

- 대상: [PR #13](https://github.com/Son2kwon/BigDataAssignment/pull/13) "Phase 5: M1 피처 모델 + SHAP" (작성 손의권)
- 리뷰어: 차승준
- 리뷰 날짜: ____
- 쓰는 법: 각 항목의 `[ ]`를 확인하면 `[x]`로 바꾸고, `메모:` 칸에 결과를 적는다. 문서·코드 위치는 PR #13 브랜치 기준이다

**진행 순서**: ⓪ Phase 6 시작 전 PLAN.md 11절 P6-D1~D4 확정 (결정자 차승준, 아래 ⑤ 전에) · ① 이 체크리스트로 리뷰 → ② 체크리스트를 채운 뒤 PR #13에 **GitHub Review**(Approve 또는 Request changes)로 요약을 남긴다 → ③ 결정 반영 (5절의 결정을 PLAN.md 11절 P5-D3~D7에 확정으로 적는다) → ④ PR #13 머지 (머지는 사람이 한다) → ⑤ 최신 main에서 Phase 6 브랜치 시작

- ②를 하는 이유: PR #1~#12에는 GitHub 리뷰 기록이 없다. 두 사람의 협업(균등 기여) 기록을 PR에 남기려고 리뷰를 GitHub 리뷰 기능으로 한다. 예: `gh pr review 13 --approve --body "체크리스트 docs/reviews/phase5-review.md 완료. 지적 사항 …"` (수정 요청이면 `--request-changes`)

준비:

```bash
git fetch origin
git switch phase/5-m1          # 또는: gh pr checkout 13
uv sync --group dl             # dl 그룹을 이미 받았다면 이 명령으로 (--group dl 없이 sync하면 torch가 지워진다)
brew install libomp            # macOS만. XGBoost가 OpenMP 런타임을 찾지 못하면 import 단계에서 멈춘다
```

---

## 1. 재현성

- [ ] **전체 재실행**: 아래 순서로 실행한다. 각 스크립트가 끝에 "완료 기준 통과"를 출력하는지 본다
  ```bash
  # steps.parquet가 없을 때만
  uv run --no-sync python code/01_parse.py
  uv run --no-sync python code/02_split.py --force
  uv run --no-sync python code/03_baselines.py --force
  uv run --no-sync python code/04_m1.py --force
  uv run --no-sync python code/04b_solution_files.py     # 5.5 사후 분석 (4초)
  uv run --no-sync pytest                                 # 129개 통과가 기대값
  ```
  메모:
- [ ] **결과 diff 없음**: `git status --short results/`와 `git diff --stat results/`가 비어 있다
  - 예외: `results/phase4/b1_grid.csv`의 고르지 않은 C ≥ 1 행에서 소수 다섯째 자리만 다른 것은 플랫폼 차이다 (PLAN.md 5절 "재현성: 결과 파일의 부동소수"). 그 밖의 파일에 diff가 있으면 파일 이름과 차이를 적는다
  - `results/phase5/figures/*.png`는 matplotlib·글꼴 버전이 다르면 바이트가 다를 수 있다 (`docs/phase5.md` 5.6 "재현성"). 그림 내용이 같은지 눈으로 확인하고, 다르면 버전을 적는다
  - `summary.json`의 float는 소수 10자리로 반올림된다 (5.0a). 마지막 자리만 다르면 반올림 경계 차이로 본다
  - 메모 (diff 난 파일):
- [ ] **실행 시간과 머신 사양**: 각 스크립트의 표준 출력 끝 `timings`와 전체 시간
  - 머신 (CPU·코어 수·메모리·OS): ____
  - `03_baselines.py`: ____ (작성자 12코어 M4 Pro 2분 59초) / `04_m1.py`: ____ (4분 28초)
- [ ] **설치 메모**: macOS에서 XGBoost에 `brew install libomp`가 필요하다는 내용이 README 설치 절(`README.md` "2. 패키지 설치")에 있는지 본다. 없으면 6절 지적 사항으로 남긴다
  - 정정 (2026-11-06): 처음 이 체크리스트에 "PR #13의 README에는 없다 [확인]"이라고 적었는데 틀렸다. README "OS별 차이" 절에는 `1f40612`(2026-10-23, 환경 설정 커밋)부터 "macOS에서 `import xgboost`가 libomp 오류를 내면 `brew install libomp`를 실행한다"가 있었다 (확인할 때 검색 결과를 앞부분만 보고 놓쳤다). PR #13 `4eb9ef5`에서 "2. 패키지 설치" 절에도 오류 증상과 함께 넣었다
  - 볼 것: 설치 절의 안내와 오류 문구가 실제 증상과 맞는지 (`brew uninstall libomp` 상태에서 `uv run --no-sync python -c "import xgboost"`를 해 보면 확인할 수 있다. 확인 뒤 다시 설치)
  - 메모:

## 2. 피처와 누수

- [ ] **금지 정보가 섞인 피처가 없다**: `docs/phase5.md` 5.1 "피처 사전" 표의 모든 행을 보고, 각 피처가 `exit_status`, `generated_patch`, `eval_logs`, 정답 패치 정보(정답 파일 열람 포함), 시점 N 이후 스텝을 쓰지 않는지 본다. 코드는 `code/afp/features.py`의 `STEP_INPUT_COLUMNS`(읽는 스텝 컬럼)와 `m1_features`(앞 N스텝 집계)
  ```bash
  uv run --no-sync pytest code/tests/test_features_leakage.py code/tests/test_leakage.py -v
  grep -n "LEAKY\|exit_status\|generated_patch\|eval_logs" code/afp/features.py code/04_m1.py   # 문서 문자열 외에는 없어야 한다
  ```
  - 확인할 테스트: `test_features_ignore_steps_after_n` (N 이후 스텝 변경·삭제·추가·순서 섞기), `test_features_ignore_n_steps_beyond_running`, `test_features_without_leaky_columns` (금지 컬럼을 넣어도 결과가 같음), `test_real_features_one_to_one_and_ignore_later_steps` (실데이터), `test_leakage.py`의 정적 점검(허용 목록 밖 파일에서 누수 컬럼 이름을 쓰면 실패)
  - `b1_score`는 B1 OOF(train) / B1 fold k(test) 점수다. B1 입력에서 "고칠 파일 목록"은 Phase 4에서 이미 지웠다
  - 메모:
- [ ] **"고칠 파일 목록" 꼬리 제거가 피처 경로에 적용됐다** (5.2에서 고침): `code/afp/features.py`의 `issue_file_mentions`가 `baselines.strip_hint`를 먼저 부르는지 본다. 이 함수를 `mentions_table` → `step_values`(`opens_issue_file`) → `m1_features`(`issue_mentions_file`, `issue_file_opened`, `n_issue_file_steps`)가 쓴다
  - 관련 테스트: `test_issue_mentions_ignore_gold_file_hint` (합성 본문), `test_real_features_one_to_one_and_ignore_later_steps`의 마지막 부분 (실데이터 13개 이슈)
  - 고치기 전후 차이: `docs/phase5.md` 5.2 "검증" (세 피처, 궤적 11개)
  - 메모:
- [ ] **5.5 사후 분석이 피처 경로와 분리돼 있다**: 패치 컬럼은 `code/afp/leakage.py`의 5.5 절(`load_patch_files` 등)만 읽는다 (`io.read_raw(..., allow_leaky=True)`). `code/04b_solution_files.py`는 결과 표만 쓰고, `code/04_m1.py`·`code/afp/features.py`는 `leakage`를 import하지 않는다
  ```bash
  grep -nE "^(from|import) " code/04_m1.py code/afp/features.py | grep leakage   # 결과가 없어야 한다 (features.py 문서 문자열에 "afp/leakage.py"가 나오는 것은 설명이다)
  grep -n "allow_leaky" code/ -r
  ```
  - 메모:

## 3. 스텝 단위 API (Phase 6에서 직접 쓸 부분)

- [ ] **M2 입력을 실제로 만들어 본다**: `docs/phase5.md` 5.1 "M2(Phase 6)가 쓸 스텝 단위 API"를 읽고 아래처럼 불러, 시점 N의 스텝 시퀀스(궤적마다 길이 N, 스텝마다 명령 종류 + 수치 값)를 만들 수 있는지 본다
  ```python
  import polars as pl
  from afp import evaluate, features, metrics
  sets = evaluate.load_eval_sets()
  trajs = pl.concat([t.select("traj_id", "instance_id", "n_steps") for t in sets.values()])
  values = features.load_step_values(20, trajs)          # traj_id, step_idx + 스텝 단위 값 (앞 20스텝)
  n = 10
  alive = metrics.running(trajs, n).select("traj_id")
  seq = values.filter(pl.col("step_idx") <= n).join(alive, on="traj_id", how="semi").sort("traj_id", "step_idx")
  print(seq.group_by("traj_id").len()["len"].unique())  # 모두 n이어야 한다
  print(seq.head())
  ```
  - 볼 것: 명령 종류 사전(`features.COMMAND_NAMES` 28종)을 임베딩 인덱스로 바꾸기 쉬운지, 수치 값(`obs_len`, `run_len` 등)의 척도, `open_file`(문자열)을 M2에서 어떻게 쓸지, 시점 N마다 다시 자르는 방식이 학습 루프에 맞는지, 속도·메모리
  - 부족한 값이나 불편한 점:

## 4. 평가와 해석

- [ ] **P4-D6 (fold별 채점)**: M1 test 운영점이 fold 모델 5개에 τ·K를 각각 적용한 결과의 평균인지 본다. `code/04_m1.py`가 `preds.write_test_folds`로 fold별 점수를 저장하고 `evaluate.operating_points(model_preds, ...)`에 `(preds, fold_preds)`를 넘기는지, `results/phase5/operating_points.csv`의 M1 test 행에 `*_fold_min`·`*_fold_max`가 있는지 (완료 기준 C7)
  - 메모:
- [ ] **P4-D7 (B1 fold k 입력)**: `code/04_m1.py`의 `design()`이 train은 `preds.load_feature("b1_n", "train")`, test fold k는 `load_feature("b1_n", "test", fold=k)`를 쓰는지, `oof.oof_predict(..., test_for_fold=...)`로 fold k 모델에 넣는지. 평균 점수(③)를 쓰지 않았다는 확인은 `results/phase5/summary.json`의 `b1_input.test_fold_input_max_spread`(0보다 큼)와 완료 기준 C5. ② 민감도는 `results/phase5/b1_input_sensitivity.csv`, `docs/phase5.md` 5.6
  - 메모:
- [ ] **판정 기준·K\*·참고 행이 표에 있다**: `results/phase5/operating_points_compare.csv`와 `docs/phase5.md` 5.6 운영점 표에 `meets_judge_ci_hi`(test 손실 95% 상한 ≤5%), `b0_cap` `cap` 행(K\*=40), `cap_proposal` 행(K=35)이 있는지. K\* 대비 차이(`d_saved_steps`와 구간)가 같은 부트스트랩 표본 기준인지 (완료 기준 C8: 이 스크립트가 다시 계산한 K\*·K=35 행이 Phase 4와 같음)
  - 메모:
- [ ] **한계가 문서에 적혀 있다**: 아래 셋이 `docs/phase5.md`에 있는지 본다
  - 개발 실행(`--skip-bootstrap --skip-shap`)이 test 지표를 출력했고 작성자가 그것을 읽었다는 사실 → 5.3 "결정과 근거"의 "한계 (그리드 경계)"와 "검증"의 "개발 실행 순서"
  - XGBoost 그리드 경계 (24개 중 23개가 가장 작은 설정) → 5.3 "결정과 근거", "미결"
  - 좋은 N이 탐색 범위 끝(N=20)이라는 점 → 5.6 "좋은 N"
  - 그 밖에: 같은 OOF로 하이퍼파라미터·운영점을 고르는 낙관 (5.3), SHAP을 인과로 읽지 않는다는 점 (5.4), 5.5가 근사라 83%/40%를 재현하지 못했다는 점 (5.5)
  - 메모:
- [ ] **5.6 "절약 토큰" 표** (PR #13 `7f077fa`로 추가): `docs/phase5.md` 5.6 "절약 토큰 (N=10·15·20)" 표의 해석, 즉 "N=10은 토큰 기준 이득이 없다 (결합 정책은 K\*와 같거나 약간 적고 단독은 크게 적다), N=15 결합·N=20은 토큰 기준으로도 K\*보다 유의하게 더 아낀다"가 표 수치와 맞는지 본다. 원 수치는 `results/phase5/operating_points_compare.csv`의 `saved_tokens`, `d_saved_tokens`, `d_saved_tokens_lo`, `d_saved_tokens_hi` (test 행)
  ```bash
  uv run --no-sync python -c "import polars as pl; o=pl.read_csv('results/phase5/operating_points_compare.csv'); print(o.filter((pl.col('split')=='test') & pl.col('model').str.starts_with('m1') & pl.col('N').is_in([10,15,20])).select('model','policy','N','saved_tokens','d_saved_tokens','d_saved_tokens_lo','d_saved_tokens_hi','meets_judge_ci_hi').sort('model','policy','N'))"
  ```
  - 메모:
- [ ] **5.6 "발표 시 주의"** (`7f077fa`): "가장 큰 절약은 N=15·20 시점의 선별이고(진행 중 궤적의 46~54%를 끊음), N=3·5에는 궤적 신호가 거의 없다(이슈 내 AUROC 0.50~0.54)"는 서술이 타당한지 본다. 근거 표: 5.6 "N별 판별 지표"(이슈 내 AUROC), 운영점 표(정지율), `results/phase5/discrimination_compare.csv`. 발표 문장 예시가 K\*=40 대비임을 밝히는지도 본다
  - 메모:
- [ ] **그림 `results/phase5/figures/auroc_by_n.png`** (`7f077fa`): `uv run --no-sync python code/04c_fig_auroc.py`로 다시 만들어지는지, 그림의 점이 5.6 "N별 판별 지표" 표(test AUROC, 이슈 내 AUROC 쌍 가중)와 같은지 본다. 다시 만든 뒤 `git checkout -- results/phase5/figures/auroc_by_n.png`로 되돌릴 수 있다
  - 참고 [확인, 2026-11-06]: 이 체크리스트 작성자의 머신(같은 `.venv`)에서 다시 만들면 **내용은 같지만 바이트가 다르다** (커밋된 파일 97,127바이트, 다시 만든 파일 91,521바이트. 글자·선의 렌더링만 다르고, 같은 머신에서 두 번 만들면 서로 같다). 커밋된 그림을 만든 환경(matplotlib·글꼴)이 다른 것으로 보인다 [가정]. 리뷰어 머신에서도 바이트 diff가 나면 그림 내용만 비교하고, 어느 환경의 그림을 커밋할지 정해서 지적 사항에 적는다
  - 메모:
- [ ] **5.7 보조 실험 결과** (PR #13 5.7): `docs/phase5.md` 5.7의 사전 등록(그리드·절차)이 결과보다 먼저 커밋됐는지(`git log -- docs/phase5.md`), 결과가 사전 등록대로인지, `--final` 없는 실행이 test를 채점하지 않는지(`code/04d_m1_supplementary.py`), 각 실험이 주 결과의 결론을 바꾸는지에 대한 한 줄 정리가 표와 맞는지 본다. 산출물 `results/phase5/supplementary/`. 기존 `results/phase4·5/`·`preds/`가 바뀌지 않았는지도 (`git diff --stat results/`)
  ```bash
  uv run --no-sync python code/04d_m1_supplementary.py            # 개발 모드: train OOF만 출력, 파일 안 씀
  uv run --no-sync python code/04d_m1_supplementary.py --final    # 최종: results/phase5/supplementary/ 생성
  ```
  - 메모:
- [ ] **결과 해석 규칙** (CLAUDE.md "판단이 필요할 때"): 기준선 비교가 K\* 대비라고 적혀 있는지, 하나만 통과한 결과를 강조하지 않고 N 전체의 경향으로 쓰는지 (5.6 "판정을 통과하면서 K\*보다 유의하게 더 아끼는 정책"의 "N 전체의 경향")
  - 메모:

## 5. 결정 필요 (PLAN.md 11절 P5-D3~D7)

PR #13 브랜치의 PLAN.md 11절에서 옮겨 적었다. 2026-11-06 손의권이 다섯 건을 결정했다 (PR #13 PLAN 11절 "확정 (손의권, 2026-11-06), 리뷰어 확인 대기"). 리뷰어는 각 항목에 동의 여부를 적는다. 반대하면 이유를 적고, PLAN.md 11절에 반영하기 전에 둘이 다시 정한다.

### P5-D3 M1 피처 정의 (PR #13에서 "확정"으로 적음)
- 내용: 스텝 단위 값(`features.step_values`)의 앞 N스텝 집계 73개. 비율 피처와 같은 N 안의 선형 변환 피처는 넣지 않는다. 에러·편집 실패는 B0 규칙 값 재사용. 이슈 본문 언급 추출 전에 정답 파일 목록 꼬리를 지운다. M2도 같은 스텝 단위 값을 쓴다
- 함께 볼 미결 (`docs/phase5.md` 5.1 "미결"): 이슈 언급 파일 매칭 규칙(휴리스틱). 선택지 (a) 지금 규칙 유지 / (b) 이름만 있는 언급에서 흔한 이름(`__init__.py`, `setup.py` 등)을 빼고 맞춤. Claude Code 권장: (a). 바꾼다면 train 분포만 보고 정한다
- **결정 (손의권, 2026-11-06)**: 이슈 본문 파일 매칭 규칙 유지
- 근거: 운영 중 계산 가능하고 단순하다. 바꾸려면 train 분포만 보고 정해야 하는데, SHAP(test) 결과를 이미 봤다
- **리뷰어 동의 여부: 동의 · 반대(이유) ___**

### P5-D4 M1 하이퍼파라미터 (그리드 경계)
- 내용: (모델, N)마다 각 8개 그리드에서 train OOF AUROC 최대, 0.002 안이면 앞쪽(단순한) 설정. XGBoost는 가장 작은 설정(깊이 3, 트리 200)이 골라져 최적점이 그리드 경계에 있다. test 결과를 본 뒤라 그리드를 넓히지 않았다
- 선택지: (a) 주 결과를 그대로 두고 경계 문제를 한계로 보고 / (b) 더 작은 모델(깊이 2, 트리 50~100개 등)까지 넓힌 그리드를 보조 실험으로 돌려 train OOF AUROC만 비교
- Claude Code 권장: (a). (b)를 하더라도 주 표는 (a)
- **결정 (손의권, 2026-11-06)**: 주 결과(현재 그리드)는 그대로. 넓힌 그리드는 보조 실험(PR #13 5.7 A2)으로만 하고, 결과로 주 결과를 바꾸지 않는다
- 근거: 그리드 경계를 확인한 개발 실행에서 test 지표를 이미 봤다 (5.3)
- **리뷰어 동의 여부: 동의 · 반대(이유) ___**

### P5-D5 좋은 N 규칙의 세부
- 내용: train OOF AUROC 최대에서 0.005 안의 N 중 train 결합 정책 절약 스텝이 큰 N. 이번에는 네 모델 모두 N=20 (OOF AUROC가 N에 따라 단조 증가해 동률 폭이 결과를 바꾸지 않음). N=20은 탐색 범위의 끝이다
- 선택지: (a) 지금 규칙과 N 후보(3~20) 유지 / (b) N=25·30 등을 보조 실험으로 추가 (train OOF로만 판단)
- Claude Code 권장: (a), N>20은 보조 실험
- **결정 (손의권, 2026-11-06)**: 좋은 N 규칙(동률 폭 0.005) 유지. N=25·30은 보조 실험(5.7 A3)
- 근거: 이번 결과에서 동률 폭이 선택을 바꾸지 않았고, 규칙을 test를 본 뒤 바꾸지 않는다
- **재검토 필요 (2026-11-06, 5.7 A3 결과 반영)**: N=25·30에서도 train OOF AUROC가 계속 올라(N=30 0.73~0.79) 규칙대로면 좋은 N이 20 → 30으로 또 바뀐다. AUROC 우선 규칙은 **주어진 N 범위의 끝을 고른다**. N이 클수록 남은 궤적의 성공률이 낮아져(실패가 대부분) 구분이 쉬워지는 효과가 섞이기 때문이다 [가정]. 반면 train 절약은 N=20(XGB)~25(LR)에서 최대이고, N=25·30 정책은 진행 중 궤적의 61~87%를 끊는다
  - 제안 (손의권): (1) 발표에서는 "좋은 N" 하나를 고르지 않고 **N에 따른 트레이드오프 곡선**(AUROC는 증가, 절약은 N≈20~25에서 최대)을 보여 준다 (2) 꼭 하나를 고른다면 train OOF **절약 스텝을 주 기준**으로 한다 (프로젝트 목적이 비용 절감이므로)
  - 주의: 규칙 변경은 train 정보만으로 정당화되지만 test를 본 뒤의 변경이므로, 바꾸면 그 사실을 PLAN 11절과 `docs/phase5.md`에 함께 기록한다. 주 결과 표(모든 N 보고)는 바뀌지 않는다
  - **리뷰어 의견 (재검토안 동의 여부): ___**
- **리뷰어 동의 여부: 동의 · 반대(이유) ___**

### P5-D6 5.5 정답 파일 출처
- 내용: 정답 패치가 데이터에 없어 근사(같은 이슈 다른 성공 궤적 패치의 비테스트 파일, leave-one-out)로 계산 → 성공 99.7% / 실패 84.3%. 데이터 카드 83% / 40%는 재현되지 않음 (근사는 성공 궤적이 있는 이슈에서만 정의됨). proposal-changes.md 18번
- 선택지: (a) 근사 결과를 한계와 함께 보고 / (b) 이슈 원본 데이터셋을 `data/download.sh`에 revision·SHA-256 고정으로 추가하고 다시 계산
- Claude Code 권장: (a)
- **결정 (손의권, 2026-11-06)**: 5.5는 대리 지표와 한계 보고로 마무리. 정답 패치 데이터는 추가하지 않는다
- 근거: 피처가 아니라 발표의 동기 설명용이고, 방향과 N별 차이는 근사로도 보인다. 다운로드 경로를 바꾸는 비용에 비해 얻는 것이 작다
- **리뷰어 동의 여부: 동의 · 반대(이유) ___**

### P5-D7 M1 단독 정책의 train → test 손실 상승
- 내용: 단독 정책의 손실이 train에서 test로 갈 때 올라간다 (N=10: train 3.50% → test 3.67~4.40%). fold 사이 변동 범위 안인지는 아직 모른다. 출처: PR #13 `docs/phase5.md` 5.6 "미결 (Phase 5 전체)", PLAN.md 11절 P5-D7 (PR #13 `7f077fa`)
- 선택지: (a) M1에도 운영점 이전성 진단을 돌린다 (`evaluate.transfer_rows`: fold k를 뺀 학습 이슈에서 고른 운영점을 fold k·test에 적용. 새 학습 없음) / (b) 하지 않고 한계로만 기록
- Claude Code 권장: (a). 진단 결과로 선택 기준이나 운영점을 바꾸지 않는다 (판정은 이미 test 손실 구간 상한으로 했으므로 판정 결과에도 영향이 없다)
- **결정 (손의권, 2026-11-06)**: M1 운영점 이전성 진단을 실행한다(5.7 A4). 진단용이고 선택 기준이나 운영점을 바꾸지 않는다
- 근거: 손실 상승이 fold 사이 변동 범위 안인지 알아야 한계를 정확히 적을 수 있고, 새 학습이 필요 없다
- **리뷰어 동의 여부: 동의 · 반대(이유) ___**

### 참고 의견
- **그리드 확장과 N>20**: 하더라도 **보조 실험으로만** 한다. train OOF로만 판단하고 주 결과(현재 표)는 바꾸지 않는다. 테스트 결과를 이미 봤다는 사실(5.3 "개발 실행 순서")을 보조 실험 기록에 함께 적는다. 새 CLAUDE.md 규칙(개발 중 테스트 지표 출력 금지, `--final` 분리)을 보조 실험 스크립트에 적용한다
- **5.5 정답 패치**: SWE-bench 원본 데이터셋의 `patch` 필드(정답 패치)를 `instance_id`로 연결하면 데이터 카드 83% / 40%와 직접 비교할 수 있다. HF Hub 직접 로드는 금지이므로 `data/download.sh`에 revision과 체크섬을 고정해 받는다 [가정: 이 데이터의 이슈가 어느 원본 데이터셋에 있는지는 확인하지 않았다. 저장소 이름(`iterative__dvc` 등)으로 보아 SWE-bench 본 데이터셋 밖의 이슈가 섞여 있을 수 있다]. 어려우면 지금의 대리 지표와 한계 보고로 충분하다 (피처가 아니라 발표의 동기 설명용)

## 6. 리뷰 결과

- [ ] 승인
- [ ] 수정 요청

지적 사항 (번호, 위치, 내용, 심각도: 필수 수정 / 권장 / 참고):

| # | 위치 | 내용 | 심각도 |
| --- | --- | --- | --- |
| 1 | `README.md` 설치 절 | macOS XGBoost용 `brew install libomp` 안내. **상태: PR #13 `4eb9ef5`에서 README "2. 패키지 설치" 절에 반영함 (오류 증상 포함), 리뷰어 확인 필요.** 참고: "OS별 차이" 절에는 원래 한 줄 있었다 (1절 정정) | ____ |
| 2 | `results/phase5/figures/auroc_by_n.png` | `04c_fig_auroc.py`로 다시 만들면 내용은 같고 바이트가 다르던 문제. **상태: PR #13 `8f8a7d0`(A0)에서 Mac `.venv`로 다시 생성함** (두 번 실행해 바이트 단위로 같음을 확인, 환경 메모는 `19ec43b`). 리뷰어 머신(다른 OS)에서는 바이트가 다를 수 있다 (플랫폼 차이) | ____ |
| 3 | | | |
| 4 | | | |

Phase 6에 넘길 메모 (스텝 단위 API, 평가 함수 사용 등):
