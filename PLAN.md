# 구현 계획 (PLAN)

[docs/proposal.md](docs/proposal.md)의 연구 계획을 실제 코드로 옮기기 위한 작업 계획이다. 제안서가 "무엇을 왜"를 정한다면 이 문서는 "어떤 순서로, 무엇이 끝나면 다음으로 넘어가는지"를 정한다. 방법·지표가 제안서와 다르게 바뀌면 이 문서와 [docs/proposal-changes.md](docs/proposal-changes.md)를 같이 고친다.

표기:
- **[확인]** 저장소 파일이나 데이터를 직접 보고 확인한 사실 (2026-10-29 기준)
- **[가정]** 확인하지 않은 추정. 진행하면서 검증하거나 팀이 정해야 한다
- **[결정 필요]** 팀이 고를 선택지. 11절에 모아 두었다

---

## 1. 프로젝트 요약

코딩 에이전트(SWE-agent)가 GitHub 이슈를 푸는 궤적의 앞 N스텝만 보고 최종 성공 여부를 예측해서, 실패할 실행을 일찍 끊어 비용을 줄인다. 핵심 질문은 두 가지다.

1. N을 얼마나 줄여도 예측이 되는가, 그리고 그 예측력 중 이슈 난이도가 아닌 **궤적 자체의 신호**는 얼마인가 (이슈 내 AUROC).
2. 예측 기반 조기 종료가 **고정 스텝 상한 K=35**(성공 손실 4.9%, 스텝 절약 24.6%)보다 같은 성공 손실(≤5%)에서 더 많이 절약하는가.

과제 일정 [확인, proposal.md §7·proposal-changes.md]: 제안서 제출 10/22 완료, 작업 8~12주차, 최종 발표 13~14주차, 2인 팀이 균등 기여.

## 2. 현재 상태

### 있는 것 [확인]

| 항목 | 상태 |
| --- | --- |
| 제안서 | `docs/proposal.md`, `docs/proposal.pdf`, 수정 내역 `docs/proposal-changes.md` |
| 환경 | `pyproject.toml` + `uv.lock` (Python 3.13). 기본 그룹: numpy, pandas, polars, pyarrow, datasets, scikit-learn, xgboost, shap, matplotlib, tqdm, ipykernel. `dl` 그룹: torch(CUDA 13.0), transformers, accelerate. 이 머신의 `.venv`에는 `dl` 그룹까지 설치되어 있다 |
| 데이터 다운로드 | `data/download.sh`. HF revision 고정 + SHA-256 검증, 이어받기 지원. 이 머신의 `data/`에 4개 데이터셋 모두 받아져 있다 (3.4GB) |
| 분석 코드 | `code/test for K/fixed_cap_k.py` 하나. 고정 상한 K별 잃는 성공·절약 스텝 계산. `code/test for K/test.py`는 `print("Hello world!")`뿐 |
| 테스트·린트·CI | 없음 |

### 데이터 구조 [확인, parquet 스키마와 샘플 행 조회]

- **swe-agent** (주 데이터): 파일 12개, 컬럼 `instance_id, model_name, target, trajectory, exit_status, generated_patch, eval_logs`. **궤적 고유 ID 컬럼이 없다.** `trajectory`는 `{role, text, mask, system_prompt, cutoff_date}` 리스트이고 순서는 `system`(시스템 프롬프트) → `user`(이슈 본문, "We're currently solving the following issue…") → `ai` / `user` 반복이다. 첫 파일 200행 샘플에서 `ai`와 `user` 수가 같았다 (이슈 메시지 1개 + 관측이 마지막 `ai` 뒤에는 없음으로 보임). `exit_status` 값: `submitted`, `submitted (exit_context)`, `exit_context`, `submitted_no_patch`, `early_exit` 등.
- **openhands**: 단일 파일 2.1GB (row group 17개), OpenAI tool-calling 형식(`system`/`user`/`assistant`/`tool`), 라벨은 `resolved`(int). 최대 100 iteration에서 끊긴 궤적이 있다.
- **terminalbench**: `reward` 라벨, 토큰·비용 컬럼, `steps`는 문자열(JSON으로 보임)인데 **null인 행이 있다** (첫 행이 null).
- **swebench-experiments**: git 얕은 클론. `evaluation/lite`에 제출작 84개, `test`에 24개. SWE-agent 제출작이 있다 (예: `20240402_sweagent_gpt4`, `20240402_sweagent_claude3opus`, `20240620_sweagent_claude3.5sonnet`). 궤적 자체는 아직 받지 않았다 (`python -m analysis.download_logs` 필요).

### 제안서에 이미 있는 수치 [확인: 문서에 기록됨 / 이 저장소 코드로 재현된 것은 K 표뿐]

궤적 80,036개, 이슈 3,591개, 에이전트 스텝 2,115,623개, 모델별 70B 74,792 / 8B 4,053 / 405B 1,191, 테스트 이슈 578개, 학습 70B 궤적 62,460개, 같은 모델 테스트 12,332개, 다른 모델 테스트 5,244개, K=35 → 4.9% / 24.6%. 이 수치들은 Phase 1·2의 **검증 기준값**으로 쓴다.

## 3. 최종 산출물 (proposal.md §3~§5 기준)

1. N ∈ {3, 5, 8, 10, 15, 20}별 AUROC·PR-AUC(성공 클래스) 곡선 — B0, B1, M1(±B1), M2, M3
2. N별 이슈 내 AUROC와 계산에 쓴 이슈 수
3. 비용 곡선(성공 손실률 대비 절약 스텝·절약 토큰 비율)과 성공 손실 ≤5%에서 K=35 대비 비교, 상한+예측기 결합 정책
4. M1 SHAP 분석 (초반 실패 신호)
5. 70B 같은 모델 테스트 vs 8B·405B 다른 모델 테스트
6. (선택) SWE-bench/experiments 외부 테스트
7. 발표 자료 (13~14주차)

최소 발표 범위 [확인, proposal.md §7]: M3·외부 테스트가 늦으면 B0·B1·M1·M2로 발표한다. 따라서 **1~5 중 M3를 뺀 부분이 반드시 끝나야 하는 범위**다.

## 4. 격차 (제안서 대비 비어 있는 것)

- 파싱·스텝 테이블, 분할, 평가 하네스, 모든 모델(B1, M1, M2, M3)이 아직 없다. 제안서 일정상 8주차 작업(파싱, 누수 점검, 분할, B0·B1)이 저장소에 없다.
- `fixed_cap_k.py`는 `load_dataset("nebius/SWE-agent-trajectories")`로 HF Hub에서 **revision 고정 없이** 받는다. `download.sh`가 고정한 로컬 parquet와 같은 데이터라는 보장이 없다.
- 궤적 ID가 없어서 스텝·피처·예측 결과를 다시 궤적에 붙일 키를 만들어야 한다.
- 제안서의 "절약 토큰 비율"에 쓸 토큰 수가 swe-agent 데이터에는 없다 (문자 길이 기반 근사 필요).
- README·download.sh는 openhands·terminalbench를 "일반화·도메인 밖 테스트"로 소개하지만 **제안서에는 없다.** 범위 결정 필요 (11절 D1).
- `proposal.md`의 팀 정보 표(이름·학번·담당)가 비어 있다.
- 테스트 프레임워크가 없다. 파싱·지표 같은 핵심 함수는 작게라도 검증 코드가 있어야 한다.

## 5. 공통 설계 결정 (권장)

### 코드 배치

```
code/
├── afp/                 공용 모듈 (agent failure prediction)
│   ├── io.py            로컬 parquet 로드, 경로 상수
│   ├── parse.py         궤적 → 스텝 테이블, 명령 파싱
│   ├── split.py         테스트 이슈 고정, GroupKFold
│   ├── metrics.py       AUROC, PR-AUC, 이슈 내 AUROC, 비용 곡선, 운영점
│   └── features.py      M1 피처
├── 01_parse.py          실행 스크립트 (번호 = 실행 순서)
├── 02_split.py
├── 03_baselines.py
├── ...
└── tests/               pytest
```

- 스크립트를 `code/` 바로 아래에 두면 `uv run python code/01_parse.py`로 실행할 때 `code/`가 `sys.path[0]`이 되어 `import afp`가 된다. **`code`를 패키지로 import하지 않는다** (`python -m code.xxx`는 표준 라이브러리 `code` 모듈과 이름이 겹친다).
- 중간 산출물은 `data/processed/`에 parquet로 저장한다. `.gitignore`의 `data/*` 규칙으로 이미 git에서 빠진다 [확인]. 결과 표·그림은 `results/`(작은 CSV·PNG, git에 올림)에 둔다 [결정 필요 D6].
- 모든 데이터 로드는 `data/` 로컬 parquet에서만 한다. HF Hub 직접 로드 금지 (revision 고정 때문).
- 난수 시드는 `afp/io.py`의 상수 하나로 통일한다.

### 정의 (제안서 그대로, 코드에서 한 곳에만 둔다)

- **궤적 ID** `traj_id`: `"{파일번호:02d}-{파일 내 행 번호:05d}"`. 파일·행 순서는 revision이 고정이라 안정적이다.
- **스텝**: `ai` 메시지 1개 + 바로 뒤 `user` 메시지(관측, 없을 수 있음). 시스템 메시지와 첫 `user`(이슈 본문)는 스텝이 아니다. 이슈 본문은 B1 입력으로 따로 저장한다.
- **시점 N의 대상**: 스텝 수 > N인 궤적만. 입력은 앞 N스텝.
- **라벨**: 모델은 실패 확률을 내고, PR-AUC는 성공 클래스 기준(실패 확률의 반대 점수)으로 잰다.

## 6. 로드맵

각 Phase는 "완료 기준"을 만족해야 다음 Phase가 그 결과를 믿고 쓸 수 있다. Phase 3(평가 하네스)을 모델보다 먼저 만드는 이유는, 모든 모델이 같은 코드로 평가되어야 비교가 성립하기 때문이다.

### Phase 0. 기반 정리 (0.5일)

| 작업 | 내용 |
| --- | --- |
| 0.1 | `code/afp/` 골격과 `io.py`(로컬 parquet 경로, 시드) 작성 |
| 0.2 | `fixed_cap_k.py`를 로컬 parquet 로드로 바꾸고 결과가 기존 값과 같은지 확인 |
| 0.3 | pytest 도입 [결정 필요 D5]: `uv add --dev pytest` 후 `pyproject.toml`에 `[tool.pytest.ini_options] pythonpath = ["code"]`, `testpaths = ["code/tests"]` |
| 0.4 | `proposal.md` 팀 정보 표 채우기, 담당 분배 (10절) |

완료 기준: `uv run python "code/test for K/fixed_cap_k.py"` 출력이 K=30 → 7.0%/29.4%, K=35 → 4.9%/24.6%, K=40 → 3.7%/20.8%, K=50 → 2.2%/15.8%와 일치. `uv run pytest`가 (빈 테스트라도) 통과.

### Phase 1. 파싱과 스텝 테이블 (8주차 몫, 1~2일)

| 작업 | 내용 |
| --- | --- |
| 1.1 | `trajectories.parquet`: 궤적당 1행. `traj_id, instance_id, model_name, target, n_steps, exit_status`(분석 전용), `issue_text` |
| 1.2 | `steps.parquet`: 스텝당 1행. `traj_id, step_idx, action_text, command`(ai 텍스트의 마지막 코드 블록), `command_name`(첫 토큰: `open`, `edit`, `search_dir`, `find_file`, `python`, `submit` …), `obs_text`(또는 길이·앞부분만), `obs_len, ai_len` |
| 1.3 | 명령 이름 사전: 상위 빈도 명령을 실제로 세어 보고 나머지는 `other`로 묶는다 |
| 1.4 | 파일 단위(12개)로 처리해서 메모리 사용을 제한한다. polars `explode` 또는 pyarrow로 `trajectory`를 펼친다 |
| 1.5 | 파서 단위 테스트: 손으로 만든 작은 궤적 2~3개 (관측 없는 마지막 스텝, 코드 블록 없는 ai 메시지, 빈 텍스트) |

완료 기준 (제안서 수치와 대조):
- 궤적 80,036개, 고유 `instance_id` 3,591개, `traj_id` 중복 0
- 전체 스텝(= `ai` 메시지 수) 2,115,623개
- 모델별 74,792 / 4,053 / 1,191, 성공률 16.7%
- 스텝 수 중앙값 성공 12 / 실패 19, 최대 408
- N=10에서 진행 중 궤적 59,708개(성공률 13.8%), N=20에서 34,059개(6.4%)
- 역할 순서가 `system, user, (ai, user)*, [ai]`가 아닌 궤적 수를 세어 보고 처리 방식을 기록

불일치가 있으면 원인을 `docs/proposal-changes.md`에 적고 제안서 수치를 고친다.

### Phase 2. 분할과 누수 점검 (8주차 몫, 1일)

| 작업 | 내용 |
| --- | --- |
| 2.1 | 테스트 이슈 = 8B 또는 405B가 시도한 이슈. `splits.parquet`: `traj_id, split ∈ {train, test_same, test_cross}, fold` |
| 2.2 | 학습 이슈(70B 궤적)에 5-fold `GroupKFold`(그룹 = `instance_id`). 성공률 편차가 크면 `StratifiedGroupKFold` 사용 |
| 2.3 | 누수 점검: (a) `exit_status, generated_patch, eval_logs`가 피처 코드에서 읽히지 않는지 (컬럼을 아예 로드하지 않는 방식 권장) (b) 앞 20스텝 관측에 채점용 숨김 테스트 결과가 섞였는지 — `eval_logs`에 나오는 테스트 이름이 앞 스텝 관측에 나타나는 비율을 샘플로 확인 (c) 시스템 프롬프트가 모든 궤적에서 같은지 해시로 확인 |
| 2.4 | 학습 이슈 3,013개 vs 테스트 이슈 578개의 저장소 구성·이슈별 성공률·궤적 길이 비교 (proposal-changes.md의 남은 일) |

완료 기준:
- 테스트 이슈 578개, 학습 이슈 3,013개, train 70B 궤적 62,460개, test_same 12,332개, test_cross 5,244개
- train과 test 이슈 교집합 0 (테스트로 검증), fold 간 이슈 교집합 0
- 누수 점검 결과와 2.4 비교표를 `results/`에 저장

### Phase 3. 평가 하네스 (모델보다 먼저, 1~2일)

모든 모델은 `traj_id, N, p_fail` 형태의 예측 파일만 내고, 평가는 이 하네스 하나로 한다.

| 작업 | 내용 |
| --- | --- |
| 3.1 | N별 AUROC, PR-AUC(성공 클래스). 무작위 기준값(해당 N의 성공률)을 함께 출력 |
| 3.2 | 이슈 내 AUROC: 성공·실패가 섞인 테스트 이슈 안에서만 계산. 이슈 수를 같이 보고. 집계 방식(쌍 가중 vs 이슈 평균) [결정 필요 D3] |
| 3.3 | 비용 곡선: 정책 "시점 N에서 p_fail ≥ τ면 종료"에 대해 τ를 훑으며 (성공 손실률, 절약 스텝 비율) 계산. 분모는 제안서와 같이 전체 궤적의 전체 스텝 |
| 3.4 | 절약 토큰 근사: 스텝 t의 입력 맥락 ≈ 이전 모든 메시지 문자 수 합. 종료로 절약되는 양 = 남은 스텝들의 맥락 합 [가정: 문자 수 ∝ 토큰 수. D4] |
| 3.5 | 운영점: 성공 손실 ≤5%에서 최대 절약. τ(와 결합 정책의 K)는 **학습 이슈의 out-of-fold 예측으로 고르고** 테스트에서는 고정해서 적용 |
| 3.6 | 결합 정책: "스텝 > K면 종료 또는 시점 N에서 p_fail ≥ τ면 종료" |
| 3.7 | (권장) 이슈 단위 클러스터 부트스트랩으로 AUROC·절약률 95% 구간 |
| 3.8 | 단위 테스트: 완벽 예측기 → AUROC 1, 상수 예측기 → 0.5, 고정 상한 정책을 하네스로 돌리면 K=35 → 4.9%/24.6% |

완료 기준: 3.8의 테스트 통과. B0(고정 상한)을 하네스로 평가한 결과가 `fixed_cap_k.py`와 일치.

### Phase 4. 기준선 B0, B1 (8주차 몫, 1일)

| 작업 | 내용 |
| --- | --- |
| 4.1 | B0: 고정 상한 K (이미 있음 → 하네스로 이전), 에러 횟수 규칙 (앞 N스텝 에러 관측 수 ≥ m이면 종료). "에러 관측" 판별 규칙은 Phase 1 사전에서 정한다 (예: `Traceback`, `Error`, `syntax error(s)` 문구) |
| 4.2 | B1: 이슈 본문 TF-IDF + 로지스틱 회귀 (`sklearn` `TfidfVectorizer` + `LogisticRegression`). 학습은 이슈 단위가 아닌 궤적 단위 라벨이므로 이슈별 성공률을 타깃으로 하는 방식과 비교 [가정: 둘 다 시도할 가치 있음] |
| 4.3 | **B1의 학습 이슈 점수는 out-of-fold로 만든다.** M1이 B1 점수를 입력으로 쓰므로, 같은 이슈로 학습한 B1 점수를 넣으면 이슈 난이도가 누수된다. 테스트 점수는 학습 이슈 전체로 학습한 B1으로 낸다 |
| 4.4 | 참고용 상한: "같은 이슈의 다른 궤적 성공률" 오라클(제안서 AUROC 0.945)을 재현만 하고 비교 기준으로 표시 |

완료 기준: N별 B0·B1 AUROC/PR-AUC/이슈 내 AUROC 표. B1의 이슈 내 AUROC는 정의상 0.5여야 한다 (이슈 안에서는 점수가 같음) — 이 값이 하네스 검증 역할도 한다.

### Phase 5. M1 피처 모델 + SHAP (9~10주차, 2~3일) — 최소 범위의 핵심

| 작업 | 내용 |
| --- | --- |
| 5.1 | 피처 (proposal.md §3): 명령 종류별 빈도, 에러 관측 수, 같은 명령 반복 수(연속·전체), 열어 본 고유 파일 수, `edit` 횟수, 관측 길이 통계. 추가 후보: 이슈 본문에 언급된 파일 경로를 열었는지(운영 중 계산 가능), 마지막 스텝의 명령·에러 여부 |
| 5.2 | 피처는 N별로 **앞 N스텝만** 보고 계산. 누수 방지 테스트: N 이후 스텝을 바꿔도 피처가 같아야 한다 |
| 5.3 | 로지스틱 회귀(표준화) → XGBoost. 각각 B1 점수 포함/제외. 하이퍼파라미터는 학습 이슈 5-fold로 선택. N마다 모델을 따로 학습할지, N을 피처로 넣은 하나의 모델로 할지 [결정 필요 D2] |
| 5.4 | SHAP: XGBoost에 `shap.TreeExplainer`. N=5, 10, 20의 상위 피처와 의존 그림 |
| 5.5 | 사후 분석 전용: 정답 파일 열람 여부(성공 83% / 실패 40%)는 피처로 쓰지 않고 별도 표로만 |

완료 기준: test_same·test_cross에서 N별 지표 표, B1 포함/제외 차이, 성공 손실 ≤5%에서 K=35(24.6%) 대비 절약률. 결과가 K=35보다 나쁘거나 이슈 내 AUROC가 0.5 근처여도 그대로 보고한다 (proposal.md §5).

### Phase 6. M2 LSTM (10주차, 2일)

| 작업 | 내용 |
| --- | --- |
| 6.1 | 입력: 스텝별 `command_name` 임베딩 + 스텝 수치 피처(에러 여부, 관측 길이 등). `torch.nn.LSTM` + `pack_padded_sequence` |
| 6.2 | 하나의 모델로 모든 N 처리: 길이 N으로 자른 시퀀스를 학습에 섞거나, 각 시점의 은닉 상태에서 예측 |
| 6.3 | 클래스 불균형: `pos_weight` 또는 샘플 가중. 검증 fold로 조기 종료 |

완료 기준: M1과 같은 표. 학습 시간 기록 (제안서 예상 10~30분/실험).

### Phase 7. M3 Transformer fine-tune (11주차, 선택 범위)

| 작업 | 내용 |
| --- | --- |
| 7.1 | `transformers` `AutoModelForSequenceClassification` + `answerdotai/ModernBERT-base`, `Trainer`, bf16 |
| 7.2 | 입력: 앞 N스텝 원문, 관측은 스텝당 앞부분만 (예: 200토큰). 최대 길이는 VRAM 12GB에 맞춰 조정 |
| 7.3 | 범위 축소안 (proposal-changes.md 남은 일): N=5, 10, 20만 학습, 학습 궤적은 이슈 균형 샘플링으로 줄인다 [결정 필요 D7] |

완료 기준: 고른 N에서 M1과 같은 표. 실패하거나 시간이 부족하면 "가능한 N까지만" 보고.

### Phase 8. 일반화 테스트 (11~12주차)

| 작업 | 내용 |
| --- | --- |
| 8.1 | 필수: test_cross(8B·405B) 결과를 test_same과 나란히 보고 — Phase 3 하네스로 바로 나온다 |
| 8.2 | 선택(제안서): SWE-bench/experiments의 SWE-agent 제출작 2~3개. `analysis.download_logs`로 궤적을 받아 Phase 1 파서가 읽을 수 있는지 먼저 확인. 형식이 다르면 어댑터 작성 |
| 8.3 | 선택(제안서 밖): openhands·terminalbench. 행동 어휘·메시지 형식이 달라 M1 피처를 그대로 못 쓴다. 하려면 프레임워크 독립 피처(스텝 수, 에러 관측 수, 관측 길이, 반복)만으로 된 축소 모델을 따로 만든다 [결정 필요 D1] |

### Phase 9. 정리와 발표 (12주차)

- 그림: N별 AUROC·PR-AUC·이슈 내 AUROC 곡선, 비용 곡선(K 상한 곡선과 겹쳐서), SHAP 요약
- 표: 성공 손실 ≤5% 운영점 비교, 운영 정책 예시 문장("N=10에서 p_fail ≥ τ면 종료 → 성공 x% 손실, 비용 y% 절감")
- 적용 범위 명시: Llama 계열 + SWE-agent
- `results/`의 표·그림을 처음부터 다시 만드는 스크립트 순서를 README에 기록

## 7. 일정

[가정] 학기 1주차가 8/31에 시작했다면 오늘(10/29)은 9주차이고, 8주차 작업(Phase 1·2·4)이 아직 저장소에 없으므로 약 1주 늦다.

| 주차 | 계획 |
| --- | --- |
| 9주차 (지금) | Phase 0~4 (밀린 8주차 몫 포함) |
| 10주차 | Phase 5 (M1, SHAP), 비용 곡선 확정 |
| 11주차 | Phase 6 (M2), Phase 8.1 |
| 12주차 | Phase 7 (M3, 선택), Phase 8.2 (선택), Phase 9 |
| 13~14주차 | 발표 |

M2를 11주차로 미루고 M3를 12주차로 미룬 것은 지연을 반영한 것이다. 10주차 말까지 Phase 5가 끝나지 않으면 M3를 포기하고 M2와 일반화 테스트에 집중한다.

## 8. 검증 명령 모음

```bash
uv sync --group dl                                   # 환경 (dl 없이 sync하면 torch가 지워짐)
bash data/download.sh                                # 데이터 (이미 받았으면 검증만)
uv run python "code/test for K/fixed_cap_k.py"       # K 기준선 재현
uv run python code/01_parse.py                       # Phase 1 → data/processed/
uv run python code/02_split.py                       # Phase 2 → 분할 수치 출력
uv run pytest                                        # 파서·지표·누수 테스트 (Phase 0.3 이후)
```

각 스크립트는 끝에 완료 기준 수치(궤적 수, 스텝 수, 분할 크기 등)를 출력하고, 기대값과 다르면 `assert`로 멈추게 한다. 무거운 단계(전체 파싱, M3 학습)는 이 명령들과 분리하고 결과 파일이 있으면 건너뛴다.

## 9. 라이브러리 선택

새 의존성 없이 이미 `pyproject.toml`에 있는 것으로 충분하다.

| 필요 | 사용 | 직접 만들지 않는 이유 / 비고 |
| --- | --- | --- |
| 파싱 | polars, pyarrow | 이미 있음. 큰 중첩 리스트를 멀티코어로 처리 |
| 그룹 분할 | `sklearn.model_selection.GroupKFold` / `StratifiedGroupKFold` | 표준 구현 |
| AUROC, PR-AUC | `sklearn.metrics.roc_auc_score`, `average_precision_score` | 이슈 내 AUROC·비용 곡선만 직접 구현 (기존 라이브러리에 이 정의가 없음) |
| B1 | `TfidfVectorizer` + `LogisticRegression` | |
| M1 | `xgboost`, `shap.TreeExplainer` | |
| M2 | `torch.nn.LSTM` | |
| M3 | `transformers` `Trainer` + ModernBERT-base | |
| 테스트 | pytest (dev 의존성 추가 필요) | `unittest`로도 가능하지만 pytest가 간단 |
| 실험 기록 | CSV/JSON 파일 | 실험 수가 적어 MLflow·W&B는 과함 |

## 10. 역할 분담 (제안)

[가정] 두 사람이 균등 기여해야 하므로 각자 모델 하나 이상과 공용 인프라 일부를 맡는다. 팀이 정해서 `proposal.md` 팀 표에 적는다.

| | A | B |
| --- | --- | --- |
| 인프라 | Phase 1 파싱, 스텝 테이블 | Phase 2 분할·누수, Phase 3 하네스 |
| 모델 | B1, M1, SHAP | B0, M2, (M3) |
| 일반화·정리 | 외부 테스트(선택) | test_cross 분석 |
| 발표 | 공동 | 공동 |

Phase 1과 Phase 3은 인터페이스(`steps.parquet` 스키마, 예측 파일 형식 `traj_id, N, p_fail`)만 먼저 합의하면 병렬로 진행할 수 있다.

## 11. 미결 사항

| ID | 결정할 것 | 권장 |
| --- | --- | --- |
| D1 | openhands·terminalbench를 범위에 넣을지 (README에는 있고 제안서에는 없음) | 최소 범위 밖 선택 과제로 둔다. 하게 되면 제안서 수정 내역에 추가 |
| D2 | M1을 N별로 따로 학습할지, N을 피처로 넣은 단일 모델로 할지 | N별 따로 (해석·비교가 쉽고 XGBoost는 빠름) |
| D3 | 이슈 내 AUROC 집계 방식 | 이슈 안 성공×실패 쌍을 모아 계산한 값(쌍 가중)을 주 지표로, 이슈 평균을 보조로 |
| D4 | 토큰 근사 방법 | 문자 수 기반. 필요하면 ModernBERT 토크나이저로 샘플 보정 |
| D5 | pytest 도입 (`pyproject.toml`·`uv.lock` 변경) | 도입 |
| D6 | `results/`를 git에 올릴지 | 작은 CSV·PNG만 올린다 |
| D7 | M3 범위 | N=5, 10, 20만 |
| D8 | 비용 곡선이 단일 시점 정책만 볼지, 여러 시점(N=5, 10, 20에서 차례로 검사) 정책도 볼지 | 단일 시점을 주 결과로, 다중 시점은 시간이 남으면 |

## 12. 리스크

| 리스크 | 영향 | 대응 |
| --- | --- | --- |
| 제안서 수치가 로컬 데이터에서 재현되지 않음 (HF Hub 로드 vs 고정 revision, 스텝 정의 차이) | 모든 비교 기준이 흔들림 | Phase 0.2·1에서 먼저 대조하고 차이를 문서화 |
| B1 점수 누수 (in-fold 점수를 M1에 넣음) | M1 성능 과대평가 | Phase 4.3 out-of-fold 강제, 테스트로 확인 |
| 운영점 τ를 테스트 셋에서 고름 | 절약률 과대평가 | Phase 3.5: τ·K는 학습 OOF에서만 선택 |
| `exit_context`(맥락 초과 종료) 궤적이 길이·라벨과 얽힘 | "길다 = 실패"만 학습 | 시점별 진행 중 궤적만 비교 + 고정 상한과 대조 (제안서 대응 유지), `exit_status`는 분석에만 |
| 테스트 이슈 578개가 무작위 표본이 아님 | 일반화 해석 왜곡 | Phase 2.4 분포 비교를 결과에 첨부 |
| N이 클수록 이슈 내 AUROC 계산 가능 이슈가 줄어 분산이 큼 | 큰 N 결과 불안정 | 이슈 수와 부트스트랩 구간을 같이 보고 |
| 일정 지연 (현재 약 1주 [가정]) | M3·외부 테스트 누락 | 최소 범위(B0·B1·M1·M2)를 먼저 끝냄. 7절 기준 |
| 2.1GB openhands 파일 메모리 사용 | 선택 과제에서만 | row group 단위로 읽기 |
| GPU 환경 (RTX 5070 12GB, CUDA 13.0 + 드라이버 580 이상) [가정: 제안서 §7의 머신이 학습 머신] | M2·M3 지연 | M2는 CPU로도 가능하도록 작게 유지 |
