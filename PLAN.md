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
| 분석 코드 | `code/afp/` 공용 모듈(`io, parse, split, leakage, metrics, preds, oof, evaluate, baselines`), 실행 스크립트 `code/01_parse.py`(Phase 1), `code/02_split.py`(Phase 2), `code/03_baselines.py`(Phase 4). 평가 하네스 `afp/metrics.py`(Phase 3). 예전 `code/test for K/fixed_cap_k.py`(HF Hub 로드)도 남아 있다 |
| 중간 산출물 | `data/processed/trajectories.parquet, steps.parquet, splits.parquet, context_costs.parquet, preds/*.parquet` (git 밖). 검증 수치·결과 표 `results/phase1/`, `results/phase2/`, `results/phase4/` (git 안) |
| 테스트·린트·CI | pytest(dev 그룹) 95개, `code/tests/`. 린트·CI 없음 |
| 진행 | Phase 0(0.2·0.4 일부 제외)~4 완료 (3.5 운영점 선택 포함). 다음은 Phase 5 (M1) |

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

2026-10-31 기준. 파싱·스텝 테이블(Phase 1), 분할·누수 점검(Phase 2), 평가 하네스(Phase 3, 3.5 포함), 기준선 B0·B1·오라클(Phase 4)은 끝났다. 궤적 ID(`traj_id`), 토큰 근사(문자 수 기반 맥락 길이), pytest, 공용 예측 파일·OOF·평가 드라이버도 생겼다. 남은 것:

- 제안 모델 M1, M2, M3가 아직 없다 (Phase 5~7). B1 OOF 점수(`preds/b1_n.parquet`)는 M1 입력으로 준비됐다.
- `fixed_cap_k.py`는 여전히 `load_dataset("nebius/SWE-agent-trajectories")`로 HF Hub에서 **revision 고정 없이** 받는다. 같은 값은 이제 `01_parse.py`(로컬 parquet)와 하네스가 재현하므로 이 스크립트는 참고용이다 (0.2).
- README·download.sh는 openhands·terminalbench를 "일반화·도메인 밖 테스트"로 소개하지만 **제안서에는 없다.** 범위 결정 필요 (11절 D1).
- `proposal.md` 팀 표의 이름·학번은 채워졌지만 **담당 칸이 비어 있다** (0.4).

## 5. 공통 설계 결정 (권장)

### 코드 배치

```
code/
├── afp/                 공용 모듈 (agent failure prediction)
│   ├── io.py            로컬 parquet 로드, 경로 상수
│   ├── parse.py         궤적 → 스텝 테이블, 명령 파싱
│   ├── split.py         테스트 이슈 고정, StratifiedGroupKFold (Phase 2 결과 참고)
│   ├── leakage.py       사후 누수 점검 전용 (채점 로그는 여기서만 읽음)
│   ├── metrics.py       평가 집합 로더, AUROC, PR-AUC, 이슈 내 AUROC, 비용 곡선, 고정 상한, 운영점
│   ├── preds.py         예측 파일 data/processed/preds/<model>.parquet 쓰기·읽기·검증 (Phase 4)
│   ├── oof.py           fold별 OOF 예측 + 테스트 fold 평균 (Phase 4, M1~M3 재사용)
│   ├── evaluate.py      모델 하나 → 분할×N 판별 표, 비용 곡선, 운영점, 부트스트랩 stat (Phase 4)
│   ├── baselines.py     B0 규칙, B1 TF-IDF+LR, 오라클 (Phase 4)
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

#### Phase 0 상태 (2026-10-29)

- 0.1 완료: `code/afp/io.py` (경로 상수, `SEED = 42`, 컬럼을 명시하는 로더)
- 0.2 목적은 달성, 스크립트는 그대로: K=30/35/40/50 값을 로컬 parquet에서 `01_parse.py`(`results/phase1/summary.json`의 `fixed_cap`)와 Phase 3 하네스가 둘 다 재현한다. `fixed_cap_k.py` 자체는 아직 HF Hub에서 로드한다
- 0.3 완료: pytest가 dev 그룹에 있고 `pyproject.toml`에 `pythonpath`·`testpaths` 설정
- 0.4 일부: `proposal.md` 팀 표에 이름·학번은 있고 담당 칸은 비어 있다

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

#### Phase 1 결과 (2026-10-29 완료)

`uv run python code/01_parse.py` (약 1.5분) → 위 완료 기준 **모두 일치** [확인]. 제안서 수치 수정 없음. 검증 수치는 `results/phase1/summary.json`, 명령 빈도표는 `results/phase1/command_counts.csv`.

- 추가로 일치 확인: 이슈당 궤적 중앙값 16·최대 125, 한 번도 성공 못 한 이슈 2,753개, 성공·실패 섞인 이슈 807개(궤적 34,922개), K=30/35/40/50 고정 상한 값(로컬 parquet 기준이라 0.2의 재현 목적도 겸함)
- 역할 순서 이상 궤적 **0개**. 마지막 `ai` 뒤 관측이 없는 궤적 80,035개(1개 빼고 전부). 스텝 0개 궤적 1개 (이슈만 있고 `ai` 없음)
- 시스템 프롬프트 해시가 80,036개 모두 같다 (2.3 (c) 완료)
- 모든 `ai` 메시지에 코드 블록이 있다 (`command` null 0개)

확정한 규칙 (코드는 `code/afp/parse.py` 한 곳):
- `step_idx`는 **1부터**. "앞 N스텝" = `step_idx <= N`
- `command` = 중첩되지 않은 마지막 ``` 블록 (SWE-agent ThoughtActionParser와 같은 규칙)
- `command_head` = 첫 토큰의 식별자 부분. 모델이 `find. -name`, `cd..`, `ls.github`처럼 공백을 빼먹은 명령을 자주 내서 `find`, `cd`, `ls`로 정규화한다 (원문은 `command`에 남음)
- `command_name` = 빈도 0.1% 이상 26개(누적 96.7%) 사전, 나머지 `other`(3.3%), 코드 블록 없음 `none`
- `is_error` (스텝의 30.0%): Traceback, `XxxError:`/`Exception:` 줄, 편집 거부("Your proposed edit has introduced new syntax error"), `command not found`, `No such file or directory`, bash `syntax error near unexpected token`, SWE-agent `File/Directory … not found`, `usage:` 줄. 샘플 2만 개에서 매칭이 모두 실제 에러임을 확인. `edit_failed`(11.6%)는 편집 거부만 따로 표시
- `open_file`, `cwd`: 관측 끝 `(Open file: …)(Current directory: …)` 줄에서 파싱. M1 "열어 본 고유 파일 수"에 쓴다

산출물 스키마 (Phase 2·3·5와의 인터페이스):
- `data/processed/trajectories.parquet` (2.4MB): `traj_id, file_idx, row_idx, instance_id, model_name, target, n_steps, exit_status`(분석 전용)`, issue_text`(ISSUE:~INSTRUCTIONS: 사이 본문)`, issue_msg_len, system_len, system_hash, role_pattern_ok, roles_head`
- `data/processed/steps.parquet` (308MB, 원문 전체 포함): `traj_id, step_idx, action_text, command, command_head, command_name, obs_text, ai_len, obs_len, is_error, edit_failed, open_file, cwd`. 텍스트가 필요 없으면 `pl.scan_parquet(...).select(...)`로 컬럼을 골라 읽는다
- 3.4 토큰 근사용 길이: `system_len + issue_msg_len + Σ(ai_len + obs_len)`

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

#### Phase 2 결과 (2026-10-29 완료)

`uv run python code/02_split.py` (약 10초, `01_parse.py` 결과 필요) → 위 완료 기준 **모두 일치** [확인]. 제안서 분할 수치 수정 없음. 검증 수치는 `results/phase2/summary.json`. 최종 검증 증거(실행 명령·종료 코드·pytest 요약·재실행 해시 비교)는 `results/phase2/verification.json`.

- 궤적 80,036개 = train 62,460 + test_same 12,332 + test_cross 5,244 (8B 4,053 + 405B 1,191). 학습 이슈 3,013개, 테스트 이슈 578개, 교집합 0, fold를 넘는 학습 이슈 0
- 출처: 원본 12개의 크기·SHA-256이 `data/download.sh` 고정값과 일치하고, `trajectories.parquet` 80,036행이 원본 (파일, 행)의 `instance_id, model_name, target`과 모두 같다
- 결정성: 입력 행 순서를 섞어도 같은 분할이 나오고, 다시 실행하면 `splits.parquet`와 `results/phase2/` 전체가 바이트 단위로 같다. 저장된 파일이 재계산과 다르면 멈춘다 (`--force`로 덮어씀)
- 모르는 `model_name`, null, `traj_id` 중복, bool이 아닌 `target`은 버리지 않고 에러로 멈춘다

확정한 규칙 (코드는 `code/afp/split.py` 한 곳):
- `splits.parquet`: `traj_id, split, fold`만. `fold`는 train 0~4, **test_same·test_cross는 -1** (`split.HELDOUT_FOLD`, 어떤 fold에도 속하지 않음). 분할 방법·seed·sklearn 버전은 parquet 메타데이터 `afp.split`에 기록
- fold는 `GroupKFold`가 아니라 **`StratifiedGroupKFold(5, shuffle=True, random_state=42)`** (그룹 = `instance_id`, 층화 = `target`). GroupKFold로는 이슈 성공률 편차 때문에 fold 성공률이 14.4%~17.5%(폭 3.1%p, 상대 19%)로 벌어져 fold마다 PR-AUC 기준값이 달라진다. StratifiedGroupKFold는 16.2%~16.3%(폭 0.09%p), fold 크기 차이 32궤적. `results/phase2/fold_balance.csv` (2.2의 "편차가 크면" 조건 해당)
- 테스트 이슈 중 70B 궤적이 없는 이슈는 0개 (모든 테스트 이슈에 test_same이 있음)

누수 점검 (2.3):
- (a) 피처 쪽 로드는 `io.load_trajectories(columns)`, `io.scan_steps(columns)`로 컬럼을 **명시**해서 읽고, 누수 컬럼(`io.LEAKY_COLUMNS`)을 요청하면 에러. `io.read_raw`도 `allow_leaky=True` 없이는 거부. 정적 점검(`leakage.scan_code`, AST 기준이라 주석·docstring은 제외)으로 누수 컬럼 이름을 쓰는 파일이 허용 목록(`afp/io.py`, `afp/leakage.py`, `01_parse.py`) 밖에 생기면 pytest가 실패한다. `splits.parquet`에는 누수 컬럼이 없다. 원본의 결과 전용 컬럼은 사후 점검 모듈 `afp/leakage.py`에서만 읽는다
- (b) 채점 로그(`eval_logs`) 구조를 전수 조회해서 파서를 만들었다: 모두 SWE-bench 형식 하네스 로그이고 테스트 실행은 전부 `pytest -rA`. 80,036개 중 null 9,397, 테스트 구간 없음(패치 적용 실패 등) 1,454, 구간이 잘림 211. 식별자 = `-rA` 요약 줄·verbose 줄의 node id, 테스트 이름, 실행 대상 테스트 파일, 테스트 패치 파일
  - **전수(1,208,185개 관측, 앞 20스텝)**: 하네스 전용 문자열(`___TESTS_SECTION___`, `test_patch.patch`, `prediction_patch.patch`, run_tests·apply_patch 안내문 등) **0건**. 하네스 디렉터리 이름 `swebench_matterhorn`은 104개 궤적·137개 관측에 나온다. 124개는 에이전트의 `ls`·`cd`·`find` 명령 결과(주로 `ls /` 루트 목록), 나머지 13개는 `python`(6)·`edit`(4)·`find_file`(2)·`create`(1) 명령 뒤 관측이다. 이 137개 관측에도 위 하네스 전용 문자열은 없다
  - **표본 4,000개** ((model, target) 층화, seed 42, 테스트 식별자를 꺼낸 궤적 3,158개, 식별자 162,733개): 채점 로그의 테스트 이름이 앞 20스텝 관측에 나온 궤적 295개(9.3%), 흔한 이름(3개 이상 저장소에 나오는 `test_repr` 등) 제외 252개(8.0%), node id 일치 30개(0.95%). 처음 나온 스텝의 명령 맥락을 규칙(`leakage.classify`, 명령 이름·정규식 기반 휴리스틱)으로 나누면: 저장소 파일 보기(`open`·`search_*`·`grep` 등, 기존 테스트 코드) 852건(테스트 파일 경로 2건 포함), 에이전트의 테스트 실행 141건, 에이전트가 직접 쓴 이름(테스트 파일 생성 등) 97건, 이슈 본문에 있는 이름 38건, 어느 범주에도 안 들어가는 경우(`other`)와 하네스 문자열이 함께 나온 경우 각 0건. pytest 결과 줄 형태로 나온 32개 궤적은 모두 그 스텝 명령이 테스트 실행 패턴(`pytest` 등)에 맞았다. 범주는 맥락이 정상 경로와 **부합한다는 뜻일 뿐**이다. 예를 들어 `repo_view`는 "기존 테스트 코드를 봤다"와 맞지만 숨김 테스트가 컨테이너에 없었다는 증명은 아니다
  - 대조군: 같은 저장소의 다른 이슈 로그를 붙이면 5.8%, 다른 저장소 로그를 붙이면 0.9%(흔한 이름 제외 0.16%)가 겹친다. 관측된 9.3%가 같은 저장소 대조군과 같은 자릿수라서 "같은 저장소의 기존 테스트가 보였다"와 부합한다 (원인을 증명하지는 않음)
  - 결론: 이 점검들에서 숨김 채점 결과가 앞 20스텝 관측에 섞였다는 **직접 증거(하네스 문자열, 테스트 실행 없이 나온 채점 테스트 결과 줄)는 찾지 못했다**. 이것이 누수가 없다는 증명은 아니다. 맥락 범주는 휴리스틱이고, 숨김 테스트 패치·채점 대상 테스트 목록이 데이터에 없으며, 글자 그대로의 겹침 대조로는 부재를 확립할 수 없다. 관측이 이슈 단위로 묶여 있고(독립 표본이 아님) 탐지 민감도를 모르므로 0건에서 누수율 상한을 계산하지 않는다
  - 한계: 데이터에 test_patch·FAIL_TO_PASS 목록이 없어 "숨김 테스트 패치가 새로 추가한 테스트"를 직접 알 수 없다. 근사로 쓴 "되돌리지 않은 .py 테스트 패치 파일 = 새 파일" 규칙은 거짓 양성이 있다 (표본에서 걸린 2개 파일은 에이전트가 1스텝 `search_dir`로 본 기존 파일이었음). 채점 로그가 없거나 테스트 구간이 없는 궤적(표본의 21%, 전부 실패 궤적)은 식별자 대조를 못 했고 전수 하네스 문자열 점검만 적용된다. 겹침은 글자 그대로 일치만 본다
- (c) 시스템 프롬프트 해시 1종(`bfc71085e702`, 80,036개) — Phase 1 메타데이터로 다시 확인

학습/테스트 이슈 비교 (2.4, 두 쪽 모두 70B 궤적, `results/phase2/issue_comparison.csv`, `repo_composition.csv`, `split_counts.csv`):

| | 학습 이슈 | 테스트 이슈 |
| --- | --- | --- |
| 이슈 / 저장소 | 3,013 / 1,148 | 578 / 377 |
| 궤적 성공률 | 16.3% | 18.7% |
| 이슈 성공률 평균 | 9.7% | 11.8% |
| 성공 0 / 성공·실패 섞임 이슈 비율 | 77.3% / 21.6% | 73.7% / 25.4% |
| 스텝 수 중앙값 (평균, p90) | 17 (26.0, 50) | 17 (25.0, 47) |

- 테스트 이슈가 약간 쉽다 (성공률 +2.4%p, 섞인 이슈 비율 +3.8%p). 이슈 내 AUROC를 계산할 수 있는 섞인 테스트 이슈는 test_same 147개, test_cross 98개
- 저장소 구성이 크게 다르다: 공통 저장소 249개, 테스트 이슈의 76.5%(442개)만 학습에도 있는 저장소에서 나온다. 저장소 비율의 총변동거리 0.63. 학습 최다 저장소 `iterative__dvc`(학습 이슈의 8.6%)는 테스트에 없고, 테스트 최다 `pydantic__pydantic`은 테스트의 8.7%(학습 5.0%). 테스트 결과는 일부 "처음 보는 저장소" 일반화를 포함한다
- test_cross 궤적은 70B보다 길다 (스텝 중앙값 8B 26, 405B 27 vs 70B 17). 405B 성공률 25.9%, 8B 15.2%

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

#### Phase 3 결과 (2026-10-29 완료)

완료 기준 **모두 충족** [확인, 2026-10-29 main `ea48b0f`에서 다시 실행]. 3.5(운영점 선택)는 아래 "Phase 3.5 결과"에서 따로 마쳤다.

- `uv run --no-sync pytest`: **56개 통과** (1.6초). `01_parse.py`·`02_split.py` 검증 모드도 "완료 기준 통과", 다시 실행해도 `results/`가 바뀌지 않는다 (git diff 없음)
- 3.8 단위 테스트: 완벽 예측기 → AUROC·PR-AUC 1, 상수 예측기 → AUROC 0.5·PR-AUC = 성공률, 고정 상한 = "N=K에서 모두 끊기"(`test_metrics.py`의 합성 데이터 테스트)
- 3.8 실데이터 (pytest 밖, 별도 스크립트로 확인): 진행 중 궤적에 모두 p_fail=1을 준 하네스 결과가 제안서·`fixed_cap_k.py` 값과 일치하고, numpy로 따로 계산한 값과 1e-12 안에서 같다

| K | 성공 손실 | 절약 스텝 | 절약 토큰(근사) |
| --- | --- | --- | --- |
| 30 | 7.0% | 29.4% | 50.0% |
| 35 | 4.9% | 24.6% | 42.2% |
| 40 | 3.7% | 20.8% | 35.8% |
| 50 | 2.2% | 15.8% | 27.1% |

- 분할별 K=35 (손실 / 절약 스텝 / 절약 토큰): train 4.3% / 24.2% / 41.6%, test_same 4.5% / 21.8% / 40.2%, test_cross 12.4% / 32.5% / 49.7%. **모델 비교의 K=35 기준값은 전체(4.9%/24.6%)가 아니라 같은 평가 집합의 값을 쓴다.** test_cross는 궤적이 길어서 K=35가 손실 5%를 넘는다
- 테스트 이슈(test_same + test_cross)에서 완벽 예측기 N=10 → AUROC·이슈 내 AUROC 1.0, 상수 예측기 → 0.5·0.5, PR-AUC 0.1526 = 성공률. 결합 정책(N=10, K=35)의 τ=inf 행이 K=35 단독(6.73% / 25.80%)과 같다

#### Phase 3.5 결과 (2026-10-29 완료)

`metrics.select_operating_point(trajs, preds, N, caps=(None, 30, 35, ...), max_loss=0.05, by="saved_steps", costs=None)`. 평가 집합 로더 `metrics.eval_set`·`eval_costs`도 같이 추가했다 (아래). 테스트 `uv run --no-sync pytest` 64개 통과.

- 상한 후보 K마다(`None` = 상한 없음) 비용 곡선을 만들고, 성공 손실 ≤ `max_loss`(`metrics.MAX_LOSS` = 5%)인 행 중 `by`(`saved_steps` 또는 `saved_tokens`)가 가장 큰 (K, τ)를 고른다. 동률이면 손실이 작은 것 → τ가 큰 것(덜 끊음) → K가 큰 것
- τ=inf가 뽑히면 예측기를 쓰지 않는 쪽(상한 K만)이 낫다는 뜻이다. 손실 ≤5%인 행이 하나도 없으면 멈춘다 (예: K 후보가 모두 상한만으로 5%를 넘을 때)
- 사용법: **학습 이슈(`eval_set("train")`) 궤적과 그 OOF 예측**을 넣어 고르고, 테스트에서는 `point_metrics(test, test_preds, N, tau=op["tau"], cap=op["cap"], costs=)`로 그대로 적용한다. 함수는 입력이 OOF인지 검사하지 않으므로 호출하는 쪽이 지킨다
- 테스트: 손으로 센 예제, 후보가 없는 경우, 궤적마다 정책을 직접 적용한 참조 구현으로 모든 (K, τ)를 훑은 최적값과 일치 (`saved_steps`·`saved_tokens` 각각), 고른 값을 `point_metrics`로 다시 적용하면 같은 값
- 실데이터 동작 확인 [확인, 무작위 점수라 결과가 아닌 동작 확인용]: train 62,460궤적, N=10, K 후보 (없음, 30, 35, 40, 50) → K=35 + τ=0.989 (1.2%를 추가로 끊음), 손실 4.94%, 절약 스텝 24.7%, 0.1초. K=35 단독의 train 손실(4.3%)과 5% 사이 여유를 예측기로 채운다. **학습 OOF에서 한도에 거의 맞춰 고르므로 테스트 손실은 5%를 넘을 수 있다.** 테스트 운영점의 손실은 부트스트랩 구간(3.7)과 함께 보고한다

평가 집합 로더 — 모든 모델이 같은 평가 집합을 쓰도록 하네스 입력은 이것으로만 만든다:

```python
trajs = metrics.eval_set("test_same")   # train / test_same / test_cross
costs = metrics.eval_costs(trajs)       # 절약 토큰이 필요할 때
op = metrics.select_operating_point(metrics.eval_set("train"), oof_preds, N, caps=(None, 30, 35, 40, 50))
metrics.point_metrics(trajs, test_preds, N, tau=op["tau"], cap=op["cap"], costs=costs)
```

- `eval_set(name)`: 출력 `traj_id, instance_id, target, n_steps, fold`. N 전에 끝난 궤적도 포함한다 (비용 곡선 분모). splits의 궤적이 trajectories에 없으면 멈춘다. `io.load_splits()`는 `splits.parquet`가 없으면 `02_split.py` 실행을 안내하고 멈춘다
- **test_same·test_cross를 합친 집합은 받지 않는다.** 두 분할은 테스트 이슈 578개를 공유해서, 합치면 578개 이슈 모두에 70B와 8B·405B 궤적이 섞이고 이슈 내 AUROC가 모델 종류 차이까지 잰다. 위 "Phase 3 결과"의 합친 테스트 집합 값(6.73% / 25.80%)은 동작 점검용이고, 결과는 분할별로 따로 보고한다
- 실데이터 테스트(`test_real_eval_sets_and_fixed_cap_baseline`, 데이터가 있을 때만): 세 집합의 궤적 수와 분할별 K=35 기준값(위 표)이 맞는지 확인한다

아래 두 절은 작업 당시 기록이다.

#### Phase 3 진행 (2026-10-29, 3.1~3.4 완료)

`code/afp/metrics.py`, 테스트 `code/tests/test_metrics.py` (13개, 전체 `uv run pytest` 20개 통과). Phase 2 분할 없이 만들었다. 평가 집합은 `trajs` 인자로 받으므로 `splits.parquet`이 나오면 `test_same`·`test_cross`로 거른 `trajs`를 넘기면 된다.

- `align(trajs, preds, N)`: 시점 N 예측이 진행 중 궤적(`n_steps > N`)과 정확히 1:1인지 확인 (빠짐·남음·중복·NaN이면 멈춤)
- 3.1 `discrimination`: AUROC(p_fail → 실패), PR-AUC(성공 클래스, 점수 −p_fail), `base_rate`(성공률)
- 3.2 `within_issue_auroc`: 이슈별 Mann–Whitney U (동점 0.5). `pair_weighted`(주)와 `issue_mean`(보조)을 둘 다 내고 이슈·궤적·쌍 수를 같이 보고 (D3 권장안대로 구현, 팀 확정 필요)
- 3.3 `cost_curve`: τ를 p_fail 고유값으로 훑고 동점은 함께 끊는다. 분모는 평가 집합 전체 성공·전체 스텝
- 3.4 `context_costs`, `tokens_after`: 스텝 t 맥락 = `system_len + issue_msg_len + Σ_{s<t}(ai_len + obs_len)` (문자 수). `cost_curve(..., costs=)`가 `saved_tokens` 열을 추가

실제 데이터 확인 [확인]: 모든 진행 중 궤적에 p_fail=1을 주면 K=30/35/40/50의 성공 손실·절약 스텝이 제안서 값과 일치한다. 같은 정책의 **절약 토큰(근사)은 K=30 50.0%, K=35 42.2%, K=40 35.8%, K=50 27.1%**이다. 맥락이 스텝마다 길어지므로 뒤쪽 스텝이 더 비싸서, 토큰 절약률이 스텝 절약률보다 훨씬 크다. 토큰 기준으로 비교할 때는 K=35 기준값이 24.6%가 아니라 42.2%다.

#### Phase 3 진행 (2026-10-29, 3.6~3.7 완료, 3.5는 Phase 2 이후)

테스트 `uv run pytest` 29개 통과.

- 3.6 `cost_curve(..., cap=K)`: 결합 정책 "스텝 > K 또는 시점 N에서 p_fail ≥ τ" (N < K). τ=inf 행이 고정 상한 K 자체다. 궤적마다 정책을 직접 적용한 참조 구현과 (N, K) 4조합 × τ 5개에서 손실·절약 스텝·절약 토큰이 일치하는지 테스트
- `at_threshold(curve, τ)`: 고정한 τ를 적용한 행 (3.5에서 학습 OOF로 고른 τ를 테스트에 적용할 때 쓴다)
- 3.7 `bootstrap(trajs, preds, stat)`: 이슈를 복원 추출하고, c번 뽑힌 이슈의 궤적에 `weight=c`를 준다. 모든 지표 함수가 선택 컬럼 `weight`를 받는다 (없으면 1). 가중치 결과가 이슈를 실제로 복제한 데이터의 결과와 같은지 테스트. 백분위 95% 구간, 기본 1,000회, `io.SEED`
- `point_metrics(trajs, preds, N, tau=, costs=, cap=)`: 부트스트랩 `stat`으로 쓰는 지표 묶음 (AUROC, PR-AUC, 이슈 내 AUROC 2종, 운영점 손실·절약)
- 재현성: polars join·group_by의 행 순서가 실행마다 달라 합이 1e-16 수준으로 흔들렸다. `align` 출력과 이슈별 표를 정렬해 결과를 비트 단위로 고정했다

실제 데이터 동작 확인 [확인, 분할 전이라 결과가 아닌 동작 확인용]: 임시 점수 "앞 10스텝 에러 관측 비율"로 N=10, K=35 결합 정책을 돌리면 τ=inf 행이 K=35 값(4.9%/24.6%/42.2%)과 같다. 테스트 셋 크기(578이슈, 13,080궤적)에서 부트스트랩 100회 2.1초 (1,000회 약 20초). 이 크기에서 운영점 **성공 손실의 95% 구간 폭이 약 ±1.4%p**(4.7% → 3.4~6.2%)라, "손실 ≤5%" 제약을 만족했는지는 구간과 함께 보고해야 한다.

> 통합 메모 (Phase 2 브랜치, 2026-10-29): 위 Phase 3 기록의 "Phase 2 분할 없이", "분할 전이라"는 하네스를 만들 당시 상황이다. 지금은 `data/processed/splits.parquet`(`code/02_split.py`)가 있다. 실제 분할로 하네스를 돌려 B0 고정 상한 K=35(N=35에서 진행 중 궤적 전부 끊기)를 확인한 동작 점검 값: 전체 4.9%/24.6%(제안서와 일치), train 4.3%/24.2%, test_same 4.5%/21.8%, test_cross 12.4%/32.5% (성공 손실/절약 스텝). test_cross 궤적이 길어서 같은 K로 성공을 훨씬 많이 잃는다. 하네스 코드와 위 기록은 바꾸지 않았다.

### Phase 4. 기준선 B0, B1 (8주차 몫, 1일)

| 작업 | 내용 |
| --- | --- |
| 4.1 | B0: 고정 상한 K (이미 있음 → 하네스로 이전), 에러 횟수 규칙 (앞 N스텝 에러 관측 수 ≥ m이면 종료). "에러 관측" 판별 규칙은 Phase 1 사전에서 정한다 (예: `Traceback`, `Error`, `syntax error(s)` 문구) |
| 4.2 | B1: 이슈 본문 TF-IDF + 로지스틱 회귀 (`sklearn` `TfidfVectorizer` + `LogisticRegression`). 학습은 이슈 단위가 아닌 궤적 단위 라벨이므로 이슈별 성공률을 타깃으로 하는 방식과 비교 [가정: 둘 다 시도할 가치 있음] |
| 4.3 | **B1의 학습 이슈 점수는 out-of-fold로 만든다.** M1이 B1 점수를 입력으로 쓰므로, 같은 이슈로 학습한 B1 점수를 넣으면 이슈 난이도가 누수된다. 테스트 점수는 학습 이슈 전체로 학습한 B1으로 낸다 |
| 4.4 | 참고용 상한: "같은 이슈의 다른 궤적 성공률" 오라클(제안서 AUROC 0.945)을 재현만 하고 비교 기준으로 표시 |

완료 기준: N별 B0·B1 AUROC/PR-AUC/이슈 내 AUROC 표. B1의 이슈 내 AUROC는 정의상 0.5여야 한다 (이슈 안에서는 점수가 같음) — 이 값이 하네스 검증 역할도 한다.

#### Phase 4 결과 (2026-10-31 완료)

상세 계획은 [PLAN_4.md](PLAN_4.md). 실행 `uv run --no-sync python code/03_baselines.py` (16코어 2분 24초: B1 그리드·OOF 54초, 운영점 선택 36초, 부트스트랩 82개 × 1,000회 50초). 완료 기준 C1~C8 **모두 충족**(스크립트가 assert, `results/phase4/summary.json`의 `criteria`). C9: `uv run --no-sync pytest` **95개 통과**, 다시 실행해도 `results/phase4/`와 `preds/*.parquet`이 바이트 단위로 같다 [확인].

| # | 기준 | 결과 |
| --- | --- | --- |
| C1 | 예측 파일 6개가 모든 분할·N에서 진행 중 궤적과 1:1 | ✓ (`preds.write`가 저장 전에 검사) |
| C2 | B1 이슈 내 AUROC = 0.5 | ✓ 최대 편차 0 (train_oof·test_same·test_cross × N 7개) |
| C3 | OOF 자기 이슈 점수 0개, 정리 후 이슈마다 본문 1개 | ✓ 0개, 3,591개 (꼬리 제거 13개) |
| C4 | 고정 상한 Phase 3 값 재현 | ✓ 전체 K=30/35/40/50 → 7.0/4.9/3.7/2.2%, 29.4/24.6/20.8/15.8%, K=35 토큰 42.2%. test_same K=35 4.47%/21.81% |
| C5 | train에서 고른 K\* = 34 | ✓ (train 손실 4.71%, 절약 25.06%, fold 4.16~5.27%) |
| C6 | 오라클 전체 LOO AUROC 0.945 | ✓ 0.9448 (궤적 2개 이상 이슈 3,581개, 80,026궤적) |
| C7 | 판별 표 + 테스트 구간 | ✓ `discrimination.csv`, `discrimination_ci.csv` |
| C8 | 단독·결합 운영점 + K\* 대비 차이 구간 | ✓ `operating_points.csv` |

확정한 규칙
- **B1 입력 정리**: 이슈 13개(모두 train)의 한쪽 본문 끝에 붙은 "고칠 파일 목록"(정답 패치 정보)을 지운다. 꼬리 앞에는 `\n\n\n`(4개는 `\r\n\n\n\n`)이 있어서, 꼬리 시작부터 지우고 끝 공백을 `rstrip()`한다. 13쌍 모두 꼬리 없는 본문과 정확히 같아진다 (`test_real_issue_text_unique_after_strip`)
- **B1 학습 단위**: 이슈마다 (실패, 성공) 2행에 궤적 수 가중치 = 궤적 단위 LR (합성 데이터에서 계수 1e-5 안에서 같음). `issue` 가중은 이슈마다 무게를 같게 하고 가중치 합을 궤적 수로 맞춰 C의 의미를 `traj`와 같게 했다. 벡터라이저(어휘·IDF)는 fold 안에서 다시 학습한다
- **B1 설정**: 20개 그리드(분석 단위 2 × C 5 × 가중 2)를 train OOF AUROC(N=0)로 골랐다 → **char_wb (3,5)-gram, C=0.1, `issue` 가중**, OOF AUROC 0.652 (word 최고 0.639). 고른 설정을 모든 N의 `b1_n`에 그대로 쓴다. 전체 표 `b1_grid.csv`
- **테스트 점수 = 5개 fold 모델 평균** (P4-D2). 전체 재학습과 비교하면 테스트 AUROC 차이 0.004 이하, 중앙값 차이 0.0012 이하이고, 재학습 쪽 q95가 0.002~0.006 높다 (더 자신 있게 냄, `b1_scale.csv`)
- **누수 크기** (4.3이 막는 것): 같은 설정을 학습 이슈 전체로 학습해 train에 채점하면 N=0 AUROC 0.907, OOF는 0.652. in-fold 점수를 M1에 넣으면 이 차이만큼 이슈 난이도가 샌다
- **오라클**: LOO(같은 집합·같은 이슈 다른 궤적 성공률)는 이슈 내 AUROC가 정의상 0이라 `discrimination.csv`에서 비웠다. `ref_oracle_xmodel`(test_cross ↔ test_same 성공률)은 이슈 안에서 상수(0.5). 점수를 못 받는 궤적은 train 이슈 10개(궤적 1개짜리)뿐이고 학습 성공률로 채웠다
- **운영점** (3.5 적용): train OOF 곡선 하나(62,460궤적)에서 손실 ≤5%·절약 스텝 최대로 고르고 테스트에 그대로 적용. 단독 = 상한 없음, 결합 = K ∈ N+1..60. 비교 기준은 같은 부트스트랩 표본의 K\*=34 (짝지은 95% 구간). 분할마다 같은 시드라 모델 사이에서도 같은 재표본이다

판별 (테스트, 이슈 단위 부트스트랩 95% 구간. B0 이슈 내 AUROC는 쌍 가중)

| 모델 | test_same N=0 | N=10 | 이슈 내 N=10 | test_cross N=0 | N=10 | 이슈 내 N=10 |
| --- | --- | --- | --- | --- | --- | --- |
| B0 에러 수 | — | 0.534 [0.502, 0.566] | 0.545 [0.519, 0.571] | — | 0.532 [0.485, 0.582] | 0.609 [0.550, 0.667] |
| B0 편집 실패 수 | — | 0.542 [0.517, 0.567] | 0.532 [0.510, 0.555] | — | 0.553 [0.518, 0.586] | 0.571 [0.529, 0.608] |
| B1 (`b1_n`) | 0.632 [0.585, 0.677] | 0.621 [0.569, 0.668] | 0.5 | 0.678 [0.614, 0.732] | 0.646 [0.585, 0.708] | 0.5 |
| 오라클 LOO (참고) | 0.941 | 0.930 | — | 0.914 | 0.905 | — |
| 오라클 다른 모델 (참고) | 0.843 | 0.830 | 0.5 | 0.917 | 0.907 | 0.5 |

- B1 PR-AUC: test_same N=0 0.260(기준값 0.187), test_cross 0.276(0.176). train OOF는 0.652 / 0.249
- B0 규칙은 전체 AUROC가 0.5 근처(0.48~0.56)지만 **이슈 내 AUROC는 0.51~0.61로 대부분 0.5 위**다. 이슈 난이도와 섞이면 묻히는 약한 궤적 신호가 있다
- 오라클과 B1 사이(0.63 vs 0.94)가 이슈 난이도 중 본문 TF-IDF로 못 잡는 부분이다

운영점 (N=10, train에서 고른 τ·K 그대로 적용. 절약 스텝 차이는 같은 표본 K\*=34 대비)

| 정책 | τ, K | train 손실 / 절약 | test_same 손실 / 절약 | test_same 차이 [95%] | test_cross 손실 / 절약 | test_cross 차이 [95%] |
| --- | --- | --- | --- | --- | --- | --- |
| K\*=34 | —, 34 | 4.71% / 25.06% | 4.95% / 22.64% | — | 12.91% / 33.61% | — |
| K=35 (제안서) | —, 35 | 4.32% / 24.19% | 4.47% / 21.81% | −0.83%p [−0.85, −0.79] | 12.36% / 32.53% | −1.09%p |
| B0 에러 단독 | 5, — | 2.59% / 9.06% | 2.22% / 8.02% | −14.6%p | 2.71% / 11.13% | −22.5%p |
| B0 에러 결합 | 8, 34 | 4.80% / 25.76% | 4.95% / 23.14% | **+0.50%p [+0.32, +0.70]** | 12.91% / 34.49% | **+0.88%p [+0.51, +1.24]** |
| B1 단독 | 0.936, — | 5.00% / 14.72% | 2.43% / 4.71% | −17.9%p | 2.17% / 7.10% | −26.5%p |
| B1 결합 | 0.949, 35 | 4.96% / 26.45% | 4.91% / 21.92% | −0.72%p [−0.84, −0.53] | 12.69% / 32.97% | −0.65%p [−1.10, +0.20] |

- **어떤 단독 정책도 K\*를 넘지 못한다** (N=3~20 모두, 차이 구간 상한 < 0). 이 기준선들로는 "N에서 한 번 보고 끊기"만으로 고정 상한의 절약을 따라가지 못한다
- **B0 결합 정책은 K\* 대비 절약 차이 구간이 0 위**다 (에러 수는 N=3~20 전부, 편집 실패 수는 N=5~20. test_same +0.2~+2.6%p, N=20 에러 결합이 최대 +2.6%p [+2.1, +3.0]). 대신 test_same 손실도 0~0.4%p 늘어서(5.0~5.3%, 구간 상한 6.0~6.5%) "같은 손실에서 더 아낀다"는 아직 아니다
- **B1의 τ가 테스트로 옮겨지지 않는다.** train OOF에서 진행 중 궤적의 18.8%를 끊던 τ(N=10)가 test_same에서는 7.8%만 끊는다. 테스트 이슈가 더 쉽고(성공률 18.7% vs 16.3%) 점수 분포가 낮아서다 (`b1_scale.csv`: OOF 중앙값 0.917, 테스트 0.908). 결합 정책도 τ가 거의 아무것도 끊지 않아 K=35 단독과 비슷해지고 K\*보다 0.7%p 적게 아낀다. M1부터는 테스트 정지율을 train 정지율과 같이 본다
- test_cross는 K\*만으로 손실 12.9%라 어떤 정책도 5%를 지키지 못한다. test_cross 운영점은 "train에서 고른 값을 그대로 쓴 결과"로만 읽는다
- 고른 운영점의 fold별 train 손실은 결합 정책에서 3.8~6.1%, B1 단독에서 2.4~9.2%까지 흔들린다 (`fold_lost_success_*`). 5% 경계 판정은 구간과 같이 본다

산출물
- `data/processed/preds/{b0_err, b0_editfail, b1_static, b1_n, ref_oracle_loo, ref_oracle_xmodel}.parquet` (git 밖). 메타데이터에 설정·N 목록·분할·git 커밋·생성 시각. 내용이 같으면 다시 쓰지 않는다
- `results/phase4/`: `discrimination.csv`, `discrimination_ci.csv`, `fixed_cap.csv`(분할·전체 × K 1..80), `operating_points.csv`, `operating_points_sensitivity.csv`(max_loss 2~10% × 목적 함수 2), `cost_curves/`(113개, 행 ≤200), `b1_grid.csv`, `b1_scale.csv`, `summary.json`
- Phase 5로 넘기는 것: `preds.load_feature("b1_n")`을 `(traj_id, N)`으로 join (`ref_*`는 거부됨), `oof.oof_predict(train, fit, predict, test=)`, `evaluate.*`, `metrics.select_operating_point / select_from_curves / select_cap / fixed_cap_curve`, 앞 N스텝 누수 테스트 틀 `test_error_count_ignores_steps_after_n`

PLAN_4.md와 다르게 한 것
- 3.5는 팀원이 먼저 `metrics.select_operating_point(..., caps=)`로 main에 넣어서(#4) 계획의 `select_combined`는 따로 만들지 않았다. 결합 정책 = `caps=range(N+1, 61)`. 같은 곡선에서 max_loss·목적 함수를 바꿔 고르려고 선택 단계만 `select_from_curves`로 떼어냈다 (시그니처 변경 없음)
- `summary.json`에 실행 시간·커밋을 넣지 않았다 (재실행 시 `git diff results/` 없음 기준 C9와 충돌). 커밋·생성 시각은 예측 파일 메타데이터에, 실행 시간은 표준 출력에만 남는다
- `--models` 옵션은 만들지 않았다 (전체 실행이 2.5분이라 필요 없음). 대신 `--jobs`(B1 그리드·부트스트랩 병렬 프로세스 수)를 추가했다
- LR은 BLAS 스레드를 1개로 묶는다 (`threadpoolctl`). 이 크기에서 다중 스레드 BLAS가 50배 느렸다 (단어 TF-IDF 1회 학습 2.4초 → 0.04초)
- 부트스트랩은 반복마다 맥락 길이를 경계 {N, K, K\*} 구간 합으로 줄여 넣는다 (`metrics.compress_costs`, cost_curve 결과가 같음을 테스트)
- 운영점 표의 `stop_rate`: 모델 행은 "시점 N 진행 중 궤적 중 예측기가 끊은 비율", `b0_cap` 행은 "전체 궤적 중 상한에 걸린 비율"


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

[가정] 학기 1주차가 8/31에 시작했다면 10/29는 9주차다. 10/31 기준 Phase 0~4가 끝났다.

| 주차 | 계획 |
| --- | --- |
| 9주차 (지금) | ~~Phase 0~3~~ 완료 (10/29, 3.5 포함). ~~Phase 4 (B0·B1)~~ 완료 (10/31) |
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
uv run --no-sync python code/03_baselines.py         # Phase 4 → preds/, results/phase4/ (16코어 약 2.5분. --n-boot 100, --skip-bootstrap은 개발용)
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
| D3 | 이슈 내 AUROC 집계 방식 | 이슈 안 성공×실패 쌍을 모아 계산한 값(쌍 가중)을 주 지표로, 이슈 평균을 보조로. **하네스는 둘 다 낸다 (Phase 3). 주 지표는 팀 확정 필요** |
| D4 | 토큰 근사 방법 | 문자 수 기반. 필요하면 ModernBERT 토크나이저로 샘플 보정. **문자 수 기반으로 구현됨 (Phase 3.4), 보정은 안 함** |
| D5 | pytest 도입 (`pyproject.toml`·`uv.lock` 변경) | **확정: 도입함** |
| D6 | `results/`를 git에 올릴지 | **확정: 작은 CSV·JSON을 올린다** (`results/phase1/`, `results/phase2/`) |
| D7 | M3 범위 | N=5, 10, 20만 |
| D8 | 비용 곡선이 단일 시점 정책만 볼지, 여러 시점(N=5, 10, 20에서 차례로 검사) 정책도 볼지 | 단일 시점을 주 결과로, 다중 시점은 시간이 남으면 |
| P4-D1 | 모델 비교의 고정 상한 기준선 | **train에서 고른 K\*=34를 주 기준으로 구현**, 제안서 K=35는 같은 표의 참고 행(`cap_proposal`). 팀 확정 필요 |
| P4-D2 | OOF 모델의 테스트 점수 | **5개 fold 모델 평균으로 구현.** 전체 재학습과의 차이는 작다 (`b1_scale.csv`: test_same N=0 AUROC 0.6318 vs 0.6316) |
| P4-D3 | B1 대표 행 | **`b1_n`을 주 결과로**, `b1_static` 보조. 두 값은 N≤8에서 거의 같다 |
| P4-D4 | 운영점 목적 함수 | **절약 스텝으로 구현.** 토큰 기준 선택은 `operating_points_sensitivity.csv`(objective=saved_tokens)에만 |
| P4-D5 | B1 가중 방식 | 그리드 OOF로 선택됨: `issue` 가중 (char_wb (3,5)-gram, C=0.1) |

## 12. 리스크

| 리스크 | 영향 | 대응 |
| --- | --- | --- |
| 제안서 수치가 로컬 데이터에서 재현되지 않음 (HF Hub 로드 vs 고정 revision, 스텝 정의 차이) | 모든 비교 기준이 흔들림 | Phase 0.2·1에서 먼저 대조하고 차이를 문서화 |
| B1 점수 누수 (in-fold 점수를 M1에 넣음) | M1 성능 과대평가 | Phase 4.3 out-of-fold 강제, 테스트로 확인. 실측 누수 크기: train N=0 in-fold AUROC 0.907 vs OOF 0.652 |
| 운영점 τ를 테스트 셋에서 고름 | 절약률 과대평가 | Phase 3.5: τ·K는 학습 OOF에서만 선택 |
| `exit_context`(맥락 초과 종료) 궤적이 길이·라벨과 얽힘 | "길다 = 실패"만 학습 | 시점별 진행 중 궤적만 비교 + 고정 상한과 대조 (제안서 대응 유지), `exit_status`는 분석에만 |
| 테스트 이슈 578개가 무작위 표본이 아님 | 일반화 해석 왜곡 | Phase 2.4 분포 비교를 결과에 첨부 |
| N이 클수록 이슈 내 AUROC 계산 가능 이슈가 줄어 분산이 큼 | 큰 N 결과 불안정 | 이슈 수와 부트스트랩 구간을 같이 보고 |
| 일정 지연 (현재 약 1주 [가정]) | M3·외부 테스트 누락 | 최소 범위(B0·B1·M1·M2)를 먼저 끝냄. 7절 기준 |
| 2.1GB openhands 파일 메모리 사용 | 선택 과제에서만 | row group 단위로 읽기 |
| GPU 환경 (RTX 5070 12GB, CUDA 13.0 + 드라이버 580 이상) [가정: 제안서 §7의 머신이 학습 머신] | M2·M3 지연 | M2는 CPU로도 가능하도록 작게 유지 |
