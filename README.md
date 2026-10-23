# AI 코딩 에이전트 실패 조기 예측

코딩 에이전트(SWE-agent)가 GitHub 이슈를 푸는 궤적의 앞 N스텝만 보고 최종 성공 여부를 예측한다. 실패할 실행을 일찍 끊어 비용을 줄이는 것이 목표다. 자세한 내용은 [docs/proposal.md](docs/proposal.md)에 있다.

## 구조

```
code/      실험 코드
data/      데이터 (git에는 download.sh만 올라감)
docs/      제안서와 수정 내역
```

## 환경 설정

Python 3.13과 패키지는 [uv](https://docs.astral.sh/uv/)로 관리한다. 패키지 버전은 `uv.lock`에 고정되어 있어 팀원 모두 같은 버전을 받는다.

### 1. uv 설치

macOS / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

설치가 끝나면 터미널을 새로 열고 `uv --version`으로 확인한다. Homebrew(`brew install uv`)나 winget(`winget install --id=astral-sh.uv -e`)으로 설치해도 된다.

### 2. 패키지 설치

저장소 루트에서 실행한다. 명령은 세 OS 모두 같다.

```bash
uv sync               # 기본 패키지 (약 1GB)
uv sync --group dl    # 기본 + PyTorch, transformers (Windows/Linux 약 6GB)
```

- Python 3.13이 없으면 uv가 받아서 `.venv/`에 환경을 만든다. 시스템 Python은 건드리지 않는다.
- `dl` 그룹은 M2(LSTM)와 M3(Transformer)에만 필요하다. 그 전까지는 `uv sync`만 해도 된다.
- `dl`을 받은 뒤 `--group dl` 없이 `uv sync`를 실행하면 torch가 지워진다. 한 번 받았으면 계속 `--group dl`을 붙인다.

### 3. 실행

```bash
uv run python "code/test for K/fixed_cap_k.py"
```

`uv run`은 `.venv`의 Python으로 실행하므로 환경을 따로 켤 필요가 없다. 직접 켜려면 macOS/Linux는 `source .venv/bin/activate`, Windows는 `.venv\Scripts\activate`를 쓴다.

- VS Code: `Ctrl+Shift+P` → `Python: Select Interpreter`에서 `.venv`를 고른다. 노트북 커널도 같은 `.venv`를 고르면 된다.
- Jupyter Lab: `uv run --with jupyterlab jupyter lab`
- GPU 확인: `uv run python -c "import torch; print(torch.cuda.is_available())"`

### 패키지

| 패키지 | 용도 |
| --- | --- |
| `numpy`, `pandas` | 수치 계산, 표 처리 |
| `pyarrow` | parquet 읽기 |
| `polars` | 궤적 파싱, 피처 추출. 멀티코어로 돌아서 큰 parquet에서 pandas보다 빠르다 |
| `datasets` | Hugging Face 데이터셋 로드 |
| `scikit-learn` | TF-IDF, 로지스틱 회귀, GroupKFold, AUROC·PR-AUC (B1, M1) |
| `xgboost` | M1. Windows/Linux에서는 `device="cuda"`로 GPU 학습 |
| `shap` | 피처 중요도 |
| `matplotlib` | 그래프 |
| `tqdm` | 진행 표시줄 |
| `ipykernel` | VS Code 노트북 커널 |
| `torch` (dl) | M2(LSTM), M3 fine-tune |
| `transformers`, `accelerate` (dl) | M3 ModernBERT-base fine-tune |

### OS별 차이

- Windows/Linux는 PyTorch CUDA 13.0 빌드를 받는다. NVIDIA 드라이버 580 이상이 필요하다(`nvidia-smi`로 확인). GPU가 없거나 드라이버가 낮으면 CPU로 돈다.
- macOS는 Apple Silicon만 지원한다. PyTorch가 Intel Mac 빌드를 더 이상 내지 않는다. CUDA 대신 `device="mps"`로 GPU를 쓴다.
- macOS에서 `import xgboost`가 libomp 오류를 내면 `brew install libomp`를 실행한다.

### 패키지 추가

`pip install` 대신 `uv add 이름`을 쓴다. `dl` 그룹에 넣으려면 `uv add --group dl 이름`. `pyproject.toml`과 `uv.lock`이 같이 바뀌므로 둘 다 커밋한다. 다른 팀원은 pull 받은 뒤 `uv sync`를 다시 실행한다.

## 데이터 받기

```bash
bash data/download.sh                # 기본 3개 (swe-agent, openhands, terminalbench), 3.4GB
bash data/download.sh openhands      # 이름을 주면 그것만 받음
bash data/download.sh all            # swebench까지 전부
```

| 이름 | 데이터셋 | 규모 | 용도 |
| --- | --- | --- | --- |
| `swe-agent` | [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) | 궤적 80,036개, 성공률 16.7%, 1.1GB | 주 데이터. SWE-agent + Llama 3종 |
| `openhands` | [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) | 궤적 67,074개, 성공률 48%, 2.1GB | 다른 프레임워크(OpenHands)·모델(Qwen3-Coder-480B) 일반화 테스트. 최대 100턴에서 끊긴다 |
| `terminalbench` | [yoonholee/terminalbench-trajectories](https://huggingface.co/datasets/yoonholee/terminalbench-trajectories) | 시도 52,104개, 성공률 39.6%, 0.2GB | 도메인 밖 테스트. 에이전트 26종 × 모델 49종, 과제 89개. 토큰·비용 컬럼 있음 |
| `swebench` | [SWE-bench/experiments](https://github.com/SWE-bench/experiments) | 약 150MB (git) | 리더보드 제출 기록. 기본값에서 빠짐 |

- macOS와 Linux는 터미널에서 실행한다. Windows는 Git Bash나 WSL에서 실행한다.
- 필요한 도구는 `curl` 또는 `wget`이다. `swebench`는 `git`도 필요하다.
- 데이터셋 버전이 스크립트에 고정되어 있어 팀원 모두 같은 파일을 받는다. 받은 파일은 SHA-256으로 검증한다.
- 중간에 끊겨도 다시 실행하면 이어서 받는다. 이미 받은 파일은 건너뛴다.
- 요청 제한에 걸리면 `HF_TOKEN`을 설정하고 다시 실행한다.

받은 뒤 구조:

```
data/
├── download.sh
├── swe-agent-trajectories/              train-000XX-of-00012.parquet
├── swe-rebench-openhands-trajectories/  trajectories.parquet
├── terminalbench-trajectories/          train-0000X-of-00002.parquet
└── swebench-experiments/                (swebench를 받은 경우)
```

