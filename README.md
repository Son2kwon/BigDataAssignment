# AI 코딩 에이전트 실패 조기 예측

코딩 에이전트(SWE-agent)가 GitHub 이슈를 푸는 궤적의 앞 N스텝만 보고 최종 성공 여부를 예측한다. 실패할 실행을 일찍 끊어 비용을 줄이는 것이 목표다. 자세한 내용은 [docs/proposal.md](docs/proposal.md)에 있다.

## 구조

```
code/      실험 코드
data/      데이터 (git에는 download.sh만 올라감)
docs/      제안서와 수정 내역
```

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

