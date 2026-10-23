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
bash data/download.sh                  # 필수: nebius/SWE-agent-trajectories (parquet 12개, 1.1GB)
bash data/download.sh --with-swebench  # 선택: SWE-bench/experiments 저장소도 받음 (외부 테스트용, 약 150MB)
```

- macOS와 Linux는 터미널에서 실행한다. Windows는 Git Bash나 WSL에서 실행한다.
- 필요한 도구는 `curl` 또는 `wget`이다. `--with-swebench`를 쓰면 `git`도 필요하다.
- 데이터셋 버전(커밋 `68195a1`)이 스크립트에 고정되어 있어 팀원 모두 같은 파일을 받는다. 받은 파일은 SHA-256으로 검증한다.
- 중간에 끊겨도 다시 실행하면 이어서 받는다. 이미 받은 파일은 건너뛴다.
- 요청 제한에 걸리면 `HF_TOKEN`을 설정하고 다시 실행한다.

받은 뒤 구조:

```
data/
├── download.sh
├── swe-agent-trajectories/   train-000XX-of-00012.parquet
└── swebench-experiments/     (--with-swebench를 쓴 경우)
```

