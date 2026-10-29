"""경로 상수와 로컬 parquet 로드.

모든 데이터는 data/download.sh가 revision을 고정해 받은 로컬 parquet에서만 읽는다.
HF Hub에서 직접 load_dataset 하지 않는다.
"""
from pathlib import Path

import pyarrow.parquet as pq

SEED = 42

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "swe-agent-trajectories"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"

N_RAW_FILES = 12

TRAJECTORIES_PATH = PROCESSED_DIR / "trajectories.parquet"
STEPS_PATH = PROCESSED_DIR / "steps.parquet"

# 라벨 누수 컬럼. 분석 전용이고 피처 코드에서는 읽지 않는다.
LEAKY_COLUMNS = ("exit_status", "generated_patch", "eval_logs")


def raw_files() -> list[Path]:
    """swe-agent parquet 12개를 파일 번호 순서로 돌려준다. 하나라도 없으면 멈춘다."""
    files = [RAW_DIR / f"train-{i:05d}-of-{N_RAW_FILES:05d}.parquet" for i in range(N_RAW_FILES)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"{RAW_DIR}에 없는 파일: {missing}. bash data/download.sh swe-agent 를 실행하세요.")
    return files


def read_raw(file_idx: int, columns: list[str]):
    """원본 파일 하나를 pyarrow Table로 읽는다."""
    return pq.read_table(raw_files()[file_idx], columns=columns)


def traj_id(file_idx: int, row_idx: int) -> str:
    """궤적 ID. revision이 고정이라 파일·행 순서는 안정적이다."""
    return f"{file_idx:02d}-{row_idx:05d}"
