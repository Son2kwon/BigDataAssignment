"""경로 상수와 로컬 parquet 로드.

모든 데이터는 data/download.sh가 revision을 고정해 받은 로컬 parquet에서만 읽는다.
HF Hub에서 직접 load_dataset 하지 않는다.
"""
import hashlib
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

SEED = 42

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "swe-agent-trajectories"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
DOWNLOAD_SCRIPT = DATA_DIR / "download.sh"

N_RAW_FILES = 12

TRAJECTORIES_PATH = PROCESSED_DIR / "trajectories.parquet"
STEPS_PATH = PROCESSED_DIR / "steps.parquet"
SPLITS_PATH = PROCESSED_DIR / "splits.parquet"

# 라벨 누수 컬럼. 결과가 나온 뒤에야 정해지는 값이라 피처 코드에서는 읽지 않는다.
# 읽어도 되는 곳은 사후 분석 전용 모듈(afp/leakage.py)과 Phase 1 파서(exit_status 보관)뿐이다.
LEAKY_COLUMNS = ("exit_status", "generated_patch", "eval_logs")


def check_safe_columns(columns) -> list[str]:
    """피처 쪽 입력 컬럼 목록을 검사한다. 컬럼을 명시하지 않거나 누수 컬럼이 있으면 멈춘다."""
    if columns is None or isinstance(columns, str):
        raise ValueError("읽을 컬럼을 리스트로 명시해야 한다 (전체 컬럼 로드 금지)")
    columns = list(columns)
    leaky = [c for c in columns if c in LEAKY_COLUMNS]
    if leaky:
        raise ValueError(f"누수 컬럼은 피처 입력으로 읽을 수 없다: {leaky}")
    return columns


def load_trajectories(columns) -> pl.DataFrame:
    """trajectories.parquet에서 명시한 안전 컬럼만 읽는다 (exit_status는 거부)."""
    return pl.read_parquet(TRAJECTORIES_PATH, columns=check_safe_columns(columns))


def scan_steps(columns) -> pl.LazyFrame:
    """steps.parquet에서 명시한 컬럼만 lazy로 읽는다."""
    return pl.scan_parquet(STEPS_PATH).select(check_safe_columns(columns))


def load_splits() -> pl.DataFrame:
    """splits.parquet (traj_id, split, fold). 없으면 02_split.py 실행을 안내하고 멈춘다."""
    if not SPLITS_PATH.exists():
        raise FileNotFoundError(f"{SPLITS_PATH}가 없음. uv run python code/02_split.py 를 실행하세요.")
    return pl.read_parquet(SPLITS_PATH, columns=["traj_id", "split", "fold"])


def raw_files() -> list[Path]:
    """swe-agent parquet 12개를 파일 번호 순서로 돌려준다. 하나라도 없으면 멈춘다."""
    files = [RAW_DIR / f"train-{i:05d}-of-{N_RAW_FILES:05d}.parquet" for i in range(N_RAW_FILES)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"{RAW_DIR}에 없는 파일: {missing}. bash data/download.sh swe-agent 를 실행하세요.")
    return files


def read_raw(file_idx: int, columns: list[str], *, allow_leaky: bool = False):
    """원본 파일 하나를 pyarrow Table로 읽는다.

    누수 컬럼은 allow_leaky=True일 때만 읽는다. 사후 누수 점검(afp/leakage.py)에서만 쓴다.
    """
    if not allow_leaky:
        check_safe_columns(columns)
    return pq.read_table(raw_files()[file_idx], columns=list(columns))


def expected_raw_checksums(script: Path = DOWNLOAD_SCRIPT) -> dict[str, tuple[int, str]]:
    """download.sh에 고정된 swe-agent 파일별 (크기, SHA-256). 키는 파일 이름."""
    out = {}
    for line in script.read_text().splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] == "swe-agent" and parts[1].endswith(".parquet"):
            out[Path(parts[1]).name] = (int(parts[2]), parts[3])
    if len(out) != N_RAW_FILES:
        raise ValueError(f"{script}에서 swe-agent 체크섬 {N_RAW_FILES}개를 찾지 못함: {len(out)}개")
    return out


def verify_raw_checksums() -> dict[str, str]:
    """로컬 원본 12개가 download.sh에 고정된 revision의 파일과 같은지 크기·SHA-256으로 확인한다."""
    expected = expected_raw_checksums()
    got = {}
    for path in raw_files():
        size, sha = expected[path.name]
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        if path.stat().st_size != size or h.hexdigest() != sha:
            raise ValueError(f"{path.name}이 download.sh에 고정된 파일과 다르다 (크기·SHA-256 불일치)")
        got[path.name] = h.hexdigest()
    return got


def traj_id(file_idx: int, row_idx: int) -> str:
    """궤적 ID. revision이 고정이라 파일·행 순서는 안정적이다."""
    return f"{file_idx:02d}-{row_idx:05d}"
