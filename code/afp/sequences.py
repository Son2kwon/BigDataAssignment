"""M2 시퀀스 입력 (Phase 6.1). 궤적마다 스텝 1..L의 입력 행렬을 만든다 (L = min(n_steps - 1, MAX_N)).

M2(LSTM)는 시점 t의 은닉 상태로 "시점 t에 진행 중인 궤적"의 실패 확률을 낸다 (모델 하나로 모든 N, P6-D2).
시점 t에 진행 중 = n_steps > t 이므로, 예측이 필요한 위치는 t = 1..n_steps-1 뿐이다. 그래서 궤적의 마지막 스텝
(보통 submit)은 입력에 넣지 않는다. 시점 N의 예측은 위치 N의 출력이고 입력 스텝 1..N만 본다 (LSTM은 단방향).

스텝 t의 입력 (모두 스텝 1..t만 보고 정해진다, features.step_values와 같은 정의):
- command: 명령 종류 인덱스 (features.COMMAND_NAMES 28종, 임베딩). 패딩 = PAD_COMMAND
- STEP_FEATURES: 스텝 t의 값 (에러, 편집 거부·성공, log1p 관측 길이, 열린 파일 유무·새 파일, 반복, log1p 연속 길이,
  이슈 언급 파일 열람, log1p 위치)
- CUM_FEATURES: 시점 t의 M1 피처 (features.m1_features(N=t))에서 스텝 t 값과 겹치는 last_*·obs_len_last와 정적
  issue_mentions_file을 뺀 누적 집계 41개, log1p. 근거: Phase 5 SHAP 상위가 횟수 피처(n_edit_ok 등)였고,
  작은 데이터에서 LSTM이 정확한 횟수를 스스로 배우기 어렵다. 누적 집계를 넣으면 M2는 매 시점 M1과 같은 정보를
  가지므로, M1과의 차이는 "순서 정보 + 모든 N을 함께 학습"에서 나온다 (구성 요소 분석은 05_m2.py 절제 실험)
- static: issue_mentions_file (궤적마다 상수)

누수: 입력은 features.load_step_values(io.scan_steps로 컬럼 명시)와 features.m1_features뿐이다. 누수 컬럼과 정답 패치 정보를
읽지 않는다. B1 점수는 b1_matrix로 따로 붙인다 (P4-D7은 05_m2.py에서).
"""
import hashlib
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from afp import evaluate, features, io

MAX_N = max(evaluate.N_GRID)
COMMAND_INDEX = {c: i for i, c in enumerate(features.COMMAND_NAMES)}
PAD_COMMAND = len(features.COMMAND_NAMES)
STEP_FEATURES = ["is_error", "edit_failed", "edit_ok", "obs_len_log", "has_file", "new_file", "repeat_prev", "repeat_any",
                 "run_len_log", "opens_issue_file", "position_log"]
_DROP_FROM_CUM = {"issue_mentions_file", "obs_len_last", "last_is_error", "last_edit_failed", *features.LAST_CMD_FEATURES}
CUM_FEATURES = [f for f in features.FEATURES if f not in _DROP_FROM_CUM]
STATIC_FEATURES = ["issue_mentions_file"]
TRAJ_COLUMNS = ["traj_id", "instance_id", "target", "n_steps", "fold"]


@dataclass
class SequenceData:
    """궤적 행(traj_id 정렬) × 위치 t(0-기반 인덱스 t-1). lengths[i] = 입력 스텝 수 L_i (>= 1)."""
    traj_id: np.ndarray
    instance_id: np.ndarray
    fold: np.ndarray
    y: np.ndarray  # 실패 = 1 (모델 타깃)
    n_steps: np.ndarray
    lengths: np.ndarray
    command: np.ndarray  # int64 [n, T], 패딩 PAD_COMMAND
    step: np.ndarray  # float32 [n, T, len(STEP_FEATURES)], 패딩 0
    cum: np.ndarray  # float32 [n, T, len(CUM_FEATURES)], 패딩 0
    static: np.ndarray  # float32 [n, len(STATIC_FEATURES)]
    max_n: int = MAX_N
    fingerprint: str = field(default="")

    @property
    def n(self) -> int:
        return len(self.traj_id)

    def valid(self) -> np.ndarray:
        """[n, T] bool: 위치 t가 진행 중 (t <= L_i)."""
        return np.arange(1, self.max_n + 1)[None, :] <= self.lengths[:, None]

    def running(self, n: int) -> np.ndarray:
        """시점 N에 진행 중인 궤적 행 인덱스 (n_steps > N ⇔ L >= N, N <= max_n)."""
        if not 1 <= n <= self.max_n:
            raise ValueError(f"N={n}은 1..{self.max_n} 밖")
        return np.flatnonzero(self.lengths >= n)


def step_inputs(values: pl.DataFrame) -> pl.DataFrame:
    """features.step_values 출력 → 스텝 입력 (traj_id, step_idx, command, STEP_FEATURES). 스텝 t 값은 1..t만 본다."""
    v = values.sort("traj_id", "step_idx")
    has_file = pl.col("open_file").is_not_null()
    first_seen = pl.col("step_idx").cum_count().over("traj_id", "open_file") == 1  # 같은 파일이 처음 열린 스텝
    return v.select(
        "traj_id", "step_idx",
        command=pl.col("command_name").replace_strict(COMMAND_INDEX, return_dtype=pl.Int64),
        is_error=pl.col("is_error").cast(pl.Float32),
        edit_failed=pl.col("edit_failed").cast(pl.Float32),
        edit_ok=((pl.col("command_name") == "edit") & ~pl.col("edit_failed")).cast(pl.Float32),
        obs_len_log=pl.col("obs_len").cast(pl.Float64).log1p().cast(pl.Float32),
        has_file=has_file.cast(pl.Float32),
        new_file=(has_file & first_seen).cast(pl.Float32),
        repeat_prev=pl.col("repeat_prev").cast(pl.Float32),
        repeat_any=pl.col("repeat_any").cast(pl.Float32),
        run_len_log=pl.col("run_len").cast(pl.Float64).log1p().cast(pl.Float32),
        opens_issue_file=pl.col("opens_issue_file").cast(pl.Float32),
        position_log=pl.col("step_idx").cast(pl.Float64).log1p().cast(pl.Float32),
    )


def _fingerprint(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes() if a.dtype != object else "\x00".join(map(str, a)).encode())
    return h.hexdigest()


def build(trajs: pl.DataFrame, values: pl.DataFrame, mentions: pl.DataFrame | None, max_n: int = MAX_N) -> SequenceData:
    """trajs(TRAJ_COLUMNS) + 스텝 단위 값(features.step_values, 앞 max_n스텝 이상) → SequenceData.

    입력 스텝이 없는 궤적(n_steps <= 1, 어떤 시점에도 진행 중이 아님)은 뺀다. 궤적마다 스텝 1..L이 모두 있어야 한다.
    """
    missing = [c for c in TRAJ_COLUMNS if c not in trajs.columns]
    if missing:
        raise ValueError(f"trajs에 없는 컬럼: {missing}")
    t = (trajs.select(TRAJ_COLUMNS).with_columns(L=(pl.col("n_steps") - 1).clip(0, max_n).cast(pl.Int64))
         .filter(pl.col("L") >= 1).sort("traj_id").with_row_index("_i"))
    n, T = t.height, max_n
    s = (step_inputs(values).join(t.select("traj_id", "L", "_i"), on="traj_id", how="inner")
         .filter(pl.col("step_idx") <= pl.col("L")))
    got = s.group_by("_i").agg(pl.len(), pl.col("step_idx").max().alias("mx")).join(t.select("_i", "L"), on="_i", how="right")
    bad = got.filter((pl.col("len").fill_null(0) != pl.col("L")) | (pl.col("mx").fill_null(0) != pl.col("L"))).height
    if bad:
        raise ValueError(f"스텝 1..L이 모두 있지 않은 궤적 {bad}개")
    cum = features.m1_features(values, t.select("traj_id", "instance_id", "n_steps"), range(1, max_n + 1), mentions)
    cum = cum.join(t.select("traj_id", "_i"), on="traj_id", how="inner")
    if cum.height != s.height:
        raise ValueError(f"누적 피처 행 {cum.height} != 입력 스텝 행 {s.height}")

    ii, tt = s["_i"].to_numpy(), s["step_idx"].to_numpy().astype(np.int64) - 1
    command = np.full((n, T), PAD_COMMAND, dtype=np.int64)
    command[ii, tt] = s["command"].to_numpy()
    step = np.zeros((n, T, len(STEP_FEATURES)), dtype=np.float32)
    step[ii, tt] = s.select(STEP_FEATURES).to_numpy().astype(np.float32)
    ci, ct = cum["_i"].to_numpy(), cum["N"].to_numpy().astype(np.int64) - 1
    cum_arr = np.zeros((n, T, len(CUM_FEATURES)), dtype=np.float32)
    cum_arr[ci, ct] = np.log1p(cum.select(CUM_FEATURES).to_numpy()).astype(np.float32)
    first = cum.filter(pl.col("N") == 1).sort("_i")
    if first.height != n:
        raise AssertionError("시점 1 누적 피처가 모든 궤적에 있어야 한다")
    static = first.select(STATIC_FEATURES).to_numpy().astype(np.float32)
    data = SequenceData(
        traj_id=t["traj_id"].to_numpy(), instance_id=t["instance_id"].to_numpy(), fold=t["fold"].to_numpy().astype(np.int64),
        y=(~t["target"]).to_numpy().astype(np.float32), n_steps=t["n_steps"].to_numpy().astype(np.int64),
        lengths=t["L"].to_numpy(), command=command, step=step, cum=cum_arr, static=static, max_n=max_n)
    for name in ("step", "cum", "static"):
        if not np.isfinite(getattr(data, name)).all():
            raise ValueError(f"{name}에 NaN/inf가 있음")
    data.fingerprint = _fingerprint(data.traj_id, data.instance_id, data.fold, data.y, data.lengths, data.command, data.step,
                                    data.cum, data.static)
    return data


def load(trajs: pl.DataFrame, max_n: int = MAX_N) -> SequenceData:
    """평가 집합 궤적(TRAJ_COLUMNS) → SequenceData. 스텝 값은 steps.parquet, 이슈 본문은 trajectories.parquet에서 읽는다."""
    values = features.load_step_values(max_n, trajs)
    t = io.load_trajectories(["traj_id", "instance_id", "issue_text"]).join(trajs.select("traj_id"), on="traj_id", how="semi")
    return build(trajs, values, features.mentions_table(t), max_n)


def b1_matrix(data: SequenceData, b1: pl.DataFrame, n_grid=evaluate.N_GRID_ISSUE) -> np.ndarray:
    """B1 점수(traj_id, N, p_fail; N ∈ n_grid) → [n, T] 위치별 점수. 위치 t = 그리드에서 t 이하 가장 큰 N의 점수.

    시점 N 진행 중 궤적은 N 이하 모든 그리드 시점에도 진행 중이라 점수가 있다. 그리드 시점(평가 위치)에서는
    b1_n(N) 그대로다. 진행 중이 아닌 위치는 NaN. 진행 중 위치에 점수가 빠지면 멈춘다.
    """
    grid = sorted(int(x) for x in n_grid)
    if grid[0] != 0:
        raise ValueError("n_grid에 N=0이 있어야 한다 (t < 첫 그리드 시점의 위치)")
    idx = pl.DataFrame({"traj_id": data.traj_id, "_i": np.arange(data.n)})
    out = np.full((data.n, data.max_n), np.nan, dtype=np.float32)
    owner = np.array([max(g for g in grid if g <= t) for t in range(1, data.max_n + 1)])
    for g in grid:
        cols = np.flatnonzero(owner == g)
        if not len(cols):
            continue
        b = b1.filter(pl.col("N") == g).select("traj_id", "p_fail").join(idx, on="traj_id", how="inner")
        vals = np.full(data.n, np.nan, dtype=np.float32)
        vals[b["_i"].to_numpy()] = b["p_fail"].to_numpy()
        out[:, cols] = vals[:, None]
    out[~data.valid()] = np.nan
    if np.isnan(out[data.valid()]).any():
        raise ValueError("진행 중 위치에 B1 점수가 없다")
    return out


def preds_frame(data: SequenceData, probs: np.ndarray, n_grid=evaluate.N_GRID) -> pl.DataFrame:
    """[n, T] 확률 → 예측 파일 형식 (traj_id, N, p_fail): 시점 N마다 진행 중 궤적 전부."""
    parts = []
    for n in n_grid:
        r = data.running(n)
        parts.append(pl.DataFrame({"traj_id": data.traj_id[r], "N": np.full(len(r), n, dtype=np.int16),
                                   "p_fail": probs[r, n - 1].astype(np.float64)}))
    return pl.concat(parts)


ARRAY_FIELDS = ("traj_id", "instance_id", "fold", "y", "n_steps", "lengths", "command", "step", "cum", "static")


def save_arrays(data: SequenceData, directory) -> None:
    """작업 프로세스가 메모리 매핑으로 읽도록 필드마다 .npy로 저장한다 (git 밖 캐시). 문자열 배열은 고정 길이 유니코드로."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in ARRAY_FIELDS:
        a = getattr(data, name)
        np.save(directory / f"{name}.npy", a.astype(str) if a.dtype == object else a)
    (directory / "meta.txt").write_text(f"{data.fingerprint}\n{data.max_n}\n")


def load_arrays(directory, mmap: bool = True) -> SequenceData:
    fingerprint, max_n = (directory / "meta.txt").read_text().split()
    kw = {name: np.load(directory / f"{name}.npy", mmap_mode="r" if mmap else None) for name in ARRAY_FIELDS}
    for name in ("traj_id", "instance_id"):
        kw[name] = np.asarray(kw[name]).astype(object)
    return SequenceData(**kw, max_n=int(max_n), fingerprint=fingerprint)
