"""예측 파일: data/processed/preds/<model_id>.parquet (git 밖).

모든 모델(B0·B1·M1·M2·M3, 참고용 오라클)의 예측을 같은 형식으로 저장한다. Phase 5~7이 그대로 쓴다.
- 컬럼은 하네스 형식 그대로 traj_id, N, p_fail만. 분할·fold는 splits.parquet와 join해서 얻는다
- 한 파일에 train(OOF) + test_same + test_cross 예측이 같이 있다. 시점 N마다 그 시점 진행 중
  궤적 전부에 정확히 1개 (metrics.align이 검사하는 조건을 저장 전에 validate가 분할마다 미리 검사)
- parquet 메타데이터 afp.preds: model_id, config, oof_scheme, n_grid, splits, git 커밋, 버전, 생성 시각
- model_id: 기준선 b0_*, b1_*, 모델 m1_*·m2_*·m3_*, 참고용 ref_*.
  ref_* 예측은 정답 라벨로 만든 값이라 다른 모델의 입력으로 쓰지 않는다 (load_feature가 거부)
"""
import datetime
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import sklearn

from afp import io, metrics

PREDS_DIR = io.PROCESSED_DIR / "preds"
SCHEMA = {"traj_id": pl.String, "N": pl.Int16, "p_fail": pl.Float64}
PREFIXES = ("b0_", "b1_", "m1_", "m2_", "m3_", "ref_")
REFERENCE_PREFIX = "ref_"
META_KEY = b"afp.preds"
MODEL_ID = re.compile(r"^[a-z0-9_]+$")


def path(model_id: str) -> Path:
    if not MODEL_ID.match(model_id) or not model_id.startswith(PREFIXES):
        raise ValueError(f"model_id는 {PREFIXES} 중 하나로 시작하는 소문자·숫자·밑줄이어야 한다: {model_id!r}")
    return PREDS_DIR / f"{model_id}.parquet"


def normalize(preds: pl.DataFrame) -> pl.DataFrame:
    """저장 형식으로 맞춘다: 세 컬럼, 고정 dtype, (N, traj_id) 순 정렬 (바이트 단위 재현)."""
    missing = [c for c in SCHEMA if c not in preds.columns]
    if missing:
        raise ValueError(f"예측에 없는 컬럼: {missing}")
    return preds.select([pl.col(c).cast(t) for c, t in SCHEMA.items()]).sort("N", "traj_id")


def validate(preds: pl.DataFrame, sets: dict[str, pl.DataFrame], n_grid) -> dict:
    """sets(분할 이름 → 궤적)의 모든 분할·N에서 예측이 진행 중 궤적과 1:1인지 검사한다.

    p_fail의 null·NaN·inf, n_grid 밖의 N, 어느 분할에도 없는 궤적의 예측이 있으면 멈춘다.
    출력: {분할: {N: 예측 수}}.
    """
    n_grid = sorted(int(n) for n in n_grid)
    p = preds["p_fail"]
    if p.null_count() or not np.isfinite(p.to_numpy()).all():
        raise ValueError("p_fail에 null/NaN/inf가 있음")
    extra_n = sorted(set(preds["N"].unique().to_list()) - set(n_grid))
    if extra_n:
        raise ValueError(f"n_grid {n_grid} 밖의 N: {extra_n}")
    known = pl.concat([t.select("traj_id") for t in sets.values()])
    outside = preds.join(known, on="traj_id", how="anti").height
    if outside:
        raise ValueError(f"주어진 분할 {list(sets)} 어디에도 없는 궤적의 예측 {outside}개")
    counts = {}
    for name, trajs in sets.items():
        sp = preds.join(trajs.select("traj_id"), on="traj_id", how="semi")
        counts[name] = {}
        for n in n_grid:
            try:
                counts[name][n] = metrics.align(trajs, sp, n).height
            except ValueError as e:
                raise ValueError(f"{name}: {e}") from None
    return counts


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=io.ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def write(model_id: str, preds: pl.DataFrame, sets: dict[str, pl.DataFrame], n_grid, config: dict,
          oof_scheme: str, force: bool = False) -> dict:
    """검사 후 저장한다. 같은 내용(예측·config·oof_scheme·n_grid·분할)이 이미 있으면 다시 쓰지 않는다
    (파일 바이트가 그대로 남는다). 내용이 다르면 force 없이는 멈춘다 (02_split.py와 같은 방식).
    """
    out = path(model_id)
    p = normalize(preds)
    counts = validate(p, sets, n_grid)
    n_grid = sorted(int(n) for n in n_grid)
    key = {"config": config, "oof_scheme": oof_scheme, "n_grid": n_grid, "splits": list(sets)}
    status = {"model_id": model_id, "rows": p.height, "coverage": counts}
    if out.exists():
        old_meta = metadata(model_id)
        same = read(model_id).equals(p) and _canonical({k: old_meta.get(k) for k in key}) == _canonical(key)
        status["matches_existing_file"] = same
        if same:
            status["written"] = False
            return status
        if not force:
            raise SystemExit(f"{out}가 다시 계산한 예측과 다르다. 확인 후 --force로 덮어쓴다.")
    meta = {
        "model_id": model_id, **key,
        "git_commit": _git_commit(), "sklearn": sklearn.__version__, "polars": pl.__version__,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    table = p.to_arrow().replace_schema_metadata({META_KEY: _canonical(meta).encode()})
    tmp = out.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(out)
    status["written"] = True
    return status


def read(model_id: str) -> pl.DataFrame:
    """저장된 예측 (traj_id, N, p_fail). 없으면 03_baselines.py 실행을 안내하고 멈춘다."""
    p = path(model_id)
    if not p.exists():
        raise FileNotFoundError(f"{p}가 없음. 예측을 만드는 스크립트(Phase 4는 code/03_baselines.py)를 먼저 실행하세요.")
    return pl.read_parquet(p, columns=list(SCHEMA))


def metadata(model_id: str) -> dict:
    return json.loads(pq.read_schema(path(model_id)).metadata[META_KEY])


def load_feature(model_id: str) -> pl.DataFrame:
    """다른 모델의 입력 피처로 쓸 예측. (traj_id, N)으로 join한다. 참고용 ref_* 예측은 거부한다."""
    if model_id.startswith(REFERENCE_PREFIX):
        raise ValueError(f"{model_id}: 참고용(ref_*) 예측은 정답 라벨로 만든 값이라 피처로 쓸 수 없다")
    return read(model_id)
