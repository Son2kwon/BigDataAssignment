"""예측 파일: data/processed/preds/<model_id>.parquet (git 밖).

모든 모델(B0·B1·M1·M2·M3, 참고용 오라클)의 예측을 같은 형식으로 저장한다. Phase 5~7이 그대로 쓴다.
- 컬럼은 하네스 형식 그대로 traj_id, N, p_fail만. 분할·fold는 splits.parquet와 join해서 얻는다
- 한 파일에 train(OOF) + test 예측이 같이 있다. 시점 N마다 그 시점 진행 중
  궤적 전부에 정확히 1개 (metrics.align이 검사하는 조건을 저장 전에 validate가 분할마다 미리 검사)
- parquet 메타데이터 afp.preds: model_id, config, oof_scheme, n_grid, splits, git 커밋, 버전, 생성 시각
- model_id: 기준선 b0_*, b1_*, 모델 m1_*·m2_*·m3_*, 참고용 ref_*.
  ref_* 예측은 정답 라벨로 만든 값이라 다른 모델의 입력으로 쓰지 않는다 (load_feature가 거부)

fold별 테스트 점수: data/processed/preds/<model_id>.test_folds.parquet (OOF로 학습하는 모델만, P4-D6·P4-D7)
- 컬럼 traj_id, N, fold, p_fail. test 궤적만, fold k 모델 하나가 준 점수. 시점 N·fold마다 진행 중 test 궤적과 1:1
- 주 예측 파일의 test 행(= 5개 fold 모델 평균, P4-D2)은 판별 지표용이고, 운영점 평가(P4-D6)와 다른 모델의
  입력(P4-D7)에는 이 파일을 쓴다. 주 파일 형식(traj_id, N, p_fail)을 바꾸지 않으려고 따로 둔다
- write_test_folds가 저장 전에 fold별 범위와 "fold 평균 = 주 파일 test 점수"를 검사한다
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
PREFIXES = ("b0_", "b1_", "m1_", "m2_", "m2x_", "m3_", "ref_")  # m2x_ = M2 보조 실험 (Phase 6.5)
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


# ---- fold별 테스트 점수 (P4-D6·P4-D7) ----

FOLD_SCHEMA = {"traj_id": pl.String, "N": pl.Int16, "fold": pl.Int8, "p_fail": pl.Float64}
FOLD_SUFFIX = ".test_folds.parquet"
FOLD_MEAN_TOL = 1e-12  # fold 평균과 주 파일 test 점수의 허용 차이 (합산 순서에 따른 부동소수 오차)


def fold_path(model_id: str) -> Path:
    """<model_id>.test_folds.parquet. model_id에는 '.'이 없어서 다른 모델의 예측 파일과 이름이 겹치지 않는다."""
    return path(model_id).with_name(model_id + FOLD_SUFFIX)


def normalize_folds(fold_preds: pl.DataFrame) -> pl.DataFrame:
    missing = [c for c in FOLD_SCHEMA if c not in fold_preds.columns]
    if missing:
        raise ValueError(f"fold별 예측에 없는 컬럼: {missing}")
    return fold_preds.select([pl.col(c).cast(t) for c, t in FOLD_SCHEMA.items()]).sort("N", "fold", "traj_id")


def validate_test_folds(fold_preds: pl.DataFrame, test: pl.DataFrame, n_grid, n_folds: int,
                        mean_preds: pl.DataFrame | None = None) -> dict:
    """fold 0..n_folds-1 각각이 test의 모든 N에서 진행 중 궤적과 1:1인지, (mean_preds가 있으면) fold 평균이
    mean_preds의 test 점수와 같은지 검사한다. 출력: {"folds": [...], "max_abs_mean_diff": 값 또는 None}."""
    folds = sorted(fold_preds["fold"].unique().to_list())
    if folds != list(range(n_folds)):
        raise ValueError(f"fold는 0..{n_folds - 1}이어야 한다: {folds}")
    for k in folds:
        try:
            validate(fold_preds.filter(pl.col("fold") == k).drop("fold"), {"test": test}, n_grid)
        except ValueError as e:
            raise ValueError(f"fold {k}: {e}") from None
    out = {"folds": folds, "max_abs_mean_diff": None}
    if mean_preds is not None:
        m = fold_preds.group_by("traj_id", "N").agg(pl.col("p_fail").mean().alias("fold_mean"))
        j = m.join(mean_preds.join(test.select("traj_id"), on="traj_id", how="semi").select("traj_id", pl.col("N").cast(pl.Int16), "p_fail"),
                   on=["traj_id", "N"], how="full", coalesce=True)
        if j["fold_mean"].null_count() or j["p_fail"].null_count():
            raise ValueError("fold별 점수와 주 파일 test 점수의 (traj_id, N)이 다르다")
        diff = float((j["fold_mean"] - j["p_fail"]).abs().max())
        if diff > FOLD_MEAN_TOL:
            raise ValueError(f"fold 평균과 주 파일 test 점수가 다르다 (최대 차이 {diff:.3g})")
        out["max_abs_mean_diff"] = diff
    return out


def write_test_folds(model_id: str, fold_preds: pl.DataFrame, test: pl.DataFrame, n_grid, n_folds: int,
                     force: bool = False) -> dict:
    """fold별 테스트 점수를 검사 후 저장한다. 주 예측 파일(write)을 먼저 저장해야 한다 (fold 평균을 대조).
    같은 내용이면 다시 쓰지 않고, 다르면 force 없이는 멈춘다 (write와 같은 방식)."""
    out = fold_path(model_id)
    p = normalize_folds(fold_preds)
    n_grid = sorted(int(n) for n in n_grid)
    check = validate_test_folds(p, test, n_grid, n_folds, mean_preds=read(model_id))
    key = {"model_id": model_id, "n_grid": n_grid, "n_folds": n_folds, "splits": ["test"]}
    status = {"model_id": model_id, "rows": p.height, **check}
    if out.exists():
        same = read_test_folds(model_id).equals(p) and _canonical({k: test_folds_metadata(model_id).get(k) for k in key}) == _canonical(key)
        status["matches_existing_file"] = same
        if same:
            status["written"] = False
            return status
        if not force:
            raise SystemExit(f"{out}가 다시 계산한 fold별 점수와 다르다. 확인 후 --force로 덮어쓴다.")
    meta = {**key, "git_commit": _git_commit(), "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    table = p.to_arrow().replace_schema_metadata({META_KEY: _canonical(meta).encode()})
    tmp = out.with_name(out.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(out)
    status["written"] = True
    return status


def read_test_folds(model_id: str) -> pl.DataFrame:
    """저장된 fold별 테스트 점수 (traj_id, N, fold, p_fail). OOF로 학습하지 않는 모델(B0 규칙 등)은 없다."""
    p = fold_path(model_id)
    if not p.exists():
        raise FileNotFoundError(f"{p}가 없음. fold별 테스트 점수는 OOF로 학습하는 모델만 있다 (Phase 4는 code/03_baselines.py).")
    return pl.read_parquet(p, columns=list(FOLD_SCHEMA))


def test_folds_metadata(model_id: str) -> dict:
    return json.loads(pq.read_schema(fold_path(model_id)).metadata[META_KEY])


# ---- 다른 모델의 입력 피처 (P4-D7) ----

def _feature_model(model_id: str) -> None:
    if model_id.startswith(REFERENCE_PREFIX):
        raise ValueError(f"{model_id}: 참고용(ref_*) 예측은 정답 라벨로 만든 값이라 피처로 쓸 수 없다")


def _split_ids(splits: pl.DataFrame | None, name: str) -> pl.DataFrame:
    s = io.load_splits() if splits is None else splits
    return s.filter(pl.col("split") == name).select("traj_id")


def load_feature(model_id: str, split: str, fold: int | None = None, splits: pl.DataFrame | None = None) -> pl.DataFrame:
    """다른 모델(M1 등)의 입력 피처로 쓸 점수 (traj_id, N, p_fail). (traj_id, N)으로 join한다.

    - split="train": OOF 점수 (fold는 주지 않는다. train 궤적은 자기 fold를 뺀 모델의 점수를 이미 받았다)
    - split="test":  fold=k를 꼭 준다 → 이 모델의 fold k 모델 하나가 준 점수 (P4-D7 ①: M1 fold k ← B1 fold k).
      fold 없이 test를 요청하면 멈춘다. 주 파일의 test 행(5개 fold 모델 평균, P4-D7 ③)은 이 함수로 꺼낼 수 없다
    ② 민감도(fold k에 대해 j≠k 4개)는 load_feature_other_folds.
    splits: traj_id, split (기본 io.load_splits()). 참고용 ref_* 예측은 거부한다.
    """
    _feature_model(model_id)
    if split == "train":
        if fold is not None:
            raise ValueError("train 피처는 OOF 점수 하나뿐이다 (fold를 주지 않는다)")
        return read(model_id).join(_split_ids(splits, "train"), on="traj_id", how="semi")
    if split == "test":
        if fold is None:
            raise ValueError("test 피처는 fold를 지정해야 한다 (P4-D7 ①: M1 fold k ← 입력 모델 fold k 점수). "
                             "fold 평균 test 점수(P4-D7 ③)는 입력으로 쓰지 않는다")
        f = read_test_folds(model_id)
        if fold not in set(f["fold"].unique().to_list()):
            raise ValueError(f"{model_id}: fold {fold}가 없음 (있는 fold: {sorted(f['fold'].unique().to_list())})")
        return f.filter(pl.col("fold") == fold).join(_split_ids(splits, "test"), on="traj_id", how="semi").drop("fold")
    raise ValueError(f"split은 train 또는 test: {split!r}")


def load_feature_other_folds(model_id: str, fold: int, splits: pl.DataFrame | None = None) -> pl.DataFrame:
    """P4-D7 ② 민감도: fold k를 뺀 j≠k 모델들의 test 점수 (traj_id, N, src_fold, p_fail).

    M1 fold k 모델을 src_fold마다 따로 채점하고 결과를 평균한다 (입력 점수를 평균하지 않는다).
    """
    _feature_model(model_id)
    f = read_test_folds(model_id)
    folds = sorted(f["fold"].unique().to_list())
    if fold not in folds:
        raise ValueError(f"{model_id}: fold {fold}가 없음 (있는 fold: {folds})")
    return (f.filter(pl.col("fold") != fold).join(_split_ids(splits, "test"), on="traj_id", how="semi")
            .rename({"fold": "src_fold"}).select("traj_id", "N", "src_fold", "p_fail"))
