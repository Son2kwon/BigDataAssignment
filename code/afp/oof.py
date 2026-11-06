"""학습 이슈 fold별 out-of-fold(OOF) 예측. B1(Phase 4)과 M1·M2·M3(Phase 5~7)이 같은 함수로 학습한다.

- train fold k 행의 점수 = fold k를 뺀 나머지 fold로 학습한 모델의 점수. 자기 이슈를 본 모델의
  점수를 받는 학습 행은 없다 (이슈는 한 fold에만 속하므로). 실행할 때마다 이 조건을 다시 센다
- 테스트 행의 점수 = fold 모델들 점수의 평균 (test_mode="fold_mean", PLAN_4 P4-D2 (a)).
  운영점 τ를 OOF 점수 위에서 고르고 테스트에 그대로 적용하므로, 학습 데이터 양이 OOF 모델과 같은
  모델들의 점수를 쓴다. test_mode="refit"은 학습 행 전체로 다시 학습한 모델 하나의 점수
- fit은 받은 행만으로 모든 것(벡터라이저 어휘·IDF 포함)을 새로 학습해야 한다. fold를 나누기 전에
  전체 train으로 만든 전처리는 검증 fold 정보를 섞는다

행 단위는 호출하는 쪽이 정한다 (B1은 이슈 행, M1은 (궤적, N) 행). 필요한 컬럼은 instance_id, fold.
"""
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from afp import split

TEST_MODES = ("fold_mean", "refit")


@dataclass
class OOFResult:
    train: pl.DataFrame  # 입력 train 행 (같은 순서) + p_fail (OOF)
    test: pl.DataFrame | None  # 입력 test 행 (같은 순서) + p_fail
    fold_issues: dict[int, int] = field(default_factory=dict)  # 검증 fold k → fold k 모델 학습에 쓴 이슈 수
    seen_issue_scores: int = 0  # 자기 이슈를 학습에 쓴 모델의 점수를 받은 train 행 수 (항상 0이어야 함)
    # test_mode="fold_mean"일 때 fold k 모델 하나가 test 행에 준 점수 (n_folds × test 행 수). test 점수 = 행 평균.
    # fold k의 OOF 점수와 같은 모델의 점수라, 평균 때문에 점수 분포가 달라지는지 확인하는 데 쓴다
    test_fold_scores: np.ndarray | None = None
    models: list | None = None  # keep_models=True일 때 fold k 모델 (SHAP·다른 입력으로 다시 채점할 때)


def check_folds(train: pl.DataFrame, n_folds: int) -> None:
    for c in ("instance_id", "fold"):
        if c not in train.columns:
            raise ValueError(f"train에 {c} 컬럼이 없음")
    folds = sorted(train["fold"].unique().to_list())
    if folds != list(range(n_folds)):
        raise ValueError(f"train fold는 0..{n_folds - 1}이어야 한다: {folds}")
    crossing = train.group_by("instance_id").agg(pl.col("fold").n_unique()).filter(pl.col("fold") > 1).height
    if crossing:
        raise ValueError(f"여러 fold에 걸친 이슈 {crossing}개 (fold는 이슈 단위여야 한다)")


def _scores(predict, model, rows: pl.DataFrame) -> np.ndarray:
    p = np.asarray(predict(model, rows), dtype=float)
    if p.shape != (rows.height,):
        raise ValueError(f"predict 출력 크기 {p.shape} != 행 수 {rows.height}")
    if not np.isfinite(p).all():
        raise ValueError("predict 출력에 NaN/inf가 있음")
    return p


def oof_predict(train: pl.DataFrame, fit, predict, test: pl.DataFrame | None = None,
                test_mode: str = "fold_mean", n_folds: int = split.N_FOLDS, test_for_fold=None,
                keep_models: bool = False) -> OOFResult:
    """fit(rows) -> model, predict(model, rows) -> p_fail 배열 (행 순서대로).

    train: 학습 행 (instance_id, fold 0..n_folds-1). test: 채점만 할 행 (instance_id 필요, train과 이슈가 겹치면 멈춤).
    test_for_fold(k, test) -> fold k 모델에 넣을 test 행 (같은 행·순서, 입력 컬럼만 다를 수 있음). 다른 모델 점수를
    입력으로 쓰는 모델이 fold k 모델에 그 모델의 fold k test 점수를 넣을 때 쓴다 (P4-D7 ①, test_mode="fold_mean"만).
    keep_models면 fold 모델들을 result.models에 남긴다.
    """
    if test_mode not in TEST_MODES:
        raise ValueError(f"test_mode는 {TEST_MODES} 중 하나: {test_mode!r}")
    check_folds(train, n_folds)
    if test_for_fold is not None and (test is None or test_mode != "fold_mean"):
        raise ValueError("test_for_fold는 test가 있고 test_mode='fold_mean'일 때만 쓴다")
    if test is not None:
        overlap = test.join(train.select("instance_id").unique(), on="instance_id", how="semi").height
        if overlap:
            raise ValueError(f"test 행 {overlap}개의 이슈가 train에도 있음")

    rows = train.with_row_index("_row")
    oof = np.full(train.height, np.nan)
    fold_scores = np.full((n_folds, test.height), np.nan) if test is not None and test_mode == "fold_mean" else None
    result = OOFResult(train=train, test=test, models=[] if keep_models else None)
    for k in range(n_folds):
        fit_rows = rows.filter(pl.col("fold") != k)
        held = rows.filter(pl.col("fold") == k)
        fit_issues = fit_rows["instance_id"].unique()
        result.seen_issue_scores += held.filter(pl.col("instance_id").is_in(fit_issues.implode())).height
        result.fold_issues[k] = fit_issues.len()
        model = fit(fit_rows.drop("_row"))
        oof[held["_row"].to_numpy()] = _scores(predict, model, held.drop("_row"))
        if keep_models:
            result.models.append(model)
        if fold_scores is not None:
            rows_k = test if test_for_fold is None else test_for_fold(k, test)
            if rows_k.height != test.height:
                raise ValueError(f"test_for_fold({k}) 행 수 {rows_k.height} != test {test.height}")
            fold_scores[k] = _scores(predict, model, rows_k)
    if result.seen_issue_scores:
        raise AssertionError(f"자기 이슈를 본 모델의 점수를 받은 train 행 {result.seen_issue_scores}개")
    if np.isnan(oof).any():
        raise AssertionError("OOF 점수를 받지 못한 train 행이 있음")
    result.train = train.with_columns(p_fail=pl.Series(oof))
    if test is not None:
        p = fold_scores.mean(axis=0) if test_mode == "fold_mean" else _scores(predict, fit(train), test)
        result.test = test.with_columns(p_fail=pl.Series(p))
        result.test_fold_scores = fold_scores
    return result
