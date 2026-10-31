"""Phase 4 기준선: B0 규칙 점수, B1 이슈 본문 TF-IDF + 로지스틱 회귀, 참고용 오라클.

- B0 (학습 없음): p_fail = 앞 N스텝의 is_error(또는 edit_failed) 개수. 에러 판별 규칙은 Phase 1의 parse.ERROR.
  고정 상한 K는 시점 N 예측이 아니라 정책이라 metrics.fixed_cap_curve로 따로 평가한다
- B1: 이슈 본문만 보는 이슈 난이도 기준선. 같은 이슈의 궤적은 입력이 같으므로 이슈마다
  (실패 행, 성공 행) 두 행에 궤적 수를 가중치로 준 LR = 궤적 단위 LR (수학적으로 같은 목적 함수)
- 오라클 (ref_*): 같은 이슈 다른 궤적의 성공률. 정답 라벨로 만든 배포 불가능한 참고 천장이다
"""
from dataclasses import asdict, dataclass

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from afp import io, metrics

# 이슈 13개(모두 train)의 한쪽 본문 끝에 붙은 "고칠 파일 목록". 정답 패치에서 나온 정보라 B1 입력에서 지운다
HINT = "Here is the list of files that is recommended to be edited"
ISSUE_COLUMNS = ["traj_id", "instance_id", "issue_text"]
STEP_FLAG_COLUMNS = ["traj_id", "step_idx", "is_error", "edit_failed"]
LR_TOL = 1e-6
LR_MAX_ITER = 10_000


# ---- 4.1 B0: 앞 N스텝 개수 규칙 ----

def load_step_flags(max_n: int) -> pl.DataFrame:
    """앞 max_n스텝의 에러 플래그만 읽는다 (원문 컬럼을 읽지 않음)."""
    return io.scan_steps(STEP_FLAG_COLUMNS).filter(pl.col("step_idx") <= max_n).collect()


def count_scores(flags: pl.DataFrame, trajs: pl.DataFrame, n_grid, column: str) -> pl.DataFrame:
    """p_fail = 앞 N스텝(step_idx <= N)의 column 개수. 시점 N 진행 중 궤적(n_steps > N)마다 한 행.

    N 뒤의 스텝은 보지 않는다 (test_error_count_ignores_steps_after_n). 출력: traj_id, N, p_fail.
    """
    out = []
    for n in n_grid:
        alive = metrics.running(trajs.select("traj_id", "n_steps"), n).select("traj_id")
        cnt = (
            flags.filter(pl.col("step_idx") <= n)
            .group_by("traj_id")
            .agg(p_fail=pl.col(column).cast(pl.Float64).sum())
        )
        out.append(alive.join(cnt, on="traj_id", how="left").fill_null(0.0).with_columns(N=pl.lit(n)))
    return pl.concat(out).select("traj_id", pl.col("N").cast(pl.Int16), "p_fail").sort("N", "traj_id")


# ---- 4.2 B1: 이슈 본문 ----

def strip_hint(text: str) -> str:
    """끝에 붙은 "고칠 파일 목록" 꼬리를 지운다. 꼬리가 없는 본문은 그대로 둔다."""
    i = text.find(HINT)
    return text if i < 0 else text[:i].rstrip()  # 꼬리 앞 구분 줄바꿈(\r\n 포함)까지


def issue_table() -> pl.DataFrame:
    """이슈마다 정리한 본문 하나 (instance_id, issue_text). 정리 후에도 본문이 2개 이상인 이슈가 있으면 멈춘다."""
    t = io.load_trajectories(ISSUE_COLUMNS)
    t = t.with_columns(issue_text=pl.col("issue_text").map_elements(strip_hint, return_dtype=pl.String))
    texts = t.group_by("instance_id").agg(pl.col("issue_text").unique())
    multi = texts.filter(pl.col("issue_text").list.len() != 1)
    if multi.height:
        raise AssertionError(f"정리 후에도 본문이 1개가 아닌 이슈 {multi.height}개: {multi['instance_id'].head(5).to_list()}")
    return texts.select("instance_id", pl.col("issue_text").list.first()).sort("instance_id")


@dataclass(frozen=True)
class B1Config:
    analyzer: str = "word"  # word (1,2)-gram 또는 char_wb (3,5)-gram
    C: float = 1.0
    weighting: str = "traj"  # traj: 궤적 단위 라벨 / issue: 이슈별 성공률 타깃, 이슈마다 무게 같음
    min_df: int = 2
    max_features: int = 200_000
    sublinear_tf: bool = True

    def ngram_range(self) -> tuple[int, int]:
        return {"word": (1, 2), "char_wb": (3, 5)}[self.analyzer]

    def as_dict(self) -> dict:
        return {**asdict(self), "ngram_range": list(self.ngram_range())}


def issue_rows(trajs: pl.DataFrame, n: int, texts: pl.DataFrame) -> pl.DataFrame:
    """시점 N에 진행 중인 궤적이 있는 이슈마다 한 행: instance_id, fold, issue_text, n_succ, n_fail.

    n_succ·n_fail = 그 이슈에서 N에 아직 진행 중인 성공·실패 궤적 수 (N=0이면 스텝 1개 이상인 궤적 전부).
    trajs에 fold가 없으면 fold=-1 (테스트).
    """
    t = metrics.running(trajs, n)
    fold = pl.col("fold").first() if "fold" in trajs.columns else pl.lit(-1, dtype=pl.Int8)
    rows = t.group_by("instance_id").agg(
        fold=fold,
        n_succ=pl.col("target").sum().cast(pl.Int64),
        n_fail=(~pl.col("target")).sum().cast(pl.Int64),
    )
    rows = rows.join(texts, on="instance_id", how="left")
    if rows["issue_text"].null_count():
        raise ValueError(f"본문이 없는 이슈 {rows['issue_text'].null_count()}개")
    return rows.sort("instance_id").select("instance_id", "fold", "issue_text", "n_succ", "n_fail")


def training_rows(rows: pl.DataFrame, weighting: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """이슈 행 → LR 학습 행 (이슈 위치, y=실패 여부, 가중치). 가중치 0인 행은 뺀다.

    traj:  (실패, 성공) 가중치 = (n_fail, n_succ). 궤적을 복제한 LR과 같은 목적 함수
    issue: (n_fail/m, n_succ/m) × (평균 m). 이슈마다 무게가 같고, 가중치 합을 traj와 같게 맞춰
           C(정규화 강도)가 두 방식에서 같은 의미가 되게 한다
    """
    succ = rows["n_succ"].to_numpy().astype(float)
    fail = rows["n_fail"].to_numpy().astype(float)
    m = succ + fail
    if weighting == "traj":
        w_fail, w_succ = fail, succ
    elif weighting == "issue":
        scale = m.mean()
        w_fail, w_succ = fail / m * scale, succ / m * scale
    else:
        raise ValueError(f"weighting은 traj 또는 issue: {weighting!r}")
    idx = np.concatenate([np.arange(len(m)), np.arange(len(m))])
    y = np.concatenate([np.ones(len(m)), np.zeros(len(m))])
    w = np.concatenate([w_fail, w_succ])
    keep = w > 0
    return idx[keep], y[keep], w[keep]


def fit_lr(X, y, w, C: float) -> LogisticRegression:
    """가중 L2 LR. 결정적(lbfgs), 수렴 허용 오차를 작게 둔다.

    BLAS 스레드를 1개로 묶는다: 이 크기(수천 행 희소 행렬)에서는 다중 스레드 BLAS가 오히려 수십 배 느렸다
    (32코어, 단어 TF-IDF 한 번 학습 2.4초 → 0.04초).
    """
    with threadpool_limits(1):
        return LogisticRegression(C=C, solver="lbfgs", tol=LR_TOL, max_iter=LR_MAX_ITER).fit(X, y, sample_weight=w)


@dataclass
class B1Model:
    vectorizer: TfidfVectorizer
    lr: LogisticRegression

    def p_fail(self, texts) -> np.ndarray:
        return self.lr.predict_proba(self.vectorizer.transform(texts))[:, 1]


def fit_b1(rows: pl.DataFrame, config: B1Config) -> B1Model:
    """이슈 행(issue_rows 출력)만으로 벡터라이저(어휘·IDF)와 LR을 학습한다. 어휘의 문서 단위는 이슈 하나."""
    vec = TfidfVectorizer(
        analyzer=config.analyzer, ngram_range=config.ngram_range(), min_df=config.min_df,
        max_features=config.max_features, sublinear_tf=config.sublinear_tf, dtype=np.float64,
    )
    X = vec.fit_transform(rows["issue_text"].to_list())
    idx, y, w = training_rows(rows, config.weighting)
    return B1Model(vec, fit_lr(X[idx], y, w, config.C))


def b1_functions(config: B1Config):
    """oof.oof_predict에 넘길 (fit, predict)."""
    return (lambda rows: fit_b1(rows, config)), (lambda model, rows: model.p_fail(rows["issue_text"].to_list()))


def to_trajectories(trajs: pl.DataFrame, issue_scores: pl.DataFrame, n: int) -> pl.DataFrame:
    """이슈 점수(instance_id, p_fail)를 시점 N 진행 중 궤적에 붙인다. 출력: traj_id, N, p_fail."""
    alive = metrics.running(trajs, n).select("traj_id", "instance_id")
    out = alive.join(issue_scores.select("instance_id", "p_fail"), on="instance_id", how="left")
    if out["p_fail"].null_count():
        raise ValueError(f"N={n}: 이슈 점수가 없는 궤적 {out['p_fail'].null_count()}개")
    return out.select("traj_id", N=pl.lit(n, dtype=pl.Int16), p_fail="p_fail").sort("traj_id")


# ---- 4.4 오라클 (참고용, 정답 라벨 사용) ----

def oracle_loo(trajs: pl.DataFrame, n: int, fill: float) -> pl.DataFrame:
    """p_fail = 1 - 같은 집합·같은 이슈의 다른 궤적 전부(N 전에 끝난 것 포함)의 성공률.

    이슈 안에서 성공 궤적은 (k-1)/(m-1), 실패 궤적은 k/(m-1)을 받으므로 이슈 내 AUROC는 정의상 0이다.
    궤적이 하나뿐인 이슈(다른 궤적 없음)는 fill을 받는다. 출력: traj_id, N, p_fail.
    """
    t = trajs.with_columns(
        m=pl.len().over("instance_id"),
        k=pl.col("target").cast(pl.Int64).sum().over("instance_id"),
    )
    p_succ = (pl.col("k") - pl.col("target").cast(pl.Int64)) / (pl.col("m") - 1)
    t = t.with_columns(p_fail=pl.when(pl.col("m") > 1).then(1 - p_succ).otherwise(pl.lit(fill)))
    return metrics.running(t, n).select("traj_id", N=pl.lit(n, dtype=pl.Int16), p_fail="p_fail").sort("traj_id")


def oracle_xmodel(trajs: pl.DataFrame, other: pl.DataFrame, n: int, fill: float) -> pl.DataFrame:
    """p_fail = 1 - 다른 집합(other)에서 같은 이슈 궤적들의 성공률. 이슈 안에서 상수라 이슈 내 AUROC = 0.5.

    test_cross는 test_same(70B), test_same은 test_cross(8B·405B)를 other로 쓴다. other에 없는 이슈는 fill.
    """
    rate = other.group_by("instance_id").agg(p=1 - pl.col("target").cast(pl.Float64).mean())
    out = metrics.running(trajs, n).join(rate, on="instance_id", how="left")
    return out.select("traj_id", N=pl.lit(n, dtype=pl.Int16), p_fail=pl.col("p").fill_null(fill)).sort("traj_id")


def oracle_full_loo_auroc(trajs: pl.DataFrame) -> dict:
    """제안서 0.945 재현: 전체 데이터에서 궤적이 2개 이상인 이슈의 궤적 전부 (스텝 수와 무관), LOO 성공률 AUROC."""
    t = trajs.with_columns(m=pl.len().over("instance_id")).filter(pl.col("m") > 1)
    p = oracle_loo(t.with_columns(n_steps=pl.lit(1)), 0, fill=np.nan)  # 스텝 0개 궤적도 넣으려고 n_steps를 덮어씀
    a = t.select("traj_id", "target").join(p, on="traj_id")
    d = metrics.discrimination(a["target"], a["p_fail"])
    return {"n_traj": a.height, "n_issues": t["instance_id"].n_unique(), "auroc": d["auroc"]}
