"""모델 하나의 예측 → 분할 × N 지표 표, 비용 곡선, 운영점 표. Phase 4~7이 같은 코드로 평가한다.

- 평가 집합은 metrics.eval_set의 train, test (둘 다 70B). train 행은 OOF 점수 기준이라 표에 train_oof로 적는다
- 운영점은 train(OOF) 곡선에서만 고르고 테스트에는 고른 (τ, K)를 그대로 적용한다 (PLAN 3.5)
  - single:   "시점 N에서 p_fail >= τ면 종료" (상한 없음)
  - combined: "스텝 > K면 종료 또는 시점 N에서 p_fail >= τ면 종료", K ∈ N+1..COMBINED_MAX_K
- 고정 상한 기준선 K*는 train에서 metrics.select_cap으로 고른 값. 모델 운영점은 같은 부트스트랩 표본의
  K*와의 차이(짝지은 구간)로 비교한다
- 손실 기준 (회의 2026-11-05 §4.1): 고를 때는 train 손실 <= metrics.SELECT_MAX_LOSS(3.5%), 테스트 충족 판정은
  손실 95% 구간 상한 <= metrics.JUDGE_MAX_LOSS(5%) (loss_flags)
- OOF로 학습하는 모델의 테스트 운영점 (P4-D6): 테스트 예측에 fold 모델별 점수 컬럼(attach_fold_scores)을 붙이면
  test_point·bootstrap_stat·model_curves가 같은 τ·K를 fold 점수마다 적용하고 결과를 평균한다 (min~max도).
  판별 지표는 그대로 p_fail(5개 fold 평균 점수, P4-D2)로 잰다. 모델 종류와 무관하게 fold별 점수만 있으면 된다
"""
import numpy as np
import polars as pl
import pyarrow.parquet as pq

from afp import io, metrics, split

N_GRID = (3, 5, 8, 10, 15, 20)  # 제안서 시점 N
N_GRID_ISSUE = (0, *N_GRID)  # 궤적을 보지 않는 모델(B1, 오라클)은 시작 전 N=0도 낸다
# 손실 기준은 metrics.SELECT_MAX_LOSS(선택 3.5%)·metrics.JUDGE_MAX_LOSS(판정 구간 상한 5%)에만 둔다.
# 민감도: 선택 기준을 바꿔 다시 고른 결과. 0.05는 제안서 "손실 <= 5%"로 골랐을 때와 비교하려고 남긴다
MAX_LOSS_GRID = (0.02, 0.03, 0.035, 0.05, 0.08, 0.10)
N_BOOT = 1000
COMBINED_MAX_K = 60
OBJECTIVES = ("saved_steps", "saved_tokens")
POLICIES = ("single", "combined")
CURVE_MAX_ROWS = 200
COSTS_PATH = io.PROCESSED_DIR / "context_costs.parquet"
SPLIT_LABEL = {"train": "train_oof", "test": "test"}
TEST_SPLITS = ("test",)
POINT_COLUMNS = ("stop_rate", "lost_success", "saved_steps", "saved_tokens")
FOLD_SCORE_PREFIX = "p_fail_fold"  # attach_fold_scores가 붙이는 fold k 점수 컬럼: p_fail_fold0, p_fail_fold1, ...


def load_eval_sets() -> dict[str, pl.DataFrame]:
    return {name: metrics.eval_set(name) for name in split.SPLITS}


def load_all_costs() -> pl.DataFrame:
    """모든 궤적(8B·405B 포함)의 스텝 맥락 길이. 한 번 계산해 COSTS_PATH에 캐시한다 (steps 행 수로 검사).

    평가에는 load_costs로 평가 집합 궤적만 골라 쓴다. 전체는 제안서 수치 재현 검사에만 쓴다.
    """
    n_steps_rows = pq.read_metadata(io.STEPS_PATH).num_rows
    if COSTS_PATH.exists() and pq.read_metadata(COSTS_PATH).num_rows == n_steps_rows:
        return pl.read_parquet(COSTS_PATH)
    costs = metrics.eval_costs(io.load_trajectories(["traj_id"]))
    if costs.height != n_steps_rows:
        raise AssertionError(f"맥락 길이 {costs.height}행 != steps {n_steps_rows}행")
    costs.write_parquet(COSTS_PATH)
    return costs


def load_costs(sets: dict[str, pl.DataFrame], all_costs: pl.DataFrame | None = None) -> dict[str, pl.DataFrame]:
    """평가 집합별 스텝 맥락 길이 (load_all_costs 캐시에서 그 집합 궤적만)."""
    costs = load_all_costs() if all_costs is None else all_costs
    return {name: costs.join(t.select("traj_id"), on="traj_id", how="semi") for name, t in sets.items()}


def split_preds(preds: pl.DataFrame, trajs: pl.DataFrame, n: int | None = None) -> pl.DataFrame:
    """trajs 궤적의 예측만 (하네스는 대상 밖 예측이 있으면 멈추므로 분할마다 거른다)."""
    p = preds if n is None else preds.filter(pl.col("N") == n)
    return p.join(trajs.select("traj_id"), on="traj_id", how="semi")


# ---- 판별 지표 ----

def discrimination_row(trajs: pl.DataFrame, preds: pl.DataFrame, n: int) -> dict:
    a = metrics.align(trajs, preds, n)
    d = metrics.discrimination(a["target"], a["p_fail"], a["weight"] if "weight" in a.columns else None)
    wi = metrics.within_issue_auroc(a)
    return {
        "n_running": d["n"], "base_rate": d["base_rate"], "auroc": d["auroc"], "pr_auc": d["pr_auc"],
        "within_pair": wi["pair_weighted"], "within_issue_mean": wi["issue_mean"],
        "within_n_issues": wi["n_issues"], "within_n_pairs": wi["n_pairs"],
    }


def discrimination_table(model_id: str, preds: pl.DataFrame, sets: dict[str, pl.DataFrame], n_grid) -> pl.DataFrame:
    """행 = (model, split, N). preds에 없는 분할(예: 테스트 전용 오라클의 train)은 건너뛴다."""
    rows = []
    for name, trajs in sets.items():
        sp = split_preds(preds, trajs)
        if sp.height == 0:
            continue
        for n in n_grid:
            rows.append({"model": model_id, "split": SPLIT_LABEL[name], "N": n, **discrimination_row(trajs, sp, n)})
    return pl.DataFrame(rows)


# ---- fold 모델별 테스트 점수 (P4-D6) ----

def attach_fold_scores(preds: pl.DataFrame, fold_preds: pl.DataFrame) -> pl.DataFrame:
    """preds(traj_id, N, p_fail)의 행에 fold k 모델 점수 컬럼 p_fail_fold{k}를 붙인다.

    fold_preds: traj_id, N, fold, p_fail (preds.read_test_folds 형식, test 궤적만). preds의 행마다 모든 fold 점수가
    있어야 한다 (테스트 행만 넘긴다). 한 프레임에 같이 두므로 부트스트랩에서 fold 점수가 모두 같은 재표본을 쓴다.
    """
    folds = sorted(fold_preds["fold"].unique().to_list())
    wide = fold_preds.select("traj_id", pl.col("N").cast(preds["N"].dtype), pl.col("fold").cast(pl.Int64), "p_fail") \
        .pivot(on="fold", index=["traj_id", "N"], values="p_fail", sort_columns=True) \
        .rename({str(k): f"{FOLD_SCORE_PREFIX}{k}" for k in folds})
    out = preds.join(wide, on=["traj_id", "N"], how="left")
    cols = fold_columns(out)
    if out.select(pl.any_horizontal(pl.col(cols).is_null())).to_series().any():
        raise ValueError("fold 점수가 없는 예측 행이 있음 (테스트 행만, 모든 fold 점수와 함께 넘긴다)")
    return out


def fold_columns(preds: pl.DataFrame) -> list[str]:
    return sorted((c for c in preds.columns if c.startswith(FOLD_SCORE_PREFIX)), key=lambda c: int(c[len(FOLD_SCORE_PREFIX):]))


def fold_preds_list(preds: pl.DataFrame) -> list[pl.DataFrame]:
    """fold 점수 컬럼마다 (traj_id, N, p_fail) 예측 하나."""
    return [preds.select("traj_id", "N", pl.col(c).alias("p_fail")) for c in fold_columns(preds)]


def test_point(trajs: pl.DataFrame, preds: pl.DataFrame, n: int, tau: float, cap: int | None, costs: pl.DataFrame | None,
               spread: bool = True) -> dict:
    """테스트에 고정한 (τ, K) 적용. fold 점수 컬럼이 있으면 fold마다 적용해 평균하고 (P4-D6),
    spread면 <지표>_fold_min/_fold_max도 낸다. 없으면(B0 규칙 등) apply_point 그대로."""
    cols = fold_columns(preds)
    if not cols:
        return apply_point(trajs, preds.select("traj_id", "N", "p_fail"), n, tau, cap, costs)
    pts = [apply_point(trajs, p, n, tau, cap, costs) for p in fold_preds_list(preds)]
    out = {c: float(np.mean([pt[c] for pt in pts])) for c in pts[0]}
    if spread:
        for c in pts[0]:
            out[f"{c}_fold_min"] = float(min(pt[c] for pt in pts))
            out[f"{c}_fold_max"] = float(max(pt[c] for pt in pts))
    return out


def fold_mean_curve(curves: list[pl.DataFrame]) -> pl.DataFrame:
    """fold별 비용 곡선(같은 trajs·N·K)을 τ마다 평균한 곡선. τ = 모든 곡선 τ의 합집합 (큰 쪽부터).

    각 곡선의 τ 값은 at_threshold와 같은 규칙(그 τ 이상인 마지막 행)으로 읽는다. n_stopped는 평균이라 실수다.
    """
    taus = np.unique(np.concatenate([c["tau"].to_numpy() for c in curves]))[::-1]
    value_cols = [c for c in curves[0].columns if c != "tau"]
    acc = {c: np.zeros(len(taus)) for c in value_cols}
    for c in curves:
        t = c["tau"].to_numpy()  # 큰 쪽부터 (inf가 맨 앞)
        idx = np.searchsorted(-t, -taus, side="right") - 1
        for col in value_cols:
            acc[col] += c[col].to_numpy().astype(float)[idx]
    return pl.DataFrame({"tau": taus, **{c: v / len(curves) for c, v in acc.items()}})


# ---- 비용 곡선 ----

def thin_curve(curve: pl.DataFrame, max_rows: int = CURVE_MAX_ROWS) -> pl.DataFrame:
    """저장용으로 줄인 곡선. 보간 없이 기존 행에서 고른다: 첫·끝 행과, 손실 축을 고르게 나눈 값마다 처음 닿는 행."""
    if curve.height <= max_rows:
        return curve
    loss = curve["lost_success"].to_numpy()
    targets = np.linspace(loss[0], loss[-1], max_rows)
    idx = np.unique(np.concatenate([[0, curve.height - 1], np.searchsorted(loss, targets, side="left").clip(0, curve.height - 1)]))
    return curve[idx]


def model_curves(preds: pl.DataFrame, sets: dict, costs: dict, n_grid,
                 fold_preds: pl.DataFrame | None = None) -> dict[tuple[str, int], pl.DataFrame]:
    """(split, N) → 상한 없는 비용 곡선 (saved_tokens 포함). preds에 없는 분할은 건너뛴다.

    fold_preds(테스트 fold별 점수)가 있으면 테스트 분할 곡선은 fold별 곡선을 τ마다 평균한 것 (P4-D6).
    train은 OOF 점수 하나의 곡선이다.
    """
    out = {}
    for name, trajs in sets.items():
        sp = split_preds(preds, trajs)
        if sp.height == 0:
            continue
        if fold_preds is not None and name in TEST_SPLITS:
            sp = attach_fold_scores(sp, fold_preds)
        for n in n_grid:
            pn = sp.filter(pl.col("N") == n)
            if fold_columns(pn):
                out[(name, n)] = fold_mean_curve([metrics.cost_curve(trajs, p, n, costs=costs[name]) for p in fold_preds_list(pn)])
            else:
                out[(name, n)] = metrics.cost_curve(trajs, pn, n, costs=costs[name])
    return out


# ---- 운영점 ----

def policy_caps(policy: str, n: int) -> tuple:
    if policy == "single":
        return (None,)
    if policy == "combined":
        return tuple(range(n + 1, COMBINED_MAX_K + 1))
    raise ValueError(f"policy는 {POLICIES} 중 하나: {policy!r}")


def candidate_curves(train: pl.DataFrame, train_preds: pl.DataFrame, n: int, policy: str, costs: pl.DataFrame | None) -> dict:
    """train(OOF) 곡선 {K: cost_curve}. max_loss·목적 함수가 달라도 같은 곡선에서 고른다."""
    p = train_preds.filter(pl.col("N") == n)
    return {cap: metrics.cost_curve(train, p, n, costs=costs, cap=cap) for cap in policy_caps(policy, n)}


def apply_point(trajs: pl.DataFrame, preds: pl.DataFrame, n: int, tau: float, cap: int | None, costs: pl.DataFrame | None) -> dict:
    """고정한 (τ, K)를 trajs에 적용한 정지율·손실·절약 (테스트에서 τ를 다시 고르지 않음)."""
    row = metrics.at_threshold(metrics.cost_curve(trajs, preds, n, costs=costs, cap=cap), tau)
    return {c: row[c] for c in POINT_COLUMNS if c in row}


def cap_point(trajs: pl.DataFrame, k: int, costs: pl.DataFrame | None) -> dict:
    """고정 상한 K 하나의 정지율·손실·절약."""
    row = metrics.fixed_cap_curve(trajs, costs, ks=[k]).row(0, named=True)
    return {c: row[c] for c in POINT_COLUMNS if c in row}


LOSS_FLAG_COLUMNS = ("meets_select_point", "meets_judge_ci_hi")


def loss_flags(row: dict) -> dict:
    """운영점 표 한 행의 손실 기준 표시. 어느 기준인지 컬럼 이름으로 구분한다 (해당 없는 쪽은 None).

    - meets_select_point (train 행): 점추정 손실 <= metrics.SELECT_MAX_LOSS. train에는 구간이 없다
    - meets_judge_ci_hi  (테스트 행): 부트스트랩 95% 구간 상한 lost_success_hi <= metrics.JUDGE_MAX_LOSS.
      점추정이 5% 이하여도 상한이 5%를 넘으면 불충족. 구간이 없으면(부트스트랩 생략) None
    """
    if row["split"] == "train":
        return {"meets_select_point": metrics.meets_select_loss(row["lost_success"]), "meets_judge_ci_hi": None}
    hi = row.get("lost_success_hi")
    return {"meets_select_point": None, "meets_judge_ci_hi": None if hi is None else metrics.meets_judge_loss(hi)}


def fold_spread(train: pl.DataFrame, preds: pl.DataFrame, n: int, tau: float, cap: int | None, costs: pl.DataFrame | None) -> dict:
    """train에서 고른 운영점의 fold별 손실·절약 (최소·최대). "train 손실 <= 선택 기준"이 fold 사이에서 얼마나 흔들리는지.

    preds가 None이면 고정 상한 K=cap 하나의 fold별 값.
    """
    vals = []
    for f in sorted(train["fold"].unique().to_list()):
        t = train.filter(pl.col("fold") == f)
        c = None if costs is None else costs.join(t.select("traj_id"), on="traj_id", how="semi")
        vals.append(cap_point(t, cap, c) if preds is None else apply_point(t, split_preds(preds, t, n), n, tau, cap, c))
    out = {}
    for c in ("lost_success", "saved_steps"):
        v = [x[c] for x in vals]
        out[f"fold_{c}_min"], out[f"fold_{c}_max"] = float(min(v)), float(max(v))
    return out


def bootstrap_stat(n: int | None, points: dict, k_ref: int | None, costs: pl.DataFrame | None, discrimination: bool):
    """부트스트랩 stat(trajs, preds) 함수. points = {이름: (τ, K)}.

    - discrimination: AUROC, PR-AUC, 이슈 내 AUROC 2종
    - points마다 <이름>.<지표>와 K_ref 대비 차이 <이름>.d_<지표> (같은 표본에서 계산 → 짝지은 구간)
    - preds에 fold 점수 컬럼(attach_fold_scores)이 있으면 운영점 지표는 test_point로 fold마다 적용한 값의 평균
      (P4-D6). 반복마다 같은 재표본에서 5개를 평균하므로 구간은 "평균"의 구간이다. 판별 지표는 p_fail(평균 점수)
    costs는 metrics.compress_costs로 경계 {N, K들, K_ref}만 남겨 넣는다 (반복마다 수십만 행을 집계하지 않게).
    """
    def stat(trajs, preds):
        out = {}
        if discrimination:
            d = discrimination_row(trajs, preds.select("traj_id", "N", "p_fail"), n)
            out.update({k: d[k] for k in ("auroc", "pr_auc", "within_pair", "within_issue_mean")})
        ref = cap_point(trajs, k_ref, costs) if k_ref is not None and points else None
        for name, (tau, cap) in points.items():
            pt = test_point(trajs, preds, n, tau, cap, costs, spread=False)
            out.update({f"{name}.{c}": v for c, v in pt.items()})
            if ref is not None:
                out.update({f"{name}.d_{c}": pt[c] - ref[c] for c in pt if c != "stop_rate"})
        return out
    return stat


def compressed_costs(costs: pl.DataFrame, n: int | None, caps, k_ref: int | None) -> pl.DataFrame:
    return metrics.compress_costs(costs, [b for b in (n, *caps, k_ref) if b is not None])


# ---- 운영점 이전성 (회의 2026-11-05 §4.3) ----

TRANSFER_COLUMNS = (
    "source", "target", "tau", "cap",
    "src_stop_rate", "src_lost_success", "src_saved_steps",
    "dst_stop_rate", "dst_lost_success", "dst_saved_steps",
)


def _transfer_row(source: str, target: str, op: dict, dst: dict) -> dict:
    return {
        "source": source, "target": target, "tau": op.get("tau"), "cap": op.get("cap"),
        **{f"src_{c}": op[c] for c in ("stop_rate", "lost_success", "saved_steps")},
        **{f"dst_{c}": dst[c] for c in ("stop_rate", "lost_success", "saved_steps")},
    }


def transfer_rows(train: pl.DataFrame, train_preds: pl.DataFrame | None, test: pl.DataFrame, test_preds: pl.DataFrame | None,
                  n: int | None, policy: str, full_op: dict, max_loss: float = metrics.SELECT_MAX_LOSS, by: str = "saved_steps",
                  test_fold_preds: list[pl.DataFrame] | None = None) -> list[dict]:
    """train에서 고른 운영점이 처음 보는 이슈로 옮겨지는지 (정지율·손실이 유지되는지).

    - source=train-fold{k}: fold k를 뺀 학습 이슈 OOF 곡선에서 고른 운영점을 target=fold{k}(같은 방식으로
      뽑힌 처음 보는 학습 이슈)와 target=test에 각각 적용한다. 같은 운영점이라 두 target을 짝지어 비교한다
    - source=train: 학습 이슈 전체에서 고른 주 운영점(full_op)을 test(평균 점수)에 적용. B0는 operating_points.csv와
      같은 값이고, OOF 모델은 2026-11-05 P4-D6부터 operating_points.csv가 fold 모델별 채점 평균이라 다르다 (이 표는 진단용)
    - test_fold_preds[k] (OOF로 학습하는 모델): 테스트를 fold k 모델 하나로만 채점한 예측. 있으면 같은 운영점을
      target=test_fold_model에도 적용한다. fold{k}와 채점 모델이 같아서, test(5개 fold 모델 평균 점수)와의 차이가
      "점수 평균" 때문인지 "이슈 차이" 때문인지 가른다
    fold 사이에서는 잘 옮겨지는데 test로만 안 옮겨지면 테스트 이슈 선택(또는 테스트 점수 계산 방식)이 원인이다.
    policy="cap"이면 예측 없이 고정 상한 K만 고른다 (select_cap, train_preds·n은 None). 비용은 스텝만 본다.
    """
    rows = []
    for k in sorted(train["fold"].unique().to_list()):
        src, held = train.filter(pl.col("fold") != k), train.filter(pl.col("fold") == k)
        if policy == "cap":
            op = metrics.select_cap(metrics.fixed_cap_curve(src), max_loss)
            op = {**op, "tau": None, "cap": int(op["K"])}
            targets = {f"fold{k}": cap_point(held, op["cap"], None), "test": cap_point(test, op["cap"], None)}
        else:
            curves = candidate_curves(src, split_preds(train_preds, src, n), n, policy, None)
            op = metrics.select_from_curves(curves, n, max_loss=max_loss, by=by)
            targets = {f"fold{k}": apply_point(held, split_preds(train_preds, held, n), n, op["tau"], op["cap"], None),
                       "test": apply_point(test, split_preds(test_preds, test, n), n, op["tau"], op["cap"], None)}
            if test_fold_preds is not None:
                targets["test_fold_model"] = apply_point(test, split_preds(test_fold_preds[k], test, n), n, op["tau"], op["cap"], None)
        rows += [_transfer_row(f"train-fold{k}", name, op, dst) for name, dst in targets.items()]
    if policy == "cap":
        dst = cap_point(test, full_op["cap"], None)
    else:
        dst = apply_point(test, split_preds(test_preds, test, n), n, full_op["tau"], full_op["cap"], None)
    rows.append(_transfer_row("train", "test", full_op, dst))
    return rows


def transfer_summary(rows: pl.DataFrame) -> pl.DataFrame:
    """transfer_rows 표 → (model, policy, N)마다 fold 이전과 test 이전의 정지율 비·손실 차이 요약.

    stop_ratio = dst_stop_rate / src_stop_rate (src가 0이면 null), d_lost = dst_lost_success - src_lost_success.
    - fold_*: source=train-fold{k} → target=fold{k} 5개의 최소·최대
    - test_from_folds_*: 같은 5개 운영점 → target=test(fold 모델 평균 점수)의 최소·최대
    - test_fold_model_*: 같은 5개 운영점 → target=test_fold_model(fold k 모델 하나의 점수). B1처럼 OOF로 학습하는 모델만
    - test_from_train_*: 주 운영점 → test
    """
    r = rows.with_columns(
        stop_ratio=pl.when(pl.col("src_stop_rate") > 0).then(pl.col("dst_stop_rate") / pl.col("src_stop_rate")),
        d_lost=pl.col("dst_lost_success") - pl.col("src_lost_success"),
        kind=pl.when(pl.col("source") == "train").then(pl.lit("test_from_train"))
        .when(pl.col("target") == "test").then(pl.lit("test_from_folds"))
        .when(pl.col("target") == "test_fold_model").then(pl.lit("test_fold_model"))
        .otherwise(pl.lit("fold")),
    )
    aggs = []
    for kind in ("fold", "test_from_folds", "test_fold_model", "test_from_train"):
        sel = pl.col("kind") == kind
        for c in ("stop_ratio", "d_lost"):
            aggs += [pl.col(c).filter(sel).min().alias(f"{kind}_{c}_min"), pl.col(c).filter(sel).max().alias(f"{kind}_{c}_max")]
    return r.group_by("model", "policy", "N", maintain_order=True).agg(aggs)
