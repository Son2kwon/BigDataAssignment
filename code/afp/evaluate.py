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


# ---- 운영점 선택·적용 루프 (P5-D2, Phase 5.0에서 03_baselines.py에서 옮김) ----
# 모델과 무관하다. 입력은 모델마다 "train OOF + test 예측 (traj_id, N, p_fail)"과, OOF로 학습한 모델이면
# "test fold별 예측 (traj_id, N, fold, p_fail)" 하나씩이다 (ModelPreds). B0·B1(03_baselines.py)과 M1~M3가 같은 함수를 쓴다.

ModelPreds = tuple  # (preds, fold_preds 또는 None)

K_PROPOSAL = 35  # 제안서의 고정 상한 (전체 데이터에서 고른 값, 참고 행 cap_proposal)
DELTA_COLUMNS = ("lost_success", "saved_steps", "saved_tokens")
OP_COLUMNS = [
    "model", "policy", "N", "tau", "cap", "split", "stop_rate", "lost_success", "lost_success_lo", "lost_success_hi",
    *LOSS_FLAG_COLUMNS, "saved_steps", "saved_steps_lo", "saved_steps_hi", "saved_tokens", "saved_tokens_lo", "saved_tokens_hi",
    "d_lost_success", "d_lost_success_lo", "d_lost_success_hi", "d_saved_steps", "d_saved_steps_lo", "d_saved_steps_hi",
    "d_saved_tokens", "d_saved_tokens_lo", "d_saved_tokens_hi",
    "fold_lost_success_min", "fold_lost_success_max", "fold_saved_steps_min", "fold_saved_steps_max",
    # test 행, OOF 모델만 (P4-D6): fold 모델별 점수 5개에 같은 τ·K를 적용한 결과의 최소·최대 (주 값은 평균)
    *(f"{c}_fold_{e}" for c in POINT_COLUMNS for e in ("min", "max")),
]
SENS_COLUMNS = ["model", "policy", "N", "objective", "max_loss", "tau", "cap", "split", *POINT_COLUMNS]
OP_SCHEMA = {"N": pl.Int64, "cap": pl.Int64, "tau": pl.Float64}


def parallel(fn, tasks, jobs):
    """작업마다 결과가 정해져 있어(고정 시드) 실행 순서와 무관하게 같은 값이 나온다."""
    if jobs <= 1:
        return [fn(*t) for t in tasks]
    import os

    from joblib import Parallel, delayed
    os.environ["POLARS_MAX_THREADS"] = "1"  # 이제 뜨는 작업 프로세스에만 적용 (이 프로세스는 이미 스레드 풀이 있음)
    return Parallel(n_jobs=jobs, backend="loky", inner_max_num_threads=1)(delayed(fn)(*t) for t in tasks)


def frame(rows, columns, schema_overrides=None) -> pl.DataFrame:
    """행 dict 목록 → columns 순서의 표. 없는 컬럼은 null(Float64)로 채운다."""
    df = pl.DataFrame(rows, infer_schema_length=None, schema_overrides=schema_overrides)
    for c in columns:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
    return df.select(columns)


def test_preds(preds: pl.DataFrame, fold_preds: pl.DataFrame | None, trajs: pl.DataFrame, n: int | None = None) -> pl.DataFrame:
    """테스트 분할 예측. fold별 점수가 있으면(OOF 모델) p_fail_fold{k} 컬럼을 붙인다 (P4-D6)."""
    p = split_preds(preds, trajs, n)
    if fold_preds is not None:
        p = attach_fold_scores(p, fold_preds if n is None else fold_preds.filter(pl.col("N") == n))
    return p


def operating_points(models: dict[str, ModelPreds], sets: dict, costs: dict, k_star: int, n_grid=N_GRID):
    """모델마다 train(OOF) 곡선에서 운영점을 고르고 테스트에 적용한다.

    models = {model_id: (preds, fold_preds 또는 None)}. preds는 train OOF + test 예측, fold_preds는 test fold별 예측.
    - 주 운영점: 목적 saved_steps, 선택 기준 metrics.SELECT_MAX_LOSS. 단독·결합 정책 각각
    - 민감도: OBJECTIVES × MAX_LOSS_GRID로 같은 곡선에서 다시 고른 결과
    - test 적용은 test_point (fold 점수가 있으면 fold마다 적용해 평균, P4-D6)
    반환: (주 운영점 행 — 부트스트랩 구간은 attach_ci로 나중에 붙임, 민감도 행, chosen {(model, N, policy): (τ, K)}).
    """
    train = sets["train"]
    main, sens, chosen = [], [], {}
    applied = {}
    ref_train = cap_point(train, k_star, costs["train"])
    for model, (p_all, fold_p) in models.items():
        sp = {name: (test_preds(p_all, fold_p, t) if name in TEST_SPLITS else split_preds(p_all, t)) for name, t in sets.items()}
        for n in n_grid:
            for policy in POLICIES:
                curves = candidate_curves(train, sp["train"], n, policy, costs["train"])
                for by in OBJECTIVES:
                    for max_loss in MAX_LOSS_GRID:
                        op = metrics.select_from_curves(curves, n, max_loss=max_loss, by=by)
                        key = (model, n, op["tau"], op["cap"])
                        for name in TEST_SPLITS:
                            if (key, name) not in applied:
                                applied[(key, name)] = test_point(sets[name], sp[name].filter(pl.col("N") == n), n, op["tau"], op["cap"], costs[name])
                        base = {"model": model, "policy": policy, "N": n, "objective": by, "max_loss": max_loss, "tau": op["tau"], "cap": op["cap"]}
                        sens.append({**base, "split": "train", **{c: op[c] for c in POINT_COLUMNS}})
                        sens += [{**base, "split": name, **{c: applied[(key, name)][c] for c in POINT_COLUMNS}} for name in TEST_SPLITS]
                        if by == "saved_steps" and max_loss == metrics.SELECT_MAX_LOSS:
                            chosen[(model, n, policy)] = (op["tau"], op["cap"])
                            spread = fold_spread(train, sp["train"], n, op["tau"], op["cap"], costs["train"])
                            tr = {c: op[c] for c in POINT_COLUMNS}
                            main.append({"model": model, "policy": policy, "N": n, "tau": op["tau"], "cap": op["cap"], "split": "train",
                                         **tr, **{f"d_{c}": tr[c] - ref_train[c] for c in DELTA_COLUMNS}, **spread})
                            main += [{"model": model, "policy": policy, "N": n, "tau": op["tau"], "cap": op["cap"], "split": name,
                                      **applied[(key, name)]} for name in TEST_SPLITS]
    return main, sens, chosen


def cap_rows(sets: dict, costs: dict, k_star: int, cap_curve_train: pl.DataFrame, k_proposal: int = K_PROPOSAL):
    """고정 상한 기준선 행: K*(policy=cap)와 제안서 K(cap_proposal) × 분할, 그리고 선택 기준별 K 민감도 행."""
    rows, sens = [], []
    ref = {name: cap_point(t, k_star, costs[name]) for name, t in sets.items()}
    for policy, k in (("cap", k_star), ("cap_proposal", k_proposal)):
        for name, t in sets.items():
            pt = cap_point(t, k, costs[name])
            row = {"model": "b0_cap", "policy": policy, "N": None, "tau": None, "cap": k, "split": name, **pt}
            if name == "train":  # K* 행은 차이의 기준이라 d_*가 없다
                if k != k_star:
                    row.update({f"d_{c}": pt[c] - ref[name][c] for c in DELTA_COLUMNS})
                row.update(fold_spread(sets["train"], None, None, None, k, costs["train"]))
            rows.append(row)
    for max_loss in MAX_LOSS_GRID:
        k = metrics.select_cap(cap_curve_train, max_loss)["K"]
        for name, t in sets.items():
            sens.append({"model": "b0_cap", "policy": "cap", "N": None, "objective": "saved_steps", "max_loss": max_loss,
                         "tau": None, "cap": k, "split": name, **cap_point(t, k, costs[name])})
    return rows, sens


def cap_stat(ks, costs):
    """고정 상한 K 목록의 부트스트랩 stat. ks[0]이 기준(K*), 나머지는 K* 대비 차이 K<k>.d_<지표>도 낸다."""
    def stat(trajs, _preds):
        out = {}
        for k in ks:
            out.update({f"K{k}.{c}": v for c, v in cap_point(trajs, k, costs).items()})
        for k in ks[1:]:
            out.update({f"K{k}.d_{c}": out[f"K{k}.{c}"] - out[f"K{ks[0]}.{c}"] for c in DELTA_COLUMNS})
        return out
    return stat


def boot_task(trajs, p, stat, n_boot):
    return metrics.bootstrap(trajs, p, stat, n_boot=n_boot)


def bootstrap_tasks(models: dict[str, ModelPreds], sets: dict, costs: dict, chosen: dict, k_star: int | None,
                    n_grid_of: dict, n_boot: int) -> tuple[list, list]:
    """테스트 분할 부트스트랩 작업 (boot_task 인자, 키 (model, split, N)). 모델·N마다 판별 지표 + 고른 운영점.

    chosen에 없는 (model, N)은 판별 지표만 (예: 오라클, N=0). 분할마다 같은 시드라 모델 사이에서도 같은 재표본을 쓴다
    (스크립트가 달라도 같다: 재표본은 평가 집합의 이슈와 io.SEED로만 정해진다).
    """
    tasks, keys = [], []
    for name in TEST_SPLITS:
        t = sets[name]
        for m, (p_all, fold_p) in models.items():
            for n in n_grid_of[m]:
                points = {pol: chosen[(m, n, pol)] for pol in POLICIES if (m, n, pol) in chosen}
                caps_used = [cap for _, cap in points.values()]
                c = compressed_costs(costs[name], n, caps_used, k_star) if points else None
                stat = bootstrap_stat(n, points, k_star if points else None, c, discrimination=True)
                tasks.append((t, test_preds(p_all, fold_p, t, n), stat, n_boot))
                keys.append((m, name, n))
    return tasks, keys


def cap_bootstrap_tasks(sets: dict, costs: dict, k_star: int, n_boot: int, k_proposal: int = K_PROPOSAL) -> tuple[list, list]:
    """고정 상한 K*·제안서 K의 테스트 부트스트랩 작업 (키 ("b0_cap", split, None))."""
    from afp import preds as preds_mod
    tasks, keys = [], []
    for name in TEST_SPLITS:
        c = compressed_costs(costs[name], None, [k_proposal], k_star)
        empty = pl.DataFrame(schema=preds_mod.SCHEMA)
        ks = list(dict.fromkeys([k_star, k_proposal]))  # K*가 제안서 K와 같으면 차이(0)를 따로 만들지 않는다
        tasks.append((sets[name], empty, cap_stat(ks, c), n_boot))
        keys.append(("b0_cap", name, None))
    return tasks, keys


def discrimination_ci(boots: dict, no_within=()) -> pl.DataFrame:
    """부트스트랩 결과 → 판별 지표 구간 표 (model, split, N, stat, estimate, lo, hi, n_boot). no_within 모델은 이슈 내 AUROC를 뺀다."""
    rows = []
    for (m, name, n), b in boots.items():
        if m == "b0_cap":
            continue
        keep = ["auroc", "pr_auc"] + ([] if m in no_within else ["within_pair", "within_issue_mean"])
        rows.append(b.filter(pl.col("stat").is_in(keep)).with_columns(model=pl.lit(m), split=pl.lit(name), N=pl.lit(n)))
    return pl.concat(rows).select("model", "split", "N", "stat", "estimate", "lo", "hi", "n_boot")


def attach_ci(main_rows: list[dict], boots: dict) -> list[dict]:
    """주 운영점 행에 테스트 부트스트랩 구간을 붙인다. boots[(model, split, N)] = bootstrap 출력."""
    out = []
    for r in main_rows:
        r = dict(r)
        if r["split"] in TEST_SPLITS and boots:
            if r["model"] == "b0_cap":
                b, prefix = boots[("b0_cap", r["split"], None)], f"K{r['cap']}."
            else:
                b, prefix = boots[(r["model"], r["split"], r["N"])], f"{r['policy']}."
            ci = {s[len(prefix):]: (e, lo, hi) for s, e, lo, hi in b.select("stat", "estimate", "lo", "hi").iter_rows() if s.startswith(prefix)}
            for c, (e, lo, hi) in ci.items():
                if c.startswith("d_") or c not in r:
                    r[c] = e
                r[f"{c}_lo"], r[f"{c}_hi"] = lo, hi
        out.append(r)
    return out


def operating_table(rows: list[dict], boots: dict) -> pl.DataFrame:
    """운영점 행(cap_rows + operating_points) → 구간·손실 기준 표시를 붙인 operating_points.csv 형식 표."""
    ops = attach_ci(rows, boots)
    for r in ops:
        r.update(loss_flags(r))
    return frame(ops, OP_COLUMNS, {**OP_SCHEMA, **dict.fromkeys(LOSS_FLAG_COLUMNS, pl.Boolean)})


def sensitivity_table(rows: list[dict]) -> pl.DataFrame:
    return frame(rows, SENS_COLUMNS, OP_SCHEMA)


def judged_and_saves_more(ops_df: pl.DataFrame) -> pl.DataFrame:
    """판정 기준 충족 + K* 대비 절약 스텝 차이 구간 하한 > 0인 모델 운영점 (test 행)."""
    return ops_df.filter(
        pl.col("split").is_in(TEST_SPLITS) & (pl.col("model") != "b0_cap")
        & pl.col("meets_judge_ci_hi") & (pl.col("d_saved_steps_lo") > 0)).select(
        "model", "policy", "N", "tau", "cap", "lost_success", "lost_success_hi", "saved_steps", "d_saved_steps",
        "d_saved_steps_lo", "d_saved_steps_hi")


def loss_ci_width(ops_df: pl.DataFrame, k_star: int, models=None) -> dict:
    """테스트 손실 95% 구간 폭 (hi - lo)과 반폭. K* 행과 모델 운영점(models가 있으면 그 모델만) 전체의 범위."""
    t = ops_df.filter(pl.col("split").is_in(TEST_SPLITS)).with_columns(w=pl.col("lost_success_hi") - pl.col("lost_success_lo"))
    k = t.filter((pl.col("policy") == "cap") & (pl.col("cap") == k_star)).row(0, named=True)
    m = t.filter(pl.col("model") != "b0_cap") if models is None else t.filter(pl.col("model").is_in(list(models)))
    return {"k_star": {"lost_success": k["lost_success"], "lo": k["lost_success_lo"], "hi": k["lost_success_hi"], "half_width": k["w"] / 2},
            "models_half_width_min": float(m["w"].min()) / 2, "models_half_width_median": float(m["w"].median()) / 2,
            "models_half_width_max": float(m["w"].max()) / 2,
            "combined_half_width_median": float(m.filter(pl.col("policy") == "combined")["w"].median()) / 2,
            "single_half_width_median": float(m.filter(pl.col("policy") == "single")["w"].median()) / 2}


# ---- 두 모델 짝지은 비교 (Phase 6, M2 vs M1. M3도 같은 함수) ----
# 두 모델의 test 예측을 한 프레임에 두고 같은 부트스트랩 재표본에서 차이(a − b)를 잰다. b의 컬럼에는 PAIR_PREFIX를 붙인다.

PAIR_PREFIX = "b__"
PAIR_DISC = ("auroc", "pr_auc", "within_pair", "within_issue_mean")


def pair_preds(a: pl.DataFrame, b: pl.DataFrame) -> pl.DataFrame:
    """두 모델의 같은 (traj_id, N) 예측을 한 프레임으로. a·b 모두 traj_id, N, p_fail(+ p_fail_fold{k}).

    b의 점수 컬럼은 b__p_fail, b__p_fail_fold{k}가 된다. 두 모델의 (traj_id, N)이 다르면 멈춘다.
    """
    score_b = [c for c in b.columns if c == "p_fail" or c.startswith(FOLD_SCORE_PREFIX)]
    bb = b.select("traj_id", pl.col("N").cast(a["N"].dtype), *[pl.col(c).alias(PAIR_PREFIX + c) for c in score_b])
    out = a.join(bb, on=["traj_id", "N"], how="full", coalesce=True)
    if out.select(pl.any_horizontal(pl.all().is_null())).to_series().any():
        raise ValueError("두 모델의 (traj_id, N) 예측이 다르다")
    return out.sort("N", "traj_id")


def unpair(preds: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    a = preds.select([c for c in preds.columns if not c.startswith(PAIR_PREFIX)])
    b = preds.select("traj_id", "N", *[pl.col(c).alias(c[len(PAIR_PREFIX):]) for c in preds.columns if c.startswith(PAIR_PREFIX)])
    return a, b


def paired_stat(n: int, points_a: dict, points_b: dict, costs: pl.DataFrame | None):
    """부트스트랩 stat: a·b의 판별 지표와 차이 d_<지표> = a − b, 운영점 points(정책 → (τ, K))마다 <정책>.d_<지표>.

    points_a·points_b는 같은 정책 이름을 가진다 (각 모델이 train에서 고른 자기 τ·K). 운영점은 test_point (fold 점수가
    있으면 fold마다 적용해 평균, P4-D6). costs는 compressed_costs로 두 모델의 K를 모두 경계에 넣어 준다.
    """
    def stat(trajs, preds):
        a, b = unpair(preds)
        da = discrimination_row(trajs, a.select("traj_id", "N", "p_fail"), n)
        db = discrimination_row(trajs, b.select("traj_id", "N", "p_fail"), n)
        out = {}
        for c in PAIR_DISC:
            out.update({f"a.{c}": da[c], f"b.{c}": db[c], f"d_{c}": da[c] - db[c]})
        for name in points_a:
            pa = test_point(trajs, a, n, *points_a[name], costs, spread=False)
            pb = test_point(trajs, b, n, *points_b[name], costs, spread=False)
            for c in POINT_COLUMNS:
                out.update({f"{name}.a.{c}": pa[c], f"{name}.b.{c}": pb[c], f"{name}.d_{c}": pa[c] - pb[c]})
        return out
    return stat


N_SUMMARY_COLUMNS = [
    "model", "policy", "N", "tau", "cap", "stop_rate", "lost_success", "lost_success_hi", "meets_judge_ci_hi",
    "saved_steps", "d_saved_steps", "d_saved_steps_lo", "d_saved_steps_hi",
    "saved_tokens", "d_saved_tokens", "d_saved_tokens_lo", "d_saved_tokens_hi",
]


def n_summary(ops_df: pl.DataFrame) -> pl.DataFrame:
    """운영점 표(operating_table) → test 행의 N 요약. 절약 스텝과 절약 토큰을 K* 대비 차이 [95%]와 함께 (CLAUDE.md 평가 규칙).

    고정 상한 행(b0_cap)은 N이 없어서 맨 앞에 둔다.
    """
    t = ops_df.filter(pl.col("split").is_in(TEST_SPLITS))
    return pl.concat([t.filter(pl.col("model") == "b0_cap").select(N_SUMMARY_COLUMNS),
                      t.filter(pl.col("model") != "b0_cap").sort("model", "policy", "N").select(N_SUMMARY_COLUMNS)])
