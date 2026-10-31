"""모델 하나의 예측 → 분할 × N 지표 표, 비용 곡선, 운영점 표. Phase 4~7이 같은 코드로 평가한다.

- 평가 집합은 metrics.eval_set의 train, test_same, test_cross. train 행은 OOF 점수 기준이라 표에 train_oof로 적는다
- 운영점은 train(OOF) 곡선에서만 고르고 테스트에는 고른 (τ, K)를 그대로 적용한다 (PLAN 3.5)
  - single:   "시점 N에서 p_fail >= τ면 종료" (상한 없음)
  - combined: "스텝 > K면 종료 또는 시점 N에서 p_fail >= τ면 종료", K ∈ N+1..COMBINED_MAX_K
- 고정 상한 기준선 K*는 train에서 metrics.select_cap으로 고른 값. 모델 운영점은 같은 부트스트랩 표본의
  K*와의 차이(짝지은 구간)로 비교한다
"""
import numpy as np
import polars as pl
import pyarrow.parquet as pq

from afp import io, metrics, split

N_GRID = (3, 5, 8, 10, 15, 20)  # 제안서 시점 N
N_GRID_ISSUE = (0, *N_GRID)  # 궤적을 보지 않는 모델(B1, 오라클)은 시작 전 N=0도 낸다
MAX_LOSS = metrics.MAX_LOSS
MAX_LOSS_GRID = (0.02, 0.03, 0.05, 0.08, 0.10)  # 민감도
N_BOOT = 1000
COMBINED_MAX_K = 60
OBJECTIVES = ("saved_steps", "saved_tokens")
POLICIES = ("single", "combined")
CURVE_MAX_ROWS = 200
COSTS_PATH = io.PROCESSED_DIR / "context_costs.parquet"
SPLIT_LABEL = {"train": "train_oof", "test_same": "test_same", "test_cross": "test_cross"}
TEST_SPLITS = ("test_same", "test_cross")
POINT_COLUMNS = ("stop_rate", "lost_success", "saved_steps", "saved_tokens")


def load_eval_sets() -> dict[str, pl.DataFrame]:
    return {name: metrics.eval_set(name) for name in split.SPLITS}


def load_costs(sets: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """평가 집합별 스텝 맥락 길이. 전체 궤적에 대해 한 번 계산해 COSTS_PATH에 캐시한다 (steps 행 수로 검사)."""
    n_steps_rows = pq.read_metadata(io.STEPS_PATH).num_rows
    if COSTS_PATH.exists() and pq.read_metadata(COSTS_PATH).num_rows == n_steps_rows:
        costs = pl.read_parquet(COSTS_PATH)
    else:
        costs = metrics.eval_costs(io.load_trajectories(["traj_id"]))
        if costs.height != n_steps_rows:
            raise AssertionError(f"맥락 길이 {costs.height}행 != steps {n_steps_rows}행")
        costs.write_parquet(COSTS_PATH)
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


# ---- 비용 곡선 ----

def thin_curve(curve: pl.DataFrame, max_rows: int = CURVE_MAX_ROWS) -> pl.DataFrame:
    """저장용으로 줄인 곡선. 보간 없이 기존 행에서 고른다: 첫·끝 행과, 손실 축을 고르게 나눈 값마다 처음 닿는 행."""
    if curve.height <= max_rows:
        return curve
    loss = curve["lost_success"].to_numpy()
    targets = np.linspace(loss[0], loss[-1], max_rows)
    idx = np.unique(np.concatenate([[0, curve.height - 1], np.searchsorted(loss, targets, side="left").clip(0, curve.height - 1)]))
    return curve[idx]


def model_curves(preds: pl.DataFrame, sets: dict, costs: dict, n_grid) -> dict[tuple[str, int], pl.DataFrame]:
    """(split, N) → 상한 없는 비용 곡선 (saved_tokens 포함). preds에 없는 분할은 건너뛴다."""
    out = {}
    for name, trajs in sets.items():
        sp = split_preds(preds, trajs)
        if sp.height == 0:
            continue
        for n in n_grid:
            out[(name, n)] = metrics.cost_curve(trajs, sp.filter(pl.col("N") == n), n, costs=costs[name])
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


def fold_spread(train: pl.DataFrame, preds: pl.DataFrame, n: int, tau: float, cap: int | None, costs: pl.DataFrame | None) -> dict:
    """train에서 고른 운영점의 fold별 손실·절약 (최소·최대). "손실 <= 5%"가 fold 사이에서 얼마나 흔들리는지.

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
    costs는 metrics.compress_costs로 경계 {N, K들, K_ref}만 남겨 넣는다 (반복마다 수십만 행을 집계하지 않게).
    """
    def stat(trajs, preds):
        out = {}
        if discrimination:
            d = discrimination_row(trajs, preds, n)
            out.update({k: d[k] for k in ("auroc", "pr_auc", "within_pair", "within_issue_mean")})
        ref = cap_point(trajs, k_ref, costs) if k_ref is not None and points else None
        for name, (tau, cap) in points.items():
            pt = apply_point(trajs, preds, n, tau, cap, costs)
            out.update({f"{name}.{c}": v for c, v in pt.items()})
            if ref is not None:
                out.update({f"{name}.d_{c}": pt[c] - ref[c] for c in pt if c != "stop_rate"})
        return out
    return stat


def compressed_costs(costs: pl.DataFrame, n: int | None, caps, k_ref: int | None) -> pl.DataFrame:
    return metrics.compress_costs(costs, [b for b in (n, *caps, k_ref) if b is not None])
