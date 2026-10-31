"""평가 하네스: 모든 모델의 예측을 같은 코드로 평가한다.

입력 형식
- trajs: 평가 대상 궤적 집합 (예: test_same 분할). 컬럼 traj_id, instance_id, target, n_steps.
  시점 N에서 끝난 궤적(n_steps <= N)도 포함해야 한다. 비용 곡선의 분모가 전체 궤적이다.
  선택 컬럼 weight: 궤적 가중치 (부트스트랩에서 이슈가 뽑힌 횟수). 없으면 모두 1.
- preds: 모델 예측. 컬럼 traj_id, N, p_fail (실패 확률).
  시점 N의 예측은 n_steps > N인 궤적 전부에 하나씩 있어야 한다.

라벨 방향: 모델은 실패 확률을 내고, AUROC는 p_fail로 실패를 맞히는 값(= -p_fail로 성공을
맞히는 값과 같음), PR-AUC는 성공 클래스 기준(점수 -p_fail)으로 잰다.
"""
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from afp import io, split

TRAJ_COLUMNS = ["traj_id", "instance_id", "target", "n_steps"]
MAX_LOSS = 0.05  # 운영점 제약: 성공 손실률 상한 (proposal.md §5)


def eval_set(name: str) -> pl.DataFrame:
    """평가 집합 하나의 궤적 = 하네스의 trajs 입력. name은 train, test_same, test_cross 중 하나.

    모든 모델이 같은 집합으로 평가받도록 trajs는 이 함수로만 만든다.
    - N 전에 끝난 궤적도 들어 있다 (비용 곡선의 분모가 전체 궤적이다). 미리 거르지 않는다
    - test_same과 test_cross는 같은 테스트 이슈를 공유하므로 합쳐서 평가하지 않는다. 합치면
      이슈 내 AUROC가 한 이슈 안의 70B 궤적과 8B·405B 궤적을 비교하게 된다
    - K 고정 상한 같은 기준값도 같은 집합에서 다시 계산한다 (전체 데이터 값을 가져오지 않는다)
    출력: traj_id, instance_id, target, n_steps, fold (train은 0..4, 테스트는 split.HELDOUT_FOLD).
    """
    if name not in split.SPLITS:
        raise ValueError(f"평가 집합은 {split.SPLITS} 중 하나: {name!r}")
    ids = io.load_splits().filter(pl.col("split") == name).select("traj_id", "fold")
    trajs = io.load_trajectories(TRAJ_COLUMNS).join(ids, on="traj_id", how="inner")
    if trajs.height != ids.height:
        raise ValueError(f"{name}: splits의 궤적 {ids.height}개 중 {ids.height - trajs.height}개가 trajectories에 없음")
    return trajs.sort("traj_id")


def eval_costs(trajs: pl.DataFrame) -> pl.DataFrame:
    """trajs 궤적들의 스텝별 맥락 길이 (cost_curve·point_metrics의 costs 입력)."""
    ids = trajs.select("traj_id")
    base = io.load_trajectories(["traj_id", "system_len", "issue_msg_len"]).join(ids, on="traj_id", how="semi")
    steps = io.scan_steps(["traj_id", "step_idx", "ai_len", "obs_len"]).join(ids.lazy(), on="traj_id", how="semi")
    return context_costs(steps, base)


def _columns(df: pl.DataFrame) -> list[str]:
    return TRAJ_COLUMNS + (["weight"] if "weight" in df.columns else [])


def _weight(df: pl.DataFrame) -> pl.Expr:
    return pl.col("weight") if "weight" in df.columns else pl.lit(1, dtype=pl.Int64)


def running(trajs: pl.DataFrame, n: int) -> pl.DataFrame:
    """시점 N에 아직 진행 중인 궤적 (스텝 수 > N)."""
    return trajs.filter(pl.col("n_steps") > n)


def align(trajs: pl.DataFrame, preds: pl.DataFrame, n: int) -> pl.DataFrame:
    """시점 N의 예측을 진행 중 궤적에 붙인다. 빠지거나 남거나 중복된 예측이 있으면 멈춘다."""
    alive = running(trajs.select(_columns(trajs)), n)
    p = preds.filter(pl.col("N") == n).select("traj_id", "p_fail")
    if p["traj_id"].n_unique() != p.height:
        raise ValueError(f"N={n}: traj_id 중복 예측 {p.height - p['traj_id'].n_unique()}개")
    if p["p_fail"].null_count() or p["p_fail"].is_nan().any():
        raise ValueError(f"N={n}: p_fail에 null/NaN이 있음")
    out = alive.join(p, on="traj_id", how="left")
    missing = out["p_fail"].null_count()
    extra = p.height - (out.height - missing)
    if missing or extra:
        raise ValueError(f"N={n}: 진행 중 궤적 {alive.height}개 중 예측 없음 {missing}개, 대상 밖 예측 {extra}개")
    return out.sort("traj_id")  # join 순서는 실행마다 달라서, 합산 순서를 고정해 결과를 재현 가능하게 한다


def discrimination(target, p_fail, weight=None) -> dict:
    """전체 AUROC, PR-AUC(성공 클래스). base_rate = 성공률 = 무작위 예측기의 PR-AUC 기댓값."""
    target = np.asarray(target, dtype=bool)
    p_fail = np.asarray(p_fail, dtype=float)
    w = np.ones(len(target)) if weight is None else np.asarray(weight, dtype=float)
    n, n_succ = w.sum(), w[target].sum()
    out = {"n": int(n), "n_success": int(n_succ), "base_rate": float(n_succ / n) if n else np.nan}
    if 0 < n_succ < n:
        out["auroc"] = float(roc_auc_score(~target, p_fail, sample_weight=weight))
        out["pr_auc"] = float(average_precision_score(target, -p_fail, sample_weight=weight))
    else:
        out["auroc"] = out["pr_auc"] = np.nan
    return out


def within_issue_auroc(aligned: pl.DataFrame) -> dict:
    """이슈 내 AUROC: 성공·실패가 섞인 이슈 안의 (성공, 실패) 쌍만 비교한다.

    이슈 난이도로는 설명되지 않는 궤적 자체의 신호를 잰다. 이슈마다 p_fail 순위로
    Mann-Whitney U를 구한다 (실패의 p_fail이 더 크면 1, 같으면 0.5).
    - pair_weighted: 모든 이슈의 쌍을 모아 계산 (쌍이 많은 이슈의 비중이 큼). 주 지표
    - issue_mean: 이슈별 AUROC의 단순 평균. 보조 지표
    aligned는 align()의 출력 (instance_id, target, p_fail 필요). weight가 있으면 이슈 가중치로 쓴다
    (부트스트랩에서 한 이슈의 궤적은 가중치가 모두 같다). 개수들도 가중치를 곱한 값이다.
    """
    per_issue = (
        aligned.with_columns(rank=pl.col("p_fail").rank("average").over("instance_id"))
        .group_by("instance_id")
        .agg(
            n=pl.len(),
            n_succ=pl.col("target").sum(),
            fail_rank_sum=pl.col("rank").filter(~pl.col("target")).sum(),
            w=_weight(aligned).first(),
        )
        .with_columns(n_fail=pl.col("n") - pl.col("n_succ"))
        .filter((pl.col("n_succ") > 0) & (pl.col("n_fail") > 0))
        .sort("instance_id")
        .with_columns(
            pairs=pl.col("n_succ") * pl.col("n_fail"),
            u=pl.col("fail_rank_sum") - pl.col("n_fail") * (pl.col("n_fail") + 1) / 2,
        )
    )
    w = per_issue["w"]
    out = {"n_issues": int(w.sum()), "n_traj": int((w * per_issue["n"]).sum()), "n_pairs": int((w * per_issue["pairs"]).sum())}
    if per_issue.height == 0:
        out["pair_weighted"] = out["issue_mean"] = np.nan
    else:
        out["pair_weighted"] = float((w * per_issue["u"]).sum() / (w * per_issue["pairs"]).sum())
        out["issue_mean"] = float((w * per_issue["u"] / per_issue["pairs"]).sum() / w.sum())
    return out


def context_costs(steps: pl.DataFrame | pl.LazyFrame, trajs: pl.DataFrame) -> pl.DataFrame:
    """스텝별 입력 맥락 길이 근사 (문자 수, 토큰 수에 비례한다고 가정 — PLAN D4).

    스텝 t에서 모델이 읽는 맥락 = 시스템 프롬프트 + 이슈 메시지 + 앞 스텝들의 (ai + 관측).
    ctx_t = system_len + issue_msg_len + sum_{s<t}(ai_len_s + obs_len_s)
    steps는 steps.parquet (traj_id, step_idx, ai_len, obs_len), trajs는 trajectories.parquet
    (traj_id, system_len, issue_msg_len). 출력: traj_id, step_idx, ctx.
    """
    base = trajs.select("traj_id", base=(pl.col("system_len") + pl.col("issue_msg_len")).cast(pl.Int64))
    return (
        steps.lazy()
        .select("traj_id", "step_idx", msg=(pl.col("ai_len") + pl.col("obs_len")).cast(pl.Int64))
        .join(base.lazy(), on="traj_id", how="inner")
        .sort("traj_id", "step_idx")  # join은 순서를 보존하지 않으므로 누적합 직전에 정렬
        .select(
            "traj_id",
            "step_idx",
            ctx=pl.col("base") + pl.col("msg").cum_sum().over("traj_id") - pl.col("msg"),
        )
        .collect()
    )


def tokens_after(costs: pl.DataFrame, n: int) -> pl.DataFrame:
    """궤적별 전체 맥락 합과 N스텝 뒤 맥락 합 (= 시점 N에서 끊으면 아끼는 양)."""
    return costs.group_by("traj_id").agg(
        tokens_total=pl.col("ctx").sum(),
        tokens_after=pl.col("ctx").filter(pl.col("step_idx") > n).sum(),
    )


def cost_curve(
    trajs: pl.DataFrame,
    preds: pl.DataFrame,
    n: int,
    costs: pl.DataFrame | None = None,
    cap: int | None = None,
) -> pl.DataFrame:
    """정책 "시점 N에서 p_fail >= tau면 종료"의 tau별 (성공 손실률, 절약 스텝 비율).

    행 하나 = tau 하나. tau는 p_fail의 고유값을 큰 쪽부터 훑고, 맨 앞에 예측기로는 아무것도
    끊지 않는 tau=inf 행을 둔다. 끊긴 궤적은 N스텝 뒤의 스텝을 아낀다.
    - lost_success = 끊긴 성공 궤적 / trajs의 전체 성공 궤적
    - saved_steps = 아낀 스텝 합 / trajs의 전체 스텝 (N 전에 끝난 궤적 포함)
    - saved_tokens (costs가 있을 때) = 아낀 맥락 합 / trajs의 전체 맥락 합
    - n_stopped, stop_rate = 예측기가 N에서 끊은 궤적 수와 진행 중 궤적 대비 비율
    고정 상한 K는 N=K에서 모든 궤적을 끊는 경우(p_fail이 모두 같음)와 같다.

    cap=K면 결합 정책 "스텝 > K면 종료 또는 시점 N에서 p_fail >= tau면 종료" (N < K).
    tau=inf 행이 고정 상한 K 그 자체이고, N에서 끊긴 궤적은 K에서 끊겼을 때보다
    min(n_steps, K) - N 스텝을 더 아낀다. K에서 이미 끊기는 성공 궤적은 손실을 두 번 세지 않는다.
    """
    if cap is not None and cap <= n:
        raise ValueError(f"cap K={cap}는 N={n}보다 커야 한다")
    k = cap if cap is not None else max(int(trajs["n_steps"].max()), n + 1)  # 상한 없음 = 아무도 안 끊기는 K
    pop = trajs.select(_columns(trajs)).with_columns(w=_weight(trajs))
    w = pl.col("w")
    if costs is not None:
        tok = costs.group_by("traj_id").agg(
            tokens_total=pl.col("ctx").sum(),
            after_n=pl.col("ctx").filter(pl.col("step_idx") > n).sum(),
            after_k=pl.col("ctx").filter(pl.col("step_idx") > k).sum(),
        )
        pop = pop.join(tok, on="traj_id", how="left").fill_null(0)
    capped = pl.col("n_steps") > k

    base = {
        "lost": (w * (pl.col("target") & capped)).sum(),
        "saved": (w * (pl.col("n_steps") - k).clip(lower_bound=0)).sum(),
        "succ": (w * pl.col("target")).sum(),
        "steps": (w * pl.col("n_steps")).sum(),
    }
    inc = {
        "tau": pl.col("p_fail"),
        "n_stopped": w.cum_sum(),
        "lost": (w * (pl.col("target") & ~capped)).cum_sum(),
        "saved": (w * (pl.min_horizontal("n_steps", pl.lit(k)) - n)).cum_sum(),
    }
    if costs is not None:
        base["saved_tok"] = (w * pl.col("after_k")).sum()
        base["tokens"] = (w * pl.col("tokens_total")).sum()
        inc["saved_tok"] = (w * (pl.col("after_n") - pl.col("after_k"))).cum_sum()
    b = pop.select(**base).row(0, named=True)

    a = align(trajs, preds, n).join(pop.drop(_columns(trajs)[1:]), on="traj_id", how="left")
    n_running = a["w"].sum()
    curve = a.sort("p_fail", descending=True).select(**inc).group_by("tau", maintain_order=True).last()
    start = pl.DataFrame({c: [float("inf") if c == "tau" else 0] for c in inc}, schema=curve.schema)
    out = [
        "tau",
        "n_stopped",
        (pl.col("n_stopped") / max(n_running, 1)).alias("stop_rate"),
        ((pl.col("lost") + b["lost"]) / b["succ"]).alias("lost_success"),
        ((pl.col("saved") + b["saved"]) / b["steps"]).alias("saved_steps"),
    ]
    if costs is not None:
        out.append(((pl.col("saved_tok") + b["saved_tok"]) / b["tokens"]).alias("saved_tokens"))
    return pl.concat([start, curve]).select(out)


def at_threshold(curve: pl.DataFrame, tau: float) -> dict:
    """고정한 tau를 적용했을 때의 행 (p_fail >= tau인 궤적을 모두 끊은 상태)."""
    return curve.filter(pl.col("tau") >= tau).row(-1, named=True)


def select_operating_point(
    trajs: pl.DataFrame,
    preds: pl.DataFrame,
    n: int,
    caps=(None,),
    max_loss: float = MAX_LOSS,
    by: str = "saved_steps",
    costs: pl.DataFrame | None = None,
) -> dict:
    """운영점 선택: 성공 손실 <= max_loss인 (K, tau) 중 절약(by)이 가장 큰 것.

    trajs·preds는 학습 이슈와 그 out-of-fold 예측이어야 한다. 테스트 예측으로 고르면 절약률이
    과대평가된다. 고른 값은 테스트에서 point_metrics(..., tau=tau, cap=cap)로 그대로 적용한다.
    - caps: 결합 정책의 상한 K 후보. None은 상한 없음 ("시점 N에서 p_fail >= tau"만)
    - tau=inf가 뽑히면 예측기를 쓰지 않는 것이 낫다는 뜻이다 (상한 K만, 또는 아무것도 안 끊음)
    - 동률이면 손실이 작은 것, 그다음 tau가 큰 것(덜 끊음), 그다음 K가 큰 것을 고른다
    출력: N, cap, 그리고 그 행의 tau, n_stopped, stop_rate, lost_success, saved_steps(, saved_tokens).
    """
    if by == "saved_tokens" and costs is None:
        raise ValueError("by='saved_tokens'는 costs가 필요하다")
    curves = {cap: cost_curve(trajs, preds, n, costs=costs, cap=cap) for cap in caps}
    return select_from_curves(curves, n, max_loss=max_loss, by=by)


def select_from_curves(curves: dict, n: int, max_loss: float = MAX_LOSS, by: str = "saved_steps") -> dict:
    """select_operating_point의 선택 단계. curves = {K 후보: cost_curve(..., cap=K)} (None = 상한 없음).

    곡선은 max_loss와 무관하므로, 같은 곡선에서 여러 max_loss·by로 고를 때 곡선을 다시 만들지 않으려고 따로 둔다.
    """
    if any(by not in c.columns for c in curves.values()):
        raise ValueError(f"곡선에 {by} 컬럼이 없음 (saved_tokens는 costs를 넣어 만든 곡선이 필요하다)")
    cand = pl.concat([
        c.filter(pl.col("lost_success") <= max_loss + 1e-12)  # 비율 계산의 부동소수 오차
        .with_columns(cap=pl.lit(cap, dtype=pl.Int64))
        for cap, c in curves.items()
    ])
    if cand.height == 0:
        raise ValueError(f"N={n}, K 후보 {list(curves)}: 성공 손실 <= {max_loss}인 운영점이 없음")
    best = cand.sort(
        [pl.col(by), pl.col("lost_success"), pl.col("tau"), pl.col("cap").fill_null(np.iinfo(np.int64).max)],
        descending=[True, False, True, True],
    ).row(0, named=True)
    return {"N": n, **best}


def fixed_cap_curve(trajs: pl.DataFrame, costs: pl.DataFrame | None = None, ks=None) -> pl.DataFrame:
    """B0 고정 상한 "스텝 > K면 종료"의 K별 (성공 손실률, 절약 스텝, 절약 맥락). 닫힌 식으로 한 번에 계산한다.

    cost_curve에 "N=K에서 진행 중 궤적 전부 p_fail=1"을 넣은 결과(그 곡선의 마지막 행)와 같다.
    - n_stopped, stop_rate = 상한에 걸린 궤적 수(n_steps > K)와 trajs 전체 대비 비율
    - lost_success, saved_steps, saved_tokens의 분모는 cost_curve와 같다
    ks: K 목록 (기본 1..최대 스텝). weight 컬럼이 있으면 궤적 가중치로 쓴다 (부트스트랩).
    """
    pop = trajs.select(_columns(trajs)).with_columns(w=_weight(trajs))
    n_steps = pop["n_steps"].to_numpy().astype(np.int64)
    w = pop["w"].to_numpy().astype(float)
    succ = pop["target"].to_numpy().astype(bool)
    ks = np.arange(1, n_steps.max() + 1) if ks is None else np.asarray(list(ks), dtype=np.int64)
    if (ks < 0).any():
        raise ValueError("K는 0 이상이어야 한다")
    size = int(max(n_steps.max(), ks.max())) + 2

    def above(x):  # above(x)[k] = sum_{n_steps > k} x
        return np.cumsum(np.bincount(n_steps, weights=x, minlength=size)[::-1])[::-1][1:]

    cnt, lost, steps = above(w), above(w * succ), above(w * n_steps)
    out = {
        "K": ks,
        "n_stopped": cnt[ks],
        "stop_rate": cnt[ks] / w.sum(),
        "lost_success": lost[ks] / (w * succ).sum(),
        "saved_steps": (steps[ks] - ks * cnt[ks]) / (w * n_steps).sum(),
    }
    if costs is not None:
        c = costs.join(pop.select("traj_id", "w"), on="traj_id", how="inner")
        step_idx = c["step_idx"].to_numpy().astype(np.int64)
        size_c = int(max(step_idx.max(initial=0), ks.max())) + 2
        tok = c["w"].to_numpy() * c["ctx"].to_numpy().astype(float)
        tok_above = np.cumsum(np.bincount(step_idx, weights=tok, minlength=size_c)[::-1])[::-1][1:]
        out["saved_tokens"] = tok_above[ks] / tok.sum()
    return pl.DataFrame(out).with_columns(pl.col("n_stopped").cast(pl.Int64))


def select_cap(cap_curve: pl.DataFrame, max_loss: float = MAX_LOSS) -> dict:
    """fixed_cap_curve에서 성공 손실 <= max_loss인 가장 작은 K (= 그 제약에서 절약이 가장 큰 고정 상한).

    학습 이슈 곡선에서 고르고 테스트에는 고른 K를 그대로 쓴다 (select_operating_point와 같은 원칙).
    """
    ok = cap_curve.filter(pl.col("lost_success") <= max_loss + 1e-12)
    if ok.height == 0:
        raise ValueError(f"성공 손실 <= {max_loss}인 K가 없음 (K {cap_curve['K'].min()}..{cap_curve['K'].max()})")
    return ok.sort("K").row(0, named=True)


def compress_costs(costs: pl.DataFrame, breakpoints) -> pl.DataFrame:
    """costs를 궤적마다 경계 구간 합으로 줄인다. 경계 b에서 "step_idx > b인 맥락 합"과 전체 합이 그대로 남는다.

    cost_curve(n=N, cap=K)는 경계 {N, K}의 이 합만 쓰므로, 같은 (N, K)를 수천 번 다시 계산하는
    부트스트랩에서 행 수를 줄이는 데 쓴다. 궤적마다 구간 (b_{j-1}, b_j] (마지막은 (b_max, ∞))의 스텝을
    한 행으로 합치고, 그 행의 step_idx는 구간 안 마지막 스텝이다 (궤적 길이를 넘지 않아서, 상한 없는
    cost_curve가 쓰는 "최대 스텝 뒤 = 0"도 그대로다).
    """
    bps = np.unique(np.asarray([b for b in breakpoints if b is not None], dtype=np.int64))
    if len(bps) == 0:
        raise ValueError("경계가 하나는 있어야 한다")
    bucket = np.searchsorted(bps, costs["step_idx"].to_numpy().astype(np.int64), side="left")
    return (
        costs.select("traj_id", "step_idx", "ctx")
        .with_columns(_bucket=pl.Series(bucket))
        .group_by("traj_id", "_bucket")
        .agg(pl.col("step_idx").max(), pl.col("ctx").sum())
        .sort("traj_id", "step_idx")
        .select("traj_id", "step_idx", "ctx")
    )


def point_metrics(
    trajs: pl.DataFrame,
    preds: pl.DataFrame,
    n: int,
    tau: float | None = None,
    costs: pl.DataFrame | None = None,
    cap: int | None = None,
) -> dict:
    """시점 N의 주요 지표. tau가 있으면 그 운영점(결합 정책이면 cap=K)의 손실·절약도 넣는다."""
    a = align(trajs, preds, n)
    d = discrimination(a["target"], a["p_fail"], a["weight"] if "weight" in a.columns else None)
    wi = within_issue_auroc(a)
    out = {
        "auroc": d["auroc"],
        "pr_auc": d["pr_auc"],
        "base_rate": d["base_rate"],
        "within_auroc": wi["pair_weighted"],
        "within_auroc_issue_mean": wi["issue_mean"],
    }
    if tau is not None:
        row = at_threshold(cost_curve(trajs, preds, n, costs=costs, cap=cap), tau)
        out.update({c: row[c] for c in ("stop_rate", "lost_success", "saved_steps", "saved_tokens") if c in row})
    return out


def bootstrap(trajs: pl.DataFrame, preds: pl.DataFrame, stat, n_boot: int = 1000, seed: int = io.SEED, level: float = 0.95) -> pl.DataFrame:
    """이슈 단위 클러스터 부트스트랩 백분위 구간.

    이슈를 복원 추출하고, c번 뽑힌 이슈의 궤적에 weight=c를 준 trajs로 stat을 다시 계산한다
    (같은 이슈의 궤적은 서로 독립이 아니므로 궤적이 아닌 이슈를 뽑는다). 궤적을 복제하지
    않으므로 traj_id로 붙는 costs를 그대로 쓸 수 있다.
    stat(trajs, preds) -> {이름: 값}, 예: lambda t, p: point_metrics(t, p, 10, tau=0.8, costs=costs)
    출력: stat, estimate(원래 데이터), lo, hi, n_boot. NaN이 나온 반복은 구간 계산에서 뺀다.
    """
    if "weight" in trajs.columns:
        raise ValueError("trajs에 이미 weight 컬럼이 있음")
    issues = trajs["instance_id"].unique().sort()
    rng = np.random.default_rng(seed)
    estimate = stat(trajs, preds)
    draws = []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, len(issues), len(issues)), minlength=len(issues))
        keep = counts > 0
        w = pl.DataFrame({"instance_id": issues.filter(pl.Series(keep)), "weight": counts[keep]})
        tb = trajs.join(w, on="instance_id", how="inner")
        draws.append(stat(tb, preds.join(tb.select("traj_id"), on="traj_id", how="semi")))
    q = [(1 - level) / 2, (1 + level) / 2]
    rows = []
    for name, value in estimate.items():
        v = np.array([d[name] for d in draws], dtype=float)
        lo, hi = np.nanquantile(v, q) if np.isfinite(v).any() else (np.nan, np.nan)
        rows.append({"stat": name, "estimate": value, "lo": float(lo), "hi": float(hi), "n_boot": n_boot})
    return pl.DataFrame(rows)
