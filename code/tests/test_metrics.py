"""afp.metrics 단위 테스트. 손으로 답을 셀 수 있는 작은 궤적 집합으로 확인한다."""
import math

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import roc_auc_score

from afp import io, metrics


def trajs_frame(rows):
    """rows: (traj_id, instance_id, target, n_steps)"""
    return pl.DataFrame(rows, schema=metrics.TRAJ_COLUMNS, orient="row")


TRAJS = trajs_frame([
    ("t1", "i1", True, 4),
    ("t2", "i1", False, 10),
    ("t3", "i1", False, 6),
    ("t4", "i2", True, 8),
    ("t5", "i2", False, 12),
    ("t6", "i3", False, 2),  # N=3 전에 끝남
])


def preds_frame(n, mapping):
    return pl.DataFrame({"traj_id": list(mapping), "N": n, "p_fail": list(mapping.values())})


# ---- 평가 집합 로더 ----

@pytest.fixture
def processed(tmp_path, monkeypatch):
    """작은 trajectories·steps·splits parquet를 만들고 io 경로를 그쪽으로 돌린다."""
    traj = pl.DataFrame({
        "traj_id": ["a", "b", "c", "d", "e"],
        "instance_id": ["i1", "i1", "i2", "i3", "i3"],
        "target": [True, False, False, True, False],
        "n_steps": [2, 1, 0, 3, 1],
        "system_len": [100] * 5,
        "issue_msg_len": [10, 10, 20, 30, 30],
    })
    steps = pl.DataFrame({
        "traj_id": ["a", "a", "b", "d", "d", "d", "e"],
        "step_idx": [1, 2, 1, 1, 2, 3, 1],
        "ai_len": [1, 2, 3, 4, 5, 6, 7],
        "obs_len": [10, 20, 30, 40, 50, 60, 70],
    })
    splits = pl.DataFrame(
        {"traj_id": ["a", "b", "c", "d", "e"], "split": ["train", "train", "train", "test_same", "test_cross"], "fold": [0, 1, 0, -1, -1]},
        schema={"traj_id": pl.String, "split": pl.String, "fold": pl.Int8},
    )
    for name, df in [("TRAJECTORIES_PATH", traj), ("STEPS_PATH", steps), ("SPLITS_PATH", splits)]:
        path = tmp_path / f"{name}.parquet"
        df.write_parquet(path)
        monkeypatch.setattr(io, name, path)
    return traj, splits


def test_eval_set_returns_one_split_with_harness_columns(processed):
    train = metrics.eval_set("train")
    assert train.columns == [*metrics.TRAJ_COLUMNS, "fold"]
    assert train["traj_id"].to_list() == ["a", "b", "c"]  # 0스텝 궤적 c도 남는다 (비용 곡선 분모)
    assert train["fold"].to_list() == [0, 1, 0]
    assert metrics.eval_set("test_same")["traj_id"].to_list() == ["d"]

    with pytest.raises(ValueError, match="중 하나"):
        metrics.eval_set("test")  # test_same·test_cross를 합친 집합은 없다


def test_eval_set_stops_on_missing_files_or_trajectories(processed, monkeypatch, tmp_path):
    traj, splits = processed
    traj.filter(pl.col("traj_id") != "b").write_parquet(io.TRAJECTORIES_PATH)
    with pytest.raises(ValueError, match="1개가 trajectories에 없음"):
        metrics.eval_set("train")

    monkeypatch.setattr(io, "SPLITS_PATH", tmp_path / "none.parquet")
    with pytest.raises(FileNotFoundError, match="02_split.py"):
        metrics.eval_set("train")


def test_eval_costs_covers_only_given_trajectories(processed):
    costs = metrics.eval_costs(metrics.eval_set("train")).sort("traj_id", "step_idx")
    # a: 110, 110+11 / b: 110. c는 스텝이 없다. 다른 분할의 d·e는 빠진다
    assert costs.rows() == [("a", 1, 110), ("a", 2, 121), ("b", 1, 110)]


# ---- 3.1 정렬, AUROC, PR-AUC ----

def test_align_keeps_only_running_and_checks_coverage():
    preds = preds_frame(3, {"t1": 0.1, "t2": 0.9, "t3": 0.5, "t4": 0.2, "t5": 0.8})
    out = metrics.align(TRAJS, preds, 3)
    assert sorted(out["traj_id"]) == ["t1", "t2", "t3", "t4", "t5"]

    with pytest.raises(ValueError, match="예측 없음 1개"):
        metrics.align(TRAJS, preds.filter(pl.col("traj_id") != "t5"), 3)
    with pytest.raises(ValueError, match="대상 밖 예측 1개"):
        metrics.align(TRAJS, pl.concat([preds, preds_frame(3, {"t6": 0.5})]), 3)
    with pytest.raises(ValueError, match="중복"):
        metrics.align(TRAJS, pl.concat([preds, preds_frame(3, {"t1": 0.5})]), 3)
    with pytest.raises(ValueError, match="NaN"):
        metrics.align(TRAJS, preds.with_columns(p_fail=pl.lit(float("nan"))), 3)


def test_discrimination_perfect_constant_and_reversed():
    target = np.array([True, False, False, True, False])
    perfect = metrics.discrimination(target, (~target).astype(float))
    assert perfect["auroc"] == 1.0 and perfect["pr_auc"] == 1.0
    assert perfect["base_rate"] == pytest.approx(0.4) and perfect["n_success"] == 2

    const = metrics.discrimination(target, np.full(5, 0.3))
    assert const["auroc"] == 0.5
    assert const["pr_auc"] == pytest.approx(0.4)  # 상수 점수 → 성공률

    assert metrics.discrimination(target, target.astype(float))["auroc"] == 0.0


def test_discrimination_single_class_is_nan():
    out = metrics.discrimination([False, False], [0.1, 0.2])
    assert math.isnan(out["auroc"]) and math.isnan(out["pr_auc"])


# ---- 3.2 이슈 내 AUROC ----

def test_within_issue_auroc_counts_pairs_by_hand():
    # i1: 성공 t1, 실패 t2·t3 → 쌍 2개. t2(0.9) > t1(0.4) 맞음, t3(0.4) = t1 동점 0.5 → 1.5/2
    # i2: 성공 t4, 실패 t5 → 쌍 1개. t5(0.1) < t4(0.3) 틀림 → 0/1
    # i3: 실패만 → 제외
    preds = preds_frame(1, {"t1": 0.4, "t2": 0.9, "t3": 0.4, "t4": 0.3, "t5": 0.1, "t6": 0.2})
    out = metrics.within_issue_auroc(metrics.align(TRAJS, preds, 1))
    assert out["n_issues"] == 2 and out["n_traj"] == 5 and out["n_pairs"] == 3
    assert out["pair_weighted"] == pytest.approx(1.5 / 3)
    assert out["issue_mean"] == pytest.approx((0.75 + 0.0) / 2)


def test_within_issue_auroc_issue_level_score_is_half():
    # B1처럼 이슈 안에서 점수가 같으면 정의상 0.5
    preds = preds_frame(1, {"t1": 0.7, "t2": 0.7, "t3": 0.7, "t4": 0.2, "t5": 0.2, "t6": 0.9})
    out = metrics.within_issue_auroc(metrics.align(TRAJS, preds, 1))
    assert out["pair_weighted"] == 0.5 and out["issue_mean"] == 0.5


def test_within_issue_auroc_matches_sklearn_per_issue():
    rng = np.random.default_rng(0)
    rows = [(f"t{k}", f"i{k % 7}", bool(rng.random() < 0.4), 20) for k in range(200)]
    trajs = trajs_frame(rows)
    p = np.round(rng.random(200), 1)  # 동점이 생기게
    preds = pl.DataFrame({"traj_id": trajs["traj_id"], "N": 5, "p_fail": p})
    out = metrics.within_issue_auroc(metrics.align(trajs, preds, 5))

    df = trajs.with_columns(p_fail=pl.Series(p))
    aucs, num, den = [], 0.0, 0
    for _, g in df.group_by("instance_id"):
        y = ~g["target"].to_numpy()
        if 0 < y.sum() < len(y):
            a = roc_auc_score(y, g["p_fail"].to_numpy())
            pairs = int(y.sum() * (~y).sum())
            aucs.append(a)
            num, den = num + a * pairs, den + pairs
    assert out["issue_mean"] == pytest.approx(np.mean(aucs))
    assert out["pair_weighted"] == pytest.approx(num / den)


def test_within_issue_auroc_no_mixed_issue_is_nan():
    preds = preds_frame(9, {"t2": 0.5, "t5": 0.5})
    out = metrics.within_issue_auroc(metrics.align(TRAJS, preds, 9))
    assert out["n_issues"] == 0 and math.isnan(out["pair_weighted"])


# ---- 3.3 비용 곡선 ----

def test_cost_curve_by_hand():
    # N=5 진행 중: t2(n=10, 실패), t3(6, 실패), t4(8, 성공), t5(12, 실패). 전체 성공 2, 전체 스텝 42
    preds = preds_frame(5, {"t2": 0.9, "t3": 0.6, "t4": 0.6, "t5": 0.2})
    curve = metrics.cost_curve(TRAJS, preds, 5)
    assert curve["tau"].to_list() == [float("inf"), 0.9, 0.6, 0.2]
    assert curve["n_stopped"].to_list() == [0, 1, 3, 4]  # 동점 t3·t4는 함께 끊김
    assert curve["stop_rate"].to_list() == pytest.approx([0, 0.25, 0.75, 1])
    assert curve["lost_success"].to_list() == pytest.approx([0, 0, 0.5, 0.5])
    # 아끼는 스텝: t2 5, t3 1, t4 3, t5 7
    assert curve["saved_steps"].to_list() == pytest.approx([0, 5 / 42, 9 / 42, 16 / 42])


def test_cost_curve_stop_all_equals_fixed_cap():
    rng = np.random.default_rng(1)
    n_steps = rng.integers(0, 80, 500)
    target = rng.random(500) < 0.2
    trajs = trajs_frame([(f"t{k}", f"i{k % 30}", bool(target[k]), int(n_steps[k])) for k in range(500)])
    for k in (5, 35, 50):
        alive = metrics.running(trajs, k)
        preds = pl.DataFrame({"traj_id": alive["traj_id"], "N": k, "p_fail": 1.0})
        last = metrics.cost_curve(trajs, preds, k).row(-1, named=True)
        assert last["lost_success"] == pytest.approx((n_steps[target] > k).mean())
        assert last["saved_steps"] == pytest.approx(np.clip(n_steps - k, 0, None).sum() / n_steps.sum())


def test_cost_curve_is_monotone():
    rng = np.random.default_rng(2)
    trajs = trajs_frame([(f"t{k}", f"i{k % 10}", bool(rng.random() < 0.3), int(rng.integers(1, 60))) for k in range(300)])
    alive = metrics.running(trajs, 10)
    preds = pl.DataFrame({"traj_id": alive["traj_id"], "N": 10, "p_fail": np.round(rng.random(alive.height), 2)})
    curve = metrics.cost_curve(trajs, preds, 10)
    for col in ("n_stopped", "lost_success", "saved_steps"):
        assert (curve[col].diff().drop_nulls() >= 0).all(), col
    assert (curve["tau"].diff().drop_nulls() < 0).all()
    assert curve["n_stopped"][-1] == alive.height


# ---- 3.4 절약 토큰 근사 ----

def test_context_costs_by_hand():
    trajs = pl.DataFrame({"traj_id": ["a", "b"], "system_len": [100, 100], "issue_msg_len": [20, 30]})
    # a: 3스텝 (ai, obs) = (5, 10), (7, 3), (4, 0). b: 0스텝이라 steps에 없음
    steps = pl.DataFrame({"traj_id": ["a"] * 3, "step_idx": [3, 1, 2], "ai_len": [4, 5, 7], "obs_len": [0, 10, 3]})
    costs = metrics.context_costs(steps, trajs).sort("step_idx")
    assert costs["ctx"].to_list() == [120, 135, 145]

    after = metrics.tokens_after(costs, 1).row(0, named=True)
    assert after == {"traj_id": "a", "tokens_total": 400, "tokens_after": 280}


def test_cost_curve_saved_tokens():
    trajs = trajs_frame([("a", "i1", False, 3), ("b", "i1", True, 2), ("c", "i2", False, 1)])
    costs = pl.DataFrame({
        "traj_id": ["a", "a", "a", "b", "b", "c"],
        "step_idx": [1, 2, 3, 1, 2, 1],
        "ctx": [10, 20, 30, 10, 15, 10],
    })
    # 전체 맥락 95. N=1에서 a를 끊으면 50, b까지 끊으면 65를 아낀다
    preds = preds_frame(1, {"a": 0.8, "b": 0.3})
    curve = metrics.cost_curve(trajs, preds, 1, costs=costs)
    assert curve["saved_tokens"].to_list() == pytest.approx([0, 50 / 95, 65 / 95])
    assert curve["saved_steps"].to_list() == pytest.approx([0, 2 / 6, 3 / 6])
    assert "saved_tokens" not in metrics.cost_curve(trajs, preds, 1).columns


# ---- 3.6 결합 정책 ----

def test_cost_curve_with_cap_by_hand():
    # N=5, K=9. K에서 끊기는 궤적: t2(10, 실패) 1스텝, t5(12, 실패) 3스텝 → 기본 절약 4/42
    preds = preds_frame(5, {"t2": 0.9, "t3": 0.6, "t4": 0.6, "t5": 0.2})
    curve = metrics.cost_curve(TRAJS, preds, 5, cap=9)
    assert curve["n_stopped"].to_list() == [0, 1, 3, 4]
    assert curve["lost_success"].to_list() == pytest.approx([0, 0, 0.5, 0.5])
    # N에서 끊으면 추가로: t2 min(10,9)-5=4, t3 1, t4 3, t5 4
    assert curve["saved_steps"].to_list() == pytest.approx([4 / 42, 8 / 42, 12 / 42, 16 / 42])

    with pytest.raises(ValueError, match="커야"):
        metrics.cost_curve(TRAJS, preds, 5, cap=5)


def simulate_policy(n_steps, target, p_fail, ctx, n, tau, k):
    """궤적마다 정책을 그대로 적용해 본 참조 구현."""
    lost = saved = saved_tok = 0
    for i in range(len(n_steps)):
        stop = None
        if n_steps[i] > n and p_fail[i] >= tau:
            stop = n
        elif k is not None and n_steps[i] > k:
            stop = k
        if stop is not None:
            lost += target[i]
            saved += n_steps[i] - stop
            saved_tok += sum(ctx[i][stop:])
    total_tok = sum(sum(c) for c in ctx)
    return lost / target.sum(), saved / n_steps.sum(), saved_tok / total_tok


@pytest.mark.parametrize("n,k", [(5, None), (5, 20), (10, 11), (3, 40)])
def test_cost_curve_matches_simulation(n, k):
    rng = np.random.default_rng(n * 100 + (k or 0))
    m = 300
    n_steps = rng.integers(0, 50, m)
    target = rng.random(m) < 0.3
    ctx = [list(rng.integers(1, 100, s)) for s in n_steps]
    trajs = trajs_frame([(f"t{i}", f"i{i % 20}", bool(target[i]), int(n_steps[i])) for i in range(m)])
    costs = pl.DataFrame(
        [(f"t{i}", s + 1, int(c)) for i in range(m) for s, c in enumerate(ctx[i])],
        schema=["traj_id", "step_idx", "ctx"], orient="row",
    )
    alive = n_steps > n
    p_fail = np.round(rng.random(m), 1)
    preds = pl.DataFrame({"traj_id": trajs["traj_id"].filter(alive), "N": n, "p_fail": p_fail[alive]})
    curve = metrics.cost_curve(trajs, preds, n, costs=costs, cap=k)
    for tau in (float("inf"), 0.9, 0.5, 0.15, 0.0):
        row = metrics.at_threshold(curve, tau)
        lost, saved, saved_tok = simulate_policy(n_steps, target, p_fail, ctx, n, tau, k)
        assert row["lost_success"] == pytest.approx(lost), tau
        assert row["saved_steps"] == pytest.approx(saved), tau
        assert row["saved_tokens"] == pytest.approx(saved_tok), tau


def test_at_threshold_picks_all_at_or_above_tau():
    preds = preds_frame(5, {"t2": 0.9, "t3": 0.6, "t4": 0.6, "t5": 0.2})
    curve = metrics.cost_curve(TRAJS, preds, 5)
    assert metrics.at_threshold(curve, 0.95)["n_stopped"] == 0
    assert metrics.at_threshold(curve, 0.6)["n_stopped"] == 3
    assert metrics.at_threshold(curve, 0.5)["n_stopped"] == 3
    assert metrics.at_threshold(curve, 0.0)["n_stopped"] == 4


# ---- 3.5 운영점 선택 ----

def test_select_operating_point_by_hand():
    # N=5 곡선: tau inf/0.9/0.6/0.2 → 손실 0/0/0.5/0.5, 절약 0/5/9/16 (분모 42)
    preds = preds_frame(5, {"t2": 0.9, "t3": 0.6, "t4": 0.6, "t5": 0.2})
    op = metrics.select_operating_point(TRAJS, preds, 5)
    assert (op["N"], op["cap"], op["tau"], op["n_stopped"]) == (5, None, 0.9, 1)
    assert op["saved_steps"] == pytest.approx(5 / 42)

    # 손실 0.5까지 허용하면 0.6과 0.2 중 절약이 큰 0.2
    assert metrics.select_operating_point(TRAJS, preds, 5, max_loss=0.5)["tau"] == 0.2

    # K=9를 더하면 tau=0.9 + K=9가 8/42로 가장 많이 아낀다 (손실 0)
    op = metrics.select_operating_point(TRAJS, preds, 5, caps=(None, 9))
    assert (op["cap"], op["tau"], op["lost_success"]) == (9, 0.9, 0)
    assert op["saved_steps"] == pytest.approx(8 / 42)


def test_select_operating_point_prefers_no_predictor_when_nothing_fits():
    # 모든 예측이 성공 궤적 t4를 같이 끊으면 손실 0.5라서, 예측기를 쓰지 않는 tau=inf만 남는다
    preds = preds_frame(5, {"t2": 0.6, "t3": 0.6, "t4": 0.6, "t5": 0.6})
    op = metrics.select_operating_point(TRAJS, preds, 5, caps=(None, 9))
    assert (op["cap"], op["tau"], op["n_stopped"]) == (9, float("inf"), 0)

    # K=6은 t4(8스텝, 성공)를 끊어 상한만으로 손실 0.5 → 후보 없음
    with pytest.raises(ValueError, match="운영점이 없음"):
        metrics.select_operating_point(TRAJS, preds, 5, caps=(6,))
    with pytest.raises(ValueError, match="costs"):
        metrics.select_operating_point(TRAJS, preds, 5, by="saved_tokens")


@pytest.mark.parametrize("by", ["saved_steps", "saved_tokens"])
def test_select_operating_point_matches_brute_force(by):
    rng = np.random.default_rng(11)
    m, n, caps = 400, 5, (None, 15, 25, 40)
    n_steps = rng.integers(0, 50, m)
    target = rng.random(m) < 0.3
    ctx = [list(rng.integers(1, 100, s)) for s in n_steps]
    trajs = trajs_frame([(f"t{i}", f"i{i % 20}", bool(target[i]), int(n_steps[i])) for i in range(m)])
    costs = pl.DataFrame(
        [(f"t{i}", s + 1, int(c)) for i in range(m) for s, c in enumerate(ctx[i])],
        schema=["traj_id", "step_idx", "ctx"], orient="row",
    )
    alive = n_steps > n
    # 실패일수록 p_fail이 높은 약한 예측기 (무작위면 손실 5% 안에서 고를 게 거의 없다)
    p_fail = np.round(np.clip(0.5 * ~target + 0.6 * rng.random(m), 0, 1), 2)
    preds = pl.DataFrame({"traj_id": trajs["traj_id"].filter(alive), "N": n, "p_fail": p_fail[alive]})

    col = {"saved_steps": 1, "saved_tokens": 2}[by]
    best = max(
        (sim[col], -sim[0])
        for k in caps
        for tau in [float("inf"), *np.unique(p_fail[alive])]
        if (sim := simulate_policy(n_steps, target, p_fail, ctx, n, tau, k))[0] <= metrics.MAX_LOSS
    )
    op = metrics.select_operating_point(trajs, preds, n, caps=caps, by=by, costs=costs)
    assert op[by] == pytest.approx(best[0])
    assert op["lost_success"] == pytest.approx(-best[1])
    assert op["n_stopped"] > 0  # 예측기를 실제로 쓰는 운영점이 뽑혀야 이 테스트가 의미 있다

    # 고른 운영점을 그대로 다시 적용하면 같은 값 (테스트 쪽 적용 경로)
    again = metrics.point_metrics(trajs, preds, n, tau=op["tau"], costs=costs, cap=op["cap"])
    assert again["lost_success"] == pytest.approx(op["lost_success"])
    assert again[by] == pytest.approx(op[by])


# ---- 3.7 가중치, 이슈 단위 부트스트랩 ----

def random_setup(m=400, n_issues=25, seed=3):
    rng = np.random.default_rng(seed)
    n_steps = rng.integers(0, 50, m)
    issue = rng.integers(0, n_issues, m)
    # 이슈 난이도가 있게: 이슈마다 성공률이 다름
    target = rng.random(m) < np.linspace(0.05, 0.6, n_issues)[issue]
    trajs = trajs_frame([(f"t{i}", f"i{issue[i]}", bool(target[i]), int(n_steps[i])) for i in range(m)])
    costs = pl.DataFrame(
        [(f"t{i}", s + 1, int(c)) for i in range(m) for s, c in enumerate(rng.integers(1, 100, n_steps[i]))],
        schema=["traj_id", "step_idx", "ctx"], orient="row",
    )
    preds = pl.concat([
        metrics.running(trajs, n).select("traj_id", N=pl.lit(n), p_fail=pl.Series(np.round(rng.random(int((n_steps > n).sum())), 2)))
        for n in (5, 10)
    ])
    return trajs, preds, costs


def test_weights_equal_physical_duplication():
    trajs, preds, costs = random_setup()
    counts = {f"i{j}": c for j, c in enumerate([0, 1, 2, 3] * 7) if j < 25}
    w = pl.DataFrame({"instance_id": list(counts), "weight": list(counts.values())}).filter(pl.col("weight") > 0)
    weighted = trajs.join(w, on="instance_id")
    wpreds = preds.join(weighted.select("traj_id"), on="traj_id", how="semi")

    # 이슈를 c번 복제하고 ID를 새로 붙인 데이터
    dup = weighted.with_columns(rep=pl.int_ranges(0, pl.col("weight"))).explode("rep").with_columns(
        traj_id=pl.col("traj_id") + "#" + pl.col("rep").cast(pl.String),
        instance_id=pl.col("instance_id") + "#" + pl.col("rep").cast(pl.String),
        src=pl.col("traj_id"),
    )
    dpreds = wpreds.join(dup.select("src", "traj_id"), left_on="traj_id", right_on="src").select(traj_id="traj_id_right", N="N", p_fail="p_fail")
    dcosts = costs.join(dup.select("src", "traj_id"), left_on="traj_id", right_on="src").select(traj_id="traj_id_right", step_idx="step_idx", ctx="ctx")
    dup = dup.select(metrics.TRAJ_COLUMNS)

    for n, tau, cap in [(5, 0.5, None), (10, 0.3, 30)]:
        got = metrics.point_metrics(weighted, wpreds, n, tau=tau, costs=costs, cap=cap)
        want = metrics.point_metrics(dup, dpreds, n, tau=tau, costs=dcosts, cap=cap)
        assert got == pytest.approx(want), (n, tau, cap)
        wi_got = metrics.within_issue_auroc(metrics.align(weighted, wpreds, n))
        wi_want = metrics.within_issue_auroc(metrics.align(dup, dpreds, n))
        assert wi_got == pytest.approx(wi_want)


def test_bootstrap_reproducible_and_brackets_estimate():
    trajs, preds, costs = random_setup()
    stat = lambda t, p: metrics.point_metrics(t, p, 5, tau=0.5, costs=costs, cap=30)
    a = metrics.bootstrap(trajs, preds, stat, n_boot=50, seed=7)
    b = metrics.bootstrap(trajs, preds, stat, n_boot=50, seed=7)
    assert a.equals(b)
    assert set(a["stat"]) == {"auroc", "pr_auc", "base_rate", "within_auroc", "within_auroc_issue_mean",
                              "stop_rate", "lost_success", "saved_steps", "saved_tokens"}
    assert (a["lo"] <= a["hi"]).all()
    # 무작위 예측이라 AUROC 구간은 0.5를 포함해야 한다
    auroc = a.filter(pl.col("stat") == "auroc").row(0, named=True)
    assert auroc["lo"] < 0.5 < auroc["hi"]


def test_bootstrap_perfect_predictor_has_degenerate_interval():
    trajs, _, _ = random_setup()
    alive = metrics.running(trajs, 5)
    preds = alive.select("traj_id", N=pl.lit(5), p_fail=(~pl.col("target")).cast(pl.Float64))
    out = metrics.bootstrap(trajs, preds, lambda t, p: metrics.point_metrics(t, p, 5), n_boot=30)
    auroc = out.filter(pl.col("stat") == "auroc").row(0, named=True)
    assert auroc["estimate"] == auroc["lo"] == auroc["hi"] == 1.0

    with pytest.raises(ValueError, match="weight"):
        metrics.bootstrap(trajs.with_columns(weight=pl.lit(1)), preds, lambda t, p: {}, n_boot=1)


# ---- 실제 데이터 (data/processed에 Phase 1·2 결과가 있을 때만) ----

@pytest.mark.skipif(
    not (io.TRAJECTORIES_PATH.exists() and io.STEPS_PATH.exists() and io.SPLITS_PATH.exists()),
    reason="data/processed에 trajectories·steps·splits가 없음 (01_parse.py, 02_split.py 실행 필요)",
)
def test_real_eval_sets_and_fixed_cap_baseline():
    sizes = {"train": 62_460, "test_same": 12_332, "test_cross": 5_244}
    # K=35 고정 상한 (성공 손실, 절약 스텝). PLAN.md Phase 3 결과의 분할별 기준값
    k35 = {"train": (0.043, 0.242), "test_same": (0.045, 0.218), "test_cross": (0.124, 0.325)}
    for name, size in sizes.items():
        trajs = metrics.eval_set(name)
        assert trajs.height == size
        alive = metrics.running(trajs, 35)
        preds = alive.select("traj_id", N=pl.lit(35), p_fail=pl.lit(1.0))
        row = metrics.cost_curve(trajs, preds, 35).row(-1, named=True)
        assert (row["lost_success"], row["saved_steps"]) == pytest.approx(k35[name], abs=5e-4), name
    assert set(metrics.eval_costs(metrics.eval_set("test_cross"))["traj_id"]) <= set(metrics.eval_set("test_cross")["traj_id"])
