"""Phase 4.5 운영점 선택 보강: 고정 상한 닫힌 식, K 선택, 곡선에서 고르기, 맥락 압축."""
import numpy as np
import polars as pl
import pytest

from afp import metrics


def trajs_frame(rows):
    return pl.DataFrame(rows, schema=metrics.TRAJ_COLUMNS, orient="row")


TRAJS = trajs_frame([
    ("t1", "i1", True, 4),
    ("t2", "i1", False, 10),
    ("t3", "i1", False, 6),
    ("t4", "i2", True, 8),
    ("t5", "i2", False, 12),
    ("t6", "i3", False, 2),
])


def preds_frame(n, mapping):
    return pl.DataFrame({"traj_id": list(mapping), "N": [n] * len(mapping), "p_fail": list(mapping.values())},
                        schema={"traj_id": pl.String, "N": pl.Int64, "p_fail": pl.Float64})


def random_population(seed=0, n_traj=300):
    rng = np.random.default_rng(seed)
    n_steps = rng.integers(0, 40, n_traj)
    trajs = pl.DataFrame({
        "traj_id": [f"t{i:04d}" for i in range(n_traj)],
        "instance_id": [f"i{i % 37}" for i in range(n_traj)],
        "target": rng.random(n_traj) < 0.3,
        "n_steps": n_steps,
    })
    costs = pl.DataFrame({
        "traj_id": [f"t{i:04d}" for i in range(n_traj) for _ in range(n_steps[i])],
        "step_idx": [s + 1 for i in range(n_traj) for s in range(n_steps[i])],
        "ctx": rng.integers(1, 1000, int(n_steps.sum())),
    })
    return trajs, costs, rng


# ---- 고정 상한 닫힌 식 ----

def test_fixed_cap_curve_by_hand():
    # K=5: 끊김 t2(10), t3(6), t4(8), t5(12) → 성공 t4 하나 손실 1/2, 절약 5+1+3+7=16 / 42
    row = metrics.fixed_cap_curve(TRAJS, ks=[5]).row(0, named=True)
    assert row["K"] == 5 and row["n_stopped"] == 4
    assert row["lost_success"] == pytest.approx(0.5)
    assert row["saved_steps"] == pytest.approx(16 / 42)
    assert row["stop_rate"] == pytest.approx(4 / 6)
    full = metrics.fixed_cap_curve(TRAJS)
    assert full["K"].to_list() == list(range(1, 13))
    assert full.filter(pl.col("K") == 12).row(0, named=True)["saved_steps"] == 0


def test_fixed_cap_curve_matches_cost_curve():
    trajs, costs, _ = random_population()
    curve = metrics.fixed_cap_curve(trajs, costs, ks=[1, 3, 7, 15, 25, 39, 45])
    for row in curve.iter_rows(named=True):
        k = row["K"]
        # 하네스: N=K에서 진행 중 궤적 전부 p_fail=1 → 마지막 행이 고정 상한 K
        alive = metrics.running(trajs, k)
        h = metrics.cost_curve(trajs, preds_frame(k, dict.fromkeys(alive["traj_id"], 1.0)), k, costs=costs).row(-1, named=True)
        for c in ("lost_success", "saved_steps", "saved_tokens"):
            assert row[c] == pytest.approx(h[c], abs=1e-12), (k, c)
        # 결합 정책의 tau=inf 행 = 고정 상한 K 그 자체
        if k > 3:
            alive3 = metrics.running(trajs, 3)
            inf_row = metrics.cost_curve(trajs, preds_frame(3, dict.fromkeys(alive3["traj_id"], 0.5)), 3, costs=costs, cap=k).row(0, named=True)
            for c in ("lost_success", "saved_steps", "saved_tokens"):
                assert row[c] == pytest.approx(inf_row[c], abs=1e-12), (k, c)


def test_fixed_cap_curve_uses_weights():
    trajs, costs, rng = random_population(seed=1)
    w = rng.integers(0, 4, trajs.height)
    weighted = metrics.fixed_cap_curve(trajs.with_columns(weight=pl.Series(w)), costs, ks=[5, 20])
    rep = trajs.with_columns(weight=pl.Series(w)).filter(pl.col("weight") > 0)
    rep = rep.select(pl.all().repeat_by("weight").explode()).drop("weight")
    rep = rep.with_columns(traj_id=pl.col("traj_id") + "_" + pl.int_range(pl.len()).over("traj_id").cast(pl.String))
    rep_costs = costs.join(rep.select("traj_id", src=pl.col("traj_id").str.split("_").list.first()), left_on="traj_id", right_on="src").select(
        pl.col("traj_id_right").alias("traj_id"), "step_idx", "ctx")
    plain = metrics.fixed_cap_curve(rep, rep_costs, ks=[5, 20])
    for c in ("lost_success", "saved_steps", "saved_tokens", "stop_rate"):
        assert weighted[c].to_list() == pytest.approx(plain[c].to_list(), abs=1e-12)


def test_select_cap_smallest_feasible_k():
    curve = metrics.fixed_cap_curve(TRAJS)
    # 성공 t1(4), t4(8): K<4 → 손실 1, 4<=K<8 → 0.5, K>=8 → 0
    assert metrics.select_cap(curve, max_loss=0.0)["K"] == 8
    assert metrics.select_cap(curve, max_loss=0.5)["K"] == 4
    assert metrics.select_cap(curve, max_loss=1.0)["K"] == 1
    with pytest.raises(ValueError, match="K가 없음"):
        metrics.select_cap(curve.filter(pl.col("K") < 8), max_loss=0.0)


# ---- 곡선에서 고르기 ----

def test_select_max_saving_under_loss():
    # N=5 진행 중: t2(F,10), t3(F,6), t4(S,8), t5(F,12). 성공 2개 → 하나 끊으면 손실 0.5
    preds = preds_frame(5, {"t2": 0.9, "t3": 0.9, "t4": 0.6, "t5": 0.2})
    curves = {None: metrics.cost_curve(TRAJS, preds, 5)}
    op = metrics.select_from_curves(curves, 5, max_loss=0.0)
    # 손실 0에서 고를 수 있는 것: tau=inf(절약 0), tau=0.9(t2·t3 → 5+1=6) → 0.9
    assert op["tau"] == 0.9 and op["saved_steps"] == pytest.approx(6 / 42) and op["cap"] is None
    # 손실 0.5까지 허용: tau=0.2(모두 끊음, 5+1+3+7=16)가 tau=0.6(9)보다 많이 아낌
    assert metrics.select_from_curves(curves, 5, max_loss=0.5)["tau"] == 0.2
    assert metrics.select_from_curves(curves, 5, max_loss=0.5) == metrics.select_operating_point(TRAJS, preds, 5, max_loss=0.5)


def test_select_tie_prefers_larger_tau():
    # 절약·손실이 같은 두 tau와 같은 곡선의 두 K 후보 → 직접 만든 곡선으로 동점 규칙을 본다
    curve = pl.DataFrame({
        "tau": [float("inf"), 0.9, 0.5], "n_stopped": [0, 1, 2], "stop_rate": [0.0, 0.5, 1.0],
        "lost_success": [0.0, 0.01, 0.01], "saved_steps": [0.0, 0.2, 0.2],
    })
    op = metrics.select_from_curves({None: curve, 30: curve}, 5, max_loss=0.05)
    assert op["tau"] == 0.9 and op["cap"] is None  # 큰 tau, 그다음 큰 K(None = 상한 없음)


def test_select_infeasible_returns_no_stop():
    # 모든 tau가 손실 > 0이면 tau=inf(아무것도 안 끊음)
    preds = preds_frame(5, {"t2": 0.1, "t3": 0.1, "t4": 0.9, "t5": 0.1})
    op = metrics.select_from_curves({None: metrics.cost_curve(TRAJS, preds, 5)}, 5, max_loss=0.0)
    assert op["tau"] == float("inf") and op["saved_steps"] == 0 and op["n_stopped"] == 0
    # 상한 자체가 손실을 넘으면 고를 행이 없다
    with pytest.raises(ValueError, match="운영점이 없음"):
        metrics.select_from_curves({6: metrics.cost_curve(TRAJS, preds, 5, cap=6)}, 5, max_loss=0.0)
    with pytest.raises(ValueError, match="saved_tokens"):
        metrics.select_from_curves({None: metrics.cost_curve(TRAJS, preds, 5)}, 5, by="saved_tokens")


def test_select_combined_inf_row_is_cap():
    # 쓸모없는 예측기(모두 같은 점수)로 결합 정책을 고르면 tau=inf + 고정 상한 K*가 된다
    trajs, costs, _ = random_population(seed=2, n_traj=500)
    n = 3
    preds = preds_frame(n, dict.fromkeys(metrics.running(trajs, n)["traj_id"], 0.5))
    k_star = metrics.select_cap(metrics.fixed_cap_curve(trajs), 0.05)
    op = metrics.select_operating_point(trajs, preds, n, caps=range(n + 1, 61), max_loss=0.05)
    assert op["tau"] == float("inf") and op["cap"] == k_star["K"]
    assert op["saved_steps"] == pytest.approx(k_star["saved_steps"])
    assert op["lost_success"] == pytest.approx(k_star["lost_success"])


# ---- 맥락 압축 ----

def test_compress_costs_preserves_cost_curve():
    trajs, costs, rng = random_population(seed=3)
    for n, cap, ref in ((3, None, 20), (5, 12, 34), (10, 25, 8)):
        alive = metrics.running(trajs, n)
        preds = preds_frame(n, dict(zip(alive["traj_id"], rng.random(alive.height))))
        small = metrics.compress_costs(costs, [n, cap, ref])
        assert small.height < costs.height
        a = metrics.cost_curve(trajs, preds, n, costs=costs, cap=cap)
        b = metrics.cost_curve(trajs, preds, n, costs=small, cap=cap)
        assert a["saved_tokens"].to_list() == pytest.approx(b["saved_tokens"].to_list(), abs=1e-12)
        assert metrics.fixed_cap_curve(trajs, costs, ks=[ref])["saved_tokens"][0] == pytest.approx(
            metrics.fixed_cap_curve(trajs, small, ks=[ref])["saved_tokens"][0], abs=1e-12)
    # 0스텝·짧은 궤적에서도 압축 행의 step_idx는 궤적 길이를 넘지 않는다
    joined = metrics.compress_costs(costs, [50]).join(trajs, on="traj_id")
    assert (joined["step_idx"] <= joined["n_steps"]).all()
