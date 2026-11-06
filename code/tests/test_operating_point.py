"""Phase 4.5 운영점 선택 보강: 고정 상한 닫힌 식, K 선택, 곡선에서 고르기, 맥락 압축."""
import inspect

import numpy as np
import polars as pl
import pytest

from afp import evaluate, metrics


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


def test_loss_criteria_values():
    # 회의 2026-11-05 §4.1: 선택 3.5%, 판정 5% (고정값). 민감도 표에 두 값이 모두 있다
    assert metrics.SELECT_MAX_LOSS == 0.035 and metrics.JUDGE_MAX_LOSS == 0.05
    assert {metrics.SELECT_MAX_LOSS, metrics.JUDGE_MAX_LOSS} <= set(evaluate.MAX_LOSS_GRID)


def test_selection_defaults_to_select_criterion():
    # (a) 선택은 3.5%를 쓴다: 손실 4%(<= 5%, > 3.5%)인 K는 기본값으로는 못 고르고 3%인 K를 고른다
    cap_curve = pl.DataFrame({"K": [10, 20, 30], "n_stopped": [3, 2, 1], "stop_rate": [0.3, 0.2, 0.1],
                              "lost_success": [0.06, 0.04, 0.03], "saved_steps": [0.4, 0.3, 0.2]})
    assert metrics.select_cap(cap_curve)["K"] == 30
    assert metrics.select_cap(cap_curve, metrics.JUDGE_MAX_LOSS)["K"] == 20
    # 경계: 정확히 3.5%는 고를 수 있다
    assert metrics.select_cap(cap_curve.with_columns(lost_success=pl.Series([0.06, 0.035, 0.03])))["K"] == 20
    # 운영점 (τ, K)도 같은 기본값: 손실 4%인 τ=0.5 대신 3%인 τ=0.9
    curve = pl.DataFrame({
        "tau": [float("inf"), 0.9, 0.5], "n_stopped": [0, 1, 2], "stop_rate": [0.0, 0.5, 1.0],
        "lost_success": [0.0, 0.03, 0.04], "saved_steps": [0.0, 0.1, 0.2],
    })
    assert metrics.select_from_curves({None: curve}, 5)["tau"] == 0.9
    assert metrics.select_from_curves({None: curve}, 5, max_loss=metrics.JUDGE_MAX_LOSS)["tau"] == 0.5
    # 이전성 분석(transfer_rows)도 같은 기본값을 쓴다
    assert inspect.signature(evaluate.transfer_rows).parameters["max_loss"].default == metrics.SELECT_MAX_LOSS


def test_judge_uses_ci_upper_not_point():
    # (b) 판정은 구간 상한: 점추정 4% <= 5%여도 상한 5.5%면 불충족
    row = {"split": "test", "lost_success": 0.04, "lost_success_lo": 0.025, "lost_success_hi": 0.055}
    assert evaluate.loss_flags(row) == {"meets_select_point": None, "meets_judge_ci_hi": False}
    assert evaluate.loss_flags({**row, "lost_success_hi": 0.05})["meets_judge_ci_hi"] is True
    # 점추정이 5%를 넘어도 판정 값은 상한만 본다 (점추정은 판정 함수에 들어가지 않음)
    assert evaluate.loss_flags({**row, "lost_success": 0.051, "lost_success_hi": 0.049})["meets_judge_ci_hi"] is True
    assert metrics.meets_judge_loss(0.0501) is False
    with pytest.raises(ValueError, match="구간 상한"):
        metrics.meets_judge_loss(float("nan"))
    # 구간이 없는 테스트 행(부트스트랩 생략)은 판정하지 않는다
    assert evaluate.loss_flags({"split": "test", "lost_success": 0.01})["meets_judge_ci_hi"] is None
    # train 행은 구간이 없어 점추정 <= 선택 기준(3.5%)으로만 표시한다
    assert evaluate.loss_flags({"split": "train", "lost_success": 0.035}) == {"meets_select_point": True, "meets_judge_ci_hi": None}
    assert evaluate.loss_flags({"split": "train", "lost_success": 0.04})["meets_select_point"] is False


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


# ---- 운영점 이전성 (회의 2026-11-05 §4.3) ----

def transfer_setup(seed=4, n_traj=900, n=5):
    """train(fold 0..4) + test. p_fail은 실패와 약하게 상관된 점수."""
    rng = np.random.default_rng(seed)
    issue = np.arange(n_traj) % 90
    trajs = pl.DataFrame({
        "traj_id": [f"t{i:04d}" for i in range(n_traj)],
        "instance_id": [f"i{k}" for k in issue],
        "target": rng.random(n_traj) < 0.3,
        "n_steps": rng.integers(0, 40, n_traj),
    })
    fold = np.where(issue < 75, issue % 5, -1)  # 이슈 75..89 = test
    trajs = trajs.with_columns(fold=pl.Series(fold, dtype=pl.Int8))
    alive = metrics.running(trajs, n)
    p = np.clip(0.5 + 0.3 * (~alive["target"].to_numpy()) + rng.normal(0, 0.3, alive.height), 0, 1)
    preds = preds_frame(n, dict(zip(alive["traj_id"], p)))
    return trajs.filter(pl.col("fold") >= 0), trajs.filter(pl.col("fold") < 0), preds


def test_transfer_rows_match_selection_on_other_folds():
    n = 5
    train, test, preds = transfer_setup(n=n)
    full = metrics.select_operating_point(train, evaluate.split_preds(preds, train, n), n, max_loss=0.05)
    rows = evaluate.transfer_rows(train, evaluate.split_preds(preds, train, n), test, evaluate.split_preds(preds, test, n),
                                  n, "single", full, max_loss=0.05)
    assert len(rows) == 2 * 5 + 1
    by = {(r["source"], r["target"]): r for r in rows}
    for k in range(5):
        src = train.filter(pl.col("fold") != k)
        held = train.filter(pl.col("fold") == k)
        op = metrics.select_operating_point(src, evaluate.split_preds(preds, src, n), n, max_loss=0.05)
        r = by[(f"train-fold{k}", f"fold{k}")]
        assert r["tau"] == op["tau"] and r["src_stop_rate"] == op["stop_rate"] and r["src_lost_success"] == op["lost_success"]
        row = metrics.at_threshold(metrics.cost_curve(held, evaluate.split_preds(preds, held, n), n), op["tau"])
        assert r["dst_stop_rate"] == row["stop_rate"] and r["dst_lost_success"] == row["lost_success"]
        assert by[(f"train-fold{k}", "test")]["tau"] == op["tau"]  # 같은 운영점을 test에도
    r = by[("train", "test")]
    assert r["tau"] == full["tau"] and r["src_stop_rate"] == full["stop_rate"]
    row = metrics.at_threshold(metrics.cost_curve(test, evaluate.split_preds(preds, test, n), n), full["tau"])
    assert r["dst_stop_rate"] == row["stop_rate"]

    # 같은 운영점을 fold k 모델 점수로 채점한 test에도 적용 (여기서는 점수를 0.1 내려 정지율이 줄어드는지)
    shifted = [evaluate.split_preds(preds, test, n).with_columns(pl.col("p_fail") - 0.1)] * 5
    rows2 = evaluate.transfer_rows(train, evaluate.split_preds(preds, train, n), test, evaluate.split_preds(preds, test, n),
                                   n, "single", full, max_loss=0.05, test_fold_preds=shifted)
    assert len(rows2) == 3 * 5 + 1
    for k in range(5):
        a = next(r for r in rows2 if r["source"] == f"train-fold{k}" and r["target"] == "test")
        b = next(r for r in rows2 if r["source"] == f"train-fold{k}" and r["target"] == "test_fold_model")
        assert b["dst_stop_rate"] <= a["dst_stop_rate"]

    s = evaluate.transfer_summary(pl.DataFrame([{"model": "m", "policy": "single", "N": n, **r} for r in rows2]))
    assert s.height == 1
    fold_ratios = [by[(f"train-fold{k}", f"fold{k}")]["dst_stop_rate"] / by[(f"train-fold{k}", f"fold{k}")]["src_stop_rate"] for k in range(5)]
    assert s["fold_stop_ratio_min"][0] == pytest.approx(min(fold_ratios))
    assert s["fold_stop_ratio_max"][0] == pytest.approx(max(fold_ratios))
    assert s["test_from_train_stop_ratio_min"][0] == pytest.approx(r["dst_stop_rate"] / r["src_stop_rate"])


def test_transfer_rows_fixed_cap():
    train, test, _ = transfer_setup()
    k = metrics.select_cap(metrics.fixed_cap_curve(train), 0.05)
    full = {"tau": None, "cap": int(k["K"]), **evaluate.cap_point(train, int(k["K"]), None)}
    rows = evaluate.transfer_rows(train, None, test, None, None, "cap", full, max_loss=0.05)
    assert len(rows) == 11
    for r in rows:
        assert r["tau"] is None and r["src_lost_success"] <= 0.05 + 1e-12
    for kf in range(5):
        r = next(x for x in rows if x["target"] == f"fold{kf}")
        src = train.filter(pl.col("fold") != kf)
        assert r["cap"] == metrics.select_cap(metrics.fixed_cap_curve(src), 0.05)["K"]
        assert r["dst_lost_success"] == evaluate.cap_point(train.filter(pl.col("fold") == kf), r["cap"], None)["lost_success"]


# ---- fold 모델별 테스트 채점 (P4-D6) ----

def fold_setup(seed=5, n_folds=5, n=5):
    trajs, costs, rng = random_population(seed=seed, n_traj=400)
    alive = metrics.running(trajs, n)
    base = rng.random(alive.height)
    # fold마다 다른 점수 (실패 쪽으로 약하게 치우침), 평균 점수 = p_fail
    fold_scores = np.clip(base[None, :] * 0.5 + 0.5 * rng.random((n_folds, alive.height)) * (~alive["target"].to_numpy() + 0.5), 0, 1).round(3)
    long = pl.concat([alive.select("traj_id", N=pl.lit(n, dtype=pl.Int64), fold=pl.lit(k, dtype=pl.Int8), p_fail=pl.Series(fold_scores[k]))
                      for k in range(n_folds)])
    mean = alive.select("traj_id", N=pl.lit(n, dtype=pl.Int64), p_fail=pl.Series(fold_scores.mean(axis=0)))
    return trajs, costs, n, long, mean


def test_test_point_averages_fold_results():
    trajs, costs, n, long, mean = fold_setup()
    wide = evaluate.attach_fold_scores(mean, long)
    assert evaluate.fold_columns(wide) == [f"p_fail_fold{k}" for k in range(5)]
    for tau, cap in ((0.6, None), (0.55, 20), (float("inf"), 15)):
        got = evaluate.test_point(trajs, wide, n, tau, cap, costs)
        per = [evaluate.apply_point(trajs, long.filter(pl.col("fold") == k).drop("fold"), n, tau, cap, costs) for k in range(5)]
        for c in evaluate.POINT_COLUMNS:
            vals = [p[c] for p in per]
            assert got[c] == pytest.approx(np.mean(vals), abs=1e-15)
            assert got[f"{c}_fold_min"] == min(vals) and got[f"{c}_fold_max"] == max(vals)
        # 평균 점수에 τ를 적용한 것과는 다르다 (P4-D6이 바꾸는 것)
        assert set(evaluate.test_point(trajs, mean, n, tau, cap, costs)) == set(evaluate.POINT_COLUMNS)
    # fold 점수가 모두 같으면 평균 점수 적용과 같다
    same = long.with_columns(p_fail=pl.col("traj_id").replace_strict(dict(mean.select("traj_id", "p_fail").rows())))
    w2 = evaluate.attach_fold_scores(mean, same)
    a, b = evaluate.test_point(trajs, w2, n, 0.6, 20, costs, spread=False), evaluate.apply_point(trajs, mean, n, 0.6, 20, costs)
    assert a == pytest.approx(b, abs=1e-15)


def test_attach_fold_scores_requires_every_row():
    trajs, costs, n, long, mean = fold_setup()
    with pytest.raises(ValueError, match="fold 점수가 없는"):
        evaluate.attach_fold_scores(mean, long.filter(~((pl.col("fold") == 3) & (pl.col("traj_id") == mean["traj_id"][0]))))


def test_bootstrap_averages_folds_within_each_resample():
    # 구간 = 반복마다 (같은 재표본에서) 5개 결과를 평균한 값의 구간. 평균 점수나 fold별 구간의 평균이 아니다
    trajs, costs, n, long, mean = fold_setup()
    wide = evaluate.attach_fold_scores(mean, long)
    k_ref = 25
    stat = evaluate.bootstrap_stat(n, {"single": (0.6, None), "combined": (0.55, 20)}, k_ref, costs, discrimination=True)
    got = metrics.bootstrap(trajs, wide, stat, n_boot=60)

    def manual(t, p):
        out = {}
        ref = evaluate.cap_point(t, k_ref, costs)
        for name, (tau, cap) in {"single": (0.6, None), "combined": (0.55, 20)}.items():
            per = [evaluate.apply_point(t, fp, n, tau, cap, costs) for fp in evaluate.fold_preds_list(p)]
            for c in evaluate.POINT_COLUMNS:
                out[f"{name}.{c}"] = float(np.mean([x[c] for x in per]))
                if c != "stop_rate":
                    out[f"{name}.d_{c}"] = out[f"{name}.{c}"] - ref[c]
        return out
    want = metrics.bootstrap(trajs, wide, manual, n_boot=60)
    g = {r["stat"]: r for r in got.iter_rows(named=True)}
    for r in want.iter_rows(named=True):
        for c in ("estimate", "lo", "hi"):
            assert g[r["stat"]][c] == pytest.approx(r[c], abs=1e-15), (r["stat"], c)
    # 판별 지표는 평균 점수(p_fail) 그대로
    plain = metrics.bootstrap(trajs, mean, evaluate.bootstrap_stat(n, {}, None, None, discrimination=True), n_boot=60)
    for r in plain.iter_rows(named=True):
        assert g[r["stat"]]["lo"] == r["lo"] and g[r["stat"]]["hi"] == r["hi"]


def test_fold_mean_curve_matches_at_threshold():
    trajs, costs, n, long, mean = fold_setup()
    curves = [metrics.cost_curve(trajs, long.filter(pl.col("fold") == k).drop("fold"), n, costs=costs) for k in range(5)]
    avg = evaluate.fold_mean_curve(curves)
    assert avg["tau"][0] == float("inf") and (avg["tau"].diff().drop_nulls() < 0).all()
    for tau in avg["tau"].to_list()[::17]:
        row = avg.filter(pl.col("tau") == tau).row(0, named=True)
        for c in ("lost_success", "saved_steps", "saved_tokens", "stop_rate"):
            assert row[c] == pytest.approx(np.mean([metrics.at_threshold(cv, tau)[c] for cv in curves]), abs=1e-15)
    # 곡선 하나면 그대로
    one = evaluate.fold_mean_curve(curves[:1])
    assert one.select(curves[0].columns).with_columns(pl.col("n_stopped").cast(pl.Int64)).equals(curves[0].with_columns(pl.col("n_stopped").cast(pl.Int64)))
    # model_curves: test 분할은 fold 곡선 평균, train은 그대로
    sets = {"train": trajs.filter(pl.col("traj_id") < "t0200"), "test": trajs.filter(pl.col("traj_id") >= "t0200")}
    test_long = long.join(sets["test"].select("traj_id"), on="traj_id", how="semi")
    allp = mean.with_columns(pl.col("N").cast(pl.Int16))
    cs = {k: costs.join(v.select("traj_id"), on="traj_id", how="semi") for k, v in sets.items()}
    mc = evaluate.model_curves(allp, sets, cs, [n], fold_preds=test_long)
    ref_test = evaluate.fold_mean_curve([metrics.cost_curve(sets["test"], test_long.filter(pl.col("fold") == k).drop("fold"), n, costs=cs["test"])
                                         for k in range(5)])
    assert mc[("test", n)].equals(ref_test)
    assert mc[("train", n)].equals(metrics.cost_curve(sets["train"], evaluate.split_preds(allp, sets["train"]), n, costs=cs["train"]))


# ---- 모델 무관 운영점 루프 (P5-D2, Phase 5.0) ----

def loop_setup(seed=7, n=5):
    trajs, costs, rng = random_population(seed=seed, n_traj=600)
    trajs = trajs.with_columns(fold=pl.Series([int(i[1:]) % 5 for i in trajs["instance_id"]], dtype=pl.Int8))
    test_issue = pl.col("instance_id").is_in([f"i{k}" for k in range(0, 37, 6)])
    sets = {"train": trajs.filter(~test_issue), "test": trajs.filter(test_issue).with_columns(fold=pl.lit(-1, dtype=pl.Int8))}
    cs = {k: costs.join(v.select("traj_id"), on="traj_id", how="semi") for k, v in sets.items()}
    alive = pl.concat([metrics.running(t, n) for t in sets.values()])
    p = alive.select("traj_id", N=pl.lit(n, dtype=pl.Int16),
                     p_fail=pl.Series((~alive["target"].to_numpy()) * 0.3 + rng.random(alive.height) * 0.7))
    test_alive = metrics.running(sets["test"], n)
    folds = pl.concat([test_alive.select("traj_id", N=pl.lit(n, dtype=pl.Int16), fold=pl.lit(k, dtype=pl.Int8),
                                         p_fail=pl.Series(rng.random(test_alive.height))) for k in range(5)])
    mean = folds.group_by("traj_id", "N").agg(pl.col("p_fail").mean())
    p = pl.concat([evaluate.split_preds(p, sets["train"]), mean.select(p.columns)]).sort("N", "traj_id")
    return sets, cs, n, p, folds


def test_operating_points_loop_matches_direct_selection():
    sets, cs, n, p, folds = loop_setup()
    k_star = metrics.select_cap(metrics.fixed_cap_curve(sets["train"], cs["train"]))["K"]
    main, sens, chosen = evaluate.operating_points({"m1_x": (p, folds), "b0_x": (p, None)}, sets, cs, k_star, n_grid=[n])
    for model, fp in (("m1_x", folds), ("b0_x", None)):
        for policy in evaluate.POLICIES:
            curves = evaluate.candidate_curves(sets["train"], evaluate.split_preds(p, sets["train"]), n, policy, cs["train"])
            op = metrics.select_from_curves(curves, n)
            assert chosen[(model, n, policy)] == (op["tau"], op["cap"])
            test_row = next(r for r in main if (r["model"], r["policy"], r["split"]) == (model, policy, "test"))
            want = evaluate.test_point(sets["test"], evaluate.test_preds(p, fp, sets["test"], n), n, op["tau"], op["cap"], cs["test"])
            assert {c: test_row[c] for c in want} == want
            assert ("lost_success_fold_min" in test_row) == (fp is not None)
    assert len(sens) == 2 * len(evaluate.POLICIES) * len(evaluate.OBJECTIVES) * len(evaluate.MAX_LOSS_GRID) * 2
    cap_main, _ = evaluate.cap_rows(sets, cs, k_star, metrics.fixed_cap_curve(sets["train"], cs["train"]), evaluate.K_PROPOSAL)
    tasks, keys = evaluate.bootstrap_tasks({"m1_x": (p, folds)}, sets, cs, chosen, k_star, {"m1_x": [n]}, n_boot=20)
    ctasks, ckeys = evaluate.cap_bootstrap_tasks(sets, cs, k_star, 20)
    assert keys == [("m1_x", "test", n)] and ckeys == [("b0_cap", "test", None)]
    boots = dict(zip(keys + ckeys, [evaluate.boot_task(*t) for t in tasks + ctasks]))
    rows = cap_main + [r for r in main if r["model"] == "m1_x"]
    table = evaluate.operating_table(rows, boots)
    assert table.columns == evaluate.OP_COLUMNS
    t = table.filter(pl.col("split") == "test")
    assert t["lost_success_hi"].null_count() == 0 and t["meets_judge_ci_hi"].null_count() == 0
    assert t.filter(pl.col("model") == "m1_x")["d_saved_steps_lo"].null_count() == 0
    ci = evaluate.discrimination_ci(boots)
    assert set(ci["stat"]) == {"auroc", "pr_auc", "within_pair", "within_issue_mean"}
