"""5e 비용 가중 학습: 가중치 누수 방지(P5e-D7 ①~⑤)·가중치 식·선택 규칙. 작은 입력만 쓴다."""
import importlib
import inspect
import json

import numpy as np
import polars as pl
import pytest

from afp import metrics, thought

runner = importlib.import_module("04h_m1_cost_weight")


def toy_train(n_issues=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_issues):
        for j in range(4):
            rows.append({"traj_id": f"t{i:03d}_{j}", "instance_id": f"org__repo-{i:03d}", "fold": i % 5,
                         "target": bool(j % 2), "n_steps": int(rng.integers(5, 70))})
    return pl.DataFrame(rows).sort("traj_id")


def toy_costs(train):
    rows = [{"traj_id": t, "step_idx": s, "ctx": 100 * s + len(t)} for t, n in train.select("traj_id", "n_steps").iter_rows()
            for s in range(1, n + 1)]
    return pl.DataFrame(rows)


def fit_rows(train, n, held):
    return metrics.running(train, n).sort("traj_id").filter(pl.col("fold") != held)


def test_1_weights_reject_held_fold_and_test_rows():
    train = toy_train()
    scope = thought.TrainScope(train)
    alive = metrics.running(train, 5).sort("traj_id")
    with pytest.raises(ValueError, match="held-out"):
        runner.fit_weights(alive, scope, 0, 5, "step", 1.0, 40)
    test_like = fit_rows(train, 5, 0).with_columns(split=pl.lit("test"))
    with pytest.raises(ValueError, match="test"):
        runner.fit_weights(test_like, scope, 0, 5, "step", 1.0, 40)
    unknown = fit_rows(train, 5, 0).with_columns(traj_id=pl.col("traj_id") + "_x")
    with pytest.raises(ValueError):
        runner.fit_weights(unknown, scope, 0, 5, "step", 1.0, 40)
    finished = train.filter(pl.col("fold") != 0)  # N 전에 끝난 궤적 포함
    if finished.filter(pl.col("n_steps") <= 5).height:
        with pytest.raises(ValueError, match="finished"):
            runner.fit_weights(finished, scope, 0, 5, "step", 1.0, 40)


def test_2_weights_do_not_change_feature_matrices(monkeypatch):
    rng = np.random.default_rng(1)
    n = 60
    D = {"X": rng.random((n, 4)), "y": rng.random(n) < 0.5, "inst": np.array([f"i{i // 2}" for i in range(n)]),
         "fold": np.array([(i // 2) % 5 for i in range(n)])}
    blocks = runner.m1g.fold_blocks(D)
    seen = {}

    def fake_fit(cfg, X, y, w, device):
        seen.setdefault(w is None, []).append(X.copy())
        return ("fake", None)

    monkeypatch.setattr(runner, "fit_weighted", fake_fit)
    monkeypatch.setattr(runner, "predict", lambda model, X: np.full(len(X), 0.5))
    D["y"][:2] = [True, False]
    runner.oof_task("k", {}, D["y"], blocks, None, "cpu")
    runner.oof_task("k", {}, D["y"], blocks, [np.full(len(f), 2.0) for f, *_ in blocks], "cpu")
    assert len(seen[True]) == len(seen[False]) == 5
    for a, b in zip(seen[True], seen[False]):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="weights do not match"):
        runner.oof_task("k", {}, D["y"], blocks, [np.ones(3)] * 5, "cpu")


def test_3_prediction_has_no_weight_argument():
    assert list(inspect.signature(runner.predict).parameters) == ["model", "X"]
    assert not any("weight" in p for p in inspect.signature(runner.m1.predict_model).parameters)


@pytest.mark.parametrize("weighting", runner.WEIGHTINGS)
def test_4_failure_weights_have_mean_one_and_success_weight_is_lambda(weighting):
    train = toy_train()
    rows = fit_rows(train, 5, 2)
    w = runner.fit_weights(rows, thought.TrainScope(train), 2, 5, weighting, 0.37, 40, toy_costs(train))
    fail = ~rows["target"].to_numpy()
    assert abs(w[fail].mean() - 1) < 1e-12
    assert (w[fail] > 0).all()
    np.testing.assert_array_equal(w[~fail], 0.37)


def test_5_fold_k_star_ignores_fold_k_labels():
    train = toy_train(40)
    for k in range(5):
        flipped = train.with_columns(target=pl.when(pl.col("fold") == k).then(~pl.col("target")).otherwise(pl.col("target")))
        assert runner.fold_k_star(flipped, k) == runner.fold_k_star(train, k)
        longer = train.with_columns(n_steps=pl.when(pl.col("fold") == k).then(pl.col("n_steps") + 50).otherwise(pl.col("n_steps")))
        assert runner.fold_k_star(longer, k) == runner.fold_k_star(train, k)


def test_raw_saving_matches_combined_policy_formula():
    rows = pl.DataFrame({"traj_id": ["a", "b"], "n_steps": [12, 50]})
    np.testing.assert_array_equal(runner.raw_saving(rows, 5, 40, "step"), [7, 35])
    costs = pl.DataFrame({"traj_id": ["b"] * 50 + ["a"] * 12,
                          "step_idx": [*range(1, 51), *range(1, 13)], "ctx": [*range(1, 51), *range(1, 13)]})
    expect_a, expect_b = sum(range(6, 13)), sum(range(6, 41))
    np.testing.assert_array_equal(runner.raw_saving(rows, 5, 40, "token", costs), [expect_a, expect_b])
    with pytest.raises(ValueError):
        runner.raw_saving(rows, 40, 40, "step")
    # √ 묶음은 원값의 제곱근을 평균 1로 맞춘다
    train = toy_train()
    r = fit_rows(train, 5, 0)
    w = runner.fit_weights(r, thought.TrainScope(train), 0, 5, "step_sqrt", 1.0, 40)
    raw = np.sqrt(runner.raw_saving(r.filter(~pl.col("target")), 5, 40, "step"))
    np.testing.assert_allclose(w[~r["target"].to_numpy()], raw / raw.mean())


def test_pick_prefers_unweighted_then_smaller_lambda():
    lams = {"s3": None, "step": 2.0, "token": 0.5}
    assert runner.pick({"s3": 0.300, "step": 0.3015, "token": 0.3019}, ["s3", "step", "token"], lams) == "s3"
    assert runner.pick({"s3": 0.290, "step": 0.3015, "token": 0.3019}, ["s3", "step", "token"], lams) == "token"
    assert runner.pick({"s3": 0.290, "step": 0.310, "token": 0.3019}, ["s3", "step", "token"], lams) == "step"
    cands = ["step_lam1", "step_lam0half", "step_lam0", "step_lam0x2"]
    lam = {"step_lam1": 1.0, "step_lam0half": 0.2, "step_lam0": 0.4, "step_lam0x2": 0.8}
    assert runner.pick(dict.fromkeys(cands, 0.3), cands, lam) == "step_lam0half"


def test_lambda_grid_and_variants():
    assert runner.lambda_value("lam1", 0.4) == 1.0
    assert [runner.lambda_value(t, 0.4) for t in runner.LAMBDA_TAGS[1:]] == [0.2, 0.4, 0.8]
    assert len(runner.VARIANTS) == 1 + 3 * 4
    assert runner.LAMBDA0_KIND["step_sqrt"] == "step"  # P5e-D9


def test_unweighted_fit_is_the_5d_learner():
    rng = np.random.default_rng(3)
    X, y = rng.random((120, 3)), rng.random(120) < 0.6
    cfg = runner.m1.XGB_GRID[0]
    a = runner.predict(runner.fit_weighted(cfg, X, y, None, "cpu"), X)
    b = runner.m1.predict_model(runner.m1w.fit_xgb(cfg, X, y, "cpu"), X)
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="invalid sample weights"):
        runner.fit_weighted(cfg, X, y, np.zeros(120), "cpu")


def test_upstream_check_stops_when_5d_used_other_s2(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.m1g, "s2_check", lambda: {"phase5c_S2": "sl5", "phase5c_device": "cpu"})
    monkeypatch.setattr(runner.m1g, "OUT_DIR", tmp_path)
    monkeypatch.setattr(runner.m1g, "DATA_DIR", tmp_path)
    sel = {"S3": "length_lexical", "t": 100, "vocab_k": 10}
    summary = {"status": "complete", "criteria": {"C1": True}, "device": "cpu", "s2": {"phase5c_S2": "chg"}, "selection": sel}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    (tmp_path / "handoff.json").write_text(json.dumps({"summary": summary}))
    with pytest.raises(ValueError, match="rerun 5d"):
        runner.upstream_check()
    summary["s2"]["phase5c_S2"] = "sl5"
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    (tmp_path / "handoff.json").write_text(json.dumps({"summary": summary}))
    upstream, _ = runner.upstream_check()
    assert upstream["phase5d_S3"] == "length_lexical" and upstream["phase5d_vocab_k"] == 10
    summary["criteria"]["C1"] = False
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="incomplete"):
        runner.upstream_check()
