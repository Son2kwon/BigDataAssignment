"""afp.baselines 단위 테스트 (합성 데이터)."""
import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression

from afp import baselines, metrics, oof, preds

TAIL = (
    "\n\n\nHere is the list of files that is recommended to be edited in order to fix the issue.\n"
    "FILES TO MODIFY:\n- pkg/module.py"
)


def test_strip_hint_removes_file_list_only():
    clean = "Bug in foo.\r\n\r\nThank you"
    assert baselines.strip_hint(clean) == clean
    assert baselines.strip_hint(clean + TAIL) == clean
    assert baselines.strip_hint(clean + "\r\n" + TAIL) == clean  # 실데이터 4개 이슈는 꼬리 앞에 \r\n이 더 있다
    assert baselines.strip_hint("no hint\n") == "no hint\n"  # 꼬리 없는 본문의 끝 공백은 건드리지 않는다


def test_issue_table_strips_and_requires_one_text(monkeypatch):
    t = pl.DataFrame({
        "traj_id": ["a", "b", "c"], "instance_id": ["i1", "i1", "i2"],
        "issue_text": ["Bug.", "Bug." + TAIL, "Other."],
    })
    monkeypatch.setattr(baselines.io, "load_trajectories", lambda cols: t.select(cols))
    assert baselines.issue_table().rows() == [("i1", "Bug."), ("i2", "Other.")]
    t = t.with_columns(issue_text=pl.Series(["Bug.", "Bug 2.", "Other."]))
    with pytest.raises(AssertionError, match="1개가 아닌 이슈 1개"):
        baselines.issue_table()


def test_issue_rows_weighted_lr_equals_trajectory_lr():
    rng = np.random.default_rng(0)
    n_issues = 40
    X_issue = rng.normal(size=(n_issues, 5))
    m = rng.integers(1, 12, n_issues)
    k = rng.binomial(m, 1 / (1 + np.exp(-X_issue[:, 0])))
    rows = pl.DataFrame({"n_succ": k, "n_fail": m - k})
    idx, y, w = baselines.training_rows(rows, "traj")
    weighted = baselines.fit_lr(X_issue[idx], y, w, C=0.7)

    # 궤적을 하나씩 복제: 이슈 i에서 실패 m-k개, 성공 k개
    X_rep = np.repeat(X_issue, m, axis=0)
    y_rep = np.concatenate([np.r_[np.ones(mi - ki), np.zeros(ki)] for mi, ki in zip(m, k)])
    rep = LogisticRegression(C=0.7, solver="lbfgs", tol=1e-10, max_iter=10_000).fit(X_rep, y_rep)
    np.testing.assert_allclose(weighted.coef_, rep.coef_, atol=1e-5)
    np.testing.assert_allclose(weighted.intercept_, rep.intercept_, atol=1e-5)
    np.testing.assert_allclose(weighted.predict_proba(X_issue), rep.predict_proba(X_issue), atol=1e-5)


def test_issue_weighting_gives_each_issue_equal_mass():
    rows = pl.DataFrame({"n_succ": [1, 0, 6], "n_fail": [1, 4, 2]})
    idx, y, w = baselines.training_rows(rows, "issue")
    mass = np.bincount(idx, weights=w)
    np.testing.assert_allclose(mass, [14 / 3] * 3)  # 이슈마다 같은 무게, 합 = 궤적 수 14
    assert (w > 0).all() and len(w) == 5  # 성공 0개 이슈의 성공 행은 빠진다
    with pytest.raises(ValueError, match="weighting"):
        baselines.training_rows(rows, "other")


def synthetic_issues(n_issues=60, seed=0):
    """이슈마다 궤적 몇 개, 본문은 난이도 단어(hard/easy)를 포함한다."""
    rng = np.random.default_rng(seed)
    rows = []
    texts = []
    for i in range(n_issues):
        hard = i % 2 == 0
        texts.append({"instance_id": f"i{i:03d}", "issue_text": f"{'hard crash' if hard else 'easy typo'} module{i % 7} token{i}"})
        for j in range(rng.integers(2, 6)):
            rows.append({
                "traj_id": f"t{i:03d}_{j}", "instance_id": f"i{i:03d}",
                "target": bool(rng.random() < (0.1 if hard else 0.6)),
                "n_steps": int(rng.integers(1, 30)), "fold": i % 5,
            })
    return pl.DataFrame(rows), pl.DataFrame(texts)


def test_b1_score_constant_within_issue():
    trajs, texts = synthetic_issues()
    rows = baselines.issue_rows(trajs, 0, texts)
    fit, predict = baselines.b1_functions(baselines.B1Config(C=1.0, min_df=1))
    res = oof.oof_predict(rows, fit, predict)
    for n in (0, 5, 10):
        p = baselines.to_trajectories(trajs, res.train.select("instance_id", "p_fail"), n)
        a = metrics.align(trajs, p, n)
        assert a.group_by("instance_id").agg(pl.col("p_fail").n_unique())["p_fail"].max() == 1
        wi = metrics.within_issue_auroc(a)
        assert abs(wi["pair_weighted"] - 0.5) < 1e-12 and abs(wi["issue_mean"] - 0.5) < 1e-12
    # 난이도 단어를 학습해서 처음 보는 이슈도 어느 정도 맞힌다
    a = metrics.align(trajs, baselines.to_trajectories(trajs, res.train.select("instance_id", "p_fail"), 0), 0)
    assert metrics.discrimination(a["target"], a["p_fail"])["auroc"] > 0.7


def test_issue_rows_count_running_trajectories():
    trajs = pl.DataFrame({
        "traj_id": ["a", "b", "c", "d"], "instance_id": ["i1", "i1", "i1", "i2"],
        "target": [True, False, False, True], "n_steps": [3, 10, 0, 2], "fold": [0, 0, 0, 1],
    })
    texts = pl.DataFrame({"instance_id": ["i1", "i2"], "issue_text": ["x", "y"]})
    assert baselines.issue_rows(trajs, 0, texts).select("instance_id", "n_succ", "n_fail").rows() == [("i1", 1, 1), ("i2", 1, 0)]
    assert baselines.issue_rows(trajs, 5, texts).select("instance_id", "n_succ", "n_fail").rows() == [("i1", 0, 1)]
    test = baselines.issue_rows(trajs.drop("fold"), 0, texts)
    assert test["fold"].to_list() == [-1, -1]
    with pytest.raises(ValueError, match="본문이 없는"):
        baselines.issue_rows(trajs, 0, texts.head(1))


def test_error_count_ignores_steps_after_n():
    rng = np.random.default_rng(1)
    trajs = pl.DataFrame({"traj_id": [f"t{i}" for i in range(30)], "n_steps": rng.integers(0, 25, 30)})
    flags = pl.DataFrame({
        "traj_id": [f"t{i}" for i in range(30) for _ in range(trajs["n_steps"][i])],
        "step_idx": [s + 1 for i in range(30) for s in range(trajs["n_steps"][i])],
    }).with_columns(is_error=pl.Series(rng.random(int(trajs["n_steps"].sum())) < 0.3))
    grid = (3, 5, 10)
    base = baselines.count_scores(flags, trajs, grid, "is_error")
    for n in grid:
        # N 뒤의 스텝을 모두 바꿔도 시점 N 점수는 같다
        flipped = flags.with_columns(is_error=pl.when(pl.col("step_idx") > n).then(~pl.col("is_error")).otherwise(pl.col("is_error")))
        assert baselines.count_scores(flipped, trajs, [n], "is_error").equals(base.filter(pl.col("N") == n))
        # 진행 중 궤적마다 하나, 값 = 앞 N스텝 에러 수
        got = base.filter(pl.col("N") == n)
        alive = metrics.running(trajs, n)
        assert sorted(got["traj_id"]) == sorted(alive["traj_id"])
        want = flags.filter(pl.col("step_idx") <= n).group_by("traj_id").agg(c=pl.col("is_error").sum())
        chk = got.join(want, on="traj_id", how="left").fill_null(0)
        assert (chk["p_fail"] == chk["c"]).all()


def test_oracle_loo_within_issue_is_zero():
    trajs, _ = synthetic_issues(seed=2)
    p = baselines.oracle_loo(trajs, 0, fill=0.8)
    a = metrics.align(trajs, p, 0)
    wi = metrics.within_issue_auroc(a)
    assert wi["n_pairs"] > 0 and wi["pair_weighted"] == 0.0  # 정의상 0 (자기 라벨을 뺀 인공물)
    # 손으로: 이슈 (S, F, F) → S는 0/2 성공 → p_fail 1, F는 1/2 → 0.5
    t = pl.DataFrame({"traj_id": ["a", "b", "c", "d"], "instance_id": ["i", "i", "i", "j"],
                      "target": [True, False, False, True], "n_steps": [5, 5, 5, 5]})
    assert dict(baselines.oracle_loo(t, 0, fill=0.9).select("traj_id", "p_fail").rows()) == {"a": 1.0, "b": 0.5, "c": 0.5, "d": 0.9}


def test_oracle_xmodel_constant_within_issue_and_fills_missing():
    t = pl.DataFrame({"traj_id": ["a", "b", "c"], "instance_id": ["i", "i", "j"], "target": [True, False, True], "n_steps": [5, 5, 5]})
    other = pl.DataFrame({"instance_id": ["i", "i", "i", "i"], "target": [True, False, False, False]})
    p = baselines.oracle_xmodel(t, other, 0, fill=0.8)
    assert dict(p.select("traj_id", "p_fail").rows()) == {"a": 0.75, "b": 0.75, "c": 0.8}


def test_oracle_full_loo_skips_single_trajectory_issues():
    t = pl.DataFrame({"traj_id": ["a", "b", "c", "d"], "instance_id": ["i", "i", "i", "j"],
                      "target": [True, False, False, True], "n_steps": [5, 0, 5, 5]})
    out = baselines.oracle_full_loo_auroc(t)
    assert out["n_traj"] == 3 and out["n_issues"] == 1 and out["auroc"] == 0.0  # 스텝 0개 궤적 b도 들어간다


def test_ref_preds_rejected_as_features(tmp_path, monkeypatch):
    monkeypatch.setattr(preds, "PREDS_DIR", tmp_path)
    trajs = pl.DataFrame({"traj_id": ["a"], "instance_id": ["i"], "target": [True], "n_steps": [3]})
    p = pl.DataFrame({"traj_id": ["a"], "N": [0], "p_fail": [0.4]})
    preds.write("ref_oracle_loo", p, {"train": trajs}, [0], {}, "none")
    preds.write("b1_n", p, {"train": trajs}, [0], {}, "none")
    with pytest.raises(ValueError, match="피처로 쓸 수 없다"):
        preds.load_feature("ref_oracle_loo")
    assert preds.load_feature("b1_n")["p_fail"].to_list() == [0.4]
