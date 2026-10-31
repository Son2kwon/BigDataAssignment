"""afp.oof 단위 테스트: fold 경계를 하나라도 어기면 깨지는 합성 데이터."""
import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from afp import baselines, oof


def memorization_data(n_issues=200, per_issue=6, seed=0):
    """라벨이 이슈로만 정해지고 피처는 이슈 ID 원-핫. 처음 보는 이슈는 구별할 수 없다.

    처음 보는 이슈의 점수는 절편뿐이고 절편은 학습 fold의 실패율을 따르므로, fold마다 실패 이슈 비율을
    똑같이 맞춘다 (안 맞추면 "검증 fold 실패율이 높을수록 절편이 낮다"는 fold 인공물로 AUROC가 0.5에서 벗어난다).
    """
    rng = np.random.default_rng(seed)
    fail_issue = np.zeros(n_issues, dtype=bool)
    for f in range(5):
        members = np.arange(f, n_issues, 5)
        fail_issue[rng.permutation(members)[: len(members) // 2]] = True
    rows = [{"row": i * per_issue + j, "instance_id": i, "fold": i % 5, "y": bool(fail_issue[i])}
            for i in range(n_issues) for j in range(per_issue)]
    return pl.DataFrame(rows), n_issues


def onehot(rows, n_issues):
    X = np.zeros((rows.height, n_issues))
    X[np.arange(rows.height), rows["instance_id"].to_numpy()] = 1
    return X


def test_oof_issue_memorization_is_chance():
    data, n_issues = memorization_data()
    fit = lambda r: LogisticRegression(C=100.0, max_iter=1000).fit(onehot(r, n_issues), r["y"].to_numpy())
    predict = lambda m, r: m.predict_proba(onehot(r, n_issues))[:, 1]
    res = oof.oof_predict(data, fit, predict)
    in_fold = roc_auc_score(data["y"], predict(fit(data), data))
    oof_auc = roc_auc_score(res.train["y"], res.train["p_fail"])
    assert in_fold > 0.99
    assert abs(oof_auc - 0.5) < 1e-12  # 검증 fold 안에서는 점수가 모두 절편으로 같다
    assert res.seen_issue_scores == 0
    assert res.train.drop("p_fail").equals(data)  # 입력 순서 그대로


def test_oof_never_scores_seen_issue():
    data, _ = memorization_data(n_issues=50)
    seen = {}

    def fit(rows):
        return set(rows["instance_id"].to_list())

    def predict(model, rows):
        for i in rows["instance_id"].to_list():
            seen.setdefault(i, []).append(i in model)
        return np.zeros(rows.height)

    res = oof.oof_predict(data, fit, predict)
    assert not any(any(v) for v in seen.values())
    assert sorted(res.fold_issues) == [0, 1, 2, 3, 4] and all(v == 40 for v in res.fold_issues.values())

    # 이슈가 두 fold에 걸치면 멈춘다
    bad = data.with_columns(fold=pl.when(pl.col("row") == 0).then(1).otherwise(pl.col("fold")))
    with pytest.raises(ValueError, match="여러 fold"):
        oof.oof_predict(bad, fit, predict)
    with pytest.raises(ValueError, match="fold는 0..4"):
        oof.oof_predict(data.filter(pl.col("fold") != 4), fit, predict)
    with pytest.raises(ValueError, match="train에도 있음"):
        oof.oof_predict(data, fit, predict, test=data.head(3))


def test_test_scores_are_fold_mean():
    data, _ = memorization_data(n_issues=50)
    test = pl.DataFrame({"instance_id": [1000, 1001], "x": [1.0, 2.0]})
    # 모델 = 학습에 쓴 행 수. 테스트 점수 = 5개 fold 모델의 평균
    fit = lambda rows: rows.height
    predict = lambda model, rows: np.full(rows.height, float(model))
    res = oof.oof_predict(data, fit, predict, test=test)
    sizes = [data.filter(pl.col("fold") != k).height for k in range(5)]
    assert res.test["p_fail"].to_list() == pytest.approx([np.mean(sizes)] * 2)
    assert res.test.drop("p_fail").equals(test)
    refit = oof.oof_predict(data, fit, predict, test=test, test_mode="refit")
    assert refit.test["p_fail"].to_list() == [float(data.height)] * 2
    with pytest.raises(ValueError, match="test_mode"):
        oof.oof_predict(data, fit, predict, test=test, test_mode="other")


def test_vectorizer_fit_inside_fold():
    # fold 0 이슈에만 있는 단어 "zebra"는 fold 0을 뺀 모델의 어휘에 없어야 한다
    rows = pl.DataFrame({
        "instance_id": [f"i{i}" for i in range(10)],
        "fold": [i % 5 for i in range(10)],
        "issue_text": [("zebra common" if i % 5 == 0 else f"common word{i % 3}") for i in range(10)],
        "n_succ": [1, 0] * 5,
        "n_fail": [1, 2] * 5,
    })
    config = baselines.B1Config(min_df=1)
    vocab = {}
    fit_b1, predict = baselines.b1_functions(config)

    def fit(r):
        model = fit_b1(r)
        vocab[tuple(sorted(set(r["fold"].to_list())))] = set(model.vectorizer.vocabulary_)
        return model

    oof.oof_predict(rows, fit, predict)
    for folds, words in vocab.items():
        assert ("zebra" in words) == (0 in folds), folds
