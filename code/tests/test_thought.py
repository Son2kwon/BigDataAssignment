"""Phase 5d fold vocabulary: 손 계산·guard·문서/스텝 차이. GPU/실데이터 없음."""
import numpy as np
import polars as pl
import pytest

from afp import thought


def fixture():
    rows, steps = [], []
    for fold in range(5):
        for success in (False, True):
            tid = f"{fold}_{int(success)}"
            rows.append({"traj_id": tid, "instance_id": f"org__repo-{fold}", "fold": fold, "target": success, "n_steps": 10})
            words = ("works", "works", "now") if success else ("stuck stuck", "stuck", "again")
            steps.extend({"traj_id": tid, "step_idx": i, "action_text": text} for i, text in enumerate(words, 1))
    train, raw = pl.DataFrame(rows), pl.DataFrame(steps)
    corpus = thought.Corpus.build(raw, train, 3)
    scope = thought.TrainScope(train)
    bodies = {f"org__repo-{i}": "stuck in issue" for i in range(5)}
    return train, raw, corpus, scope, bodies


def fitted(held=0):
    train, raw, corpus, scope, bodies = fixture()
    fit = train.filter(pl.col("fold") != held)
    stats = thought.occurrences(corpus, fit.select("traj_id", "instance_id", "fold"), scope, held, bodies)
    scores = thought.association(corpus, fit, scope, held, stats)
    return train, raw, corpus, scope, bodies, fit, stats, scores


def test_tokenization_removes_whole_paths_and_numbers_with_boundaries():
    grams = thought.ngrams(thought.token_segments("HELLO src/parser.py world 123 retry v2 finally foo.txt done"))
    assert {"hello", "world", "retry", "finally", "done"} <= grams
    assert not ({"src", "parser", "py", "foo", "txt", "123", "v", "hello world", "world retry", "retry finally"} & grams)
    assert "works now" in thought.ngrams(thought.token_segments("WORKS now"))
    assert "works now" not in thought.ngrams(thought.token_segments("works. now"))


def test_failure_residual_sign_score_and_issue_frequency_by_hand():
    *_, scores = fitted()
    by = {r["ngram"]: r for r in scores}
    # 4 fit 이슈에 각 성공/실패 1개. 실패 잔차 +0.5, 성공 -0.5.
    assert by["stuck"]["score"] == pytest.approx(2 / np.sqrt(4))
    assert by["works"]["score"] == pytest.approx(-2 / np.sqrt(4))
    assert by["stuck"]["n_trajectories"] == 4
    assert by["stuck"]["n_issues"] == 4
    vocabulary = thought.select_vocabulary(scores, 4, 100)
    assert {r["direction"] for r in vocabulary} >= {"failure", "success"}
    assert thought.select_vocabulary(scores, 5, 10) == []


def test_repeated_word_counts_once_per_step_and_document_boundary_bigram():
    _, _, corpus, *_ = fixture()
    vocab = [{"ngram": "stuck"}, {"ngram": "stuck stuck"}, {"ngram": "stuck again"}]
    X = corpus.transform(["0_0"], vocab, 3)
    np.testing.assert_array_equal(X, [[2, 1, 0]])
    assert "stuck again" in corpus.documents["0_0"]
    # 선택 문서는 스텝2→3 경계 bigram도 포함하지만 모델은 스텝 안만 센다.
    assert thought.lexical_columns(3, 3) == ["thought_g00", "thought_g01", "thought_g02"]


def test_lexical_window_boundary_and_padding():
    train, raw, _, _, _ = fixture()
    extra = pl.DataFrame([{"traj_id": tid, "step_idx": i, "action_text": ""} for tid in train["traj_id"] for i in (4, 5)])
    corpus = thought.Corpus.build(pl.concat([raw, extra]), train, 5)
    X = corpus.transform(["0_0"], [{"ngram": "stuck"}], 2)
    np.testing.assert_array_equal(X, [[2, 0, 0, 0]])  # step1/2 only; window=3..5
    assert len(thought.lexical_columns(5, 2)) == 4


def test_held_labels_and_future_text_do_not_change_fit_vocabulary():
    train, raw, corpus, scope, bodies, fit, stats, scores = fitted()
    altered = train.with_columns(target=pl.when(pl.col("fold") == 0).then(~pl.col("target")).otherwise(pl.col("target")))
    assert thought.association(corpus, altered.filter(pl.col("fold") != 0), scope, 0, stats) == scores
    future = pl.DataFrame([{"traj_id": t, "step_idx": 9, "action_text": "FUTURE secret"} for t in train["traj_id"]])
    other = thought.Corpus.build(pl.concat([raw, future]).reverse(), train, 3)
    assert other.documents == corpus.documents
    assert other.steps == corpus.steps
    assert thought.association(other, fit.reverse(), scope, 0, stats) == scores
    # fit 라벨이 전부 실패면 이슈 내 잔차는 0이다.
    all_fail = thought.association(corpus, fit.with_columns(target=pl.lit(False)), scope, 0, stats)
    assert all(r["score"] == 0 for r in all_fail)


@pytest.mark.parametrize("kind", ("held", "test", "unknown", "forged_fold", "forged_issue", "mixed_test"))
def test_fit_rejects_held_test_unknown_and_forged_rows(kind):
    train, _, corpus, scope, bodies, fit, *_ = fitted()
    if kind == "held":
        bad = train
    elif kind == "test":
        bad = fit.with_columns(split=pl.lit("test"))
    elif kind == "mixed_test":
        bad = fit.with_columns(split=pl.when(pl.col("traj_id") == "1_0").then(pl.lit("test")).otherwise(pl.lit("train")))
    elif kind == "unknown":
        bad = fit.with_columns(traj_id=pl.col("traj_id") + "unknown")
    elif kind == "forged_fold":
        bad = fit.with_columns(fold=pl.lit(3))
    else:
        bad = fit.with_columns(instance_id=pl.lit("forged__repo-1"))
    with pytest.raises(ValueError):
        thought.association(corpus, bad, scope, 0, thought.occurrences(
            corpus, fit.select("traj_id", "instance_id", "fold"), scope, 0, bodies))


def test_stats_cannot_be_reused_from_another_fold_and_diagnostics_reject_labels():
    _, _, corpus, scope, bodies, fit, stats, _ = fitted()
    with pytest.raises(ValueError, match="different fit"):
        thought.association(corpus, fit.head(1), scope, 0, stats)
    with pytest.raises(ValueError, match="unlabelled"):
        thought.occurrences(corpus, fit, scope, 0, bodies)


def test_unlabelled_diagnostics_denominators_and_thresholds():
    *_, stats, _ = fitted()
    result = thought.diagnostics(stats, (4, 5))
    assert result[0]["n_candidates"] > 0
    assert result[0]["repo_focused_ratio"] == 1
    assert result[0]["issue_overlap_n"] == 1  # stuck only
    assert result[1]["n_candidates"] == 0
    assert result[1]["repo_focused_ratio"] is None


def test_score_tie_lexical_order_and_overlap_empty_sets():
    scores = [{"ngram": g, "score": s, "n_issues": 10, "n_trajectories": 10} for g, s in (("z", -1), ("a", 1))]
    assert [r["ngram"] for r in thought.select_vocabulary(scores, 10, 2)] == ["a", "z"]
    rows = thought.overlap_rows({0: [{"ngram": "a"}], 1: [{"ngram": "a"}, {"ngram": "b"}], 2: []})
    assert rows[0]["jaccard"] == 0.5
    assert thought.overlap_rows({0: [], 1: []})[0]["jaccard"] is None
    with pytest.raises(ValueError, match="owner__repo"):
        thought.repo_id("unknown")


def test_scope_rejects_issue_crossing_folds():
    train, *_ = fixture()
    with pytest.raises(ValueError, match="registry"):
        thought.TrainScope(train.with_columns(instance_id=pl.lit("same__repo-1")))
