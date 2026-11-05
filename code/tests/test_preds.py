"""afp.preds: 예측 파일 검증·저장·재현."""
import polars as pl
import pytest

from afp import preds

TRAIN = pl.DataFrame({"traj_id": ["a", "b", "c"], "instance_id": ["i", "i", "j"], "target": [True, False, False], "n_steps": [2, 5, 0]})
TEST = pl.DataFrame({"traj_id": ["d"], "instance_id": ["k"], "target": [True], "n_steps": [4]})
SETS = {"train": TRAIN, "test": TEST}


def good():
    # N=0: a, b, d / N=3: b, d (c는 스텝 0개, a는 N=3 전에 끝남)
    return pl.DataFrame({"traj_id": ["d", "a", "b", "b", "d"], "N": [0, 0, 0, 3, 3], "p_fail": [0.1, 0.2, 0.3, 0.4, 0.5]})


@pytest.fixture
def tmp_preds(tmp_path, monkeypatch):
    monkeypatch.setattr(preds, "PREDS_DIR", tmp_path)
    return tmp_path


def test_validate_counts_and_rejects_bad_coverage():
    assert preds.validate(good(), SETS, [0, 3]) == {"train": {0: 2, 3: 1}, "test": {0: 1, 3: 1}}
    with pytest.raises(ValueError, match="test: N=3: .*예측 없음 1개"):
        preds.validate(good().filter(~((pl.col("traj_id") == "d") & (pl.col("N") == 3))), SETS, [0, 3])
    with pytest.raises(ValueError, match="대상 밖 예측 1개"):
        preds.validate(pl.concat([good(), pl.DataFrame({"traj_id": ["a"], "N": [3], "p_fail": [0.5]})]), SETS, [0, 3])
    with pytest.raises(ValueError, match="어디에도 없는 궤적"):
        preds.validate(pl.concat([good(), pl.DataFrame({"traj_id": ["zz"], "N": [0], "p_fail": [0.5]})]), SETS, [0, 3])
    with pytest.raises(ValueError, match="n_grid"):
        preds.validate(good(), SETS, [0])
    with pytest.raises(ValueError, match="NaN/inf"):
        preds.validate(good().with_columns(p_fail=pl.lit(float("inf"))), SETS, [0, 3])


def test_write_is_idempotent_and_guards_changes(tmp_preds):
    st = preds.write("b0_err", good(), SETS, [0, 3], {"rule": "is_error"}, "rule")
    assert st["written"] and st["rows"] == 5
    path = preds.path("b0_err")
    before = path.read_bytes()
    st = preds.write("b0_err", good().sample(fraction=1.0, shuffle=True, seed=3), SETS, [0, 3], {"rule": "is_error"}, "rule")
    assert not st["written"] and st["matches_existing_file"]
    assert path.read_bytes() == before  # 같은 내용이면 바이트 그대로

    changed = good().with_columns(p_fail=pl.col("p_fail") + 0.01)
    with pytest.raises(SystemExit, match="--force"):
        preds.write("b0_err", changed, SETS, [0, 3], {"rule": "is_error"}, "rule")
    with pytest.raises(SystemExit):
        preds.write("b0_err", good(), SETS, [0, 3], {"rule": "edit_failed"}, "rule")  # config가 달라도 멈춤
    assert preds.write("b0_err", changed, SETS, [0, 3], {"rule": "is_error"}, "rule", force=True)["written"]

    got = preds.read("b0_err")
    assert got.columns == ["traj_id", "N", "p_fail"] and got["N"].dtype == pl.Int16
    assert got.select("N", "traj_id").rows() == [(0, "a"), (0, "b"), (0, "d"), (3, "b"), (3, "d")]
    meta = preds.metadata("b0_err")
    assert meta["model_id"] == "b0_err" and meta["n_grid"] == [0, 3] and meta["splits"] == ["train", "test"]
    assert meta["config"] == {"rule": "is_error"} and meta["oof_scheme"] == "rule"


def test_model_id_rules(tmp_preds):
    for bad in ("x_err", "B0_err", "b0-err", "../b0_x"):
        with pytest.raises(ValueError, match="model_id"):
            preds.path(bad)
    with pytest.raises(FileNotFoundError, match="03_baselines.py"):
        preds.read("b1_missing")


# ---- fold별 test 점수와 M1 입력 API (P4-D6·P4-D7) ----

SPLITS = pl.DataFrame({"traj_id": ["a", "b", "c", "d"], "split": ["train", "train", "train", "test"]})
FOLD_SCORES = {0: (0.1, 0.4), 1: (0.3, 0.6), 2: (0.2, 0.5)}  # fold k → (d의 N=0 점수, N=3 점수)


def fold_frame():
    return pl.DataFrame([("d", n, k, v[i]) for k, v in FOLD_SCORES.items() for i, n in enumerate((0, 3))],
                        schema=["traj_id", "N", "fold", "p_fail"], orient="row")


def write_with_folds():
    mean = {n: sum(v[i] for v in FOLD_SCORES.values()) / 3 for i, n in enumerate((0, 3))}
    main = good().with_columns(p_fail=pl.when(pl.col("traj_id") == "d").then(pl.col("N").replace_strict(mean, return_dtype=pl.Float64))
                               .otherwise(pl.col("p_fail")))
    preds.write("b1_n", main, SETS, [0, 3], {}, "5fold-oof/test-fold-mean")
    return preds.write_test_folds("b1_n", fold_frame(), TEST, [0, 3], n_folds=3)


def test_feature_test_requires_fold(tmp_preds):
    write_with_folds()
    # (a) fold k를 주면 fold k 모델 점수
    for k, (v0, v3) in FOLD_SCORES.items():
        got = preds.load_feature("b1_n", "test", fold=k, splits=SPLITS)
        assert got.select("traj_id", "N", "p_fail").rows() == [("d", 0, v0), ("d", 3, v3)]
    # (b) fold 없이 test를 요청하면 에러 (fold 평균 점수 = P4-D7 ③을 꺼낼 수 없다)
    with pytest.raises(ValueError, match="fold를 지정"):
        preds.load_feature("b1_n", "test", splits=SPLITS)
    with pytest.raises(ValueError, match="fold 5가 없음"):
        preds.load_feature("b1_n", "test", fold=5, splits=SPLITS)
    # train은 OOF 점수 (fold를 주면 에러), test 궤적은 섞이지 않는다
    tr = preds.load_feature("b1_n", "train", splits=SPLITS)
    assert set(tr["traj_id"]) == {"a", "b"} and tr.filter(pl.col("traj_id") == "a")["p_fail"].to_list() == [0.2]
    with pytest.raises(ValueError, match="train 피처"):
        preds.load_feature("b1_n", "train", fold=0, splits=SPLITS)
    with pytest.raises(ValueError, match="split은"):
        preds.load_feature("b1_n", "valid", splits=SPLITS)


def test_feature_other_folds_excludes_own(tmp_preds):
    write_with_folds()
    # (c) ② 민감도: fold k에 대해 j≠k만
    for k in FOLD_SCORES:
        got = preds.load_feature_other_folds("b1_n", k, splits=SPLITS)
        assert sorted(set(got["src_fold"].to_list())) == [j for j in FOLD_SCORES if j != k]
        assert got.height == 2 * (len(FOLD_SCORES) - 1)
        for j in set(got["src_fold"].to_list()):
            assert got.filter(pl.col("src_fold") == j).sort("N")["p_fail"].to_list() == list(FOLD_SCORES[j])
    with pytest.raises(ValueError, match="fold 9가 없음"):
        preds.load_feature_other_folds("b1_n", 9, splits=SPLITS)


def test_test_folds_mean_matches_main_and_write_is_idempotent(tmp_preds):
    st = write_with_folds()
    # (d) fold 점수 평균 = 주 파일 test 점수 (저장 전에 검사, 차이 기록)
    assert st["written"] and st["folds"] == [0, 1, 2] and st["max_abs_mean_diff"] <= preds.FOLD_MEAN_TOL
    f = preds.read_test_folds("b1_n")
    mean = f.group_by("N").agg(pl.col("p_fail").mean()).sort("N")["p_fail"].to_list()
    main = preds.read("b1_n").filter(pl.col("traj_id") == "d").sort("N")["p_fail"].to_list()
    assert mean == pytest.approx(main, abs=1e-15)
    before = preds.fold_path("b1_n").read_bytes()
    again = preds.write_test_folds("b1_n", fold_frame().reverse(), TEST, [0, 3], n_folds=3)
    assert not again["written"] and preds.fold_path("b1_n").read_bytes() == before
    with pytest.raises(SystemExit, match="--force"):
        # fold 0·1 점수를 바꿔 넣음: 평균은 같아 검사는 통과하지만 내용이 달라 force 없이는 멈춘다
        preds.write_test_folds("b1_n", fold_frame().with_columns(fold=pl.col("fold").replace_strict({0: 1, 1: 0, 2: 2})), TEST, [0, 3], n_folds=3)
    # fold 평균이 주 파일과 다르면 저장하지 않는다
    with pytest.raises(ValueError, match="fold 평균과 주 파일"):
        preds.write_test_folds("b1_n", fold_frame().with_columns(p_fail=pl.col("p_fail") + 0.01), TEST, [0, 3], n_folds=3, force=True)
    # fold마다 진행 중 test 궤적과 1:1
    with pytest.raises(ValueError, match="fold 1: .*예측 없음"):
        preds.validate_test_folds(fold_frame().filter(~((pl.col("fold") == 1) & (pl.col("N") == 3))), TEST, [0, 3], 3)
    with pytest.raises(ValueError, match="fold는 0..3"):
        preds.validate_test_folds(fold_frame(), TEST, [0, 3], 4)
    # 모델 이름과 겹치지 않는 경로
    assert preds.fold_path("b1_n").name == "b1_n.test_folds.parquet"
