"""afp.preds: 예측 파일 검증·저장·재현."""
import polars as pl
import pytest

from afp import preds

TRAIN = pl.DataFrame({"traj_id": ["a", "b", "c"], "instance_id": ["i", "i", "j"], "target": [True, False, False], "n_steps": [2, 5, 0]})
TEST = pl.DataFrame({"traj_id": ["d"], "instance_id": ["k"], "target": [True], "n_steps": [4]})
SETS = {"train": TRAIN, "test_same": TEST}


def good():
    # N=0: a, b, d / N=3: b, d (c는 스텝 0개, a는 N=3 전에 끝남)
    return pl.DataFrame({"traj_id": ["d", "a", "b", "b", "d"], "N": [0, 0, 0, 3, 3], "p_fail": [0.1, 0.2, 0.3, 0.4, 0.5]})


@pytest.fixture
def tmp_preds(tmp_path, monkeypatch):
    monkeypatch.setattr(preds, "PREDS_DIR", tmp_path)
    return tmp_path


def test_validate_counts_and_rejects_bad_coverage():
    assert preds.validate(good(), SETS, [0, 3]) == {"train": {0: 2, 3: 1}, "test_same": {0: 1, 3: 1}}
    with pytest.raises(ValueError, match="test_same: N=3: .*예측 없음 1개"):
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
    assert meta["model_id"] == "b0_err" and meta["n_grid"] == [0, 3] and meta["splits"] == ["train", "test_same"]
    assert meta["config"] == {"rule": "is_error"} and meta["oof_scheme"] == "rule"


def test_model_id_rules(tmp_preds):
    for bad in ("x_err", "B0_err", "b0-err", "../b0_x"):
        with pytest.raises(ValueError, match="model_id"):
            preds.path(bad)
    with pytest.raises(FileNotFoundError, match="03_baselines.py"):
        preds.read("b1_missing")
