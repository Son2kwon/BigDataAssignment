"""afp.split 단위 테스트. 작은 합성 궤적 표로 분할 규칙과 불변식을 확인한다."""
import numpy as np
import polars as pl
import pytest

from afp import split

M70, M8, M405 = split.TRAIN_MODEL, *split.EXCLUDED_MODELS


def make_traj(n_issues=60, per_issue=6, n_other_issues=12, seed=0):
    """모든 이슈에 70B 궤적이 있고, 앞쪽 n_other_issues개 이슈에는 8B·405B 궤적도 있는 표."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_issues):
        iid = f"own{i % 4}__repo{i % 4}-{i}"
        p = rng.choice([0.0, 0.0, 0.3, 0.8])  # 이슈 난이도 편차
        models = [M70] * per_issue
        if i < n_other_issues:
            models += [M8 if i % 2 else M405] * 2
        for m in models:
            rows.append({"instance_id": iid, "model_name": m, "target": bool(rng.random() < p), "n_steps": int(rng.integers(1, 60))})
    df = pl.DataFrame(rows).with_row_index("r")
    return df.with_columns(traj_id=pl.format("00-{}", pl.col("r").cast(pl.String).str.zfill(5))).drop("r")


def test_test_issues_are_a_random_70b_part_and_other_models_are_dropped():
    traj = make_traj()
    test_issues = set(split.select_test_issues(traj))
    n_issues = traj["instance_id"].n_unique()
    assert 0 < len(test_issues) < n_issues
    assert abs(len(test_issues) - n_issues / split.TEST_PARTS) <= 3  # 약 1/6
    # 8B·405B가 시도했는지와 무관하다: 8B·405B 행을 지워도 같은 테스트 이슈
    assert set(split.select_test_issues(traj.filter(pl.col("model_name") == M70))) == test_issues
    out = split.assign_splits(traj)
    assert out.height == traj.filter(pl.col("model_name") == M70).height
    assert (out["model_name"] == M70).all()
    for r in out.iter_rows(named=True):
        assert r["split"] == ("test" if r["instance_id"] in test_issues else "train")


def test_test_selection_is_stratified_and_fixed():
    traj = make_traj(n_issues=600, per_issue=8, seed=3)
    out = split.assign_splits(traj)
    rate = dict(out.group_by("split").agg(pl.col("target").mean()).iter_rows())
    assert abs(rate["train"] - rate["test"]) < 0.02  # target 층화
    # 테스트 이슈는 고정 시드로 뽑는다 (입력 순서와 무관)
    assert split.select_test_issues(traj.sample(fraction=1.0, shuffle=True, seed=5)) == split.select_test_issues(traj)
    assert split.select_test_issues(traj, seed=7) != split.select_test_issues(traj)


def test_make_splits_invariants():
    traj = make_traj()
    splits = split.make_splits(traj)
    assert dict(splits.schema) == split.SPLIT_SCHEMA
    assert splits.columns == ["traj_id", "split", "fold"]  # target·exit_status 같은 컬럼이 섞이지 않는다
    n_test = len(split.select_test_issues(traj))
    stats = split.validate_splits(traj, splits)
    assert stats["trajectories"] == 60 * 6 == splits.height  # 70B만
    assert stats["excluded_trajectories"] == {M8: 6 * 2, M405: 6 * 2}
    assert stats["split_trajectories"] == {"train": (60 - n_test) * 6, "test": n_test * 6}
    assert stats["train_issues"] == 60 - n_test and stats["test_issues"] == n_test
    assert stats["train_test_issue_overlap"] == 0 and stats["train_issues_crossing_folds"] == 0

    df = traj.join(splits, on="traj_id")
    assert (df["model_name"] == M70).all()
    assert (df.filter(pl.col("split") != "train")["fold"] == split.HELDOUT_FOLD).all()
    train = df.filter(pl.col("split") == "train")
    assert sorted(train["fold"].unique()) == list(range(split.N_FOLDS))
    assert train.group_by("instance_id").agg(pl.col("fold").n_unique())["fold"].max() == 1


def test_make_splits_is_deterministic_and_order_invariant():
    traj = make_traj()
    a = split.make_splits(traj)
    assert split.make_splits(traj).equals(a)
    assert split.make_splits(traj.sample(fraction=1.0, shuffle=True, seed=7)).equals(a)
    assert a["traj_id"].is_sorted()
    # seed가 바뀌면 fold는 바뀔 수 있어도 split(테스트 이슈)은 같다
    b = split.make_splits(traj, seed=123)
    assert b["split"].equals(a["split"])
    split.validate_splits(traj, b)


def test_issue_attempted_only_by_other_models_is_not_in_splits():
    traj = make_traj()
    extra = pl.DataFrame({"traj_id": ["99-00000"], "instance_id": ["solo__x-1"], "model_name": [M8], "target": [True], "n_steps": [5]})
    traj = pl.concat([traj, extra.select(traj.columns)])
    splits = split.make_splits(traj)
    assert "99-00000" not in splits["traj_id"].to_list()
    stats = split.validate_splits(traj, splits)
    assert stats["train_issues"] + stats["test_issues"] == 60


@pytest.mark.parametrize("mutate, message", [
    (lambda t: t.with_columns(model_name=pl.when(pl.col("traj_id") == "00-00000").then(pl.lit("swe-agent-gpt4")).otherwise("model_name")), "모르는 model_name"),
    (lambda t: t.with_columns(instance_id=pl.when(pl.col("traj_id") == "00-00000").then(None).otherwise("instance_id")), "null"),
    (lambda t: pl.concat([t, t.head(1)]), "중복"),
    (lambda t: t.with_columns(pl.col("target").cast(pl.Int8)), "bool"),
    (lambda t: t.drop("model_name"), "없는 컬럼"),
])
def test_invalid_input_fails_loudly(mutate, message):
    with pytest.raises(ValueError, match=message):
        split.make_splits(mutate(make_traj()))


def test_too_few_issues():
    with pytest.raises(ValueError, match="테스트 조각"):
        split.make_splits(make_traj(n_issues=5, n_other_issues=0))
    assert split.validate_splits(t := make_traj(n_issues=6, n_other_issues=0), split.make_splits(t))["train_issues"] == 5
    with pytest.raises(ValueError, match="fold"):  # 학습 이슈가 fold 수보다 적다
        split.train_folds(make_traj(n_issues=4, n_other_issues=0).sort("traj_id"))


def _tamper(splits, traj_id, **values):
    return splits.with_columns(**{
        k: pl.when(pl.col("traj_id") == traj_id).then(pl.lit(v, dtype=splits.schema[k])).otherwise(k) for k, v in values.items()
    })


def test_validate_splits_detects_violations():
    traj = make_traj()
    splits = split.make_splits(traj)
    train_row = splits.filter(pl.col("split") == "train").row(0, named=True)
    test_row = splits.filter(pl.col("split") == "test").row(0, named=True)
    other_fold = (train_row["fold"] + 1) % split.N_FOLDS
    other_model_id = traj.filter(pl.col("model_name") != M70)["traj_id"][0]
    bad = {
        "이슈가 fold를 넘음": _tamper(splits, train_row["traj_id"], fold=other_fold),
        "테스트 fold": _tamper(splits, test_row["traj_id"], fold=0),
        "split 뒤바뀜": _tamper(splits, test_row["traj_id"], split="train", fold=0),
        "모르는 split": _tamper(splits, test_row["traj_id"], split="test_same"),
        "행 빠짐": splits.slice(1),
        "행 추가": pl.concat([splits, pl.DataFrame({"traj_id": ["77-00000"], "split": ["train"], "fold": [0]}, schema=split.SPLIT_SCHEMA)]),
        "8B·405B 궤적 추가": pl.concat([splits, pl.DataFrame({"traj_id": [other_model_id], "split": ["test"], "fold": [-1]}, schema=split.SPLIT_SCHEMA)]),
        "스키마": splits.with_columns(pl.col("fold").cast(pl.Int64)),
    }
    for name, s in bad.items():
        with pytest.raises(ValueError):
            split.validate_splits(traj, s)
        assert name  # 실패 시 어떤 경우인지 보이도록


def test_stratified_folds_balance_success_rate():
    traj = make_traj(n_issues=360, per_issue=8, n_other_issues=0, seed=1)
    train = split.assign_splits(traj).filter(pl.col("split") == "train").sort("traj_id")
    ranges = {}
    for stratified in (False, True):
        bal = split.fold_balance(train, split.train_folds(train, stratified=stratified))
        assert bal["n_traj"].sum() == train.height and bal.height == split.N_FOLDS
        ranges[stratified] = bal["success_rate"].max() - bal["success_rate"].min()
    assert ranges[True] <= ranges[False]


def test_compare_issue_sets():
    traj = make_traj()
    splits = split.make_splits(traj)
    n_test = len(split.select_test_issues(traj))
    metrics, repos = split.compare_issue_sets(traj, splits)
    m = {r["metric"]: (r["train"], r["test"]) for r in metrics.iter_rows(named=True)}
    assert m["issues"] == (60 - n_test, n_test)
    assert m["trajectories_70b"] == ((60 - n_test) * 6, n_test * 6)  # 8B·405B는 비교에서 빠진다
    assert repos["train_issues"].sum() == 60 - n_test and repos["test_issues"].sum() == n_test
    assert abs(repos["train_share"].sum() - 1) < 1e-9 and abs(repos["test_share"].sum() - 1) < 1e-9
    assert pl.select(split.repo_of(pl.lit("owner__my-repo-123"))).item() == "owner__my-repo"
