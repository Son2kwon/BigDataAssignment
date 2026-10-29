"""학습/테스트 분할과 학습 이슈 fold.

- 테스트 이슈 = 8B 또는 405B가 한 번이라도 시도한 instance_id 전부
- 테스트 이슈의 70B 궤적 → test_same, 8B·405B 궤적 → test_cross
- 나머지 이슈의 70B 궤적 → train. 이슈(instance_id) 단위 5-fold StratifiedGroupKFold
- fold: train은 0..4, test_same·test_cross는 HELDOUT_FOLD(-1). 테스트는 어떤 fold에도 속하지 않는다

GroupKFold 대신 StratifiedGroupKFold를 쓰는 이유: 이슈 성공률 편차가 커서(76.7%가 성공 0)
GroupKFold로는 fold 성공률이 14.4%~17.5%로 벌어진다. 비교 수치는 results/phase2/fold_balance.csv.
"""
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from afp.io import SEED

TRAIN_MODEL = "swe-agent-llama-70b"
HELDOUT_MODELS = ("swe-agent-llama-8b", "swe-agent-llama-405b")
KNOWN_MODELS = (TRAIN_MODEL, *HELDOUT_MODELS)

SPLITS = ("train", "test_same", "test_cross")
N_FOLDS = 5
HELDOUT_FOLD = -1

# 분할에 필요한 컬럼. target은 fold 층화에만 쓰고 splits.parquet에는 넣지 않는다.
INPUT_COLUMNS = ["traj_id", "instance_id", "model_name", "target"]
SPLIT_SCHEMA = {"traj_id": pl.String, "split": pl.String, "fold": pl.Int8}


def validate_input(traj: pl.DataFrame) -> None:
    """분할 입력 검사. 모르는 모델·null·중복 ID는 조용히 버리지 않고 멈춘다."""
    missing = [c for c in INPUT_COLUMNS if c not in traj.columns]
    if missing:
        raise ValueError(f"분할 입력에 없는 컬럼: {missing}")
    nulls = {c: n for c in INPUT_COLUMNS if (n := traj[c].null_count())}
    if nulls:
        raise ValueError(f"분할 입력에 null이 있다: {nulls}")
    if traj["target"].dtype != pl.Boolean:
        raise ValueError(f"target은 bool이어야 한다: {traj['target'].dtype}")
    dup = traj.height - traj["traj_id"].n_unique()
    if dup:
        raise ValueError(f"traj_id 중복 {dup}개")
    unknown = sorted(set(traj["model_name"].unique()) - set(KNOWN_MODELS))
    if unknown:
        raise ValueError(f"모르는 model_name: {unknown}. 알려진 모델: {list(KNOWN_MODELS)}")


def heldout_issues(traj: pl.DataFrame) -> list[str]:
    """8B 또는 405B가 시도한 instance_id (정렬)."""
    return sorted(traj.filter(pl.col("model_name").is_in(HELDOUT_MODELS))["instance_id"].unique())


def assign_splits(traj: pl.DataFrame) -> pl.DataFrame:
    """traj에 split 컬럼을 붙인다. 모든 행이 정확히 하나의 split을 받는다."""
    validate_input(traj)
    is_test = pl.col("instance_id").is_in(heldout_issues(traj))
    is_70b = pl.col("model_name") == TRAIN_MODEL
    return traj.with_columns(
        split=pl.when(is_test & is_70b).then(pl.lit("test_same"))
        .when(is_test).then(pl.lit("test_cross"))
        .otherwise(pl.lit("train"))
    )


def fold_splitter(n_folds: int = N_FOLDS, seed: int = SEED, stratified: bool = True):
    if stratified:
        return StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return GroupKFold(n_splits=n_folds)


def train_folds(train: pl.DataFrame, n_folds: int = N_FOLDS, seed: int = SEED, stratified: bool = True) -> np.ndarray:
    """train 행(traj_id 순 정렬된 상태여야 함)의 fold 번호. 그룹 = instance_id."""
    n_issues = train["instance_id"].n_unique()
    if n_issues < n_folds:
        raise ValueError(f"학습 이슈 {n_issues}개로는 {n_folds}-fold를 만들 수 없다")
    y = train["target"].to_numpy()
    groups = train["instance_id"].to_numpy()
    fold = np.full(train.height, HELDOUT_FOLD, dtype=np.int8)
    for k, (_, val) in enumerate(fold_splitter(n_folds, seed, stratified).split(np.zeros(len(y)), y, groups)):
        fold[val] = k
    return fold


def make_splits(traj: pl.DataFrame, n_folds: int = N_FOLDS, seed: int = SEED, stratified: bool = True) -> pl.DataFrame:
    """궤적 표 → splits 표 (traj_id, split, fold). 입력 행 순서와 무관하게 같은 결과를 낸다."""
    validate_input(traj)
    df = assign_splits(traj.select(INPUT_COLUMNS)).sort("traj_id")
    is_train = (df["split"] == "train").to_numpy()
    fold = np.full(df.height, HELDOUT_FOLD, dtype=np.int8)
    fold[is_train] = train_folds(df.filter(pl.col("split") == "train"), n_folds, seed, stratified)
    return df.select("traj_id", "split").with_columns(fold=pl.Series(fold, dtype=pl.Int8))


def validate_splits(traj: pl.DataFrame, splits: pl.DataFrame, n_folds: int = N_FOLDS) -> dict:
    """splits가 분할 규칙을 지키는지 확인하고 크기를 돌려준다. 어기면 ValueError."""
    validate_input(traj)
    errors = []
    if dict(splits.schema) != SPLIT_SCHEMA:
        errors.append(f"스키마가 다르다: {dict(splits.schema)} != {SPLIT_SCHEMA}")
    if splits.null_count().sum_horizontal().item():
        errors.append("splits에 null이 있다")
    if splits.height != splits["traj_id"].n_unique():
        errors.append("splits에 traj_id 중복")
    df = traj.select(INPUT_COLUMNS).join(splits, on="traj_id", how="full", coalesce=True)
    if df.height != traj.height or df.height != splits.height or df["split"].null_count() or df["instance_id"].null_count():
        errors.append(f"traj_id가 일대일로 대응하지 않는다 (궤적 {traj.height}, 분할 {splits.height}, 합집합 {df.height})")
        raise ValueError("; ".join(errors))
    bad_split = sorted(set(df["split"].unique()) - set(SPLITS))
    if bad_split:
        errors.append(f"모르는 split 값: {bad_split}")

    expected = assign_splits(traj.select(INPUT_COLUMNS)).select("traj_id", expected_split="split")
    wrong = df.join(expected, on="traj_id").filter(pl.col("split") != pl.col("expected_split")).height
    if wrong:
        errors.append(f"규칙과 다른 split {wrong}개")

    train = df.filter(pl.col("split") == "train")
    test = df.filter(pl.col("split") != "train")
    if (test["fold"] != HELDOUT_FOLD).any():
        errors.append(f"테스트 행의 fold는 {HELDOUT_FOLD}이어야 한다")
    if train.height and not train["fold"].is_between(0, n_folds - 1).all():
        errors.append(f"train fold는 0..{n_folds - 1}이어야 한다")
    if train.height and train["fold"].n_unique() != n_folds:
        errors.append(f"train fold가 {n_folds}개가 아니다: {sorted(train['fold'].unique())}")
    if (train["model_name"] != TRAIN_MODEL).any() or (df.filter(pl.col("split") == "test_same")["model_name"] != TRAIN_MODEL).any():
        errors.append("train·test_same에 70B가 아닌 궤적이 있다")
    if (df.filter(pl.col("split") == "test_cross")["model_name"] == TRAIN_MODEL).any():
        errors.append("test_cross에 70B 궤적이 있다")

    train_issues, test_issues = set(train["instance_id"]), set(test["instance_id"])
    overlap = train_issues & test_issues
    if overlap:
        errors.append(f"train·test 이슈 교집합 {len(overlap)}개")
    if test_issues != set(heldout_issues(traj)):
        errors.append("테스트 이슈가 8B·405B 시도 이슈와 다르다")
    fold_per_issue = train.group_by("instance_id").agg(pl.col("fold").n_unique())
    crossing = fold_per_issue.filter(pl.col("fold") > 1).height
    if crossing:
        errors.append(f"여러 fold에 걸친 학습 이슈 {crossing}개")
    if errors:
        raise ValueError("; ".join(errors))

    return {
        "trajectories": df.height,
        "split_trajectories": {s: df.filter(pl.col("split") == s).height for s in SPLITS},
        "train_issues": len(train_issues),
        "test_issues": len(test_issues),
        "train_test_issue_overlap": len(overlap),
        "train_issues_crossing_folds": crossing,
        "heldout_fold": HELDOUT_FOLD,
    }


def fold_balance(train: pl.DataFrame, fold: np.ndarray) -> pl.DataFrame:
    """fold별 궤적 수·이슈 수·성공률·성공 궤적이 있는 이슈 수."""
    return (
        train.with_columns(fold=pl.Series(fold, dtype=pl.Int8))
        .group_by("fold")
        .agg(
            n_traj=pl.len(),
            n_issues=pl.col("instance_id").n_unique(),
            success_rate=pl.col("target").mean(),
            issues_with_success=pl.col("instance_id").filter(pl.col("target")).n_unique(),
        )
        .sort("fold")
    )


def repo_of(instance_id: pl.Expr) -> pl.Expr:
    """'owner__repo-123' → 'owner__repo'."""
    return instance_id.str.replace(r"-\d+$", "")


def compare_issue_sets(traj: pl.DataFrame, splits: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """학습 이슈 vs 테스트 이슈의 구성 비교 (2.4).

    공정한 비교를 위해 두 쪽 모두 70B 궤적만 쓴다 (train vs test_same).
    traj에는 traj_id, instance_id, model_name, target, n_steps가 있어야 한다.
    돌려주는 것: (지표 표 metric/train/test, 저장소 구성 표)
    """
    df = traj.join(splits, on="traj_id").filter(pl.col("model_name") == TRAIN_MODEL)
    df = df.with_columns(side=pl.when(pl.col("split") == "train").then(pl.lit("train")).otherwise(pl.lit("test")))
    issues = df.group_by("side", "instance_id").agg(
        n=pl.len(), sr=pl.col("target").mean(), med_steps=pl.col("n_steps").median()
    ).with_columns(repo=repo_of(pl.col("instance_id")))

    rows = {}
    for side in ("train", "test"):
        d, i = df.filter(pl.col("side") == side), issues.filter(pl.col("side") == side)
        rows[side] = {
            "issues": i.height,
            "repos": i["repo"].n_unique(),
            "trajectories_70b": d.height,
            "traj_per_issue_median": i["n"].median(),
            "traj_per_issue_mean": i["n"].mean(),
            "traj_success_rate": d["target"].mean(),
            "issue_success_rate_mean": i["sr"].mean(),
            "issue_success_rate_median": i["sr"].median(),
            "issues_never_success_share": (i["sr"] == 0).mean(),
            "issues_always_success_share": (i["sr"] == 1).mean(),
            "issues_mixed_share": ((i["sr"] > 0) & (i["sr"] < 1)).mean(),
            "n_steps_median": d["n_steps"].median(),
            "n_steps_mean": d["n_steps"].mean(),
            "n_steps_p90": d["n_steps"].quantile(0.9),
            "n_steps_median_success": d.filter(pl.col("target"))["n_steps"].median(),
            "n_steps_median_failure": d.filter(~pl.col("target"))["n_steps"].median(),
        }
    repos = (
        issues.group_by("repo").agg(
            train_issues=(pl.col("side") == "train").sum(),
            test_issues=(pl.col("side") == "test").sum(),
        )
        .with_columns(
            train_share=pl.col("train_issues") / rows["train"]["issues"],
            test_share=pl.col("test_issues") / rows["test"]["issues"],
        )
        .sort(["test_issues", "train_issues", "repo"], descending=[True, True, False])
    )
    shared = repos.filter((pl.col("train_issues") > 0) & (pl.col("test_issues") > 0)).height
    tvd = 0.5 * (repos["train_share"] - repos["test_share"]).abs().sum()
    rows["train"]["repos_only_this_side"] = repos.filter(pl.col("test_issues") == 0).height
    rows["test"]["repos_only_this_side"] = repos.filter(pl.col("train_issues") == 0).height
    for side in rows:
        rows[side]["repos_shared"] = shared
        rows[side]["repo_share_total_variation"] = tvd
    metrics = pl.DataFrame({
        "metric": list(rows["train"]),
        "train": [float(v) for v in rows["train"].values()],
        "test": [float(rows["test"][k]) for k in rows["train"]],
    })
    return metrics, repos
