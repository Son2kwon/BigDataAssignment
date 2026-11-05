"""학습/테스트 분할과 학습 이슈 fold.

- 평가 대상은 70B 궤적뿐이다. 8B·405B 궤적(EXCLUDED_MODELS)은 splits.parquet에 넣지 않는다
- 테스트 이슈 = 70B 이슈 전체를 이슈 단위 StratifiedGroupKFold(TEST_PARTS, shuffle, SEED)로 나눈 조각 중
  TEST_PART번째 (약 1/6). 그룹 = instance_id, 층화 = target. 나머지 이슈의 궤적 → train
- 학습 이슈는 다시 5-fold StratifiedGroupKFold로 나눈다. 테스트 이슈는 학습 fold와 같은 방식으로 뽑힌
  "한 조각 더"라서, fold 사이의 운영점 이전과 train → test 이전을 같은 조건으로 비교할 수 있다
- fold: train은 0..4, test는 HELDOUT_FOLD(-1). 테스트는 어떤 fold에도 속하지 않는다

예전 분할(2026-10-29~11-05)은 "8B·405B가 시도한 578개 이슈"를 테스트로 고정했다. 이 이슈들은 무작위 표본이
아니어서(성공률 18.7% vs train 16.3%, 저장소 구성 차이) train에서 고른 τ가 테스트로 옮겨지지 않는 원인
후보였다 (회의 2026-11-05 §4.3). 그래서 70B만으로 테스트 이슈를 무작위로 다시 뽑는다.

GroupKFold 대신 StratifiedGroupKFold를 쓰는 이유: 이슈 성공률 편차가 커서(76.7%가 성공 0)
GroupKFold로는 fold 성공률이 크게 벌어진다. 비교 수치는 results/phase2/fold_balance.csv.
"""
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from afp.io import SEED

TRAIN_MODEL = "swe-agent-llama-70b"
EXCLUDED_MODELS = ("swe-agent-llama-8b", "swe-agent-llama-405b")  # 분할·평가에서 뺀다
KNOWN_MODELS = (TRAIN_MODEL, *EXCLUDED_MODELS)

SPLITS = ("train", "test")
N_FOLDS = 5
TEST_PARTS = 6  # 70B 이슈를 6조각으로 나눠 한 조각(약 16.7%)을 테스트로
TEST_PART = 0
HELDOUT_FOLD = -1

# 분할에 필요한 컬럼. target은 층화에만 쓰고 splits.parquet에는 넣지 않는다.
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


def in_scope(traj: pl.DataFrame) -> pl.DataFrame:
    """분할 대상 궤적 (70B만), traj_id 순."""
    return traj.filter(pl.col("model_name") == TRAIN_MODEL).sort("traj_id")


def select_test_issues(traj: pl.DataFrame, seed: int = SEED) -> list[str]:
    """테스트 이슈 (정렬). 70B 궤적을 이슈 단위로 층화해 TEST_PARTS조각으로 나눈 것 중 TEST_PART번째.

    입력 행 순서와 무관하다 (traj_id 순으로 정렬한 뒤 나눈다).
    """
    t = in_scope(traj)
    if t["instance_id"].n_unique() < TEST_PARTS:
        raise ValueError(f"70B 이슈 {t['instance_id'].n_unique()}개로는 테스트 조각 {TEST_PARTS}개를 만들 수 없다")
    y, groups = t["target"].to_numpy(), t["instance_id"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=TEST_PARTS, shuffle=True, random_state=seed)
    _, idx = list(splitter.split(np.zeros(len(y)), y, groups))[TEST_PART]
    return sorted(set(groups[idx]))


def assign_splits(traj: pl.DataFrame) -> pl.DataFrame:
    """70B 궤적에 split 컬럼을 붙인다 (8B·405B 행은 빠진다). 모든 행이 정확히 하나의 split을 받는다.

    테스트 이슈는 항상 고정 시드(io.SEED)로 뽑는다. make_splits의 seed는 train fold에만 쓴다.
    """
    validate_input(traj)
    is_test = pl.col("instance_id").is_in(select_test_issues(traj))
    return in_scope(traj).with_columns(split=pl.when(is_test).then(pl.lit("test")).otherwise(pl.lit("train")))


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
    """궤적 표 → splits 표 (traj_id, split, fold). 70B 궤적만 들어간다. 입력 행 순서와 무관하게 같은 결과를 낸다.

    seed는 train fold에만 쓴다 (테스트 이슈는 assign_splits가 고정 시드로 뽑는다).
    """
    validate_input(traj)
    df = assign_splits(traj.select(INPUT_COLUMNS)).sort("traj_id")
    is_train = (df["split"] == "train").to_numpy()
    fold = np.full(df.height, HELDOUT_FOLD, dtype=np.int8)
    fold[is_train] = train_folds(df.filter(pl.col("split") == "train"), n_folds, seed, stratified)
    return df.select("traj_id", "split").with_columns(fold=pl.Series(fold, dtype=pl.Int8))


def validate_splits(traj: pl.DataFrame, splits: pl.DataFrame, n_folds: int = N_FOLDS) -> dict:
    """splits가 분할 규칙을 지키는지 확인하고 크기를 돌려준다. 어기면 ValueError.

    traj는 전체 궤적 표 (8B·405B 포함). splits에는 70B 궤적이 정확히 한 번씩, 다른 모델 궤적은 하나도 없어야 한다.
    """
    validate_input(traj)
    errors = []
    if dict(splits.schema) != SPLIT_SCHEMA:
        errors.append(f"스키마가 다르다: {dict(splits.schema)} != {SPLIT_SCHEMA}")
    if splits.null_count().sum_horizontal().item():
        errors.append("splits에 null이 있다")
    if splits.height != splits["traj_id"].n_unique():
        errors.append("splits에 traj_id 중복")
    scope = in_scope(traj.select(INPUT_COLUMNS))
    excluded = traj.filter(pl.col("model_name") != TRAIN_MODEL)
    n_excluded_in_splits = splits.join(excluded.select("traj_id"), on="traj_id", how="semi").height
    if n_excluded_in_splits:
        errors.append(f"splits에 8B·405B 궤적 {n_excluded_in_splits}개")
    df = scope.join(splits, on="traj_id", how="full", coalesce=True)
    if df.height != scope.height or df.height != splits.height or df["split"].null_count() or df["instance_id"].null_count():
        errors.append(f"70B traj_id가 일대일로 대응하지 않는다 (70B 궤적 {scope.height}, 분할 {splits.height}, 합집합 {df.height})")
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

    train_issues, test_issue_set = set(train["instance_id"]), set(test["instance_id"])
    overlap = train_issues & test_issue_set
    if overlap:
        errors.append(f"train·test 이슈 교집합 {len(overlap)}개")
    if test_issue_set != set(select_test_issues(traj)):
        errors.append("테스트 이슈가 무작위 추출 규칙(select_test_issues)과 다르다")
    fold_per_issue = train.group_by("instance_id").agg(pl.col("fold").n_unique())
    crossing = fold_per_issue.filter(pl.col("fold") > 1).height
    if crossing:
        errors.append(f"여러 fold에 걸친 학습 이슈 {crossing}개")
    if errors:
        raise ValueError("; ".join(errors))

    return {
        "trajectories": df.height,
        "excluded_trajectories": {m: excluded.filter(pl.col("model_name") == m).height for m in EXCLUDED_MODELS},
        "split_trajectories": {s: df.filter(pl.col("split") == s).height for s in SPLITS},
        "train_issues": len(train_issues),
        "test_issues": len(test_issue_set),
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

    두 쪽 모두 70B 궤적만 쓴다 (splits에 70B만 있음).
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
