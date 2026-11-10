"""Phase 8.1 보조 실험: 70B로 학습한 모델을 다른 모델(8B·405B) 궤적에 그대로 적용할 평가 집합.

주 평가(70B train / test)는 바꾸지 않는다. 8B·405B 궤적은 splits.parquet에 없고(2026-11-05 결정) 여기서만 읽는다.
8B·405B가 시도한 이슈는 모두 70B도 시도했다. 그래서 이슈 기준으로 두 종류로 나눈다.
- <모델>_unseen: 70B **test** 이슈의 궤적. M1 fold 모델이 이 이슈의 70B 궤적을 학습에 쓰지 않았다 (주 결과)
- <모델>_all: 그 모델의 궤적 전부. 70B train 이슈가 섞여 있어서, 모델이 같은 이슈의 70B 궤적을 학습에 썼다 (참고)
이 집합에서는 아무것도 고르지 않는다. τ·K·K*·하이퍼파라미터는 모두 70B train에서 고른 값을 그대로 쓴다.
"""
import polars as pl

from afp import io, metrics, split

MODEL_KEYS = {"x8b": "swe-agent-llama-8b", "x405b": "swe-agent-llama-405b"}
SCOPES = ("unseen", "all")
CROSS_SETS = tuple(f"{k}_{s}" for s in SCOPES for k in MODEL_KEYS)  # x8b_unseen, x405b_unseen, x8b_all, x405b_all


def cross_sets(trajs: pl.DataFrame, splits: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """trajs: traj_id, instance_id, model_name, target, n_steps (모든 모델). splits: 70B traj_id, split, fold.

    출력: {집합 이름: traj_id, instance_id, target, n_steps, fold(= split.HELDOUT_FOLD)} (traj_id 정렬, 하네스 trajs 형식).
    """
    if set(MODEL_KEYS.values()) != set(split.EXCLUDED_MODELS):
        raise AssertionError("MODEL_KEYS가 split.EXCLUDED_MODELS와 다르다")
    issue_split = (trajs.join(splits.select("traj_id", "split"), on="traj_id", how="inner")
                   .group_by("instance_id").agg(pl.col("split").unique()))
    if issue_split.filter(pl.col("split").list.len() > 1).height:
        raise ValueError("70B train과 test에 함께 있는 이슈가 있다")
    issue_split = issue_split.with_columns(pl.col("split").list.first())
    test_issues = issue_split.filter(pl.col("split") == "test").select("instance_id")
    out = {}
    for key, name in MODEL_KEYS.items():
        t = trajs.filter(pl.col("model_name") == name)
        if t.height == 0:
            raise ValueError(f"{name} 궤적이 없다")
        if t.join(splits.select("traj_id"), on="traj_id", how="semi").height:
            raise ValueError(f"{name} 궤적이 splits.parquet에 있다 (8B·405B는 분할에 넣지 않는다)")
        orphan = t.join(issue_split.select("instance_id"), on="instance_id", how="anti").height
        if orphan:
            raise ValueError(f"{name}: 70B가 시도하지 않은 이슈의 궤적 {orphan}개")
        t = t.select(metrics.TRAJ_COLUMNS).with_columns(fold=pl.lit(split.HELDOUT_FOLD, dtype=pl.Int8)).sort("traj_id")
        out[f"{key}_unseen"] = t.join(test_issues, on="instance_id", how="semi")
        out[f"{key}_all"] = t
    return {name: out[name] for name in CROSS_SETS}


def load_cross_sets() -> dict[str, pl.DataFrame]:
    trajs = io.load_trajectories([*metrics.TRAJ_COLUMNS, "model_name"])
    return cross_sets(trajs, io.load_splits())


def describe(sets: dict[str, pl.DataFrame], n_grid) -> pl.DataFrame:
    """집합별 궤적·이슈 수, 성공률, 스텝 수 중앙값, 시점 N 진행 중 궤적 수와 그 성공률 (모델 점수와 무관)."""
    rows = []
    for name, t in sets.items():
        row = {"set": name, "n_traj": t.height, "n_issues": t["instance_id"].n_unique(), "success_rate": float(t["target"].mean()),
               "median_steps": float(t["n_steps"].median())}
        for n in n_grid:
            r = metrics.running(t, n)
            row[f"running_N{n}"] = r.height
            row[f"running_success_N{n}"] = float(r["target"].mean()) if r.height else None
        rows.append(row)
    return pl.DataFrame(rows)
