"""Phase 8.1 다른 모델 궤적 평가 집합 (afp/cross.py)과 evaluate.operating_table의 test_splits."""
import polars as pl
import pytest

from afp import cross, evaluate, split


def population():
    """70B: 이슈 a·b는 train, c·d는 test. 8B·405B는 a·c·d 일부."""
    rows = [("t70_0", "a", "swe-agent-llama-70b"), ("t70_1", "b", "swe-agent-llama-70b"), ("t70_2", "c", "swe-agent-llama-70b"),
            ("t70_3", "d", "swe-agent-llama-70b"),
            ("x8_0", "a", "swe-agent-llama-8b"), ("x8_1", "c", "swe-agent-llama-8b"), ("x8_2", "d", "swe-agent-llama-8b"),
            ("x405_0", "a", "swe-agent-llama-405b"), ("x405_1", "d", "swe-agent-llama-405b")]
    trajs = pl.DataFrame({"traj_id": [r[0] for r in rows], "instance_id": [r[1] for r in rows], "model_name": [r[2] for r in rows],
                          "target": [i % 2 == 0 for i in range(len(rows))], "n_steps": [10 + i for i in range(len(rows))]})
    splits = pl.DataFrame({"traj_id": ["t70_0", "t70_1", "t70_2", "t70_3"], "split": ["train", "train", "test", "test"],
                           "fold": pl.Series([0, 1, split.HELDOUT_FOLD, split.HELDOUT_FOLD], dtype=pl.Int8)})
    return trajs, splits


def test_cross_sets_split_by_70b_issue_split():
    trajs, splits = population()
    s = cross.cross_sets(trajs, splits)
    assert tuple(s) == cross.CROSS_SETS
    assert s["x8b_unseen"]["traj_id"].to_list() == ["x8_1", "x8_2"]
    assert s["x405b_unseen"]["traj_id"].to_list() == ["x405_1"]
    assert s["x8b_all"]["traj_id"].to_list() == ["x8_0", "x8_1", "x8_2"]
    assert s["x405b_all"]["traj_id"].to_list() == ["x405_0", "x405_1"]
    for t in s.values():
        assert t.columns == ["traj_id", "instance_id", "target", "n_steps", "fold"]
        assert (t["fold"] == split.HELDOUT_FOLD).all()


@pytest.mark.parametrize("case", ["in_splits", "orphan_issue", "issue_in_both_splits", "missing_model"])
def test_cross_sets_rejects_invalid_inputs(case):
    trajs, splits = population()
    if case == "in_splits":
        splits = pl.concat([splits, pl.DataFrame({"traj_id": ["x8_0"], "split": ["train"], "fold": pl.Series([0], dtype=pl.Int8)})])
    elif case == "orphan_issue":
        trajs = trajs.with_columns(pl.when(pl.col("traj_id") == "x8_0").then(pl.lit("z")).otherwise("instance_id").alias("instance_id"))
    elif case == "issue_in_both_splits":
        trajs = trajs.with_columns(pl.when(pl.col("traj_id") == "t70_1").then(pl.lit("c")).otherwise("instance_id").alias("instance_id"))
    else:
        trajs = trajs.filter(pl.col("model_name") != "swe-agent-llama-405b")
    with pytest.raises(ValueError):
        cross.cross_sets(trajs, splits)


def test_describe_counts_running_runs():
    trajs, splits = population()
    d = cross.describe(cross.cross_sets(trajs, splits), (14,))
    row = d.filter(pl.col("set") == "x8b_all").row(0, named=True)
    assert (row["n_traj"], row["n_issues"]) == (3, 3)
    assert (row["running_N14"], row["running_success_N14"]) == (2, 0.5)  # n_steps 15(실패)·16(성공)만 > 14


def test_operating_table_attaches_ci_to_named_sets():
    boots = {("m", "x8b_unseen", 5): pl.DataFrame({"stat": ["single.lost_success", "single.d_saved_steps"], "estimate": [0.02, 0.1],
                                                    "lo": [0.0, 0.05], "hi": [0.06, 0.15]})}
    rows = [{"model": "m", "policy": "single", "N": 5, "tau": 0.9, "cap": None, "split": s, "stop_rate": 0.1, "lost_success": 0.02,
             "saved_steps": 0.2, "saved_tokens": 0.2} for s in ("x8b_unseen", "train")]
    default = evaluate.operating_table(rows, boots)
    named = evaluate.operating_table(rows, boots, test_splits=("x8b_unseen",))
    assert default["lost_success_hi"].null_count() == 2
    x = named.filter(pl.col("split") == "x8b_unseen").row(0, named=True)
    assert (x["lost_success_hi"], x["d_saved_steps"], x["d_saved_steps_lo"]) == (0.06, 0.1, 0.05)
    assert x["meets_judge_ci_hi"] is False
    assert named.filter(pl.col("split") == "train")["lost_success_hi"].null_count() == 1
