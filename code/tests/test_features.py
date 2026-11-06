"""M1 피처 (Phase 5.1) 단위 테스트. 누수 테스트(5.2)는 test_features_leakage.py."""
import polars as pl
import pytest

from afp import features, metrics, parse
from feature_helpers import STEP_SCHEMA, features_of, random_steps


def toy():
    steps = pl.DataFrame({
        "traj_id": ["a"] * 6, "step_idx": [1, 2, 3, 4, 5, 6],
        "command": ["ls", "ls", " ls ", "open x.py", "ls", "edit 1:2"],
        "command_name": ["ls", "ls", "ls", "open", "ls", "edit"], "obs_len": [10, 0, 5, 7, 3, 2],
        "is_error": [False, True, False, False, True, False], "edit_failed": [False] * 5 + [True],
        "open_file": [None, None, None, "/r/pkg/x.py", "/r/pkg/x.py", "/r/y.py"],
    }, schema=STEP_SCHEMA)
    trajs = pl.DataFrame({"traj_id": ["a"], "instance_id": ["i"], "issue_text": ["bug in pkg/x.py and setup.cfg"], "n_steps": [7]})
    return steps, trajs


def test_step_values_by_hand():
    steps, trajs = toy()
    v = features.step_values(steps, features.mentions_table(trajs), trajs)
    assert v.columns == features.STEP_VALUE_COLUMNS
    assert v["repeat_prev"].to_list() == [False, True, True, False, False, False]  # 공백만 다른 명령도 같은 명령
    assert v["repeat_any"].to_list() == [False, True, True, False, True, False]
    assert v["run_len"].to_list() == [1, 2, 3, 1, 1, 1]
    assert v["opens_issue_file"].to_list() == [False, False, False, True, True, False]


def test_m1_features_by_hand():
    steps, trajs = toy()
    m = features.mentions_table(trajs)
    f = features.m1_features(features.step_values(steps, m, trajs), trajs, [3, 5], m)
    assert f.columns == ["traj_id", "N", *features.FEATURES]
    r3, r5 = f.row(0, named=True), f.row(1, named=True)
    assert (r3["cmd_ls"], r3["n_error"], r3["n_repeat_consec"], r3["max_run_len"], r3["n_unique_files"]) == (3, 1, 2, 3, 0)
    assert (r3["issue_file_opened"], r3["obs_len_last"], r3["last_cmd_ls"], r3["last_is_error"]) == (0, 5, 1, 0)
    assert (r5["cmd_ls"], r5["cmd_open"], r5["n_repeat_any"], r5["n_unique_files"], r5["n_steps_no_file"]) == (4, 1, 3, 1, 3)
    assert (r5["issue_file_opened"], r5["n_issue_file_steps"], r5["issue_mentions_file"], r5["last_is_error"]) == (1, 2, 1, 1)
    assert r5["obs_len_mean"] == pytest.approx(25 / 5) and r5["obs_len_max"] == 10 and r5["obs_len_min"] == 0
    assert set(features.COUNT_FEATURES) <= set(f.columns) and sum(r5[c] for c in features.COUNT_FEATURES) == 5


def test_issue_file_mentions_and_matching():
    assert features.issue_file_mentions("see `pkg/a.py`, utils.py and http://x.org/b.html; v1.2") == ["pkg/a.py", "utils.py"]  # URL 안의 경로는 앞에 "/"가 있어 빠진다
    assert features.issue_file_mentions(None) == []
    steps = pl.DataFrame({"traj_id": ["t"] * 3, "step_idx": [1, 2, 3], "command": ["open a", "open b", "open c"],
                          "command_name": ["open"] * 3, "obs_len": [1, 1, 1], "is_error": [False] * 3, "edit_failed": [False] * 3,
                          "open_file": ["/repo/src/pkg/a.py", "/repo/other/pkg/a.py.bak", "/repo/lib/utils.py"]}, schema=STEP_SCHEMA)
    ti = pl.DataFrame({"traj_id": ["t"], "instance_id": ["i"], "issue_text": ["pkg/a.py and utils.py"]})
    v = features.step_values(steps, features.mentions_table(ti), ti)
    assert v["opens_issue_file"].to_list() == [True, False, True]  # 경로 언급은 끝 경로, 이름만 있으면 파일 이름


def test_step_definitions_reuse_b0_rules():
    """에러·편집 실패는 B0 규칙의 값(steps.parquet의 is_error / edit_failed = parse.is_error / parse.EDIT_FAILED)을 그대로 쓴다."""
    assert "is_error" in features.STEP_INPUT_COLUMNS and "edit_failed" in features.STEP_INPUT_COLUMNS
    assert features.COMMAND_NAMES[:len(parse.COMMANDS)] == parse.COMMANDS
    steps, trajs, _ = random_steps(5)
    f = features_of(steps, trajs, [5])
    want = steps.filter(pl.col("step_idx") <= 5).group_by("traj_id").agg(e=pl.col("is_error").sum(), x=pl.col("edit_failed").sum())
    j = f.join(want, on="traj_id")
    assert (j["n_error"] == j["e"]).all() and (j["n_edit_failed"] == j["x"]).all()


def test_check_features_rejects_bad_rows():
    steps, trajs, _ = random_steps(6)
    f = features_of(steps, trajs, [3])
    assert features.check_features(f, trajs, [3]) == {3: metrics.running(trajs, 3).height}
    with pytest.raises(ValueError, match="1:1"):
        features.check_features(f.head(f.height - 1), trajs, [3])
    with pytest.raises(ValueError, match="무한값"):
        features.check_features(f.with_columns(obs_len_mean=pl.lit(float("inf"))), trajs, [3])
