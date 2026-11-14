"""Phase 5b 창·최근성 피처 (afp.features.window_features) 단위·누수 테스트. PLAN "5b" 테스트 ①~⑤."""
import polars as pl
import pytest

from afp import features
from feature_helpers import STEP_SCHEMA, features_of, random_steps


def values_of(steps, trajs):
    m = features.mentions_table(trajs)
    return features.step_values(steps, m, trajs.select("traj_id", "instance_id"))


def one_traj(n_steps, **cols):
    """궤적 하나 (스텝 1..n_steps). cols로 스텝 값 목록을 덮어쓴다. 기본: ls 명령 (서로 다른 원문), 에러 없음, 파일 없음."""
    base = {"traj_id": ["a"] * n_steps, "step_idx": list(range(1, n_steps + 1)),
            "command": [f"ls {i}" for i in range(n_steps)], "command_name": ["ls"] * n_steps, "obs_len": [10] * n_steps,
            "is_error": [False] * n_steps, "edit_failed": [False] * n_steps, "open_file": [None] * n_steps}
    base.update(cols)
    steps = pl.DataFrame(base, schema=STEP_SCHEMA)
    trajs = pl.DataFrame({"traj_id": ["a"], "instance_id": ["i"], "issue_text": ["no file"], "n_steps": [n_steps + 5]})
    return steps, trajs


def row(steps, trajs, n, ks=features.WINDOW_KS, recency=True):
    f = features.window_features(values_of(steps, trajs), trajs, n, ks, recency)
    assert f.height == 1
    return f.row(0, named=True)


def test_window_features_by_hand():
    #            1      2      3      4      5      6      7      8
    cmds = ["open a", "edit 1", "edit 1", "ls", "open a", "edit 2", "ls", "ls"]  # 스텝 5는 스텝 1과 같은 원문
    names = ["open", "edit", "edit", "ls", "open", "edit", "ls", "ls"]
    steps, trajs = one_traj(
        8, command=cmds, command_name=names, obs_len=[1, 2, 3, 4, 5, 6, 7, 8],
        is_error=[False, False, True, False, True, False, True, True],
        edit_failed=[False, False, True, False, False, True, False, False],
        open_file=["/r/a.py", "/r/a.py", "/r/a.py", None, "/r/b.py", "/r/b.py", "/r/c.py", "/r/a.py"])
    r = row(steps, trajs, 8)
    # 창 k=3 = 스텝 6..8, k=5 = 스텝 4..8
    assert (r["n_error_w3"], r["n_error_w5"]) == (2, 3)
    assert (r["n_edit_failed_w3"], r["n_edit_failed_w5"]) == (1, 1)
    assert (r["n_edit_ok_w3"], r["n_edit_ok_w5"]) == (0, 0)  # 성공 편집은 스텝 2뿐
    assert (r["n_repeat_consec_w3"], r["n_repeat_consec_w5"]) == (1, 1)  # 스텝 8 "ls" = 스텝 7 "ls" (스텝 3은 창 밖)
    assert (r["n_repeat_any_w3"], r["n_repeat_any_w5"]) == (2, 3)  # 스텝 7(= 스텝 4), 8 / + 스텝 5(= 스텝 1, 창 밖 명령과 같아도 센다)
    assert (r["n_new_files_w3"], r["n_new_files_w5"]) == (1, 2)  # c.py(스텝 7) / b.py(5), c.py(7). a.py는 스텝 1에 처음 열림
    assert r["obs_len_mean_w3"] == pytest.approx(7) and r["obs_len_mean_w5"] == pytest.approx(6)
    assert r["steps_since_edit_ok"] == 6  # 8 − 2
    assert r["steps_since_new_file"] == 1  # 8 − 7
    assert r["cur_error_run"] == 2  # 스텝 7, 8


def test_window_columns_absent_when_k_ge_n():
    """② N ≤ k이면 그 k의 창 열이 없다 (N=3 창 없음, N=5는 k=3만). 최근성은 N=3에도 있다."""
    steps, trajs, _ = random_steps(0)
    v = values_of(steps, trajs)
    for n, want_ks in ((3, []), (5, [3]), (8, [3, 5])):
        f = features.window_features(v, trajs, n)
        assert f.columns == ["traj_id", *features.window_columns(n, want_ks), *features.RECENCY_FEATURES]
        assert features.window_columns(n, features.WINDOW_KS) == features.window_columns(n, want_ks)
    assert features.window_columns(3, (3, 5)) == [] and features.window_columns(5, (5,)) == []
    assert features.window_features(v, trajs, 5, ks=(5,), recency=False).columns == ["traj_id"]


@pytest.mark.parametrize("k", features.WINDOW_KS)
def test_window_boundary(k):
    """③ 창 경계: 에러가 스텝 N−k에만 있으면 0, N−k+1에 있으면 1, N에 있으면 1."""
    n = 10
    for err_step, want in ((n - k, 0), (n - k + 1, 1), (n, 1)):
        steps, trajs = one_traj(n, is_error=[s == err_step for s in range(1, n + 1)])
        assert row(steps, trajs, n, ks=(k,))[f"n_error_w{k}"] == want, err_step


def test_recency_without_events_is_n():
    """④ 사건이 없으면 steps_since_* = N. 스텝 N이 에러가 아니면 cur_error_run = 0, 전부 에러면 N."""
    for n in (3, 5, 10):
        steps, trajs = one_traj(n)
        r = row(steps, trajs, n)
        assert (r["steps_since_edit_ok"], r["steps_since_new_file"], r["cur_error_run"]) == (n, n, 0)
        steps, trajs = one_traj(n, is_error=[True] * n, command_name=["edit"] * n, edit_failed=[True] * n)
        r = row(steps, trajs, n)
        assert (r["steps_since_edit_ok"], r["cur_error_run"]) == (n, n)  # 실패 편집은 성공 편집이 아니다
    steps, trajs = one_traj(6, command_name=["ls"] * 5 + ["edit"], open_file=["/r/a.py"] + [None] * 5)
    r = row(steps, trajs, 6)
    assert (r["steps_since_edit_ok"], r["steps_since_new_file"]) == (0, 5)  # 스텝 N의 성공 편집 → 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_window_features_ignore_steps_after_n(seed):
    """① 스텝 N 이후를 무작위로 바꾸거나 지우거나 덧붙여도 새 피처 값이 같다. 궤적 길이도 피처에 들어가지 않는다."""
    steps, trajs, rng = random_steps(seed)
    for n in (3, 5, 8, 10, 15):
        base = features.window_features(values_of(steps, trajs), trajs, n)
        after = pl.col("step_idx") > n
        k = steps.height
        changed = steps.with_columns(
            command=pl.when(after).then(pl.Series([f"x {i}" for i in rng.integers(0, 3, k)])).otherwise(pl.col("command")),
            command_name=pl.when(after).then(pl.lit("edit")).otherwise(pl.col("command_name")),
            obs_len=pl.when(after).then(pl.Series(rng.integers(0, 10 ** 6, k))).otherwise(pl.col("obs_len")),
            is_error=pl.when(after).then(pl.Series(rng.random(k) < 0.5)).otherwise(pl.col("is_error")),
            edit_failed=pl.when(after).then(pl.Series(rng.random(k) < 0.5)).otherwise(pl.col("edit_failed")),
            open_file=pl.when(after).then(pl.Series([f"/r/new{i}.py" for i in rng.integers(0, 9, k)])).otherwise(pl.col("open_file")),
        )
        removed = steps.filter(~after)
        extra = pl.concat([steps, steps.filter(pl.col("step_idx") == 1).with_columns(step_idx=pl.lit(999, dtype=pl.Int64))])
        longer = trajs.with_columns(n_steps=pl.when(pl.col("n_steps") > n).then(pl.col("n_steps") + 100).otherwise(pl.col("n_steps")))
        for s_, t_ in ((changed, trajs), (removed, trajs), (extra, trajs), (steps.sample(fraction=1.0, shuffle=True, seed=seed), trajs),
                       (steps, longer)):
            assert features.window_features(values_of(s_, t_), t_, n).equals(base), n


def test_window_features_pass_check_features():
    """⑤ 새 피처를 붙여도 check_features(진행 중 궤적과 1:1, 결측·무한값 없음)를 통과한다. 없는 열은 검사에서 걸린다."""
    steps, trajs, _ = random_steps(7)
    n_grid = [3, 5, 8, 10]
    base = features_of(steps, trajs, n_grid)
    v = values_of(steps, trajs)
    parts = [base.filter(pl.col("N") == n).join(features.window_features(v, trajs, n), on="traj_id", how="left") for n in n_grid]
    joined = pl.concat(parts, how="diagonal")

    def cols(n):
        return [*features.FEATURES, *features.window_columns(n, features.WINDOW_KS), *features.RECENCY_FEATURES]
    counts = features.check_features(joined, trajs, n_grid, columns=cols)
    assert counts == features.check_features(base, trajs, n_grid)
    with pytest.raises(ValueError, match="결측"):  # N=3에 k=3 창 열을 요구하면 (없어야 할 열이라 null) 걸린다
        features.check_features(joined, trajs, [3], columns=[*features.FEATURES, *features.window_columns(5, (3,))])


def test_proportional_k():
    assert [features.proportional_k(n) for n in (3, 5, 8, 10, 15, 20)] == [None, 2, 2, 3, 4, 5]
