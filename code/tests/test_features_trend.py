"""Phase 5c 가중·변화율·감쇠·기울기 피처 (afp.features.trend_features) 단위·누수 테스트. PLAN "5c": 테스트는 5b와 같다 (①~⑤)."""
import numpy as np
import polars as pl
import pytest

from afp import features
from feature_helpers import features_of, random_steps
from test_features_window import one_traj, values_of


def trow(steps, trajs, n, **kw):
    f = features.trend_features(values_of(steps, trajs), trajs, n, **kw)
    assert f.height == 1
    return f.row(0, named=True)


HAND = dict(
    #         1         2         3         4     5         6         7     8
    command=["open a", "edit 1", "edit 1", "ls", "open a", "edit 2", "ls", "ls"],
    command_name=["open", "edit", "edit", "ls", "open", "edit", "ls", "ls"],
    obs_len=[1, 2, 3, 4, 5, 6, 7, 8],
    is_error=[False, False, True, False, True, False, True, True],
    edit_failed=[False, False, True, False, False, True, False, False],
    open_file=["/r/a.py", "/r/a.py", "/r/a.py", None, "/r/b.py", "/r/b.py", "/r/c.py", "/r/a.py"])


def test_trend_features_by_hand():
    """test_features_window의 8스텝 궤적. N=8, k=3: 이전 = 스텝 1..5, 창 = 6..8."""
    steps, trajs = one_traj(8, **HAND)
    r = trow(steps, trajs, 8)
    # 에러: 이전 {3, 5} = 2, 창 {7, 8} = 2
    assert r["n_error_a2_w3"] == 2 + 2 * 2 and r["n_error_a8_w3"] == 2 + 8 * 2
    assert r["n_error_chg_w3"] == pytest.approx(2 / 3 - 2 / 5)
    # 새 파일: a.py(1), b.py(5) 이전 2, c.py(7) 창 1 (P5c-D3: 이전 값 = 창 시작 전에 처음 연 파일 수)
    assert r["n_new_files_a4_w3"] == 2 + 4 * 1
    assert r["n_new_files_chg_w3"] == pytest.approx(1 / 3 - 2 / 5)
    # 성공 편집: 스텝 2뿐 (이전 1, 창 0). 연속 반복: 스텝 3(이전), 8(창). 아무 반복: 스텝 3·5(이전), 7·8(창)
    assert r["n_edit_ok_a2_w3"] == 1 and r["n_repeat_consec_a2_w3"] == 1 + 2 and r["n_repeat_any_a2_w3"] == 2 + 2 * 2
    # obs_len: 이전 합 15 (평균 3), 창 합 21 (평균 7)
    assert r["obs_len_mean_a2_w3"] == pytest.approx((15 + 2 * 21) / (5 + 2 * 3))
    assert r["obs_len_mean_chg_w3"] == pytest.approx(7 - 3)
    # 감쇠: 에러 스텝 3, 5, 7, 8 → γ^5 + γ^3 + γ^1 + γ^0
    g = 0.5
    assert r["n_error_g05"] == pytest.approx(g ** 5 + g ** 3 + g + 1)
    assert r["obs_len_sum_g07"] == pytest.approx(sum(0.7 ** (8 - t) * t for t in range(1, 9)))
    # 기울기: obs_len = 스텝 번호라 1
    assert r["obs_len_slope_w3"] == pytest.approx(1) and r["obs_len_slope_w5"] == pytest.approx(1)


def test_slope_matches_numpy_polyfit():
    steps, trajs = one_traj(9, obs_len=[3, 1, 4, 1, 5, 9, 2, 6, 5])
    r = trow(steps, trajs, 9)
    for k in (3, 5):
        y = [3, 1, 4, 1, 5, 9, 2, 6, 5][9 - k:]
        assert r[f"obs_len_slope_w{k}"] == pytest.approx(np.polyfit(np.arange(10 - k, 10), y, 1)[0])


def test_trend_matches_window_and_m1_totals():
    """(a)·(c)의 창·이전 값이 5b 창 피처, M1 합계(n_error 등)와 맞는다: 이전 = 합계 − 창."""
    steps, trajs, _ = random_steps(3)
    v = values_of(steps, trajs)
    for n in (5, 8, 10):
        t = features.trend_features(v, trajs, n)
        w = features.window_features(v, trajs, n, ks=(3,), recency=False)
        m = features_of(steps, trajs, [n]).drop("N")
        j = t.join(w, on="traj_id").join(m, on="traj_id")
        assert j.height == t.height == w.height == m.height
        for base, total in (("n_error", "n_error"), ("n_edit_failed", "n_edit_failed"), ("n_edit_ok", "n_edit_ok"),
                            ("n_repeat_consec", "n_repeat_consec"), ("n_repeat_any", "n_repeat_any")):
            win, a2 = j[f"{base}_w3"].to_numpy(), j[f"{base}_a2_w3"].to_numpy()
            prev = j[total].to_numpy() - win
            np.testing.assert_allclose(a2, prev + 2 * win)
            np.testing.assert_allclose(j[f"{base}_chg_w3"].to_numpy(), win / 3 - prev / (n - 3))
        # 새 파일: 이전 + 창 = 열어 본 고유 파일 수
        np.testing.assert_allclose(j["n_new_files_a2_w3"] - j["n_new_files_w3"], j["n_unique_files"])
        # obs_len: 창 평균은 5b와 같고, 이전 평균 = (N × 전체 평균 − 3 × 창 평균) / (N − 3)
        prev_mean = (n * j["obs_len_mean"] - 3 * j["obs_len_mean_w3"]) / (n - 3)
        np.testing.assert_allclose(j["obs_len_mean_chg_w3"], j["obs_len_mean_w3"] - prev_mean, rtol=1e-12, atol=1e-9)


def test_trend_columns_absent_when_k_ge_n():
    """② k ≥ N이면 그 k의 (a)·(c)·기울기 열이 없다. 감쇠는 N=3에도 있다."""
    steps, trajs, _ = random_steps(0)
    v = values_of(steps, trajs)
    decay = [c for g in features.GAMMAS for c in features.decay_columns(g)]
    for n, ks, sks in ((3, [], []), (5, [3], [3]), (8, [3], [3, 5])):
        f = features.trend_features(v, trajs, n)
        want = ([c for a in features.ALPHAS for c in features.alpha_columns(n, ks, a)] + features.change_columns(n, ks)
                + decay + features.slope_columns(n, sks))
        assert f.columns == ["traj_id", *want], n
    assert features.slope_columns(5, (5,)) == [] and features.alpha_columns(3, (3,), 2) == []


@pytest.mark.parametrize("k", (3, 5))
def test_trend_window_boundary(k):
    """③ 창 경계: 에러가 스텝 N−k에만 있으면 창 0 (이전 1), N−k+1이나 N에 있으면 창 1 (이전 0)."""
    n = 10
    for err_step, win in ((n - k, 0), (n - k + 1, 1), (n, 1)):
        steps, trajs = one_traj(n, is_error=[s == err_step for s in range(1, n + 1)])
        r = trow(steps, trajs, n, ks=(k,), alphas=(4,))
        assert r[f"n_error_a4_w{k}"] == (1 - win) + 4 * win, err_step
        assert r[f"n_error_chg_w{k}"] == pytest.approx(win / k - (1 - win) / (n - k)), err_step
    # 기울기 경계: 스텝 N−k의 관측 길이를 바꿔도 창 k 기울기는 같다
    a = trow(*one_traj(n, obs_len=list(range(n))), n)
    b = trow(*one_traj(n, obs_len=[999 if s == n - k else s - 1 for s in range(1, n + 1)]), n)
    assert a[f"obs_len_slope_w{k}"] == b[f"obs_len_slope_w{k}"]


def test_trend_without_events():
    """④ 사건이 없으면 횟수 계열은 모두 0, 관측 길이가 일정하면 변화율·기울기 0."""
    for n in (3, 5, 10):
        r = trow(*one_traj(n), n)
        for c, val in r.items():
            if c == "traj_id" or "obs_len" in c:
                continue
            assert val == 0, (n, c)
        for c in (c for c in r if c.startswith("obs_len_mean_chg") or c.startswith("obs_len_slope")):
            assert r[c] == pytest.approx(0), (n, c)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_trend_features_ignore_steps_after_n(seed):
    """① 스텝 N 이후를 무작위로 바꾸거나 지우거나 덧붙여도 값이 같다. 궤적 길이도 피처에 들어가지 않는다."""
    steps, trajs, rng = random_steps(seed)
    for n in (3, 5, 8, 10, 15):
        base = features.trend_features(values_of(steps, trajs), trajs, n)
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
            assert features.trend_features(values_of(s_, t_), t_, n).equals(base), n


def test_trend_features_pass_check_features():
    """⑤ 새 피처를 붙여도 check_features(진행 중 궤적과 1:1, 결측·무한값 없음)를 통과한다."""
    steps, trajs, _ = random_steps(7)
    n_grid = [3, 5, 8, 10]
    base = features_of(steps, trajs, n_grid)
    v = values_of(steps, trajs)
    parts = [base.filter(pl.col("N") == n).join(features.trend_features(v, trajs, n), on="traj_id", how="left") for n in n_grid]
    joined = pl.concat(parts, how="diagonal")

    def cols(n):
        return [c for c in features.trend_features(v, trajs, n).columns if c != "traj_id"]
    assert features.check_features(joined, trajs, n_grid, columns=cols) == features.check_features(base, trajs, n_grid)
    with pytest.raises(ValueError, match="결측"):  # N=3에 기울기 k=3 열을 요구하면 (없어야 할 열) 걸린다
        features.check_features(joined, trajs, [3], columns=[features.slope_name(3)])


def test_names():
    assert features.decay_name("obs_len_mean", 0.7) == "obs_len_sum_g07"
    assert features.decay_name("n_error", 0.5) == "n_error_g05"
    assert features.alpha_name("n_error", 8, 3) == "n_error_a8_w3"
