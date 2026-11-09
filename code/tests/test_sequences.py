"""M2 시퀀스 입력 (Phase 6.1): N 뒤 스텝·마지막 스텝을 보지 않음, M1 피처와 같은 정의, B1 위치 채우기, 실데이터."""
import numpy as np
import polars as pl
import pytest

from afp import evaluate, features, io, metrics, sequences
from feature_helpers import random_steps


def seq_inputs(seed, n_traj=60, max_steps=28):
    steps, trajs, rng = random_steps(seed, n_traj=n_traj, max_steps=max_steps)
    trajs = trajs.with_columns(target=pl.Series(rng.random(trajs.height) < 0.3),
                               fold=pl.col("instance_id").str.slice(1).cast(pl.Int64) % 5)
    return steps, trajs


def build(steps, trajs, max_n=sequences.MAX_N):
    m = features.mentions_table(trajs)
    values = features.step_values(steps, m, trajs.select("traj_id", "instance_id"))
    return sequences.build(trajs, values, m, max_n)


def arrays(d, n=None):
    sl = slice(None) if n is None else slice(0, n)
    return [d.command[:, sl], d.step[:, sl], d.cum[:, sl], d.static]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sequences_ignore_steps_after_n(seed):
    """N 뒤의 스텝을 바꿔도 위치 1..N의 입력이 같다 (명령·스텝 값·누적 집계·정적 값)."""
    steps, trajs = seq_inputs(seed)
    base = build(steps, trajs)
    for n in (3, 5, 10, 15):
        after = pl.col("step_idx") > n
        changed = steps.with_columns(
            command=pl.when(after).then(pl.lit("submit")).otherwise(pl.col("command")),
            command_name=pl.when(after).then(pl.lit("submit")).otherwise(pl.col("command_name")),
            obs_len=pl.when(after).then(pl.lit(10 ** 6)).otherwise(pl.col("obs_len")),
            is_error=pl.when(after).then(pl.lit(True)).otherwise(pl.col("is_error")),
            edit_failed=pl.when(after).then(pl.lit(True)).otherwise(pl.col("edit_failed")),
            open_file=pl.when(after).then(pl.lit("/r/pkg/f0.py")).otherwise(pl.col("open_file")),
        )
        other = build(changed, trajs)
        assert list(other.traj_id) == list(base.traj_id)
        for a, b in zip(arrays(base, n), arrays(other, n)):
            assert np.array_equal(a, b), n
        # 앞 N스텝만 넣고 만든 입력(max_n=N)도 같다 (시점 N에 진행 중인 궤적)
        only = build(steps.filter(~after), trajs.with_columns(n_steps=pl.min_horizontal("n_steps", pl.lit(n + 1))), max_n=n)
        r = base.running(n)
        idx = {t: i for i, t in enumerate(only.traj_id)}
        j = np.array([idx[t] for t in base.traj_id[r]])
        for a, b in zip(arrays(base, n), arrays(only)):
            assert np.array_equal(a[r], b[j]), n


def test_last_step_is_not_input():
    """궤적의 마지막 스텝(n_steps번째)은 어느 시점에도 진행 중 예측에 쓰이지 않으므로 입력에 넣지 않는다."""
    steps, trajs = seq_inputs(3)
    base = build(steps, trajs)
    assert np.array_equal(base.lengths, np.minimum(base.n_steps - 1, sequences.MAX_N))
    last = pl.col("step_idx") == pl.col("n_steps")
    s = steps.join(trajs.select("traj_id", "n_steps"), on="traj_id").with_columns(
        command_name=pl.when(last).then(pl.lit("submit")).otherwise(pl.col("command_name")),
        is_error=pl.when(last).then(pl.lit(True)).otherwise(pl.col("is_error"))).drop("n_steps")
    other = build(s, trajs)
    for a, b in zip(arrays(base), arrays(other)):
        assert np.array_equal(a, b)
    assert (base.command[~base.valid()] == sequences.PAD_COMMAND).all()
    assert (base.step[~base.valid()] == 0).all() and (base.cum[~base.valid()] == 0).all()


def test_cumulative_inputs_equal_m1_features():
    """위치 t의 누적 입력 = 시점 t의 M1 피처(log1p). 스텝 값은 features.step_values 그대로."""
    steps, trajs = seq_inputs(4)
    d = build(steps, trajs)
    m = features.mentions_table(trajs)
    values = features.step_values(steps, m, trajs.select("traj_id", "instance_id"))
    idx = {t: i for i, t in enumerate(d.traj_id)}
    for n in (1, 4, 10, 20):
        f = features.m1_features(values, trajs, [n], m)
        i = np.array([idx[t] for t in f["traj_id"]])
        assert np.allclose(d.cum[i, n - 1], np.log1p(f.select(sequences.CUM_FEATURES).to_numpy()).astype(np.float32))
        assert np.array_equal(d.static[i], f.select(sequences.STATIC_FEATURES).to_numpy().astype(np.float32))
        v = values.filter(pl.col("step_idx") == n).join(f.select("traj_id"), on="traj_id", how="semi").sort("traj_id")
        j = np.array([idx[t] for t in v["traj_id"]])
        assert np.array_equal(d.command[j, n - 1], v["command_name"].replace_strict(sequences.COMMAND_INDEX).to_numpy())
        assert np.array_equal(d.step[j, n - 1, sequences.STEP_FEATURES.index("is_error")], v["is_error"].cast(pl.Float32).to_numpy())


def test_new_file_and_edit_ok():
    rows = [("t0", 1, "open a", "open", 5, False, False, "/r/a.py"), ("t0", 2, "edit", "edit", 5, False, True, "/r/a.py"),
            ("t0", 3, "edit", "edit", 5, False, False, "/r/a.py"), ("t0", 4, "open b", "open", 5, False, False, "/r/b.py"),
            ("t0", 5, "ls", "ls", 5, False, False, None), ("t0", 6, "open a", "open", 5, False, False, "/r/a.py")]
    steps = pl.DataFrame(rows, schema=features.STEP_INPUT_COLUMNS, orient="row").with_columns(
        pl.col("step_idx").cast(pl.Int64), pl.col("obs_len").cast(pl.Int64))
    v = sequences.step_inputs(features.step_values(steps))
    assert v["new_file"].to_list() == [1, 0, 0, 1, 0, 0]
    assert v["edit_ok"].to_list() == [0, 0, 1, 0, 0, 0]
    assert v["has_file"].to_list() == [1, 1, 1, 1, 0, 1]


def test_b1_matrix_forward_fill():
    """위치 t = 그리드에서 t 이하 가장 큰 N의 B1 점수. 그리드 시점에서는 그 N 점수 그대로, 진행 중이 아닌 위치는 NaN."""
    steps, trajs = seq_inputs(5)
    d = build(steps, trajs)
    grid = evaluate.N_GRID_ISSUE
    rows = [{"traj_id": t, "N": n, "p_fail": 0.5 + n / 100 + i / 1e4}
            for i, (t, ns) in enumerate(zip(trajs["traj_id"], trajs["n_steps"])) for n in grid if ns > n]
    b1 = pl.DataFrame(rows)
    m = sequences.b1_matrix(d, b1)
    for i, t in enumerate(d.traj_id):
        k = int(t[1:])
        for pos in range(1, d.lengths[i] + 1):
            g = max(n for n in grid if n <= pos)
            assert m[i, pos - 1] == pytest.approx(0.5 + g / 100 + k / 1e4, abs=1e-6)
    assert np.isnan(m[~d.valid()]).all()
    with pytest.raises(ValueError):
        sequences.b1_matrix(d, b1.filter(pl.col("N") != 5))


def test_preds_frame_matches_running_trajectories():
    steps, trajs = seq_inputs(6)
    d = build(steps, trajs)
    p = sequences.preds_frame(d, np.random.default_rng(0).random((d.n, d.max_n)))
    for n in evaluate.N_GRID:
        a = metrics.align(trajs.select(metrics.TRAJ_COLUMNS), p, n)  # 빠지거나 남는 예측이 있으면 ValueError
        assert a.height == metrics.running(trajs, n).height


def test_array_cache_roundtrip(tmp_path):
    steps, trajs = seq_inputs(7)
    d = build(steps, trajs)
    sequences.save_arrays(d, tmp_path / "seq")
    e = sequences.load_arrays(tmp_path / "seq")
    assert e.fingerprint == d.fingerprint
    for name in sequences.ARRAY_FIELDS:
        assert np.array_equal(np.asarray(getattr(e, name)), getattr(d, name)), name


def test_sequences_read_no_leaky_columns():
    assert not set(io.LEAKY_COLUMNS) & set(features.STEP_INPUT_COLUMNS)
    assert not set(io.LEAKY_COLUMNS) & set(sequences.TRAJ_COLUMNS)
    assert set(sequences.CUM_FEATURES) <= set(features.FEATURES)
    assert not any(c.startswith("last_") for c in sequences.CUM_FEATURES)


@pytest.mark.skipif(not (io.STEPS_PATH.exists() and io.SPLITS_PATH.exists()), reason="steps.parquet / splits.parquet 없음")
def test_real_sequences_ignore_later_steps():
    """실데이터 표본 300궤적: 앞 N스텝만 넣어 만든 입력이 전체로 만든 입력의 위치 1..N과 같다."""
    t = metrics.eval_set("train").filter(pl.col("n_steps") > 12).sample(n=300, seed=io.SEED).sort("traj_id")
    d = sequences.load(t)
    assert d.n == 300 and (d.lengths == np.minimum(d.n_steps - 1, sequences.MAX_N)).all()
    raw = io.scan_steps(features.STEP_INPUT_COLUMNS).join(t.select("traj_id").lazy(), on="traj_id", how="semi").collect()
    tt = io.load_trajectories(["traj_id", "instance_id", "issue_text"]).join(t.select("traj_id"), on="traj_id", how="semi")
    m = features.mentions_table(tt)
    n = 10
    only = features.step_values(raw.filter(pl.col("step_idx") <= n), m, tt)
    e = sequences.build(t.with_columns(n_steps=pl.lit(n + 1)), only, m, max_n=n)
    for a, b in zip(arrays(d, n), arrays(e)):
        assert np.array_equal(a, b)
