"""M2 LSTM (Phase 6.2·6.3): 인과성(위치 t 출력은 1..t 입력만), fit 행 밖 데이터를 보지 않음, 결정성, 저장."""
import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from afp import lstm, sequences  # noqa: E402
from test_sequences import build, seq_inputs  # noqa: E402

TINY = lstm.M2Config(hidden=4, dropout=0.1, batch_size=64, max_epochs=2, patience=1, n_seeds=2, emb_dim=3)


def tiny_data(seed=0, n_traj=320):
    steps, trajs = seq_inputs(seed, n_traj=n_traj, max_steps=30)
    trajs = trajs.with_columns(instance_id=pl.format("i{}", pl.int_range(pl.len()) % 48))
    trajs = trajs.with_columns(fold=pl.col("instance_id").str.slice(1).cast(pl.Int64) % 5)
    return build(steps, trajs)


@pytest.mark.parametrize("recurrent", [True, False])
def test_output_at_t_ignores_later_inputs(recurrent):
    """위치 N의 logit은 위치 N 뒤 입력(명령·스텝 값·누적·B1)을 바꿔도 같다. 길이(패딩 위치)를 바꿔도 같다."""
    torch.manual_seed(0)
    d = tiny_data()
    cfg = lstm.copy_config(TINY, recurrent=recurrent, use_b1=True)
    net = lstm.Net(cfg, d.step.shape[-1], d.cum.shape[-1], d.static.shape[-1]).eval()
    rows = np.flatnonzero(d.lengths >= 15)[:32]
    scaler = lstm.Scaler.fit(d, rows, np.full((d.n, d.max_n), 0.8, dtype=np.float32))
    b1 = np.random.default_rng(1).uniform(0.6, 0.95, (d.n, d.max_n)).astype(np.float32)
    inp, _ = lstm._batch(d, rows, scaler, b1, "cpu", True)
    n = 8
    alt = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inp.items()}
    alt["command"][:, n:] = 0
    alt["step"][:, n:] += 5.0
    alt["cum"][:, n:] -= 3.0
    alt["b1"][:, n:] = 9.0
    short = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inp.items()}
    short["lengths"] = torch.full_like(inp["lengths"], n)
    with torch.no_grad():
        a, b, c = net(**inp), net(**alt), net(**short)
    assert torch.equal(a[:, :n], b[:, :n])  # 같은 길이: 비트 단위로 같다
    # 길이를 바꾸면 packed 배치의 시점별 크기가 달라져 행렬곱 커널이 바뀐다 → 마지막 비트만 다를 수 있다 (정보 차이 아님)
    assert torch.allclose(a[:, :n], c[:, :n], atol=1e-6, rtol=0)


def test_fit_sees_only_given_rows_and_is_deterministic():
    """fold 0을 뺀 행으로 학습: fold 0 궤적의 입력·라벨을 바꿔도 학습 결과가 같다 (조기 종료·표준화 포함).
    같은 seed로 다시 학습하면 비트 단위로 같다. 검증 조각은 fit 행 안의 이슈 단위 부분집합이다."""
    d = tiny_data()
    rows = np.flatnonzero(d.fold != 0)
    seeds = lstm.seeds_for(0, TINY.n_seeds)
    m1 = lstm.fit(TINY, d, rows, seeds, "cpu")
    m2 = lstm.fit(TINY, d, rows, seeds, "cpu")
    held = np.flatnonzero(d.fold == 0)
    p1 = lstm.predict(m1, d, np.arange(d.n), "cpu")
    assert np.array_equal(p1, lstm.predict(m2, d, np.arange(d.n), "cpu"), equal_nan=True)

    e = sequences.SequenceData(**{**d.__dict__})
    e.y, e.step, e.cum = d.y.copy(), d.step.copy(), d.cum.copy()
    e.y[held] = 1 - e.y[held]
    e.step[held] += 7.0
    e.cum[held] *= 0.0
    m3 = lstm.fit(TINY, e, rows, seeds, "cpu")
    assert all(np.array_equal(m1.scaler.__dict__[k], m3.scaler.__dict__[k]) for k in ("step_mean", "step_std", "cum_mean", "cum_std"))
    assert np.array_equal(p1[rows], lstm.predict(m3, d, rows, "cpu"), equal_nan=True)
    assert [h["val_history"] for h in m1.history] == [h["val_history"] for h in m3.history]

    tr, va = lstm.inner_split(d, rows)
    assert set(tr) | set(va) == set(rows) and not set(tr) & set(va)
    assert not set(d.instance_id[tr]) & set(d.instance_id[va])
    assert m1.fit_issue_hash == lstm.issue_hash(d.instance_id[rows])


def test_predictions_valid_positions_only_and_seed_mean():
    d = tiny_data(1)
    rows = np.flatnonzero(d.fold != 1)
    m = lstm.fit(TINY, d, rows, lstm.seeds_for(1, 2), "cpu")
    p = lstm.predict(m, d, np.arange(d.n), "cpu")
    assert np.isnan(p[~d.valid()]).all() and np.isfinite(p[d.valid()]).all()
    assert ((p[d.valid()] > 0) & (p[d.valid()] < 1)).all()
    single = [lstm._probs(net, d, np.arange(d.n), m.scaler, None, "cpu") for net in m.nets(d, "cpu")]
    assert np.allclose(p, np.mean(single, axis=0), equal_nan=True)


def test_b1_model_requires_scores():
    d = tiny_data(2)
    with pytest.raises(ValueError):
        lstm.fit(lstm.copy_config(TINY, use_b1=True), d, np.flatnonzero(d.fold != 0), [1], "cpu")


def test_save_load_roundtrip(tmp_path):
    d = tiny_data(3)
    rows = np.flatnonzero(d.fold != 2)
    b1 = np.where(d.valid(), 0.8, np.nan).astype(np.float32)
    m = lstm.fit(lstm.copy_config(TINY, use_b1=True), d, rows, [7], "cpu", b1)
    lstm.save(m, tmp_path / "m.pt", {"k": 1})
    back, meta = lstm.load(tmp_path / "m.pt")
    assert meta == {"k": 1}
    assert np.array_equal(lstm.predict(m, d, rows, "cpu", b1), lstm.predict(back, d, rows, "cpu", b1), equal_nan=True)


def test_seeds_depend_only_on_fold():
    assert lstm.seeds_for(0, 3) == lstm.seeds_for(0, 3)
    assert lstm.seeds_for(0, 3) != lstm.seeds_for(1, 3)
    assert lstm.seeds_for(2, 2) == lstm.seeds_for(2, 3)[:2]
