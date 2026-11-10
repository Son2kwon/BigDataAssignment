"""M3 ModernBERT (Phase 7.3): 예 뽑기·검증 조각·배치 규칙, fit 행 밖 데이터를 보지 않음, 결정성·저장, B1 결합기의 fold 분리."""
import dataclasses
import importlib

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from afp import m3  # noqa: E402
from test_m3_text import build, fake_inputs  # noqa: E402

TINY = m3.M3Config(tiny=True, max_epochs=1, per_issue=1, val_per_issue=2, eff_batch=8, tokens_per_batch=4096)


@pytest.fixture(autouse=True)
def _restore_determinism():
    yield
    torch.use_deterministic_algorithms(False)


@pytest.fixture(scope="module")
def data():
    trajs, steps = fake_inputs(0, n_traj=300, max_steps=26, n_issues=60)
    return build(trajs, steps)


def test_sample_examples(data):
    rows = np.flatnonzero(data.fold != 0)
    a = m3.sample_examples(data, rows, m3.N_GRID, 2, np.random.default_rng([7, 1]))
    b = m3.sample_examples(data, rows[::-1], m3.N_GRID, 2, np.random.default_rng([7, 1]))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))  # 행 순서와 무관, seed로만 정해진다
    er, en = a
    assert set(er) <= set(rows) and (data.lengths[er] >= en).all()
    df = pl.DataFrame({"iid": data.instance_id[er].astype(str), "N": en})
    got = df.group_by("iid", "N").len()
    avail = pl.DataFrame({"iid": data.instance_id[rows].astype(str), "L": data.lengths[rows]})
    for r in got.iter_rows(named=True):
        n_run = avail.filter((pl.col("iid") == r["iid"]) & (pl.col("L") >= r["N"])).height
        assert r["len"] == min(2, n_run)
    c = m3.sample_examples(data, rows, m3.N_GRID, 2, np.random.default_rng([7, 2]))
    assert not np.array_equal(c[0], er)  # epoch마다 다른 뽑기


def test_inner_split_by_issue(data):
    rows = np.flatnonzero(data.fold != 1)
    tr, va = m3.inner_split(data, rows)
    assert set(tr) | set(va) == set(rows) and not set(tr) & set(va)
    assert not set(data.instance_id[tr]) & set(data.instance_id[va])
    tr2, va2 = m3.inner_split(data, rows)
    assert np.array_equal(va, va2)


def test_plan_epoch_and_micro_batches(data):
    er, en = m3.all_examples(data, np.arange(data.n))
    cfg = dataclasses.replace(TINY, tokens_per_batch=1500)
    plan = m3.plan_epoch(data, er, en, cfg, np.random.default_rng(0))
    lens = data.seq_len(er, en)
    flat = np.concatenate([mb for step in plan for mb in step])
    assert np.array_equal(np.sort(flat), np.arange(len(er)))  # 모든 예가 한 번씩
    for step in plan:
        assert sum(len(mb) for mb in step) <= cfg.eff_batch
        for mb in step:
            assert len(mb) == 1 or len(mb) * lens[mb].max() <= cfg.tokens_per_batch


def _fit(data, rows, seed=3):
    return m3.fit(TINY, data, rows, seed, "cpu", log=lambda *_: None)


def test_fit_deterministic_save_load_predict(data, tmp_path):
    rows = np.flatnonzero(data.fold != 2)
    a, b = _fit(data, rows), _fit(data, rows)
    assert a.state.keys() == b.state.keys() and all(torch.equal(a.state[k], b.state[k]) for k in a.state)
    assert a.history == [{**h, "seconds": a.history[i]["seconds"]} for i, h in enumerate(b.history)]
    assert a.fit_issue_hash == m3.issue_hash(data.instance_id[rows])
    m3.save(a, tmp_path / "f.pt", {"x": 1})
    c, meta = m3.load(tmp_path / "f.pt")
    assert meta == {"x": 1}
    er, en = m3.all_examples(data, np.flatnonzero(data.fold == 2))
    p = m3.predict_examples(a.model("cpu"), data, er, en, "cpu")
    q = m3.predict_examples(c.model("cpu"), data, er, en, "cpu")
    assert np.isfinite(p).all() and np.array_equal(p, q)
    perm = np.random.default_rng(0).permutation(len(er))
    r = m3.predict_examples(c.model("cpu"), data, er[perm], en[perm], "cpu")
    assert np.array_equal(r, p[perm])  # 같은 예 집합이면 넣는 순서와 무관


def test_fit_ignores_rows_outside_fit(data):
    """fit 행 밖(= OOF로 채점할 fold) 궤적의 라벨·토큰을 바꿔도 학습 결과가 같다."""
    rows = np.flatnonzero(data.fold != 3)
    out = np.flatnonzero(data.fold == 3)
    y = data.y.copy()
    y[out] = 1 - y[out]
    tokens = data.tokens.copy()
    for i in out:
        s, e = data.start[i], data.start[i] + data.ends[i].max()
        tokens[s:e] = 11
    other = dataclasses.replace(data, y=y, tokens=tokens)
    a, b = _fit(data, rows), _fit(other, rows)
    assert all(torch.equal(a.state[k], b.state[k]) for k in a.state)


def test_stacker_fold_k_ignores_fold_k_labels():
    """m3_bert_b1: fold k의 OOF 점수는 fold k 라벨을 바꿔도 같다 (결합기 k는 fold k 밖에서만 학습)."""
    run = importlib.import_module("06_m3")
    rng = np.random.default_rng(0)
    n = 400
    ts = pl.DataFrame({"traj_id": [f"t{i:03d}" for i in range(n)], "instance_id": [f"i{i % 50}" for i in range(n)],
                       "fold": [i % 50 % 5 for i in range(n)], "target": rng.random(n) < 0.4})
    rows = pl.DataFrame({"traj_id": np.repeat(ts["traj_id"].to_numpy(), 3), "N": np.tile(np.array(m3.N_GRID, dtype=np.int16), n)})
    m3p = rows.with_columns(p_fail=pl.Series(rng.uniform(0.05, 0.95, rows.height)))
    b1p = rows.with_columns(p_fail=pl.Series(rng.uniform(0.05, 0.95, rows.height)))
    base = run.fuse(m3p, b1p, ts)
    assert base["seen_issue_scores"] == 0 and len(base["stackers"]) == 5
    for k in range(5):
        flipped = ts.with_columns(target=pl.when(pl.col("fold") == k).then(~pl.col("target")).otherwise("target"))
        other = run.fuse(m3p, b1p, flipped)
        ids = ts.filter(pl.col("fold") == k)["traj_id"]
        a = base["train"].filter(pl.col("traj_id").is_in(ids.to_list())).sort("N", "traj_id")
        b = other["train"].filter(pl.col("traj_id").is_in(ids.to_list())).sort("N", "traj_id")
        assert np.array_equal(a["p_fail"].to_numpy(), b["p_fail"].to_numpy())
        assert not base["train"].equals(other["train"])
