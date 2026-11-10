"""M3 입력 (Phase 7.2): 시점 N 입력은 스텝 1..N만 본다, 관측 자르기 규칙, 길이 상한, 캐시 왕복."""
import re
import zlib

import numpy as np
import polars as pl
import pytest

from afp import m3_text, parse

WORDS = ["fix", "the", "bug", "in", "parser", "Traceback", "Error", "line", "file", "ok", "def", "return", "x", "=", "1"]


def fake_encoder() -> m3_text.Encoder:
    """단어·기호 단위 가짜 토크나이저 ("\\n[20]" = 3토큰, " $" = 1토큰)."""
    def encode(texts):
        return [[zlib.crc32(w.encode()) % 40000 + 10 for w in re.findall(r"\w+|[^\w\s]", s)] for s in texts]
    return m3_text.Encoder(encode=encode, cls_id=1, sep_id=2, pad_id=3, name="fake")


def fake_inputs(seed=0, n_traj=40, max_steps=26, long=False, n_issues=9):
    rng = np.random.default_rng(seed)
    n_steps = rng.integers(1, max_steps + 1, n_traj)
    trajs = pl.DataFrame({"traj_id": [f"t{i:03d}" for i in range(n_traj)], "instance_id": [f"i{i % n_issues}" for i in range(n_traj)],
                          "target": rng.random(n_traj) < 0.3, "n_steps": n_steps,
                          "fold": [i % n_issues % 5 for i in range(n_traj)]})

    def text(k):
        return " ".join(rng.choice(WORDS, k))

    rows = []
    for i, n in enumerate(n_steps):
        for t in range(1, int(n) + 1):
            k = 400 if long else int(rng.integers(0, 60))
            obs = text(k) + "\n(Open file: /a.py)\n(Current directory: /repo)\nbash-$ "
            rows.append({"traj_id": f"t{i:03d}", "step_idx": t, "action_text": f"{text(k)}\n```\n{text(5)}\n```",
                         "command": text(max(1, k // 3)), "obs_text": None if rng.random() < 0.1 else obs})
    return trajs, pl.DataFrame(rows)


def build(trajs, steps, enc=None):
    return m3_text.build(trajs, m3_text.prepare_steps(steps).collect(), enc or fake_encoder())


@pytest.mark.parametrize("seed", [0, 1])
def test_input_at_n_ignores_later_steps(seed):
    """N 뒤 스텝의 원문을 바꾸거나 지워도 시점 N 입력이 같다 (진행 중인 궤적 모두)."""
    trajs, steps = fake_inputs(seed)
    base = build(trajs, steps)
    for n in (1, 5, 10, 20):
        after = pl.col("step_idx") > n
        changed = steps.with_columns(action_text=pl.when(after).then(pl.lit("Error Error Error")).otherwise("action_text"),
                                     command=pl.when(after).then(pl.lit("submit")).otherwise("command"),
                                     obs_text=pl.when(after).then(pl.lit("Traceback line 1")).otherwise("obs_text"))
        cut = trajs.with_columns(n_steps=pl.min_horizontal("n_steps", pl.lit(n + 1)))
        for other in (build(trajs, changed), build(cut, steps.filter(pl.col("step_idx") <= n + 1))):
            assert np.array_equal(other.traj_id, base.traj_id)
            for i in base.running(n):
                assert np.array_equal(other.ids(i, n), base.ids(i, n))


def test_lengths_and_last_step():
    """L = min(n_steps − 1, 20). 마지막 스텝은 들어가지 않는다. ends는 L 뒤 −1, seq_len = len(ids)."""
    trajs, steps = fake_inputs(2)
    d = build(trajs, steps)
    t = trajs.filter(pl.col("n_steps") >= 2).sort("traj_id")
    assert d.traj_id.tolist() == t["traj_id"].to_list()
    assert np.array_equal(d.lengths, np.minimum(t["n_steps"].to_numpy() - 1, 20))
    assert np.array_equal(d.y, (~t["target"]).to_numpy().astype(np.float32))
    for i in range(d.n):
        L = d.lengths[i]
        assert (d.ends[i, L:] == -1).all() and (np.diff(d.ends[i, :L]) > 0).all()
        assert d.seq_len(np.array([i]), np.array([L]))[0] == len(d.ids(i, L))
        with pytest.raises(ValueError):
            d.ids(i, L + 1) if L < 20 else d.running(21)
    # 마지막 스텝을 바꿔도 같다
    last = steps.join(trajs.select("traj_id", "n_steps"), on="traj_id").with_columns(
        command=pl.when(pl.col("step_idx") == pl.col("n_steps")).then(pl.lit("submit now")).otherwise("command")).drop("n_steps")
    assert build(trajs, last).fingerprint == d.fingerprint


def test_missing_step_raises():
    trajs, steps = fake_inputs(3)
    tid = trajs.filter(pl.col("n_steps") >= 3)["traj_id"][0]
    with pytest.raises(ValueError):
        build(trajs, steps.filter(~((pl.col("traj_id") == tid) & (pl.col("step_idx") == 2))))


def test_long_text_stays_within_budget():
    """원문이 아주 길어도 스텝 조각 <= STEP_MAX_TOKENS, 시점 20 입력 <= MAX_LEN."""
    trajs, steps = fake_inputs(4, n_traj=12, max_steps=30, long=True)
    d = build(trajs, steps)
    pieces = np.diff(np.concatenate([np.zeros((d.n, 1), dtype=np.int64), np.where(d.ends < 0, 0, d.ends)], axis=1), axis=1)
    assert pieces.max() <= m3_text.STEP_MAX_TOKENS
    assert (d.ends.max(axis=1) + 2 <= m3_text.MAX_LEN).all()
    assert 2 + m3_text.MAX_N * m3_text.STEP_MAX_TOKENS <= m3_text.MAX_LEN


def test_trailer_regex_matches_parse():
    """polars용 TRAILER_RS로 뗀 결과 = parse.TRAILER로 뗀 결과."""
    obs = ["out\n(Open file: /a.py)\n(Current directory: /r)\nbash-$ ", "x\n(Open file: n/a)\n(Current directory: /r)bash-$",
           "no trailer here", "(Open file: a)\n(Current directory: b)\nbash-$ \n\n", "mid (Open file: a)\n(Current directory: b)\nbash-$ tail",
           "", None]
    got = m3_text.prepare_steps(pl.DataFrame({"traj_id": ["t"] * len(obs), "step_idx": list(range(1, len(obs) + 1)),
                                              "action_text": [""] * len(obs), "command": [""] * len(obs), "obs_text": obs}))
    got = got.collect()["obs_head"].to_list()
    want = [parse.TRAILER.sub("", o or "").rstrip()[:m3_text.OBS_HEAD_CHARS] for o in obs]
    assert got == want


def test_obs_head_tail_rule():
    gap = [7]
    H, T = m3_text.OBS_HEAD_TOKENS, m3_text.OBS_TAIL_TOKENS
    short = list(range(H + T))
    assert m3_text._obs_ids(short, short, 100, gap) == short  # 짧으면 전부
    long_tok = list(range(H + T + 5))
    assert m3_text._obs_ids(long_tok, long_tok, 500, gap) == long_tok[:H] + gap + long_tok[-T:]  # 문자는 짧고 토큰이 많음
    head, tail = list(range(100)), list(range(1000, 1050))
    assert m3_text._obs_ids(head, tail, 5000, gap) == head[:H] + gap + tail[-T:]  # 원문이 길면 뒷부분은 끝에서
    assert m3_text._obs_ids(head[:3], tail, 5000, gap) == head[:3] + gap + tail[-T:]


def test_cache_roundtrip(tmp_path):
    trajs, steps = fake_inputs(5)
    d = build(trajs, steps)
    m3_text.save_arrays(d, tmp_path, "k")
    e, key = m3_text.load_arrays(tmp_path)
    assert key == "k" and e.fingerprint == d.fingerprint == m3_text.fingerprint(e, "fake")
    for i in e.running(5):
        assert np.array_equal(e.ids(i, 5), d.ids(i, 5))


def test_real_tokenizer_budget():
    """ModernBERT 토크나이저 (HF 캐시에 있을 때만): 스텝 표시 <= 4토큰, 구분 기호 1토큰."""
    try:
        enc = m3_text.hf_encoder()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"토크나이저 없음: {e}")
    assert max(len(m) for m in enc.markers()) <= m3_text.MARKER_MAX_TOKENS
    assert all(len(v) == 1 for v in enc.seps().values())
    assert (enc.cls_id, enc.sep_id, enc.pad_id) == (50281, 50282, 50283)
