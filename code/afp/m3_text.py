"""M3 입력 (Phase 7.2): 궤적 앞 N스텝 원문 → ModernBERT 토큰 id. 시점 N의 입력은 스텝 1..N만 본다.

스텝 t 조각 (토큰 단위로 자른다, 스텝 하나 최대 STEP_MAX_TOKENS):
    "\\n[t]" + 생각 앞 THOUGHT_TOKENS + " $" + 명령 앞 COMMAND_TOKENS + " >" + 관측 (앞 OBS_HEAD_TOKENS + " …" + 뒤 OBS_TAIL_TOKENS)
- 생각 = action_text에서 첫 코드 펜스(```) 앞부분. 명령 = steps.command (파싱된 명령)
- 관측 = obs_text에서 SWE-agent 상태 줄(parse.TRAILER: 열린 파일·현재 디렉터리·프롬프트)을 뗀 것. 짧으면 전부,
  길면 앞부분 + 뒷부분 (파이썬 에러 메시지는 보통 끝에 있다)
- 시점 N 입력 = [CLS] + 조각 1..N 이어 붙임 + [SEP]. N=20에서도 MAX_LEN 안이라 자르지 않는다 (build가 검사)
- 이슈 본문은 넣지 않는다 (B1과의 결합은 m3_bert_b1 stacking으로, docs/phase7.md 7.0)

저장 형식 (TextData): 궤적마다 조각 1..L을 이어 붙인 토큰 열 하나(tokens[start_i:start_i+ends[i, L-1]])와
누적 길이 ends[i, t-1] = 조각 1..t 길이 합. 시점 N 입력은 이 열의 앞부분(prefix)이라 N 이후 스텝은 구조상 들어갈 수 없다.
L = min(n_steps − 1, MAX_N): 궤적의 마지막 스텝은 넣지 않는다 (시점 N 진행 중 ⇔ n_steps > N ⇔ L >= N, sequences와 같다).
"""
import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from afp import io

MODEL_NAME = "answerdotai/ModernBERT-base"
REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"  # HF Hub 커밋 고정 (2026-11-09 확인)
MAX_N = 20
MAX_LEN = 2048
THOUGHT_TOKENS = 32
COMMAND_TOKENS = 24
OBS_HEAD_TOKENS = 26
OBS_TAIL_TOKENS = 12
MARKER_MAX_TOKENS = 4  # "\n[t]" (t = 1..20) 토큰 수 상한 [확인: ModernBERT 토크나이저에서 모두 4]
STEP_MAX_TOKENS = MARKER_MAX_TOKENS + THOUGHT_TOKENS + 1 + COMMAND_TOKENS + 1 + OBS_HEAD_TOKENS + 1 + OBS_TAIL_TOKENS
# 토크나이즈 전에 문자열을 먼저 자른다 (메모리). 토큰 하나는 대개 2~5자라 이 길이면 위 토큰 수보다 충분히 길다
THOUGHT_CHARS = 600
COMMAND_CHARS = 400
OBS_HEAD_CHARS = 600
OBS_TAIL_CHARS = 300
TEXT_VERSION = 1
TRAJ_COLUMNS = ["traj_id", "instance_id", "target", "n_steps", "fold"]
STEP_COLUMNS = ["traj_id", "step_idx", "action_text", "command", "obs_text"]
# parse.TRAILER와 같은 패턴 (Rust regex: \Z 대신 \z). tests/test_m3_text.py가 두 패턴의 결과를 대조한다
TRAILER_RS = r"\(Open file: .*\)\n\(Current directory: .*\)\n?bash-\$\s*\z"
ENCODE_CHUNK = 50_000

assert 2 + MAX_N * STEP_MAX_TOKENS <= MAX_LEN, "스텝 예산 × MAX_N이 MAX_LEN을 넘는다"


# ---- 토크나이저 ----

@dataclass
class Encoder:
    """문자열 묶음 → 토큰 id 목록 (특수 토큰 없음). 테스트는 가짜 encode로 만든다."""
    encode: object  # Callable[[list[str]], list[list[int]]]
    cls_id: int
    sep_id: int
    pad_id: int
    name: str = "custom"

    def ids(self, s: str) -> list[int]:
        return list(self.encode([s])[0])

    def markers(self, max_n: int = MAX_N) -> list[list[int]]:
        out = [self.ids(f"\n[{t}]") for t in range(1, max_n + 1)]
        if max(len(m) for m in out) > MARKER_MAX_TOKENS:
            raise ValueError(f"스텝 표시 토큰 수가 {MARKER_MAX_TOKENS}를 넘는다")
        return out

    def seps(self) -> dict[str, list[int]]:
        out = {"cmd": self.ids(" $"), "obs": self.ids(" >"), "gap": self.ids(" …")}
        if any(len(v) != 1 for v in out.values()):
            raise ValueError(f"구분 기호는 토큰 1개여야 한다: {out}")
        return out


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_NAME, revision=REVISION)


def hf_encoder() -> Encoder:
    """고정 revision의 ModernBERT 토크나이저. Rust 토크나이저의 묶음 인코딩을 쓴다 (자르기·패딩 없음)."""
    tok = load_tokenizer()
    bt = tok.backend_tokenizer
    bt.no_truncation()
    bt.no_padding()

    def encode(texts):
        return [e.ids for e in bt.encode_batch(list(texts), add_special_tokens=False)]

    return Encoder(encode=encode, cls_id=tok.cls_token_id, sep_id=tok.sep_token_id, pad_id=tok.pad_token_id,
                   name=f"{MODEL_NAME}@{REVISION}")


# ---- 스텝 원문 ----

def prepare_steps(steps: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """steps(STEP_COLUMNS) → (traj_id, step_idx, thought, command, obs_head, obs_tail, obs_chars). 문자 단위로 먼저 자른다."""
    obs = pl.col("obs_text").fill_null("").str.replace(TRAILER_RS, "").str.strip_chars_end()
    return steps.lazy().select(
        "traj_id", "step_idx",
        thought=pl.col("action_text").fill_null("").str.split("```").list.first().str.strip_chars().str.slice(0, THOUGHT_CHARS),
        command=pl.col("command").fill_null("").str.strip_chars().str.slice(0, COMMAND_CHARS),
        obs=obs,
    ).with_columns(
        obs_head=pl.col("obs").str.slice(0, OBS_HEAD_CHARS),
        obs_tail=pl.col("obs").str.slice(-OBS_TAIL_CHARS),
        obs_chars=pl.col("obs").str.len_chars(),
    ).drop("obs")


def load_step_text(trajs: pl.DataFrame, max_n: int = MAX_N) -> pl.DataFrame:
    """trajs의 스텝 1..min(n_steps−1, max_n) 원문 (prepare_steps 형식). 누수 컬럼은 io.scan_steps가 거부한다."""
    t = trajs.select("traj_id", L=(pl.col("n_steps") - 1).clip(0, max_n))
    s = (io.scan_steps(STEP_COLUMNS).filter(pl.col("step_idx") <= max_n).join(t.lazy(), on="traj_id", how="inner")
         .filter(pl.col("step_idx") <= pl.col("L")).drop("L"))
    return prepare_steps(s).sort("traj_id", "step_idx").collect()


def _obs_ids(head: list[int], tail: list[int], chars: int, gap: list[int]) -> list[int]:
    """관측 토큰: 앞 OBS_HEAD_CHARS자 안에 전체가 있고 짧으면 전부, 아니면 앞 OBS_HEAD_TOKENS + " …" + 뒤 OBS_TAIL_TOKENS."""
    if chars <= OBS_HEAD_CHARS and len(head) <= OBS_HEAD_TOKENS + OBS_TAIL_TOKENS:
        return head
    tail_ids = head[-OBS_TAIL_TOKENS:] if chars <= OBS_HEAD_CHARS else tail[-OBS_TAIL_TOKENS:]
    return head[:OBS_HEAD_TOKENS] + gap + tail_ids


def step_pieces(steps: pl.DataFrame, enc: Encoder) -> list[np.ndarray]:
    """prepare_steps 행마다 조각 토큰 (int32). 스텝 t 조각은 그 스텝의 원문만 본다."""
    markers, seps = enc.markers(), enc.seps()
    out = []
    for s in range(0, steps.height, ENCODE_CHUNK):
        c = steps.slice(s, ENCODE_CHUNK)
        th, cm, oh, ot = (enc.encode(c[col].to_list()) for col in ("thought", "command", "obs_head", "obs_tail"))
        for t, a, b, h, tl, n in zip(c["step_idx"].to_list(), th, cm, oh, ot, c["obs_chars"].to_list()):
            ids = (markers[t - 1] + list(a[:THOUGHT_TOKENS]) + seps["cmd"] + list(b[:COMMAND_TOKENS]) + seps["obs"]
                   + _obs_ids(list(h), list(tl), n, seps["gap"]))
            out.append(np.asarray(ids, dtype=np.int32))
    return out


# ---- 궤적 단위 배열 ----

@dataclass
class TextData:
    """궤적 행(traj_id 정렬). 시점 N 입력 = [CLS] + tokens[start[i] : start[i] + ends[i, N-1]] + [SEP]."""
    traj_id: np.ndarray
    instance_id: np.ndarray
    fold: np.ndarray
    y: np.ndarray  # 실패 = 1
    n_steps: np.ndarray
    lengths: np.ndarray  # L
    tokens: np.ndarray  # int32 [총 토큰]
    start: np.ndarray  # int64 [n]
    ends: np.ndarray  # int32 [n, max_n], 위치 t > L은 -1
    cls_id: int
    sep_id: int
    pad_id: int
    max_n: int = MAX_N
    fingerprint: str = field(default="")

    @property
    def n(self) -> int:
        return len(self.traj_id)

    def running(self, n: int) -> np.ndarray:
        if not 1 <= n <= self.max_n:
            raise ValueError(f"N={n}은 1..{self.max_n} 밖")
        return np.flatnonzero(self.lengths >= n)

    def seq_len(self, rows: np.ndarray, ns: np.ndarray) -> np.ndarray:
        """(행, N) 입력 길이 (CLS·SEP 포함)."""
        e = self.ends[np.asarray(rows), np.asarray(ns) - 1].astype(np.int64)
        if (e < 0).any():
            raise ValueError("시점 N에 진행 중이 아닌 궤적의 입력을 요청했다")
        return e + 2

    def ids(self, i: int, n: int) -> np.ndarray:
        if self.lengths[i] < n:
            raise ValueError(f"궤적 {self.traj_id[i]}는 시점 {n}에 진행 중이 아니다 (L={self.lengths[i]})")
        s = int(self.start[i])
        body = self.tokens[s:s + int(self.ends[i, n - 1])]
        return np.concatenate([[self.cls_id], body, [self.sep_id]]).astype(np.int64)


def build(trajs: pl.DataFrame, steps: pl.DataFrame, enc: Encoder, max_n: int = MAX_N) -> TextData:
    """trajs(TRAJ_COLUMNS) + 스텝 원문(prepare_steps 형식) → TextData. 궤적마다 스텝 1..L이 모두 있어야 한다."""
    missing = [c for c in TRAJ_COLUMNS if c not in trajs.columns]
    if missing:
        raise ValueError(f"trajs에 없는 컬럼: {missing}")
    t = (trajs.select(TRAJ_COLUMNS).with_columns(L=(pl.col("n_steps") - 1).clip(0, max_n).cast(pl.Int64))
         .filter(pl.col("L") >= 1).sort("traj_id").with_row_index("_i"))
    s = (steps.join(t.select("traj_id", "L", "_i"), on="traj_id", how="inner").filter(pl.col("step_idx") <= pl.col("L"))
         .sort("_i", "step_idx"))
    got = s.group_by("_i").agg(pl.len(), pl.col("step_idx").max().alias("mx")).join(t.select("_i", "L"), on="_i", how="right")
    bad = got.filter((pl.col("len").fill_null(0) != pl.col("L")) | (pl.col("mx").fill_null(0) != pl.col("L"))).height
    if bad:
        raise ValueError(f"스텝 1..L이 모두 있지 않은 궤적 {bad}개")
    pieces = step_pieces(s, enc)
    plen = np.array([len(p) for p in pieces], dtype=np.int64)
    if plen.max(initial=0) > STEP_MAX_TOKENS:
        raise AssertionError(f"스텝 조각 {plen.max()}토큰 > STEP_MAX_TOKENS {STEP_MAX_TOKENS}")
    ii = s["_i"].to_numpy()
    tt = s["step_idx"].to_numpy().astype(np.int64) - 1
    n = t.height
    L = t["L"].to_numpy()
    totals = np.zeros(n, dtype=np.int64)
    np.add.at(totals, ii, plen)
    start = np.concatenate([[0], np.cumsum(totals)[:-1]]).astype(np.int64)
    ends = np.full((n, max_n), -1, dtype=np.int32)
    # 행은 (_i, step_idx) 정렬이라 궤적 안 누적합 = 전체 누적합 − 궤적 시작
    cum = np.cumsum(plen) - start[ii]
    ends[ii, tt] = cum.astype(np.int32)
    tokens = np.concatenate(pieces).astype(np.int32) if pieces else np.zeros(0, dtype=np.int32)
    if (ends.max(axis=1) + 2 > MAX_LEN).any():
        raise AssertionError("입력 길이가 MAX_LEN을 넘는다")
    data = TextData(
        traj_id=t["traj_id"].to_numpy(), instance_id=t["instance_id"].to_numpy(), fold=t["fold"].to_numpy().astype(np.int64),
        y=(~t["target"]).to_numpy().astype(np.float32), n_steps=t["n_steps"].to_numpy().astype(np.int64), lengths=L,
        tokens=tokens, start=start, ends=ends, cls_id=int(enc.cls_id), sep_id=int(enc.sep_id), pad_id=int(enc.pad_id), max_n=max_n)
    data.fingerprint = fingerprint(data, enc.name)
    return data


def fingerprint(data: TextData, encoder_name: str) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({"v": TEXT_VERSION, "enc": encoder_name, "budget": budget(), "special": [data.cls_id, data.sep_id, data.pad_id],
                         "max_n": data.max_n}, sort_keys=True).encode())
    for a in (data.traj_id, data.instance_id, data.fold, data.y, data.lengths, data.tokens, data.start, data.ends):
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes() if a.dtype != object else "\x00".join(map(str, a)).encode())
    return h.hexdigest()


def budget() -> dict:
    return {"thought": THOUGHT_TOKENS, "command": COMMAND_TOKENS, "obs_head": OBS_HEAD_TOKENS, "obs_tail": OBS_TAIL_TOKENS,
            "marker_max": MARKER_MAX_TOKENS, "step_max": STEP_MAX_TOKENS, "max_len": MAX_LEN,
            "chars": [THOUGHT_CHARS, COMMAND_CHARS, OBS_HEAD_CHARS, OBS_TAIL_CHARS]}


# ---- 캐시 ----

ARRAY_FIELDS = ("traj_id", "instance_id", "fold", "y", "n_steps", "lengths", "tokens", "start", "ends")


def input_key(trajs: pl.DataFrame, encoder_name: str) -> str:
    """캐시 열쇠: 궤적 목록·라벨·fold + 예산 + 토크나이저 + steps.parquet 크기. 같으면 저장된 배열을 다시 쓴다."""
    t = trajs.select(TRAJ_COLUMNS).sort("traj_id")
    h = hashlib.sha256()
    h.update(json.dumps({"v": TEXT_VERSION, "enc": encoder_name, "budget": budget(), "steps_bytes": io.STEPS_PATH.stat().st_size},
                        sort_keys=True).encode())
    h.update(t.hash_rows(seed=0).to_numpy().tobytes())
    return h.hexdigest()


def save_arrays(data: TextData, directory, key: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in ARRAY_FIELDS:
        a = getattr(data, name)
        np.save(directory / f"{name}.npy", a.astype(str) if a.dtype == object else a)
    meta = {"key": key, "fingerprint": data.fingerprint, "max_n": data.max_n, "cls_id": data.cls_id, "sep_id": data.sep_id,
            "pad_id": data.pad_id}
    (directory / "meta.json").write_text(json.dumps(meta, indent=1))


def load_arrays(directory) -> tuple[TextData, str]:
    meta = json.loads((directory / "meta.json").read_text())
    kw = {name: np.load(directory / f"{name}.npy") for name in ARRAY_FIELDS}
    for name in ("traj_id", "instance_id"):
        kw[name] = kw[name].astype(object)
    data = TextData(**kw, cls_id=meta["cls_id"], sep_id=meta["sep_id"], pad_id=meta["pad_id"], max_n=meta["max_n"],
                    fingerprint=meta["fingerprint"])
    return data, meta["key"]


def load(trajs: pl.DataFrame, enc: Encoder, cache_dir=None, max_n: int = MAX_N) -> TextData:
    """평가 집합 궤적 → TextData. cache_dir에 같은 열쇠의 배열이 있으면 읽고, 없으면 만들어 저장한다."""
    key = input_key(trajs, enc.name)
    if cache_dir is not None and (cache_dir / "meta.json").exists():
        data, old = load_arrays(cache_dir)
        if old == key and data.max_n == max_n:
            return data
    data = build(trajs, load_step_text(trajs, max_n), enc, max_n)
    if cache_dir is not None:
        save_arrays(data, cache_dir, key)
    return data
