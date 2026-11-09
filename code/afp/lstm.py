"""M2 LSTM (Phase 6.2·6.3): 모델 하나로 모든 시점 N의 실패 확률을 낸다 (P6-D2).

- 입력: afp.sequences.SequenceData (스텝 t = 명령 임베딩 + 스텝 값 + 누적 집계). 단방향 LSTM + pack_padded_sequence
- 출력: 위치 t(1..max_n)마다 logit. 시점 N의 예측 = 위치 N의 출력이고 입력 스텝 1..N만 본다
- 머리(head): [h_t, 정적 피처(issue_mentions_file), (B1 점수 logit)] → 선형. B1은 순환부에 넣지 않는다 (h_t는 궤적 정보만)
- 손실: 진행 중인 모든 위치(t <= L)의 BCE 평균. 클래스 가중 없음 (M1에서 balanced가 OOF AUROC를 올리지 않았다, 5.3)
- 조기 종료·검증 (CLAUDE.md 누수 방지): fit에 들어온 행(= fold k를 뺀 train)만 쓴다. 그 안에서 이슈 단위로 1/VAL_PARTS를
  검증 조각으로 떼고(StratifiedGroupKFold, io.SEED), 검증 점수 = N_GRID 시점 AUROC 평균. fold k와 test는 보지 않는다
- 표준화 통계도 검증 조각을 뺀 학습 조각에서만 구한다
- seed 앙상블: fold 모델 하나 = seed n_seeds개 망의 확률 평균. seed는 io.SEED와 fold 번호로만 정한다
- 재현성: torch.use_deterministic_algorithms(True), cuDNN deterministic, 스레드 1개. 같은 기기에서 다시 학습하면 비트 단위로
  같다. CPU와 GPU(또는 다른 GPU) 사이에서는 부동소수 결과가 다르다 (플랫폼 차이)
"""
import hashlib
import os
from dataclasses import asdict, dataclass

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # cuBLAS 결정적 모드 (CUDA 초기화 전에 있어야 한다)

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from afp import evaluate, io, sequences, split

VAL_PARTS = 8  # 검증 조각 = fit 이슈의 1/8
PRED_BATCH = 4096
GRAD_CLIP = 1.0
MIN_DELTA = 1e-6  # 검증 점수 개선으로 보는 최소 차이
B1_CLIP = 1e-6


@dataclass(frozen=True)
class M2Config:
    hidden: int = 32
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    max_epochs: int = 40
    patience: int = 4
    n_seeds: int = 3
    emb_dim: int = 8
    recurrent: bool = True  # False: 순환 없이 스텝마다 같은 층 (절제 실험)
    use_cum: bool = True  # False: 누적 집계 없이 스텝 값만 (절제 실험)
    use_b1: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def configure_torch(device: str, threads: int = 1) -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(threads)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 쓸 수 없다. --device cpu로 실행한다")


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def seeds_for(fold: int, n_seeds: int) -> list[int]:
    """fold 모델의 seed 목록. io.SEED와 fold 번호로만 정한다 (설정·모델과 무관, 같은 fold면 같은 seed)."""
    return [int(s) for s in np.random.SeedSequence([io.SEED, int(fold)]).generate_state(n_seeds)]


def issue_hash(instance_ids) -> str:
    return hashlib.sha256("\x00".join(sorted(set(map(str, instance_ids)))).encode()).hexdigest()


def inner_split(data: sequences.SequenceData, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """fit 행 → (학습 조각, 검증 조각). 이슈 단위, 실패 비율 층화, io.SEED 고정."""
    rows = np.asarray(rows)
    splitter = split.fold_splitter(VAL_PARTS, seed=io.SEED)
    _, va = next(splitter.split(np.zeros(len(rows)), data.y[rows], data.instance_id[rows]))
    mask = np.zeros(len(rows), dtype=bool)
    mask[va] = True
    return rows[~mask], rows[mask]


# ---- 표준화 ----

@dataclass
class Scaler:
    step_mean: np.ndarray
    step_std: np.ndarray
    cum_mean: np.ndarray
    cum_std: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray
    b1_mean: float | None = None
    b1_std: float | None = None

    @staticmethod
    def _stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = x.astype(np.float64)
        m, s = x.mean(0), x.std(0)
        s[s < 1e-8] = 1.0  # 상수 피처
        return m.astype(np.float32), s.astype(np.float32)

    @classmethod
    def fit(cls, data: sequences.SequenceData, rows: np.ndarray, b1: np.ndarray | None = None) -> "Scaler":
        valid = data.valid()[rows]
        sm, ss = cls._stats(data.step[rows][valid])
        cm, cs = cls._stats(data.cum[rows][valid])
        tm, ts = cls._stats(data.static[rows])
        out = cls(sm, ss, cm, cs, tm, ts)
        if b1 is not None:
            z = b1_logit(b1[rows][valid])
            sd = float(z.std())
            out.b1_mean, out.b1_std = float(z.mean()), (sd if sd > 1e-8 else 1.0)
        return out


def b1_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), B1_CLIP, 1 - B1_CLIP)
    return np.log(p / (1 - p))


# ---- 망 ----

class Net(nn.Module):
    def __init__(self, cfg: M2Config, n_step: int, n_cum: int, n_static: int):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(sequences.PAD_COMMAND + 1, cfg.emb_dim, padding_idx=sequences.PAD_COMMAND)
        d_in = cfg.emb_dim + n_step + (n_cum if cfg.use_cum else 0)
        if cfg.recurrent:
            self.rnn = nn.LSTM(d_in, cfg.hidden, batch_first=True)
        else:
            self.rnn = nn.Sequential(nn.Linear(d_in, cfg.hidden), nn.Tanh())
        self.drop = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden + n_static + (1 if cfg.use_b1 else 0), 1)

    def forward(self, command, step, cum, lengths, static, b1=None):
        """[B, T] 입력 → [B, T] logit. 위치 t의 출력은 위치 1..t 입력만 본다 (단방향, 패딩은 뒤쪽)."""
        parts = [self.emb(command), step] + ([cum] if self.cfg.use_cum else [])
        x = torch.cat(parts, dim=-1)
        T = x.shape[1]
        if self.cfg.recurrent:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.rnn(packed)
            h, _ = pad_packed_sequence(out, batch_first=True, total_length=T)
        else:
            h = self.rnn(x)
        z = [self.drop(h), static[:, None, :].expand(-1, T, -1)]
        if self.cfg.use_b1:
            z.append(b1[..., None])
        return self.head(torch.cat(z, dim=-1)).squeeze(-1)


def _batch(data: sequences.SequenceData, idx: np.ndarray, scaler: Scaler, b1: np.ndarray | None, device: str, use_b1: bool):
    """행 idx의 입력 텐서. 배치 안 최대 길이로 자른다. 패딩 위치 값은 0 (출력은 손실·점수에서 버린다)."""
    lengths = data.lengths[idx]
    T = int(lengths.max())
    valid = np.arange(1, T + 1)[None, :] <= lengths[:, None]
    step = (data.step[idx, :T] - scaler.step_mean) / scaler.step_std
    cum = (data.cum[idx, :T] - scaler.cum_mean) / scaler.cum_std
    step[~valid], cum[~valid] = 0.0, 0.0
    static = (data.static[idx] - scaler.static_mean) / scaler.static_std
    out = {
        "command": torch.from_numpy(np.ascontiguousarray(data.command[idx, :T])).to(device),
        "step": torch.from_numpy(np.ascontiguousarray(step, dtype=np.float32)).to(device),
        "cum": torch.from_numpy(np.ascontiguousarray(cum, dtype=np.float32)).to(device),
        "lengths": torch.from_numpy(lengths.astype(np.int64)),
        "static": torch.from_numpy(np.ascontiguousarray(static, dtype=np.float32)).to(device),
        "b1": None,
    }
    if use_b1:
        z = np.zeros((len(idx), T), dtype=np.float32)
        z[valid] = ((b1_logit(b1[idx, :T][valid]) - scaler.b1_mean) / scaler.b1_std).astype(np.float32)
        out["b1"] = torch.from_numpy(z).to(device)
    return out, torch.from_numpy(valid).to(device)


def _probs(net: Net, data, rows, scaler, b1, device) -> np.ndarray:
    """[len(rows), max_n] 실패 확률 (진행 중이 아닌 위치는 NaN)."""
    net.eval()
    out = np.full((len(rows), data.max_n), np.nan, dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(rows), PRED_BATCH):
            idx = rows[s:s + PRED_BATCH]
            inp, valid = _batch(data, idx, scaler, b1, device, net.cfg.use_b1)
            p = torch.sigmoid(net(**inp)).double().cpu().numpy()
            v = valid.cpu().numpy()
            block = np.full((len(idx), data.max_n), np.nan)
            block[:, :p.shape[1]][v] = p[v]
            out[s:s + len(idx)] = block
    return out


def grid_score(data: sequences.SequenceData, rows: np.ndarray, probs: np.ndarray, n_grid=evaluate.N_GRID) -> float:
    """검증 점수: N_GRID 시점마다 진행 중 행의 AUROC, 그 평균."""
    aucs = []
    y = np.asarray(data.y)[rows]
    for n in n_grid:
        r = np.flatnonzero(np.asarray(data.lengths)[rows] >= n)
        if len(np.unique(y[r])) == 2:  # 실데이터에서는 항상 참. 작은 합성 데이터에서 한 클래스뿐인 시점은 뺀다
            aucs.append(roc_auc_score(y[r], probs[r, n - 1]))
    return float(np.mean(aucs)) if aucs else 0.5


def train_net(cfg: M2Config, data, tr_rows, va_rows, scaler, b1, seed: int, device: str) -> tuple[dict, dict]:
    """망 하나를 학습한다. 검증 점수가 patience번 연속 오르지 않으면 멈추고 최고 epoch의 가중치를 돌려준다."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net(cfg, data.step.shape[-1], data.cum.shape[-1], data.static.shape[-1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best, best_state, best_epoch, wait, history = -np.inf, None, 0, 0, []
    for epoch in range(1, cfg.max_epochs + 1):
        net.train()
        perm = rng.permutation(tr_rows)
        for s in range(0, len(perm), cfg.batch_size):
            idx = perm[s:s + cfg.batch_size]
            inp, valid = _batch(data, idx, scaler, b1, device, cfg.use_b1)
            logits = net(**inp)
            y = torch.from_numpy(data.y[idx]).to(device)[:, None].expand_as(logits)
            loss = nn.functional.binary_cross_entropy_with_logits(logits[valid], y[valid])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            opt.step()
        score = grid_score(data, va_rows, _probs(net, data, va_rows, scaler, b1, device))
        history.append(score)
        if score > best + MIN_DELTA:
            best, best_epoch, wait = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    return best_state, {"seed": seed, "best_epoch": best_epoch, "n_epochs": len(history), "best_val": float(best),
                        "val_history": [float(h) for h in history]}


@dataclass
class FittedM2:
    cfg: dict
    scaler: Scaler
    states: list  # seed마다 state_dict (CPU 텐서)
    history: list  # seed마다 학습 기록
    fit_issue_hash: str
    n_fit_rows: int
    n_val_rows: int

    def nets(self, data: sequences.SequenceData, device: str) -> list[Net]:
        cfg = M2Config(**self.cfg)
        out = []
        for st in self.states:
            net = Net(cfg, data.step.shape[-1], data.cum.shape[-1], data.static.shape[-1])
            net.load_state_dict(st)
            out.append(net.to(device))
        return out


def fit(cfg: M2Config, data: sequences.SequenceData, rows: np.ndarray, seeds: list[int], device: str,
        b1: np.ndarray | None = None) -> FittedM2:
    """fit 행(rows)만으로 학습한다: 검증 조각 분리 → 표준화 → seed마다 망 학습 (조기 종료)."""
    if cfg.use_b1 and b1 is None:
        raise ValueError("use_b1 설정에는 B1 점수 행렬이 필요하다")
    configure_torch(device)
    rows = np.asarray(rows)
    tr, va = inner_split(data, rows)
    scaler = Scaler.fit(data, tr, b1 if cfg.use_b1 else None)
    states, history = [], []
    for seed in seeds:
        st, h = train_net(cfg, data, tr, va, scaler, b1, seed, device)
        states.append(st)
        history.append(h)
    return FittedM2(cfg=cfg.as_dict(), scaler=scaler, states=states, history=history,
                    fit_issue_hash=issue_hash(data.instance_id[rows]), n_fit_rows=len(tr), n_val_rows=len(va))


def predict(model: FittedM2, data: sequences.SequenceData, rows: np.ndarray, device: str, b1: np.ndarray | None = None) -> np.ndarray:
    """[len(rows), max_n] 실패 확률 = seed 망들의 확률 평균. 진행 중이 아닌 위치는 NaN."""
    configure_torch(device)
    rows = np.asarray(rows)
    if model.cfg["use_b1"] and b1 is None:
        raise ValueError("B1 포함 모델에는 B1 점수 행렬이 필요하다")
    ps = [_probs(net, data, rows, model.scaler, b1, device) for net in model.nets(data, device)]
    return np.mean(ps, axis=0)


def save(model: FittedM2, path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"meta": meta, "model": {**model.__dict__, "scaler": model.scaler.__dict__}}, tmp)
    tmp.replace(path)


def load(path) -> tuple[FittedM2, dict]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    m = dict(obj["model"])
    m["scaler"] = Scaler(**m["scaler"])
    return FittedM2(**m), obj["meta"]


def copy_config(cfg: M2Config, **changes) -> M2Config:
    return M2Config(**{**cfg.as_dict(), **changes})


