"""M3 ModernBERT fine-tune (Phase 7): 궤적 앞 N스텝 원문(afp.m3_text) → 실패 확률. 모델 하나로 N ∈ N_GRID (P6-D2와 같은 방식).

- 망: AutoModelForSequenceClassification(ModernBERT-base, 고정 revision, num_labels=1, 평균 풀링) → logit 하나. 손실 BCE
- 학습 예 (D7 범위 축소): epoch마다 시점 N ∈ n_grid별로 이슈당 최대 per_issue개 궤적(그 시점 진행 중)을 새로 뽑는다.
  같은 궤적이 다른 N 예로 같이 뽑힐 수 있다. 뽑기는 seed와 epoch로만 정한다
- 검증·조기 종료 (CLAUDE.md 누수 방지): fit 행(= fold k를 뺀 train)만 쓴다. 그 안에서 이슈 단위로 1/VAL_PARTS를 검증 조각으로
  뗀다 (lstm.inner_split과 같은 분할기). 검증 예는 이슈당 최대 val_per_issue개 (고정). 검증 점수 = N별 AUROC 평균.
  epoch마다 evals_per_epoch번 재고 최고 점수의 가중치를 쓴다. fold k와 test는 보지 않는다
- 배치: 길이가 비슷한 예를 묶는다 (덩어리 안 정렬). 최적화 한 걸음 = eff_batch개 예, 메모리는 토큰 예산(tokens_per_batch)으로
  잘게 나눠 기울기를 모은다. 손실 = BCE 합 / 걸음의 예 수
- 재현성: torch.use_deterministic_algorithms(True). 같은 기기·같은 설정이면 다시 학습해도 가중치가 같다 [확인: 작은 실험 2회].
  예측은 저장된 가중치로만 내고 정렬·배치가 예 집합으로 정해져 같은 예 집합이면 비트 단위로 같다
"""
import hashlib
import os
import time
from dataclasses import asdict, dataclass

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# 길이가 제각각인 배치가 캐싱 할당자를 조각내 12GB에서 OOM 재시도가 났다. 값에는 영향 없음
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from afp import io, m3_text, split

N_GRID = (5, 10, 20)  # D7
VAL_PARTS = 8
GRAD_CLIP = 1.0
CHUNK = 1024  # 길이 정렬 덩어리 (예 수)
PRED_TOKENS = 32768  # 예측 배치 토큰 예산
LOGIT_CLIP = 1e-6
WARN_VAL_AUROC = 0.55


@dataclass(frozen=True)
class M3Config:
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_frac: float = 0.06
    max_epochs: int = 2
    per_issue: int = 4
    val_per_issue: int = 4
    eff_batch: int = 32
    tokens_per_batch: int = 8192
    evals_per_epoch: int = 2
    n_grid: tuple = N_GRID
    tiny: bool = False  # 점검용: 무작위 초기화한 아주 작은 ModernBERT (학습된 가중치를 읽지 않는다)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["n_grid"] = list(self.n_grid)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "M3Config":
        return cls(**{**d, "n_grid": tuple(d["n_grid"])})


def configure_torch(device: str) -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 쓸 수 없다. --device cpu로 실행한다")


def issue_hash(instance_ids) -> str:
    return hashlib.sha256("\x00".join(sorted(set(map(str, instance_ids)))).encode()).hexdigest()


# ---- 망 ----

def tiny_config():
    from transformers import ModernBertConfig
    return ModernBertConfig(vocab_size=50368, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=2,
                            global_attn_every_n_layers=2, local_attention=128, max_position_embeddings=m3_text.MAX_LEN,
                            num_labels=1, pad_token_id=50283, bos_token_id=50281, eos_token_id=50282, cls_token_id=50281,
                            sep_token_id=50282)


def new_model(cfg: M3Config, seed: int):
    """학습 시작 모델. 분류 머리 초기화는 seed로 정한다."""
    from transformers import AutoModelForSequenceClassification, ModernBertForSequenceClassification
    torch.manual_seed(seed)
    if cfg.tiny:
        c = tiny_config()
        c._attn_implementation = "sdpa"
        return ModernBertForSequenceClassification(c)
    return AutoModelForSequenceClassification.from_pretrained(m3_text.MODEL_NAME, revision=m3_text.REVISION, num_labels=1,
                                                              attn_implementation="sdpa", dtype=torch.float32)


def model_from_state(model_config: dict, state: dict):
    from transformers import ModernBertConfig, ModernBertForSequenceClassification
    c = ModernBertConfig.from_dict(model_config)
    c._attn_implementation = "sdpa"
    m = ModernBertForSequenceClassification(c)
    m.load_state_dict(state)
    return m


# ---- 예 뽑기 ----

def inner_split(data: m3_text.TextData, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """fit 행 → (학습 조각, 검증 조각). 이슈 단위, 실패 비율 층화, io.SEED 고정 (lstm.inner_split과 같은 규칙)."""
    rows = np.asarray(rows)
    splitter = split.fold_splitter(VAL_PARTS, seed=io.SEED)
    _, va = next(splitter.split(np.zeros(len(rows)), data.y[rows], data.instance_id[rows]))
    mask = np.zeros(len(rows), dtype=bool)
    mask[va] = True
    return rows[~mask], rows[mask]


def all_examples(data: m3_text.TextData, rows: np.ndarray, n_grid=N_GRID) -> tuple[np.ndarray, np.ndarray]:
    """rows 중 시점 N에 진행 중인 궤적 전부 → (행, N) 배열."""
    rows = np.asarray(rows)
    rr, nn = [], []
    for n in n_grid:
        r = rows[data.lengths[rows] >= n]
        rr.append(r)
        nn.append(np.full(len(r), n, dtype=np.int64))
    return np.concatenate(rr), np.concatenate(nn)


def sample_examples(data: m3_text.TextData, rows: np.ndarray, n_grid, per_issue: int, rng: np.random.Generator):
    """시점 N마다 이슈당 최대 per_issue개 (진행 중 궤적 중 무작위). (행, N) 배열, 행 순서로 정렬."""
    rr, nn = [], []
    rows = np.sort(np.asarray(rows))
    for n in n_grid:
        r = rows[data.lengths[rows] >= n]
        if not len(r):
            continue
        _, inst = np.unique(data.instance_id[r].astype(str), return_inverse=True)
        u = rng.random(len(r))
        order = np.lexsort((u, inst))
        g = inst[order]
        first = np.searchsorted(g, g, side="left")
        rank = np.arange(len(g)) - first
        pick = np.sort(r[order[rank < per_issue]])
        rr.append(pick)
        nn.append(np.full(len(pick), n, dtype=np.int64))
    return np.concatenate(rr), np.concatenate(nn)


def plan_epoch(data, ex_rows, ex_n, cfg: M3Config, rng) -> list[list[np.ndarray]]:
    """한 epoch의 최적화 걸음 목록. 걸음 = 예 인덱스 묶음(micro-batch) 목록. 섞기 → CHUNK 안 길이 정렬 → eff_batch개씩 → 토큰 예산으로 나눔."""
    lens = data.seq_len(ex_rows, ex_n)
    perm = rng.permutation(len(ex_rows))
    steps = []
    for s in range(0, len(perm), CHUNK):
        c = perm[s:s + CHUNK]
        c = c[np.argsort(lens[c], kind="stable")]
        for b in range(0, len(c), cfg.eff_batch):
            steps.append(micro_batches(c[b:b + cfg.eff_batch], lens, cfg.tokens_per_batch))
    order = rng.permutation(len(steps))
    return [steps[i] for i in order]


def micro_batches(idx: np.ndarray, lens: np.ndarray, tokens: int) -> list[np.ndarray]:
    """길이 오름차순 idx → 토큰 예산(배치 크기 × 배치 최대 길이 <= tokens) 안의 묶음들. 예 하나는 항상 들어간다."""
    out, cur = [], []
    for i in idx:
        if cur and (len(cur) + 1) * max(lens[cur[-1]], lens[i]) > tokens:
            out.append(np.array(cur))
            cur = []
        cur.append(i)
    if cur:
        out.append(np.array(cur))
    return out


def collate(data: m3_text.TextData, rows: np.ndarray, ns: np.ndarray, device: str):
    seqs = [data.ids(int(r), int(n)) for r, n in zip(rows, ns)]
    T = max(len(s) for s in seqs)
    ids = np.full((len(seqs), T), data.pad_id, dtype=np.int64)
    mask = np.zeros((len(seqs), T), dtype=np.int64)
    for j, s in enumerate(seqs):
        ids[j, :len(s)] = s
        mask[j, :len(s)] = 1
    return torch.from_numpy(ids).to(device), torch.from_numpy(mask).to(device)


def _logits(model, ids, mask, device):
    with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16,
                        enabled=device.startswith("cuda")):
        return model(input_ids=ids, attention_mask=mask).logits.squeeze(-1).float()


def predict_examples(model, data: m3_text.TextData, rows: np.ndarray, ns: np.ndarray, device: str,
                     tokens: int = PRED_TOKENS) -> np.ndarray:
    """(행, N) 예마다 실패 확률 (float64). 길이 정렬 후 토큰 예산 배치. 같은 예 집합이면 같은 배치라 결과가 같다."""
    rows, ns = np.asarray(rows), np.asarray(ns)
    out = np.full(len(rows), np.nan)
    if not len(rows):
        return out
    lens = data.seq_len(rows, ns)
    order = np.lexsort((ns, rows, lens))  # 길이 → 행 → N (같은 예 집합이면 순서가 같다)
    model.eval()
    with torch.no_grad():
        for mb in micro_batches(order, lens, tokens):
            ids, mask = collate(data, rows[mb], ns[mb], device)
            out[mb] = torch.sigmoid(_logits(model, ids, mask, device)).double().cpu().numpy()
    return out


def val_score(data, rows, ns, p, n_grid) -> tuple[float, dict]:
    aucs = {}
    for n in n_grid:
        m = ns == n
        y = data.y[rows[m]]
        if len(np.unique(y)) == 2:
            aucs[n] = float(roc_auc_score(y, p[m]))
    return (float(np.mean(list(aucs.values()))) if aucs else 0.5), aucs


# ---- 학습 ----

@dataclass
class FittedM3:
    cfg: dict
    model_config: dict
    state: dict  # CPU 텐서 (fp32)
    history: list  # 검증 기록
    seed: int
    best: dict
    fit_issue_hash: str
    n_train_rows: int
    n_val_rows: int
    examples_per_epoch: list
    n_val_examples: int
    seconds: float

    def model(self, device: str):
        return model_from_state(self.model_config, self.state).to(device)


def fit(cfg: M3Config, data: m3_text.TextData, rows: np.ndarray, seed: int, device: str, log=print) -> FittedM3:
    """fit 행(rows)만으로 학습: 검증 조각 분리 → epoch마다 예 뽑기 → AdamW + 선형 warmup·감소 → 검증 최고 가중치."""
    configure_torch(device)
    t0 = time.time()
    rows = np.asarray(rows)
    tr, va = inner_split(data, rows)
    va_rows, va_n = sample_examples(data, va, cfg.n_grid, cfg.val_per_issue, np.random.default_rng([seed, 0]))
    plans, n_ex = [], []
    for e in range(cfg.max_epochs):
        rng = np.random.default_rng([seed, 1 + e])
        er, en = sample_examples(data, tr, cfg.n_grid, cfg.per_issue, rng)
        plans.append((er, en, plan_epoch(data, er, en, cfg, rng)))
        n_ex.append(len(er))
    total = sum(len(p[2]) for p in plans)
    warm = max(1, int(round(cfg.warmup_frac * total)))

    model = new_model(cfg, seed).to(device)
    model_config = model.config.to_dict()
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=cfg.lr, foreach=False)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    best, best_state, history, step = {"val": -np.inf}, None, [], 0
    for e, (er, en, steps) in enumerate(plans):
        marks = {int(round(len(steps) * (j + 1) / cfg.evals_per_epoch)) - 1: j for j in range(cfg.evals_per_epoch)}
        loss_sum, loss_n = 0.0, 0
        for si, mbs in enumerate(steps):
            model.train()
            n_step = sum(len(mb) for mb in mbs)
            for mb in mbs:
                ids, mask = collate(data, er[mb], en[mb], device)
                y = torch.from_numpy(data.y[er[mb]]).to(device)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(_logits(model, ids, mask, device), y, reduction="sum")
                (loss / n_step).backward()
                loss_sum += float(loss.detach())
                loss_n += len(mb)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if si in marks:
                p = predict_examples(model, data, va_rows, va_n, device)
                score, aucs = val_score(data, va_rows, va_n, p, cfg.n_grid)
                rec = {"epoch": e + 1, "eval": marks[si] + 1, "step": step, "train_loss": loss_sum / max(1, loss_n), "val": score,
                       **{f"val_auroc_N{n}": aucs.get(n) for n in cfg.n_grid}, "seconds": round(time.time() - t0, 1)}
                history.append(rec)
                log(f"    epoch {e + 1} 검증 {marks[si] + 1}/{cfg.evals_per_epoch}: 걸음 {step}/{total}, 학습 손실 {rec['train_loss']:.4f}, "
                    f"검증 AUROC {score:.4f} {{{', '.join(f'{n}: {v:.3f}' for n, v in aucs.items())}}} [{rec['seconds']:.0f}s]")
                loss_sum, loss_n = 0.0, 0
                if score > best["val"]:
                    best = {"val": score, "epoch": e + 1, "eval": marks[si] + 1, "step": step}
                    best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    del model, opt
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return FittedM3(cfg=cfg.as_dict(), model_config=model_config, state=best_state, history=history, seed=seed, best=best,
                    fit_issue_hash=issue_hash(data.instance_id[rows]), n_train_rows=len(tr), n_val_rows=len(va),
                    examples_per_epoch=n_ex, n_val_examples=len(va_rows), seconds=round(time.time() - t0, 1))


def save(model: FittedM3, path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"meta": meta, "model": model.__dict__}, tmp)
    tmp.replace(path)


def load(path) -> tuple[FittedM3, dict]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return FittedM3(**obj["model"]), obj["meta"]


# ---- B1 결합 (m3_bert_b1, late-fusion stacking) ----

def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), LOGIT_CLIP, 1 - LOGIT_CLIP)
    return np.log(p / (1 - p))


def fit_stacker(z_m3: np.ndarray, z_b1: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """[logit M3, logit B1] → 실패. 시점 N 하나의 행으로 학습한다 (호출하는 쪽이 N별로 나눈다)."""
    return LogisticRegression(C=1.0, max_iter=1000).fit(np.column_stack([z_m3, z_b1]), y.astype(int))


def apply_stacker(lr: LogisticRegression, z_m3: np.ndarray, z_b1: np.ndarray) -> np.ndarray:
    return lr.predict_proba(np.column_stack([z_m3, z_b1]))[:, 1]
