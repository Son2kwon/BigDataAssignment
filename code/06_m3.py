"""Phase 7: M3 ModernBERT fine-tune (N=5, 10, 20, D7) — 개발(train OOF만)과 최종(--final, test 한 번 채점)을 나눈다.

실행:
  uv run --no-sync python code/06_m3.py                       # 개발: fold 모델 5개 학습 + train OOF만 (GPU 약 4~4.5시간, test를 보지 않는다)
  uv run --no-sync python code/06_m3.py --final               # 최종: 저장된 fold 모델로 test 채점·평가 (약 40분)
  uv run --no-sync python code/06_m3.py --smoke [--final]     # 점검: 무작위 초기화 작은 모델 + train 이슈 일부 (실제 test 안 씀)

중간에 멈추면 같은 명령을 다시 실행한다. 학습이 끝난 fold 모델과 채점 결과는 다시 쓴다 (meta가 같을 때).

입력: data/processed/trajectories.parquet, steps.parquet (01), splits.parquet (02), preds/b1_n*.parquet (03),
      results/phase4·5·6 (비교 표), 최종은 preds/m1_xgb*·m2_lstm*·m2x_nocum* (짝지은 비교, 이 머신의 예측 파일)
      ModernBERT-base 가중치·토크나이저 (HF 캐시, revision 고정 afp.m3_text.REVISION)
출력:
- 개발: data/processed/m3/ (토큰 배열, fold 모델 약 0.6GB × 5, 채점 캐시, OOF), results/phase7/train_log.csv, fold_models.csv,
  stacker.csv, oof_compare.csv, train_operating_points.csv, summary_dev.json (모두 train OOF 값)
- 최종: data/processed/preds/m3_*.parquet + .test_folds.parquet, results/phase7/ 판별·운영점·비교·짝지은 비교·N 요약·summary.json

방법 (docs/phase7.md 7.0 사전 등록):
- m3_bert: ModernBERT-base 하나로 N ∈ {5, 10, 20} (afp.m3). fold k 모델 = fold k를 뺀 train으로 학습 (조기 종료·검증은 그 안의
  이슈 단위 검증 조각). 학습 예는 epoch마다 N별 이슈당 최대 4궤적. 하이퍼파라미터 그리드 없음 (고정값, 비용)
- m3_bert_b1: N별 로지스틱 회귀 [logit M3, logit B1] (late fusion). fold k 결합기는 fold k를 뺀 train OOF로 학습해 fold k에 적용.
  test fold k = M3 fold k 점수 + B1 fold k 점수 → 결합기 k (P4-D7 ①). ②는 민감도
- 평가: afp/evaluate.py 공용 함수 (n_grid = (5, 10, 20)). 운영점은 train OOF에서만 고르고 test는 fold 모델별 채점 평균 (P4-D6)
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from afp import evaluate, io, lstm, m3, m3_text, metrics, oof, preds, split

MODEL_ID = "m3_bert"
FUSED_ID = "m3_bert_b1"
MODELS = (MODEL_ID, FUSED_ID)
N_GRID = m3.N_GRID
B1_MODEL = "b1_n"
P4_MODELS = ("b0_err", "b0_editfail", "b1_static", "b1_n")
M1_MODELS = ("m1_lr", "m1_lr_b1", "m1_xgb", "m1_xgb_b1")
M2_MODELS = ("m2_lstm", "m2_lstm_b1")
M2X_MODELS = ("m2x_nocum", "m2x_nocum_b1")
# 짝지은 비교 (사전 등록): 주 = 같은 B1 조건의 M1 XGB, 보조 = 공식 M2와 누적 집계 없는 M2 (P6-D11)
PAIRS = [(MODEL_ID, "m1_xgb"), (FUSED_ID, "m1_xgb_b1"), (MODEL_ID, "m2_lstm"), (FUSED_ID, "m2_lstm_b1"),
         (MODEL_ID, "m2x_nocum"), (FUSED_ID, "m2x_nocum_b1")]
CACHE_VERSION = 1
FLOAT = 6
DEFAULT_CFG = m3.M3Config()
SMOKE_CFG = m3.M3Config(tiny=True, max_epochs=1, per_issue=1, val_per_issue=2, eff_batch=8, tokens_per_batch=8192)


class Paths:
    def __init__(self, smoke: bool):
        base = io.PROCESSED_DIR / ("m3_smoke" if smoke else "m3")
        self.cache = base
        self.text_train = base / "text_train"
        self.text_test = base / "text_test"
        self.models = base / "models"
        self.scores = base / "scores"
        self.out = base / "results" if smoke else io.RESULTS_DIR / "phase7"
        self.preds = base / "preds" if smoke else preds.PREDS_DIR

    def model(self, k: int) -> Path:
        return self.models / f"fold{k}.pt"

    def model_info(self, k: int) -> Path:
        return self.models / f"fold{k}.json"

    def score(self, which: str, k: int) -> Path:
        return self.scores / f"{which}_fold{k}.parquet"

    def oof(self, model_id: str) -> Path:
        return self.cache / f"oof_{model_id}.parquet"


def _canon(x) -> str:
    return json.dumps(x, sort_keys=True, default=str)


def _hash(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(np.ascontiguousarray(p).tobytes() if isinstance(p, np.ndarray) else str(p).encode())
    return h.hexdigest()


def meta_for(cfg: m3.M3Config, k: int, seed: int, data: m3_text.TextData, device: str) -> dict:
    return {"version": CACHE_VERSION, "model_id": MODEL_ID, "cfg": cfg.as_dict(), "fold": int(k), "seed": int(seed),
            "data_fingerprint": data.fingerprint, "fit_issue_hash": m3.issue_hash(data.instance_id[data.fold != k]),
            "model": m3_text.MODEL_NAME, "revision": m3_text.REVISION, "budget": m3_text.budget(), "device": device}


def seed_for(fold: int) -> int:
    """fold 모델의 seed (M2와 같은 규칙: io.SEED와 fold 번호로만 정한다)."""
    return lstm.seeds_for(fold, 1)[0]


# ---- 학습 ----

def train_folds(paths: Paths, data: m3_text.TextData, cfg: m3.M3Config, device: str, force: bool) -> list[dict]:
    """fold 모델 5개를 차례로 학습한다 (GPU 하나). 같은 meta의 모델이 있으면 건너뛴다. 학습 기록(fold.json)을 돌려준다."""
    infos = []
    for k in range(split.N_FOLDS):
        meta = meta_for(cfg, k, seed_for(k), data, device)
        p, pi = paths.model(k), paths.model_info(k)
        if p.exists() and pi.exists() and not force:
            info = json.loads(pi.read_text())
            if _canon(info["meta"]) == _canon(meta):
                print(f"  fold {k}: 저장된 모델 사용 (검증 AUROC {info['best']['val']:.4f})", flush=True)
                infos.append({**info, "trained": False})
                continue
        rows = np.flatnonzero(data.fold != k)
        print(f"  fold {k}: 학습 시작 (fit 궤적 {len(rows)}, seed {meta['seed']})", flush=True)
        model = m3.fit(cfg, data, rows, meta["seed"], device)
        if model.fit_issue_hash != meta["fit_issue_hash"]:
            raise AssertionError("fit 이슈 집합이 meta와 다르다")
        m3.save(model, p, meta)
        info = {"meta": meta, "history": model.history, "best": model.best, "seconds": model.seconds,
                "n_train_rows": model.n_train_rows, "n_val_rows": model.n_val_rows,
                "examples_per_epoch": model.examples_per_epoch, "n_val_examples": model.n_val_examples}
        pi.write_text(json.dumps(io.round_floats(info), indent=1, default=str))
        print(f"  fold {k}: 완료 {model.seconds / 60:.1f}분, 최고 검증 AUROC {model.best['val']:.4f} "
              f"(epoch {model.best['epoch']}, 검증 {model.best['eval']})", flush=True)
        infos.append({**info, "trained": True})
        del model
    return infos


# ---- 채점 (캐시) ----

class Scorer:
    """fold 모델 채점. (모델 meta, 데이터, 예 집합)이 같으면 저장된 점수를 다시 쓴다. 모델은 필요할 때만 GPU에 올린다."""

    def __init__(self, paths: Paths, device: str, force: bool = False):
        self.paths, self.device, self.force = paths, device, force
        self._k, self._model = None, None

    def model(self, k: int):
        if self._k != k:
            self._model = None
            fitted, _ = m3.load(self.paths.model(k))
            self._model, self._k = fitted.model(self.device), k
        return self._model

    def score(self, which: str, k: int, data: m3_text.TextData, rows: np.ndarray, ns: np.ndarray, use_cache: bool = True) -> np.ndarray:
        info = json.loads(self.paths.model_info(k).read_text())
        key = _hash(_canon(info["meta"]), data.fingerprint, rows.astype(np.int64), ns.astype(np.int64))
        f = self.paths.score(which, k)
        kf = f.with_suffix(".key")
        if use_cache and not self.force and f.exists() and kf.exists() and kf.read_text() == key:
            return pl.read_parquet(f)["p_fail"].to_numpy()
        t0 = time.time()
        p = m3.predict_examples(self.model(k), data, rows, ns, self.device)
        if use_cache:
            f.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame({"traj_id": data.traj_id[rows].astype(str), "N": ns.astype(np.int16), "p_fail": p}).write_parquet(f)
            kf.write_text(key)
        print(f"    채점 {which} fold {k}: {len(rows)}예 {time.time() - t0:.0f}초", flush=True)
        return p


def long_rows(data: m3_text.TextData, n_grid=N_GRID) -> pl.DataFrame:
    """(궤적, N) 행: 시점 N 진행 중 궤적마다 하나. oof.oof_predict의 행 단위."""
    parts = []
    for n in n_grid:
        r = data.running(n)
        parts.append(pl.DataFrame({"instance_id": data.instance_id[r].astype(str), "fold": np.asarray(data.fold)[r],
                                   "_i": r, "N": np.full(len(r), n, dtype=np.int64)}))
    return pl.concat(parts)


def collect(paths: Paths, data_tr, scorer: Scorer, data_te=None) -> dict:
    """저장된 fold 모델로 train OOF와 (data_te가 있으면) test fold 모델별 점수.

    fit 콜백은 학습하지 않고 fold k 모델의 기록을 읽는다: 받은 행에 없는 fold가 k이고, 학습 이슈 해시가 "fold k를 뺀 train
    이슈"와 같아야 한다. 받은 행의 이슈는 그 집합 안에 있어야 한다 (oof.oof_predict가 seen 검사).
    """
    train_rows = long_rows(data_tr)
    test_rows = None if data_te is None else long_rows(data_te).with_columns(_src=pl.lit(-1))
    allowed = {k: set(data_tr.instance_id[np.asarray(data_tr.fold) != k]) for k in range(split.N_FOLDS)}

    def fit(rows):
        missing = set(range(split.N_FOLDS)) - set(rows["fold"].unique().to_list())
        if len(missing) != 1:
            raise AssertionError(f"fit 행에서 빠진 fold가 하나가 아니다: {missing}")
        k = missing.pop()
        meta = json.loads(paths.model_info(k).read_text())["meta"]
        if meta["fit_issue_hash"] != m3.issue_hash(allowed[k]) or meta["data_fingerprint"] != data_tr.fingerprint:
            raise AssertionError(f"fold {k}: 저장된 모델의 학습 이슈·데이터가 다르다")
        if not set(rows["instance_id"].unique().to_list()) <= allowed[k]:
            raise AssertionError("fit 행에 학습 이슈 밖 이슈가 있다")
        return k

    def predict(k, rows):
        is_test = "_src" in rows.columns
        data = data_te if is_test else data_tr
        return scorer.score("test" if is_test else "train", k, data, rows["_i"].to_numpy(), rows["N"].to_numpy())

    res = oof.oof_predict(train_rows, fit, predict, test=test_rows)
    tr = res.train.select(traj_id=pl.Series(data_tr.traj_id[res.train["_i"].to_numpy()].astype(str)),
                          N=pl.col("N").cast(pl.Int16), p_fail="p_fail").sort("N", "traj_id")
    out = {"train": tr, "seen_issue_scores": res.seen_issue_scores}
    if test_rows is not None:
        ids = data_te.traj_id[res.test["_i"].to_numpy()].astype(str)
        n = res.test["N"].cast(pl.Int16)
        out["test_mean"] = pl.DataFrame({"traj_id": ids, "N": n, "p_fail": res.test["p_fail"]}).sort("N", "traj_id")
        out["test_folds"] = pl.concat([pl.DataFrame({"traj_id": ids, "N": n, "fold": np.full(len(ids), k, dtype=np.int8),
                                                     "p_fail": res.test_fold_scores[k]}) for k in range(split.N_FOLDS)])
    return out


def verify_rescore(paths: Paths, scorer: Scorer, data_tr, k: int = 0) -> dict:
    """재현성 점검: fold k train 점수를 캐시 없이 다시 계산해 개발 실행의 캐시와 비교한다 (같은 예 집합 → 비트 단위로 같아야 한다)."""
    r = long_rows(data_tr).filter(pl.col("fold") == k)
    rows, ns = r["_i"].to_numpy(), r["N"].to_numpy()
    cached = pl.read_parquet(paths.score("train", k))["p_fail"].to_numpy()
    fresh = scorer.score("train", k, data_tr, rows, ns, use_cache=False)
    return {"fold": k, "n": len(rows), "equal": bool(np.array_equal(cached, fresh)), "max_abs_diff": float(np.abs(cached - fresh).max())}


# ---- B1 결합 (m3_bert_b1) ----

def _with_inputs(m3p: pl.DataFrame, b1p: pl.DataFrame, trajs: pl.DataFrame) -> pl.DataFrame:
    """(traj_id, N) 행에 z_m3, z_b1, y, instance_id, fold를 붙인다. 순서는 (N, traj_id)."""
    b = b1p.select("traj_id", pl.col("N").cast(pl.Int16), pl.col("p_fail").alias("b1"))
    d = (m3p.select("traj_id", pl.col("N").cast(pl.Int16), "p_fail").join(b, on=["traj_id", "N"], how="left")
         .join(trajs.select("traj_id", "instance_id", "fold", "target"), on="traj_id", how="left").sort("N", "traj_id"))
    if d["b1"].null_count() or d["instance_id"].null_count():
        raise ValueError("B1 점수 또는 궤적 정보가 빠진 행이 있다")
    return d.with_columns(z_m3=pl.Series(m3.logit(d["p_fail"].to_numpy())), z_b1=pl.Series(m3.logit(d["b1"].to_numpy())),
                          y=(~pl.col("target")).cast(pl.Int8)).drop("p_fail", "b1", "target")


def stack_predict(stackers: dict, rows: pl.DataFrame) -> np.ndarray:
    out = np.full(rows.height, np.nan)
    nn = rows["N"].to_numpy()
    for n, lr in stackers.items():
        m = nn == n
        if m.any():
            out[m] = m3.apply_stacker(lr, rows["z_m3"].to_numpy()[m], rows["z_b1"].to_numpy()[m])
    return out


def fuse(m3_train, b1_train, train_set, m3_test_folds=None, b1_test_folds=None, test_set=None) -> dict:
    """m3_bert_b1 OOF (+ test). 결합기 k는 fold k를 뺀 train의 OOF 점수로만 학습한다 (oof.oof_predict)."""
    rows = _with_inputs(m3_train, b1_train, train_set)

    def fit(r):
        return {n: m3.fit_stacker(g["z_m3"].to_numpy(), g["z_b1"].to_numpy(), g["y"].to_numpy())
                for n in N_GRID for g in [r.filter(pl.col("N") == n)]}

    test, per_fold = None, None
    if m3_test_folds is not None:
        per_fold = [_with_inputs(m3_test_folds.filter(pl.col("fold") == k).drop("fold"), b1_test_folds[k], test_set)
                    for k in range(split.N_FOLDS)]
        for k in range(1, split.N_FOLDS):
            if not per_fold[k].select("traj_id", "N").equals(per_fold[0].select("traj_id", "N")):
                raise AssertionError("fold별 test 행 순서가 다르다")
        test = per_fold[0]
    res = oof.oof_predict(rows, fit, stack_predict, test=test,
                          test_for_fold=None if test is None else (lambda k, t: per_fold[k]), keep_models=True)
    out = {"train": res.train.select("traj_id", "N", "p_fail"), "stackers": res.models, "seen_issue_scores": res.seen_issue_scores}
    if test is not None:
        out["test_mean"] = res.test.select("traj_id", "N", "p_fail")
        out["test_folds"] = pl.concat([test.select("traj_id", "N", fold=pl.lit(k, dtype=pl.Int8), p_fail=pl.Series(res.test_fold_scores[k]))
                                       for k in range(split.N_FOLDS)])
    return out


def stacker_rows(stackers) -> list[dict]:
    rows = []
    for k, st in enumerate(stackers):
        for n, lr in st.items():
            rows.append({"fold": k, "N": n, "coef_m3": float(lr.coef_[0][0]), "coef_b1": float(lr.coef_[0][1]),
                         "intercept": float(lr.intercept_[0])})
    return rows


# ---- 개발 (train OOF만) ----

def oof_scores(model_id, train_preds, train_set) -> dict:
    d = evaluate.discrimination_table(model_id, train_preds, {"train": train_set}, N_GRID)
    out = {f"oof_auroc_N{r['N']}": r["auroc"] for r in d.iter_rows(named=True)}
    out.update({f"oof_within_pair_N{r['N']}": r["within_pair"] for r in d.iter_rows(named=True)})
    out["oof_auroc_mean"] = float(d["auroc"].mean())
    out["oof_within_pair_mean"] = float(d["within_pair"].mean())
    return out


def train_ops(train_set, p_train, costs_train) -> list[dict]:
    rows = []
    for n in N_GRID:
        for policy in evaluate.POLICIES:
            op = metrics.select_from_curves(evaluate.candidate_curves(train_set, p_train, n, policy, costs_train), n)
            rows.append({"N": n, "policy": policy, **{c: op[c] for c in ("tau", "cap", "stop_rate", "lost_success", "saved_steps", "saved_tokens")}})
    return rows


RESULT_SOURCES = (("phase5", M1_MODELS), ("phase6", M2_MODELS), ("phase6/supplementary", M2X_MODELS))


def other_train_discrimination() -> pl.DataFrame | None:
    """M1·M2 train OOF 판별 지표 (results/phase5·6 CSV, N_GRID만)."""
    parts = []
    for d, models in RESULT_SOURCES:
        p = io.RESULTS_DIR / d / "discrimination.csv"
        if p.exists():
            t = pl.read_csv(p).filter((pl.col("split") == "train_oof") & pl.col("model").is_in(list(models)) & pl.col("N").is_in(list(N_GRID)))
            parts.append(t.select("model", pl.col("N").cast(pl.Int64), pl.col("auroc").alias("train_oof_auroc"),
                                  pl.col("within_pair").alias("train_oof_within_pair")))
    return pl.concat(parts) if parts else None


def other_train_operating_points(columns) -> pl.DataFrame | None:
    parts = []
    for d, models in RESULT_SOURCES:
        p = io.RESULTS_DIR / d / "operating_points.csv"
        if p.exists():
            o = pl.read_csv(p, schema_overrides={"N": pl.Int64, "cap": pl.Int64})
            parts.append(o.filter((pl.col("split") == "train") & pl.col("model").is_in(list(models)) & pl.col("N").is_in(list(N_GRID)))
                         .select(columns))
    return pl.concat(parts, how="vertical_relaxed") if parts else None


def dev(args, paths, sets, costs, data_tr, frames, device, cfg, lap):
    t = sets["train"]
    infos = train_folds(paths, data_tr, cfg, device, args.force)
    lap("fold 모델 학습")
    scorer = Scorer(paths, device, args.force)
    r = collect(paths, data_tr, scorer)
    assert r["seen_issue_scores"] == 0
    fz = fuse(r["train"], frames["train"], t)
    assert fz["seen_issue_scores"] == 0
    trains = {MODEL_ID: r["train"], FUSED_ID: fz["train"]}
    lap("train OOF 채점·결합")

    scores, ops_rows = {}, []
    for m, p in trains.items():
        p.write_parquet(paths.oof(m))
        scores[m] = oof_scores(m, p, t)
        ops_rows += [{"model": m, **x} for x in train_ops(t, p, costs["train"])]
    cap_train = metrics.fixed_cap_curve(t, costs["train"], range(1, 81))
    k_star = metrics.select_cap(cap_train, metrics.SELECT_MAX_LOSS)["K"]
    kstar_train = evaluate.cap_point(t, k_star, costs["train"])

    out = paths.out
    out.mkdir(parents=True, exist_ok=True)
    # 실행 시간(초)은 재실행마다 달라서 results/에 쓰지 않는다 (models/fold*.json과 표준 출력에만)
    log_rows = [{"fold": i["meta"]["fold"], **{c: v for c, v in h.items() if c != "seconds"}} for i in infos for h in i["history"]]
    pl.DataFrame(log_rows).write_csv(out / "train_log.csv", float_precision=FLOAT)
    fold_rows = [{"fold": i["meta"]["fold"], "seed": i["meta"]["seed"], "best_val": i["best"]["val"], "best_epoch": i["best"]["epoch"],
                  "best_eval": i["best"]["eval"], "best_step": i["best"]["step"], "n_train_rows": i["n_train_rows"],
                  "n_val_rows": i["n_val_rows"], "examples_per_epoch": json.dumps(i["examples_per_epoch"]),
                  "n_val_examples": i["n_val_examples"]} for i in infos]
    pl.DataFrame(fold_rows).write_csv(out / "fold_models.csv", float_precision=FLOAT)
    st = pl.DataFrame(stacker_rows(fz["stackers"]))
    st.write_csv(out / "stacker.csv", float_precision=FLOAT)
    cmp_rows = [{"model": m, "N": n, "train_oof_auroc": s[f"oof_auroc_N{n}"], "train_oof_within_pair": s[f"oof_within_pair_N{n}"]}
                for m, s in scores.items() for n in N_GRID]
    other = other_train_discrimination()
    cmp_df = pl.concat([pl.DataFrame(cmp_rows), other]) if other is not None else pl.DataFrame(cmp_rows)
    cmp_df.write_csv(out / "oof_compare.csv", float_precision=FLOAT)
    ops_df = pl.DataFrame(ops_rows).with_columns(d_saved_steps=pl.col("saved_steps") - kstar_train["saved_steps"],
                                                 d_saved_tokens=pl.col("saved_tokens") - kstar_train["saved_tokens"])
    other_ops = other_train_operating_points(ops_df.columns)
    if other_ops is not None:
        ops_df = pl.concat([ops_df, other_ops], how="vertical_relaxed")
    ops_df.write_csv(out / "train_operating_points.csv", float_precision=FLOAT)

    weak = [i["meta"]["fold"] for i in infos if i["best"]["val"] < m3.WARN_VAL_AUROC]
    summary = {
        "mode": "dev (train OOF only, test not scored)",
        "device": device, "model": m3_text.MODEL_NAME, "revision": m3_text.REVISION,
        "cfg": cfg.as_dict(), "budget": m3_text.budget(),
        "fold_models": fold_rows,
        "train_oof": scores,
        "stacker_coef_mean": st.group_by("N").agg(pl.col("coef_m3").mean(), pl.col("coef_b1").mean()).sort("N").to_dicts(),
        "k_star_train": {"K": k_star, **kstar_train},
        "weak_folds": weak,
        "data": {"train_trajectories": data_tr.n, "fingerprint": data_tr.fingerprint, "tokens": int(len(data_tr.tokens))},
    }
    io.write_summary(out / "summary_dev.json", summary)

    with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=220, float_precision=4):
        print(pl.DataFrame(fold_rows).with_columns(minutes=pl.Series([round(i["seconds"] / 60, 1) for i in infos])))
        print(cmp_df.pivot(on="N", index="model", values="train_oof_auroc"))
        print(cmp_df.pivot(on="N", index="model", values="train_oof_within_pair"))
        print(ops_df.sort("N", "policy", "model"))
        print(st.group_by("N").agg(pl.col("coef_m3").mean(), pl.col("coef_b1").mean()).sort("N"))
    if weak:
        print(f"\n[주의] 검증 AUROC < {m3.WARN_VAL_AUROC}인 fold 모델: {weak} (학습이 안 됐을 수 있음). 보고 후 --final")
    else:
        print("\n개발 실행 완료. 결과를 보고한 뒤 --final 을 실행한다")


# ---- 최종 ----

def load_b1_frames(final: bool, smoke: bool) -> dict:
    """B1 점수 (traj_id, N, p_fail; N_GRID만). train = OOF. 최종이면 test fold k 점수(P4-D7 ①)와 ② 민감도용 j≠k 점수도 읽는다.
    점검 모드의 가짜 test는 train 궤적이라 OOF 점수를 모든 fold에 쓴다."""
    keep = pl.col("N").is_in(list(N_GRID))
    tr = preds.load_feature(B1_MODEL, "train").filter(keep)
    out = {"train": tr, "test": None, "other": None}
    if final and smoke:
        out["test"] = [tr] * split.N_FOLDS
        out["other"] = {k: pl.concat([tr.with_columns(src_fold=pl.lit(j, dtype=pl.Int8)) for j in range(split.N_FOLDS) if j != k])
                        for k in range(split.N_FOLDS)}
    elif final:
        out["test"] = [preds.load_feature(B1_MODEL, "test", fold=k).filter(keep) for k in range(split.N_FOLDS)]
        out["other"] = {k: preds.load_feature_other_folds(B1_MODEL, k).filter(keep) for k in range(split.N_FOLDS)}
    return out


def b1_input_sensitivity(fz, m3_test_folds, frames, sets, costs, chosen, model_preds) -> list[dict]:
    """P4-D7 ② vs ①: 결합기 k에 M3 fold k 점수와 B1 fold j≠k 점수를 넣는다. 판별 = 20개 점수 평균, 운영점 = 20개 결과 평균."""
    test = sets["test"]
    parts = []
    for k in range(split.N_FOLDS):
        mk = m3_test_folds.filter(pl.col("fold") == k).drop("fold")
        o = frames["other"][k]
        for j in sorted(o["src_fold"].unique().to_list()):
            rows = _with_inputs(mk, o.filter(pl.col("src_fold") == j).select("traj_id", "N", "p_fail"), test)
            parts.append(rows.select("traj_id", "N", fold=pl.lit(10 * k + j, dtype=pl.Int16),
                                     p_fail=pl.Series(stack_predict(fz["stackers"][k], rows))))
    long = pl.concat(parts)
    mean2 = long.group_by("traj_id", "N").agg(pl.col("p_fail").mean()).sort("N", "traj_id")
    rows = []
    for n in N_GRID:
        p1 = evaluate.test_preds(*model_preds[FUSED_ID], test, n)
        p2 = evaluate.attach_fold_scores(mean2.filter(pl.col("N") == n), long.filter(pl.col("N") == n))
        d1 = evaluate.discrimination_row(test, p1.select("traj_id", "N", "p_fail"), n)
        d2 = evaluate.discrimination_row(test, mean2.filter(pl.col("N") == n), n)
        rows += [{"model": FUSED_ID, "N": n, "kind": "discrimination", "policy": None, "metric": c, "main_1": d1[c], "sens_2": d2[c]}
                 for c in evaluate.PAIR_DISC]
        for policy in evaluate.POLICIES:
            tau, cap = chosen[(FUSED_ID, n, policy)]
            a = evaluate.test_point(test, p1, n, tau, cap, costs["test"], spread=False)
            b = evaluate.test_point(test, p2, n, tau, cap, costs["test"], spread=False)
            rows += [{"model": FUSED_ID, "N": n, "kind": "operating_point", "policy": policy, "metric": c, "main_1": a[c], "sens_2": b[c]}
                     for c in evaluate.POINT_COLUMNS]
    return rows


COMPARE_SOURCES = (("phase4", P4_MODELS), ("phase5", M1_MODELS), ("phase6", M2_MODELS), ("phase6/supplementary", M2X_MODELS))


def comparison_tables(out_dir, disc, ops_df, smoke: bool) -> dict:
    """B0·B1(phase4) + M1(phase5) + M2(phase6, 공식 + 보조) + M3 → 판별 비교, 운영점 비교, N 요약, 손실 구간 폭 (N_GRID만).
    다른 모델 값은 결과 파일 그대로."""
    so = {"N": pl.Int64}
    in_grid = pl.col("N").is_in(list(N_GRID))
    d_parts, ci_parts, o_parts = [], [], []
    for d, models in COMPARE_SOURCES:
        base = io.RESULTS_DIR / d
        d_parts.append(pl.read_csv(base / "discrimination.csv", schema_overrides=so).filter(pl.col("model").is_in(list(models)) & in_grid))
        ci_parts.append(pl.read_csv(base / "discrimination_ci.csv", schema_overrides=so).filter(pl.col("model").is_in(list(models)) & in_grid))
        o = pl.read_csv(base / "operating_points.csv", schema=ops_df.schema)
        o_parts.append(o.filter(pl.col("model").is_in(list(models)) & in_grid) if d != "phase4" else
                       o.filter((pl.col("model") == "b0_cap") | (pl.col("model").is_in(list(models)) & in_grid)))
    dall = pl.concat([x.select(disc.columns).cast(disc.schema) for x in d_parts] + [disc])
    ci_parts.append(pl.read_csv(out_dir / "discrimination_ci.csv", schema_overrides=so))
    ci = pl.concat([c.select("model", "split", "N", "stat", "lo", "hi") for c in ci_parts]).pivot(
        on="stat", index=["model", "split", "N"], values=["lo", "hi"])
    ci = ci.rename({c: (c.split("_", 1)[1] + "_" + c.split("_", 1)[0]) for c in ci.columns if c.startswith(("lo_", "hi_"))})
    dcomp = dall.join(ci, on=["model", "split", "N"], how="left")
    keep = ["model", "split", "N", "n_running", "base_rate", "within_n_issues"]
    for c in evaluate.PAIR_DISC:
        keep += [c] + [x for x in (f"{c}_lo", f"{c}_hi") if x in dcomp.columns]
    dcomp = dcomp.select(keep).sort("split", "N", "model")
    dcomp.write_csv(out_dir / "discrimination_compare.csv", float_precision=FLOAT)

    cap4, cap7 = o_parts[0].filter(pl.col("model") == "b0_cap"), ops_df.filter(pl.col("model") == "b0_cap")
    num = [c for c, t in ops_df.schema.items() if t == pl.Float64]
    cap_same = cap4.height == cap7.height and all(
        np.allclose(cap4[c].fill_null(np.nan).to_numpy(), cap7[c].fill_null(np.nan).to_numpy(), atol=1e-6, equal_nan=True) for c in num)
    ocomp = pl.concat(o_parts + [ops_df.filter(pl.col("model") != "b0_cap")])
    ocomp.write_csv(out_dir / "operating_points_compare.csv", float_precision=FLOAT)
    nsum = evaluate.n_summary(ocomp.filter(pl.col("model").is_in(["b0_cap", *M1_MODELS, *M2_MODELS, *M2X_MODELS, *MODELS])))
    nsum.write_csv(out_dir / "n_summary.csv", float_precision=FLOAT)

    t = ocomp.filter(pl.col("split") == "test").with_columns(half_width=(pl.col("lost_success_hi") - pl.col("lost_success_lo")) / 2)
    width = t.select("model", "policy", "N", "tau", "cap", "stop_rate", "lost_success", "lost_success_lo", "lost_success_hi", "half_width",
                     "saved_steps", "saved_tokens", "meets_judge_ci_hi")
    width.write_csv(out_dir / "loss_ci_width_compare.csv", float_precision=FLOAT)
    width_sum = width.filter(pl.col("model") != "b0_cap").group_by("model", "policy").agg(
        n=pl.len(), half_width_median=pl.col("half_width").median(), half_width_min=pl.col("half_width").min(),
        half_width_max=pl.col("half_width").max(), stop_rate_median=pl.col("stop_rate").median(),
        lost_success_median=pl.col("lost_success").median()).sort("policy", "model")
    width_sum.write_csv(out_dir / "loss_ci_width_summary.csv", float_precision=FLOAT)
    return {"discrimination": dcomp, "operating": ocomp, "n_summary": nsum, "width_summary": width_sum,
            "cap_rows_match_phase4": bool(cap_same or smoke)}


def paired_tasks(pairs, model_preds: dict, sets, costs, chosen, n_boot) -> tuple[list, list]:
    """짝지은 부트스트랩 작업 (짝·N마다). 같은 시드라 다른 표와 같은 재표본이다."""
    test = sets["test"]
    tasks, keys = [], []
    for a, b in pairs:
        for n in N_GRID:
            pa = {pol: chosen[(a, n, pol)] for pol in evaluate.POLICIES}
            pb = {pol: chosen[(b, n, pol)] for pol in evaluate.POLICIES}
            c = evaluate.compressed_costs(costs["test"], n, [k for _, k in (*pa.values(), *pb.values())], None)
            p = evaluate.pair_preds(evaluate.test_preds(*model_preds[a], test, n), evaluate.test_preds(*model_preds[b], test, n))
            tasks.append((test, p, evaluate.paired_stat(n, pa, pb, c), n_boot))
            keys.append((a, b, n))
    return tasks, keys


def final(args, paths, sets, costs, data_tr, frames, enc, device, timings, lap):
    dev_path = paths.out / "summary_dev.json"
    if not dev_path.exists():
        sys.exit(f"{dev_path}가 없다. 먼저 개발 실행(--final 없이)을 한다.")
    dev_summary = json.loads(dev_path.read_text())
    if dev_summary["data"]["fingerprint"] != data_tr.fingerprint:
        sys.exit("train 토큰 배열이 개발 실행 때와 다르다. 개발 실행을 다시 한다.")
    if dev_summary["device"] != device:
        sys.exit(f"개발 실행은 {dev_summary['device']}로 학습했다. 같은 --device로 실행한다.")
    for k in range(split.N_FOLDS):
        if not paths.model(k).exists():
            sys.exit(f"{paths.model(k)}가 없다. 개발 실행을 끝까지 한다.")
    data_te = m3_text.load(sets["test"], enc, paths.text_test)
    lap(f"test 토큰 배열 ({data_te.n}궤적)")

    scorer = Scorer(paths, device, args.force)
    r = collect(paths, data_tr, scorer, data_te)
    assert r["seen_issue_scores"] == 0
    lap("test 채점 (fold 모델 5개)")
    rescore = verify_rescore(paths, scorer, data_tr, 0)
    lap("재현성 점검 (fold 0 train 다시 채점)")
    fz = fuse(r["train"], frames["train"], sets["train"], r["test_folds"], frames["test"], sets["test"])
    assert fz["seen_issue_scores"] == 0
    b1_test = [_with_inputs(r["test_folds"].filter(pl.col("fold") == k).drop("fold"), frames["test"][k], sets["test"])["z_b1"].to_numpy()
               for k in range(split.N_FOLDS)]
    spread = float((np.max(b1_test, axis=0) - np.min(b1_test, axis=0)).max())

    results = {MODEL_ID: r, FUSED_ID: fz}
    oof_same = {m: results[m]["train"].equals(pl.read_parquet(paths.oof(m))) for m in MODELS}
    written, written_folds, model_preds = {}, {}, {}
    for m in MODELS:
        config = {"learner": "modernbert" if m == MODEL_ID else "modernbert+b1 logistic stacking",
                  "model": m3_text.MODEL_NAME, "revision": m3_text.REVISION, "cfg": dev_summary["cfg"], "budget": m3_text.budget(),
                  "b1_input": f"train={B1_MODEL} OOF, test fold k={B1_MODEL} fold k (P4-D7 ①)" if m == FUSED_ID else None, "device": device}
        p_all = pl.concat([results[m]["train"], results[m]["test_mean"]])
        written[m] = preds.write(m, p_all, sets, N_GRID, config, "5fold-oof/test-fold-mean", force=args.force)
        written_folds[m] = preds.write_test_folds(m, results[m]["test_folds"], sets["test"], N_GRID, split.N_FOLDS, force=args.force)
        model_preds[m] = (preds.read(m), preds.read_test_folds(m))
    lap("결합·예측 저장")

    out = paths.out
    disc = pl.concat([evaluate.discrimination_table(m, model_preds[m][0], sets, N_GRID) for m in MODELS])
    disc.write_csv(out / "discrimination.csv", float_precision=FLOAT)
    cap_train = metrics.fixed_cap_curve(sets["train"], costs["train"], range(1, 81))
    k_star = metrics.select_cap(cap_train, metrics.SELECT_MAX_LOSS)["K"]
    main_rows, sens_rows, chosen = evaluate.operating_points(model_preds, sets, costs, k_star, n_grid=N_GRID)
    cap_main, cap_sens = evaluate.cap_rows(sets, costs, k_star, cap_train)
    curve_dir = out / "cost_curves"
    if curve_dir.exists():
        shutil.rmtree(curve_dir)
    curve_dir.mkdir(parents=True)
    for m in MODELS:
        for (name, n), c in evaluate.model_curves(model_preds[m][0], sets, costs, N_GRID, model_preds[m][1]).items():
            evaluate.thin_curve(c).write_csv(curve_dir / f"{m}_{name}_N{n}.csv", float_precision=FLOAT)
    lap("판별 지표·운영점 선택·적용")

    # 짝지은 비교 상대 (같은 평가 함수로 운영점을 다시 고른다. 이 머신의 예측 파일, P6-D10)
    if args.smoke:
        pairs = [(FUSED_ID, MODEL_ID)]
    else:
        others = sorted({b for _, b in PAIRS})
        missing = [b for b in others if not preds.path(b).exists()]
        if missing:
            sys.exit(f"짝지은 비교 상대의 예측 파일이 없다: {missing}. 04_m1.py·05_m2.py --final·05b_m2_supplementary.py --final 을 먼저 실행한다.")
        other_preds = {b: (preds.read(b), preds.read_test_folds(b)) for b in others}
        _, _, chosen_o = evaluate.operating_points(other_preds, sets, costs, k_star, n_grid=N_GRID)
        chosen.update(chosen_o)
        model_preds.update(other_preds)
        pairs = list(PAIRS)

    tasks_b, keys = evaluate.bootstrap_tasks({m: model_preds[m] for m in MODELS}, sets, costs, chosen, k_star,
                                             {m: N_GRID for m in MODELS}, args.n_boot)
    cap_tasks, cap_keys = evaluate.cap_bootstrap_tasks(sets, costs, k_star, args.n_boot)
    ptasks, pkeys = paired_tasks(pairs, model_preds, sets, costs, chosen, args.n_boot)
    res = evaluate.parallel(evaluate.boot_task, tasks_b + cap_tasks + ptasks, args.cpu_jobs)
    boots = dict(zip(keys + cap_keys, res[:len(keys) + len(cap_keys)]))
    pair_boots = dict(zip(pkeys, res[len(keys) + len(cap_keys):]))
    evaluate.discrimination_ci(boots).write_csv(out / "discrimination_ci.csv", float_precision=FLOAT)
    ops_df = evaluate.operating_table(cap_main + [x for x in main_rows if x["model"] in MODELS], boots)
    ops_df.write_csv(out / "operating_points.csv", float_precision=FLOAT)
    evaluate.sensitivity_table(cap_sens + [x for x in sens_rows if x["model"] in MODELS]).write_csv(
        out / "operating_points_sensitivity.csv", float_precision=FLOAT)
    pair_df = pl.concat([b.with_columns(a=pl.lit(a), b=pl.lit(bm), N=pl.lit(n)) for (a, bm, n), b in pair_boots.items()]) \
        .select("a", "b", "N", "stat", "estimate", "lo", "hi", "n_boot")
    pair_df.write_csv(out / "paired.csv", float_precision=FLOAT)
    lap(f"부트스트랩 ({len(res)}개 × {args.n_boot}회)")

    sens2 = pl.DataFrame(b1_input_sensitivity(fz, r["test_folds"], frames, sets, costs, chosen, model_preds))
    sens2 = sens2.with_columns(diff=pl.col("sens_2") - pl.col("main_1"))
    sens2.write_csv(out / "b1_input_sensitivity.csv", float_precision=FLOAT)
    lap("② 민감도")

    compare = comparison_tables(out, disc, ops_df, args.smoke) if not args.smoke else None
    check(args, out, data_tr, data_te, written, written_folds, oof_same, rescore, spread, disc, ops_df, k_star, compare, pair_df,
          sens2, dev_summary, timings)


def check(args, out, data_tr, data_te, written, written_folds, oof_same, rescore, spread, disc, ops_df, k_star, compare, pair_df, sens2,
          dev_summary, timings):
    test_ops = ops_df.filter((pl.col("split") == "test") & pl.col("model").is_in(list(MODELS)))
    judged = evaluate.judged_and_saves_more(ops_df).to_dicts()
    keep = ["d_auroc", "d_within_pair", "combined.d_saved_steps", "combined.d_saved_tokens", "single.d_saved_steps",
            "single.d_saved_tokens", "combined.d_lost_success", "single.d_lost_success"]
    pair_sum = pair_df.filter(pl.col("stat").is_in(keep)).sort("a", "b", "stat", "N")
    nsum = evaluate.n_summary(ops_df) if compare is None else compare["n_summary"]
    s = {
        "data": {"train_trajectories": data_tr.n, "test_trajectories": data_te.n, "train_fingerprint": data_tr.fingerprint,
                 "test_fingerprint": data_te.fingerprint, "budget": m3_text.budget()},
        "cfg": dev_summary["cfg"], "revision": m3_text.REVISION,
        "fold_models": dev_summary["fold_models"],
        "preds": {m: {"rows": w["rows"], "written": w["written"]} for m, w in written.items()},
        "preds_test_folds": {m: {"rows": w["rows"], "max_abs_mean_diff": w["max_abs_mean_diff"]} for m, w in written_folds.items()},
        "rescore_fold0": rescore,
        "b1_test_fold_input_max_spread": spread,
        "k_star": k_star,
        "discrimination": disc.select("model", "split", "N", "auroc", "pr_auc", "within_pair", "within_issue_mean").to_dicts(),
        "n_summary_m3": nsum.filter(pl.col("model").is_in(list(MODELS))).to_dicts(),
        "judged_and_saves_more_than_k_star": judged,
        "loss_ci_width_summary": None if compare is None else compare["width_summary"].to_dicts(),
        "paired": pair_sum.to_dicts(),
        "b1_input_sensitivity_max_abs_diff": sens2.with_columns(a=pl.col("diff").abs()).group_by("kind", "metric").agg(pl.col("a").max())
        .sort("kind", "metric").to_dicts(),
        "bootstrap": {"n_boot": args.n_boot, "seed": io.SEED},
        "smoke": args.smoke,
    }
    crit = {
        "C1_preds_cover_running_trajectories": all(w["coverage"] for w in written.values()),
        "C2_no_seen_issue": True,  # collect·fuse에서 assert
        "C3_final_oof_equals_dev_oof": all(oof_same.values()) and rescore["equal"],
        "C4_k_star": args.smoke or k_star == 40,
        "C5_b1_test_input_differs_by_fold": args.smoke or spread > 0,
        "C6_fold_files_mean_matches": all(w["max_abs_mean_diff"] is not None and w["max_abs_mean_diff"] <= preds.FOLD_MEAN_TOL
                                          for w in written_folds.values()),
        "C7_operating_points": test_ops.height == len(MODELS) * len(N_GRID) * len(evaluate.POLICIES)
        and all(test_ops[f"{c}_fold_min"].null_count() == 0 for c in evaluate.POINT_COLUMNS)
        and test_ops["lost_success_hi"].null_count() == 0 and test_ops["meets_judge_ci_hi"].null_count() == 0,
        "C8_cap_rows_match_phase4": args.smoke or compare["cap_rows_match_phase4"],
        "C9_b1_sensitivity": sens2.height > 0 and sens2["sens_2"].null_count() == 0,
        "C10_paired": pair_df["estimate"].null_count() == 0,
    }
    s["criteria"] = crit
    s["criteria_note"] = "pytest 전체 통과와 재실행 시 results/·preds/ 변화 없음은 스크립트 밖에서 확인한다. 실행 시간은 표준 출력에만 남긴다"
    io.write_summary(out / "summary.json", s)
    with pl.Config(tbl_rows=80, tbl_cols=20, tbl_width_chars=240, float_precision=4):
        if compare is not None:
            print(compare["discrimination"].filter(pl.col("split") == "test").select("model", "N", "auroc", "auroc_lo", "auroc_hi",
                                                                                     "within_pair", "pr_auc").sort("N", "model"))
        print(nsum)
        print(pair_sum)
    print(json.dumps({"criteria": crit, "rescore_fold0": rescore, "timings": timings}, ensure_ascii=False, indent=2, default=str))
    failed = [k for k, v in crit.items() if not v]
    assert not failed, failed
    print("\n완료 기준 통과")


# ---- 점검 모드 ----

def smoke_sets(sets, n_train_issues=360, n_test_issues=120) -> dict:
    """train 이슈 일부로 가짜 train/test (실제 test 궤적은 쓰지 않는다). 가짜 train fold는 split.train_folds로 다시 나눈다."""
    tr = sets["train"]
    issues = np.array(sorted(tr["instance_id"].unique().to_list()))
    pick = np.random.default_rng(io.SEED).choice(issues, n_train_issues + n_test_issues, replace=False)
    te_iss, tr_iss = pick[:n_test_issues].tolist(), pick[n_test_issues:].tolist()
    train = tr.filter(pl.col("instance_id").is_in(tr_iss)).sort("traj_id")
    train = train.with_columns(fold=pl.Series(split.train_folds(train)).cast(tr["fold"].dtype))
    test = tr.filter(pl.col("instance_id").is_in(te_iss)).sort("traj_id").with_columns(fold=pl.lit(split.HELDOUT_FOLD, dtype=tr["fold"].dtype))
    return {"train": train, "test": test}


# ---- 실행 ----

def main():
    ap = argparse.ArgumentParser(description="Phase 7: M3 ModernBERT fine-tune")
    ap.add_argument("--final", action="store_true", help="저장된 fold 모델로 test를 한 번 채점하고 결과를 쓴다")
    ap.add_argument("--smoke", action="store_true", help="무작위 초기화 작은 모델과 train 이슈 일부로 전체 경로를 점검한다 (data/processed/m3_smoke/)")
    ap.add_argument("--device", default="auto", help="auto(GPU 있으면 cuda) | cuda | cpu. 개발·최종은 같은 장치여야 한다")
    ap.add_argument("--cpu-jobs", type=int, default=min(16, os.cpu_count() or 1), help="부트스트랩 병렬 수 (최종)")
    ap.add_argument("--n-boot", type=int, default=evaluate.N_BOOT)
    ap.add_argument("--force", action="store_true", help="저장된 fold 모델·채점·예측 파일을 무시하고 다시 만든다")
    args = ap.parse_args()
    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    paths = Paths(args.smoke)
    for p, script in ((io.TRAJECTORIES_PATH, "01_parse.py"), (io.STEPS_PATH, "01_parse.py"), (io.SPLITS_PATH, "02_split.py"),
                      (preds.fold_path(B1_MODEL), "03_baselines.py"), (io.RESULTS_DIR / "phase5" / "operating_points.csv", "04_m1.py"),
                      (io.RESULTS_DIR / "phase6" / "operating_points.csv", "05_m2.py --final")):
        if not Path(p).exists():
            sys.exit(f"{p}가 없다. 먼저 uv run --no-sync python code/{script} 를 실행한다.")
    frames = load_b1_frames(args.final, args.smoke)  # 실제 예측 파일에서 읽는다 (점검 모드의 경로 변경 전)
    if args.smoke:
        preds.PREDS_DIR = paths.preds  # 점검 모드는 실제 예측 파일을 건드리지 않는다
        args.n_boot = min(args.n_boot, 20)
    cfg = SMOKE_CFG if args.smoke else DEFAULT_CFG
    t0 = time.time()
    timings = {}

    def lap(name):
        timings[name] = round(time.time() - t0 - sum(timings.values()), 1)
        print(f"[{timings[name]:>7.1f}s] {name}", flush=True)

    print(f"모드: {'최종' if args.final else '개발'}{' (점검)' if args.smoke else ''}, 장치 {device}", flush=True)
    if device == "cpu" and not args.smoke:
        print("[주의] CPU 학습은 매우 느리다 (GPU 기준 약 4시간)", flush=True)
    sets = evaluate.load_eval_sets()
    if args.smoke:
        sets = smoke_sets(sets)
    costs = evaluate.load_costs(sets)
    enc = m3_text.hf_encoder()
    data_tr = m3_text.load(sets["train"], enc, paths.text_train)
    lap(f"train 토큰 배열 ({data_tr.n}궤적, 토큰 {len(data_tr.tokens):,})")
    if not args.final:
        dev(args, paths, sets, costs, data_tr, frames, device, cfg, lap)
        lap("개발 실행")
        print(json.dumps({"timings": timings}, ensure_ascii=False))
        return
    final(args, paths, sets, costs, data_tr, frames, enc, device, timings, lap)


if __name__ == "__main__":
    main()
