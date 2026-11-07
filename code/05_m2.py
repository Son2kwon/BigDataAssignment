"""Phase 6: M2 LSTM (모델 하나로 모든 N, B1 포함/제외) — 개발(train OOF만)과 최종(--final, test 한 번 채점)을 나눈다.

실행:
  uv run --no-sync python code/05_m2.py [--jobs 4] [--device auto] [--force]             # 개발: train OOF만, test를 보지 않는다
  uv run --no-sync python code/05_m2.py --final [--jobs 4] [--n-boot 1000] [--force]     # 최종: 고른 설정으로 test 채점·평가
  uv run --no-sync python code/05_m2.py --smoke [--final]                                # 점검: train 이슈 일부로 전체 경로 (실제 test 안 씀)

입력: data/processed/trajectories.parquet, steps.parquet (01), splits.parquet (02), preds/b1_n*.parquet (03),
      results/phase4·phase5/ (비교 표), 최종은 preds/m1_*.parquet (04_m1.py, M2 vs M1 짝지은 비교)
출력:
- 개발: data/processed/m2/ (시퀀스 배열, fold 모델, OOF), results/phase6/m2_grid.csv, m2_train_log.csv, ablation_oof.csv,
  train_operating_points.csv, good_n.csv, summary_dev.json (모두 train OOF 값)
- 최종: data/processed/preds/m2_*.parquet + .test_folds.parquet, results/phase6/ 판별·운영점·비교·짝지은 비교·N 요약·summary.json

방법 (docs/phase6.md 6.0 사전 등록):
- 모델 2종 (P6-D1 둘 다): m2_lstm, m2_lstm_b1 (B1 점수 b1_n을 머리에 넣는다. train = OOF, test fold k = B1 fold k, P4-D7 ①)
- 학습: fold k 모델 = fold k를 뺀 train으로 afp.lstm.fit (조기 종료는 그 안의 이슈 단위 검증 조각). OOF·test 점수는
  oof.oof_predict로 모은다 (fit 콜백이 저장된 fold 모델을 읽고 학습 이슈 집합을 대조한다)
- 그리드: hidden HIDDEN_START × dropout DROPOUTS. 선택 = train OOF AUROC의 N_GRID 평균 최대, GRID_TIE 안이면 앞쪽(작은 hidden,
  큰 dropout). 고른 hidden이 그리드 끝이면 HIDDEN_LADDER의 다음 값을 더해 다시 고른다 (최대 MAX_EXTENSIONS번, train만 봄)
- 절제 실험 (train OOF만, test 안 봄): m2_lstm의 고른 설정에서 누적 집계 빼기(m2x_nocum), 순환 빼기(m2x_norec)
- 평가: afp/evaluate.py 공용 함수 (03·04와 같은 코드). 운영점은 train OOF에서만 고르고 test는 fold 모델별 채점 평균 (P4-D6)
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from afp import evaluate, io, lstm, metrics, oof, preds, sequences, split

MODEL_SPECS = {"m2_lstm": False, "m2_lstm_b1": True}  # model_id → B1 점수 포함
B1_MODEL = "b1_n"  # P4-D3 주 결과
HIDDEN_START = (16, 32, 64)
HIDDEN_LADDER = (8, 16, 32, 64, 128, 256)
DROPOUTS = (0.3, 0.1)  # 앞쪽 = 더 규제
MAX_EXTENSIONS = 2
GRID_TIE = 0.002  # M1·B1과 같은 값
GOOD_N_TIE = 0.005  # M1 좋은 N 규칙과 같은 값 (P5-D5 재검토 중이라 두 규칙을 같이 낸다)
ABLATION_BASE = "m2_lstm"
ABLATIONS = {"m2x_nocum": {"use_cum": False}, "m2x_norec": {"recurrent": False}}
M1_PAIR = {"m2_lstm": "m1_xgb", "m2_lstm_b1": "m1_xgb_b1"}  # 짝지은 비교 대상 (사전 등록: 같은 B1 조건의 가장 강한 M1)
M1_MODELS = ("m1_lr", "m1_lr_b1", "m1_xgb", "m1_xgb_b1")
P4_MODELS = ("b0_err", "b0_editfail", "b1_static", "b1_n")
CACHE_VERSION = 1
FLOAT = 6
DEFAULT_CFG = lstm.M2Config()


class Paths:
    def __init__(self, smoke: bool):
        base = io.PROCESSED_DIR / ("m2_smoke" if smoke else "m2")
        self.cache = base
        self.models = base / "models"
        self.seq_train = base / "seq_train"
        self.out = base / "results" if smoke else io.RESULTS_DIR / "phase6"
        self.preds = base / "preds" if smoke else preds.PREDS_DIR

    def model(self, model_id: str, key: str, k: int) -> Path:
        return self.models / model_id / key / f"fold{k}.pt"

    def oof(self, model_id: str) -> Path:
        return self.cache / f"oof_{model_id}.parquet"


# ---- 설정 ----

def make_cfg(hidden, dropout, use_b1, base: lstm.M2Config = DEFAULT_CFG, **changes) -> lstm.M2Config:
    return lstm.copy_config(base, hidden=hidden, dropout=dropout, use_b1=use_b1, **changes)


def cfg_key(cfg: lstm.M2Config) -> str:
    key = f"h{cfg.hidden}_d{round(cfg.dropout * 100):02d}"
    if not cfg.use_cum:
        key += "_nocum"
    if not cfg.recurrent:
        key += "_norec"
    return key


def grid_order(cfg: lstm.M2Config) -> tuple:
    """그리드 순서 = 단순한 것부터: 작은 hidden, 같은 hidden이면 큰 dropout."""
    return (cfg.hidden, -cfg.dropout)


def meta_for(model_id, cfg, k, seeds, data, device) -> dict:
    return {"version": CACHE_VERSION, "model_id": model_id, "cfg": cfg.as_dict(), "fold": int(k), "seeds": seeds,
            "data_fingerprint": data.fingerprint, "fit_issue_hash": lstm.issue_hash(data.instance_id[data.fold != k]),
            "device": device}


def _canon(x) -> str:
    return json.dumps(x, sort_keys=True, default=str)


# ---- 학습 작업 (작업 프로세스) ----

def train_task(path: str, meta: dict, seq_dir: str, b1_path: str | None, device: str, force: bool) -> dict:
    """fold 모델 하나. 같은 meta의 파일이 있으면 다시 학습하지 않는다. 학습 기록을 돌려준다."""
    p = Path(path)
    if p.exists() and not force:
        model, old = lstm.load(p)
        if _canon(old) == _canon(meta):
            return {"path": path, "history": model.history, "trained": False, "seconds": 0.0}
    t0 = time.time()
    data = sequences.load_arrays(Path(seq_dir))
    if data.fingerprint != meta["data_fingerprint"]:
        raise AssertionError("시퀀스 캐시가 다르다")
    b1 = np.load(b1_path, mmap_mode="r") if b1_path else None
    rows = np.flatnonzero(np.asarray(data.fold) != meta["fold"])
    model = lstm.fit(lstm.M2Config(**meta["cfg"]), data, rows, meta["seeds"], device, b1)
    if model.fit_issue_hash != meta["fit_issue_hash"]:
        raise AssertionError("fit 이슈 집합이 meta와 다르다")
    lstm.save(model, p, meta)
    return {"path": path, "history": model.history, "trained": True, "seconds": round(time.time() - t0, 1)}


def run_training(specs: list, paths: Paths, data, b1_file, device, jobs, force) -> list[dict]:
    """specs = [(model_id, cfg)] → 모든 fold 모델 학습 (병렬). 학습 기록 행."""
    tasks, keys = [], []
    for model_id, cfg in specs:
        for k in range(split.N_FOLDS):
            seeds = lstm.seeds_for(k, cfg.n_seeds)
            meta = meta_for(model_id, cfg, k, seeds, data, device)
            tasks.append((str(paths.model(model_id, cfg_key(cfg), k)), meta, str(paths.seq_train),
                          str(b1_file) if cfg.use_b1 else None, device, force))
            keys.append((model_id, cfg, k))
    out = evaluate.parallel(train_task, tasks, jobs)
    rows = []
    for (model_id, cfg, k), r in zip(keys, out):
        for h in r["history"]:
            rows.append({"model": model_id, "config": cfg_key(cfg), "fold": k, "seed": h["seed"], "best_epoch": h["best_epoch"],
                         "n_epochs": h["n_epochs"], "hit_max_epochs": h["best_epoch"] == cfg.max_epochs, "best_val": h["best_val"],
                         "val_history": json.dumps([round(v, 6) for v in h["val_history"]])})
    trained = sum(r["trained"] for r in out)
    secs = [r["seconds"] for r in out if r["trained"]]
    print(f"  fold 모델 {len(out)}개 (새로 학습 {trained}개{', 평균 %.0f초' % np.mean(secs) if secs else ''})", flush=True)
    return rows


# ---- OOF·test 점수 모으기 (oof.oof_predict) ----

def long_rows(data: sequences.SequenceData, n_grid=evaluate.N_GRID) -> pl.DataFrame:
    """(궤적, N) 행: 시점 N 진행 중 궤적마다 하나. oof.oof_predict의 행 단위."""
    parts = []
    for n in n_grid:
        r = data.running(n)
        parts.append(pl.DataFrame({"instance_id": data.instance_id[r].astype(str), "fold": np.asarray(data.fold)[r],
                                   "_i": r, "N": np.full(len(r), n, dtype=np.int64)}))
    return pl.concat(parts)


def collect(model_id, cfg, paths, data_tr, b1_tr, device, data_te=None, b1_te_folds=None) -> dict:
    """저장된 fold 모델로 train OOF와 (data_te가 있으면) test fold 모델별 점수를 낸다.

    fit 콜백은 학습하지 않고 fold k 모델 파일을 읽는다: 받은 행에 없는 fold가 k이고, 파일의 학습 이슈 해시가
    "fold k를 뺀 train 이슈"와 같아야 한다. 받은 행의 이슈는 그 집합 안에 있어야 한다 (oof.oof_predict가 seen 검사).
    test fold k 모델은 B1 fold k 점수로 채점한다 (P4-D7 ①).
    """
    train_rows = long_rows(data_tr)
    test_rows = None if data_te is None else long_rows(data_te).with_columns(_src=pl.lit(-1))
    allowed = {k: set(data_tr.instance_id[np.asarray(data_tr.fold) != k]) for k in range(split.N_FOLDS)}

    def fit(rows):
        missing = set(range(split.N_FOLDS)) - set(rows["fold"].unique().to_list())
        if len(missing) != 1:
            raise AssertionError(f"fit 행에서 빠진 fold가 하나가 아니다: {missing}")
        k = missing.pop()
        model, meta = lstm.load(paths.model(model_id, cfg_key(cfg), k))
        if meta["fit_issue_hash"] != lstm.issue_hash(allowed[k]) or meta["data_fingerprint"] != data_tr.fingerprint:
            raise AssertionError(f"{model_id} fold {k}: 저장된 모델의 학습 이슈·데이터가 다르다")
        if not set(rows["instance_id"].unique().to_list()) <= allowed[k]:
            raise AssertionError("fit 행에 학습 이슈 밖 이슈가 있다")
        return (k, model)

    def predict(model, rows):
        k, m = model
        src = int(rows["_src"][0]) if "_src" in rows.columns else None
        data = data_tr if src is None else data_te
        b1 = (b1_tr if src is None else (b1_te_folds[src] if b1_te_folds is not None else None)) if cfg.use_b1 else None
        i = rows["_i"].to_numpy()
        uniq = np.unique(i)
        P = lstm.predict(m, data, uniq, device, b1)
        out = P[np.searchsorted(uniq, i), rows["N"].to_numpy() - 1]
        if np.isnan(out).any():
            raise AssertionError("진행 중 위치에 점수가 없다")
        return out

    res = oof.oof_predict(train_rows, fit, predict, test=test_rows,
                          test_for_fold=None if test_rows is None else (lambda k, t: t.with_columns(_src=pl.lit(k))))
    tr = res.train.select(traj_id=pl.Series(data_tr.traj_id[res.train["_i"].to_numpy()].astype(str)),
                          N=pl.col("N").cast(pl.Int16), p_fail="p_fail").sort("N", "traj_id")
    out = {"train": tr, "seen_issue_scores": res.seen_issue_scores}
    if test_rows is not None:
        ids = data_te.traj_id[res.test["_i"].to_numpy()].astype(str)
        n = res.test["N"].cast(pl.Int16)
        out["test_mean"] = pl.DataFrame({"traj_id": ids, "N": n, "p_fail": res.test["p_fail"]})
        out["test_folds"] = pl.concat([pl.DataFrame({"traj_id": ids, "N": n, "fold": np.full(len(ids), k, dtype=np.int8),
                                                     "p_fail": res.test_fold_scores[k]}) for k in range(split.N_FOLDS)])
    return out


def oof_scores(model_id, train_preds, train_set) -> dict:
    d = evaluate.discrimination_table(model_id, train_preds, {"train": train_set}, evaluate.N_GRID)
    out = {f"oof_auroc_N{r['N']}": r["auroc"] for r in d.iter_rows(named=True)}
    out.update({f"oof_within_pair_N{r['N']}": r["within_pair"] for r in d.iter_rows(named=True)})
    out["oof_auroc_mean"] = float(d["auroc"].mean())
    out["oof_within_pair_mean"] = float(d["within_pair"].mean())
    return out


# ---- 개발: 그리드 + 절제 (train OOF만) ----

def select(rows: list[dict]) -> dict:
    """그리드 결과 → 고른 행. N_GRID 평균 OOF AUROC 최대에서 GRID_TIE 안이면 그리드 앞쪽."""
    best = max(r["oof_auroc_mean"] for r in rows)
    cand = sorted((r for r in rows if r["oof_auroc_mean"] >= best - GRID_TIE), key=lambda r: grid_order(r["cfg"]))
    return cand[0]


def extension(cfgs: list[lstm.M2Config], chosen: lstm.M2Config) -> list[int]:
    """고른 hidden이 지금 그리드의 끝이면 사다리에서 바깥 값 하나 (없으면 빈 목록)."""
    hs = sorted({c.hidden for c in cfgs})
    i = HIDDEN_LADDER.index(chosen.hidden)
    if chosen.hidden == hs[0] and i > 0:
        return [HIDDEN_LADDER[i - 1]]
    if chosen.hidden == hs[-1] and i < len(HIDDEN_LADDER) - 1:
        return [HIDDEN_LADDER[i + 1]]
    return []


def train_ops(train_set, p_train, costs_train) -> list[dict]:
    rows = []
    for n in evaluate.N_GRID:
        for policy in evaluate.POLICIES:
            op = metrics.select_from_curves(evaluate.candidate_curves(train_set, p_train, n, policy, costs_train), n)
            rows.append({"N": n, "policy": policy, **{c: op[c] for c in ("tau", "cap", "stop_rate", "lost_success", "saved_steps", "saved_tokens")}})
    return rows


def good_n_rows(model_id, oof_row, ops_rows) -> list[dict]:
    """좋은 N 두 규칙 (train OOF만): auroc_first = M1 규칙 (P5-D5 현재), saved_first = train 결합 절약 스텝 최대 (P5-D5 재검토안)."""
    j = pl.DataFrame([{"N": n, "train_oof_auroc": oof_row[f"oof_auroc_N{n}"],
                       "train_saved_steps_combined": next(r["saved_steps"] for r in ops_rows if r["N"] == n and r["policy"] == "combined")}
                      for n in evaluate.N_GRID])
    best = j["train_oof_auroc"].max()
    a = j.filter(pl.col("train_oof_auroc") >= best - GOOD_N_TIE).sort(["train_saved_steps_combined", "N"], descending=[True, False])["N"][0]
    s = j.sort(["train_saved_steps_combined", "N"], descending=[True, False])["N"][0]
    return [{"model": model_id, **r, "in_tie_band": r["train_oof_auroc"] >= best - GOOD_N_TIE,
             "selected_auroc_first": r["N"] == a, "selected_saved_first": r["N"] == s} for r in j.iter_rows(named=True)]


def dev(args, paths, sets, costs, data_tr, b1_tr, b1_file, device, smoke_cfg):
    t = sets["train"]
    log_rows, grid_rows, selected, path_log = [], [], {}, {}
    cfgs = {m: [make_cfg(h, d, b1, **smoke_cfg) for h in HIDDEN_START for d in DROPOUTS] for m, b1 in MODEL_SPECS.items()}
    if args.smoke:
        cfgs = {m: [make_cfg(h, 0.3, b1, **smoke_cfg) for h in (16, 32)] for m, b1 in MODEL_SPECS.items()}
    done = {m: {} for m in MODEL_SPECS}
    pending = {m: list(c) for m, c in cfgs.items()}
    for round_i in range(MAX_EXTENSIONS + 1):
        specs = [(m, c) for m, cs in pending.items() for c in cs]
        if not specs:
            break
        print(f"그리드 {round_i}회차: {[(m, cfg_key(c)) for m, c in specs]}", flush=True)
        log_rows += run_training(specs, paths, data_tr, b1_file, device, args.jobs, args.force)
        for m, c in specs:
            r = collect(m, c, paths, data_tr, b1_tr, device)
            assert r["seen_issue_scores"] == 0
            done[m][cfg_key(c)] = {"cfg": c, "round": round_i, "train": r["train"], **oof_scores(m, r["train"], t)}
        pending = {}
        for m in MODEL_SPECS:
            chosen = select(list(done[m].values()))
            path_log.setdefault(m, []).append(cfg_key(chosen["cfg"]))
            ext = extension([v["cfg"] for v in done[m].values()], chosen["cfg"]) if round_i < MAX_EXTENSIONS else []
            if ext:
                pending[m] = [make_cfg(h, d, MODEL_SPECS[m], **smoke_cfg) for h in ext for d in DROPOUTS]
                print(f"  [경계] {m}: 고른 hidden={chosen['cfg'].hidden}이 그리드 끝 → hidden {ext} 추가 (train OOF만 봄)", flush=True)
    for m in MODEL_SPECS:
        chosen = select(list(done[m].values()))
        selected[m] = chosen
        hs = sorted({v["cfg"].hidden for v in done[m].values()})
        chosen["at_boundary"] = chosen["cfg"].hidden in (hs[0], hs[-1])
        for key, v in sorted(done[m].items(), key=lambda kv: grid_order(kv[1]["cfg"])):
            grid_rows.append({"model": m, "config": key, "hidden": v["cfg"].hidden, "dropout": v["cfg"].dropout, "round": v["round"],
                              **{c: v[c] for c in v if c.startswith("oof_")}, "selected": key == cfg_key(chosen["cfg"])})
        v = selected[m]
        print(f"  고름 {m}: {cfg_key(v['cfg'])} (train OOF AUROC 평균 {v['oof_auroc_mean']:.4f}), 경로 {path_log[m]}", flush=True)

    # 절제 실험 (m2_lstm의 고른 설정 기준, train OOF만)
    base = selected[ABLATION_BASE]["cfg"]
    abl_specs = [(a, lstm.copy_config(base, **ch)) for a, ch in ABLATIONS.items()]
    log_rows += run_training(abl_specs, paths, data_tr, b1_file, device, args.jobs, args.force)
    abl = {}
    for a, c in abl_specs:
        r = collect(a, c, paths, data_tr, b1_tr, device)
        abl[a] = {"cfg": c, **oof_scores(a, r["train"], t)}

    # train 운영점·좋은 N (train OOF만)
    ops_rows, good_rows = [], []
    for m, v in selected.items():
        rows = train_ops(t, v["train"], costs["train"])
        ops_rows += [{"model": m, **r} for r in rows]
        good_rows += good_n_rows(m, v, rows)
        v["train"].write_parquet(paths.oof(m))
    cap_train = metrics.fixed_cap_curve(t, costs["train"], range(1, 81))
    k_star = metrics.select_cap(cap_train, metrics.SELECT_MAX_LOSS)["K"]
    kstar_train = evaluate.cap_point(t, k_star, costs["train"])

    paths.out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(grid_rows).write_csv(paths.out / "m2_grid.csv", float_precision=FLOAT)
    pl.DataFrame(log_rows).sort("model", "config", "fold", "seed").write_csv(paths.out / "m2_train_log.csv", float_precision=FLOAT)
    # 절제 표: M2 고른 설정 + 절제 2종 + M1 train OOF (results/phase5에서 train_oof 행만 읽는다)
    abl_rows = []
    for name, v in [(ABLATION_BASE, selected[ABLATION_BASE]), ("m2_lstm_b1", selected["m2_lstm_b1"])] + list(abl.items()):
        abl_rows += [{"model": name, "config": cfg_key(v["cfg"]), "N": n, "train_oof_auroc": v[f"oof_auroc_N{n}"],
                      "train_oof_within_pair": v[f"oof_within_pair_N{n}"]} for n in evaluate.N_GRID]
    m1 = m1_train_discrimination()
    abl_df = pl.concat([pl.DataFrame(abl_rows), m1]) if m1 is not None else pl.DataFrame(abl_rows)
    abl_df.write_csv(paths.out / "ablation_oof.csv", float_precision=FLOAT)
    ops_df = pl.DataFrame(ops_rows).with_columns(d_saved_steps=pl.col("saved_steps") - kstar_train["saved_steps"],
                                                 d_saved_tokens=pl.col("saved_tokens") - kstar_train["saved_tokens"])
    m1o = m1_train_operating_points()
    if m1o is not None:
        ops_df = pl.concat([ops_df, m1o.select(ops_df.columns)], how="vertical_relaxed")
    ops_df.write_csv(paths.out / "train_operating_points.csv", float_precision=FLOAT)
    pl.DataFrame(good_rows).write_csv(paths.out / "good_n.csv", float_precision=FLOAT)
    hit_max = pl.DataFrame(log_rows).filter(pl.col("hit_max_epochs")).height
    summary = {
        "mode": "dev (train OOF only, test not scored)",
        "device": device, "jobs": args.jobs,
        "fixed": {k: v for k, v in DEFAULT_CFG.as_dict().items() if k not in ("hidden", "dropout", "use_b1")} | smoke_cfg,
        "grid": {"hidden_start": list(HIDDEN_START), "hidden_ladder": list(HIDDEN_LADDER), "dropouts": list(DROPOUTS),
                 "max_extensions": MAX_EXTENSIONS, "tie": GRID_TIE},
        "selection_path": path_log,
        "selected": {m: {"config": cfg_key(v["cfg"]), "cfg": v["cfg"].as_dict(), "oof_auroc_mean": v["oof_auroc_mean"],
                         "at_boundary": v["at_boundary"]} for m, v in selected.items()},
        "ablation": {a: {"config": cfg_key(v["cfg"]), "oof_auroc_mean": v["oof_auroc_mean"],
                         "d_vs_base": v["oof_auroc_mean"] - selected[ABLATION_BASE]["oof_auroc_mean"]} for a, v in abl.items()},
        "k_star_train": {"K": k_star, **kstar_train},
        "fold_models_hit_max_epochs": hit_max,
        "data": {"train_sequences": data_tr.n, "fingerprint": data_tr.fingerprint},
    }
    io.write_summary(paths.out / "summary_dev.json", summary)

    with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=220, float_precision=4):
        print(pl.DataFrame(grid_rows).select("model", "config", "round", "oof_auroc_mean", "oof_within_pair_mean",
                                             *[f"oof_auroc_N{n}" for n in evaluate.N_GRID], "selected"))
        print(abl_df.pivot(on="N", index="model", values="train_oof_auroc"))
        print(ops_df.filter(pl.col("N").is_in([10, 15, 20])).sort("N", "policy", "model"))
        print(pl.DataFrame(good_rows).filter(pl.col("selected_auroc_first") | pl.col("selected_saved_first")))
    flags = [m for m, v in selected.items() if v["at_boundary"]]
    print(json.dumps({"selected": summary["selected"], "ablation": summary["ablation"], "fold_models_hit_max_epochs": hit_max},
                     ensure_ascii=False, indent=2, default=str))
    if flags:
        print(f"\n[주의] 확장 뒤에도 그리드 끝에서 고른 모델: {flags} (사다리 끝이거나 확장 횟수 초과). 보고 후 --final")
    if hit_max:
        print(f"\n[주의] 최고 epoch가 max_epochs인 망 {hit_max}개 (학습이 덜 됐을 수 있음). 보고 후 --final")
    if not flags and not hit_max:
        print("\n개발 실행 완료: 그리드 경계·epoch 상한 문제 없음. --final 을 실행해도 된다")


def m1_train_discrimination() -> pl.DataFrame | None:
    p = io.RESULTS_DIR / "phase5" / "discrimination.csv"
    if not p.exists():
        return None
    d = pl.read_csv(p).filter((pl.col("split") == "train_oof") & pl.col("model").is_in(list(M1_MODELS)))
    return d.select("model", pl.lit("phase5").alias("config"), pl.col("N").cast(pl.Int64),
                    pl.col("auroc").alias("train_oof_auroc"), pl.col("within_pair").alias("train_oof_within_pair"))


def m1_train_operating_points() -> pl.DataFrame | None:
    p = io.RESULTS_DIR / "phase5" / "operating_points.csv"
    if not p.exists():
        return None
    o = pl.read_csv(p, schema_overrides={"N": pl.Int64, "cap": pl.Int64})
    return o.filter((pl.col("split") == "train") & pl.col("model").is_in(list(M1_MODELS))).select(
        "model", "N", "policy", "tau", "cap", "stop_rate", "lost_success", "saved_steps", "saved_tokens", "d_saved_steps", "d_saved_tokens")


# ---- 최종 ----

def load_b1_frames(final: bool, smoke: bool) -> dict:
    """B1 점수 (traj_id, N, p_fail). train = OOF. 최종이면 test fold k 점수(P4-D7 ①)와 ② 민감도용 j≠k 점수도 읽는다.
    점검 모드의 가짜 test는 train 궤적이라 OOF 점수를 모든 fold에 쓴다 (② 없음)."""
    tr = preds.load_feature(B1_MODEL, "train")
    out = {"train": tr, "test": None, "other": None}
    if final and smoke:
        out["test"] = [tr] * split.N_FOLDS
        out["other"] = {k: pl.concat([tr.with_columns(src_fold=pl.lit(j, dtype=pl.Int8)) for j in range(split.N_FOLDS) if j != k])
                        for k in range(split.N_FOLDS)}
    elif final:
        out["test"] = [preds.load_feature(B1_MODEL, "test", fold=k) for k in range(split.N_FOLDS)]
        out["other"] = {k: preds.load_feature_other_folds(B1_MODEL, k) for k in range(split.N_FOLDS)}
    return out


def b1_other_folds(data_te, frames, k) -> dict[int, np.ndarray]:
    """② 민감도: fold k 모델에 넣을 B1 fold j(j≠k) 점수."""
    o = frames["other"][k]
    return {int(j): sequences.b1_matrix(data_te, o.filter(pl.col("src_fold") == j).select("traj_id", "N", "p_fail"))
            for j in sorted(o["src_fold"].unique().to_list())}


def b1_input_sensitivity(m, cfg, paths, data_te, frames, sets, costs, chosen, model_preds, device) -> list[dict]:
    """P4-D7 ② vs ①: M2 fold k를 B1 fold j≠k 4개로 각각 채점. 판별 = 20개 점수 평균, 운영점 = 20개 결과 평균."""
    test = sets["test"]
    parts = []
    for k in range(split.N_FOLDS):
        model, _ = lstm.load(paths.model(m, cfg_key(cfg), k))
        for j, b1 in b1_other_folds(data_te, frames, k).items():
            P = lstm.predict(model, data_te, np.arange(data_te.n), device, b1)
            parts.append(sequences.preds_frame(data_te, P).with_columns(fold=pl.lit(10 * k + j, dtype=pl.Int16)))
    long = pl.concat(parts)
    mean2 = long.group_by("traj_id", "N").agg(pl.col("p_fail").mean()).sort("N", "traj_id")
    rows = []
    for n in evaluate.N_GRID:
        p1 = evaluate.test_preds(*model_preds[m], test, n)
        p2 = evaluate.attach_fold_scores(mean2.filter(pl.col("N") == n), long.filter(pl.col("N") == n))
        d1 = evaluate.discrimination_row(test, p1.select("traj_id", "N", "p_fail"), n)
        d2 = evaluate.discrimination_row(test, mean2.filter(pl.col("N") == n), n)
        rows += [{"model": m, "N": n, "kind": "discrimination", "policy": None, "metric": c, "main_1": d1[c], "sens_2": d2[c]}
                 for c in evaluate.PAIR_DISC]
        for policy in evaluate.POLICIES:
            tau, cap = chosen[(m, n, policy)]
            a = evaluate.test_point(test, p1, n, tau, cap, costs["test"], spread=False)
            b = evaluate.test_point(test, p2, n, tau, cap, costs["test"], spread=False)
            rows += [{"model": m, "N": n, "kind": "operating_point", "policy": policy, "metric": c, "main_1": a[c], "sens_2": b[c]}
                     for c in evaluate.POINT_COLUMNS]
    return rows


def comparison_tables(out_dir, disc, ops_df, boots) -> dict:
    """B0·B1(phase4) + M1(phase5) + M2 → 판별 비교, 운영점 비교, N 요약, 손실 구간 폭. B0·B1·M1 값은 결과 파일 그대로."""
    p4, p5 = io.RESULTS_DIR / "phase4", io.RESULTS_DIR / "phase5"
    so = {"N": pl.Int64}
    d_parts = [pl.read_csv(p4 / "discrimination.csv", schema_overrides=so).filter(pl.col("model").is_in(list(P4_MODELS))),
               pl.read_csv(p5 / "discrimination.csv", schema_overrides=so).filter(pl.col("model").is_in(list(M1_MODELS)))]
    dall = pl.concat([d.filter(pl.col("N").is_in(list(evaluate.N_GRID))).select(disc.columns).cast(disc.schema) for d in d_parts] + [disc])
    ci_parts = [pl.read_csv(p4 / "discrimination_ci.csv", schema_overrides=so).filter(pl.col("model").is_in(list(P4_MODELS))),
                pl.read_csv(p5 / "discrimination_ci.csv", schema_overrides=so).filter(pl.col("model").is_in(list(M1_MODELS)))]
    if boots:
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

    o4 = pl.read_csv(p4 / "operating_points.csv", schema=ops_df.schema)
    o5 = pl.read_csv(p5 / "operating_points.csv", schema=ops_df.schema).filter(pl.col("model").is_in(list(M1_MODELS)))
    cap4, cap6 = o4.filter(pl.col("model") == "b0_cap"), ops_df.filter(pl.col("model") == "b0_cap")
    num = [c for c, t in ops_df.schema.items() if t == pl.Float64]
    cap_same = cap4.height == cap6.height and all(
        np.allclose(cap4[c].fill_null(np.nan).to_numpy(), cap6[c].fill_null(np.nan).to_numpy(), atol=1e-6, equal_nan=True) for c in num)
    ocomp = pl.concat([o4, o5, ops_df.filter(pl.col("model") != "b0_cap")])
    ocomp.write_csv(out_dir / "operating_points_compare.csv", float_precision=FLOAT)
    nsum = evaluate.n_summary(ocomp.filter(pl.col("model").is_in(["b0_cap", *M1_MODELS, *MODEL_SPECS])))
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
    return {"discrimination": dcomp, "operating": ocomp, "n_summary": nsum, "width_summary": width_sum, "cap_rows_match_phase4": bool(cap_same)}


def paired_tasks(pairs, model_preds: dict, sets, costs, chosen, n_boot) -> tuple[list, list]:
    """짝지은 부트스트랩 작업 (짝·N마다). pairs = {a: b} 또는 [(a, b)]. 같은 시드라 다른 표와 같은 재표본이다."""
    test = sets["test"]
    tasks, keys = [], []
    for a, b in (pairs.items() if isinstance(pairs, dict) else pairs):
        for n in evaluate.N_GRID:
            pa = {pol: chosen[(a, n, pol)] for pol in evaluate.POLICIES}
            pb = {pol: chosen[(b, n, pol)] for pol in evaluate.POLICIES}
            c = evaluate.compressed_costs(costs["test"], n, [k for _, k in (*pa.values(), *pb.values())], None)
            p = evaluate.pair_preds(evaluate.test_preds(*model_preds[a], test, n), evaluate.test_preds(*model_preds[b], test, n))
            tasks.append((test, p, evaluate.paired_stat(n, pa, pb, c), n_boot))
            keys.append((a, b, n))
    return tasks, keys


def final(args, paths, sets, costs, data_tr, b1_tr, frames, device, timings, lap):
    grid_path = paths.out / "m2_grid.csv"
    if not grid_path.exists():
        sys.exit(f"{grid_path}가 없다. 먼저 개발 실행(--final 없이)을 한다.")
    grid = pl.read_csv(grid_path).filter(pl.col("selected"))
    dev_summary = json.loads((paths.out / "summary_dev.json").read_text())
    if dev_summary["data"]["fingerprint"] != data_tr.fingerprint:
        sys.exit("train 시퀀스가 개발 실행 때와 다르다. 개발 실행을 다시 한다 (--force).")
    if dev_summary["device"] != device:
        sys.exit(f"개발 실행은 {dev_summary['device']}로 학습했다. 같은 --device로 실행한다.")
    data_te = sequences.load(sets["test"])
    b1_te = [sequences.b1_matrix(data_te, f) for f in frames["test"]]
    stacked = np.stack(b1_te)[:, data_te.valid()]  # fold × 진행 중 위치
    spread = float((stacked.max(0) - stacked.min(0)).max())
    lap(f"test 시퀀스 ({data_te.n}궤적)")

    written, written_folds, model_preds, cfgs, oof_same = {}, {}, {}, {}, {}
    for m, use_b1 in MODEL_SPECS.items():
        cfg = lstm.M2Config(**dev_summary["selected"][m]["cfg"])
        assert cfg_key(cfg) == grid.filter(pl.col("model") == m)["config"][0]
        cfgs[m] = cfg
        r = collect(m, cfg, paths, data_tr, b1_tr, device, data_te, b1_te)
        assert r["seen_issue_scores"] == 0
        oof_same[m] = r["train"].equals(pl.read_parquet(paths.oof(m)))
        config = {"learner": "lstm", "cfg": cfg.as_dict(), "selected_from": "results/phase6/m2_grid.csv (train OOF AUROC N 평균)",
                  "step_features": sequences.STEP_FEATURES, "cum_features": sequences.CUM_FEATURES, "static": sequences.STATIC_FEATURES,
                  "b1_input": f"train={B1_MODEL} OOF, test fold k={B1_MODEL} fold k (P4-D7 ①)" if use_b1 else None, "device": device}
        p_all = pl.concat([r["train"], r["test_mean"]])
        written[m] = preds.write(m, p_all, sets, evaluate.N_GRID, config, "5fold-oof/test-fold-mean", force=args.force)
        written_folds[m] = preds.write_test_folds(m, r["test_folds"], sets["test"], evaluate.N_GRID, split.N_FOLDS, force=args.force)
        model_preds[m] = (preds.read(m), preds.read_test_folds(m))
    lap("test 채점·예측 저장")

    out = paths.out
    disc = pl.concat([evaluate.discrimination_table(m, model_preds[m][0], sets, evaluate.N_GRID) for m in MODEL_SPECS])
    disc.write_csv(out / "discrimination.csv", float_precision=FLOAT)
    cap_train = metrics.fixed_cap_curve(sets["train"], costs["train"], range(1, 81))
    k_star = metrics.select_cap(cap_train, metrics.SELECT_MAX_LOSS)["K"]
    main_rows, sens_rows, chosen = evaluate.operating_points(model_preds, sets, costs, k_star)
    cap_main, cap_sens = evaluate.cap_rows(sets, costs, k_star, cap_train)
    curve_dir = out / "cost_curves"
    if curve_dir.exists():
        shutil.rmtree(curve_dir)
    curve_dir.mkdir(parents=True)
    for m in MODEL_SPECS:
        for (name, n), c in evaluate.model_curves(model_preds[m][0], sets, costs, evaluate.N_GRID, model_preds[m][1]).items():
            evaluate.thin_curve(c).write_csv(curve_dir / f"{m}_{name}_N{n}.csv", float_precision=FLOAT)
    lap("판별 지표·운영점 선택·적용")

    # M1 짝 (같은 평가 함수로 M1 운영점을 다시 고른다. M1 예측 파일이 이 머신의 것이므로)
    pairs = {}
    if args.smoke:
        pairs = {"m2_lstm_b1": "m2_lstm"}  # 점검: 경로만 확인
    else:
        missing = [b for b in M1_PAIR.values() if not preds.path(b).exists()]
        if missing:
            sys.exit(f"M1 예측 파일이 없다: {missing}. 먼저 uv run --no-sync python code/04_m1.py 를 실행한다.")
        m1_preds = {b: (preds.read(b), preds.read_test_folds(b)) for b in M1_PAIR.values()}
        _, _, chosen_m1 = evaluate.operating_points(m1_preds, sets, costs, k_star)
        chosen.update(chosen_m1)
        model_preds.update(m1_preds)
        pairs = dict(M1_PAIR)

    boots, pair_boots = {}, {}
    tasks_b, keys = evaluate.bootstrap_tasks({m: model_preds[m] for m in MODEL_SPECS}, sets, costs, chosen, k_star,
                                             {m: evaluate.N_GRID for m in MODEL_SPECS}, args.n_boot)
    cap_tasks, cap_keys = evaluate.cap_bootstrap_tasks(sets, costs, k_star, args.n_boot)
    ptasks, pkeys = paired_tasks(pairs, model_preds, sets, costs, chosen, args.n_boot)
    res = evaluate.parallel(evaluate.boot_task, tasks_b + cap_tasks + ptasks, args.cpu_jobs)
    boots = dict(zip(keys + cap_keys, res[:len(keys) + len(cap_keys)]))
    pair_boots = dict(zip(pkeys, res[len(keys) + len(cap_keys):]))
    evaluate.discrimination_ci(boots).write_csv(out / "discrimination_ci.csv", float_precision=FLOAT)
    ops_df = evaluate.operating_table(cap_main + [r for r in main_rows if r["model"] in MODEL_SPECS], boots)
    ops_df.write_csv(out / "operating_points.csv", float_precision=FLOAT)
    evaluate.sensitivity_table(cap_sens + [r for r in sens_rows if r["model"] in MODEL_SPECS]).write_csv(
        out / "operating_points_sensitivity.csv", float_precision=FLOAT)
    pair_df = pl.concat([b.with_columns(a=pl.lit(a), b=pl.lit(bm), N=pl.lit(n)) for (a, bm, n), b in pair_boots.items()]) \
        .select("a", "b", "N", "stat", "estimate", "lo", "hi", "n_boot") if pair_boots else None
    if pair_df is not None:
        pair_df.write_csv(out / "m2_vs_m1.csv", float_precision=FLOAT)
    lap(f"부트스트랩 ({len(res)}개 × {args.n_boot}회)")

    sens2 = pl.DataFrame(b1_input_sensitivity("m2_lstm_b1", cfgs["m2_lstm_b1"], paths, data_te, frames, sets, costs,
                                                                     chosen, model_preds, device))
    if sens2.height:
        sens2 = sens2.with_columns(diff=pl.col("sens_2") - pl.col("main_1"))
        sens2.write_csv(out / "b1_input_sensitivity.csv", float_precision=FLOAT)
    lap("② 민감도")

    compare = comparison_tables(out, disc, ops_df, boots)
    check(args, out, sets, data_tr, data_te, written, written_folds, oof_same, spread, disc, ops_df, k_star, compare, pair_df, sens2,
          dev_summary, timings)


def check(args, out, sets, data_tr, data_te, written, written_folds, oof_same, spread, disc, ops_df, k_star, compare, pair_df, sens2,
          dev_summary, timings):
    test_ops = ops_df.filter((pl.col("split") == "test") & pl.col("model").is_in(list(MODEL_SPECS)))
    judged = evaluate.judged_and_saves_more(ops_df).to_dicts()
    pair_sum = None
    if pair_df is not None:
        keep = ["d_auroc", "d_within_pair", "combined.d_saved_steps", "combined.d_saved_tokens", "single.d_saved_steps",
                "single.d_saved_tokens", "combined.d_lost_success", "single.d_lost_success"]
        pair_sum = pair_df.filter(pl.col("stat").is_in(keep)).sort("a", "stat", "N")
    s = {
        "data": {"train_sequences": data_tr.n, "test_sequences": data_te.n, "train_fingerprint": data_tr.fingerprint,
                 "step_features": sequences.STEP_FEATURES, "cum_features": sequences.CUM_FEATURES},
        "selected": dev_summary["selected"],
        "preds": {m: {"rows": w["rows"], "written": w["written"]} for m, w in written.items()},
        "preds_test_folds": {m: {"rows": w["rows"], "max_abs_mean_diff": w["max_abs_mean_diff"]} for m, w in written_folds.items()},
        "b1_test_fold_input_max_spread": spread,
        "k_star": k_star,
        "discrimination_N10": disc.filter(pl.col("N") == 10).select("model", "split", "auroc", "pr_auc", "within_pair", "within_issue_mean").to_dicts(),
        "n_summary_m2": compare["n_summary"].filter(pl.col("model").is_in(list(MODEL_SPECS))).to_dicts(),
        "judged_and_saves_more_than_k_star": judged,
        "loss_ci_width_summary": compare["width_summary"].to_dicts(),
        "m2_vs_m1": None if pair_sum is None else pair_sum.to_dicts(),
        "b1_input_sensitivity_max_abs_diff": None if not sens2.height else
        sens2.with_columns(a=pl.col("diff").abs()).group_by("kind", "metric").agg(pl.col("a").max()).sort("kind", "metric").to_dicts(),
        "bootstrap": {"n_boot": args.n_boot, "seed": io.SEED},
        "smoke": args.smoke,
    }
    crit = {
        "C1_preds_cover_running_trajectories": all(w["coverage"] for w in written.values()),  # preds.write가 1:1을 검사한다
        "C2_no_seen_issue": True,  # collect에서 assert
        "C3_final_oof_equals_dev_oof": all(oof_same.values()),
        "C4_k_star": args.smoke or k_star == 40,
        "C5_b1_test_input_differs_by_fold": args.smoke or spread > 0,
        "C6_fold_files_mean_matches": all(w["max_abs_mean_diff"] is not None and w["max_abs_mean_diff"] <= preds.FOLD_MEAN_TOL
                                          for w in written_folds.values()),
        "C7_operating_points": test_ops.height == len(MODEL_SPECS) * len(evaluate.N_GRID) * len(evaluate.POLICIES)
        and all(test_ops[f"{c}_fold_min"].null_count() == 0 for c in evaluate.POINT_COLUMNS)
        and test_ops["lost_success_hi"].null_count() == 0 and test_ops["meets_judge_ci_hi"].null_count() == 0,
        "C8_cap_rows_match_phase4": args.smoke or compare["cap_rows_match_phase4"],
        "C9_b1_sensitivity": sens2.height > 0 and sens2["sens_2"].null_count() == 0,
        "C10_m2_vs_m1_paired": pair_df is not None and pair_df["estimate"].null_count() == 0,
    }
    s["criteria"] = crit
    s["criteria_note"] = "pytest 전체 통과와 재실행 시 results/·preds/ 변화 없음은 스크립트 밖에서 확인한다. 실행 시간은 표준 출력에만 남긴다"
    io.write_summary(out / "summary.json", s)
    if not args.smoke:
        with pl.Config(tbl_rows=80, tbl_cols=20, tbl_width_chars=240, float_precision=4):
            print(compare["discrimination"].filter(pl.col("split") == "test").select("model", "N", "auroc", "auroc_lo", "auroc_hi",
                                                                                     "within_pair", "pr_auc").sort("N", "model"))
            print(compare["n_summary"].filter(pl.col("model").is_in(["b0_cap", *MODEL_SPECS, *M1_PAIR.values()])))
            if pair_sum is not None:
                print(pair_sum)
    print(json.dumps({"criteria": crit, "timings": timings}, ensure_ascii=False, indent=2, default=str))
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
    ap = argparse.ArgumentParser(description="Phase 6: M2 LSTM")
    ap.add_argument("--final", action="store_true", help="개발 실행에서 고른 설정으로 test를 한 번 채점하고 결과를 쓴다")
    ap.add_argument("--smoke", action="store_true", help="train 이슈 일부를 가짜 test로 써서 전체 경로를 점검한다 (data/processed/m2_smoke/)")
    ap.add_argument("--device", default="auto", help="auto(GPU 있으면 cuda) | cuda | cpu. 개발·최종은 같은 장치여야 한다")
    ap.add_argument("--jobs", type=int, default=None, help="학습 병렬 프로세스 수 (기본: cuda 4, cpu min(16, 코어 수))")
    ap.add_argument("--cpu-jobs", type=int, default=min(16, os.cpu_count() or 1), help="부트스트랩 병렬 수 (최종)")
    ap.add_argument("--n-boot", type=int, default=evaluate.N_BOOT)
    ap.add_argument("--force", action="store_true", help="저장된 fold 모델·예측 파일을 무시하고 다시 만든다")
    ap.add_argument("--smoke-epochs", type=int, default=2, help="점검 모드의 max_epochs (학습이 되는지 볼 때 늘린다)")
    args = ap.parse_args()
    device = lstm.resolve_device(args.device)
    if args.jobs is None:
        args.jobs = 4 if device == "cuda" else min(16, os.cpu_count() or 1)
    paths = Paths(args.smoke)
    for p, script in ((io.TRAJECTORIES_PATH, "01_parse.py"), (io.STEPS_PATH, "01_parse.py"), (io.SPLITS_PATH, "02_split.py"),
                      (preds.fold_path(B1_MODEL), "03_baselines.py"), (io.RESULTS_DIR / "phase5" / "operating_points.csv", "04_m1.py")):
        if not Path(p).exists():
            sys.exit(f"{p}가 없다. 먼저 uv run --no-sync python code/{script} 를 실행한다.")
    frames = load_b1_frames(args.final, args.smoke)  # 실제 예측 파일에서 읽는다 (점검 모드의 경로 변경 전)
    if args.smoke:
        preds.PREDS_DIR = paths.preds  # 점검 모드는 실제 예측 파일을 건드리지 않는다
        args.n_boot = min(args.n_boot, 20)
    t0 = time.time()
    timings = {}

    def lap(name):
        timings[name] = round(time.time() - t0 - sum(timings.values()), 1)
        print(f"[{timings[name]:>6.1f}s] {name}", flush=True)

    print(f"모드: {'최종' if args.final else '개발'}{' (점검)' if args.smoke else ''}, 장치 {device}, 학습 병렬 {args.jobs}", flush=True)
    sets = evaluate.load_eval_sets()
    if args.smoke:
        sets = smoke_sets(sets)
    costs = evaluate.load_costs(sets)
    data_tr = sequences.load(sets["train"])
    paths.cache.mkdir(parents=True, exist_ok=True)
    seq_meta = paths.seq_train / "meta.txt"
    if not seq_meta.exists() or seq_meta.read_text().split()[0] != data_tr.fingerprint:
        sequences.save_arrays(data_tr, paths.seq_train)
    b1_tr = sequences.b1_matrix(data_tr, frames["train"])
    b1_file = paths.cache / "b1_train.npy"
    np.save(b1_file, b1_tr)
    lap(f"train 시퀀스 ({data_tr.n}궤적, 스텝 입력 {len(sequences.STEP_FEATURES)} + 누적 {len(sequences.CUM_FEATURES)} + 명령 임베딩)")

    smoke_cfg = {"max_epochs": args.smoke_epochs, "n_seeds": 1, "patience": 1 if args.smoke_epochs <= 2 else 3} if args.smoke else {}
    if not args.final:
        dev(args, paths, sets, costs, data_tr, b1_tr, b1_file, device, smoke_cfg)
        lap("개발 실행")
        print(json.dumps({"timings": timings}, ensure_ascii=False))
        return
    final(args, paths, sets, costs, data_tr, b1_tr, frames, device, timings, lap)



if __name__ == "__main__":
    main()
