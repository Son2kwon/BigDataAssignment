"""Phase 5.7 보조 실험 (PLAN 11절 P5-D4·D5·D7, docs/phase5.md 5.7 사전 등록). 주 결과는 바꾸지 않는다.

실행: uv run --no-sync python code/04d_m1_supplementary.py [--final] [--n-boot 1000] [--jobs 12]

- --final 없이: train OOF 지표만 계산·출력한다 (탐색·개발용, test 행은 채점도 하지 않는다). 파일은 쓰지 않는다
- --final: 같은 선택(train OOF로 정해짐)을 다시 학습해 test를 한 번 채점하고, 판별·운영점·이전성 진단을 계산해
  results/phase5/supplementary/에 쓴다
- A2 넓힌 XGBoost 그리드 (m1_xgb, m1_xgb_b1 × N 6개, 72개 설정)
- A3 N=25·30 (M1 4종, 주 실험과 같은 그리드). B1 점수는 Phase 4 b1_n 설정으로 다시 만든다 (data/processed/supplementary/)
- A4 M1 운영점 이전성 진단 (evaluate.transfer_rows, 주 예측 파일 그대로, 새 학습 없음)

읽기만 하는 것: results/phase4·5/, data/processed/preds/, data/processed/features/. 쓰는 곳: results/phase5/supplementary/,
data/processed/supplementary/ (git 밖).
"""
import argparse
import dataclasses
import importlib
import json
import os
import sys
import time

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from afp import baselines, evaluate, features, io, metrics, oof, preds, split

m1 = importlib.import_module("04_m1")  # 주 실험의 학습기·그리드·선택 규칙을 그대로 쓴다

OUT_DIR = io.RESULTS_DIR / "phase5" / "supplementary"
DATA_DIR = io.PROCESSED_DIR / "supplementary"
# A2 (사전 등록 그대로): 축 순서·값 순서 = 그리드 순서 (동률이면 앞의 것)
WIDE_AXES = {"max_depth": (1, 2, 3), "n_estimators": (50, 100, 200), "learning_rate": (0.05, 0.1),
             "min_child_weight": (1, 10), "subsample": (0.8, 0.5)}
WIDE_GRID = tuple(dict(zip(WIDE_AXES, v)) for v in __import__("itertools").product(*WIDE_AXES.values()))
WIDE_MODELS = {"m1w_xgb": False, "m1w_xgb_b1": True}  # 보조 실험 모델 id (주 m1_xgb·m1_xgb_b1과 구분)
MAIN_OF_WIDE = {"m1w_xgb": "m1_xgb", "m1w_xgb_b1": "m1_xgb_b1"}
EXTRA_N = (25, 30)
CHECK_N = 20  # B1 재생성 코드가 Phase 4 b1_n과 같은지 확인하는 N
B1_MODEL = "b1_n"
FLOAT = 6


# ---- 학습 (test 채점은 final에서만) ----

def fit_wide(cfg, X, y):
    from xgboost import XGBClassifier
    return ("xgb", XGBClassifier(**{**m1.XGB_FIXED, **cfg}).fit(X, y))


def fit_main(kind, cfg):
    return lambda X, y: m1.fit_model(kind, cfg, X, y)


def run(fit_xy, X, y, inst, fold, X_test=None, test_inst=None, keep_models=False):
    """oof.oof_predict. X_test가 없으면 train OOF만 (test를 채점하지 않는다). fold k 모델은 X_test[k]로 채점 (P4-D7 ①)."""
    train_rows = pl.DataFrame({"instance_id": inst, "fold": fold, "_i": np.arange(len(y))})
    test_rows = None if X_test is None else pl.DataFrame({"instance_id": test_inst, "_i": np.arange(len(test_inst)), "_src": np.full(len(test_inst), -1)})

    def fit(rows):
        idx = rows["_i"].to_numpy()
        return fit_xy(X[idx], y[idx])

    def predict(model, rows):
        idx = rows["_i"].to_numpy()
        src = X if "_src" not in rows.columns else X_test[int(rows["_src"][0])]
        return m1.predict_model(model, src[idx])

    res = oof.oof_predict(train_rows, fit, predict, test=test_rows, keep_models=keep_models,
                          test_for_fold=None if X_test is None else (lambda k, t: t.with_columns(_src=pl.lit(k))))
    p = res.train["p_fail"].to_numpy()
    assert res.seen_issue_scores == 0
    return {"oof": p, "oof_auroc": float(roc_auc_score(y, p)), "test_folds": res.test_fold_scores}


def task(key, fit_xy, X, y, inst, fold, X_test, test_inst):
    return {"key": key, **run(fit_xy, X, y, inst, fold, X_test, test_inst)}


def task_args(key, fit_xy, D, final):
    """작업자에 넘길 인자: numpy 배열만 (loky가 큰 배열을 메모리 매핑한다). 이슈는 정수 코드 (oof는 묶음에만 쓴다)."""
    codes = {v: i for i, v in enumerate(sorted(set(D["train"]["instance_id"].to_list()) | set(D["test"]["instance_id"].to_list())))}
    inst = np.array([codes[v] for v in D["train"]["instance_id"].to_list()])
    test_inst = np.array([codes[v] for v in D["test"]["instance_id"].to_list()]) if final else None
    return (key, fit_xy, D["X"], D["y"], inst, D["train"]["fold"].to_numpy(), D["X_test"] if final else None, test_inst)


# ---- 설계 행렬 (B1 출처를 받는 일반형) ----

def design(sets, feats, n, b1_train, b1_test_folds, final):
    """04_m1.design과 같은 행렬. b1_train(traj_id, N, p_fail)·b1_test_folds({k: traj_id, N, p_fail})가 None이면 B1 제외.
    final이 아니면 test 행렬을 만들지 않는다."""
    cols = [*features.FEATURES, *([features.B1_FEATURE] if b1_train is not None else [])]
    f = feats.filter(pl.col("N") == n)
    out = {}
    for name in ("train", "test"):
        t = metrics.running(sets[name], n).select("traj_id", "instance_id", "target", "fold").sort("traj_id")
        out[name] = t.join(f, on="traj_id", how="left")
        if out[name].select(features.FEATURES).null_count().sum_horizontal().item():
            raise AssertionError(f"N={n} {name}: 피처가 없는 진행 중 궤적")
    train, test = out["train"], out["test"]
    if b1_train is not None:
        b = b1_train.filter(pl.col("N") == n).select("traj_id", pl.col("p_fail").alias(features.B1_FEATURE))
        train = train.join(b, on="traj_id", how="left")
        assert train[features.B1_FEATURE].null_count() == 0, f"N={n}: B1 OOF 점수가 없는 train 궤적"
    xs = None
    if final:
        if b1_train is None:
            xs = [test.select(cols).to_numpy()] * split.N_FOLDS
        else:
            xs = []
            for k in range(split.N_FOLDS):
                bk = b1_test_folds[k].filter(pl.col("N") == n).select("traj_id", pl.col("p_fail").alias(features.B1_FEATURE))
                tk = test.join(bk, on="traj_id", how="left")
                assert tk[features.B1_FEATURE].null_count() == 0, f"N={n}: B1 fold {k} 점수가 없는 test 궤적"
                xs.append(tk.select(cols).to_numpy())
    return {"cols": cols, "train": train, "test": test, "X": train.select(cols).to_numpy(), "y": (~train["target"]).to_numpy(), "X_test": xs}


# ---- A3용 B1 점수 (Phase 4 b1_n과 같은 설정) ----

def b1_config():
    cfg = preds.metadata(B1_MODEL)["config"]
    fields = {f.name for f in dataclasses.fields(baselines.B1Config)}
    return baselines.B1Config(**{k: v for k, v in cfg.items() if k in fields})


def b1_scores(sets, texts, n, config, final):
    """시점 N의 B1: train OOF (traj_id, N, p_fail)와 final이면 test fold별 {k: (traj_id, N, p_fail)}."""
    rows = baselines.issue_rows(sets["train"], n, texts)
    test_rows = baselines.issue_rows(sets["test"], n, texts) if final else None
    fit, predict = baselines.b1_functions(config)
    res = oof.oof_predict(rows, fit, predict, test=test_rows)
    train = baselines.to_trajectories(sets["train"], res.train.select("instance_id", "p_fail"), n)
    folds = None
    if final:
        folds = {k: baselines.to_trajectories(sets["test"], test_rows.select("instance_id", p_fail=pl.Series(v)), n)
                 for k, v in enumerate(res.test_fold_scores)}
    return {"n": n, "train": train, "folds": folds}


def b1_task(sets, texts, n, config, final):
    return b1_scores(sets, texts, n, config, final)


# ---- 운영점 (train만 / final) ----

def train_ops(train, p_train, n, costs_train):
    out = {}
    for policy in evaluate.POLICIES:
        op = metrics.select_from_curves(evaluate.candidate_curves(train, p_train, n, policy, costs_train), n)
        out[policy] = {c: op[c] for c in ("tau", "cap", "stop_rate", "lost_success", "saved_steps", "saved_tokens")}
    return out


def transfer_task(model, n, policy, train, train_preds, test, test_preds, full_op, test_fold_preds):
    rows = evaluate.transfer_rows(train, train_preds, test, test_preds, n, policy, full_op, test_fold_preds=test_fold_preds)
    return [{"model": model, "policy": policy, "N": n, **r} for r in rows]


def boundary(cfg):
    """고른 설정이 끝값에 있는 축: {축: 'min'|'max'} (값이 1개뿐인 축은 없음)."""
    out = {}
    for ax, vals in WIDE_AXES.items():
        lo, hi = min(vals), max(vals)
        if cfg[ax] == lo:
            out[ax] = "min"
        elif cfg[ax] == hi:
            out[ax] = "max"
    return out


def assemble(D, oof_p, test_folds, n):
    """train OOF + test 평균 (preds 형식)과 test fold별 (traj_id, N, fold, p_fail)."""
    p = pl.concat([D["train"].select("traj_id", N=pl.lit(n, dtype=pl.Int16), p_fail=pl.Series(oof_p)),
                   D["test"].select("traj_id", N=pl.lit(n, dtype=pl.Int16), p_fail=pl.Series(test_folds.mean(axis=0)))])
    f = pl.concat([D["test"].select("traj_id", N=pl.lit(n, dtype=pl.Int16), fold=pl.lit(k, dtype=pl.Int8), p_fail=pl.Series(test_folds[k]))
                   for k in range(split.N_FOLDS)])
    return p, f


def write_parquet(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and pl.read_parquet(path).equals(df):
        return
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(df.to_arrow(), tmp, compression="zstd")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description="Phase 5.7 보조 실험")
    ap.add_argument("--final", action="store_true", help="고른 설정으로 test를 한 번 채점하고 결과 파일을 쓴다 (없으면 train OOF만 출력)")
    ap.add_argument("--n-boot", type=int, default=evaluate.N_BOOT)
    ap.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1))
    args = ap.parse_args()
    final = args.final
    for p in (m1.FEATURES_PATH, preds.path("m1_xgb"), preds.fold_path(B1_MODEL)):
        if not p.exists():
            sys.exit(f"{p}가 없다. 먼저 code/04_m1.py(와 03_baselines.py)를 실행한다.")
    t0 = time.time()
    timings = {}

    def lap(name):
        timings[name] = round(time.time() - t0 - sum(timings.values()), 1)
        print(f"[{timings[name]:>6.1f}s] {name}", flush=True)

    sets = evaluate.load_eval_sets()
    costs = evaluate.load_costs(sets)
    feats_main = pl.read_parquet(m1.FEATURES_PATH)
    b1_train_main = preds.load_feature(B1_MODEL, "train")
    b1_folds_main = {k: preds.load_feature(B1_MODEL, "test", fold=k) for k in range(split.N_FOLDS)} if final else None
    cap_train = metrics.fixed_cap_curve(sets["train"], costs["train"], range(1, 81))
    k_star = metrics.select_cap(cap_train, metrics.SELECT_MAX_LOSS)["K"]
    assert k_star == 40
    lap("입력")

    # ---- A2: 넓힌 그리드 (train OOF로 선택) ----
    D2 = {(m, n): design(sets, feats_main, n, b1_train_main if b1 else None, b1_folds_main if b1 else None, final)
          for m, b1 in WIDE_MODELS.items() for n in evaluate.N_GRID}
    tasks = [task_args((m, n, i), (lambda c: (lambda X, y: fit_wide(c, X, y)))(cfg), D, False)
             for (m, n), D in D2.items() for i, cfg in enumerate(WIDE_GRID)]
    res = {r["key"]: r for r in evaluate.parallel(task, tasks, args.jobs)}
    grid_rows, sel2 = [], {}
    for (m, n) in D2:
        rs = [res[(m, n, i)] for i in range(len(WIDE_GRID))]
        best = max(r["oof_auroc"] for r in rs)
        pick = min(i for i, r in enumerate(rs) if r["oof_auroc"] >= best - m1.GRID_TIE)
        argmax = int(np.argmax([r["oof_auroc"] for r in rs]))
        sel2[(m, n)] = {"i": pick, "cfg": WIDE_GRID[pick], "oof_auroc": rs[pick]["oof_auroc"], "oof": rs[pick]["oof"],
                        "argmax_cfg": WIDE_GRID[argmax], "argmax_auroc": best}
        grid_rows += [{"model": m, "N": n, "i": i, **WIDE_GRID[i], "oof_auroc": r["oof_auroc"], "selected": i == pick,
                       "argmax": i == argmax} for i, r in enumerate(rs)]
    main_grid = pl.read_csv(io.RESULTS_DIR / "phase5" / "m1_grid.csv").filter(pl.col("selected"))
    main_oof = {(r["model"], r["N"]): r["oof_auroc"] for r in main_grid.iter_rows(named=True)}
    sel2_rows = [{"model": m, "N": n, "selected_cfg": json.dumps(s["cfg"], sort_keys=True), "oof_auroc": s["oof_auroc"],
                  "argmax_cfg": json.dumps(s["argmax_cfg"], sort_keys=True), "argmax_oof_auroc": s["argmax_auroc"],
                  "boundary": json.dumps(boundary(s["cfg"]), sort_keys=True), "main_oof_auroc": main_oof[(MAIN_OF_WIDE[m], n)],
                  "d_oof_auroc_vs_main": s["oof_auroc"] - main_oof[(MAIN_OF_WIDE[m], n)]}
                 for (m, n), s in sel2.items()]
    with pl.Config(tbl_rows=20, tbl_width_chars=230, fmt_str_lengths=110, float_precision=4):
        print(pl.DataFrame(sel2_rows).drop("argmax_cfg"))
    lap(f"A2 그리드 OOF ({len(tasks)}개)")

    # ---- A3: N=25·30 ----
    trajs = pl.concat([t.select("traj_id", "instance_id", "n_steps") for t in sets.values()])
    values = features.load_step_values(max(EXTRA_N), trajs)
    tt = io.load_trajectories(["traj_id", "instance_id", "issue_text"]).join(trajs.select("traj_id"), on="traj_id", how="semi")
    feats_x = features.m1_features(values, trajs, EXTRA_N, features.mentions_table(tt))
    feat_counts = features.check_features(feats_x, trajs, EXTRA_N)
    texts = baselines.issue_table()
    config = b1_config()
    b1 = {r["n"]: r for r in evaluate.parallel(b1_task, [(sets, texts, n, config, final) for n in (CHECK_N, *EXTRA_N)], args.jobs)}
    # 재생성 코드 확인: N=20 점수가 Phase 4 b1_n과 같아야 한다
    ref = b1_train_main.filter(pl.col("N") == CHECK_N).sort("traj_id")
    b1_same = b1[CHECK_N]["train"].sort("traj_id").select("traj_id", "p_fail").equals(ref.select("traj_id", "p_fail"))
    if final:
        b1_same &= all(b1[CHECK_N]["folds"][k].sort("traj_id").select("traj_id", "p_fail").equals(
            b1_folds_main[k].filter(pl.col("N") == CHECK_N).sort("traj_id").select("traj_id", "p_fail")) for k in range(split.N_FOLDS))
    print(f"B1 재생성 확인 (N={CHECK_N}, Phase 4 b1_n과 같음): {b1_same}")
    assert b1_same, "B1 재생성 결과가 Phase 4 b1_n과 다르다 (B1 포함 모델을 하지 않아야 한다)"
    b1x_train = pl.concat([b1[n]["train"] for n in EXTRA_N])
    b1x_folds = {k: pl.concat([b1[n]["folds"][k] for n in EXTRA_N]) for k in range(split.N_FOLDS)} if final else None
    D3 = {(m, n): design(sets, feats_x, n, b1x_train if b1f else None, b1x_folds if b1f else None, final)
          for m, (kind, b1f) in m1.MODEL_SPECS.items() for n in EXTRA_N}
    tasks3 = [task_args((m, n, i), fit_main(m1.MODEL_SPECS[m][0], cfg), D, False)
              for (m, n), D in D3.items() for i, cfg in enumerate(m1.LR_GRID if m1.MODEL_SPECS[m][0] == "lr" else m1.XGB_GRID)]
    res3 = {r["key"]: r for r in evaluate.parallel(task, tasks3, args.jobs)}
    sel3 = {}
    grid3_rows = []
    for (m, n) in D3:
        grid = m1.LR_GRID if m1.MODEL_SPECS[m][0] == "lr" else m1.XGB_GRID
        rs = [res3[(m, n, i)] for i in range(len(grid))]
        pick = m1.choose([{"i": i, "oof_auroc": r["oof_auroc"]} for i, r in enumerate(rs)])
        sel3[(m, n)] = {"i": pick, "cfg": grid[pick], "oof_auroc": rs[pick]["oof_auroc"], "oof": rs[pick]["oof"]}
        grid3_rows += [{"model": m, "N": n, "i": i, "config": m1.cfg_str(grid[i]), "oof_auroc": r["oof_auroc"], "selected": i == pick}
                       for i, r in enumerate(rs)]
    # 진행 중 궤적·성공률·이슈 내 계산 이슈 수 (train), train OOF 판별·운영점
    pop_rows, a3_train = [], []
    for n in EXTRA_N:
        for name in ("train", "test") if final else ("train",):
            a = metrics.running(sets[name], n)
            mixed = a.group_by("instance_id").agg(s=pl.col("target").sum(), c=pl.len()).filter((pl.col("s") > 0) & (pl.col("s") < pl.col("c")))
            pop_rows.append({"split": name, "N": n, "n_running": a.height, "success_rate": float(a["target"].mean()),
                             "within_n_issues": mixed.height, "share_of_all_traj": a.height / sets[name].height})
    for (m, n), s in sel3.items():
        D = D3[(m, n)]
        p_tr = D["train"].select("traj_id", N=pl.lit(n, dtype=pl.Int16), p_fail=pl.Series(s["oof"]))
        d = evaluate.discrimination_row(sets["train"], p_tr, n)
        ops = train_ops(sets["train"], p_tr, n, costs["train"])
        a3_train.append({"model": m, "N": n, "config": m1.cfg_str(s["cfg"]), "train_oof_auroc": d["auroc"], "train_oof_within_pair": d["within_pair"],
                         **{f"{pol}_{c}": v for pol, o in ops.items() for c, v in o.items()}})
    # 좋은 N 규칙(5.6)을 N 3~30에 다시 적용 (보고만)
    good = pl.read_csv(io.RESULTS_DIR / "phase5" / "good_n.csv").select("model", "N", "train_oof_auroc", "train_saved_steps_combined")
    ext = pl.concat([good, pl.DataFrame([{"model": r["model"], "N": r["N"], "train_oof_auroc": r["train_oof_auroc"],
                                          "train_saved_steps_combined": r["combined_saved_steps"]} for r in a3_train]).cast(good.schema)])
    good_rows = []
    for m in m1.MODEL_SPECS:
        j = ext.filter(pl.col("model") == m).sort("N")
        best = j["train_oof_auroc"].max()
        cand = j.filter(pl.col("train_oof_auroc") >= best - m1.GOOD_N_TIE).sort(["train_saved_steps_combined", "N"], descending=[True, False])
        pick = int(cand["N"][0])
        good_rows += [{**r, "in_tie_band": r["train_oof_auroc"] >= best - m1.GOOD_N_TIE, "selected_extended": r["N"] == pick}
                      for r in j.iter_rows(named=True)]
    with pl.Config(tbl_rows=40, tbl_width_chars=230, float_precision=4, fmt_str_lengths=60):
        print(pl.DataFrame(pop_rows))
        print(pl.DataFrame(a3_train).select("model", "N", "config", "train_oof_auroc", "train_oof_within_pair", "single_saved_steps",
                                            "combined_cap", "combined_saved_steps", "combined_saved_tokens"))
        print(pl.DataFrame(good_rows).filter(pl.col("selected_extended") | pl.col("in_tie_band")))
    lap(f"A3 N=25·30 (그리드 {len(tasks3)}개)")

    if not final:
        print(json.dumps({"mode": "dev (train OOF only, no test scored, no files written)", "timings": timings}, ensure_ascii=False, indent=2))
        return

    # ================= final: 고른 설정으로 test를 한 번 채점 =================
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_parquet(feats_x, DATA_DIR / "m1_n25_30_features.parquet")
    write_parquet(pl.concat([b1x_train, *[b1x_folds[k].with_columns(fold=pl.lit(k, dtype=pl.Int8)) for k in range(split.N_FOLDS)]],
                            how="diagonal"), DATA_DIR / "b1_n_n25_30.parquet")
    refit = [task_args((m, n), (lambda c: (lambda X, y: fit_wide(c, X, y)))(s["cfg"]), D2[(m, n)], True) for (m, n), s in sel2.items()]
    refit += [task_args((m, n), fit_main(m1.MODEL_SPECS[m][0], s["cfg"]), D3[(m, n)], True) for (m, n), s in sel3.items()]
    rr = {r["key"]: r for r in evaluate.parallel(task, refit, args.jobs)}
    same = all(np.array_equal(rr[k]["oof"], s["oof"]) for k, s in {**sel2, **sel3}.items())
    assert same, "final 재학습 OOF가 탐색 때와 다르다"
    model_preds, n_grid_of = {}, {}
    for (m, n), D in {**D2, **D3}.items():
        p, f = assemble(D, rr[(m, n)]["oof"], rr[(m, n)]["test_folds"], n)
        prev = model_preds.get(m)
        model_preds[m] = (p, f) if prev is None else (pl.concat([prev[0], p]), pl.concat([prev[1], f]))
        n_grid_of.setdefault(m, []).append(n)
    for m, (p, f) in model_preds.items():
        write_parquet(p.sort("N", "traj_id"), DATA_DIR / f"{m if m.startswith('m1w') else m + '_n25_30'}.parquet")
        write_parquet(f.sort("N", "fold", "traj_id"), DATA_DIR / f"{m if m.startswith('m1w') else m + '_n25_30'}.test_folds.parquet")
    lap("final 재학습·test 채점")

    disc = pl.concat([evaluate.discrimination_table(m, p, sets, n_grid_of[m]) for m, (p, _) in model_preds.items()])
    main_rows, chosen = [], {}
    for m, (p, f) in model_preds.items():
        rows, _, ch = evaluate.operating_points({m: (p, f)}, sets, costs, k_star, n_grid=n_grid_of[m])
        main_rows += rows
        chosen.update(ch)
    cap_main, _ = evaluate.cap_rows(sets, costs, k_star, cap_train)
    tasks_b, keys = evaluate.bootstrap_tasks(model_preds, sets, costs, chosen, k_star, n_grid_of, args.n_boot)
    cap_tasks, cap_keys = evaluate.cap_bootstrap_tasks(sets, costs, k_star, args.n_boot)
    boots = dict(zip(keys + cap_keys, evaluate.parallel(evaluate.boot_task, tasks_b + cap_tasks, args.jobs)))
    ops = evaluate.operating_table(cap_main + main_rows, boots)
    dci = evaluate.discrimination_ci(boots)
    lap(f"final 판별·운영점·부트스트랩 ({len(tasks_b) + len(cap_tasks)}개)")

    # ---- A4: 주 M1 예측의 운영점 이전성 ----
    main_preds = {m: (preds.read(m), preds.read_test_folds(m)) for m in m1.MODEL_SPECS}
    rows_main, _, _ = evaluate.operating_points(main_preds, sets, costs, k_star)
    full = {(r["model"], r["N"], r["policy"]): r for r in rows_main if r["split"] == "train"}
    train, test = sets["train"], sets["test"]
    ttasks = []
    for m, (p, f) in main_preds.items():
        for n in evaluate.N_GRID:
            fl = [f.filter((pl.col("N") == n) & (pl.col("fold") == k)).select("traj_id", "N", "p_fail").sort("traj_id") for k in range(split.N_FOLDS)]
            for policy in evaluate.POLICIES:
                ttasks.append((m, n, policy, train, evaluate.split_preds(p, train, n), test, evaluate.split_preds(p, test, n),
                               full[(m, n, policy)], fl))
    transfer = evaluate.frame([r for out in evaluate.parallel(transfer_task, ttasks, args.jobs) for r in out],
                              ["model", "policy", "N", *evaluate.TRANSFER_COLUMNS], evaluate.OP_SCHEMA)
    tsum = evaluate.transfer_summary(transfer)
    # P4-D6 기준 (fold 모델 5개 평균에 적용한) 주 운영점의 test 손실 − train 손실
    main_ops = pl.read_csv(io.RESULTS_DIR / "phase5" / "operating_points.csv", schema_overrides={"N": pl.Int64})
    d6 = (main_ops.filter(pl.col("model").is_in(list(m1.MODEL_SPECS)))
          .pivot(on="split", index=["model", "policy", "N"], values="lost_success")
          .with_columns(d_lost_p4d6=pl.col("test") - pl.col("train")).select("model", "policy", "N", "d_lost_p4d6"))
    tsum2 = tsum.join(d6, on=["model", "policy", "N"], how="left").with_columns(
        p4d6_within_fold_range=(pl.col("d_lost_p4d6") >= pl.col("fold_d_lost_min")) & (pl.col("d_lost_p4d6") <= pl.col("fold_d_lost_max")),
        test_fold_model_within_fold_range=(pl.col("test_fold_model_d_lost_min") >= pl.col("fold_d_lost_min"))
        & (pl.col("test_fold_model_d_lost_max") <= pl.col("fold_d_lost_max")))
    lap(f"A4 이전성 진단 ({len(ttasks)}개)")

    # ---- 저장 ----
    pl.DataFrame(grid_rows).write_csv(OUT_DIR / "wide_grid.csv", float_precision=FLOAT)
    pl.DataFrame(grid3_rows).write_csv(OUT_DIR / "n25_30_grid.csv", float_precision=FLOAT)
    disc.write_csv(OUT_DIR / "discrimination.csv", float_precision=FLOAT)
    dci.write_csv(OUT_DIR / "discrimination_ci.csv", float_precision=FLOAT)
    ops.write_csv(OUT_DIR / "operating_points.csv", float_precision=FLOAT)
    pl.DataFrame(pop_rows).write_csv(OUT_DIR / "n25_30_population.csv", float_precision=FLOAT)
    pl.DataFrame(good_rows).write_csv(OUT_DIR / "good_n_extended.csv", float_precision=FLOAT)
    transfer.write_csv(OUT_DIR / "tau_transfer_m1.csv", float_precision=FLOAT)
    tsum2.write_csv(OUT_DIR / "tau_transfer_m1_summary.csv", float_precision=FLOAT)

    # A2 비교 표: 주 결과 vs 넓힌 그리드 (train OOF AUROC, test AUROC, 이슈 내 AUROC, N=10·20 운영점과 토큰)
    dmain = pl.read_csv(io.RESULTS_DIR / "phase5" / "discrimination.csv", schema_overrides={"N": pl.Int64})
    omain = main_ops
    cmp_rows = []
    for (m, n), s in sel2.items():
        mm = MAIN_OF_WIDE[m]
        def drow(df, model, split_name):
            return df.filter((pl.col("model") == model) & (pl.col("N") == n) & (pl.col("split") == split_name)).row(0, named=True)
        a, b = drow(dmain, mm, "test"), drow(disc, m, "test")
        row = {"model_main": mm, "N": n, "main_cfg": main_grid.filter((pl.col("model") == mm) & (pl.col("N") == n))["config"][0],
               "wide_cfg": json.dumps(s["cfg"], sort_keys=True),
               "train_oof_auroc_main": main_oof[(mm, n)], "train_oof_auroc_wide": s["oof_auroc"],
               "test_auroc_main": a["auroc"], "test_auroc_wide": b["auroc"], "test_within_pair_main": a["within_pair"], "test_within_pair_wide": b["within_pair"]}
        cmp_rows.append(row)
    pl.DataFrame(cmp_rows).write_csv(OUT_DIR / "wide_vs_main_discrimination.csv", float_precision=FLOAT)
    keep = ["model", "policy", "N", "tau", "cap", "split", "stop_rate", "lost_success", "lost_success_hi", "meets_judge_ci_hi",
            "saved_steps", "d_saved_steps", "d_saved_steps_lo", "d_saved_steps_hi", "saved_tokens", "d_saved_tokens", "d_saved_tokens_lo", "d_saved_tokens_hi"]
    wv = pl.concat([omain.filter(pl.col("model").is_in(list(MAIN_OF_WIDE.values())) & pl.col("N").is_in([10, 20])).select(keep),
                    ops.filter(pl.col("model").is_in(list(WIDE_MODELS)) & pl.col("N").is_in([10, 20])).select(keep).cast(omain.select(keep).schema)])
    wv.sort("N", "policy", "model", "split").write_csv(OUT_DIR / "wide_vs_main_operating_points.csv", float_precision=FLOAT)

    s = {
        "preregistered": "docs/phase5.md 5.7 사전 등록",
        "a2": {"grid": {k: list(v) for k, v in WIDE_AXES.items()}, "n_configs": len(WIDE_GRID), "fixed": {k: v for k, v in m1.XGB_FIXED.items() if k not in WIDE_AXES},
               "selected": sel2_rows},
        "a3": {"extra_n": list(EXTRA_N), "feature_rows": {str(k): v for k, v in feat_counts.items()}, "b1_config": config.as_dict(),
               "b1_regeneration_matches_phase4_at_N20": bool(b1_same), "population": pop_rows, "train": a3_train,
               "good_n_extended": [r for r in good_rows if r["selected_extended"]]},
        "a4_summary_N10": tsum2.filter(pl.col("N") == 10).to_dicts(),
        "final_refit_oof_matches_search": bool(same),
        "k_star": k_star,
        "judged_and_saves_more_than_k_star": evaluate.judged_and_saves_more(ops).to_dicts(),
        "bootstrap": {"n_boot": args.n_boot, "tasks": len(boots), "seed": io.SEED},
        "note": "주 결과는 바꾸지 않는다. 개발 실행(--final 없음)은 train OOF만 출력했다. 실행 시간은 표준 출력에만 남긴다",
    }
    io.write_summary(OUT_DIR / "summary.json", s)
    with pl.Config(tbl_rows=60, tbl_width_chars=240, float_precision=4, fmt_str_lengths=90):
        print(pl.DataFrame(cmp_rows).drop("main_cfg"))
        print(wv.filter(pl.col("split") == "test"))
        print(ops.filter(pl.col("N").is_in(list(EXTRA_N)) & (pl.col("split") == "test")).select(keep))
        print(tsum2.filter(pl.col("N") == 10).select("model", "policy", "fold_d_lost_min", "fold_d_lost_max", "test_fold_model_d_lost_min",
                                                     "test_fold_model_d_lost_max", "test_from_train_d_lost_min", "d_lost_p4d6", "p4d6_within_fold_range"))
    print(json.dumps({"timings": timings}, ensure_ascii=False))


if __name__ == "__main__":
    main()
